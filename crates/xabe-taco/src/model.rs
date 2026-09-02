//! Tacotron2: symbols in, mel out.
//!
//! An encoder that reads the whole line at once, then a decoder that emits one
//! mel frame at a time until a gate says stop. Three things about it are worth
//! knowing before reading the loop.
//!
//! # The prenet keeps its dropout at inference
//!
//! `Prenet.forward` in the reference passes `training=True` to `F.dropout`
//! unconditionally, while every other dropout in the model is conditioned on
//! the module's mode. That is deliberate and load-bearing: without the noise
//! the decoder learns to copy its own previous frame and the alignment
//! collapses. So this samples a mask per layer per step, and synthesis is
//! stochastic even before WaveGlow's noise. The seed is the caller's, so a
//! given seed and text give a given utterance.
//!
//! # The gate is compared before the sigmoid
//!
//! `sigmoid(x) > p` is `x > ln(p / (1 - p))`, and the threshold is a constant.
//! Doing it that way keeps a `sigmoid` kernel out of the crate for the sake of
//! one number per step.
//!
//! # The attention query is added through a bias
//!
//! The energies are `v · tanh(query + location + memory)`, where the query is
//! one `[128]` vector added to every one of the `[tokens, 128]` rows. That is
//! exactly what a linear layer's bias does, so the query is passed as the
//! location projection's bias rather than broadcast by a kernel of its own.

use crate::clock::Clock;
use crate::weights::{Lstm, Taco2};
use crate::{Config, TacoError};
use xabe_cuda::{CudaSlice, Gpu, Operand, OutLayout};

/// An LSTM's carried state.
pub(crate) struct Cell {
    c: CudaSlice<f32>,
    /// `[hidden + extra]`: the hidden state, then whatever the projection
    /// that reads it wants concatenated after - the attention context, for
    /// both of the decoder's cells. The gates write the first `hidden`; the
    /// attention writes the rest in place, so no launch concatenates.
    h: CudaSlice<f32>,
    /// The recurrent side of the gates, `[4 * hidden]`, kept so a step
    /// allocates nothing; see `synthesize`.
    gh: CudaSlice<f32>,
}

impl Cell {
    fn zeros(gpu: &Gpu, hidden: usize, extra: usize) -> Result<Self, TacoError> {
        Ok(Self {
            c: gpu.zeros(hidden)?,
            h: gpu.zeros(hidden + extra)?,
            // SAFETY: written whole by the mat-vec before the gates read it.
            gh: unsafe { gpu.uninit(4 * hidden) }?,
        })
    }
}

/// Advances one LSTM step. `gi` is the input side, bias already in it.
fn step_cell(gpu: &Gpu, w: &Lstm, gi: &CudaSlice<f32>, st: &mut Cell) -> Result<(), TacoError> {
    // `gemv_into` rather than `linear`: one row is the `gemv` kernel, which
    // keeps the activation f32 end to end and reads the weight at whichever
    // width it was bound - see `Lstm` - and the into-form writes the gates
    // into the cell's own buffer rather than a fresh one.
    let g = 4 * w.hidden;
    gpu.gemv_into(
        &st.h,
        w.w_hh.operand(),
        Some(&w.b_hh),
        w.hidden,
        g,
        false,
        OutLayout::Row,
        &mut st.gh,
    )?;
    gpu.lstm_gates(gi, &st.gh, &mut st.c, &mut st.h, w.hidden)?;
    Ok(())
}

/// Dropout masks, and nothing else.
///
/// xorshift64* rather than anything better: this decides which half of 256
/// units survive, five hundred times an utterance, and a generator whose
/// spectral properties matter would be a generator that needs testing.
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator. Zero is remapped, since xorshift is stuck there.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }

    /// A standard normal, by Box-Muller.
    ///
    /// The cosine branch only; the sine one is discarded rather than cached,
    /// because caching would make the stream depend on how many values a caller
    /// asked for at a time.
    pub fn normal(&mut self) -> f32 {
        let u = self.uniform().max(f32::MIN_POSITIVE);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// The two halves of a synthesis: what came out, and why it stopped.
pub(crate) struct Mel {
    /// `[n_mel, frames]`, postnet already added.
    pub(crate) data: CudaSlice<f32>,
    /// How many frames.
    pub(crate) frames: usize,
    /// Whether the gate fired rather than the step limit being hit.
    pub(crate) stopped: bool,
}

/// Reads the whole line: embedding, three convolutions, one bidirectional LSTM.
///
/// Split out from the decode loop because it is the deterministic half. No
/// dropout survives to inference here - the reference conditions this one on
/// training mode, unlike the prenet's - so this is the part that can be
/// compared against the reference tensor for tensor.
///
/// Returns the memory, `[tokens, 512]`.
pub(crate) fn encode(
    gpu: &Gpu,
    w: &Taco2,
    c: &Config,
    ids: &[i64],
) -> Result<CudaSlice<f32>, TacoError> {
    let (e, t) = (c.encoder_dim, ids.len());

    // The embedding gather, on the host. `[tokens, 512]` up, then transposed
    // into the convolutions' `[channels, time]`.
    let mut emb = vec![0.0f32; t * e];
    for (i, &id) in ids.iter().enumerate() {
        let row = id as usize * e;
        emb[i * e..(i + 1) * e].copy_from_slice(&w.embedding[row..row + e]);
    }
    let flat = gpu.upload(&emb)?;
    let mut x = gpu.transpose(&flat, t, e)?;

    // Three convolutions, each with its batch norm already folded in. The
    // reference's dropout here is conditioned on training mode and so is absent.
    let pad = c.encoder_kernel / 2;
    for conv in &w.enc_convs {
        let (mut y, _) = gpu.conv1d(
            &x,
            conv.w.full(),
            Some(&conv.bias),
            e,
            t,
            e,
            conv.k,
            pad,
            pad,
            1,
        )?;
        gpu.relu(&mut y, e * t)?;
        x = y;
    }
    let seq = gpu.transpose(&x, e, t)?;

    // The bidirectional encoder. Each direction's input projection is one
    // matmul over the whole sequence; only the recurrence is a loop.
    let h = c.lstm_hidden;
    let mut memory = gpu.zeros(t * 2 * h)?;
    for (lstm, offset) in [(&w.enc_fwd, 0), (&w.enc_rev, h)] {
        let gi_all = gpu.linear(&seq, lstm.w_ih.full(), Some(&lstm.b_ih), t, e, 4 * h)?;
        let mut st = Cell::zeros(gpu, h, 0)?;
        for i in 0..t {
            let at = if offset == 0 { i } else { t - 1 - i };
            let gi = gpu.copy_range(&gi_all, at * 4 * h, 4 * h)?;
            step_cell(gpu, lstm, &gi, &mut st)?;
            // Written straight into the concatenated output, so the forward and
            // backward passes never need a buffer of their own.
            gpu.copy_into(&mut memory, &st.h, at * 2 * h + offset, h)?;
        }
    }
    Ok(memory)
}

/// Frames decoded between reads of the stop gate. See the loop in
/// [`synthesize`].
pub(crate) const GATE_LOOKAHEAD: usize = 8;

/// What one decoder frame reads and writes, held across the frames so the
/// frame allocates nothing: the two cells, each carrying the context after
/// its state; the running attention weights; the output and gate buffers
/// the frame writes at its index; the prenet's input buffer; and the
/// frame's temporaries.
struct FrameState {
    att: Cell,
    dec: Cell,
    weights_cat: CudaSlice<f32>,
    out: CudaSlice<f32>,
    gates: CudaSlice<f32>,
    /// `[prenet_dim + e]`: the prenet's output, then the context.
    cell_in: CudaSlice<f32>,
    /// The frame's temporaries, one buffer each, so that a frame allocates
    /// nothing: the prenet's layer outputs but the last, which is the head
    /// of `cell_in`; the two input-side gate vectors; the location
    /// features; the query; the alignment energies; and the stacked
    /// projection-and-gate row, which is also the next frame's input.
    pre: Vec<CudaSlice<f32>>,
    gi_att: CudaSlice<f32>,
    gi_dec: CudaSlice<f32>,
    loc: CudaSlice<f32>,
    query: CudaSlice<f32>,
    score: CudaSlice<f32>,
    next: CudaSlice<f32>,
}

/// Runs the encoder and then the decode loop.
///
/// `lookahead` is how many frames are decoded between reads of the stop
/// gate; [`GATE_LOOKAHEAD`] in service, and one in the test that holds the
/// batched read to the frame-by-frame one. Zero is taken as one.
pub(crate) fn synthesize(
    gpu: &Gpu,
    w: &Taco2,
    c: &Config,
    ids: &[i64],
    rng: &mut Rng,
    clock: &mut Clock,
    lookahead: usize,
) -> Result<Mel, TacoError> {
    let lookahead = lookahead.max(1);
    let (e, mel, t) = (c.encoder_dim, c.n_mel, ids.len());
    let at = clock.start();
    let memory = encode(gpu, w, c, ids)?;
    clock.stop(gpu, "encoder", at)?;

    let processed = gpu.linear(&memory, &w.memory.w, None, t, e, c.attention_dim)?;

    // Decoder state. The go frame is zeros, and so is everything else -
    // including the context halves of the three concatenated buffers, which
    // the first frame reads before the attention has written them.
    let att = Cell::zeros(gpu, c.attention_rnn_dim, e)?;
    let dec = Cell::zeros(gpu, c.decoder_rnn_dim, e)?;
    let weights_cat = gpu.zeros(2 * t)?;
    let cell_in = gpu.zeros(c.prenet_dim + e)?;
    let next = gpu.zeros(mel + 1)?;

    let out = gpu.zeros(c.max_decoder_steps * mel)?;
    let dhac = c.decoder_rnn_dim + e;
    let lpad = c.location_kernel / 2;
    // `sigmoid(x) > p` without the sigmoid.
    let gate_logit = (c.gate_threshold / (1.0 - c.gate_threshold)).ln();

    // The prenet's dropout masks for the whole utterance, drawn now in the
    // order the loop used to draw them - layer by layer, frame by frame - and
    // uploaded once. They were two host draws and two uploads a frame, each
    // upload a synchronous copy that held the stream. A frame reads its own
    // row of the table, so the masks a frame gets are the ones it always got.
    let mask_w: usize = w.prenet.iter().map(|l| l.out_c).sum();
    let mut mask_table = Vec::with_capacity(c.max_decoder_steps * mask_w);
    for _ in 0..c.max_decoder_steps {
        for layer in &w.prenet {
            mask_table
                .extend((0..layer.out_c).map(|_| if rng.uniform() < 0.5 { 0.0 } else { 2.0 }));
        }
    }
    let masks = gpu.upload(&mask_table)?;
    drop(mask_table);

    let gates = gpu.zeros(c.max_decoder_steps)?;
    let steps = c.max_decoder_steps;

    // The frame's temporaries. Every one is written whole by the launch that
    // produces it before the launch that consumes it, which is what lets
    // them be uninitialised and reused.
    // SAFETY: as above.
    let last = w.prenet.len().saturating_sub(1);
    let pre = w.prenet[..last]
        .iter()
        .map(|l| unsafe { gpu.uninit(l.out_c) })
        .collect::<Result<Vec<_>, _>>()?;
    let gi_att = unsafe { gpu.uninit(4 * c.attention_rnn_dim) }?;
    let gi_dec = unsafe { gpu.uninit(4 * c.decoder_rnn_dim) }?;
    let loc = unsafe { gpu.uninit(c.location_filters * t) }?;
    let query = unsafe { gpu.uninit(c.attention_dim) }?;
    let score = unsafe { gpu.uninit(t) }?;

    // Everything a frame reads and writes, in one place so the frame can be
    // a function of it: the buffers persist and the frame index lives on the
    // device, which is what lets the frame be recorded once and replayed.
    let mut st = FrameState {
        att,
        dec,
        weights_cat,
        out,
        gates,
        cell_in,
        pre,
        gi_att,
        gi_dec,
        loc,
        query,
        score,
        next,
    };

    // One decoder frame, sixteen launches, and no allocation: every
    // temporary is a field of `st`, written whole each frame.
    //
    // Two things about its shape are measured rather than chosen, and both
    // are in docs/BENCHMARKS.md. The frame was recorded as a CUDA graph and
    // replayed, and that bought nothing: with its allocations it replayed
    // at the cost of issuing it, and without them the card still left
    // three to eight microseconds between kernels that finish in one or
    // two, exactly as it does when they are issued one by one. So what a
    // frame costs is its launch count, and every buffer here is laid out so
    // that nothing has to be concatenated.
    let step = |gpu: &Gpu,
                st: &mut FrameState,
                frame: usize,
                clock: &mut Clock|
     -> Result<(), TacoError> {
        let at = clock.start();
        // The single-row projections are `gemv_into`: one row is the `gemv`
        // kernel, a warp per output column and a shuffle reduction, where
        // `linear` gives one *thread* per output element. The gate is the
        // extreme case: `n` of one is one thread walking 1536 weights while
        // the rest of the card idles.
        //
        // The prenet reads the last frame where `taco_emit` left it and its
        // last layer lands at the head of `cell_in`, ahead of the context.
        let mut base = 0usize;
        for (i, layer) in w.prenet.iter().enumerate() {
            let (done, rest) = st.pre.split_at_mut(i);
            let x = if i == 0 { &st.next } else { &done[i - 1] };
            let y = if i == last {
                &mut st.cell_in
            } else {
                &mut rest[0]
            };
            gpu.gemv_into(
                x,
                Operand::F32(&layer.w),
                None,
                layer.in_c,
                layer.out_c,
                false,
                OutLayout::Row,
                y,
            )?;
            gpu.relu_mask(y, &masks, frame * mask_w + base, layer.out_c)?;
            base += layer.out_c;
        }
        if w.prenet.is_empty() {
            gpu.copy_into(&mut st.cell_in, &st.next, 0, mel)?;
        }
        clock.stop(gpu, "  prenet", at)?;

        let at = clock.start();
        gpu.gemv_into(
            &st.cell_in,
            w.attention_rnn.w_ih.operand(),
            Some(&w.attention_rnn.b_ih),
            c.prenet_dim + e,
            4 * c.attention_rnn_dim,
            false,
            OutLayout::Row,
            &mut st.gi_att,
        )?;
        step_cell(gpu, &w.attention_rnn, &st.gi_att, &mut st.att)?;
        clock.stop(gpu, "  attention_rnn", at)?;

        let at = clock.start();
        // Location-sensitive attention.
        gpu.conv1d_into(
            &st.weights_cat,
            w.location_conv.w.full(),
            None,
            2,
            t,
            c.location_filters,
            c.location_kernel,
            lpad,
            lpad,
            1,
            &mut st.loc,
        )?;
        gpu.gemv_into(
            &st.att.h,
            Operand::F32(&w.query.w),
            None,
            c.attention_rnn_dim,
            c.attention_dim,
            false,
            OutLayout::Row,
            &mut st.query,
        )?;
        // The energies, the scores, the softmax, the context and the running
        // weights - the query riding in as the bias of the energies - in two
        // launches. See `Gpu::taco_attention`; it was seven launches and a
        // transpose, each mostly its own floor at these sizes. The context
        // lands after the prenet output, after the attention cell's state
        // and after the decoder cell's state, which is where the three
        // projections that read it want it.
        gpu.taco_attention(
            &st.loc,
            &w.location_dense.w,
            &st.query,
            &processed,
            &w.v.w,
            &memory,
            t,
            c.location_filters,
            c.attention_dim,
            e,
            &mut st.score,
            &mut st.weights_cat,
            &mut st.cell_in,
            c.prenet_dim,
            [
                Some((&mut st.att.h, c.attention_rnn_dim)),
                Some((&mut st.dec.h, c.decoder_rnn_dim)),
            ],
        )?;
        clock.stop(gpu, "  attention", at)?;

        let at = clock.start();
        gpu.gemv_into(
            &st.att.h,
            w.decoder_rnn.w_ih.operand(),
            Some(&w.decoder_rnn.b_ih),
            c.attention_rnn_dim + e,
            4 * c.decoder_rnn_dim,
            false,
            OutLayout::Row,
            &mut st.gi_dec,
        )?;
        step_cell(gpu, &w.decoder_rnn, &st.gi_dec, &mut st.dec)?;
        clock.stop(gpu, "  decoder_rnn", at)?;

        let at = clock.start();
        // The projection and the gate as one stacked mat-vec: `[mel + 1]`,
        // the frame then its logit. `taco_emit` places the frame at its row
        // of the output and the logit at its position of the gate buffer;
        // the frame stays in `next` for the next step's prenet.
        gpu.gemv_into(
            &st.dec.h,
            Operand::F32(&w.proj_gate.w),
            w.proj_gate.bias.as_ref(),
            dhac,
            mel + 1,
            false,
            OutLayout::Row,
            &mut st.next,
        )?;
        gpu.taco_emit(&mut st.out, &mut st.gates, frame, &st.next, mel)?;
        clock.stop(gpu, "  projection", at)?;
        Ok(())
    };

    let mut frames = 0usize;
    let mut checked = 0usize;
    let mut stopped = false;
    while frames < steps {
        step(gpu, &mut st, frames, clock)?;
        frames += 1;

        // The stop gate, read in batches: a download is a synchronisation
        // and the loop is otherwise free of them. Reading every eighth frame
        // decodes at most seven frames past the stop, which the frame count
        // then discards; the frames themselves are unaffected.
        if frames.is_multiple_of(lookahead) || frames == steps {
            let at = clock.start();
            let logits = gpu.download(&st.gates)?;
            clock.stop(gpu, "  gate sync", at)?;
            if let Some(i) = logits[checked..frames].iter().position(|&g| g > gate_logit) {
                frames = checked + i + 1;
                stopped = true;
                break;
            }
            checked = frames;
        }
    }
    let out = st.out;

    if !stopped {
        tracing::warn!(frames, "the decoder hit its step limit without a stop");
    }

    clock.steps = frames;

    let at = clock.start();
    // `[frames, mel]` to the postnet's `[mel, frames]`.
    let flat = gpu.copy_range(&out, 0, frames * mel)?;
    let base = gpu.transpose(&flat, frames, mel)?;

    let ppad = c.postnet_kernel / 2;
    // An explicit copy rather than a clone: the postnet chain overwrites its
    // input, and `base` is still needed for the residual at the end.
    let mut y = gpu.copy_range(&base, 0, mel * frames)?;
    let last = w.postnet.len() - 1;
    for (i, conv) in w.postnet.iter().enumerate() {
        let (mut z, _) = gpu.conv1d(
            &y,
            conv.w.full(),
            Some(&conv.bias),
            conv.in_ch,
            frames,
            conv.out_ch,
            conv.k,
            ppad,
            ppad,
            1,
        )?;
        // Every convolution but the last is squashed; the last is the residual
        // itself and is added raw.
        if i != last {
            gpu.tanh(&mut z, conv.out_ch * frames)?;
        }
        y = z;
    }
    let mut data = base;
    gpu.add_inplace(&mut data, &y, mel * frames)?;
    clock.stop(gpu, "postnet", at)?;

    Ok(Mel {
        data,
        frames,
        stopped,
    })
}
