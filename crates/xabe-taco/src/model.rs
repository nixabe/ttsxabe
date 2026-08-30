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
use xabe_cuda::{CudaSlice, Gpu};

/// An LSTM's carried state.
pub(crate) struct Cell {
    c: CudaSlice<f32>,
    h: CudaSlice<f32>,
}

impl Cell {
    fn zeros(gpu: &Gpu, hidden: usize) -> Result<Self, TacoError> {
        Ok(Self {
            c: gpu.zeros(hidden)?,
            h: gpu.zeros(hidden)?,
        })
    }
}

/// Advances one LSTM step. `gi` is the input side, bias already in it.
fn step(gpu: &Gpu, w: &Lstm, gi: &CudaSlice<f32>, st: &mut Cell) -> Result<(), TacoError> {
    // `gemm` rather than `linear`: with one row it dispatches to the `gemv`
    // kernel, which keeps f32 end to end - only the tiled kernel stages f16 -
    // so this is the faster path at identical precision.
    let gh = gpu.gemm(&st.h, &w.w_hh, Some(&w.b_hh), 1, w.hidden, 4 * w.hidden)?;
    gpu.lstm_gates(gi, &gh, &mut st.c, &mut st.h, w.hidden)?;
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
        let gi_all = gpu.linear(&seq, &lstm.w_ih, Some(&lstm.b_ih), t, e, 4 * h)?;
        let mut st = Cell::zeros(gpu, h)?;
        for i in 0..t {
            let at = if offset == 0 { i } else { t - 1 - i };
            let gi = gpu.copy_range(&gi_all, at * 4 * h, 4 * h)?;
            step(gpu, lstm, &gi, &mut st)?;
            // Written straight into the concatenated output, so the forward and
            // backward passes never need a buffer of their own.
            gpu.copy_into(&mut memory, &st.h, at * 2 * h + offset, h)?;
        }
    }
    Ok(memory)
}

/// Runs the encoder and then the decode loop.
pub(crate) fn synthesize(
    gpu: &Gpu,
    w: &Taco2,
    c: &Config,
    ids: &[i64],
    rng: &mut Rng,
    clock: &mut Clock,
) -> Result<Mel, TacoError> {
    let (e, mel, t) = (c.encoder_dim, c.n_mel, ids.len());
    let at = clock.start();
    let memory = encode(gpu, w, c, ids)?;
    clock.stop(gpu, "encoder", at)?;

    let processed = gpu.linear(&memory, &w.memory.w, None, t, e, c.attention_dim)?;
    let memory_t = gpu.transpose(&memory, t, e)?;

    // Decoder state. The go frame is zeros, and so is everything else.
    let mut att = Cell::zeros(gpu, c.attention_rnn_dim)?;
    let mut dec = Cell::zeros(gpu, c.decoder_rnn_dim)?;
    let mut context = gpu.zeros(e)?;
    let mut weights_cat = gpu.zeros(2 * t)?;
    let mut frame = gpu.zeros(mel)?;

    let mut out = gpu.zeros(c.max_decoder_steps * mel)?;
    let dhac = c.decoder_rnn_dim + e;
    let lpad = c.location_kernel / 2;
    // `sigmoid(x) > p` without the sigmoid.
    let gate_logit = (c.gate_threshold / (1.0 - c.gate_threshold)).ln();

    let mut frames = 0usize;
    let mut stopped = false;
    while frames < c.max_decoder_steps {
        let at = clock.start();
        // Prenet, dropout included.
        let mut p = frame;
        for layer in &w.prenet {
            let mut y = gpu.gemm(&p, &layer.w, None, 1, layer.in_c, layer.out_c)?;
            gpu.relu(&mut y, layer.out_c)?;
            let mask: Vec<f32> = (0..layer.out_c)
                .map(|_| if rng.uniform() < 0.5 { 0.0 } else { 2.0 })
                .collect();
            let m = gpu.upload(&mask)?;
            gpu.mul_inplace(&mut y, &m, layer.out_c)?;
            p = y;
        }

        clock.stop(gpu, "  prenet", at)?;

        let at = clock.start();
        let mut cell_in = gpu.zeros(c.prenet_dim + e)?;
        gpu.copy_into(&mut cell_in, &p, 0, c.prenet_dim)?;
        gpu.copy_into(&mut cell_in, &context, c.prenet_dim, e)?;
        let gi = gpu.gemm(
            &cell_in,
            &w.attention_rnn.w_ih,
            Some(&w.attention_rnn.b_ih),
            1,
            c.prenet_dim + e,
            4 * c.attention_rnn_dim,
        )?;
        step(gpu, &w.attention_rnn, &gi, &mut att)?;
        clock.stop(gpu, "  attention_rnn", at)?;

        let at = clock.start();
        // Location-sensitive attention.
        let (loc, _) = gpu.conv1d(
            &weights_cat,
            w.location_conv.w.full(),
            None,
            2,
            t,
            c.location_filters,
            c.location_kernel,
            lpad,
            lpad,
            1,
        )?;
        let loc_t = gpu.transpose(&loc, c.location_filters, t)?;
        // `gemm` and not `linear` for the single-row projections. Both keep
        // f32 - one row dispatches to `gemv`, which is a warp per output column
        // and a shuffle reduction - where `linear` gives one *thread* per output
        // element. The gate is the extreme case: `n` of one is one thread
        // walking 1536 weights while the rest of the card idles.
        let query = gpu.gemm(
            &att.h,
            &w.query.w,
            None,
            1,
            c.attention_rnn_dim,
            c.attention_dim,
        )?;
        // The query rides in as the bias: one `[128]` vector added to every row.
        let mut energies = gpu.linear(
            &loc_t,
            &w.location_dense.w,
            Some(&query),
            t,
            c.location_filters,
            c.attention_dim,
        )?;
        gpu.add_inplace(&mut energies, &processed, t * c.attention_dim)?;
        gpu.tanh(&mut energies, t * c.attention_dim)?;
        let mut alignment = gpu.linear(&energies, &w.v.w, None, t, c.attention_dim, 1)?;
        gpu.softmax_rows(&mut alignment, 1, t)?;

        context = gpu.gemm(&alignment, &memory_t, None, 1, t, e)?;

        // The cumulative row grows by this step's weights; the current row is
        // replaced by them. Order matters: the sum includes the new weights.
        let mut cum = gpu.copy_range(&weights_cat, t, t)?;
        gpu.add_inplace(&mut cum, &alignment, t)?;
        gpu.copy_into(&mut weights_cat, &alignment, 0, t)?;
        gpu.copy_into(&mut weights_cat, &cum, t, t)?;

        clock.stop(gpu, "  attention", at)?;

        let at = clock.start();
        let mut dec_in = gpu.zeros(c.attention_rnn_dim + e)?;
        gpu.copy_into(&mut dec_in, &att.h, 0, c.attention_rnn_dim)?;
        gpu.copy_into(&mut dec_in, &context, c.attention_rnn_dim, e)?;
        let gi = gpu.gemm(
            &dec_in,
            &w.decoder_rnn.w_ih,
            Some(&w.decoder_rnn.b_ih),
            1,
            c.attention_rnn_dim + e,
            4 * c.decoder_rnn_dim,
        )?;
        step(gpu, &w.decoder_rnn, &gi, &mut dec)?;

        clock.stop(gpu, "  decoder_rnn", at)?;

        let at = clock.start();
        let mut both = gpu.zeros(dhac)?;
        gpu.copy_into(&mut both, &dec.h, 0, c.decoder_rnn_dim)?;
        gpu.copy_into(&mut both, &context, c.decoder_rnn_dim, e)?;

        let next = gpu.gemm(
            &both,
            &w.projection.w,
            w.projection.bias.as_ref(),
            1,
            dhac,
            mel,
        )?;
        let gate = gpu.gemm(&both, &w.gate.w, w.gate.bias.as_ref(), 1, dhac, 1)?;

        gpu.copy_into(&mut out, &next, frames * mel, mel)?;
        frames += 1;
        clock.stop(gpu, "  projection", at)?;

        // The one host round trip per step, and the reason the loop is a loop.
        let at = clock.start();
        let stop = gpu.download(&gate)?[0] > gate_logit;
        clock.stop(gpu, "  gate sync", at)?;
        if stop {
            stopped = true;
            break;
        }
        frame = next;
    }

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
