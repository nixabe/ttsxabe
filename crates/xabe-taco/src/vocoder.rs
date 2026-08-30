//! WaveGlow: mel in, waveform out, running the flow backwards.
//!
//! A normalising flow is trained forwards - audio to noise - and used
//! backwards. So the only direction implemented here is the inverse: start from
//! Gaussian noise shaped like the output, and undo twelve couplings.
//!
//! # Everything happens in groups of eight
//!
//! The waveform is folded so that eight consecutive samples become eight
//! channels of one step, and the upsampled conditioning is folded to match:
//! `cond[c * 8 + j][step] = upsampled[c][step * 8 + j]`. That fold is the one
//! place an index error would be invisible - it produces audio either way -
//! so it is done as eighty explicit `[steps, 8]` transposes rather than by
//! reasoning about strides.
//!
//! # It is stochastic, and that is the model
//!
//! The output is a function of the noise as much as of the mel. Two calls with
//! different seeds are two different renderings of the same sentence, both
//! correct. Comparing against a captured reference means replaying the
//! reference's noise, not hoping the difference is small.
//!
//! # The denoiser is not here
//!
//! The reference script follows this with a bias-spectral-subtraction
//! post-filter at strength 0.01. It is not part of WaveGlow, it needs a
//! 1024-point STFT and its inverse, and at that strength it is a polish rather
//! than a fix - so it is left out, and left out visibly.

use crate::clock::Clock;
use crate::model::Rng;
use crate::weights::{Conv, Glow, Wn};
use crate::{Config, TacoError};
use xabe_cuda::{Batch, Operand};
use xabe_cuda::{CudaSlice, Gpu};

/// `x @ conv^T + bias`, taking the weight at whatever width it was stored.
fn matmul(
    gpu: &Gpu,
    x: &CudaSlice<f32>,
    conv: &Conv,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>, TacoError> {
    Ok(gpu.gemm_batched(
        Operand::F32(x),
        conv.w.operand(),
        Some(&conv.bias),
        Batch::single(m * n),
        m,
        k,
        n,
    )?)
}

/// One coupling network, entirely in `[steps, channels]`.
///
/// Every operation in here is a matmul, and a matmul wants its contracted axis
/// last - so the whole chain stays time-major and not one transpose is needed
/// between the start projection and the end one. That is the reason the
/// conditioning and the residual/skip weights are split at load: in this layout
/// their output slices are strides rather than ranges, and slicing a stride
/// would cost more than the split saves.
///
/// `audio_t` is `[steps, half]`, `cond_t` is `[steps, mel * group]`, and the
/// result is `[steps, 2 * half]`.
#[allow(clippy::too_many_arguments)]
fn coupling(
    gpu: &Gpu,
    wn: &Wn,
    c: &Config,
    audio_t: &CudaSlice<f32>,
    cond_t: &CudaSlice<f32>,
    half: usize,
    steps: usize,
    clock: &mut Clock,
) -> Result<CudaSlice<f32>, TacoError> {
    let ch = c.wn_channels;
    let at = clock.start();
    let mut a = matmul(gpu, audio_t, &wn.start, steps, half, ch)?;
    let mut skip = gpu.zeros(steps * ch)?;
    clock.stop(gpu, "    wn start", at)?;

    // Every layer's conditioning at once. It depends only on the mel, not on
    // the audio the loop below is transforming, so there is nothing to wait
    // for and one wide matmul beats `layers` narrow ones - see `Wn::cond`.
    let at = clock.start();
    let cond_all = matmul(gpu, cond_t, &wn.cond, steps, wn.cond.in_ch, wn.cond.out_ch)?;
    clock.stop(gpu, "    wn cond", at)?;

    for i in 0..c.wn_layers {
        // Dilation doubles per layer and the padding follows it, so the length
        // is unchanged. `im2col` gathers the window as `channel * k + tap`,
        // which is exactly how a `[out, in, k]` weight flattens.
        let at = clock.start();
        let dil = 1usize << i;
        let layer = &wn.in_layers[i];
        let (col, out_t) = gpu.im2col(&a, steps, ch, c.wn_kernel, 1, dil, dil)?;
        debug_assert_eq!(out_t, steps, "the dilated padding did not preserve length");
        let mut act = matmul(gpu, &col, layer, steps, ch * c.wn_kernel, 2 * ch)?;
        clock.stop(gpu, "    wn dilated", at)?;

        let at = clock.start();
        gpu.add_strided(
            &mut act,
            &cond_all,
            2 * ch,
            wn.cond.out_ch,
            i * 2 * ch,
            steps,
        )?;
        clock.stop(gpu, "    wn cond add", at)?;

        let at = clock.start();
        let gated = gpu.gated_activation_rows(&act, ch, steps)?;
        if let Some(res) = &wn.res[i] {
            let r = matmul(gpu, &gated, res, steps, ch, ch)?;
            gpu.add_inplace(&mut a, &r, steps * ch)?;
        }
        let sk = &wn.skip[i];
        let s = matmul(gpu, &gated, sk, steps, ch, ch)?;
        gpu.add_inplace(&mut skip, &s, steps * ch)?;
        clock.stop(gpu, "    wn res_skip", at)?;
    }

    let at = clock.start();
    let end = matmul(gpu, &skip, &wn.end, steps, ch, 2 * half)?;
    clock.stop(gpu, "    wn end", at)?;
    Ok(end)
}

/// Noise of `[rows, steps]`, scaled by sigma and uploaded.
fn noise(
    gpu: &Gpu,
    rng: &mut Rng,
    rows: usize,
    steps: usize,
    sigma: f32,
) -> Result<CudaSlice<f32>, TacoError> {
    let z: Vec<f32> = (0..rows * steps).map(|_| rng.normal() * sigma).collect();
    Ok(gpu.upload(&z)?)
}

/// Runs the vocoder. Returns the waveform, peak-normalised by the caller.
// Shapes and the clock are arguments rather than a descriptor, the same
// convention the rest of the workspace uses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn infer(
    gpu: &Gpu,
    w: &Glow,
    c: &Config,
    mel: &CudaSlice<f32>,
    frames: usize,
    sigma: f32,
    rng: &mut Rng,
    clock: &mut Clock,
) -> Result<Vec<f32>, TacoError> {
    let (n_mel, group) = (c.n_mel, c.n_group);

    let at = clock.start();
    // Mel to sample rate, then the reference's trim of the transposed
    // convolution's tail: `kernel - stride` samples that have no full support.
    let (up, up_t) = gpu.transposed_conv1d(
        mel,
        w.upsample.w.full(),
        Some(&w.upsample.bias),
        n_mel,
        frames,
        n_mel,
        c.filter_length,
        c.hop_length,
        0,
    )?;
    clock.stop(gpu, "upsample", at)?;
    let cut = c.filter_length - c.hop_length;
    let samples = up_t - cut;
    let steps = samples / group;

    // The fold, one mel channel at a time: `[steps, 8]` becomes `[8, steps]`
    // and lands as eight consecutive rows of the conditioning.
    let at = clock.start();
    let mut cond = gpu.zeros(n_mel * group * steps)?;
    for ch in 0..n_mel {
        let row = gpu.copy_range(&up, ch * up_t, samples)?;
        let folded = gpu.transpose(&row, steps, group)?;
        gpu.copy_into(&mut cond, &folded, ch * group * steps, group * steps)?;
    }

    clock.stop(gpu, "fold", at)?;

    // Once, not once per flow: every coupling network reads the same
    // conditioning and only the weights differ.
    let cond_t = gpu.transpose(&cond, n_mel * group, steps)?;

    let channels = c.flow_channels();
    let mut width = *channels.last().expect("at least one flow");
    let mut audio = noise(gpu, rng, width, steps, sigma)?;

    for k in (0..c.n_flows).rev() {
        let flow = &w.flows[k];
        let half = flow.channels / 2;
        let a0 = gpu.copy_range(&audio, 0, half * steps)?;
        let a1 = gpu.copy_range(&audio, half * steps, half * steps)?;

        // The only transposes left in the vocoder: two per flow, at the
        // boundary between the flow's `[channels, steps]` and the coupling
        // network's `[steps, channels]`.
        let a0t = gpu.transpose(&a0, half, steps)?;
        let st_t = coupling(gpu, &flow.wn, c, &a0t, &cond_t, half, steps, clock)?;
        let st = gpu.transpose(&st_t, steps, 2 * half)?;
        let at = clock.start();
        let fixed = gpu.coupling_inverse(&a1, &st, half, steps)?;

        let mut joined = gpu.zeros(flow.channels * steps)?;
        gpu.copy_into(&mut joined, &a0, 0, half * steps)?;
        gpu.copy_into(&mut joined, &fixed, half * steps, half * steps)?;

        // The mixing convolution, already inverted at load. A 1x1 convolution
        // rather than a matmul so the `[channels, steps]` layout is untouched.
        let (mixed, _) = gpu.conv1d(
            &joined,
            &flow.inv,
            None,
            flow.channels,
            steps,
            flow.channels,
            1,
            0,
            0,
            1,
        )?;
        audio = mixed;
        width = flow.channels;
        clock.stop(gpu, "  convinv", at)?;

        // Channels that left early on the way in rejoin here, as fresh noise.
        if k % c.n_early_every == 0 && k > 0 {
            let z = noise(gpu, rng, c.n_early_size, steps, sigma)?;
            let grown = width + c.n_early_size;
            let mut next = gpu.zeros(grown * steps)?;
            gpu.copy_into(&mut next, &z, 0, c.n_early_size * steps)?;
            gpu.copy_into(&mut next, &audio, c.n_early_size * steps, width * steps)?;
            audio = next;
            width = grown;
        }
    }

    // `[8, steps]` unfolds back to a waveform by transposing: sample
    // `step * 8 + j` is channel `j` of step `step`.
    let at = clock.start();
    let flat = gpu.transpose(&audio, width, steps)?;
    let out = gpu.download(&flat)?;
    clock.stop(gpu, "download", at)?;
    Ok(out)
}
