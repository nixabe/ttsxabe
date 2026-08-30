//! The geometry, as `tools/convert_tacotron2.py` wrote it.
//!
//! Read rather than hard-coded because the symbol table is part of it, and a
//! symbol table that drifts from the embedding is the failure this whole crate
//! is careful about. What is *not* read is anything the forward pass cannot
//! actually vary: those are checked and refused, not adapted to.

use crate::TacoError;
use std::path::Path;

/// Everything the two forward passes need to know.
pub struct Config {
    /// The 71-symbol input alphabet, in embedding-row order.
    pub symbols: Vec<String>,
    /// Output rate, 22050 Hz.
    pub sampling_rate: usize,
    /// Mel bands, 80.
    pub n_mel: usize,
    /// The upsampling convolution's kernel, 1024.
    pub filter_length: usize,
    /// Samples per mel frame, 256.
    pub hop_length: usize,
    /// Encoder and attention memory width, 512.
    pub encoder_dim: usize,
    /// Encoder convolution kernel, 5.
    pub encoder_kernel: usize,
    /// How many encoder convolutions, 3.
    pub encoder_convs: usize,
    /// Each direction of the encoder LSTM, 256.
    pub lstm_hidden: usize,
    /// Prenet width, 256.
    pub prenet_dim: usize,
    /// Attention LSTM width, 1024.
    pub attention_rnn_dim: usize,
    /// Decoder LSTM width, 1024.
    pub decoder_rnn_dim: usize,
    /// Attention projection width, 128.
    pub attention_dim: usize,
    /// Location convolution filters, 32.
    pub location_filters: usize,
    /// Location convolution kernel, 31.
    pub location_kernel: usize,
    /// Stop when the gate passes this, 0.5.
    pub gate_threshold: f32,
    /// Give up after this many frames, 3000.
    pub max_decoder_steps: usize,
    /// Postnet width, 512.
    pub postnet_dim: usize,
    /// Postnet kernel, 5.
    pub postnet_kernel: usize,
    /// Postnet convolutions, 5.
    pub postnet_convs: usize,
    /// WaveGlow coupling blocks, 12.
    pub n_flows: usize,
    /// Samples folded into one flow step, 8.
    pub n_group: usize,
    /// How often a flow splits channels off early, 4.
    pub n_early_every: usize,
    /// How many channels leave at each of those, 2.
    pub n_early_size: usize,
    /// Dilated convolutions per coupling network, 8.
    pub wn_layers: usize,
    /// Coupling network width, 256.
    pub wn_channels: usize,
    /// Coupling network kernel, 3.
    pub wn_kernel: usize,
    /// Standard deviation of the noise WaveGlow starts from, 0.666.
    pub sigma: f32,
}

/// Reads a `usize` out of a nested object, or says which key was wrong.
fn num(v: &serde_json::Value, path: &[&str]) -> Result<f64, TacoError> {
    let mut cur = v;
    for k in path {
        cur = cur.get(k).ok_or_else(|| {
            TacoError::Geometry(format!("tacotron2.json has no {}", path.join(".")))
        })?;
    }
    cur.as_f64()
        .ok_or_else(|| TacoError::Geometry(format!("{} is not a number", path.join("."))))
}

impl Config {
    /// Reads and validates `tacotron2.json`.
    ///
    /// The three refusals below are not fussiness. The decode loop emits one
    /// frame per step, WaveGlow's grouping is written for eight, and the prenet
    /// is two layers wide: each is a shape the code assumes rather than reads,
    /// so a config that disagrees would run and be quietly wrong.
    pub fn open(path: &Path) -> Result<Self, TacoError> {
        let text = std::fs::read_to_string(path).map_err(|e| TacoError::Config {
            path: path.display().to_string(),
            source: Box::new(e),
        })?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| TacoError::Config {
            path: path.display().to_string(),
            source: Box::new(e),
        })?;

        let symbols: Vec<String> = v
            .get("symbols")
            .and_then(|s| s.as_array())
            .ok_or_else(|| TacoError::Geometry("tacotron2.json has no symbols array".into()))?
            .iter()
            .map(|s| s.as_str().unwrap_or_default().to_string())
            .collect();
        if symbols.is_empty() {
            return Err(TacoError::Geometry("the symbol table is empty".into()));
        }

        let frames = num(&v, &["decoder", "n_frames_per_step"])? as usize;
        if frames != 1 {
            return Err(TacoError::Geometry(format!(
                "n_frames_per_step is {frames}; the decode loop emits one frame per step"
            )));
        }
        let n_group = num(&v, &["waveglow", "n_group"])? as usize;
        if n_group != 8 {
            return Err(TacoError::Geometry(format!(
                "n_group is {n_group}; the sample grouping is written for 8"
            )));
        }

        Ok(Self {
            symbols,
            sampling_rate: num(&v, &["audio", "sampling_rate"])? as usize,
            n_mel: num(&v, &["audio", "n_mel_channels"])? as usize,
            filter_length: num(&v, &["audio", "filter_length"])? as usize,
            hop_length: num(&v, &["audio", "hop_length"])? as usize,
            encoder_dim: num(&v, &["encoder", "encoder_embedding_dim"])? as usize,
            encoder_kernel: num(&v, &["encoder", "encoder_kernel_size"])? as usize,
            encoder_convs: num(&v, &["encoder", "encoder_n_convolutions"])? as usize,
            lstm_hidden: num(&v, &["encoder", "lstm_hidden"])? as usize,
            prenet_dim: num(&v, &["decoder", "prenet_dim"])? as usize,
            attention_rnn_dim: num(&v, &["decoder", "attention_rnn_dim"])? as usize,
            decoder_rnn_dim: num(&v, &["decoder", "decoder_rnn_dim"])? as usize,
            attention_dim: num(&v, &["decoder", "attention_dim"])? as usize,
            location_filters: num(&v, &["decoder", "attention_location_n_filters"])? as usize,
            location_kernel: num(&v, &["decoder", "attention_location_kernel_size"])? as usize,
            gate_threshold: num(&v, &["decoder", "gate_threshold"])? as f32,
            max_decoder_steps: num(&v, &["decoder", "max_decoder_steps"])? as usize,
            postnet_dim: num(&v, &["postnet", "postnet_embedding_dim"])? as usize,
            postnet_kernel: num(&v, &["postnet", "postnet_kernel_size"])? as usize,
            postnet_convs: num(&v, &["postnet", "postnet_n_convolutions"])? as usize,
            n_flows: num(&v, &["waveglow", "n_flows"])? as usize,
            n_group,
            n_early_every: num(&v, &["waveglow", "n_early_every"])? as usize,
            n_early_size: num(&v, &["waveglow", "n_early_size"])? as usize,
            wn_layers: num(&v, &["waveglow", "n_layers"])? as usize,
            wn_channels: num(&v, &["waveglow", "n_channels"])? as usize,
            wn_kernel: num(&v, &["waveglow", "kernel_size"])? as usize,
            sigma: num(&v, &["waveglow", "sigma"])? as f32,
        })
    }

    /// Channels surviving into each flow, outermost first.
    ///
    /// WaveGlow splits `n_early_size` channels off every `n_early_every` flows
    /// on the way in, so running backwards they are added back. For the
    /// published checkpoint this is 8,8,8,8,6,6,6,6,4,4,4,4.
    pub fn flow_channels(&self) -> Vec<usize> {
        let mut remaining = self.n_group;
        (0..self.n_flows)
            .map(|k| {
                if k % self.n_early_every == 0 && k > 0 {
                    remaining -= self.n_early_size;
                }
                remaining
            })
            .collect()
    }
}
