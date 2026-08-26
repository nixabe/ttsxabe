//! Binds the real mms-tts-nan checkpoint against its own config.
//!
//! Skips loudly when the checkpoint is absent. Set `XABE_TTS_MODEL`, or let it
//! find the HuggingFace cache copy.

use xabe_st::StFile;
use xabe_vits::{VitsConfig, VitsWeights};

/// Locates the snapshot directory holding both the weights and the config.
fn find_snapshot() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("XABE_TTS_MODEL") {
        return std::path::PathBuf::from(p).parent().map(Into::into);
    }
    let home = std::env::var("HOME").ok()?;
    let root = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--facebook--mms-tts-nan/snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    snap.join("model.safetensors").is_file().then_some(snap)
}

fn load() -> Option<(StFile, VitsConfig)> {
    let snap = find_snapshot()?;
    let f = StFile::open(snap.join("model.safetensors")).expect("open checkpoint");
    let cfg = VitsConfig::from_json_path(snap.join("config.json")).expect("read config");
    Some((f, cfg))
}

#[test]
fn config_matches_the_published_geometry() {
    let Some((_, cfg)) = load() else {
        eprintln!("SKIP: mms-tts-nan not found; set XABE_TTS_MODEL");
        return;
    };
    assert_eq!(cfg.hidden_size, 192);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert_eq!(cfg.num_attention_heads, 2);
    assert_eq!(cfg.head_dim(), 96);
    assert_eq!(cfg.vocab_size, 48);
    assert_eq!(cfg.sampling_rate, 16_000);
    assert_eq!(cfg.flow_size, 192);
    assert_eq!(cfg.flow_half(), 96);
    assert_eq!(cfg.rel_window(), 9);
    assert_eq!(cfg.upsample_rates, vec![8, 8, 2, 2]);
    assert_eq!(cfg.hop_length(), 256, "256 samples per frame at 16 kHz");
    assert_eq!(cfg.num_resblocks(), 12);
    assert_eq!(cfg.upsample_out_channels(0), 256);
    assert_eq!(cfg.upsample_out_channels(3), 32);
}

#[test]
fn binds_every_inference_tensor() {
    let Some((f, cfg)) = load() else {
        eprintln!("SKIP: mms-tts-nan not found; set XABE_TTS_MODEL");
        return;
    };
    let w = VitsWeights::load(&f, &cfg).expect("bind weights");

    assert_eq!(w.text_encoder.layers.len(), 6);
    assert_eq!(w.text_encoder.embed.len(), 48 * 192);
    assert_eq!(w.text_encoder.project.out_ch, 384);
    assert_eq!(w.flow.len(), 4);
    assert_eq!(w.flow[0].wavenet.len(), 4);
    assert_eq!(w.decoder.upsampler.len(), 4);
    assert_eq!(w.decoder.resblocks.len(), 12);
    assert_eq!(w.decoder.conv_post.in_ch, 32);
    assert!(w.decoder.conv_post.bias.is_none(), "conv_post has no bias");

    // The duration predictor stores one affine plus `num_flows` splines.
    assert_eq!(
        w.duration_predictor.flows.len(),
        cfg.duration_predictor_num_flows + 1
    );
    assert_eq!(
        w.duration_predictor.post_flows.len(),
        cfg.duration_predictor_num_flows + 1
    );
}

#[test]
fn the_inference_subset_is_exactly_the_checkpoint_minus_the_posterior() {
    let Some((f, cfg)) = load() else {
        eprintln!("SKIP: mms-tts-nan not found; set XABE_TTS_MODEL");
        return;
    };
    VitsWeights::load(&f, &cfg).expect("bind weights");

    // Everything not under posterior_encoder must be reachable by the schema.
    // Counting is the cheap version of that claim: if the schema missed a
    // tensor, this number moves and the test says which side is short.
    let total = f.tensors().count();
    let posterior = f
        .tensors()
        .filter(|(n, _)| n.starts_with("posterior_encoder."))
        .count();
    assert_eq!(total, 762, "checkpoint tensor count");
    assert_eq!(posterior, 100, "posterior_encoder is training-only");
    assert_eq!(total - posterior, 662, "inference tensors");
}

#[test]
fn a_wrong_geometry_is_rejected_by_name() {
    let Some((f, mut cfg)) = load() else {
        eprintln!("SKIP: mms-tts-nan not found; set XABE_TTS_MODEL");
        return;
    };
    // Claim a wider model than the checkpoint holds. This must fail at load,
    // naming a tensor - not succeed and produce wrong audio later.
    cfg.hidden_size = 256;
    let err = VitsWeights::load(&f, &cfg).expect_err("a wrong width must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("text_encoder") || msg.contains("shape"),
        "the error should name the offending tensor, got: {msg}"
    );
}

#[test]
fn config_validation_rejects_impossible_geometry() {
    let base = r#"{
        "hidden_size": 192, "num_hidden_layers": 6, "num_attention_heads": 5,
        "ffn_dim": 768, "ffn_kernel_size": 3, "vocab_size": 48,
        "sampling_rate": 16000, "flow_size": 192, "prior_encoder_num_flows": 4,
        "prior_encoder_num_wavenet_layers": 4, "wavenet_kernel_size": 5,
        "wavenet_dilation_rate": 1, "upsample_rates": [8,8,2,2],
        "upsample_kernel_sizes": [16,16,4,4], "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
        "depth_separable_channels": 2, "duration_predictor_kernel_size": 3,
        "duration_predictor_num_flows": 4, "window_size": 4,
        "depth_separable_num_layers": 3, "duration_predictor_flow_bins": 10,
        "duration_predictor_tail_bound": 5.0,
        "noise_scale": 0.667, "noise_scale_duration": 0.8, "speaking_rate": 1.0,
        "leaky_relu_slope": 0.1, "use_stochastic_duration_prediction": true
    }"#;
    // 192 is not divisible by 5 heads.
    let err = VitsConfig::from_json_str(base).expect_err("head split must be rejected");
    assert!(err.to_string().contains("divisible"), "got: {err}");

    let ok = base.replace("\"num_attention_heads\": 5", "\"num_attention_heads\": 2");
    VitsConfig::from_json_str(&ok).expect("valid geometry");

    let det = ok.replace(
        "\"use_stochastic_duration_prediction\": true",
        "\"use_stochastic_duration_prediction\": false",
    );
    VitsConfig::from_json_str(&det).expect_err("deterministic duration is unsupported");
}

#[test]
fn the_schema_reads_every_inference_parameter() {
    let Some((f, cfg)) = load() else {
        eprintln!("SKIP: mms-tts-nan not found; set XABE_TTS_MODEL");
        return;
    };
    let w = VitsWeights::load(&f, &cfg).expect("bind weights");

    // A tensor the schema forgets to bind raises no error - it just never gets
    // read. Comparing parameter counts against the file's own inference subset
    // is the only way to notice, and it is what caught the decoder's fused
    // weight-norm being loaded as if it were unfused.
    let want: usize = f
        .tensors()
        .filter(|(n, _)| !n.starts_with("posterior_encoder."))
        .map(|(_, i)| i.numel())
        .sum();

    assert_eq!(
        w.total_elements(),
        want,
        "schema binds {} parameters, the inference subset holds {want}",
        w.total_elements()
    );
}
