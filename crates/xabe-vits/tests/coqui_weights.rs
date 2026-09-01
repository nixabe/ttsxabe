//! Binds the real Coqui SuiSiann checkpoint against its own config.
//!
//! Skips loudly when the checkpoint is absent. Set `XABE_COQUI_MODEL`, or let
//! it find `models/tts/coqui-vits-suisiann`.

use xabe_pt::PtFile;
use xabe_vits::{CoquiConfig, CoquiTokenizer, VitsConfig, VitsWeights};

/// Locates the model directory.
fn find_model() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("XABE_COQUI_MODEL") {
        let p = std::path::PathBuf::from(p);
        return p.join("best_model.pth").is_file().then_some(p);
    }
    let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/coqui-vits-suisiann");
    local.join("best_model.pth").is_file().then_some(local)
}

fn load() -> Option<(PtFile, CoquiConfig, VitsConfig)> {
    let dir = find_model()?;
    let raw = CoquiConfig::from_json_path(dir.join("config.json")).expect("read config");
    let cfg = raw.to_vits().expect("convert geometry");
    let f = PtFile::open_section(dir.join("best_model.pth"), "model").expect("open checkpoint");
    Some((f, raw, cfg))
}

#[test]
fn config_matches_the_published_geometry() {
    let Some((_, raw, cfg)) = load() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    assert_eq!(raw.model, "vits");
    assert_eq!(raw.phoneme_language.as_deref(), Some("MinnanHokkien2"));
    assert!(raw.use_phonemes, "the model was trained on phonemes");
    assert!(raw.add_blank);

    assert_eq!(cfg.hidden_size, 192);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert_eq!(cfg.num_attention_heads, 2);
    assert_eq!(cfg.head_dim(), 96);
    assert_eq!(cfg.vocab_size, 137);
    // 22.05 kHz, against `mms-tts-nan`'s 16 kHz. Same 256 samples per frame,
    // so a frame is shorter in time here.
    assert_eq!(cfg.sampling_rate, 22_050);
    assert_eq!(cfg.flow_size, 192);
    assert_eq!(cfg.rel_window(), 9);
    assert_eq!(cfg.upsample_rates, vec![8, 8, 2, 2]);
    assert_eq!(cfg.hop_length(), 256);
    assert_eq!(cfg.num_resblocks(), 12);
    assert_eq!(cfg.upsample_out_channels(3), 32);
    assert_eq!(cfg.noise_scale, 0.667);
    assert_eq!(cfg.noise_scale_duration, 1.0);
    assert_eq!(cfg.speaking_rate, 1.0);
    assert!(cfg.use_stochastic_duration_prediction);
}

#[test]
fn binds_every_inference_tensor() {
    let Some((f, _, cfg)) = load() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    let w = VitsWeights::load_coqui(&f, &cfg).expect("bind weights");

    // The checkpoint holds 949 tensors and 83.1 M parameters. Removing the
    // posterior encoder, which encodes ground-truth spectrograms during
    // training, and the discriminator, which is only a training signal, leaves
    // 738 tensors and 29,075,184 parameters - and the schema must bind every
    // one of them. A tensor the schema forgets to read raises no error; it
    // simply never appears, so counting is the only way to notice.
    assert_eq!(w.total_elements(), 29_075_184);
    assert_eq!(w.text_encoder.layers.len(), 6);
    assert_eq!(w.flow.len(), 4);
    assert_eq!(w.flow[0].wavenet.len(), 4);
    assert_eq!(w.duration_predictor.flows.len(), 5);
    assert_eq!(w.duration_predictor.post_flows.len(), 5);
    assert_eq!(w.decoder.upsampler.len(), 4);
    assert_eq!(w.decoder.resblocks.len(), 12);
}

#[test]
fn the_decoder_arrives_weight_normalised() {
    let Some((f, _, cfg)) = load() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    let w = VitsWeights::load_coqui(&f, &cfg).expect("bind weights");

    // This is the one place the two dialects differ by more than a name, so it
    // is asserted rather than assumed: the 🤗 export fused these and the Coqui
    // save does not.
    for up in &w.decoder.upsampler {
        assert!(
            matches!(up, xabe_vits::MaybeWn::Normalised(_)),
            "upsamplers are stored unfused in this checkpoint",
        );
    }
    // A transposed convolution's magnitude is per *input* channel, because the
    // kernel is stored `[in, out, k]` and weight norm keeps the first axis.
    let xabe_vits::MaybeWn::Normalised(first) = &w.decoder.upsampler[0] else {
        unreachable!("checked above");
    };
    assert_eq!(first.weight_g.len(), 512);
    assert_eq!(first.weight_v.len(), 512 * 256 * 16);
    assert_eq!(first.bias.len(), 256);

    // conv_pre and conv_post are built with weight norm off, so they are plain.
    assert!(w.decoder.conv_pre.bias.is_some());
    assert!(
        w.decoder.conv_post.bias.is_none(),
        "conv_post_bias=False in the reference",
    );
}

#[test]
fn the_vocabulary_is_the_reference_order() {
    let Some((_, raw, _)) = load() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    let vocab = raw.characters.vocab().expect("build vocab");
    assert_eq!(vocab.len(), 137);
    // The four specials come first, in this order, which is what puts the
    // blank at 3 rather than at 0.
    assert_eq!(vocab[0], "<PAD>");
    assert_eq!(vocab[1], "<EOS>");
    assert_eq!(vocab[2], "<BOS>");
    assert_eq!(vocab[3], "<BLNK>");
    assert_eq!(vocab[4], "a");
    // Punctuation is appended after the alphabet, not mixed into it, so the
    // space is the very last symbol.
    assert_eq!(vocab[136], " ");

    let tok = CoquiTokenizer::new(&raw).expect("build tokenizer");
    assert_eq!(tok.vocab_size(), 137);
    assert_eq!(tok.blank(), 3);
}

#[test]
fn encoding_intersperses_the_blank() {
    let Some((_, raw, _)) = load() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    let tok = CoquiTokenizer::new(&raw).expect("build tokenizer");

    // Three symbols become blank, a, blank, b, blank, ... - 2n + 1 ids.
    let ids = tok.encode("aba");
    assert_eq!(ids.len(), 7);
    assert_eq!(ids[0], 3);
    assert_eq!(ids[6], 3);
    assert_eq!(ids[1], ids[5], "the two a's are the same symbol");

    // Han text is outside the table entirely: it is silently dropped, which is
    // why a caller has to check for an empty result rather than trust it.
    assert!(tok.encode("你好").is_empty());
}
