//! Reads the real Coqui VITS checkpoint through the container.
//!
//! Skips loudly when the checkpoint is absent. Set `XABE_COQUI_MODEL` to the
//! model directory, or let it find `models/tts/coqui-vits-suisiann`.

use xabe_pt::{Dtype, PtFile};

/// Locates the directory holding `best_model.pth`.
fn find_model() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("XABE_COQUI_MODEL") {
        let p = std::path::PathBuf::from(p);
        return p.join("best_model.pth").is_file().then_some(p);
    }
    let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/coqui-vits-suisiann");
    local.join("best_model.pth").is_file().then_some(local)
}

fn open() -> Option<PtFile> {
    let dir = find_model()?;
    Some(PtFile::open_section(dir.join("best_model.pth"), "model").expect("open checkpoint"))
}

#[test]
fn reads_the_whole_state_dict() {
    let Some(f) = open() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    // The trainer saved the generator, the discriminator and the posterior
    // encoder together; inference reads a subset, but the container binds all
    // of it, so the count is the whole `model` section.
    assert_eq!(f.len(), 949);
    let elements: usize = f.tensors().map(|(_, i)| i.numel()).sum();
    assert_eq!(elements, 83_060_332);
}

#[test]
fn every_tensor_is_f32_and_borrowable() {
    let Some(f) = open() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    for (name, info) in f.tensors() {
        assert_eq!(info.dtype, Dtype::F32, "{name} is not f32");
        let data = f.tensor(name).expect("borrow tensor");
        assert_eq!(data.len(), info.numel(), "{name} is the wrong length");
    }
}

#[test]
fn shapes_match_the_published_geometry() {
    let Some(f) = open() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    for (name, shape) in [
        ("text_encoder.emb.weight", vec![137, 192]),
        (
            "text_encoder.encoder.attn_layers.0.emb_rel_k",
            vec![1, 9, 96],
        ),
        ("text_encoder.proj.weight", vec![384, 192, 1]),
        (
            "flow.flows.0.enc.in_layers.0.parametrizations.weight.original1",
            vec![384, 192, 5],
        ),
        ("duration_predictor.flows.1.proj.weight", vec![29, 192, 1]),
        (
            "waveform_decoder.ups.0.parametrizations.weight.original1",
            vec![512, 256, 16],
        ),
        ("waveform_decoder.conv_post.weight", vec![1, 32, 7]),
    ] {
        let info = f.info(name).unwrap_or_else(|| panic!("{name} is missing"));
        assert_eq!(info.shape, shape, "{name}");
        f.tensor_shaped(name, &shape).expect("bind by shape");
    }
}

#[test]
fn a_wrong_shape_names_the_tensor() {
    let Some(f) = open() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    let err = f
        .tensor_shaped("text_encoder.emb.weight", &[136, 192])
        .expect_err("a wrong shape must be refused");
    let text = err.to_string();
    assert!(text.contains("text_encoder.emb.weight"), "{text}");
    assert!(text.contains("137"), "{text}");
}

#[test]
fn the_optimizer_section_is_a_different_state_dict() {
    let Some(dir) = find_model() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    // The root object holds `model` beside `optimizer`, `scheduler` and the
    // run's config. Naming a section that is not a mapping of tensors has to
    // fail rather than bind half of it.
    let err = PtFile::open_section(dir.join("best_model.pth"), "not_a_section")
        .expect_err("an absent section must be refused");
    assert!(err.to_string().contains("not_a_section"), "{err}");
}

/// The values `torch.load` reads for the same six tensors, in the same order:
/// first element, middle element, last element, and the sum in double
/// precision.
///
/// Captured with the reference rather than described - see `docs/ORACLE.md`.
/// The first three are exact equality: the container either hands back the
/// checkpoint's own bytes or it does not. The sum is compared loosely because
/// it is accumulated in a different order here than in torch.
const REFERENCE: &[(&str, [f64; 4])] = &[
    (
        "text_encoder.emb.weight",
        [
            -0.028_603_224_083_781_242,
            0.095_064_476_132_392_88,
            -0.032_779_544_591_903_687,
            17.313_644_512_154_042,
        ],
    ),
    (
        "text_encoder.proj.bias",
        [
            -0.036_614_589_393_138_885,
            2.594_052_602_944_43e-5,
            -0.056_096_229_702_234_27,
            -3.025_438_518_781_811_6,
        ],
    ),
    (
        "flow.flows.0.enc.in_layers.0.parametrizations.weight.original1",
        [
            0.023_888_848_721_981_05,
            -0.065_031_014_382_839_2,
            -0.007_298_257_201_910_019,
            10.634_832_122_984_122,
        ],
    ),
    (
        "duration_predictor.flows.1.proj.weight",
        [
            -0.012_148_500_420_153_141,
            -0.006_349_370_814_859_867,
            0.010_509_991_087_019_444,
            -1.365_279_611_793_539_5,
        ],
    ),
    (
        "waveform_decoder.ups.0.parametrizations.weight.original1",
        [
            0.018_435_414_880_514_145,
            -0.010_817_266_069_352_627,
            0.069_382_183_253_765_1,
            -6_183.214_919_364_389,
        ],
    ),
    (
        "waveform_decoder.conv_post.weight",
        [
            0.065_111_197_531_223_3,
            -0.015_742_883_086_204_53,
            0.025_591_218_844_056_13,
            0.604_062_542_501_196_7,
        ],
    ),
];

#[test]
fn values_match_torch_load_exactly() {
    let Some(f) = open() else {
        eprintln!("SKIP: coqui-vits-suisiann not found; set XABE_COQUI_MODEL");
        return;
    };
    for (name, [first, middle, last, sum]) in REFERENCE {
        let t = f.tensor(name).expect("borrow tensor");
        assert_eq!(f64::from(t[0]), *first, "{name} first element");
        assert_eq!(f64::from(t[t.len() / 2]), *middle, "{name} middle element");
        assert_eq!(f64::from(t[t.len() - 1]), *last, "{name} last element");

        let total: f64 = t.iter().map(|v| f64::from(*v)).sum();
        assert!(
            (total - sum).abs() < 1e-6 * sum.abs().max(1.0),
            "{name} sums to {total}, torch says {sum}",
        );
    }
}
