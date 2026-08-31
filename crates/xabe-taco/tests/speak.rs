//! Tacotron2 + WaveGlow end to end, plus the text path that needs no device.
//!
//! # What this can and cannot assert
//!
//! Synthesis here is stochastic twice over - the prenet keeps its dropout at
//! inference and WaveGlow starts from noise - so there is no sample-for-sample
//! comparison to make against anything, including a second run of itself. What
//! is checkable without an oracle is arithmetic and finiteness: the waveform is
//! a whole number of mel frames long at 256 samples each, every sample is
//! finite and in range, the decoder stopped rather than ran out, and a fixed
//! seed reproduces a run exactly.
//!
//! A reference comparison needs the captured mel *and* the captured noise -
//! see the module header in `src/vocoder.rs` - and belongs in a golden test
//! once `tools/oracle` can capture this model.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn poj_becomes_the_alphabet_the_model_reads() {
    use xabe_taco::poj_to_tlpa;

    // The spellings that differ, and the tone marks that become digits.
    assert_eq!(poj_to_tlpa("góa"), "gua2");
    assert_eq!(poj_to_tlpa("lí hó"), "li2 ho2");
    assert_eq!(poj_to_tlpa("chhùi"), "tshui3");
    assert_eq!(poj_to_tlpa("chi̍t"), "tsit8");
    assert_eq!(poj_to_tlpa("Tâi-oân"), "tai5-uan5");
    assert_eq!(poj_to_tlpa("hoⁿ"), "honn1");
    // Unmarked and stop-final is tone 4, unmarked otherwise is tone 1.
    assert_eq!(poj_to_tlpa("sit"), "sit4");
    assert_eq!(poj_to_tlpa("si"), "si1");
    // `o` with the POJ dot becomes `oo`.
    assert_eq!(poj_to_tlpa("o\u{0358}"), "oo1");
    // Already numeric: left alone rather than converted twice.
    assert_eq!(
        poj_to_tlpa("gua2 si7 tai5-uan5-lang5"),
        "gua2 si7 tai5-uan5-lang5"
    );
    // Mixed: the decision is per syllable, so the POJ half is still converted.
    // A line-wide test passed the whole thing through and the tokeniser then
    // dropped every diacritic in it without a word.
    assert_eq!(
        poj_to_tlpa("gua2 tuà tī tâi-pak"),
        "gua2 tua3 ti7 tai5-pak4"
    );
}

#[test]
fn the_alphabet_drops_what_it_cannot_say() {
    use xabe_taco::Tokenizer;

    let symbols: Vec<String> = ["_", "-", " ", "a", "i", "l", "o", "h", "2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let tok = Tokenizer::new(&symbols);

    let (ids, dropped) = tok.encode("li2 ho2");
    assert_eq!(ids.len(), 7);
    assert_eq!(dropped, 0);

    // Han is not in the table and goes silently, which is the checkpoint's
    // behaviour and the reason `synthesize` refuses an empty sequence.
    let (ids, dropped) = tok.encode("你好");
    assert!(ids.is_empty());
    assert_eq!(dropped, 2);
}

#[test]
fn it_speaks() {
    let dir = root().join("models/tts/tacotron2-nan");
    if !dir.join("tacotron2.safetensors").is_file() {
        println!("SKIP: no models/tts/tacotron2-nan; run tools/convert_tacotron2.py");
        return;
    }
    let Some(dev) = std::env::var("XABE_TACO_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        println!("SKIP: set XABE_TACO_DEVICE=<free card>; see docs/TESTING.md");
        return;
    };

    let taco = xabe_taco::Taco::open(&dir, dev, None, 7).expect("open");
    assert_eq!(taco.sample_rate(), 22050);

    let audio = taco
        .synthesize("gua2 si7 tai5-uan5-lang5")
        .expect("synthesize");
    assert!(!audio.is_empty(), "produced no samples");

    // WaveGlow emits whole groups of eight and the upsampling is exactly one
    // hop per mel frame, so the length is a multiple of the hop. No tolerance:
    // this is arithmetic, and a fold done wrong lands somewhere else.
    assert_eq!(
        audio.len() % 256,
        0,
        "length {} is not whole frames",
        audio.len()
    );
    let seconds = audio.len() as f32 / 22050.0;
    assert!(
        (0.3..12.0).contains(&seconds),
        "{seconds:.2}s is not a plausible reading of five syllables"
    );

    assert!(
        audio.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
        "samples left [-1, 1] or went non-finite"
    );
    // Peak-normalised, so the loudest sample is at full scale by construction.
    let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        peak > 0.99,
        "peak {peak} - the normalisation did not happen"
    );

    // Not a constant: a flow that collapsed would still pass everything above.
    let mean = audio.iter().sum::<f32>() / audio.len() as f32;
    let var = audio.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / audio.len() as f32;
    assert!(var > 1e-4, "variance {var} - this is silence, not speech");
}

#[test]
fn a_seed_reproduces_a_run() {
    let dir = root().join("models/tts/tacotron2-nan");
    if !dir.join("tacotron2.safetensors").is_file() {
        println!("SKIP: no models/tts/tacotron2-nan; run tools/convert_tacotron2.py");
        return;
    }
    let Some(dev) = std::env::var("XABE_TACO_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        println!("SKIP: set XABE_TACO_DEVICE=<free card>; see docs/TESTING.md");
        return;
    };

    // Two synthesisers on the same seed, each speaking once: the dropout masks
    // and the vocoder noise are then the same stream, so the waveforms must
    // agree exactly. This is what makes the stochasticity a property rather
    // than an excuse.
    let a = xabe_taco::Taco::open(&dir, dev, None, 99).expect("open");
    let b = xabe_taco::Taco::open(&dir, dev, None, 99).expect("open");
    let x = a.synthesize("li2 ho2").expect("a");
    let y = b.synthesize("li2 ho2").expect("b");
    assert_eq!(x.len(), y.len(), "same seed, different lengths");
    assert!(
        x.iter().zip(&y).all(|(p, q)| p == q),
        "same seed, different samples"
    );
}

// ----------------------------------------------------- the runaway decoder

/// Full-width punctuation reaches the alphabet instead of vanishing.
///
/// The chat model punctuates Han full-width and the translator carries it into
/// POJ, so this is what nearly every line arriving here ends in. None of these
/// marks is among the checkpoint's 71 symbols, and before they were folded the
/// tokeniser dropped them without a word - taking the decoder's only
/// end-of-utterance cue with them.
#[test]
fn full_width_punctuation_is_folded_rather_than_dropped() {
    use xabe_taco::poj_to_tlpa;

    assert_eq!(poj_to_tlpa("lí hó。"), "li2 ho2.");
    assert_eq!(poj_to_tlpa("lí hó，sī"), "li2 ho2,si7");
    assert_eq!(poj_to_tlpa("lí hó！"), "li2 ho2!");
    assert_eq!(poj_to_tlpa("lí hó？"), "li2 ho2?");
    assert_eq!(poj_to_tlpa("lí hó、sī"), "li2 ho2,si7");
    // ASCII is already right and must not be touched twice.
    assert_eq!(poj_to_tlpa("li2 ho2."), "li2 ho2.");
}

/// A line with no punctuation gets some, and one that has some is left alone.
///
/// Without this the gate can fail to fire at all and the decoder runs to
/// `max_decoder_steps` - 34.8 seconds of held tone on this checkpoint, which
/// is the "stuck voice". Any of the six marks satisfies it, measured, so a
/// clause ending in a comma is not repunctuated into a sentence it is not.
#[test]
fn a_line_with_no_punctuation_is_given_a_stop() {
    use xabe_taco::{poj_to_tlpa, with_gate_cue};

    assert_eq!(with_gate_cue("li2 ho2"), "li2 ho2.");
    assert_eq!(with_gate_cue("tsiah8-pa2--bo5"), "tsiah8-pa2--bo5.");
    // Trailing whitespace is not punctuation and must not hide the absence.
    assert_eq!(with_gate_cue("li2 ho2  "), "li2 ho2.");

    for already in [
        "li2 ho2.", "li2 ho2,", "li2 ho2?", "li2 ho2!", "li2 ho2;", "li2 ho2:",
    ] {
        assert_eq!(
            with_gate_cue(already),
            already,
            "{already} was repunctuated"
        );
    }

    // The pipeline converts first for exactly this reason: a full-width mark
    // is already ASCII by the time the cue looks, so it is not stopped twice.
    assert_eq!(with_gate_cue(&poj_to_tlpa("lí hó。")), "li2 ho2.");
    assert_eq!(with_gate_cue(&poj_to_tlpa("lí hó")), "li2 ho2.");

    // Nothing speakable: left alone, so the caller still reaches the
    // empty-sequence path and its warning rather than synthesising one stop.
    assert_eq!(with_gate_cue(""), "");
    assert_eq!(with_gate_cue("   "), "   ");
}
