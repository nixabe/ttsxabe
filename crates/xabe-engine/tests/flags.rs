//! What the flag surface accepts and what it refuses.
//!
//! These run in microseconds and need no model, which is the point: every
//! combination that cannot mean anything is rejected before a checkpoint is
//! opened. The stage flags are the part of this engine most likely to be typed
//! wrong, because a working deployment spreads them across six processes and
//! an equal number of environment variables.

use clap::Parser;

/// Parses a command line as the binary would, minus the program name.
fn parse(argv: &[&str]) -> Result<xabe_engine::Args, clap::Error> {
    let mut full = vec!["xabe-engine"];
    full.extend_from_slice(argv);
    xabe_engine::Args::try_parse_from(full)
}

/// Resolves stages from a command line, or panics with the clap error.
fn stages(argv: &[&str]) -> Result<xabe_engine::Stages, xabe_engine::StageError> {
    parse(argv).expect("flags parse").stages()
}

/// Resolves the action too, which is what a run actually does.
fn action(argv: &[&str]) -> Result<xabe_engine::Action, xabe_engine::ActionError> {
    let args = parse(argv).expect("flags parse");
    let stages = args.stages().expect("stages resolve");
    xabe_engine::Action::resolve(
        &stages,
        args.serve.as_deref(),
        args.input.as_ref(),
        args.text.as_deref(),
        args.out.as_ref(),
    )
}

// ---------------------------------------------------------------- the symmetry

#[test]
fn a_stage_is_satisfied_either_locally_or_over_http() {
    let local = stages(&["--tts-model", "/m", "--text", "x"]).expect("local");
    let remote = stages(&["--tts-url", "http://h:1", "--text", "x"]).expect("remote");

    assert!(local.tts.is_on() && remote.tts.is_on());
    assert!(matches!(local.tts, xabe_engine::Stage::Local { .. }));
    assert!(matches!(remote.tts, xabe_engine::Stage::Remote { .. }));
    // Whatever is downstream cannot tell them apart, which is the whole design.
    assert_eq!(local.full_chain(), remote.full_chain());
}

#[test]
fn giving_both_halves_of_a_stage_is_refused_by_name() {
    let e = stages(&["--tts-model", "/m", "--tts-url", "http://h:1"]).unwrap_err();
    assert_eq!(e, xabe_engine::StageError::Both(xabe_engine::Kind::Tts));
    assert!(e.to_string().contains("--tts-model"), "{e}");
}

#[test]
fn a_device_for_a_delegated_stage_is_refused_rather_than_ignored() {
    // The likely typo is a stage that was meant to be local: --asr-device got
    // typed and --asr-model did not. Ignoring it starts a process that quietly
    // uses somebody else's GPU assignment.
    let e = stages(&["--asr-url", "http://h:1", "--asr-device", "1"]).unwrap_err();
    assert_eq!(
        e,
        xabe_engine::StageError::DeviceWithoutModel(xabe_engine::Kind::Asr)
    );
}

#[test]
fn a_device_with_no_stage_at_all_is_refused() {
    let e = stages(&["--tts-model", "/m", "--vad-device", "0"]).unwrap_err();
    assert_eq!(
        e,
        xabe_engine::StageError::DeviceWithoutModel(xabe_engine::Kind::Vad)
    );
}

#[test]
fn the_device_defaults_to_the_first_card_not_to_the_cpu() {
    // The CPU path is the scalar reference and roughly 45x slower than real
    // time. Defaulting to it would make the engine look broken rather than slow.
    match stages(&["--tts-model", "/m"]).expect("stages").tts {
        xabe_engine::Stage::Local { device, .. } => {
            assert_eq!(device, xabe_engine::Device::Cuda(0));
        }
        other => panic!("expected Local, got {other:?}"),
    }
}

#[test]
fn cpu_and_an_ordinal_both_parse_and_anything_else_is_named() {
    assert_eq!(
        xabe_engine::Device::parse("cpu"),
        Some(xabe_engine::Device::Cpu)
    );
    assert_eq!(
        xabe_engine::Device::parse("2"),
        Some(xabe_engine::Device::Cuda(2))
    );
    assert_eq!(xabe_engine::Device::parse("cuda:2"), None);

    let e = stages(&["--tts-model", "/m", "--tts-device", "gpu0"]).unwrap_err();
    assert!(e.to_string().contains("gpu0"), "{e}");
}

// ------------------------------------------------------------- cross-stage rules

#[test]
fn a_run_with_no_stage_says_which_flags_would_give_it_one() {
    let e = stages(&[]).unwrap_err();
    assert_eq!(e, xabe_engine::StageError::Nothing);
    for flag in [
        "--asr-model",
        "--vad-url",
        "--tts-model",
        "--translator-url",
    ] {
        assert!(e.to_string().contains(flag), "{flag} missing from: {e}");
    }
}

#[test]
fn a_served_vad_needs_an_asr_to_gate() {
    let e = stages(&["--serve", "127.0.0.1:8080", "--vad-model", "/v"]).unwrap_err();
    assert_eq!(e, xabe_engine::StageError::VadWithoutAsr);
}

#[test]
fn a_vad_over_a_file_is_a_tool_rather_than_a_gate_and_needs_no_asr() {
    // Same flags, no --serve: this is the documented `--vad-model ... --in
    // clip.wav` segment dump, and refusing it would contradict the CLI docs.
    let a = action(&["--vad-model", "/v", "--in", "clip.wav"]).expect("action");
    assert!(matches!(a, xabe_engine::Action::Segment { .. }));
}

#[test]
fn the_full_chain_needs_speech_in_a_reply_and_speech_out() {
    let partial = stages(&["--asr-model", "/a", "--tts-model", "/t"]).expect("stages");
    assert!(!partial.full_chain(), "no LLM is not a full chain");

    let full = stages(&[
        "--asr-model",
        "/a",
        "--tts-model",
        "/t",
        "--llm-url",
        "http://h:8082",
    ])
    .expect("stages");
    assert!(full.full_chain());

    // The translator is deliberately not required: --direct-taigi takes it out
    // of the reply path, and that is the default the measured pipeline runs.
    assert!(!full.translator.is_on());
}

// ------------------------------------------------------------------ the actions

#[test]
fn serve_and_a_one_shot_input_are_not_the_same_run() {
    let e = action(&["--serve", "1.2.3.4:1", "--tts-model", "/t", "--text", "x"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::ServeWithInput);
}

#[test]
fn text_without_a_tts_stage_names_the_flags_that_would_fix_it() {
    let e = action(&["--asr-model", "/a", "--text", "x", "--out", "o.wav"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::TextWithoutTts);
    assert!(e.to_string().contains("--tts-model"), "{e}");
}

#[test]
fn text_without_an_output_is_refused_before_the_model_loads() {
    let e = action(&["--tts-model", "/t", "--text", "x"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::SpeakWithoutOut);
}

#[test]
fn audio_in_with_nothing_that_reads_audio_is_refused() {
    let e = action(&["--tts-model", "/t", "--in", "clip.wav"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::InputWithoutReader);
}

#[test]
fn asr_wins_over_vad_because_the_vad_is_the_gate_in_front_of_it() {
    let a = action(&["--asr-model", "/a", "--vad-model", "/v", "--in", "c.wav"]).expect("action");
    assert!(matches!(a, xabe_engine::Action::Transcribe { .. }));
}

#[test]
fn configured_stages_with_no_work_asked_of_them_is_refused_not_a_silent_success() {
    // The shape of this is a serve command with --serve forgotten, which would
    // otherwise load every model and exit 0.
    let e = action(&["--tts-model", "/t"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::Nothing);
    assert!(e.to_string().contains("--serve"), "{e}");
}

#[test]
fn in_and_text_together_are_refused() {
    let e = action(&["--tts-model", "/t", "--in", "c.wav", "--text", "x"]).unwrap_err();
    assert_eq!(e, xabe_engine::ActionError::BothInputs);
}

// ------------------------------------------------------------------- env twins

#[test]
fn every_stage_flag_has_an_environment_twin() {
    // A container is configured by environment, not by rewriting a command
    // line, so a flag without a twin is a flag that cannot be set there.
    let help = {
        let mut buf = Vec::new();
        <xabe_engine::Args as clap::CommandFactory>::command()
            .write_long_help(&mut buf)
            .expect("render help");
        String::from_utf8(buf).expect("utf8")
    };
    for var in [
        "XABE_SERVE",
        "XABE_ASR_MODEL",
        "XABE_ASR_URL",
        "XABE_ASR_DEVICE",
        "XABE_VAD_MODEL",
        "XABE_VAD_URL",
        "XABE_VAD_DEVICE",
        "XABE_TTS_MODEL",
        "XABE_TTS_URL",
        "XABE_TTS_DEVICE",
        "XABE_TRANSLATOR_MODEL",
        "XABE_TRANSLATOR_URL",
        "XABE_TRANSLATOR_DEVICE",
        "XABE_LLM_URL",
    ] {
        assert!(help.contains(var), "{var} is not in --help");
    }
}
