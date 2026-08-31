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
fn a_gpu_only_stage_refuses_the_cpu_rather_than_taking_twenty_minutes() {
    // The mirror of the VAD's rule, for the mirror-image reason. One
    // 30-second window is about 2.2 TFLOP through Whisper's encoder alone, and
    // the scalar kernels run at something under 2 GFLOP/s - twenty minutes an
    // utterance, which is not a slow option but a fictional one.
    let e = stages(&["--asr-model", "/a", "--asr-device", "cpu", "--in", "c.wav"]).unwrap_err();
    assert_eq!(e, xabe_engine::StageError::GpuOnly(xabe_engine::Kind::Asr));

    // A card is accepted, and is also what happens with no device flag.
    let ok = stages(&["--asr-model", "/a", "--asr-device", "1", "--in", "c.wav"]);
    assert!(ok.is_ok(), "{ok:?}");

    // The translator too: 27 GB of weights and 26 GFLOP a token.
    let e = stages(&[
        "--serve",
        "127.0.0.1:8000",
        "--asr-url",
        "http://a",
        "--llm-url",
        "http://l",
        "--tts-url",
        "http://t",
        "--translator-model",
        "/t",
        "--translator-device",
        "cpu",
    ])
    .unwrap_err();
    assert_eq!(
        e,
        xabe_engine::StageError::GpuOnly(xabe_engine::Kind::Translator)
    );
    match stages(&["--asr-model", "/a", "--in", "c.wav"])
        .expect("stages")
        .asr
    {
        xabe_engine::Stage::Local { device, .. } => {
            assert_eq!(
                device,
                xabe_engine::Device::Cuda(0),
                "and it is the default"
            );
        }
        other => panic!("expected Local, got {other:?}"),
    }
}

#[test]
fn a_cpu_only_stage_refuses_a_card_rather_than_quietly_ignoring_it() {
    // The VAD has no CUDA implementation and is not going to: 15 tensors and a
    // millisecond of work, where the transfer would cost more than the
    // arithmetic. Accepting the flag and running on the CPU anyway made the
    // startup log announce `device=cuda:0` for a stage that was on the CPU.
    let e = stages(&["--vad-model", "/v", "--vad-device", "0", "--in", "c.wav"]).unwrap_err();
    assert_eq!(e, xabe_engine::StageError::CpuOnly(xabe_engine::Kind::Vad));

    // Explicit `cpu` is accepted, because it is true.
    let ok = stages(&["--vad-model", "/v", "--vad-device", "cpu", "--in", "c.wav"]);
    assert!(ok.is_ok(), "{ok:?}");

    match stages(&["--vad-model", "/v", "--in", "c.wav"])
        .expect("stages")
        .vad
    {
        xabe_engine::Stage::Local { device, .. } => {
            assert_eq!(device, xabe_engine::Device::Cpu, "and it is the default");
        }
        other => panic!("expected Local, got {other:?}"),
    }
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

// ----------------------------------------------------------- registered engines

#[test]
fn a_named_engine_is_a_tts_stage_on_its_own() {
    // Serving one synthesiser without loading the rest: `--tts-engine` names
    // the checkpoint and there is no `--tts-model` for it to be an extra to.
    let s = stages(&[
        "--tts-engine",
        "taco=/models/tts/tacotron2-nan",
        "--serve",
        "h:1",
    ])
    .expect("stages");

    assert!(
        s.has_tts(),
        "a registered engine is a synthesiser in this process"
    );
    assert!(s.any_on(), "so the run is not stageless");
    // The unnamed slot stays empty: what `--tts-engine` filled has a name, and
    // nothing downstream should mistake it for the `--tts-model` one.
    assert_eq!(s.tts, xabe_engine::Stage::Off);
}

#[test]
fn a_device_is_meaningful_for_a_named_engine_with_no_tts_model() {
    // The card a registered engine lands on is `--tts-device`, and with no
    // `--tts-model` there is no other flag that says which one.
    let s = stages(&[
        "--tts-engine",
        "taco=/models/tts/tacotron2-nan",
        "--tts-device",
        "1",
        "--serve",
        "h:1",
    ])
    .expect("stages");
    assert!(s.has_tts());

    // Still validated, though: the stage being off is not a licence to accept
    // a device string that names nothing.
    let e = stages(&[
        "--tts-engine",
        "taco=/t",
        "--tts-device",
        "bogus",
        "--serve",
        "h:1",
    ])
    .unwrap_err();
    assert_eq!(
        e,
        xabe_engine::StageError::BadDevice(xabe_engine::Kind::Tts, "bogus".into())
    );
}

#[test]
fn a_named_engine_completes_the_full_chain() {
    let full = stages(&[
        "--asr-model",
        "/a",
        "--tts-engine",
        "taco=/t",
        "--llm-url",
        "http://h:8082",
        "--serve",
        "h:1",
    ])
    .expect("stages");
    assert!(full.full_chain(), "speech in, a reply, and speech out");
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
        "XABE_TTS_ENGINES",
    ] {
        assert!(help.contains(var), "{var} is not in --help");
    }
}

// ------------------------------------------------------------- system prompt

/// Resolves the system prompt from a command line, as `serve` would.
fn prompt(argv: &[&str]) -> Result<String, xabe_engine::EngineError> {
    xabe_engine::serve::system_prompt(&parse(argv).expect("flags parse"))
}

#[test]
fn an_inline_prompt_replaces_the_built_in_one_whole() {
    let built_in = prompt(&["--llm-url", "http://h:1", "--serve", "h:1"]).expect("built-in");
    let given = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--system-prompt",
        "你是小哇，用台語講話。",
    ])
    .expect("inline");

    assert_eq!(given, "你是小哇，用台語講話。");
    // Replaced, not prepended: a system prompt is one instruction, and two
    // stacked is the shape that produces a model following neither.
    assert!(
        !given.contains(&built_in),
        "the built-in prompt survived alongside the given one",
    );
}

#[test]
fn the_two_ways_of_giving_a_prompt_are_alternatives() {
    let e = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--system-prompt",
        "x",
        "--prompt-file",
        "/nonexistent",
    ])
    .expect_err("both should be refused");

    assert!(
        matches!(e, xabe_engine::EngineError::BothPrompts),
        "wanted BothPrompts, got {e}",
    );
    // Refused on the flags alone: the path is never opened, so the error is
    // about the combination rather than about a file that happens to be
    // missing too.
    assert!(!format!("{e}").contains("nonexistent"));
}

#[test]
fn a_prompt_that_says_nothing_is_refused_rather_than_used() {
    for empty in ["", "   ", "\n\t  \n"] {
        let e = prompt(&[
            "--llm-url",
            "http://h:1",
            "--serve",
            "h:1",
            "--system-prompt",
            empty,
        ])
        .expect_err("an empty prompt should be refused");
        assert!(
            matches!(e, xabe_engine::EngineError::EmptyPrompt { flag } if flag == "--system-prompt"),
            "wanted EmptyPrompt, got {e}",
        );
    }
}

#[test]
fn a_given_prompt_is_trimmed_but_not_otherwise_rewritten() {
    // Braces stay braces. The built-ins interpolate `person` and `bot`
    // because they are the engine's own text; a prompt from outside is not
    // the engine's to rewrite, and one containing a brace would otherwise
    // change meaning depending on where it was written.
    let given = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--person",
        "阿公",
        "--system-prompt",
        "  {person} 佮 {bot} 咧講話。  ",
    ])
    .expect("inline");

    assert_eq!(given, "{person} 佮 {bot} 咧講話。");
}

#[test]
fn which_built_in_is_chosen_follows_who_produces_taigi() {
    let mandarin = prompt(&["--llm-url", "http://h:1", "--serve", "h:1"]).expect("mandarin");
    let taigi = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--direct-taigi",
    ])
    .expect("direct taigi");

    assert_ne!(mandarin, taigi, "--direct-taigi did not change the prompt");
    assert!(
        taigi.contains("台語"),
        "the direct prompt should ask for Taigi"
    );
}

#[test]
fn direct_taigi_still_places_the_translator_when_the_prompt_is_given() {
    // `--direct-taigi` decides the pipeline; giving a prompt only takes over
    // which text is used, so the two are not the same switch.
    let given = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--direct-taigi",
        "--system-prompt",
        "講台語。",
    ])
    .expect("inline");

    assert_eq!(given, "講台語。");
    let args = parse(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--direct-taigi",
    ])
    .expect("flags parse");
    assert!(args.direct_taigi, "the flag still means what it meant");
}

#[test]
fn a_prompt_file_is_read_and_trimmed() {
    let dir = std::env::temp_dir().join("xabe-prompt-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("prompt.txt");
    std::fs::write(&path, "\n  你是小哇。  \n\n").expect("write");

    let given = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--prompt-file",
        path.to_str().expect("utf-8 path"),
    ])
    .expect("file");
    assert_eq!(given, "你是小哇。");

    // The same emptiness rule, from the other source. This used to be
    // accepted and produce a completion opening on a blank line.
    std::fs::write(&path, "   \n ").expect("write");
    let e = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--prompt-file",
        path.to_str().expect("utf-8 path"),
    ])
    .expect_err("an empty file should be refused");
    assert!(
        matches!(e, xabe_engine::EngineError::EmptyPrompt { flag } if flag == "--prompt-file"),
        "wanted EmptyPrompt, got {e}",
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_prompt_file_that_is_not_there_names_the_path() {
    let e = prompt(&[
        "--llm-url",
        "http://h:1",
        "--serve",
        "h:1",
        "--prompt-file",
        "/nonexistent/prompt.txt",
    ])
    .expect_err("a missing file should be refused");
    assert!(
        format!("{e}").contains("/nonexistent/prompt.txt"),
        "the error should name the path, got {e}",
    );
}
