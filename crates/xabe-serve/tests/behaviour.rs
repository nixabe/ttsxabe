//! The behaviours the pipeline was tuned into, each tested against the case
//! that produced it.
//!
//! None of this is arbitrary. Every rule here was added after a real failure on
//! real speech, and the failures are recorded in the test names so that a
//! future change that breaks one is told what it is breaking.

use xabe_serve::{Chunker, Decision, Endpointer, TurnPolicy, sanitize_asr};
use xabe_serve::{clean, normalize_for_mms, split_poj, split_sentences};

// ------------------------------------------------------- hallucination defence

#[test]
fn the_phrases_whisper_invents_out_of_silence_are_dropped() {
    // Every one of these is YouTube subtitle boilerplate that Breeze-ASR-26
    // emits confidently when it has nothing to transcribe. Answering them is
    // the assistant replying to its own hallucination.
    for phrase in [
        "謝謝觀看",
        "謝謝大家",
        "感謝收看",
        "請訂閱",
        "請不吝點贊 訂閱",
        "字幕由 Amara.org 社群提供",
        "下集再見",
        "我們下次再見",
        "Thanks for watching!",
        // The pattern this was ported from read `Thanks? for watching`, which
        // matches "Thank for watching" and misses the form people actually say.
        "Thank you for watching.",
        "Subtitles by the Amara.org community",
        "Please subscribe to my channel",
    ] {
        assert_eq!(sanitize_asr(phrase), "", "should have dropped {phrase:?}");
    }
}

#[test]
fn subtitle_annotations_are_stripped_and_what_is_left_is_judged() {
    // On room noise the model produced "(我會陪你一起走)" - parenthesised,
    // because that is how its training data marks non-speech.
    assert_eq!(sanitize_asr("(音樂)"), "");
    assert_eq!(sanitize_asr("【掌聲】"), "");
    // Stripping the annotation must not throw away real speech beside it.
    assert_eq!(sanitize_asr("(音樂) 今天天氣很好"), "今天天氣很好");
}

#[test]
fn punctuation_alone_is_not_a_turn() {
    for noise in ["。", "…", "、、、", "  ", "...", "！？"] {
        assert_eq!(sanitize_asr(noise), "", "should have dropped {noise:?}");
    }
}

#[test]
fn a_lone_character_is_noise_unless_it_is_a_word_that_is_a_whole_turn() {
    // The rule earns its keep: one character after cleaning is far more often
    // a fragment of noise than a reply.
    assert_eq!(sanitize_asr("我"), "");
    assert_eq!(sanitize_asr("啊"), "");

    // But the Python applied it unconditionally, which silently discarded
    // legitimate one-character answers. A user saying 好 was ignored.
    for reply in ["好", "是", "對", "嗯", "有", "無"] {
        assert_eq!(sanitize_asr(reply), reply, "{reply} is a real turn");
    }
}

#[test]
fn ordinary_speech_survives_untouched_apart_from_whitespace() {
    assert_eq!(sanitize_asr("今天天氣很好"), "今天天氣很好");
    assert_eq!(sanitize_asr("  你好   世界 "), "你好 世界");
    assert_eq!(
        sanitize_asr("謝謝你的幫忙，我知道了"),
        "謝謝你的幫忙，我知道了"
    );
}

#[test]
fn a_thank_you_that_continues_into_a_sentence_is_not_boilerplate() {
    // The pattern is anchored at both ends for exactly this reason: 謝謝 is a
    // real thing to say, and only the bare subtitle forms are hallucinations.
    let real = "謝謝你今天來幫我";
    assert_eq!(sanitize_asr(real), real);
}

// --------------------------------------------------------------- reply chunking

#[test]
fn the_first_chunk_breaks_at_a_clause_and_later_ones_at_a_sentence() {
    let mut c = Chunker::new(4, 6);
    // A Taigi reply is often one long sentence. Waiting for 。 means waiting
    // for all of it before any audio exists - measured 4.1 s to first audio.
    assert_eq!(c.push("台北今仔日"), None, "no boundary yet");
    let first = c.push("好天，").expect("a comma ends the first chunk");
    assert_eq!(first, "台北今仔日好天，");

    // From here on only a sentence boundary will do, so a comma is not enough.
    assert_eq!(c.push("溫度差不多，"), None, "a comma no longer suffices");
    let second = c
        .push("二十五度。")
        .expect("a full stop ends a later chunk");
    assert_eq!(second, "溫度差不多，二十五度。");
}

#[test]
fn a_boundary_that_arrives_too_early_does_not_produce_a_chunk() {
    let mut c = Chunker::new(4, 6);
    // "好，" is two characters. Synthesising it would spend a whole round trip
    // on a syllable and then leave a gap before the rest.
    assert_eq!(c.push("好，"), None);
    assert_eq!(c.push("今仔日好天。").as_deref(), Some("好，今仔日好天。"));
}

#[test]
fn a_chunk_ends_at_the_boundary_and_not_at_the_end_of_the_piece() {
    let mut c = Chunker::new(4, 6);
    // A streamed piece is several characters, so it straddles the boundary.
    // Emitting all of `你好！天` would tear `天` off `天氣`, and the orphan is
    // then translated alone as "sky" and synthesised as its own segment -
    // which is what a listener hears as speaking word by word.
    let first = c.push("你好！天氣的確很好，").expect("the comma ends it");
    assert_eq!(first, "你好！天氣的確很好，");
    assert_eq!(c.push("我已經吃飽了，"), None, "a comma no longer suffices");
    assert_eq!(
        c.push("謝謝關心。").as_deref(),
        Some("我已經吃飽了，謝謝關心。")
    );
}

#[test]
fn text_after_the_boundary_is_kept_for_the_next_chunk() {
    let mut c = Chunker::new(4, 6);
    // `了` arrives in the same piece as the full stop that ends the clause
    // before it, and must still be spoken - as part of what follows, not alone.
    assert_eq!(
        c.push("今仔日好天。食飽").as_deref(),
        Some("今仔日好天。"),
        "the chunk stops at the full stop"
    );
    assert_eq!(c.finish().as_deref(), Some("食飽"), "and the rest survives");
}

#[test]
fn the_minimum_length_is_counted_in_characters_not_bytes() {
    // Every Han character is three UTF-8 bytes, so a byte count would fire the
    // first chunk after one character and the whole tuning would be wrong.
    let mut c = Chunker::new(4, 6);
    assert_eq!(
        c.push("好啊，"),
        None,
        "three characters is below the minimum of four"
    );
}

#[test]
fn whatever_is_left_when_the_reply_ends_is_still_spoken() {
    let mut c = Chunker::new(4, 6);
    assert_eq!(c.push("食飽矣"), None);
    assert_eq!(c.finish().as_deref(), Some("食飽矣"));
    assert_eq!(c.finish(), None, "and only once");
}

#[test]
fn a_reply_that_is_only_whitespace_produces_nothing() {
    let mut c = Chunker::new(4, 6);
    assert_eq!(c.push("  \n "), None);
    assert_eq!(c.finish(), None);
}

// -------------------------------------------------------- preparing text to say

#[test]
fn decoration_that_is_written_but_not_spoken_is_removed() {
    assert_eq!(clean("**你好** `世界`"), "你好 世界");
    assert_eq!(clean("你好（笑）世界"), "你好世界");
    assert_eq!(clean("你好   \n  世界"), "你好 世界");
}

#[test]
fn the_nasal_is_the_one_poj_symbol_the_vocabulary_lacks() {
    // The model's 48 symbols are POJ, not Tâi-lô. Everything is left as the
    // translator wrote it except ⁿ, which becomes nn. See docs/MODEL.md for
    // the round trip that established this.
    assert_eq!(normalize_for_mms("hoⁿ"), "honn");
    assert_eq!(
        normalize_for_mms("chin hó"),
        "chin hó",
        "c and ó are in the vocabulary"
    );
}

#[test]
fn long_text_is_split_so_the_synthesiser_does_not_degrade() {
    let long = "一二三四五六七八九十".repeat(8); // 80 characters, no punctuation
    let parts = split_sentences(&long, 60);
    assert!(parts.len() >= 2, "80 characters must not be one chunk");
    for p in &parts {
        assert!(
            p.chars().count() <= 60,
            "chunk too long: {}",
            p.chars().count()
        );
    }
}

#[test]
fn a_sentence_with_no_internal_punctuation_still_terminates() {
    // The comma search finds nothing here. Without the hard cut this loops.
    let parts = split_sentences(&"甲".repeat(200), 60);
    assert_eq!(parts.len(), 4);
    assert_eq!(parts.iter().map(|p| p.chars().count()).sum::<usize>(), 200);
}

#[test]
fn romanised_text_splits_on_ascii_punctuation_and_never_mid_syllable() {
    let text = "li hó, kin-á-ji̍t thinn-khì chin hó. góa beh khì chhī-tiûⁿ bé mih-kiāⁿ.";
    let parts = split_poj(text, 40);
    assert!(parts.len() >= 2);
    for p in &parts {
        assert!(!p.starts_with('-'), "cut mid-syllable: {p}");
        assert!(p.chars().count() <= 41, "chunk too long: {p}");
    }
}

#[test]
fn a_chunk_already_short_enough_is_not_split_at_its_sentence_marks() {
    // The limit is a ceiling, not a target. Two short sentences inside it are
    // one waveform, not two with a gap in the middle.
    let parts = split_poj("Li hó! Thinn-khì chin hó,", 120);
    assert_eq!(parts, vec!["Li hó! Thinn-khì chin hó,"]);

    let han = split_sentences("你好！天氣真好。", 60);
    assert_eq!(han, vec!["你好！天氣真好。"]);
}

#[test]
fn packing_never_carries_a_part_past_the_limit() {
    // Three sentences of 10 characters against a limit of 25: the first two
    // fit together and the third starts a new part rather than overflowing.
    let parts = split_poj("aaaaaaaaa. bbbbbbbbb. ccccccccc.", 25);
    for p in &parts {
        assert!(p.chars().count() <= 25, "over the limit: {p}");
    }
    assert_eq!(parts.len(), 2, "{parts:?}");
}

#[test]
fn splitting_preserves_every_character_that_was_not_whitespace() {
    let text = "第一句。第二句，還有更多的內容，最後一句。";
    let joined: String = split_sentences(text, 8).join("");
    let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    assert_eq!(strip(&joined), strip(text));
}

// ------------------------------------------------------------------ turn-taking

/// Feeds frames of one energy at 32 ms each until something happens.
///
/// Returns the *first* decision that is not `Idle` or `Listening`, and stops
/// there. Those two mean "nothing happened", and a run of frames is fed
/// precisely to find the frame where something did. Returning the last decision
/// instead would hide a commit behind whatever the next frames started, which
/// on continuing loud audio is a fresh turn.
fn feed(e: &mut Endpointer, energy: f32, n: usize) -> Decision {
    let mut last = Decision::Idle;
    for _ in 0..n {
        last = e.push(energy, 32);
        if !matches!(last, Decision::Idle | Decision::Listening) {
            return last;
        }
    }
    last
}

/// Frames of silence needed to arm the end of a turn, plus a margin.
fn to_arm(p: TurnPolicy) -> usize {
    (p.silence_ms / 32 + 2) as usize
}

/// Further frames of silence needed to commit it, plus a margin.
fn to_commit(p: TurnPolicy) -> usize {
    (p.grace_ms / 32 + 2) as usize
}

#[test]
fn a_single_loud_frame_does_not_open_a_turn() {
    // A click, a door, a chair. onset_frames exists for exactly this.
    let mut e = Endpointer::new(TurnPolicy::default());
    assert_eq!(e.push(0.9, 32), Decision::Idle);
    assert_eq!(e.push(0.001, 32), Decision::Idle);
    assert_eq!(e.push(0.9, 32), Decision::Idle, "the run was broken");
}

#[test]
fn sustained_speech_opens_a_turn() {
    let mut e = Endpointer::new(TurnPolicy::default());
    assert_eq!(feed(&mut e, 0.2, 3), Decision::Opened);
}

#[test]
fn room_noise_at_the_old_threshold_no_longer_opens_a_turn() {
    // The trigger used to be 0.012, which sat at room-noise level: the mic
    // fired on silence and Whisper hallucinated a sentence out of it.
    let mut e = Endpointer::new(TurnPolicy::default());
    assert_eq!(feed(&mut e, 0.02, 50), Decision::Idle);
}

#[test]
fn a_pause_arms_the_end_of_turn_but_does_not_commit_it() {
    let p = TurnPolicy::default();
    let mut e = Endpointer::new(p);
    feed(&mut e, 0.2, 3);
    feed(&mut e, 0.2, 30); // ~1 s of speech

    // 700 ms of silence arms it...
    let armed = feed(&mut e, 0.0, to_arm(p));
    assert_eq!(armed, Decision::Armed, "the pause should arm, not commit");

    // ...and speech inside the grace window resumes the same turn.
    assert_eq!(e.push(0.2, 32), Decision::Resumed);
    assert!(matches!(e.push(0.2, 32), Decision::Listening));
}

#[test]
fn a_mid_sentence_thinking_pause_costs_nothing() {
    // "台北的天氣…嗯…今天如何" was being cut in half and sent as a question the
    // user had not finished asking.
    let p = TurnPolicy::default();
    let mut e = Endpointer::new(p);
    feed(&mut e, 0.2, 3);
    feed(&mut e, 0.2, 20);
    feed(&mut e, 0.0, to_arm(p)); // arms
    feed(&mut e, 0.2, 20); // resumes and continues
    let d = feed(&mut e, 0.2, 5);
    assert!(
        matches!(d, Decision::Listening),
        "the turn must still be open, got {d:?}"
    );
}

#[test]
fn silence_past_the_grace_window_commits_the_turn() {
    let p = TurnPolicy::default();
    let mut e = Endpointer::new(p);
    feed(&mut e, 0.2, 3);
    feed(&mut e, 0.2, 30);
    assert_eq!(feed(&mut e, 0.0, to_arm(p)), Decision::Armed);
    let d = feed(&mut e, 0.0, to_commit(p));
    assert!(
        matches!(
            d,
            Decision::Committed {
                voiced: true,
                truncated: false
            }
        ),
        "expected a committed turn, got {d:?}",
    );
}

#[test]
fn a_turn_of_nothing_but_brief_noise_commits_as_unvoiced() {
    let p = TurnPolicy::default();
    let mut e = Endpointer::new(p);
    feed(&mut e, 0.2, 3); // opens
    // Just below vad_stop from here: never accumulates voiced audio.
    assert_eq!(feed(&mut e, 0.01, to_arm(p)), Decision::Armed);
    let d = feed(&mut e, 0.01, to_commit(p));
    assert!(
        matches!(d, Decision::Committed { voiced: false, .. }),
        "expected an unvoiced commit, got {d:?}",
    );
}

#[test]
fn a_stuck_microphone_is_cut_rather_than_growing_without_bound() {
    let p = TurnPolicy::default();
    let mut e = Endpointer::new(p);
    feed(&mut e, 0.2, 3);
    let d = feed(&mut e, 0.2, (p.max_ms / 32 + 4) as usize);
    assert!(
        matches!(
            d,
            Decision::Committed {
                truncated: true,
                ..
            }
        ),
        "expected a truncated commit, got {d:?}",
    );
}

#[test]
fn push_to_talk_commits_regardless_of_how_loud_it_was() {
    // Push-to-talk never accumulates voiced audio the way the detector does,
    // so applying the energy test to it discarded every push-to-talk utterance
    // whenever auto-detect was checked - which is the default.
    let mut e = Endpointer::new(TurnPolicy::default());
    feed(&mut e, 0.2, 3);
    feed(&mut e, 0.001, 40); // quiet, but the button is held
    assert_eq!(
        e.release(),
        Decision::Committed {
            voiced: true,
            truncated: false
        },
        "a held button is an explicit statement that this is a turn",
    );
}

#[test]
fn a_released_button_that_was_held_for_an_instant_is_still_too_short() {
    let mut e = Endpointer::new(TurnPolicy::default());
    feed(&mut e, 0.2, 3);
    assert_eq!(
        e.release(),
        Decision::Committed {
            voiced: false,
            truncated: false
        },
        "under min_ms, even from the button",
    );
}

#[test]
fn the_frames_that_proved_speech_was_starting_count_towards_the_turn() {
    // The page used to reset its buffer to a single frame on onset, throwing
    // away the ~512 ms that proved speech had begun and clipping the first
    // syllable off every utterance.
    let mut e = Endpointer::new(TurnPolicy::default());
    feed(&mut e, 0.2, 3);
    assert_eq!(e.turn_ms(), 96, "all three onset frames belong to the turn");
}
