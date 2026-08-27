//! The chat model's byte-level BPE against llama.cpp's own.
//!
//! This is an *exact* comparison, not a close one. A tokenizer that is nearly
//! right is worse than one that is obviously wrong: the model still produces
//! fluent text from slightly-wrong ids, so the failure arrives as a reply that
//! reads oddly rather than as an error. The only way that shows up in a test
//! is id-for-id equality on inputs chosen for where a reimplementation
//! diverges - see `tools/oracle/capture_chat_tokenizer.py` for why each case
//! is in the corpus.
//!
//! The reference is llama.cpp reading the same GGUF, which makes it the thing
//! being replaced rather than a stand-in for it.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct Golden {
    vocab_size: usize,
    specials: std::collections::HashMap<String, u32>,
    cases: Vec<Case>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model() -> Option<PathBuf> {
    let p = root().join("models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf");
    p.is_file().then_some(p)
}

fn golden() -> Option<Golden> {
    let p = root().join(".golden/chat/tokenizer.json");
    let s = std::fs::read_to_string(p).ok()?;
    Some(serde_json::from_str(&s).expect("parse the capture"))
}

/// Opens the vocabulary once; every test below shares it.
fn bpe() -> Option<(xabe_llama::Bpe, Golden)> {
    let (m, g) = (model()?, golden()?);
    let f = xabe_gguf::GgufFile::open(&m).expect("open the chat GGUF");
    // The file outlives nothing here - `Bpe` copies what it needs out of the
    // metadata rather than borrowing the mapping, which is what lets it be
    // returned past the end of this function.
    Some((
        xabe_llama::Bpe::from_gguf(&f).expect("read the vocabulary"),
        g,
    ))
}

#[test]
fn every_captured_case_tokenizes_identically() {
    let Some((bpe, g)) = bpe() else {
        println!("SKIP: run tools/oracle/capture_chat_tokenizer.py");
        return;
    };

    let mut wrong = Vec::new();
    for c in &g.cases {
        // `parse_special = false`, matching how the reference was captured:
        // `<|eot_id|>` in the input is ordinary text there, and this is the
        // path every character of user text takes.
        let got = bpe.encode(&c.text, false);
        if got != c.ids {
            wrong.push((c, got));
        }
    }

    if !wrong.is_empty() {
        // Printing all of them rather than the first: a pre-tokenizer mistake
        // fails a whole class of cases at once, and the class is what names
        // the bug. One failure would only name a symptom.
        for (c, got) in &wrong {
            println!("  {:?}\n    want {:?}\n    got  {:?}", c.text, c.ids, got);
        }
        panic!("{} of {} cases differ", wrong.len(), g.cases.len());
    }
    println!("  {} cases, exact", g.cases.len());
    assert!(g.cases.len() >= 50, "the corpus should not have shrunk");
}

#[test]
fn the_vocabulary_is_the_size_the_file_declares() {
    let Some((bpe, g)) = bpe() else {
        println!("SKIP: run tools/oracle/capture_chat_tokenizer.py");
        return;
    };
    assert_eq!(bpe.len(), g.vocab_size);
    assert_eq!(bpe.len(), 128_256, "Llama-3's vocabulary");
}

#[test]
fn parsing_specials_collapses_what_not_parsing_them_spells_out() {
    // The two readings of `<|eot_id|>`, and the reason the flag exists. With
    // it off the input is nine ordinary tokens; with it on it is one. A
    // default either way would be wrong half the time: a chat template needs
    // the first, and user text that happens to contain the spelling needs the
    // second, or a user ends the model's turn from inside the prompt.
    let Some((bpe, g)) = bpe() else {
        println!("SKIP: run tools/oracle/capture_chat_tokenizer.py");
        return;
    };

    for (spelling, &id) in &g.specials {
        let parsed = bpe.encode(spelling, true);
        assert_eq!(parsed, vec![id], "{spelling} should parse to one token");

        let literal = bpe.encode(spelling, false);
        assert!(
            literal.len() > 1,
            "{spelling} should spell out when not parsed, got {literal:?}"
        );
        assert_eq!(bpe.special(spelling), Some(id));
    }
    assert!(g.specials.len() >= 5, "the header and turn markers");

    // Surrounding text survives the split rather than being eaten with the
    // token, which is the part a naive scan gets wrong.
    let eot = g.specials["<|eot_id|>"];
    let got = bpe.encode("Hi<|eot_id|>there", true);
    let want: Vec<u32> = [
        bpe.encode("Hi", false),
        vec![eot],
        bpe.encode("there", false),
    ]
    .concat();
    assert_eq!(got, want);
}

#[test]
fn a_bare_opener_does_not_stall_the_scan() {
    // `<|` that begins no known token. The scan has to consume through it or
    // loop forever on the same position - which is a hang, not a wrong answer,
    // and so would not show up as a failed comparison anywhere else.
    let Some((bpe, _)) = bpe() else {
        println!("SKIP: run tools/oracle/capture_chat_tokenizer.py");
        return;
    };
    for text in ["<|", "<|not_a_token|>", "a <| b", "<|<|<|", "<|unclosed"] {
        let ids = bpe.encode(text, true);
        assert!(!ids.is_empty(), "{text:?} produced nothing");
        assert_eq!(bpe.decode(&ids, false), text, "{text:?} did not round trip");
    }
}

#[test]
fn decoding_inverts_encoding_on_the_whole_corpus() {
    // Byte-level BPE is exactly invertible - there is no normalisation step to
    // lose anything to - so anything less than equality is a bug in the byte
    // alphabet, and the multi-byte cases are where it would be.
    let Some((bpe, g)) = bpe() else {
        println!("SKIP: run tools/oracle/capture_chat_tokenizer.py");
        return;
    };
    for c in &g.cases {
        let round = bpe.decode(&bpe.encode(&c.text, false), false);
        assert_eq!(round, c.text, "round trip");
        // And from the reference's ids, not just our own - so a tokenizer
        // that is self-consistently wrong still fails here.
        assert_eq!(bpe.decode(&c.ids, false), c.text, "decode of the reference");
    }
}

#[test]
fn digit_runs_split_at_three() {
    // The single clearest difference from the GPT-2 pattern, asserted on its
    // own so a regression names itself instead of arriving as "eleven cases
    // differ". A tokenizer that borrowed GPT-2's pattern passes prose and
    // fails here.
    let split = xabe_llama::pre_tokenize("1234567890");
    assert_eq!(split, ["123", "456", "789", "0"]);
    assert_eq!(xabe_llama::pre_tokenize("12"), ["12"]);
    assert_eq!(xabe_llama::pre_tokenize("1234"), ["123", "4"]);
}

#[test]
fn a_whitespace_run_hands_its_last_space_to_the_word_after_it() {
    // What `\s+(?!\S)` does, reconstructed without lookahead. Three spaces
    // before a word are two spaces plus a word owning the third; three spaces
    // at the end of the input are three spaces, because there is nothing to
    // hand one to.
    assert_eq!(xabe_llama::pre_tokenize("a   b"), ["a", "  ", " b"]);
    assert_eq!(xabe_llama::pre_tokenize("a  b"), ["a", " ", " b"]);
    assert_eq!(xabe_llama::pre_tokenize("a b"), ["a", " b"]);
    assert_eq!(xabe_llama::pre_tokenize("a   "), ["a", "   "]);
}

#[test]
fn a_run_containing_a_newline_keeps_all_of_it() {
    // The rule above applies to the *last* alternative only. A run with a
    // newline in it was matched further up the pattern, which keeps what it
    // took - so a blank line is one piece, not two newlines, and `\r\n` is
    // one piece, not a carriage return and a line feed.
    //
    // This is the case that failed when the trim was written without the
    // distinction, and it failed *quietly*: the ids were all real tokens and
    // the text round-tripped, so only the comparison against llama.cpp caught
    // it.
    assert_eq!(
        xabe_llama::pre_tokenize("para\n\npara"),
        ["para", "\n\n", "para"]
    );
    assert_eq!(
        xabe_llama::pre_tokenize("windows\r\nline"),
        ["windows", "\r\n", "line"]
    );
    // And the two rules compose: the newline run keeps its whole self, and
    // the ordinary run *after* it still gives up its last space.
    assert_eq!(
        xabe_llama::pre_tokenize("a  \n  b"),
        ["a", "  \n", " ", " b"]
    );
}

#[test]
fn a_sentencepiece_vocabulary_is_refused_rather_than_half_read() {
    // The translator's GGUF is `tokenizer.ggml.model = llama` - SentencePiece,
    // a different algorithm with no merge table. Reading it here would build
    // an empty rank map and silently tokenize every input one byte at a time,
    // which is fluent-looking nonsense rather than an error.
    let p = root().join("models/llm/taigi-translator-13b-f16.gguf");
    if !p.is_file() {
        println!("SKIP: the translator GGUF is missing");
        return;
    }
    let f = xabe_gguf::GgufFile::open(&p).expect("open");
    match xabe_llama::Bpe::from_gguf(&f) {
        Err(xabe_llama::LlamaError::Vocab { what, .. }) => {
            assert!(what.contains("gpt2"), "{what}");
        }
        other => panic!("wanted a Vocab refusal, got {:?}", other.map(|b| b.len())),
    }
}
