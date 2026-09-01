//! The spelling tables, checked by hand against what each model reads.
//!
//! These need no checkpoint and no capture: they are the table's structure,
//! and every case here is one somebody has to be able to read and confirm
//! against a description of the orthography. The corpus differential in
//! `correspondence.rs` is the other half - it checks the table against what
//! goruut actually emitted, over thousands of syllables nobody wrote down.

use xabe_taigi::{poj_to_ipa, poj_to_tailo, tailo_to_ipa};

#[test]
fn poj_becomes_tailo_with_numeric_tones() {
    // The spellings that differ, and the tone marks that become digits. These
    // came from `xabe-taco`, where this conversion used to live.
    assert_eq!(poj_to_tailo("góa"), "gua2");
    assert_eq!(poj_to_tailo("lí hó"), "li2 ho2");
    assert_eq!(poj_to_tailo("chhùi"), "tshui3");
    assert_eq!(poj_to_tailo("chi̍t"), "tsit8");
    assert_eq!(poj_to_tailo("Tâi-oân"), "tai5-uan5");
    assert_eq!(poj_to_tailo("hoⁿ"), "honn1");
    // Unmarked and stop-final is tone 4; unmarked otherwise is tone 1.
    assert_eq!(poj_to_tailo("sit"), "sit4");
    assert_eq!(poj_to_tailo("si"), "si1");
    // `o` with the POJ dot becomes `oo`.
    assert_eq!(poj_to_tailo("o\u{0358}"), "oo1");
    // Already numeric: left alone rather than converted twice.
    assert_eq!(poj_to_tailo("gua2 si7"), "gua2 si7");
}

#[test]
fn punctuation_is_left_alone_by_the_shared_table() {
    // The fold to ASCII is Tacotron2's requirement and lives in `xabe-taco`,
    // because the Coqui checkpoint wants the opposite. Asserting it here is
    // what keeps the two from being merged back together by someone tidying.
    assert_eq!(poj_to_tailo("lí hó。"), "li2 ho2。");
    assert_eq!(poj_to_tailo("sī--bô？"), "si7--bo5？");
}

#[test]
fn the_initials_are_the_ones_goruut_writes() {
    for (tailo, ipa) in [
        ("pa1", "pa˥˥"),
        ("pha1", "pʰa˥˥"),
        ("ba1", "ba˥˥"),
        ("ma1", "ma˥˥"),
        ("ta1", "ta˥˥"),
        ("tha1", "tʰa˥˥"),
        ("na1", "na˥˥"),
        ("la1", "la˥˥"),
        ("tsa1", "tsa˥˥"),
        ("tsha1", "tsʰa˥˥"),
        ("sa1", "sa˥˥"),
        ("ja1", "dza˥˥"),
        ("ka1", "ka˥˥"),
        ("kha1", "kʰa˥˥"),
        ("ga1", "\u{0261}a˥˥"),
        ("nga1", "ŋa˥˥"),
        ("ha1", "ha˥˥"),
        ("a1", "a˥˥"),
    ] {
        assert_eq!(tailo_to_ipa(tailo).text, ipa, "{tailo}");
    }
}

#[test]
fn g_is_the_script_g_not_the_ascii_one() {
    // U+0261, not U+0067. The model's vocabulary contains only the former -
    // ASCII `g` is not in it at all - and Coqui's own wrapper translates
    // goruut's output for exactly this reason. Getting it wrong drops every
    // `g` in the utterance without a word.
    let out = tailo_to_ipa("gua2").text;
    assert!(out.starts_with('\u{0261}'), "{out:?}");
    assert!(!out.contains('g'), "{out:?}");
}

#[test]
fn the_seven_tones_are_chao_letters() {
    for (tailo, ipa) in [
        ("si1", "si˥˥"),
        ("si2", "si˥˧"),
        ("si3", "si˨˩"),
        ("sit4", "sit˨"),
        ("si5", "si˨˦"),
        ("si7", "si˧˧"),
        ("sit8", "sit˦"),
    ] {
        assert_eq!(tailo_to_ipa(tailo).text, ipa, "{tailo}");
    }
    // Tone 6 merged with 2 everywhere this checkpoint saw, and goruut has no
    // letter for it. Mapping it rather than dropping it keeps a marked
    // syllable from vanishing.
    assert_eq!(tailo_to_ipa("si6").text, "si˥˧");
}

#[test]
fn the_rimes_that_are_easy_to_get_wrong() {
    for (tailo, ipa) in [
        // `oo` is the open o, `o` the close one.
        ("oo1", "ɔ˥˥"),
        ("o1", "o˥˥"),
        ("hoo1", "hɔ˥˥"),
        // `-h` is a glottal stop, not an `h`.
        ("ah4", "aʔ˨"),
        ("beh4", "beʔ˨"),
        // `-ng` is one segment.
        ("ang1", "aŋ˥˥"),
        ("ing1", "iŋ˥˥"),
        ("iong1", "ioŋ˥˥"),
        // The nasal marker comes off before the coda, or `ainn` reads as
        // `ai` + `n` and the nasalisation is silently lost.
        ("ann1", "ã˥˥"),
        ("ainn1", "ãĩ˥˥"),
        ("iann1", "ĩã˥˥"),
        ("uann1", "ũã˥˥"),
        ("iunn1", "ĩũ˥˥"),
        ("uainn1", "ũãĩ˥˥"),
        ("onn1", "õ˥˥"),
        ("annh4", "ãʔ˨"),
        // Syllabic nasals, alone and after an initial. `thng` is a syllable
        // and reading it as an unanalysable run drops it.
        ("m7", "m˧˧"),
        ("ng2", "ŋ˥˧"),
        ("hng1", "hŋ˥˥"),
        ("thng1", "tʰŋ˥˥"),
        ("tshng1", "tsʰŋ˥˥"),
        ("mh4", "mʔ˨"),
    ] {
        assert_eq!(tailo_to_ipa(tailo).text, ipa, "{tailo}");
    }
}

#[test]
fn syllables_run_together_with_no_separator() {
    // goruut is called with an empty separator, so the model has never seen a
    // space between phonemes - nor a hyphen, which is a syllable boundary in
    // romanisation and not a sound.
    let p = tailo_to_ipa("li2 ho2");
    assert_eq!(p.text, "li˥˧ho˥˧");
    assert_eq!(p.syllables, 2);
    assert_eq!(p.dropped, 0);
    assert_eq!(tailo_to_ipa("tai5-uan5").text, "tai˨˦uan˨˦");
}

#[test]
fn punctuation_is_dropped_rather_than_carried() {
    // The one deliberate divergence from goruut, and it is measured: what
    // survives its tokenizer is 0.337% of the training characters, with `,`
    // seen four times in the whole corpus. See the module header in `ipa.rs`.
    assert_eq!(tailo_to_ipa("li2 ho2, li2 ho2.").text, "li˥˧ho˥˧li˥˧ho˥˧");
    assert_eq!(tailo_to_ipa("li2 ho2。").text, "li˥˧ho˥˧");
}

#[test]
fn poj_reaches_ipa_in_one_call() {
    // The conversion the pipeline actually makes: the translator emits POJ and
    // the model eats IPA.
    let p = poj_to_ipa("lí hó");
    assert_eq!(p.text, "li˥˧ho˥˧");
    assert_eq!(p.syllables, 2);
    assert_eq!(
        poj_to_ipa("góa sī Tâi-oân-lâng").text,
        "\u{0261}ua˥˧si˧˧tai˨˦uan˨˦laŋ˨˦"
    );
}

#[test]
fn already_phonemised_text_passes_through() {
    // Without this a run of `li` inside `li˥˧` has no digit, takes the default
    // tone, and comes back `li˥˥˥˧`.
    let ipa = "li˥˧ho˥˧";
    let p = tailo_to_ipa(ipa);
    assert_eq!(p.text, ipa);
    assert_eq!(p.syllables, 0);
    assert_eq!(poj_to_ipa(ipa).text, ipa);
}

#[test]
fn what_is_not_a_syllable_is_dropped_and_counted() {
    // Dropped rather than passed on: a Latin letter *is* in this model's
    // vocabulary, so passing one through would put a phoneme in the sequence
    // rather than a gap. The count is what lets a caller notice.
    let p = tailo_to_ipa("li2 russia 42 ho2");
    assert_eq!(p.text, "li˥˧ho˥˧");
    assert_eq!(p.syllables, 2);
    assert_eq!(p.dropped, 2);

    // Han is not romanisation and never was.
    let p = tailo_to_ipa("你好");
    assert!(p.text.is_empty());
    assert_eq!(p.syllables, 0);
}

#[test]
fn an_empty_input_produces_nothing_rather_than_a_panic() {
    for s in ["", " ", "-", "。", "--"] {
        let p = tailo_to_ipa(s);
        assert!(p.text.is_empty(), "{s:?}");
        assert_eq!(p.syllables, 0, "{s:?}");
    }
}
