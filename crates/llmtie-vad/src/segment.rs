//! Turning a sequence of per-frame probabilities into speech segments.
//!
//! This is a port of `whisper_vad_segments_from_probs`, and the fidelity
//! matters more than it looks: every threshold the pipeline runs with was tuned
//! against *this* segmenter's output, so a segmenter that is merely reasonable
//! would invalidate the tuning without failing any test.
//!
//! Two of the rules are whisper.cpp's own additions rather than upstream
//! Silero's - the 200 ms merge and the second minimum-duration sweep - and are
//! marked below so a future comparison against Python silero-vad knows where
//! the two diverge on purpose.

/// Frames per second of audio the probabilities were computed from.
const SAMPLE_RATE: usize = 16_000;

/// Samples per probability.
const WINDOW: usize = crate::weights::WINDOW;

/// How the probabilities become segments.
#[derive(Debug, Clone, Copy)]
pub struct SegmentParams {
    /// Probability at or above which a frame is speech.
    pub threshold: f32,
    /// Shortest segment kept, in milliseconds.
    pub min_speech_duration_ms: usize,
    /// Silence needed to end a segment, in milliseconds.
    pub min_silence_duration_ms: usize,
    /// Longest segment before it is split, in seconds.
    pub max_speech_duration_s: f32,
    /// Padding added around each segment, in milliseconds.
    pub speech_pad_ms: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        // whisper.cpp's defaults, except the threshold, which run.sh raises to
        // 0.6 because 0.5 let room noise through on this microphone.
        SegmentParams {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            max_speech_duration_s: f32::MAX,
            speech_pad_ms: 30,
        }
    }
}

/// A span of speech, in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// First sample.
    pub start: usize,
    /// One past the last sample.
    pub end: usize,
}

impl Segment {
    /// Start, in seconds.
    pub fn start_s(&self) -> f64 {
        self.start as f64 / SAMPLE_RATE as f64
    }

    /// End, in seconds.
    pub fn end_s(&self) -> f64 {
        self.end as f64 / SAMPLE_RATE as f64
    }
}

/// Finds speech segments in a probability sequence.
pub fn segments(probs: &[f32], params: SegmentParams) -> Vec<Segment> {
    let min_silence_samples = SAMPLE_RATE * params.min_silence_duration_ms / 1000;
    let min_speech_samples = SAMPLE_RATE * params.min_speech_duration_ms / 1000;
    let speech_pad_samples = SAMPLE_RATE * params.speech_pad_ms / 1000;
    let audio_length = probs.len() * WINDOW;

    let max_speech_samples = if params.max_speech_duration_s > 100_000.0 {
        usize::MAX / 2
    } else {
        let temp = SAMPLE_RATE as i64 * params.max_speech_duration_s as i64
            - WINDOW as i64
            - 2 * speech_pad_samples as i64;
        if temp < 0 {
            usize::MAX / 2
        } else {
            temp as usize
        }
    };

    // 98 ms, taken from upstream silero-vad. It marks where a segment *could*
    // be split if it ever runs past max_speech_samples, which is a different
    // question from whether the segment has ended.
    let min_silence_at_max_speech = SAMPLE_RATE * 98 / 1000;

    // Hysteresis. Ending on the same threshold that started the segment chops
    // a turn up at every unvoiced consonant.
    let neg_threshold = (params.threshold - 0.15).max(0.01);

    let mut speeches: Vec<Segment> = Vec::new();
    let mut in_speech = false;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    let mut speech_start = 0usize;
    let mut has_speech = false;

    for (i, &prob) in probs.iter().enumerate() {
        let sample = WINDOW * i;

        // Back above the threshold: whatever silence had begun is not the end.
        if prob >= params.threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = sample;
            }
        }

        if prob >= params.threshold && !in_speech {
            in_speech = true;
            speech_start = sample;
            has_speech = true;
            continue;
        }

        if in_speech && sample.saturating_sub(speech_start) > max_speech_samples {
            if prev_end != 0 {
                speeches.push(Segment {
                    start: speech_start,
                    end: prev_end,
                });
                has_speech = true;
                if next_start < prev_end {
                    in_speech = false;
                    has_speech = false;
                } else {
                    speech_start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push(Segment {
                    start: speech_start,
                    end: sample,
                });
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                in_speech = false;
                has_speech = false;
                continue;
            }
        }

        if prob < neg_threshold && in_speech {
            if temp_end == 0 {
                temp_end = sample;
            }
            if sample.saturating_sub(temp_end) > min_silence_at_max_speech {
                prev_end = temp_end;
            }
            if sample.saturating_sub(temp_end) < min_silence_samples {
                continue;
            }
            if temp_end.saturating_sub(speech_start) > min_speech_samples {
                speeches.push(Segment {
                    start: speech_start,
                    end: temp_end,
                });
            }
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            in_speech = false;
            has_speech = false;
            continue;
        }
    }

    // Still speaking when the audio ran out.
    if has_speech && audio_length.saturating_sub(speech_start) > min_speech_samples {
        speeches.push(Segment {
            start: speech_start,
            end: audio_length,
        });
    }

    // whisper.cpp's addition, not upstream Silero's: adjacent segments closer
    // than 200 ms are one segment. Without it a single sentence arrives as
    // several transcription requests and the ASR loses the context between them.
    let max_merge_gap = SAMPLE_RATE * 200 / 1000;
    let mut i = 0;
    while i + 1 < speeches.len() {
        if speeches[i + 1].start.saturating_sub(speeches[i].end) < max_merge_gap {
            speeches[i].end = speeches[i + 1].end;
            speeches.remove(i + 1);
        } else {
            i += 1;
        }
    }

    // Also whisper.cpp's: a second minimum-duration sweep, after merging. The
    // first sweep can pass a short segment that the merge then fails to absorb.
    speeches.retain(|s| s.end - s.start >= min_speech_samples);

    pad(&mut speeches, speech_pad_samples, audio_length);
    speeches
}

/// Widens each segment, splitting the difference where two are close together.
fn pad(speeches: &mut [Segment], pad: usize, audio_length: usize) {
    for i in 0..speeches.len() {
        if i == 0 {
            speeches[i].start = speeches[i].start.saturating_sub(pad);
        }
        if i + 1 < speeches.len() {
            let gap = speeches[i + 1].start.saturating_sub(speeches[i].end);
            if gap < 2 * pad {
                // Too close to pad both fully without overlapping, so the gap
                // is split. Overlapping segments would transcribe the same
                // audio twice and duplicate words across the boundary.
                speeches[i].end += gap / 2;
                speeches[i + 1].start = speeches[i + 1].start.saturating_sub(gap / 2);
            } else {
                speeches[i].end = (speeches[i].end + pad).min(audio_length);
                speeches[i + 1].start = speeches[i + 1].start.saturating_sub(pad);
            }
        } else {
            speeches[i].end = (speeches[i].end + pad).min(audio_length);
        }
    }
}
