// Captures whisper.cpp's VAD output as binary, for the differential tests.
//
//   g++ -O2 -std=c++17 -I $W/include -I $W/ggml/include \
//       tools/oracle/vad_capture.cpp -L $W/build/bin -lwhisper -o /tmp/vad_capture
//   LD_LIBRARY_PATH=$W/build/bin /tmp/vad_capture <vad.bin> <clip.wav> <out-dir>
//
// where $W is a built whisper.cpp checkout. It writes:
//
//   probs.bin      n_probs float32, one per 512 samples
//   segments.bin   2*n_segments float32, (start, end) in seconds
//
// Captured, never transcribed: the point of the differential tests is to
// compare against what the reference actually did, and a constant copied into
// a test is a copy of what someone believed it does.
//
// The reference here is whisper.cpp rather than Python silero-vad, deliberately
// and against the usual rule that the upstream author's implementation is the
// oracle. Every threshold in the pipeline was tuned against whisper.cpp's
// probabilities, and whisper.cpp differs from upstream on purpose: it parses
// n_context and then ignores it, substituting a reflective pad. Matching Python
// instead would invalidate the tuning. See docs/MODEL.md.

#include "whisper.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// A minimal 16-bit mono WAV reader. The corpus is written by this repository,
// so it need only read what this repository writes - and walking the chunk list
// is still cheaper than linking the examples library for it.
static bool read_wav(const char * path, std::vector<float> & out, uint32_t & rate) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (fread(b.data(), 1, n, f) != (size_t) n) { fclose(f); return false; }
    fclose(f);

    if (n < 12 || memcmp(b.data(), "RIFF", 4) || memcmp(b.data() + 8, "WAVE", 4)) {
        fprintf(stderr, "%s is not RIFF/WAVE\n", path);
        return false;
    }
    uint16_t channels = 0, bits = 0;
    const uint8_t * data = nullptr;
    uint32_t data_len = 0;
    size_t at = 12;
    while (at + 8 <= (size_t) n) {
        uint32_t size;
        memcpy(&size, b.data() + at + 4, 4);
        const uint8_t * body = b.data() + at + 8;
        if (at + 8 + size > (size_t) n) break;
        if (!memcmp(b.data() + at, "fmt ", 4) && size >= 16) {
            memcpy(&channels, body + 2, 2);
            memcpy(&rate,     body + 4, 4);
            memcpy(&bits,     body + 14, 2);
        } else if (!memcmp(b.data() + at, "data", 4)) {
            data = body;
            data_len = size;
        }
        at += 8 + size + (size & 1);   // chunks are word-aligned
    }
    if (!data || channels != 1 || bits != 16) {
        fprintf(stderr, "%s: need 16-bit mono, got %u channels %u bits\n", path, channels, bits);
        return false;
    }
    out.resize(data_len / 2);
    for (size_t i = 0; i < out.size(); i++) {
        int16_t s;
        memcpy(&s, data + 2 * i, 2);
        out[i] = s / 32768.0f;
    }
    return true;
}

static bool write_f32(const std::string & path, const float * v, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return false; }
    bool ok = fwrite(v, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <vad-model.bin> <clip.wav> <out-dir>\n", argv[0]);
        return 1;
    }
    const char * model = argv[1];
    const char * clip  = argv[2];
    std::string  out   = argv[3];

    std::vector<float> samples;
    uint32_t rate = 0;
    if (!read_wav(clip, samples, rate)) return 1;
    if (rate != 16000) {
        fprintf(stderr, "%s is %u Hz; the VAD is 16 kHz only\n", clip, rate);
        return 1;
    }

    auto cparams = whisper_vad_default_context_params();
    cparams.use_gpu = false;   // the CPU path is the reference
    cparams.n_threads = 1;     // float32 reduction order is not thread-invariant

    whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(model, cparams);
    if (!vctx) { fprintf(stderr, "cannot load %s\n", model); return 1; }

    if (!whisper_vad_detect_speech(vctx, samples.data(), (int) samples.size())) {
        fprintf(stderr, "detect_speech failed\n");
        return 1;
    }

    const int     n_probs = whisper_vad_n_probs(vctx);
    const float * probs   = whisper_vad_probs(vctx);
    if (!write_f32(out + "/probs.bin", probs, n_probs)) return 1;

    auto vparams = whisper_vad_default_params();
    // The pipeline's own settings, from run.sh - the segmenter is only worth
    // comparing at the thresholds it actually runs with.
    vparams.threshold               = 0.6f;
    vparams.min_speech_duration_ms  = 250;
    vparams.min_silence_duration_ms = 200;

    whisper_vad_segments * segs = whisper_vad_segments_from_probs(vctx, vparams);
    std::vector<float> flat;
    if (segs) {
        const int n = whisper_vad_segments_n_segments(segs);
        for (int i = 0; i < n; i++) {
            flat.push_back(whisper_vad_segments_get_segment_t0(segs, i) / 100.0f);
            flat.push_back(whisper_vad_segments_get_segment_t1(segs, i) / 100.0f);
        }
        whisper_vad_free_segments(segs);
    }
    if (!write_f32(out + "/segments.bin", flat.data(), flat.size())) return 1;

    printf("%s: %d samples, %d probs, %zu segments\n",
           clip, (int) samples.size(), n_probs, flat.size() / 2);

    whisper_vad_free(vctx);
    return 0;
}
