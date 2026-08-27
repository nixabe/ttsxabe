//! The DiT estimator against CosyVoice3's own, one evaluation at a time.
//!
//! The solver runs ten of these and a mistake in any one looks identical from
//! outside, so the estimator is compared on its own first - at the solver's
//! first timestep, with the classifier-free pair batched exactly as
//! `solve_euler` batches it: row 0 conditioned, row 1 with the conditioning
//! zeroed.
//!
//! That the two rows differ is itself worth asserting. Classifier-free
//! guidance extrapolates *between* them, so an estimator that ignored its
//! conditioning would give two identical rows, a zero difference, and a
//! guidance term that does nothing - and would still produce a mel.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn npy(p: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let (hlen, at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let head = std::str::from_utf8(&b[at..at + hlen]).expect("ascii");
    // See `tools/oracle/capture_cosyvoice.py`: a Fortran-order file has the
    // right shape in its header and its axes the other way round in its bytes,
    // so a reader that trusts the shape gets a permutation of the right values.
    assert!(
        head.contains("'fortran_order': False"),
        "{}: fortran order",
        p.display()
    );
    let o = head.find('(').expect("shape");
    let c = head[o..].find(')').expect("shape") + o;
    let shape: Vec<usize> = head[o + 1..c]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("dim"))
        .collect();
    let v = b[at + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (shape, v)
}

fn capture() -> Option<PathBuf> {
    let d = root().join(".golden/cosyvoice");
    d.join("dit_step0.npy").is_file().then_some(d)
}

fn device() -> Option<usize> {
    std::env::var("XABE_COSY_DEVICE").ok()?.parse().ok()
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

#[test]
fn one_estimator_evaluation_matches_the_reference() {
    let (Some(dir), Some(dev)) = (capture(), device()) else {
        println!(
            "SKIP: needs .golden/cosyvoice (tools/oracle/capture_cosyvoice.py) \
             and XABE_COSY_DEVICE=<free card>"
        );
        return;
    };
    let model = root().join("models/tts/cosyvoice3-0.5b/flow.safetensors");
    if !model.is_file() {
        println!("SKIP: run tools/convert_cosyvoice.py");
        return;
    }

    let flow = xabe_cosy::Flow::open(&model, dev).expect("open the flow");
    let m = flow.config().mel_dim;

    let (noise_shape, noise) = npy(&dir.join("cfm_noise.npy"));
    let (_, mu) = npy(&dir.join("flow_mu.npy"));
    let (_, cond) = npy(&dir.join("flow_cond.npy"));
    let (_, spk) = npy(&dir.join("spk80.npy"));
    let (_, t_span) = npy(&dir.join("t_span.npy"));
    let (want_shape, want) = npy(&dir.join("dit_step0.npy"));

    let n = *noise_shape.last().expect("frames");
    assert_eq!(want_shape, vec![2, m, n], "the capture is a batch of two");

    // The cosine schedule, checked against the reference's own rather than
    // assumed: an off-by-one in the linspace shifts every timestep.
    let mine = flow.t_span();
    assert_eq!(mine.len(), t_span.len());
    for (a, b) in mine.iter().zip(&t_span) {
        assert!((a - b).abs() < 1e-6, "t_span differs: {a} against {b}");
    }

    // The classifier-free pair, exactly as `solve_euler` builds it: both rows
    // get the noise, and only row 0 gets the conditioning.
    let mut x2 = vec![0.0f32; 2 * m * n];
    x2[..m * n].copy_from_slice(&noise);
    x2[m * n..].copy_from_slice(&noise);
    let mut mu2 = vec![0.0f32; 2 * m * n];
    mu2[..m * n].copy_from_slice(&mu);
    let mut cond2 = vec![0.0f32; 2 * m * n];
    cond2[..m * n].copy_from_slice(&cond);
    let mut spk2 = vec![0.0f32; 2 * m];
    spk2[..m].copy_from_slice(&spk);

    let got = flow
        .estimate(&x2, &mu2, &cond2, &spk2, t_span[0], n)
        .expect("estimate");
    assert_eq!(got.len(), want.len());

    for (row, name) in [(0usize, "conditioned"), (1, "unconditioned")] {
        let a = &got[row * m * n..(row + 1) * m * n];
        let b = &want[row * m * n..(row + 1) * m * n];
        let corr = correlation(a, b);
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let rms =
            (b.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / b.len() as f64).sqrt();
        println!("  {name:14} correlation {corr:.6}, worst {worst:.5}, rms {rms:.5}");
        assert!(corr > 0.999, "{name} correlates {corr:.6}");
        // Measured at correlation 0.999967 with a worst element of 0.35
        // against an rms of 6.14. The bound is on the worst *element* and is
        // loose for a reason worth writing down: the residual stream through
        // twenty-two blocks carries a handful of activations near 280 while
        // its rms is 8.7, and float32 error concentrates on exactly those.
        // Tightening this to the rms-relative figure would be measuring
        // something the arithmetic does not promise; loosening it further
        // would stop catching a wrong block.
        assert!(worst < 0.6, "{name} worst element differs by {worst}");
    }

    // The two rows must actually differ, or the guidance term is a no-op and
    // the estimator is ignoring what it is conditioned on.
    let spread = correlation(&got[..m * n], &got[m * n..]);
    println!("  the two rows correlate {spread:.4} with each other");
    assert!(
        spread < 0.99,
        "the conditioned and unconditioned rows are {spread:.4} alike; \
         classifier-free guidance would do nothing"
    );
}

#[test]
fn the_whole_flow_reproduces_the_reference_mel() {
    // The estimator above is one evaluation. This is all of it: the token
    // embedding, the look-ahead pair, the speaker projection, the condition,
    // and ten Euler steps of classifier-free guidance - against the mel the
    // reference handed its vocoder.
    //
    // Worth having separately, because everything the estimator test does not
    // touch lives in front of it, and a wrong `repeat_interleave` or a
    // condition laid at the wrong end of the timeline produces an estimator
    // that is exactly right and a mel that is not.
    let (Some(dir), Some(dev)) = (capture(), device()) else {
        println!("SKIP: see the test above");
        return;
    };
    let model = root().join("models/tts/cosyvoice3-0.5b/flow.safetensors");
    if !model.is_file() {
        println!("SKIP: run tools/convert_cosyvoice.py");
        return;
    }

    let flow = xabe_cosy::Flow::open(&model, dev).expect("open the flow");
    let m = flow.config().mel_dim;

    let ids = |name: &str| -> Vec<u32> {
        npy(&dir.join(name))
            .1
            .iter()
            .map(|v| v.round() as u32)
            .collect()
    };
    let prompt_tokens = ids("flow_prompt_speech_token.npy");
    let tokens = ids("speech_token.npy");
    let (feat_shape, prompt_feat) = npy(&dir.join("prompt_speech_feat.npy"));
    let (_, embedding) = npy(&dir.join("flow_embedding.npy"));
    let (_, noise) = npy(&dir.join("cfm_noise.npy"));
    let (want_shape, want) = npy(&dir.join("mel.npy"));

    // The bundle's own arithmetic, asserted here because the flow's condition
    // depends on it: two mel frames per speech token, and the generated part
    // starts exactly where the prompt's mel ends.
    assert_eq!(feat_shape, vec![1, prompt_tokens.len() * 2, m]);
    let frames = (prompt_tokens.len() + tokens.len()) * 2 - prompt_tokens.len() * 2;
    assert_eq!(want_shape, vec![1, m, frames], "the reference mel");

    let (got, got_frames) = flow
        .mel(&prompt_tokens, &tokens, &prompt_feat, &embedding, &noise)
        .expect("mel");
    assert_eq!(got_frames, frames, "generated frames");
    assert_eq!(got.len(), want.len());

    let corr = correlation(&got, &want);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = (want
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / want.len() as f64)
        .sqrt();
    println!("  mel: correlation {corr:.6}, worst {worst:.5}, rms {rms:.5}");

    // Correlation first, because it is what separates "the arithmetic rounds
    // differently over ten steps" from "the timeline is assembled wrong". A
    // condition at the wrong end, or a `repeat_interleave` along the wrong
    // axis, lands far from 1 here while leaving the rms untouched.
    assert!(corr > 0.9999, "the mel correlates {corr:.6}");
    assert!(worst < 0.5, "worst mel bin differs by {worst}");
    assert!(got.iter().all(|v| v.is_finite()), "a mel bin is not finite");
}
