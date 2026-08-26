//! Proves the relative-position simplification against the reference's own
//! index gymnastics.
//!
//! `xabe_dsp::self_attention` does not implement the reference's three
//! pad-reshape-slice manoeuvres. It implements the closed form they compose to:
//! embedding entry `W + j - i`, inside a window of `|i - j| <= W`, zero
//! outside. That derivation is written out in `attention.rs`, and a derivation
//! is exactly the kind of thing that is convincing and wrong.
//!
//! So this file executes the reference's version - literally, pads and
//! reshapes and slices, on index arrays rather than values - and checks that
//! composing it gives the closed form, for every length from 1 to 32 and for
//! several window sizes. It needs no model and no capture, because it is a
//! statement about indices rather than about weights.

/// The window this model uses.
const WINDOW: usize = 4;

/// `_get_relative_embeddings`, on indices.
///
/// Returns, for each of the `2L-1` output rows, which row of the stored
/// `[2W+1, D]` table it reads - or `None` where the reference's zero padding
/// means it reads nothing.
fn get_relative_embeddings(window: usize, length: usize) -> Vec<Option<isize>> {
    let span = 2 * window + 1;
    let pad = (length as isize - (window as isize + 1)).max(0);
    // After padding, row `r` of the padded table is stored row `r - pad`.
    let padded: Vec<Option<isize>> = (0..span as isize + 2 * pad)
        .map(|r| {
            let src = r - pad;
            (src >= 0 && src < span as isize).then_some(src)
        })
        .collect();

    let start = ((window as isize + 1) - length as isize).max(0) as usize;
    let end = start + 2 * length - 1;
    padded[start..end].to_vec()
}

/// `_relative_position_to_absolute_position`, on indices.
///
/// Written the reference's way: pad the last axis by one, flatten, pad by
/// `L-1`, reshape to `[L+1, 2L-1]`, then slice `[:L, L-1:]`. The values carried
/// through are positions in the `[L, 2L-1]` input, so composing this with
/// `get_relative_embeddings` says which table row ends up at `(i, j)`.
fn relative_to_absolute(length: usize) -> Vec<Vec<Option<usize>>> {
    let l = length;
    // `[L, 2L]`: each row is the `2L-1` relative slots plus one pad column.
    let mut padded: Vec<Option<usize>> = Vec::with_capacity(l * 2 * l);
    for row in 0..l {
        for col in 0..2 * l {
            padded.push((col < 2 * l - 1).then_some(row * (2 * l - 1) + col));
        }
    }
    // Flatten and pad `L-1` at the end.
    padded.extend(std::iter::repeat_n(None, l - 1));
    // Reshape `[L+1, 2L-1]` and slice.
    (0..l)
        .map(|i| {
            (0..l)
                .map(|j| padded[i * (2 * l - 1) + (l - 1 + j)])
                .collect()
        })
        .collect()
}

// `i` and `j` are grid coordinates that appear in the arithmetic, not merely
// cursors into `abs`, so iterating the slice would lose the thing being tested.
#[allow(clippy::needless_range_loop)]
#[test]
fn the_reference_index_dance_composes_to_a_windowed_bias() {
    for window in [1usize, 2, 4, 8] {
        for length in 1usize..=32 {
            let table = get_relative_embeddings(window, length);
            let abs = relative_to_absolute(length);

            for i in 0..length {
                for j in 0..length {
                    // What the reference reads at (i, j), if anything.
                    let slot = abs[i][j].expect("the slice never lands on a pad");
                    // `slot` is a flat index into `[L, 2L-1]`; the relative
                    // axis is what selects the table row.
                    let rel = slot % (2 * length - 1);
                    let reference = table[rel];

                    // What `attention.rs` computes instead.
                    let r = window as isize + j as isize - i as isize;
                    let ours = (r >= 0 && r < 2 * window as isize + 1).then_some(r);

                    assert_eq!(
                        reference, ours,
                        "window {window}, length {length}, ({i}, {j})",
                    );
                }
            }
        }
    }
}

// `i` and `j` are grid coordinates that appear in the arithmetic, not merely
// cursors into `abs`, so iterating the slice would lose the thing being tested.
#[allow(clippy::needless_range_loop)]
#[test]
fn the_reference_reads_a_row_and_a_column_consistently() {
    // The composed map must also be *only* a function of `j - i`, which is what
    // makes it expressible as a window at all. Asserting it separately catches
    // a derivation that happens to agree on the diagonal.
    let length = 20;
    let table = get_relative_embeddings(WINDOW, length);
    let abs = relative_to_absolute(length);

    for i in 0..length {
        for j in 0..length {
            let rel = abs[i][j].unwrap() % (2 * length - 1);
            let d = j as isize - i as isize;
            let expected = if d.abs() <= WINDOW as isize {
                Some(WINDOW as isize + d)
            } else {
                None
            };
            assert_eq!(table[rel], expected, "({i}, {j}) has offset {d}");
        }
    }
}

#[test]
fn positions_outside_the_window_still_attend() {
    // The windowing applies to the positional bias, not to attention itself.
    // Reading the reference as sliding-window attention is an easy mistake -
    // the machinery looks exactly like one - and it would silently change what
    // the model can see. Nothing in the index maths above forbids `(0, 19)`
    // from being attended; it merely gets no positional bias.
    let length = 20;
    let abs = relative_to_absolute(length);
    let table = get_relative_embeddings(WINDOW, length);
    assert!(
        table[abs[0][length - 1].unwrap() % (2 * length - 1)].is_none(),
        "the far corner should have no positional bias",
    );
}
