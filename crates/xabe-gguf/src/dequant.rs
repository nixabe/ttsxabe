//! Block-quantized ggml formats, unpacked to f32.
//!
//! # Why these are transcribed rather than invented
//!
//! Every layout here is read off `gguf-py/gguf/quants.py` in upstream
//! llama.cpp, which is the same code that *wrote* the files being read. The
//! numpy there expresses each format as a reshape-and-shift over whole blocks;
//! the loops below walk one element at a time and therefore have to get the
//! element *ordering* right by hand, which is the only hard part and the only
//! place a plausible-looking mistake hides.
//!
//! Take Q4_0. The reference does
//!
//! ```text
//! qs.reshape((n, -1, 1, 16)) >> [0, 4]   ->   (n, 1, 2, 16)   ->   (n, 32)
//! ```
//!
//! so the first sixteen values of a block are the **low** nibbles of bytes
//! 0..16 and the last sixteen are the **high** nibbles of the same bytes. The
//! obvious reading - low then high of byte 0, low then high of byte 1 - is
//! wrong and produces a tensor that is a permutation of the right one. Every
//! format here has a trap of that shape, so every one is checked against
//! `gguf-py`'s own output on random data rather than reasoned about; see
//! `tests/dequant.rs`.
//!
//! # What is not here
//!
//! The `IQ*` formats, `Q8_K`, and `TQ*`. `Q8_K` is an intermediate used during
//! quantization and never appears in a stored tensor. The rest are refused by
//! name and id, as the whole table used to be.

use crate::error::GgufError;
use crate::types::GgmlType;

/// Reads a little-endian f16 at `b[i..i + 2]` as f32.
#[inline]
fn f16(b: &[u8], i: usize) -> f32 {
    f32::from(half::f16::from_le_bytes([b[i], b[i + 1]]))
}

/// The 6-bit scale and minimum pairs shared by Q4_K and Q5_K.
///
/// Twelve bytes hold eight scales and eight minimums at six bits each, packed
/// as the reference comments:
///
/// ```text
///  0 EEAAAAAA   4 eeaaaaaa   8 eeeeEEEE
///  1 FFBBBBBB   5 ffbbbbbb   9 ffffFFFF
///  2 GGCCCCCC   6 ggcccccc  10 ggggGGGG
///  3 HHDDDDDD   7 hhhhdddd  11 hhhhHHHH
/// ```
///
/// The first four scales are the low six bits of bytes 0..4; the last four are
/// the low nibble of bytes 8..12 with two more bits borrowed from the top of
/// bytes 0..4. Minimums mirror that through bytes 4..8 and the high nibbles.
fn scale_min(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = s[j] & 0x3F;
        mn[j] = s[j + 4] & 0x3F;
        sc[j + 4] = (s[j + 8] & 0x0F) | ((s[j] >> 2) & 0x30);
        mn[j + 4] = (s[j + 8] >> 4) | ((s[j + 4] >> 2) & 0x30);
    }
    (sc, mn)
}

/// Unpacks `n` elements of `ty` from `raw` into `out`.
///
/// `out` is filled exactly, and `raw` must hold whole blocks - both are
/// guaranteed by the caller, which sized them from the same tensor entry.
pub(crate) fn dequantize(ty: GgmlType, raw: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    let bs = ty.block_size() as usize;
    let ts = ty.type_size() as usize;
    let blocks = n / bs;
    let mut out = vec![0.0f32; n];

    for b in 0..blocks {
        let blk = &raw[b * ts..(b + 1) * ts];
        let dst = &mut out[b * bs..(b + 1) * bs];
        match ty {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
                unreachable!("the unquantized widths never reach the block path")
            }

            // d(2) + 16 packed nibbles. Low nibbles first, then high.
            GgmlType::Q4_0 => {
                let d = f16(blk, 0);
                let qs = &blk[2..];
                for (j, slot) in dst.iter_mut().enumerate() {
                    let nib = if j < 16 {
                        qs[j] & 0x0F
                    } else {
                        qs[j - 16] >> 4
                    };
                    *slot = d * (i32::from(nib) - 8) as f32;
                }
            }

            // d(2) + m(2) + nibbles. Offset rather than centred: no -8.
            GgmlType::Q4_1 => {
                let (d, m) = (f16(blk, 0), f16(blk, 2));
                let qs = &blk[4..];
                for (j, slot) in dst.iter_mut().enumerate() {
                    let nib = if j < 16 {
                        qs[j] & 0x0F
                    } else {
                        qs[j - 16] >> 4
                    };
                    *slot = d * f32::from(nib) + m;
                }
            }

            // d(2) + qh(4, one bit per element) + nibbles.
            GgmlType::Q5_0 => {
                let d = f16(blk, 0);
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..];
                for (j, slot) in dst.iter_mut().enumerate() {
                    let lo = if j < 16 {
                        qs[j] & 0x0F
                    } else {
                        qs[j - 16] >> 4
                    };
                    let hi = ((qh >> j) & 1) as u8;
                    *slot = d * (i32::from(lo | (hi << 4)) - 16) as f32;
                }
            }

            GgmlType::Q5_1 => {
                let (d, m) = (f16(blk, 0), f16(blk, 2));
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                let qs = &blk[8..];
                for (j, slot) in dst.iter_mut().enumerate() {
                    let lo = if j < 16 {
                        qs[j] & 0x0F
                    } else {
                        qs[j - 16] >> 4
                    };
                    let hi = ((qh >> j) & 1) as u8;
                    *slot = d * f32::from(lo | (hi << 4)) + m;
                }
            }

            // The simple one: a scale and 32 signed bytes.
            GgmlType::Q8_0 => {
                let d = f16(blk, 0);
                for (j, slot) in dst.iter_mut().enumerate() {
                    *slot = d * f32::from(blk[2 + j] as i8);
                }
            }

            // scales(16) + qs(64) + d(2) + dmin(2). Two bits per element, one
            // 4-bit scale and one 4-bit minimum per group of sixteen.
            GgmlType::Q2K => {
                let scales = &blk[..16];
                let qs = &blk[16..80];
                let (d, dmin) = (f16(blk, 80), f16(blk, 82));
                for (flat, slot) in dst.iter_mut().enumerate() {
                    let g = flat / 16;
                    let dl = d * f32::from(scales[g] & 0x0F);
                    let ml = dmin * f32::from(scales[g] >> 4);
                    let (hi, r) = (flat / 128, flat % 128);
                    let (s, k) = (r / 32, r % 32);
                    let q = (qs[hi * 32 + k] >> (2 * s)) & 3;
                    *slot = dl * f32::from(q) - ml;
                }
            }

            // hmask(32) + qs(64) + scales(12) + d(2). Six-bit scales split
            // across two runs, and a high bit whose sense is inverted.
            GgmlType::Q3K => {
                let hmask = &blk[..32];
                let qs = &blk[32..96];
                let sc = &blk[96..108];
                let d = f16(blk, 108);

                // 0..8 are low nibbles of bytes 0..8, then high nibbles of the
                // same; the top two bits come from bytes 8..12, two per byte.
                let mut scales = [0i8; 16];
                for (i, s) in scales.iter_mut().enumerate() {
                    let low = if i < 8 { sc[i] & 0x0F } else { sc[i - 8] >> 4 };
                    let high = (sc[8 + (i % 4)] >> (2 * (i / 4))) & 0x03;
                    *s = ((low | (high << 4)) as i8) - 32;
                }

                for (flat, slot) in dst.iter_mut().enumerate() {
                    let g = flat / 16;
                    let (hi, r) = (flat / 128, flat % 128);
                    let (s, k) = (r / 32, r % 32);
                    let ql = (qs[hi * 32 + k] >> (2 * s)) & 3;
                    // The reference notes it too: the offset is applied when
                    // the mask bit is *zero*, so the bit is inverted first.
                    let qh = ((hmask[k] >> (flat / 32)) & 1) ^ 1;
                    let q = ql as i8 - ((qh << 2) as i8);
                    *slot = d * f32::from(scales[g]) * f32::from(q);
                }
            }

            // d(2) + dmin(2) + scales(12) + qs(128). Eight sub-blocks of 32,
            // each with its own six-bit scale and minimum.
            GgmlType::Q4K => {
                let (d, dmin) = (f16(blk, 0), f16(blk, 2));
                let (sc, mn) = scale_min(&blk[4..16]);
                let qs = &blk[16..144];
                for (flat, slot) in dst.iter_mut().enumerate() {
                    let j = flat / 32;
                    let (hi, r) = (flat / 64, flat % 64);
                    let (s, k) = (r / 32, r % 32);
                    let q = (qs[hi * 32 + k] >> (4 * s)) & 0x0F;
                    *slot = d * f32::from(sc[j]) * f32::from(q) - dmin * f32::from(mn[j]);
                }
            }

            // Q4_K plus one high bit per element.
            GgmlType::Q5K => {
                let (d, dmin) = (f16(blk, 0), f16(blk, 2));
                let (sc, mn) = scale_min(&blk[4..16]);
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                for (flat, slot) in dst.iter_mut().enumerate() {
                    let j = flat / 32;
                    let (hi, r) = (flat / 64, flat % 64);
                    let (s, k) = (r / 32, r % 32);
                    let lo = (qs[hi * 32 + k] >> (4 * s)) & 0x0F;
                    let bit = (qh[flat % 32] >> j) & 1;
                    *slot =
                        d * f32::from(sc[j]) * f32::from(lo | (bit << 4)) - dmin * f32::from(mn[j]);
                }
            }

            // ql(128) + qh(64) + scales(16, signed) + d(2). Note the low part
            // runs in groups of 64 and the high part in groups of 32.
            GgmlType::Q6K => {
                let ql = &blk[..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208];
                let d = f16(blk, 208);
                for (flat, slot) in dst.iter_mut().enumerate() {
                    let g = flat / 16;
                    let (hi, r) = (flat / 128, flat % 128);
                    let (sl, k64) = (r / 64, r % 64);
                    let (sh, k32) = (r / 32, r % 32);
                    let lo = (ql[hi * 64 + k64] >> (4 * sl)) & 0x0F;
                    let bits = (qh[hi * 32 + k32] >> (2 * sh)) & 0x03;
                    let q = ((lo | (bits << 4)) as i8) - 32;
                    *slot = d * f32::from(scales[g] as i8) * f32::from(q);
                }
            }
        }
    }
    Ok(out)
}
