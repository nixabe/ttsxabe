//! The zip container a modern torch checkpoint is.
//!
//! # What is (and isn't) here
//!
//! Enough of the zip format to say where an entry's bytes are: the end-of-
//! central-directory record, the central directory, the zip64 extensions that
//! any checkpoint above 4 GiB needs, and the local header that has to be read
//! to skip its own variable-length fields. Nothing decompresses, because
//! nothing is compressed: torch stores every entry and aligns the storages so
//! that they can be mapped rather than inflated. An entry that *is* compressed
//! is refused by name.
//!
//! This module knows a byte range. It has no idea a tensor lives there - that
//! is [`crate::file`], which reads the pickle that says so.

use std::path::Path;

use crate::error::PtError;

/// End of central directory.
const EOCD_SIG: u32 = 0x0605_4b50;
/// Zip64 end of central directory locator.
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4b50;
/// Zip64 end of central directory.
const EOCD64_SIG: u32 = 0x0606_4b50;
/// A central directory file header.
const CENTRAL_SIG: u32 = 0x0201_4b50;
/// A local file header.
const LOCAL_SIG: u32 = 0x0403_4b50;

/// Fixed size of the end-of-central-directory record, before its comment.
const EOCD_LEN: usize = 22;
/// Fixed size of a local file header, before its name and extra fields.
const LOCAL_HEADER_LEN: usize = 30;
/// Size of the zip64 end-of-central-directory locator, which sits immediately
/// before the ordinary record when there is one.
const ZIP64_LOCATOR_LEN: usize = 20;
/// The only compression method this reader accepts.
const STORED: u16 = 0;
/// Tag of the zip64 extended-information extra field.
const ZIP64_EXTRA: u16 = 1;

/// Where one archive entry's data begins and how long it is.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The entry's name, as the archive spells it.
    pub name: String,
    /// Offset of the entry's data within the whole mapping.
    pub start: usize,
    /// Length of that data in bytes.
    pub len: usize,
}

/// Reads the central directory and resolves every entry to a byte range.
///
/// The local headers are read too, because the central directory records where
/// an entry's *header* starts and the header's own name and extra fields are
/// variable-length - so the data offset cannot be known without it.
pub fn entries(path: &Path, map: &[u8]) -> Result<Vec<Entry>, PtError> {
    let eocd = find_eocd(path, map)?;
    let (count, dir_start) = directory(path, map, eocd)?;

    let mut out = Vec::with_capacity(count);
    let mut at = dir_start;
    for _ in 0..count {
        let (entry, next) = central_entry(path, map, at)?;
        out.push(entry);
        at = next;
    }
    Ok(out)
}

/// Scans backwards for the end-of-central-directory signature.
///
/// The record is last in the file unless the archive carries a comment, which
/// torch's writer does not - but the scan is written the general way so that a
/// checkpoint rewritten by some other zip tool still opens.
fn find_eocd(path: &Path, map: &[u8]) -> Result<usize, PtError> {
    if map.len() < EOCD_LEN {
        return Err(PtError::NotAnArchive {
            path: path.to_path_buf(),
        });
    }
    // A zip comment is a 16-bit length, so the record cannot be further back
    // than 64 KiB plus its own size.
    let horizon = map.len().saturating_sub(EOCD_LEN + u16::MAX as usize);
    for at in (horizon..=map.len() - EOCD_LEN).rev() {
        if u32(map, at) == Some(EOCD_SIG) {
            return Ok(at);
        }
    }
    Err(PtError::NotAnArchive {
        path: path.to_path_buf(),
    })
}

/// Returns the entry count and the offset of the central directory, following
/// the zip64 records when the 32-bit fields are saturated.
fn directory(path: &Path, map: &[u8], eocd: usize) -> Result<(usize, usize), PtError> {
    // The zip64 record, when there is one, is authoritative. Deciding by
    // saturation instead would be wrong at exactly 65,535 entries, which is a
    // legal count and not a marker - and a checkpoint of a large enough model
    // can reach it.
    if let Some(v) = zip64_directory(path, map, eocd)? {
        return Ok(v);
    }
    let count = u16(map, eocd + 10).ok_or_else(|| malformed(path, "end-of-central-directory"))?;
    let start = u32(map, eocd + 16).ok_or_else(|| malformed(path, "end-of-central-directory"))?;
    Ok((count as usize, start as usize))
}

/// Follows the zip64 locator, if the archive has one.
///
/// `Ok(None)` means it does not, which is the ordinary case: a checkpoint under
/// 4 GiB with under 65,535 entries needs none. Any checkpoint above 4 GiB - the
/// 13 B translator would be one - needs it, so this is written rather than
/// assumed unreachable.
fn zip64_directory(
    path: &Path,
    map: &[u8],
    eocd: usize,
) -> Result<Option<(usize, usize)>, PtError> {
    let Some(locator) = eocd.checked_sub(ZIP64_LOCATOR_LEN) else {
        return Ok(None);
    };
    if u32(map, locator) != Some(EOCD64_LOCATOR_SIG) {
        return Ok(None);
    }
    let eocd64 = u64(map, locator + 8).ok_or_else(|| malformed(path, "zip64 locator"))? as usize;
    if u32(map, eocd64) != Some(EOCD64_SIG) {
        return Err(malformed(path, "zip64 end-of-central-directory"));
    }
    let count = u64(map, eocd64 + 32).ok_or_else(|| malformed(path, "zip64 directory"))?;
    let start = u64(map, eocd64 + 48).ok_or_else(|| malformed(path, "zip64 directory"))?;
    Ok(Some((count as usize, start as usize)))
}

/// Reads one central directory record, and follows it to the entry's data.
///
/// Returns the entry and the offset of the next record.
fn central_entry(path: &Path, map: &[u8], at: usize) -> Result<(Entry, usize), PtError> {
    if u32(map, at) != Some(CENTRAL_SIG) {
        return Err(malformed(path, format!("central directory record at {at}")));
    }
    let method = u16(map, at + 10).ok_or_else(|| malformed(path, "central record"))?;
    let compressed = u32(map, at + 20).ok_or_else(|| malformed(path, "central record"))?;
    let uncompressed = u32(map, at + 24).ok_or_else(|| malformed(path, "central record"))?;
    let name_len = u16(map, at + 28).ok_or_else(|| malformed(path, "central record"))? as usize;
    let extra_len = u16(map, at + 30).ok_or_else(|| malformed(path, "central record"))? as usize;
    let comment_len = u16(map, at + 32).ok_or_else(|| malformed(path, "central record"))? as usize;
    let header = u32(map, at + 42).ok_or_else(|| malformed(path, "central record"))?;

    let name_at = at + 46;
    let name = map
        .get(name_at..name_at + name_len)
        .ok_or_else(|| malformed(path, "central record name"))?;
    let name = String::from_utf8_lossy(name).into_owned();

    if method != STORED {
        return Err(PtError::Compressed { name, method });
    }

    let extra = map
        .get(name_at + name_len..name_at + name_len + extra_len)
        .ok_or_else(|| malformed(path, "central record extra field"))?;
    let (size, local) = widen(path, &name, extra, compressed, uncompressed, header)?;

    if u32(map, local) != Some(LOCAL_SIG) {
        return Err(malformed(path, format!("local header for {name}")));
    }
    let local_name_len =
        u16(map, local + 26).ok_or_else(|| malformed(path, "local header"))? as usize;
    let local_extra_len =
        u16(map, local + 28).ok_or_else(|| malformed(path, "local header"))? as usize;
    let start = local + LOCAL_HEADER_LEN + local_name_len + local_extra_len;
    let len = size as usize;
    // Checked, because `size` comes out of the file: a corrupt length near
    // `u64::MAX` would wrap the sum and pass a plain comparison.
    if start.checked_add(len).is_none_or(|end| end > map.len()) {
        return Err(malformed(
            path,
            format!("{name} runs past the end of the file"),
        ));
    }

    Ok((
        Entry { name, start, len },
        at + 46 + name_len + extra_len + comment_len,
    ))
}

/// Resolves the three fields zip64 can move out of the central record.
///
/// A 32-bit field saturated to `0xFFFFFFFF` is the format's way of saying "the
/// real value is in the zip64 extra field". The extra field carries *only* the
/// saturated ones, in a fixed order, so which offset to read from depends on
/// which were saturated - which is why this is a walk rather than three
/// indexes. Any checkpoint above 4 GiB reaches this path.
fn widen(
    path: &Path,
    name: &str,
    extra: &[u8],
    compressed: u32,
    uncompressed: u32,
    header: u32,
) -> Result<(u64, usize), PtError> {
    let saturated = compressed == u32::MAX || uncompressed == u32::MAX || header == u32::MAX;
    if !saturated {
        return Ok((u64::from(compressed), header as usize));
    }

    let mut cursor = 0;
    while cursor + 4 <= extra.len() {
        let tag = u16(extra, cursor).ok_or_else(|| malformed(path, "extra field"))?;
        let len = u16(extra, cursor + 2).ok_or_else(|| malformed(path, "extra field"))? as usize;
        let body = extra
            .get(cursor + 4..cursor + 4 + len)
            .ok_or_else(|| malformed(path, "extra field"))?;
        if tag != ZIP64_EXTRA {
            cursor += 4 + len;
            continue;
        }

        let mut at = 0;
        let mut take = |present: bool| -> Result<Option<u64>, PtError> {
            if !present {
                return Ok(None);
            }
            let v = u64(body, at)
                .ok_or_else(|| malformed(path, format!("zip64 extra field for {name} is short")))?;
            at += 8;
            Ok(Some(v))
        };
        // The order is fixed by the format: uncompressed, compressed, offset.
        let big_uncompressed = take(uncompressed == u32::MAX)?;
        let big_compressed = take(compressed == u32::MAX)?;
        let big_header = take(header == u32::MAX)?;

        let size = big_compressed
            .or(big_uncompressed)
            .unwrap_or(u64::from(compressed));
        let local = big_header.map_or(header as usize, |v| v as usize);
        return Ok((size, local));
    }

    Err(malformed(
        path,
        format!("{name} needs a zip64 extra field and has none"),
    ))
}

/// Builds a [`PtError::MalformedArchive`].
fn malformed(path: &Path, what: impl Into<String>) -> PtError {
    PtError::MalformedArchive {
        path: path.to_path_buf(),
        what: what.into(),
    }
}

/// Reads a little-endian `u16`, or `None` if it would run off the end.
fn u16(map: &[u8], at: usize) -> Option<u16> {
    map.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("slice is 2 bytes")))
}

/// Reads a little-endian `u32`, or `None` if it would run off the end.
fn u32(map: &[u8], at: usize) -> Option<u32> {
    map.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("slice is 4 bytes")))
}

/// Reads a little-endian `u64`, or `None` if it would run off the end.
fn u64(map: &[u8], at: usize) -> Option<u64> {
    map.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("slice is 8 bytes")))
}
