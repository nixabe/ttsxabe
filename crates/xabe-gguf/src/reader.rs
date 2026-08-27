//! A bounds-checked byte cursor over GGUF's flat binary layout.
//!
//! Adapted from `llmxabe/crates/xabe-gguf`, which has been reading GGUF on
//! this machine for a while; see `docs/TOOLCHAIN.md` for what was taken and
//! what was dropped.

use crate::error::GgufError;

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn pos(&self) -> u64 {
        self.pos as u64
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// A capacity hint that never exceeds the bytes actually left.
    ///
    /// A truncated file claiming a billion-element array must not drive a
    /// billion-element allocation before the per-element bounds check below
    /// gets a chance to fail.
    pub(crate) fn hint(&self, n: u64) -> usize {
        (n as u128).min(self.remaining() as u128) as usize
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], GgufError> {
        if n > self.remaining() {
            return Err(GgufError::UnexpectedEof {
                context,
                offset: self.pos(),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub(crate) fn magic(&mut self) -> Result<[u8; 4], GgufError> {
        let s = self.take(4, "magic")?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    pub(crate) fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1, "u8")?[0])
    }

    pub(crate) fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.take(1, "i8")?[0] as i8)
    }

    pub(crate) fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2, "u16")?.try_into().unwrap()))
    }

    pub(crate) fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.take(2, "i16")?.try_into().unwrap()))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4, "u32")?.try_into().unwrap()))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.take(4, "i32")?.try_into().unwrap()))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8, "u64")?.try_into().unwrap()))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.take(8, "i64")?.try_into().unwrap()))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4, "f32")?.try_into().unwrap()))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.take(8, "f64")?.try_into().unwrap()))
    }

    /// GGUF stores `bool` as one `int8_t`, zero being false.
    pub(crate) fn bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.i8()? != 0)
    }

    /// A `u64` byte length followed by raw, non-terminated UTF-8.
    pub(crate) fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::UnexpectedEof {
            context: "string length",
            offset: self.pos(),
        })?;
        let bytes = self.take(len, "string bytes")?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }
}
