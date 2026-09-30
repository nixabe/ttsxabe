//! Host-upload allocation policy and Windows memory diagnostics.
//!
//! Uploaded weights live for the model's lifetime and do not benefit from
//! scratch-pool reuse. Windows defaults these uploads to ordinary CUDA device
//! allocations, avoiding the non-migratable memory-pool path under WDDM.
//! Scratch allocations still use cudarc's stream pool. This does not bypass
//! Windows commit or residency budgets. Start with `UploadAllocator::parse`.

use crate::CudaError;
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError, result, sys};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UploadAllocator {
    Pool,
    Legacy,
}

impl UploadAllocator {
    pub(crate) fn parse(value: &str, windows: bool) -> Result<Self, CudaError> {
        match value {
            "auto" if windows => Ok(Self::Legacy),
            "auto" | "pool" => Ok(Self::Pool),
            "legacy" => Ok(Self::Legacy),
            _ => Err(CudaError::UploadAllocator(value.to_string())),
        }
    }

    pub(crate) fn from_env() -> Result<Self, CudaError> {
        let value = match std::env::var("LLMTIE_CUDA_UPLOAD_ALLOCATOR") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "auto".to_string(),
            Err(e) => return Err(CudaError::UploadAllocator(e.to_string())),
        };
        Self::parse(&value, cfg!(target_os = "windows"))
    }

    /// Uninitialized device memory: the caller fills it before exposing it.
    pub(crate) unsafe fn alloc<T: DeviceRepr>(
        self,
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
        if self == Self::Pool || bytes == 0 {
            return unsafe { stream.alloc(len) };
        }
        stream.context().bind_to_thread()?;
        let mut guard = LegacyPtr(Some(unsafe { result::malloc_sync(bytes) }?));
        // Transfer ownership to cudarc. CUDA explicitly permits free_async of
        // an ordinary allocation; CudaSlice records dependencies before free.
        // If cudarc's event creation unwinds, the guard releases the pointer.
        let slice = unsafe { stream.upgrade_device_ptr(guard.0.unwrap(), len) };
        guard.0 = None;
        Ok(slice)
    }
}

/// Owns the raw allocation until cudarc has constructed its event bookkeeping.
struct LegacyPtr(Option<sys::CUdeviceptr>);

impl Drop for LegacyPtr {
    fn drop(&mut self) {
        if let Some(ptr) = self.0 {
            // Construction and this guard run on the already-bound thread.
            if let Err(e) = unsafe { result::free_sync(ptr) } {
                tracing::warn!(error = ?e, "releasing an unadopted CUDA upload allocation");
            }
        }
    }
}

#[cfg(target_os = "windows")]
pub(crate) fn host_memory_status() -> Option<String> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Some(format!(
            "Windows memory info unavailable: {}",
            std::io::Error::last_os_error()
        ));
    }
    Some(format!(
        "Windows physical available={} total={} bytes; process commit available={} limit={} bytes",
        status.ullAvailPhys, status.ullTotalPhys, status.ullAvailPageFile, status.ullTotalPageFile
    ))
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn host_memory_status() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_policy_defaults_and_explicit_overrides() {
        assert_eq!(
            UploadAllocator::parse("auto", true).unwrap(),
            UploadAllocator::Legacy
        );
        assert_eq!(
            UploadAllocator::parse("auto", false).unwrap(),
            UploadAllocator::Pool
        );
        for windows in [false, true] {
            assert_eq!(
                UploadAllocator::parse("pool", windows).unwrap(),
                UploadAllocator::Pool
            );
            assert_eq!(
                UploadAllocator::parse("legacy", windows).unwrap(),
                UploadAllocator::Legacy
            );
            assert!(UploadAllocator::parse("", windows).is_err());
            assert!(UploadAllocator::parse("typo", windows).is_err());
        }
    }
}
