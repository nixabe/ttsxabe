//! Errors from the GPU path.

/// Something went wrong talking to the device.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// No usable CUDA device, or the driver could not be loaded.
    ///
    /// This is the variant every caller should be prepared for: the driver is
    /// loaded dynamically, so a machine with no toolkit and no card reaches
    /// here rather than failing to link.
    #[error("no usable CUDA device: {0}")]
    NoDevice(String),

    /// The kernel source did not compile.
    ///
    /// Only reachable if the kernels themselves are wrong, since they are a
    /// compile-time constant - but NVRTC's message is worth keeping rather than
    /// collapsing into "compilation failed".
    #[error("kernel compilation failed: {0}")]
    Compile(String),

    /// A kernel could not be found in the compiled module.
    #[error("kernel {name} is not in the module")]
    MissingKernel {
        /// The kernel that was asked for.
        name: String,
    },

    /// A driver call failed.
    #[error("{what}: {source}")]
    Driver {
        /// What was being attempted.
        what: &'static str,
        /// The driver's own error.
        #[source]
        source: cudarc::driver::DriverError,
    },

    /// A contraction length the tensor-core instruction cannot express.
    ///
    /// `m16n8k8` steps the contraction in eights. Padding a ragged one to the
    /// next multiple would work and would hide a caller that has miscomputed a
    /// shape, which in this workspace is the more likely of the two.
    #[error("contraction length {k} is not a multiple of 8")]
    RaggedContraction {
        /// The length asked for.
        k: usize,
    },
}
