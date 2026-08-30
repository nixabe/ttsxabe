//! Errors from the GPU path.

/// Something went wrong talking to the device.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// An odd contraction with an f16 weight, which packs two to a word.
    ///
    /// The F32 path takes any length - see the history in
    /// `gemm_accepts_every_contraction_length`. This one cannot: an f16 weight
    /// is addressed as 32-bit words, so an odd `k` would put the boundary in
    /// the middle of one. Every contraction in a transformer is even, so this
    /// is a check rather than a limitation.
    #[error("contraction length {k} is odd, and an f16 weight packs two to a word")]
    RaggedContraction {
        /// The length asked for.
        k: usize,
    },

    /// A contraction that is not a whole number of quantization blocks.
    ///
    /// A packed row starts on a block boundary, so `q_at` derives the block
    /// index by dividing. A `k` that is not a multiple of the block size makes
    /// that division cross into the previous row's last block: in bounds, and
    /// wrong. GGUF guarantees the fastest-varying dimension is a whole number
    /// of blocks, and this refuses the shape rather than trusting it.
    #[error("contraction length {k} is not a multiple of the {block}-element block")]
    RaggedBlock {
        /// The length asked for.
        k: usize,
        /// Elements per block in the format handed over.
        block: usize,
    },

    /// A quantized left operand, which no kernel accepts.
    ///
    /// Weights come from a checkpoint and can be stored packed; activations are
    /// produced by the previous kernel at f32. Quantizing one would mean
    /// quantizing at runtime, which is a different piece of work and not one
    /// this path pretends to do.
    #[error("the left operand is quantized, and only a weight may be")]
    QuantizedActivation,

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
}
