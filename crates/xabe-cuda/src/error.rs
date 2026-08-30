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

    /// A packed tensor whose byte count is not a whole number of blocks.
    ///
    /// Checked at upload because Q6_K is restrided on the way to the device -
    /// see [`crate::Quant::device_stride`]. A ragged tail would be restrided
    /// into the next block's slot and decode to plausible numbers rather than
    /// failing, so the length is the thing to refuse.
    #[error("{len} bytes is not a whole number of {ty}-byte blocks")]
    RaggedBlockBytes {
        /// The length handed over.
        len: usize,
        /// Bytes per block in the format handed over.
        ty: usize,
    },

    /// An [`crate::Operand::F32Q`] whose int8 twin was taken at another shape.
    ///
    /// The twin is addressed as a dense `[rows, k]` of its own, so one taken at
    /// a different shape is read in bounds and produces numbers. There is no
    /// downstream check that would catch it - which is the whole reason this
    /// one exists at the boundary.
    #[error("the int8 twin is {rows}x{k}, and this matmul is {want_rows}x{want_k}")]
    MismatchedQ8 {
        /// Rows the twin was taken over.
        rows: usize,
        /// Contraction the twin was taken along.
        k: usize,
        /// Rows this matmul has.
        want_rows: usize,
        /// Contraction this matmul has.
        want_k: usize,
    },

    /// A cache append that would run past the buffer's capacity.
    ///
    /// The cache is one allocation holding every head, so a `past` beyond its
    /// capacity writes into the next head's positions rather than off the end:
    /// in bounds, and wrong for every step after it.
    #[error("appending at {at} to a cache with room for {cap}")]
    CacheOverrun {
        /// Where the write would end.
        at: usize,
        /// Positions the buffer has room for.
        cap: usize,
    },

    /// A row stride asked for on an operand that has no rows to stride.
    ///
    /// The packed and f16 paths derive addressing from the block or word
    /// layout, so a stride handed to them would be ignored without a word.
    #[error("a row stride of {stride} on a weight that is not f32")]
    StridedNonF32Weight {
        /// The stride asked for.
        stride: usize,
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
