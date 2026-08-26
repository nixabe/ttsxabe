//! What the serving layer refuses, and what it reports upward.

use thiserror::Error;

/// A failure in the serving layer.
#[derive(Debug, Error)]
pub enum ServeError {
    /// A stage this process delegates to failed.
    ///
    /// Carries the stage name because a turn touches four services and "HTTP
    /// 500" alone sends the reader to the wrong log.
    #[error("{stage}: {message}")]
    Upstream {
        /// Which delegated stage failed.
        stage: &'static str,
        /// What it said.
        message: String,
    },

    /// The HTTP client could not be built or a request could not be formed.
    #[error("http client: {0}")]
    Client(String),

    /// The listen address could not be bound.
    #[error("binding {addr}: {source}")]
    Bind {
        /// The address as given.
        addr: String,
        /// Why it failed.
        source: std::io::Error,
    },

    /// The server stopped serving.
    #[error("serving: {0}")]
    Serve(std::io::Error),

    /// A browser frame carried audio that is not decodable base64.
    #[error("frame audio is not base64: {0}")]
    BadPcm(String),

    /// A turn needed a stage this process has no way to reach.
    ///
    /// Distinct from [`ServeError::Upstream`]: that stage exists and failed,
    /// this one was never configured. The fix is a flag, not a restart.
    #[error("this process has no {0} stage; it was not configured")]
    NoStage(&'static str),
}
