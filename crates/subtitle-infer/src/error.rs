//! Error type for the inference engine (`subtitle-infer`).

use thiserror::Error;

pub type Result<T> = std::result::Result<T, InferError>;

/// Everything that can go wrong while loading or driving the GGUF runtime.
#[derive(Debug, Error)]
pub enum InferError {
    /// The model spec (path / alias / HF repo id) could not be resolved.
    #[error("model spec resolution failed: {0}")]
    Spec(String),

    /// The resolved GGUF file does not exist on disk and no downloader is
    /// wired in yet (network fetch arrives with the Phase 4 pipeline).
    #[error("model file not found: {path} (hint: {hint})")]
    ModelNotFound { path: String, hint: String },

    /// GPU backend was never a candidate (not compiled in, or the runtime
    /// reports it unavailable) — recorded by the fallback state machine.
    #[error("gpu backend {backend} unavailable: {reason}")]
    BackendUnavailable { backend: String, reason: String },

    /// GPU memory allocation or kernel launch ran out of VRAM.
    #[error("gpu out of memory on {backend}: {reason}")]
    Oom { backend: String, reason: String },

    /// The native runtime returned a non-recoverable error code.
    #[error("native runtime error {code}: {message}")]
    Runtime { code: i32, message: String },

    /// The compiled runtime cannot perform the operation at all (e.g. the
    /// stub build without vendored C++ sources).
    #[error("operation not supported by the compiled runtime: {0}")]
    Unsupported(String),

    /// Bad caller input (empty audio, wrong model kind for the session, ...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl InferError {
    /// Whether the fallback state machine should treat this error as "this
    /// GPU backend did not work, move down the ladder" rather than a hard
    /// failure. `BackendUnavailable` and `Oom` both qualify; anything else
    /// (bad model file, invalid argument, runtime bug) fails immediately.
    pub fn is_gpu_init_failure(&self) -> bool {
        matches!(
            self,
            InferError::Oom { .. } | InferError::BackendUnavailable { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_init_failures_are_retryable() {
        let oom = InferError::Oom {
            backend: "cuda".into(),
            reason: "alloc 2 GiB failed".into(),
        };
        let unavailable = InferError::BackendUnavailable {
            backend: "vulkan".into(),
            reason: "no device with compute queue".into(),
        };
        let hard = InferError::InvalidArgument("empty samples".into());
        assert!(oom.is_gpu_init_failure());
        assert!(unavailable.is_gpu_init_failure());
        assert!(!hard.is_gpu_init_failure());
    }
}
