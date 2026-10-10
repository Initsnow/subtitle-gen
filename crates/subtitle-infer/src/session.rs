//! ASR session plumbing: resolve the model, run the GPU → CPU fallback state
//! machine around `Session::open`, and hand back the handle plus the device
//! plan it landed on.
//!
//! Forced alignment and VAD are stateless per-call FFI functions in the
//! CrispASR ABI (they take the model path each time, exactly like the Python
//! binding's `align_words`), so they do not need this machinery — see
//! [`crate::aligner`] and [`crate::vad`].

use std::path::PathBuf;

use crate::backend::{self, GpuBudget};
use crate::device::{DevicePlan, FallbackSelector, GpuBackend};
use crate::error::Result;
use crate::model::resolve_model_path;

/// Per-session load knobs.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Worker threads for the stateless VAD / aligner FFI calls; `0` = all
    /// logical cores. ASR session open delegates threading to the runtime.
    pub n_threads: i32,
    /// Override the detected backend ladder (advanced use); `None` = auto.
    pub ladder: Option<Vec<GpuBackend>>,
}

impl LoadOptions {
    /// Thread count for stateless FFI calls: explicit or all logical cores.
    pub fn effective_threads(&self) -> i32 {
        if self.n_threads > 0 {
            self.n_threads
        } else {
            crate::device::probe_cpu().default_threads()
        }
    }
}

/// A successfully opened ASR session plus the device plan it landed on.
pub struct AsrHandle {
    pub session: backend::RawAsrSession,
    pub plan: DevicePlan,
    pub model_path: PathBuf,
}

impl std::fmt::Debug for AsrHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrHandle")
            .field("model_path", &self.model_path)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

/// Resolve the model spec, then open the session through the fallback state
/// machine: GPU budget first (runtime default), CPU-only budget (`0`) as the
/// final rung. GPU initialization failures degrade with a logged warning;
/// hard errors (missing model file, corrupt GGUF) propagate immediately.
pub fn open_asr_session(
    model_spec: &str,
    quant: &str,
    options: &LoadOptions,
) -> Result<AsrHandle> {
    let model_path = resolve_model_path(model_spec, quant)?;
    let selector = match &options.ladder {
        Some(ladder) => FallbackSelector::from_candidates(ladder.clone()),
        None => FallbackSelector::detect(),
    };

    let attempt = |_backend: GpuBackend, n_gpu_layers: i32| -> Result<backend::RawAsrSession> {
        let budget = if n_gpu_layers == DevicePlan::GPU_LAYERS_NONE {
            GpuBudget::CpuOnly
        } else {
            GpuBudget::Default
        };
        backend::RawAsrSession::open(&model_path, budget)
    };

    let (session, plan) = selector.select(attempt)?;
    Ok(AsrHandle { session, plan, model_path })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::InferError;

    #[test]
    fn effective_threads_falls_back_to_core_count() {
        let opts = LoadOptions::default();
        assert!(opts.effective_threads() >= 1);
        let explicit = LoadOptions { n_threads: 4, ..LoadOptions::default() };
        assert_eq!(explicit.effective_threads(), 4);
    }

    #[test]
    fn open_session_reports_missing_model_file_not_gpu_trouble() {
        // Stub mode: the model is missing first, and that hard error must
        // surface before any GPU ladder run.
        let err = open_asr_session(
            "cstr/definitely-not-a-real-model-GGUF",
            crate::model::DEFAULT_QUANT,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, InferError::ModelNotFound { .. }));
    }
}
