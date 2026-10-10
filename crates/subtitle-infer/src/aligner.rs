//! Qwen3-ForcedAligner (GGUF) session: audio waveform + transcript in,
//! word-level `TimedToken` timestamps out.
//!
//! Uses the raw `crispasr_align_words_abi` FFI — the same stateless
//! per-call model the Python binding's `crispasr.align_words` drives. The
//! "session" here only resolves the model path once and records the device
//! plan for diagnostics; GPU selection inside the call is the runtime's own.
//! Mirrors the Python `CrispASRAligner.align`: an empty transcript short-
//! circuits to an empty result, and tokens with empty text are filtered.

use std::path::{Path, PathBuf};

use subtitle_core::types::TimedToken;

use crate::backend;
use crate::device::{DevicePlan, GpuBackend};
use crate::error::Result;
use crate::model::{resolve_model_path, DEFAULT_ALIGNER_GGUF, DEFAULT_QUANT};
use crate::session::LoadOptions;

/// A resolved Qwen3-ForcedAligner "session".
#[derive(Debug)]
pub struct AlignerSession {
    model_path: PathBuf,
    plan: DevicePlan,
    n_threads: i32,
}

impl AlignerSession {
    /// Resolve the aligner model (default spec:
    /// `qwen3-forced-aligner-0.6b-q4_k`) and record the device plan.
    pub fn load(model_spec: &str, load: &LoadOptions) -> Result<Self> {
        let model_path = resolve_model_path(model_spec, DEFAULT_QUANT)?;
        let plan = DevicePlan {
            backend: GpuBackend::first_compiled().unwrap_or(GpuBackend::Cpu),
            n_gpu_layers: DevicePlan::GPU_LAYERS_ALL,
            attempts: Vec::new(),
        };
        tracing::info!(
            model = %model_spec,
            backend = %plan.backend,
            n_gpu_layers = plan.n_gpu_layers,
            "aligner model resolved"
        );
        Ok(Self {
            model_path,
            plan,
            n_threads: load.effective_threads(),
        })
    }

    /// Resolve with the registry default spec.
    pub fn load_default(load: &LoadOptions) -> Result<Self> {
        Self::load(DEFAULT_ALIGNER_GGUF, load)
    }

    /// The device plan recorded for this session (informational: the
    /// runtime picks its backend internally per call).
    pub fn plan(&self) -> &DevicePlan {
        &self.plan
    }

    /// Resolved local model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Force-align `transcript` against a 16 kHz mono waveform, producing
    /// word-level timestamps. Times are relative to the start of `samples`;
    /// the caller shifts them onto the global timeline using the owning
    /// window offset (Phase 4 pipeline).
    pub fn align(
        &self,
        samples: &[f32],
        transcript: &str,
        _language: Option<&str>,
    ) -> Result<Vec<TimedToken>> {
        // The FFI takes no language hint; the aligner infers it from the
        // transcript (same as the Python binding).
        let words = backend::align_words(&self.model_path, transcript, samples, self.n_threads, 0)?;
        Ok(words
            .into_iter()
            .filter(|(text, _, _)| !text.is_empty())
            .map(|(text, start, end)| TimedToken::new(text, start, end))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_matches_python() {
        assert_eq!(DEFAULT_ALIGNER_GGUF, "qwen3-forced-aligner-0.6b-q4_k");
    }
}
