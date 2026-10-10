//! Qwen3-ASR (GGUF) session: autoregressive transcription with optional
//! language hints and automatic language detection.
//!
//! Wraps the official `crispasr::Session` (safe) API. Mirrors the Python
//! `CrispASRASR` behaviour: segment texts are joined with a single space,
//! trimmed, and the detected language (ISO 639-1) is reported alongside the
//! transcript.

use std::path::Path;

use crate::error::{InferError, Result};
use crate::model::{iso_language, DEFAULT_ASR_GGUF};
use crate::session::{open_asr_session, AsrHandle, LoadOptions};

/// One transcription result.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrOutput {
    pub text: String,
    /// ISO 639-1 code when the model could detect it, else the normalized
    /// hint, else `None`.
    pub language: Option<String>,
}

/// A loaded Qwen3-ASR session. `!Send` — pinned to the creating thread.
pub struct AsrSession {
    handle: AsrHandle,
}

impl std::fmt::Debug for AsrSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrSession")
            .field("model_path", &self.handle.model_path)
            .field("plan", &self.handle.plan)
            .finish_non_exhaustive()
    }
}

impl AsrSession {
    /// Load the ASR model (default spec: `qwen3-asr-0.6b-q4_k`), going
    /// through the GPU → CPU fallback state machine.
    pub fn load(model_spec: &str, load: &LoadOptions) -> Result<Self> {
        let handle = open_asr_session(model_spec, crate::model::DEFAULT_QUANT, load)?;
        tracing::info!(
            model = %model_spec,
            backend = %handle.plan.backend,
            n_gpu_layers = handle.plan.n_gpu_layers,
            "ASR session loaded"
        );
        Ok(Self { handle })
    }

    /// Load with the registry default spec.
    pub fn load_default(load: &LoadOptions) -> Result<Self> {
        Self::load(DEFAULT_ASR_GGUF, load)
    }

    /// The device plan this session landed on.
    pub fn plan(&self) -> &crate::device::DevicePlan {
        &self.handle.plan
    }

    /// Resolved local model path.
    pub fn model_path(&self) -> &Path {
        &self.handle.model_path
    }

    /// Transcribe a 16 kHz mono waveform. `language` accepts ISO codes or
    /// full names (`"ja"`, `"Japanese"`); `None` enables automatic detection.
    pub fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<AsrOutput> {
        if samples.is_empty() {
            return Err(InferError::InvalidArgument("empty audio samples".into()));
        }
        let hint = iso_language(language);
        let raw = self.handle.session.transcribe(samples, hint.as_deref())?;

        // Join non-empty segment texts with single spaces, trimmed — the
        // Python implementation's exact behaviour.
        let text = raw
            .segments
            .iter()
            .filter(|(text, _, _)| !text.is_empty())
            .map(|(text, _, _)| text.clone())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();

        Ok(AsrOutput {
            text,
            language: raw.detected_language.or(hint),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_matches_python() {
        assert_eq!(DEFAULT_ASR_GGUF, "qwen3-asr-0.6b-q4_k");
    }
}
