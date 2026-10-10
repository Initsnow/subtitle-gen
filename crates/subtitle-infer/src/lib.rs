//! subtitle-infer: the GGUF inference engine.
//!
//! Phase 2 — the core inference engine over the official CrispASR Rust
//! binding (`crispasr` safe wrapper + `crispasr-sys` raw FFI):
//!
//! * [`device`] — hardware detection (CPU features, GPU feature gates) and
//!   the GPU → CPU automatic fallback state machine. GPU initialization
//!   failures (OOM, missing drivers) degrade to `CRISPASR_N_GPU_LAYERS = 0`
//!   with a logged warning, never a crash.
//! * [`vad`] — VAD session: 16 kHz `&[f32]` in, speech regions out, via the
//!   raw `crispasr_vad_segments` FFI (GPU flag with lazy CPU fallback).
//! * [`asr`] — Qwen3-ASR (GGUF) session over `crispasr::Session`:
//!   autoregressive transcription with language hints and automatic language
//!   detection.
//! * [`aligner`] — Qwen3-ForcedAligner (GGUF) session via the raw
//!   `crispasr_align_words_abi` FFI: audio + text in, word-level
//!   [`TimedToken`] timestamps out.
//! * [`model`] — model spec resolution (local path / registry alias /
//!   Hugging Face repo id, with native-mode download) and language-hint
//!   normalization.
//! * [`session`] — ASR session plumbing through the fallback state machine.
//! * [`backend`] — the feature-gated access layer to the official binding.
//!
//! Cargo features:
//!
//! * `native` — link the real CrispASR runtime (requires a prebuilt
//!   `libcrispasr` with `CRISPASR_LIB_DIR`, or the git-dependency mode where
//!   `crispasr-sys`'s build.rs cmakes it from source). Without it, a stub
//!   returns [`InferError::Unsupported`] at runtime so the workspace stays
//!   buildable everywhere.
//! * `cuda` / `vulkan` / `metal` — markers for the GPU backends
//!   `libcrispasr` was built with; they shape the default backend ladder.

pub mod aligner;
pub mod asr;
pub mod backend;
pub mod device;
pub mod engine;
pub mod error;
pub mod model;
pub mod session;
pub mod vad;

pub use aligner::AlignerSession;
pub use asr::{AsrOutput, AsrSession};
pub use device::{CpuInfo, DevicePlan, FallbackSelector, GpuBackend};
pub use engine::InferenceEngine;
pub use error::{InferError, Result};
pub use model::{cache_dir, resolve_model_path, DEFAULT_ALIGNER_GGUF, DEFAULT_ASR_GGUF};
pub use session::LoadOptions;
pub use vad::{SpeechRegion, VadOptions, VadSession};

pub use subtitle_core::types::{AudioWindow, TimedToken, TranscriptChunk};

/// Name of the inference runtime reported alongside device plans.
pub const RUNTIME_NAME: &str = "crispasr-ggml";

/// True when the real CrispASR runtime is linked in (the `native` feature);
/// false for the stub build.
pub fn native_runtime_available() -> bool {
    cfg!(feature = "native")
}
