//! subtitle-core: pure-Rust business logic for subtitle generation.
//!
//! This crate owns everything that does not touch the inference runtime:
//!
//! * [`audio`] — container decoding, mono downmix, and resampling via
//!   `symphonia` + `rubato`, with zero-copy window slicing.
//! * [`formats`] — SRT / WebVTT / LRC / JSON / TXT parsing and rendering.
//! * [`types`] — the domain entities (`TimedToken`, `SubtitleItem`,
//!   `AudioWindow`, `TranscriptChunk`).
//!
//! The inference engine (VAD / ASR / forced alignment over GGUF weights)
//! lives in `subtitle-infer`; LLM proofreading/translation and the async
//! pipeline will be added in later phases.

pub mod audio;
pub mod cache;
pub mod config;
pub mod error;
pub mod formats;
pub mod llm;
pub mod pipeline;
pub mod text;
pub mod types;

pub use error::{Error, Result};
