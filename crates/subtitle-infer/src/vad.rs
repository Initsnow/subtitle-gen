//! VAD session: 16 kHz `&[f32]` in, speech regions out.
//!
//! Backed by the raw `crispasr_vad_segments` FFI (standalone voice-activity
//! detection, no transcription). The runtime returns raw speech/silence
//! boundaries; this module applies the deterministic post-processing in Rust
//! (padding, minimum speech duration, region merging) with defaults mirroring
//! the Python `VADConfig`:
//!
//! * `min_speech_duration = 0.25 s` (250 ms)
//! * `min_silence_duration = 0.6 s` (600 ms)
//! * `speech_padding = 0.25 s` (250 ms)
//!
//! GPU usage: the FFI takes an explicit `use_gpu` flag. The first `detect`
//! call tries GPU when a backend is compiled in; on a GPU initialization
//! failure it retries on CPU and remembers the choice for subsequent calls.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use crate::backend;
use crate::device::{DevicePlan, GpuBackend};
use crate::error::{InferError, Result};
use crate::model::{resolve_model_path, DEFAULT_QUANT};
use crate::session::LoadOptions;

/// VAD tuning knobs, matching the Python `VADConfig` defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct VadOptions {
    /// Speech probability threshold (model dependent).
    pub threshold: f32,
    /// Regions shorter than this are discarded.
    pub min_speech_duration: f64,
    /// Gaps shorter than this are bridged when merging adjacent regions.
    pub min_silence_duration: f64,
    /// Padding added on both sides of each speech region.
    pub speech_padding: f64,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration: 0.25,
            min_silence_duration: 0.6,
            speech_padding: 0.25,
        }
    }
}

/// One speech region in seconds (half-open interval).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechRegion {
    pub start: f64,
    pub end: f64,
}

impl SpeechRegion {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

impl From<SpeechRegion> for (f64, f64) {
    fn from(region: SpeechRegion) -> Self {
        (region.start, region.end)
    }
}

/// A resolved VAD model with lazy GPU fallback. `!Send` (the fallback state
/// is per-session).
pub struct VadSession {
    model_path: PathBuf,
    options: VadOptions,
    n_threads: i32,
    /// Current GPU preference, flipped to `false` after the first GPU
    /// initialization failure.
    use_gpu: Cell<bool>,
    plan: std::cell::RefCell<DevicePlan>,
}

impl std::fmt::Debug for VadSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadSession")
            .field("model_path", &self.model_path)
            .field("options", &self.options)
            .field("plan", &self.plan.borrow().clone())
            .finish_non_exhaustive()
    }
}

impl VadSession {
    /// Resolve the VAD model and prepare the session.
    pub fn load(model_spec: &str, options: VadOptions, load: &LoadOptions) -> Result<Self> {
        let model_path = resolve_model_path(model_spec, DEFAULT_QUANT)?;
        let gpu = GpuBackend::first_compiled();
        let plan = DevicePlan {
            backend: gpu.unwrap_or(GpuBackend::Cpu),
            n_gpu_layers: if gpu.is_some() {
                DevicePlan::GPU_LAYERS_ALL
            } else {
                DevicePlan::GPU_LAYERS_NONE
            },
            attempts: Vec::new(),
        };
        tracing::info!(
            model = %model_spec,
            backend = %plan.backend,
            n_gpu_layers = plan.n_gpu_layers,
            "VAD session resolved"
        );
        Ok(Self {
            model_path,
            options,
            n_threads: load.effective_threads(),
            use_gpu: Cell::new(gpu.is_some()),
            plan: std::cell::RefCell::new(plan),
        })
    }

    /// The device plan currently in effect.
    pub fn plan(&self) -> DevicePlan {
        self.plan.borrow().clone()
    }

    /// Resolved local model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Detect speech regions in a 16 kHz mono waveform. Retries once on CPU
    /// when the GPU attempt fails to initialize.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<SpeechRegion>> {
        if samples.is_empty() {
            return Err(InferError::InvalidArgument("empty audio samples".into()));
        }
        let opts = &self.options;
        let call = |use_gpu: bool| {
            backend::vad_regions(
                &self.model_path,
                samples,
                opts.threshold,
                (opts.min_speech_duration * 1000.0) as i32,
                (opts.min_silence_duration * 1000.0) as i32,
                use_gpu,
                self.n_threads,
            )
        };

        let raw = match call(self.use_gpu.get()) {
            Err(err) if err.is_gpu_init_failure() && self.use_gpu.get() => {
                tracing::warn!(
                    reason = %err,
                    "GPU VAD failed to initialize; falling back to CPU"
                );
                self.use_gpu.set(false);
                let mut plan = self.plan.borrow().clone();
                plan.backend = GpuBackend::Cpu;
                plan.n_gpu_layers = DevicePlan::GPU_LAYERS_NONE;
                plan.attempts.push(crate::device::AttemptNote {
                    backend: GpuBackend::Cpu,
                    n_gpu_layers: DevicePlan::GPU_LAYERS_NONE,
                    outcome: crate::device::AttemptOutcome::Ok,
                });
                self.plan.replace(plan);
                call(false)?
            }
            result => result?,
        };

        let duration = samples.len() as f64 / 16_000.0;
        Ok(normalize_regions(&raw, opts, duration))
    }
}

/// Deterministic post-processing of raw VAD boundaries (silero ordering:
/// merge → filter short speech → pad last):
///
/// 1. clamp to `[0, duration]`, drop empty/inverted regions, sort by start,
/// 2. merge regions separated by less than `min_silence_duration`,
/// 3. drop regions shorter than `min_speech_duration` (pre-padding),
/// 4. pad both sides by `speech_padding`, clamped to `[0, duration]`.
pub(crate) fn normalize_regions(
    raw: &[(f64, f64)],
    options: &VadOptions,
    duration: f64,
) -> Vec<SpeechRegion> {
    let mut regions: Vec<(f64, f64)> = raw
        .iter()
        .map(|&(start, end)| (start.min(end), start.max(end)))
        .filter(|&(start, end)| end > start)
        .collect();
    regions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Merge across short silences.
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(regions.len());
    for (start, end) in regions {
        match merged.last_mut() {
            Some(last) if start - last.1 < options.min_silence_duration => {
                last.1 = last.1.max(end);
            }
            _ => merged.push((start, end)),
        }
    }

    merged
        .into_iter()
        // Filter before padding, matching silero's `get_speech_timestamps`.
        .filter(|&(start, end)| end - start >= options.min_speech_duration)
        .map(|(start, end)| {
            (
                (start - options.speech_padding).max(0.0),
                (end + options.speech_padding).min(duration),
            )
        })
        .map(|(start, end)| SpeechRegion { start, end })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> VadOptions {
        VadOptions {
            threshold: 0.5,
            min_speech_duration: 0.25,
            min_silence_duration: 0.6,
            speech_padding: 0.25,
        }
    }

    #[test]
    fn regions_are_padded_clamped_and_filtered() {
        let raw = vec![(0.1, 1.0), (1.2, 2.0), (5.0, 5.1)];
        let out = normalize_regions(&raw, &opts(), 6.0);
        // First two merge (gap 0.2 < 0.6), pad to [0, 2.25]; third is only
        // 0.1 s before padding → dropped.
        assert_eq!(out.len(), 1);
        assert!((out[0].start - 0.0).abs() < 1e-9);
        assert!((out[0].end - 2.25).abs() < 1e-9);
    }

    #[test]
    fn padding_respects_duration_bounds() {
        let raw = vec![(5.5, 5.9)];
        let out = normalize_regions(&raw, &opts(), 6.0);
        assert_eq!(out.len(), 1);
        assert!((out[0].end - 6.0).abs() < 1e-9);
    }

    #[test]
    fn inverted_and_empty_regions_are_dropped() {
        let raw = vec![(3.0, 3.0), (2.5, 1.0)];
        let out = normalize_regions(&raw, &opts(), 10.0);
        assert_eq!(out.len(), 1);
        assert!((out[0].start - 0.75).abs() < 1e-9);
        assert!((out[0].end - 2.75).abs() < 1e-9);
    }

    #[test]
    fn region_tuple_conversion() {
        let region = SpeechRegion {
            start: 1.0,
            end: 2.5,
        };
        let (start, end) = region.into();
        assert_eq!((start, end), (1.0, 2.5));
    }
}
