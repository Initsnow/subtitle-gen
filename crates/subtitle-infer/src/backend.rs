//! Backend layer: thin, feature-gated access to the official CrispASR Rust
//! binding (`crispasr` safe wrapper + `crispasr-sys` raw FFI).
//!
//! Two compilation modes:
//!
//! * **`feature = "native"`** — the real C++ ggml runtime is linked (see the
//!   crate docs for the two linking modes). ASR goes through the safe
//!   `crispasr::Session` API; forced alignment and standalone VAD go through
//!   the raw `crispasr-sys` FFI, which the safe wrapper does not expose.
//! * **default (stub)** — the same signatures return
//!   [`InferError::Unsupported`], keeping the workspace buildable and
//!   unit-testable on machines without `libcrispasr`.
//!
//! GPU policy: the runtime selects its best compiled-in backend at session
//! open, honouring the `CRISPASR_N_GPU_LAYERS` environment variable (the same
//! knob the Python binding uses): unset = runtime default (full offload on a
//! GPU build), `0` = CPU only. The fallback state machine in [`crate::device`]
//! drives this variable around each open attempt.

use std::path::Path;

use crate::error::{InferError, Result};

/// Environment variable read by the CrispASR runtime at session open.
pub const GPU_ENV_VAR: &str = "CRISPASR_N_GPU_LAYERS";

/// GPU layer budget for an open attempt. `Default` = runtime decides (full
/// offload when built with a GPU backend); `CpuOnly` forces `n_gpu_layers = 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBudget {
    Default,
    CpuOnly,
}

impl GpuBudget {
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    pub(crate) fn apply_env(self) {
        match self {
            GpuBudget::Default => std::env::remove_var(GPU_ENV_VAR),
            GpuBudget::CpuOnly => std::env::set_var(GPU_ENV_VAR, "0"),
        }
    }
}

/// Transcript segments from one ASR pass: `(text, start_seconds, end_seconds)`
/// plus the acoustically detected language, when the backend reports one.
#[derive(Debug, Clone, Default)]
pub struct RawTranscript {
    pub segments: Vec<(String, f64, f64)>,
    pub detected_language: Option<String>,
}

// ---------------------------------------------------------------------------
// Native mode — the real runtime.
// ---------------------------------------------------------------------------

#[cfg(feature = "native")]
mod imp {
    use super::*;
    use std::ffi::CString;

    /// Classify an error string from `Session::open` so the fallback state
    /// machine can tell "try the next device" from "stop".
    pub(crate) fn classify_open_error(message: String) -> InferError {
        let lower = message.to_lowercase();
        if lower.contains("out of memory") || lower.contains("oom") || lower.contains("alloc") {
            InferError::Oom { backend: "gpu".into(), reason: message }
        } else if lower.contains("driver")
            || lower.contains("device")
            || lower.contains("backend")
            || lower.contains("cuda")
            || lower.contains("vulkan")
            || lower.contains("metal")
        {
            InferError::BackendUnavailable { backend: "gpu".into(), reason: message }
        } else {
            InferError::Runtime { code: -1, message }
        }
    }

    /// ASR session over the safe `crispasr::Session` API. `!Send`: the
    /// underlying ggml context is not thread-safe.
    pub struct RawAsrSession {
        inner: crispasr::Session,
    }

    impl RawAsrSession {
        /// Open a model with runtime auto backend selection. `GpuBudget`
        /// goes through `CRISPASR_N_GPU_LAYERS`; note the variable is
        /// process-global and only read at open time.
        pub fn open(model_path: &Path, budget: GpuBudget) -> Result<Self> {
            budget.apply_env();
            let path = model_path.to_string_lossy().into_owned();
            let inner = crispasr::Session::open(&path).map_err(classify_open_error)?;
            tracing::info!(model = %path, "ASR session opened (runtime auto backend)");
            Ok(Self { inner })
        }

        /// Transcribe 16 kHz mono samples with an optional language hint.
        pub fn transcribe(
            &self,
            samples: &[f32],
            language: Option<&str>,
        ) -> Result<RawTranscript> {
            if samples.is_empty() {
                return Err(InferError::InvalidArgument("empty audio samples".into()));
            }
            let segments = self
                .inner
                .transcribe_with_language(samples, language)
                .map_err(|msg| InferError::Runtime { code: -4, message: msg })?;

            let detected = self.detected_language();
            Ok(RawTranscript {
                segments: segments
                    .iter()
                    .map(|seg| (seg.text.trim().to_string(), seg.start, seg.end))
                    .collect(),
                detected_language: detected,
            })
        }

        /// ISO 639-1 code detected on the last transcribe, or `None`.
        fn detected_language(&self) -> Option<String> {
            let value = self.inner.detected_language();
            let value = value.trim();
            (!value.is_empty() && value != "unknown").then(|| value.to_string())
        }
    }

    /// Forced alignment of `transcript` against 16 kHz mono samples via the
    /// raw `crispasr_align_words_abi` FFI (the safe wrapper does not expose
    /// it). Timestamps come back in centiseconds and are converted to
    /// seconds; `t_offset_cs` shifts them onto an absolute timeline.
    pub fn align_words(
        model_path: &Path,
        transcript: &str,
        samples: &[f32],
        n_threads: i32,
        t_offset_cs: i64,
    ) -> Result<Vec<(String, f64, f64)>> {
        if transcript.trim().is_empty() {
            return Ok(Vec::new());
        }
        if samples.is_empty() {
            return Err(InferError::InvalidArgument("empty audio samples".into()));
        }
        let model_c = cstring(model_path.to_string_lossy().as_ref())?;
        let transcript_c = cstring(transcript)?;
        let mut out: Vec<(String, f64, f64)> = Vec::new();

        // SAFETY: all C strings outlive the call, `samples` is a valid
        // read-only buffer, and the result handle (if any) is read via the
        // accessors below and then released with `crispasr_align_result_free`.
        unsafe {
            let result = crispasr_sys::crispasr_align_words_abi(
                model_c.as_ptr(),
                transcript_c.as_ptr(),
                samples.as_ptr(),
                samples.len() as i32,
                t_offset_cs,
                n_threads,
            );
            if result.is_null() {
                return Err(InferError::Runtime {
                    code: -5,
                    message: "crispasr_align_words_abi failed (null result)".into(),
                });
            }
            let n_words = crispasr_sys::crispasr_align_result_n_words(result);
            for i in 0..n_words {
                let text = ptr_to_string(crispasr_sys::crispasr_align_result_word_text(result, i));
                let start = crispasr_sys::crispasr_align_result_word_t0(result, i) as f64 / 100.0;
                let end = crispasr_sys::crispasr_align_result_word_t1(result, i) as f64 / 100.0;
                if !text.is_empty() {
                    out.push((text, start, end));
                }
            }
            crispasr_sys::crispasr_align_result_free(result);
        }
        Ok(out)
    }

    /// Standalone VAD over the raw `crispasr_vad_segments` FFI: returns
    /// speech regions in seconds without transcribing. `use_gpu` maps to the
    /// FFI's `use_gpu` int.
    ///
    /// Note: the sys crate documents no unit for the span buffer; the spans
    /// are treated as `(start_seconds, end_seconds)` float pairs, and the
    /// return value as the span count (negative = error).
    pub fn vad_regions(
        model_path: &Path,
        samples: &[f32],
        threshold: f32,
        min_speech_ms: i32,
        min_silence_ms: i32,
        use_gpu: bool,
        n_threads: i32,
    ) -> Result<Vec<(f64, f64)>> {
        if samples.is_empty() {
            return Err(InferError::InvalidArgument("empty audio samples".into()));
        }
        let model_c = cstring(model_path.to_string_lossy().as_ref())?;
        let mut spans: *mut f32 = std::ptr::null_mut();

        if !use_gpu {
            std::env::set_var(GPU_ENV_VAR, "0");
        }

        // SAFETY: `model_c` and `samples` outlive the call; `spans` is
        // written by the callee and released with `crispasr_vad_free`.
        let n = unsafe {
            crispasr_sys::crispasr_vad_slices(
                model_c.as_ptr(),
                samples.as_ptr(),
                samples.len() as i32,
                16_000,
                threshold,
                min_speech_ms,
                min_silence_ms,
                0,
                60.0,
                n_threads,
                &mut spans,
            )
        };
        if !use_gpu {
            std::env::remove_var(GPU_ENV_VAR);
        }
        if n < 0 {
            return Err(InferError::Runtime {
                code: -6,
                message: format!("crispasr_vad_slices failed (status {n})"),
            });
        }
        let mut regions = Vec::with_capacity(n.max(0) as usize);
        if n > 0 {
            // SAFETY: the callee allocated 2*n floats; copied before freeing.
            unsafe {
                let slice = std::slice::from_raw_parts(spans, (n as usize) * 2);
                for pair in slice.chunks_exact(2) {
                    regions.push((pair[0] as f64, pair[1] as f64));
                }
                crispasr_sys::crispasr_vad_free(spans);
            }
        }
        Ok(regions)
    }

    fn cstring(value: &str) -> Result<CString> {
        CString::new(value).map_err(|_| InferError::InvalidArgument("string contains NUL".into()))
    }

    unsafe fn ptr_to_string(ptr: *const std::ffi::c_char) -> String {
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn open_error_classification_is_retryable_for_gpu_trouble() {
            let oom = classify_open_error("CUDA error: out of memory".into());
            assert!(oom.is_gpu_init_failure());
            let driver = classify_open_error("no vulkan device with compute queue".into());
            assert!(driver.is_gpu_init_failure());
            let hard = classify_open_error("invalid gguf magic".into());
            assert!(!hard.is_gpu_init_failure());
        }

        #[test]
        fn gpu_budget_env_roundtrip() {
            std::env::remove_var(GPU_ENV_VAR);
            GpuBudget::CpuOnly.apply_env();
            assert_eq!(std::env::var(GPU_ENV_VAR).as_deref(), Ok("0"));
            GpuBudget::Default.apply_env();
            assert!(std::env::var(GPU_ENV_VAR).is_err());
        }
    }
}

// ---------------------------------------------------------------------------
// Stub mode — no native runtime linked.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "native"))]
mod imp {
    use super::*;

    /// Placeholder session; every operation reports
    /// [`InferError::Unsupported`].
    #[derive(Debug, Default)]
    pub struct RawAsrSession;

    impl RawAsrSession {
        pub fn open(_model_path: &Path, _budget: GpuBudget) -> Result<Self> {
            Err(unsupported())
        }

        pub fn transcribe(
            &self,
            _samples: &[f32],
            _language: Option<&str>,
        ) -> Result<RawTranscript> {
            Err(unsupported())
        }
    }

    pub fn align_words(
        _model_path: &Path,
        _transcript: &str,
        _samples: &[f32],
        _n_threads: i32,
        _t_offset_cs: i64,
    ) -> Result<Vec<(String, f64, f64)>> {
        Err(unsupported())
    }

    pub fn vad_regions(
        _model_path: &Path,
        _samples: &[f32],
        _threshold: f32,
        _min_speech_ms: i32,
        _min_silence_ms: i32,
        _use_gpu: bool,
        _n_threads: i32,
    ) -> Result<Vec<(f64, f64)>> {
        Err(unsupported())
    }

    fn unsupported() -> InferError {
        InferError::Unsupported(
            "the CrispASR native runtime is not linked in; rebuild with \
             --features native (see crates/subtitle-infer/Cargo.toml for the \
             libcrispasr linking modes)"
                .into(),
        )
    }
}

pub use imp::{align_words, vad_regions, RawAsrSession};

#[cfg(all(test, not(feature = "native")))]
mod stub_tests {
    use super::*;

    #[test]
    fn stub_transcribe_is_unsupported_not_panic() {
        let session = RawAsrSession::open(Path::new("x.gguf"), GpuBudget::Default);
        assert!(session.is_err());
    }
}
