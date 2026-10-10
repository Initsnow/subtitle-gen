//! Hardware detection and the GPU → CPU automatic fallback state machine.
//!
//! The CrispASR runtime selects its backend internally at session open and
//! exposes exactly two externally controllable budget states — the
//! `CRISPASR_N_GPU_LAYERS` environment variable: unset (runtime default =
//! full GPU offload when built with a GPU backend) or `0` (CPU only). The
//! ladder therefore has at most one GPU rung (the highest-preference backend
//! compiled into `libcrispasr`, chosen via the Cargo features) plus the
//! final CPU rung.
//!
//! [`FallbackSelector::select`] walks the ladder around an injected attempt
//! closure. GPU initialization failures — out-of-memory, missing driver, no
//! usable device — are recorded with a `tracing` warning and the next state
//! is tried; the final attempt is always CPU with the budget pinned to `0`.
//! The state machine never panics on GPU trouble: it either lands on a
//! working plan or propagates the last error to the caller.

use crate::error::InferError;

/// Compute backends understood by the runtime, in preference order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuBackend {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl GpuBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuBackend::Cpu => "cpu",
            GpuBackend::Cuda => "cuda",
            GpuBackend::Vulkan => "vulkan",
            GpuBackend::Metal => "metal",
        }
    }

    /// Whether this backend's Cargo feature is compiled in *and* legal for
    /// the current target OS. This reflects what `libcrispasr` was built
    /// with — the final word is the runtime's own device probe at open time.
    pub fn compiled_in(&self) -> bool {
        match self {
            GpuBackend::Cpu => true,
            // CUDA ships on Windows and Linux only.
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            GpuBackend::Cuda => true,
            #[cfg(not(all(feature = "cuda", any(target_os = "windows", target_os = "linux"))))]
            GpuBackend::Cuda => false,
            // Vulkan on Windows/Linux; macOS uses Metal instead.
            #[cfg(all(feature = "vulkan", any(target_os = "windows", target_os = "linux")))]
            GpuBackend::Vulkan => true,
            #[cfg(not(all(feature = "vulkan", any(target_os = "windows", target_os = "linux"))))]
            GpuBackend::Vulkan => false,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            GpuBackend::Metal => true,
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            GpuBackend::Metal => false,
        }
    }

    /// The highest-preference GPU backend compiled into this build, if any.
    pub fn first_compiled() -> Option<GpuBackend> {
        [GpuBackend::Cuda, GpuBackend::Vulkan, GpuBackend::Metal]
            .into_iter()
            .find(|b| b.compiled_in())
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// CPU capabilities relevant to the ggml kernel dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInfo {
    /// Logical core count (`available_parallelism`), the default `n_threads`.
    pub logical_cores: usize,
    pub avx2: bool,
    pub avx512: bool,
    /// Always `true` on aarch64 (NEON is baseline), `false` elsewhere.
    pub neon: bool,
}

impl CpuInfo {
    /// Sensible default thread count: all logical cores, at least 1.
    pub fn default_threads(&self) -> i32 {
        self.logical_cores.clamp(1, i32::MAX as usize) as i32
    }
}

/// Probe the local CPU. Cheap and side-effect free.
pub fn probe_cpu() -> CpuInfo {
    let logical_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    CpuInfo {
        logical_cores,
        avx2: is_x86_feature_detected!("avx2"),
        avx512: is_x86_feature_detected!("avx512f"),
        neon: cfg!(target_arch = "aarch64"),
    }
}

/// One recorded attempt on the ladder, kept for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptNote {
    pub backend: GpuBackend,
    pub n_gpu_layers: i32,
    pub outcome: AttemptOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// Attempt succeeded on this rung.
    Ok,
    /// Attempt rejected before or during the call.
    Failed(String),
}

/// The device plan a session ended up running on.
///
/// Note: when the GPU rung succeeds, `backend` records the *requested*
/// preference; the runtime may internally pick any backend it was built
/// with. `n_gpu_layers` reflects the budget actually applied (`-1` default,
/// `0` CPU).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePlan {
    pub backend: GpuBackend,
    pub n_gpu_layers: i32,
    pub attempts: Vec<AttemptNote>,
}

impl DevicePlan {
    /// `n_gpu_layers` budget meaning "runtime default" (GPU when available).
    pub const GPU_LAYERS_ALL: i32 = -1;
    /// `n_gpu_layers` budget meaning "keep all compute on the CPU".
    pub const GPU_LAYERS_NONE: i32 = 0;

    /// True when compute runs (even partially) on a GPU.
    pub fn uses_gpu(&self) -> bool {
        self.backend != GpuBackend::Cpu && self.n_gpu_layers != Self::GPU_LAYERS_NONE
    }
}

/// Ordered backend ladder — pure function of the candidate list, so it is
/// unit-testable without touching real hardware.
///
/// Preference order: CUDA (fastest, most mature ggml path) → Vulkan → Metal →
/// CPU. The CPU backend is always last and always present.
pub fn build_ladder(candidates: &[GpuBackend]) -> Vec<GpuBackend> {
    let order = [GpuBackend::Cuda, GpuBackend::Vulkan, GpuBackend::Metal];
    let mut ladder: Vec<GpuBackend> =
        order.iter().filter(|b| candidates.contains(b)).copied().collect();
    ladder.push(GpuBackend::Cpu);
    ladder
}

/// The fallback state machine. Walks the ladder around an injected attempt
/// closure so tests can simulate OOM / driver failures without hardware.
#[derive(Debug, Clone)]
pub struct FallbackSelector {
    ladder: Vec<GpuBackend>,
}

impl FallbackSelector {
    /// Detect hardware and compile the ladder: the single highest-preference
    /// GPU rung compiled into this build (the runtime itself has only two
    /// budget states — default and CPU-only — so extra GPU rungs would only
    /// repeat the same attempt), followed by the CPU rung.
    pub fn detect() -> Self {
        let gpu = GpuBackend::first_compiled();
        let ladder: Vec<GpuBackend> =
            gpu.into_iter().chain(std::iter::once(GpuBackend::Cpu)).collect();
        Self { ladder }
    }

    /// Ladder from an explicit candidate list (used by tests and by callers
    /// that already know their backend situation).
    pub fn from_candidates(candidates: Vec<GpuBackend>) -> Self {
        Self { ladder: build_ladder(&candidates) }
    }

    pub fn ladder(&self) -> &[GpuBackend] {
        &self.ladder
    }

    /// Walk the ladder. `attempt` receives the backend rung and the GPU
    /// layer budget ([`DevicePlan::GPU_LAYERS_ALL`] for the GPU rung,
    /// [`DevicePlan::GPU_LAYERS_NONE`] for the final CPU attempt). Errors
    /// matching [`InferError::is_gpu_init_failure`] advance the state
    /// machine; other errors fail fast.
    pub fn select<T>(
        &self,
        mut attempt: impl FnMut(GpuBackend, i32) -> Result<T, InferError>,
    ) -> Result<(T, DevicePlan), InferError> {
        let mut attempts: Vec<AttemptNote> = Vec::new();
        let mut last_gpu_error: Option<InferError> = None;

        for &backend in &self.ladder {
            if backend == GpuBackend::Cpu {
                break;
            }
            match attempt(backend, DevicePlan::GPU_LAYERS_ALL) {
                Ok(value) => {
                    attempts.push(AttemptNote {
                        backend,
                        n_gpu_layers: DevicePlan::GPU_LAYERS_ALL,
                        outcome: AttemptOutcome::Ok,
                    });
                    return Ok((
                        value,
                        DevicePlan {
                            backend,
                            n_gpu_layers: DevicePlan::GPU_LAYERS_ALL,
                            attempts,
                        },
                    ));
                }
                Err(err) => {
                    if err.is_gpu_init_failure() {
                        tracing::warn!(
                            backend = %backend,
                            reason = %err,
                            "GPU backend failed to initialize; falling back"
                        );
                        attempts.push(AttemptNote {
                            backend,
                            n_gpu_layers: DevicePlan::GPU_LAYERS_ALL,
                            outcome: AttemptOutcome::Failed(err.to_string()),
                        });
                        last_gpu_error = Some(err);
                        continue;
                    }
                    // Hard failure (corrupt model, bad argument, runtime
                    // bug): retrying on another device cannot help.
                    return Err(err);
                }
            }
        }

        // Final state: CPU, all layers resident in system memory.
        match attempt(GpuBackend::Cpu, DevicePlan::GPU_LAYERS_NONE) {
            Ok(value) => {
                if let Some(err) = &last_gpu_error {
                    tracing::warn!(
                        reason = %err,
                        "no GPU backend available; running on CPU (n_gpu_layers = 0)"
                    );
                }
                attempts.push(AttemptNote {
                    backend: GpuBackend::Cpu,
                    n_gpu_layers: DevicePlan::GPU_LAYERS_NONE,
                    outcome: AttemptOutcome::Ok,
                });
                Ok((
                    value,
                    DevicePlan {
                        backend: GpuBackend::Cpu,
                        n_gpu_layers: DevicePlan::GPU_LAYERS_NONE,
                        attempts,
                    },
                ))
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::InferError;

    fn oom(backend: GpuBackend) -> InferError {
        InferError::Oom { backend: backend.as_str().into(), reason: "alloc failed".into() }
    }

    #[test]
    fn ladder_orders_gpu_backends_then_cpu() {
        let ladder = build_ladder(&[
            GpuBackend::Metal,
            GpuBackend::Cpu,
            GpuBackend::Cuda,
            GpuBackend::Vulkan,
        ]);
        assert_eq!(
            ladder,
            vec![GpuBackend::Cuda, GpuBackend::Vulkan, GpuBackend::Metal, GpuBackend::Cpu]
        );
    }

    #[test]
    fn ladder_without_gpus_is_cpu_only() {
        assert_eq!(build_ladder(&[]), vec![GpuBackend::Cpu]);
    }

    #[test]
    fn select_prefers_first_working_backend() {
        let selector =
            FallbackSelector::from_candidates(vec![GpuBackend::Cuda, GpuBackend::Vulkan]);
        let (value, plan) = selector
            .select(|backend, _layers| match backend {
                GpuBackend::Cuda => Err(oom(backend)),
                _ => Ok(backend.as_str().to_string()),
            })
            .unwrap();
        assert_eq!(value, "vulkan");
        assert!(plan.uses_gpu());
        assert_eq!(plan.backend, GpuBackend::Vulkan);
        assert_eq!(plan.n_gpu_layers, DevicePlan::GPU_LAYERS_ALL);
        assert_eq!(plan.attempts.len(), 2);
    }

    #[test]
    fn select_degrades_to_cpu_zero_layers_on_total_gpu_failure() {
        let selector = FallbackSelector::from_candidates(vec![GpuBackend::Cuda]);
        let (value, plan) = selector
            .select(|backend, layers| match backend {
                GpuBackend::Cpu => {
                    assert_eq!(layers, DevicePlan::GPU_LAYERS_NONE);
                    Ok("cpu".to_string())
                }
                _ => Err(oom(backend)),
            })
            .unwrap();
        assert_eq!(value, "cpu");
        assert!(!plan.uses_gpu());
        assert_eq!(plan.backend, GpuBackend::Cpu);
        assert_eq!(plan.n_gpu_layers, DevicePlan::GPU_LAYERS_NONE);
        // CUDA attempt (failed) + CPU attempt (ok).
        assert_eq!(plan.attempts.len(), 2);
        assert!(matches!(
            &plan.attempts[0].outcome,
            AttemptOutcome::Failed(msg) if msg.contains("out of memory")
        ));
    }

    #[test]
    fn select_fails_fast_on_hard_errors() {
        let selector = FallbackSelector::from_candidates(vec![GpuBackend::Cuda]);
        let err = selector
            .select(|backend, _layers| {
                if backend == GpuBackend::Cuda {
                    Err(InferError::InvalidArgument("corrupt gguf".into()))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert!(matches!(err, InferError::InvalidArgument(_)));
    }

    #[test]
    fn select_propagates_cpu_error_when_everything_fails() {
        let selector = FallbackSelector::from_candidates(vec![]);
        let err = selector
            .select(|_backend, _layers| -> Result<String, InferError> {
                Err(InferError::Runtime { code: -1, message: "boom".into() })
            })
            .unwrap_err();
        assert!(matches!(err, InferError::Runtime { .. }));
    }

    #[test]
    fn cpu_probe_reports_positive_core_count() {
        let cpu = probe_cpu();
        assert!(cpu.logical_cores >= 1);
        assert!(cpu.default_threads() >= 1);
    }

    #[test]
    fn cpu_backend_is_always_compiled_in() {
        assert!(GpuBackend::Cpu.compiled_in());
    }

    #[test]
    fn detect_ladder_has_at_most_one_gpu_rung() {
        let ladder = FallbackSelector::detect().ladder().to_vec();
        assert_eq!(*ladder.last().unwrap(), GpuBackend::Cpu);
        assert!(ladder.len() <= 2);
    }
}
