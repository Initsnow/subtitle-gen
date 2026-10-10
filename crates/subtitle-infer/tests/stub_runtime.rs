//! End-to-end degradation tests for the stub build (no `native` feature).
//!
//! Without the linked CrispASR runtime, every session load must fail with a
//! *graceful* error — never a panic, never a link failure — so the CLI can
//! surface a clear message. Skipped automatically when built with
//! `--features native`.

#![cfg(not(feature = "native"))]

use subtitle_infer::{AsrSession, LoadOptions, VadSession};

#[test]
fn asr_load_fails_gracefully_without_native_runtime() {
    let err = AsrSession::load_default(&LoadOptions::default()).unwrap_err();
    assert!(!err.is_gpu_init_failure());
    let message = err.to_string();
    assert!(
        message.contains("runtime") || message.contains("model file not found"),
        "unexpected error: {message}"
    );
}

#[test]
fn vad_fails_gracefully_without_native_runtime() {
    let res = VadSession::load("firered-vad", Default::default(), &LoadOptions::default());
    match res {
        Err(err) => {
            assert!(!err.to_string().is_empty());
        }
        Ok(session) => {
            let err = session.detect(&[0.0; 16000]).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("support") || msg.contains("native") || msg.contains("runtime"),
                "unexpected error: {msg}"
            );
        }
    }
}

#[test]
fn stub_mode_is_reported() {
    assert!(!subtitle_infer::native_runtime_available());
}
