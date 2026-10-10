//! Unified inference engine implementing `subtitle_core::pipeline::InferenceBackend`.
//!
//! Uses a dedicated worker thread to maintain thread affinity for ggml / CrispASR
//! sessions while presenting a thread-safe (`Send + Sync`) interface to the Tokio runtime.

use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use subtitle_core::error::{Error as CoreError, Result as CoreResult};
use subtitle_core::pipeline::InferenceBackend;
use subtitle_core::types::TimedToken;

use crate::aligner::AlignerSession;
use crate::asr::AsrSession;
use crate::model::{DEFAULT_ALIGNER_GGUF, DEFAULT_ASR_GGUF, DEFAULT_VAD_GGUF};
use crate::session::LoadOptions;
use crate::vad::{VadOptions, VadSession};

enum InferenceCommand {
    DetectVad {
        samples: Vec<f32>,
        resp: Sender<CoreResult<Vec<(f64, f64)>>>,
    },
    Transcribe {
        samples: Vec<f32>,
        hint: Option<String>,
        resp: Sender<CoreResult<(String, Option<String>)>>,
    },
    Align {
        samples: Vec<f32>,
        transcript: String,
        lang: Option<String>,
        resp: Sender<CoreResult<Vec<TimedToken>>>,
    },
}

/// The unified GGUF inference engine.
pub struct InferenceEngine {
    tx: Mutex<Sender<InferenceCommand>>,
    _handle: Option<JoinHandle<()>>,
}

impl InferenceEngine {
    /// Load VAD, ASR, and forced-aligner models on a dedicated inference worker thread.
    pub fn load(
        vad_spec: Option<&str>,
        asr_spec: Option<&str>,
        aligner_spec: Option<&str>,
        load_opts: &LoadOptions,
    ) -> CoreResult<Self> {
        let vad_spec = vad_spec.unwrap_or(DEFAULT_VAD_GGUF).to_string();
        let asr_spec = asr_spec.unwrap_or(DEFAULT_ASR_GGUF).to_string();
        let aligner_spec = aligner_spec.unwrap_or(DEFAULT_ALIGNER_GGUF).to_string();
        let opts = load_opts.clone();

        let (cmd_tx, cmd_rx) = channel::<InferenceCommand>();
        let (init_tx, init_rx) = channel::<CoreResult<()>>();

        let handle = thread::spawn(move || {
            // Load sessions inside worker thread
            let vad_res = VadSession::load(&vad_spec, VadOptions::default(), &opts);
            let asr_res = AsrSession::load(&asr_spec, &opts);
            let aligner_res = AlignerSession::load(&aligner_spec, &opts);

            let (vad, asr, aligner) = match (vad_res, asr_res, aligner_res) {
                (Ok(v), Ok(a), Ok(al)) => {
                    let _ = init_tx.send(Ok(()));
                    (v, a, al)
                }
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                    let _ = init_tx.send(Err(CoreError::Pipeline(format!(
                        "Failed to load inference models: {e}"
                    ))));
                    return;
                }
            };

            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    InferenceCommand::DetectVad { samples, resp } => {
                        let res = vad
                            .detect(&samples)
                            .map(|regions| regions.into_iter().map(|r| (r.start, r.end)).collect())
                            .map_err(|e| CoreError::Pipeline(e.to_string()));
                        let _ = resp.send(res);
                    }
                    InferenceCommand::Transcribe { samples, hint, resp } => {
                        let res = asr
                            .transcribe(&samples, hint.as_deref())
                            .map(|out| (out.text, out.language))
                            .map_err(|e| CoreError::Pipeline(e.to_string()));
                        let _ = resp.send(res);
                    }
                    InferenceCommand::Align {
                        samples,
                        transcript,
                        lang,
                        resp,
                    } => {
                        let res = aligner
                            .align(&samples, &transcript, lang.as_deref())
                            .map_err(|e| CoreError::Pipeline(e.to_string()));
                        let _ = resp.send(res);
                    }
                }
            }
        });

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx: Mutex::new(cmd_tx),
                _handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(CoreError::Pipeline(format!(
                "Inference worker thread initialization failed: {e}"
            ))),
        }
    }
}

impl InferenceBackend for InferenceEngine {
    fn detect_speech_regions(&self, samples: &[f32]) -> CoreResult<Vec<(f64, f64)>> {
        let (resp_tx, resp_rx) = channel();
        let cmd = InferenceCommand::DetectVad {
            samples: samples.to_vec(),
            resp: resp_tx,
        };
        self.tx
            .lock()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
            .send(cmd)
            .map_err(|e| CoreError::Pipeline(e.to_string()))?;

        resp_rx
            .recv()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
    }

    fn transcribe(
        &self,
        samples: &[f32],
        language_hint: Option<&str>,
    ) -> CoreResult<(String, Option<String>)> {
        let (resp_tx, resp_rx) = channel();
        let cmd = InferenceCommand::Transcribe {
            samples: samples.to_vec(),
            hint: language_hint.map(String::from),
            resp: resp_tx,
        };
        self.tx
            .lock()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
            .send(cmd)
            .map_err(|e| CoreError::Pipeline(e.to_string()))?;

        resp_rx
            .recv()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
    }

    fn align(
        &self,
        samples: &[f32],
        transcript: &str,
        language: Option<&str>,
    ) -> CoreResult<Vec<TimedToken>> {
        let (resp_tx, resp_rx) = channel();
        let cmd = InferenceCommand::Align {
            samples: samples.to_vec(),
            transcript: transcript.to_string(),
            lang: language.map(String::from),
            resp: resp_tx,
        };
        self.tx
            .lock()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
            .send(cmd)
            .map_err(|e| CoreError::Pipeline(e.to_string()))?;

        resp_rx
            .recv()
            .map_err(|e| CoreError::Pipeline(e.to_string()))?
    }
}
