//! End-to-end streaming subtitle generation and lyrics alignment pipeline.
//!
//! Uses Tokio MPSC channels to overlap stages:
//! audio decode/resample -> VAD windowing -> ASR transcription ->
//! forced alignment -> Lindera local text segmentation -> optional LLM proofread/translate.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::audio::AudioBuffer;
use crate::cache::{
    asr_cache_path, cleanup_cache, compute_audio_sha256, load_cached_alignment_tokens,
    load_cached_asr_result, refine_cache_path, save_alignment_tokens, save_asr_result,
};
use crate::config::{AppConfig, VadConfig};
use crate::error::{Error, Result};
use crate::llm::SubtitleLlm;
use crate::text::{
    filter_spurious_tokens, finalize_spans, match_lines_to_tokens, plan_fixed_windows,
    LocalSegmenter,
};
use crate::types::{AudioWindow, SubtitleItem, TimedToken, TranscriptChunk};

pub type ProgressCallback = Arc<dyn Fn(&str) + Send + Sync>;

fn report(progress: &Option<ProgressCallback>, msg: &str) {
    if let Some(p) = progress {
        p(msg);
    }
}

/// Abstract speech inference backend implemented by `subtitle-infer`.
pub trait InferenceBackend: Send + Sync {
    /// Voice activity detection: returns speech intervals `(start_sec, end_sec)`.
    fn detect_speech_regions(&self, samples: &[f32]) -> Result<Vec<(f64, f64)>>;

    /// Automatic speech recognition: returns `(transcript_text, detected_language)`.
    fn transcribe(
        &self,
        samples: &[f32],
        language_hint: Option<&str>,
    ) -> Result<(String, Option<String>)>;

    /// Forced alignment: returns word-level timestamps relative to `samples[0]`.
    fn align(
        &self,
        samples: &[f32],
        transcript: &str,
        language: Option<&str>,
    ) -> Result<Vec<TimedToken>>;
}

#[derive(Clone)]
pub struct PipelineOptions {
    pub input_path: PathBuf,
    pub translate: Option<String>,
    pub segment_mode: Option<String>,
    pub overwrite_cache: bool,
    pub progress: Option<ProgressCallback>,
    pub stream_token: Option<crate::llm::StreamCallback>,
}

impl PipelineOptions {
    pub fn new(input_path: impl Into<PathBuf>) -> Self {
        Self {
            input_path: input_path.into(),
            translate: None,
            segment_mode: None,
            overwrite_cache: false,
            progress: None,
            stream_token: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub subtitles: Vec<SubtitleItem>,
    pub transcript_chunks: Vec<TranscriptChunk>,
    pub detected_language: Option<String>,
}

#[derive(Debug, Clone)]
struct ProcessedWindowData {
    window: AudioWindow,
    transcript: String,
    language: Option<String>,
    local_tokens: Option<Vec<TimedToken>>,
}

/// The main subtitle generation pipeline.
pub struct SubtitlePipeline<B: InferenceBackend> {
    pub config: AppConfig,
    pub backend: Arc<B>,
}

impl<B: InferenceBackend + 'static> SubtitlePipeline<B> {
    pub fn new(config: AppConfig, backend: Arc<B>) -> Self {
        Self { config, backend }
    }

    /// Run end-to-end subtitle generation on a media file.
    pub async fn run(&self, options: PipelineOptions) -> Result<PipelineResult> {
        let input_path = &options.input_path;
        report(
            &options.progress,
            &format!("input: {}", input_path.display()),
        );

        let segment_mode = options
            .segment_mode
            .as_deref()
            .unwrap_or(&self.config.segment.mode)
            .to_string();

        let use_cache = self.config.cache.enabled;
        let cache_root = if use_cache {
            PathBuf::from(&self.config.cache_dir)
        } else {
            std::env::temp_dir().join(format!("subgen-tmp-{}", std::process::id()))
        };

        report(
            &options.progress,
            &format!(
                "mode: {}; cache: {}",
                segment_mode,
                if use_cache { "enabled" } else { "disabled" }
            ),
        );
        report(&options.progress, "preparing 16 kHz mono audio");

        let audio = AudioBuffer::load(input_path)?;
        let duration = audio.duration();
        let samples = audio.samples();
        let media_id = compute_audio_sha256(samples);

        report(
            &options.progress,
            &format!(
                "audio duration: {:.2}s; media id: {}",
                duration,
                &media_id[..12.min(media_id.len())]
            ),
        );

        report(&options.progress, "planning speech windows");
        let windows = if self.config.vad.enabled {
            let regions = self.backend.detect_speech_regions(samples)?;
            plan_vad_windows(&regions, duration, &self.config.vad, Some(samples))
        } else {
            plan_fixed_windows(duration, 30.0, Some(1.0))
        };

        report(
            &options.progress,
            &format!("planned {} chunk window(s)", windows.len()),
        );

        if use_cache && self.config.cache.cleanup_enabled {
            report(&options.progress, "cleaning stale cache entries");
            let mut active_map = HashMap::new();
            active_map.insert(media_id.clone(), windows.clone());
            let mut keep_set = HashSet::new();
            keep_set.insert(media_id.clone());
            cleanup_cache(
                &cache_root,
                &keep_set,
                Some(&active_map),
                None,
                self.config.cache.max_media_entries,
            );
        }

        // Tokio MPSC streaming execution for chunk processing
        let (tx, mut rx) = mpsc::channel::<ProcessedWindowData>(32);
        let backend = Arc::clone(&self.backend);
        let config = self.config.clone();
        let media_id_clone = media_id.clone();
        let cache_root_clone = cache_root.clone();
        let windows_clone = windows.clone();
        let samples_vec = samples.to_vec();
        let with_alignment = segment_mode != "none";
        let overwrite = options.overwrite_cache;
        let progress = options.progress.clone();

        tokio::task::spawn_blocking(move || {
            let total = windows_clone.len();
            for (idx, window) in windows_clone.iter().enumerate() {
                let chunk_num = idx + 1;
                let start_idx = ((window.start * 16000.0) as usize).min(samples_vec.len());
                let end_idx = ((window.end * 16000.0) as usize).min(samples_vec.len());
                let window_samples = &samples_vec[start_idx..end_idx];

                let cache_file = asr_cache_path(&cache_root_clone, &media_id_clone, window);

                // 1. ASR
                report(
                    &progress,
                    &format!(
                        "chunk {chunk_num}/{total}: transcribing [{:.1}s - {:.1}s]",
                        window.start, window.end
                    ),
                );

                let (transcript, language) = if use_cache && !overwrite {
                    if let Some(cached) = load_cached_asr_result(
                        &cache_file,
                        &config.model.asr_model,
                        config.model.language.as_deref(),
                    ) {
                        report(
                            &progress,
                            &format!("chunk {chunk_num}/{total}: ASR cache hit"),
                        );
                        cached
                    } else {
                        match backend.transcribe(window_samples, config.model.language.as_deref()) {
                            Ok((text, lang)) => {
                                let _ = save_asr_result(
                                    &cache_file,
                                    window,
                                    &config.model.asr_model,
                                    config.model.language.as_deref(),
                                    &text,
                                    lang.as_deref(),
                                );
                                (text, lang)
                            }
                            Err(_) => (String::new(), None),
                        }
                    }
                } else {
                    match backend.transcribe(window_samples, config.model.language.as_deref()) {
                        Ok((text, lang)) => {
                            if use_cache {
                                let _ = save_asr_result(
                                    &cache_file,
                                    window,
                                    &config.model.asr_model,
                                    config.model.language.as_deref(),
                                    &text,
                                    lang.as_deref(),
                                );
                            }
                            (text, lang)
                        }
                        Err(_) => (String::new(), None),
                    }
                };

                // 2. Alignment
                let local_tokens = if with_alignment && !transcript.trim().is_empty() {
                    report(
                        &progress,
                        &format!("chunk {chunk_num}/{total}: forced alignment"),
                    );
                    let alignment_lang = language.as_deref().unwrap_or("en");
                    if use_cache && !overwrite {
                        if let Some(cached_toks) = load_cached_alignment_tokens(
                            &cache_file,
                            &config.model.aligner_model,
                            alignment_lang,
                            &transcript,
                        ) {
                            report(
                                &progress,
                                &format!("chunk {chunk_num}/{total}: alignment cache hit"),
                            );
                            Some(cached_toks)
                        } else {
                            match backend.align(window_samples, &transcript, Some(alignment_lang)) {
                                Ok(toks) => {
                                    let _ = save_alignment_tokens(
                                        &cache_file,
                                        window,
                                        &config.model.aligner_model,
                                        alignment_lang,
                                        &transcript,
                                        &toks,
                                    );
                                    Some(toks)
                                }
                                Err(_) => Some(Vec::new()),
                            }
                        }
                    } else {
                        match backend.align(window_samples, &transcript, Some(alignment_lang)) {
                            Ok(toks) => {
                                if use_cache {
                                    let _ = save_alignment_tokens(
                                        &cache_file,
                                        window,
                                        &config.model.aligner_model,
                                        alignment_lang,
                                        &transcript,
                                        &toks,
                                    );
                                }
                                Some(toks)
                            }
                            Err(_) => Some(Vec::new()),
                        }
                    }
                } else {
                    None
                };

                let item = ProcessedWindowData {
                    window: *window,
                    transcript,
                    language,
                    local_tokens,
                };

                if tx.blocking_send(item).is_err() {
                    break;
                }
            }
        });

        let mut processed_windows = Vec::new();
        while let Some(item) = rx.recv().await {
            processed_windows.push(item);
        }

        // Sort by window index
        processed_windows.sort_by_key(|w| w.window.index);

        let mut detected_language: Option<String> = None;
        let mut chunks = Vec::new();
        let mut segmented_subtitles = Vec::new();

        let refine_mode = if segment_mode == "local" {
            "hard"
        } else {
            "none"
        };
        let segmenter = LocalSegmenter::new(Some(self.config.segment.clone()), Some(refine_mode));

        for pw in &processed_windows {
            if detected_language.is_none() && pw.language.is_some() {
                detected_language = pw.language.clone();
            }

            chunks.push(TranscriptChunk {
                index: pw.window.index,
                start: pw.window.start,
                end: pw.window.end,
                text: pw.transcript.clone(),
                language: pw.language.clone(),
            });

            if segment_mode != "none" {
                if let Some(ref local_toks) = pw.local_tokens {
                    let global_tokens: Vec<TimedToken> = local_toks
                        .iter()
                        .map(|t| t.shifted(pw.window.start))
                        .collect();

                    let items = segmenter.segment(
                        &global_tokens,
                        Some(&pw.transcript),
                        pw.language.as_deref().or(detected_language.as_deref()),
                    );
                    segmented_subtitles.extend(items);
                }
            }
        }

        let mut subtitles = if segment_mode != "none" {
            segmented_subtitles
        } else {
            subtitles_from_chunks(&chunks)
        };

        // Renumber IDs sequentially starting from 1
        for (i, item) in subtitles.iter_mut().enumerate() {
            item.id = i + 1;
        }

        report(
            &options.progress,
            &format!("generated {} subtitle cue(s)", subtitles.len()),
        );

        // Optional LLM proofreading and translation
        if let Some(target_lang) = options.translate {
            if !self.config.llm.api_key.is_empty() {
                let llm = SubtitleLlm::new(self.config.llm.clone());
                report(&options.progress, "proofreading subtitles with LLM");
                subtitles = llm
                    .proofread(
                        &subtitles,
                        options.stream_token.clone(),
                        options.progress.clone(),
                    )
                    .await?;
                report(
                    &options.progress,
                    &format!("translating subtitles to {target_lang} with LLM"),
                );
                subtitles = llm
                    .translate(
                        &subtitles,
                        &target_lang,
                        options.stream_token.clone(),
                        options.progress.clone(),
                    )
                    .await?;
                report(&options.progress, "LLM translation complete");
            } else {
                report(
                    &options.progress,
                    "warning: --translate requested but no [llm].api_key configured; skipping LLM stage",
                );
            }
        }

        Ok(PipelineResult {
            subtitles,
            transcript_chunks: chunks,
            detected_language,
        })
    }

    /// Add timestamps to untimed lyrics or reference text lines by aligning them against audio.
    #[allow(clippy::too_many_arguments)]
    pub async fn align_lines_to_audio(
        &self,
        audio_path: &Path,
        lines: &[String],
        language: Option<&str>,
        refine: bool,
        use_vad: bool,
        progress: Option<ProgressCallback>,
        overwrite_cache: bool,
    ) -> Result<Vec<SubtitleItem>> {
        if lines.is_empty() {
            return Ok(Vec::new());
        }

        report(
            &progress,
            &format!(
                "aligning {} line(s) against {}",
                lines.len(),
                audio_path.display()
            ),
        );
        report(&progress, "preparing 16 kHz mono audio");

        let audio = AudioBuffer::load(audio_path)?;
        let duration = audio.duration();
        let samples = audio.samples();
        let media_id = compute_audio_sha256(samples);

        let windows = if use_vad && self.config.vad.enabled {
            let regions = self.backend.detect_speech_regions(samples)?;
            plan_vad_windows(&regions, duration, &self.config.vad, Some(samples))
        } else {
            plan_fixed_windows(duration, 30.0, Some(1.0))
        };

        report(
            &progress,
            &format!("planned {} alignment window(s)", windows.len()),
        );

        let use_cache = self.config.cache.enabled;
        let cache_root = if use_cache {
            PathBuf::from(&self.config.cache_dir)
        } else {
            std::env::temp_dir().join(format!("subgen-tmp-align-{}", std::process::id()))
        };

        // Pass 1: Transcribe and force-align speech windows to obtain rough token timestamps
        let mut window_tokens: Vec<(AudioWindow, Vec<TimedToken>)> = Vec::new();
        let mut detected_language: Option<String> = None;
        let total_windows = windows.len();

        for (idx, window) in windows.iter().enumerate() {
            let chunk_num = idx + 1;
            report(
                &progress,
                &format!(
                    "window {chunk_num}/{total_windows}: transcribing [{:.1}s - {:.1}s]",
                    window.start, window.end
                ),
            );
            let start_idx = ((window.start * 16000.0) as usize).min(samples.len());
            let end_idx = ((window.end * 16000.0) as usize).min(samples.len());
            let window_samples = &samples[start_idx..end_idx];

            let cache_file = asr_cache_path(&cache_root, &media_id, window);
            let (transcript, lang) = if use_cache && !overwrite_cache {
                if let Some(c) =
                    load_cached_asr_result(&cache_file, &self.config.model.asr_model, language)
                {
                    report(
                        &progress,
                        &format!("window {chunk_num}/{total_windows}: ASR cache hit"),
                    );
                    c
                } else {
                    let res = self.backend.transcribe(window_samples, language)?;
                    let _ = save_asr_result(
                        &cache_file,
                        window,
                        &self.config.model.asr_model,
                        language,
                        &res.0,
                        res.1.as_deref(),
                    );
                    res
                }
            } else {
                let res = self.backend.transcribe(window_samples, language)?;
                if use_cache {
                    let _ = save_asr_result(
                        &cache_file,
                        window,
                        &self.config.model.asr_model,
                        language,
                        &res.0,
                        res.1.as_deref(),
                    );
                }
                res
            };

            if detected_language.is_none() && lang.is_some() {
                detected_language = lang.clone();
            }

            if !transcript.trim().is_empty() {
                report(
                    &progress,
                    &format!("window {chunk_num}/{total_windows}: forced alignment"),
                );
                let alignment_lang = language.or(detected_language.as_deref()).unwrap_or("en");
                let tokens = if use_cache && !overwrite_cache {
                    if let Some(c_toks) = load_cached_alignment_tokens(
                        &cache_file,
                        &self.config.model.aligner_model,
                        alignment_lang,
                        &transcript,
                    ) {
                        report(
                            &progress,
                            &format!("window {chunk_num}/{total_windows}: alignment cache hit"),
                        );
                        c_toks
                    } else {
                        let toks = self.backend.align(
                            window_samples,
                            &transcript,
                            Some(alignment_lang),
                        )?;
                        let _ = save_alignment_tokens(
                            &cache_file,
                            window,
                            &self.config.model.aligner_model,
                            alignment_lang,
                            &transcript,
                            &toks,
                        );
                        toks
                    }
                } else {
                    let toks =
                        self.backend
                            .align(window_samples, &transcript, Some(alignment_lang))?;
                    if use_cache {
                        let _ = save_alignment_tokens(
                            &cache_file,
                            window,
                            &self.config.model.aligner_model,
                            alignment_lang,
                            &transcript,
                            &toks,
                        );
                    }
                    toks
                };
                window_tokens.push((*window, tokens));
            }
        }

        let merged_tokens = crate::text::merge_window_tokens(&window_tokens);
        let viable_tokens = filter_spurious_tokens(&merged_tokens);
        let match_lang = language.or(detected_language.as_deref());

        let mut spans = match_lines_to_tokens(lines, &viable_tokens, match_lang);
        if !spans.iter().any(|s| s.is_some()) && !merged_tokens.is_empty() {
            spans = match_lines_to_tokens(lines, &merged_tokens, match_lang);
        }

        if !spans.iter().any(|s| s.is_some()) {
            return Err(Error::Pipeline(
                "No recognized speech tokens could be matched to the input lyrics/text lines"
                    .into(),
            ));
        }

        let rough = finalize_spans(&spans, duration);
        report(
            &progress,
            &format!("matched rough spans for {} lines", lines.len()),
        );

        if !refine {
            let items = lines
                .iter()
                .zip(rough)
                .enumerate()
                .map(|(i, (text, (s, e)))| SubtitleItem::new(i + 1, s, e, text.clone()))
                .collect();
            return Ok(items);
        }

        // Pass 2: Refine each line's exact text in a short window
        report(&progress, "refining line boundaries via forced alignment");
        let total = lines.len();
        let mut refined_items = Vec::with_capacity(total);

        for (idx, (line, &(rough_start, rough_end))) in lines.iter().zip(&rough).enumerate() {
            if (idx + 1) % 5 == 0 || idx == 0 || idx + 1 == total {
                report(
                    &progress,
                    &format!("refining line {}/{total} via forced alignment", idx + 1),
                );
            }
            let mut start = rough_start;
            let mut end = rough_end;

            if !line.trim().is_empty() {
                let window = refine_window(idx + 1, &rough, duration);
                let start_idx = ((window.start * 16000.0) as usize).min(samples.len());
                let end_idx = ((window.end * 16000.0) as usize).min(samples.len());
                let window_samples = &samples[start_idx..end_idx];

                let refine_cache = refine_cache_path(&cache_root, &media_id, &window);
                let align_lang = match_lang.unwrap_or("en");

                let local_tokens = if use_cache && !overwrite_cache {
                    if let Some(t) = load_cached_alignment_tokens(
                        &refine_cache,
                        &self.config.model.aligner_model,
                        align_lang,
                        line,
                    ) {
                        t
                    } else {
                        let t = self
                            .backend
                            .align(window_samples, line, Some(align_lang))
                            .unwrap_or_default();
                        let _ = save_alignment_tokens(
                            &refine_cache,
                            &window,
                            &self.config.model.aligner_model,
                            align_lang,
                            line,
                            &t,
                        );
                        t
                    }
                } else {
                    let t = self
                        .backend
                        .align(window_samples, line, Some(align_lang))
                        .unwrap_or_default();
                    if use_cache {
                        let _ = save_alignment_tokens(
                            &refine_cache,
                            &window,
                            &self.config.model.aligner_model,
                            align_lang,
                            line,
                            &t,
                        );
                    }
                    t
                };

                let viable: Vec<TimedToken> = local_tokens
                    .into_iter()
                    .filter(|t| !t.text.trim().is_empty() && t.end > t.start)
                    .collect();

                if let (Some(first), Some(last)) = (viable.first(), viable.last()) {
                    let ref_start = (first.start + window.start).clamp(0.0, duration);
                    let ref_end = (last.end + window.start).clamp(0.0, duration);
                    if ref_end > ref_start {
                        start = ref_start;
                        end = ref_end;
                    }
                }
            }

            refined_items.push(SubtitleItem::new(
                idx + 1,
                (start * 1000.0).round() / 1000.0,
                (end * 1000.0).round() / 1000.0,
                line.clone(),
            ));
        }

        Ok(refined_items)
    }
}

fn subtitles_from_chunks(chunks: &[TranscriptChunk]) -> Vec<SubtitleItem> {
    chunks
        .iter()
        .filter(|c| !c.text.trim().is_empty())
        .enumerate()
        .map(|(i, c)| SubtitleItem::new(i + 1, c.start, c.end, c.text.clone()))
        .collect()
}

fn refine_window(index: usize, rough: &[(f64, f64)], duration: f64) -> AudioWindow {
    let (start, end) = rough[index - 1];
    let low = if index > 1 {
        (rough[index - 2].1 + start) / 2.0
    } else {
        0.0
    };
    let high = if index < rough.len() {
        (end + rough[index].0) / 2.0
    } else {
        duration
    };

    let mut window_start = low.max(start - 0.5);
    let mut window_end = high.min(end + 0.5);
    if window_end <= window_start {
        window_start = start;
        window_end = end;
    }

    AudioWindow::new(
        index,
        (window_start * 1000.0).round() / 1000.0,
        (window_end * 1000.0).round() / 1000.0,
    )
}

const SNAP_FRAME_SECONDS: f64 = 0.05;
const SNAP_TOLERANCE_SECONDS: f64 = 4.0;
const SNAP_MIN_PART_SECONDS: f64 = 5.0;

/// Group raw VAD speech intervals into chunks conforming to `VadConfig`.
///
/// When a single continuous speech interval exceeds `VadConfig.hard_max_chunk_duration` (e.g. 60s),
/// instead of slicing blindly through active speech, boundaries are snapped to the quietest nearby
/// audio frame (RMS energy valley) within a ±4.0s window to avoid cutting through words or syllables.
pub fn plan_vad_windows(
    regions: &[(f64, f64)],
    duration: f64,
    config: &VadConfig,
    samples: Option<&[f32]>,
) -> Vec<AudioWindow> {
    let max_len = if config.hard_max_chunk_duration > 0.0 {
        config.hard_max_chunk_duration
    } else {
        60.0
    };

    if regions.is_empty() {
        return split_span(0.0, duration, max_len, samples);
    }

    // Filter regions shorter than min_speech_duration
    let mut valid_regions: Vec<(f64, f64)> = regions
        .iter()
        .filter(|(s, e)| e - s >= config.min_speech_duration)
        .copied()
        .collect();

    if valid_regions.is_empty() {
        return split_span(0.0, duration, max_len, samples);
    }

    // Pad regions
    for (s, e) in &mut valid_regions {
        *s = (*s - config.speech_padding).max(0.0);
        *e = (*e + config.speech_padding).min(duration);
    }

    // Split any continuous speech regions exceeding hard_max_chunk_duration,
    // snapping boundaries to the quietest nearby frame (energy minimum)
    let mut split_regions: Vec<(f64, f64)> = Vec::new();
    for (s, e) in valid_regions {
        if e - s <= max_len {
            split_regions.push((s, e));
        } else {
            let mut parts = Vec::new();
            let mut cur = s;
            while cur < e {
                let next = (cur + max_len).min(e);
                parts.push((cur, next));
                cur = next;
            }
            let snapped = snap_boundaries_to_quiet(&parts, samples, 16_000);
            split_regions.extend(snapped);
        }
    }

    // Merge adjacent regions that overlap or gap < min_silence_duration
    let mut merged: Vec<(f64, f64)> = Vec::new();
    let mut current = split_regions[0];

    for &next in &split_regions[1..] {
        let gap = next.0 - current.1;
        let chunk_dur = next.1 - current.0;
        if gap < config.min_silence_duration && chunk_dur <= config.max_chunk_duration {
            current.1 = next.1.max(current.1);
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);

    merged
        .into_iter()
        .enumerate()
        .map(|(i, (s, e))| {
            AudioWindow::new(
                i + 1,
                (s * 1000.0).round() / 1000.0,
                (e * 1000.0).round() / 1000.0,
            )
        })
        .collect()
}

fn split_span(start: f64, end: f64, max_len: f64, samples: Option<&[f32]>) -> Vec<AudioWindow> {
    let mut parts = Vec::new();
    let mut cur = start;
    while cur < end {
        let next = (cur + max_len).min(end);
        parts.push((cur, next));
        cur = next;
    }
    let snapped = snap_boundaries_to_quiet(&parts, samples, 16_000);
    snapped
        .into_iter()
        .enumerate()
        .map(|(i, (s, e))| {
            AudioWindow::new(
                i + 1,
                (s * 1000.0).round() / 1000.0,
                (e * 1000.0).round() / 1000.0,
            )
        })
        .collect()
}

/// Snap interior hard-split boundaries to the quietest nearby frame.
///
/// For each interior boundary, search a ±4.0s window for the minimum RMS energy frame
/// and cut there instead of slicing through a word.
pub fn snap_boundaries_to_quiet(
    parts: &[(f64, f64)],
    samples: Option<&[f32]>,
    sample_rate: u32,
) -> Vec<(f64, f64)> {
    let Some(samples) = samples else {
        return parts.to_vec();
    };
    if parts.len() < 2 || samples.is_empty() {
        return parts.to_vec();
    }

    let rate = sample_rate as f64;
    let total_samples = samples.len();
    let frame_size = ((SNAP_FRAME_SECONDS * rate).round() as usize).max(1);

    let quietest_boundary = |low: f64, high: f64| -> f64 {
        let lo = ((low * rate).round() as usize).min(total_samples);
        let hi = (((high * rate).round() as usize) + frame_size).min(total_samples);
        if hi <= lo || hi - lo < frame_size {
            return (low + high) / 2.0;
        }

        let mut min_rms = f32::MAX;
        let mut best_sample_idx = lo;

        let mut idx = lo;
        while idx + frame_size <= hi {
            let frame = &samples[idx..idx + frame_size];
            let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / frame_size as f32).sqrt();
            if rms < min_rms {
                min_rms = rms;
                best_sample_idx = idx + frame_size / 2;
            }
            idx += frame_size;
        }

        best_sample_idx as f64 / rate
    };

    let mut snapped: Vec<f64> = vec![parts[0].0];
    for part in &parts[1..] {
        let nominal = part.0;
        let last_snapped = *snapped.last().unwrap();
        let low = (nominal - SNAP_TOLERANCE_SECONDS).max(last_snapped + 0.1);
        let high =
            (nominal + SNAP_TOLERANCE_SECONDS).min(parts.last().unwrap().1 - SNAP_MIN_PART_SECONDS);

        if high <= low {
            snapped.push(nominal);
            continue;
        }

        let candidate = quietest_boundary(low, high).clamp(low, high);
        snapped.push(candidate.max(last_snapped + 0.1));
    }
    snapped.push(parts.last().unwrap().1);

    let mut rebuilt = Vec::with_capacity(parts.len());
    for win in snapped.windows(2) {
        if win[1] > win[0] {
            rebuilt.push((win[0], win[1]));
        }
    }

    if rebuilt.is_empty() {
        parts.to_vec()
    } else {
        rebuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;
    impl InferenceBackend for MockBackend {
        fn detect_speech_regions(&self, _samples: &[f32]) -> Result<Vec<(f64, f64)>> {
            Ok(vec![(0.5, 3.0), (3.5, 6.0)])
        }
        fn transcribe(
            &self,
            _samples: &[f32],
            _hint: Option<&str>,
        ) -> Result<(String, Option<String>)> {
            Ok(("hello world test".into(), Some("en".into())))
        }
        fn align(
            &self,
            _samples: &[f32],
            _transcript: &str,
            _lang: Option<&str>,
        ) -> Result<Vec<TimedToken>> {
            Ok(vec![
                TimedToken::new("hello", 0.0, 0.5),
                TimedToken::new("world", 0.6, 1.2),
                TimedToken::new("test", 1.3, 1.8),
            ])
        }
    }

    #[test]
    fn test_plan_vad_windows() {
        let regions = vec![(1.0, 3.0), (3.2, 5.0)];
        let config = VadConfig::default();
        let windows = plan_vad_windows(&regions, 10.0, &config, None);
        assert_eq!(windows.len(), 1); // bridged because gap 0.2 < min_silence_duration (0.6)
    }

    #[tokio::test]
    async fn test_pipeline_with_mock_backend() {
        let backend = Arc::new(MockBackend);
        let mut config = AppConfig::default();
        config.cache.enabled = false;
        let _pipeline = SubtitlePipeline::new(config, backend);

        let lines = vec!["hello world".to_string(), "test".to_string()];
        let tokens = vec![
            TimedToken::new("hello", 0.0, 0.5),
            TimedToken::new("world", 0.6, 1.2),
            TimedToken::new("test", 1.3, 1.8),
        ];

        let spans = match_lines_to_tokens(&lines, &tokens, Some("en"));
        assert_eq!(spans.len(), 2);
    }
}
