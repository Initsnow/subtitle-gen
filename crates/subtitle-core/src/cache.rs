//! Persistent disk cache for 16 kHz mono audio waveforms, ASR transcripts,
//! and word-level alignment tokens.
//!
//! Fully compatible with the Python cache schema (`CACHE_SCHEMA_VERSION = 1`),
//! enabling breakpoint resumes and instant skips on unchanged audio files.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::types::{AudioWindow, TimedToken};

pub const CACHE_SCHEMA_VERSION: usize = 1;
pub const CHUNK_AUDIO_SUFFIX: &str = ".wav";
pub const ASR_CACHE_SUFFIX: &str = ".json";
pub const AUDIO_CACHE_SUFFIX: &str = ".16k-mono.wav";

/// Configuration for persistent disk caching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_cache_enabled")]
    pub enabled: bool,
    #[serde(default = "default_cleanup_enabled")]
    pub cleanup_enabled: bool,
    #[serde(default = "default_max_media_entries")]
    pub max_media_entries: usize,
}

fn default_cache_enabled() -> bool { false }
fn default_cleanup_enabled() -> bool { true }
fn default_max_media_entries() -> usize { 12 }

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_cache_enabled(),
            cleanup_enabled: default_cleanup_enabled(),
            max_media_entries: default_max_media_entries(),
        }
    }
}

/// Compute SHA-256 hex digest of byte data.
pub fn compute_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Compute SHA-256 of f32 audio samples.
pub fn compute_audio_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    #[cfg(target_endian = "little")]
    {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                samples.as_ptr() as *const u8,
                samples.len() * std::mem::size_of::<f32>(),
            )
        };
        hasher.update(byte_slice);
    }
    #[cfg(target_endian = "big")]
    {
        for sample in samples {
            hasher.update(sample.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Compute SHA-1 hex digest of text (for transcript validation).
pub fn text_sha1(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())[..40].to_string()
}

/// Derive media ID from file path.
pub fn media_cache_id(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if let Some(stripped) = name.strip_suffix(AUDIO_CACHE_SUFFIX) {
        stripped.to_string()
    } else {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }
}

/// Filename stem for window chunks: `chunk-00001-0.000-5.200`.
pub fn chunk_cache_stem(window: &AudioWindow) -> String {
    format!("chunk-{:05}-{:.3}-{:.3}", window.index, window.start, window.end)
}

/// Cache file path for a window chunk's ASR transcript and alignment.
pub fn asr_cache_path(cache_root: &Path, media_id: &str, window: &AudioWindow) -> PathBuf {
    cache_root
        .join("asr")
        .join(media_id)
        .join(format!("{}{}", chunk_cache_stem(window), ASR_CACHE_SUFFIX))
}

/// Cache file path for a refine window chunk.
pub fn refine_cache_path(cache_root: &Path, media_id: &str, window: &AudioWindow) -> PathBuf {
    cache_root
        .join("refine")
        .join(media_id)
        .join(format!("{}{}", chunk_cache_stem(window), ASR_CACHE_SUFFIX))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedWindow {
    pub index: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAsr {
    pub model_id: String,
    pub language_hint: Option<String>,
    pub text: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToken {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAlignment {
    pub model_id: String,
    pub language: String,
    pub transcript_sha1: String,
    pub tokens: Vec<CachedToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub schema_version: usize,
    pub window: CachedWindow,
    pub asr: Option<CachedAsr>,
    pub alignment: Option<CachedAlignment>,
}

/// Load cached ASR transcript from disk.
pub fn load_cached_asr_result(
    path: &Path,
    model_id: &str,
    language_hint: Option<&str>,
) -> Option<(String, Option<String>)> {
    let entry = read_cache_entry(path)?;
    let asr = entry.asr?;
    if asr.model_id != model_id {
        return None;
    }
    if asr.language_hint.as_deref() != language_hint {
        return None;
    }
    Some((asr.text, asr.language))
}

/// Save ASR transcript to cache.
pub fn save_asr_result(
    path: &Path,
    window: &AudioWindow,
    model_id: &str,
    language_hint: Option<&str>,
    text: &str,
    language: Option<&str>,
) -> std::io::Result<()> {
    let mut entry = read_cache_entry(path).unwrap_or(CacheEntry {
        schema_version: CACHE_SCHEMA_VERSION,
        window: CachedWindow {
            index: window.index,
            start: window.start,
            end: window.end,
        },
        asr: None,
        alignment: None,
    });

    let previous_text = entry.asr.as_ref().map(|a| a.text.clone());
    entry.schema_version = CACHE_SCHEMA_VERSION;
    entry.window = CachedWindow {
        index: window.index,
        start: window.start,
        end: window.end,
    };
    entry.asr = Some(CachedAsr {
        model_id: model_id.to_string(),
        language_hint: language_hint.map(String::from),
        text: text.to_string(),
        language: language.map(String::from),
    });

    // Invalidate alignment if transcript text changed
    if let Some(prev) = previous_text {
        if prev != text {
            entry.alignment = None;
        }
    }

    write_cache_entry(path, &entry)
}

/// Load cached alignment tokens from disk.
pub fn load_cached_alignment_tokens(
    path: &Path,
    model_id: &str,
    language: &str,
    transcript: &str,
) -> Option<Vec<TimedToken>> {
    let entry = read_cache_entry(path)?;
    let alignment = entry.alignment?;
    if alignment.model_id != model_id {
        return None;
    }
    if alignment.language != language {
        return None;
    }
    if alignment.transcript_sha1 != text_sha1(transcript) {
        return None;
    }

    Some(
        alignment
            .tokens
            .into_iter()
            .map(|t| TimedToken::new(t.text, t.start, t.end))
            .collect(),
    )
}

/// Save alignment tokens to cache.
pub fn save_alignment_tokens(
    path: &Path,
    window: &AudioWindow,
    model_id: &str,
    language: &str,
    transcript: &str,
    tokens: &[TimedToken],
) -> std::io::Result<()> {
    let mut entry = read_cache_entry(path).unwrap_or(CacheEntry {
        schema_version: CACHE_SCHEMA_VERSION,
        window: CachedWindow {
            index: window.index,
            start: window.start,
            end: window.end,
        },
        asr: None,
        alignment: None,
    });

    entry.schema_version = CACHE_SCHEMA_VERSION;
    entry.window = CachedWindow {
        index: window.index,
        start: window.start,
        end: window.end,
    };
    entry.alignment = Some(CachedAlignment {
        model_id: model_id.to_string(),
        language: language.to_string(),
        transcript_sha1: text_sha1(transcript),
        tokens: tokens
            .iter()
            .map(|t| CachedToken {
                text: t.text.clone(),
                start: t.start,
                end: t.end,
            })
            .collect(),
    });

    write_cache_entry(path, &entry)
}

fn read_cache_entry(path: &Path) -> Option<CacheEntry> {
    if !path.exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    let entry: CacheEntry = serde_json::from_str(&content).ok()?;
    if entry.schema_version != CACHE_SCHEMA_VERSION {
        return None;
    }
    Some(entry)
}

fn write_cache_entry(path: &Path, entry: &CacheEntry) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(entry)?;
    fs::write(path, format!("{json}\n"))
}

/// Cleanup stale cache entries and enforce LRU retention bounds.
pub fn cleanup_cache(
    cache_root: &Path,
    keep_media_ids: &HashSet<String>,
    active_windows_by_media: Option<&HashMap<String, Vec<AudioWindow>>>,
    active_refine_windows_by_media: Option<&HashMap<String, Vec<AudioWindow>>>,
    max_media_entries: usize,
) {
    if !cache_root.exists() {
        return;
    }

    if let Some(active_map) = active_windows_by_media {
        for (media_id, windows) in active_map {
            let active_stems: HashSet<String> = windows.iter().map(chunk_cache_stem).collect();
            let asr_dir = cache_root.join("asr").join(media_id);
            remove_stale_children(&asr_dir, &active_stems, ASR_CACHE_SUFFIX);
        }
    }

    if let Some(active_refine) = active_refine_windows_by_media {
        for (media_id, windows) in active_refine {
            let active_stems: HashSet<String> = windows.iter().map(chunk_cache_stem).collect();
            let refine_dir = cache_root.join("refine").join(media_id);
            remove_stale_children(&refine_dir, &active_stems, ASR_CACHE_SUFFIX);
        }
    }

    if max_media_entries == 0 {
        return;
    }

    let all_media_ids = collect_media_ids(cache_root);
    let mut removable: Vec<String> = all_media_ids
        .into_iter()
        .filter(|id| !keep_media_ids.contains(id))
        .collect();

    removable.sort_by_key(|id| media_cache_mtime(cache_root, id));

    let excess = removable.len().saturating_sub(max_media_entries);
    for media_id in removable.into_iter().take(excess) {
        remove_media_cache(cache_root, &media_id);
    }
}

fn remove_stale_children(dir: &Path, active_stems: &HashSet<String>, suffix: &str) {
    if !dir.exists() {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(stem) = name.strip_suffix(suffix) {
                    if !active_stems.contains(stem) {
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }
    }
}

fn collect_media_ids(root: &Path) -> HashSet<String> {
    let mut ids = HashSet::new();
    for sub in &["audio", "asr", "refine"] {
        let dir = root.join(sub);
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        ids.insert(name.to_string());
                    }
                } else if path.is_file() {
                    ids.insert(media_cache_id(&path));
                }
            }
        }
    }
    ids
}

fn media_cache_mtime(root: &Path, media_id: &str) -> std::time::SystemTime {
    for sub in &["asr", "refine", "audio"] {
        let path = root.join(sub).join(media_id);
        if let Ok(meta) = fs::metadata(&path) {
            if let Ok(mtime) = meta.modified() {
                return mtime;
            }
        }
    }
    std::time::SystemTime::UNIX_EPOCH
}

fn remove_media_cache(root: &Path, media_id: &str) {
    let _ = fs::remove_dir_all(root.join("asr").join(media_id));
    let _ = fs::remove_dir_all(root.join("refine").join(media_id));
    let _ = fs::remove_file(root.join("audio").join(format!("{media_id}{AUDIO_CACHE_SUFFIX}")));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asr_cache_save_and_load() {
        let temp_dir = std::env::temp_dir().join(format!("test_asr_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        let path = temp_dir.join("test.json");

        let window = AudioWindow::new(1, 0.0, 5.0);
        save_asr_result(
            &path,
            &window,
            "qwen3-asr",
            Some("en"),
            "Hello world",
            Some("en"),
        )
        .unwrap();

        let loaded = load_cached_asr_result(&path, "qwen3-asr", Some("en"));
        assert_eq!(loaded, Some(("Hello world".to_string(), Some("en".to_string()))));

        // Wrong model or language returns None
        assert!(load_cached_asr_result(&path, "other-model", Some("en")).is_none());
        assert!(load_cached_asr_result(&path, "qwen3-asr", Some("ja")).is_none());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_alignment_cache_save_and_load() {
        let temp_dir = std::env::temp_dir().join(format!("test_align_cache_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        let path = temp_dir.join("test.json");

        let window = AudioWindow::new(1, 0.0, 5.0);
        let tokens = vec![
            TimedToken::new("Hello", 0.0, 0.5),
            TimedToken::new("world", 0.5, 1.0),
        ];

        save_alignment_tokens(
            &path,
            &window,
            "qwen3-aligner",
            "en",
            "Hello world",
            &tokens,
        )
        .unwrap();

        let loaded = load_cached_alignment_tokens(&path, "qwen3-aligner", "en", "Hello world");
        assert_eq!(loaded, Some(tokens));

        // Changed transcript invalidates cache
        assert!(load_cached_alignment_tokens(&path, "qwen3-aligner", "en", "Hello Earth").is_none());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
