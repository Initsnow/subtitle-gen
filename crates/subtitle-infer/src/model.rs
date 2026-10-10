//! Model spec resolution and language-hint normalization.
//!
//! Mirrors the Python implementation in `src/subtitle_gen/crispasr_models.py`
//! so both binaries accept the same specs:
//!
//! * a path to a local `.gguf` file
//! * a registry alias (`qwen3`, `qwen3-1.7b`, `qwen3-forced-aligner`, …)
//! * a known GGUF filename (`qwen3-asr-0.6b-q4_k.gguf`)
//! * a Hugging Face repo id or URL (`cstr/qwen3-asr-0.6b-GGUF`, optionally
//!   `repo:file.gguf`)
//!
//! Resolved files are looked up in the CrispASR cache (`~/.cache/crispasr`,
//! overridable via `CRISPASR_CACHE`). In native mode a cache miss triggers a
//! download through the official binding's `crispasr::cache_ensure_file`
//! (same cache directory the CLI uses); in stub mode the resolver reports a
//! precise error with the expected cache path instead.

use std::path::{Path, PathBuf};

use crate::error::{InferError, Result};

/// Default GGUF specs, matching the CrispASR registry defaults.
pub const DEFAULT_ASR_GGUF: &str = "qwen3-asr-0.6b-q4_k";
pub const DEFAULT_ALIGNER_GGUF: &str = "qwen3-forced-aligner-0.6b-q4_k";
pub const DEFAULT_VAD_GGUF: &str = "firered-vad";

/// `quant` suffix used when a HF repo id is given without a file.
pub const DEFAULT_QUANT: &str = "q4_k";

/// Friendly spec → (default GGUF filename in the CrispASR cache, Hugging
/// Face repo used for download).
///
/// The Python backend additionally consults the runtime's registry lookup to
/// resolve download URLs; the filenames/repos below pin the same registry
/// entries so both resolve to the same cache files.
const ALIASES: &[(&str, &str, &str)] = &[
    (
        "qwen3",
        "qwen3-asr-0.6b-q4_k.gguf",
        "cstr/qwen3-asr-0.6b-GGUF",
    ),
    (
        "qwen3-asr",
        "qwen3-asr-0.6b-q4_k.gguf",
        "cstr/qwen3-asr-0.6b-GGUF",
    ),
    (
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b-q4_k.gguf",
        "cstr/qwen3-asr-0.6b-GGUF",
    ),
    (
        "qwen3-asr-0.6b-q4_k",
        "qwen3-asr-0.6b-q4_k.gguf",
        "cstr/qwen3-asr-0.6b-GGUF",
    ),
    (
        "qwen3-asr-0.6b-q8_0",
        "qwen3-asr-0.6b-q8_0.gguf",
        "cstr/qwen3-asr-0.6b-GGUF",
    ),
    (
        "qwen3-1.7b",
        "qwen3-asr-1.7b-q4_k.gguf",
        "cstr/qwen3-asr-1.7b-GGUF",
    ),
    (
        "qwen3-asr-1.7b",
        "qwen3-asr-1.7b-q4_k.gguf",
        "cstr/qwen3-asr-1.7b-GGUF",
    ),
    (
        "qwen3-forced-aligner",
        "qwen3-forced-aligner-0.6b-q4_k.gguf",
        "cstr/qwen3-forced-aligner-0.6b-GGUF",
    ),
    (
        "qwen3-fa",
        "qwen3-forced-aligner-0.6b-q4_k.gguf",
        "cstr/qwen3-forced-aligner-0.6b-GGUF",
    ),
    (
        "qwen3-forced-aligner-0.6b",
        "qwen3-forced-aligner-0.6b-q4_k.gguf",
        "cstr/qwen3-forced-aligner-0.6b-GGUF",
    ),
    (
        "qwen3-forced-aligner-0.6b-q4_k",
        "qwen3-forced-aligner-0.6b-q4_k.gguf",
        "cstr/qwen3-forced-aligner-0.6b-GGUF",
    ),
    ("firered-vad", "firered-vad.gguf", "cstr/firered-vad-GGUF"),
    (
        "cstr/firered-vad-gguf",
        "firered-vad.gguf",
        "cstr/firered-vad-GGUF",
    ),
];

/// Full language names → ISO 639-1, for the runtime `language` hint.
const ISO_LANGUAGES: &[(&str, &str)] = &[
    ("english", "en"),
    ("chinese", "zh"),
    ("mandarin", "zh"),
    ("cantonese", "yue"),
    ("japanese", "ja"),
    ("korean", "ko"),
    ("french", "fr"),
    ("german", "de"),
    ("italian", "it"),
    ("portuguese", "pt"),
    ("russian", "ru"),
    ("spanish", "es"),
    ("arabic", "ar"),
    ("dutch", "nl"),
    ("polish", "pl"),
    ("turkish", "tr"),
    ("vietnamese", "vi"),
    ("thai", "th"),
];

/// Normalize a language hint to ISO 639-1. Accepts two-letter codes and the
/// full names above; anything else is passed through unchanged (the runtime
/// performs final validation).
pub fn iso_language(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    let lower = value.to_lowercase();
    if lower.len() == 2 && lower.chars().all(|c| c.is_ascii_alphabetic()) {
        return Some(lower);
    }
    Some(
        ISO_LANGUAGES
            .iter()
            .find(|(name, _)| *name == lower)
            .map(|(_, code)| (*code).to_string())
            .unwrap_or_else(|| lower),
    )
}

/// CrispASR model cache directory.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CRISPASR_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("crispasr")
}

/// Normalize a model spec to a canonical lookup name, applying the alias
/// table. Pure helper shared by the tests and [`resolve_model_path`].
pub fn canonical_name(spec: &str) -> String {
    let mut name = spec.trim().to_string();

    // Strip a full Hugging Face URL down to "<owner>/<repo>" (and honour
    // "repo:file.gguf" after it).
    if name.starts_with("http://") || name.starts_with("https://") {
        let rest: String = match name.find("huggingface.co/") {
            Some(idx) => name[idx + "huggingface.co/".len()..].to_string(),
            None => name.clone(),
        };
        let parts: Vec<&str> = rest
            .split('/')
            .filter(|p: &&str| !p.is_empty() && *p != "resolve" && *p != "main")
            .collect();
        // Keep "<owner>/<repo>"; an explicit file inside the URL is handled
        // by the HF downloader in a later phase (same rule as Python).
        name = if parts.len() >= 2 {
            parts[..2].join("/")
        } else {
            parts.join("/")
        };
    }

    // "owner/repo:file.gguf" → keep the file part for exact cache lookup.
    // (The split itself is done by the caller via `hf_file_in_spec`.)

    let lowered = name.to_lowercase();
    match ALIASES.iter().find(|(alias, _, _)| *alias == lowered) {
        Some((_, filename, _)) => filename.to_string(),
        None => name,
    }
}

/// Extract an explicit `<file>.gguf` from a `repo:file.gguf` spec.
pub fn hf_file_in_spec(spec: &str) -> Option<String> {
    let spec = spec.trim();
    let colon = spec.rfind(':')?;
    // Windows drive letters ("C:\...") are handled earlier: local paths are
    // returned before this helper is consulted.
    let file = &spec[colon + 1..];
    if file.ends_with(".gguf") && !file.is_empty() {
        Some(file.to_string())
    } else {
        None
    }
}

/// Strip a trailing "-GGUF" from a HF repo's last segment (case-insensitive),
/// mirroring the Python default-filename construction.
pub fn default_gguf_filename(repo: &str, quant: &str) -> String {
    let last = repo.trim_matches('/').rsplit('/').next().unwrap_or(repo);
    let stem = match last.rsplit_once('-') {
        Some((stem, suffix)) if suffix.eq_ignore_ascii_case("gguf") => stem,
        _ => last,
    };
    format!("{stem}-{quant}.gguf")
}

/// Resolve a model spec to an existing local `.gguf` path.
///
/// Resolution order: existing local path → (filename, HF repo) derived from
/// the spec → cache lookup → native-mode download via the official binding's
/// `cache_ensure_file`. Missing files yield [`InferError::ModelNotFound`]
/// with the exact expected path.
pub fn resolve_model_path(spec: &str, quant: &str) -> Result<PathBuf> {
    let raw = spec.trim();
    if raw.is_empty() {
        return Err(InferError::Spec("empty model spec".into()));
    }

    // 1. Local path (checked before `:` handling so Windows drive letters
    //    such as `C:\models\x.gguf` work).
    let candidate = Path::new(raw);
    if candidate.is_file() {
        return Ok(candidate.to_path_buf());
    }

    // 2. Derive (cache filename, HF repo) from the spec.
    let (filename, repo) = spec_to_filename_and_repo(raw, quant).ok_or_else(|| {
        InferError::Spec(format!(
            "unknown model spec {raw:?}: use a local .gguf path, a registry alias \
             (qwen3, qwen3-forced-aligner, ...), or a Hugging Face repo id such as \
             cstr/qwen3-asr-0.6b-GGUF"
        ))
    })?;

    resolve_in_cache(&filename, &repo, raw)
}

/// Map a non-local spec to `(cache filename, HF repo)` for cache lookup and
/// (native-mode) download.
fn spec_to_filename_and_repo(spec: &str, quant: &str) -> Option<(String, String)> {
    // Collapse a full HF URL down to "<owner>/<repo>" first, mirroring the
    // Python implementation (explicit file inside the URL is not honoured).
    let mut spec = spec.trim().to_string();
    if spec.starts_with("http://") || spec.starts_with("https://") {
        let rest = match spec.find("huggingface.co/") {
            Some(idx) => spec[idx + "huggingface.co/".len()..].to_string(),
            None => spec.clone(),
        };
        let parts: Vec<&str> = rest
            .split('/')
            .filter(|p: &&str| !p.is_empty() && *p != "resolve" && *p != "main")
            .collect();
        spec = if parts.len() >= 2 {
            parts[..2].join("/")
        } else {
            parts.join("/")
        };
    }

    // Explicit "repo:file.gguf".
    if let Some(file) = hf_file_in_spec(&spec) {
        let repo = spec[..spec.rfind(':').unwrap()].trim().to_string();
        return Some((file, repo));
    }

    // Registry alias.
    let lowered = spec.to_lowercase();
    if let Some((_, filename, repo)) = ALIASES.iter().find(|(alias, _, _)| *alias == lowered) {
        return Some(((*filename).to_string(), (*repo).to_string()));
    }

    // Bare known filename — no repo known, cache lookup only.
    if spec.ends_with(".gguf") {
        return Some((spec.to_string(), String::new()));
    }

    // Hugging Face repo id (or an already-collapsed HF URL).
    let repo = spec.trim_start_matches("https://huggingface.co/");
    if repo.contains('/') {
        return Some((default_gguf_filename(repo, quant), repo.to_string()));
    }

    None
}

/// Look the filename up in the cache; in native mode, download on miss.
fn resolve_in_cache(filename: &str, repo: &str, spec: &str) -> Result<PathBuf> {
    let path = cache_dir().join(filename);
    if path.is_file() {
        return Ok(path);
    }

    #[cfg(feature = "native")]
    if !repo.is_empty() {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
        tracing::info!(filename, %url, "model not in cache; downloading");
        match crispasr::cache_ensure_file(filename, &url, false, None) {
            Ok(Some(cached)) if !cached.is_empty() => return Ok(PathBuf::from(cached)),
            Ok(_) => {}
            Err(err) => tracing::warn!(filename, %err, "model download failed"),
        }
    }

    Err(InferError::ModelNotFound {
        path: path.display().to_string(),
        hint: if repo.is_empty() {
            format!(
                "spec {spec:?} maps to a bare filename with no known repo; place \
                 the weights at the path above or pass an explicit local .gguf path"
            )
        } else {
            format!(
                "download the weights for spec {spec:?} into the CrispASR cache \
                 (override with CRISPASR_CACHE), or pass an explicit local .gguf path"
            )
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_resolve_to_registry_filenames() {
        assert_eq!(canonical_name("qwen3"), "qwen3-asr-0.6b-q4_k.gguf");
        assert_eq!(canonical_name("QWEN3-ASR-0.6B"), "qwen3-asr-0.6b-q4_k.gguf");
        assert_eq!(
            canonical_name("qwen3-fa"),
            "qwen3-forced-aligner-0.6b-q4_k.gguf"
        );
        assert_eq!(canonical_name("cstr/firered-vad-GGUF"), "firered-vad.gguf");
    }

    #[test]
    fn hf_repo_maps_to_default_quant_filename() {
        assert_eq!(
            default_gguf_filename("cstr/qwen3-asr-0.6b-GGUF", "q4_k"),
            "qwen3-asr-0.6b-q4_k.gguf"
        );
        assert_eq!(
            default_gguf_filename("cstr/qwen3-asr-0.6b-GGUF", "q8_0"),
            "qwen3-asr-0.6b-q8_0.gguf"
        );
        // Repo without the -GGUF suffix keeps its name.
        assert_eq!(default_gguf_filename("acme/foo", "q4_k"), "foo-q4_k.gguf");
    }

    #[test]
    fn hf_urls_collapse_to_repo_ids() {
        assert_eq!(
            canonical_name("https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/x.gguf"),
            "cstr/qwen3-asr-0.6b-GGUF"
        );
    }

    #[test]
    fn explicit_file_spec_is_extracted() {
        assert_eq!(
            hf_file_in_spec("cstr/qwen3-asr-0.6b-GGUF:qwen3-asr-0.6b-q8_0.gguf").as_deref(),
            Some("qwen3-asr-0.6b-q8_0.gguf")
        );
        assert_eq!(hf_file_in_spec("cstr/qwen3-asr-0.6b-GGUF"), None);
    }

    #[test]
    fn local_path_wins_over_cache() {
        let dir = std::env::temp_dir();
        let file = dir.join(format!("subtitle-infer-test-{}.gguf", std::process::id()));
        std::fs::write(&file, b"fake").unwrap();
        let resolved = resolve_model_path(file.to_str().unwrap(), "q4_k").unwrap();
        assert_eq!(resolved, file);
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn missing_file_reports_expected_cache_path() {
        let err = resolve_model_path("cstr/nonexistent-model-GGUF", "q4_k").unwrap_err();
        match err {
            InferError::ModelNotFound { path, hint } => {
                assert!(path.ends_with("nonexistent-model-q4_k.gguf"));
                assert!(hint.contains("CRISPASR_CACHE"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn unknown_spec_is_rejected() {
        assert!(matches!(
            resolve_model_path("not a model", "q4_k"),
            Err(InferError::Spec(_))
        ));
        assert!(matches!(
            resolve_model_path("   ", "q4_k"),
            Err(InferError::Spec(_))
        ));
    }

    #[test]
    fn language_hints_normalize_to_iso_codes() {
        assert_eq!(iso_language(Some("ja")).as_deref(), Some("ja"));
        assert_eq!(iso_language(Some("Japanese")).as_deref(), Some("ja"));
        assert_eq!(iso_language(Some("Mandarin")).as_deref(), Some("zh"));
        assert_eq!(iso_language(Some("Cantonese")).as_deref(), Some("yue"));
        assert_eq!(iso_language(Some("fr")).as_deref(), Some("fr"));
        assert_eq!(iso_language(None), None);
        assert_eq!(iso_language(Some("  ")), None);
    }
}
