//! Application configuration matching Python's `config.toml` schema.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cache::CacheConfig;
use crate::error::{Error, Result};
use crate::llm::LLMConfig;
use crate::text::SegmentConfig;

pub const DEFAULT_CONFIG_FILE: &str = "config.toml";
pub const APP_CONFIG_DIR_NAME: &str = "subtitle-gen";
pub const ENV_CONFIG_FILE: &str = "SUBTITLE_GEN_CONFIG";
pub const DEFAULT_ASR_MODEL: &str = "qwen3-asr-0.6b-q4_k";
pub const DEFAULT_ALIGNER_MODEL: &str = "qwen3-forced-aligner-0.6b-q4_k";
pub const DEFAULT_QUANT: &str = "q4_k";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_asr_model")]
    pub asr_model: String,
    #[serde(default = "default_aligner_model")]
    pub aligner_model: String,
    #[serde(default = "default_quant")]
    pub quant: String,
    #[serde(default)]
    pub n_threads: usize,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub language: Option<String>,
}

fn default_backend() -> String {
    "crispasr".into()
}
fn default_asr_model() -> String {
    DEFAULT_ASR_MODEL.into()
}
fn default_aligner_model() -> String {
    DEFAULT_ALIGNER_MODEL.into()
}
fn default_quant() -> String {
    DEFAULT_QUANT.into()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            asr_model: default_asr_model(),
            aligner_model: default_aligner_model(),
            quant: default_quant(),
            n_threads: 0,
            gpu_layers: None,
            language: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadConfig {
    #[serde(default = "default_vad_enabled")]
    pub enabled: bool,
    #[serde(default = "default_short_audio_threshold")]
    pub short_audio_threshold: f64,
    #[serde(default = "default_min_speech_duration")]
    pub min_speech_duration: f64,
    #[serde(default = "default_min_silence_duration")]
    pub min_silence_duration: f64,
    #[serde(default = "default_speech_padding")]
    pub speech_padding: f64,
    #[serde(default = "default_max_chunk_duration")]
    pub max_chunk_duration: f64,
    #[serde(default = "default_hard_max_chunk_duration")]
    pub hard_max_chunk_duration: f64,
    #[serde(default = "default_skip_silence_longer_than")]
    pub skip_silence_longer_than: f64,
}

fn default_vad_enabled() -> bool {
    true
}
fn default_short_audio_threshold() -> f64 {
    180.0
}
fn default_min_speech_duration() -> f64 {
    0.25
}
fn default_min_silence_duration() -> f64 {
    0.6
}
fn default_speech_padding() -> f64 {
    0.25
}
fn default_max_chunk_duration() -> f64 {
    120.0
}
fn default_hard_max_chunk_duration() -> f64 {
    240.0
}
fn default_skip_silence_longer_than() -> f64 {
    8.0
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            enabled: default_vad_enabled(),
            short_audio_threshold: default_short_audio_threshold(),
            min_speech_duration: default_min_speech_duration(),
            min_silence_duration: default_min_silence_duration(),
            speech_padding: default_speech_padding(),
            max_chunk_duration: default_max_chunk_duration(),
            hard_max_chunk_duration: default_hard_max_chunk_duration(),
            skip_silence_longer_than: default_skip_silence_longer_than(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_formats")]
    pub default_formats: Vec<String>,
    #[serde(default = "default_bilingual_separator")]
    pub bilingual_separator: String,
    #[serde(default = "default_strip_punctuation")]
    pub strip_punctuation: bool,
}

fn default_formats() -> Vec<String> {
    vec!["srt".into()]
}
fn default_bilingual_separator() -> String {
    "\n".into()
}
fn default_strip_punctuation() -> bool {
    true
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            default_formats: default_formats(),
            bilingual_separator: default_bilingual_separator(),
            strip_punctuation: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub vad: VadConfig,
    #[serde(default)]
    pub segment: SegmentConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub llm: LLMConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
}

fn default_cache_dir() -> String {
    ".subtitle-gen-cache".into()
}

/// Returns the platform-specific configuration directory for `subtitle-gen`, if resolvable.
///
/// Typical locations:
/// - Windows: `%APPDATA%\subtitle-gen`
/// - Linux: `$XDG_CONFIG_HOME/subtitle-gen` or `~/.config/subtitle-gen`
/// - macOS: `~/Library/Application Support/subtitle-gen`
pub fn system_config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join(APP_CONFIG_DIR_NAME))
}

/// Returns the platform-specific default configuration file path for `subtitle-gen`.
///
/// E.g. `<system_config_dir>/config.toml`
pub fn system_config_path() -> Option<PathBuf> {
    system_config_dir().map(|d| d.join(DEFAULT_CONFIG_FILE))
}

/// Resolves configuration file path according to the following priority:
/// 1. Explicit path passed as argument (e.g. `--config <path>`)
/// 2. `SUBTITLE_GEN_CONFIG` environment variable
/// 3. Current working directory (`./config.toml`)
/// 4. Platform-specific system configuration directory
pub fn resolve_config_path(path: Option<&Path>) -> Result<Option<PathBuf>> {
    let env_override = std::env::var_os(ENV_CONFIG_FILE).map(PathBuf::from);
    let local_file = Path::new(DEFAULT_CONFIG_FILE);
    let sys_file = system_config_path();

    resolve_config_path_internal(
        path,
        env_override.as_deref(),
        local_file,
        sys_file.as_deref(),
    )
}

fn resolve_config_path_internal(
    explicit: Option<&Path>,
    env_override: Option<&Path>,
    local_file: &Path,
    sys_file: Option<&Path>,
) -> Result<Option<PathBuf>> {
    if let Some(p) = explicit {
        if !p.exists() {
            return Err(Error::Config(format!(
                "Config file not found: {}",
                p.display()
            )));
        }
        return Ok(Some(p.to_path_buf()));
    }

    if let Some(p) = env_override {
        if !p.exists() {
            return Err(Error::Config(format!(
                "Config file specified by {ENV_CONFIG_FILE} not found: {}",
                p.display()
            )));
        }
        return Ok(Some(p.to_path_buf()));
    }

    if local_file.is_file() {
        return Ok(Some(local_file.to_path_buf()));
    }

    if let Some(p) = sys_file {
        if p.is_file() {
            return Ok(Some(p.to_path_buf()));
        }
    }

    Ok(None)
}

/// Load configuration from a TOML file path or auto-detect from:
/// 1. `SUBTITLE_GEN_CONFIG` environment variable
/// 2. `./config.toml` in the current working directory
/// 3. Platform-specific system configuration directory (`config.toml`)
///
/// If no configuration file is found, returns [`AppConfig::default()`].
pub fn load_config(path: Option<&Path>) -> Result<AppConfig> {
    match resolve_config_path(path)? {
        Some(resolved_path) => {
            tracing::debug!("Loading configuration from: {}", resolved_path.display());
            let content = fs::read_to_string(&resolved_path)?;
            load_config_from_str(&content)
        }
        None => {
            tracing::debug!("No configuration file found; using default settings");
            Ok(AppConfig::default())
        }
    }
}

/// Parse configuration from a TOML string.
pub fn load_config_from_str(s: &str) -> Result<AppConfig> {
    toml::from_str(s).map_err(|e| Error::Config(format!("TOML parse error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_parses() {
        let toml_str = r#"
[model]
asr_model = "my-asr-model"
quant = "q8_0"

[segment]
max_duration = 5.5

[vad]
enabled = false
"#;
        let config = load_config_from_str(toml_str).unwrap();
        assert_eq!(config.model.asr_model, "my-asr-model");
        assert_eq!(config.model.quant, "q8_0");
        assert_eq!(config.segment.max_duration, 5.5);
        assert!(!config.vad.enabled);
        assert!(!config.cache.enabled);
        assert!(config.output.strip_punctuation);
    }

    #[test]
    fn test_system_config_path_resolves() {
        if let Some(path) = system_config_path() {
            let path_str = path.to_string_lossy();
            assert!(
                path_str.contains(APP_CONFIG_DIR_NAME) && path_str.contains(DEFAULT_CONFIG_FILE),
                "Unexpected system config path: {path_str}"
            );
        }
    }

    #[test]
    fn test_resolve_config_path_explicit_missing() {
        let missing = Path::new("non_existent_config_file_12345.toml");
        assert!(resolve_config_path(Some(missing)).is_err());
    }

    #[test]
    fn test_resolve_config_path_priority_internal() {
        let temp_dir = std::env::temp_dir().join("subtitle_gen_test_config");
        let _ = fs::create_dir_all(&temp_dir);
        let explicit = temp_dir.join("explicit.toml");
        let env_p = temp_dir.join("env.toml");
        let local = temp_dir.join("local.toml");
        let sys = temp_dir.join("sys.toml");

        fs::write(&explicit, "[model]\nquant = \"q8_0\"\n").unwrap();
        fs::write(&env_p, "[model]\nquant = \"q5_k\"\n").unwrap();
        fs::write(&local, "[model]\nquant = \"q4_k\"\n").unwrap();
        fs::write(&sys, "[model]\nquant = \"q2_k\"\n").unwrap();

        // 1. Explicit wins over everything
        let res = resolve_config_path_internal(Some(&explicit), Some(&env_p), &local, Some(&sys))
            .unwrap();
        assert_eq!(res, Some(explicit.clone()));

        // 2. Env wins over local and sys
        let res = resolve_config_path_internal(None, Some(&env_p), &local, Some(&sys)).unwrap();
        assert_eq!(res, Some(env_p.clone()));

        // 3. Local wins over sys
        let res = resolve_config_path_internal(None, None, &local, Some(&sys)).unwrap();
        assert_eq!(res, Some(local.clone()));

        // 4. Sys is selected when local missing
        let missing_local = temp_dir.join("non_existent_local.toml");
        let res = resolve_config_path_internal(None, None, &missing_local, Some(&sys)).unwrap();
        assert_eq!(res, Some(sys.clone()));

        // 5. Fallback to None when nothing exists
        let missing_sys = temp_dir.join("non_existent_sys.toml");
        let res =
            resolve_config_path_internal(None, None, &missing_local, Some(&missing_sys)).unwrap();
        assert_eq!(res, None);

        // Missing explicit returns Err
        let missing_explicit = temp_dir.join("non_existent_explicit.toml");
        assert!(
            resolve_config_path_internal(Some(&missing_explicit), None, &local, Some(&sys))
                .is_err()
        );

        // Missing env returns Err
        let missing_env = temp_dir.join("non_existent_env.toml");
        assert!(
            resolve_config_path_internal(None, Some(&missing_env), &local, Some(&sys)).is_err()
        );

        // Cleanup
        let _ = fs::remove_file(explicit);
        let _ = fs::remove_file(env_p);
        let _ = fs::remove_file(local);
        let _ = fs::remove_file(sys);
        let _ = fs::remove_dir(temp_dir);
    }
}
