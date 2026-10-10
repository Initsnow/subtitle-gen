//! Application configuration matching Python's `config.toml` schema.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cache::CacheConfig;
use crate::error::{Error, Result};
use crate::llm::LLMConfig;
use crate::text::SegmentConfig;

pub const DEFAULT_CONFIG_FILE: &str = "config.toml";
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

fn default_backend() -> String { "crispasr".into() }
fn default_asr_model() -> String { DEFAULT_ASR_MODEL.into() }
fn default_aligner_model() -> String { DEFAULT_ALIGNER_MODEL.into() }
fn default_quant() -> String { DEFAULT_QUANT.into() }

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

fn default_vad_enabled() -> bool { true }
fn default_short_audio_threshold() -> f64 { 180.0 }
fn default_min_speech_duration() -> f64 { 0.25 }
fn default_min_silence_duration() -> f64 { 0.6 }
fn default_speech_padding() -> f64 { 0.25 }
fn default_max_chunk_duration() -> f64 { 120.0 }
fn default_hard_max_chunk_duration() -> f64 { 240.0 }
fn default_skip_silence_longer_than() -> f64 { 8.0 }

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
    #[serde(default)]
    pub strip_punctuation: bool,
}

fn default_formats() -> Vec<String> { vec!["srt".into()] }
fn default_bilingual_separator() -> String { "\n".into() }

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            default_formats: default_formats(),
            bilingual_separator: default_bilingual_separator(),
            strip_punctuation: false,
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

fn default_cache_dir() -> String { ".subtitle-gen-cache".into() }

/// Load configuration from a TOML file path or default `config.toml` if found.
pub fn load_config(path: Option<&Path>) -> Result<AppConfig> {
    let resolved_path = match path {
        Some(p) => {
            if !p.exists() {
                return Err(Error::Config(format!("Config file not found: {}", p.display())));
            }
            Some(p.to_path_buf())
        }
        None => {
            let default_path = PathBuf::from(DEFAULT_CONFIG_FILE);
            if default_path.exists() {
                Some(default_path)
            } else {
                None
            }
        }
    };

    match resolved_path {
        Some(p) => {
            let content = fs::read_to_string(&p)?;
            load_config_from_str(&content)
        }
        None => Ok(AppConfig::default()),
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
    }
}
