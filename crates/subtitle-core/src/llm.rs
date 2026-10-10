//! Async OpenAI-compatible LLM client for optional subtitle proofreading
//! and translation.
//!
//! Disabled by default; only called when `--proofread` or `--translate` is requested
//! and an API key is configured.

use std::collections::HashMap;

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::SubtitleItem;

pub const PROOFREAD_SYSTEM_PROMPT: &str = "\
You are a subtitle proofreading engine.
Fix ASR errors and transcription artifacts.
Return only valid JSON in compact format: [{\"i\":<id>,\"t\":\"<corrected text>\"}, ...].
Only include items you actually corrected; omit unchanged ones.
Do not change meaning.";

use std::sync::Arc;
use futures_util::StreamExt;

pub type StreamCallback = Arc<dyn Fn(&str) + Send + Sync>;

pub const TRANSLATION_SYSTEM_PROMPT: &str = "\
You are a subtitle translation engine.
Return only valid JSON in compact format: [{\"i\":<id>,\"t\":\"<translation>\"}, ...].
Preserve ids exactly and translate each subtitle line concisely.";

/// LLM configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LLMConfig {
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_api_key")]
    pub api_key: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: f64,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub thinking_effort: Option<String>,
}

fn default_base_url() -> String { "https://api.openai.com/v1".into() }
fn default_api_key() -> String { String::new() }
fn default_timeout() -> f64 { 60.0 }
fn default_temperature() -> f32 { 0.0 }
fn default_batch_size() -> usize { 40 }
fn default_concurrency() -> usize { 2 }
fn default_max_retries() -> usize { 2 }

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            api_key: default_api_key(),
            model: None,
            timeout_secs: default_timeout(),
            temperature: default_temperature(),
            batch_size: default_batch_size(),
            concurrency: default_concurrency(),
            max_retries: default_max_retries(),
            thinking: false,
            thinking_effort: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LlmJsonItem {
    i: usize,
    t: String,
}

/// Asynchronous LLM client for subtitle proofreading and translation.
#[derive(Clone)]
pub struct SubtitleLlm {
    client: Client<OpenAIConfig>,
    pub config: LLMConfig,
}

impl SubtitleLlm {
    pub fn new(config: LLMConfig) -> Self {
        let mut openai_config = OpenAIConfig::new();
        if !config.api_key.is_empty() {
            openai_config = openai_config.with_api_key(&config.api_key);
        }
        if !config.base_url.is_empty() {
            openai_config = openai_config.with_api_base(&config.base_url);
        }
        let client = Client::with_config(openai_config);
        Self { client, config }
    }

    /// Asynchronously proofread a list of subtitles in batches.
    pub async fn proofread(
        &self,
        items: &[SubtitleItem],
        on_token: Option<StreamCallback>,
    ) -> Result<Vec<SubtitleItem>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let model = self
            .config
            .model
            .as_deref()
            .unwrap_or("gpt-4o-mini");

        let batch_size = if self.config.batch_size == 0 {
            items.len()
        } else {
            self.config.batch_size
        };

        let mut result = items.to_vec();
        for chunk in result.chunks_mut(batch_size) {
            let chunk_items: Vec<serde_json::Value> = chunk
                .iter()
                .map(|item| serde_json::json!({ "i": item.id, "t": item.text }))
                .collect();

            let payload = serde_json::json!({
                "requirements": [
                    "Fix ASR errors and transcription artifacts in the following subtitle lines.",
                    "Only return items that actually need correction; omit unchanged ones.",
                    "Do not merge, split, remove, or reorder ids.",
                    "Do not change meaning or rewrite for style."
                ],
                "items": chunk_items
            });

            let response_text = self
                .complete_json(PROOFREAD_SYSTEM_PROMPT, &payload.to_string(), model, on_token.as_ref())
                .await?;
            let corrections = parse_proofread_json(&response_text)?;

            for item in chunk.iter_mut() {
                if let Some(corr) = corrections.get(&item.id) {
                    item.proofread = Some(corr.clone());
                }
            }
        }

        Ok(result)
    }

    /// Asynchronously translate a list of subtitles in batches.
    pub async fn translate(
        &self,
        items: &[SubtitleItem],
        target_language: &str,
        on_token: Option<StreamCallback>,
    ) -> Result<Vec<SubtitleItem>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let model = self
            .config
            .model
            .as_deref()
            .unwrap_or("gpt-4o-mini");

        let batch_size = if self.config.batch_size == 0 {
            items.len()
        } else {
            self.config.batch_size
        };

        let mut result = items.to_vec();
        for chunk in result.chunks_mut(batch_size) {
            let expected_ids: Vec<usize> = chunk.iter().map(|item| item.id).collect();
            let chunk_items: Vec<serde_json::Value> = chunk
                .iter()
                .map(|item| {
                    let source = item.proofread.as_deref().unwrap_or(&item.text);
                    serde_json::json!({ "i": item.id, "t": source })
                })
                .collect();

            let payload = serde_json::json!({
                "target_language": target_language,
                "requirements": [
                    "Translate the following subtitle lines as one coherent context.",
                    "Return exactly one item for each id.",
                    "Do not merge, split, remove, or reorder ids.",
                    "Keep translations concise and suitable for subtitles."
                ],
                "items": chunk_items
            });

            let response_text = self
                .complete_json(TRANSLATION_SYSTEM_PROMPT, &payload.to_string(), model, on_token.as_ref())
                .await?;
            let translations = parse_translation_json(&response_text, &expected_ids)?;

            for item in chunk.iter_mut() {
                if let Some(trans) = translations.get(&item.id) {
                    item.translation = Some(trans.clone());
                }
            }
        }

        Ok(result)
    }

    async fn complete_json(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        model: &str,
        on_token: Option<&StreamCallback>,
    ) -> Result<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .temperature(self.config.temperature)
            .stream(on_token.is_some())
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()
                    .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_prompt)
                    .build()
                    .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?
                    .into(),
            ])
            .build()
            .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

        if let Some(cb) = on_token {
            let mut stream = self
                .client
                .chat()
                .create_stream(request)
                .await
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

            let mut text = String::new();
            while let Some(chunk_res) = stream.next().await {
                let chunk = chunk_res.map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
                for choice in chunk.choices {
                    if let Some(delta) = choice.delta.content {
                        cb(&delta);
                        text.push_str(&delta);
                    }
                }
            }
            Ok(text)
        } else {
            let response = self
                .client
                .chat()
                .create(request)
                .await
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;

            let text = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();

            Ok(text)
        }
    }
}

/// Extract clean JSON from possibly markdown-fenced text.
pub fn clean_json_markdown(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(stripped) = trimmed.strip_prefix("```json") {
        if let Some(end) = stripped.rfind("```") {
            return stripped[..end].trim();
        }
    } else if let Some(stripped) = trimmed.strip_prefix("```") {
        if let Some(end) = stripped.rfind("```") {
            return stripped[..end].trim();
        }
    }
    trimmed
}

/// Parse and validate proofread JSON output.
pub fn parse_proofread_json(raw: &str) -> Result<HashMap<usize, String>> {
    let clean = clean_json_markdown(raw);
    let items: Vec<LlmJsonItem> = serde_json::from_str(clean)
        .map_err(|e| Error::Llm(format!("Failed to parse LLM proofread JSON: {e}")))?;

    let mut map = HashMap::new();
    for item in items {
        map.insert(item.i, item.t.trim().to_string());
    }
    Ok(map)
}

/// Parse and validate translation JSON output, ensuring all expected IDs match.
pub fn parse_translation_json(raw: &str, expected_ids: &[usize]) -> Result<HashMap<usize, String>> {
    let clean = clean_json_markdown(raw);
    let items: Vec<LlmJsonItem> = serde_json::from_str(clean)
        .map_err(|e| Error::Llm(format!("Failed to parse LLM translation JSON: {e}")))?;

    let mut map = HashMap::new();
    for item in items {
        map.insert(item.i, item.t.trim().to_string());
    }

    for &id in expected_ids {
        if !map.contains_key(&id) {
            return Err(Error::Llm(format!("LLM translation missing expected cue id: {id}")));
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_proofread_json() {
        let raw = r#"```json
[
  {"i": 2, "t": "World"},
  {"i": 5, "t": "Test"}
]
```"#;
        let parsed = parse_proofread_json(raw).unwrap();
        assert_eq!(parsed.get(&2), Some(&"World".to_string()));
        assert_eq!(parsed.get(&5), Some(&"Test".to_string()));
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_parse_translation_json() {
        let raw = r#"[
  {"i": 1, "t": "Bonjour"},
  {"i": 2, "t": "Monde"}
]"#;
        let expected = vec![1, 2];
        let parsed = parse_translation_json(raw, &expected).unwrap();
        assert_eq!(parsed.get(&1), Some(&"Bonjour".to_string()));
        assert_eq!(parsed.get(&2), Some(&"Monde".to_string()));

        // Missing id returns error
        let expected_with_missing = vec![1, 2, 3];
        assert!(parse_translation_json(raw, &expected_with_missing).is_err());
    }
}
