//! Subtitle format parsing and rendering: SRT, WebVTT, LRC, JSON, and TXT.
//!
//! Ported from the Python implementation (`subtitle_gen/formats.py`). Output
//! is byte-compatible with the Python version, including timestamp rounding,
//! key ordering in JSON, and LRC metadata handling.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

use crate::error::Result;
use crate::types::{round_half_even, SubtitleItem};

/// Punctuation stripped from line ends when `strip_punctuation` is set.
pub const STRIP_PUNCTUATION_CHARS: &[char] =
    &[',', '.', ';', ':', '!', '?', '，', '。', '；', '：', '！', '？', '、'];

/// Which text variant of a [`SubtitleItem`] a renderer emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubtitleMode {
    #[default]
    Original,
    Translation,
    Bilingual,
    Proofread,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct FormatError(pub String);

/// Untimed LRC content: metadata tags plus plain text lines.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LrcData {
    pub metadata: Vec<String>,
    pub lines: Vec<String>,
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

fn split_timestamp(seconds: f64) -> (i64, i64, i64, i64) {
    let total_millis = round_half_even(seconds.max(0.0) * 1000.0) as i64;
    let millis = total_millis % 1000;
    let total_seconds = total_millis / 1000;
    let secs = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    (hours, minutes, secs, millis)
}

/// `00:01:23,456`
pub fn format_srt_timestamp(seconds: f64) -> String {
    let (h, m, s, ms) = split_timestamp(seconds);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// `00:01:23.456`
pub fn format_vtt_timestamp(seconds: f64) -> String {
    let (h, m, s, ms) = split_timestamp(seconds);
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// `[mm:ss.xx]` with explicit carry so `[mm:60.xx]` never appears.
pub fn format_lrc_timestamp(seconds: f64) -> String {
    let total_centis = (seconds.max(0.0) * 100.0).round() as i64;
    let (total_seconds, centis) = (total_centis / 100, total_centis % 100);
    let (minutes, secs) = (total_seconds / 60, total_seconds % 60);
    format!("[{minutes:02}:{secs:02}.{centis:02}]")
}

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

fn strip_punctuation(text: &str) -> String {
    text.split_terminator('\n')
        .map(|line| line.trim_end_matches(STRIP_PUNCTUATION_CHARS))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolve the text variant of `item` selected by `mode`.
pub fn render_subtitle_text(item: &SubtitleItem, mode: SubtitleMode, strip: bool) -> String {
    let text = match mode {
        SubtitleMode::Original => item.text.clone(),
        SubtitleMode::Proofread => item.proofread.clone().unwrap_or_else(|| item.text.clone()),
        SubtitleMode::Translation => item.translation.clone().unwrap_or_default(),
        SubtitleMode::Bilingual => {
            let base = item.proofread.clone().unwrap_or_else(|| item.text.clone());
            match &item.translation {
                Some(translation) => format!("{base}\n{translation}"),
                None => base,
            }
        }
    };
    if strip {
        strip_punctuation(&text)
    } else {
        text
    }
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

pub fn render_srt(items: &[SubtitleItem], mode: SubtitleMode, strip: bool) -> String {
    let mut blocks: Vec<String> = Vec::new();
    for (offset, item) in items.iter().enumerate() {
        let text = render_subtitle_text(item, mode, strip);
        if text.is_empty() {
            continue;
        }
        blocks.push(format!(
            "{}\n{} --> {}\n{}",
            offset + 1,
            format_srt_timestamp(item.start),
            format_srt_timestamp(item.end),
            text
        ));
    }
    let suffix = if blocks.is_empty() { "" } else { "\n" };
    format!("{}{}", blocks.join("\n\n"), suffix)
}

pub fn render_vtt(items: &[SubtitleItem], mode: SubtitleMode, strip: bool) -> String {
    let mut blocks: Vec<String> = vec!["WEBVTT".to_string(), String::new()];
    for item in items {
        let text = render_subtitle_text(item, mode, strip);
        if text.is_empty() {
            continue;
        }
        blocks.push(format!(
            "{} --> {}",
            format_vtt_timestamp(item.start),
            format_vtt_timestamp(item.end)
        ));
        blocks.push(text);
        blocks.push(String::new());
    }
    blocks.join("\n")
}

pub fn render_txt(items: &[SubtitleItem], mode: SubtitleMode, strip: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    for item in items {
        let text = render_subtitle_text(item, mode, strip);
        if text.is_empty() {
            continue;
        }
        // One line per cue: collapse cue-internal line breaks into spaces.
        let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() {
            lines.push(line);
        }
    }
    let suffix = if lines.is_empty() { "" } else { "\n" };
    format!("{}{}", lines.join("\n"), suffix)
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

pub fn render_json(items: &[SubtitleItem], mode: SubtitleMode, strip: bool) -> String {
    let data: Vec<Value> = match mode {
        SubtitleMode::Original => items.iter().map(|i| Value::Object(i.to_json_map())).collect(),
        SubtitleMode::Proofread => items
            .iter()
            .map(|i| {
                let text = i.proofread.clone().unwrap_or_else(|| i.text.clone());
                let text = if strip { strip_punctuation(&text) } else { text };
                serde_json::json!({
                    "id": i.id,
                    "start": round3(i.start),
                    "end": round3(i.end),
                    "text": text,
                })
            })
            .collect(),
        SubtitleMode::Translation => items
            .iter()
            .map(|i| {
                let text = i.translation.clone().unwrap_or_default();
                let text = if strip { strip_punctuation(&text) } else { text };
                serde_json::json!({
                    "id": i.id,
                    "start": round3(i.start),
                    "end": round3(i.end),
                    "text": text,
                })
            })
            .collect(),
        SubtitleMode::Bilingual => items
            .iter()
            .map(|i| {
                let mut map = i.to_json_map();
                if strip {
                    for key in ["text", "translation", "proofread"] {
                        if let Some(Value::String(text)) = map.get(key) {
                            let stripped = strip_punctuation(text);
                            map.insert(key.into(), Value::String(stripped));
                        }
                    }
                }
                Value::Object(map)
            })
            .collect(),
    };
    format!("{}\n", serde_json::to_string_pretty(&data).unwrap())
}

pub fn render_lrc(
    items: &[SubtitleItem],
    metadata: Option<&[String]>,
    strip: bool,
) -> String {
    let mut lines: Vec<String> = metadata.map(|m| m.to_vec()).unwrap_or_default();
    for item in items {
        let text = if strip { strip_punctuation(&item.text) } else { item.text.clone() };
        lines.push(format!("{}{}", format_lrc_timestamp(item.start), text));
    }
    format!("{}\n", lines.join("\n"))
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Write text with the same newline behaviour as the Python implementation's
/// `Path.write_text`: `\n` is translated to `\r\n` on Windows, kept as-is
/// elsewhere. Keeps outputs byte-identical across the two implementations.
fn write_text(path: &Path, content: &str) -> std::io::Result<()> {
    #[cfg(windows)]
    let content = content.replace('\n', "\r\n");
    fs::write(path, content)
}

/// Write `items` to `path`, dispatching on the file extension.
pub fn write_subtitles(
    path: impl AsRef<Path>,
    items: &[SubtitleItem],
    mode: SubtitleMode,
    strip: bool,
) -> Result<PathBuf> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let content = match extension.as_str() {
        "srt" => render_srt(items, mode, strip),
        "vtt" => render_vtt(items, mode, strip),
        "json" => render_json(items, mode, strip),
        "txt" => render_txt(items, mode, strip),
        other => {
            return Err(FormatError(format!("Unsupported subtitle format: {other}")).into());
        }
    };
    ensure_parent(path)?;
    write_text(path, &content)?;
    Ok(path.to_path_buf())
}

/// Write the full output set (`*.original.*`, plus translation/bilingual/
/// proofread variants when requested) for each requested format.
#[allow(clippy::too_many_arguments)]
pub fn write_output_set(
    out_dir: impl AsRef<Path>,
    stem: &str,
    items: &[SubtitleItem],
    formats: &[SubtitleFormat],
    include_translation: bool,
    include_bilingual: bool,
    include_proofread: bool,
    strip: bool,
) -> Result<Vec<PathBuf>> {
    let out_dir = out_dir.as_ref();
    let mut written: Vec<PathBuf> = Vec::new();
    for format in formats {
        let ext = format.extension();
        written.push(write_subtitles(
            out_dir.join(format!("{stem}.original.{ext}")),
            items,
            SubtitleMode::Original,
            strip,
        )?);
        if include_proofread {
            written.push(write_subtitles(
                out_dir.join(format!("{stem}.proofread.{ext}")),
                items,
                SubtitleMode::Proofread,
                strip,
            )?);
        }
        if include_translation {
            written.push(write_subtitles(
                out_dir.join(format!("{stem}.translation.{ext}")),
                items,
                SubtitleMode::Translation,
                strip,
            )?);
        }
        if include_bilingual {
            written.push(write_subtitles(
                out_dir.join(format!("{stem}.bilingual.{ext}")),
                items,
                SubtitleMode::Bilingual,
                strip,
            )?);
        }
    }
    Ok(written)
}

pub fn write_lrc(
    path: impl AsRef<Path>,
    items: &[SubtitleItem],
    metadata: Option<&[String]>,
    strip: bool,
) -> Result<PathBuf> {
    let path = path.as_ref();
    ensure_parent(path)?;
    write_text(path, &render_lrc(items, metadata, strip))?;
    Ok(path.to_path_buf())
}

/// Output file formats understood by [`write_output_set`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    Vtt,
    Json,
    Txt,
}

impl SubtitleFormat {
    pub fn extension(self) -> &'static str {
        match self {
            SubtitleFormat::Srt => "srt",
            SubtitleFormat::Vtt => "vtt",
            SubtitleFormat::Json => "json",
            SubtitleFormat::Txt => "txt",
        }
    }

    /// Parse a CLI/config format name (case-insensitive, leading dot tolerated).
    pub fn parse(name: &str) -> Result<Self> {
        let normalized = name.trim_start_matches('.').to_ascii_lowercase();
        match normalized.as_str() {
            "srt" => Ok(SubtitleFormat::Srt),
            "vtt" => Ok(SubtitleFormat::Vtt),
            "json" => Ok(SubtitleFormat::Json),
            "txt" => Ok(SubtitleFormat::Txt),
            other => Err(FormatError(format!("Unsupported subtitle format: {other}")).into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

fn normalize_newlines(content: &str) -> String {
    content
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

struct SrtRegexes {
    timing: Regex,
    block: Regex,
    timestamp: Regex,
}

impl SrtRegexes {
    fn new() -> Self {
        Self {
            timing: Regex::new(r"\d\d:\d\d:\d\d[,.]\d\d\d\s+-->\s+\d\d:\d\d:\d\d[,.]\d\d\d")
                .unwrap(),
            block: Regex::new(r"\n\n+").unwrap(),
            timestamp: Regex::new(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})").unwrap(),
        }
    }
}

fn parse_srt_timestamp(regexes: &SrtRegexes, ts: &str) -> std::result::Result<f64, FormatError> {
    let caps = regexes.timestamp.captures(ts).ok_or_else(|| {
        FormatError(format!("Invalid SRT timestamp: {ts}"))
    })?;
    let hours: f64 = caps[1].parse().unwrap();
    let minutes: f64 = caps[2].parse().unwrap();
    let secs: f64 = caps[3].parse().unwrap();
    let millis: f64 = caps[4].parse().unwrap();
    Ok(hours * 3600.0 + minutes * 60.0 + secs + millis / 1000.0)
}

fn parse_srt_timing_line(regexes: &SrtRegexes, line: &str) -> std::result::Result<(f64, f64), FormatError> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(FormatError(format!("Invalid SRT timing line: {line}")));
    }
    let start = parse_srt_timestamp(regexes, parts[0].trim())?;
    let end = parse_srt_timestamp(regexes, parts[1].trim())?;
    Ok((start, end))
}

/// Parse SRT content (already read from disk).
pub fn parse_srt_str(content: &str) -> std::result::Result<Vec<SubtitleItem>, FormatError> {
    let content = normalize_newlines(content);
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }

    let regexes = SrtRegexes::new();
    let mut items: Vec<SubtitleItem> = Vec::new();
    let mut seen_ids: HashSet<usize> = HashSet::new();

    let trimmed = content.trim_matches('\n');
    for (ordinal, raw_block) in regexes.block.split(trimmed).enumerate() {
        let lines: Vec<&str> = raw_block.split('\n').collect();
        let timing_idx = lines.iter().position(|line| regexes.timing.is_match(line));
        let Some(timing_idx) = timing_idx else {
            continue;
        };

        let mut cue_id = ordinal + 1;
        if timing_idx > 0 {
            let first = lines[0].trim();
            if !first.is_empty() && first.bytes().all(|b| b.is_ascii_digit()) {
                cue_id = first.parse().unwrap();
            }
        }
        if !seen_ids.insert(cue_id) {
            return Err(FormatError(format!("Duplicate cue id in SRT: {cue_id}")));
        }

        let (start, end) = parse_srt_timing_line(&regexes, lines[timing_idx])?;
        let text = lines[timing_idx + 1..].join("\n").trim().to_string();
        if text.is_empty() {
            continue;
        }
        items.push(SubtitleItem::new(cue_id, start, end, text));
    }

    Ok(items)
}

/// Parse an SRT file into [`SubtitleItem`]s.
pub fn parse_srt(path: impl AsRef<Path>) -> std::result::Result<Vec<SubtitleItem>, FormatError> {
    let content = fs::read_to_string(path.as_ref())
        .map_err(|err| FormatError(format!("could not read SRT file: {err}")))?;
    parse_srt_str(&content)
}

struct LrcRegexes {
    metadata: Regex,
    timestamp: Regex,
    word_tag: Regex,
}

impl LrcRegexes {
    fn new() -> Self {
        Self {
            metadata: Regex::new(r"^\[([A-Za-z]+):(.*)\]$").unwrap(),
            timestamp: Regex::new(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]").unwrap(),
            word_tag: Regex::new(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>").unwrap(),
        }
    }
}

/// Parse LRC content (already read from disk) into metadata tags and untimed
/// text lines.
pub fn parse_lrc_str(content: &str) -> LrcData {
    let content = normalize_newlines(content);
    let regexes = LrcRegexes::new();
    let mut data = LrcData::default();

    for raw_line in content.split('\n') {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if regexes.metadata.is_match(line) {
            data.metadata.push(line.to_string());
            continue;
        }
        let line = regexes.timestamp.replace_all(line, "");
        let line = regexes.word_tag.replace_all(&line, "");
        let line = line.trim();
        if !line.is_empty() {
            data.lines.push(line.to_string());
        }
    }
    data
}

/// Parse an LRC file into metadata tags and untimed text lines.
pub fn parse_lrc(path: impl AsRef<Path>) -> std::result::Result<LrcData, FormatError> {
    let content = fs::read_to_string(path.as_ref())
        .map_err(|err| FormatError(format!("could not read LRC file: {err}")))?;
    Ok(parse_lrc_str(&content))
}

/// Load untimed subtitle text from a file (LRC, TXT, etc.).
/// Returns `(metadata_lines, text_lines)`.
pub fn load_text_input(path: impl AsRef<Path>) -> Result<(Vec<String>, Vec<String>)> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    if ext == "lrc" {
        let lrc = parse_lrc(path)?;
        let metadata: Vec<String> = lrc
            .metadata
            .into_iter()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                !lower.starts_with("[length:") && !lower.starts_with("[offset:")
            })
            .collect();
        Ok((metadata, lrc.lines))
    } else {
        let content = fs::read_to_string(path)?;
        let lines: Vec<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        Ok((Vec::new(), lines))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_formatting_rounds_millis() {
        assert_eq!(format_srt_timestamp(3661.2345), "01:01:01,234");
        assert_eq!(format_vtt_timestamp(0.9996), "00:00:01.000");
    }

    #[test]
    fn render_srt_bilingual() {
        let items = vec![SubtitleItem::new(1, 0.0, 1.2, "Hello")
            .with_translation(Some("你好".to_string()))];

        let content = render_srt(&items, SubtitleMode::Bilingual, false);

        assert_eq!(content, "1\n00:00:00,000 --> 00:00:01,200\nHello\n你好\n");
    }

    #[test]
    fn render_vtt_original() {
        let items = vec![SubtitleItem::new(1, 0.0, 1.2, "Hello")];

        let content = render_vtt(&items, SubtitleMode::Original, false);

        assert!(content.starts_with("WEBVTT\n\n"));
        assert!(content.contains("00:00:00.000 --> 00:00:01.200\nHello"));
    }

    #[test]
    fn render_txt_one_line_per_cue_without_timestamps() {
        let items = vec![
            SubtitleItem::new(1, 0.0, 1.2, "Hello"),
            SubtitleItem::new(2, 1.4, 2.0, "   "),
            SubtitleItem::new(3, 2.0, 3.0, "World again"),
        ];

        let content = render_txt(&items, SubtitleMode::Original, false);

        assert_eq!(content, "Hello\nWorld again\n");
    }

    #[test]
    fn render_txt_bilingual_collapses_cue_line_breaks() {
        let items = vec![SubtitleItem::new(1, 0.0, 1.2, "Hello!")
            .with_translation(Some("你好。".to_string()))];

        let content = render_txt(&items, SubtitleMode::Bilingual, true);

        assert_eq!(content, "Hello 你好\n");
    }

    #[test]
    fn render_json_includes_optional_translation() {
        let items = vec![SubtitleItem::new(1, 0.0, 1.2, "Hello")
            .with_translation(Some("你好".to_string()))];

        let data: Vec<serde_json::Value> =
            serde_json::from_str(&render_json(&items, SubtitleMode::Original, false)).unwrap();

        assert_eq!(
            data,
            vec![serde_json::json!({
                "id": 1,
                "start": 0.0,
                "end": 1.2,
                "text": "Hello",
                "translation": "你好"
            })]
        );
    }

    #[test]
    fn format_lrc_timestamp_centiseconds() {
        assert_eq!(format_lrc_timestamp(0.0), "[00:00.00]");
        assert_eq!(format_lrc_timestamp(65.789), "[01:05.79]");
    }

    #[test]
    fn format_lrc_timestamp_carries_rounded_seconds_into_minutes() {
        assert_eq!(format_lrc_timestamp(59.996), "[01:00.00]");
        assert_eq!(format_lrc_timestamp(3599.999), "[60:00.00]");
    }

    #[test]
    fn render_lrc_keeps_metadata_and_uses_start_times() {
        let items = vec![
            SubtitleItem::new(1, 5.0, 7.0, "first line"),
            SubtitleItem::new(2, 9.25, 11.0, "second line"),
        ];

        let content = render_lrc(
            &items,
            Some(&["[ti:Title]".to_string(), "[ar:Artist]".to_string()]),
            false,
        );

        assert_eq!(
            content,
            "[ti:Title]\n[ar:Artist]\n[00:05.00]first line\n[00:09.25]second line\n"
        );
    }

    #[test]
    fn render_lrc_honors_strip_punctuation() {
        let items = vec![SubtitleItem::new(1, 5.0, 7.0, "first line!")];

        let content = render_lrc(&items, None, true);

        assert_eq!(content, "[00:05.00]first line\n");
    }

    #[test]
    fn parse_lrc_strips_timestamps_and_keeps_metadata() {
        let content = "[ti:きっとそう]\n[ar:ルサンチマン]\n[00:12.34]ああ、きっとそう\n[00:15.00]間違ってないような\n";

        let data = parse_lrc_str(content);

        assert_eq!(data.metadata, vec!["[ti:きっとそう]", "[ar:ルサンチマン]"]);
        assert_eq!(data.lines, vec!["ああ、きっとそう", "間違ってないような"]);
    }

    #[test]
    fn parse_lrc_strips_inline_word_tags() {
        let data = parse_lrc_str("[00:01.00]<00:01.00>あ<00:01.50>あ\n");

        assert_eq!(data.lines, vec!["ああ"]);
    }

    #[test]
    fn parse_srt_round_trip() {
        let content = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,500\r\nHello world\r\n\r\n2\n00:00:03.000 --> 00:00:04.000\nSecond\ncue\n";

        let items = parse_srt_str(content).unwrap();

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, 1);
        assert!((items[0].start - 1.0).abs() < 1e-9);
        assert!((items[0].end - 2.5).abs() < 1e-9);
        assert_eq!(items[0].text, "Hello world");
        assert_eq!(items[1].text, "Second\ncue");

        let rendered = render_srt(&items, SubtitleMode::Original, false);
        assert!(rendered.contains("00:00:01,000 --> 00:00:02,500"));
        assert!(rendered.contains("Second\ncue"));
    }

    #[test]
    fn parse_srt_rejects_duplicate_ids() {
        let content = "1\n00:00:01,000 --> 00:00:02,000\na\n\n1\n00:00:03,000 --> 00:00:04,000\nb\n";

        let err = parse_srt_str(content).unwrap_err();

        assert!(err.0.contains("Duplicate cue id"));
    }

    #[test]
    fn parse_srt_skips_blocks_without_timing() {
        let content = "some junk\n\n1\n00:00:01,000 --> 00:00:02,000\nreal\n";

        let items = parse_srt_str(content).unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "real");
    }
}
