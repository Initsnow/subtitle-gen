//! Local NLP text segmentation, Lindera Japanese morphological analysis,
//! and Needleman-Wunsch global lyrics alignment.
//!
//! Fully ported and enhanced from Python's `local_segmenter.py`, `segmenter.py`,
//! and `text_align.py`.

use std::borrow::Cow;
use std::sync::OnceLock;

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

use crate::types::{AudioWindow, SubtitleItem, TimedToken};

pub const MAJOR_BOUNDARY_PUNCTUATION: &str = ".!?。！？";
pub const SOFT_BOUNDARY_PUNCTUATION: &str = ",;:，；：、";
pub const TRAILING_QUOTES: &str = "\"'”’)]}）】」』";
pub const TRAILING_PUNCTUATION: &str = ",.;:!?，。；：！？、)]}）】」』";
pub const LEADING_PUNCTUATION: &str = "([{（【「『";
pub const CJK_PUNCTUATION: &str = "，。；：！？、";
pub const ATTACHABLE_PUNCTUATION: &str = ",.;:!?，。；：！？、)]}）】」』\"'”’";

const BREAK_BEFORE_WORDS: &[&str] = &[
    "and", "as", "because", "but", "for", "if", "in", "nor", "of", "on", "or", "so", "that",
    "to", "when", "while", "with", "yet",
];

const SPURIOUS_GAP: f64 = 3.0;
const ALIGN_FULL_MAX_CELLS: usize = 2_000_000;
const ALIGN_BAND_MAX_CELLS: usize = 24_000_000;
const ALIGN_BAND_MIN: usize = 64;
const ALIGN_BAND_RATIO: f64 = 0.05;

/// Configuration for subtitle segmentation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SegmentConfig {
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_min_duration")]
    pub min_duration: f64,
    #[serde(default = "default_max_duration")]
    pub max_duration: f64,
    #[serde(default = "default_max_lines")]
    pub max_lines: usize,
    #[serde(default = "default_max_chars_en")]
    pub max_chars_en: usize,
    #[serde(default = "default_max_chars_zh")]
    pub max_chars_zh: usize,
    #[serde(default = "default_max_chars_ja")]
    pub max_chars_ja: usize,
    #[serde(default = "default_max_cps_en")]
    pub max_cps_en: f64,
    #[serde(default = "default_max_cps_zh")]
    pub max_cps_zh: f64,
    #[serde(default = "default_max_cps_ja")]
    pub max_cps_ja: f64,
    #[serde(default = "default_pause_threshold")]
    pub pause_threshold: f64,
}

fn default_mode() -> String { "local".into() }
fn default_min_duration() -> f64 { 0.833 }
fn default_max_duration() -> f64 { 7.0 }
fn default_max_lines() -> usize { 2 }
fn default_max_chars_en() -> usize { 42 }
fn default_max_chars_zh() -> usize { 16 }
fn default_max_chars_ja() -> usize { 13 }
fn default_max_cps_en() -> f64 { 20.0 }
fn default_max_cps_zh() -> f64 { 9.0 }
fn default_max_cps_ja() -> f64 { 4.0 }
fn default_pause_threshold() -> f64 { 0.65 }

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            min_duration: default_min_duration(),
            max_duration: default_max_duration(),
            max_lines: default_max_lines(),
            max_chars_en: default_max_chars_en(),
            max_chars_zh: default_max_chars_zh(),
            max_chars_ja: default_max_chars_ja(),
            max_cps_en: default_max_cps_en(),
            max_cps_zh: default_max_cps_zh(),
            max_cps_ja: default_max_cps_ja(),
            pause_threshold: default_pause_threshold(),
        }
    }
}

/// Atomic token unit used for segmentation groupings.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentAtom {
    pub id: usize,
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl SegmentAtom {
    pub fn new(id: usize, text: impl Into<String>, start: f64, end: f64) -> Self {
        Self { id, text: text.into(), start, end }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalSegmentation {
    pub atoms: Vec<SegmentAtom>,
    pub groups: Vec<Vec<usize>>,
    pub language: String,
}

impl LocalSegmentation {
    pub fn subtitles(&self) -> Vec<SubtitleItem> {
        segments_from_atom_groups(&self.groups, &self.atoms)
    }
}

#[derive(Debug, Clone)]
struct GroupMetrics {
    #[allow(dead_code)]
    text: String,
    duration: f64,
    length: usize,
    cps: f64,
}

// ---------------------------------------------------------------------------
// Lindera morphological analysis helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MorphToken {
    pub surface: String,
    pub pos: String,
    pub pos_sub1: String,
    pub conjugation_form: String,
}

static LINDERA_SEGMENTER: OnceLock<Option<lindera::segmenter::Segmenter>> = OnceLock::new();

fn get_lindera_segmenter() -> Option<&'static lindera::segmenter::Segmenter> {
    LINDERA_SEGMENTER
        .get_or_init(|| {
            match lindera::dictionary::load_dictionary("embedded://ipadic") {
                Ok(dict) => {
                    Some(lindera::segmenter::Segmenter::new(
                        lindera::mode::Mode::Normal,
                        dict,
                        None,
                    ))
                }
                Err(err) => {
                    tracing::warn!("Failed to load embedded IPADIC dictionary: {err}");
                    None
                }
            }
        })
        .as_ref()
}

/// Perform morphological analysis on Japanese text using Lindera (IPADIC).
pub fn analyze_japanese(text: &str) -> Vec<MorphToken> {
    let segmenter = match get_lindera_segmenter() {
        Some(s) => s,
        None => return Vec::new(),
    };

    let mut tokens = match segmenter.segment(Cow::Borrowed(text)) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::with_capacity(tokens.len());
    for token in &mut tokens {
        let surface = token.surface.to_string();
        let details = token.details();
        let pos = details.first().copied().unwrap_or("").to_string();
        let pos_sub1 = details.get(1).copied().unwrap_or("").to_string();
        let conjugation_form = details.get(5).copied().unwrap_or("").to_string();
        result.push(MorphToken {
            surface,
            pos,
            pos_sub1,
            conjugation_form,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Local Segmenter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LocalSegmenter {
    pub config: SegmentConfig,
    pub refine_mode: String,
}

impl LocalSegmenter {
    pub fn new(config: Option<SegmentConfig>, refine_mode: Option<&str>) -> Self {
        let config = config.unwrap_or_default();
        let refine_mode = refine_mode
            .map(String::from)
            .unwrap_or_else(|| refine_mode_for_segment_mode(&config.mode));
        Self { config, refine_mode }
    }

    pub fn segment(
        &self,
        tokens: &[TimedToken],
        context_text: Option<&str>,
        language: Option<&str>,
    ) -> Vec<SubtitleItem> {
        let result = self.segment_groups(tokens, context_text, language);
        result.subtitles()
    }

    pub fn segment_groups(
        &self,
        tokens: &[TimedToken],
        context_text: Option<&str>,
        language: Option<&str>,
    ) -> LocalSegmentation {
        let projected = project_context_punctuation(tokens, context_text);
        let atoms = atomize_tokens(&projected);
        if atoms.is_empty() {
            return LocalSegmentation {
                atoms: Vec::new(),
                groups: Vec::new(),
                language: normalize_language(language, ""),
            };
        }

        let source_text = join_text_parts(atoms.iter().map(|a| a.text.as_str()));
        let resolved_language =
            normalize_language(language, context_text.unwrap_or(&source_text));
        let groups = sentence_groups(&atoms, &self.config);

        if self.refine_mode == "none" {
            return LocalSegmentation {
                atoms,
                groups,
                language: resolved_language,
            };
        }

        let by_id: std::collections::HashMap<usize, &SegmentAtom> =
            atoms.iter().map(|a| (a.id, a)).collect();

        let mut refined = Vec::new();
        let allow_weak_fallback = self.refine_mode == "hard";
        for group in groups {
            refined.extend(split_group(
                &group,
                &by_id,
                &self.config,
                &resolved_language,
                allow_weak_fallback,
            ));
        }

        let merged = merge_short_groups(&refined, &by_id, &self.config, &resolved_language);
        LocalSegmentation {
            atoms,
            groups: merged,
            language: resolved_language,
        }
    }
}

fn refine_mode_for_segment_mode(mode: &str) -> String {
    if mode == "local" {
        "hard".into()
    } else {
        "none".into()
    }
}

pub fn atomize_tokens(tokens: &[TimedToken]) -> Vec<SegmentAtom> {
    let cleaned = clean_tokens(tokens);
    cleaned
        .into_iter()
        .enumerate()
        .map(|(index, token)| SegmentAtom {
            id: index + 1,
            text: token.text,
            start: token.start,
            end: token.end,
        })
        .collect()
}

pub fn clean_tokens(tokens: &[TimedToken]) -> Vec<TimedToken> {
    let mut cleaned: Vec<TimedToken> = tokens
        .iter()
        .filter(|t| !t.text.trim().is_empty() && t.end >= t.start)
        .cloned()
        .collect();
    cleaned.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.end
                    .partial_cmp(&b.end)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    cleaned
}

// ---------------------------------------------------------------------------
// Sentence Splitting & Boundaries
// ---------------------------------------------------------------------------

fn sentence_groups(atoms: &[SegmentAtom], config: &SegmentConfig) -> Vec<Vec<usize>> {
    let source_text = join_text_parts(atoms.iter().map(|a| a.text.as_str()));
    let sentence_parts = split_sentences(&source_text);
    if !sentence_parts.is_empty() {
        let groups = groups_from_text_parts(&sentence_parts, atoms);
        if !groups.is_empty() {
            return groups;
        }
    }
    fallback_sentence_groups(atoms, config)
}

fn split_sentences(text: &str) -> Vec<String> {
    // Unicode sentence segmentation
    text.unicode_sentences()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn fallback_sentence_groups(atoms: &[SegmentAtom], config: &SegmentConfig) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut current = Vec::new();
    for (index, atom) in atoms.iter().enumerate() {
        current.push(atom.id);
        let next_atom = atoms.get(index + 1);
        if let Some(next) = next_atom {
            let pause = (next.start - atom.end).max(0.0);
            if is_major_boundary(&atom.text) || pause >= config.pause_threshold {
                groups.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    if groups.is_empty() {
        vec![atoms.iter().map(|a| a.id).collect()]
    } else {
        groups
    }
}

fn split_group(
    group: &[usize],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    config: &SegmentConfig,
    language: &str,
    allow_weak_fallback: bool,
) -> Vec<Vec<usize>> {
    if group.len() <= 1 || !needs_split(group, by_id, config, language) {
        return vec![group.to_vec()];
    }

    let split_index = best_split_index(group, by_id, config, language, allow_weak_fallback);
    let split_index = match split_index {
        Some(idx) => idx,
        None => return vec![group.to_vec()],
    };

    let left = &group[..split_index];
    let right = &group[split_index..];
    if left.is_empty() || right.is_empty() {
        return vec![group.to_vec()];
    }

    let mut result = split_group(left, by_id, config, language, allow_weak_fallback);
    result.extend(split_group(
        right,
        by_id,
        config,
        language,
        allow_weak_fallback,
    ));
    result
}

fn best_split_index(
    group: &[usize],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    config: &SegmentConfig,
    language: &str,
    allow_weak_fallback: bool,
) -> Option<usize> {
    let metrics = group_metrics(group, by_id, language);
    let allow_weak = allow_weak_fallback && is_hard_over_limit(&metrics, config, language);
    let mut best_index = None;
    let mut best_score = f64::NEG_INFINITY;

    for split_index in 1..group.len() {
        let left = &group[..split_index];
        let right = &group[split_index..];
        let left_atoms = group_atoms(left, by_id);
        let right_atoms = group_atoms(right, by_id);
        if left_atoms.is_empty() || right_atoms.is_empty() {
            continue;
        }

        let left_last = left_atoms[left_atoms.len() - 1];
        let right_first = right_atoms[0];

        let boundary_score = calculate_boundary_score(left_last, right_first, config, language);
        if boundary_score <= 0.0 && !allow_weak {
            continue;
        }

        let left_metrics = group_metrics(left, by_id, language);
        let right_metrics = group_metrics(right, by_id, language);

        let left_share = left_metrics.length as f64 / (metrics.length.max(1) as f64);
        let balance_penalty = (left_share - 0.5).abs() * 30.0;
        let min_duration_penalty = min_duration_penalty(&left_metrics, &right_metrics, config);
        let overflow_penalty = overflow_penalty(&left_metrics, config, language)
            + overflow_penalty(&right_metrics, config, language);
        let weak_bonus = if boundary_score <= 0.0 { 1.0 } else { 0.0 };

        let mut score = boundary_score + weak_bonus - balance_penalty - min_duration_penalty;
        score -= overflow_penalty * 25.0;

        if score > best_score {
            best_score = score;
            best_index = Some(split_index);
        }
    }

    best_index
}

/// Calculate boundary score at the split between `left` and `right`.
/// Incorporates punctuation, physical acoustic pause, Lindera morphology (Japanese),
/// and English break-before words.
fn calculate_boundary_score(
    left: &SegmentAtom,
    right: &SegmentAtom,
    config: &SegmentConfig,
    language: &str,
) -> f64 {
    let pause = (right.start - left.end).max(0.0);
    let mut score = 0.0f64;

    // 1. Punctuation scores
    if is_major_boundary(&left.text) {
        score = score.max(100.0);
    } else if is_soft_boundary(&left.text) {
        score = score.max(80.0);
    }

    // 2. Physical acoustic pause (Task 3.2: pause >= 150ms grants high break tendency)
    if pause >= config.pause_threshold {
        score = score.max(70.0 + pause.min(1.5) * 5.0);
    } else if pause >= 0.150 {
        // Physical silence >= 150ms gives strong acoustic split tendency
        score = score.max(55.0 + pause.min(1.5) * 10.0);
    }

    // 3. Language-specific grammar / POS rules (Task 3.1)
    if language == "ja" {
        // Check Lindera POS tags
        let ja_score = japanese_pos_boundary_score(&left.text, &right.text);
        score = score.max(ja_score);
    } else if language == "en" && starts_with_break_before_word(&right.text) {
        score = score.max(45.0);
    }

    score
}

/// Check POS boundary score for Japanese using Lindera.
/// High priority: after particle (助詞: は, が, を, に, ので, etc.) -> 65.0
/// Mid priority: before conjunction (接続詞: そして, しかし, etc.) -> 50.0
/// Aux priority: after verb continuative form (て/で) -> 40.0
fn japanese_pos_boundary_score(left_text: &str, right_text: &str) -> f64 {
    let mut score: f64 = 0.0;

    // Check right token: starts with conjunction?
    let right_morphs = analyze_japanese(right_text);
    if let Some(first) = right_morphs.first() {
        if first.pos == "接続詞" {
            score = score.max(50.0);
        }
    }

    // Check left token: ends with particle or verb continuative form?
    let left_morphs = analyze_japanese(left_text);
    if let Some(last) = left_morphs.last() {
        if last.pos == "助詞" {
            score = score.max(65.0);
        } else if last.conjugation_form.contains("連用形") || last.surface == "て" || last.surface == "で" {
            score = score.max(40.0);
        }
    }

    score
}

fn merge_short_groups(
    groups: &[Vec<usize>],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    config: &SegmentConfig,
    language: &str,
) -> Vec<Vec<usize>> {
    let mut merged: Vec<Vec<usize>> = Vec::new();
    let mut index = 0;
    while index < groups.len() {
        let group = &groups[index];
        let metrics = group_metrics(group, by_id, language);
        if metrics.duration >= config.min_duration || groups.len() == 1 {
            merged.push(group.clone());
            index += 1;
            continue;
        }

        if index + 1 < groups.len()
            && can_merge(group, &groups[index + 1], by_id, config, language)
        {
            let mut combined = group.clone();
            combined.extend_from_slice(&groups[index + 1]);
            merged.push(combined);
            index += 2;
            continue;
        }

        if let Some(last) = merged.last_mut() {
            if can_merge(last, group, by_id, config, language) {
                last.extend_from_slice(group);
                index += 1;
                continue;
            }
        }

        merged.push(group.clone());
        index += 1;
    }

    merged
}

fn can_merge(
    left: &[usize],
    right: &[usize],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    config: &SegmentConfig,
    language: &str,
) -> bool {
    let mut combined = left.to_vec();
    combined.extend_from_slice(right);
    let metrics = group_metrics(&combined, by_id, language);
    let max_chars = max_total_chars(config, language);
    metrics.duration <= config.max_duration && metrics.length <= max_chars
}

fn needs_split(
    group: &[usize],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    config: &SegmentConfig,
    language: &str,
) -> bool {
    let metrics = group_metrics(group, by_id, language);
    metrics.duration > config.max_duration || metrics.length > max_total_chars(config, language)
}

fn is_hard_over_limit(metrics: &GroupMetrics, config: &SegmentConfig, language: &str) -> bool {
    metrics.duration > config.max_duration * 1.10
        || metrics.length > max_total_chars(config, language)
}

fn min_duration_penalty(
    left: &GroupMetrics,
    right: &GroupMetrics,
    config: &SegmentConfig,
) -> f64 {
    let mut penalty = 0.0;
    if left.duration < config.min_duration {
        penalty += (config.min_duration - left.duration) / config.min_duration.max(0.001) * 35.0;
    }
    if right.duration < config.min_duration {
        penalty += (config.min_duration - right.duration) / config.min_duration.max(0.001) * 35.0;
    }
    penalty
}

fn overflow_penalty(metrics: &GroupMetrics, config: &SegmentConfig, language: &str) -> f64 {
    let mut penalty = 0.0;
    if metrics.duration > config.max_duration {
        penalty += metrics.duration / config.max_duration.max(0.001) - 1.0;
    }
    let max_chars = max_total_chars(config, language) as f64;
    if metrics.length as f64 > max_chars {
        penalty += (metrics.length as f64) / max_chars.max(1.0) - 1.0;
    }
    let max_cps = max_cps_for_lang(config, language);
    if metrics.cps > max_cps {
        penalty += metrics.cps / max_cps.max(0.001) - 1.0;
    }
    penalty
}

fn group_metrics(
    group: &[usize],
    by_id: &std::collections::HashMap<usize, &SegmentAtom>,
    language: &str,
) -> GroupMetrics {
    let atoms = group_atoms(group, by_id);
    if atoms.is_empty() {
        return GroupMetrics {
            text: String::new(),
            duration: 0.0,
            length: 0,
            cps: 0.0,
        };
    }
    let text = join_text_parts(atoms.iter().map(|a| a.text.as_str()))
        .trim()
        .to_string();
    let duration = (atoms[atoms.len() - 1].end - atoms[0].start).max(0.001);
    let length = measure_text(&text, language);
    GroupMetrics {
        text,
        duration,
        length,
        cps: length as f64 / duration,
    }
}

fn group_atoms<'a>(
    group: &[usize],
    by_id: &std::collections::HashMap<usize, &'a SegmentAtom>,
) -> Vec<&'a SegmentAtom> {
    group.iter().filter_map(|id| by_id.get(id).copied()).collect()
}

pub fn is_major_boundary(text: &str) -> bool {
    let stripped = strip_trailing_quotes(text);
    stripped
        .chars()
        .last()
        .map(|c| MAJOR_BOUNDARY_PUNCTUATION.contains(c))
        .unwrap_or(false)
}

pub fn is_soft_boundary(text: &str) -> bool {
    let stripped = strip_trailing_quotes(text);
    stripped
        .chars()
        .last()
        .map(|c| SOFT_BOUNDARY_PUNCTUATION.contains(c))
        .unwrap_or(false)
}

fn strip_trailing_quotes(text: &str) -> &str {
    text.trim().trim_end_matches(|c| TRAILING_QUOTES.contains(c))
}

fn starts_with_break_before_word(text: &str) -> bool {
    let trimmed = text.trim();
    let word = trimmed
        .split(|c: char| !c.is_ascii_alphabetic() && c != '\'')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    BREAK_BEFORE_WORDS.contains(&word.as_str())
}

pub fn normalize_language(language: Option<&str>, text: &str) -> String {
    let norm = language.unwrap_or("").trim().to_ascii_lowercase();
    if norm.starts_with("ja") || norm.starts_with("jp") || norm.contains("日本") {
        return "ja".into();
    }
    if norm.starts_with("zh")
        || norm.starts_with("ch")
        || norm.contains("中文")
        || norm.contains("汉")
        || norm.contains("漢")
    {
        return "zh".into();
    }
    if norm.starts_with("en") || norm.contains("english") {
        return "en".into();
    }

    static JA_RE: OnceLock<Regex> = OnceLock::new();
    static CJK_RE: OnceLock<Regex> = OnceLock::new();
    let ja_re = JA_RE.get_or_init(|| Regex::new(r"[\u3040-\u30ff]").unwrap());
    let cjk_re = CJK_RE.get_or_init(|| Regex::new(r"[\u3400-\u9fff]").unwrap());

    if ja_re.is_match(text) {
        "ja".into()
    } else if cjk_re.is_match(text) {
        "zh".into()
    } else {
        "en".into()
    }
}

pub fn max_chars_per_line(config: &SegmentConfig, language: &str) -> usize {
    match language {
        "ja" => config.max_chars_ja,
        "zh" => config.max_chars_zh,
        _ => config.max_chars_en,
    }
}

pub fn max_total_chars(config: &SegmentConfig, language: &str) -> usize {
    config.max_lines.max(1) * max_chars_per_line(config, language)
}

pub fn max_cps_for_lang(config: &SegmentConfig, language: &str) -> f64 {
    match language {
        "ja" => config.max_cps_ja,
        "zh" => config.max_cps_zh,
        _ => config.max_cps_en,
    }
}

pub fn measure_text(text: &str, language: &str) -> usize {
    if language == "ja" || language == "zh" {
        text.chars().filter(|c| !c.is_whitespace()).count()
    } else {
        text.split_whitespace().collect::<Vec<_>>().join(" ").len()
    }
}

// ---------------------------------------------------------------------------
// Text Formatting & Punctuation Projection
// ---------------------------------------------------------------------------

pub fn join_text_parts<I, S>(parts: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut output = String::new();
    for raw_part in parts {
        let part = raw_part.as_ref().trim();
        if part.is_empty() {
            continue;
        }
        if output.is_empty() {
            output.push_str(part);
            continue;
        }
        let left = output.chars().last().unwrap();
        let right = part.chars().next().unwrap();
        if needs_space(left, right) {
            output.push(' ');
        }
        output.push_str(part);
    }
    output
}

fn needs_space(left: char, right: char) -> bool {
    if LEADING_PUNCTUATION.contains(left) || TRAILING_PUNCTUATION.contains(right) {
        return false;
    }
    if CJK_PUNCTUATION.contains(left) {
        return false;
    }
    if left == '\'' || right == '\'' {
        return false;
    }
    if left.is_ascii_digit() && right.is_ascii_digit() {
        return false;
    }
    if is_cjk(left) || is_cjk(right) {
        return false;
    }
    true
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}' | // Hiragana & Katakana
        '\u{3400}'..='\u{9fff}' | // CJK Unified Ideographs
        '\u{ac00}'..='\u{d7af}'   // Hangul Syllables
    )
}

fn is_content_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || is_cjk(c)
}

/// Project punctuation from `context_text` onto word tokens.
pub fn project_context_punctuation(
    tokens: &[TimedToken],
    context_text: Option<&str>,
) -> Vec<TimedToken> {
    let cleaned = clean_tokens(tokens);
    let context_text = match context_text {
        Some(t) if !t.is_empty() => t,
        _ => return cleaned,
    };
    if cleaned.is_empty() {
        return cleaned;
    }

    let spans = match_context_spans(&cleaned, context_text);
    if !spans.iter().any(|s| s.is_some()) {
        return cleaned;
    }

    let mut projected = Vec::with_capacity(cleaned.len());
    for (index, token) in cleaned.iter().enumerate() {
        let span = match spans[index] {
            Some(s) => s,
            None => {
                projected.push(token.clone());
                continue;
            }
        };

        let (_start, end) = span;
        let next_span = spans.get(index + 1).copied().flatten();
        if next_span.is_none() && index + 1 < spans.len() {
            projected.push(token.clone());
            continue;
        }

        let next_start = next_span.map(|(s, _)| s).unwrap_or(context_text.len());
        if end > next_start || end > context_text.len() || next_start > context_text.len() {
            projected.push(token.clone());
            continue;
        }

        let between = &context_text[end..next_start];
        if between.chars().any(is_content_char) {
            projected.push(token.clone());
            continue;
        }

        let trailing: String = between
            .chars()
            .filter(|c| ATTACHABLE_PUNCTUATION.contains(*c))
            .collect();
        let text = append_missing_trailing_punctuation(&token.text, &trailing);
        projected.push(TimedToken::new(text, token.start, token.end));
    }

    projected
}

fn append_missing_trailing_punctuation(text: &str, trailing: &str) -> String {
    let mut missing = trailing;
    while !missing.is_empty() && text.trim_end().ends_with(missing.chars().next().unwrap()) {
        missing = &missing[missing.chars().next().unwrap().len_utf8()..];
    }
    if !missing.is_empty() && !text.trim_end().ends_with(missing) {
        format!("{text}{missing}")
    } else {
        text.to_string()
    }
}

fn match_context_spans(tokens: &[TimedToken], context_text: &str) -> Vec<Option<(usize, usize)>> {
    let context_units: Vec<(char, usize)> = context_text
        .char_indices()
        .filter(|&(_, c)| is_content_char(c))
        .map(|(idx, c)| (c.to_ascii_lowercase(), idx))
        .collect();

    if context_units.is_empty() {
        return vec![None; tokens.len()];
    }

    let mut cursor = 0;
    let mut spans = Vec::with_capacity(tokens.len());
    for token in tokens {
        let target = normalize_for_context_mapping(&token.text);
        if target.is_empty() {
            spans.push(None);
            continue;
        }
        let target_chars: Vec<char> = target.chars().collect();
        let start_cursor = find_context_units(&context_units, &target_chars, cursor);
        match start_cursor {
            Some(sc) => {
                let end_cursor = sc + target_chars.len();
                let start = context_units[sc].1;
                let last_unit = &context_units[end_cursor - 1];
                let end = last_unit.1 + last_unit.0.len_utf8();
                spans.push(Some((start, end)));
                cursor = end_cursor;
            }
            None => {
                spans.push(None);
            }
        }
    }
    spans
}

fn find_context_units(
    units: &[(char, usize)],
    target: &[char],
    start_cursor: usize,
) -> Option<usize> {
    if target.len() > units.len() {
        return None;
    }
    let max_start = units.len() - target.len();
    for cur in start_cursor..=max_start {
        let matches = target
            .iter()
            .enumerate()
            .all(|(i, &tc)| units[cur + i].0 == tc);
        if matches {
            return Some(cur);
        }
    }
    None
}

fn normalize_for_context_mapping(text: &str) -> String {
    text.chars()
        .filter(|c| is_content_char(*c))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

pub fn segments_from_atom_groups(
    groups: &[Vec<usize>],
    atoms: &[SegmentAtom],
) -> Vec<SubtitleItem> {
    let by_id: std::collections::HashMap<usize, &SegmentAtom> =
        atoms.iter().map(|a| (a.id, a)).collect();
    let mut segments = Vec::new();

    for group in groups {
        let group_atoms: Vec<&SegmentAtom> =
            group.iter().filter_map(|id| by_id.get(id).copied()).collect();
        if group_atoms.is_empty() {
            continue;
        }
        let text = join_text_parts(group_atoms.iter().map(|a| a.text.as_str()))
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }
        let start = group_atoms[0].start;
        let end = group_atoms[group_atoms.len() - 1].end;
        if end <= start {
            continue;
        }
        segments.push(SubtitleItem::new(segments.len() + 1, start, end, text));
    }

    segments
}

pub fn groups_from_text_parts(parts: &[String], atoms: &[SegmentAtom]) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut atom_index = 0;

    for part in parts {
        let target = normalize_for_mapping(part);
        if target.is_empty() {
            continue;
        }

        let start_index = atom_index;
        let mut consumed = String::new();
        while atom_index < atoms.len() && consumed.len() < target.len() {
            consumed.push_str(&normalize_for_mapping(&atoms[atom_index].text));
            atom_index += 1;
        }

        if consumed != target {
            return Vec::new();
        }

        let group: Vec<usize> = atoms[start_index..atom_index].iter().map(|a| a.id).collect();
        if group.is_empty() {
            return Vec::new();
        }
        groups.push(group);
    }

    if atom_index != atoms.len() {
        return Vec::new();
    }
    groups
}

pub fn normalize_for_mapping(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace() && *c != '|').collect()
}

// ---------------------------------------------------------------------------
// Needleman-Wunsch Global Lyrics Alignment (Task 3.3)
// ---------------------------------------------------------------------------

/// Normalize reference / ASR text for global Needleman-Wunsch alignment:
/// NFKC normalization, lowercase, alphanumeric + CJK only, and Katakana -> Hiragana for Japanese.
pub fn normalize_for_alignment(text: &str, language: Option<&str>) -> String {
    use unicode_normalization::UnicodeNormalization;
    let nfkc: String = text.nfkc().collect();
    let lowered = nfkc.to_lowercase();
    let filtered: String = lowered.chars().filter(|c: &char| c.is_alphanumeric()).collect();
    if let Some(lang) = language {
        let l = lang.to_ascii_lowercase();
        if l.starts_with("ja") || l.starts_with("jp") {
            return katakana_to_hiragana(&filtered);
        }
    }
    filtered
}

pub fn katakana_to_hiragana(text: &str) -> String {
    text.chars()
        .map(|c| {
            let cp = c as u32;
            if (0x30A1..=0x30F6).contains(&cp) {
                char::from_u32(cp - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Global Needleman-Wunsch alignment mapping each ref char index to an ASR char index,
/// or `None` if aligned to a gap.
pub fn align_chars(ref_text: &str, asr_text: &str) -> Vec<Option<usize>> {
    let n = ref_text.chars().count();
    let m = asr_text.chars().count();
    if n == 0 {
        return Vec::new();
    }
    if m == 0 {
        return vec![None; n];
    }

    let ref_chars: Vec<char> = ref_text.chars().collect();
    let asr_chars: Vec<char> = asr_text.chars().collect();

    if n * m <= ALIGN_FULL_MAX_CELLS {
        align_chars_full(&ref_chars, &asr_chars)
    } else {
        let band = choose_alignment_band(n, m);
        match band {
            Some(b) => align_chars_banded(&ref_chars, &asr_chars, b).unwrap_or_else(|| vec![None; n]),
            None => vec![None; n],
        }
    }
}

fn choose_alignment_band(n: usize, m: usize) -> Option<usize> {
    let mut band = ALIGN_BAND_MIN.max((n.max(m) as f64 * ALIGN_BAND_RATIO) as usize);
    if m > n {
        band = band.max(m.div_ceil(n) + 1);
    }
    let max_by_cells = 1.max((ALIGN_BAND_MAX_CELLS / n.max(1) - 1) / 2);
    if band > max_by_cells {
        band = max_by_cells;
        if m > n && band < m.div_ceil(n) + 1 {
            return None;
        }
    }
    Some(1.max(band))
}

fn align_chars_full(ref_chars: &[char], asr_chars: &[char]) -> Vec<Option<usize>> {
    let n = ref_chars.len();
    let m = asr_chars.len();
    let match_score = 2i32;
    let mismatch_score = -1i32;
    let gap_score = -2i32;
    let width = m + 1;

    let mut directions = vec![0u8; (n + 1) * width];
    let mut prev = vec![0i32; width];
    for j in 1..width {
        prev[j] = j as i32 * gap_score;
        directions[j] = 3; // left
    }

    for i in 1..=n {
        let mut curr = vec![0i32; width];
        curr[0] = i as i32 * gap_score;
        directions[i * width] = 2; // up
        let row_base = i * width;
        let ref_char = ref_chars[i - 1];

        for j in 1..width {
            let diag = prev[j - 1]
                + if ref_char == asr_chars[j - 1] {
                    match_score
                } else {
                    mismatch_score
                };
            let up = prev[j] + gap_score;
            let left = curr[j - 1] + gap_score;

            if diag >= up && diag >= left {
                curr[j] = diag;
                directions[row_base + j] = 1;
            } else if up >= left {
                curr[j] = up;
                directions[row_base + j] = 2;
            } else {
                curr[j] = left;
                directions[row_base + j] = 3;
            }
        }
        prev = curr;
    }

    let mut mapping = vec![None; n];
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        let direction = directions[i * width + j];
        if direction == 1 {
            i -= 1;
            j -= 1;
            mapping[i] = Some(j);
        } else if direction == 2 {
            i -= 1;
        } else if direction == 3 {
            j -= 1;
        } else {
            break;
        }
    }

    mapping
}

fn band_bounds(i: usize, n: usize, m: usize, band: usize) -> (usize, usize) {
    let center = (i * m + n / 2) / n.max(1);
    let lo = center.saturating_sub(band);
    let hi = m.min(center + band);
    (lo, hi)
}

fn align_chars_banded(
    ref_chars: &[char],
    asr_chars: &[char],
    band: usize,
) -> Option<Vec<Option<usize>>> {
    let n = ref_chars.len();
    let m = asr_chars.len();
    let match_score = 2i64;
    let mismatch_score = -1i64;
    let gap_score = -2i64;
    let neg = -1_000_000_000_000_000_000i64;

    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n + 1);
    let mut prev: Option<(usize, usize, Vec<i64>)> = None;

    for i in 0..=n {
        let (lo, hi) = band_bounds(i, n, m, band);
        let width = hi - lo + 1;
        let mut curr = vec![0i64; width];
        let mut directions = vec![0u8; width];

        if i == 0 {
            for (index, j) in (lo..=hi).enumerate() {
                curr[index] = j as i64 * gap_score;
                if j > 0 {
                    directions[index] = 3;
                }
            }
        } else {
            let (plo, phi, prev_scores) = prev.as_ref()?;
            let (plo, phi) = (*plo, *phi);
            let ref_char = ref_chars[i - 1];

            for (index, j) in (lo..=hi).enumerate() {
                let mut diag = neg;
                if j > 0 && (plo..=phi).contains(&(j - 1)) {
                    diag = prev_scores[j - 1 - plo]
                        + if ref_char == asr_chars[j - 1] {
                            match_score
                        } else {
                            mismatch_score
                        };
                }
                let mut up = neg;
                if (plo..=phi).contains(&j) {
                    up = prev_scores[j - plo] + gap_score;
                }
                let mut left = neg;
                if index > 0 {
                    left = curr[index - 1] + gap_score;
                }

                if diag >= up && diag >= left {
                    curr[index] = diag;
                    directions[index] = 1;
                } else if up >= left {
                    curr[index] = up;
                    directions[index] = 2;
                } else {
                    curr[index] = left;
                    directions[index] = 3;
                }
            }
        }

        rows.push(directions);
        prev = Some((lo, hi, curr));
    }

    let mut mapping = vec![None; n];
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        let (lo, hi) = band_bounds(i, n, m, band);
        if j < lo || j > hi {
            return None;
        }
        let direction = rows[i][j - lo];
        if direction == 1 {
            i -= 1;
            j -= 1;
            mapping[i] = Some(j);
        } else if direction == 2 {
            i -= 1;
        } else if direction == 3 {
            j -= 1;
        } else {
            return None;
        }
    }

    Some(mapping)
}

/// Map each reference line to a `(start, end)` span in the timed token stream.
pub fn match_lines_to_tokens(
    lines: &[String],
    tokens: &[TimedToken],
    language: Option<&str>,
) -> Vec<Option<(f64, f64)>> {
    if lines.is_empty() {
        return Vec::new();
    }
    if tokens.is_empty() {
        return vec![None; lines.len()];
    }

    let mut ref_chars = String::new();
    let mut ref_char_line = Vec::new();
    for (line_index, line) in lines.iter().enumerate() {
        let norm = normalize_for_alignment(line, language);
        for c in norm.chars() {
            ref_chars.push(c);
            ref_char_line.push(line_index);
        }
    }

    let mut asr_chars = String::new();
    let mut asr_char_token = Vec::new();
    for (token_index, token) in tokens.iter().enumerate() {
        let norm = normalize_for_alignment(&token.text, language);
        for c in norm.chars() {
            asr_chars.push(c);
            asr_char_token.push(token_index);
        }
    }

    let mapping = align_chars(&ref_chars, &asr_chars);

    let mut line_min_token: Vec<Option<usize>> = vec![None; lines.len()];
    let mut line_max_token: Vec<Option<usize>> = vec![None; lines.len()];

    for (ref_index, &asr_index) in mapping.iter().enumerate() {
        let asr_idx = match asr_index {
            Some(idx) => idx,
            None => continue,
        };
        let line_index = ref_char_line[ref_index];
        let token_index = asr_char_token[asr_idx];

        line_min_token[line_index] = Some(match line_min_token[line_index] {
            Some(prev) => prev.min(token_index),
            None => token_index,
        });
        line_max_token[line_index] = Some(match line_max_token[line_index] {
            Some(prev) => prev.max(token_index),
            None => token_index,
        });
    }

    let mut spans = Vec::with_capacity(lines.len());
    for line_index in 0..lines.len() {
        match (line_min_token[line_index], line_max_token[line_index]) {
            (Some(min_tok), Some(max_tok)) => {
                spans.push(Some((tokens[min_tok].start, tokens[max_tok].end)));
            }
            _ => spans.push(None),
        }
    }

    spans
}

/// Filter isolated tokens that are hallucinations on silent / instrumental stretches.
pub fn filter_spurious_tokens(tokens: &[TimedToken]) -> Vec<TimedToken> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut kept = Vec::new();
    let total = tokens.len();
    for (index, token) in tokens.iter().enumerate() {
        let prev_gap = if index > 0 {
            Some(token.start - tokens[index - 1].end)
        } else {
            None
        };
        let next_gap = if index + 1 < total {
            Some(tokens[index + 1].start - token.end)
        } else {
            None
        };
        let isolated = (prev_gap.is_none() || prev_gap.unwrap() > SPURIOUS_GAP)
            && (next_gap.is_none() || next_gap.unwrap() > SPURIOUS_GAP);
        if isolated {
            continue;
        }
        kept.push(token.clone());
    }
    kept
}

/// Finalize rough spans into monotonic `(start, end)` timestamps,
/// interpolating missing spans between anchors.
pub fn finalize_spans(spans: &[Option<(f64, f64)>], duration: f64) -> Vec<(f64, f64)> {
    let total = spans.len();
    let mut starts: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut ends: Vec<Option<f64>> = Vec::with_capacity(total);

    for span in spans {
        match span {
            Some((start, end)) if *end > *start => {
                starts.push(Some(*start));
                ends.push(Some(*end));
            }
            _ => {
                starts.push(None);
                ends.push(None);
            }
        }
    }

    let mut index = 0;
    while index < total {
        if starts[index].is_some() {
            index += 1;
            continue;
        }
        let mut run_end = index;
        while run_end < total && starts[run_end].is_none() {
            run_end += 1;
        }
        let count = run_end - index;
        let prev_end = if index > 0 { ends[index - 1] } else { None };
        let next_start = if run_end < total { starts[run_end] } else { None };

        match (prev_end, next_start) {
            (Some(pe), Some(ns)) => {
                let step = (ns - pe) / (count as f64 + 1.0);
                for offset in 0..count {
                    let pos = pe + step * (offset as f64 + 1.0);
                    starts[index + offset] = Some(pos);
                    ends[index + offset] = Some(pos + step.clamp(0.0, 1.0));
                }
            }
            (Some(pe), None) => {
                for offset in 0..count {
                    let pos = pe + 0.5 * (offset as f64 + 1.0);
                    starts[index + offset] = Some(pos);
                    ends[index + offset] = Some(pos + 1.0);
                }
            }
            (None, Some(ns)) => {
                for offset in 0..count {
                    let pos = (ns - 0.5 * (count - offset) as f64).max(0.0);
                    starts[index + offset] = Some(pos);
                    ends[index + offset] = Some(pos + 0.5);
                }
            }
            (None, None) => {
                for offset in 0..count {
                    starts[index + offset] = Some(0.0);
                    ends[index + offset] = Some(1.0);
                }
            }
        }
        index = run_end;
    }

    let mut final_starts: Vec<f64> = starts.into_iter().map(|s| s.unwrap_or(0.0)).collect();
    let mut final_ends: Vec<f64> = ends.into_iter().map(|e| e.unwrap_or(1.0)).collect();

    // Enforce monotonic starts
    for pos in 1..total {
        if final_starts[pos] < final_starts[pos - 1] {
            final_starts[pos] = final_starts[pos - 1];
        }
    }

    // Clamp to media duration
    for pos in 0..total {
        if final_starts[pos] >= duration {
            final_starts[pos] = (duration - 0.5).max(0.0);
        }
        if final_ends[pos] > duration {
            final_ends[pos] = duration;
        }
        if final_ends[pos] <= final_starts[pos] {
            final_ends[pos] = (final_starts[pos] + 0.5).min(duration);
        }
    }

    for pos in 1..total {
        if final_starts[pos] < final_starts[pos - 1] {
            final_starts[pos] = final_starts[pos - 1];
            if final_ends[pos] <= final_starts[pos] {
                final_ends[pos] = (final_starts[pos] + 0.5).min(duration);
            }
        }
    }

    // Clamp overlaps
    for pos in 0..(total.saturating_sub(1)) {
        if final_ends[pos] > final_starts[pos + 1] {
            final_ends[pos] = final_starts[pos + 1];
        }
    }

    for pos in 0..total {
        if final_ends[pos] <= final_starts[pos] {
            final_ends[pos] = (final_starts[pos] + 0.5).min(duration);
        }
    }

    final_starts
        .into_iter()
        .zip(final_ends)
        .map(|(s, e)| (round_to_3(s), round_to_3(e)))
        .collect()
}

fn round_to_3(val: f64) -> f64 {
    (val * 1000.0).round() / 1000.0
}

/// Tile duration with fixed-size audio windows (for song alignment without VAD).
pub fn plan_fixed_windows(
    duration: f64,
    window_size: f64,
    overlap: Option<f64>,
) -> Vec<AudioWindow> {
    let window_size = if window_size <= 0.0 { duration.max(1.0) } else { window_size };
    let overlap = overlap.unwrap_or(1.0).clamp(0.0, (window_size - 0.001).max(0.0));

    let mut windows = Vec::new();
    let mut cursor = 0.0;
    let mut index = 1;
    while cursor < duration {
        let end = duration.min(cursor + window_size);
        if end > cursor {
            windows.push(AudioWindow::new(index, round_to_3(cursor), round_to_3(end)));
            index += 1;
        }
        cursor = if end >= duration { end } else { end - overlap };
    }
    if windows.is_empty() {
        windows.push(AudioWindow::new(1, 0.0, round_to_3(duration.max(0.0))));
    }
    windows
}

/// Merge shifted tokens from multiple windows and deduplicate overlaps.
pub fn merge_window_tokens(
    window_tokens: &[(AudioWindow, Vec<TimedToken>)],
) -> Vec<TimedToken> {
    let mut tokens: Vec<TimedToken> = Vec::new();
    let mut prev_window_end: Option<f64> = None;

    for (window, local_tokens) in window_tokens {
        let mut cleaned = clean_tokens(local_tokens);
        cleaned.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal))
        });
        let shifted: Vec<TimedToken> = cleaned.into_iter().map(|t| t.shifted(window.start)).collect();

        if !tokens.is_empty() {
            if let Some(p_end) = prev_window_end {
                if window.start < p_end - 1e-9 {
                    while let Some(last) = tokens.last() {
                        if last.start >= window.start - 1e-9 {
                            tokens.pop();
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        tokens.extend(shifted);
        prev_window_end = Some(window.end);
    }

    tokens
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_text_parts_handles_cjk_and_punctuation() {
        assert_eq!(
            join_text_parts(["你", "好", "，", "world", "!"]),
            "你好，world!"
        );
    }

    #[test]
    fn test_local_segmenter_projects_context_punctuation() {
        let tokens = vec![
            TimedToken::new("Hello", 0.0, 0.4),
            TimedToken::new("world", 0.4, 1.0),
            TimedToken::new("Next", 1.2, 1.6),
            TimedToken::new("one", 1.6, 2.2),
        ];

        let config = SegmentConfig { min_duration: 0.2, ..Default::default() };
        let segmenter = LocalSegmenter::new(Some(config), None);
        let segments = segmenter.segment(&tokens, Some("Hello world. Next one."), Some("English"));

        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["Hello world.", "Next one."]);
    }

    #[test]
    fn test_project_context_punctuation_keeps_timing() {
        let tokens = vec![
            TimedToken::new("Hello", 0.0, 0.4),
            TimedToken::new("world", 0.4, 1.0),
        ];
        let projected = project_context_punctuation(&tokens, Some("Hello world!"));
        assert_eq!(
            projected,
            vec![
                TimedToken::new("Hello", 0.0, 0.4),
                TimedToken::new("world!", 0.4, 1.0),
            ]
        );
    }

    #[test]
    fn test_project_context_punctuation_does_not_duplicate_partial_suffixes() {
        let projected = project_context_punctuation(
            &[TimedToken::new("Hello.", 0.0, 0.4)],
            Some("Hello.\""),
        );
        assert_eq!(projected, vec![TimedToken::new("Hello.\"", 0.0, 0.4)]);
    }

    #[test]
    fn test_local_segmenter_splits_overlong_sentence_on_soft_boundary() {
        let tokens = vec![
            TimedToken::new("This", 0.0, 0.3),
            TimedToken::new("is", 0.3, 0.6),
            TimedToken::new("quite", 0.6, 0.9),
            TimedToken::new("long,", 0.9, 1.2),
            TimedToken::new("but", 1.2, 1.5),
            TimedToken::new("it", 1.5, 1.8),
            TimedToken::new("can", 1.8, 2.1),
            TimedToken::new("split.", 2.1, 2.6),
        ];

        let config = SegmentConfig {
            mode: "local".into(),
            min_duration: 0.2,
            max_chars_en: 20,
            max_lines: 1,
            ..Default::default()
        };
        let segmenter = LocalSegmenter::new(Some(config), Some("hard"));
        let segments = segmenter.segment(&tokens, None, Some("English"));

        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec!["This is quite long,", "but it can split."]);
    }

    #[test]
    fn test_normalize_lowercases_and_strips_punctuation() {
        assert_eq!(normalize_for_alignment("Hello, World!", None), "helloworld");
    }

    #[test]
    fn test_normalize_folds_fullwidth_and_katakana() {
        assert_eq!(normalize_for_alignment("ABC１２３", None), "abc123");
        assert_eq!(normalize_for_alignment("ベタな愛", Some("ja")), "べたな愛");
        assert_eq!(normalize_for_alignment("ベタな愛", Some("en")), "ベタな愛");
    }

    #[test]
    fn test_align_chars_matches_identical_sequences() {
        assert_eq!(align_chars("abc", "abc"), vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_align_chars_handles_gaps_on_either_side() {
        assert_eq!(align_chars("ac", "abc"), vec![Some(0), Some(2)]);
        assert_eq!(align_chars("abc", "ac"), vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn test_align_chars_empty_inputs() {
        assert_eq!(align_chars("", "abc"), Vec::<Option<usize>>::new());
        assert_eq!(align_chars("abc", ""), vec![None, None, None]);
    }

    #[test]
    fn test_align_chars_switches_to_banded_for_large_inputs() {
        let text = "ab".repeat(1200);
        let expected: Vec<Option<usize>> = (0..text.len()).map(Some).collect();
        assert_eq!(align_chars(&text, &text), expected);
    }

    #[test]
    fn test_match_lines_to_tokens_assigns_spans() {
        let tokens = vec![
            TimedToken::new("Hello", 0.0, 0.5),
            TimedToken::new("world", 0.5, 1.0),
            TimedToken::new("goodbye", 2.0, 2.5),
            TimedToken::new("moon", 2.5, 3.0),
        ];

        let spans = match_lines_to_tokens(
            &["Hello world".into(), "Goodbye moon".into()],
            &tokens,
            None,
        );
        assert_eq!(spans, vec![Some((0.0, 1.0)), Some((2.0, 3.0))]);
    }

    #[test]
    fn test_match_lines_to_tokens_gaps_unmatched() {
        let tokens = vec![
            TimedToken::new("hello", 0.0, 0.5),
            TimedToken::new("world", 1.0, 1.5),
        ];

        let spans = match_lines_to_tokens(
            &["hello".into(), "missing".into(), "world".into()],
            &tokens,
            None,
        );
        assert_eq!(spans, vec![Some((0.0, 0.5)), None, Some((1.0, 1.5))]);
    }

    #[test]
    fn test_finalize_spans_interpolates_between_anchors() {
        let finalized = finalize_spans(&[Some((0.0, 1.0)), None, None, Some((4.0, 5.0))], 10.0);
        assert_eq!(
            finalized,
            vec![(0.0, 1.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
        );
    }

    #[test]
    fn test_finalize_spans_clamps_overlaps_and_duration() {
        let finalized = finalize_spans(&[Some((0.0, 8.0)), Some((2.0, 3.0))], 5.0);
        assert_eq!(finalized, vec![(0.0, 2.0), (2.0, 3.0)]);
    }

    #[test]
    fn test_lindera_japanese_split_bonus() {
        // Test that Lindera analysis finds particles and conjunctions
        let morphs = analyze_japanese("これはテストです。そして次へ。");
        assert!(!morphs.is_empty());
        let has_particle = morphs.iter().any(|m| m.pos == "助詞");
        let has_conj = morphs.iter().any(|m| m.pos == "接続詞");
        assert!(has_particle);
        assert!(has_conj);
    }
}
