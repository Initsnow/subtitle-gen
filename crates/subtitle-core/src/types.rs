//! Domain entities shared across the subtitle generation pipeline.
//!
//! Ported from the Python implementation (`subtitle_gen/types.py`); field
//! semantics and helper behaviour are kept identical so both implementations
//! can process the same data interchangeably.

use serde_json::{json, Map, Value};

/// A word-level token with precise start/end times produced by the forced
/// aligner.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedToken {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl TimedToken {
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self { text: text.into(), start, end }
    }

    /// Return a copy shifted by `offset` seconds (may be negative).
    pub fn shifted(&self, offset: f64) -> Self {
        Self { text: self.text.clone(), start: self.start + offset, end: self.end + offset }
    }
}

/// One subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleItem {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub translation: Option<String>,
    pub proofread: Option<String>,
}

impl SubtitleItem {
    pub fn new(id: usize, start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            id,
            start,
            end,
            text: text.into(),
            translation: None,
            proofread: None,
        }
    }

    pub fn with_translation(mut self, translation: Option<String>) -> Self {
        self.translation = translation;
        self
    }

    pub fn with_proofread(mut self, proofread: Option<String>) -> Self {
        self.proofread = proofread;
        self
    }

    /// Serialize to the JSON object shape used by the JSON subtitle output.
    ///
    /// Key order matches the Python implementation: `id`, `start`, `end`,
    /// `text`, then optional `proofread` and `translation`.
    pub fn to_json_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("id".into(), json!(self.id));
        map.insert("start".into(), json!(round_millis(self.start, 3)));
        map.insert("end".into(), json!(round_millis(self.end, 3)));
        map.insert("text".into(), json!(self.text));
        if let Some(proofread) = &self.proofread {
            map.insert("proofread".into(), json!(proofread));
        }
        if let Some(translation) = &self.translation {
            map.insert("translation".into(), json!(translation));
        }
        map
    }
}

/// A half-open time window over the decoded audio waveform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioWindow {
    pub index: usize,
    pub start: f64,
    pub end: f64,
}

impl AudioWindow {
    pub fn new(index: usize, start: f64, end: f64) -> Self {
        Self { index, start, end }
    }

    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// One ASR transcript chunk over an audio window.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptChunk {
    pub index: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub language: Option<String>,
}

impl TranscriptChunk {
    pub fn new(index: usize, start: f64, end: f64, text: impl Into<String>) -> Self {
        Self { index, start, end, text: text.into(), language: None }
    }
}

/// Round `value` to `digits` decimal places (mirrors Python's JSON output of
/// `round(x, 3)` for timestamps, including banker's rounding).
fn round_millis(value: f64, digits: i32) -> f64 {
    let factor = 10f64.powi(digits);
    round_half_even(value * factor) / factor
}

/// Round to the nearest integer, ties to even — the behaviour of Python's
/// built-in `round()`, which the original implementation relied on for
/// timestamp formatting.
pub(crate) fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac == 0.5 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}
