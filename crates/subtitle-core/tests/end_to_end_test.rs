use std::fs;
use std::sync::Arc;

use subtitle_core::config::{AppConfig, VadConfig};
use subtitle_core::error::Result;
use subtitle_core::formats::{parse_srt, write_output_set, SubtitleFormat};
use subtitle_core::pipeline::{InferenceBackend, PipelineOptions, SubtitlePipeline};
use subtitle_core::types::TimedToken;

struct TestBackend;

impl InferenceBackend for TestBackend {
    fn detect_speech_regions(&self, samples: &[f32]) -> Result<Vec<(f64, f64)>> {
        let dur = samples.len() as f64 / 16000.0;
        // Generate two realistic speech regions
        Ok(vec![
            (0.1, (dur * 0.45).min(2.0)),
            ((dur * 0.5).min(2.5), (dur * 0.9).min(4.5)),
        ])
    }

    fn transcribe(
        &self,
        _samples: &[f32],
        _hint: Option<&str>,
    ) -> Result<(String, Option<String>)> {
        Ok((
            "こんにちは、世界。そして、新しい字幕です。".to_string(),
            Some("ja".to_string()),
        ))
    }

    fn align(
        &self,
        _samples: &[f32],
        _transcript: &str,
        _lang: Option<&str>,
    ) -> Result<Vec<TimedToken>> {
        // Return timed Japanese tokens with particles and conjunctions
        Ok(vec![
            TimedToken::new("こんにちは", 0.1, 0.8),
            TimedToken::new("、", 0.8, 0.9),
            TimedToken::new("世界", 0.9, 1.5),
            TimedToken::new("。", 1.5, 1.6),
            TimedToken::new("そして", 2.0, 2.5),
            TimedToken::new("、", 2.5, 2.6),
            TimedToken::new("新しい", 2.6, 3.2),
            TimedToken::new("字幕", 3.2, 3.8),
            TimedToken::new("です", 3.8, 4.2),
            TimedToken::new("。", 4.2, 4.3),
        ])
    }
}

#[tokio::test]
async fn test_end_to_end_pipeline_and_file_output() {
    let wav_path = std::path::Path::new("target/media-smoke/test.wav");
    if !wav_path.exists() {
        return;
    }

    let out_dir = std::env::temp_dir().join(format!("test_e2e_subgen_{}", std::process::id()));
    let _ = fs::create_dir_all(&out_dir);

    let mut config = AppConfig::default();
    config.cache.enabled = false;
    config.segment.mode = "local".into();
    config.vad = VadConfig {
        enabled: true,
        min_speech_duration: 0.1,
        min_silence_duration: 0.2,
        speech_padding: 0.1,
        ..Default::default()
    };

    let backend = Arc::new(TestBackend);
    let pipeline = SubtitlePipeline::new(config.clone(), backend);

    let options = PipelineOptions {
        input_path: wav_path.to_path_buf(),
        translate: None,
        segment_mode: Some("local".into()),
        overwrite_cache: false,
        progress: None,
        stream_token: None,
    };

    let result = pipeline.run(options).await.expect("pipeline run");
    assert!(
        !result.subtitles.is_empty(),
        "subtitles should be generated"
    );
    assert_eq!(result.detected_language, Some("ja".to_string()));

    // Verify file output in multiple formats: SRT, VTT, JSON, TXT
    let formats = [
        SubtitleFormat::Srt,
        SubtitleFormat::Vtt,
        SubtitleFormat::Json,
        SubtitleFormat::Txt,
    ];

    let written = write_output_set(
        &out_dir,
        "test_output",
        &result.subtitles,
        &formats,
        false,
        false,
        false,
        false,
    )
    .expect("write outputs");

    assert_eq!(written.len(), 4);

    for path in &written {
        assert!(path.exists(), "Output file {:?} should exist", path);
        let content = fs::read_to_string(path).expect("read content");
        assert!(!content.is_empty(), "File {:?} should not be empty", path);
    }

    // Verify SRT parses back
    let srt_path = &written[0];
    let parsed_cues = parse_srt(srt_path).expect("parse srt");
    assert_eq!(parsed_cues.len(), result.subtitles.len());

    // Clean up
    let _ = fs::remove_dir_all(&out_dir);
}
