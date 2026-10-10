//! Audio decoding, mono downmix, and resampling.
//!
//! Replaces the Python implementation's `ffmpeg` subprocess pipeline
//! (`subtitle_gen/media.py`): instead of shelling out to extract a 16 kHz mono
//! WAV and then cutting per-window chunks on disk, the whole file is decoded
//! in memory once (any container/codec supported by the `symphonia` feature
//! set), downmixed to mono `f32`, resampled to the target rate with `rubato`,
//! and every later stage slices the buffer by reference — zero copies, zero
//! disk round-trips.

use std::fs::File;
use std::path::Path;

use std::sync::OnceLock;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};
use crate::types::AudioWindow;

/// Returns the shared [`CodecRegistry`] containing all built-in Symphonia codecs
/// plus the `libopus` decoder adapter.
pub fn get_codec_registry() -> &'static CodecRegistry {
    static REGISTRY: OnceLock<CodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut registry);
        registry.register_audio_decoder::<symphonia_adapter_libopus::OpusDecoder>();
        registry
    })
}

/// The sample rate every inference model in the pipeline expects.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// In-memory mono PCM waveform at a fixed sample rate.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl AudioBuffer {
    /// Decode `path` and resample to [`DEFAULT_SAMPLE_RATE`] (16 kHz).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_rate(path, DEFAULT_SAMPLE_RATE)
    }

    /// Decode `path` and resample to `target_rate`.
    ///
    /// If `ffmpeg` is available on the system PATH, it is preferred for hardware/SIMD
    /// accelerated decoding and resampling in seconds. Otherwise, seamlessly falls back
    /// to the bundled pure-Rust Symphonia + Rubato engine.
    pub fn load_with_rate(path: impl AsRef<Path>, target_rate: u32) -> Result<Self> {
        let path = path.as_ref();
        if let Some(buf) = Self::load_via_ffmpeg(path, target_rate) {
            return Ok(buf);
        }

        let src = File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(src), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let mut format = symphonia::default::get_probe().probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )?;

        // Pick the first track we can actually build a decoder for.
        let mut track_id = None;
        let mut decoder = None;
        for track in format.tracks() {
            let Some(CodecParameters::Audio(ref audio_params)) = track.codec_params else {
                continue;
            };
            if let Ok(dec) = get_codec_registry()
                .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
            {
                track_id = Some(track.id);
                decoder = Some(dec);
                break;
            }
        }
        let track_id = track_id.ok_or(Error::NoAudioTrack)?;
        let mut decoder = decoder.expect("decoder built alongside track_id");

        let mut samples: Vec<f32> = Vec::new();
        let mut sample_rate: Option<u32> = None;
        let mut interleaved: Vec<f32> = Vec::new();

        while let Some(packet) = format.next_packet()? {
            if packet.track_id != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    let spec = decoded.spec();
                    sample_rate = Some(spec.rate());
                    let channels = spec.channels().count();

                    let count = decoded.samples_interleaved();
                    interleaved.resize(count, 0.0);
                    decoded.copy_to_slice_interleaved(&mut interleaved);

                    if channels <= 1 {
                        samples.extend_from_slice(&interleaved);
                    } else {
                        samples.reserve(count / channels);
                        for frame in interleaved.chunks_exact(channels) {
                            let sum: f32 = frame.iter().sum();
                            samples.push(sum / channels as f32);
                        }
                    }
                }
                // A corrupt packet skips; the stream continues.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(err) => return Err(err.into()),
            }
        }

        let sample_rate = sample_rate.ok_or(Error::NoAudioTrack)?;
        let samples = if sample_rate == target_rate {
            samples
        } else {
            resample_mono(&samples, sample_rate, target_rate)?
        };

        Ok(Self {
            samples,
            sample_rate: target_rate,
        })
    }

    /// Try decoding and resampling using system `ffmpeg` if available.
    fn load_via_ffmpeg(path: &Path, target_rate: u32) -> Option<Self> {
        let output = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(path)
            .args([
                "-vn",
                "-ac",
                "1",
                "-ar",
                &target_rate.to_string(),
                "-f",
                "f32le",
                "pipe:1",
            ])
            .output()
            .ok()?;

        if !output.status.success() || output.stdout.is_empty() {
            return None;
        }

        let bytes = output.stdout;
        let sample_count = bytes.len() / 4;
        let mut samples = vec![0.0f32; sample_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                samples.as_mut_ptr() as *mut u8,
                sample_count * 4,
            );
        }

        Some(Self {
            samples,
            sample_rate: target_rate,
        })
    }

    /// Wrap already-decoded mono samples.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Total duration in seconds.
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Cut `[start_sec, end_sec)` out of the waveform with zero copies.
    ///
    /// Mirrors the Python `slice_wav_window`: bounds are rounded to the
    /// nearest sample and clamped so the returned slice is always valid.
    pub fn slice(&self, start_sec: f64, end_sec: f64) -> &[f32] {
        let total = self.samples.len() as i64;
        let rate = self.sample_rate as f64;
        let start = ((start_sec * rate).round() as i64).clamp(0, total) as usize;
        let end = ((end_sec * rate).round() as i64).clamp(start as i64, total) as usize;
        &self.samples[start..end]
    }

    /// Slice by [`AudioWindow`].
    pub fn slice_window(&self, window: &AudioWindow) -> &[f32] {
        self.slice(window.start, window.end)
    }
}

/// Resample a mono signal with an FFT resampler.
pub fn resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    let mut resampler = Fft::<f32>::new(
        in_rate as usize,
        out_rate as usize,
        1024,
        1, // nbr_channels (mono)
        FixedSync::Both,
    )?;

    let input_adapter = InterleavedSlice::new(input, 1, input.len())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    let output = resampler.process_all(&input_adapter, input.len(), None)?;
    let mut out = output.take_data();
    let expected_out =
        ((input.len() as f64) * (out_rate as f64) / (in_rate as f64)).round() as usize;
    out.resize(expected_out, 0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_wav(path: &Path, samples: &[i16], sample_rate: u32) -> std::io::Result<()> {
        let data_len = (samples.len() * 2) as u32;
        let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(&buf)?;
        Ok(())
    }

    #[test]
    fn decodes_wav_at_native_rate() {
        let dir = std::env::temp_dir().join("subtitle-core-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("native-16k.wav");

        let samples: Vec<i16> = vec![0, 8192, -8192, 4096, -4096, 0];
        write_test_wav(&path, &samples, 16_000).unwrap();

        let buffer = AudioBuffer::load(&path).unwrap();
        assert_eq!(buffer.sample_rate(), 16_000);
        assert_eq!(buffer.samples().len(), samples.len());
        assert!((buffer.samples()[1] - 8192.0 / 32768.0).abs() < 1e-6);
        assert!((buffer.samples()[2] + 8192.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn resamples_to_target_rate_and_preserves_duration() {
        let dir = std::env::temp_dir().join("subtitle-core-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sine-8k.wav");

        // 1 second of 440 Hz sine at 8 kHz.
        let sample_rate = 8_000u32;
        let samples: Vec<i16> = (0..sample_rate)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 32767.0) as i16
            })
            .collect();
        write_test_wav(&path, &samples, sample_rate).unwrap();

        let buffer = AudioBuffer::load(&path).unwrap();
        assert_eq!(buffer.sample_rate(), 16_000);
        // 2x samples, allowing small tolerance for chunk rounding.
        let len = buffer.samples().len();
        assert!((len as i64 - 16_000).abs() < 64, "unexpected length {len}");
        assert!((buffer.duration() - 1.0).abs() < 0.01);
        // The sine should survive resampling with reasonable amplitude.
        let peak = buffer.samples().iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(peak > 0.4 && peak < 0.6, "unexpected peak {peak}");
    }

    #[test]
    fn slice_clamps_and_rounds() {
        let buffer = AudioBuffer::from_samples(vec![0.0; 16_000], 16_000);
        assert_eq!(buffer.slice(0.0, 0.5).len(), 8_000);
        assert_eq!(buffer.slice(-1.0, 0.25).len(), 4_000);
        assert_eq!(buffer.slice(0.9, 5.0).len(), 1_600);
        assert_eq!(buffer.slice(0.5, 0.5).len(), 0);
        // Reversed bounds collapse to an empty slice, never panic.
        assert_eq!(buffer.slice(0.9, 0.1).len(), 0);
    }

    #[test]
    fn slice_window_matches_slice() {
        let buffer = AudioBuffer::from_samples((0..32_000).map(|i| i as f32).collect(), 16_000);
        let window = AudioWindow::new(3, 0.25, 0.75);
        assert_eq!(buffer.slice_window(&window), buffer.slice(0.25, 0.75));
    }

    #[test]
    fn codec_registry_registers_opus_decoder() {
        use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
        let registry = get_codec_registry();
        assert!(
            registry.get_audio_decoder(CODEC_ID_OPUS).is_some(),
            "Opus decoder must be registered in codec registry"
        );
    }
}
