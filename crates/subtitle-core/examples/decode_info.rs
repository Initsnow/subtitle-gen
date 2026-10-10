//! Smoke tool: decode a media file and print waveform stats.
//!
//! Usage: `cargo run -p subtitle-core --example decode_info -- <file>`

use subtitle_core::audio::AudioBuffer;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: decode_info <media-file>");
    match AudioBuffer::load(&path) {
        Ok(buffer) => {
            let peak = buffer.samples().iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            println!(
                "file: {}\nsample_rate: {}\nsamples: {}\nduration: {:.3}s\npeak: {:.4}",
                path,
                buffer.sample_rate(),
                buffer.samples().len(),
                buffer.duration(),
                peak
            );
        }
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}
