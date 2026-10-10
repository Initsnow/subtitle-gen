//! Smoke tool: render the same demo items the Python test suite uses, so the
//! Rust and Python outputs can be diffed byte for byte.
//!
//! Usage: `cargo run -p subtitle-core --example render_demo -- <outdir>`

use subtitle_core::formats::{write_lrc, write_subtitles, SubtitleMode};
use subtitle_core::types::SubtitleItem;

fn main() {
    let out_dir = std::env::args().nth(1).expect("usage: render_demo <outdir>");
    std::fs::create_dir_all(&out_dir).expect("create outdir");

    let items = vec![
        SubtitleItem::new(1, 0.0, 1.2, "Hello").with_translation(Some("你好".to_string())),
        SubtitleItem::new(2, 2.5, 4.0, "Second line,\nwith a break"),
        SubtitleItem::new(3, 5.0, 6.5, "こんにちは世界。"),
    ];

    let outputs = [
        ("demo.srt", SubtitleMode::Original),
        ("demo.bilingual.srt", SubtitleMode::Bilingual),
        ("demo.vtt", SubtitleMode::Original),
        ("demo.txt", SubtitleMode::Original),
        ("demo.json", SubtitleMode::Original),
    ];

    for (name, mode) in outputs {
        let path = write_subtitles(format!("{out_dir}/{name}"), &items, mode, false)
            .expect("write subtitles");
        println!("wrote {}", path.display());
    }
    let path = write_lrc(
        format!("{out_dir}/demo.lrc"),
        &items,
        Some(&["[ti:Title]".to_string()]),
        false,
    )
    .expect("write lrc");
    println!("wrote {}", path.display());
}
