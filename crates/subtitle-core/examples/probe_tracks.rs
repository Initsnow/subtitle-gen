use std::fs::File;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let src = File::open(&path).unwrap();
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .unwrap();
    for t in probed.tracks() {
        match t.codec_params {
            Some(CodecParameters::Audio(ref audio)) => {
                println!(
                    "track {} audio codec={:?} rate={:?} channels={:?}",
                    t.id,
                    audio.codec,
                    audio.sample_rate,
                    audio.channels.as_ref().map(|c| c.count())
                );
                match subtitle_core::audio::get_codec_registry()
                    .make_audio_decoder(audio, &AudioDecoderOptions::default())
                {
                    Ok(_) => println!("  -> decoder OK"),
                    Err(e) => println!("  -> decoder ERR: {e}"),
                }
            }
            Some(ref other) => {
                println!("track {} non-audio={:?}", t.id, other);
            }
            None => {
                println!("track {} unknown params", t.id);
            }
        }
    }
}
