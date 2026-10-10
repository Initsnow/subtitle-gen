//! Global error type for `subtitle-core`.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    /// The input container has no track decodable by the bundled codecs.
    #[error("no decodable audio track found in input")]
    NoAudioTrack,
    /// Failure while demuxing or decoding a media container.
    #[error("audio decode error: {0}")]
    Decode(#[from] symphonia::core::errors::Error),
    /// Failure inside the resampler.
    #[error("resampling error: {0}")]
    Resample(#[from] rubato::ResampleError),
    /// Failure while constructing the resampler.
    #[error("resampler construction error: {0}")]
    ResamplerInit(#[from] rubato::ResamplerConstructionError),
    /// Malformed subtitle file or unsupported format.
    #[error("subtitle format error: {0}")]
    Format(#[from] crate::formats::FormatError),
    /// LLM completion or parsing error.
    #[error("llm error: {0}")]
    Llm(String),
    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),
    /// Pipeline orchestration error.
    #[error("pipeline error: {0}")]
    Pipeline(String),
    /// Underlying filesystem error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
