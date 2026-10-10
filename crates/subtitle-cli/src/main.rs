//! subtitle-cli: High-performance CLI for subtitle generation, alignment,
//! proofreading, and translation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::EnvFilter;

use subtitle_core::config::{load_config, AppConfig};
use subtitle_core::formats::{
    load_text_input, parse_srt, write_lrc, write_output_set, write_subtitles, SubtitleFormat,
    SubtitleMode,
};
use subtitle_core::llm::{StreamCallback, SubtitleLlm};
use subtitle_core::pipeline::{PipelineOptions, SubtitlePipeline};
use subtitle_core::types::SubtitleItem;
use subtitle_infer::session::LoadOptions;
use subtitle_infer::InferenceEngine;
#[derive(Debug, Parser)]
#[command(
    name = "subtitle-gen",
    version,
    about = "Generate subtitles from audio or video files with GGUF ASR, forced alignment, and local text segmentation.",
    after_help = "Subcommands:\n  align         Add timestamps to untimed text/LRC against audio\n  proofread     Proofread an existing SRT subtitle file\n  translate     Translate an existing SRT subtitle file"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    pipeline: PipelineArgs,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Add timestamps to untimed text or LRC against audio
    Align(AlignArgs),
    /// Proofread an existing SRT subtitle file using an LLM
    Proofread(ProofreadArgs),
    /// Translate an existing SRT subtitle file using an LLM
    Translate(TranslateArgs),
}

#[derive(Debug, Args, Clone)]
struct PipelineArgs {
    /// Input audio/video file(s)
    #[arg(required = false)]
    input: Vec<PathBuf>,

    /// TOML config file path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Single output file path
    #[arg(long)]
    out: Option<PathBuf>,

    /// Directory for output subtitle set
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Output format for --out-dir (srt, vtt, lrc, json, txt)
    #[arg(long = "format", action = clap::ArgAction::Append)]
    formats: Vec<String>,

    /// Subtitle segmentation mode (none, local)
    #[arg(long)]
    segment_mode: Option<String>,

    /// Target language for translation
    #[arg(long)]
    translate: Option<String>,

    /// Also write a bilingual subtitle file alongside translation
    #[arg(long)]
    bilingual: bool,

    /// Override ASR model ID
    #[arg(long)]
    asr_model: Option<String>,

    /// Override forced-aligner model ID
    #[arg(long)]
    aligner_model: Option<String>,

    /// Quantization for Hugging Face repo-id models
    #[arg(long)]
    quant: Option<String>,

    /// CPU threads (0 = auto)
    #[arg(long)]
    n_threads: Option<usize>,

    /// Layers offloaded to the GPU (0 = CPU only)
    #[arg(long)]
    gpu_layers: Option<i32>,

    /// Source language hint
    #[arg(long)]
    language: Option<String>,

    /// Enable persistent cache
    #[arg(long, overrides_with = "no_cache")]
    cache: bool,

    /// Disable persistent cache
    #[arg(long, overrides_with = "cache")]
    no_cache: bool,

    /// Cache directory path
    #[arg(long)]
    cache_dir: Option<String>,

    /// Regenerate cached audio chunks and transcripts
    #[arg(long)]
    overwrite_cache: bool,
}

#[derive(Debug, Args)]
struct AlignArgs {
    /// Input audio or video file
    audio: PathBuf,

    /// Input untimed text or LRC file
    text: PathBuf,

    /// TOML config file path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Output subtitle file path
    #[arg(long)]
    out: Option<PathBuf>,

    /// Language hint for alignment
    #[arg(long)]
    language: Option<String>,

    /// Perform precise second-pass boundary refinement
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    refine: bool,

    /// Use VAD windowing (speech) vs fixed windows (songs)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    vad: bool,

    /// Strip punctuation from end of lines
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    strip_punctuation: bool,
}

#[derive(Debug, Args)]
struct ProofreadArgs {
    /// Input SRT file
    input: PathBuf,

    /// Output SRT file path
    #[arg(long)]
    out: Option<PathBuf>,

    /// TOML config file path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override LLM model
    #[arg(long)]
    llm_model: Option<String>,
}

#[derive(Debug, Args)]
struct TranslateArgs {
    /// Input SRT file
    input: PathBuf,

    /// Target language
    #[arg(long)]
    target: String,

    /// Output SRT file path
    #[arg(long)]
    out: Option<PathBuf>,

    /// Write bilingual subtitle file (original + translation)
    #[arg(long)]
    bilingual: bool,

    /// TOML config file path
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override LLM model
    #[arg(long)]
    llm_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Align(args)) => cmd_align(args).await,
        Some(Commands::Proofread(args)) => cmd_proofread(args).await,
        Some(Commands::Translate(args)) => cmd_translate(args).await,
        None => cmd_pipeline(cli.pipeline).await,
    }
}

#[derive(Clone)]
pub struct StreamDisplay {
    spinner: ProgressBar,
    state: Arc<std::sync::Mutex<StreamState>>,
}

struct StreamState {
    label: String,
    content: String,
    token_count: usize,
    start_time: std::time::Instant,
    last_token_time: std::time::Instant,
}

impl StreamDisplay {
    pub fn new(spinner: ProgressBar) -> Self {
        let now = std::time::Instant::now();
        Self {
            spinner,
            state: Arc::new(std::sync::Mutex::new(StreamState {
                label: String::new(),
                content: String::new(),
                token_count: 0,
                start_time: now,
                last_token_time: now,
            })),
        }
    }

    pub fn callback(&self, default_label: &'static str) -> StreamCallback {
        let spinner = self.spinner.clone();
        let state = Arc::clone(&self.state);
        Arc::new(move |token: &str| {
            if let Ok(mut s) = state.lock() {
                let now = std::time::Instant::now();
                if now.duration_since(s.last_token_time) > std::time::Duration::from_millis(1500) {
                    s.content.clear();
                    s.token_count = 0;
                    s.start_time = now;
                }
                s.last_token_time = now;
                s.content.push_str(token);
                s.token_count += 1;

                let elapsed = s.start_time.elapsed().as_secs_f32();
                let char_count = s.content.chars().count();
                let tail = if char_count > 60 {
                    let snippet: String = s.content.chars().skip(char_count - 57).collect();
                    format!("...{snippet}")
                } else {
                    s.content.clone()
                };
                let clean = tail.replace(['\r', '\n'], " ");
                let label = if s.label.is_empty() {
                    default_label
                } else {
                    &s.label
                };
                spinner.set_message(format!(
                    "{} [{} toks, {:.1}s]: {}",
                    label, s.token_count, elapsed, clean
                ));
            }
        })
    }
}

type ProgressReporter = (ProgressBar, Arc<dyn Fn(&str) + Send + Sync>);

fn create_progress_reporter() -> ProgressReporter {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .expect("valid template"),
    );
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let sp = spinner.clone();
    let cb = Arc::new(move |msg: &str| {
        sp.set_message(msg.to_string());
    });

    (spinner, cb)
}

async fn cmd_pipeline(args: PipelineArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.input.is_empty() {
        eprintln!(
            "subtitle-gen: error: at least one input file is required. Use --help for usage."
        );
        std::process::exit(1);
    }

    if args.out.is_some() && args.input.len() > 1 {
        eprintln!("subtitle-gen: error: --out requires exactly one input file; use --out-dir for batches.");
        std::process::exit(1);
    }

    let mut config = load_config(args.config.as_deref())?;
    apply_pipeline_overrides(&mut config, &args);

    let (spinner, progress) = create_progress_reporter();
    let stream_display = StreamDisplay::new(spinner.clone());

    // Initialize inference engine
    progress("loading inference models");
    let load_opts = LoadOptions {
        n_threads: config.model.n_threads as i32,
        ladder: None,
    };

    let backend = match InferenceEngine::load(
        None,
        Some(&config.model.asr_model),
        Some(&config.model.aligner_model),
        &load_opts,
    ) {
        Ok(engine) => Arc::new(engine),
        Err(e) => {
            spinner.finish_and_clear();
            eprintln!("subtitle-gen: inference initialization failed: {e}");
            std::process::exit(1);
        }
    };

    let pipeline = SubtitlePipeline::new(config.clone(), backend);
    let mut failures = Vec::new();
    let total_inputs = args.input.len();

    for (idx, input_path) in args.input.iter().enumerate() {
        if total_inputs > 1 {
            progress(&format!(
                "processing file {}/{}: {}",
                idx + 1,
                total_inputs,
                input_path.display()
            ));
        }

        let options = PipelineOptions {
            input_path: input_path.clone(),
            translate: args.translate.clone(),
            segment_mode: args.segment_mode.clone(),
            overwrite_cache: args.overwrite_cache,
            progress: Some(Arc::clone(&progress)),
            stream_token: Some(stream_display.callback("LLM")),
        };

        match pipeline.run(options).await {
            Ok(result) => {
                progress("writing subtitle output");
                let written =
                    write_pipeline_outputs(&args, input_path, &result.subtitles, &config)?;
                spinner.suspend(|| {
                    for p in written {
                        println!("{}", p.display());
                    }
                });
            }
            Err(e) => {
                eprintln!("subtitle-gen: {}: {e}", input_path.display());
                failures.push(input_path.clone());
            }
        }
    }

    spinner.finish_and_clear();

    if !failures.is_empty() {
        eprintln!(
            "subtitle-gen: {}/{} input(s) failed",
            failures.len(),
            total_inputs
        );
        std::process::exit(1);
    }

    Ok(())
}

fn apply_pipeline_overrides(config: &mut AppConfig, args: &PipelineArgs) {
    if let Some(ref m) = args.asr_model {
        config.model.asr_model = m.clone();
    }
    if let Some(ref m) = args.aligner_model {
        config.model.aligner_model = m.clone();
    }
    if let Some(ref q) = args.quant {
        config.model.quant = q.clone();
    }
    if let Some(n) = args.n_threads {
        config.model.n_threads = n;
    }
    if let Some(l) = args.gpu_layers {
        config.model.gpu_layers = Some(l);
    }
    if let Some(ref lang) = args.language {
        config.model.language = Some(lang.clone());
    }
    if args.cache {
        config.cache.enabled = true;
    } else if args.no_cache {
        config.cache.enabled = false;
    }
    if let Some(ref dir) = args.cache_dir {
        config.cache_dir = dir.clone();
    }
    if let Some(ref mode) = args.segment_mode {
        config.segment.mode = mode.clone();
    }
}

fn write_pipeline_outputs(
    args: &PipelineArgs,
    input_path: &Path,
    subtitles: &[SubtitleItem],
    config: &AppConfig,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let strip = config.output.strip_punctuation;
    let has_translation = subtitles.iter().any(|s| s.translation.is_some());
    let has_proofread = subtitles.iter().any(|s| s.proofread.is_some());

    if let Some(ref out_path) = args.out {
        let mode = if has_translation {
            SubtitleMode::Translation
        } else if has_proofread {
            SubtitleMode::Proofread
        } else {
            SubtitleMode::Original
        };
        let p = write_subtitles(out_path, subtitles, mode, strip)?;
        let mut written = vec![p];
        if args.bilingual && has_translation {
            let bi_path = bilingual_out_path(out_path);
            let bp = write_subtitles(&bi_path, subtitles, SubtitleMode::Bilingual, strip)?;
            written.push(bp);
        }
        return Ok(written);
    }

    let out_dir = args
        .out_dir
        .clone()
        .unwrap_or_else(|| input_path.parent().unwrap_or(Path::new("")).to_path_buf());

    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("subtitles");

    let formats = if !args.formats.is_empty() {
        args.formats
            .iter()
            .filter_map(|f| match f.to_ascii_lowercase().as_str() {
                "srt" => Some(SubtitleFormat::Srt),
                "vtt" => Some(SubtitleFormat::Vtt),
                "json" => Some(SubtitleFormat::Json),
                "txt" => Some(SubtitleFormat::Txt),
                _ => None,
            })
            .collect::<Vec<_>>()
    } else {
        vec![SubtitleFormat::Srt]
    };

    let written = write_output_set(
        &out_dir,
        stem,
        subtitles,
        &formats,
        has_translation,
        args.bilingual && has_translation,
        has_proofread,
        strip,
    )?;

    Ok(written)
}

fn bilingual_out_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or(Path::new(""));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("subtitles");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("srt");
    parent.join(format!("{stem}.bilingual.{ext}"))
}

async fn cmd_align(args: AlignArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (metadata, lines) = load_text_input(&args.text)?;
    if lines.is_empty() {
        eprintln!("subtitle-gen align: no text lines found in input file.");
        std::process::exit(1);
    }

    let config = load_config(args.config.as_deref())?;
    let (spinner, progress) = create_progress_reporter();

    progress("loading inference models");
    let load_opts = LoadOptions {
        n_threads: config.model.n_threads as i32,
        ladder: None,
    };

    let backend = match InferenceEngine::load(
        None,
        Some(&config.model.asr_model),
        Some(&config.model.aligner_model),
        &load_opts,
    ) {
        Ok(engine) => Arc::new(engine),
        Err(e) => {
            spinner.finish_and_clear();
            eprintln!("subtitle-gen align: inference initialization failed: {e}");
            std::process::exit(1);
        }
    };

    let pipeline = SubtitlePipeline::new(config.clone(), backend);
    let items = pipeline
        .align_lines_to_audio(
            &args.audio,
            &lines,
            args.language.as_deref(),
            args.refine,
            args.vad,
            Some(Arc::clone(&progress)),
            false,
        )
        .await?;

    let out_path = args.out.unwrap_or_else(|| {
        let ext = args
            .text
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("srt");
        args.audio.with_extension(ext)
    });

    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "lrc" {
        write_lrc(&out_path, &items, Some(&metadata), args.strip_punctuation)?;
    } else {
        write_subtitles(
            &out_path,
            &items,
            SubtitleMode::Original,
            args.strip_punctuation,
        )?;
    }

    spinner.finish_and_clear();
    println!("{}", out_path.display());
    Ok(())
}

async fn cmd_proofread(args: ProofreadArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = load_config(args.config.as_deref())?;
    if let Some(m) = args.llm_model {
        config.llm.model = Some(m);
    }

    let items = parse_srt(&args.input)?;
    if items.is_empty() {
        eprintln!("subtitle-gen proofread: no subtitle cues found in input SRT.");
        std::process::exit(1);
    }

    let (spinner, progress) = create_progress_reporter();
    let stream_display = StreamDisplay::new(spinner.clone());
    progress(&format!("proofreading {} cues with LLM", items.len()));

    let llm = SubtitleLlm::new(config.llm.clone());
    let proofread_items = llm
        .proofread(&items, Some(stream_display.callback("LLM proofread")))
        .await?;

    let out_path = args.out.unwrap_or_else(|| {
        let stem = args
            .input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("subtitles");
        args.input.with_file_name(format!("{stem}.proofread.srt"))
    });

    write_subtitles(
        &out_path,
        &proofread_items,
        SubtitleMode::Proofread,
        config.output.strip_punctuation,
    )?;
    spinner.finish_and_clear();
    println!("{}", out_path.display());
    Ok(())
}

async fn cmd_translate(args: TranslateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = load_config(args.config.as_deref())?;
    if let Some(m) = args.llm_model {
        config.llm.model = Some(m);
    }

    let items = parse_srt(&args.input)?;
    if items.is_empty() {
        eprintln!("subtitle-gen translate: no subtitle cues found in input SRT.");
        std::process::exit(1);
    }

    let (spinner, progress) = create_progress_reporter();
    let stream_display = StreamDisplay::new(spinner.clone());
    progress(&format!(
        "translating {} cues to {} with LLM",
        items.len(),
        args.target
    ));

    let llm = SubtitleLlm::new(config.llm.clone());
    let translated_items = llm
        .translate(
            &items,
            &args.target,
            Some(stream_display.callback("LLM translate")),
        )
        .await?;

    let target_suffix = args.target.to_ascii_lowercase().replace(' ', "-");
    let out_path = args.out.unwrap_or_else(|| {
        let stem = args
            .input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("subtitles");
        args.input
            .with_file_name(format!("{stem}.{target_suffix}.srt"))
    });

    write_subtitles(
        &out_path,
        &translated_items,
        SubtitleMode::Translation,
        config.output.strip_punctuation,
    )?;
    if args.bilingual {
        let bi_path = bilingual_out_path(&out_path);
        write_subtitles(
            &bi_path,
            &translated_items,
            SubtitleMode::Bilingual,
            config.output.strip_punctuation,
        )?;
        println!("{}", bi_path.display());
    }

    spinner.finish_and_clear();
    println!("{}", out_path.display());
    Ok(())
}
