//! subtitle-cli: High-performance CLI for subtitle generation, alignment,
//! proofreading, and translation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use console::{style, Term};
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

fn wrap_lines(text: &str, max_width: usize, max_lines: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut current_width = 0;

    for ch in text.chars() {
        if ch == '\r' {
            continue;
        }
        if ch == '\n' {
            lines.push(current_line);
            current_line = String::new();
            current_width = 0;
            continue;
        }
        let ch_width = console::measure_text_width(&ch.to_string());
        if current_width + ch_width > max_width && !current_line.is_empty() {
            lines.push(current_line);
            current_line = String::new();
            current_width = 0;
        }
        current_line.push(ch);
        current_width += ch_width;
    }
    if !current_line.is_empty() || lines.is_empty() {
        lines.push(current_line);
    }

    if lines.len() > max_lines {
        lines.split_off(lines.len() - max_lines)
    } else {
        lines
    }
}

pub struct StreamDisplay {
    spinner: ProgressBar,
    state: Arc<std::sync::Mutex<StreamState>>,
}

struct StreamState {
    label: String,
    content: String,
    token_count: usize,
    start_time: std::time::Instant,
    last_render_time: std::time::Instant,
    rendered_lines: usize,
    active: bool,
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
                last_render_time: now,
                rendered_lines: 0,
                active: false,
            })),
        }
    }

    pub fn start_batch(&self, label: &str) {
        if let Ok(mut s) = self.state.lock() {
            let term = Term::stderr();
            if s.active && s.rendered_lines > 0 {
                s.render(&term);
                s.rendered_lines = 0;
            }

            self.spinner.disable_steady_tick();
            self.spinner.finish_and_clear();

            s.label = label.to_string();
            s.content.clear();
            s.token_count = 0;
            s.start_time = std::time::Instant::now();
            s.last_render_time = s.start_time;
            s.rendered_lines = 0;
            s.active = true;
            s.render(&term);
        }
    }

    pub fn finish(&self) {
        if let Ok(mut s) = self.state.lock() {
            if s.active {
                let term = Term::stderr();
                s.render(&term);
                s.rendered_lines = 0;
                s.active = false;
            }
        }
        self.spinner.reset();
        self.spinner
            .enable_steady_tick(std::time::Duration::from_millis(100));
    }

    pub fn callback(self: &Arc<Self>, default_label: &'static str) -> StreamCallback {
        let display = Arc::clone(self);
        let default_label = default_label.to_string();
        Arc::new(move |token: &str| {
            if let Ok(mut s) = display.state.lock() {
                if !s.active {
                    display.spinner.disable_steady_tick();
                    display.spinner.finish_and_clear();
                    s.label = default_label.clone();
                    s.content.clear();
                    s.token_count = 0;
                    s.start_time = std::time::Instant::now();
                    s.last_render_time = s.start_time;
                    s.rendered_lines = 0;
                    s.active = true;
                }

                s.content.push_str(token);
                s.token_count += 1;

                let now = std::time::Instant::now();
                if now.duration_since(s.last_render_time) >= std::time::Duration::from_millis(40) {
                    let term = Term::stderr();
                    s.render(&term);
                    s.last_render_time = now;
                }
            }
        })
    }
}

impl StreamState {
    fn render(&mut self, term: &Term) {
        if !term.is_term() {
            return;
        }

        let (_, cols) = term.size();
        let width = (cols as usize).clamp(40, 100);
        let inner_width = width.saturating_sub(4);

        let display_text = if self.content.chars().count() > 800 {
            let count = self.content.chars().count();
            let tail: String = self
                .content
                .chars()
                .skip(count.saturating_sub(799))
                .collect();
            format!("…{tail}")
        } else {
            self.content.clone()
        };

        let content_lines = wrap_lines(&display_text, inner_width, 5);
        let mut out = Vec::with_capacity(content_lines.len() + 2);

        // Top border
        let title = if self.label.is_empty() {
            " LLM ".to_string()
        } else {
            format!(" LLM · {} ", self.label)
        };
        let max_title_w = width.saturating_sub(6);
        let title = if console::measure_text_width(&title) > max_title_w {
            console::truncate_str(&title, max_title_w, "…").to_string()
        } else {
            title
        };
        let title_w = console::measure_text_width(&title);
        let dashes = width.saturating_sub(title_w + 3);
        out.push(format!(
            "{}{}{}{}",
            style("╭─").cyan(),
            style(&title).cyan().bold(),
            style("─".repeat(dashes)).cyan(),
            style("╮").cyan()
        ));

        // Content lines
        for line in &content_lines {
            let line_w = console::measure_text_width(line);
            let pad = inner_width.saturating_sub(line_w);
            out.push(format!(
                "{} {}{}{}",
                style("│").cyan(),
                line,
                " ".repeat(pad),
                style(" │").cyan()
            ));
        }

        // Bottom border
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let footer = if self.token_count == 0 {
            " waiting for response... ".to_string()
        } else {
            format!(" {} tokens · {:.1}s ", self.token_count, elapsed)
        };
        let max_footer_w = width.saturating_sub(6);
        let footer = if console::measure_text_width(&footer) > max_footer_w {
            console::truncate_str(&footer, max_footer_w, "…").to_string()
        } else {
            footer
        };
        let footer_w = console::measure_text_width(&footer);
        let dashes = width.saturating_sub(footer_w + 3);
        out.push(format!(
            "{}{}{}{}",
            style("╰─").cyan(),
            style(&footer).dim(),
            style("─".repeat(dashes)).cyan(),
            style("╯").cyan()
        ));

        if self.rendered_lines > 0 {
            let _ = term.clear_last_lines(self.rendered_lines);
        }
        for line in &out {
            let _ = term.write_line(line);
        }
        self.rendered_lines = out.len();
    }
}

type ProgressReporter = (
    ProgressBar,
    Arc<dyn Fn(&str) + Send + Sync>,
    Arc<StreamDisplay>,
);

fn create_progress_reporter() -> ProgressReporter {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .expect("valid template"),
    );
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let stream_display = Arc::new(StreamDisplay::new(spinner.clone()));
    let sp = spinner.clone();
    let sd = Arc::clone(&stream_display);
    let cb = Arc::new(move |msg: &str| {
        if msg.starts_with("proofreading batch ") || msg.starts_with("translating batch ") {
            sd.start_batch(msg);
        } else {
            sp.set_message(msg.to_string());
        }
    });

    (spinner, cb, stream_display)
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

    // When only TXT output is requested (no timestamps required) and the user didn't explicitly
    // specify a segment_mode, default to "none" so forced alignment can be skipped.
    if args.segment_mode.is_none() {
        let only_txt = if let Some(ref out_path) = args.out {
            out_path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        } else if !args.formats.is_empty() {
            args.formats.iter().all(|f| f.eq_ignore_ascii_case("txt"))
        } else {
            config
                .output
                .default_formats
                .iter()
                .all(|f| f.eq_ignore_ascii_case("txt"))
        };
        if only_txt {
            config.segment.mode = "none".to_string();
        }
    }

    let (spinner, progress, stream_display) = create_progress_reporter();

    // Initialize inference engine: only load forced aligner if segmentation needs it
    progress("loading inference models");
    let load_opts = LoadOptions {
        n_threads: config.model.n_threads as i32,
        ladder: None,
    };

    let needs_aligner = config.segment.mode != "none";
    let aligner_spec = if needs_aligner {
        Some(config.model.aligner_model.as_str())
    } else {
        None
    };

    let backend = match InferenceEngine::load(
        None,
        Some(&config.model.asr_model),
        aligner_spec,
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
            segment_mode: Some(config.segment.mode.clone()),
            overwrite_cache: args.overwrite_cache,
            progress: Some(Arc::clone(&progress)),
            stream_token: Some(stream_display.callback("LLM")),
        };

        match pipeline.run(options).await {
            Ok(result) => {
                stream_display.finish();
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
                stream_display.finish();
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
    let (spinner, progress, _stream_display) = create_progress_reporter();

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

    let (spinner, progress, stream_display) = create_progress_reporter();
    progress(&format!("proofreading {} cues with LLM", items.len()));

    let llm = SubtitleLlm::new(config.llm.clone());
    let proofread_items = llm
        .proofread(
            &items,
            Some(stream_display.callback("LLM proofread")),
            Some(Arc::clone(&progress)),
        )
        .await?;
    stream_display.finish();

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

    let (spinner, progress, stream_display) = create_progress_reporter();
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
            Some(Arc::clone(&progress)),
        )
        .await?;
    stream_display.finish();

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
