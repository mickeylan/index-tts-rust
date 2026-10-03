//! IndexTTS-2.5 CLI Tool
//!
//! Command-line interface for speech synthesis using IndexTTS-2.5.
//!
//! **Note**: This is a placeholder implementation.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use indextts_core::{GenerationConfig, Language, ModelConfig, DeviceConfig, Precision};
use indextts_audio::save_wav;
use indextts_pipeline::{IndexTtsPipeline, SemanticRuntime};
use std::path::PathBuf;
use tracing::info;

/// IndexTTS-2.5 Command Line Interface
#[derive(Parser)]
#[command(
    name = "indextts",
    about = "Pure Rust IndexTTS-2.5 inference (placeholder)",
    version,
    author
)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Log file directory
    #[arg(long, global = true)]
    log_dir: Option<PathBuf>,

    /// Command to execute
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Synthesize speech from text
    Synth {
        /// Model directory path
        #[arg(short, long)]
        model: PathBuf,

        /// Reference voice audio file
        #[arg(short, long)]
        voice: PathBuf,

        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Language code
        #[arg(short, long, default_value = "zh")]
        language: LanguageArg,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Random seed for reproducibility
        #[arg(long, default_value = "0")]
        seed: u64,

        /// Duration factor (0.5 - 2.0)
        #[arg(long, default_value = "1.0")]
        duration_factor: f32,

        /// Use sampling instead of greedy
        #[arg(long, default_value = "false")]
        do_sample: bool,

        /// Number of beams for beam search
        #[arg(long, default_value = "1")]
        num_beams: usize,

        /// Sampling temperature
        #[arg(long, default_value = "1.0")]
        temperature: f32,

        /// Top-k for sampling
        #[arg(long, default_value = "50")]
        top_k: usize,

        /// Top-p for sampling
        #[arg(long, default_value = "0.95")]
        top_p: f32,
    },

    /// Generate semantic tokens only (for debugging)
    Tokens {
        /// Model directory path
        #[arg(short, long)]
        model: PathBuf,

        /// Reference voice audio file
        #[arg(short, long)]
        voice: PathBuf,

        /// Text to tokenize
        #[arg(short, long)]
        text: String,

        /// Language code
        #[arg(short, long, default_value = "zh")]
        language: LanguageArg,

        /// Output file for tokens (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Print version information
    Version,
}

#[derive(Clone, ValueEnum, Debug)]
enum LanguageArg {
    Zh,
    En,
    Ja,
    Es,
    Ar,
}

impl From<LanguageArg> for Language {
    fn from(arg: LanguageArg) -> Self {
        match arg {
            LanguageArg::Zh => Language::Zh,
            LanguageArg::En => Language::En,
            LanguageArg::Ja => Language::Ja,
            LanguageArg::Es => Language::Es,
            LanguageArg::Ar => Language::Ar,
        }
    }
}

fn setup_logging(_verbose: bool, _log_dir: Option<PathBuf>) {
    // Placeholder: initialize tracing
    let _ = tracing_subscriber::fmt::try_init();
}

fn cmd_synth(
    model: PathBuf,
    voice: PathBuf,
    text: String,
    language: LanguageArg,
    output: Option<PathBuf>,
    seed: u64,
    duration_factor: f32,
    do_sample: bool,
    num_beams: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    info!("Synthesizing: {}", text);
    info!("Language: {:?}", language);
    info!("Seed: {}", seed);

    // Create pipeline
    let config = ModelConfig {
        model_dir: model,
        device: DeviceConfig::default(),
        precision: Precision::default(),
    };

    let mut pipeline = IndexTtsPipeline::new(config);
    pipeline.load()?;

    // Create generation config
    let gen_config = GenerationConfig {
        text: text.clone(),
        language: language.into(),
        seed,
        duration_factor,
        do_sample,
        num_beams,
        temperature,
        top_k,
        top_p,
        repetition_penalty: 1.0,
        max_length: None,
    };

    let audio = pipeline.synthesize(&text, &voice, &gen_config)?;
    info!("Generated {} samples ({}s)", audio.len(), audio.duration());

    let output_path = output.unwrap_or_else(|| {
        let stem = sanitize_filename(&text);
        PathBuf::from(format!("{}.wav", stem))
    });

    save_wav(&output_path, &audio)?;
    info!("Saved output to: {:?}", output_path);
    println!("{}", output_path.display());
    Ok(())
}

fn cmd_tokens(
    model: PathBuf,
    voice: PathBuf,
    text: String,
    language: LanguageArg,
    output: Option<PathBuf>,
) -> Result<()> {
    let language: Language = language.into();
    let mut runtime = SemanticRuntime::load(&model)?;
    let codes = runtime.generate(&text, language, &voice, 1815)?;
    let json = serde_json::to_string_pretty(&codes.tokens)?;
    if let Some(path) = output {
        std::fs::write(&path, format!("{json}\n"))?;
        info!("Saved semantic codes to: {:?}", path);
    } else {
        println!("{json}");
    }
    Ok(())
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .take(50)
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    setup_logging(cli.verbose, cli.log_dir);

    match cli.command {
        Commands::Synth {
            model,
            voice,
            text,
            language,
            output,
            seed,
            duration_factor,
            do_sample,
            num_beams,
            temperature,
            top_k,
            top_p,
        } => {
            cmd_synth(
                model, voice, text, language, output,
                seed, duration_factor, do_sample, num_beams,
                temperature, top_k, top_p,
            )?;
        }
        Commands::Tokens {
            model,
            voice,
            text,
            language,
            output,
        } => {
            cmd_tokens(model, voice, text, language, output)?;
        }
        Commands::Version => {
            println!("indextts {}", env!("CARGO_PKG_VERSION"));
            println!("IndexTTS-2.5 Rust runtime");
        }
    }

    Ok(())
}
