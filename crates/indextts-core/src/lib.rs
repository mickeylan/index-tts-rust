//! Core data structures and error types for IndexTTS-2.5
//!
//! This crate defines the fundamental types used across the entire IndexTTS-Rs stack:
//! - Model configuration and generation parameters
//! - Speaker and emotion condition types
//! - Semantic codes representation
//! - Audio buffer types
//! - Unified error types

// ============================================================================
// Error Types
// ============================================================================

/// Unified error type for all IndexTTS operations
#[derive(Debug, thiserror::Error)]
pub enum IndexTtsError {
    /// Model loading failed
    #[error("Failed to load model: {0}")]
    InvalidModel(String),

    /// Invalid or empty text input
    #[error("Invalid text input: {0}")]
    InvalidText(String),

    /// Invalid reference audio
    #[error("Invalid reference audio: {0}")]
    InvalidReferenceAudio(String),

    /// ONNX or CUDA backend failure
    #[error("Backend failure: {0}")]
    BackendFailure(String),

    /// Generated empty semantic codes
    #[error("Empty semantic codes generated")]
    EmptySemanticCodes,

    /// Invalid audio output
    #[error("Invalid audio output: {0}")]
    InvalidAudio(String),

    /// Generation was cancelled
    #[error("Generation cancelled")]
    Cancelled,

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type alias for IndexTTS operations
pub type Result<T> = std::result::Result<T, IndexTtsError>;

// ============================================================================
// Model Configuration
// ============================================================================

/// Configuration for loading an IndexTTS model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Path to the model directory
    pub model_dir: std::path::PathBuf,
    /// Device to run inference on (e.g., "cuda:0", "cpu")
    pub device: DeviceConfig,
    /// Precision mode (float32, bfloat16)
    pub precision: Precision,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_dir: std::path::PathBuf::new(),
            device: DeviceConfig::default(),
            precision: Precision::default(),
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeviceConfig {
    /// Device type
    pub kind: DeviceKind,
    /// Device index (for multi-GPU)
    pub index: usize,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            kind: DeviceKind::Auto,
            index: 0,
        }
    }
}

impl DeviceConfig {
    /// Create a new device config
    pub fn new(kind: DeviceKind, index: usize) -> Self {
        Self { kind, index }
    }

    /// Resolve to actual device string
    pub fn resolve(&self) -> String {
        match self.kind {
            DeviceKind::Auto => "cuda:0".to_string(),
            DeviceKind::Cpu => "cpu".to_string(),
            DeviceKind::Cuda => format!("cuda:{}", self.index),
        }
    }
}

/// Device kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// Auto-detect (prefer CUDA if available)
    #[default]
    Auto,
    /// CPU only
    Cpu,
    /// CUDA GPU
    Cuda,
}

/// Precision mode for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 32-bit float
    #[default]
    Float32,
    /// 16-bit brain float
    BFloat16,
}

impl Precision {
    /// Check if this precision is half-precision
    pub fn is_half(&self) -> bool {
        matches!(self, Precision::BFloat16)
    }
}

// ============================================================================
// Generation Configuration
// ============================================================================

/// Configuration for speech generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationConfig {
    /// Text to synthesize
    pub text: String,
    /// Language code (ZH, EN, JA, ES, AR)
    pub language: Language,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Duration factor (0.5 to 2.0)
    pub duration_factor: f32,
    /// Maximum number of tokens to generate (None = model default)
    pub max_length: Option<usize>,
    /// Do not use sampling (greedy)
    pub do_sample: bool,
    /// Number of beams for beam search
    pub num_beams: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k for sampling
    pub top_k: usize,
    /// Top-p (nucleus) for sampling
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            text: String::new(),
            language: Language::Zh,
            seed: 0,
            duration_factor: 1.0,
            max_length: None,
            do_sample: false,
            num_beams: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.0,
        }
    }
}

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Chinese
    #[default]
    Zh,
    /// English
    En,
    /// Japanese
    Ja,
    /// Spanish
    Es,
    /// Arabic
    Ar,
}

impl Language {
    /// Get the language code string
    pub fn code(&self) -> &'static str {
        match self {
            Language::Zh => "ZH",
            Language::En => "EN",
            Language::Ja => "JA",
            Language::Es => "ES",
            Language::Ar => "AR",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "ZH" | "CN" | "CHINESE" => Some(Language::Zh),
            "EN" | "ENGLISH" => Some(Language::En),
            "JA" | "JAPANESE" => Some(Language::Ja),
            "ES" | "SPANISH" => Some(Language::Es),
            "AR" | "ARABIC" => Some(Language::Ar),
            _ => None,
        }
    }
}

// ============================================================================
// Speaker and Emotion Conditions
// ============================================================================

/// Speaker condition derived from reference audio
#[derive(Debug, Clone)]
pub struct SpeakerCondition {
    /// CAMPPlus embedding (192-dim)
    pub campplus_embedding: Vec<f32>,
    /// Conformer/Perceiver latent (32, 1024)
    pub latent: Vec<f32>,
}

impl SpeakerCondition {
    /// Create a new speaker condition
    pub fn new(campplus_embedding: Vec<f32>, latent: Vec<f32>) -> Self {
        Self {
            campplus_embedding,
            latent,
        }
    }

    /// Check if the condition is valid
    pub fn is_valid(&self) -> bool {
        !self.campplus_embedding.is_empty() && !self.latent.is_empty()
    }
}

/// Emotion condition derived from reference audio
#[derive(Debug, Clone)]
pub struct EmotionCondition {
    /// Emotion vector (1024-dim)
    pub emotion_vector: Vec<f32>,
}

impl EmotionCondition {
    /// Create a new emotion condition
    pub fn new(emotion_vector: Vec<f32>) -> Self {
        Self { emotion_vector }
    }
}

// ============================================================================
// Semantic Codes
// ============================================================================

/// Semantic tokens generated by the GPT model
#[derive(Debug, Clone)]
pub struct SemanticCodes {
    /// Token IDs (0-8191 are semantic codes, 8192=start, 8193=stop)
    pub tokens: Vec<u32>,
    /// Number of tokens
    pub len: usize,
}

impl SemanticCodes {
    /// Create new semantic codes from tokens
    pub fn new(tokens: Vec<u32>) -> Self {
        let len = tokens.len();
        Self { tokens, len }
    }

    /// Check if codes are empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the token at index
    pub fn get(&self, index: usize) -> Option<u32> {
        self.tokens.get(index).copied()
    }

    /// Iterate over tokens (excluding special tokens)
    pub fn semantic_tokens(&self) -> impl Iterator<Item = u32> + '_ {
        self.tokens.iter().copied().filter(|&t| t < 8192)
    }

    /// Find the stop token position
    pub fn find_stop_token(&self) -> Option<usize> {
        self.tokens.iter().position(|&t| t == 8193)
    }

    /// Get semantic tokens up to (but not including) stop token
    pub fn semantic_tokens_until_stop(&self) -> Vec<u32> {
        match self.find_stop_token() {
            Some(pos) => self.tokens[..pos]
                .iter()
                .copied()
                .filter(|&t| t < 8192)
                .collect(),
            None => self.tokens.iter().copied().filter(|&t| t < 8192).collect(),
        }
    }
}

/// Special token IDs
pub mod tokens {
    /// Start of mel sequence token
    pub const START_MEL_TOKEN: u32 = 8192;
    /// End of mel sequence token
    pub const STOP_MEL_TOKEN: u32 = 8193;
    /// Start of text sequence token
    pub const START_TEXT_TOKEN: u32 = 0;
    /// End of text sequence token
    pub const STOP_TEXT_TOKEN: u32 = 1;
    /// Number of mel codes (semantic vocab size)
    pub const NUM_MEL_CODES: u32 = 8194;
    /// Number of text tokens
    pub const NUM_TEXT_TOKENS: u32 = 256;
}

// ============================================================================
// Audio Buffer
// ============================================================================

/// Audio buffer containing synthesized speech
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples (mono, normalized to [-1, 1])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (always 1 for now)
    pub channels: u16,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    /// Get the duration in seconds
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Get the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Validate audio buffer
    pub fn validate(&self) -> Result<()> {
        if self.samples.is_empty() {
            return Err(IndexTtsError::InvalidAudio("Empty audio buffer".into()));
        }
        if self.sample_rate == 0 {
            return Err(IndexTtsError::InvalidAudio("Invalid sample rate".into()));
        }
        // Check for NaN/Inf
        for (i, &sample) in self.samples.iter().enumerate() {
            if !sample.is_finite() {
                return Err(IndexTtsError::InvalidAudio(format!(
                    "Non-finite sample at index {}: {}",
                    i, sample
                )));
            }
        }
        Ok(())
    }

    /// Convert to WAV bytes
    pub fn to_wav(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut buffer = Vec::new();
        {
            let spec = hound::WavSpec {
                channels: self.channels,
                sample_rate: self.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buffer), spec)
                .map_err(|e| IndexTtsError::InvalidAudio(e.to_string()))?;

            for sample in &self.samples {
                let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer
                    .write_sample(sample_i16)
                    .map_err(|e| IndexTtsError::InvalidAudio(e.to_string()))?;
            }
            writer
                .finalize()
                .map_err(|e| IndexTtsError::InvalidAudio(e.to_string()))?;
        }
        Ok(buffer)
    }
}

// ============================================================================
// Model Manifest
// ============================================================================

/// Model manifest containing metadata about the model package
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelManifest {
    /// Model version
    pub version: String,
    /// IndexTTS version this model was created from
    pub indextts_version: String,
    /// Original weight SHA-256 hashes
    pub weight_hashes: std::collections::HashMap<String, String>,
    /// Converter version
    pub converter_version: String,
    /// ONNX opset version
    pub onnx_opset: u32,
    /// Input/output tensor names
    pub tensor_info: TensorInfo,
    /// Sample rate
    pub sample_rate: u32,
    /// Semantic token frequency
    pub semantic_token_frequency: f32,
    /// License information
    pub license: String,
    /// Creation timestamp
    pub created_at: String,
}

/// Tensor information for ONNX models
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    /// Input tensor names and shapes
    pub inputs: std::collections::HashMap<String, TensorShape>,
    /// Output tensor names and shapes
    pub outputs: std::collections::HashMap<String, TensorShape>,
}

/// Tensor shape information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorShape {
    /// Shape dimensions (None for dynamic)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub dtype: String,
}

// ============================================================================
// Voice Profile
// ============================================================================

/// Voice profile for pre-configured voices
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceProfile {
    /// Unique voice identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Reference audio path
    pub reference_audio: String,
    /// Language
    pub language: Language,
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

// ============================================================================
// Re-exports
// ============================================================================

pub use tokens::*;
