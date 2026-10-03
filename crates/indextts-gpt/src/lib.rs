//! IndexTTS-2.5 GPT semantic token generation
//!
//! Implements the core GPT model for generating semantic tokens from
//! text and speaker conditions.
//!
//! ## Architecture
//!
//! The IndexTTS-2.5 GPT model is based on GPT-2 architecture with:
//! - 24 transformer layers
//! - 20 attention heads
//! - 1280 model dimension
//! - GELU New activation
//! - Learned position embeddings
//!
//! ## Modules
//!
//! - `config`: Model configuration
//! - `model`: Main GPT model implementation
//! - `attention`: GPT2 attention and MLP layers
//! - `generation`: Token generation strategies
//! - `embedding`: Input embedding layers
//! - `cache`: KV cache management

pub mod attention;
pub mod cache;
pub mod config;
pub mod embedding;
pub mod generation;
pub mod model;
pub mod weights;

pub use attention::{Gpt2Attention, Gpt2Block, Gpt2MLP};
pub use cache::KvCache;
pub use config::{GptConfig, GptModelType};
pub use generation::{BeamGenerator, Generator, GreedyGenerator, SamplingGenerator};
pub use model::IndexGpt;
pub use weights::{LoadedWeights, WeightError, Weights};
