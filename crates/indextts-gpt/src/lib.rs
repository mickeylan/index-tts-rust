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

pub mod config;
pub mod model;
pub mod attention;
pub mod generation;
pub mod embedding;
pub mod cache;
pub mod weights;

pub use config::{GptConfig, GptModelType};
pub use model::IndexGpt;
pub use attention::{Gpt2Attention, Gpt2Block, Gpt2MLP};
pub use generation::{Generator, GreedyGenerator, SamplingGenerator, BeamGenerator};
pub use cache::KvCache;
pub use weights::{Weights, LoadedWeights, WeightError};
