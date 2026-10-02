//! IndexTTS-2.5 GPT semantic token generation
//!
//! Implements the core GPT model for generating semantic tokens from
//! text and speaker conditions.
//!
//! **Note**: This is a placeholder implementation. Full GPT model
//! requires careful mapping of Candle API with the actual model weights.

pub mod config;
pub mod model;
pub mod generation;
pub mod embedding;
pub mod cache;

pub use config::{GptConfig, GptModelType};
pub use model::IndexGpt;
pub use generation::{Generator, GreedyGenerator, SamplingGenerator, BeamGenerator};
pub use cache::KvCache;
