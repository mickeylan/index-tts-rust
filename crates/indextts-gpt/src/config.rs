//! GPT configuration for IndexTTS-2.5
//!
//! Based on the official IndexTTS-2.5 config.yaml and Python implementation.

use serde::{Deserialize, Serialize};

/// GPT model configuration (matching IndexTTS-2.5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    /// Model dimension (hidden size) - 1280
    pub n_embd: usize,
    /// Number of attention heads - 20
    pub n_head: usize,
    /// Number of layers - 24
    pub n_layer: usize,
    /// Vocabulary size (mel codes) - 8194
    pub n_vocab: usize,
    /// Maximum position embeddings
    pub n_positions: usize,
    /// Context length
    pub n_ctx: usize,
    /// Inner dimension for FFN - 5120
    pub n_inner: usize,
    /// Activation function
    pub activation_function: String,
    /// Dropout probability
    pub resid_dropout: f32,
    /// Attention dropout
    pub attn_dropout: f32,
    /// Maximum mel tokens - 1815
    pub max_mel_tokens: usize,
    /// Maximum text tokens - 600
    pub max_text_tokens: usize,
    /// Number of conditioning latents (perceiver) - 32
    pub cond_num_latent: usize,
    /// Number of text tokens - 60509
    pub number_text_tokens: usize,
    /// Start mel token ID - 8192
    pub start_mel_token: u32,
    /// Stop mel token ID - 8193
    pub stop_mel_token: u32,
    /// Start text token ID - 0
    pub start_text_token: u32,
    /// Stop text token ID - 1
    pub stop_text_token: u32,
    /// Number of mel codes - 8194
    pub num_mel_codes: u32,
    /// Use mel codes as input - true
    pub use_mel_codes_as_input: bool,
    /// Mel length compression - 1024
    pub mel_length_compression: usize,
    /// Condition type - "conformer_perceiver"
    pub condition_type: String,
    /// Speaker condition mode - "campplus"
    pub spk_cond_mode: String,
}

impl Default for GptConfig {
    fn default() -> Self {
        // IndexTTS-2.5 default configuration from config.yaml
        Self {
            n_embd: 1280,
            n_head: 20,
            n_layer: 24,
            n_vocab: 8194,
            n_positions: 2048,
            n_ctx: 2048,
            n_inner: 5120, // 4 * n_embd
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            attn_dropout: 0.0,
            max_mel_tokens: 1815,
            max_text_tokens: 600,
            cond_num_latent: 32,
            number_text_tokens: 60509,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            start_text_token: 0,
            stop_text_token: 1,
            num_mel_codes: 8194,
            use_mel_codes_as_input: true,
            mel_length_compression: 1024,
            condition_type: "conformer_perceiver".to_string(),
            spk_cond_mode: "campplus".to_string(),
        }
    }
}

impl GptConfig {
    /// Load config from YAML file (simplified parsing)
    pub fn from_yaml(path: &std::path::Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        
        let mut config = Self::default();
        
        for line in content.lines() {
            let line = line.trim();
            
            if let Some(rest) = line.strip_prefix("model_dim:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.n_embd = val;
                    config.n_inner = val * 4;
                }
            } else if let Some(rest) = line.strip_prefix("max_mel_tokens:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.max_mel_tokens = val;
                }
            } else if let Some(rest) = line.strip_prefix("max_text_tokens:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.max_text_tokens = val;
                }
            } else if let Some(rest) = line.strip_prefix("heads:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.n_head = val;
                }
            } else if let Some(rest) = line.strip_prefix("layers:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.n_layer = val;
                }
            } else if let Some(rest) = line.strip_prefix("number_text_tokens:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.number_text_tokens = val;
                }
            } else if let Some(rest) = line.strip_prefix("number_mel_codes:") {
                if let Ok(val) = rest.trim().parse::<usize>() {
                    config.num_mel_codes = val as u32;
                    config.n_vocab = val;
                }
            } else if let Some(rest) = line.strip_prefix("start_mel_token:") {
                if let Ok(val) = rest.trim().parse::<u32>() {
                    config.start_mel_token = val;
                }
            } else if let Some(rest) = line.strip_prefix("stop_mel_token:") {
                if let Ok(val) = rest.trim().parse::<u32>() {
                    config.stop_mel_token = val;
                }
            } else if let Some(rest) = line.strip_prefix("condition_type:") {
                config.condition_type = rest.trim().trim_matches('"').to_string();
            }
        }
        
        // Update dependent values
        config.n_positions = config.max_mel_tokens + config.max_text_tokens + 50;
        config.n_ctx = config.n_positions;
        
        Ok(config)
    }

    /// Get head dimension
    pub fn n_head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Get scale for attention
    pub fn scale(&self) -> f32 {
        1.0 / (self.n_head_dim() as f32).sqrt()
    }

    /// Check if config is valid
    pub fn validate(&self) -> bool {
        self.n_embd % self.n_head == 0
            && self.n_embd > 0
            && self.n_head > 0
            && self.n_layer > 0
    }
}

/// Model type for loading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptModelType {
    /// Standard GPT (PyTorch safetensors)
    Safetensors,
    /// PyTorch checkpoint (.pth)
    Pytorch,
    /// ONNX exported model
    Onnx,
}

impl GptModelType {
    /// Get weight filename
    pub fn weight_filename(&self) -> &'static str {
        match self {
            GptModelType::Safetensors => "model.safetensors",
            GptModelType::Pytorch => "gpt.pth",
            GptModelType::Onnx => "model.onnx",
        }
    }
}

/// Weight file entry
#[derive(Debug, Clone)]
pub struct WeightEntry {
    /// Parameter name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
}

impl WeightEntry {
    /// Load from safetensors file (placeholder)
    #[allow(dead_code)]
    pub fn load_from_safetensors(_path: &std::path::Path) -> std::io::Result<Vec<Self>> {
        // TODO: Implement actual safetensors loading
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GptConfig::default();
        assert!(config.validate());
        assert_eq!(config.n_embd, 1280);
        assert_eq!(config.n_head, 20);
        assert_eq!(config.n_head_dim(), 64);
        assert_eq!(config.n_layer, 24);
    }

    #[test]
    fn test_special_tokens() {
        let config = GptConfig::default();
        assert_eq!(config.start_mel_token, 8192);
        assert_eq!(config.stop_mel_token, 8193);
        assert_eq!(config.start_text_token, 0);
        assert_eq!(config.stop_text_token, 1);
    }

    #[test]
    fn test_max_tokens() {
        let config = GptConfig::default();
        assert_eq!(config.max_mel_tokens, 1815);
        assert_eq!(config.max_text_tokens, 600);
    }

    #[test]
    fn test_condition_config() {
        let config = GptConfig::default();
        assert_eq!(config.condition_type, "conformer_perceiver");
        assert_eq!(config.spk_cond_mode, "campplus");
    }
}
