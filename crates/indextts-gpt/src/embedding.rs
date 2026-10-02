//! Embedding layers for IndexTTS GPT
//!
//! This module implements the embedding layers for the IndexTTS-2.5 GPT model.

use candle_core::{Device, Module, Result as CandleResult, Tensor};

/// Text embedding layer
#[derive(Debug, Clone)]
pub struct TextEmbedding {
    /// Number of tokens
    pub num_tokens: usize,
    /// Embedding dimension
    pub dim: usize,
}

impl TextEmbedding {
    /// Create a new text embedding (placeholder)
    pub fn new(num_tokens: usize, dim: usize) -> Self {
        Self { num_tokens, dim }
    }

    /// Forward pass: lookup embeddings
    /// 
    /// Input: (batch, seq_len) token IDs
    /// Output: (batch, seq_len, dim) embeddings
    pub fn forward(&self, ids: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        if weight.dims() != [self.num_tokens, self.dim] {
            candle_core::bail!(
                "text embedding weight shape {:?}, expected [{}, {}]",
                weight.dims(),
                self.num_tokens,
                self.dim
            )
        }
        candle_nn::Embedding::new(weight.clone(), self.dim).forward(ids)
    }
}

/// Learned position embedding
#[derive(Debug, Clone)]
pub struct LearnedPositionEmbedding {
    /// Maximum sequence length
    pub max_len: usize,
    /// Embedding dimension
    pub dim: usize,
}

impl LearnedPositionEmbedding {
    /// Create a new position embedding
    pub fn new(max_len: usize, dim: usize) -> Self {
        Self { max_len, dim }
    }

    /// Forward pass
    pub fn forward(&self, position_ids: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
        if weight.dims() != [self.max_len, self.dim] {
            candle_core::bail!(
                "position embedding weight shape {:?}, expected [{}, {}]",
                weight.dims(),
                self.max_len,
                self.dim
            )
        }
        candle_nn::Embedding::new(weight.clone(), self.dim).forward(position_ids)
    }
}

/// Create position IDs for a sequence
pub fn create_position_ids(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    let ids: Vec<i64> = (0..seq_len as i64).collect();
    Tensor::new(ids.as_slice(), device)?.reshape((1, seq_len))
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Weight
    pub weight: Tensor,
    /// Bias
    pub bias: Option<Tensor>,
    /// Variance epsilon
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer norm
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let bias = match &self.bias {
            Some(bias) => bias.clone(),
            None => Tensor::zeros(self.weight.dims(), self.weight.dtype(), self.weight.device())?,
        };
        candle_nn::LayerNorm::new(self.weight.clone(), bias, self.eps as f64).forward(x)
    }
}

/// Linear projection
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix
    pub weight: Tensor,
    /// Bias vector
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Forward pass: y = x @ W^T + b
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        candle_nn::Linear::new(self.weight.clone(), self.bias.clone()).forward(x)
    }
}

/// GPT Embeddings combining text and position
#[derive(Debug, Clone)]
pub struct GptEmbeddings {
    /// Text embedding
    pub text_emb: TextEmbedding,
    /// Position embedding
    pub pos_emb: LearnedPositionEmbedding,
    /// Dropout probability
    pub dropout: f32,
}

impl GptEmbeddings {
    /// Create new GPT embeddings
    pub fn new(num_tokens: usize, dim: usize, max_len: usize) -> Self {
        Self {
            text_emb: TextEmbedding::new(num_tokens, dim),
            pos_emb: LearnedPositionEmbedding::new(max_len, dim),
            dropout: 0.0,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        input_ids: &Tensor,
        text_weight: &Tensor,
        pos_weight: &Tensor,
    ) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let pos_ids = create_position_ids(seq_len, &input_ids.device())?;
        
        let text_emb = self.text_emb.forward(input_ids, text_weight)?;
        let pos_emb = self.pos_emb.forward(&pos_ids, pos_weight)?;
        
        // Add embeddings.
        text_emb + pos_emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_ids() {
        let device = Device::Cpu;
        let ids = create_position_ids(10, &device).unwrap();
        assert_eq!(ids.dims(), &[1, 10]);
    }

    #[test]
    fn test_text_embedding() {
        let device = Device::Cpu;
        let emb = TextEmbedding::new(3, 2);
        let weight = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device).unwrap();
        let ids = Tensor::new(&[[2u32, 0]], &device).unwrap();
        let output = emb.forward(&ids, &weight).unwrap();
        assert_eq!(output.dims(), &[1, 2, 2]);
        assert_eq!(output.to_vec3::<f32>().unwrap(), vec![vec![vec![5., 6.], vec![1., 2.]]]);
    }

    #[test]
    fn test_linear_uses_pytorch_layout() {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 0.], [0., 2.], [1., 1.]], &device).unwrap();
        let bias = Tensor::new(&[1f32, -1., 0.5], &device).unwrap();
        let linear = Linear::new(weight, Some(bias));
        let input = Tensor::new(&[[2f32, 3.]], &device).unwrap();
        assert_eq!(linear.forward(&input).unwrap().to_vec2::<f32>().unwrap(), vec![vec![3., 5., 5.5]]);
    }

    #[test]
    fn test_layer_norm_matches_expected_values() {
        let device = Device::Cpu;
        let norm = LayerNorm::new(
            Tensor::ones(2, candle_core::DType::F32, &device).unwrap(),
            Some(Tensor::zeros(2, candle_core::DType::F32, &device).unwrap()),
            1e-5,
        );
        let input = Tensor::new(&[[1f32, 3.]], &device).unwrap();
        let output = norm.forward(&input).unwrap().to_vec2::<f32>().unwrap();
        assert!((output[0][0] + 1.0).abs() < 1e-4);
        assert!((output[0][1] - 1.0).abs() < 1e-4);
    }
}
