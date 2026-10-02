//! GPT-2 transformer primitives used by IndexTTS-2.5.

use candle_core::{D, Device, Module, Result as CandleResult, Tensor};

use crate::embedding::{LayerNorm, Linear};

/// GPT-2 self-attention. Official Hugging Face Conv1D matrices are accepted
/// in their checkpoint layout `[in, out]` and transposed once at construction.
#[derive(Debug, Clone)]
pub struct Gpt2Attention {
    pub n_head: usize,
    pub head_dim: usize,
    c_attn: Option<Linear>,
    c_proj: Option<Linear>,
}

impl Gpt2Attention {
    pub fn new(n_embd: usize, n_head: usize) -> CandleResult<Self> {
        if n_embd % n_head != 0 {
            candle_core::bail!("embedding dimension {n_embd} is not divisible by {n_head} heads")
        }
        Ok(Self {
            n_head,
            head_dim: n_embd / n_head,
            c_attn: None,
            c_proj: None,
        })
    }

    pub fn from_weights(
        n_embd: usize,
        n_head: usize,
        c_attn_weight: Tensor,
        c_attn_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,
    ) -> CandleResult<Self> {
        if c_attn_weight.dims() != [n_embd, 3 * n_embd] {
            candle_core::bail!("invalid c_attn weight shape {:?}", c_attn_weight.dims())
        }
        if c_proj_weight.dims() != [n_embd, n_embd] {
            candle_core::bail!("invalid c_proj weight shape {:?}", c_proj_weight.dims())
        }
        let mut attention = Self::new(n_embd, n_head)?;
        attention.c_attn = Some(Linear::new(c_attn_weight.t()?, Some(c_attn_bias)));
        attention.c_proj = Some(Linear::new(c_proj_weight.t()?, Some(c_proj_bias)));
        Ok(attention)
    }

    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let c_attn = self.c_attn.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("attention weights are not loaded".into()))?;
        let c_proj = self.c_proj.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("attention weights are not loaded".into()))?;
        let (batch, seq_len, n_embd) = hidden_states.dims3()?;
        let qkv = c_attn.forward(hidden_states)?;
        let query = qkv.narrow(2, 0, n_embd)?;
        let key = qkv.narrow(2, n_embd, n_embd)?;
        let value = qkv.narrow(2, 2 * n_embd, n_embd)?;

        let split = |tensor: Tensor| -> CandleResult<Tensor> {
            tensor.reshape((batch, seq_len, self.n_head, self.head_dim))?
                .permute((0, 2, 1, 3))
        };
        let query = split(query)?;
        let key = split(key)?;
        let value = split(value)?;

        let scores = query.matmul(&key.transpose(2, 3)?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let mask = create_causal_mask(seq_len, scores.device())?
            .reshape((1, 1, seq_len, seq_len))?;
        let scores = scores.broadcast_add(&mask)?;
        let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = probabilities.matmul(&value)?
            .permute((0, 2, 1, 3))?
            .reshape((batch, seq_len, n_embd))?;
        c_proj.forward(&context)
    }

    pub fn n_head(&self) -> usize { self.n_head }
    pub fn head_dim(&self) -> usize { self.head_dim }
}

#[derive(Debug, Clone)]
pub struct Gpt2MLP {
    pub n_inner: usize,
    c_fc: Option<Linear>,
    c_proj: Option<Linear>,
}

impl Gpt2MLP {
    pub fn new(n_inner: usize) -> CandleResult<Self> {
        Ok(Self { n_inner, c_fc: None, c_proj: None })
    }

    pub fn from_weights(
        n_embd: usize,
        n_inner: usize,
        c_fc_weight: Tensor,
        c_fc_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,
    ) -> CandleResult<Self> {
        if c_fc_weight.dims() != [n_embd, n_inner] {
            candle_core::bail!("invalid c_fc weight shape {:?}", c_fc_weight.dims())
        }
        if c_proj_weight.dims() != [n_inner, n_embd] {
            candle_core::bail!("invalid MLP c_proj weight shape {:?}", c_proj_weight.dims())
        }
        Ok(Self {
            n_inner,
            c_fc: Some(Linear::new(c_fc_weight.t()?, Some(c_fc_bias))),
            c_proj: Some(Linear::new(c_proj_weight.t()?, Some(c_proj_bias))),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let c_fc = self.c_fc.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("MLP weights are not loaded".into()))?;
        let c_proj = self.c_proj.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("MLP weights are not loaded".into()))?;
        let hidden_states = c_fc.forward(hidden_states)?;
        let hidden_states = candle_nn::Activation::NewGelu.forward(&hidden_states)?;
        c_proj.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct Gpt2Block {
    pub layer_idx: usize,
    pub attn: Gpt2Attention,
    pub mlp: Gpt2MLP,
    ln_1: Option<LayerNorm>,
    ln_2: Option<LayerNorm>,
}

impl Gpt2Block {
    pub fn new(layer_idx: usize, n_embd: usize, n_head: usize, n_inner: usize) -> CandleResult<Self> {
        Ok(Self {
            layer_idx,
            attn: Gpt2Attention::new(n_embd, n_head)?,
            mlp: Gpt2MLP::new(n_inner)?,
            ln_1: None,
            ln_2: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        layer_idx: usize,
        n_embd: usize,
        n_head: usize,
        n_inner: usize,
        ln_1_weight: Tensor,
        ln_1_bias: Tensor,
        c_attn_weight: Tensor,
        c_attn_bias: Tensor,
        c_proj_weight: Tensor,
        c_proj_bias: Tensor,
        ln_2_weight: Tensor,
        ln_2_bias: Tensor,
        c_fc_weight: Tensor,
        c_fc_bias: Tensor,
        mlp_c_proj_weight: Tensor,
        mlp_c_proj_bias: Tensor,
    ) -> CandleResult<Self> {
        Ok(Self {
            layer_idx,
            attn: Gpt2Attention::from_weights(
                n_embd, n_head, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias,
            )?,
            mlp: Gpt2MLP::from_weights(
                n_embd, n_inner, c_fc_weight, c_fc_bias, mlp_c_proj_weight, mlp_c_proj_bias,
            )?,
            ln_1: Some(LayerNorm::new(ln_1_weight, Some(ln_1_bias), 1e-5)),
            ln_2: Some(LayerNorm::new(ln_2_weight, Some(ln_2_bias), 1e-5)),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let ln_1 = self.ln_1.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("block weights are not loaded".into()))?;
        let ln_2 = self.ln_2.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("block weights are not loaded".into()))?;
        let attention = self.attn.forward(&ln_1.forward(hidden_states)?)?;
        let hidden_states = (hidden_states + attention)?;
        let feed_forward = self.mlp.forward(&ln_2.forward(&hidden_states)?)?;
        hidden_states + feed_forward
    }
}

pub fn create_causal_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|row| (0..seq_len).map(move |column| if column <= row { 0.0 } else { f32::MIN }))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_attention_creation() {
        let attn = Gpt2Attention::new(1280, 20).unwrap();
        assert_eq!(attn.n_head(), 20);
        assert_eq!(attn.head_dim(), 64);
    }

    #[test]
    fn test_gpt2_block_creation() {
        let block = Gpt2Block::new(0, 1280, 20, 5120).unwrap();
        assert_eq!(block.layer_idx, 0);
    }

    #[test]
    fn test_attention_with_zero_weights_returns_projection_bias() {
        let device = Device::Cpu;
        let attention = Gpt2Attention::from_weights(
            2,
            1,
            Tensor::zeros((2, 6), candle_core::DType::F32, &device).unwrap(),
            Tensor::zeros(6, candle_core::DType::F32, &device).unwrap(),
            Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap(),
            Tensor::new(&[0.25f32, -0.5], &device).unwrap(),
        ).unwrap();
        let input = Tensor::new(&[[[1f32, 2.], [3., 4.]]], &device).unwrap();
        assert_eq!(
            attention.forward(&input).unwrap().to_vec3::<f32>().unwrap(),
            vec![vec![vec![0.25, -0.5], vec![0.25, -0.5]]]
        );
    }

    #[test]
    fn test_mlp_applies_gelu_and_projection() {
        let device = Device::Cpu;
        let mlp = Gpt2MLP::from_weights(
            1,
            1,
            Tensor::new(&[[1f32]], &device).unwrap(),
            Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
            Tensor::new(&[[1f32]], &device).unwrap(),
            Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
        ).unwrap();
        let output = mlp.forward(&Tensor::new(&[[1f32]], &device).unwrap())
            .unwrap().to_vec2::<f32>().unwrap();
        assert!((output[0][0] - 0.841_192).abs() < 1e-4);
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(2, &device).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(mask[0][0], 0.0);
        assert!(mask[0][1] < -1e30);
        assert_eq!(mask[1], vec![0.0, 0.0]);
    }
}
