//! IndexTTS-2.5 semantic GPT model.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::{collections::HashMap, path::Path};

use crate::{
    attention::Gpt2Block,
    cache::KvCache,
    config::GptConfig,
    embedding::{LayerNorm, LearnedPositionEmbedding, Linear, TextEmbedding},
    weights::Weights,
};

/// GPT inference model for a precomputed conditioning/text prefix.
#[derive(Debug, Clone)]
pub struct IndexGpt {
    config: GptConfig,
    device: Device,
    cached_prefix: Option<Tensor>,
    mel_embedding: TextEmbedding,
    mel_embedding_weight: Option<Tensor>,
    mel_position: LearnedPositionEmbedding,
    mel_position_weight: Option<Tensor>,
    blocks: Vec<Gpt2Block>,
    gpt_final_norm: Option<LayerNorm>,
    final_norm: Option<LayerNorm>,
    mel_head: Option<Linear>,
    loaded: bool,
    num_params: usize,
}

impl IndexGpt {
    pub fn new(config: GptConfig, device: Device) -> CandleResult<Self> {
        let num_params = Self::calculate_params(&config);
        let mel_positions = config.max_mel_tokens + 3;
        Ok(Self {
            mel_embedding: TextEmbedding::new(config.n_vocab, config.n_embd),
            mel_position: LearnedPositionEmbedding::new(mel_positions, config.n_embd),
            config,
            device,
            cached_prefix: None,
            mel_embedding_weight: None,
            mel_position_weight: None,
            blocks: Vec::new(),
            gpt_final_norm: None,
            final_norm: None,
            mel_head: None,
            loaded: false,
            num_params,
        })
    }

    fn calculate_params(config: &GptConfig) -> usize {
        let embedding_params = config.n_vocab * config.n_embd
            + (config.max_mel_tokens + 3) * config.n_embd;
        let per_layer = 4 * config.n_embd * config.n_embd
            + 2 * config.n_embd * config.n_inner
            + 4 * config.n_embd
            + config.n_inner;
        embedding_params
            + per_layer * config.n_layer
            + 4 * config.n_embd
            + config.n_embd * config.n_vocab
            + config.n_vocab
    }

    pub fn load_weights(&mut self, path: &Path) -> CandleResult<()> {
        let weights = Weights::load(path, &self.device)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        weights.validate_greedy_contract()
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.load_state_dict(weights.as_map())
    }

    pub fn load_state_dict(&mut self, weights: &HashMap<String, Tensor>) -> CandleResult<()> {
        let get = |name: &str| -> CandleResult<Tensor> {
            weights.get(name).cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
        };

        self.mel_embedding_weight = Some(get("mel_embedding.weight")?);
        self.mel_position_weight = Some(get("mel_pos_embedding.emb.weight")?);

        let mut blocks = Vec::with_capacity(self.config.n_layer);
        for layer in 0..self.config.n_layer {
            let prefix = format!("gpt.h.{layer}");
            blocks.push(Gpt2Block::from_weights(
                layer,
                self.config.n_embd,
                self.config.n_head,
                self.config.n_inner,
                get(&format!("{prefix}.ln_1.weight"))?,
                get(&format!("{prefix}.ln_1.bias"))?,
                get(&format!("{prefix}.attn.c_attn.weight"))?,
                get(&format!("{prefix}.attn.c_attn.bias"))?,
                get(&format!("{prefix}.attn.c_proj.weight"))?,
                get(&format!("{prefix}.attn.c_proj.bias"))?,
                get(&format!("{prefix}.ln_2.weight"))?,
                get(&format!("{prefix}.ln_2.bias"))?,
                get(&format!("{prefix}.mlp.c_fc.weight"))?,
                get(&format!("{prefix}.mlp.c_fc.bias"))?,
                get(&format!("{prefix}.mlp.c_proj.weight"))?,
                get(&format!("{prefix}.mlp.c_proj.bias"))?,
            )?);
        }
        self.blocks = blocks;
        self.gpt_final_norm = Some(LayerNorm::new(
            get("gpt.ln_f.weight")?,
            Some(get("gpt.ln_f.bias")?),
            1e-5,
        ));
        self.final_norm = Some(LayerNorm::new(
            get("final_norm.weight")?,
            Some(get("final_norm.bias")?),
            1e-5,
        ));
        self.mel_head = Some(Linear::new(
            get("mel_head.weight")?,
            Some(get("mel_head.bias")?),
        ));
        self.loaded = true;
        Ok(())
    }

    pub fn store_mel_emb(&mut self, prefix: Tensor) {
        self.cached_prefix = Some(prefix);
    }

    pub fn clear_mel_emb(&mut self) {
        self.cached_prefix = None;
    }

    pub fn get_cached_mel_emb(&self) -> Option<&Tensor> {
        self.cached_prefix.as_ref()
    }

    /// Store a caller-produced prefix and create the fake IDs/mask expected by
    /// the generation API. `prefix` already contains conditioning and text embeddings.
    pub fn prepare_inputs(
        &mut self,
        prefix: &Tensor,
        _text_tokens: &Tensor,
        _lang_ids: Option<&Tensor>,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (batch, prefix_len, hidden) = prefix.dims3()?;
        if hidden != self.config.n_embd {
            candle_core::bail!("prefix hidden size {hidden}, expected {}", self.config.n_embd)
        }
        let target_len = prefix_len + 1;
        let mut fake_ids = vec![1u32; batch * target_len];
        for row in 0..batch {
            fake_ids[row * target_len + target_len - 1] = self.config.start_mel_token;
        }
        let fake_inputs = Tensor::new(fake_ids.as_slice(), &self.device)?
            .reshape((batch, target_len))?;
        self.store_mel_emb(prefix.clone());
        let mask = Tensor::ones((batch, target_len), DType::U32, &self.device)?;
        Ok((fake_inputs, mask))
    }

    pub fn forward(&self, mel_ids: &Tensor) -> CandleResult<Tensor> {
        self.require_loaded()?;
        let seq_len = mel_ids.dim(1)?;
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let hidden = self.embed_mel(mel_ids, &positions)?;
        self.forward_hidden(hidden, None)
    }

    pub fn prefill(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        kv_cache: Option<&mut KvCache>,
    ) -> CandleResult<Tensor> {
        self.require_loaded()?;
        let prefix = self.cached_prefix.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("prefix embeddings are not prepared".into()))?;
        let (batch, total_len) = input_ids.dims2()?;
        let (prefix_batch, prefix_len, _) = prefix.dims3()?;
        if batch != prefix_batch || total_len != prefix_len + 1 {
            candle_core::bail!(
                "fake input shape {:?} is incompatible with prefix shape {:?}",
                input_ids.dims(),
                prefix.dims()
            )
        }
        let start_id = input_ids.narrow(1, total_len - 1, 1)?;
        let start_position = Tensor::new(&[0u32], &self.device)?;
        let start_embedding = self.embed_mel(&start_id, &start_position)?;
        let hidden = Tensor::cat(&[prefix, &start_embedding], 1)?;
        self.forward_hidden(hidden, kv_cache)
    }

    pub fn decode(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut KvCache,
        _position: usize,
    ) -> CandleResult<Tensor> {
        self.require_loaded()?;
        let prefix_len = self.cached_prefix.as_ref()
            .ok_or_else(|| candle_core::Error::Msg("prefix embeddings are not prepared".into()))?
            .dim(1)?;
        // The official wrapper leaves position 1 unused after the prefill start
        // token: decode uses attention_mask_len - cached_prefix_len, yielding 2
        // for the first generated semantic token.
        let mel_position = kv_cache.seq_len().checked_sub(prefix_len)
            .ok_or_else(|| candle_core::Error::Msg("KV cache is shorter than prefix".into()))?
            + 1;
        if mel_position >= self.mel_position.max_len {
            candle_core::bail!("mel position {mel_position} exceeds position table")
        }
        let position = Tensor::new(&[mel_position as u32], &self.device)?;
        let hidden = self.embed_mel(input_ids, &position)?;
        self.forward_hidden(hidden, Some(kv_cache))
    }

    fn embed_mel(&self, ids: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let token = self.mel_embedding.forward(
            ids,
            self.mel_embedding_weight.as_ref().unwrap(),
        )?;
        let position = self.mel_position.forward(
            positions,
            self.mel_position_weight.as_ref().unwrap(),
        )?;
        token.broadcast_add(&position)
    }

    fn forward_hidden(
        &self,
        mut hidden: Tensor,
        mut cache: Option<&mut KvCache>,
    ) -> CandleResult<Tensor> {
        for block in &self.blocks {
            hidden = match cache.as_deref_mut() {
                Some(cache) => block.forward_with_cache(&hidden, cache)?,
                None => block.forward(&hidden)?,
            };
        }
        hidden = self.gpt_final_norm.as_ref().unwrap().forward(&hidden)?;
        hidden = self.final_norm.as_ref().unwrap().forward(&hidden)?;
        self.mel_head.as_ref().unwrap().forward(&hidden)
    }

    fn require_loaded(&self) -> CandleResult<()> {
        if !self.loaded {
            candle_core::bail!("model weights are not loaded")
        }
        Ok(())
    }

    pub fn config(&self) -> &GptConfig { &self.config }
    pub fn device(&self) -> &Device { &self.device }
    pub fn is_loaded(&self) -> bool { self.loaded }
    pub fn num_parameters(&self) -> usize { self.num_params }
}

pub fn create_position_ids(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    let ids: Vec<u32> = (0..seq_len as u32).collect();
    Tensor::new(ids.as_slice(), device)?.reshape((1, seq_len))
}

pub fn create_attention_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    Tensor::ones((1, seq_len), DType::U32, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> GptConfig {
        GptConfig {
            n_embd: 2,
            n_head: 1,
            n_layer: 1,
            n_vocab: 3,
            n_positions: 8,
            n_ctx: 8,
            n_inner: 4,
            max_mel_tokens: 5,
            num_mel_codes: 3,
            start_mel_token: 1,
            stop_mel_token: 2,
            ..GptConfig::default()
        }
    }

    fn tiny_weights(device: &Device) -> HashMap<String, Tensor> {
        let z1 = |size| Tensor::zeros(size, DType::F32, device).unwrap();
        let z2 = |shape| Tensor::zeros(shape, DType::F32, device).unwrap();
        let o1 = |size| Tensor::ones(size, DType::F32, device).unwrap();
        HashMap::from([
            ("mel_embedding.weight".into(), z2((3, 2))),
            ("mel_pos_embedding.emb.weight".into(), z2((8, 2))),
            ("gpt.h.0.ln_1.weight".into(), o1(2)),
            ("gpt.h.0.ln_1.bias".into(), z1(2)),
            ("gpt.h.0.attn.c_attn.weight".into(), z2((2, 6))),
            ("gpt.h.0.attn.c_attn.bias".into(), z1(6)),
            ("gpt.h.0.attn.c_proj.weight".into(), z2((2, 2))),
            ("gpt.h.0.attn.c_proj.bias".into(), z1(2)),
            ("gpt.h.0.ln_2.weight".into(), o1(2)),
            ("gpt.h.0.ln_2.bias".into(), z1(2)),
            ("gpt.h.0.mlp.c_fc.weight".into(), z2((2, 4))),
            ("gpt.h.0.mlp.c_fc.bias".into(), z1(4)),
            ("gpt.h.0.mlp.c_proj.weight".into(), z2((4, 2))),
            ("gpt.h.0.mlp.c_proj.bias".into(), z1(2)),
            ("gpt.ln_f.weight".into(), o1(2)),
            ("gpt.ln_f.bias".into(), z1(2)),
            ("final_norm.weight".into(), o1(2)),
            ("final_norm.bias".into(), z1(2)),
            ("mel_head.weight".into(), z2((3, 2))),
            ("mel_head.bias".into(), Tensor::new(&[1f32, 2., 3.], device).unwrap()),
        ])
    }

    #[test]
    fn model_wires_prefill_and_decode_through_cache() {
        let device = Device::Cpu;
        let mut model = IndexGpt::new(tiny_config(), device.clone()).unwrap();
        model.load_state_dict(&tiny_weights(&device)).unwrap();
        let prefix = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
        let text = Tensor::new(&[[0u32]], &device).unwrap();
        let (input_ids, mask) = model.prepare_inputs(&prefix, &text, None).unwrap();
        let mut cache = KvCache::new(8, device.clone());
        let logits = model.prefill(&input_ids, Some(&mask), None, Some(&mut cache)).unwrap();
        assert_eq!(logits.dims(), &[1, 2, 3]);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(logits.to_vec3::<f32>().unwrap()[0][1], vec![1., 2., 3.]);

        let next = Tensor::new(&[[1u32]], &device).unwrap();
        let logits = model.decode(&next, &mut cache, 2).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 3]);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_model_creation() {
        let model = IndexGpt::new(GptConfig::default(), Device::Cpu).unwrap();
        assert_eq!(model.config().n_layer, 24);
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_position_ids() {
        let ids = create_position_ids(10, &Device::Cpu).unwrap();
        assert_eq!(ids.dims(), &[1, 10]);
    }
}
