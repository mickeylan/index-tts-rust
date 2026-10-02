//! Generation strategies for IndexTTS GPT
//!
//! This module implements token generation strategies matching the Python implementation.
//!
//! ## Generation Flow
//!
//! 1. **Prefill Phase**: Process conditioning + text tokens in one forward pass
//! 2. **Decode Phase**: Autoregressive generation
//!    - Get logits for next token
//!    - Apply sampling strategy (greedy, temperature, top-k, top-p)
//!    - Check for EOS (stop_mel_token = 8193)
//!    - Update KV cache
//!
//! ## Python Equivalents
//!
//! - `do_sample=False`: GreedyGenerator
//! - `do_sample=True`: SamplingGenerator with temperature
//! - `num_beams > 1`: BeamGenerator (not yet implemented)

use candle_core::{Tensor, Result as CandleResult, Device};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::config::GptConfig;
use super::model::IndexGpt;
use super::cache::KvCache;

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Generated token IDs (semantic codes)
    pub tokens: Vec<u32>,
    /// Number of tokens generated
    pub num_tokens: usize,
    /// Whether generation stopped due to EOS
    pub stopped: bool,
    /// Stop token position
    pub stop_position: Option<usize>,
}

impl GenerationOutput {
    /// Create from tokens
    pub fn new(tokens: Vec<u32>, stopped: bool, stop_position: Option<usize>) -> Self {
        let num_tokens = tokens.len();
        Self {
            tokens,
            num_tokens,
            stopped,
            stop_position,
        }
    }

    /// Check if output is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Generator trait
pub trait Generator {
    /// Generate tokens autoregressively
    fn generate(
        &mut self,
        model: &IndexGpt,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        max_length: usize,
    ) -> CandleResult<GenerationOutput>;

    /// Reset generator state (clear KV cache)
    fn reset(&mut self);
}

/// Greedy generator (equivalent to `do_sample=False`)
#[derive(Debug, Clone)]
pub struct GreedyGenerator {
    /// KV cache for generation
    kv_cache: Option<KvCache>,
}

impl GreedyGenerator {
    /// Create new greedy generator
    pub fn new(device: &Device, max_seq_len: usize) -> Self {
        Self {
            kv_cache: Some(KvCache::new(max_seq_len, device.clone())),
        }
    }

    /// Get a reference to the KV cache
    pub fn kv_cache(&self) -> Option<&KvCache> {
        self.kv_cache.as_ref()
    }
}

impl Generator for GreedyGenerator {
    fn generate(
        &mut self,
        model: &IndexGpt,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        max_length: usize,
    ) -> CandleResult<GenerationOutput> {
        let config = model.config();
        let device = model.device();
        let stop_token = config.stop_mel_token;
        
        // Initial prefill
        let mut logits = model.prefill(input_ids, attention_mask, None, self.kv_cache.as_mut())?;
        
        let mut tokens = Vec::new();
        let seq_len = input_ids.dim(1)?;
        let mut current_pos = seq_len;
        
        loop {
            // Get logits for last position
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?;
            let last_logits_flat = last_logits.reshape(())?;
            
            // Greedy: find max logit
            let probs = softmax(&last_logits_flat)?;
            let probs_vec = probs.to_vec1::<f32>()?;
            
            let next_token = probs_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            
            tokens.push(next_token);
            
            // Check for stop token
            if next_token == stop_token {
                // Remove stop token from output
                tokens.pop();
                let result_tokens = tokens.clone();
                return Ok(GenerationOutput::new(result_tokens, true, Some(tokens.len())));
            }
            
            // Check max length
            if current_pos >= max_length {
                let result_tokens = tokens.clone();
                return Ok(GenerationOutput::new(result_tokens, false, None));
            }
            
            // Decode next token
            let next_input = Tensor::new(&[next_token as i64], device)?.reshape((1, 1))?;
            logits = model.decode(&next_input, self.kv_cache.as_mut().unwrap(), current_pos)?;
            current_pos += 1;
        }
    }

    fn reset(&mut self) {
        if let Some(cache) = &mut self.kv_cache {
            cache.clear();
        }
    }
}

/// Sampling generator (equivalent to `do_sample=True`)
#[derive(Debug, Clone)]
pub struct SamplingGenerator {
    /// KV cache for generation
    kv_cache: Option<KvCache>,
    /// Random number generator
    rng: StdRng,
    /// Temperature (default: 1.0)
    temperature: f32,
    /// Top-k sampling (default: 0 = disabled)
    top_k: usize,
    /// Top-p (nucleus) sampling (default: 1.0 = disabled)
    top_p: f32,
    /// Repetition penalty (default: 1.0)
    repetition_penalty: f32,
    /// Generated tokens for repetition check
    generated_tokens: Vec<u32>,
}

impl SamplingGenerator {
    /// Create new sampling generator
    pub fn new(device: &Device, max_seq_len: usize, seed: u64) -> Self {
        Self {
            kv_cache: Some(KvCache::new(max_seq_len, device.clone())),
            rng: StdRng::seed_from_u64(seed),
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            generated_tokens: Vec::new(),
        }
    }

    /// Set temperature (default: 1.0)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling (0 = disabled)
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p (nucleus) sampling (1.0 = disabled)
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set repetition penalty (default: 1.0, >1.0 to penalize repetition)
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Sample a token from logits
    fn sample_token(&mut self, logits: &[f32]) -> u32 {
        let mut logits = logits.to_vec();
        
        // Apply repetition penalty
        if self.repetition_penalty != 1.0 {
            for &token in &self.generated_tokens {
                let idx = token as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.repetition_penalty;
                    } else {
                        logits[idx] *= self.repetition_penalty;
                    }
                }
            }
        }
        
        // Apply temperature
        if (self.temperature - 1.0).abs() > 1e-6 {
            for logit in &mut logits {
                *logit /= self.temperature;
            }
        }
        
        // Convert to probabilities
        let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();
        
        // Apply top-k
        if self.top_k > 0 && self.top_k < probs.len() {
            // Keep only top-k
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            indices.truncate(self.top_k);
            
            let mask: Vec<f32> = (0..probs.len())
                .map(|i| if indices.contains(&i) { 1.0 } else { 0.0 })
                .collect();
            
            for (i, m) in mask.iter().enumerate() {
                probs[i] *= m;
            }
            
            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }
        
        // Apply top-p (nucleus)
        if self.top_p < 1.0 && self.top_p > 0.0 {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut cumsum = 0.0f32;
            let mut selected = Vec::new();
            for &idx in &indices {
                cumsum += probs[idx];
                selected.push(idx);
                if cumsum >= self.top_p {
                    break;
                }
            }
            
            // Zero out non-selected and renormalize
            let mask: Vec<f32> = (0..probs.len())
                .map(|i| if selected.contains(&i) { 1.0 } else { 0.0 })
                .collect();
            
            for (i, m) in mask.iter().enumerate() {
                probs[i] *= m;
            }
            
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }
        
        // Sample from distribution
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0f32;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return i as u32;
            }
        }
        
        // Fallback to highest probability
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

impl Generator for SamplingGenerator {
    fn generate(
        &mut self,
        model: &IndexGpt,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        max_length: usize,
    ) -> CandleResult<GenerationOutput> {
        let config = model.config();
        let device = model.device();
        let stop_token = config.stop_mel_token;
        
        // Initial prefill
        let mut logits = model.prefill(input_ids, attention_mask, None, self.kv_cache.as_mut())?;
        
        let mut tokens = Vec::new();
        let seq_len = input_ids.dim(1)?;
        let mut current_pos = seq_len;
        self.generated_tokens.clear();
        
        loop {
            // Get logits for last position
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?;
            let last_logits_flat = last_logits.reshape(())?;
            let logits_vec = last_logits_flat.to_vec1::<f32>()?;
            
            // Sample next token
            let next_token = self.sample_token(&logits_vec);
            
            tokens.push(next_token);
            self.generated_tokens.push(next_token);
            
            // Check for stop token
            if next_token == stop_token {
                tokens.pop();
                let result_tokens = tokens.clone();
                return Ok(GenerationOutput::new(result_tokens, true, Some(tokens.len())));
            }
            
            // Check max length
            if current_pos >= max_length {
                let result_tokens = tokens.clone();
                return Ok(GenerationOutput::new(result_tokens, false, None));
            }
            
            // Decode next token
            let next_input = Tensor::new(&[next_token as i64], device)?.reshape((1, 1))?;
            logits = model.decode(&next_input, self.kv_cache.as_mut().unwrap(), current_pos)?;
            current_pos += 1;
        }
    }

    fn reset(&mut self) {
        if let Some(cache) = &mut self.kv_cache {
            cache.clear();
        }
        self.generated_tokens.clear();
    }
}

/// Beam search generator (not yet implemented)
#[derive(Debug, Clone)]
pub struct BeamGenerator {
    /// Number of beams
    num_beams: usize,
    /// Device
    device: Device,
    /// Temperature
    temperature: f32,
}

impl BeamGenerator {
    /// Create new beam search generator
    pub fn new(device: Device, _max_seq_len: usize, num_beams: usize) -> Self {
        Self {
            num_beams,
            device,
            temperature: 1.0,
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
}

impl Generator for BeamGenerator {
    fn generate(
        &mut self,
        _model: &IndexGpt,
        _input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
        _max_length: usize,
    ) -> CandleResult<GenerationOutput> {
        if self.num_beams > 1 {
            candle_core::bail!("Beam search not yet implemented. Use greedy (num_beams=1) or sampling instead.");
        }
        
        // Fall back to greedy for single beam
        let mut greedy = GreedyGenerator::new(&self.device, 512);
        greedy.generate(_model, _input_ids, _attention_mask, _max_length)
    }

    fn reset(&mut self) {
        // No cache in beam search yet
    }
}

/// Helper: compute softmax over last dimension
fn softmax(logits: &Tensor) -> CandleResult<Tensor> {
    // Softmax over the last dimension
    let exp = logits.exp()?;
    let sum = exp.sum_all()?;
    exp.broadcast_div(&sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_generator_creation() {
        let device = candle_core::Device::Cpu;
        let gen = GreedyGenerator::new(&device, 512);
        assert_eq!(gen.kv_cache().map(|c| c.is_empty()), Some(true));
    }

    #[test]
    fn test_sampling_generator_creation() {
        let device = candle_core::Device::Cpu;
        let gen = SamplingGenerator::new(&device, 512, 42)
            .with_temperature(0.8)
            .with_top_k(50)
            .with_top_p(0.9)
            .with_repetition_penalty(1.1);
        
        assert!((gen.temperature - 0.8).abs() < 0.001);
        assert_eq!(gen.top_k, 50);
        assert!((gen.top_p - 0.9).abs() < 0.001);
        assert!((gen.repetition_penalty - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_generation_output() {
        let output = GenerationOutput::new(vec![1, 2, 3], true, Some(3));
        assert_eq!(output.num_tokens, 3);
        assert!(output.stopped);
        assert_eq!(output.stop_position, Some(3));
    }
}
