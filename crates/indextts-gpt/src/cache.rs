//! KV Cache implementation for efficient generation

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use std::collections::HashMap;

/// Key-Value cache for transformer layers
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cache for key tensors [layer, 2, batch, heads, seq_len, head_dim]
    keys: HashMap<String, Tensor>,
    /// Cache for value tensors
    values: HashMap<String, Tensor>,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Device
    device: Device,
}

impl KvCache {
    /// Create a new KV cache
    pub fn new(max_seq_len: usize, device: Device) -> Self {
        Self {
            keys: HashMap::new(),
            values: HashMap::new(),
            max_seq_len,
            device,
        }
    }

    /// Get key tensor for a layer
    pub fn get_key(&self, layer: usize) -> Option<&Tensor> {
        self.keys.get(&format!("layer_{}", layer))
    }

    /// Get value tensor for a layer
    pub fn get_value(&self, layer: usize) -> Option<&Tensor> {
        self.values.get(&format!("layer_{}", layer))
    }

    /// Set key tensor for a layer
    pub fn set_key(&mut self, layer: usize, key: Tensor) {
        self.keys.insert(format!("layer_{}", layer), key);
    }

    /// Set value tensor for a layer
    pub fn set_value(&mut self, layer: usize, value: Tensor) {
        self.values.insert(format!("layer_{}", layer), value);
    }

    /// Update cache with new key-value pairs
    pub fn update(&mut self, layer: usize, new_key: &Tensor, new_value: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let key_name = format!("layer_{}", layer);
        let value_name = format!("layer_{}", layer);
        
        if let (Some(existing_key), Some(existing_value)) = (
            self.keys.get(&key_name),
            self.values.get(&value_name)
        ) {
            // Concatenate along sequence dimension
            let key = candle_core::Tensor::cat(&[existing_key, new_key], 2)?;
            let value = candle_core::Tensor::cat(&[existing_value, new_value], 2)?;
            
            self.keys.insert(key_name, key.clone());
            self.values.insert(value_name, value.clone());
            
            Ok((key, value))
        } else {
            // First entry
            self.keys.insert(key_name, new_key.clone());
            self.values.insert(value_name, new_value.clone());
            Ok((new_key.clone(), new_value.clone()))
        }
    }

    /// Get all keys as a single tensor
    pub fn all_keys(&self) -> Option<Tensor> {
        let mut layers: Vec<_> = self.keys.keys().collect();
        layers.sort();
        
        if layers.is_empty() {
            return None;
        }
        
        let mut tensors: Vec<Tensor> = layers
            .iter()
            .filter_map(|k| self.keys.get(*k).cloned())
            .collect();
        
        if tensors.is_empty() {
            return None;
        }
        
        // Stack along a new dimension
        candle_core::Tensor::stack(&tensors, 0).ok()
    }

    /// Get all values as a single tensor
    pub fn all_values(&self) -> Option<Tensor> {
        let mut layers: Vec<_> = self.values.keys().collect();
        layers.sort();
        
        if layers.is_empty() {
            return None;
        }
        
        let mut tensors: Vec<Tensor> = layers
            .iter()
            .filter_map(|v| self.values.get(*v).cloned())
            .collect();
        
        if tensors.is_empty() {
            return None;
        }
        
        candle_core::Tensor::stack(&tensors, 0).ok()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.values.is_empty()
    }

    /// Get cache length (current sequence length)
    pub fn seq_len(&self) -> usize {
        self.keys.values().next()
            .and_then(|t| t.dims().get(2).copied())
            .unwrap_or(0)
    }

    /// Reset to empty state
    pub fn reset(&mut self) {
        self.clear();
    }
}

/// Attention mask for causal attention
pub fn create_causal_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    // Create lower triangular mask (causal)
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::MIN }))
        .collect();
    
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

/// Create attention mask for batch
pub fn create_batch_attention_mask(
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> CandleResult<Tensor> {
    // Create attention mask where valid positions are 1, padded positions are 0
    let mask: Vec<f32> = std::iter::repeat(1.0_f32)
        .take(batch_size * seq_len)
        .collect();
    
    Tensor::from_slice(&mask, (batch_size, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cache_creation() {
        let device = Device::Cpu;
        let cache = KvCache::new(512, device);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let mask = create_causal_mask(10, &device).unwrap();
        assert_eq!(mask.dims(), &[10, 10]);
    }
}
