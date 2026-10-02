//! Weight loading utilities for IndexTTS GPT
//!
//! This module provides placeholder for weight loading.
//! Full implementation requires PyTorch interop or ONNX export.

use candle_core::{Tensor, Device};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Weight loading not implemented: {0}")]
    NotImplemented(String),
    #[error("Missing weight: {0}")]
    MissingWeight(String),
    #[error("Tensor error: {0}")]
    Tensor(String),
    #[error("Shape mismatch for {name}: actual {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
}

/// Weight storage
pub struct Weights {
    /// Named tensors
    tensors: HashMap<String, Tensor>,
    /// Device
    device: Device,
}

impl Weights {
    /// Load tensors from a safetensors file using Candle's dtype-aware loader.
    pub fn load(path: &Path, device: &Device) -> Result<Self, WeightError> {
        if !path.is_file() {
            return Err(WeightError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("weight file not found: {}", path.display()),
            )));
        }
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|error| WeightError::Tensor(error.to_string()))?;
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Get a weight tensor.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Borrow the complete state dictionary.
    pub fn as_map(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Check if weight exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all weight names
    pub fn names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get number of weights
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Insert a weight.
    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.tensors.insert(name, tensor);
    }

    /// Validate all tensors needed by the greedy semantic GPT path.
    pub fn validate_greedy_contract(&self) -> Result<(), WeightError> {
        for spec in greedy_gpt_weight_specs() {
            let tensor = self.tensors.get(&spec.name)
                .ok_or_else(|| WeightError::MissingWeight(spec.name.clone()))?;
            if tensor.dims() != spec.shape.as_slice() {
                return Err(WeightError::ShapeMismatch {
                    name: spec.name,
                    actual: tensor.dims().to_vec(),
                    expected: spec.shape,
                });
            }
        }
        Ok(())
    }
}

/// Weight loading result with metadata
pub struct LoadedWeights {
    /// The weights
    pub weights: Weights,
    /// Total parameters
    pub num_params: usize,
    /// Weight names
    pub weight_names: Vec<String>,
}

impl LoadedWeights {
    /// Load weights and calculate stats
    pub fn load(path: &Path, device: &Device) -> Result<Self, WeightError> {
        let weights = Weights::load(path, device)?;
        
        let num_params: usize = weights.tensors.values()
            .map(|t| t.elem_count())
            .sum();
        
        let mut weight_names: Vec<String> = weights.tensors.keys()
            .cloned()
            .collect();
        weight_names.sort();
        
        Ok(Self {
            weights,
            num_params,
            weight_names,
        })
    }
}

/// Print weight info
#[allow(dead_code)]
pub fn print_weight_info(loaded: &LoadedWeights) {
    eprintln!("Loaded {} weights, {} parameters", loaded.weights.len(), loaded.num_params);
    eprintln!("\nWeight names:");
    for name in &loaded.weight_names {
        if let Some(tensor) = loaded.weights.get(name) {
            eprintln!("  {}: {:?}", name, tensor.dims());
        }
    }
}

/// One required tensor in the IndexTTS-2.5 greedy GPT contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
    /// PyTorch `Conv1D` stores matrices as `[in, out]`; regular linear
    /// layers and embeddings use their native PyTorch layouts.
    pub conv1d_input_major: bool,
}

/// Exact tensors required to construct text/language prefixes and perform
/// semantic-token generation.
pub fn greedy_gpt_weight_specs() -> Vec<WeightSpec> {
    let mut specs = vec![
        WeightSpec { name: "mel_embedding.weight".into(), shape: vec![8194, 1280], conv1d_input_major: false },
        WeightSpec { name: "mel_pos_embedding.emb.weight".into(), shape: vec![1818, 1280], conv1d_input_major: false },
        WeightSpec { name: "text_embedding.weight".into(), shape: vec![60510, 1280], conv1d_input_major: false },
        WeightSpec { name: "text_pos_embedding.emb.weight".into(), shape: vec![602, 1280], conv1d_input_major: false },
        WeightSpec { name: "lang_embedding.weight".into(), shape: vec![107, 1280], conv1d_input_major: false },
    ];

    for layer in 0..24 {
        let prefix = format!("gpt.h.{layer}");
        specs.extend([
            WeightSpec { name: format!("{prefix}.ln_1.weight"), shape: vec![1280], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.ln_1.bias"), shape: vec![1280], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.attn.c_attn.weight"), shape: vec![1280, 3840], conv1d_input_major: true },
            WeightSpec { name: format!("{prefix}.attn.c_attn.bias"), shape: vec![3840], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.attn.c_proj.weight"), shape: vec![1280, 1280], conv1d_input_major: true },
            WeightSpec { name: format!("{prefix}.attn.c_proj.bias"), shape: vec![1280], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.ln_2.weight"), shape: vec![1280], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.ln_2.bias"), shape: vec![1280], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.mlp.c_fc.weight"), shape: vec![1280, 5120], conv1d_input_major: true },
            WeightSpec { name: format!("{prefix}.mlp.c_fc.bias"), shape: vec![5120], conv1d_input_major: false },
            WeightSpec { name: format!("{prefix}.mlp.c_proj.weight"), shape: vec![5120, 1280], conv1d_input_major: true },
            WeightSpec { name: format!("{prefix}.mlp.c_proj.bias"), shape: vec![1280], conv1d_input_major: false },
        ]);
    }

    specs.extend([
        WeightSpec { name: "gpt.ln_f.weight".into(), shape: vec![1280], conv1d_input_major: false },
        WeightSpec { name: "gpt.ln_f.bias".into(), shape: vec![1280], conv1d_input_major: false },
        WeightSpec { name: "final_norm.weight".into(), shape: vec![1280], conv1d_input_major: false },
        WeightSpec { name: "final_norm.bias".into(), shape: vec![1280], conv1d_input_major: false },
        WeightSpec { name: "mel_head.weight".into(), shape: vec![8194, 1280], conv1d_input_major: false },
        WeightSpec { name: "mel_head.bias".into(), shape: vec![8194], conv1d_input_major: false },
    ]);
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_structures() {
        let device = Device::Cpu;
        let weights = HashMap::new();
        let _ = Weights {
            tensors: weights,
            device,
        };
        assert!(true);
    }

    #[test]
    fn greedy_contract_matches_official_architecture() {
        let specs = greedy_gpt_weight_specs();
        assert_eq!(specs.len(), 299);
        assert_eq!(specs[0].name, "mel_embedding.weight");
        assert_eq!(specs[0].shape, vec![8194, 1280]);

        let qkv = specs.iter()
            .find(|spec| spec.name == "gpt.h.0.attn.c_attn.weight")
            .unwrap();
        assert_eq!(qkv.shape, vec![1280, 3840]);
        assert!(qkv.conv1d_input_major);

        assert!(!specs.iter().any(|spec| spec.name == "wte.weight"));
        assert!(!specs.iter().any(|spec| spec.name == "wpe.weight"));
    }

    #[test]
    fn empty_weights_report_first_required_tensor() {
        let weights = Weights {
            tensors: HashMap::new(),
            device: Device::Cpu,
        };
        let error = weights.validate_greedy_contract().unwrap_err();
        assert!(matches!(
            error,
            WeightError::MissingWeight(name) if name == "mel_embedding.weight"
        ));
    }
}
