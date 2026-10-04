//! CPU-only Qwen3 emotion-text classifier used by IndexTTS-2.5.

use candle_core_qwen::{DType, Device, Tensor};
use candle_nn_qwen::VarBuilder;
use candle_transformers_qwen::models::qwen3::{Config, ModelForCausalLM};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use tokenizers::Tokenizer;

const EOS: u32 = 151_643;
const END_THINK: u32 = 151_668;
const LABELS: [(&str, &str); 8] = [
    ("高兴", "happy"),
    ("愤怒", "angry"),
    ("悲伤", "sad"),
    ("恐惧", "afraid"),
    ("反感", "disgusted"),
    ("低落", "melancholic"),
    ("惊讶", "surprised"),
    ("自然", "calm"),
];

#[derive(Debug, thiserror::Error)]
pub enum EmotionTextError {
    #[error("emotion classifier asset error: {0}")]
    Asset(String),
    #[error("emotion classifier inference error: {0}")]
    Inference(String),
    #[error("emotion classification cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, EmotionTextError>;

pub struct EmotionTextClassifier {
    tokenizer: Tokenizer,
    model: ModelForCausalLM,
    device: Device,
    max_new_tokens: usize,
}

impl EmotionTextClassifier {
    /// Load official `qwen0.6bemo4-merge` assets on CPU. This intentionally avoids GPU VRAM.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json")).map_err(asset)?)
                .map_err(asset)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(asset)?;
        let weights = model_dir.join("model.safetensors");
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .map_err(inference)?;
        let model = ModelForCausalLM::new(&config, vb).map_err(inference)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            max_new_tokens: 256,
        })
    }

    pub fn classify(&mut self, text: &str, cancelled: &AtomicBool) -> Result<[f32; 8]> {
        let response = self.classify_response(text, cancelled)?;
        Ok(parse_emotion_response(text, &response))
    }

    pub fn classify_response(&mut self, text: &str, cancelled: &AtomicBool) -> Result<String> {
        if text.trim().is_empty() {
            return Err(EmotionTextError::Inference("emotion text is empty".into()));
        }
        self.model.clear_kv_cache();
        let prompt = emotion_prompt(text);
        let encoding = self.tokenizer.encode(prompt, true).map_err(inference)?;
        let input_ids = encoding.get_ids();
        let mut input = Tensor::new(input_ids, &self.device)
            .and_then(|value| value.unsqueeze(0))
            .map_err(inference)?;
        let mut offset = 0;
        let mut generated = Vec::new();
        for _ in 0..self.max_new_tokens {
            if cancelled.load(Ordering::Acquire) {
                return Err(EmotionTextError::Cancelled);
            }
            let logits = self.model.forward(&input, offset).map_err(inference)?;
            offset += input.dim(1).map_err(inference)?;
            let logits = logits
                .squeeze(0)
                .and_then(|value| value.squeeze(0))
                .map_err(inference)?;
            let token = logits
                .argmax(0)
                .and_then(|value| value.to_scalar::<u32>())
                .map_err(inference)?;
            if token == EOS {
                break;
            }
            generated.push(token);
            input = Tensor::new(&[token], &self.device)
                .and_then(|value| value.unsqueeze(0))
                .map_err(inference)?;
        }
        let start = generated
            .iter()
            .rposition(|token| *token == END_THINK)
            .map_or(0, |index| index + 1);
        let response = self
            .tokenizer
            .decode(&generated[start..], true)
            .map_err(inference)?;
        Ok(response)
    }
}

pub fn emotion_prompt(text: &str) -> String {
    format!("System: 文本情感分类<|endoftext|>\nHuman: {text}<|endoftext|>\nAssistant:")
}

pub fn parse_emotion_response(input: &str, response: &str) -> [f32; 8] {
    let parsed = serde_json::from_str::<serde_json::Value>(response).ok();
    let mut output = [0.0; 8];
    for (index, (cn, en)) in LABELS.iter().enumerate() {
        let value = parsed
            .as_ref()
            .and_then(|root| root.get(cn).or_else(|| root.get(en)).and_then(json_score))
            .or_else(|| fallback_score(response, cn))
            .or_else(|| fallback_score(response, en));
        output[index] = value.unwrap_or(0.0).clamp(0.0, 1.2);
    }
    for (index, (cn, en)) in LABELS.iter().enumerate() {
        if response.trim().eq_ignore_ascii_case(cn) || response.trim().eq_ignore_ascii_case(en) {
            output[index] = 1.0;
        }
    }
    let lower = input.to_lowercase();
    if [
        "低落",
        "melancholy",
        "melancholic",
        "depression",
        "depressed",
        "gloomy",
    ]
    .iter()
    .any(|word| lower.contains(word))
    {
        output.swap(2, 5);
    }
    if output.iter().all(|value| *value <= 0.0) {
        output[7] = 1.0;
    }
    output
}

fn json_score(value: &serde_json::Value) -> Option<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .or_else(|| value.as_str().and_then(|text| text.parse::<f32>().ok()))
}

fn fallback_score(text: &str, label: &str) -> Option<f32> {
    let start = text.find(label)? + label.len();
    let tail = &text[start..];
    let colon = tail.find(':').or_else(|| tail.find('：'))?;
    let number: String = tail[colon + 1..]
        .chars()
        .skip_while(|ch| ch.is_whitespace() || *ch == '"')
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.' || *ch == '-')
        .collect();
    number.parse().ok()
}

fn asset(error: impl std::fmt::Display) -> EmotionTextError {
    EmotionTextError::Asset(error.to_string())
}
fn inference(error: impl std::fmt::Display) -> EmotionTextError {
    EmotionTextError::Inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_matches_official_template() {
        assert_eq!(
            emotion_prompt("温柔"),
            "System: 文本情感分类<|endoftext|>\nHuman: 温柔<|endoftext|>\nAssistant:"
        );
    }

    #[test]
    fn parses_json_aliases_clamps_and_defaults() {
        let values = parse_emotion_response("高兴", r#"{"高兴":0.5,"angry":2.0}"#);
        assert_eq!(values[0], 0.5);
        assert_eq!(values[1], 1.2);
        assert_eq!(parse_emotion_response("未知", "invalid")[7], 1.0);
    }

    #[test]
    fn melancholic_workaround_swaps_sadness() {
        let values = parse_emotion_response("非常低落", r#"{"悲伤":0.7}"#);
        assert_eq!(values[2], 0.0);
        assert_eq!(values[5], 0.7);
    }
}
