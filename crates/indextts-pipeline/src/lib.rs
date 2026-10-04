//! Complete IndexTTS-2.5 inference pipeline
//!
//! Orchestrates the full TTS pipeline:
//! ```text
//! Text -> Normalize -> Tokenize -> Speaker/Emotion Condition
//!      -> GPT Semantic Codes -> Semantic Codec
//!      -> Length Regulator -> S2Mel Diffusion
//!      -> BigVGAN -> PCM
//! ```
//!

mod manifest;
pub use manifest::{validate_model_manifest, ValidatedManifest};

use candle_core::{Device, Tensor as CandleTensor};
use indextts_audio::{
    campplus_fbank, process_reference_audio, process_reference_buffer, reference_mel,
    seamless_m4t_features,
};
use indextts_core::{
    AudioBuffer, DeviceConfig, GenerationConfig, Language, ModelConfig, Result as TtsResult,
    SemanticCodes,
};
use indextts_emotion_qwen::EmotionTextClassifier;
use indextts_gpt::{GptConfig, GreedyGenerator, IndexGpt};
use indextts_ort::{
    run_campplus, run_emotion_conditioner, run_gpt_conditioning, run_length_regulator,
    run_semantic_codec, run_wav2vec2bert, solve_cfm_bucketed_cancellable, BigVganBuckets,
    DitBuckets, OnnxModel, OnnxSession, Tensor, Wav2VecStats,
};
use indextts_text::{segment_text, TextNormalizer};
use indextts_tokenizer::{language_token_id, IndexTtsTokenizer};
use std::{
    ops::Range,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Instant,
};
use tracing::{info, instrument};

/// Features extracted from a user-provided reference voice.
#[derive(Debug, Clone)]
pub struct ReferenceConditioning {
    /// Normalized Wav2Vec2-BERT hidden state `[1, semantic_frames, 1024]`.
    pub semantic: Tensor,
    /// CAMPPlus speaker style `[1, 192]`.
    pub speaker_style: Tensor,
    /// Three GPT conditioning tokens `[1, 3, 1280]`.
    pub gpt_conditioning: Tensor,
    /// Voice-reference emotion embedding `[1, 1280]`, when packaged.
    pub voice_emotion: Option<Tensor>,
    /// Reference mel `[1, 80, mel_frames]`.
    pub reference_mel: Tensor,
    /// Length-regulated S2Mel prompt `[1, mel_frames, 512]`.
    pub prompt_condition: Tensor,
}

/// Runtime for the two reference-audio encoder branches.
#[derive(Debug)]
pub struct PreparedEmotion {
    pub embedding: Tensor,
}

#[derive(Debug)]
pub struct ReferenceEncoder {
    wav2vec: OnnxSession,
    campplus: OnnxSession,
    gpt_conditioning: OnnxSession,
    emotion_conditioner: Option<OnnxSession>,
    length_regulator: OnnxSession,
    stats: Wav2VecStats,
}

impl ReferenceEncoder {
    pub fn load(model_dir: &Path) -> TtsResult<Self> {
        Self::load_with_device(model_dir, None)
    }

    fn load_with_device(model_dir: &Path, cuda_device: Option<usize>) -> TtsResult<Self> {
        let wav2vec =
            OnnxSession::load_with_device(&OnnxModel::Wav2Vec2Bert.path(model_dir), cuda_device)?;
        wav2vec.validate_contract(
            &[("input_features", "Float32"), ("attention_mask", "Int64")],
            &[("hidden_states_17", "Float32")],
        )?;
        let campplus =
            OnnxSession::load_with_device(&OnnxModel::Campplus.path(model_dir), cuda_device)?;
        campplus.validate_contract(&[("x", "Float32")], &[("style", "Float32")])?;
        let gpt_conditioning = OnnxSession::load_with_device(
            &OnnxModel::GptConditioning.path(model_dir),
            cuda_device,
        )?;
        gpt_conditioning.validate_contract(
            &[
                ("speaker_style", "Float32"),
                ("semantic_features", "Float32"),
            ],
            &[("conditioning", "Float32")],
        )?;
        let emotion_conditioner = OnnxSession::load_with_device(
            &OnnxModel::EmotionConditioner.path(model_dir),
            cuda_device,
        )?;
        emotion_conditioner.validate_contract(
            &[("semantic_features", "Float32")],
            &[("emotion", "Float32")],
        )?;
        let length_regulator = OnnxSession::load_with_device(
            &OnnxModel::LengthRegulator.path(model_dir),
            cuda_device,
        )?;
        length_regulator.validate_contract(
            &[("semantic_features", "Float32"), ("target_length", "Int64")],
            &[("condition", "Float32")],
        )?;
        Ok(Self {
            wav2vec,
            campplus,
            gpt_conditioning,
            emotion_conditioner: Some(emotion_conditioner),
            length_regulator,
            stats: Wav2VecStats::load(&model_dir.join("wav2vec2bert_stats.safetensors"))?,
        })
    }

    pub fn encode(&self, reference_audio: &Path) -> TtsResult<ReferenceConditioning> {
        let (audio_16k, audio_22k) = process_reference_audio(reference_audio)?;
        self.encode_processed(&audio_16k, &audio_22k)
    }

    pub fn encode_buffer(&self, audio: AudioBuffer) -> TtsResult<ReferenceConditioning> {
        let (audio_16k, audio_22k) = process_reference_buffer(audio)?;
        self.encode_processed(&audio_16k, &audio_22k)
    }

    fn encode_processed(
        &self,
        audio_16k: &AudioBuffer,
        audio_22k: &AudioBuffer,
    ) -> TtsResult<ReferenceConditioning> {
        let seamless = seamless_m4t_features(audio_16k)?;
        let semantic = run_wav2vec2bert(
            &self.wav2vec,
            Tensor::new(
                seamless.input_features,
                vec![1, seamless.frames as i64, 160],
            ),
            Tensor::new_i64(seamless.attention_mask, vec![1, seamless.frames as i64]),
        )?;
        let semantic = self.stats.normalize(semantic)?;

        let fbank = campplus_fbank(audio_16k)?;
        let fbank_frames = fbank.len() / 80;
        let speaker_style = run_campplus(
            &self.campplus,
            Tensor::new(fbank, vec![1, fbank_frames as i64, 80]),
        )?;
        let gpt_conditioning = run_gpt_conditioning(
            &self.gpt_conditioning,
            speaker_style.clone(),
            semantic.clone(),
        )?;
        let voice_emotion = self
            .emotion_conditioner
            .as_ref()
            .map(|session| run_emotion_conditioner(session, semantic.clone()))
            .transpose()?;
        let mel = reference_mel(audio_22k)?;
        let mel_frames = mel.len() / 80;
        let reference_mel = Tensor::new(mel, vec![1, 80, mel_frames as i64]);
        let prompt_condition =
            run_length_regulator(&self.length_regulator, semantic.clone(), mel_frames)?;
        if semantic.shape().first() != Some(&1) || semantic.shape().last() != Some(&1024) {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "invalid semantic conditioning shape {:?}",
                semantic.shape()
            )));
        }
        if speaker_style.shape() != [1, 192] {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "invalid speaker style shape {:?}",
                speaker_style.shape()
            )));
        }
        if gpt_conditioning.shape() != [1, 3, 1280] {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "invalid GPT conditioning shape {:?}",
                gpt_conditioning.shape()
            )));
        }
        if prompt_condition.shape() != [1, mel_frames as i64, 512] {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "invalid prompt condition shape {:?}",
                prompt_condition.shape()
            )));
        }
        Ok(ReferenceConditioning {
            semantic,
            speaker_style,
            gpt_conditioning,
            voice_emotion,
            reference_mel,
            prompt_condition,
        })
    }

    pub fn encode_emotion(&self, reference_audio: &Path) -> TtsResult<PreparedEmotion> {
        let session = self.emotion_conditioner.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "model package has no emotion-conditioner component".into(),
            )
        })?;
        let (audio_16k, _) = process_reference_audio(reference_audio)?;
        let seamless = seamless_m4t_features(&audio_16k)?;
        let semantic = run_wav2vec2bert(
            &self.wav2vec,
            Tensor::new(
                seamless.input_features,
                vec![1, seamless.frames as i64, 160],
            ),
            Tensor::new_i64(seamless.attention_mask, vec![1, seamless.frames as i64]),
        )?;
        let embedding = run_emotion_conditioner(session, self.stats.normalize(semantic)?)?;
        if embedding.shape() != [1, 1280] {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "invalid emotion embedding shape {:?}",
                embedding.shape()
            )));
        }
        Ok(PreparedEmotion { embedding })
    }
}

#[derive(Debug)]
struct EmotionPrototypeBank {
    speaker: Vec<f32>,
    emotion: Vec<f32>,
    offsets: Vec<usize>,
}

impl EmotionPrototypeBank {
    fn load(path: &Path) -> TtsResult<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?;
        let speaker = tensors.get("speaker_prototypes").ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "emotion prototypes missing speaker_prototypes".into(),
            )
        })?;
        let emotion = tensors.get("emotion_prototypes").ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "emotion prototypes missing emotion_prototypes".into(),
            )
        })?;
        let offsets = tensors.get("group_offsets").ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "emotion prototypes missing group_offsets".into(),
            )
        })?;
        if speaker.dims() != [73, 192] || emotion.dims() != [73, 1280] || offsets.dims() != [9] {
            return Err(indextts_core::IndexTtsError::InvalidModel(
                "invalid emotion prototype shapes".into(),
            ));
        }
        Ok(Self {
            speaker: speaker
                .flatten_all()
                .and_then(|value| value.to_vec1::<f32>())
                .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?,
            emotion: emotion
                .flatten_all()
                .and_then(|value| value.to_vec1::<f32>())
                .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?,
            offsets: offsets
                .to_vec1::<i64>()
                .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
        })
    }

    fn mix(
        &self,
        speaker_style: &Tensor,
        voice_emotion: &Tensor,
        weights: &[f32; 8],
        strength: f32,
    ) -> TtsResult<PreparedEmotion> {
        if speaker_style.shape() != [1, 192] || voice_emotion.shape() != [1, 1280] {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "invalid emotion vector inputs".into(),
            ));
        }
        if !strength.is_finite()
            || !(0.0..=1.0).contains(&strength)
            || weights
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "invalid emotion vector or strength".into(),
            ));
        }
        let scaled: Vec<f32> = weights
            .iter()
            .map(|value| (value * strength * 10_000.0).trunc() / 10_000.0)
            .collect();
        let total: f32 = scaled.iter().sum();
        let style = speaker_style.as_slice();
        let mut output: Vec<f32> = voice_emotion
            .as_slice()
            .iter()
            .map(|value| value * (1.0 - total))
            .collect();
        let style_norm = style
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for (group, group_weight) in scaled.iter().enumerate().take(8) {
            let mut best = self.offsets[group];
            let mut best_score = f32::NEG_INFINITY;
            for row in self.offsets[group]..self.offsets[group + 1] {
                let candidate = &self.speaker[row * 192..(row + 1) * 192];
                let norm = candidate
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                let score = style.iter().zip(candidate).map(|(a, b)| a * b).sum::<f32>()
                    / (style_norm * norm);
                if score > best_score {
                    best_score = score;
                    best = row;
                }
            }
            let prototype = &self.emotion[best * 1280..(best + 1) * 1280];
            for (value, prototype) in output.iter_mut().zip(prototype) {
                *value += group_weight * prototype;
            }
        }
        Ok(PreparedEmotion {
            embedding: Tensor::new(output, vec![1, 1280]),
        })
    }
}

/// Runtime for reference-audio + text to greedy semantic codes.
pub struct SemanticRuntime {
    reference: ReferenceEncoder,
    tokenizer: IndexTtsTokenizer,
    normalizer: TextNormalizer,
    gpt: IndexGpt,
    semantic_codec: OnnxSession,
    length_regulator: OnnxSession,
    dit_buckets: DitBuckets,
    bigvgan_buckets: BigVganBuckets,
    emotion_prototypes: Option<EmotionPrototypeBank>,
    emotion_text: Option<EmotionTextClassifier>,
    emotion_text_dir: std::path::PathBuf,
}

impl SemanticRuntime {
    pub fn load(model_dir: &Path) -> TtsResult<Self> {
        Self::load_with_device(
            model_dir,
            indextts_core::DeviceConfig::new(indextts_core::DeviceKind::Cpu, 0),
        )
    }

    pub fn load_with_device(
        model_dir: &Path,
        device_config: indextts_core::DeviceConfig,
    ) -> TtsResult<Self> {
        let cuda_device = match device_config.kind {
            indextts_core::DeviceKind::Cpu => None,
            indextts_core::DeviceKind::Cuda => Some(device_config.index),
            indextts_core::DeviceKind::Auto => {
                #[cfg(feature = "cuda")]
                {
                    Some(device_config.index)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    None
                }
            }
        };
        let device = if let Some(index) = cuda_device {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(index).map_err(|error| {
                    indextts_core::IndexTtsError::BackendFailure(error.to_string())
                })?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = index;
                return Err(indextts_core::IndexTtsError::BackendFailure(
                    "CUDA requested, but indextts-pipeline was built without the cuda feature"
                        .into(),
                ));
            }
        } else {
            Device::Cpu
        };
        let mut gpt = IndexGpt::new(GptConfig::default(), device)
            .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?;
        gpt.load_weights(&model_dir.join("gpt.safetensors"))
            .map_err(|error| indextts_core::IndexTtsError::InvalidModel(error.to_string()))?;
        Ok(Self {
            reference: ReferenceEncoder::load_with_device(model_dir, cuda_device)?,
            tokenizer: IndexTtsTokenizer::from_dir(model_dir)?,
            normalizer: TextNormalizer::new(),
            gpt,
            semantic_codec: OnnxSession::load_with_device(
                &OnnxModel::SemanticCodec.path(model_dir),
                cuda_device,
            )?,
            length_regulator: OnnxSession::load_with_device(
                &OnnxModel::LengthRegulator.path(model_dir),
                cuda_device,
            )?,
            // Loading every exported bucket eagerly duplicates ORT model weights and can
            // exhaust desktop VRAM. Keep only the buckets verified safe in production.
            dit_buckets: DitBuckets::load_with_device(model_dir, &[256, 512, 1024], cuda_device)?,
            // BigVGAN 1024 remains excluded because it crashes on Windows.
            bigvgan_buckets: BigVganBuckets::load_with_device(model_dir, &[256, 512], cuda_device)?,
            emotion_prototypes: {
                let path = model_dir.join("emotion-prototypes.safetensors");
                if path.is_file() {
                    Some(EmotionPrototypeBank::load(&path)?)
                } else {
                    None
                }
            },
            emotion_text: None,
            emotion_text_dir: model_dir.join("qwen0.6bemo4-merge"),
        })
    }

    pub fn generate(
        &mut self,
        text: &str,
        language: Language,
        reference_audio: &Path,
        max_tokens: usize,
    ) -> TtsResult<SemanticCodes> {
        if text.trim().is_empty() {
            return Err(indextts_core::IndexTtsError::InvalidText(
                "text is empty".into(),
            ));
        }
        let reference = self.reference.encode(reference_audio)?;
        self.generate_with_reference(text, language, &reference, max_tokens)
    }

    pub fn generate_with_reference(
        &mut self,
        text: &str,
        language: Language,
        reference: &ReferenceConditioning,
        max_tokens: usize,
    ) -> TtsResult<SemanticCodes> {
        let cancelled = AtomicBool::new(false);
        self.generate_with_reference_cancellable(text, language, reference, max_tokens, &cancelled)
    }

    pub fn generate_with_reference_cancellable(
        &mut self,
        text: &str,
        language: Language,
        reference: &ReferenceConditioning,
        max_tokens: usize,
        cancelled: &AtomicBool,
    ) -> TtsResult<SemanticCodes> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if text.trim().is_empty() {
            return Err(indextts_core::IndexTtsError::InvalidText(
                "text is empty".into(),
            ));
        }
        let normalized = self.normalizer.normalize(text, language)?;
        self.generate_normalized_with_reference_cancellable(
            &normalized,
            language,
            reference,
            max_tokens,
            cancelled,
        )
    }

    fn generate_normalized_with_reference_cancellable(
        &mut self,
        normalized_text: &str,
        language: Language,
        reference: &ReferenceConditioning,
        max_tokens: usize,
        cancelled: &AtomicBool,
    ) -> TtsResult<SemanticCodes> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if normalized_text.trim().is_empty() {
            return Err(indextts_core::IndexTtsError::InvalidText(
                "text is empty after normalization".into(),
            ));
        }
        let tokens = self.tokenizer.tokenize_for_gpt(normalized_text, language)?;
        let conditioning = ort_f32_to_candle(&reference.gpt_conditioning, self.gpt.device())?;
        let prefix = self
            .gpt
            .build_prefix(&conditioning, &tokens, language_token_id(language))
            .map_err(|error| indextts_core::IndexTtsError::BackendFailure(error.to_string()))?;
        let dummy_text = CandleTensor::zeros((1, 0), candle_core::DType::U32, self.gpt.device())
            .map_err(|error| indextts_core::IndexTtsError::BackendFailure(error.to_string()))?;
        let (input_ids, attention_mask) = self
            .gpt
            .prepare_inputs(&prefix, &dummy_text, None)
            .map_err(|error| indextts_core::IndexTtsError::BackendFailure(error.to_string()))?;
        let input_len = input_ids
            .dim(1)
            .map_err(|error| indextts_core::IndexTtsError::BackendFailure(error.to_string()))?;
        let mut generator = GreedyGenerator::new(self.gpt.device(), input_len + max_tokens + 1);
        let output = generator
            .generate_cancellable(
                &self.gpt,
                &input_ids,
                Some(&attention_mask),
                input_len + max_tokens - 1,
                cancelled,
            )
            .map_err(|error| {
                if cancelled.load(Ordering::Acquire) {
                    indextts_core::IndexTtsError::Cancelled
                } else {
                    indextts_core::IndexTtsError::BackendFailure(error.to_string())
                }
            })?;
        if output.tokens.is_empty() {
            return Err(indextts_core::IndexTtsError::EmptySemanticCodes);
        }
        Ok(SemanticCodes::new(output.tokens))
    }

    pub fn generate_mel(
        &self,
        codes: &SemanticCodes,
        reference: &ReferenceConditioning,
        duration_factor: f32,
        seed: u64,
    ) -> TtsResult<Tensor> {
        let cancelled = AtomicBool::new(false);
        self.generate_mel_cancellable(codes, reference, duration_factor, seed, &cancelled)
    }

    pub fn generate_mel_cancellable(
        &self,
        codes: &SemanticCodes,
        reference: &ReferenceConditioning,
        duration_factor: f32,
        seed: u64,
        cancelled: &AtomicBool,
    ) -> TtsResult<Tensor> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if !(0.5..=2.0).contains(&duration_factor) {
            return Err(indextts_core::IndexTtsError::BackendFailure(format!(
                "duration factor {duration_factor} is outside [0.5, 2.0]"
            )));
        }
        let decoded = run_semantic_codec(&self.semantic_codec, &codes.tokens)?;
        let decoded_frames = decoded.shape()[1] as usize;
        let generated_frames = (decoded_frames as f32 * 1.72 * duration_factor) as usize;
        let generated_condition =
            run_length_regulator(&self.length_regulator, decoded, generated_frames)?;
        let condition = concat_conditions(&reference.prompt_condition, &generated_condition)?;
        let full_mel = solve_cfm_bucketed_cancellable(
            &self.dit_buckets,
            &condition,
            &reference.reference_mel,
            &reference.speaker_style,
            25,
            0.7,
            seed,
            cancelled,
        )?;
        crop_reference_mel(&full_mel, reference.reference_mel.shape()[2] as usize)
    }

    pub fn vocode(&self, mel: &Tensor) -> TtsResult<AudioBuffer> {
        let waveform = self.bigvgan_buckets.synthesize(mel)?;
        let samples: Vec<f32> = waveform
            .as_slice()
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect();
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(indextts_core::IndexTtsError::InvalidAudio(
                "BigVGAN returned empty or non-finite audio".into(),
            ));
        }
        Ok(AudioBuffer::new(samples, 22_050))
    }
}

pub fn apply_emotion_reference(
    conditioning: &Tensor,
    voice_emotion: &Tensor,
    reference_emotion: &Tensor,
    strength: f32,
) -> TtsResult<Tensor> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(indextts_core::IndexTtsError::BackendFailure(
            "emotion strength must be finite and in [0, 1]".into(),
        ));
    }
    if conditioning.shape() != [1, 3, 1280]
        || voice_emotion.shape() != [1, 1280]
        || reference_emotion.shape() != [1, 1280]
    {
        return Err(indextts_core::IndexTtsError::BackendFailure(format!(
            "invalid emotion conditioning shapes {:?}, {:?}, {:?}",
            conditioning.shape(),
            voice_emotion.shape(),
            reference_emotion.shape()
        )));
    }
    let mut data = conditioning.as_slice().to_vec();
    for (index, value) in data.iter_mut().take(1280).enumerate() {
        *value +=
            strength * (reference_emotion.as_slice()[index] - voice_emotion.as_slice()[index]);
    }
    Ok(Tensor::new(data, vec![1, 3, 1280]))
}

fn concat_conditions(left: &Tensor, right: &Tensor) -> TtsResult<Tensor> {
    let left_frames = left.shape()[1] as usize;
    let right_frames = right.shape()[1] as usize;
    if left.shape() != [1, left_frames as i64, 512]
        || right.shape() != [1, right_frames as i64, 512]
    {
        return Err(indextts_core::IndexTtsError::BackendFailure(format!(
            "invalid condition shapes {:?} and {:?}",
            left.shape(),
            right.shape()
        )));
    }
    let mut data = Vec::with_capacity((left_frames + right_frames) * 512);
    data.extend_from_slice(left.as_slice());
    data.extend_from_slice(right.as_slice());
    Ok(Tensor::new(
        data,
        vec![1, (left_frames + right_frames) as i64, 512],
    ))
}

fn crop_reference_mel(mel: &Tensor, prompt_frames: usize) -> TtsResult<Tensor> {
    let total_frames = mel.shape()[2] as usize;
    if mel.shape()[..2] != [1, 80] || prompt_frames >= total_frames {
        return Err(indextts_core::IndexTtsError::BackendFailure(format!(
            "cannot crop mel {:?} at frame {prompt_frames}",
            mel.shape()
        )));
    }
    let generated_frames = total_frames - prompt_frames;
    let mut data = Vec::with_capacity(80 * generated_frames);
    for channel in 0..80 {
        let offset = channel * total_frames + prompt_frames;
        data.extend_from_slice(&mel.as_slice()[offset..offset + generated_frames]);
    }
    Ok(Tensor::new(data, vec![1, 80, generated_frames as i64]))
}

fn ort_f32_to_candle(tensor: &Tensor, device: &Device) -> TtsResult<CandleTensor> {
    let shape: Vec<usize> = tensor
        .shape()
        .iter()
        .map(|dimension| *dimension as usize)
        .collect();
    CandleTensor::from_slice(tensor.as_slice(), shape, device)
        .map_err(|error| indextts_core::IndexTtsError::BackendFailure(error.to_string()))
}

#[derive(Debug, Clone, Default)]
pub struct GenerationDiagnostics {
    pub semantic_token_count: u32,
    pub generated_seconds: f32,
    pub reference_encode_ms: f32,
    pub gpt_ms: f32,
    pub semantic_codec_ms: f32,
    pub s2mel_ms: f32,
    pub bigvgan_ms: f32,
    pub total_ms: f32,
    pub peak: f32,
    pub rms: f32,
    pub silence_ratio: f32,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub audio: AudioBuffer,
    pub diagnostics: GenerationDiagnostics,
}

/// Segmentation and joining options for deterministic long-text synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTextConfig {
    /// Maximum number of Unicode scalar values in one segment.
    pub max_chars: usize,
    /// Silence inserted between adjacent segments. An unsigned value makes it nonnegative.
    pub pause_ms: u64,
}

impl Default for LongTextConfig {
    fn default() -> Self {
        Self {
            max_chars: 200,
            pause_ms: 200,
        }
    }
}

/// Metadata for one synthesized segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongTextSegment {
    /// Character range in `LongTextSynthesisResult::normalized_text`.
    pub text_range: Range<usize>,
    pub semantic_token_count: u32,
    /// Start of segment speech in the aggregate 22,050 Hz mono buffer.
    pub audio_offset_samples: usize,
    /// Segment speech length, excluding inter-segment silence.
    pub audio_duration_samples: usize,
    pub seed: u64,
}

/// Audio, aggregate diagnostics, and segment metadata from long-text synthesis.
#[derive(Debug, Clone)]
pub struct LongTextSynthesisResult {
    pub audio: AudioBuffer,
    pub diagnostics: GenerationDiagnostics,
    pub normalized_text: String,
    pub segments: Vec<LongTextSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedSegment {
    text: String,
    text_range: Range<usize>,
    seed: u64,
}

/// Normalize-independent deterministic planning. Seed addition is checked rather than wrapping.
fn plan_long_text(
    normalized_text: &str,
    max_chars: usize,
    base_seed: u64,
) -> TtsResult<Vec<PlannedSegment>> {
    if normalized_text.trim().is_empty() {
        return Err(indextts_core::IndexTtsError::InvalidText(
            "text is empty after normalization".into(),
        ));
    }
    segment_text(normalized_text, max_chars)?
        .into_iter()
        .enumerate()
        .map(|(index, segment)| {
            let seed_offset = u64::try_from(index).map_err(|_| {
                indextts_core::IndexTtsError::InvalidText(
                    "long-text segment index does not fit in u64".into(),
                )
            })?;
            let seed = base_seed.checked_add(seed_offset).ok_or_else(|| {
                indextts_core::IndexTtsError::InvalidText(format!(
                    "long-text seed overflow at segment {index}"
                ))
            })?;
            Ok(PlannedSegment {
                text: segment.text,
                text_range: segment.start_char..segment.end_char,
                seed,
            })
        })
        .collect()
}

fn pause_samples(pause_ms: u64) -> TtsResult<usize> {
    let samples = u128::from(pause_ms) * 22_050 / 1_000;
    usize::try_from(samples).map_err(|_| {
        indextts_core::IndexTtsError::InvalidAudio(
            "long-text pause is too large for this platform".into(),
        )
    })
}

fn concatenate_long_text(
    planned: &[PlannedSegment],
    results: Vec<SynthesisResult>,
    pause_ms: u64,
) -> TtsResult<(AudioBuffer, GenerationDiagnostics, Vec<LongTextSegment>)> {
    if planned.len() != results.len() {
        return Err(indextts_core::IndexTtsError::InvalidAudio(
            "long-text plan/result count mismatch".into(),
        ));
    }
    let pause = pause_samples(pause_ms)?;
    let speech_samples = results.iter().try_fold(0usize, |total, result| {
        if result.audio.sample_rate != 22_050 || result.audio.channels != 1 {
            return Err(indextts_core::IndexTtsError::InvalidAudio(format!(
                "long-text segment audio must be 22050 Hz mono, got {} Hz and {} channels",
                result.audio.sample_rate, result.audio.channels
            )));
        }
        total
            .checked_add(result.audio.samples.len())
            .ok_or_else(|| {
                indextts_core::IndexTtsError::InvalidAudio("long-text audio is too large".into())
            })
    })?;
    let total_pause = pause
        .checked_mul(results.len().saturating_sub(1))
        .ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidAudio("long-text audio is too large".into())
        })?;
    let capacity = speech_samples.checked_add(total_pause).ok_or_else(|| {
        indextts_core::IndexTtsError::InvalidAudio("long-text audio is too large".into())
    })?;
    let mut samples = Vec::with_capacity(capacity);
    let mut segments = Vec::with_capacity(results.len());
    let mut diagnostics = GenerationDiagnostics::default();

    for (index, (plan, result)) in planned.iter().zip(results).enumerate() {
        if index != 0 {
            samples.resize(samples.len() + pause, 0.0);
        }
        let audio_offset_samples = samples.len();
        let audio_duration_samples = result.audio.samples.len();
        diagnostics.semantic_token_count = diagnostics
            .semantic_token_count
            .saturating_add(result.diagnostics.semantic_token_count);
        diagnostics.reference_encode_ms += result.diagnostics.reference_encode_ms;
        diagnostics.gpt_ms += result.diagnostics.gpt_ms;
        diagnostics.semantic_codec_ms += result.diagnostics.semantic_codec_ms;
        diagnostics.s2mel_ms += result.diagnostics.s2mel_ms;
        diagnostics.bigvgan_ms += result.diagnostics.bigvgan_ms;
        diagnostics.total_ms += result.diagnostics.total_ms;
        samples.extend(result.audio.samples);
        segments.push(LongTextSegment {
            text_range: plan.text_range.clone(),
            semantic_token_count: result.diagnostics.semantic_token_count,
            audio_offset_samples,
            audio_duration_samples,
            seed: plan.seed,
        });
    }

    let peak = samples
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let (rms, silence_ratio) = if samples.is_empty() {
        (0.0, 0.0)
    } else {
        (
            (samples.iter().map(|value| value * value).sum::<f32>() / samples.len() as f32).sqrt(),
            samples.iter().filter(|value| value.abs() < 1e-4).count() as f32 / samples.len() as f32,
        )
    };
    diagnostics.generated_seconds = samples.len() as f32 / 22_050.0;
    diagnostics.peak = peak;
    diagnostics.rms = rms;
    diagnostics.silence_ratio = silence_ratio;
    diagnostics.seed = planned.first().map_or(0, |segment| segment.seed);
    Ok((AudioBuffer::new(samples, 22_050), diagnostics, segments))
}

/// End-to-end IndexTTS pipeline.
pub struct IndexTtsPipeline {
    config: ModelConfig,
    runtime: Option<Mutex<SemanticRuntime>>,
    manifest: Option<ValidatedManifest>,
}

impl IndexTtsPipeline {
    /// Create a new pipeline
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            runtime: None,
            manifest: None,
        }
    }

    /// Load all runtime models.
    #[instrument(skip(self))]
    pub fn load(&mut self) -> TtsResult<()> {
        info!(
            "Loading IndexTTS-2.5 models from {:?}",
            self.config.model_dir
        );
        let manifest = validate_model_manifest(&self.config.model_dir)
            .map_err(indextts_core::IndexTtsError::InvalidModel)?;
        self.runtime = Some(Mutex::new(SemanticRuntime::load_with_device(
            &self.config.model_dir,
            self.config.device,
        )?));
        self.manifest = Some(manifest);
        info!("Model loading complete");
        Ok(())
    }

    pub fn prepare_voice(&self, reference_audio_path: &Path) -> TtsResult<ReferenceConditioning> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        runtime.reference.encode(reference_audio_path)
    }

    pub fn prepare_voice_buffer(&self, audio: AudioBuffer) -> TtsResult<ReferenceConditioning> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        runtime.reference.encode_buffer(audio)
    }

    pub fn prepare_emotion_reference(
        &self,
        reference_audio_path: &Path,
    ) -> TtsResult<PreparedEmotion> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        runtime.reference.encode_emotion(reference_audio_path)
    }

    pub fn prepare_emotion_text(
        &self,
        voice: &ReferenceConditioning,
        text: &str,
        strength: f32,
        cancelled: &AtomicBool,
    ) -> TtsResult<PreparedEmotion> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let mut runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        if runtime.emotion_text.is_none() {
            runtime.emotion_text = Some(
                EmotionTextClassifier::load(&runtime.emotion_text_dir).map_err(|error| {
                    indextts_core::IndexTtsError::InvalidModel(error.to_string())
                })?,
            );
        }
        let weights = runtime
            .emotion_text
            .as_mut()
            .expect("initialized above")
            .classify(text, cancelled)
            .map_err(|error| match error {
                indextts_emotion_qwen::EmotionTextError::Cancelled => {
                    indextts_core::IndexTtsError::Cancelled
                }
                error => indextts_core::IndexTtsError::BackendFailure(error.to_string()),
            })?;
        let bank = runtime.emotion_prototypes.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "model package has no emotion prototype bank".into(),
            )
        })?;
        let voice_emotion = voice.voice_emotion.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "prepared voice has no emotion embedding".into(),
            )
        })?;
        bank.mix(&voice.speaker_style, voice_emotion, &weights, strength)
    }

    pub fn prepare_emotion_vector(
        &self,
        voice: &ReferenceConditioning,
        weights: &[f32; 8],
        strength: f32,
    ) -> TtsResult<PreparedEmotion> {
        if weights.iter().sum::<f32>() * strength > 0.8001 {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "emotion vector sum after strength must not exceed 0.8".into(),
            ));
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        let bank = runtime.emotion_prototypes.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "model package has no emotion prototype bank".into(),
            )
        })?;
        let voice_emotion = voice.voice_emotion.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "prepared voice has no emotion embedding".into(),
            )
        })?;
        bank.mix(&voice.speaker_style, voice_emotion, weights, strength)
    }

    /// Synthesize speech from text and a reference voice.
    #[instrument(skip(self, reference_audio_path))]
    pub fn synthesize(
        &self,
        text: &str,
        reference_audio_path: &Path,
        gen_config: &GenerationConfig,
    ) -> TtsResult<AudioBuffer> {
        let cancelled = AtomicBool::new(false);
        self.synthesize_cancellable(text, reference_audio_path, gen_config, &cancelled)
    }

    pub fn synthesize_cancellable(
        &self,
        text: &str,
        reference_audio_path: &Path,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<AudioBuffer> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if gen_config.do_sample || gen_config.num_beams != 1 {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "the end-to-end runtime currently supports greedy generation only".into(),
            ));
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let mut runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        let reference = runtime.reference.encode(reference_audio_path)?;
        Self::synthesize_with_runtime(&mut runtime, text, &reference, gen_config, cancelled)
    }

    pub fn synthesize_prepared_cancellable(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<AudioBuffer> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if gen_config.do_sample || gen_config.num_beams != 1 {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "the end-to-end runtime currently supports greedy generation only".into(),
            ));
        }
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let mut runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        Self::synthesize_with_runtime(&mut runtime, text, reference, gen_config, cancelled)
    }

    pub fn synthesize_prepared_with_emotion_cancellable(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        emotion: &PreparedEmotion,
        strength: f32,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<AudioBuffer> {
        let voice_emotion = reference.voice_emotion.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "prepared voice has no emotion embedding; re-prepare it with an emotion-enabled model package".into(),
            )
        })?;
        let mut adjusted = reference.clone();
        adjusted.gpt_conditioning = apply_emotion_reference(
            &reference.gpt_conditioning,
            voice_emotion,
            &emotion.embedding,
            strength,
        )?;
        self.synthesize_prepared_cancellable(text, &adjusted, gen_config, cancelled)
    }

    pub fn synthesize_prepared_with_emotion_result_cancellable(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        emotion: &PreparedEmotion,
        strength: f32,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<SynthesisResult> {
        let voice_emotion = reference.voice_emotion.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel(
                "prepared voice has no emotion embedding".into(),
            )
        })?;
        let mut adjusted = reference.clone();
        adjusted.gpt_conditioning = apply_emotion_reference(
            &reference.gpt_conditioning,
            voice_emotion,
            &emotion.embedding,
            strength,
        )?;
        self.synthesize_prepared_result_cancellable(text, &adjusted, gen_config, cancelled)
    }

    pub fn synthesize_prepared_result_cancellable(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<SynthesisResult> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let mut runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        Self::synthesize_prepared_result_with_runtime(
            &mut runtime,
            text,
            false,
            reference,
            gen_config,
            cancelled,
        )
    }

    /// Deterministically normalize, segment, synthesize, and join long text.
    pub fn synthesize_long_text_prepared_result(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        long_config: &LongTextConfig,
    ) -> TtsResult<LongTextSynthesisResult> {
        let cancelled = AtomicBool::new(false);
        self.synthesize_long_text_prepared_result_cancellable(
            text,
            reference,
            gen_config,
            long_config,
            &cancelled,
        )
    }

    /// Cancellable deterministic long-text synthesis.
    ///
    /// The prepared voice and loaded runtime are reused for every segment. Segment `i` uses
    /// `gen_config.seed + i`; overflow is rejected rather than wrapped.
    pub fn synthesize_long_text_prepared_result_cancellable(
        &self,
        text: &str,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        long_config: &LongTextConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<LongTextSynthesisResult> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        if gen_config.do_sample || gen_config.num_beams != 1 {
            return Err(indextts_core::IndexTtsError::BackendFailure(
                "the end-to-end runtime currently supports greedy generation only".into(),
            ));
        }
        let total_start = Instant::now();
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            indextts_core::IndexTtsError::InvalidModel("pipeline is not loaded".into())
        })?;
        let mut runtime = runtime.lock().map_err(|_| {
            indextts_core::IndexTtsError::BackendFailure(
                "pipeline runtime lock was poisoned".into(),
            )
        })?;
        let normalized_text = runtime.normalizer.normalize(text, gen_config.language)?;
        let planned = plan_long_text(&normalized_text, long_config.max_chars, gen_config.seed)?;
        let mut results = Vec::with_capacity(planned.len());
        for (index, segment) in planned.iter().enumerate() {
            // This explicit boundary check avoids starting another segment after cancellation.
            if cancelled.load(Ordering::Acquire) {
                return Err(indextts_core::IndexTtsError::Cancelled);
            }
            let mut segment_config = gen_config.clone();
            segment_config.seed = segment.seed;
            let result = Self::synthesize_prepared_result_with_runtime(
                &mut runtime,
                &segment.text,
                true,
                reference,
                &segment_config,
                cancelled,
            )
            .map_err(|error| match error {
                indextts_core::IndexTtsError::Cancelled => indextts_core::IndexTtsError::Cancelled,
                error => indextts_core::IndexTtsError::BackendFailure(format!(
                    "long-text segment {index} failed: {error}"
                )),
            })?;
            results.push(result);
        }
        let (audio, mut diagnostics, segments) =
            concatenate_long_text(&planned, results, long_config.pause_ms)?;
        diagnostics.total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
        Ok(LongTextSynthesisResult {
            audio,
            diagnostics,
            normalized_text,
            segments,
        })
    }

    fn synthesize_prepared_result_with_runtime(
        runtime: &mut SemanticRuntime,
        text: &str,
        text_is_normalized: bool,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<SynthesisResult> {
        let total_start = Instant::now();
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        let gpt_start = Instant::now();
        let max_tokens = gen_config.max_length.unwrap_or(1815);
        let codes = if text_is_normalized {
            runtime.generate_normalized_with_reference_cancellable(
                text,
                gen_config.language,
                reference,
                max_tokens,
                cancelled,
            )?
        } else {
            runtime.generate_with_reference_cancellable(
                text,
                gen_config.language,
                reference,
                max_tokens,
                cancelled,
            )?
        };
        let gpt_ms = gpt_start.elapsed().as_secs_f32() * 1000.0;
        let codec_start = Instant::now();
        let decoded = run_semantic_codec(&runtime.semantic_codec, &codes.tokens)?;
        let decoded_frames = decoded.shape()[1] as usize;
        let generated_frames = (decoded_frames as f32 * 1.72 * gen_config.duration_factor) as usize;
        let generated_condition =
            run_length_regulator(&runtime.length_regulator, decoded, generated_frames)?;
        let semantic_codec_ms = codec_start.elapsed().as_secs_f32() * 1000.0;
        let s2mel_start = Instant::now();
        let condition = concat_conditions(&reference.prompt_condition, &generated_condition)?;
        let full_mel = solve_cfm_bucketed_cancellable(
            &runtime.dit_buckets,
            &condition,
            &reference.reference_mel,
            &reference.speaker_style,
            25,
            0.7,
            gen_config.seed,
            cancelled,
        )?;
        let mel = crop_reference_mel(&full_mel, reference.reference_mel.shape()[2] as usize)?;
        let s2mel_ms = s2mel_start.elapsed().as_secs_f32() * 1000.0;
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        let vocoder_start = Instant::now();
        let audio = runtime.vocode(&mel)?;
        let bigvgan_ms = vocoder_start.elapsed().as_secs_f32() * 1000.0;
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        let peak = audio
            .samples
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let rms = (audio.samples.iter().map(|value| value * value).sum::<f32>()
            / audio.samples.len() as f32)
            .sqrt();
        let silence_ratio = audio
            .samples
            .iter()
            .filter(|value| value.abs() < 1e-4)
            .count() as f32
            / audio.samples.len() as f32;
        let diagnostics = GenerationDiagnostics {
            semantic_token_count: codes.tokens.len() as u32,
            generated_seconds: audio.duration() as f32,
            reference_encode_ms: 0.0,
            gpt_ms,
            semantic_codec_ms,
            s2mel_ms,
            bigvgan_ms,
            total_ms: total_start.elapsed().as_secs_f32() * 1000.0,
            peak,
            rms,
            silence_ratio,
            seed: gen_config.seed,
        };
        Ok(SynthesisResult { audio, diagnostics })
    }

    fn synthesize_with_runtime(
        runtime: &mut SemanticRuntime,
        text: &str,
        reference: &ReferenceConditioning,
        gen_config: &GenerationConfig,
        cancelled: &AtomicBool,
    ) -> TtsResult<AudioBuffer> {
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        let max_tokens = gen_config.max_length.unwrap_or(1815);
        let codes = runtime.generate_with_reference_cancellable(
            text,
            gen_config.language,
            reference,
            max_tokens,
            cancelled,
        )?;
        let mel = runtime.generate_mel_cancellable(
            &codes,
            reference,
            gen_config.duration_factor,
            gen_config.seed,
            cancelled,
        )?;
        if cancelled.load(Ordering::Acquire) {
            return Err(indextts_core::IndexTtsError::Cancelled);
        }
        runtime.vocode(&mel)
    }

    pub fn validated_manifest(&self) -> Option<&ValidatedManifest> {
        self.manifest.as_ref()
    }

    /// Check if models are loaded
    pub fn is_loaded(&self) -> bool {
        self.runtime.is_some()
    }

    /// Get model directory
    pub fn model_dir(&self) -> &Path {
        &self.config.model_dir
    }
}

/// Convenience function to synthesize speech.
pub fn synthesize(
    model_dir: &Path,
    text: &str,
    reference_audio: &Path,
    config: &GenerationConfig,
) -> TtsResult<AudioBuffer> {
    let model_config = ModelConfig {
        model_dir: model_dir.to_path_buf(),
        device: DeviceConfig::default(),
        precision: indextts_core::Precision::default(),
    };

    let mut pipeline = IndexTtsPipeline::new(model_config);
    pipeline.load()?;
    pipeline.synthesize(text, reference_audio, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indextts_core::ModelConfig;

    #[test]
    fn test_pipeline_creation() {
        let config = ModelConfig::default();
        let pipeline = IndexTtsPipeline::new(config);
        assert!(!pipeline.is_loaded());
    }

    #[test]
    fn cancellation_preempts_unloaded_pipeline() {
        let pipeline = IndexTtsPipeline::new(ModelConfig::default());
        let cancelled = AtomicBool::new(true);
        let error = pipeline
            .synthesize_cancellable(
                "test",
                Path::new("unused.wav"),
                &GenerationConfig::default(),
                &cancelled,
            )
            .unwrap_err();
        assert!(matches!(error, indextts_core::IndexTtsError::Cancelled));
    }

    #[test]
    fn emotion_reference_blends_only_first_token() {
        let conditioning = Tensor::new(vec![10.0; 3 * 1280], vec![1, 3, 1280]);
        let voice = Tensor::new(vec![2.0; 1280], vec![1, 1280]);
        let reference = Tensor::new(vec![6.0; 1280], vec![1, 1280]);
        let output = apply_emotion_reference(&conditioning, &voice, &reference, 0.5).unwrap();
        assert!(output.as_slice()[..1280].iter().all(|value| *value == 12.0));
        assert!(output.as_slice()[1280..].iter().all(|value| *value == 10.0));
        let unchanged = apply_emotion_reference(&conditioning, &voice, &reference, 0.0).unwrap();
        assert_eq!(unchanged, conditioning);
        assert!(apply_emotion_reference(&conditioning, &voice, &reference, f32::NAN).is_err());
    }

    #[test]
    fn long_text_plan_has_stable_ranges_and_checked_seeds() {
        let planned = plan_long_text("甲乙，丙丁。戊己", 4, 41).unwrap();
        assert_eq!(
            planned,
            vec![
                PlannedSegment {
                    text: "甲乙，".into(),
                    text_range: 0..3,
                    seed: 41,
                },
                PlannedSegment {
                    text: "丙丁。".into(),
                    text_range: 3..6,
                    seed: 42,
                },
                PlannedSegment {
                    text: "戊己".into(),
                    text_range: 6..8,
                    seed: 43,
                },
            ]
        );
        let error = plan_long_text("甲乙", 1, u64::MAX).unwrap_err();
        assert!(error.to_string().contains("segment 1"));
        assert!(plan_long_text("", 10, 0).is_err());
        assert!(plan_long_text("text", 0, 0).is_err());
    }

    #[test]
    fn long_text_concatenation_inserts_pause_and_aggregates_metadata() {
        let planned = vec![
            PlannedSegment {
                text: "one".into(),
                text_range: 0..3,
                seed: 7,
            },
            PlannedSegment {
                text: "two".into(),
                text_range: 3..6,
                seed: 8,
            },
        ];
        let make_result = |samples: Vec<f32>, tokens, seed| SynthesisResult {
            audio: AudioBuffer::new(samples, 22_050),
            diagnostics: GenerationDiagnostics {
                semantic_token_count: tokens,
                gpt_ms: 1.0,
                total_ms: 2.0,
                seed,
                ..GenerationDiagnostics::default()
            },
        };
        let (audio, diagnostics, segments) = concatenate_long_text(
            &planned,
            vec![
                make_result(vec![0.5, -0.5], 2, 7),
                make_result(vec![0.25], 3, 8),
            ],
            100,
        )
        .unwrap();

        assert_eq!(audio.sample_rate, 22_050);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 2 + 2_205 + 1);
        assert_eq!(&audio.samples[..2], &[0.5, -0.5]);
        assert!(audio.samples[2..2 + 2_205]
            .iter()
            .all(|sample| *sample == 0.0));
        assert_eq!(audio.samples.last(), Some(&0.25));
        assert_eq!(segments[0].audio_offset_samples, 0);
        assert_eq!(segments[0].audio_duration_samples, 2);
        assert_eq!(segments[1].audio_offset_samples, 2 + 2_205);
        assert_eq!(segments[1].audio_duration_samples, 1);
        assert_eq!(segments[1].text_range, 3..6);
        assert_eq!(segments[1].seed, 8);
        assert_eq!(diagnostics.semantic_token_count, 5);
        assert_eq!(diagnostics.gpt_ms, 2.0);
        assert_eq!(diagnostics.total_ms, 4.0);
        assert_eq!(diagnostics.seed, 7);
        assert_eq!(diagnostics.peak, 0.5);
    }

    #[test]
    fn long_text_concatenation_rejects_wrong_audio_format() {
        let planned = vec![PlannedSegment {
            text: "text".into(),
            text_range: 0..4,
            seed: 0,
        }];
        let result = SynthesisResult {
            audio: AudioBuffer::new(vec![0.0], 16_000),
            diagnostics: GenerationDiagnostics::default(),
        };
        assert!(concatenate_long_text(&planned, vec![result], 0).is_err());
    }

    #[test]
    fn test_semantic_codes() {
        let codes = SemanticCodes::new(vec![100, 200, 300, 8193]);
        assert_eq!(codes.len, 4);
        assert_eq!(codes.find_stop_token(), Some(3));
        assert_eq!(codes.semantic_tokens_until_stop(), vec![100, 200, 300]);
    }
}
