//! ONNX Runtime integration for the non-autoregressive IndexTTS-2.5 stages.

use indextts_core::{IndexTtsError, Result};
use ort::{
    session::{Session, SessionInputValue},
    value::{DynTensor, Tensor as OrtTensor},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

/// Owned host tensor accepted by the runtime wrapper.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    F32 { data: Vec<f32>, shape: Vec<i64> },
    I64 { data: Vec<i64>, shape: Vec<i64> },
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        Self::F32 { data, shape }
    }

    pub fn new_i64(data: Vec<i64>, shape: Vec<i64>) -> Self {
        Self::I64 { data, shape }
    }

    pub fn shape(&self) -> &[i64] {
        match self {
            Self::F32 { shape, .. } | Self::I64 { shape, .. } => shape,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::F32 { data, .. } => data,
            Self::I64 { .. } => panic!("requested f32 data from an i64 tensor"),
        }
    }

    pub fn as_i64_slice(&self) -> &[i64] {
        match self {
            Self::I64 { data, .. } => data,
            Self::F32 { .. } => panic!("requested i64 data from an f32 tensor"),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.shape().iter().any(|dimension| *dimension < 0) {
            return Err(IndexTtsError::BackendFailure(format!(
                "runtime tensor has a negative shape: {:?}",
                self.shape()
            )));
        }
        let expected = self
            .shape()
            .iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(*dimension as usize)
            })
            .ok_or_else(|| IndexTtsError::BackendFailure("runtime tensor shape overflow".into()))?;
        let actual = match self {
            Self::F32 { data, .. } => data.len(),
            Self::I64 { data, .. } => data.len(),
        };
        if actual != expected {
            return Err(IndexTtsError::BackendFailure(format!(
                "runtime tensor has {actual} values but shape {:?} requires {expected}",
                self.shape()
            )));
        }
        Ok(())
    }

    fn into_ort(self) -> Result<DynTensor> {
        self.validate()?;
        match self {
            Self::F32 { data, shape } => OrtTensor::from_array((shape, data))
                .map(|tensor| tensor.upcast())
                .map_err(ort_error),
            Self::I64 { data, shape } => OrtTensor::from_array((shape, data))
                .map(|tensor| tensor.upcast())
                .map_err(ort_error),
        }
    }
}

fn ort_error(error: ort::Error) -> IndexTtsError {
    IndexTtsError::BackendFailure(format!("ONNX Runtime: {error}"))
}

/// Thread-safe handle. `ort::Session::run` is serialized because ORT requires
/// mutable access and some execution-provider internals are not thread-safe.
#[derive(Debug, Clone)]
pub struct OnnxSession {
    path: PathBuf,
    inner: Arc<Mutex<Session>>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxSession {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_device(path, None)
    }

    pub fn load_with_device(path: &Path, cuda_device: Option<usize>) -> Result<Self> {
        if !path.is_file() {
            return Err(IndexTtsError::InvalidModel(format!(
                "ONNX model not found: {}",
                path.display()
            )));
        }
        let mut builder = Session::builder().map_err(ort_error)?;
        if let Some(device) = cuda_device {
            #[cfg(feature = "cuda")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default()
                        .with_device_id(device as i32)
                        .build()])
                    .map_err(|error| {
                        IndexTtsError::BackendFailure(format!("ONNX Runtime: {error}"))
                    })?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
                return Err(IndexTtsError::BackendFailure(
                    "CUDA requested, but indextts-ort was built without the cuda feature".into(),
                ));
            }
        }
        let session = builder.commit_from_file(path).map_err(ort_error)?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_owned())
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            inner: Arc::new(Mutex::new(session)),
            input_names,
            output_names,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
    pub fn has_input(&self, name: &str) -> bool {
        self.input_names.iter().any(|item| item == name)
    }
    pub fn has_output(&self, name: &str) -> bool {
        self.output_names.iter().any(|item| item == name)
    }

    pub fn run<IT, OT>(&self, inputs: IT, requested_outputs: OT) -> Result<Vec<Tensor>>
    where
        IT: IntoIterator<Item = (String, Tensor)>,
        OT: IntoIterator<Item = String>,
    {
        let inputs: Vec<(String, SessionInputValue<'static>)> = inputs
            .into_iter()
            .map(|(name, tensor)| Ok((name, SessionInputValue::from(tensor.into_ort()?))))
            .collect::<Result<_>>()?;
        for (name, _) in &inputs {
            if !self.has_input(name) {
                return Err(IndexTtsError::BackendFailure(format!(
                    "model {} has no input named {name}; available: {:?}",
                    self.path.display(),
                    self.input_names
                )));
            }
        }
        let requested: Vec<String> = requested_outputs.into_iter().collect();
        for name in &requested {
            if !self.has_output(name) {
                return Err(IndexTtsError::BackendFailure(format!(
                    "model {} has no output named {name}; available: {:?}",
                    self.path.display(),
                    self.output_names
                )));
            }
        }

        let mut session = self.inner.lock().map_err(|_| {
            IndexTtsError::BackendFailure("ONNX Runtime session lock was poisoned".into())
        })?;
        let outputs = session.run(inputs).map_err(ort_error)?;
        let names = if requested.is_empty() {
            self.output_names.clone()
        } else {
            requested
        };
        names
            .into_iter()
            .map(|name| {
                let value = outputs.get(&name).ok_or_else(|| {
                    IndexTtsError::BackendFailure(format!("ONNX Runtime omitted output {name}"))
                })?;
                if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                    return Ok(Tensor::F32 {
                        data: data.to_vec(),
                        shape: shape.iter().copied().collect(),
                    });
                }
                if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                    return Ok(Tensor::I64 {
                        data: data.to_vec(),
                        shape: shape.iter().copied().collect(),
                    });
                }
                Err(IndexTtsError::BackendFailure(format!(
                    "output {name} is not a supported f32/i64 CPU tensor"
                )))
            })
            .collect()
    }

    pub fn run_tensors(
        &self,
        inputs: Vec<(&str, Tensor)>,
        outputs: Vec<String>,
    ) -> Result<Vec<Tensor>> {
        self.run(
            inputs
                .into_iter()
                .map(|(name, tensor)| (name.to_owned(), tensor)),
            outputs,
        )
    }
}

#[derive(Debug, Default)]
pub struct SessionCache {
    sessions: HashMap<PathBuf, OnnxSession>,
}

impl SessionCache {
    pub fn get(&mut self, path: &Path) -> Result<OnnxSession> {
        let path = path.to_path_buf();
        if let Some(session) = self.sessions.get(&path) {
            return Ok(session.clone());
        }
        let session = OnnxSession::load(&path)?;
        self.sessions.insert(path, session.clone());
        Ok(session)
    }
    pub fn clear(&mut self) {
        self.sessions.clear();
    }
    pub fn len(&self) -> usize {
        self.sessions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxModel {
    Wav2Vec2Bert,
    Campplus,
    SpeakerConditioner,
    EmotionConditioner,
    GptConditioning,
    SemanticCodec,
    LengthRegulator,
    S2Mel,
    BigVGAN,
}

impl OnnxModel {
    pub fn filename(&self) -> &'static str {
        "model.onnx"
    }
    pub fn subdir(&self) -> &'static str {
        match self {
            Self::Wav2Vec2Bert => "wav2vec2bert",
            Self::Campplus => "campplus",
            Self::SpeakerConditioner => "speaker-conditioner",
            Self::EmotionConditioner => "emotion-conditioner",
            Self::GptConditioning => "gpt-conditioning",
            Self::SemanticCodec => "semantic-codec",
            Self::LengthRegulator => "length-regulator",
            Self::S2Mel => "s2mel",
            Self::BigVGAN => "bigvgan",
        }
    }
    pub fn path(&self, model_dir: &Path) -> PathBuf {
        model_dir
            .join("onnx")
            .join(self.subdir())
            .join(self.filename())
    }
}

pub fn run_wav2vec2bert(session: &OnnxSession, features: Tensor, mask: Tensor) -> Result<Tensor> {
    session
        .run_tensors(
            vec![("input_features", features), ("attention_mask", mask)],
            vec!["hidden_states_17".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing Wav2Vec2-BERT output".into()))
}

#[derive(Debug, Clone)]
pub struct Wav2VecStats {
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
}

impl Wav2VecStats {
    pub fn load(path: &Path) -> Result<Self> {
        let tensors =
            candle_core::safetensors::load(path, &candle_core::Device::Cpu).map_err(|error| {
                IndexTtsError::InvalidModel(format!(
                    "failed to load Wav2Vec statistics {}: {error}",
                    path.display()
                ))
            })?;
        let mean = tensors
            .get("mean")
            .ok_or_else(|| IndexTtsError::InvalidModel("Wav2Vec statistics missing mean".into()))?
            .to_vec1::<f32>()
            .map_err(|error| IndexTtsError::InvalidModel(error.to_string()))?;
        let variance = tensors
            .get("var")
            .ok_or_else(|| IndexTtsError::InvalidModel("Wav2Vec statistics missing var".into()))?
            .to_vec1::<f32>()
            .map_err(|error| IndexTtsError::InvalidModel(error.to_string()))?;
        if mean.len() != 1024
            || variance.len() != 1024
            || variance.iter().any(|value| *value <= 0.0)
        {
            return Err(IndexTtsError::InvalidModel(format!(
                "invalid Wav2Vec statistics: mean={}, var={}",
                mean.len(),
                variance.len()
            )));
        }
        Ok(Self {
            mean,
            inverse_std: variance
                .into_iter()
                .map(|value| 1.0 / value.sqrt())
                .collect(),
        })
    }

    pub fn normalize(&self, tensor: Tensor) -> Result<Tensor> {
        let Tensor::F32 { mut data, shape } = tensor else {
            return Err(IndexTtsError::BackendFailure(
                "Wav2Vec hidden states must be f32".into(),
            ));
        };
        if shape.last() != Some(&1024) {
            return Err(IndexTtsError::BackendFailure(format!(
                "Wav2Vec hidden width must be 1024, got {shape:?}"
            )));
        }
        for row in data.chunks_exact_mut(1024) {
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value - self.mean[index]) * self.inverse_std[index];
            }
        }
        Ok(Tensor::F32 { data, shape })
    }
}

pub fn run_campplus(session: &OnnxSession, features: Tensor) -> Result<Tensor> {
    session
        .run_tensors(vec![("x", features)], vec!["style".into()])?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing CAMPPlus output".into()))
}

pub fn run_emotion_conditioner(session: &OnnxSession, semantic_features: Tensor) -> Result<Tensor> {
    session
        .run_tensors(
            vec![("semantic_features", semantic_features)],
            vec!["emotion".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing emotion output".into()))
}

pub fn run_gpt_conditioning(
    session: &OnnxSession,
    speaker_style: Tensor,
    semantic_features: Tensor,
) -> Result<Tensor> {
    session
        .run_tensors(
            vec![
                ("speaker_style", speaker_style),
                ("semantic_features", semantic_features),
            ],
            vec!["conditioning".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing GPT conditioning output".into()))
}

pub fn run_semantic_codec(session: &OnnxSession, codes: &[u32]) -> Result<Tensor> {
    if codes.is_empty() {
        return Err(IndexTtsError::EmptySemanticCodes);
    }
    let codes: Vec<i64> = codes.iter().map(|code| *code as i64).collect();
    session
        .run_tensors(
            vec![(
                "codes",
                Tensor::new_i64(codes.clone(), vec![1, codes.len() as i64]),
            )],
            vec!["semantic_features".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing semantic codec output".into()))
}

pub fn run_length_regulator(
    session: &OnnxSession,
    semantic_features: Tensor,
    target_length: usize,
) -> Result<Tensor> {
    if target_length == 0 {
        return Err(IndexTtsError::BackendFailure(
            "length regulator target is zero".into(),
        ));
    }
    session
        .run_tensors(
            vec![
                ("semantic_features", semantic_features),
                (
                    "target_length",
                    Tensor::new_i64(vec![target_length as i64], vec![1]),
                ),
            ],
            vec!["condition".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing length regulator output".into()))
}

/// Fixed-length DiT session selected from exported frame buckets.
#[derive(Debug, Clone)]
pub struct DitBucket {
    pub frames: usize,
    pub session: OnnxSession,
}

#[derive(Debug, Clone)]
pub struct DitBuckets {
    buckets: Vec<DitBucket>,
}

impl DitBuckets {
    pub fn load(model_dir: &Path, frame_sizes: &[usize]) -> Result<Self> {
        Self::load_with_device(model_dir, frame_sizes, None)
    }

    pub fn load_with_device(
        model_dir: &Path,
        frame_sizes: &[usize],
        cuda_device: Option<usize>,
    ) -> Result<Self> {
        let mut buckets = Vec::new();
        for &frames in frame_sizes {
            let path = model_dir
                .join("onnx")
                .join("s2mel")
                .join(format!("model-{frames}.onnx"));
            if path.is_file() {
                buckets.push(DitBucket {
                    frames,
                    session: OnnxSession::load_with_device(&path, cuda_device)?,
                });
            }
        }
        buckets.sort_by_key(|bucket| bucket.frames);
        if buckets.is_empty() {
            return Err(IndexTtsError::InvalidModel(format!(
                "no DiT frame buckets found under {}",
                model_dir.join("onnx/s2mel").display()
            )));
        }
        Ok(Self { buckets })
    }

    pub fn select(&self, required_frames: usize) -> Result<&DitBucket> {
        self.buckets
            .iter()
            .find(|bucket| bucket.frames >= required_frames)
            .ok_or_else(|| {
                IndexTtsError::BackendFailure(format!(
                    "no DiT bucket can fit {required_frames} frames; largest is {}",
                    self.buckets.last().map(|bucket| bucket.frames).unwrap_or(0)
                ))
            })
    }

    pub fn frame_sizes(&self) -> Vec<usize> {
        self.buckets.iter().map(|bucket| bucket.frames).collect()
    }
}

pub fn run_dit(
    session: &OnnxSession,
    x: Tensor,
    prompt: Tensor,
    lengths: Tensor,
    time: Tensor,
    style: Tensor,
    condition: Tensor,
) -> Result<Tensor> {
    session
        .run_tensors(
            vec![
                ("x", x),
                ("prompt_x", prompt),
                ("x_lens", lengths),
                ("t", time),
                ("style", style),
                ("condition", condition),
            ],
            vec!["velocity".into()],
        )?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing DiT velocity output".into()))
}

/// Run the official 25-step Euler CFM solver with classifier-free guidance.
pub fn solve_cfm_bucketed(
    buckets: &DitBuckets,
    condition: &Tensor,
    prompt_mel: &Tensor,
    style: &Tensor,
    steps: usize,
    cfg_rate: f32,
    seed: u64,
) -> Result<Tensor> {
    let cancelled = AtomicBool::new(false);
    solve_cfm_bucketed_cancellable(
        buckets, condition, prompt_mel, style, steps, cfg_rate, seed, &cancelled,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_cfm_bucketed_cancellable(
    buckets: &DitBuckets,
    condition: &Tensor,
    prompt_mel: &Tensor,
    style: &Tensor,
    steps: usize,
    cfg_rate: f32,
    seed: u64,
    cancelled: &AtomicBool,
) -> Result<Tensor> {
    let required_frames = *condition.shape().get(1).ok_or_else(|| {
        IndexTtsError::BackendFailure("CFM condition has no time dimension".into())
    })? as usize;
    let bucket = buckets.select(required_frames)?;
    let padded_condition = pad_condition(condition, bucket.frames)?;
    let output = solve_cfm_inner(
        &bucket.session,
        &padded_condition,
        prompt_mel,
        style,
        steps,
        cfg_rate,
        seed,
        required_frames,
        cancelled,
    )?;
    crop_mel_frames(&output, required_frames)
}

fn pad_condition(condition: &Tensor, frames: usize) -> Result<Tensor> {
    let Tensor::F32 { data, shape } = condition else {
        return Err(IndexTtsError::BackendFailure(
            "DiT condition must be f32".into(),
        ));
    };
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 512 || shape[1] as usize > frames {
        return Err(IndexTtsError::BackendFailure(format!(
            "cannot pad condition {shape:?} to {frames} frames"
        )));
    }
    let source_frames = shape[1] as usize;
    let mut padded = vec![0f32; frames * 512];
    for frame in 0..source_frames {
        padded[frame * 512..(frame + 1) * 512]
            .copy_from_slice(&data[frame * 512..(frame + 1) * 512]);
    }
    Ok(Tensor::new(padded, vec![1, frames as i64, 512]))
}

fn crop_mel_frames(mel: &Tensor, frames: usize) -> Result<Tensor> {
    let Tensor::F32 { data, shape } = mel else {
        return Err(IndexTtsError::BackendFailure(
            "DiT output must be f32".into(),
        ));
    };
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 80 || frames > shape[2] as usize {
        return Err(IndexTtsError::BackendFailure(format!(
            "cannot crop DiT output {shape:?} to {frames} frames"
        )));
    }
    let bucket_frames = shape[2] as usize;
    let mut cropped = Vec::with_capacity(80 * frames);
    for channel in 0..80 {
        let offset = channel * bucket_frames;
        cropped.extend_from_slice(&data[offset..offset + frames]);
    }
    Ok(Tensor::new(cropped, vec![1, 80, frames as i64]))
}

pub fn solve_cfm(
    session: &OnnxSession,
    condition: &Tensor,
    prompt_mel: &Tensor,
    style: &Tensor,
    steps: usize,
    cfg_rate: f32,
    seed: u64,
) -> Result<Tensor> {
    let active_frames = *condition.shape().get(1).ok_or_else(|| {
        IndexTtsError::BackendFailure("CFM condition has no time dimension".into())
    })? as usize;
    let cancelled = AtomicBool::new(false);
    solve_cfm_inner(
        session,
        condition,
        prompt_mel,
        style,
        steps,
        cfg_rate,
        seed,
        active_frames,
        &cancelled,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_cfm_inner(
    session: &OnnxSession,
    condition: &Tensor,
    prompt_mel: &Tensor,
    style: &Tensor,
    steps: usize,
    cfg_rate: f32,
    seed: u64,
    active_frames: usize,
    cancelled: &AtomicBool,
) -> Result<Tensor> {
    if steps == 0 {
        return Err(IndexTtsError::BackendFailure(
            "CFM steps must be positive".into(),
        ));
    }
    let Tensor::F32 {
        data: condition_data,
        shape: condition_shape,
    } = condition
    else {
        return Err(IndexTtsError::BackendFailure(
            "CFM condition must be f32".into(),
        ));
    };
    let Tensor::F32 {
        data: prompt_data,
        shape: prompt_shape,
    } = prompt_mel
    else {
        return Err(IndexTtsError::BackendFailure(
            "CFM prompt must be f32".into(),
        ));
    };
    let Tensor::F32 {
        data: style_data,
        shape: style_shape,
    } = style
    else {
        return Err(IndexTtsError::BackendFailure(
            "CFM style must be f32".into(),
        ));
    };
    if condition_shape.len() != 3
        || condition_shape[0] != 1
        || condition_shape[2] != 512
        || prompt_shape.len() != 3
        || prompt_shape[0] != 1
        || prompt_shape[1] != 80
        || style_shape.as_slice() != [1, 192]
    {
        return Err(IndexTtsError::BackendFailure(format!(
            "invalid CFM shapes: condition={condition_shape:?}, prompt={prompt_shape:?}, style={style_shape:?}"
        )));
    }
    let total_frames = condition_shape[1] as usize;
    let prompt_frames = prompt_shape[2] as usize;
    if prompt_frames > active_frames || active_frames > total_frames {
        return Err(IndexTtsError::BackendFailure(format!(
            "invalid CFM active length: prompt={prompt_frames}, active={active_frames}, total={total_frames}"
        )));
    }
    let frame_values = 80 * total_frames;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x: Vec<f32> = (0..frame_values)
        .map(|_| rng.sample(StandardNormal))
        .collect();
    let mut prompt = vec![0f32; frame_values];
    for channel in 0..80 {
        let source = &prompt_data[channel * prompt_frames..(channel + 1) * prompt_frames];
        prompt[channel * total_frames..channel * total_frames + prompt_frames]
            .copy_from_slice(source);
        x[channel * total_frames..channel * total_frames + prompt_frames].fill(0.0);
    }
    let dt = 1.0 / steps as f32;
    for step in 0..steps {
        if cancelled.load(Ordering::Acquire) {
            return Err(IndexTtsError::Cancelled);
        }
        let mut stacked_x = x.clone();
        stacked_x.extend_from_slice(&x);
        let mut stacked_prompt = prompt.clone();
        stacked_prompt.extend(vec![0f32; frame_values]);
        let mut stacked_style = style_data.clone();
        stacked_style.extend(vec![0f32; 192]);
        let mut stacked_condition = condition_data.clone();
        stacked_condition.extend(vec![0f32; condition_data.len()]);
        let velocity = run_dit(
            session,
            Tensor::new(stacked_x, vec![2, 80, total_frames as i64]),
            Tensor::new(stacked_prompt, vec![2, 80, total_frames as i64]),
            Tensor::new_i64(vec![active_frames as i64, active_frames as i64], vec![2]),
            Tensor::new(vec![step as f32 * dt; 2], vec![2]),
            Tensor::new(stacked_style, vec![2, 192]),
            Tensor::new(stacked_condition, vec![2, total_frames as i64, 512]),
        )?;
        let values = velocity.as_slice();
        if values.len() != 2 * frame_values {
            return Err(IndexTtsError::BackendFailure(format!(
                "unexpected DiT output shape {:?}",
                velocity.shape()
            )));
        }
        for index in 0..frame_values {
            let guided = (1.0 + cfg_rate) * values[index] - cfg_rate * values[frame_values + index];
            x[index] += dt * guided;
        }
        if step + 1 < steps {
            for channel in 0..80 {
                x[channel * total_frames..channel * total_frames + prompt_frames].fill(0.0);
            }
        }
    }
    Ok(Tensor::new(x, vec![1, 80, total_frames as i64]))
}

pub fn run_bigvgan(session: &OnnxSession, mel: Tensor) -> Result<Tensor> {
    session
        .run_tensors(vec![("mel", mel)], vec!["audio".into()])?
        .into_iter()
        .next()
        .ok_or_else(|| IndexTtsError::BackendFailure("missing BigVGAN output".into()))
}

#[derive(Debug, Clone)]
pub struct BigVganBuckets {
    buckets: Vec<DitBucket>,
}

impl BigVganBuckets {
    pub fn load(model_dir: &Path, frame_sizes: &[usize]) -> Result<Self> {
        Self::load_with_device(model_dir, frame_sizes, None)
    }

    pub fn load_with_device(
        model_dir: &Path,
        frame_sizes: &[usize],
        cuda_device: Option<usize>,
    ) -> Result<Self> {
        let mut buckets = Vec::new();
        for &frames in frame_sizes {
            let path = model_dir
                .join("onnx")
                .join("bigvgan")
                .join(format!("model-{frames}.onnx"));
            if path.is_file() {
                buckets.push(DitBucket {
                    frames,
                    session: OnnxSession::load_with_device(&path, cuda_device)?,
                });
            }
        }
        buckets.sort_by_key(|bucket| bucket.frames);
        if buckets.is_empty() {
            return Err(IndexTtsError::InvalidModel(format!(
                "no BigVGAN frame buckets found under {}",
                model_dir.join("onnx/bigvgan").display()
            )));
        }
        Ok(Self { buckets })
    }

    pub fn synthesize(&self, mel: &Tensor) -> Result<Tensor> {
        let frames = *mel
            .shape()
            .get(2)
            .ok_or_else(|| IndexTtsError::BackendFailure("mel has no time dimension".into()))?
            as usize;
        let bucket = self
            .buckets
            .iter()
            .find(|bucket| bucket.frames >= frames)
            .ok_or_else(|| {
                IndexTtsError::BackendFailure(format!("no BigVGAN bucket can fit {frames} frames"))
            })?;
        let padded = pad_mel(mel, bucket.frames)?;
        let audio = run_bigvgan(&bucket.session, padded)?;
        let required_samples = frames * 256;
        let Tensor::F32 { data, shape } = audio else {
            return Err(IndexTtsError::BackendFailure(
                "BigVGAN output must be f32".into(),
            ));
        };
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 || data.len() < required_samples {
            return Err(IndexTtsError::BackendFailure(format!(
                "invalid BigVGAN output {shape:?}"
            )));
        }
        Ok(Tensor::new(
            data[..required_samples].to_vec(),
            vec![1, 1, required_samples as i64],
        ))
    }
}

fn pad_mel(mel: &Tensor, frames: usize) -> Result<Tensor> {
    let Tensor::F32 { data, shape } = mel else {
        return Err(IndexTtsError::BackendFailure("mel must be f32".into()));
    };
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 80 || shape[2] as usize > frames {
        return Err(IndexTtsError::BackendFailure(format!(
            "cannot pad mel {shape:?} to {frames}"
        )));
    }
    let source_frames = shape[2] as usize;
    let mut padded = vec![0f32; 80 * frames];
    for channel in 0..80 {
        padded[channel * frames..channel * frames + source_frames]
            .copy_from_slice(&data[channel * source_frames..(channel + 1) * source_frames]);
    }
    Ok(Tensor::new(padded, vec![1, 80, frames as i64]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_model_paths() {
        let model_dir = Path::new("/models/index-tts");
        assert_eq!(
            OnnxModel::Wav2Vec2Bert.path(model_dir),
            Path::new("/models/index-tts/onnx/wav2vec2bert/model.onnx")
        );
        assert_eq!(
            OnnxModel::BigVGAN.path(model_dir),
            Path::new("/models/index-tts/onnx/bigvgan/model.onnx")
        );
    }

    #[test]
    fn test_session_cache() {
        let cache = SessionCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert_eq!(tensor.shape(), &[1, 3]);
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0]);
        tensor.validate().unwrap();
    }

    #[test]
    fn rejects_invalid_tensor_shape() {
        let tensor = Tensor::new_i64(vec![1, 2], vec![1, 3]);
        assert!(tensor.validate().is_err());
    }

    #[test]
    fn condition_padding_and_mel_cropping_preserve_layout() {
        let condition = Tensor::new(
            (0..3 * 512).map(|value| value as f32).collect(),
            vec![1, 3, 512],
        );
        let padded = pad_condition(&condition, 5).unwrap();
        assert_eq!(padded.shape(), &[1, 5, 512]);
        assert_eq!(&padded.as_slice()[..3 * 512], condition.as_slice());
        assert!(padded.as_slice()[3 * 512..]
            .iter()
            .all(|value| *value == 0.0));

        let mel = Tensor::new(
            (0..80 * 5).map(|value| value as f32).collect(),
            vec![1, 80, 5],
        );
        let cropped = crop_mel_frames(&mel, 3).unwrap();
        assert_eq!(cropped.shape(), &[1, 80, 3]);
        assert_eq!(&cropped.as_slice()[..6], &[0., 1., 2., 5., 6., 7.]);
    }
}
