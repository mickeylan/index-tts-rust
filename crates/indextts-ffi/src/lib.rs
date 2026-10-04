//! Stable C ABI for IndexTTS-2.5.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

mod manifest;

use indextts_audio::{process_reference_audio, process_reference_buffer};
use indextts_core::{
    AudioBuffer, DeviceConfig, DeviceKind, GenerationConfig, Language, ModelConfig, Precision,
};
use indextts_pipeline::{
    GenerationDiagnostics, IndexTtsPipeline, PreparedEmotion, ReferenceConditioning,
    SynthesisResult,
};
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

pub struct IndexTtsModelHandle {
    pipeline: IndexTtsPipeline,
    cancelled: Arc<AtomicBool>,
    info: ModelInfo,
    voice_cache: Mutex<VoiceCache>,
    active_requests: AtomicU64,
}
pub struct IndexTtsVoiceHandle {
    conditioning: Arc<ReferenceConditioning>,
    info: VoiceInfo,
}

struct ModelInfo {
    model_version: String,
    manifest_sha256: String,
    device: String,
}

struct VoiceInfo {
    reference_sha256: String,
    duration_seconds: f32,
    source_sample_rate: u32,
    source_channels: u32,
    cache_bytes: u64,
}

#[derive(Default)]
struct VoiceCache {
    entries: HashMap<String, Arc<ReferenceConditioning>>,
    order: VecDeque<String>,
}

impl VoiceCache {
    const MAX_ENTRIES: usize = 16;

    fn total_bytes(&self) -> u64 {
        self.entries
            .values()
            .map(|entry| conditioning_bytes(entry))
            .sum()
    }

    fn get(&mut self, key: &str) -> Option<Arc<ReferenceConditioning>> {
        let value = self.entries.get(key)?.clone();
        self.order.retain(|item| item != key);
        self.order.push_back(key.to_owned());
        Some(value)
    }
    fn insert(&mut self, key: String, value: Arc<ReferenceConditioning>) {
        self.entries.insert(key.clone(), value);
        self.order.retain(|item| item != &key);
        self.order.push_back(key);
        while self.entries.len() > Self::MAX_ENTRIES {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }
}
pub struct IndexTtsEmotionHandle {
    emotion: PreparedEmotion,
}
pub struct IndexTtsRequestHandle {
    model: usize,
    cancelled: Arc<AtomicBool>,
}

pub type indextts_model_t = *mut IndexTtsModelHandle;
pub type indextts_voice_t = *mut IndexTtsVoiceHandle;
pub type indextts_emotion_t = *mut IndexTtsEmotionHandle;
pub type indextts_request_t = *mut IndexTtsRequestHandle;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_model_options_t {
    pub model_dir: *const libc::c_char,
    /// -1 = CPU, 0 or greater = CUDA device. CUDA is rejected until built in.
    pub device_index: i32,
    pub precision: i32,
    pub reserved: [u64; 8],
}
impl Default for indextts_model_options_t {
    fn default() -> Self {
        Self {
            model_dir: std::ptr::null(),
            device_index: -1,
            precision: 0,
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_generate_options_t {
    pub text: *const libc::c_char,
    pub language: *const libc::c_char,
    pub seed: u64,
    pub duration_factor: f32,
    pub do_sample: i32,
    pub num_beams: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub reserved: [u64; 4],
}
impl Default for indextts_generate_options_t {
    fn default() -> Self {
        Self {
            text: std::ptr::null(),
            language: std::ptr::null(),
            seed: 0,
            duration_factor: 1.0,
            do_sample: 0,
            num_beams: 1,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.0,
            reserved: [0; 4],
        }
    }
}

pub const INDEXTTS_EMOTION_NONE: i32 = 0;
pub const INDEXTTS_EMOTION_TEXT: i32 = 1;
pub const INDEXTTS_EMOTION_REFERENCE: i32 = 2;
pub const INDEXTTS_EMOTION_VECTOR: i32 = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_emotion_options_t {
    pub mode: i32,
    pub text: *const libc::c_char,
    pub reference: indextts_emotion_t,
    pub vector: *const f32,
    pub vector_length: usize,
    pub strength: f32,
    pub reserved: [u64; 4],
}

impl Default for indextts_emotion_options_t {
    fn default() -> Self {
        Self {
            mode: INDEXTTS_EMOTION_NONE,
            text: std::ptr::null(),
            reference: std::ptr::null_mut(),
            vector: std::ptr::null(),
            vector_length: 0,
            strength: 1.0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct indextts_generate_options_v2_t {
    pub base: indextts_generate_options_t,
    pub emotion: indextts_emotion_options_t,
    pub reserved: [u64; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_audio_out_t {
    pub samples: *mut f32,
    pub sample_count: usize,
    pub sample_rate: u32,
    pub channels: u32,
    pub reserved: [u64; 4],
}
impl Default for indextts_audio_out_t {
    fn default() -> Self {
        Self {
            samples: std::ptr::null_mut(),
            sample_count: 0,
            sample_rate: 0,
            channels: 0,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct indextts_generation_info_t {
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
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct indextts_generation_result_t {
    pub audio: indextts_audio_out_t,
    pub info: indextts_generation_info_t,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct indextts_capabilities_t {
    pub abi_major: u32,
    pub abi_minor: u32,
    pub sample_rate: u32,
    pub max_reference_seconds: u32,
    pub max_semantic_tokens: u32,
    pub max_concurrent_requests_per_model: u32,
    pub supports_cuda: i32,
    pub supports_cpu: i32,
    pub supports_cancellation: i32,
    pub supports_request_cancellation: i32,
    pub supports_voice_cache: i32,
    pub supports_emotion_text: i32,
    pub supports_emotion_reference: i32,
    pub supports_emotion_vector: i32,
    pub supports_target_duration: i32,
    pub supports_sampling: i32,
    pub supports_beam_search: i32,
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_health_t {
    pub loaded: i32,
    pub device_healthy: i32,
    pub voice_cache_entries: u64,
    pub voice_cache_bytes: u64,
    pub active_requests: u64,
    pub queued_requests: u64,
    pub last_error: [libc::c_char; 512],
    pub reserved: [u64; 8],
}

impl Default for indextts_health_t {
    fn default() -> Self {
        Self {
            loaded: 0,
            device_healthy: 0,
            voice_cache_entries: 0,
            voice_cache_bytes: 0,
            active_requests: 0,
            queued_requests: 0,
            last_error: [0; 512],
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_model_info_t {
    pub runtime_version: [libc::c_char; 64],
    pub model_version: [libc::c_char; 64],
    pub model_manifest_sha256: [libc::c_char; 65],
    pub backend: [libc::c_char; 32],
    pub device: [libc::c_char; 32],
    pub reserved: [u64; 8],
}

impl Default for indextts_model_info_t {
    fn default() -> Self {
        Self {
            runtime_version: [0; 64],
            model_version: [0; 64],
            model_manifest_sha256: [0; 65],
            backend: [0; 32],
            device: [0; 32],
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_voice_info_t {
    pub reference_sha256: [libc::c_char; 65],
    pub duration_seconds: f32,
    pub source_sample_rate: u32,
    pub source_channels: u32,
    pub cache_bytes: u64,
    pub reserved: [u64; 8],
}

impl Default for indextts_voice_info_t {
    fn default() -> Self {
        Self {
            reference_sha256: [0; 65],
            duration_seconds: 0.0,
            source_sample_rate: 0,
            source_channels: 0,
            cache_bytes: 0,
            reserved: [0; 8],
        }
    }
}

const ABI_MAJOR: u32 = 1;
const ABI_MINOR: u32 = 4;

static LAST_ERROR: Lazy<Mutex<Option<CString>>> = Lazy::new(|| Mutex::new(None));
static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

fn set_last_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', "\\0");
    if let Ok(mut error) = LAST_ERROR.lock() {
        *error = CString::new(sanitized).ok();
    }
}
fn clear_last_error() {
    if let Ok(mut error) = LAST_ERROR.lock() {
        *error = None;
    }
}
fn ffi_status(operation: impl FnOnce() -> Result<(), String>) -> i32 {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            let status = if error == indextts_core::IndexTtsError::Cancelled.to_string() {
                -3
            } else {
                -1
            };
            set_last_error(error);
            status
        }
        Err(_) => {
            set_last_error("panic contained at IndexTTS C ABI boundary");
            -2
        }
    }
}
fn ffi_void(operation: impl FnOnce()) {
    if catch_unwind(AssertUnwindSafe(operation)).is_err() {
        set_last_error("panic contained at IndexTTS C ABI boundary");
    }
}
unsafe fn required_utf8<'a>(pointer: *const libc::c_char, name: &str) -> Result<&'a str, String> {
    if pointer.is_null() {
        return Err(format!("{name} is required"));
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map_err(|_| format!("{name} is not valid UTF-8"))
}
fn validate_reserved(values: &[u64]) -> Result<(), String> {
    if values.iter().any(|value| *value != 0) {
        Err("reserved fields must be zero".into())
    } else {
        Ok(())
    }
}

fn write_c_string<const N: usize>(
    target: &mut [libc::c_char; N],
    value: &str,
) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.len() >= N {
        return Err(format!("value is too long for {N}-byte ABI string"));
    }
    target.fill(0);
    for (slot, byte) in target.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn conditioning_bytes(conditioning: &ReferenceConditioning) -> u64 {
    [
        &conditioning.semantic,
        &conditioning.speaker_style,
        &conditioning.gpt_conditioning,
        &conditioning.reference_mel,
        &conditioning.prompt_condition,
    ]
    .iter()
    .map(|tensor| std::mem::size_of_val(tensor.as_slice()) as u64)
    .sum()
}

fn load_model_info(device: DeviceConfig, manifest: manifest::ValidatedManifest) -> ModelInfo {
    ModelInfo {
        model_version: manifest.model,
        manifest_sha256: manifest.sha256,
        device: match device.kind {
            DeviceKind::Cpu => "cpu".into(),
            DeviceKind::Cuda => format!("cuda:{}", device.index),
            DeviceKind::Auto => format!("auto:{}", device.index),
        },
    }
}

#[no_mangle]
pub extern "C" fn indextts_abi_version() -> u32 {
    (ABI_MAJOR << 16) | ABI_MINOR
}

#[no_mangle]
pub unsafe extern "C" fn indextts_get_capabilities(
    capabilities: *mut indextts_capabilities_t,
) -> i32 {
    ffi_status(|| {
        if capabilities.is_null() {
            return Err("capabilities is required".into());
        }
        *capabilities = indextts_capabilities_t {
            abi_major: ABI_MAJOR,
            abi_minor: ABI_MINOR,
            sample_rate: 22_050,
            max_reference_seconds: 15,
            max_semantic_tokens: 1815,
            max_concurrent_requests_per_model: 1,
            supports_cuda: cfg!(feature = "cuda") as i32,
            supports_cpu: 1,
            supports_cancellation: 1,
            supports_request_cancellation: 1,
            supports_voice_cache: 1,
            supports_emotion_text: 0,
            supports_emotion_reference: 1,
            supports_emotion_vector: 1,
            supports_target_duration: 0,
            supports_sampling: 0,
            supports_beam_search: 0,
            reserved: [0; 8],
        };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_model_get_info(
    model: indextts_model_t,
    info: *mut indextts_model_info_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || info.is_null() {
            return Err("invalid arguments".into());
        }
        let model = &*model;
        let mut output = indextts_model_info_t::default();
        write_c_string(&mut output.runtime_version, env!("CARGO_PKG_VERSION"))?;
        write_c_string(&mut output.model_version, &model.info.model_version)?;
        write_c_string(
            &mut output.model_manifest_sha256,
            &model.info.manifest_sha256,
        )?;
        write_c_string(&mut output.backend, "candle+onnxruntime")?;
        write_c_string(&mut output.device, &model.info.device)?;
        *info = output;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_model_health(
    model: indextts_model_t,
    health: *mut indextts_health_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || health.is_null() {
            return Err("invalid arguments".into());
        }
        let model = &*model;
        let cache = model
            .voice_cache
            .lock()
            .map_err(|_| "voice cache lock was poisoned")?;
        let mut output = indextts_health_t::default();
        output.loaded = model.pipeline.is_loaded() as i32;
        // This is a non-invasive health query: successful model loading is the only safe
        // device signal available without allocating tensors or running inference.
        output.device_healthy = output.loaded;
        output.voice_cache_entries = cache.entries.len() as u64;
        output.voice_cache_bytes = cache.total_bytes();
        output.active_requests = model.active_requests.load(Ordering::Acquire);
        output.queued_requests = 0;
        *health = output;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_voice_get_info(
    voice: indextts_voice_t,
    info: *mut indextts_voice_info_t,
) -> i32 {
    ffi_status(|| {
        if voice.is_null() || info.is_null() {
            return Err("invalid arguments".into());
        }
        let voice = &*voice;
        let mut output = indextts_voice_info_t::default();
        write_c_string(&mut output.reference_sha256, &voice.info.reference_sha256)?;
        output.duration_seconds = voice.info.duration_seconds;
        output.source_sample_rate = voice.info.source_sample_rate;
        output.source_channels = voice.info.source_channels;
        output.cache_bytes = voice.info.cache_bytes;
        *info = output;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_model_options_init(options: *mut indextts_model_options_t) {
    ffi_void(|| {
        if !options.is_null() {
            *options = indextts_model_options_t::default();
        }
    });
}
#[no_mangle]
pub unsafe extern "C" fn indextts_generate_options_init(options: *mut indextts_generate_options_t) {
    ffi_void(|| {
        if !options.is_null() {
            *options = indextts_generate_options_t::default();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn indextts_generate_options_v2_init(
    options: *mut indextts_generate_options_v2_t,
) {
    ffi_void(|| {
        if !options.is_null() {
            *options = indextts_generate_options_v2_t::default();
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn indextts_model_load(
    options: *const indextts_model_options_t,
    out_model: *mut indextts_model_t,
) -> i32 {
    ffi_status(|| {
        if options.is_null() || out_model.is_null() {
            return Err("invalid arguments".into());
        }
        *out_model = std::ptr::null_mut();
        let options = &*options;
        validate_reserved(&options.reserved)?;
        let model_dir = required_utf8(options.model_dir, "model_dir")?;
        let device = match options.device_index {
            -1 => DeviceConfig::new(DeviceKind::Cpu, 0),
            index if index >= 0 => {
                #[cfg(feature = "cuda")]
                {
                    DeviceConfig::new(DeviceKind::Cuda, index as usize)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(
                        "CUDA device requested, but indextts was built without the cuda feature"
                            .into(),
                    );
                }
            }
            index => {
                return Err(format!(
                    "invalid device_index {index}; use -1 for CPU or >= 0 for CUDA"
                ))
            }
        };
        let precision = match options.precision {
            0 => Precision::Float32,
            value => {
                return Err(format!(
                    "unsupported precision {value}; this build requires float32"
                ))
            }
        };
        let model_dir = PathBuf::from(model_dir);
        let manifest = manifest::validate_model_manifest(&model_dir)?;
        let info = load_model_info(device, manifest);
        let config = ModelConfig {
            model_dir,
            device,
            precision,
        };
        let mut pipeline = IndexTtsPipeline::new(config);
        pipeline.load().map_err(|error| error.to_string())?;
        *out_model = Box::into_raw(Box::new(IndexTtsModelHandle {
            pipeline,
            cancelled: Arc::new(AtomicBool::new(false)),
            info,
            voice_cache: Mutex::new(VoiceCache::default()),
            active_requests: AtomicU64::new(0),
        }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_voice_prepare(
    model: indextts_model_t,
    reference_audio_path: *const libc::c_char,
    out_voice: *mut indextts_voice_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || out_voice.is_null() {
            return Err("invalid arguments".into());
        }
        *out_voice = std::ptr::null_mut();
        let path = PathBuf::from(required_utf8(reference_audio_path, "reference_audio_path")?);
        let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
        let reader = hound::WavReader::open(&path).map_err(|error| error.to_string())?;
        let spec = reader.spec();
        let (_, processed) = process_reference_audio(&path).map_err(|error| error.to_string())?;
        let reference_sha256 = sha256_hex(&bytes);
        let model = &*model;
        let conditioning = {
            let mut cache = model
                .voice_cache
                .lock()
                .map_err(|_| "voice cache lock was poisoned")?;
            cache.get(&reference_sha256)
        };
        let conditioning = match conditioning {
            Some(value) => value,
            None => {
                let value = Arc::new(
                    model
                        .pipeline
                        .prepare_voice(&path)
                        .map_err(|error| error.to_string())?,
                );
                let mut cache = model
                    .voice_cache
                    .lock()
                    .map_err(|_| "voice cache lock was poisoned")?;
                cache.insert(reference_sha256.clone(), value.clone());
                value
            }
        };
        let info = VoiceInfo {
            reference_sha256,
            duration_seconds: processed.duration() as f32,
            source_sample_rate: spec.sample_rate,
            source_channels: spec.channels as u32,
            cache_bytes: conditioning_bytes(&conditioning),
        };
        *out_voice = Box::into_raw(Box::new(IndexTtsVoiceHandle { conditioning, info }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_voice_prepare_pcm(
    model: indextts_model_t,
    samples: *const f32,
    sample_count: usize,
    sample_rate: u32,
    channels: u32,
    out_voice: *mut indextts_voice_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || samples.is_null() || sample_count == 0 || out_voice.is_null() {
            return Err("invalid arguments".into());
        }
        if sample_rate == 0 || channels == 0 || sample_count % channels as usize != 0 {
            return Err("invalid PCM shape or sample rate".into());
        }
        *out_voice = std::ptr::null_mut();
        let interleaved = std::slice::from_raw_parts(samples, sample_count);
        if interleaved.iter().any(|sample| !sample.is_finite()) {
            return Err("PCM contains a non-finite sample".into());
        }
        let mono: Vec<f32> = interleaved
            .chunks_exact(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        let hash_bytes = unsafe {
            std::slice::from_raw_parts(
                samples.cast::<u8>(),
                sample_count * std::mem::size_of::<f32>(),
            )
        };
        let mut digest = Sha256::new();
        digest.update(sample_rate.to_le_bytes());
        digest.update(channels.to_le_bytes());
        digest.update(hash_bytes);
        let reference_sha256 = format!("{:x}", digest.finalize());
        let source = AudioBuffer::new(mono, sample_rate);
        let (_, processed) =
            process_reference_buffer(source.clone()).map_err(|error| error.to_string())?;
        let model = &*model;
        let conditioning = {
            let mut cache = model
                .voice_cache
                .lock()
                .map_err(|_| "voice cache lock was poisoned")?;
            cache.get(&reference_sha256)
        };
        let conditioning = match conditioning {
            Some(value) => value,
            None => {
                let value = Arc::new(
                    model
                        .pipeline
                        .prepare_voice_buffer(source)
                        .map_err(|error| error.to_string())?,
                );
                let mut cache = model
                    .voice_cache
                    .lock()
                    .map_err(|_| "voice cache lock was poisoned")?;
                cache.insert(reference_sha256.clone(), value.clone());
                value
            }
        };
        let info = VoiceInfo {
            reference_sha256,
            duration_seconds: processed.duration() as f32,
            source_sample_rate: sample_rate,
            source_channels: channels,
            cache_bytes: conditioning_bytes(&conditioning),
        };
        *out_voice = Box::into_raw(Box::new(IndexTtsVoiceHandle { conditioning, info }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_emotion_prepare_reference(
    model: indextts_model_t,
    reference_audio_path: *const libc::c_char,
    out_emotion: *mut indextts_emotion_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || reference_audio_path.is_null() || out_emotion.is_null() {
            return Err("invalid arguments".into());
        }
        *out_emotion = std::ptr::null_mut();
        let path = PathBuf::from(required_utf8(reference_audio_path, "reference_audio_path")?);
        let emotion = (&*model)
            .pipeline
            .prepare_emotion_reference(&path)
            .map_err(|error| error.to_string())?;
        *out_emotion = Box::into_raw(Box::new(IndexTtsEmotionHandle { emotion }));
        Ok(())
    })
}

unsafe fn parse_generation_options(
    options: &indextts_generate_options_t,
) -> Result<(&str, GenerationConfig), String> {
    validate_reserved(&options.reserved)?;
    let text = required_utf8(options.text, "text")?;
    let language = if options.language.is_null() {
        "ZH"
    } else {
        required_utf8(options.language, "language")?
    };
    let language =
        Language::parse(language).ok_or_else(|| format!("unsupported language {language}"))?;
    if options.do_sample != 0 || options.num_beams != 1 {
        return Err("this build supports greedy generation only".into());
    }
    if !(0.5..=2.0).contains(&options.duration_factor) || !options.duration_factor.is_finite() {
        return Err("duration_factor must be finite and in [0.5, 2.0]".into());
    }
    if !options.temperature.is_finite()
        || !options.top_p.is_finite()
        || !options.repetition_penalty.is_finite()
        || options.top_k < 0
    {
        return Err("invalid generation option".into());
    }
    Ok((
        text,
        GenerationConfig {
            text: text.into(),
            language,
            seed: options.seed,
            duration_factor: options.duration_factor,
            do_sample: false,
            num_beams: 1,
            temperature: options.temperature,
            top_k: options.top_k as usize,
            top_p: options.top_p,
            repetition_penalty: options.repetition_penalty,
            max_length: None,
        },
    ))
}

unsafe fn store_audio(audio: indextts_core::AudioBuffer, out_audio: *mut indextts_audio_out_t) {
    let mut samples = audio.samples.into_boxed_slice();
    *out_audio = indextts_audio_out_t {
        samples: samples.as_mut_ptr(),
        sample_count: samples.len(),
        sample_rate: audio.sample_rate,
        channels: audio.channels as u32,
        reserved: [0; 4],
    };
    std::mem::forget(samples);
}

fn generation_info(value: GenerationDiagnostics) -> indextts_generation_info_t {
    indextts_generation_info_t {
        semantic_token_count: value.semantic_token_count,
        generated_seconds: value.generated_seconds,
        reference_encode_ms: value.reference_encode_ms,
        gpt_ms: value.gpt_ms,
        semantic_codec_ms: value.semantic_codec_ms,
        s2mel_ms: value.s2mel_ms,
        bigvgan_ms: value.bigvgan_ms,
        total_ms: value.total_ms,
        peak: value.peak,
        rms: value.rms,
        silence_ratio: value.silence_ratio,
        seed: value.seed,
        reserved: [0; 8],
    }
}

#[no_mangle]
pub unsafe extern "C" fn indextts_generate(
    model: indextts_model_t,
    voice: indextts_voice_t,
    options: *const indextts_generate_options_t,
    out_audio: *mut indextts_audio_out_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || voice.is_null() || options.is_null() || out_audio.is_null() {
            return Err("invalid arguments".into());
        }
        *out_audio = indextts_audio_out_t::default();
        let (text, config) = parse_generation_options(&*options)?;
        let model = &*model;
        let _active = ActiveRequestGuard::new(&model.active_requests);
        model.cancelled.store(false, Ordering::Release);
        let audio = model
            .pipeline
            .synthesize_prepared_cancellable(
                text,
                &(&*voice).conditioning,
                &config,
                &model.cancelled,
            )
            .map_err(|error| error.to_string())?;
        store_audio(audio, out_audio);
        Ok(())
    })
}

struct ActiveRequestGuard<'a>(&'a AtomicU64);

impl<'a> ActiveRequestGuard<'a> {
    fn new(counter: &'a AtomicU64) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for ActiveRequestGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

unsafe fn generate_v2_inner(
    model: &IndexTtsModelHandle,
    voice: &IndexTtsVoiceHandle,
    options: &indextts_generate_options_v2_t,
    cancelled: &AtomicBool,
) -> Result<AudioBuffer, String> {
    validate_reserved(&options.reserved)?;
    validate_reserved(&options.emotion.reserved)?;
    let (text, config) = parse_generation_options(&options.base)?;
    match options.emotion.mode {
        INDEXTTS_EMOTION_NONE => model.pipeline.synthesize_prepared_cancellable(
            text,
            &voice.conditioning,
            &config,
            cancelled,
        ),
        INDEXTTS_EMOTION_REFERENCE => {
            if options.emotion.reference.is_null() {
                return Err("emotion reference handle is required".into());
            }
            model.pipeline.synthesize_prepared_with_emotion_cancellable(
                text,
                &voice.conditioning,
                &(&*options.emotion.reference).emotion,
                options.emotion.strength,
                &config,
                cancelled,
            )
        }
        INDEXTTS_EMOTION_TEXT => return Err("emotion text is not supported by this runtime".into()),
        INDEXTTS_EMOTION_VECTOR => {
            if options.emotion.vector.is_null() || options.emotion.vector_length != 8 {
                return Err("emotion vector must contain exactly 8 values".into());
            }
            let values = std::slice::from_raw_parts(options.emotion.vector, 8);
            let weights: [f32; 8] = values.try_into().map_err(|_| "invalid emotion vector")?;
            let emotion = model
                .pipeline
                .prepare_emotion_vector(&voice.conditioning, &weights, options.emotion.strength)
                .map_err(|error| error.to_string())?;
            model.pipeline.synthesize_prepared_with_emotion_cancellable(
                text,
                &voice.conditioning,
                &emotion,
                1.0,
                &config,
                cancelled,
            )
        }
        mode => return Err(format!("unsupported emotion mode {mode}")),
    }
    .map_err(|error| error.to_string())
}

unsafe fn generate_result_v2_inner(
    model: &IndexTtsModelHandle,
    voice: &IndexTtsVoiceHandle,
    options: &indextts_generate_options_v2_t,
    cancelled: &AtomicBool,
) -> Result<SynthesisResult, String> {
    validate_reserved(&options.reserved)?;
    validate_reserved(&options.emotion.reserved)?;
    let (text, config) = parse_generation_options(&options.base)?;
    match options.emotion.mode {
        INDEXTTS_EMOTION_NONE => model.pipeline.synthesize_prepared_result_cancellable(
            text,
            &voice.conditioning,
            &config,
            cancelled,
        ),
        INDEXTTS_EMOTION_REFERENCE => {
            if options.emotion.reference.is_null() {
                return Err("emotion reference handle is required".into());
            }
            model
                .pipeline
                .synthesize_prepared_with_emotion_result_cancellable(
                    text,
                    &voice.conditioning,
                    &(&*options.emotion.reference).emotion,
                    options.emotion.strength,
                    &config,
                    cancelled,
                )
        }
        INDEXTTS_EMOTION_TEXT => return Err("emotion text is not supported by this runtime".into()),
        INDEXTTS_EMOTION_VECTOR => {
            if options.emotion.vector.is_null() || options.emotion.vector_length != 8 {
                return Err("emotion vector must contain exactly 8 values".into());
            }
            let values = std::slice::from_raw_parts(options.emotion.vector, 8);
            let weights: [f32; 8] = values.try_into().map_err(|_| "invalid emotion vector")?;
            let emotion = model
                .pipeline
                .prepare_emotion_vector(&voice.conditioning, &weights, options.emotion.strength)
                .map_err(|error| error.to_string())?;
            model
                .pipeline
                .synthesize_prepared_with_emotion_result_cancellable(
                    text,
                    &voice.conditioning,
                    &emotion,
                    1.0,
                    &config,
                    cancelled,
                )
        }
        mode => return Err(format!("unsupported emotion mode {mode}")),
    }
    .map_err(|error| error.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn indextts_generate_v2(
    model: indextts_model_t,
    voice: indextts_voice_t,
    options: *const indextts_generate_options_v2_t,
    out_audio: *mut indextts_audio_out_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || voice.is_null() || options.is_null() || out_audio.is_null() {
            return Err("invalid arguments".into());
        }
        *out_audio = indextts_audio_out_t::default();
        let model = &*model;
        let _active = ActiveRequestGuard::new(&model.active_requests);
        model.cancelled.store(false, Ordering::Release);
        let audio = generate_v2_inner(model, &*voice, &*options, &model.cancelled)?;
        store_audio(audio, out_audio);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_request_create(
    model: indextts_model_t,
    out_request: *mut indextts_request_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null() || out_request.is_null() {
            return Err("invalid arguments".into());
        }
        *out_request = Box::into_raw(Box::new(IndexTtsRequestHandle {
            model: model as usize,
            cancelled: Arc::new(AtomicBool::new(false)),
        }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_generate_request_v2(
    model: indextts_model_t,
    request: indextts_request_t,
    voice: indextts_voice_t,
    options: *const indextts_generate_options_v2_t,
    out_audio: *mut indextts_audio_out_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null()
            || request.is_null()
            || voice.is_null()
            || options.is_null()
            || out_audio.is_null()
        {
            return Err("invalid arguments".into());
        }
        if (&*request).model != model as usize {
            return Err("request belongs to another model".into());
        }
        *out_audio = indextts_audio_out_t::default();
        let model_ref = &*model;
        let _active = ActiveRequestGuard::new(&model_ref.active_requests);
        (&*request).cancelled.store(false, Ordering::Release);
        let audio = generate_v2_inner(model_ref, &*voice, &*options, &(&*request).cancelled)?;
        store_audio(audio, out_audio);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_generate_result_request_v2(
    model: indextts_model_t,
    request: indextts_request_t,
    voice: indextts_voice_t,
    options: *const indextts_generate_options_v2_t,
    out_result: *mut indextts_generation_result_t,
) -> i32 {
    ffi_status(|| {
        if model.is_null()
            || request.is_null()
            || voice.is_null()
            || options.is_null()
            || out_result.is_null()
        {
            return Err("invalid arguments".into());
        }
        if (&*request).model != model as usize {
            return Err("request belongs to another model".into());
        }
        *out_result = indextts_generation_result_t::default();
        let model_ref = &*model;
        let _active = ActiveRequestGuard::new(&model_ref.active_requests);
        (&*request).cancelled.store(false, Ordering::Release);
        let result =
            generate_result_v2_inner(model_ref, &*voice, &*options, &(&*request).cancelled)?;
        store_audio(result.audio, &mut (*out_result).audio);
        (*out_result).info = generation_info(result.diagnostics);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn indextts_request_cancel(request: indextts_request_t) -> i32 {
    if request.is_null() {
        return -1;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        (&*request).cancelled.store(true, Ordering::Release)
    })) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn indextts_request_free(request: indextts_request_t) {
    ffi_void(|| {
        if !request.is_null() {
            drop(Box::from_raw(request));
        }
    });
}

/// Request cooperative cancellation of the model's active generation.
/// Returns immediately; the generation call returns `INDEXTTS_CANCELLED` at
/// the next token, pipeline stage, or CFM-step cancellation point.
#[no_mangle]
pub unsafe extern "C" fn indextts_model_cancel(model: indextts_model_t) -> i32 {
    if model.is_null() {
        return -1;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        (&*model).cancelled.store(true, Ordering::Release);
    })) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn indextts_audio_free(audio: *mut indextts_audio_out_t) {
    ffi_void(|| {
        if audio.is_null() {
            return;
        }
        let output = &mut *audio;
        if !output.samples.is_null() {
            drop(Vec::from_raw_parts(
                output.samples,
                output.sample_count,
                output.sample_count,
            ));
        }
        *output = indextts_audio_out_t::default();
    });
}
#[no_mangle]
pub unsafe extern "C" fn indextts_emotion_free(emotion: indextts_emotion_t) {
    ffi_void(|| {
        if !emotion.is_null() {
            drop(Box::from_raw(emotion));
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn indextts_voice_free(voice: indextts_voice_t) {
    ffi_void(|| {
        if !voice.is_null() {
            drop(Box::from_raw(voice));
        }
    });
}
#[no_mangle]
pub unsafe extern "C" fn indextts_model_free(model: indextts_model_t) {
    ffi_void(|| {
        if !model.is_null() {
            drop(Box::from_raw(model));
        }
    });
}
#[no_mangle]
pub extern "C" fn indextts_last_error() -> *const libc::c_char {
    LAST_ERROR
        .lock()
        .ok()
        .and_then(|error| error.as_ref().map(|value| value.as_ptr()))
        .unwrap_or(std::ptr::null())
}
#[no_mangle]
pub extern "C" fn indextts_version() -> *const libc::c_char {
    VERSION.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_are_valid() {
        let model = indextts_model_options_t::default();
        assert_eq!(model.device_index, -1);
        let generation = indextts_generate_options_t::default();
        assert_eq!(generation.num_beams, 1);
        assert_eq!(generation.duration_factor, 1.0);
    }
    #[test]
    fn abi_version_and_capabilities_are_truthful() {
        assert_eq!(indextts_abi_version(), 0x0001_0004);
        let mut capabilities = indextts_capabilities_t::default();
        assert_eq!(unsafe { indextts_get_capabilities(&mut capabilities) }, 0);
        assert_eq!(capabilities.abi_major, 1);
        assert_eq!(capabilities.abi_minor, 4);
        assert_eq!(capabilities.sample_rate, 22_050);
        assert_eq!(capabilities.max_concurrent_requests_per_model, 1);
        assert_eq!(capabilities.supports_cuda, cfg!(feature = "cuda") as i32);
        assert_eq!(capabilities.supports_request_cancellation, 1);
        assert_eq!(capabilities.supports_voice_cache, 1);
        assert_eq!(capabilities.supports_emotion_reference, 1);
        assert_eq!(capabilities.supports_emotion_text, 0);
        let options = indextts_generate_options_v2_t::default();
        assert_eq!(options.emotion.mode, INDEXTTS_EMOTION_NONE);
        assert_eq!(options.emotion.strength, 1.0);
    }

    #[test]
    fn request_cancellation_is_independent() {
        let first = Box::into_raw(Box::new(IndexTtsRequestHandle {
            model: 1,
            cancelled: Arc::new(AtomicBool::new(false)),
        }));
        let second = Box::into_raw(Box::new(IndexTtsRequestHandle {
            model: 1,
            cancelled: Arc::new(AtomicBool::new(false)),
        }));
        assert_eq!(unsafe { indextts_request_cancel(first) }, 0);
        assert!(unsafe { (&*first).cancelled.load(Ordering::Acquire) });
        assert!(!unsafe { (&*second).cancelled.load(Ordering::Acquire) });
        unsafe { indextts_request_free(first) };
        unsafe { indextts_request_free(second) };
    }

    #[test]
    fn version_and_error_pointers_are_stable() {
        assert_eq!(
            unsafe { CStr::from_ptr(indextts_version()) }
                .to_str()
                .unwrap(),
            env!("CARGO_PKG_VERSION")
        );
        set_last_error("example");
        let pointer = indextts_last_error();
        assert_eq!(
            unsafe { CStr::from_ptr(pointer) }.to_str().unwrap(),
            "example"
        );
    }
    #[test]
    fn cancellation_sets_model_flag() {
        let handle = Box::new(IndexTtsModelHandle {
            pipeline: IndexTtsPipeline::new(ModelConfig::default()),
            cancelled: Arc::new(AtomicBool::new(false)),
            info: ModelInfo {
                model_version: String::new(),
                manifest_sha256: String::new(),
                device: "cpu".into(),
            },
            voice_cache: Mutex::new(VoiceCache::default()),
            active_requests: AtomicU64::new(0),
        });
        let pointer = Box::into_raw(handle);
        assert_eq!(unsafe { indextts_model_cancel(pointer) }, 0);
        assert!(unsafe { &*pointer }.cancelled.load(Ordering::Acquire));
        unsafe { indextts_model_free(pointer) };
    }

    #[test]
    fn audio_free_reclaims_and_zeros() {
        let mut samples = vec![1.0f32, 2.0].into_boxed_slice();
        let mut audio = indextts_audio_out_t {
            samples: samples.as_mut_ptr(),
            sample_count: 2,
            sample_rate: 22_050,
            channels: 1,
            reserved: [0; 4],
        };
        std::mem::forget(samples);
        unsafe {
            indextts_audio_free(&mut audio);
        }
        assert!(audio.samples.is_null());
        assert_eq!(audio.sample_count, 0);
        unsafe {
            indextts_audio_free(&mut audio);
        }
    }
}
