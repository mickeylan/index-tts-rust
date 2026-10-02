//! C ABI for IndexTTS-2.5
//!
//! Provides stable C API for use by Go, C/C++, and other languages.
//!
//! # Safety
//!
//! All FFI functions are unsafe and require proper handling of pointers.

use indextts_core::{
    AudioBuffer, DeviceConfig, DeviceKind, GenerationConfig, IndexTtsError,
    Language, ModelConfig, Precision, Result as TtsResult, SemanticCodes,
};
use indextts_pipeline::IndexTtsPipeline;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::Mutex;

/// Opaque handle to an IndexTTS model
pub type indextts_model_t = *mut IndexTtsModelHandle;

/// Opaque handle to a voice (preprocessed speaker condition)
pub type indextts_voice_t = *mut IndexTtsVoiceHandle;

/// Opaque handle to generated audio
pub type indextts_audio_t = *mut IndexTtsAudioHandle;

/// Internal model handle
struct IndexTtsModelHandle {
    pipeline: IndexTtsPipeline,
}

/// Internal voice handle
struct IndexTtsVoiceHandle {
    reference_path: PathBuf,
    // Precomputed conditions would go here
}

/// Internal audio handle
struct IndexTtsAudioHandle {
    audio: AudioBuffer,
}

// ============================================================================
// Model Options
// ============================================================================

/// Model loading options
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_model_options_t {
    /// Model directory path (UTF-8)
    pub model_dir: *const libc::c_char,
    /// CUDA device index
    pub device_index: libc::int32_t,
    /// Precision: 0=float32, 1=bfloat16
    pub precision: libc::int32_t,
    /// Reserved for future use
    pub reserved: [libc::uint64_t; 8],
}

impl Default for indextts_model_options_t {
    fn default() -> Self {
        Self {
            model_dir: std::ptr::null(),
            device_index: 0,
            precision: 0,
            reserved: [0; 8],
        }
    }
}

/// Generation options
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_generate_options_t {
    /// Text to synthesize (UTF-8)
    pub text: *const libc::c_char,
    /// Language code (ZH, EN, JA, ES, AR)
    pub language: *const libc::c_char,
    /// Random seed (0 = random)
    pub seed: libc::uint64_t,
    /// Duration factor (1.0 = default)
    pub duration_factor: libc::c_float,
    /// Do sample (0 = greedy, 1 = sample)
    pub do_sample: libc::int32_t,
    /// Number of beams
    pub num_beams: libc::int32_t,
    /// Temperature for sampling
    pub temperature: libc::c_float,
    /// Top-k for sampling
    pub top_k: libc::int32_t,
    /// Top-p for sampling
    pub top_p: libc::c_float,
    /// Repetition penalty
    pub repetition_penalty: libc::c_float,
    /// Reserved
    pub reserved: [libc::uint64_t; 4],
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

/// Audio output structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct indextts_audio_out_t {
    /// Audio samples (normalized to [-1, 1])
    pub samples: *mut libc::c_float,
    /// Number of samples
    pub sample_count: libc::size_t,
    /// Sample rate in Hz
    pub sample_rate: libc::uint32_t,
    /// Number of channels
    pub channels: libc::uint32_t,
    /// Reserved
    pub reserved: [libc::uint64_t; 4],
}

impl Default for indextts_audio_out_t {
    fn default() -> Self {
        Self {
            samples: std::ptr::null_mut(),
            sample_count: 0,
            sample_rate: 22050,
            channels: 1,
            reserved: [0; 4],
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

/// Last error message (thread-local)
thread_local! {
    static LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);
}

/// Set the last error message
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        let mut guard = e.lock().unwrap();
        *guard = CString::new(msg).ok();
    });
}

/// Get the last error message
fn get_last_error() -> Option<CString> {
    LAST_ERROR.with(|e| {
        let guard = e.lock().unwrap();
        guard.clone()
    })
}

// ============================================================================
// FFI Functions
// ============================================================================

/// Load an IndexTTS model
///
/// # Safety
/// - `options` must be a valid pointer
/// - `out_model` must point to valid memory
#[no_mangle]
pub unsafe extern "C" fn indextts_model_load(
    options: *const indextts_model_options_t,
    out_model: *mut indextts_model_t,
) -> libc::int32_t {
    // Validate inputs
    if options.is_null() || out_model.is_null() {
        set_last_error("Invalid arguments");
        return -1;
    }

    let options = &*options;

    // Get model directory
    if options.model_dir.is_null() {
        set_last_error("model_dir is required");
        return -1;
    }

    let model_dir = unsafe {
        match CStr::from_ptr(options.model_dir).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid model_dir encoding");
                return -1;
            }
        }
    };

    // Create config
    let device = match options.device_index {
        0 => DeviceConfig::new(DeviceKind::Auto, 0),
        _ => DeviceConfig::new(DeviceKind::Cuda, options.device_index as usize),
    };

    let precision = match options.precision {
        0 => Precision::Float32,
        1 => Precision::BFloat16,
        _ => {
            set_last_error("Invalid precision");
            return -1;
        }
    };

    let config = ModelConfig {
        model_dir: PathBuf::from(model_dir),
        device,
        precision,
    };

    // Create and load pipeline
    let mut pipeline = IndexTtsPipeline::new(config);
    match pipeline.load() {
        Ok(_) => {
            let handle = Box::new(IndexTtsModelHandle { pipeline });
            unsafe { *out_model = Box::into_raw(handle) };
            0
        }
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Prepare a voice from reference audio
///
/// # Safety
/// - `model` must be a valid model handle
/// - `reference_audio_path` must be valid UTF-8 string
/// - `out_voice` must point to valid memory
#[no_mangle]
pub unsafe extern "C" fn indextts_voice_prepare(
    model: indextts_model_t,
    reference_audio_path: *const libc::c_char,
    out_voice: *mut indextts_voice_t,
) -> libc::int32_t {
    if model.is_null() || reference_audio_path.is_null() || out_voice.is_null() {
        set_last_error("Invalid arguments");
        return -1;
    }

    let path = unsafe {
        match CStr::from_ptr(reference_audio_path).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid path encoding");
                return -1;
            }
        }
    };

    let handle = unsafe { &*model };
    let voice = IndexTtsVoiceHandle {
        reference_path: PathBuf::from(path),
    };

    let voice_box = Box::new(voice);
    unsafe { *out_voice = Box::into_raw(voice_box) };
    0
}

/// Generate speech
///
/// # Safety
/// - All pointer arguments must be valid
/// - Returned audio must be freed with indextts_audio_free
#[no_mangle]
pub unsafe extern "C" fn indextts_generate(
    model: indextts_model_t,
    voice: indextts_voice_t,
    options: *const indextts_generate_options_t,
    out_audio: *mut indextts_audio_out_t,
) -> libc::int32_t {
    if model.is_null() || voice.is_null() || options.is_null() || out_audio.is_null() {
        set_last_error("Invalid arguments");
        return -1;
    }

    let options = &*options;

    // Parse text
    let text = unsafe {
        if options.text.is_null() {
            set_last_error("text is required");
            return -1;
        }
        match CStr::from_ptr(options.text).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_last_error("Invalid text encoding");
                return -1;
            }
        }
    };

    // Parse language
    let language = unsafe {
        if options.language.is_null() {
            "ZH"
        } else {
            match CStr::from_ptr(options.language).to_str() {
                Ok(s) => s,
                Err(_) => {
                    set_last_error("Invalid language encoding");
                    return -1;
                }
            }
        }
    };

    let lang = Language::parse(language).unwrap_or(Language::Zh);

    // Create generation config
    let gen_config = GenerationConfig {
        text: text.to_string(),
        language: lang,
        seed: options.seed,
        duration_factor: options.duration_factor,
        do_sample: options.do_sample != 0,
        num_beams: options.num_beams as usize,
        temperature: options.temperature,
        top_k: options.top_k as usize,
        top_p: options.top_p,
        repetition_penalty: options.repetition_penalty,
        max_length: None,
    };

    let model_handle = unsafe { &*model };
    let voice_handle = unsafe { &*voice };

    match model_handle.pipeline.synthesize(
        text,
        &voice_handle.reference_path,
        &gen_config,
    ) {
        Ok(audio) => {
            let samples = audio.samples.clone();
            let sample_rate = audio.sample_rate;
            
            // Allocate output
            let out = indextts_audio_out_t {
                samples: samples.as_ptr() as *mut libc::c_float,
                sample_count: samples.len(),
                sample_rate,
                channels: 1,
                reserved: [0; 4],
            };
            
            // Store audio for later cleanup
            let audio_handle = Box::new(IndexTtsAudioHandle { audio });
            // TODO: Store this somewhere for cleanup
            
            unsafe { *out_audio = out };
            0
        }
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free audio data
///
/// # Safety
/// - `audio` must be a valid audio out pointer
#[no_mangle]
pub unsafe extern "C" fn indextts_audio_free(audio: *mut indextts_audio_out_t) {
    if !audio.is_null() {
        // Audio samples were owned by IndexTtsAudioHandle
        // This needs proper memory management
    }
}

/// Free voice handle
///
/// # Safety
/// - `voice` must be a valid voice handle
#[no_mangle]
pub unsafe extern "C" fn indextts_voice_free(voice: indextts_voice_t) {
    if !voice.is_null() {
        unsafe { drop(Box::from_raw(voice)) };
    }
}

/// Free model handle
///
/// # Safety
/// - `model` must be a valid model handle
#[no_mangle]
pub unsafe extern "C" fn indextts_model_free(model: indextts_model_t) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}

/// Get the last error message
///
/// Returns null if no error occurred
#[no_mangle]
pub unsafe extern "C" fn indextts_last_error() -> *const libc::c_char {
    match get_last_error() {
        Some(cstr) => cstr.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Get the library version
#[no_mangle]
pub extern "C" fn indextts_version() -> *const libc::c_char {
    CString::new(env!("CARGO_PKG_VERSION")).unwrap().as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = indextts_model_options_t::default();
        assert!(opts.model_dir.is_null());
        assert_eq!(opts.device_index, 0);
        assert_eq!(opts.precision, 0);
    }

    #[test]
    fn test_default_gen_options() {
        let opts = indextts_generate_options_t::default();
        assert!(opts.text.is_null());
        assert_eq!(opts.seed, 0);
        assert_eq!(opts.duration_factor, 1.0);
        assert_eq!(opts.num_beams, 1);
    }
}
