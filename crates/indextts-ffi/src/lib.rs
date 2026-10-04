//! Stable C ABI for IndexTTS-2.5.

#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use indextts_audio::process_reference_audio;
use indextts_core::{DeviceConfig, DeviceKind, GenerationConfig, Language, ModelConfig, Precision};
use indextts_pipeline::IndexTtsPipeline;
use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Mutex;

pub struct IndexTtsModelHandle {
    pipeline: IndexTtsPipeline,
}
pub struct IndexTtsVoiceHandle {
    reference_path: PathBuf,
}
pub type indextts_model_t = *mut IndexTtsModelHandle;
pub type indextts_voice_t = *mut IndexTtsVoiceHandle;

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
            set_last_error(error);
            -1
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
        let config = ModelConfig {
            model_dir: PathBuf::from(model_dir),
            device,
            precision,
        };
        let mut pipeline = IndexTtsPipeline::new(config);
        pipeline.load().map_err(|error| error.to_string())?;
        *out_model = Box::into_raw(Box::new(IndexTtsModelHandle { pipeline }));
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
        process_reference_audio(&path).map_err(|error| error.to_string())?;
        *out_voice = Box::into_raw(Box::new(IndexTtsVoiceHandle {
            reference_path: path,
        }));
        Ok(())
    })
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
        let options = &*options;
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
        let config = GenerationConfig {
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
        };
        let audio = (&*model)
            .pipeline
            .synthesize(text, &(&*voice).reference_path, &config)
            .map_err(|error| error.to_string())?;
        let mut samples = audio.samples.into_boxed_slice();
        let output = indextts_audio_out_t {
            samples: samples.as_mut_ptr(),
            sample_count: samples.len(),
            sample_rate: audio.sample_rate,
            channels: audio.channels as u32,
            reserved: [0; 4],
        };
        std::mem::forget(samples);
        *out_audio = output;
        Ok(())
    })
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
