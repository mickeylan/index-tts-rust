use indextts_core::Language;
use indextts_pipeline::{ReferenceEncoder, SemanticRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: generate_mel <model-dir> <reference.wav> <text>")?;
    let reference_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: generate_mel <model-dir> <reference.wav> <text>")?;
    let text = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or("usage: generate_mel <model-dir> <reference.wav> <text>")?;

    let reference = ReferenceEncoder::load(&model_dir)?.encode(&reference_path)?;
    let mut runtime = SemanticRuntime::load(&model_dir)?;
    let codes = runtime.generate(&text, Language::Zh, &reference_path, 1815)?;
    let mel = runtime.generate_mel(&codes, &reference, 1.0, 0)?;
    println!("semantic_codes={}", codes.len);
    println!("generated_mel_shape={:?}", mel.shape());
    println!(
        "generated_mel_finite={}",
        mel.as_slice().iter().all(|value| value.is_finite())
    );
    Ok(())
}
