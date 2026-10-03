use indextts_audio::save_wav;
use indextts_core::Language;
use indextts_pipeline::{ReferenceEncoder, SemanticRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: generate_audio <model-dir> <reference.wav> <text> <output.wav>")?;
    let reference_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: generate_audio <model-dir> <reference.wav> <text> <output.wav>")?;
    let text = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or("usage: generate_audio <model-dir> <reference.wav> <text> <output.wav>")?;
    let output = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: generate_audio <model-dir> <reference.wav> <text> <output.wav>")?;

    let reference = ReferenceEncoder::load(&model_dir)?.encode(&reference_path)?;
    let mut runtime = SemanticRuntime::load(&model_dir)?;
    let codes = runtime.generate(&text, Language::Zh, &reference_path, 1815)?;
    let mel = runtime.generate_mel(&codes, &reference, 1.0, 0)?;
    let audio = runtime.vocode(&mel)?;
    save_wav(&output, &audio)?;
    println!("semantic_codes={}", codes.len);
    println!("mel_shape={:?}", mel.shape());
    println!(
        "audio_samples={} sample_rate={}",
        audio.samples.len(),
        audio.sample_rate
    );
    println!("wrote={}", output.display());
    Ok(())
}
