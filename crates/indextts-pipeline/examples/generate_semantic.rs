use indextts_core::Language;
use indextts_pipeline::SemanticRuntime;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from)
        .ok_or("usage: generate_semantic <model-dir> <reference.wav> <text>")?;
    let reference = args.next().map(PathBuf::from)
        .ok_or("usage: generate_semantic <model-dir> <reference.wav> <text>")?;
    let text = args.next().and_then(|value| value.into_string().ok())
        .ok_or("usage: generate_semantic <model-dir> <reference.wav> <text>")?;
    let mut runtime = SemanticRuntime::load(&model_dir)?;
    let codes = runtime.generate(&text, Language::Zh, &reference, 1815)?;
    println!("semantic_codes={} {:?}", codes.len, codes.tokens);
    Ok(())
}
