use indextts_pipeline::ReferenceEncoder;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let model_dir = args.next().ok_or("usage: encode_reference <model-dir> <reference.wav>")?;
    let reference = args.next().ok_or("usage: encode_reference <model-dir> <reference.wav>")?;
    let encoder = ReferenceEncoder::load(&model_dir)?;
    let conditioning = encoder.encode(&reference)?;
    println!("semantic_shape={:?}", conditioning.semantic.shape());
    println!("speaker_style_shape={:?}", conditioning.speaker_style.shape());
    println!("gpt_conditioning_shape={:?}", conditioning.gpt_conditioning.shape());
    println!("reference_samples_22k={}", conditioning.reference_samples_22k);
    Ok(())
}
