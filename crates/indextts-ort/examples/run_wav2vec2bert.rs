use indextts_ort::{run_wav2vec2bert, OnnxSession, Tensor, Wav2VecStats};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let model = args.next().ok_or("usage: run_wav2vec2bert <model.onnx> <stats.safetensors>")?;
    let stats = args.next().ok_or("usage: run_wav2vec2bert <model.onnx> <stats.safetensors>")?;
    let session = OnnxSession::load(&model)?;
    let frames = 64usize;
    let features: Vec<f32> = (0..frames * 160)
        .map(|index| ((index as f32) * 0.013).sin())
        .collect();
    let mut mask = vec![1i64; frames];
    mask[frames - 3..].fill(0);
    let output = run_wav2vec2bert(
        &session,
        Tensor::new(features, vec![1, frames as i64, 160]),
        Tensor::new_i64(mask, vec![1, frames as i64]),
    )?;
    let output = Wav2VecStats::load(&stats)?.normalize(output)?;
    println!("inputs={:?}", session.input_names());
    println!("outputs={:?}", session.output_names());
    println!("hidden_shape={:?}", output.shape());
    if output.shape() != [1, frames as i64, 1024] {
        return Err(format!("unexpected Wav2Vec2-BERT shape: {:?}", output.shape()).into());
    }
    if !output.as_slice().iter().all(|value| value.is_finite()) {
        return Err("Wav2Vec2-BERT returned non-finite values".into());
    }
    Ok(())
}
