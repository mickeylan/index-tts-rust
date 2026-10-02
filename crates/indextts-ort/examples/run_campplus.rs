use indextts_ort::{run_campplus, OnnxSession, Tensor};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os().nth(1).map(PathBuf::from)
        .ok_or("usage: run_campplus <model.onnx>")?;
    let session = OnnxSession::load(&model)?;
    let frames = 197usize;
    let features: Vec<f32> = (0..frames * 80)
        .map(|index| ((index as f32) * 0.017).sin())
        .collect();
    let output = run_campplus(
        &session,
        Tensor::new(features, vec![1, frames as i64, 80]),
    )?;
    println!("inputs={:?}", session.input_names());
    println!("outputs={:?}", session.output_names());
    println!("style_shape={:?}", output.shape());
    if output.shape() != [1, 192] {
        return Err(format!("unexpected CAMPPlus shape: {:?}", output.shape()).into());
    }
    if !output.as_slice().iter().all(|value| value.is_finite()) {
        return Err("CAMPPlus returned non-finite values".into());
    }
    Ok(())
}
