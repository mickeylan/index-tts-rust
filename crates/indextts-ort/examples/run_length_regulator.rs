use indextts_ort::{run_length_regulator, OnnxSession, Tensor};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: run_length_regulator <model.onnx>")?;
    let session = OnnxSession::load(&model)?;
    let semantic = Tensor::new(vec![0.01; 34 * 1024], vec![1, 34, 1024]);
    let output = run_length_regulator(&session, semantic, 58)?;
    println!("inputs={:?}", session.input_names());
    println!("outputs={:?}", session.output_names());
    println!("condition_shape={:?}", output.shape());
    if output.shape() != [1, 58, 512] {
        return Err(format!("unexpected length regulator shape: {:?}", output.shape()).into());
    }
    if !output.as_slice().iter().all(|value| value.is_finite()) {
        return Err("length regulator returned non-finite values".into());
    }
    Ok(())
}
