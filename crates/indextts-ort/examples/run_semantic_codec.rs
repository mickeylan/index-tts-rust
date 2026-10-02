use indextts_ort::{run_semantic_codec, OnnxSession};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os().nth(1).map(PathBuf::from)
        .ok_or("usage: run_semantic_codec <model.onnx>")?;
    let session = OnnxSession::load(&model)?;
    let codes: Vec<u32> = (0..17).map(|index| (index * 431) % 8192).collect();
    let output = run_semantic_codec(&session, &codes)?;
    println!("inputs={:?}", session.input_names());
    println!("outputs={:?}", session.output_names());
    println!("semantic_shape={:?}", output.shape());
    if output.shape() != [1, 34, 1024] {
        return Err(format!("unexpected semantic codec shape: {:?}", output.shape()).into());
    }
    if !output.as_slice().iter().all(|value| value.is_finite()) {
        return Err("semantic codec returned non-finite values".into());
    }
    Ok(())
}
