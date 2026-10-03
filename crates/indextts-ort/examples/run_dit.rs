use indextts_ort::{run_dit, OnnxSession, Tensor};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: run_dit <model.onnx>")?;
    let session = OnnxSession::load(&model)?;
    let frames = 64i64;
    let output = run_dit(
        &session,
        Tensor::new(vec![0.01; 2 * 80 * frames as usize], vec![2, 80, frames]),
        Tensor::new(vec![0.0; 2 * 80 * frames as usize], vec![2, 80, frames]),
        Tensor::new_i64(vec![frames, frames], vec![2]),
        Tensor::new(vec![0.4, 0.4], vec![2]),
        Tensor::new(vec![0.02; 2 * 192], vec![2, 192]),
        Tensor::new(vec![0.03; 2 * frames as usize * 512], vec![2, frames, 512]),
    )?;
    println!("velocity_shape={:?}", output.shape());
    if output.shape() != [2, 80, frames] || !output.as_slice().iter().all(|value| value.is_finite())
    {
        return Err("invalid DiT output".into());
    }
    Ok(())
}
