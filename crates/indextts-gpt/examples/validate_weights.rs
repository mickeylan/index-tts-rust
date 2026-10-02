use candle_core::Device;
use indextts_gpt::LoadedWeights;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: validate_weights <gpt.safetensors>")?;
    let loaded = LoadedWeights::load(&path, &Device::Cpu)?;
    loaded.weights.validate_greedy_contract()?;
    println!(
        "validated {} tensors, {} parameters",
        loaded.weights.len(),
        loaded.num_params
    );
    Ok(())
}
