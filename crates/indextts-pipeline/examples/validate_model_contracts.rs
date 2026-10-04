use indextts_core::{DeviceConfig, DeviceKind, ModelConfig, Precision};
use indextts_pipeline::IndexTtsPipeline;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: validate_model_contracts <model-dir>")?,
    );
    let mut pipeline = IndexTtsPipeline::new(ModelConfig {
        model_dir,
        device: DeviceConfig::new(DeviceKind::Cpu, 0),
        precision: Precision::Float32,
    });
    pipeline.load()?;
    println!("model contracts validated");
    Ok(())
}
