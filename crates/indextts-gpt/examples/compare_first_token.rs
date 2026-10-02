use candle_core::{Device, IndexOp};
use indextts_gpt::{Generator, GptConfig, GreedyGenerator, IndexGpt, KvCache};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let weights_path = args.next().ok_or("usage: compare_first_token <weights> <fixture>")?;
    let fixture_path = args.next().ok_or("usage: compare_first_token <weights> <fixture>")?;
    let device = Device::Cpu;

    let fixture = candle_core::safetensors::load(&fixture_path, &device)?;
    let prefix = fixture.get("prefix").ok_or("fixture missing prefix")?.clone();
    let fake_ids = fixture.get("fake_ids").ok_or("fixture missing fake_ids")?.clone();
    let expected_logits = fixture.get("first_logits").ok_or("fixture missing first_logits")?;
    let metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_path.with_extension("json"))?
    )?;
    let expected_codes: Vec<u32> = serde_json::from_value(
        metadata.get("semantic_codes").cloned().ok_or("metadata missing semantic_codes")?
    )?;

    let mut model = IndexGpt::new(GptConfig::default(), device.clone())?;
    model.load_weights(&weights_path)?;
    model.store_mel_emb(prefix);
    let mut cache = KvCache::new(model.config().n_positions, device);
    let logits = model.prefill(&fake_ids, None, None, Some(&mut cache))?;
    let actual = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>()?;
    let expected = expected_logits.i(0)?.to_vec1::<f32>()?;
    let actual_token = actual.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).map(|(index, _)| index).unwrap();
    let expected_token = expected.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).map(|(index, _)| index).unwrap();
    let max_abs_error = actual.iter().zip(&expected)
        .map(|(left, right)| (left - right).abs()).fold(0f32, f32::max);

    println!("expected_token={expected_token}");
    println!("actual_token={actual_token}");
    println!("max_abs_error={max_abs_error}");
    if actual_token != expected_token {
        return Err("first token mismatch".into());
    }

    let mut generator = GreedyGenerator::new(model.device(), model.config().n_positions);
    let max_length = fake_ids.dim(1)? + expected_codes.len() - 1;
    let generated = generator.generate(&model, &fake_ids, None, max_length)?;
    println!("expected_codes={expected_codes:?}");
    println!("actual_codes={:?}", generated.tokens);
    if generated.tokens != expected_codes {
        return Err("greedy semantic code mismatch".into());
    }
    Ok(())
}
