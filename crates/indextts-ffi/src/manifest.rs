use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Component, Path},
};

const REQUIRED_FILES: &[&str] = &[
    "gpt.safetensors",
    "wav2vec2bert_stats.safetensors",
    "emotion-prototypes.safetensors",
    "bpe.model",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "pinyin.vocab",
    "onnx/wav2vec2bert/model.onnx",
    "onnx/campplus/model.onnx",
    "onnx/gpt-conditioning/model.onnx",
    "onnx/emotion-conditioner/model.onnx",
    "onnx/semantic-codec/model.onnx",
    "onnx/length-regulator/model.onnx",
    "onnx/s2mel/model-256.onnx",
    "onnx/s2mel/model-512.onnx",
    "onnx/s2mel/model-1024.onnx",
    "onnx/bigvgan/model-256.onnx",
    "onnx/bigvgan/model-512.onnx",
];

#[derive(Debug)]
pub struct ValidatedManifest {
    pub model: String,
    pub sha256: String,
}

pub fn validate_model_manifest(model_dir: &Path) -> Result<ValidatedManifest, String> {
    let manifest_path = model_dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|error| {
        format!(
            "model manifest {} cannot be read: {error}",
            manifest_path.display()
        )
    })?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("model manifest is invalid JSON: {error}"))?;
    if manifest.get("format_version").and_then(Value::as_u64) != Some(1) {
        return Err("unsupported model manifest format_version; expected 1".into());
    }
    if manifest.get("runtime").and_then(Value::as_str) != Some("index-tts-rust") {
        return Err("model manifest runtime must be index-tts-rust".into());
    }
    if manifest.get("sample_rate").and_then(Value::as_u64) != Some(22_050) {
        return Err("model manifest sample_rate must be 22050".into());
    }
    let minimum_runtime = manifest
        .get("minimum_runtime_version")
        .and_then(Value::as_str)
        .ok_or_else(|| "model manifest has no minimum_runtime_version".to_string())?;
    if parse_version(minimum_runtime)? > parse_version(env!("CARGO_PKG_VERSION"))? {
        return Err(format!(
            "model requires runtime {minimum_runtime}; this runtime is {}",
            env!("CARGO_PKG_VERSION")
        ));
    }
    let s2mel_buckets = validate_u64_array(&manifest, "s2mel_buckets", &[256, 512, 1024])?;
    let bigvgan_buckets = validate_u64_array(&manifest, "bigvgan_buckets", &[256, 512])?;
    let semantic = manifest
        .get("semantic")
        .and_then(Value::as_object)
        .ok_or_else(|| "model manifest has no semantic contract".to_string())?;
    for (name, expected) in [
        ("start_token", 8192),
        ("stop_token", 8193),
        ("max_tokens", 1815),
    ] {
        if semantic.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(format!("model manifest semantic {name} must be {expected}"));
        }
    }
    let model = manifest
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "model manifest has no model identifier".to_string())?
        .to_owned();
    let files = manifest
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| "model manifest files must be an object".to_string())?;
    for required in REQUIRED_FILES {
        if !files.contains_key(*required) {
            return Err(format!(
                "model manifest is missing required component {required}"
            ));
        }
    }
    for bucket in s2mel_buckets {
        require_manifest_file(files, &format!("onnx/s2mel/model-{bucket}.onnx"))?;
    }
    for bucket in bigvgan_buckets {
        require_manifest_file(files, &format!("onnx/bigvgan/model-{bucket}.onnx"))?;
    }
    for (relative, metadata) in files {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(format!("model manifest contains unsafe path {relative}"));
        }
        let expected_bytes = metadata
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("model manifest entry {relative} has no valid byte size"))?;
        let expected_hash = metadata
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .ok_or_else(|| format!("model manifest entry {relative} has no valid SHA-256"))?;
        let path = model_dir.join(relative_path);
        let actual_bytes = path
            .metadata()
            .map_err(|error| format!("model component {} cannot be read: {error}", path.display()))?
            .len();
        if actual_bytes != expected_bytes {
            return Err(format!(
                "model component {relative} size mismatch: expected {expected_bytes}, got {actual_bytes}"
            ));
        }
        let actual_hash = sha256_file(&path)?;
        if !actual_hash.eq_ignore_ascii_case(expected_hash) {
            return Err(format!("model component {relative} SHA-256 mismatch"));
        }
    }
    Ok(ValidatedManifest {
        model,
        sha256: format!("{:x}", Sha256::digest(&manifest_bytes)),
    })
}

fn require_manifest_file(
    files: &serde_json::Map<String, Value>,
    relative: &str,
) -> Result<(), String> {
    if files.contains_key(relative) {
        Ok(())
    } else {
        Err(format!(
            "model manifest is missing declared bucket {relative}"
        ))
    }
}

fn parse_version(value: &str) -> Result<(u64, u64, u64), String> {
    let core = value.split_once('-').map_or(value, |(core, _)| core);
    let components: Vec<_> = core.split('.').collect();
    if components.len() != 3 {
        return Err(format!("invalid runtime version {value}"));
    }
    Ok((
        components[0]
            .parse()
            .map_err(|_| format!("invalid runtime version {value}"))?,
        components[1]
            .parse()
            .map_err(|_| format!("invalid runtime version {value}"))?,
        components[2]
            .parse()
            .map_err(|_| format!("invalid runtime version {value}"))?,
    ))
}

fn validate_u64_array(manifest: &Value, name: &str, expected: &[u64]) -> Result<Vec<u64>, String> {
    let actual: Vec<u64> = manifest
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("model manifest has no {name}"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| format!("model manifest {name} must contain integers"))
        })
        .collect::<Result<_, _>>()?;
    if actual != expected {
        return Err(format!("model manifest {name} must be {expected:?}"));
    }
    Ok(actual)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "model component {} cannot be opened: {error}",
            path.display()
        )
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|error| {
            format!(
                "model component {} cannot be hashed: {error}",
                path.display()
            )
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn runtime_version_uses_minimum_semantics() {
        assert!(
            parse_version("0.1.0").unwrap() <= parse_version(env!("CARGO_PKG_VERSION")).unwrap()
        );
        assert!(
            parse_version("0.2.0").unwrap() > parse_version(env!("CARGO_PKG_VERSION")).unwrap()
        );
        assert!(parse_version("invalid").is_err());
    }

    #[test]
    fn rejects_hash_mismatch_and_unsafe_paths() {
        let directory = tempdir().unwrap();
        let mut files = serde_json::Map::new();
        for required in REQUIRED_FILES {
            let path = directory.path().join(required);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, required.as_bytes()).unwrap();
            files.insert(
                (*required).into(),
                json!({"bytes": required.len(), "sha256": format!("{:x}", Sha256::digest(required.as_bytes()))}),
            );
        }
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&json!({
                "format_version": 1,
                "model": "IndexTTS-2.5",
                "runtime": "index-tts-rust",
                "minimum_runtime_version": env!("CARGO_PKG_VERSION"),
                "s2mel_buckets": [256, 512, 1024],
                "bigvgan_buckets": [256, 512],
                "sample_rate": 22050,
                "semantic": {"start_token": 8192, "stop_token": 8193, "max_tokens": 1815},
                "files": files,
            }))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            validate_model_manifest(directory.path()).unwrap().model,
            "IndexTTS-2.5"
        );
        std::fs::write(directory.path().join(REQUIRED_FILES[0]), b"corrupt").unwrap();
        assert!(validate_model_manifest(directory.path())
            .unwrap_err()
            .contains("size mismatch"));
    }
}
