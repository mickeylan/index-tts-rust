//! Audio processing for IndexTTS-2.5
//!
//! Handles:
//! - WAV reading via hound
//! - Mono conversion
//! - Reference audio truncation to 15 seconds
//! - NaN/Inf/silence validation
//! - PCM/WAV output

use indextts_core::{AudioBuffer, IndexTtsError, Result};
use std::path::Path;

/// Default sample rate for IndexTTS (22.05 kHz)
pub const INDEXTTS_SAMPLE_RATE: u32 = 22_050;

/// Maximum reference audio duration in seconds
pub const MAX_REFERENCE_DURATION: f64 = 15.0;

/// Minimum reference audio duration in seconds
pub const MIN_REFERENCE_DURATION: f64 = 0.25;

/// Target sample rate for Wav2Vec2-BERT (16 kHz)
pub const WAV2VEC_SAMPLE_RATE: u32 = 16_000;

/// Read audio file and return samples (WAV only for now)
pub fn read_audio(path: &Path) -> Result<AudioBuffer> {
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    
    match extension.as_str() {
        "wav" => read_wav(path),
        _ => {
            // For non-WAV files, return an error with helpful message
            Err(IndexTtsError::InvalidAudio(format!(
                "Unsupported audio format: {}. Currently only WAV is supported. \
                 Use ffmpeg to convert: ffmpeg -i input.mp3 output.wav",
                extension
            )))
        }
    }
}

/// Read WAV file using hound
fn read_wav(path: &Path) -> Result<AudioBuffer> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to open WAV: {}", e)))?;
    
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    
    // Convert to mono f32 samples
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            match spec.bits_per_sample {
                8 => {
                    let samples: Vec<i8> = reader.into_samples::<i8>()
                        .filter_map(|s| s.ok())
                        .collect();
                    samples.chunks(channels)
                        .map(|chunk| {
                            let sum: f32 = chunk.iter().map(|&s| s as f32 / 128.0).sum();
                            sum / channels as f32
                        })
                        .collect()
                }
                16 => {
                    let samples: Vec<i16> = reader.into_samples::<i16>()
                        .filter_map(|s| s.ok())
                        .collect();
                    samples.chunks(channels)
                        .map(|chunk| {
                            let sum: f32 = chunk.iter().map(|&s| s as f32 / 32768.0).sum();
                            sum / channels as f32
                        })
                        .collect()
                }
                32 => {
                    let samples: Vec<i32> = reader.into_samples::<i32>()
                        .filter_map(|s| s.ok())
                        .collect();
                    samples.chunks(channels)
                        .map(|chunk| {
                            let sum: f32 = chunk.iter().map(|&s| s as f32 / 2147483648.0).sum();
                            sum / channels as f32
                        })
                        .collect()
                }
                _ => {
                    return Err(IndexTtsError::InvalidAudio(format!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    )));
                }
            }
        }
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect();
            samples.chunks(channels)
                .map(|chunk| {
                    let sum: f32 = chunk.iter().sum();
                    sum / channels as f32
                })
                .collect()
        }
    };
    
    if samples.is_empty() {
        return Err(IndexTtsError::InvalidAudio("No samples in WAV file".into()));
    }
    
    Ok(AudioBuffer::new(samples, sample_rate))
}

/// Process reference audio for IndexTTS
pub fn process_reference_audio(path: &Path) -> Result<(AudioBuffer, AudioBuffer)> {
    let mut audio = read_audio(path)?;
    
    // Check minimum duration
    let min_samples = (MIN_REFERENCE_DURATION * audio.sample_rate as f64) as usize;
    if audio.samples.len() < min_samples {
        return Err(IndexTtsError::InvalidReferenceAudio(format!(
            "Audio too short: {} seconds, minimum {} seconds",
            audio.duration(),
            MIN_REFERENCE_DURATION
        )));
    }
    
    // Truncate to 15 seconds
    let max_samples = (MAX_REFERENCE_DURATION * audio.sample_rate as f64) as usize;
    if audio.samples.len() > max_samples {
        audio.samples.truncate(max_samples);
    }
    
    // Clamp to [-1, 1]
    for sample in &mut audio.samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
    
    // Validate no NaN/Inf
    for (i, &sample) in audio.samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(IndexTtsError::InvalidReferenceAudio(format!(
                "Non-finite sample at index {}: {}",
                i, sample
            )));
        }
    }
    
    // For now, return the same audio (no resampling)
    // TODO: Add proper resampling using rubato
    let audio_22050 = if audio.sample_rate != INDEXTTS_SAMPLE_RATE {
        // Placeholder: would need resampling
        audio.clone()
    } else {
        audio.clone()
    };
    
    let audio_16000 = audio; // Placeholder: would need resampling
    
    Ok((audio_16000, audio_22050))
}

/// Validate audio buffer
pub fn validate_audio(audio: &AudioBuffer) -> Result<()> {
    if audio.samples.is_empty() {
        return Err(IndexTtsError::InvalidAudio("Empty audio".into()));
    }
    
    if audio.sample_rate == 0 {
        return Err(IndexTtsError::InvalidAudio("Invalid sample rate".into()));
    }
    
    // Check for NaN/Inf
    for (i, &sample) in audio.samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(IndexTtsError::InvalidAudio(format!(
                "Non-finite sample at index {}: {}",
                i, sample
            )));
        }
    }
    
    // Check for silence
    let max_amplitude = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_amplitude < 0.001 {
        return Err(IndexTtsError::InvalidAudio("Audio is silent or near-silent".into()));
    }
    
    // Check for clipping
    let clipping_threshold = 0.99;
    let clipping_count = audio.samples.iter()
        .filter(|&&s| s.abs() > clipping_threshold)
        .count();
    let clipping_ratio = clipping_count as f32 / audio.samples.len() as f32;
    if clipping_ratio > 0.01 {
        return Err(IndexTtsError::InvalidAudio(format!(
            "Audio has excessive clipping: {:.2}%",
            clipping_ratio * 100.0
        )));
    }
    
    Ok(())
}

/// Save audio to WAV file
pub fn save_wav(path: &Path, audio: &AudioBuffer) -> Result<()> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to create WAV writer: {}", e)))?;
    
    for sample in &audio.samples {
        let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(sample_i16)
            .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to write sample: {}", e)))?;
    }
    
    writer.finalize()
        .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to finalize WAV: {}", e)))?;
    
    Ok(())
}

/// Convert AudioBuffer to Vec<i16> PCM
pub fn to_pcm_i16(audio: &AudioBuffer) -> Vec<i16> {
    audio.samples.iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Convert AudioBuffer to Vec<u8> 8-bit PCM
pub fn to_pcm_u8(audio: &AudioBuffer) -> Vec<u8> {
    audio.samples.iter()
        .map(|&s| ((s + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_rate_constants() {
        assert_eq!(INDEXTTS_SAMPLE_RATE, 22050);
        assert_eq!(WAV2VEC_SAMPLE_RATE, 16000);
        assert_eq!(MAX_REFERENCE_DURATION, 15.0);
        assert_eq!(MIN_REFERENCE_DURATION, 0.25);
    }
}
