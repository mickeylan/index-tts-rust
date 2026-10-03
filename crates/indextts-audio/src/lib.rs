//! Audio processing for IndexTTS-2.5
//!
//! Handles:
//! - WAV reading via hound
//! - Mono conversion
//! - Reference audio truncation to 15 seconds
//! - NaN/Inf/silence validation
//! - PCM/WAV output

use indextts_core::{AudioBuffer, IndexTtsError, Result};
use rubato::{FftFixedInOut, Resampler};
use rustfft::{num_complex::Complex, FftPlanner};
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
    let extension = path
        .extension()
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
        hound::SampleFormat::Int => match spec.bits_per_sample {
            8 => {
                let samples: Vec<i8> = reader.into_samples::<i8>().filter_map(|s| s.ok()).collect();
                samples
                    .chunks(channels)
                    .map(|chunk| {
                        let sum: f32 = chunk.iter().map(|&s| s as f32 / 128.0).sum();
                        sum / channels as f32
                    })
                    .collect()
            }
            16 => {
                let samples: Vec<i16> = reader
                    .into_samples::<i16>()
                    .filter_map(|s| s.ok())
                    .collect();
                samples
                    .chunks(channels)
                    .map(|chunk| {
                        let sum: f32 = chunk.iter().map(|&s| s as f32 / 32768.0).sum();
                        sum / channels as f32
                    })
                    .collect()
            }
            32 => {
                let samples: Vec<i32> = reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .collect();
                samples
                    .chunks(channels)
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
        },
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect();
            samples
                .chunks(channels)
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

    validate_audio(&audio)
        .map_err(|error| IndexTtsError::InvalidReferenceAudio(error.to_string()))?;

    let audio_16000 = resample_mono(&audio, WAV2VEC_SAMPLE_RATE)?;
    let audio_22050 = resample_mono(&audio, INDEXTTS_SAMPLE_RATE)?;
    Ok((audio_16000, audio_22050))
}

/// Resample a mono audio buffer with Rubato's band-limited FFT resampler.
pub fn resample_mono(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    if audio.sample_rate == target_rate {
        return Ok(audio.clone());
    }
    if audio.sample_rate == 0 || target_rate == 0 || audio.samples.is_empty() {
        return Err(IndexTtsError::InvalidAudio(
            "cannot resample empty audio or a zero sample rate".into(),
        ));
    }

    let mut resampler =
        FftFixedInOut::<f32>::new(audio.sample_rate as usize, target_rate as usize, 1024, 1)
            .map_err(|error| {
                IndexTtsError::InvalidAudio(format!("resampler setup failed: {error}"))
            })?;
    let delay = resampler.output_delay();
    let expected_len = ((audio.samples.len() as u64 * target_rate as u64
        + audio.sample_rate as u64 / 2)
        / audio.sample_rate as u64) as usize;
    let mut remaining = audio.samples.as_slice();
    let mut output = Vec::with_capacity(expected_len + delay);

    while remaining.len() >= resampler.input_frames_next() {
        let input = [remaining];
        let chunk = resampler
            .process(&input, None)
            .map_err(|error| IndexTtsError::InvalidAudio(format!("resampling failed: {error}")))?;
        remaining = &remaining[resampler.input_frames_next()..];
        output.extend_from_slice(&chunk[0]);
    }
    if !remaining.is_empty() {
        let input = [remaining];
        let chunk = resampler
            .process_partial(Some(&input), None)
            .map_err(|error| {
                IndexTtsError::InvalidAudio(format!("resampling tail failed: {error}"))
            })?;
        output.extend_from_slice(&chunk[0]);
    }

    let end = (delay + expected_len).min(output.len());
    let mut samples = output.get(delay..end).unwrap_or_default().to_vec();
    samples.resize(expected_len, 0.0);
    Ok(AudioBuffer::new(samples, target_rate))
}

/// Features consumed by the Wav2Vec2-BERT ONNX graph.
#[derive(Debug, Clone)]
pub struct SeamlessM4tFeatures {
    pub input_features: Vec<f32>,
    pub attention_mask: Vec<i64>,
    pub frames: usize,
}

/// Reproduce Hugging Face `SeamlessM4TFeatureExtractor` for one 16 kHz waveform.
pub fn seamless_m4t_features(audio: &AudioBuffer) -> Result<SeamlessM4tFeatures> {
    if audio.sample_rate != WAV2VEC_SAMPLE_RATE {
        return Err(IndexTtsError::InvalidAudio(format!(
            "SeamlessM4T requires 16000 Hz audio, got {}",
            audio.sample_rate
        )));
    }
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const BINS: usize = FFT / 2 + 1;
    const MELS: usize = 80;
    if audio.samples.len() < FRAME {
        return Err(IndexTtsError::InvalidAudio(
            "audio is too short for one feature frame".into(),
        ));
    }
    let raw_frames = 1 + (audio.samples.len() - FRAME) / HOP;
    let frames = raw_frames - raw_frames % 2;
    if frames < 2 {
        return Err(IndexTtsError::InvalidAudio(
            "audio is too short after stride-2 stacking".into(),
        ));
    }

    let window: Vec<f64> = (0..FRAME)
        .map(|index| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / (FRAME - 1) as f64).cos();
            hann.powf(0.85)
        })
        .collect();
    let mel_min = 1127.0f64 * (1.0f64 + 20.0 / 700.0).ln();
    let mel_max = 1127.0f64 * (1.0f64 + 8000.0 / 700.0).ln();
    let mel_points: Vec<f64> = (0..MELS + 2)
        .map(|index| mel_min + (mel_max - mel_min) * index as f64 / (MELS + 1) as f64)
        .collect();
    let fft_mels: Vec<f64> = (0..BINS)
        .map(|index| {
            let hz = WAV2VEC_SAMPLE_RATE as f64 / FFT as f64 * index as f64;
            1127.0 * (1.0 + hz / 700.0).ln()
        })
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT);
    let mut mel_frames = vec![0f32; frames * MELS];
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT];
    for frame_index in 0..frames {
        buffer.fill(Complex::new(0.0, 0.0));
        let offset = frame_index * HOP;
        let mean = audio.samples[offset..offset + FRAME]
            .iter()
            .map(|value| *value as f64 * 32768.0)
            .sum::<f64>()
            / FRAME as f64;
        let mut previous = audio.samples[offset] as f64 * 32768.0 - mean;
        buffer[0].re = previous * (1.0 - 0.97) * window[0];
        for index in 1..FRAME {
            let current = audio.samples[offset + index] as f64 * 32768.0 - mean;
            buffer[index].re = (current - 0.97 * previous) * window[index];
            previous = current;
        }
        fft.process(&mut buffer);
        for mel in 0..MELS {
            let left = mel_points[mel];
            let center = mel_points[mel + 1];
            let right = mel_points[mel + 2];
            let mut energy = 0.0f64;
            for bin in 0..BINS {
                let frequency = fft_mels[bin];
                let weight = if frequency >= left && frequency <= center {
                    (frequency - left) / (center - left)
                } else if frequency > center && frequency <= right {
                    (right - frequency) / (right - center)
                } else {
                    0.0
                };
                energy += buffer[bin].norm_sqr() * weight;
            }
            mel_frames[frame_index * MELS + mel] = energy.max(f32::EPSILON as f64).ln() as f32;
        }
    }

    // Normalize each mel channel using sample variance (ddof=1).
    for mel in 0..MELS {
        let mean = (0..frames)
            .map(|frame| mel_frames[frame * MELS + mel] as f64)
            .sum::<f64>()
            / frames as f64;
        let variance = (0..frames)
            .map(|frame| {
                let delta = mel_frames[frame * MELS + mel] as f64 - mean;
                delta * delta
            })
            .sum::<f64>()
            / (frames - 1) as f64;
        let scale = (variance + 1e-7).sqrt();
        for frame in 0..frames {
            mel_frames[frame * MELS + mel] =
                ((mel_frames[frame * MELS + mel] as f64 - mean) / scale) as f32;
        }
    }

    // Stride-2 stacking: [T, 80] -> [T/2, 160].
    let stacked_frames = frames / 2;
    let mut input_features = vec![0f32; stacked_frames * 160];
    for frame in 0..stacked_frames {
        input_features[frame * 160..frame * 160 + 80]
            .copy_from_slice(&mel_frames[(frame * 2) * 80..(frame * 2 + 1) * 80]);
        input_features[frame * 160 + 80..(frame + 1) * 160]
            .copy_from_slice(&mel_frames[(frame * 2 + 1) * 80..(frame * 2 + 2) * 80]);
    }
    Ok(SeamlessM4tFeatures {
        input_features,
        attention_mask: vec![1; stacked_frames],
        frames: stacked_frames,
    })
}

/// Compute the 80-bin reference mel used by S2Mel.
///
/// This matches `s2mel.modules.audio.mel_spectrogram`: reflect padding,
/// periodic Hann window, magnitude STFT, Slaney-normalized mel filters, and
/// natural-log compression.
pub fn reference_mel(audio: &AudioBuffer) -> Result<Vec<f32>> {
    if audio.sample_rate != INDEXTTS_SAMPLE_RATE {
        return Err(IndexTtsError::InvalidAudio(format!(
            "reference mel requires 22050 Hz audio, got {}",
            audio.sample_rate
        )));
    }
    const FFT: usize = 1024;
    const HOP: usize = 256;
    const BINS: usize = FFT / 2 + 1;
    const MELS: usize = 80;
    const PAD: usize = (FFT - HOP) / 2;
    if audio.samples.len() <= PAD {
        return Err(IndexTtsError::InvalidAudio(
            "audio is too short for reference mel padding".into(),
        ));
    }
    let mut padded = Vec::with_capacity(audio.samples.len() + 2 * PAD);
    padded.extend((1..=PAD).rev().map(|index| audio.samples[index]));
    padded.extend_from_slice(&audio.samples);
    padded.extend((1..=PAD).map(|index| audio.samples[audio.samples.len() - 1 - index]));
    let frames = 1 + (padded.len() - FFT) / HOP;

    fn hz_to_slaney_mel(hz: f64) -> f64 {
        if hz < 1000.0 {
            hz / (200.0 / 3.0)
        } else {
            15.0 + (hz / 1000.0).ln() / ((6.4f64).ln() / 27.0)
        }
    }
    fn slaney_mel_to_hz(mel: f64) -> f64 {
        if mel < 15.0 {
            mel * (200.0 / 3.0)
        } else {
            1000.0 * (((6.4f64).ln() / 27.0) * (mel - 15.0)).exp()
        }
    }
    let mel_max = hz_to_slaney_mel(INDEXTTS_SAMPLE_RATE as f64 / 2.0);
    let frequencies: Vec<f64> = (0..MELS + 2)
        .map(|index| slaney_mel_to_hz(mel_max * index as f64 / (MELS + 1) as f64))
        .collect();
    let fft_hz: Vec<f64> = (0..BINS)
        .map(|index| INDEXTTS_SAMPLE_RATE as f64 / FFT as f64 * index as f64)
        .collect();
    let window: Vec<f64> = (0..FFT)
        .map(|index| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / FFT as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT);
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT];
    let mut output = vec![0f32; MELS * frames];
    for frame in 0..frames {
        let offset = frame * HOP;
        for index in 0..FFT {
            buffer[index] = Complex::new(padded[offset + index] as f64 * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for mel in 0..MELS {
            let left = frequencies[mel];
            let center = frequencies[mel + 1];
            let right = frequencies[mel + 2];
            let area_norm = 2.0 / (right - left);
            let mut magnitude = 0.0f64;
            for bin in 0..BINS {
                let hz = fft_hz[bin];
                let weight = ((hz - left) / (center - left))
                    .min((right - hz) / (right - center))
                    .max(0.0)
                    * area_norm;
                magnitude += (buffer[bin].norm_sqr() + 1e-9).sqrt() * weight;
            }
            output[mel * frames + frame] = magnitude.max(1e-5).ln() as f32;
        }
    }
    Ok(output)
}

/// Compute the mean-centered Kaldi fbank consumed by CAMPPlus.
pub fn campplus_fbank(audio: &AudioBuffer) -> Result<Vec<f32>> {
    if audio.sample_rate != WAV2VEC_SAMPLE_RATE {
        return Err(IndexTtsError::InvalidAudio(format!(
            "CAMPPlus requires 16000 Hz audio, got {}",
            audio.sample_rate
        )));
    }
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const BINS: usize = FFT / 2;
    const MELS: usize = 80;
    if audio.samples.len() < FRAME {
        return Err(IndexTtsError::InvalidAudio(
            "audio is too short for CAMPPlus fbank".into(),
        ));
    }
    let frames = 1 + (audio.samples.len() - FRAME) / HOP;
    let window: Vec<f64> = (0..FRAME)
        .map(|index| {
            let hann =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / (FRAME - 1) as f64).cos();
            hann.powf(0.85)
        })
        .collect();
    let mel_min = 1127.0f64 * (1.0f64 + 20.0 / 700.0).ln();
    let mel_max = 1127.0f64 * (1.0f64 + 8000.0 / 700.0).ln();
    let mel_points: Vec<f64> = (0..MELS + 2)
        .map(|index| mel_min + (mel_max - mel_min) * index as f64 / (MELS + 1) as f64)
        .collect();
    let fft_mels: Vec<f64> = (0..BINS)
        .map(|index| {
            let hz = WAV2VEC_SAMPLE_RATE as f64 / FFT as f64 * index as f64;
            1127.0 * (1.0 + hz / 700.0).ln()
        })
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT);
    let mut features = vec![0f32; frames * MELS];
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT];
    for frame_index in 0..frames {
        buffer.fill(Complex::new(0.0, 0.0));
        let offset = frame_index * HOP;
        let mean = audio.samples[offset..offset + FRAME]
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / FRAME as f64;
        let mut previous = audio.samples[offset] as f64 - mean;
        buffer[0].re = previous * (1.0 - 0.97) * window[0];
        for index in 1..FRAME {
            let current = audio.samples[offset + index] as f64 - mean;
            buffer[index].re = (current - 0.97 * previous) * window[index];
            previous = current;
        }
        fft.process(&mut buffer);
        for mel in 0..MELS {
            let left = mel_points[mel];
            let center = mel_points[mel + 1];
            let right = mel_points[mel + 2];
            let mut energy = 0.0f64;
            for bin in 0..BINS {
                let frequency = fft_mels[bin];
                let weight = ((frequency - left) / (center - left))
                    .min((right - frequency) / (right - center))
                    .max(0.0);
                energy += buffer[bin].norm_sqr() * weight;
            }
            features[frame_index * MELS + mel] = energy.max(f32::EPSILON as f64).ln() as f32;
        }
    }
    for mel in 0..MELS {
        let mean = (0..frames)
            .map(|frame| features[frame * MELS + mel] as f64)
            .sum::<f64>()
            / frames as f64;
        for frame in 0..frames {
            features[frame * MELS + mel] -= mean as f32;
        }
    }
    Ok(features)
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
        return Err(IndexTtsError::InvalidAudio(
            "Audio is silent or near-silent".into(),
        ));
    }

    // Check for clipping
    let clipping_threshold = 0.99;
    let clipping_count = audio
        .samples
        .iter()
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
        writer
            .write_sample(sample_i16)
            .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| IndexTtsError::InvalidAudio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Convert AudioBuffer to Vec<i16> PCM
pub fn to_pcm_i16(audio: &AudioBuffer) -> Vec<i16> {
    audio
        .samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Convert AudioBuffer to Vec<u8> 8-bit PCM
pub fn to_pcm_u8(audio: &AudioBuffer) -> Vec<u8> {
    audio
        .samples
        .iter()
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

    #[test]
    fn resampling_produces_exact_target_length_and_rate() {
        let source_rate = 48_000;
        let samples: Vec<f32> = (0..source_rate)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 440.0 * index as f32 / source_rate as f32).sin() * 0.5
            })
            .collect();
        let source = AudioBuffer::new(samples, source_rate);
        let output = resample_mono(&source, WAV2VEC_SAMPLE_RATE).unwrap();
        assert_eq!(output.sample_rate, WAV2VEC_SAMPLE_RATE);
        assert_eq!(output.samples.len(), WAV2VEC_SAMPLE_RATE as usize);
        assert!(output.samples.iter().all(|sample| sample.is_finite()));
        assert!(
            output
                .samples
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0, f32::max)
                > 0.4
        );
    }

    fn test_waveform() -> Vec<f32> {
        (0..WAV2VEC_SAMPLE_RATE)
            .map(|index| {
                let time = index as f32 / WAV2VEC_SAMPLE_RATE as f32;
                0.3 * (2.0 * std::f32::consts::PI * 220.0 * time).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 630.0 * time).sin()
            })
            .collect()
    }

    #[test]
    fn seamless_features_have_expected_shape_and_normalization() {
        let features =
            seamless_m4t_features(&AudioBuffer::new(test_waveform(), WAV2VEC_SAMPLE_RATE)).unwrap();
        assert_eq!(features.frames, 49);
        assert_eq!(features.input_features.len(), 49 * 160);
        assert_eq!(features.attention_mask, vec![1; 49]);
        assert!(features
            .input_features
            .iter()
            .all(|value| value.is_finite()));
        let official = [
            (0, 0.90362066f32),
            (1, 0.80721146),
            (79, -3.3372314),
            (80, -0.2189388),
            (159, -2.5282848),
            (160, 0.40543014),
            (777, -1.7934725),
            (7839, 0.6679816),
        ];
        for (index, expected) in official {
            assert!(
                (features.input_features[index] - expected).abs() < 1e-2,
                "feature {index}: Rust={}, Python={expected}",
                features.input_features[index]
            );
        }
    }

    #[test]
    fn reference_mel_matches_official_implementation() {
        let samples: Vec<f32> = (0..INDEXTTS_SAMPLE_RATE)
            .map(|index| {
                let time = index as f32 / INDEXTTS_SAMPLE_RATE as f32;
                0.3 * (2.0 * std::f32::consts::PI * 220.0 * time).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 630.0 * time).sin()
            })
            .collect();
        let mel = reference_mel(&AudioBuffer::new(samples, INDEXTTS_SAMPLE_RATE)).unwrap();
        assert_eq!(mel.len(), 80 * 86);
        let official = [
            (0, -0.81806934f32),
            (1, -2.5950215),
            (79, -6.8224835),
            (80, -6.619554),
            (159, -5.701_24),
            (160, -5.6991324),
            (777, -6.9009333),
            (6879, -7.3257093),
        ];
        for (index, expected) in official {
            assert!(
                (mel[index] - expected).abs() < 2e-3,
                "mel {index}: Rust={}, Python={expected}",
                mel[index]
            );
        }
    }

    #[test]
    fn campplus_fbank_matches_torchaudio_reference() {
        let features =
            campplus_fbank(&AudioBuffer::new(test_waveform(), WAV2VEC_SAMPLE_RATE)).unwrap();
        assert_eq!(features.len(), 98 * 80);
        let official = [
            (0, 1.4589176f32),
            (1, 1.5607796),
            (79, -0.7501602),
            (80, -0.35348177),
            (159, -0.7501602),
            (160, 0.65457535),
            (777, -0.0000009536743),
            (7839, 0.5490694),
        ];
        for (index, expected) in official {
            assert!(
                (features[index] - expected).abs() < 1e-2,
                "fbank {index}: Rust={}, Python={expected}",
                features[index]
            );
        }
    }
}
