//go:build windows && cgo && indextts_native

package indextts

/*
#cgo windows LDFLAGS: -lindextts
#include "indextts.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"time"
	"unsafe"
)

func nativeError() error {
	if p := C.indextts_last_error(); p != nil {
		return errors.New(C.GoString(p))
	}
	return errors.New("IndexTTS native call failed")
}

// nativeCall keeps the C call and last-error lookup on the same OS thread.
// The native runtime stores its last error in thread-local storage.
func nativeCall(call func() C.int32_t) (C.int32_t, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	status := call()
	if status != C.INDEXTTS_OK {
		return status, nativeError()
	}
	return status, nil
}

func fixedString(pointer *C.char, length int) string {
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(pointer)), length)
	end := 0
	for end < len(bytes) && bytes[end] != 0 {
		end++
	}
	return string(bytes[:end])
}

func ABIVersion() uint32 { return uint32(C.indextts_abi_version()) }

func GetCapabilities() (Capabilities, error) {
	var value C.indextts_capabilities_t
	if _, err := nativeCall(func() C.int32_t { return C.indextts_get_capabilities(&value) }); err != nil {
		return Capabilities{}, err
	}
	return Capabilities{
		ABIMajor: uint32(value.abi_major), ABIMinor: uint32(value.abi_minor), SampleRate: uint32(value.sample_rate),
		MaxReferenceSeconds: uint32(value.max_reference_seconds), MaxSemanticTokens: uint32(value.max_semantic_tokens),
		MaxConcurrentRequests: uint32(value.max_concurrent_requests_per_model), SupportsCUDA: value.supports_cuda != 0,
		SupportsCPU: value.supports_cpu != 0, SupportsCancellation: value.supports_cancellation != 0,
		SupportsRequestCancel: value.supports_request_cancellation != 0, SupportsVoiceCache: value.supports_voice_cache != 0,
		SupportsEmotionText: value.supports_emotion_text != 0, SupportsEmotionReference: value.supports_emotion_reference != 0,
		SupportsEmotionVector: value.supports_emotion_vector != 0, SupportsTargetDuration: value.supports_target_duration != 0,
		SupportsSampling: value.supports_sampling != 0, SupportsBeamSearch: value.supports_beam_search != 0,
	}, nil
}

func Load(modelDir string) (*Model, error) {
	return LoadWithOptions(LoadOptions{ModelDir: modelDir, Device: DeviceCPU})
}

func LoadWithOptions(loadOptions LoadOptions) (*Model, error) {
	if loadOptions.ModelDir == "" {
		return nil, errors.New("IndexTTS model directory is empty")
	}
	path := C.CString(loadOptions.ModelDir)
	defer C.free(unsafe.Pointer(path))
	var options C.indextts_model_options_t
	C.indextts_model_options_init(&options)
	options.model_dir = path
	switch loadOptions.Device {
	case "", DeviceCPU:
		options.device_index = -1
	case DeviceCUDA:
		if loadOptions.DeviceIndex < 0 {
			return nil, errors.New("IndexTTS CUDA device index must be non-negative")
		}
		options.device_index = C.int32_t(loadOptions.DeviceIndex)
	default:
		return nil, errors.New("unsupported IndexTTS device: " + string(loadOptions.Device))
	}
	var handle C.indextts_model_t
	if _, err := nativeCall(func() C.int32_t { return C.indextts_model_load(&options, &handle) }); err != nil {
		return nil, err
	}
	model := &Model{h: unsafe.Pointer(handle)}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

func (m *Model) Info() (ModelInfo, error) {
	m.life.RLock()
	defer m.life.RUnlock()
	if m.h == nil {
		return ModelInfo{}, errors.New("IndexTTS model is closed")
	}
	var value C.indextts_model_info_t
	if _, err := nativeCall(func() C.int32_t {
		return C.indextts_model_get_info(C.indextts_model_t(m.h), &value)
	}); err != nil {
		return ModelInfo{}, err
	}
	return ModelInfo{
		RuntimeVersion: fixedString(&value.runtime_version[0], 64), ModelVersion: fixedString(&value.model_version[0], 64),
		ManifestSHA256: fixedString(&value.model_manifest_sha256[0], 65), Backend: fixedString(&value.backend[0], 32),
		Device: fixedString(&value.device[0], 32),
	}, nil
}

func (m *Model) Health() (Health, error) {
	m.life.RLock()
	defer m.life.RUnlock()
	if m.h == nil {
		return Health{}, errors.New("IndexTTS model is closed")
	}
	var value C.indextts_health_t
	if _, err := nativeCall(func() C.int32_t {
		return C.indextts_model_health(C.indextts_model_t(m.h), &value)
	}); err != nil {
		return Health{}, err
	}
	return Health{
		Loaded: value.loaded != 0, DeviceHealthy: value.device_healthy != 0,
		VoiceCacheEntries: uint64(value.voice_cache_entries), VoiceCacheBytes: uint64(value.voice_cache_bytes),
		ActiveRequests: uint64(value.active_requests), QueuedRequests: uint64(value.queued_requests),
		LastError: fixedString(&value.last_error[0], 512),
	}, nil
}

func (m *Model) Close() error {
	m.life.Lock()
	defer m.life.Unlock()
	m.call.Lock()
	defer m.call.Unlock()
	if m.h != nil {
		C.indextts_model_free(C.indextts_model_t(m.h))
		m.h = nil
	}
	return nil
}

func (m *Model) PrepareVoicePCM(samples []float32, sampleRate, channels uint32) (*Voice, error) {
	m.life.RLock()
	defer m.life.RUnlock()
	m.call.Lock()
	defer m.call.Unlock()
	if m.h == nil {
		return nil, errors.New("IndexTTS model is closed")
	}
	if len(samples) == 0 || sampleRate == 0 || channels == 0 || len(samples)%int(channels) != 0 {
		return nil, errors.New("invalid PCM samples, sample rate, or channel count")
	}
	memory := C.malloc(C.size_t(len(samples)) * C.size_t(unsafe.Sizeof(C.float(0))))
	if memory == nil {
		return nil, errors.New("failed to allocate native PCM buffer")
	}
	defer C.free(memory)
	copy(unsafe.Slice((*float32)(memory), len(samples)), samples)
	var handle C.indextts_voice_t
	_, err := nativeCall(func() C.int32_t {
		return C.indextts_voice_prepare_pcm(C.indextts_model_t(m.h), (*C.float)(memory), C.size_t(len(samples)), C.uint32_t(sampleRate), C.uint32_t(channels), &handle)
	})
	if err != nil {
		return nil, err
	}
	voice := &Voice{model: m, h: unsafe.Pointer(handle)}
	runtime.SetFinalizer(voice, (*Voice).Close)
	return voice, nil
}

func (m *Model) PrepareVoice(path string) (*Voice, error) {
	m.life.RLock()
	defer m.life.RUnlock()
	m.call.Lock()
	defer m.call.Unlock()
	if m.h == nil {
		return nil, errors.New("IndexTTS model is closed")
	}
	value := C.CString(path)
	defer C.free(unsafe.Pointer(value))
	var handle C.indextts_voice_t
	_, err := nativeCall(func() C.int32_t {
		return C.indextts_voice_prepare(C.indextts_model_t(m.h), value, &handle)
	})
	if err != nil {
		return nil, err
	}
	voice := &Voice{model: m, h: unsafe.Pointer(handle)}
	runtime.SetFinalizer(voice, (*Voice).Close)
	return voice, nil
}

func (v *Voice) Info() (VoiceInfo, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.h == nil {
		return VoiceInfo{}, errors.New("IndexTTS voice is closed")
	}
	var value C.indextts_voice_info_t
	if _, err := nativeCall(func() C.int32_t {
		return C.indextts_voice_get_info(C.indextts_voice_t(v.h), &value)
	}); err != nil {
		return VoiceInfo{}, err
	}
	return VoiceInfo{
		ReferenceSHA256: fixedString(&value.reference_sha256[0], 65), DurationSeconds: float32(value.duration_seconds),
		SourceSampleRate: uint32(value.source_sample_rate), SourceChannels: uint32(value.source_channels),
		CacheBytes: uint64(value.cache_bytes),
	}, nil
}

func (v *Voice) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.h != nil {
		C.indextts_voice_free(C.indextts_voice_t(v.h))
		v.h = nil
	}
	return nil
}

func (m *Model) PrepareEmotionReference(path string) (*Emotion, error) {
	m.life.RLock()
	defer m.life.RUnlock()
	m.call.Lock()
	defer m.call.Unlock()
	if m.h == nil {
		return nil, errors.New("IndexTTS model is closed")
	}
	value := C.CString(path)
	defer C.free(unsafe.Pointer(value))
	var handle C.indextts_emotion_t
	_, err := nativeCall(func() C.int32_t {
		return C.indextts_emotion_prepare_reference(C.indextts_model_t(m.h), value, &handle)
	})
	if err != nil {
		return nil, err
	}
	emotion := &Emotion{model: m, h: unsafe.Pointer(handle)}
	runtime.SetFinalizer(emotion, (*Emotion).Close)
	return emotion, nil
}

func (e *Emotion) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.h != nil {
		C.indextts_emotion_free(C.indextts_emotion_t(e.h))
		e.h = nil
	}
	return nil
}

func (m *Model) Generate(v *Voice, text string, options Options) (Audio, error) {
	return m.GenerateContext(context.Background(), v, text, options)
}

func (m *Model) Cancel() error {
	m.life.RLock()
	defer m.life.RUnlock()
	if m.h == nil {
		return errors.New("IndexTTS model is closed")
	}
	if _, err := nativeCall(func() C.int32_t {
		return C.indextts_model_cancel(C.indextts_model_t(m.h))
	}); err != nil {
		return err
	}
	return nil
}

func (m *Model) GenerateContext(ctx context.Context, v *Voice, text string, options Options) (Audio, error) {
	if ctx == nil {
		return Audio{}, errors.New("context is nil")
	}
	m.life.RLock()
	defer m.life.RUnlock()
	m.call.Lock()
	defer m.call.Unlock()
	if v == nil {
		return Audio{}, errors.New("model or voice is closed")
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if m.h == nil || v.h == nil || v.model != m {
		return Audio{}, errors.New("model or voice is closed or mismatched")
	}
	if options.Language == "" {
		options.Language = "ZH"
	}
	if options.DurationFactor == 0 {
		options.DurationFactor = 1
	}
	ctext, clang := C.CString(text), C.CString(options.Language)
	defer C.free(unsafe.Pointer(ctext))
	defer C.free(unsafe.Pointer(clang))
	var config C.indextts_generate_options_t
	C.indextts_generate_options_init(&config)
	config.text, config.language = ctext, clang
	config.seed, config.duration_factor = C.uint64_t(options.Seed), C.float(options.DurationFactor)
	if err := ctx.Err(); err != nil {
		return Audio{}, err
	}
	done, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		select {
		case <-ctx.Done():
			_ = m.Cancel()
		case <-done:
		}
	}()
	var output C.indextts_audio_out_t
	status, nativeErr := nativeCall(func() C.int32_t {
		return C.indextts_generate(C.indextts_model_t(m.h), C.indextts_voice_t(v.h), &config, &output)
	})
	close(done)
	<-stopped
	if status == C.INDEXTTS_CANCELLED {
		if err := ctx.Err(); err != nil {
			return Audio{}, err
		}
		return Audio{}, ErrCancelled
	}
	if status != C.INDEXTTS_OK {
		return Audio{}, nativeErr
	}
	defer C.indextts_audio_free(&output)
	count := int(output.sample_count)
	if count < 0 || C.size_t(count) != output.sample_count {
		return Audio{}, errors.New("native audio is too large for this Go process")
	}
	floats := append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(output.samples)), count)...)
	runtime.KeepAlive(v)
	return Audio{Samples: floats, SampleRate: uint32(output.sample_rate), Channels: uint32(output.channels)}, nil
}

func (m *Model) GenerateV2(v *Voice, text string, options OptionsV2) (Audio, error) {
	return m.GenerateV2Context(context.Background(), v, text, options)
}

func (m *Model) GenerateV2Context(ctx context.Context, v *Voice, text string, options OptionsV2) (Audio, error) {
	result, err := m.GenerateV2ResultContext(ctx, v, text, options)
	return result.Audio, err
}

func (m *Model) GenerateV2Result(v *Voice, text string, options OptionsV2) (GenerationResult, error) {
	return m.GenerateV2ResultContext(context.Background(), v, text, options)
}

func (m *Model) GenerateV2ResultContext(ctx context.Context, v *Voice, text string, options OptionsV2) (GenerationResult, error) {
	if ctx == nil {
		return GenerationResult{}, errors.New("context is nil")
	}
	m.life.RLock()
	defer m.life.RUnlock()
	m.call.Lock()
	defer m.call.Unlock()
	if v == nil {
		return GenerationResult{}, errors.New("model or voice is closed")
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if m.h == nil || v.h == nil || v.model != m {
		return GenerationResult{}, errors.New("model or voice is closed or mismatched")
	}
	if options.Language == "" {
		options.Language = "ZH"
	}
	if options.DurationFactor == 0 {
		options.DurationFactor = 1
	}
	ctext, clang := C.CString(text), C.CString(options.Language)
	defer C.free(unsafe.Pointer(ctext))
	defer C.free(unsafe.Pointer(clang))
	var config C.indextts_generate_options_v2_t
	C.indextts_generate_options_v2_init(&config)
	config.base.text, config.base.language = ctext, clang
	config.base.seed, config.base.duration_factor = C.uint64_t(options.Seed), C.float(options.DurationFactor)
	config.emotion.mode = C.int32_t(options.Emotion.Mode)
	config.emotion.strength = C.float(options.Emotion.Strength)
	var emotion *Emotion
	if options.Emotion.Mode == EmotionReference {
		emotion = options.Emotion.Reference
		if emotion == nil {
			return GenerationResult{}, errors.New("emotion reference is required")
		}
		emotion.mu.Lock()
		defer emotion.mu.Unlock()
		if emotion.h == nil || emotion.model != m {
			return GenerationResult{}, errors.New("emotion is closed or belongs to another model")
		}
		config.emotion.reference = C.indextts_emotion_t(emotion.h)
	} else if options.Emotion.Mode == EmotionText {
		return GenerationResult{}, errors.New("emotion text is not supported by this runtime")
	} else if options.Emotion.Mode == EmotionVector {
		if len(options.Emotion.Vector) != 8 {
			return GenerationResult{}, errors.New("emotion vector must contain exactly 8 values")
		}
		vectorMemory := C.malloc(C.size_t(len(options.Emotion.Vector)) * C.size_t(unsafe.Sizeof(C.float(0))))
		if vectorMemory == nil {
			return GenerationResult{}, errors.New("failed to allocate native emotion vector")
		}
		defer C.free(vectorMemory)
		copy(unsafe.Slice((*float32)(vectorMemory), 8), options.Emotion.Vector)
		config.emotion.vector = (*C.float)(vectorMemory)
		config.emotion.vector_length = 8
	}
	if err := ctx.Err(); err != nil {
		return GenerationResult{}, err
	}
	var request C.indextts_request_t
	if _, err := nativeCall(func() C.int32_t {
		return C.indextts_request_create(C.indextts_model_t(m.h), &request)
	}); err != nil {
		return GenerationResult{}, err
	}
	defer C.indextts_request_free(request)
	done, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		select {
		case <-ctx.Done():
			_, _ = nativeCall(func() C.int32_t { return C.indextts_request_cancel(request) })
		case <-done:
		}
	}()
	var output C.indextts_generation_result_t
	status, nativeErr := nativeCall(func() C.int32_t {
		return C.indextts_generate_result_request_v2(C.indextts_model_t(m.h), request, C.indextts_voice_t(v.h), &config, &output)
	})
	close(done)
	<-stopped
	if status == C.INDEXTTS_CANCELLED {
		if err := ctx.Err(); err != nil {
			return GenerationResult{}, err
		}
		return GenerationResult{}, ErrCancelled
	}
	if status != C.INDEXTTS_OK {
		return GenerationResult{}, nativeErr
	}
	defer C.indextts_audio_free(&output.audio)
	count := int(output.audio.sample_count)
	if count < 0 || C.size_t(count) != output.audio.sample_count {
		return GenerationResult{}, errors.New("native audio is too large for this Go process")
	}
	floats := append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(output.audio.samples)), count)...)
	runtime.KeepAlive(v)
	runtime.KeepAlive(emotion)
	runtime.KeepAlive(options.Emotion.Vector)
	milliseconds := func(value C.float) time.Duration { return time.Duration(float64(value) * float64(time.Millisecond)) }
	return GenerationResult{
		Audio: Audio{Samples: floats, SampleRate: uint32(output.audio.sample_rate), Channels: uint32(output.audio.channels)},
		Info: GenerationInfo{
			SemanticTokens: uint32(output.info.semantic_token_count), GeneratedSeconds: float32(output.info.generated_seconds),
			ReferenceEncode: milliseconds(output.info.reference_encode_ms), GPT: milliseconds(output.info.gpt_ms),
			SemanticCodec: milliseconds(output.info.semantic_codec_ms), S2Mel: milliseconds(output.info.s2mel_ms),
			BigVGAN: milliseconds(output.info.bigvgan_ms), Total: milliseconds(output.info.total_ms),
			Peak: float32(output.info.peak), RMS: float32(output.info.rms), SilenceRatio: float32(output.info.silence_ratio),
			Seed: uint64(output.info.seed),
		},
	}, nil
}

func Version() string { return C.GoString(C.indextts_version()) }
