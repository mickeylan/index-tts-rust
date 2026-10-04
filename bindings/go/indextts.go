package indextts

/*
#cgo CFLAGS: -I../../crates/indextts-ffi
#include "indextts.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

var nativeMu sync.Mutex

var ErrCancelled = errors.New("IndexTTS generation cancelled")

func nativeError() error {
	if p := C.indextts_last_error(); p != nil {
		return errors.New(C.GoString(p))
	}
	return errors.New("IndexTTS native call failed")
}

type Device string

const (
	DeviceCPU  Device = "cpu"
	DeviceCUDA Device = "cuda"
)

type LoadOptions struct {
	ModelDir    string
	Device      Device
	DeviceIndex int
}

type Capabilities struct {
	ABIMajor, ABIMinor                                                   uint32
	SampleRate, MaxReferenceSeconds                                      uint32
	MaxSemanticTokens, MaxConcurrentRequests                             uint32
	SupportsCUDA, SupportsCPU                                            bool
	SupportsCancellation, SupportsRequestCancel                          bool
	SupportsVoiceCache                                                   bool
	SupportsEmotionText, SupportsEmotionReference, SupportsEmotionVector bool
	SupportsTargetDuration, SupportsSampling, SupportsBeamSearch         bool
}

type ModelInfo struct {
	RuntimeVersion, ModelVersion, ManifestSHA256 string
	Backend, Device                              string
}

type VoiceInfo struct {
	ReferenceSHA256                  string
	DurationSeconds                  float32
	SourceSampleRate, SourceChannels uint32
	CacheBytes                       uint64
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
	if C.indextts_get_capabilities(&value) != C.INDEXTTS_OK {
		return Capabilities{}, nativeError()
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

type Model struct {
	life sync.RWMutex
	call sync.Mutex
	h    C.indextts_model_t
}

func Load(modelDir string) (*Model, error) {
	return LoadWithOptions(LoadOptions{ModelDir: modelDir, Device: DeviceCPU})
}

func LoadWithOptions(loadOptions LoadOptions) (*Model, error) {
	nativeMu.Lock()
	defer nativeMu.Unlock()
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
	if C.indextts_model_load(&options, &handle) != C.INDEXTTS_OK {
		return nil, nativeError()
	}
	model := &Model{h: handle}
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
	if C.indextts_model_get_info(m.h, &value) != C.INDEXTTS_OK {
		return ModelInfo{}, nativeError()
	}
	return ModelInfo{
		RuntimeVersion: fixedString(&value.runtime_version[0], 64), ModelVersion: fixedString(&value.model_version[0], 64),
		ManifestSHA256: fixedString(&value.model_manifest_sha256[0], 65), Backend: fixedString(&value.backend[0], 32),
		Device: fixedString(&value.device[0], 32),
	}, nil
}

func (m *Model) Close() error {
	m.life.Lock()
	defer m.life.Unlock()
	m.call.Lock()
	defer m.call.Unlock()
	if m.h != nil {
		C.indextts_model_free(m.h)
		m.h = nil
	}
	return nil
}

type Voice struct {
	mu    sync.Mutex
	model *Model
	h     C.indextts_voice_t
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
	nativeMu.Lock()
	status := C.indextts_voice_prepare(m.h, value, &handle)
	if status != C.INDEXTTS_OK {
		err := nativeError()
		nativeMu.Unlock()
		return nil, err
	}
	nativeMu.Unlock()
	voice := &Voice{model: m, h: handle}
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
	if C.indextts_voice_get_info(v.h, &value) != C.INDEXTTS_OK {
		return VoiceInfo{}, nativeError()
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
		C.indextts_voice_free(v.h)
		v.h = nil
	}
	return nil
}

type Options struct {
	Language       string
	Seed           uint64
	DurationFactor float32
}

type Audio struct {
	Samples    []float32
	SampleRate uint32
	Channels   uint32
}

type EmotionMode int32

const (
	EmotionNone EmotionMode = iota
	EmotionText
	EmotionReference
	EmotionVector
)

type Emotion struct {
	mu    sync.Mutex
	model *Model
	h     C.indextts_emotion_t
}

type EmotionOptions struct {
	Mode      EmotionMode
	Text      string
	Reference *Emotion
	Vector    []float32
	Strength  float32
}

type OptionsV2 struct {
	Options
	Emotion EmotionOptions
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
	nativeMu.Lock()
	status := C.indextts_emotion_prepare_reference(m.h, value, &handle)
	if status != C.INDEXTTS_OK {
		err := nativeError()
		nativeMu.Unlock()
		return nil, err
	}
	nativeMu.Unlock()
	emotion := &Emotion{model: m, h: handle}
	runtime.SetFinalizer(emotion, (*Emotion).Close)
	return emotion, nil
}

func (e *Emotion) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.h != nil {
		C.indextts_emotion_free(e.h)
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
	if C.indextts_model_cancel(m.h) != C.INDEXTTS_OK {
		return errors.New("IndexTTS cancellation request failed")
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
	cancelWatchDone := make(chan struct{})
	cancelWatchStopped := make(chan struct{})
	go func() {
		defer close(cancelWatchStopped)
		select {
		case <-ctx.Done():
			_ = m.Cancel()
		case <-cancelWatchDone:
		}
	}()
	var output C.indextts_audio_out_t
	nativeMu.Lock()
	status := C.indextts_generate(m.h, v.h, &config, &output)
	var nativeErr error
	if status != C.INDEXTTS_OK && status != C.INDEXTTS_CANCELLED {
		nativeErr = nativeError()
	}
	nativeMu.Unlock()
	close(cancelWatchDone)
	<-cancelWatchStopped
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
	nativeSamples := unsafe.Slice((*float32)(unsafe.Pointer(output.samples)), count)
	floats := append([]float32(nil), nativeSamples...)
	runtime.KeepAlive(v)
	return Audio{Samples: floats, SampleRate: uint32(output.sample_rate), Channels: uint32(output.channels)}, nil
}

func (m *Model) GenerateV2(v *Voice, text string, options OptionsV2) (Audio, error) {
	return m.GenerateV2Context(context.Background(), v, text, options)
}

func (m *Model) GenerateV2Context(ctx context.Context, v *Voice, text string, options OptionsV2) (Audio, error) {
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
			return Audio{}, errors.New("emotion reference is required")
		}
		emotion.mu.Lock()
		defer emotion.mu.Unlock()
		if emotion.h == nil || emotion.model != m {
			return Audio{}, errors.New("emotion is closed or belongs to another model")
		}
		config.emotion.reference = emotion.h
	} else if options.Emotion.Mode == EmotionText || options.Emotion.Mode == EmotionVector {
		return Audio{}, errors.New("emotion text/vector is not supported by this runtime")
	}
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
	nativeMu.Lock()
	status := C.indextts_generate_v2(m.h, v.h, &config, &output)
	var nativeErr error
	if status != C.INDEXTTS_OK && status != C.INDEXTTS_CANCELLED {
		nativeErr = nativeError()
	}
	nativeMu.Unlock()
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
	runtime.KeepAlive(emotion)
	return Audio{Samples: floats, SampleRate: uint32(output.sample_rate), Channels: uint32(output.channels)}, nil
}

func Version() string { return C.GoString(C.indextts_version()) }
