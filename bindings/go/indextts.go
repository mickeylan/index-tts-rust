package indextts

/*
#cgo CFLAGS: -I../../crates/indextts-ffi
#include "indextts.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

var nativeMu sync.Mutex

func nativeError() error {
	if p := C.indextts_last_error(); p != nil {
		return errors.New(C.GoString(p))
	}
	return errors.New("IndexTTS native call failed")
}

type Model struct {
	mu sync.Mutex
	h  C.indextts_model_t
}

func Load(modelDir string) (*Model, error) {
	nativeMu.Lock()
	defer nativeMu.Unlock()
	path := C.CString(modelDir)
	defer C.free(unsafe.Pointer(path))
	var options C.indextts_model_options_t
	C.indextts_model_options_init(&options)
	options.model_dir = path
	var handle C.indextts_model_t
	if C.indextts_model_load(&options, &handle) != C.INDEXTTS_OK {
		return nil, nativeError()
	}
	model := &Model{h: handle}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.h != nil {
		C.indextts_model_free(m.h)
		m.h = nil
	}
	return nil
}

type Voice struct {
	model *Model
	h     C.indextts_voice_t
}

func (m *Model) PrepareVoice(path string) (*Voice, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
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

func (v *Voice) Close() error {
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

func (m *Model) Generate(v *Voice, text string, options Options) (Audio, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.h == nil || v == nil || v.h == nil {
		return Audio{}, errors.New("model or voice is closed")
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
	var output C.indextts_audio_out_t
	nativeMu.Lock()
	status := C.indextts_generate(m.h, v.h, &config, &output)
	if status != C.INDEXTTS_OK {
		err := nativeError()
		nativeMu.Unlock()
		return Audio{}, err
	}
	nativeMu.Unlock()
	defer C.indextts_audio_free(&output)
	samples := C.GoBytes(unsafe.Pointer(output.samples), C.int(output.sample_count*C.size_t(4)))
	floats := make([]float32, output.sample_count)
	for i := range floats {
		floats[i] = *(*float32)(unsafe.Pointer(&samples[i*4]))
	}
	runtime.KeepAlive(v)
	return Audio{Samples: floats, SampleRate: uint32(output.sample_rate), Channels: uint32(output.channels)}, nil
}

func Version() string { return C.GoString(C.indextts_version()) }
