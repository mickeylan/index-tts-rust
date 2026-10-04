package indextts

import (
	"errors"
	"sync"
	"time"
	"unsafe"
)

var (
	ErrCancelled         = errors.New("IndexTTS generation cancelled")
	ErrNativeUnavailable = errors.New("IndexTTS native runtime is unavailable; build on Windows with cgo and the indextts_native tag")
)

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

type Health struct {
	Loaded, DeviceHealthy              bool
	VoiceCacheEntries, VoiceCacheBytes uint64
	ActiveRequests, QueuedRequests     uint64
	LastError                          string
}

type VoiceInfo struct {
	ReferenceSHA256                  string
	DurationSeconds                  float32
	SourceSampleRate, SourceChannels uint32
	CacheBytes                       uint64
}

// Model is an opaque loaded IndexTTS model.
type Model struct {
	life sync.RWMutex
	call sync.Mutex
	h    unsafe.Pointer
}

// Voice is opaque conditioning prepared from reference audio.
type Voice struct {
	mu    sync.Mutex
	model *Model
	h     unsafe.Pointer
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

// Emotion is an opaque prepared emotion reference.
type Emotion struct {
	mu    sync.Mutex
	model *Model
	h     unsafe.Pointer
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

type GenerationInfo struct {
	SemanticTokens                                             uint32
	GeneratedSeconds                                           float32
	ReferenceEncode, GPT, SemanticCodec, S2Mel, BigVGAN, Total time.Duration
	Peak, RMS, SilenceRatio                                    float32
	Seed                                                       uint64
}

type GenerationResult struct {
	Audio Audio
	Info  GenerationInfo
}
