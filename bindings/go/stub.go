//go:build !windows || !cgo || !indextts_native

package indextts

import "context"

func ABIVersion() uint32 { return 0 }

func GetCapabilities() (Capabilities, error) { return Capabilities{}, ErrNativeUnavailable }

func Load(string) (*Model, error) { return nil, ErrNativeUnavailable }

func LoadWithOptions(LoadOptions) (*Model, error) { return nil, ErrNativeUnavailable }

func (m *Model) Info() (ModelInfo, error) { return ModelInfo{}, ErrNativeUnavailable }

func (m *Model) Health() (Health, error) { return Health{}, ErrNativeUnavailable }

func (m *Model) Close() error { return nil }

func (m *Model) PrepareVoicePCM([]float32, uint32, uint32) (*Voice, error) {
	return nil, ErrNativeUnavailable
}

func (m *Model) PrepareVoice(string) (*Voice, error) { return nil, ErrNativeUnavailable }

func (v *Voice) Info() (VoiceInfo, error) { return VoiceInfo{}, ErrNativeUnavailable }

func (v *Voice) Close() error { return nil }

func (m *Model) PrepareEmotionReference(string) (*Emotion, error) {
	return nil, ErrNativeUnavailable
}

func (e *Emotion) Close() error { return nil }

func (m *Model) Generate(*Voice, string, Options) (Audio, error) {
	return Audio{}, ErrNativeUnavailable
}

func (m *Model) Cancel() error { return ErrNativeUnavailable }

func (m *Model) GenerateContext(context.Context, *Voice, string, Options) (Audio, error) {
	return Audio{}, ErrNativeUnavailable
}

func (m *Model) GenerateV2(*Voice, string, OptionsV2) (Audio, error) {
	return Audio{}, ErrNativeUnavailable
}

func (m *Model) GenerateV2Context(context.Context, *Voice, string, OptionsV2) (Audio, error) {
	return Audio{}, ErrNativeUnavailable
}

func (m *Model) GenerateV2Result(*Voice, string, OptionsV2) (GenerationResult, error) {
	return GenerationResult{}, ErrNativeUnavailable
}

func (m *Model) GenerateV2ResultContext(context.Context, *Voice, string, OptionsV2) (GenerationResult, error) {
	return GenerationResult{}, ErrNativeUnavailable
}

func (m *Model) GenerateLongTextResult(*Voice, string, Options, LongTextOptions) (LongTextResult, error) {
	return LongTextResult{}, ErrNativeUnavailable
}

func (m *Model) GenerateLongTextResultContext(context.Context, *Voice, string, Options, LongTextOptions) (LongTextResult, error) {
	return LongTextResult{}, ErrNativeUnavailable
}

func Version() string { return "" }
