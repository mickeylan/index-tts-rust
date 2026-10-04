//go:build !windows || !cgo || !indextts_native

package indextts

import (
	"context"
	"errors"
	"testing"
)

func TestStubReportsNativeUnavailable(t *testing.T) {
	if ABIVersion() != 0 {
		t.Fatalf("stub ABI version = %#x, want 0", ABIVersion())
	}
	if Version() != "" {
		t.Fatalf("stub version = %q, want empty", Version())
	}

	checks := []struct {
		name string
		err  error
	}{
		{"capabilities", capabilityError()},
		{"load", loadError()},
		{"model info", modelInfoError()},
		{"model health", modelHealthError()},
		{"prepare voice", prepareVoiceError()},
		{"prepare PCM voice", prepareVoicePCMError()},
		{"voice info", voiceInfoError()},
		{"prepare emotion", prepareEmotionError()},
		{"generate", generateError()},
		{"generate context", generateContextError()},
		{"generate V2", generateV2Error()},
		{"generate V2 context", generateV2ContextError()},
		{"generate V2 result", generateV2ResultError()},
		{"generate V2 result context", generateV2ResultContextError()},
		{"cancel", (&Model{}).Cancel()},
	}
	for _, check := range checks {
		if !errors.Is(check.err, ErrNativeUnavailable) {
			t.Errorf("%s error = %v, want ErrNativeUnavailable", check.name, check.err)
		}
	}
	if err := (&Model{}).Close(); err != nil {
		t.Errorf("close model: %v", err)
	}
	if err := (&Voice{}).Close(); err != nil {
		t.Errorf("close voice: %v", err)
	}
	if err := (&Emotion{}).Close(); err != nil {
		t.Errorf("close emotion: %v", err)
	}
}

func capabilityError() error {
	_, err := GetCapabilities()
	return err
}
func loadError() error {
	_, err := Load("model")
	return err
}
func modelInfoError() error {
	_, err := (&Model{}).Info()
	return err
}
func modelHealthError() error {
	_, err := (&Model{}).Health()
	return err
}
func prepareVoiceError() error {
	_, err := (&Model{}).PrepareVoice("voice.wav")
	return err
}
func prepareVoicePCMError() error {
	_, err := (&Model{}).PrepareVoicePCM([]float32{0}, 22050, 1)
	return err
}
func voiceInfoError() error {
	_, err := (&Voice{}).Info()
	return err
}
func prepareEmotionError() error {
	_, err := (&Model{}).PrepareEmotionReference("emotion.wav")
	return err
}
func generateError() error {
	_, err := (&Model{}).Generate(&Voice{}, "text", Options{})
	return err
}
func generateContextError() error {
	_, err := (&Model{}).GenerateContext(context.Background(), &Voice{}, "text", Options{})
	return err
}
func generateV2Error() error {
	_, err := (&Model{}).GenerateV2(&Voice{}, "text", OptionsV2{})
	return err
}
func generateV2ContextError() error {
	_, err := (&Model{}).GenerateV2Context(context.Background(), &Voice{}, "text", OptionsV2{})
	return err
}
func generateV2ResultError() error {
	_, err := (&Model{}).GenerateV2Result(&Voice{}, "text", OptionsV2{})
	return err
}
func generateV2ResultContextError() error {
	_, err := (&Model{}).GenerateV2ResultContext(context.Background(), &Voice{}, "text", OptionsV2{})
	return err
}
