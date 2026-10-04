//go:build windows && cgo && indextts_native

package indextts

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"
)

func TestVersion(t *testing.T) {
	if ABIVersion() != 0x00010005 {
		t.Fatalf("unexpected ABI version %#x", ABIVersion())
	}
	capabilities, err := GetCapabilities()
	if err != nil {
		t.Fatal(err)
	}
	if capabilities.ABIMajor != 1 || capabilities.ABIMinor != 5 || capabilities.SampleRate != 22050 || !capabilities.SupportsCPU || !capabilities.SupportsEmotionReference {
		t.Fatalf("unexpected capabilities: %+v", capabilities)
	}
	if Version() == "" {
		t.Fatal("native version is empty")
	}
}

func TestLoadRejectsUnknownDevice(t *testing.T) {
	_, err := LoadWithOptions(LoadOptions{ModelDir: "unused", Device: Device("invalid")})
	if err == nil {
		t.Fatal("expected unknown device error")
	}
}

func TestEndToEnd(t *testing.T) {
	modelDir := os.Getenv("INDEXTTS_TEST_MODEL")
	voicePath := os.Getenv("INDEXTTS_TEST_VOICE")
	if modelDir == "" || voicePath == "" {
		t.Skip("set INDEXTTS_TEST_MODEL and INDEXTTS_TEST_VOICE for native integration test")
	}
	device := Device(os.Getenv("INDEXTTS_TEST_DEVICE"))
	if device == "" {
		device = DeviceCPU
	}
	model, err := LoadWithOptions(LoadOptions{
		ModelDir:    modelDir,
		Device:      device,
		DeviceIndex: 0,
	})
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	defer model.Close()
	modelInfo, err := model.Info()
	if err != nil {
		t.Fatalf("model info: %v", err)
	}
	if modelInfo.RuntimeVersion == "" || modelInfo.Backend == "" || modelInfo.Device == "" {
		t.Fatalf("invalid model info: %+v", modelInfo)
	}

	voiceBytes, err := os.ReadFile(voicePath)
	if err != nil {
		t.Fatalf("read voice fixture: %v", err)
	}
	cachedVoicePath := t.TempDir() + `\voice.wav`
	if err := os.WriteFile(cachedVoicePath, voiceBytes, 0o600); err != nil {
		t.Fatalf("copy voice fixture: %v", err)
	}
	voice, err := model.PrepareVoice(cachedVoicePath)
	if err != nil {
		t.Fatalf("prepare voice: %v", err)
	}
	defer voice.Close()
	voiceInfo, err := voice.Info()
	if err != nil {
		t.Fatalf("voice info: %v", err)
	}
	if len(voiceInfo.ReferenceSHA256) != 64 || voiceInfo.DurationSeconds <= 0 || voiceInfo.SourceSampleRate == 0 || voiceInfo.CacheBytes == 0 {
		t.Fatalf("invalid voice info: %+v", voiceInfo)
	}
	health, err := model.Health()
	if err != nil {
		t.Fatalf("model health: %v", err)
	}
	if !health.Loaded || !health.DeviceHealthy || health.VoiceCacheEntries == 0 || health.VoiceCacheBytes == 0 {
		t.Fatalf("invalid model health: %+v", health)
	}

	if err := os.Remove(cachedVoicePath); err != nil {
		t.Fatalf("delete prepared voice source: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancelled := make(chan error, 1)
	go func() {
		_, generateErr := model.GenerateContext(
			ctx,
			voice,
			"这是一段用于验证底层协作取消的较长文本，调用方取消以后推理应当尽快停止，而不是继续生成完整音频。",
			Options{Language: "ZH", Seed: 1234, DurationFactor: 1},
		)
		cancelled <- generateErr
	}()
	time.Sleep(500 * time.Millisecond)
	cancelStarted := time.Now()
	cancel()
	select {
	case cancelErr := <-cancelled:
		t.Logf("cancelled generation in %s", time.Since(cancelStarted))
		if !errors.Is(cancelErr, context.Canceled) && !errors.Is(cancelErr, ErrCancelled) {
			t.Fatalf("expected cancellation error, got %v", cancelErr)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("cancellation did not stop generation within 30 seconds")
	}

	emotion, err := model.PrepareEmotionReference(voicePath)
	if err != nil {
		t.Fatalf("prepare emotion reference: %v", err)
	}
	defer emotion.Close()
	v2Audio, err := model.GenerateV2(voice, "你好世界，这是一次Go情感参考测试。", OptionsV2{
		Options: Options{Language: "ZH", Seed: 1234, DurationFactor: 1},
		Emotion: EmotionOptions{Mode: EmotionReference, Reference: emotion, Strength: 1},
	})
	if err != nil {
		t.Fatalf("generate v2: %v", err)
	}
	if len(v2Audio.Samples) == 0 || v2Audio.SampleRate != 22_050 {
		t.Fatalf("invalid V2 audio: rate=%d samples=%d", v2Audio.SampleRate, len(v2Audio.Samples))
	}
	vectorResult, err := model.GenerateV2Result(voice, "你好世界，这是一次Go情感向量测试。", OptionsV2{
		Options: Options{Language: "ZH", Seed: 1234, DurationFactor: 1},
		Emotion: EmotionOptions{Mode: EmotionVector, Vector: []float32{0.3, 0, 0, 0, 0, 0, 0, 0.2}, Strength: 1},
	})
	if err != nil {
		t.Fatalf("generate emotion vector: %v", err)
	}
	if len(vectorResult.Audio.Samples) == 0 || vectorResult.Audio.SampleRate != 22_050 {
		t.Fatalf("invalid vector audio: rate=%d samples=%d", vectorResult.Audio.SampleRate, len(vectorResult.Audio.Samples))
	}
	if vectorResult.Info.SemanticTokens == 0 || vectorResult.Info.Total <= 0 || vectorResult.Info.GPT <= 0 || vectorResult.Info.BigVGAN <= 0 || vectorResult.Info.Seed != 1234 {
		t.Fatalf("invalid generation diagnostics: %+v", vectorResult.Info)
	}
	if vectorResult.Info.Peak <= 0 || vectorResult.Info.RMS <= 0 || vectorResult.Info.GeneratedSeconds <= 0 {
		t.Fatalf("invalid audio diagnostics: %+v", vectorResult.Info)
	}

	longText, err := model.GenerateLongTextResult(
		voice,
		"第一段用于验证长文本。第二段用于验证分段元数据和音频拼接。",
		Options{Language: "ZH", Seed: 1234, DurationFactor: 1},
		LongTextOptions{MaxChars: 12, Pause: 100 * time.Millisecond},
	)
	if err != nil {
		t.Fatalf("generate long text: %v", err)
	}
	if len(longText.Audio.Samples) == 0 || longText.NormalizedText == "" || len(longText.Segments) < 2 {
		t.Fatalf("invalid long-text result: text=%q segments=%d samples=%d", longText.NormalizedText, len(longText.Segments), len(longText.Audio.Samples))
	}
	for index, segment := range longText.Segments {
		if segment.StartChar >= segment.EndChar || segment.AudioDurationSamples == 0 || segment.SemanticTokens == 0 || segment.Seed != 1234+uint64(index) {
			t.Fatalf("invalid long-text segment %d: %+v", index, segment)
		}
		if segment.AudioOffsetSamples+segment.AudioDurationSamples > len(longText.Audio.Samples) {
			t.Fatalf("long-text segment %d exceeds audio", index)
		}
	}

	audio, err := model.Generate(voice, "你好世界，这是一次Go端到端测试。", Options{
		Language:       "ZH",
		Seed:           1234,
		DurationFactor: 1,
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if audio.SampleRate != 22_050 || audio.Channels != 1 || len(audio.Samples) == 0 {
		t.Fatalf("invalid audio: rate=%d channels=%d samples=%d", audio.SampleRate, audio.Channels, len(audio.Samples))
	}
	peak := float32(0)
	for _, sample := range audio.Samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatal("audio contains a non-finite sample")
		}
		if value := float32(math.Abs(float64(sample))); value > peak {
			peak = value
		}
	}
	if peak == 0 {
		t.Fatal("audio is silent")
	}
	t.Logf("generated samples=%d seconds=%.3f peak=%.6f device=%s", len(audio.Samples), float64(len(audio.Samples))/float64(audio.SampleRate), peak, device)
}
