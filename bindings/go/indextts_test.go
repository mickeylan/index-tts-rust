package indextts

import (
	"math"
	"os"
	"testing"
)

func TestVersion(t *testing.T) {
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

	voice, err := model.PrepareVoice(voicePath)
	if err != nil {
		t.Fatalf("prepare voice: %v", err)
	}
	defer voice.Close()

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
