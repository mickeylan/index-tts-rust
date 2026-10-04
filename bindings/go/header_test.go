package indextts

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestPublicHeaderMatchesFFISource(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate Go binding source directory")
	}
	moduleHeader := filepath.Join(filepath.Dir(filename), "indextts.h")
	sourceHeader := filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "crates", "indextts-ffi", "indextts.h"))

	got, err := os.ReadFile(moduleHeader)
	if err != nil {
		t.Fatalf("read module header: %v", err)
	}
	want, err := os.ReadFile(sourceHeader)
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("FFI source header is not present in the independently published module")
	}
	if err != nil {
		t.Fatalf("read FFI source header: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("bindings/go/indextts.h drifted from crates/indextts-ffi/indextts.h")
	}
}
