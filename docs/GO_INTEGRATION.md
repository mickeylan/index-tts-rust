# Go 对接 IndexTTS Rust 详细指南

本文说明如何在 Windows 上通过 **cgo + IndexTTS C ABI**，从 Go 程序调用 `index-tts-rust`。

> 当前接口状态：CPU 和 CUDA 均已通过 Go → cgo → C ABI → Rust 的真实端到端测试。`Load` 默认使用 CPU；GPU 使用 `LoadWithOptions` 和 CUDA 版 `indextts.dll`。

## 1. 调用架构

```text
Go 业务程序
  ↓ Go package: bindings/go
cgo
  ↓ indextts.h
indextts.dll / indextts.lib
  ↓ stable C ABI
Rust IndexTtsPipeline
  ↓
Candle GPT + ONNX Runtime + Rust audio/text processing
```

Go 不直接调用 Rust ABI，也不直接持有 Rust 对象。Rust 模型和 voice 通过不透明 C handle 管理：

```c
indextts_model_t
indextts_voice_t
```

生成的音频先由 Rust 分配，Go wrapper 立即复制为 Go `[]float32`，然后调用 `indextts_audio_free` 释放原生内存。

## 2. 当前目录结构

```text
index-tts-rust/
├─ bindings/go/
│  ├─ go.mod
│  └─ indextts.go
├─ crates/indextts-ffi/
│  ├─ indextts.h
│  └─ src/lib.rs
├─ target/release/
│  ├─ indextts.dll
│  ├─ indextts.dll.lib
│  └─ indextts.lib
└─ docs/GO_INTEGRATION.md
```

Go module 名称：

```text
github.com/mickeylan/index-tts-rust/bindings/go
```

## 3. 前置要求

开发机需要：

- Go 1.22 或更新版本；
- `CGO_ENABLED=1`；
- Windows x64；
- Rust MSVC 工具链；
- Visual C++ linker/toolchain；
- 已导出的 IndexTTS Rust 模型包；
- 与构建匹配的 ONNX Runtime DLL；
- CPU 版本无需 CUDA；GPU 版本需要 CUDA 12.8、cuDNN 9、CUDA ORT DLL 和 `indextts-ffi/cuda` 构建。

检查：

```powershell
go version
go env GOOS GOARCH CGO_ENABLED CC CXX
rustc --version
cargo --version
```

预期至少包含：

```text
GOOS=windows
GOARCH=amd64
CGO_ENABLED=1
```

若 cgo 被关闭：

```powershell
$env:CGO_ENABLED = '1'
```

## 4. 构建 Rust C ABI

在仓库根目录执行：

```powershell
Set-Location E:\mickeylan\ai\index-tts-rust
cargo build --release -p indextts-ffi
if ($LASTEXITCODE -ne 0) { throw "indextts-ffi 构建失败：$LASTEXITCODE" }
```

确认产物：

```powershell
Get-Item target\release\indextts.dll
Get-Item target\release\indextts.dll.lib
Get-Item target\release\indextts.lib
```

含义：

| 文件 | 用途 |
|---|---|
| `indextts.dll` | Rust 动态库实现，程序运行时需要 |
| `indextts.dll.lib` | DLL 导入库 |
| `indextts.lib` | Rust staticlib；不建议直接由普通 cgo 项目手工链接全部依赖 |
| `indextts.h` | C API 声明 |

MSVC C/C++ 使用 `indextts.dll + indextts.dll.lib`；Windows cgo 默认由 MinGW GCC 链接，需使用 `indextts.dll + libindextts.dll.a`。

## 5. 编译仓库自带 Go wrapper

PowerShell：

```powershell
Set-Location E:\mickeylan\ai\index-tts-rust
$env:CGO_ENABLED = '1'
$env:CGO_LDFLAGS = "-L$PWD/target/release"

go -C bindings/go test -tags indextts_native ./...
if ($LASTEXITCODE -ne 0) { throw "Go wrapper 编译失败：$LASTEXITCODE" }
```

Go module 内含同步的 `bindings/go/indextts.h`，不再引用仓库外 Header。默认构建使用安全 stub，不要求 DLL；Windows 原生实现必须显式启用 `indextts_native` build tag。`CGO_LDFLAGS` 只需指定包含 `libindextts.dll.a` 的目录，`-lindextts` 已由 module 声明。

## 6. 在独立 Go 项目中引用

假设业务项目：

```text
E:\work\tts-service
```

创建：

```powershell
New-Item -ItemType Directory -Force E:\work\tts-service | Out-Null
Set-Location E:\work\tts-service
go mod init example.com/tts-service
```

开发阶段用 `replace` 指向本地 wrapper：

```powershell
go mod edit -require=github.com/mickeylan/index-tts-rust/bindings/go@v0.0.0
go mod edit -replace=github.com/mickeylan/index-tts-rust/bindings/go=E:\mickeylan\ai\index-tts-rust\bindings\go
go mod tidy
```

生成的 `go.mod` 类似：

```go
module example.com/tts-service

go 1.22

require github.com/mickeylan/index-tts-rust/bindings/go v0.0.0

replace github.com/mickeylan/index-tts-rust/bindings/go => E:\mickeylan\ai\index-tts-rust\bindings\go
```

> 若以后为 Go module 发布独立版本，可移除 `replace`，改用正式 tag。

## 7. 最小 Go 程序

`main.go`：

```go
package main

import (
    "fmt"
    "log"

    indextts "github.com/mickeylan/index-tts-rust/bindings/go"
)

func main() {
    fmt.Println("IndexTTS native version:", indextts.Version())

    model, err := indextts.Load(`E:\models\indextts25-rust`)
    if err != nil {
        log.Fatal("load model: ", err)
    }
    defer model.Close()

    voice, err := model.PrepareVoice(
        `K:\ComfyUI\models\TTS\IndexTTS-2.5\voices\official-demo\voice_03.wav`,
    )
    if err != nil {
        log.Fatal("prepare voice: ", err)
    }
    defer voice.Close()

    audio, err := model.Generate(
        voice,
        "你好世界，这是一次 Go 调用测试。",
        indextts.Options{
            Language:       "ZH",
            Seed:           1234,
            DurationFactor: 1.0,
        },
    )
    if err != nil {
        log.Fatal("generate: ", err)
    }

    fmt.Printf(
        "samples=%d sampleRate=%d channels=%d seconds=%.3f\n",
        len(audio.Samples),
        audio.SampleRate,
        audio.Channels,
        float64(len(audio.Samples))/float64(audio.SampleRate*audio.Channels),
    )
}
```

## 8. 编译独立 Go 程序

业务项目中设置 linker 搜索路径：

```powershell
$rustRepo = 'E:\mickeylan\ai\index-tts-rust'
$env:CGO_ENABLED = '1'
$env:CGO_LDFLAGS = "-L$rustRepo/target/release"

go build -tags indextts_native -o bin\tts-service.exe .
if ($LASTEXITCODE -ne 0) { throw "Go build 失败：$LASTEXITCODE" }
```

检查：

```powershell
Get-Item bin\tts-service.exe
```

## 9. 运行时 DLL 布局

Windows 加载 `tts-service.exe` 时必须能找到：

```text
indextts.dll
onnxruntime.dll
```

CPU 版还需 ORT 自身依赖和 MSVC runtime。最简单布局：

```text
release/
├─ tts-service.exe
├─ indextts.dll
├─ onnxruntime.dll
├─ onnxruntime_providers_shared.dll   # 若当前 ORT 分发包需要
└─ models/                            # 可放外部，不强制同目录
```

复制示例：

```powershell
$release = 'E:\work\tts-service\release'
$rustRepo = 'E:\mickeylan\ai\index-tts-rust'
$ortDir = 'D:\Python3\Lib\site-packages\onnxruntime\capi'

New-Item -ItemType Directory -Force $release | Out-Null
Copy-Item bin\tts-service.exe $release -Force
Copy-Item "$rustRepo\target\release\indextts.dll" $release -Force
Copy-Item "$ortDir\onnxruntime.dll" $release -Force
Copy-Item "$ortDir\onnxruntime_providers_shared.dll" $release -Force
```

也可在启动前设置：

```powershell
$env:Path = "$rustRepo\target\release;$ortDir;$env:Path"
$env:ORT_DYLIB_PATH = "$ortDir\onnxruntime.dll"
.\bin\tts-service.exe
```

正式发布推荐 DLL 与 EXE 放在同一目录，不要求用户手工设置全局 PATH。

## 10. 将 `Audio` 保存为 WAV

Go wrapper 返回：

```go
type Audio struct {
    Samples    []float32
    SampleRate uint32
    Channels   uint32
}
```

`Samples` 是归一化浮点 PCM，通常范围 `[-1, 1]`。下面使用 Go 标准库写 16-bit PCM WAV。

```go
package wav

import (
    "encoding/binary"
    "fmt"
    "io"
    "math"
    "os"
)

func WritePCM16(path string, samples []float32, sampleRate, channels uint32) error {
    if sampleRate == 0 || channels == 0 {
        return fmt.Errorf("invalid WAV format: rate=%d channels=%d", sampleRate, channels)
    }
    if uint64(len(samples)) > math.MaxUint32/2 {
        return fmt.Errorf("audio too large for RIFF/WAV")
    }

    f, err := os.Create(path)
    if err != nil {
        return err
    }
    defer f.Close()

    dataBytes := uint32(len(samples) * 2)
    byteRate := sampleRate * channels * 2
    blockAlign := uint16(channels * 2)

    write := func(value any) error {
        return binary.Write(f, binary.LittleEndian, value)
    }

    if _, err = io.WriteString(f, "RIFF"); err != nil { return err }
    if err = write(uint32(36) + dataBytes); err != nil { return err }
    if _, err = io.WriteString(f, "WAVEfmt "); err != nil { return err }
    if err = write(uint32(16)); err != nil { return err }
    if err = write(uint16(1)); err != nil { return err } // PCM
    if err = write(uint16(channels)); err != nil { return err }
    if err = write(sampleRate); err != nil { return err }
    if err = write(byteRate); err != nil { return err }
    if err = write(blockAlign); err != nil { return err }
    if err = write(uint16(16)); err != nil { return err }
    if _, err = io.WriteString(f, "data"); err != nil { return err }
    if err = write(dataBytes); err != nil { return err }

    for _, sample := range samples {
        if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
            return fmt.Errorf("non-finite audio sample")
        }
        if sample > 1 { sample = 1 }
        if sample < -1 { sample = -1 }
        pcm := int16(math.Round(float64(sample) * 32767.0))
        if err = write(pcm); err != nil { return err }
    }
    return f.Sync()
}
```

调用：

```go
if err := wav.WritePCM16(
    `F:\test\go-result.wav`,
    audio.Samples,
    audio.SampleRate,
    audio.Channels,
); err != nil {
    log.Fatal(err)
}
```

## 11. Go API 说明

### `Version()`

```go
func Version() string
```

返回 Rust C ABI crate 版本，例如 `0.1.0`。该字符串从 Rust 静态内存复制为 Go string。

### `Load(modelDir)`

```go
func Load(modelDir string) (*Model, error)
```

功能：

1. 将模型目录转为临时 C 字符串；
2. 初始化 `indextts_model_options_t`；
3. 调用 `indextts_model_load`；
4. 返回持有原生 handle 的 `Model`。

当前行为：

- `Load` 默认 CPU、Float32；
- `LoadWithOptions` 可选择 CPU 或 CUDA；
- 加载 GPT 权重与全部 ONNX session；
- 模型目录必须是已转换的 Rust runtime package；
- 不是官方原始 checkpoint 目录。

### `LoadWithOptions(options)`

```go
model, err := indextts.LoadWithOptions(indextts.LoadOptions{
    ModelDir:    `E:\models\indextts25-rust`,
    Device:      indextts.DeviceCUDA,
    DeviceIndex: 0,
})
```

设备常量：

```go
indextts.DeviceCPU
indextts.DeviceCUDA
```

CUDA 需要使用 `cargo build --release -p indextts-ffi --features cuda` 生成的 DLL。CPU DLL 收到 CUDA 设备请求时会明确返回错误，不会静默回退 CPU。

### `(*Model).PrepareVoice(path)`

```go
func (m *Model) PrepareVoice(path string) (*Voice, error)
```

当前实现会立即完成 Wav2Vec2-BERT、CAMPPlus、GPT conditioning、reference mel 和 prompt conditioning，并按原始内容 SHA-256 在 Model 内进行有界 LRU 缓存。后续生成不再读取原 WAV；源文件删除后，已准备的 Voice 仍可继续使用。

参考 WAV 建议 3–15 秒、单人、无音乐、非静音。

### `(*Model).Generate(...)`

```go
func (m *Model) Generate(
    voice *Voice,
    text string,
    options Options,
) (Audio, error)
```

`Options`：

```go
type Options struct {
    Language       string
    Seed           uint64
    DurationFactor float32
}
```

| 字段 | 含义 | 默认行为 |
|---|---|---|
| `Language` | `ZH`、`EN`、`JA`、`ES`、`AR` | 空字符串变为 `ZH` |
| `Seed` | CFM 随机种子 | `0` 是固定值，不是系统随机 |
| `DurationFactor` | 时长因子，范围 `0.5..2.0` | `0` 自动替换为 `1.0` |

当前生成固定为 greedy：

```text
do_sample=false
num_beams=1
```

返回的 `Audio.Samples` 已属于 Go，可在 `Generate` 返回后长期使用。

### 协作取消：`GenerateContext` 与 `Cancel`

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

audio, err := model.GenerateContext(ctx, voice, text, options)
if errors.Is(err, context.DeadlineExceeded) {
    // 底层推理已收到取消请求并在安全检查点退出。
}
```

也可从另一个 goroutine 主动请求取消：

```go
if err := model.Cancel(); err != nil {
    log.Printf("cancel failed: %v", err)
}
```

C ABI 对应函数：

```c
int32_t indextts_model_cancel(indextts_model_t model);
```

取消是协作式而不是强制终止线程：

- GPT 每个 decode token 前检查；
- pipeline 神经阶段之间检查；
- CFM 每个 Euler step 前检查；
- 已经进入的一次 ONNX Runtime/CUDA kernel 不会被中途破坏，而是在该调用返回后的下一个检查点退出；
- 取消返回 `INDEXTTS_CANCELLED (-3)`；
- 取消完成后同一个 Model 可以继续执行下一次生成；
- `Close` 会等待正在执行的生成退出，不能用来代替取消。

### `Close()`

```go
func (m *Model) Close() error
func (v *Voice) Close() error
```

- 可以重复调用；handle 置空后不再释放。
- 不应依赖 finalizer；应显式 `defer Close()`。
- 必须先停止所有生成调用，再关闭 Voice/Model。
- 推荐先关 Voice，再关 Model。

## 12. 正确生命周期

```go
model, err := indextts.Load(modelDir)
if err != nil { return err }
defer model.Close()

voice, err := model.PrepareVoice(voicePath)
if err != nil { return err }
defer voice.Close()

audio, err := model.Generate(voice, text, options)
if err != nil { return err }
```

不要：

- 在 `Generate` 运行时并发调用 `Close`；
- 将 Voice 用于另一个 Model；
- 复制 `Model` 或 `Voice` struct 值；
- 直接访问 cgo handle；
- 依赖垃圾回收器决定显存/内存释放时机。

## 13. 并发模型

当前 wrapper 内部：

- 每个 `Model` 有互斥锁；
- 同一 Model 的 `PrepareVoice` 和 `Generate` 串行；
- C ABI 错误读取由全局 `nativeMu` 保护；
- 单个 Model 上并发调用不会并行推理。

推荐服务端设计：

```text
一个 Model
  + 多个 Voice 配置/路径
  + 有界请求队列
  + 单个推理 worker
```

示例：

```go
type Request struct {
    Voice *indextts.Voice
    Text  string
    Opt   indextts.Options
    Reply chan Result
}

type Result struct {
    Audio indextts.Audio
    Err   error
}

func worker(model *indextts.Model, queue <-chan Request) {
    for request := range queue {
        audio, err := model.Generate(request.Voice, request.Text, request.Opt)
        request.Reply <- Result{Audio: audio, Err: err}
    }
}
```

不要通过创建很多 Model 来盲目提高并发：每个 Model 会重复加载大模型和占用大量内存。

## 14. HTTP 服务示例

以下示例只展示接口结构。生产服务还需要鉴权、限流、超时、请求大小限制和安全路径策略。

```go
package main

import (
    "encoding/json"
    "log"
    "net/http"

    indextts "github.com/mickeylan/index-tts-rust/bindings/go"
)

type Server struct {
    model *indextts.Model
    voice *indextts.Voice
}

type synthRequest struct {
    Text           string  `json:"text"`
    Language       string  `json:"language"`
    Seed           uint64  `json:"seed"`
    DurationFactor float32 `json:"duration_factor"`
}

func (s *Server) synth(w http.ResponseWriter, r *http.Request) {
    var request synthRequest
    decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
    if err := decoder.Decode(&request); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    if request.Text == "" {
        http.Error(w, "text is required", http.StatusBadRequest)
        return
    }

    audio, err := s.model.Generate(s.voice, request.Text, indextts.Options{
        Language:       request.Language,
        Seed:           request.Seed,
        DurationFactor: request.DurationFactor,
    })
    if err != nil {
        log.Printf("synthesis failed: %v", err)
        http.Error(w, "synthesis failed", http.StatusInternalServerError)
        return
    }

    // 生产代码可直接编码到 response，避免临时文件。
    w.Header().Set("Content-Type", "application/json")
    _ = json.NewEncoder(w).Encode(map[string]any{
        "sample_rate": audio.SampleRate,
        "channels":    audio.Channels,
        "samples":     len(audio.Samples),
    })
}
```

不要把底层错误、模型路径或系统路径原样返回给互联网客户端；记录到服务端日志即可。

## 15. 错误处理

Go wrapper 会将 `indextts_last_error()` 复制成 Go `error`。

典型错误：

| 错误 | 原因 |
|---|---|
| model file not found | 模型包不完整或路径错误 |
| ONNX Runtime DLL load failure | ORT DLL 或依赖未找到 |
| invalid reference audio | WAV 太短、静音、损坏或格式不支持 |
| no DiT/BigVGAN bucket can fit | 文本对应帧数超过最大 bucket |
| model or voice is closed | 生命周期错误 |
| greedy generation only | 传入未支持的 sampling/beam 配置 |

Go 中包装上下文：

```go
model, err := indextts.Load(modelDir)
if err != nil {
    return fmt.Errorf("load IndexTTS model %q: %w", modelDir, err)
}
```

## 16. DLL 找不到的排查

查看 EXE 所需 DLL：

```powershell
& 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\dumpbin.exe' `
  /dependents .\release\tts-service.exe
```

检查关键文件：

```powershell
@(
  '.\release\tts-service.exe',
  '.\release\indextts.dll',
  '.\release\onnxruntime.dll'
) | ForEach-Object {
  [pscustomobject]@{ Path=$_; Exists=Test-Path -LiteralPath $_ }
}
```

安装 Microsoft Visual C++ Redistributable：

```text
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

## 17. CPU 发布目录示例

```powershell
$release = 'E:\work\tts-service\release'
$rustRepo = 'E:\mickeylan\ai\index-tts-rust'
$ortDir = 'D:\Python3\Lib\site-packages\onnxruntime\capi'

New-Item -ItemType Directory -Force $release | Out-Null
Copy-Item '.\bin\tts-service.exe' $release -Force
Copy-Item "$rustRepo\target\release\indextts.dll" $release -Force
Copy-Item "$ortDir\onnxruntime.dll" $release -Force
Copy-Item "$ortDir\onnxruntime_providers_shared.dll" $release -Force
```

启动：

```powershell
Set-Location $release
$env:ORT_DYLIB_PATH = "$release\onnxruntime.dll"
.\tts-service.exe
```

## 18. GPU 对接

CUDA C ABI 构建：

```powershell
cargo build --release -p indextts-ffi --features cuda
```

Go 使用：

```go
model, err := indextts.LoadWithOptions(indextts.LoadOptions{
    ModelDir:    modelDir,
    Device:      indextts.DeviceCUDA,
    DeviceIndex: 0,
})
```

C ABI 设备约定：

```c
int32_t device_index; /* -1 = CPU; >= 0 = CUDA device */
```

CUDA feature 未启用时请求 GPU会明确失败。CUDA feature 启用后，Candle GPT 与全部 ONNX session 使用指定 GPU。发布目录必须包含 CUDA、cuDNN 和 ORT CUDA DLL；`tools/package_windows_cuda.ps1` 会同时打包 CUDA CLI、CUDA `indextts.dll`、MSVC import library、MinGW/cgo import library和 header。

已验证链路：

```text
Go test → cgo → indextts.dll → Candle CUDA + ORT CUDA → waveform
```

实测输出：52,736 samples、22,050 Hz、有限非静音音频。

## 19. 测试策略

### Wrapper 编译测试

MinGW cgo 需要 GNU import library。发布脚本会自动生成 `libindextts.dll.a`；手工生成命令：

```powershell
Push-Location target\release
& 'D:\mingw-w64\bin\gendef.exe' indextts.dll
& 'D:\mingw-w64\bin\dlltool.exe' -d indextts.def -D indextts.dll -l libindextts.dll.a -m i386:x86-64
Pop-Location

$env:CGO_LDFLAGS = "-L$PWD/target/release"
go -C bindings/go test -tags indextts_native ./...
```

### 已有测试与真实 CUDA 集成测试

默认测试不加载大模型；真实测试通过环境变量启用：

```powershell
$env:INDEXTTS_TEST_MODEL = 'E:\models\indextts25-rust'
$env:INDEXTTS_TEST_VOICE = 'K:\ComfyUI\models\TTS\IndexTTS-2.5\voices\official-demo\voice_03.wav'
$env:INDEXTTS_TEST_DEVICE = 'cuda'
$env:CGO_LDFLAGS = "-L$PWD/target/release -lindextts.dll"
go -C bindings/go test -tags indextts_native -v -count=1 ./...
```

已覆盖 `Version()`、错误设备和真实 CUDA 音频生成。后续仍建议补充：

- 错误模型路径返回错误；
- 错误模型路径返回错误；
- 错误 voice 路径返回错误；
- 真实模型加载；
- 真实参考音频准备；
- 生成非空、有限值音频；
- `SampleRate == 22050`；
- 重复 `Close()` 不崩溃；
- `Close()` 后 Generate 返回错误；
- 多 goroutine 请求在同一 Model 上安全串行；
- 长时间循环生成没有明显 native memory 泄漏。

### 运行 race detector

```powershell
go -C bindings/go test -tags indextts_native -race ./...
```

cgo 和 Windows race 支持受 Go 版本/工具链约束；即使 race detector 通过，也不能证明 C/Rust 内存完全安全。

## 20. 生产服务建议

- 服务启动时加载一次 Model，不要每个请求加载。
- Voice 尽量复用；当前版本仍会在生成时重新编码参考音频。
- 使用有界队列限制并发和内存峰值。
- 将请求 context 传给 `GenerateContext`；超时会触发底层协作取消，但当前 ONNX 调用会先执行到安全检查点。
- 优雅关闭顺序：停止接收请求 → 等待推理结束 → 关闭 Voice → 关闭 Model。
- 记录耗时、文本长度、semantic token 数、输出秒数和错误类别，不记录私密参考音频内容。
- 模型和 DLL 版本应与发布包固定，启动时记录 `indextts.Version()`。
- 不要允许外部用户任意传服务器本地 voice/model 路径。

## 21. 推荐业务封装

业务层不应到处直接持有 native Model。建议定义接口：

```go
type Synthesizer interface {
    Synthesize(ctx context.Context, text string, options SynthesisOptions) ([]byte, error)
    Close() error
}
```

由一个实现负责：

- native Model/Voice 生命周期；
- 请求串行化；
- WAV 编码；
- 指标；
- 超时状态；
- 错误归类；
- 服务退出。

实现应将 `context.Context` 传给 `GenerateContext`，由 C ABI 取消标志停止 GPT/CFM 后续迭代。

## 22. 快速检查清单

```text
[ ] Go 1.22+
[ ] CGO_ENABLED=1
[ ] indextts-ffi release build 成功
[ ] CGO_LDFLAGS 指向 target/release
[ ] Go wrapper 编译成功
[ ] EXE 同目录可找到 indextts.dll
[ ] ORT DLL 可加载
[ ] 模型包完整
[ ] reference WAV 可读取
[ ] 显式关闭 Voice 和 Model
[ ] 不并发 Close 与 Generate
[ ] 按需求选择 `Load`（CPU）或 `LoadWithOptions`（CUDA）
[ ] CUDA 发布包含 `libindextts.dll.a`
[ ] 真实 Go CUDA 集成测试通过
```
