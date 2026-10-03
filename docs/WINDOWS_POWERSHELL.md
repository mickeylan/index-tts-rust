# Windows PowerShell 使用手册

本文给出从源码构建、模型导出、CPU/GPU 合成、测试和 CUDA 发布包制作的完整 PowerShell 命令。以下路径按当前已验证环境编写，请按实际安装位置修改。

## 1. 已验证环境

```text
Windows 11 x64
NVIDIA GeForce RTX 4070
CUDA Toolkit 12.8
cuDNN 9
ONNX Runtime GPU 1.23.2
Visual Studio 2019 MSVC 14.29
Rust stable / MSVC target
Python D:\Python3（仅用于离线模型导出）
```

运行时不依赖 Python/PyTorch；Python 只用于模型转换。预打包 CUDA 版本只要求目标机器安装兼容的 NVIDIA 驱动和 Microsoft Visual C++ Redistributable。

## 2. 打开项目目录

```powershell
Set-Location E:\mickeylan\ai\index-tts-rust
```

PowerShell 中连续命令请使用 `;`，或者逐条执行。不要在 Windows PowerShell 5.1 中使用 Bash 的 `&&`。

## 3. 设置构建环境

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_ROOT = $env:CUDA_PATH
$env:NVCC_CCBIN = 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
$env:Path = @(
  "$env:CUDA_PATH\bin",
  (Split-Path -Parent $env:NVCC_CCBIN),
  $env:Path
) -join ';'
```

检查编译工具：

```powershell
nvcc --version
& $env:NVCC_CCBIN 2>&1 | Select-Object -First 2
rustc --version
cargo --version
nvidia-smi
```

`nvcc --version` 应显示 `release 12.8`。

## 4. 设置 GPU 运行时 DLL

当前已验证环境的 DLL 路径：

```powershell
$runtimePaths = @(
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
  'D:\Python3\Lib\site-packages\nvidia\cudnn\bin',
  'D:\Python3\Lib\site-packages\nvidia\cu13\bin\x86_64',
  'D:\Python3\Lib\site-packages\onnxruntime\capi'
)

$env:Path = ($runtimePaths -join ';') + ';' + $env:Path
$env:ORT_DYLIB_PATH = 'D:\Python3\Lib\site-packages\onnxruntime\capi\onnxruntime.dll'
```

说明：当前 Candle 0.8/cudarc 组合在此机器上还会探测 CUDA 13 cuBLAS，因此运行时路径包含 `nvidia\cu13\bin\x86_64`。使用 `tools\package_windows_cuda.ps1` 打包后，用户无需手工设置这些路径。

验证 ONNX Runtime CUDA 是否真正执行，不能只检查 provider 列表：

```powershell
python -c "import onnxruntime as ort,numpy as np; p=r'E:\models\indextts25-rust\onnx\semantic-codec\model.onnx'; s=ort.InferenceSession(p,providers=[('CUDAExecutionProvider',{'device_id':0})]); print(s.get_providers()); print(s.run(None,{'codes':np.array([[1,2,3]],dtype=np.int64)})[0].shape)"
```

预期包含：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
(1, 6, 1024)
```

若只显示 `CPUExecutionProvider`，说明 CUDA/cuDNN DLL 未完整加载。

## 5. 导出模型包

模型转换需要官方源码和 checkpoint：

```powershell
python tools\export_all.py `
  --source E:\mickeylan\ai\index-tts `
  --checkpoints E:\mickeylan\ai\index-tts\checkpoints `
  --output E:\models\indextts25-rust `
  --buckets 256 512

if ($LASTEXITCODE -ne 0) { throw "模型导出失败，退出码 $LASTEXITCODE" }
```

建议先导出 `256 512`。当前环境中 BigVGAN 1024 导出曾发生 Windows 原生访问异常；短、中等句子使用 256/512 bucket 已可正常合成。

检查模型包：

```powershell
$modelDir = 'E:\models\indextts25-rust'

@(
  "$modelDir\gpt.safetensors",
  "$modelDir\wav2vec2bert_stats.safetensors",
  "$modelDir\multilingual_zh_ja_yue_char_del.tiktoken",
  "$modelDir\pinyin.vocab",
  "$modelDir\onnx\s2mel\model-256.onnx",
  "$modelDir\onnx\bigvgan\model-256.onnx",
  "$modelDir\manifest.json"
) | ForEach-Object {
  [pscustomobject]@{ Path = $_; Exists = Test-Path -LiteralPath $_ }
}
```

所有 `Exists` 应为 `True`。

## 6. 构建 CPU 版本

```powershell
cargo build --release -p indextts-cli
if ($LASTEXITCODE -ne 0) { throw "CPU CLI 构建失败" }
```

## 7. 构建 CUDA 版本

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_ROOT = $env:CUDA_PATH
$env:NVCC_CCBIN = 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
$env:Path = "$env:CUDA_PATH\bin;$(Split-Path -Parent $env:NVCC_CCBIN);$env:Path"

cargo build --release -p indextts-cli --features cuda
if ($LASTEXITCODE -ne 0) { throw "CUDA CLI 构建失败" }
```

检查：

```powershell
Get-Item target\release\indextts.exe
```

## 8. 生成 semantic tokens

### GPU

```powershell
$modelDir = 'E:\models\indextts25-rust'
$voice = 'K:\ComfyUI\models\TTS\IndexTTS-2.5\voices\official-demo\voice_03.wav'
$outputDir = 'F:\test'
New-Item -ItemType Directory -Force $outputDir | Out-Null

target\release\indextts.exe `
  --device cuda `
  --device-index 0 `
  tokens `
  --model $modelDir `
  --voice $voice `
  --text '你好世界，这是一次GPU语音合成测试。' `
  --language zh `
  --output "$outputDir\tokens-gpu.json"

if ($LASTEXITCODE -ne 0) { throw "GPU semantic token 生成失败" }
Get-Content "$outputDir\tokens-gpu.json"
```

### CPU

CPU 运行前应使用未启用 CUDA feature 的 CPU 构建，或直接显式指定：

```powershell
target\release\indextts.exe `
  --device cpu `
  tokens `
  --model $modelDir `
  --voice $voice `
  --text '你好世界。' `
  --language zh `
  --output "$outputDir\tokens-cpu.json"
```

## 9. 完整 GPU 语音合成

```powershell
$stopwatch = [Diagnostics.Stopwatch]::StartNew()

target\release\indextts.exe `
  --device cuda `
  --device-index 0 `
  synth `
  --model $modelDir `
  --voice $voice `
  --text '你好世界，这是一次GPU语音合成测试。' `
  --language zh `
  --seed 1234 `
  --duration-factor 1.0 `
  --output "$outputDir\gpu-result.wav"

$exitCode = $LASTEXITCODE
$stopwatch.Stop()
"exit=$exitCode elapsed_seconds=$($stopwatch.Elapsed.TotalSeconds)"
if ($exitCode -ne 0) { throw "GPU 语音合成失败" }
```

播放：

```powershell
Start-Process "$outputDir\gpu-result.wav"
```

查看 GPU 使用情况可另开一个 PowerShell：

```powershell
nvidia-smi --loop=1
```

## 10. 检查 WAV

```powershell
python -c "import wave,array,math; p=r'F:\test\gpu-result.wav'; w=wave.open(p,'rb'); a=array.array('h',w.readframes(w.getnframes())); print({'channels':w.getnchannels(),'rate':w.getframerate(),'frames':len(a),'seconds':len(a)/w.getframerate(),'peak':max(map(abs,a))/32768,'finite':all(math.isfinite(x) for x in a)})"
```

预期：

- `channels = 1`
- `rate = 22050`
- `frames > 0`
- `finite = True`
- `peak > 0`

## 11. CPU/GPU 确定性比较

相同模型、参考音频、文本和 seed 分别生成：

```powershell
target\release\indextts.exe --device cpu synth `
  --model $modelDir --voice $voice --text '你好世界。' --language zh `
  --seed 1234 --output "$outputDir\cpu.wav"

target\release\indextts.exe --device cuda --device-index 0 synth `
  --model $modelDir --voice $voice --text '你好世界。' --language zh `
  --seed 1234 --output "$outputDir\gpu.wav"
```

浮点后端可能产生微小数值差异，因此最终 WAV 哈希不要求相等。应比较 semantic tokens、输出时长、有限值、听感和波形统计。

## 12. 运行完整回归测试

```powershell
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) { throw "rustfmt 失败" }

cargo clippy --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { throw "clippy 失败" }

cargo test --workspace
if ($LASTEXITCODE -ne 0) { throw "Rust 测试失败" }

python -m unittest discover -s tools -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw "Python 离线工具测试失败" }
```

CUDA 编译检查：

```powershell
cargo build --release -p indextts-cli --features cuda
if ($LASTEXITCODE -ne 0) { throw "CUDA release 构建失败" }
```

## 13. 构建 C ABI

CPU C ABI：

```powershell
cargo build --release -p indextts-ffi
```

产物：

```powershell
Get-Item target\release\indextts.dll
Get-Item target\release\indextts.dll.lib
Get-Item target\release\indextts.lib
```

## 14. 编译 Go binding

```powershell
$env:CGO_LDFLAGS = "-L$PWD/target/release -lindextts"
go -C bindings/go test ./...
if ($LASTEXITCODE -ne 0) { throw "Go binding 编译失败" }
```

## 15. 制作自包含 CUDA 包

自动构建并收集所有已验证 DLL：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -CudaRoot 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  -PythonRoot 'D:\Python3' `
  -MsvcCl 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe' `
  -Output 'dist\index-tts-rust-win64-cuda' `
  -Zip

if ($LASTEXITCODE -ne 0) { throw "CUDA 发布包制作失败" }
```

若已经构建好 CUDA CLI，只重新收集文件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -SkipBuild `
  -Output 'dist\index-tts-rust-win64-cuda' `
  -Zip
```

生成：

```text
dist\index-tts-rust-win64-cuda\
dist\index-tts-rust-win64-cuda.zip
```

包内启动器会自动指定同目录的 `onnxruntime.dll`：

```powershell
.\dist\index-tts-rust-win64-cuda\indextts-cuda.cmd `
  --device cuda `
  --device-index 0 `
  synth `
  --model E:\models\indextts25-rust `
  --voice $voice `
  --text '这是打包后的GPU测试。' `
  --language zh `
  --output F:\test\packaged-gpu.wav
```

发布包不包含模型。模型目录通过 `--model` 单独指定。

## 16. 常见错误

### `nvcc` 找不到

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
```

### `cl.exe` 找不到

```powershell
$env:NVCC_CCBIN = 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
$env:Path = "$(Split-Path -Parent $env:NVCC_CCBIN);$env:Path"
```

### `CUDAExecutionProvider` 回退 CPU

确认以下目录都在 `PATH`：

```text
CUDA 12.8 bin
cuDNN 9 bin
ONNX Runtime capi
当前环境的 CUDA 13 cuBLAS bin
```

然后执行第 4 节的实际 ONNX inference 测试。

### 找不到 `pinyin.vocab`

重新运行最新的 `tools\export_all.py`，或确认：

```powershell
Test-Path E:\models\indextts25-rust\pinyin.vocab
```

### `Unsupported bit depth: 24`

更新到包含 24-bit PCM WAV 支持的最新版本，并重新构建 CLI。

### 没有足够大的 bucket

缩短输入文本，或导出更大的同尺寸 DiT 和 BigVGAN bucket。二者必须同时存在。
