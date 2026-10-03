# IndexTTS Windows PowerShell 命令详解

本文不是简单的“复制粘贴清单”，而是逐项解释本项目在 Windows PowerShell 中会用到的命令、参数、变量、退出码和常见组合。完整操作流程见 [Windows PowerShell 使用手册](WINDOWS_POWERSHELL.md)。

## 目录

1. [PowerShell 基础语法](#1-powershell-基础语法)
2. [路径和文件命令](#2-路径和文件命令)
3. [环境变量命令](#3-环境变量命令)
4. [环境检查命令](#4-环境检查命令)
5. [Cargo 构建命令](#5-cargo-构建命令)
6. [CLI 全局参数](#6-cli-全局参数)
7. [`synth` 合成命令](#7-synth-合成命令)
8. [`tokens` 语义码命令](#8-tokens-语义码命令)
9. [模型导出命令](#9-模型导出命令)
10. [测试和静态检查命令](#10-测试和静态检查命令)
11. [CUDA 发布包命令](#11-cuda-发布包命令)
12. [C ABI 和 Go 命令](#12-c-abi-和-go-命令)
13. [诊断、计时和监控命令](#13-诊断计时和监控命令)
14. [常用完整命令块](#14-常用完整命令块)

## 1. PowerShell 基础语法

### 1.1 切换目录

```powershell
Set-Location E:\mickeylan\ai\index-tts-rust
```

短写：

```powershell
cd E:\mickeylan\ai\index-tts-rust
```

- `Set-Location` 是 PowerShell cmdlet。
- 路径包含空格时必须加引号。
- 本项目后续相对路径都以仓库根目录为起点。

确认当前目录：

```powershell
Get-Location
```

### 1.2 字符串引号

单引号不展开变量：

```powershell
$text = '你好，$name'
```

值仍然是 `你好，$name`。

双引号会展开变量：

```powershell
$name = '小明'
$text = "你好，$name"
```

值是 `你好，小明`。

路径通常推荐单引号，含变量的路径使用双引号：

```powershell
$modelDir = 'E:\models\indextts25-rust'
$gpt = "$modelDir\gpt.safetensors"
```

### 1.3 多行命令

PowerShell 使用反引号 `` ` `` 续行：

```powershell
target\release\indextts.exe `
  --device cuda `
  version
```

注意：反引号必须是该行最后一个字符，后面不能有空格或注释。

也可以写成一行：

```powershell
target\release\indextts.exe --device cuda version
```

### 1.4 连续执行和错误控制

Windows PowerShell 5.1 不支持 Bash 风格的 `&&`。使用分号：

```powershell
cargo build; cargo test
```

但分号不会因前一个命令失败而停止。严谨写法：

```powershell
cargo build
if ($LASTEXITCODE -ne 0) { throw "cargo build 失败：$LASTEXITCODE" }

cargo test
if ($LASTEXITCODE -ne 0) { throw "cargo test 失败：$LASTEXITCODE" }
```

- `$LASTEXITCODE`：最近一个原生程序的退出码。
- `0`：通常代表成功。
- 非 `0`：失败。
- `throw`：立即终止当前脚本并显示错误。

对于 PowerShell cmdlet，可设置：

```powershell
$ErrorActionPreference = 'Stop'
```

它会让多数 PowerShell 非终止错误变成终止错误，但不会替代对原生程序 `$LASTEXITCODE` 的检查。

### 1.5 管道

将左侧结果传给右侧：

```powershell
Get-ChildItem target\release | Select-Object Name, Length
```

过滤：

```powershell
Get-ChildItem target\release | Where-Object Name -Like '*.dll'
```

排序：

```powershell
Get-ChildItem target\release | Sort-Object Length -Descending
```

## 2. 路径和文件命令

### 2.1 检查文件或目录

```powershell
Test-Path -LiteralPath 'E:\models\indextts25-rust\gpt.safetensors'
```

- `True`：存在。
- `False`：不存在。
- `-LiteralPath` 不把 `[`、`]`、`*` 等字符解释成通配符。

### 2.2 显示文件信息

```powershell
Get-Item -LiteralPath 'target\release\indextts.exe'
```

只显示关键信息：

```powershell
Get-Item target\release\indextts.exe | Select-Object FullName, Length, LastWriteTime
```

### 2.3 列出目录

```powershell
Get-ChildItem 'E:\models\indextts25-rust'
```

递归列出 ONNX 文件：

```powershell
Get-ChildItem 'E:\models\indextts25-rust\onnx' -Recurse -Filter '*.onnx'
```

### 2.4 创建目录

```powershell
New-Item -ItemType Directory -Force 'F:\test' | Out-Null
```

- `-Force`：目录已经存在时不报错。
- `Out-Null`：隐藏创建结果输出。

### 2.5 复制文件

```powershell
Copy-Item -LiteralPath 'source.dll' -Destination 'dist\runtime\source.dll' -Force
```

### 2.6 删除生成目录

```powershell
Remove-Item -LiteralPath 'dist\index-tts-rust-win64-cuda' -Recurse -Force
```

只应删除确认可重建的生成文件，不要对模型和源码目录使用未经检查的递归删除命令。

### 2.7 文件哈希

```powershell
Get-FileHash 'F:\test\result.wav' -Algorithm SHA256
```

比较两个文件：

```powershell
$a = (Get-FileHash 'F:\test\a.wav' -Algorithm SHA256).Hash
$b = (Get-FileHash 'F:\test\b.wav' -Algorithm SHA256).Hash
$a -eq $b
```

## 3. 环境变量命令

### 3.1 设置当前终端变量

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
```

只对当前 PowerShell 进程和它启动的子进程有效。关闭窗口后失效。

### 3.2 CUDA 构建变量

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_ROOT = $env:CUDA_PATH
$env:NVCC_CCBIN = 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
```

- `CUDA_PATH`：CUDA Toolkit 根目录。
- `CUDA_ROOT`：部分 Rust/CUDA 构建脚本读取的别名。
- `NVCC_CCBIN`：指定 `nvcc` 使用的 MSVC host compiler。

### 3.3 将目录加入 PATH

```powershell
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
```

多个目录：

```powershell
$runtimePaths = @(
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
  'D:\Python3\Lib\site-packages\nvidia\cudnn\bin',
  'D:\Python3\Lib\site-packages\nvidia\cu13\bin\x86_64',
  'D:\Python3\Lib\site-packages\onnxruntime\capi'
)
$env:Path = ($runtimePaths -join ';') + ';' + $env:Path
```

### 3.4 指定 ONNX Runtime DLL

```powershell
$env:ORT_DYLIB_PATH = 'D:\Python3\Lib\site-packages\onnxruntime\capi\onnxruntime.dll'
```

本项目使用 ORT 动态加载模式。设置后，Rust 运行时直接加载指定文件。

### 3.5 查看和撤销变量

```powershell
$env:CUDA_PATH
$env:ORT_DYLIB_PATH
```

删除当前进程变量：

```powershell
Remove-Item Env:ORT_DYLIB_PATH
```

## 4. 环境检查命令

### 4.1 检查 NVIDIA 驱动和 GPU

```powershell
nvidia-smi
```

重点看：

- GPU 型号；
- 驱动版本；
- 显存总量和已用量；
- 进程列表。

每秒刷新：

```powershell
nvidia-smi --loop=1
```

只查询常用字段：

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
```

### 4.2 检查 CUDA 编译器

```powershell
nvcc --version
```

本项目当前应使用 CUDA 12.8。若输出 13.x，说明 `PATH` 中 CUDA 13 位于 12.8 前面。

查找实际执行文件：

```powershell
where.exe nvcc
```

### 4.3 检查 MSVC

```powershell
Test-Path -LiteralPath $env:NVCC_CCBIN
& $env:NVCC_CCBIN 2>&1 | Select-Object -First 2
```

`cl.exe` 不带参数运行会返回非零退出码，但只要能显示编译器版本，就证明路径有效。

### 4.4 检查 Rust

```powershell
rustc --version
cargo --version
rustup show
```

### 4.5 检查 Python

```powershell
python --version
python -m pip --version
```

必须使用 `python -m pip`，避免 `pip.exe` 指向另一个 Python。

### 4.6 检查 ONNX Runtime provider 列表

```powershell
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

仅出现 `CUDAExecutionProvider` 名称还不代表 DLL 能真正加载。必须执行模型：

```powershell
python -c "import onnxruntime as ort,numpy as np; p=r'E:\models\indextts25-rust\onnx\semantic-codec\model.onnx'; s=ort.InferenceSession(p,providers=[('CUDAExecutionProvider',{'device_id':0})]); print(s.get_providers()); print(s.run(None,{'codes':np.array([[1,2,3]],dtype=np.int64)})[0].shape)"
```

结果第一项应为 `CUDAExecutionProvider`，且推理输出应为 `(1, 6, 1024)`。

## 5. Cargo 构建命令

### 5.1 CPU CLI

```powershell
cargo build --release -p indextts-cli
```

- `build`：编译。
- `--release`：使用优化配置。
- `-p indextts-cli`：只构建 CLI package。

产物：

```text
target\release\indextts.exe
```

### 5.2 CUDA CLI

```powershell
cargo build --release -p indextts-cli --features cuda
```

- `--features cuda` 向下启用 pipeline、GPT、Candle 和 ORT 的 CUDA feature。
- 必须先设置 CUDA 12.8、`NVCC_CCBIN` 和 PATH。

### 5.3 C ABI

```powershell
cargo build --release -p indextts-ffi
```

产物：

```text
target\release\indextts.dll
target\release\indextts.dll.lib
target\release\indextts.lib
```

### 5.4 编译整个 workspace

```powershell
cargo build --workspace
```

包含所有 crate，但不会替代 CUDA feature 构建验证。

### 5.5 清理

清理整个 target：

```powershell
cargo clean
```

清理某个 package 的构建缓存：

```powershell
cargo clean -p indextts-cli
```

这会增加下次构建时间，通常只在 feature/工具链缓存异常时使用。

## 6. CLI 全局参数

查看帮助：

```powershell
target\release\indextts.exe --help
```

基本语法：

```text
indextts.exe [全局参数] <子命令> [子命令参数]
```

### `--device`

```powershell
--device auto
--device cpu
--device cuda
```

- `auto`：CUDA feature 构建优先使用 CUDA；CPU 构建使用 CPU。
- `cpu`：强制 CPU。
- `cuda`：强制 CUDA；没有 CUDA feature 或运行库时应报错。

### `--device-index`

```powershell
--device-index 0
```

GPU 序号，从 0 开始。可用 GPU：

```powershell
nvidia-smi -L
```

### `--verbose`

```powershell
--verbose
```

布尔开关，不需要写 `true`。当前日志实现较简化。

### `--log-dir`

```powershell
--log-dir F:\test\logs
```

当前参数保留，但日志文件输出尚未完整实现；不要依赖它进行生产日志归档。

### `version`

```powershell
target\release\indextts.exe version
```

也可查看 Cargo/Clap 版本：

```powershell
target\release\indextts.exe --version
```

## 7. `synth` 合成命令

帮助：

```powershell
target\release\indextts.exe synth --help
```

完整语法：

```powershell
target\release\indextts.exe `
  --device cuda `
  --device-index 0 `
  synth `
  --model E:\models\indextts25-rust `
  --voice F:\voices\speaker.wav `
  --text '你好世界。' `
  --language zh `
  --seed 1234 `
  --duration-factor 1.0 `
  --output F:\test\result.wav
```

### `--model` / `-m`

```powershell
--model E:\models\indextts25-rust
```

完整运行时模型包目录，不是官方原始 checkpoints 目录。必须包含 GPT、tokenizer、stats 和 ONNX 文件。

### `--voice` / `-v`

```powershell
--voice F:\voices\speaker.wav
```

参考音频。当前公开合同为 WAV；已支持常见 16/24/32-bit PCM。建议：

- 单人；
- 3–15 秒；
- 人声清晰；
- 无背景音乐；
- 不削波；
- 非静音。

注意：`-v` 同时被全局 verbose 和 synth voice 使用，容易产生歧义，建议始终写完整的 `--voice` 和 `--verbose`。

### `--text` / `-t`

```powershell
--text '你好世界，这是一次语音合成测试。'
```

PowerShell 中中文建议使用单引号。过长文本可能超过已导出的固定 bucket。

### `--language` / `-l`

支持：

```text
zh en ja es ar
```

当前主要验证中文 `zh`。其他语言“可选”不等于已完成质量验证。

### `--output` / `-o`

```powershell
--output F:\test\result.wav
```

省略时根据文本生成当前目录下的文件名。生产调用建议总是明确指定。

### `--seed`

```powershell
--seed 1234
```

控制 CFM 初始噪声。相同模型、输入、设备和 seed 应具有可重复性；CPU/GPU 浮点实现可能造成最终 WAV 微差。

### `--duration-factor`

```powershell
--duration-factor 1.0
```

允许范围：`0.5` 到 `2.0`。

- 小于 `1.0`：更短；
- 大于 `1.0`：更长；
- 它改变声学目标帧数，可能导致选择更大的 bucket。

### 采样和 beam 参数

CLI 暴露：

```text
--do-sample
--num-beams
--temperature
--top-k
--top-p
```

当前端到端运行时只支持：

```text
do_sample = false
num_beams = 1
```

因此当前不要传 `--do-sample`，也不要将 `--num-beams` 设置为其他值。其他采样参数目前不会形成已验证的采样链路。

## 8. `tokens` 语义码命令

```powershell
target\release\indextts.exe `
  --device cuda `
  --device-index 0 `
  tokens `
  --model E:\models\indextts25-rust `
  --voice F:\voices\speaker.wav `
  --text '你好世界。' `
  --language zh `
  --output F:\test\tokens.json
```

该命令执行：

```text
参考音频编码 → tokenizer → Candle GPT → semantic codes
```

不会运行 DiT 和 BigVGAN，适合排查前半链路。

省略 `--output` 时直接打印 JSON：

```powershell
target\release\indextts.exe --device cuda tokens `
  --model $modelDir --voice $voice --text '你好'
```

## 9. 模型导出命令

这些命令属于离线工具，依赖 Python/PyTorch。部署运行时不调用它们。

### 9.1 一键导出全部模型

```powershell
python tools\export_all.py `
  --source E:\mickeylan\ai\index-tts `
  --checkpoints E:\mickeylan\ai\index-tts\checkpoints `
  --output E:\models\indextts25-rust `
  --buckets 256 512
```

参数：

- `--source`：官方 IndexTTS 源码根目录。
- `--checkpoints`：官方 checkpoint 目录。
- `--output`：Rust 运行时模型包输出目录。
- `--buckets`：固定帧 bucket 列表；每个尺寸同时导出 DiT 和 BigVGAN。

当前建议从 `256 512` 开始。BigVGAN 1024 在当前环境曾触发导出器原生崩溃。

### 9.2 仅验证 GPT checkpoint

```powershell
python tools\export_weights.py `
  --input E:\mickeylan\ai\index-tts\checkpoints\gpt.pth `
  --validate-only
```

只检查权重键、shape 和参数合同，不输出文件。

### 9.3 导出 GPT safetensors

```powershell
python tools\export_weights.py `
  --input E:\mickeylan\ai\index-tts\checkpoints\gpt.pth `
  --output E:\models\indextts25-rust\gpt.safetensors `
  --manifest E:\models\indextts25-rust\gpt.manifest.json
```

### 9.4 导出 Wav2Vec 统计量

```powershell
python tools\export_wav2vec_stats.py `
  --input E:\mickeylan\ai\index-tts\checkpoints\wav2vec2bert_stats.pt `
  --output E:\models\indextts25-rust\wav2vec2bert_stats.safetensors
```

### 9.5 单独导出普通 ONNX 模块

组件名：

```text
campplus
wav2vec2bert
gpt-conditioning
semantic-codec
length-regulator
```

示例：

```powershell
python tools\export_indextts25_onnx.py semantic-codec `
  --source E:\mickeylan\ai\index-tts `
  --model-dir E:\mickeylan\ai\index-tts\checkpoints `
  --output E:\models\indextts25-rust\onnx\semantic-codec\model.onnx
```

### 9.6 单独导出 DiT bucket

```powershell
python tools\export_indextts25_onnx.py dit `
  --frames 512 `
  --source E:\mickeylan\ai\index-tts `
  --model-dir E:\mickeylan\ai\index-tts\checkpoints `
  --output E:\models\indextts25-rust\onnx\s2mel\model-512.onnx
```

### 9.7 单独导出 BigVGAN bucket

```powershell
python tools\export_indextts25_onnx.py bigvgan `
  --frames 512 `
  --source E:\mickeylan\ai\index-tts `
  --model-dir E:\mickeylan\ai\index-tts\checkpoints `
  --output E:\models\indextts25-rust\onnx\bigvgan\model-512.onnx
```

同一 bucket 尺寸的 DiT 和 BigVGAN 应成对存在。

### 9.8 捕获 GPT parity fixture

```powershell
python tools\capture_gpt_baseline.py `
  --source E:\mickeylan\ai\index-tts `
  --model-dir E:\mickeylan\ai\index-tts\checkpoints `
  --output target\model-export\gpt-baseline.safetensors `
  --steps 64
```

用于开发验证，不是普通用户合成步骤。

## 10. 测试和静态检查命令

### 10.1 格式检查

```powershell
cargo fmt --all -- --check
```

- `--all`：整个 workspace。
- 第二个 `--`：后面的参数传给 rustfmt。
- `--check`：只检查，不改文件。

自动格式化：

```powershell
cargo fmt --all
```

### 10.2 Clippy

```powershell
cargo clippy --workspace --all-targets -- -D warnings
```

- `--workspace`：所有 crate。
- `--all-targets`：lib、bin、example、test 等。
- `-D warnings`：任何警告都视为失败。

### 10.3 Rust 测试

```powershell
cargo test --workspace
```

单 crate：

```powershell
cargo test -p indextts-gpt
```

单模块并显示输出：

```powershell
cargo test -p indextts-gpt attention::tests -- --nocapture
```

### 10.4 Python 离线工具测试

```powershell
python -m unittest discover -s tools -p 'test_*.py' -v
```

- `discover`：自动发现测试。
- `-s tools`：测试目录。
- `-p 'test_*.py'`：文件模式。
- `-v`：详细输出。

### 10.5 文档测试

```powershell
cargo test --doc --workspace
```

### 10.6 Git diff 健康检查

```powershell
git diff --check
```

检查尾随空格、冲突标记等问题。

查看状态：

```powershell
git status --short --branch
```

## 11. CUDA 发布包命令

脚本：

```text
tools\package_windows_cuda.ps1
```

### 11.1 默认构建并打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1
```

### 11.2 构建并生成 ZIP

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -Zip
```

### 11.3 完整参数

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -Output 'dist\index-tts-rust-win64-cuda' `
  -CudaRoot 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  -PythonRoot 'D:\Python3' `
  -MsvcCl 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe' `
  -Zip
```

参数：

- `-Output`：发布目录；相对路径以仓库根目录解析。
- `-CudaRoot`：CUDA 12.8 Toolkit 根目录。
- `-PythonRoot`：当前包含 ORT、cuDNN、cu13 DLL 的 Python 根目录。
- `-MsvcCl`：MSVC `cl.exe` 完整路径。
- `-SkipBuild`：不运行 Cargo，只收集现有 release EXE/DLL。
- `-Zip`：额外创建 ZIP。

### 11.4 只重新打包

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -SkipBuild `
  -Zip
```

### 11.5 打包产物

```text
dist\index-tts-rust-win64-cuda\indextts.exe
dist\index-tts-rust-win64-cuda\indextts-cuda.cmd
dist\index-tts-rust-win64-cuda\*.dll
dist\index-tts-rust-win64-cuda\manifest.json
dist\index-tts-rust-win64-cuda.zip
```

`manifest.json` 包含每个文件的大小和 SHA-256。

### 11.6 使用打包启动器

```powershell
.\dist\index-tts-rust-win64-cuda\indextts-cuda.cmd `
  --device cuda `
  --device-index 0 `
  synth `
  --model E:\models\indextts25-rust `
  --voice F:\voices\speaker.wav `
  --text '打包运行测试。' `
  --language zh `
  --output F:\test\packaged.wav
```

启动器自动设置同目录 `onnxruntime.dll`，用户无需设置 `ORT_DYLIB_PATH`。

## 12. C ABI 和 Go 命令

### 12.1 C ABI release build

```powershell
cargo build --release -p indextts-ffi
```

Header：

```text
crates\indextts-ffi\indextts.h
```

### 12.2 Go binding 编译

```powershell
$env:CGO_LDFLAGS = "-L$PWD/target/release -lindextts"
go -C bindings/go test ./...
```

- `CGO_LDFLAGS`：告诉 Go linker 到哪里找 `indextts.lib`。
- `go -C bindings/go`：先切换到 Go module 目录再执行。
- `test ./...`：编译并测试 module 中所有 package。

### 12.3 Go 工具检查

```powershell
go version
go env CGO_ENABLED
```

Windows 上 `CGO_ENABLED` 应为 `1`。

## 13. 诊断、计时和监控命令

### 13.1 计时

```powershell
$timer = [Diagnostics.Stopwatch]::StartNew()

target\release\indextts.exe --device cuda synth `
  --model $modelDir --voice $voice --text '计时测试。' --output F:\test\timed.wav

$code = $LASTEXITCODE
$timer.Stop()
[pscustomobject]@{
  ExitCode = $code
  Seconds = $timer.Elapsed.TotalSeconds
}
```

### 13.2 查看输出 WAV

```powershell
Get-Item F:\test\result.wav | Select-Object FullName, Length, LastWriteTime
Start-Process F:\test\result.wav
```

### 13.3 WAV 结构和幅度检查

```powershell
python -c "import wave,array,math; p=r'F:\test\result.wav'; w=wave.open(p,'rb'); a=array.array('h',w.readframes(w.getnframes())); print({'channels':w.getnchannels(),'rate':w.getframerate(),'frames':len(a),'seconds':len(a)/w.getframerate(),'peak':max(map(abs,a))/32768,'finite':all(math.isfinite(x) for x in a)})"
```

### 13.4 检查模型包必要文件

```powershell
$modelDir = 'E:\models\indextts25-rust'
$required = @(
  'gpt.safetensors',
  'wav2vec2bert_stats.safetensors',
  'multilingual_zh_ja_yue_char_del.tiktoken',
  'pinyin.vocab',
  'onnx\wav2vec2bert\model.onnx',
  'onnx\campplus\model.onnx',
  'onnx\gpt-conditioning\model.onnx',
  'onnx\semantic-codec\model.onnx',
  'onnx\length-regulator\model.onnx',
  'onnx\s2mel\model-256.onnx',
  'onnx\bigvgan\model-256.onnx'
)
$required | ForEach-Object {
  $path = Join-Path $modelDir $_
  [pscustomobject]@{ File = $_; Exists = Test-Path -LiteralPath $path }
}
```

### 13.5 检查 DLL

```powershell
Get-ChildItem dist\index-tts-rust-win64-cuda -Filter '*.dll' |
  Sort-Object Name |
  Select-Object Name, Length
```

### 13.6 追踪退出码

一定要在目标命令后立即读取：

```powershell
target\release\indextts.exe version
$code = $LASTEXITCODE
"exit=$code"
```

不要在读取前运行另一个原生程序，否则 `$LASTEXITCODE` 会被覆盖。

## 14. 常用完整命令块

### 14.1 新终端初始化 GPU 环境

```powershell
Set-Location E:\mickeylan\ai\index-tts-rust

$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_ROOT = $env:CUDA_PATH
$env:NVCC_CCBIN = 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
$env:ORT_DYLIB_PATH = 'D:\Python3\Lib\site-packages\onnxruntime\capi\onnxruntime.dll'

$env:Path = @(
  "$env:CUDA_PATH\bin",
  (Split-Path -Parent $env:NVCC_CCBIN),
  'D:\Python3\Lib\site-packages\nvidia\cudnn\bin',
  'D:\Python3\Lib\site-packages\nvidia\cu13\bin\x86_64',
  'D:\Python3\Lib\site-packages\onnxruntime\capi',
  $env:Path
) -join ';'

nvcc --version
nvidia-smi
```

### 14.2 构建并合成

```powershell
cargo build --release -p indextts-cli --features cuda
if ($LASTEXITCODE -ne 0) { throw "CUDA 构建失败" }

$modelDir = 'E:\models\indextts25-rust'
$voice = 'K:\ComfyUI\models\TTS\IndexTTS-2.5\voices\official-demo\voice_03.wav'
$output = 'F:\test\result.wav'
New-Item -ItemType Directory -Force (Split-Path -Parent $output) | Out-Null

target\release\indextts.exe `
  --device cuda `
  --device-index 0 `
  synth `
  --model $modelDir `
  --voice $voice `
  --text '你好世界，这是一次完整测试。' `
  --language zh `
  --seed 1234 `
  --duration-factor 1.0 `
  --output $output

if ($LASTEXITCODE -ne 0) { throw "合成失败" }
Start-Process $output
```

### 14.3 一次性质量门禁

```powershell
cargo fmt --all -- --check
if ($LASTEXITCODE -ne 0) { throw 'rustfmt failed' }

cargo clippy --workspace --all-targets -- -D warnings
if ($LASTEXITCODE -ne 0) { throw 'clippy failed' }

cargo test --workspace
if ($LASTEXITCODE -ne 0) { throw 'cargo test failed' }

python -m unittest discover -s tools -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'python tools tests failed' }

cargo build --release -p indextts-cli --features cuda
if ($LASTEXITCODE -ne 0) { throw 'cuda release build failed' }
```

### 14.4 制作最终 ZIP

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools\package_windows_cuda.ps1 `
  -CudaRoot 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' `
  -PythonRoot 'D:\Python3' `
  -MsvcCl 'D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe' `
  -Output 'dist\index-tts-rust-win64-cuda' `
  -Zip

if ($LASTEXITCODE -ne 0) { throw 'package failed' }
Get-Item dist\index-tts-rust-win64-cuda.zip | Select-Object FullName, Length
```

## 15. 已知限制

- 当前端到端仅支持 greedy：不要启用 sampling 或 beam search。
- 当前主要验证中文。
- DiT/BigVGAN 使用固定 bucket；文本过长时需要更大且成对的 bucket。
- CUDA feature 构建目前针对 CUDA 12 ABI；已在 RTX 4070、CUDA 12.8 上验证。
- 模型文件不放进 CUDA runtime ZIP，需要单独通过 `--model` 指定。
- `--log-dir` 当前未形成完整的文件日志功能。
