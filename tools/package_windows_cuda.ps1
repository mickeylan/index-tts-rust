param(
    [string]$Output = "dist/index-tts-rust-win64-cuda",
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    [string]$PythonRoot = "D:\Python3",
    [string]$MsvcCl = "D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe",
    [string]$MingwBin = "D:\mingw-w64\bin",
    [switch]$SkipBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $repo $Output))
$cudaBin = Join-Path $CudaRoot "bin"
$sitePackages = Join-Path $PythonRoot "Lib\site-packages"
$ortBin = Join-Path $sitePackages "onnxruntime\capi"
$cudnnBin = Join-Path $sitePackages "nvidia\cudnn\bin"
$cu13Bin = Join-Path $sitePackages "nvidia\cu13\bin\x86_64"

$requiredFiles = @(
    @{ Source = (Join-Path $repo "target\release\indextts.exe"); Name = "indextts.exe" },
    @{ Source = (Join-Path $repo "target\release\indextts.dll"); Name = "indextts.dll" },
    @{ Source = (Join-Path $repo "target\release\indextts.dll.lib"); Name = "indextts.dll.lib" },
    @{ Source = (Join-Path $repo "target\release\libindextts.dll.a"); Name = "libindextts.dll.a" },
    @{ Source = (Join-Path $repo "crates\indextts-ffi\indextts.h"); Name = "indextts.h" },
    @{ Source = (Join-Path $ortBin "onnxruntime.dll"); Name = "onnxruntime.dll" },
    @{ Source = (Join-Path $ortBin "onnxruntime_providers_shared.dll"); Name = "onnxruntime_providers_shared.dll" },
    @{ Source = (Join-Path $ortBin "onnxruntime_providers_cuda.dll"); Name = "onnxruntime_providers_cuda.dll" },
    @{ Source = (Join-Path $cudaBin "cudart64_12.dll"); Name = "cudart64_12.dll" },
    @{ Source = (Join-Path $cudaBin "cublas64_12.dll"); Name = "cublas64_12.dll" },
    @{ Source = (Join-Path $cudaBin "cublasLt64_12.dll"); Name = "cublasLt64_12.dll" },
    @{ Source = (Join-Path $cudaBin "cufft64_11.dll"); Name = "cufft64_11.dll" },
    @{ Source = (Join-Path $cudaBin "curand64_10.dll"); Name = "curand64_10.dll" },
    @{ Source = (Join-Path $cudaBin "cusolver64_11.dll"); Name = "cusolver64_11.dll" },
    @{ Source = (Join-Path $cudaBin "cusparse64_12.dll"); Name = "cusparse64_12.dll" },
    @{ Source = (Join-Path $cudaBin "nvJitLink_120_0.dll"); Name = "nvJitLink_120_0.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn64_9.dll"); Name = "cudnn64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_adv64_9.dll"); Name = "cudnn_adv64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_cnn64_9.dll"); Name = "cudnn_cnn64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_engines_precompiled64_9.dll"); Name = "cudnn_engines_precompiled64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_engines_runtime_compiled64_9.dll"); Name = "cudnn_engines_runtime_compiled64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_graph64_9.dll"); Name = "cudnn_graph64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_heuristic64_9.dll"); Name = "cudnn_heuristic64_9.dll" },
    @{ Source = (Join-Path $cudnnBin "cudnn_ops64_9.dll"); Name = "cudnn_ops64_9.dll" },
    # Candle 0.8/cudarc currently also probes CUDA 13 cuBLAS on this machine.
    @{ Source = (Join-Path $cu13Bin "cublas64_13.dll"); Name = "cublas64_13.dll" },
    @{ Source = (Join-Path $cu13Bin "cublasLt64_13.dll"); Name = "cublasLt64_13.dll" }
)

if (-not $SkipBuild) {
    foreach ($path in @($CudaRoot, $MsvcCl)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required build dependency not found: $path" }
    }
    $env:CUDA_PATH = $CudaRoot
    $env:CUDA_ROOT = $CudaRoot
    $env:NVCC_CCBIN = $MsvcCl
    $env:Path = "$cudaBin;$(Split-Path -Parent $MsvcCl);$env:Path"
    Push-Location $repo
    try {
        cargo build --release -p indextts-cli --features cuda
        if ($LASTEXITCODE -ne 0) { throw "CUDA CLI build failed with exit code $LASTEXITCODE" }
        cargo build --release -p indextts-ffi --features cuda
        if ($LASTEXITCODE -ne 0) { throw "CUDA C ABI build failed with exit code $LASTEXITCODE" }
        $gendef = Join-Path $MingwBin "gendef.exe"
        $dlltool = Join-Path $MingwBin "dlltool.exe"
        foreach ($tool in @($gendef, $dlltool)) {
            if (-not (Test-Path -LiteralPath $tool)) { throw "Required cgo tool not found: $tool" }
        }
        Push-Location (Join-Path $repo "target\release")
        try {
            & $gendef "indextts.dll"
            if ($LASTEXITCODE -ne 0) { throw "gendef failed with exit code $LASTEXITCODE" }
            & $dlltool -d "indextts.def" -D "indextts.dll" -l "libindextts.dll.a" -m "i386:x86-64"
            if ($LASTEXITCODE -ne 0) { throw "dlltool failed with exit code $LASTEXITCODE" }
        } finally {
            Pop-Location
        }
    } finally {
        Pop-Location
    }
}

$missing = $requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_.Source) }
if ($missing) {
    $list = ($missing | ForEach-Object { $_.Source }) -join "`n"
    throw "Required runtime files are missing:`n$list"
}

if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Recurse -Force }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
foreach ($file in $requiredFiles) {
    Copy-Item -LiteralPath $file.Source -Destination (Join-Path $outputPath $file.Name) -Force
}
Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination $outputPath
Copy-Item -LiteralPath (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination $outputPath

$launcher = @'
@echo off
setlocal
set "ORT_DYLIB_PATH=%~dp0onnxruntime.dll"
"%~dp0indextts.exe" %*
exit /b %ERRORLEVEL%
'@
Set-Content -LiteralPath (Join-Path $outputPath "indextts-cuda.cmd") -Value $launcher -Encoding Ascii

$manifest = Get-ChildItem -LiteralPath $outputPath -File | Sort-Object Name | ForEach-Object {
    [pscustomobject]@{
        name = $_.Name
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath "manifest.json") -Encoding UTF8

$megabytes = [math]::Round(((Get-ChildItem $outputPath -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "Packaged $($manifest.Count) files ($megabytes MiB) to $outputPath"
Write-Host "Run: $outputPath\indextts-cuda.cmd --device cuda --device-index 0 synth ..."

if ($Zip) {
    $zipPath = "$outputPath.zip"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Compress-Archive -Path "$outputPath\*" -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "Wrote $zipPath"
}
