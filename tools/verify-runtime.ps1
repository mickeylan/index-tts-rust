param(
    [string]$RuntimeDir = $PSScriptRoot,
    [string]$ModelDir,
    [int]$DeviceIndex = 0
)

$ErrorActionPreference = "Stop"
$runtime = [IO.Path]::GetFullPath($RuntimeDir)
$manifestPath = Join-Path $runtime "runtime-manifest.json"
if (-not (Test-Path $manifestPath)) { throw "runtime-manifest.json is missing" }
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($property in $manifest.files.PSObject.Properties) {
    $path = Join-Path $runtime $property.Name
    if (-not (Test-Path -LiteralPath $path)) { throw "Runtime file is missing: $($property.Name)" }
    $file = Get-Item $path
    if ($file.Length -ne [int64]$property.Value.bytes) { throw "Size mismatch: $($property.Name)" }
    $hash = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $property.Value.sha256) { throw "SHA-256 mismatch: $($property.Name)" }
}
if ($ModelDir) {
    $modelManifest = Join-Path $ModelDir "manifest.json"
    if (-not (Test-Path $modelManifest)) { throw "Model manifest is missing: $modelManifest" }
    $model = Get-Content $modelManifest -Raw | ConvertFrom-Json
    if ($model.runtime -ne "index-tts-rust" -or $model.sample_rate -ne 22050) {
        throw "Model manifest is incompatible"
    }
}
if ($manifest.backend -eq "cuda") {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) { throw "NVIDIA driver utility nvidia-smi.exe is unavailable" }
    $gpuCount = [int](& $nvidiaSmi.Source --query-gpu=count --format=csv,noheader,nounits | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $DeviceIndex -lt 0 -or $DeviceIndex -ge $gpuCount) {
        throw "CUDA device index $DeviceIndex is unavailable"
    }
    foreach ($dll in @("onnxruntime_providers_cuda.dll", "cudnn64_9.dll", "cudart64_12.dll")) {
        if (-not (Test-Path (Join-Path $runtime $dll))) { throw "CUDA dependency is missing: $dll" }
    }
}
Write-Host "Runtime verification passed: $($manifest.package) ABI $($manifest.abi_version)"
