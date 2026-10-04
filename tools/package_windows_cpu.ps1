param(
    [string]$Output = "dist/index-tts-rust-win64-cpu",
    [string]$OrtBin = "D:\Python3\Lib\site-packages\onnxruntime\capi",
    [string]$MingwBin = "D:\mingw-w64\bin",
    [switch]$SkipBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath((Join-Path $repo $Output))

if (-not $SkipBuild) {
    Push-Location $repo
    try {
        cargo build --release -p indextts-cli
        if ($LASTEXITCODE -ne 0) { throw "CPU CLI build failed: $LASTEXITCODE" }
        cargo build --release -p indextts-ffi
        if ($LASTEXITCODE -ne 0) { throw "CPU C ABI build failed: $LASTEXITCODE" }
        Push-Location (Join-Path $repo "target\release")
        try {
            & (Join-Path $MingwBin "gendef.exe") "indextts.dll"
            if ($LASTEXITCODE -ne 0) { throw "gendef failed: $LASTEXITCODE" }
            & (Join-Path $MingwBin "dlltool.exe") -d "indextts.def" -D "indextts.dll" -l "libindextts.dll.a" -m "i386:x86-64"
            if ($LASTEXITCODE -ne 0) { throw "dlltool failed: $LASTEXITCODE" }
        } finally { Pop-Location }
    } finally { Pop-Location }
}

$files = @(
    @{ Source = (Join-Path $repo "target\release\indextts.exe"); Name = "indextts.exe" },
    @{ Source = (Join-Path $repo "target\release\indextts.dll"); Name = "indextts.dll" },
    @{ Source = (Join-Path $repo "target\release\indextts.dll.lib"); Name = "indextts.dll.lib" },
    @{ Source = (Join-Path $repo "target\release\libindextts.dll.a"); Name = "libindextts.dll.a" },
    @{ Source = (Join-Path $repo "crates\indextts-ffi\indextts.h"); Name = "indextts.h" },
    @{ Source = (Join-Path $OrtBin "onnxruntime.dll"); Name = "onnxruntime.dll" }
)
$missing = $files | Where-Object { -not (Test-Path -LiteralPath $_.Source) }
if ($missing) { throw "Missing package files: $($missing.Source -join ', ')" }
if (Test-Path -LiteralPath $outputPath) { Remove-Item $outputPath -Recurse -Force }
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
foreach ($file in $files) { Copy-Item $file.Source (Join-Path $outputPath $file.Name) -Force }
Copy-Item (Join-Path $repo "LICENSE") $outputPath
Copy-Item (Join-Path $repo "THIRD_PARTY_NOTICES.md") $outputPath
Copy-Item (Join-Path $repo "tools\verify-runtime.ps1") $outputPath
$manifest = [ordered]@{
    package = "index-tts-rust-win64-cpu"
    architecture = "x86_64-pc-windows-msvc"
    abi_version = "1.5"
    backend = "cpu"
    files = @{}
}
Get-ChildItem $outputPath -File | Sort-Object Name | ForEach-Object {
    $manifest.files[$_.Name] = [ordered]@{ bytes = $_.Length; sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputPath "runtime-manifest.json") -Encoding UTF8
Write-Host "Packaged CPU runtime to $outputPath"
if ($Zip) {
    $zipPath = "$outputPath.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive "$outputPath\*" $zipPath -CompressionLevel Optimal
    Write-Host "Wrote $zipPath"
}
