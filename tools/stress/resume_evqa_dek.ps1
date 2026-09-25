# Resume EVQA DEK
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
Set-Location $Root
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_evqa_dek.ps1") -Resume
exit $LASTEXITCODE
