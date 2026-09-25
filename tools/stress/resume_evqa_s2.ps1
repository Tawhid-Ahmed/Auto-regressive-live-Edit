# Resume EVQA S2
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
Set-Location $Root
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_evqa_s2.ps1") -Resume
exit $LASTEXITCODE
