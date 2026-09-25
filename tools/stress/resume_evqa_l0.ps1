# Resume FULL EVQA L0 from latest checkpoint
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "Resuming full EVQA L0..."
& $Py -u tools\stress\run_evqa_l0.py --resume
exit $LASTEXITCODE
