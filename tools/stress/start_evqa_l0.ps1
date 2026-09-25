# Fresh start EVQA L0 on FULL train set (AAAI protocol)
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "AAAI protocol: full EVQA L0 (no -dn subset)"
Write-Host "Monitor: Get-Content `"$Root\doc\results\meeting_validation\step6\stress\logs\EVQA_L0_full.train.log`" -Wait -Tail 40"
Write-Host "Resume:  powershell -File tools\stress\resume_evqa_l0.ps1"
& $Py -u tools\stress\run_evqa_l0.py
exit $LASTEXITCODE
