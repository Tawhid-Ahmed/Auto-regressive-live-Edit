# Fresh start LLaVA L0 dn=500 eps=20 (deadline-aware backbone stress)
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "AAAI Phase 2: VLKEB LLaVA L0 dn=500 eps=3 (disclose short L0 in paper)"
Write-Host "ETA ~9-10 hours wall-clock; resume: tools\stress\resume_llava_l0.ps1"
Write-Host "Monitor: Get-Content doc\results\meeting_validation\step6\stress\logs\VLKEB_llava_L0_dn500.train.log -Wait -Tail 40"
Write-Host "Resume: powershell -File tools\stress\resume_llava_l0.ps1"
& $Py -u tools\stress\run_llava_l0.py
exit $LASTEXITCODE
