# Resume LLaVA L0 dn=500 from latest good checkpoint
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "Resuming VLKEB_llava_L0_dn500 ..."
Write-Host "Monitor: Get-Content `"$Root\doc\results\meeting_validation\step6\stress\logs\VLKEB_llava_L0_dn500.train.log`" -Wait -Tail 40"
& $Py -u tools\stress\run_llava_l0.py --resume
exit $LASTEXITCODE
