# T1.4 EVQA DEK — fresh or resume
param([switch]$Resume)
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "Monitor: Get-Content `"$Root\doc\results\meeting_validation\step6\stress\logs\EVQA_avbt_DEK_stress.train.log`" -Wait -Tail 40"
if ($Resume) {
  & $Py -u tools\stress\run_evqa_finetune.py --mode dek --resume
} else {
  & $Py -u tools\stress\run_evqa_finetune.py --mode dek
}
exit $LASTEXITCODE
