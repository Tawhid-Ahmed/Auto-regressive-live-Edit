# T1.3 EVQA S2 (AVBT) — fresh or resume
param([switch]$Resume)
$Root = "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
$Py = "E:\anaconda\envs\liveedit\python.exe"
Set-Location $Root
Write-Host "Monitor: Get-Content `"$Root\doc\results\meeting_validation\step6\stress\logs\EVQA_avbt_S2_stress.train.log`" -Wait -Tail 40"
if ($Resume) {
  & $Py -u tools\stress\run_evqa_finetune.py --mode s2 --resume
} else {
  & $Py -u tools\stress\run_evqa_finetune.py --mode s2
}
exit $LASTEXITCODE
