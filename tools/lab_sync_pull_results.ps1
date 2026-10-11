# Pull G5 experiment tracker, logs, and small result artifacts from lab -> this repo.
# Does NOT pull full multi-GB checkpoints by default.
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1
#   powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta

param(
    [string]$LabHost = "user@100.89.137.40",
    [string]$LabRepo = "/home/user/TWD/LiveEdit_lab_usb/LiveEdit",
    [switch]$IncludeCkptMeta
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LocalG5 = Join-Path $Root "doc\results\cvpr2027\g5"
New-Item -ItemType Directory -Force -Path $LocalG5 | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $LocalG5 "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $LocalG5 "lab_mirror") | Out-Null

Write-Host "=== PULL RESULTS from $LabHost ===" -ForegroundColor Cyan

# Tracker + logs
scp -q "${LabHost}:${LabRepo}/doc/results/cvpr2027/g5/EXPERIMENT_TRACKER.md" (Join-Path $LocalG5 "EXPERIMENT_TRACKER.md") 2>$null
scp -q "${LabHost}:${LabRepo}/doc/results/cvpr2027/g5/experiments.jsonl" (Join-Path $LocalG5 "experiments.jsonl") 2>$null
scp -q "${LabHost}:${LabRepo}/doc/results/cvpr2027/g5/LAST_CODE_SYNC.txt" (Join-Path $LocalG5 "LAST_CODE_SYNC.txt") 2>$null
ssh -o BatchMode=yes $LabHost "mkdir -p '$LabRepo/doc/results/cvpr2027/g5/logs'; ls '$LabRepo/doc/results/cvpr2027/g5/logs' 2>/dev/null | head"
# Per-file scp (avoid scp -r with spaces in Windows local path)
$logNames = ssh -o BatchMode=yes $LabHost "ls '$LabRepo/doc/results/cvpr2027/g5/logs' 2>/dev/null"
foreach ($name in ($logNames -split "`n")) {
    $n = $name.Trim()
    if (-not $n) { continue }
    scp -q "${LabHost}:${LabRepo}/doc/results/cvpr2027/g5/logs/$n" (Join-Path $LocalG5 "logs\$n") 2>$null
}

# Taxonomy + any small G5 result JSONs written under doc/results/cvpr2027/g5/
scp -q "${LabHost}:${LabRepo}/doc/results/cvpr2027/g5/leakage_taxonomy_llava.json" (Join-Path $LocalG5 "leakage_taxonomy_llava.json") 2>$null

# Lightweight eval summaries for g5_llava_* runs (not full generations)
$evalMirror = Join-Path $LocalG5 "lab_mirror\eval_results"
New-Item -ItemType Directory -Force -Path $evalMirror | Out-Null
$evalFiles = ssh -o BatchMode=yes $LabHost "find '$LabRepo/eval_results/meeting_validation/step6' -type f \( -name seed_42_mean_results.json -o -name routing_diagnostics.json -o -name meta.json \) -path '*g5_llava*' 2>/dev/null"
foreach ($line in ($evalFiles -split "`n")) {
    $rp = $line.Trim()
    if (-not $rp) { continue }
    $rel = $rp.Substring($LabRepo.Length).TrimStart('/')
    $dest = Join-Path $evalMirror ($rel -replace '/', '\')
    $destDir = Split-Path $dest -Parent
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    scp -q "${LabHost}:$rp" $dest
}

# Lab backup copy
$backupLocal = Join-Path $LocalG5 "lab_mirror\experiment_backup"
New-Item -ItemType Directory -Force -Path $backupLocal | Out-Null
scp -q "${LabHost}:/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5/EXPERIMENT_TRACKER.md" (Join-Path $backupLocal "EXPERIMENT_TRACKER.md") 2>$null
scp -q "${LabHost}:/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5/experiments.jsonl" (Join-Path $backupLocal "experiments.jsonl") 2>$null

# Lightweight run metadata (config.yaml, runtime.json, pipeline_event.json) — not weights
if ($IncludeCkptMeta) {
    $metaDir = Join-Path $LocalG5 "lab_mirror\run_meta"
    New-Item -ItemType Directory -Force -Path $metaDir | Out-Null
    $remoteList = ssh -o BatchMode=yes $LabHost "find '$LabRepo/records/lab_g5' -type f \( -name config.yaml -o -name runtime.json -o -name pipeline_event.json \) 2>/dev/null"
    foreach ($line in ($remoteList -split "`n")) {
        $rp = $line.Trim()
        if (-not $rp) { continue }
        $rel = $rp.Substring($LabRepo.Length).TrimStart('/')
        $dest = Join-Path $metaDir ($rel -replace '/', '\')
        $destDir = Split-Path $dest -Parent
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        scp -q "${LabHost}:$rp" $dest
    }
}

$stamp = Get-Date -Format "o"
Set-Content -Path (Join-Path $LocalG5 "LAST_RESULTS_PULL.txt") -Value "pulled_from_lab $stamp"
Write-Host "=== PULL DONE $stamp ===" -ForegroundColor Green
Write-Host "See: doc/results/cvpr2027/g5/EXPERIMENT_TRACKER.md"
