param(
    [string]$EnvName = "liveedit",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$wd = (Get-Location).Path

function Resolve-CondaExe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    foreach ($c in @(
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
            "$env:USERPROFILE\miniconda3\condabin\conda.bat",
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
            "$env:USERPROFILE\anaconda3\condabin\conda.bat",
            "E:\anaconda\Scripts\conda.exe",
            "E:\anaconda\condabin\conda.bat"
        )) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$condaExe = Resolve-CondaExe
if (-not $condaExe) {
    Write-Error "conda not found. Set CONDA_EXE to your conda.exe path, or add conda to PATH."
    exit 1
}

Write-Host "WorkingDirectory: $wd"
Write-Host "Conda: $condaExe | env: $EnvName (default: liveedit) | device: $Device"

$logAll = Join-Path $wd "_diag_blip2_run.log"
$scriptPath = Join-Path $wd "_diag_blip2_load.py"
$runner = Join-Path $wd "_diag_conda_run.cmd"
Remove-Item $logAll, $runner -ErrorAction SilentlyContinue

# Wrapper batch: conda + quoted paths + log (avoids conda.bat redirect quirks)
$batch = @"
@echo off
call "$condaExe" run -n $EnvName python "$scriptPath" $Device 1> "$logAll" 2>&1
exit /b %ERRORLEVEL%
"@
Set-Content -Path $runner -Value $batch -Encoding ASCII

Write-Host "Starting wrapper: $runner"
$proc = Start-Process -FilePath $runner -WorkingDirectory $wd -PassThru -WindowStyle Hidden
$start = Get-Date

while (-not $proc.HasExited) {
    $pyMax = Get-Process python -ErrorAction SilentlyContinue | Measure-Object -Property WorkingSet64 -Maximum
    $mb = if ($pyMax.Maximum) { [math]::Round($pyMax.Maximum / 1MB) } else { 0 }
    Write-Host ("{0:HH:mm:ss} max python WorkingSet={1} MB (conda wrapper PID={2})" -f (Get-Date), $mb, $proc.Id)
    if (((Get-Date) - $start).TotalSeconds -gt 900) {
        Write-Host "Timeout 900s - stopping"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        break
    }
    Start-Sleep -Seconds 2
}

try { $null = $proc.WaitForExit(30000) } catch {}
Write-Host ("Wrapper ExitCode: {0}" -f $proc.ExitCode)
Write-Host '---- _diag_blip2_run.log ----'
if (Test-Path $logAll) { Get-Content $logAll -ErrorAction SilentlyContinue } else { Write-Host "(no log file)" }
