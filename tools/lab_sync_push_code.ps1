# DEPRECATED for routine code sync.
# Prefer: commit on lab/g5-code-only -> git push -> on lab: bash tools/lab_git_pull_on_lab.sh
#
# This SCP script is only an emergency fallback if git on the lab PC is unavailable.
# It will refuse to run unless -ForceEmergencyScp is passed.

param([switch]$ForceEmergencyScp)

Write-Host @"
Use the git workflow instead:

  1) On this PC (after explicit approval to commit/push):
       git checkout lab/g5-code-only
       git add ...
       git commit -m "..."    # no Cursor co-author
       git push -u origin lab/g5-code-only

  2) On lab PC:
       cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
       bash tools/lab_git_pull_on_lab.sh

Emergency SCP only: re-run with -ForceEmergencyScp
"@ -ForegroundColor Yellow

if (-not $ForceEmergencyScp) {
    exit 2
}

Write-Host "Emergency SCP not implemented in this stub; restore from git history if needed." -ForegroundColor Red
exit 1
