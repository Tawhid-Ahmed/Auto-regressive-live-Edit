#!/usr/bin/env bash
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
chmod +x "$REPO/tools/lab_g5_m7_delayed_start.sh" "$REPO/tools/lab_g5_llava_m7_bs_pipeline.sh"
CRON_LINE='10 19 11 10 * /home/user/TWD/LiveEdit_lab_usb/LiveEdit/tools/lab_g5_m7_delayed_start.sh'
tmp=$(mktemp)
crontab -l 2>/dev/null | grep -v lab_g5_m7_delayed_start >"$tmp" || true
echo "$CRON_LINE" >>"$tmp"
crontab "$tmp"
rm -f "$tmp"
echo "--- crontab ---"
crontab -l
date
ls -la "$REPO/tools/lab_g5_m7_delayed_start.sh" "$REPO/tools/lab_g5_llava_m7_bs_pipeline.sh"
