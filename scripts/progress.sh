#!/bin/bash
# Check progress of the running PhishProof panel run. Usage: bash scripts/progress.sh
cd "$(dirname "$0")/.." || exit 1

echo "=== Process ==="
if pgrep -f "build_results_bundle.py" >/dev/null; then
  pid=$(pgrep -f "build_results_bundle.py" | head -1)
  cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
  echo "ĐANG CHẠY (PID $pid, CPU%=$cpu)"
else
  echo "Không có process — đã xong hoặc chưa bắt đầu"
fi

echo ""
echo "=== Tiến độ mới nhất ==="
# pick the newest task log that is an actual panel run (contains 'running panel')
latest=$(grep -l "running panel" /private/tmp/claude-*/*phishing*/*/tasks/*.output 2>/dev/null \
         | xargs ls -t 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  now=$(date +%s); lm=$(stat -f %m "$latest" 2>/dev/null)
  echo "(cập nhật $((now-lm))s trước)"
  grep -E "sweeping|[0-9]+/[0-9]+|wrote|fitted|running panel|rror|Traceback" "$latest" | tail -5
else
  echo "(chưa tìm thấy log panel run)"
fi

echo ""
echo "=== Bundle output (chỉ có khi XONG) ==="
done_any=0
for b in results/bundle_pilot_v3.jsonl results/bundle_4000.jsonl; do
  if [ -f "$b" ]; then echo "  $b: $(wc -l < "$b" | tr -d ' ') dòng  ✅ XONG"; done_any=1; fi
done
[ "$done_any" = 0 ] && echo "  (chưa có bundle — đang chạy)"
