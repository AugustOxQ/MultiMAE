#!/usr/bin/env bash
# Emit each new actionable line of the Stage 1 queue log (from a persisted offset, so re-arming misses nothing),
# and a warning if the queue process is not running.
L=/project/MultiAlign/MultiMAE/tests/20261003_ml_improve/queue.log
O=/project/MultiAlign/MultiMAE/tests/20261003_ml_improve/.monitor_offset
warned=0
while true; do
  n=$(cat "$O" 2>/dev/null || echo 0); m=$(wc -l < "$L")
  if [ "$m" -gt "$n" ]; then
    sed -n "$((n + 1)),${m}p" "$L" | grep -E "SUCCEEDED|FAILED|failed|LAUNCHED|GAVE UP|not relaunching|ALL DONE"
    echo "$m" > "$O"
  fi
  if pgrep -f "[t]ests/20261003_ml_improve/queue.sh" > /dev/null; then warned=0
  elif [ $warned -eq 0 ]; then echo "QUEUE NOT RUNNING (run tests/20261003_ml_improve/resume.sh)"; warned=1; fi
  sleep 30
done
