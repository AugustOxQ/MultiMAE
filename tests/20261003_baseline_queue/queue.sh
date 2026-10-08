#!/usr/bin/env bash
# Overnight queue (2026-10-03): as each seed-43 fusion run on node403 ends, pull it and launch the same
# model with seed 44 on the freed GPU; pull every finished run; relaunch a failed run once; when all are
# done, write summary.md. Run detached: setsid nohup bash tests/20261003_baseline_queue/queue.sh &
set -u
cd /project/MultiAlign/MultiMAE
C=~/.claude/skills/cluster-run/cluster
PY=/root/miniconda3/envs/MultiMAE/bin/python
DIR=tests/20261003_baseline_queue
LOG=$DIR/queue.log

queue=("fusion_none 44" "fusion_concat 44" "fusion_multilearner 44")
# tag:model:seed:attempt
running=(
  "20261002-191350-88e2663:fusion_none:43:1"
  "20261002-191517-88e2663:fusion_concat:43:1"
  "20261002-192224-88e2663:fusion_multilearner:43:1"
)

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

launch() {  # model seed -> prints the new tag; up to 5 tries ("another launch in progress", ssh hiccups)
  local model=$1 seed=$2 out tag try
  for try in 1 2 3 4 5; do
    out=$(timeout 900 "$C" launch -- python train.py model="$model" data=coco_cluster seed="$seed" 2>&1)
    tag=$(echo "$out" | grep '^RESULT' | "$PY" -c 'import sys, json
r = json.loads(sys.stdin.read()[7:])
print(r.get("tag", "") if r.get("ok") else "")' 2>/dev/null)
    if [ -n "$tag" ]; then echo "$tag"; return 0; fi
    log "launch $model seed=$seed failed (try $try): $(echo "$out" | tail -n 3 | tr '\n' ' ' | cut -c1-300)"
    sleep 120
  done
  return 1
}

start() {  # model seed attempt -> appends to next[]
  local tag
  if tag=$(launch "$1" "$2"); then
    log "LAUNCHED $1 seed=$2 attempt=$3 as $tag"
    next+=("$tag:$1:$2:$3")
  else
    log "GAVE UP launching $1 seed=$2 (5 failed tries)"
  fi
}

log "queue started: running ${running[*]}; queued ${queue[*]}"
while [ ${#running[@]} -gt 0 ]; do
  next=()
  for entry in "${running[@]}"; do
    IFS=: read -r tag model seed attempt <<< "$entry"
    timeout 180 "$C" watch "$tag" --once > /dev/null 2>&1
    code=$?
    if [ $code -ne 0 ] && [ $code -ne 1 ]; then
      [ $code -ne 10 ] && log "$tag $model seed=$seed: watch exit $code (unknown/unreachable), keep watching"
      next+=("$entry")
      continue
    fi
    timeout 3600 "$C" pull --tag "$tag" > /dev/null 2>&1 || log "pull $tag failed (exit $?); retry by hand"
    if [ $code -eq 0 ]; then
      log "SUCCEEDED $tag $model seed=$seed (pulled)"
    else
      log "FAILED $tag $model seed=$seed attempt=$attempt (pulled logs)"
      if [ "$attempt" -lt 2 ]; then
        start "$model" "$seed" $((attempt + 1))
        continue
      fi
      log "not relaunching $model seed=$seed again (2 failures)"
    fi
    if [ ${#queue[@]} -gt 0 ]; then
      read -r m s <<< "${queue[0]}"
      queue=("${queue[@]:1}")
      start "$m" "$s" 1
    fi
  done
  running=("${next[@]+"${next[@]}"}")
  [ ${#running[@]} -gt 0 ] && sleep 300
done
"$PY" "$DIR/summarize.py" > "$DIR/summary.md" 2>> "$LOG" && log "ALL DONE: summary in $DIR/summary.md" \
  || log "ALL DONE, but summarize.py failed (see above)"
