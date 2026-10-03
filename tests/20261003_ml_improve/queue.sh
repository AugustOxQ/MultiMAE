#!/usr/bin/env bash
# Stage 1 queue of the multilearner line (spec docs/superpowers/specs/2026-10-03-improve-multilearner-design.md,
# section 6). Every 5 minutes: pull each finished run (a failed one goes back to the front of the queue once), then
# fill every free GPU on node403 and node405 with the next line of queue.txt. Launches run from a pinned git
# worktree ($WT, its res/ a symlink to the main checkout's), so commits in the main checkout never move the
# launch commit; each queue line names the commit it runs at (it must already be synced to both nodes).
#
#   queue.txt line:   <sha> <name> <seed> <attempt> <hydra overrides...>
#   A name starting with "diag" runs `bash scripts/run_diagnostics.sh <overrides>` (seed unused) and only on
#   $DIAG_NODE, where the baseline checkpoints are; the first line eligible for a node is launched there.
#   running.txt line: <tag> <node> <sha> <name> <seed> <attempt> <hydra overrides...>
#
# Start detached:  setsid nohup bash tests/20261003_ml_improve/queue.sh > /dev/null 2>&1 &
# Stop launching:  touch tests/20261003_ml_improve/STOP   (running jobs are still watched and pulled)
set -u
MAIN=/project/MultiAlign/MultiMAE
WT=/project/MultiAlign/MultiMAE-queue
DIR=$MAIN/tests/20261003_ml_improve
C=~/.claude/skills/cluster-run/cluster
PY=/root/miniconda3/envs/MultiMAE/bin/python
LOG=$DIR/queue.log
QUEUE=$DIR/queue.txt
RUNNING=$DIR/running.txt
DONE=$DIR/done.txt
NODES="node403 node405"
DIAG_NODE=node403
touch "$QUEUE" "$RUNNING" "$DONE"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

result() {  # <key> : print RESULT[key] from stdin (lists joined by spaces, missing -> empty)
  "$PY" -c 'import json, sys
for line in sys.stdin:
    if line.startswith("RESULT "):
        value = json.loads(line[7:]).get(sys.argv[1])
        print(" ".join(map(str, value)) if isinstance(value, list) else ("" if value is None else value))' "$1"
}

free_slots() { timeout 240 "$C" status --node "$1" 2>/dev/null | result free_slots | wc -w; }

launch() {  # node sha name seed overrides... -> prints the tag, or nothing
  local node=$1 sha=$2 name=$3 seed=$4
  shift 4
  # launch needs a named branch (any name); eval-baselines is checked out in the main worktree
  git -C "$WT" checkout -q -B queue-launch "$sha" || { log "checkout $sha failed"; return 1; }
  if [[ $name == diag* ]]; then
    (cd "$WT" && timeout 900 "$C" launch --node "$node" -- bash scripts/run_diagnostics.sh "$@" 2>&1) > "$DIR/.launch.out"
  else
    (cd "$WT" && timeout 900 "$C" launch --node "$node" -- python train.py data=coco_cluster seed="$seed" \
       wandb.group=ml_improve wandb.name="$name" "$@" 2>&1) > "$DIR/.launch.out"
  fi
  result tag < "$DIR/.launch.out"
}

log "queue started: $(wc -l < "$RUNNING") running, $(wc -l < "$QUEUE") queued"
while true; do
  # 1. finished runs
  : > "$RUNNING.next"
  while read -r tag node sha name seed attempt overrides; do
    [ -z "$tag" ] && continue
    timeout 240 "$C" watch "$tag" --node "$node" --once > /dev/null 2>&1
    code=$?
    if [ $code -ne 0 ] && [ $code -ne 1 ]; then
      echo "$tag $node $sha $name $seed $attempt $overrides" >> "$RUNNING.next"
      continue
    fi
    (cd "$WT" && timeout 3600 "$C" pull --node "$node" --tag "$tag" > /dev/null 2>&1) || log "pull $tag failed (exit $?)"
    if [ $code -eq 0 ]; then
      log "SUCCEEDED $tag $node $name seed=$seed (pulled)"
      echo "$tag $node $sha $name $seed $attempt succeeded $overrides" >> "$DONE"
    else
      log "FAILED $tag $node $name seed=$seed attempt=$attempt (logs pulled)"
      echo "$tag $node $sha $name $seed $attempt failed $overrides" >> "$DONE"
      if [ "$attempt" -lt 2 ]; then
        { echo "$sha $name $seed $((attempt + 1)) $overrides"; cat "$QUEUE"; } > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
      else
        log "not relaunching $name seed=$seed (2 failures)"
      fi
    fi
  done < "$RUNNING"
  mv "$RUNNING.next" "$RUNNING"

  # 2. fill free GPUs
  if [ ! -e "$DIR/STOP" ]; then
    for node in $NODES; do
      free=$(free_slots "$node")
      while [ "${free:-0}" -gt 0 ] && [ -s "$QUEUE" ]; do
        # first line eligible for this node: diag* lines only on $DIAG_NODE
        lineno=$(awk -v node="$node" -v dnode="$DIAG_NODE" '$2 !~ /^diag/ || node == dnode {print NR; exit}' "$QUEUE")
        [ -z "$lineno" ] && break
        read -r sha name seed attempt overrides < <(sed -n "${lineno}p" "$QUEUE")
        tag=$(launch "$node" "$sha" "$name" "$seed" $overrides)
        if [ -z "$tag" ]; then
          log "launch $name seed=$seed on $node failed: $(tail -n 3 "$DIR/.launch.out" | tr '\n' ' ' | cut -c1-300)"
          break  # retry next round
        fi
        sed -i "${lineno}d" "$QUEUE"
        echo "$tag $node $sha $name $seed $attempt $overrides" >> "$RUNNING"
        log "LAUNCHED $tag on $node: $name seed=$seed attempt=$attempt at $sha ($overrides)"
        free=$((free - 1))
      done
    done
  fi

  if [ ! -s "$RUNNING" ] && { [ ! -s "$QUEUE" ] || [ -e "$DIR/STOP" ]; }; then
    log "ALL DONE: nothing running, nothing to launch"
    exit 0
  fi
  sleep 300
done
