#!/usr/bin/env bash
# Restart the Stage 1 queue after a container restart (or whenever it is not running).
# The cluster jobs keep running on DAS6 meanwhile; the queue's state (queue.txt, running.txt, done.txt,
# queue.log) lives in this folder on /project, so nothing is lost. Safe to run twice: it refuses to start
# a second queue.
#   bash /project/MultiAlign/MultiMAE/tests/20261003_ml_improve/resume.sh
set -u
cd /project/MultiAlign/MultiMAE
DIR=tests/20261003_ml_improve
if pgrep -f "$DIR/queue.sh" > /dev/null; then
  echo "queue already running:"; pgrep -af "$DIR/queue.sh"; exit 0
fi
[ -d /project/MultiAlign/MultiMAE-queue ] || { echo "missing worktree /project/MultiAlign/MultiMAE-queue"; exit 1; }
[ -e "$DIR/STOP" ] && echo "note: $DIR/STOP exists, so the queue will watch and pull but not launch"
rm -f "$DIR/running.txt.next" "$DIR/queue.txt.tmp"   # leftovers of a round cut off by the restart
setsid nohup bash "$DIR/queue.sh" > /dev/null 2>&1 < /dev/null &
sleep 2
pgrep -af "$DIR/queue.sh" || { echo "queue failed to start"; exit 1; }
echo "running: $(wc -l < "$DIR/running.txt") jobs; queued: $(wc -l < "$DIR/queue.txt")"
tail -n 1 "$DIR/queue.log"
