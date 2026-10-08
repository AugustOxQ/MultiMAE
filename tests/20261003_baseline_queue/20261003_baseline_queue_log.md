# 2026-10-03 baseline queue: the remaining fusion seeds, run overnight unattended

## Problem

By the evening of 2026-10-02 the contrastive baseline had finished three seeds (42, 43, 44) on node403, and the three fusion models (`fusion_none`, `fusion_concat`, `fusion_multilearner`) had seed 42 (trained at commit 2056ea5, re-evaluated with `evaluate.py` at 88e2663 for the extended test metrics) and seed 43 running (jobs `20261002-191350-88e2663`, `20261002-191517-88e2663`, `20261002-192224-88e2663`, started 21:15 to 21:23 CEST). Seed 44 of each fusion model still had to run. Each fusion run takes about 7 hours on one RTX A6000, so the seed-44 runs had to start, one per GPU, as each seed-43 run freed its GPU during the night, with nobody watching.

## How

- `queue.sh`, started detached (`setsid nohup bash tests/20261003_baseline_queue/queue.sh &`, output in `nohup.out`, which stayed empty), polled the three running jobs every 5 minutes with `cluster watch <tag> --once`.
- When a job ended it pulled the run folder (`cluster pull --tag`), logged SUCCEEDED or FAILED, and launched the next queued run (`fusion_none 44`, `fusion_concat 44`, `fusion_multilearner 44`, in that order) with `cluster launch -- python train.py model=<m> data=coco_cluster seed=44`, which takes the lowest free GPU. A failed run would have been relaunched once; a launch would have been retried up to 5 times, 2 minutes apart.
- When no job was left it ran `summarize.py`, which reads every completed run folder under `res/coco/multimae/default/` that has PMRP in its test metrics (the seed-42 fusion runs from `run.json["eval"]["test"]`, the others from `results.test`), and wrote `summary.md` (mean ± std per model over seeds 42, 43, 44, plus a per-seed table).

## Outcome

All six runs succeeded on the first attempt; no relaunch and no launch retry was needed. Times from `queue.log` (CEST):

| Event | Time |
|---|---|
| queue started, three seed-43 runs in progress | 2026-10-03 03:04:28 |
| `fusion_none` seed 43 succeeded and pulled; seed 44 launched | 04:07:56; 04:08:51 |
| `fusion_concat` seed 43 succeeded and pulled; seed 44 launched | 04:14:39; 04:15:33 |
| `fusion_multilearner` seed 43 succeeded and pulled; seed 44 launched | 04:42:19; 04:43:15 |
| `fusion_none` seed 44 succeeded and pulled | 11:05:34 |
| `fusion_concat` seed 44 succeeded and pulled | 11:11:18 |
| `fusion_multilearner` seed 44 succeeded and pulled | 11:57:37 |
| `summary.md` written | 11:57:37 |

Every run trained 10 epochs on one GPU (`num_processes` 1) and picked epoch 10 as its best epoch. Training took 6.84 to 7.23 hours per fusion run (`run.json` `duration_s`). The results, re-derived from the run folders, are analysed in `docs/reports/auto/v1/2026-10-03_baselines.md`; its build script `docs/reports/assets/build_2026-10-03_baselines.py` reproduces every number in `summary.md`.

Files: `queue.sh` (the runner), `queue.log` (its log), `summarize.py`, `summary.md`, `nohup.out` (empty).
