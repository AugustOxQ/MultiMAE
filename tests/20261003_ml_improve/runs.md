# Run registry: improving fusion_multilearner (spec docs/superpowers/specs/2026-10-03-improve-multilearner-design.md)

All runs: `data=coco_cluster wandb.group=ml_improve`, one GPU each. Results pull to `res/coco/multimae/ml_improve/`.

| Tag | Node/slot | Stage | Arm | Seed | Commit | Overrides | Status |
|---|---|---|---|---|---|---|---|
| 20261003-162109-eb85340 | node403/0 | 1 | M2a multilearner text 40% | 42 | eb85340 | `model=fusion_multilearner model.masking.text_ratio=0.4 wandb.name=multilearner_txt40` | running (launched 18:21) |
| 20261003-162213-eb85340 | node403/1 | 1 | R3 contrastive mean pooling | 42 | eb85340 | `model=contrastive model.pooling=mean wandb.name=contrastive_meanpool` | running (launched 18:22) |
| 20261003-162314-eb85340 | node403/2 | 1 | R5 contrastive 15 epochs | 42 | eb85340 | `model=contrastive train.epochs=15 wandb.name=contrastive_ep15` | running (launched 18:23) |
| 20261003-170713-9203048 | node405/0 | 1 | M5 multilearner MAE off (weight 0), MLM kept | 42 | 9203048 | `model=fusion_multilearner model.loss.weights.mae=0 wandb.name=multilearner_mae0` | never started (job.sh path lost its first character in the tmux pane); relaunched below |
| 20261003-170850-9203048 | node405/1 | 1 | M2a multilearner text 40% | 43 | 9203048 | `model=fusion_multilearner model.masking.text_ratio=0.4 wandb.name=multilearner_txt40` | running (launched 19:08) |
| 20261003-170954-9203048 | node405/2 | 1 | R3 contrastive mean pooling | 43 | 9203048 | `model=contrastive model.pooling=mean wandb.name=contrastive_meanpool` | running (launched 19:09) |
| (relaunch) | node405/0 | 1 | M5 multilearner MAE off (weight 0), MLM kept | 42 | this commit | `model=fusion_multilearner model.loss.weights.mae=0 wandb.name=multilearner_mae0` | to launch |

From 2026-10-03 ~20:30 the Stage 1 queue (`queue.sh`, log `queue.log`, state `queue.txt` / `running.txt` / `done.txt`) launches the remaining arms on both nodes as GPUs free and pulls every finished run; tags are in `queue.log` and `done.txt`. Queued at e8eb0ad's successor (M1 from fad0c6c, M2b from e8eb0ad): M1 clean, M2b content, M1 clean_detached (seed 42); M5 MAE off, M1 clean, M2b content, R5 15 epochs, M1 clean_detached (seed 43).
Added to the queue at the next cluster-run commit (M3 from ce60916, M6 from f196a61): M3 pooled conditioning and M6 masked-view InfoNCE (weight 0.25), seeds 42 and 43.
Added to the queue at the next cluster-run commit (R2 from 9005ac1): R2 contrastive with text lr 5e-5, vision lr 5e-6, layer decay 0.7, vision frozen 2 epochs, seeds 42 and 43.
Stage 0 diagnostics (`scripts/run_diagnostics.sh`, Task 10 at 2261333) queued at the next cluster-run commit as `diag_stage0`, first in the queue, node403 only (where the 12 baseline checkpoints are): `+diag.runs_root=/local/wding/res/MultiMAE/coco/multimae/default +diag.out=/local/wding/res/MultiMAE/coco/diagnostics/stage0 eval.vwsd_dir=/local/wding/Dataset/vwsd`. CPU dry run with zero-shot only on the real data (10-03 21:39): per-query PMRP 55.31 (package 55.32), rsum 361.97, VWSD Hit@1 58.10 / MRR 72.79.
