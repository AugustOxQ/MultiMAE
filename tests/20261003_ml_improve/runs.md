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
