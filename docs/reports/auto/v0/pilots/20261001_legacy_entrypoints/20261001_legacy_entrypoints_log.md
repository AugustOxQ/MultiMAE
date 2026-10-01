# 20261001 legacy entry points

## Problem
`src/hook/__init__.py` exports only the fusion trainers, so `from src.hook import train_mae` (main_mae.py), `train_mlm` (main_mlm.py), `train_mmae` (main_mmae.py, src/test/test_mmae_updated.py) bind the submodule, and calling it raises `TypeError: 'module' object is not callable`.

## Fix
Import the function from its submodule (`from src.hook.train_mae import train_mae`, same for mlm/mmae). `__init__.py` untouched.

Two further blocking bugs found while smoke-running:
1. main_mae.py: after training, `wandb_logger.log_metrics(...)` is called unguarded; with `wandb.enabled=false` -> `AttributeError: 'NoneType' object has no attribute 'log_metrics'`. Fixed with a `if wandb_logger is None: break` guard in the epoch loop (the later block was already guarded).
2. test_mmae_updated.py used `image_size=32` with patch 16 (only 4 patches). `MultiModalMAE.encode_image_split` (src/model/mmae.py, not mine) then returns a non-finite `features_masked`, so the "test" printed `nan` for every loss yet reported success. Changed to `image_size=64` (finite). Underlying edge case in mmae.py left alone (not my file).

## Not fixed / notes
- (Fixed later, see Follow-up fixes: the config now points at `data/cifar`.) The original default `train.data_root: dataset/cifar` did not hold the data (the repo has data/cifar), so main_mae.py with the old default started downloading CIFAR-10 to dataset/cifar. I ran with `train.data_root=data/cifar`. That early run with the old config left a partial download, `dataset/cifar/cifar-10-python.tar.gz` (about 93 MB, untracked), so `dataset/cifar/` now exists; my rm was denied by the permission classifier, so it is still there and can be deleted.
- (Fixed later, see Follow-up fixes: finite-loss check added.) test_mmae_updated.py returned True even when losses were NaN (no finite check).
- The test's second function `test_wandb_logging` only prints mock numbers.

## Commands and outcomes (cwd = repo root)
Envs: first main_mae/main_mlm runs used CoSiR + PYTHONPATH (outputs mae_cifar10_run*.out, mlm_try1.out). Final results below are from /root/miniconda3/envs/MultiMAE/bin/python (no PYTHONPATH).

| Entry | Command (MultiMAE env) | Result |
|---|---|---|
| main_mae.py | `python main_mae.py wandb.enabled=false train.epochs=1 train.batch_size=128 train.num_workers=2 train.data_root=data/cifar` | rc 0. Epoch 1/1 Train loss 0.2252, Val loss 0.1874, "Training finished. Last LR: 0.0" (mae_MultiMAE_env.out). Before the guard fix: AttributeError after training (mae_cifar10_run.out). |
| main_mlm.py | `python main_mlm.py wandb.enabled=false train.epochs=1 train.batch_size=64 train.num_workers=2` | rc 0. Epoch 1/1 Train loss 2.7584, Val loss 2.3446 (wikitext-2, bert-base-uncased) (mlm_MultiMAE_env.out). |
| main_mmae.py, subset | `python run_small_coco.py 512 main_mmae wandb.enabled=false train.epochs=2 train.batch_size=32 train.num_workers=2` (driver truncates COCOImageTextDataset.data to 512 items, then runpy's main_mmae.py as __main__) | rc 0, 2 epochs + test pass; test losses finite (Total 8.7381, MAE 0.4237, MLM 4.6536, Contrastive 3.6608) (mmae_small_mm3.out). |
| main_mmae.py, full COCO | `timeout 150 python main_mmae.py wandb.enabled=false train.batch_size=64 train.num_workers=4` | killed by timeout (rc 124) after 731/8856 steps of epoch 1, step loss total 16.6124, mae 0.4143, mlm 2.5881, finite (main_mmae_full_timeout.out). |
| test_mmae_updated.py, subset | `python run_small_coco.py 64 test_mmae_updated` (after image_size=64) | rc 0; train 15.2997, val 6.5651, test total 6.0902, mae 0.4552, mlm 4.1028, contrastive 1.5321 (test_mmae_updated_small64.out). With image_size=32: all losses nan (test_mmae_updated_small.out). |
| test_mmae_updated.py, full COCO | `timeout 120 python src/test/test_mmae_updated.py` | rc 124 by timeout at step 1415/141687, total_loss 7.0705 finite (test_mmae_updated_full_timeout64.out). With image_size=32 the same run showed nan from step 1 (..._full_timeout.out). |

Other files: run_small_coco.py (driver), nan_probe.py (shows img split non-finite only at image_size=32).

## Follow-up fixes (controller fix wave)

Three small edits to address issues found during testing:

1. **configs/mae_config.yaml**: Changed `train.data_root: dataset/cifar` to `train.data_root: data/cifar`. The dataset directory at `data/cifar/` exists; `dataset/cifar/` held no usable data (only the partial download described above, created by an early run with the old config). This prevents accidental fresh downloads to the wrong path when main_mae.py runs with default config.

2. **main_mae.py (lines 105-118)**: Wrapped the epoch-wise metric logging loop in `if wandb_logger is not None:`. The previous code used a guard *inside* the loop (`if wandb_logger is None: break`), which meant the loop would still attempt one iteration before exiting. The new guard is cleaner and matches the style used later in the same function (line 122, `if wandb_logger is not None:`).

3. **src/test/test_mmae_updated.py**: 
   - Added `import math` at the top (line 6).
   - Inserted a finite-value check before the final `return True` (after line 55): collects all loss values and returns `False` if any contain NaN or inf. Previously the test reported success even when all losses were NaN.

### Verification results

- **Import check**: `/root/miniconda3/envs/MultiMAE/bin/python -c "import main_mae, src.test.test_mmae_updated"` → success.
- **Config check**: `OmegaConf.load('configs/mae_config.yaml').train.data_root` → `data/cifar`.
- **Guard check (NaN detection)**:
  - With `image_size=64` (correct): `python run_small_coco.py 10 test_mmae_updated` → test losses finite (Total 6.3019, MAE 0.5291, MLM 4.4472, Contrastive 1.3256), TEST_RESULT True.
  - With `image_size=32` (triggers NaN): same command → losses all NaN, error message printed ("❌ 损失中出现非有限值 (NaN/inf): [nan, nan, ...]"), TEST_RESULT False.
  - Restored `image_size=64`: test passes again (finite losses).
- **Smoke run of main_mae.py**: `python main_mae.py wandb.enabled=false train.epochs=1 train.batch_size=128 train.num_workers=2` (no data_root override; uses config value data/cifar) → rc 0, "Training finished. Last LR: 0.0". Directory `data/cifar` mtime unchanged before and after (1755607704), confirming no download or modification occurred.
