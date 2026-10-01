"""How train.save_dir parses in the baseline and fixed configs, and what the hook does with it."""
import os

from hydra import compose, initialize_config_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
S = os.environ.get("HARNESS_DIR") or os.path.join(_HERE, "out", "work")  # needs make_trees.py first
R = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for name, cfg_dir in [
    ("baseline", f"{S}/baseline/configs"),
    ("fixed", f"{R}/configs"),
]:
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="fusion_mmae_config")
        v = cfg.train.save_dir
        # the hooks do `if save_dir: os.makedirs(save_dir)` and save when `save_dir` is truthy
        print(
            f"{name}: train.save_dir={v!r} type={type(v).__name__} "
            f"-> hook saves checkpoints: {bool(v)}"
            + (f" (into ./{v}/)" if v else "")
        )
        cfg2 = compose(config_name="fusion_mmae_config", overrides=["train.save_dir=ckpts"])
        print(f"{name}: with override train.save_dir=ckpts -> {cfg2.train.save_dir!r}")
