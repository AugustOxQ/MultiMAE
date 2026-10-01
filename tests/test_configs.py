import pytest

from helpers import MODEL_NAMES, compose_cfg


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_config_composes(name):
    cfg = compose_cfg(f"model={name}")
    assert cfg.model.name == name
    assert cfg.model.backbone.type == "hf_clip"
    both = set(cfg.model.modalities) == {"image", "text"}
    assert cfg.model.monitor.metric == ("val/retrieval/rsum" if both else "val/loss")
    assert cfg.train.lr == 1e-4 and cfg.train.lr_backbone == 1e-5
    assert cfg.wandb.project == "multimae" and cfg.paths.res_dir == "res"
