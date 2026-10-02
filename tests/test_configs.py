import pytest

from helpers import CLIP_NAME, MODEL_NAMES, compose_cfg
from mmae.models.backbones import processor_name


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_config_composes(name):
    cfg = compose_cfg(f"model={name}")
    assert cfg.model.name == name
    assert cfg.model.backbone.type == "hf_clip"
    both = set(cfg.model.modalities) == {"image", "text"}
    assert cfg.model.monitor.metric == ("val/retrieval/rsum" if both else "val/loss")
    assert cfg.train.lr == 1e-4 and cfg.train.lr_backbone == 1e-5
    assert cfg.wandb.project == "multimae" and cfg.paths.res_dir == "res"


def test_processor_follows_pretrained():
    """Swapping the checkpoint swaps the tokenizer and preprocessing; the tiny CLIP uses B/32's; an explicit
    processor still wins."""
    assert processor_name(compose_cfg().model.backbone) == CLIP_NAME
    large = compose_cfg("model.backbone.pretrained=openai/clip-vit-large-patch14").model.backbone
    assert large.processor == "openai/clip-vit-large-patch14" == processor_name(large)
    assert processor_name(compose_cfg("model.backbone.pretrained=tiny-random-clip").model.backbone) == CLIP_NAME
    explicit = compose_cfg("model.backbone.pretrained=tiny-random-clip", "model.backbone.processor=openai/clip-vit-base-patch16")
    assert processor_name(explicit.model.backbone) == "openai/clip-vit-base-patch16"


def test_cluster_data_config_points_at_the_node():
    local = compose_cfg()
    cluster = compose_cfg("data=coco_cluster")
    assert set(cluster.data) == set(local.data)  # coco_cluster.yaml repeats every data key
    assert cluster.data.images_dir == "/local/wding/Dataset/coco/images"
    assert cluster.data.annotations_dir == "/local/wding/Dataset/coco/annotations"
    assert cluster.data.max_text_len == local.data.max_text_len
    assert cluster.paths.res_dir == "/local/wding/res/MultiMAE/coco"
    debug = compose_cfg("data=coco_cluster", "train=debug")
    assert debug.data.limit_train == 256 and debug.paths.res_dir == "/local/wding/res/MultiMAE/coco"
