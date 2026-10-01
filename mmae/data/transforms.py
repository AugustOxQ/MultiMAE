"""Image preprocessing identical to OpenAI CLIP's: bicubic resize of the shortest side, center crop,
CLIP mean/std. Sizes and statistics come from the backbone's image processor config. (HF's own
image processor resizes slightly differently; the published zero-shot numbers use this pipeline.)"""
from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor


def build_image_transform(processor_name: str) -> T.Compose:
    processor = AutoImageProcessor.from_pretrained(processor_name)
    size = processor.size["shortest_edge"]
    crop = (processor.crop_size["height"], processor.crop_size["width"])
    return T.Compose(
        [
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
