import json
import os
import pathlib

import torch
from datasets import config
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


class COCOImageTextDataset(Dataset):

    def __init__(
        self,
        root: str = "/data/SSD/coco/images/",
        annotation_file: str = "/data/SSD/coco/annotations/xxx.json",
        split: str = "train",
        image_size: int = 224,
        tokenizer_name: str = "bert-base-uncased",
        max_len: int = 64,
    ):
        self.root = root
        self.annotation_file = annotation_file
        self.split = split
        self.image_size = image_size
        self.max_len = max_len
        self.annotation_file_prefix = pathlib.Path(root).parent / "annotations"
        if split == "train":
            self.annotation_file = (
                self.annotation_file_prefix / "coco_karpathy_train.json"
            )
        elif split == "val":
            self.annotation_file = (
                self.annotation_file_prefix / "coco_karpathy_val_one_caption.json"
            )
        elif split == "test":
            self.annotation_file = (
                self.annotation_file_prefix / "coco_karpathy_test_one_caption.json"
            )

        with open(self.annotation_file, "r") as f:
            self.data = json.load(f)

        # Setup transforms
        self.transform = Compose(
            [
                Resize((self.image_size, self.image_size)),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.root, item["image"])
        image = Image.open(image_path).convert("RGB")
        caption = item["caption"]

        # Apply image transforms
        image = self.transform(image)

        # Tokenize caption
        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        token_ids = toks["input_ids"].squeeze(0)

        return image, token_ids


class MSCOCOTestDataset(Dataset):

    def __init__(
        self,
        root: str = "/data/SSD/coco/images/",
        annotation_file: str = "/data/SSD/coco/annotations/xxx.json",
        split: str = "train",
        image_size: int = 224,
        tokenizer_name: str = "bert-base-uncased",
        max_len: int = 64,
    ):
        self.root = root
        self.annotation_file = annotation_file
        self.split = split
        self.image_size = image_size
        self.max_len = max_len
        self.annotation_file_prefix = pathlib.Path(root).parent / "annotations"

        # Setup transforms
        self.transform = Compose(
            [
                Resize((self.image_size, self.image_size)),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if split == "test":
            self.annotation_file = (
                self.annotation_file_prefix / "coco_karpathy_test.json"
            )
        else:
            self.annotation_file = (
                self.annotation_file_prefix / "coco_karpathy_val.json"
            )

        with open(self.annotation_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.root, item["image"])
        image = Image.open(image_path).convert("RGB")
        caption = item["caption"][
            :5
        ]  # Here is a list of five captions, different from the training set

        # Apply image transforms
        image = self.transform(image)

        # Tokenize caption
        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        token_ids = toks["input_ids"]

        return image, token_ids
