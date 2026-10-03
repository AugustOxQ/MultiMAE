"""Batch collation: stack images and tokenize captions per batch with the backbone's tokenizer."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from mmae.data.stopwords import content_token_table


class Collator:
    def __init__(self, tokenizer_name: str, max_text_len: int, content_words: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = int(max_text_len)
        self.content_table = content_token_table(self.tokenizer) if content_words else None

    def tokenize(self, captions: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            captions,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        out = {key: enc[key] for key in ("input_ids", "attention_mask", "special_tokens_mask")}
        if self.content_table is not None:
            out["content_tokens_mask"] = (
                self.content_table[out["input_ids"]] & out["attention_mask"].bool() & ~out["special_tokens_mask"].bool()
            )
        return out

    def pairs(self, batch: list[tuple[torch.Tensor, str]]) -> dict[str, torch.Tensor]:
        images, captions = zip(*batch)
        return {"pixel_values": torch.stack(images), **self.tokenize(list(captions))}

    def retrieval(self, batch: list[tuple[torch.Tensor, list[str]]]) -> dict[str, torch.Tensor]:
        images, captions = zip(*batch)
        per_image = len(captions[0])
        flat = self.tokenize([c for group in captions for c in group])
        return {
            "pixel_values": torch.stack(images),
            **{key: value.view(len(images), per_image, -1) for key, value in flat.items()},
        }
