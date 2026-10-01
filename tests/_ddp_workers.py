"""Multi-process workers launched by tests through torchrun (CPU, gloo). Not collected by pytest."""
import argparse
from pathlib import Path

import torch
import torch.distributed as dist


def contrastive(out: Path) -> None:
    from mmae.losses import contrastive_loss

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    g = torch.Generator().manual_seed(0)
    image = torch.nn.functional.normalize(torch.randn(8, 6, generator=g), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(8, 6, generator=g), dim=-1)
    local = slice(rank * 4, (rank + 1) * 4)
    img = image[local].clone().requires_grad_(True)
    txt = text[local].clone().requires_grad_(True)
    loss = contrastive_loss(img, txt, torch.tensor(2.0), gather=True)
    loss.backward()
    torch.save({"loss": loss.detach(), "img_grad": img.grad, "txt_grad": txt.grad}, out / f"rank{rank}.pt")
    dist.destroy_process_group()


class DummyEmbedder(torch.nn.Module):
    """Deterministic stand-in for MultiMAE's embed_image / embed_text."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.image = torch.nn.Linear(12, 8)
        self.token = torch.nn.Embedding(100, 8)

    def embed_image(self, pixel_values):
        return torch.nn.functional.normalize(self.image(pixel_values), dim=-1)

    def embed_text(self, input_ids, attention_mask):
        return torch.nn.functional.normalize(self.token(input_ids).mean(dim=1), dim=-1)


def retrieval_dataset(n: int = 43):
    g = torch.Generator().manual_seed(1)
    images = torch.randn(n, 12, generator=g)
    ids = torch.randint(0, 100, (n, 5, 6), generator=g)
    return [
        {"pixel_values": images[i], "input_ids": ids[i], "attention_mask": torch.ones(5, 6, dtype=torch.long)}
        for i in range(n)
    ]


def retrieval(out: Path, unprepared: bool) -> None:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from mmae.engine.retrieval import evaluate_retrieval, encode_retrieval_set

    accelerator = Accelerator(cpu=True)
    loader = DataLoader(retrieval_dataset(), batch_size=8, collate_fn=torch.utils.data.default_collate)
    model = DummyEmbedder()
    if unprepared:
        model = accelerator.prepare(model)
        try:
            evaluate_retrieval(model, loader, accelerator)
        except ValueError:
            (out / f"raised{accelerator.process_index}").touch()
        return
    model, loader = accelerator.prepare(model, loader)
    images, captions = encode_retrieval_set(model, loader, accelerator)
    metrics = evaluate_retrieval(model, loader, accelerator)
    if accelerator.is_main_process:
        torch.save({"images": images, "captions": captions, "metrics": metrics}, out / "retrieval.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unprepared", action="store_true")
    args = parser.parse_args()
    {"contrastive": lambda: contrastive(args.out), "retrieval": lambda: retrieval(args.out, args.unprepared)}[args.command]()


if __name__ == "__main__":
    main()
