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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unprepared", action="store_true")
    args = parser.parse_args()
    {"contrastive": contrastive}[args.command](args.out)


if __name__ == "__main__":
    main()
