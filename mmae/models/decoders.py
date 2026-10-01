"""Query decoders: learned queries (one per output position) cross-attending to a memory sequence."""
from __future__ import annotations

import torch
from torch import nn


class QueryDecoder(nn.Module):
    """Predicts one output vector per query; works with a memory of any length."""

    def __init__(
        self, num_queries: int, dim: int, out_dim: int, depth: int = 4, heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        self.pos_embed = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerDecoderLayer(
            dim, heads, 4 * dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(dim))
        self.head = nn.Linear(dim, out_dim)

    @property
    def num_queries(self) -> int:
        return self.queries.shape[0]

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        query_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """memory (B, L, dim); paddings are bool, True = padded. Returns (B, n, out_dim)."""
        n = self.num_queries if query_padding is None else query_padding.shape[1]
        if n > self.num_queries:
            raise ValueError(f"{n} queries requested but the decoder has {self.num_queries}")
        target = (self.queries[:n] + self.pos_embed[:n]).unsqueeze(0).expand(memory.shape[0], -1, -1)
        out = self.decoder(
            target, memory, tgt_key_padding_mask=query_padding, memory_key_padding_mask=memory_padding
        )
        return self.head(out)
