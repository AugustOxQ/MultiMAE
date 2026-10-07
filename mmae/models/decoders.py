"""Query decoders: learned queries (one per output position) cross-attending to a memory sequence."""
from __future__ import annotations

import torch
from torch import nn


class QueryDecoder(nn.Module):
    """Predicts one output vector per query; works with a memory of any length.

    With `prefix_queries` > 0, that many extra learned queries sit ahead of the positional queries (no position
    embedding, never padded) and read out through their own head of size `prefix_out_dim`; forward then returns
    (main outputs, prefix outputs). H-b's emotion slot is one such query (spec 2026-10-07, section 5.1).
    """

    def __init__(
        self, num_queries: int, dim: int, out_dim: int, depth: int = 4, heads: int = 8, dropout: float = 0.1,
        prefix_queries: int = 0, prefix_out_dim: int = 0,
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
        self.prefix_queries = None
        if prefix_queries > 0:
            if prefix_out_dim <= 0:
                raise ValueError("prefix_queries needs prefix_out_dim > 0")
            self.prefix_queries = nn.Parameter(torch.empty(prefix_queries, dim))
            nn.init.normal_(self.prefix_queries, std=0.02)
            self.prefix_head = nn.Linear(dim, prefix_out_dim)

    @property
    def num_queries(self) -> int:
        return self.queries.shape[0]

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        query_padding: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """memory (B, L, dim); paddings are bool, True = padded. Returns (B, n, out_dim), plus
        (B, prefix_queries, prefix_out_dim) when the decoder has prefix queries."""
        n = self.num_queries if query_padding is None else query_padding.shape[1]
        if n > self.num_queries:
            raise ValueError(f"{n} queries requested but the decoder has {self.num_queries}")
        batch = memory.shape[0]
        target = (self.queries[:n] + self.pos_embed[:n]).unsqueeze(0).expand(batch, -1, -1)
        p = 0
        if self.prefix_queries is not None:
            p = self.prefix_queries.shape[0]
            target = torch.cat([self.prefix_queries.unsqueeze(0).expand(batch, -1, -1), target], dim=1)
            if query_padding is not None:
                query_padding = torch.cat([query_padding.new_zeros(batch, p), query_padding], dim=1)
        out = self.decoder(
            target, memory, tgt_key_padding_mask=query_padding, memory_key_padding_mask=memory_padding
        )
        if self.prefix_queries is None:
            return self.head(out)
        return self.head(out[:, p:]), self.prefix_head(out[:, :p])
