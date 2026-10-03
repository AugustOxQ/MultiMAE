"""Content words for content-word masking (M2b): a token is a content token when its word piece is alphabetic
and not an English stop word. Stop words are NLTK's English list (179 words)."""
from __future__ import annotations

import torch

STOP_WORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his
himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom
this that that'll these those am is are was were be been being have has had having do does did doing a an the
and but if or because as until while of at by for with about against between into through during before after
above below to from up down in out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too very s t can will just don
don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn
hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())


def is_content_word(word: str) -> bool:
    return word.isalpha() and word.lower() not in STOP_WORDS


def content_token_table(tokenizer) -> torch.Tensor:
    """(vocab,) bool: True for the token ids whose word piece (CLIP BPE, "</w>" marks a word end) is a content word."""
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return torch.tensor([is_content_word(p.replace("</w>", "")) for p in pieces], dtype=torch.bool)
