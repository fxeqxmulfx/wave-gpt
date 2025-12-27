import torch
from torch import nn

from diff_gpt.model.mha import MultiHeadAttention
from diff_gpt.model.feed_forward import FeedForward
from diff_gpt.model.rms_norm import rms_norm
from diff_gpt.model.kv_cache import KVCache


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        swiglu_alpha: float,
        swiglu_limit: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
        )
        self.ffwd = FeedForward(
            n_embd=n_embd, swiglu_alpha=swiglu_alpha, swiglu_limit=swiglu_limit
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_cache: KVCache
    ) -> torch.Tensor:
        x = x + self.sa(rms_norm(x), freqs_cis=freqs_cis, kv_cache=kv_cache)
        x = x + self.ffwd(rms_norm(x))
        return x
