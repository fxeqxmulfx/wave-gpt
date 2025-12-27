import torch
from torch import nn

from diff_gpt.model.swi_glu import SwiGLU


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, swiglu_alpha: float, swiglu_limit: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            SwiGLU(alpha=swiglu_alpha, limit=swiglu_limit),
            nn.Linear(2 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.net(x)
        return result
