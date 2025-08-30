import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(
        torch.ones_like(freqs, dtype=torch.float32), freqs
    )  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(
        1, 2
    )
    return x_out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # RoPE
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        k = apply_rotary_emb(k, freqs_cis)
        q = apply_rotary_emb(q, freqs_cis)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class FeedFoward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.net(x)
        return result


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, rmsnorm_eps: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
        )
        self.ffwd = FeedFoward(n_embd=n_embd)
        self.norm_1 = RMSNorm(num_features=n_embd, eps=rmsnorm_eps)
        self.norm_2 = RMSNorm(num_features=n_embd, eps=rmsnorm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.norm_1(x), freqs_cis=freqs_cis)
        x = x + self.ffwd(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        rmsnorm_eps: float = 1e-5,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(n_embd=n_embd, n_head=n_head, rmsnorm_eps=rmsnorm_eps)
            for _ in range(n_layer)
        )
        self.norm = RMSNorm(num_features=n_embd, eps=rmsnorm_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(n_embd // n_head, block_size * 2, theta=rope_theta),
        )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B, T, C)
        freqs_cis = (
            self.freqs_cis[:T]  # pyright: ignore[reportIndexIssue]
        )  # (T, hs/2)
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)  # (B, T, C)
        x = self.norm(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        block_size = self.block_size
        device = idx.device
        B, T = idx.shape
        all_tokens = torch.full(
            (B, T + max_new_tokens),
            0,
            dtype=torch.int64,
            device=device,
        )
        all_tokens[:, :T] = idx[:, :T]
        for current_pos in range(T, T + max_new_tokens):
            start_slice = max(0, current_pos - block_size)
            idx_cond = all_tokens[:, start_slice:current_pos]
            logits, _ = self(idx_cond)
            logits = logits[:, -1]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            all_tokens[:, current_pos] = idx_next.squeeze()
        self.train()
        return all_tokens
