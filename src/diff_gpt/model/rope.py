import torch


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_shaped)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.flatten(3)
    result = x_out.type_as(x)
    return result


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
