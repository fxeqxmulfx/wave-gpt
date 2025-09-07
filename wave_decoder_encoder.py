import torch


def encode(
    inp: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    diff = torch.diff(inp, dim=0)
    max_diff, _ = torch.max(torch.abs(diff), dim=0)
    scale = torch.ones(max_diff.shape, dtype=max_diff.dtype)
    _filter = torch.abs(max_diff) > torch.finfo(max_diff.dtype).eps
    _scale = (vocab_size - 2) / max_diff[_filter] / 2
    scale[_filter] = _scale
    scaled_diff = diff * scale
    residual = scaled_diff - torch.trunc(scaled_diff)
    residual = torch.diff(
        torch.trunc(torch.round(torch.cumsum(residual, dim=0))),
        dim=0,
        prepend=torch.zeros(residual[:1].shape, dtype=residual.dtype),
    )
    result = torch.trunc(scaled_diff) + vocab_size // 2 + residual
    return inp[:1], scale, result


def decode(
    start: torch.Tensor, scale: torch.Tensor, inp: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    diff = (inp - vocab_size // 2) / scale
    result = torch.concat((start, diff), dim=0).cumsum(dim=0)
    return result
