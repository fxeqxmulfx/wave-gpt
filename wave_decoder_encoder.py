import torch


def encode(
    inp: torch.Tensor, vocab_size: int, diff_domain_of_definition: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert diff_domain_of_definition.shape[0] == inp.shape[1]
    scale = torch.ones(
        diff_domain_of_definition.shape, dtype=diff_domain_of_definition.dtype
    )
    _filter = (
        torch.abs(diff_domain_of_definition)
        > torch.finfo(diff_domain_of_definition.dtype).eps
    )
    _scale = (vocab_size - 2) / diff_domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    diff = torch.diff(inp, dim=0)
    scaled_diff = diff * scale
    residual = scaled_diff - torch.trunc(scaled_diff)
    residual = torch.diff(
        torch.round(torch.cumsum(residual, dim=0)),
        dim=0,
        prepend=torch.zeros(residual[:1].shape, dtype=residual.dtype),
    )
    encoded_data = torch.trunc(scaled_diff) + vocab_size // 2 + residual
    start = inp[:1]
    result = start, scale, encoded_data
    return result


def decode(
    start: torch.Tensor, scale: torch.Tensor, inp: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    diff = (inp - vocab_size // 2) / scale
    result = torch.concat((start, diff), dim=0).cumsum(dim=0)
    return result


def get_diff_domain_of_definition(inp: torch.Tensor) -> torch.Tensor:
    diff = torch.diff(inp, dim=0)
    result = torch.max(torch.abs(diff), dim=0).values
    return result
