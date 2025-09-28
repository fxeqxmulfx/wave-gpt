import math
import numpy as np
import torch


def _sum(a: float, b: float) -> float:
    return math.fsum((a, b))


_custom_sum = np.vectorize(_sum, signature="(),()->()")


def custom_sum(a: torch.Tensor, b: int | torch.Tensor) -> torch.Tensor:
    if isinstance(b, int):
        return torch.from_numpy(_custom_sum(a.numpy(), b))
    return torch.from_numpy(_custom_sum(a.numpy(), b.numpy()))


def _cumsum(inp: np.ndarray) -> np.ndarray:
    size = inp.size
    result = np.zeros(shape=size, dtype=inp.dtype)
    if size == 0:
        return result
    result[0] = inp[0]
    for i in range(1, size):
        result[i] = math.fsum((inp[i], result[i - 1]))
    return result


_custom_cumsum = np.vectorize(_cumsum, signature="(n)->(n)")


def custom_cumsum(inp: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(_custom_cumsum(inp.numpy()))


def _diff(inp: np.ndarray) -> np.ndarray:
    size = inp.size - 1
    result = np.zeros(shape=size, dtype=inp.dtype)
    for i in range(size):
        result[i] = math.fsum((-inp[i], inp[i + 1]))
    return result


_custom_diff = np.vectorize(_diff, signature="(n)->(m)")


def custom_diff(inp: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(_custom_diff(inp.numpy()))


def encode(
    inp: torch.Tensor,
    vocab_size: int,
    domain_of_definition: torch.Tensor,
    order_of_derivative: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    assert order_of_derivative >= 0
    assert inp.dtype == domain_of_definition.dtype == torch.float64
    scale = torch.ones(domain_of_definition.shape, dtype=domain_of_definition.dtype)
    _filter = domain_of_definition.abs() > torch.finfo(domain_of_definition.dtype).eps
    _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    diff = inp
    start = torch.zeros((0, inp.shape[1]), dtype=inp.dtype)
    for _ in range(order_of_derivative):
        start = torch.concat((start, diff[0].unsqueeze(0)))
        diff = custom_diff(diff.T).T
    scaled_diff = diff * scale
    trunced_scaled_diff = scaled_diff.trunc()
    residual = custom_sum(scaled_diff, -trunced_scaled_diff)
    residual = custom_cumsum(residual.T).T
    residual = residual.round()
    residual = torch.concat(
        (torch.zeros((1, residual.shape[1]), dtype=residual.dtype), residual)
    )
    residual = custom_diff(residual.T).T
    encoded_data = custom_sum(
        custom_sum(trunced_scaled_diff, vocab_size // 2), residual
    )
    result = start, scale, encoded_data.to(dtype=torch.int64)
    return result


def decode(
    start: torch.Tensor,
    scale: torch.Tensor,
    inp: torch.Tensor,
    vocab_size: int,
    order_of_derivative: int,
) -> torch.Tensor:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert order_of_derivative >= 0
    assert start.dtype == scale.dtype == torch.float64
    assert inp.dtype == torch.int64
    inp = custom_sum(inp, -vocab_size // 2) / scale
    result = inp
    for i in range(order_of_derivative):
        _start = start[-i - 1]
        result = torch.concat((_start.unsqueeze(0), result), dim=0)
        result = custom_cumsum(result.T).T
    return result


def get_domain_of_definition(
    inp: torch.Tensor,
    order_of_derivative: int,
) -> torch.Tensor:
    assert order_of_derivative >= 0
    assert inp.dtype == torch.float64
    diff = inp
    for _ in range(order_of_derivative):
        diff = custom_diff(diff.T).T
    result = diff.abs().max(dim=0).values
    return result
