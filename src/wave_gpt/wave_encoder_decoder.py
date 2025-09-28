import math
import numpy as np


def _sum(a: float, b: float) -> float:
    return math.fsum((a, b))


custom_sum = np.vectorize(_sum, signature="(),()->()")


def _cumsum(inp: np.ndarray) -> np.ndarray:
    size = inp.size
    result = np.zeros(shape=size, dtype=inp.dtype)
    if size == 0:
        return result
    result[0] = inp[0]
    for i in range(1, size):
        result[i] = math.fsum((inp[i], result[i - 1]))
    return result


custom_cumsum = np.vectorize(_cumsum, signature="(n)->(n)")


def _diff(inp: np.ndarray) -> np.ndarray:
    size = inp.size - 1
    result = np.zeros(shape=size, dtype=inp.dtype)
    for i in range(size):
        result[i] = math.fsum((-inp[i], inp[i + 1]))
    return result


custom_diff = np.vectorize(_diff, signature="(n)->(m)")


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    assert order_of_derivative >= 0
    assert inp.dtype == domain_of_definition.dtype == np.float128
    scale = np.ones(shape=domain_of_definition.shape, dtype=domain_of_definition.dtype)
    _filter = np.abs(domain_of_definition) > np.finfo(domain_of_definition.dtype).eps
    _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    diff = inp
    start = np.zeros((0, inp.shape[1]), dtype=inp.dtype)
    for _ in range(order_of_derivative):
        start = np.concat((start, np.expand_dims(diff[0], axis=0)))
        diff = custom_diff(diff.T).T
    scaled_diff = diff * scale
    trunced_scaled_diff = np.trunc(scaled_diff)
    residual = custom_sum(scaled_diff, -trunced_scaled_diff)
    residual = custom_cumsum(residual.T).T
    residual = np.round(residual)
    residual = np.concat(
        (np.zeros(shape=(1, residual.shape[1]), dtype=residual.dtype), residual)
    )
    residual = custom_diff(residual.T).T
    encoded_data = custom_sum(
        custom_sum(trunced_scaled_diff, vocab_size // 2), residual
    )
    result = start, scale, encoded_data.astype(np.int64)
    return result


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int,
) -> np.ndarray:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert order_of_derivative >= 0
    assert start.dtype == scale.dtype == np.float128
    assert inp.dtype == np.int64
    inp = custom_sum(inp, -vocab_size // 2) / scale
    result = inp
    for i in range(order_of_derivative):
        _start = start[-i - 1]
        result = np.concat((np.expand_dims(_start, axis=0), result))
        result = custom_cumsum(result.T).T
    return result


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int,
) -> np.ndarray:
    assert order_of_derivative >= 0
    assert inp.dtype == np.float128
    diff = inp
    for _ in range(order_of_derivative):
        diff = custom_diff(diff.T).T
    result = np.max(np.abs(diff), axis=0)
    return result
