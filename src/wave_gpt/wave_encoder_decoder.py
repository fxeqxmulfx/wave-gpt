import numpy as np


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    assert order_of_derivative >= 0
    assert inp.dtype == domain_of_definition.dtype == np.float64
    scale = np.ones(shape=domain_of_definition.shape, dtype=domain_of_definition.dtype)
    _filter = np.abs(domain_of_definition) > np.finfo(domain_of_definition.dtype).eps
    _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    diff = inp
    start = np.zeros((0, inp.shape[1]), dtype=inp.dtype)
    for _ in range(order_of_derivative):
        start = np.concat((start, np.expand_dims(diff[0], axis=0)), axis=0)
        diff = np.diff(diff, axis=0)
    scaled_diff = diff * scale
    trunced_scaled_diff = np.trunc(scaled_diff)
    residual = scaled_diff - trunced_scaled_diff
    residual = np.cumsum(residual, axis=0)
    residual = np.round(residual)
    residual = np.concat(
        (np.zeros(shape=(1, residual.shape[1]), dtype=residual.dtype), residual), axis=0
    )
    residual = np.diff(residual, axis=0)
    encoded_data = trunced_scaled_diff + vocab_size // 2 + residual
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
    assert start.dtype == scale.dtype == np.float64
    assert inp.dtype == np.int64
    inp = (inp - vocab_size // 2) / scale
    result = inp
    for i in range(order_of_derivative):
        _start = start[-i - 1]
        result = np.concat((np.expand_dims(_start, axis=0), result), axis=0)
        result = np.cumsum(result, axis=0)
    return result


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int,
) -> np.ndarray:
    assert order_of_derivative >= 0
    assert inp.dtype == np.float64
    diff = np.diff(inp, n=order_of_derivative, axis=0)
    result = np.max(np.abs(diff), axis=0)
    return result
