from decimal import Decimal

import numpy as np

ufunc_decimal = np.frompyfunc(Decimal, 1, 1)


def np_to_decimal(inp: np.ndarray) -> np.ndarray:
    result = ufunc_decimal(inp)
    if isinstance(result, Decimal):
        result = np.array(result)
    return result


def np_is_decimal(inp: np.ndarray) -> bool:
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 0 or inp.shape[0] == 0:
        return inp.dtype == "object"
    if len_inp_shape == 1:
        return isinstance(inp[0], Decimal)
    return isinstance(inp[0, 0], Decimal)


ufunc_round = np.frompyfunc(round, 1, 1)


def differentiate(
    inp: np.ndarray, order_of_derivative: int
) -> tuple[np.ndarray, np.ndarray]:
    assert order_of_derivative >= 0
    assert order_of_derivative <= inp.shape[0]
    assert np_is_decimal(inp)
    diff = inp
    start = np.zeros((0, inp.shape[1]), dtype=inp.dtype)
    for _ in range(order_of_derivative):
        start = np.concat((start, np.expand_dims(diff[0], axis=0)), axis=0)
        diff = np.diff(diff, axis=0)
    assert np_is_decimal(start)
    assert np_is_decimal(diff)
    result = start, diff
    return result


def integrate(
    start: np.ndarray, diff: np.ndarray, order_of_derivative: int
) -> np.ndarray:
    assert order_of_derivative == start.shape[0]
    assert np_is_decimal(start)
    assert np_is_decimal(diff)
    result = diff
    for i in range(order_of_derivative):
        _start = start[-i - 1]
        result = np.concat((np.expand_dims(_start, axis=0), result), axis=0)
        result = np.cumsum(result, axis=0)
    assert np_is_decimal(result)
    return result


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    assert order_of_derivative >= 0
    assert np_is_decimal(inp)
    assert np_is_decimal(domain_of_definition)
    scale = np_to_decimal(np.ones(shape=domain_of_definition.shape, dtype=np.float64))
    _filter = np.abs(domain_of_definition) > np.finfo(np.float64).eps
    _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    start, diff = differentiate(inp, order_of_derivative)
    scaled_diff = diff * scale
    trunced_scaled_diff = np.trunc(scaled_diff)
    residual = scaled_diff - trunced_scaled_diff
    residual = np.cumsum(residual, axis=0)
    residual = ufunc_round(residual)
    residual = np.concat(
        (
            np.zeros(shape=(1, residual.shape[1]), dtype=np.int64),
            residual,
        ),
        axis=0,
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
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    assert order_of_derivative >= 0
    assert np_is_decimal(start)
    assert np_is_decimal(scale)
    assert inp.dtype == np.int64
    inp = (inp - vocab_size // 2) / scale
    result = integrate(
        start=start,
        diff=inp,
        order_of_derivative=order_of_derivative,
    )
    return result


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int,
) -> np.ndarray:
    assert order_of_derivative >= 0
    assert np_is_decimal(inp)
    diff = np.diff(inp, n=order_of_derivative, axis=0)
    result = np.max(np.abs(diff), axis=0)
    return result
