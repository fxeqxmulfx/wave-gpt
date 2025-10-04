import numpy as np

from wave_gpt.wave_encoder_decoder import differentiate, integrate


def test_zero_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 0
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) == 0.0


def test_first_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 1
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= 4.9460435747050724e-14


def test_second_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 2
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= 2.6157809251969866e-10


def test_third_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 3
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= 0.00019781837839571992


def test_fourth_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 4
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= 57.53735976698562


def test_fifth_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 5
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= 22888174.67337878
