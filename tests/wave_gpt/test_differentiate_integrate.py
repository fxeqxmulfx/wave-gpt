from decimal import Decimal

import numpy as np
import pytest

from wave_gpt.wave_encoder_decoder import differentiate, integrate, np_to_decimal


def test_zero_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 0
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) == Decimal("0")


def test_first_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("7.802992583885788917541503906E-26")


def test_second_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 2
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_third_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 3
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_fourth_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 4
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_fifth_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("7.802992583885788917541503906E-26")


def test_sixth_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 6
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_seventh_order_differentiate_integrate_sin_cos():
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 7
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_overflow_differentiate_integrate_square_x():
    idx = np.arange(7, dtype=np.float64)
    inp = np.stack(
        (np.square(idx),),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 7
    start, diff = differentiate(inp=inp, order_of_derivative=order_of_derivative)
    result = integrate(start=start, diff=diff, order_of_derivative=order_of_derivative)
    assert np.max(np.abs(result - inp)) == Decimal("0")
    order_of_derivative = 8
    with pytest.raises((AssertionError,)):
        differentiate(inp=inp, order_of_derivative=order_of_derivative)
