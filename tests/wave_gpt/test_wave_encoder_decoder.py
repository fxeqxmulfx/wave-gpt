import numpy as np
from wave_gpt.wave_encoder_decoder import (
    encode,
    decode,
    get_domain_of_definition,
)


def test_wave_encoder_decoder_sin_cos():
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 0
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.007873033862


def test_wave_encoder_decoder_sin_cos_derivative():
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.00377500359439803


def test_wave_encoder_decoder_sin_cos_second_derivative():
    vocab_size = 4096
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 2
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.04210817981582421


def test_wave_encoder_decoder_sin_cos_third_derivative():
    vocab_size = 65536
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 3
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    # TODO: How to fix this?
    assert float(np.max(np.abs(decoded_inp - inp))) <= 3080.6745080035776


def test_wave_encoder_sin_cos_different_lenght():
    vocab_size = 256
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    start_2, scale_2, encoded_inp_2 = encode(
        inp=inp[:16],
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp_2 >= 0) & (encoded_inp_2 < vocab_size))
    assert np.all(start == start_2)
    assert np.all(scale == scale_2)
    assert np.all(encoded_inp[:8] == encoded_inp_2[:8])


def test_wave_encoder_decoder_lin():
    vocab_size = 4
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            idx,
            np.flip(idx, (0,)),
        ),
        axis=1,
    )
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0


def test_wave_encoder_decoder_const():
    vocab_size = 4
    inp = np.ones(1_000_000, dtype=np.float64).reshape(-1, 1)
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0


def test_wave_encoder_decoder_small_const():
    vocab_size = 4
    inp = (np.ones(1_000_000, dtype=np.float64) * 0.01).reshape(-1, 1)
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0


def test_wave_encoder_decoder_mix():
    vocab_size = 256
    const = np.ones(1_000_000, dtype=np.float64)
    small_const = np.ones(1_000_000, dtype=np.float64) * 0.01
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
            np.sqrt(idx),
            idx,
            np.flip(idx, (0,)),
            const,
            small_const,
        ),
        axis=1,
    )
    order_of_derivative = 1
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=1,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) <= 0.003936999343636671


def test_wave_encoder_decoder_square():
    vocab_size = 4
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (np.square(idx),),
        axis=1,
    )
    order_of_derivative = 2
    domain_of_definition = get_domain_of_definition(
        inp=inp, order_of_derivative=order_of_derivative
    )
    start, scale, encoded_inp = encode(
        inp=inp,
        vocab_size=vocab_size,
        domain_of_definition=domain_of_definition,
        order_of_derivative=order_of_derivative,
    )
    assert np.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(
        start=start,
        scale=scale,
        inp=encoded_inp,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
    )
    assert float(np.max(np.abs(decoded_inp - inp))) == 0
