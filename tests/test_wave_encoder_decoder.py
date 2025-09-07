import torch
from wave_decoder_encoder import encode, decode


def test_wave_encoder_decoder_sin_cos():
    vocab_size = 256
    idx = torch.arange(1_000_000)
    inp = torch.stack(
        (
            torch.sin(idx),
            torch.cos(idx),
        ),
        dim=1,
    )
    start, scale, encoded_inp = encode(inp, vocab_size)
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert torch.max(torch.abs(decoded_inp - inp)) <= 0.0038102269172668457


def test_wave_encoder_decoder_lin():
    vocab_size = 4
    idx = torch.arange(1_000_000, dtype=torch.float32)
    inp = torch.stack(
        (
            idx,
            torch.flip(idx, (0,)),
        ),
        dim=1,
    )
    start, scale, encoded_inp = encode(inp, vocab_size)
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert torch.max(torch.abs(decoded_inp - inp)) == 0


def test_wave_encoder_decoder_const():
    vocab_size = 4
    inp = torch.ones(1_000_000, dtype=torch.float32)
    start, scale, encoded_inp = encode(inp, vocab_size)
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert torch.max(torch.abs(decoded_inp - inp)) == 0


def test_wave_encoder_decoder_small_const():
    vocab_size = 4
    inp = torch.ones(1_000_000, dtype=torch.float32) * 0.01
    start, scale, encoded_inp = encode(inp, vocab_size)
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert torch.max(torch.abs(decoded_inp - inp)) == 0


def test_wave_encoder_decoder_mix():
    vocab_size = 256
    const = torch.ones(1_000_000)
    small_const = torch.ones(1_000_000) * 0.01
    idx = torch.arange(1_000_000)
    inp = torch.stack(
        (
            torch.sin(idx),
            torch.cos(idx),
            idx,
            torch.flip(idx, (0,)),
            const,
            small_const,
        ),
        dim=1,
    )
    start, scale, encoded_inp = encode(inp, vocab_size)
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert torch.max(torch.abs(decoded_inp - inp)) <= 0.0038102269172668457
