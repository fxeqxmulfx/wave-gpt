import torch
from wave_decoder_encoder import encode, decode, get_diff_domain_of_definition


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
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp, vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert float(torch.max(torch.abs(decoded_inp - inp))) <= 0.0038102269172668457


def test_wave_encoder_sin_cos_different_lenght():
    vocab_size = 256
    idx = torch.arange(1_000_000)
    inp = torch.stack(
        (
            torch.sin(idx),
            torch.cos(idx),
        ),
        dim=1,
    )
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp, vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    start_2, scale_2, encoded_inp_2 = encode(
        inp[:16], vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp_2 >= 0) & (encoded_inp_2 < vocab_size))
    assert torch.all(start == start_2)
    assert torch.all(scale == scale_2)
    assert torch.all(encoded_inp[:8] == encoded_inp_2[:8])


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
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp, vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert float(torch.max(torch.abs(decoded_inp - inp))) == 0


def test_wave_encoder_decoder_const():
    vocab_size = 4
    inp = torch.ones(1_000_000, dtype=torch.float32).reshape(-1, 1)
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp, vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert float(torch.max(torch.abs(decoded_inp - inp))) == 0


def test_wave_encoder_decoder_small_const():
    vocab_size = 4
    inp = (torch.ones(1_000_000, dtype=torch.float32) * 0.01).reshape(-1, 1)
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp, vocab_size, diff_domain_of_definition=diff_domain_of_definition
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert float(torch.max(torch.abs(decoded_inp - inp))) == 0


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
    diff_domain_of_definition = get_diff_domain_of_definition(inp)
    start, scale, encoded_inp = encode(
        inp,
        vocab_size,
        diff_domain_of_definition=diff_domain_of_definition,
    )
    assert torch.all((encoded_inp >= 0) & (encoded_inp < vocab_size))
    decoded_inp = decode(start, scale, encoded_inp, vocab_size)
    assert float(torch.max(torch.abs(decoded_inp - inp))) <= 0.0038102269172668457
