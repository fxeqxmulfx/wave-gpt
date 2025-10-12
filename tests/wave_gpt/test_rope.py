import torch

from wave_gpt.model import MultiHeadAttention


def test_apply_rope():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    freqs_cis = (
        torch.tensor([[0 + 1j, 1 + 0j]], dtype=torch.complex64)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    actual_output = MultiHeadAttention.apply_rotary_emb(x, freqs_cis)
    expected_output = torch.tensor([[[[-2.0, 1.0, 3.0, 4.0]]]])
    assert torch.all(actual_output == expected_output)
