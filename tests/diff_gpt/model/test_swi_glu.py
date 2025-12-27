import torch
from torch.testing import assert_close

from diff_gpt.model.swi_glu import SwiGLU


def test_shape_reduction():
    r"""
    Verify dimensionality reduction.
    x \in \mathbb{R}^{B \times D} \implies y \in \mathbb{R}^{B \times D/2}
    """
    # D_{in} = 10, D_{out} = 5
    x = torch.randn(2, 10)
    model = SwiGLU(alpha=1.0, limit=10.0)

    y = model(x)

    assert y.shape == (2, 5)


def test_explicit_math_no_clamp():
    r"""
    Verify calculation without clamping boundaries.
    x \in \mathbb{R}^{1 \times 2} \to y \in \mathbb{R}^{1 \times 1}
    """
    alpha = 2.0
    limit = 100.0
    model = SwiGLU(alpha=alpha, limit=limit)

    # x = [[x_g, x_l]]
    x = torch.tensor([[1.0, 2.0]])

    # y_{scalar} = (1.0 \cdot \sigma(2.0 \cdot 1.0)) \cdot (2.0 + 1)
    scalar_val = 1.0 * torch.sigmoid(torch.tensor(2.0)) * 3.0

    # y_{tensor} \in \mathbb{R}^{1 \times 1}
    expected = scalar_val.view(1, 1)

    out = model(x)
    assert_close(out, expected)


def test_clamping_logic_upper():
    r"""
    Verify upper bound clamping ($x > L$).
    x \in \mathbb{R}^{1 \times 2}
    """
    limit = 5.0
    model = SwiGLU(alpha=1.0, limit=limit)

    # x_g = 10, x_l = 10
    x = torch.tensor([[10.0, 10.0]])

    out = model(x)

    # \hat{x}_g = 5, \hat{x}_l = 5
    # y = (5 \cdot \sigma(1 \cdot 5)) \cdot (5 + 1)
    scalar_val = (5.0 * torch.sigmoid(torch.tensor(5.0))) * 6.0
    expected = scalar_val.view(1, 1)

    assert_close(out, expected)


def test_clamping_logic_linear_lower():
    r"""
    Verify lower bound clamping for linear part ($x_l < -L$).
    """
    limit = 5.0
    model = SwiGLU(alpha=1.0, limit=limit)

    # x_g = 1, x_l = -10
    x = torch.tensor([[1.0, -10.0]])

    out = model(x)

    # \hat{x}_l = -5
    # y = \text{Swish}(1) \cdot (-5 + 1)
    swish_part = 1.0 * torch.sigmoid(torch.tensor(1.0))
    scalar_val = swish_part * -4.0
    expected = scalar_val.view(1, 1)

    assert_close(out, expected)


def test_clamping_glu_no_lower_bound():
    r"""
    Verify GLU part has NO lower bound clamping ($min=None$).
    """
    limit = 5.0
    model = SwiGLU(alpha=1.0, limit=limit)

    # x_g = -10, x_l = 0
    x = torch.tensor([[-10.0, 0.0]])

    out = model(x)

    # \hat{x}_g = -10 (\text{not } -5)
    # y = (-10 \cdot \sigma(-10)) \cdot (0 + 1)
    scalar_val = (-10.0 * torch.sigmoid(torch.tensor(-10.0))) * 1.0
    expected = scalar_val.view(1, 1)

    assert_close(out, expected)


def test_gradients():
    r"""
    Verify \nabla_x L \neq \emptyset.
    """
    x = torch.randn(1, 4, requires_grad=True)
    model = SwiGLU(alpha=1.0, limit=10.0)

    y = model(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)
