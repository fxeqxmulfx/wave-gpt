import pytest
import torch
from torch.testing import assert_close

from diff_gpt.model.rms_norm import rms_norm


def test_explicit_formula_verification():
    r"""
    Verify against explicit definition.
    y_i = \frac{x_i}{\text{RMS}(x)}
    \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{j=1}^{n} x_j^2 + \epsilon}
    """
    B, D = 2, 5
    x = torch.randn(B, D)

    # Default epsilon in F.rms_norm is usually 1e-6
    epsilon = 1e-6

    # \mu_{sq} = \mathbb{E}[x^2]
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)

    # rms = \sqrt{\mu_{sq} + \epsilon}
    rms = torch.sqrt(mean_sq + epsilon)

    # y_{ref} = x \odot rms^{-1}
    expected = x / rms

    out = rms_norm(x)

    # y \approx y_{ref}
    assert_close(out, expected)


def test_scaling_invariance():
    r"""
    Verify scale invariance property.
    \text{RMSNorm}(\alpha \cdot x) \approx \text{RMSNorm}(x) \cdot \text{sign}(\alpha)
    \text{If } \epsilon \to 0, \frac{\alpha x}{\sqrt{(\alpha x)^2}} = \frac{\alpha x}{|\alpha| \sqrt{x^2}} = \text{sign}(\alpha) \frac{x}{\text{RMS}(x)}
    """
    x = torch.randn(2, 10)
    alpha = 5.0

    # y_1 = f(x)
    y1 = rms_norm(x)

    # y_2 = f(\alpha \cdot x)
    y2 = rms_norm(x * alpha)

    # y_1 \approx y_2 (Assuming \mu_{sq} \gg \epsilon)
    assert_close(y1, y2, atol=1e-5, rtol=1e-4)


def test_output_magnitude():
    r"""
    Verify output RMS is approximately 1.
    \text{RMS}(y) = \sqrt{\frac{\mu_{sq}}{\mu_{sq} + \epsilon}} \approx 1
    """
    x = torch.randn(10, 100) * 10  # Scale up to negate epsilon effect

    y = rms_norm(x)

    # \sqrt{\mathbb{E}[y^2]}
    y_rms = torch.sqrt(y.pow(2).mean(dim=-1))

    # \forall i: \text{RMS}(y_i) \approx 1
    target = torch.ones_like(y_rms)
    assert_close(y_rms, target, atol=1e-3, rtol=1e-3)


def test_gradients_flow():
    r"""
    Verify differentiability.
    \exists \nabla_x L
    """
    x = torch.randn(2, 4, requires_grad=True)

    y = rms_norm(x)
    loss = y.sum()
    loss.backward()

    # \frac{\partial L}{\partial x} \neq \emptyset
    assert x.grad is not None
    assert torch.norm(x.grad) > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dtype_preservation(dtype):
    r"""
    Verify Input Dtype == Output Dtype.
    x \in \mathbb{D} \implies y \in \mathbb{D}
    """
    # Skip bfloat16 on CPU if not supported
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_bf16_supported() or torch.ops.mkldnn._is_mkldnn_bf16_supported()
    ):
        return

    x = torch.randn(2, 4).to(dtype)
    y = rms_norm(x)

    assert y.dtype == dtype
