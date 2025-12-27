import torch
from torch.testing import assert_close

from diff_gpt.model.block import Block
from diff_gpt.model.kv_cache import KVCache


def get_freqs_cis(seq_len: int, head_dim: int) -> torch.Tensor:
    dims = head_dim // 2
    theta = torch.randn(1, seq_len, 1, dims)
    return torch.polar(torch.ones_like(theta), theta)


def test_block_forward_and_gradients():
    r"""
    $x \in \mathbb{R}^{B \times T \times C} \xrightarrow{\text{Block}} y \in \mathbb{R}^{B \times T \times C}$

    Check Residual Property:
    $y = x + \text{SA}(\text{RMS}(x)) + \text{FFN}(\text{RMS}(x'))$

    Check Gradient Flow:
    $\nabla_{\theta_{SA}} \mathcal{L} \neq \mathbf{0}, \quad \nabla_{\theta_{FFN}} \mathcal{L} \neq \mathbf{0}$
    """
    B, T, C, H = 2, 8, 32, 4
    head_dim = C // H

    model = Block(n_embd=C, n_head=H, swiglu_alpha=1.0, swiglu_limit=10.0, layer_idx=0)

    x = torch.randn(B, T, C, requires_grad=True)
    freqs = get_freqs_cis(T, head_dim)

    # Forward
    y = model(x, freqs_cis=freqs, kv_cache=None)

    # $\mathbb{R}^{B \times T \times C}$ preserved
    assert y.shape == (B, T, C)

    # Backward
    loss = y.sum()
    loss.backward()

    # $\exists \nabla_x$
    assert x.grad is not None

    # Check Sub-modules parameters
    # $\forall w \in \text{SA}, \|\nabla w\| > 0$
    for param in model.sa.parameters():
        assert param.grad is not None
        assert torch.norm(param.grad) > 0

    # $\forall w \in \text{FFN}, \|\nabla w\| > 0$
    for param in model.ffwd.parameters():
        assert param.grad is not None
        assert torch.norm(param.grad) > 0


def test_block_causality_preservation():
    r"""
    Verify Autoregressive Property via Block.
    $y_t = f(x_{0:t})$
    $\frac{\partial y_t}{\partial x_{t+k}} = 0, \quad \forall k > 0$
    """
    B, T, C, H = 1, 5, 16, 2
    head_dim = C // H

    model = Block(n_embd=C, n_head=H, swiglu_alpha=1.0, swiglu_limit=10.0, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)
    freqs = get_freqs_cis(T, head_dim)

    # 1. Forward Original
    with torch.no_grad():
        y_orig = model(x, freqs, kv_cache=None)

    # 2. Perturb $x_T$
    x_pert = x.clone()
    x_pert[:, -1, :] += 5.0

    with torch.no_grad():
        y_new = model(x_pert, freqs, kv_cache=None)

    # $y_{0:T-1} \approx y'_{0:T-1}$
    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])

    # $y_T \neq y'_T$
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_block_kv_cache_step_consistency():
    r"""
    Verify $KV$ Cache integration through Residual Stream.

    Let $\Phi(x_{0:T})$ be full forward.
    Let $\phi(x_t, s_{t-1}) \to (y_t, s_t)$ be step forward.

    Verify: $\Phi(x)_{t} \equiv \phi(x_t, s_{t-1})$
    """
    B, T, C, H = 1, 6, 16, 2
    head_dim = C // H

    model = Block(n_embd=C, n_head=H, swiglu_alpha=1.0, swiglu_limit=10.0, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)
    freqs_full = get_freqs_cis(T, head_dim)

    # 1. Parallel (Batch) Forward
    with torch.no_grad():
        y_batch = model(x, freqs_full, kv_cache=None)

    # 2. Serial (Step) Forward
    # Assuming KVCache(..., num_layers=1) for single block test
    kv = KVCache(batch_size=B, num_heads=H, seq_len=20, head_dim=head_dim, num_layers=1)

    outputs = []
    for t in range(T):
        x_step = x[:, t : t + 1, :]
        freqs_step = freqs_full[:, t : t + 1, :, :]

        with torch.no_grad():
            y_step = model(x_step, freqs_step, kv_cache=kv)
            outputs.append(y_step)

    y_serial = torch.cat(outputs, dim=1)

    # $|y_{batch} - y_{serial}| < \epsilon$
    assert_close(y_batch, y_serial, atol=1e-5, rtol=1e-5)
