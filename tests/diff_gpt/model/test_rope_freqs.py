import torch
from torch.testing import assert_close

from diff_gpt.model.rope import precompute_freqs_cis


def test_output_shape():
    r"""
    Verify dimensions: $T \times (D/2)$.
    $T = \text{end}, D = \text{dim}$
    Output $\in \mathbb{C}^{T \times D/2}$
    """
    dim, end, base = 64, 100, 10000
    cis = precompute_freqs_cis(dim, end, base)

    # Shape check
    assert cis.shape == (end, dim // 2)
    # Type check (Complex)
    assert torch.is_complex(cis)


def test_zero_timestep_identity():
    r"""
    Verify $t=0$ boundary condition.
    $m=0 \implies \text{angle} = 0 \cdot \theta_j = 0$
    $e^{i0} = 1 + 0i$
    """
    dim, end, base = 10, 5, 10000
    cis = precompute_freqs_cis(dim, end, base)

    # Row 0 ($t=0$)
    row_0 = cis[0]

    # \text{Re} \approx 1, \text{Im} \approx 0
    assert_close(row_0.real, torch.ones_like(row_0.real))
    assert_close(row_0.imag, torch.zeros_like(row_0.imag))


def test_explicit_formula_values():
    r"""
    Verify numerical correctness against scalar formula.
    $\theta_j = b^{-2j/d}, \quad b=10000$
    $\text{cis}_{m, j} = \cos(m \theta_j) + i \sin(m \theta_j)$
    """
    dim, end, base = 4, 2, 10000
    cis = precompute_freqs_cis(dim, end, theta=base)

    # Indices $j \in \{0, 1\}$ ($D/2 = 2$)

    # --- Check $j=0$ ---
    # $2j/d = 0 \implies \theta_0 = 1.0$
    # $m=1 \implies e^{i \cdot 1 \cdot 1} = \cos(1) + i\sin(1)$
    val_t1_j0 = cis[1, 0]
    expected_t1_j0 = torch.complex(
        torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))
    )
    assert_close(val_t1_j0, expected_t1_j0)

    # --- Check $j=1$ ---
    # $2j/d = 2/4 = 0.5$
    # $\theta_1 = 1 / 10000^{0.5} = 1/100 = 0.01$
    # $m=1 \implies e^{i \cdot 1 \cdot 0.01}$
    val_t1_j1 = cis[1, 1]
    expected_t1_j1 = torch.complex(
        torch.cos(torch.tensor(0.01)), torch.sin(torch.tensor(0.01))
    )
    assert_close(val_t1_j1, expected_t1_j1)


def test_frequency_decay():
    r"""
    Verify monotonicity of frequencies.
    $\theta_0 > \theta_1 > \dots > \theta_{d/2-1}$
    """
    dim, end, base = 128, 2, 10000
    cis = precompute_freqs_cis(dim, end, base)

    # Extract angles at $t=1$
    # $\angle = \text{atan2}(Im, Re)$
    angles = cis[1].angle()

    # $\theta_j$ is decreasing, so angles should be decreasing (assuming range $(0, \pi)$)
    # Check first few to avoid phase wrap-around issues
    assert angles[0] > angles[1]
    assert angles[1] > angles[2]


def test_magnitude_unity():
    r"""
    Verify unitarity.
    $|e^{ix}| = 1$
    """
    dim, end, base = 20, 10, 10000
    cis = precompute_freqs_cis(dim, end, base)

    # $|z| = \sqrt{Re^2 + Im^2}$
    magnitudes = cis.abs()

    # $|z| \approx 1$
    assert_close(magnitudes, torch.ones_like(magnitudes))
