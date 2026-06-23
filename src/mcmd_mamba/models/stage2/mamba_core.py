"""
Sequence operator core for MD-SSM scan paths.
Implements Eq. (4)–(6): continuous SSM, ZOH discretization (Eq. 5), discrete recurrence (Eq. 6).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


# ---------- ZOH discretization (Eq. 5) ----------
def _zoh_discretize(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-order hold: Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I)(ΔB).
    A: (E, N, N), B: (E, N), delta: (E,) -> A_bar (E, N, N), B_bar (E, N).
    """
    E, N, _ = A.shape
    device, dtype = A.device, A.dtype
    deltaA = delta.view(E, 1, 1) * A
    A_bar = torch.linalg.matrix_exp(deltaA)
    deltaB = delta.view(E, 1) * B
    expm_minus_I = A_bar - torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    rhs = torch.bmm(expm_minus_I, deltaB.unsqueeze(-1))
    deltaA_safe = deltaA + 1e-5 * torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
    B_bar = torch.linalg.solve(deltaA_safe, rhs).squeeze(-1)
    return A_bar, B_bar


# ---------- SSM recurrence (Eq. 6) ----------
def _ssm_recurrence(
    x: torch.Tensor,
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete recurrence: h_t = Ā h_{t-1} + B̄ x_t, y_t = C h_t + D x_t.
    x: (B, L, E), A_bar (E, N, N), B_bar (E, N), C (E, N), D (E).
    Returns y (B, L, E). Causal, order-sensitive.
    """
    B, L, E = x.shape
    N = A_bar.shape[1]
    device, dtype = x.device, x.dtype
    h = torch.zeros(B, E, N, device=device, dtype=dtype)
    ys = []
    for t in range(L):
        x_t = x[:, t, :]  # (B, E)
        # h_new = A_bar @ h + B_bar * x_t
        h = torch.einsum("enm,ben->bem", A_bar, h) + x_t.unsqueeze(-1) * B_bar.unsqueeze(0)
        # y_t = C @ h + D * x_t
        y_t = torch.einsum("en,ben->be", C, h) + D.unsqueeze(0) * x_t
        ys.append(y_t)
    return torch.stack(ys, dim=1)


# ---------- Interface ----------
class SequenceCore(nn.Module):
    """
    Strict interface for any scan-path core operator (doc.md).
    Contract: (B, L, D) → (B, L, D); preserve dtype/device; order-sensitive.
    """

    def __init__(self, d_model: int, **kwargs: Any) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)."""
        raise NotImplementedError


class DummyCore(SequenceCore):
    """Lightweight placeholder: single linear for scan/reorder/merge validation."""

    def __init__(self, d_model: int, dropout: float = 0.0, **kwargs: Any) -> None:
        super().__init__(d_model, **kwargs)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.proj(x))
        return y.to(dtype=x.dtype, device=x.device)


class MambaSSMCore(SequenceCore):
    """
    Real SSM core: Eq. (4)–(6). ZOH discretization + discrete recurrence.
    Order-sensitive, causal. (B, L, D) → (B, L, D).
    Optional: expand to inner dim E = d_model * expand for capacity.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(d_model, **kwargs)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.E = E = d_model * expand
        self.N = d_state

        self.in_proj = nn.Linear(d_model, E)
        # Continuous A: (E, N, N) stable (e.g. -I per channel)
        A_init = -torch.eye(d_state).unsqueeze(0).expand(E, -1, -1).clone()
        self.A = nn.Parameter(A_init)
        self.B = nn.Parameter(torch.randn(E, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(E, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(E) * 0.01)
        self.log_delta = nn.Parameter(torch.zeros(E))

        self.out_proj = nn.Linear(E, d_model)

    def _get_A_bar_B_bar(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ZOH: Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA)-I)(ΔB)."""
        delta = torch.nn.functional.softplus(self.log_delta) + 1e-4
        return _zoh_discretize(self.A, self.B, delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) → (B, L, D). Causal SSM recurrence in scan order."""
        B, L, D = x.shape
        x_in = self.in_proj(x)  # (B, L, E)
        A_bar, B_bar = self._get_A_bar_B_bar()
        y_in = _ssm_recurrence(x_in, A_bar, B_bar, self.C, self.D)  # (B, L, E)
        y = self.out_proj(y_in)
        return y.to(dtype=x.dtype, device=x.device)


def build_core(kind: str, d_model: int, **kwargs: Any) -> SequenceCore:
    """Factory: 'dummy' → DummyCore; 'mamba' → MambaSSMCore (real SSM)."""
    if kind == "dummy":
        return DummyCore(d_model=d_model, **kwargs)
    if kind == "mamba":
        return MambaSSMCore(d_model=d_model, **kwargs)
    raise ValueError(f"Unknown core kind: {kind!r}. Use 'dummy' or 'mamba'.")


def assert_core_io(x: torch.Tensor, y: torch.Tensor, d_model: int) -> None:
    """Validation: shape, ndim, last dim d_model, finite."""
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape == y.shape
    assert x.shape[2] == d_model
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
