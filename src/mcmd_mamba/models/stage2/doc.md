# Stage 2 — Multi-Direction Mamba (Paper §2.2)

Stage 2 implements the paper’s **MD-Mamba** module, which captures long-range dependencies over the **concatenated multi-scale token sequence** produced by Stage 1.

**Stage 1 output expected by Stage 2**
- `Tconcat`: shape `(B, 3L, D)`
- `L = Htok * Wtok` where `(Htok, Wtok)` is the token grid size used in Stage 1 tokenization.

Stage 2 is organized into three architectural layers (and three files):

1) `mamba_core.py` — the **sequence operator** used inside each scan path (Ψ).  
2) `md_ssm.py` — **MD-SSM** (Equation 8): p-branch + multi-direction scan paths + merge + gating.  
3) `md_mamba_block.py` — **MD-Mamba Block** (Equation 7): pre-norm residual wrapper around MD-SSM + FFN.

---

## Equations: what they mean and where they live

### Equation (7): MD-Mamba Block (pre-norm residual)
The paper defines an MD-Mamba block as a stable pre-norm residual block:

**(7a)** `T' = MD-SSM(LayerNorm(T)) + T`  
**(7b)** `Tout = FFN(LayerNorm(T')) + T'`

Interpretation:
- Run **LayerNorm → MD-SSM**, then add a residual connection (7a).
- Run **LayerNorm → FFN**, then add another residual connection (7b).

**Where this is implemented**
- `md_mamba_block.py` implements Eq. (7a–7b).
- It calls `MDSSM` from `md_ssm.py`.

---

### Equation (8): MD-SSM (Multi-Direction State Space Model)
Equation (8) defines **MD-SSM**, the key novelty. It builds two branches (`p` processing, `g` gating), runs a shared sequence operator under three scan orders (horizontal/vertical/spiral), merges them, and gates the result.

Let the per-group input token sequence be:
- `x ∈ ℝ^{B × L × D}`  
(Within Stage 2, we typically split `Tconcat` into three groups `xs, xc, xf`, each `(B, L, D)`.)

**(8a) Processing branch**
`p = SiLU(Conv1D(Linear(x)))`

Meaning:
- `Linear`: channel mixing per token
- `Conv1D`: local sequence context injection (short-range inductive bias)
- `SiLU`: nonlinearity
- Output `p` has shape `(B, L, D)`.

**(8b) Multi-direction scans + merge**
`Σ = LayerNorm( Merge( Ψx(p), Ψy(p), Ψθ(p) ) )`

Meaning:
- `Ψx`: horizontal scan path (row-wise order)
- `Ψy`: vertical scan path (column-wise order)
- `Ψθ`: spiral scan path (top-right start, CCW rotation, periphery→center)
- Each Ψ performs:
  1) reorder tokens into a scan order  
  2) apply the **sequence core** (SSM/Mamba operator)  
  3) undo reorder back to original positions
- `Merge` in this paper is **summation**: `Ψx(p) + Ψy(p) + Ψθ(p)`
- `LayerNorm` stabilizes multi-path fusion.
- Output `Σ` has shape `(B, L, D)`.

**(8c) Gating branch**
`g = SiLU(Linear(x))`

Meaning:
- A learned gate per token feature dimension.
- Output `g` has shape `(B, L, D)`.

**(8d) Gated output projection**
`yout = Linear( Σ ⊙ g )`

Meaning:
- `⊙` is Hadamard (element-wise) product.
- Gate the aggregated multi-direction context `Σ` using `g`, then project.
- Output `yout` has shape `(B, L, D)`.

**Where this is implemented**
- `md_ssm.py` implements Eq. (8a–8d).
- The scan ordering utilities are in `scans.py`.
- The per-scan sequence operator is implemented by `SequenceCore` in `mamba_core.py`.

---

## File: `mamba_core.py`

### Architectural role
This file contains the **sequence operator core** applied inside each scan path Ψ.  
It must be **order-sensitive** (because scan permutations only matter if the operator depends on sequence order).

### Required names / interface

#### `class SequenceCore(nn.Module)`
**Purpose**
Defines the strict interface for any scan-path core operator.

**Contract**
- Input: `x` of shape `(B, L, D)`
- Output: `y` of shape `(B, L, D)`
- Must preserve dtype/device; no assumptions about `L` beyond being the sequence length.
- Does NOT implement scans, gating, residuals, or 3-scale splitting.

#### `class DummyCore(SequenceCore)`
**Purpose**
A lightweight placeholder operator used to validate scan/reorder/merge correctness before integrating a real Mamba core.

**Must**
- Preserve `(B, L, D) → (B, L, D)`
- Produce finite outputs on random tensors

#### `build_core(kind: str, d_model: int, **kwargs) -> SequenceCore`
**Purpose**
Factory that creates the requested core implementation (`"dummy"` now; optionally `"mamba"` later) without changing MD-SSM code.

#### `assert_core_io(x: torch.Tensor, y: torch.Tensor, d_model: int) -> None`
**Purpose**
Validation helper:
- check `x.shape == y.shape`
- check `x.ndim == y.ndim == 3`
- check last dim equals `d_model`
- check outputs are finite

---

## File: `md_ssm.py`

### Architectural role
Implements **MD-SSM (Eq. 8)**.

### Required names / responsibilities

#### `class MDSSM(nn.Module)`
**Purpose**
Compute MD-SSM over the concatenated sequence `(B, 3L, D)` by applying Eq. (8a–8d) on each of the three scale groups.

**Input / Output**
- Input: `(B, 3L, D)`
- Output: `(B, 3L, D)`

**Key invariants**
- Must not mix scan orders across scale boundaries: scanning is applied within each group `(B, L, D)` based on `(Htok, Wtok)`.
- Merge is by sum (paper default).
- Spiral scan must match paper description: start top-right, rotate counterclockwise, move periphery→center.

#### `split_concat_sequence(x: torch.Tensor, L: int)`
**Purpose**
Split concatenated tokens:
- Input `(B, 3L, D)`
- Output `(xs, xc, xf)` each `(B, L, D)`

#### `merge_concat_sequence(xs, xc, xf)`
**Purpose**
Concatenate back to `(B, 3L, D)`.

#### `compute_p_branch(xg, p_linear, p_conv1d)`
**Purpose**
Implement Eq. (8a) for one scale group `(B, L, D)`:
- `p = SiLU(Conv1D(Linear(xg)))`

#### `compute_g_branch(xg, g_linear)`
**Purpose**
Implement Eq. (8c) for one scale group `(B, L, D)`:
- `g = SiLU(Linear(xg))`

#### `run_scan_path(p, idx, core)`
**Purpose**
Implement one scan operator Ψ for one scale group:
1) reorder by `idx`
2) apply `core`
3) undo reorder using inverse permutation
Return `(B, L, D)` aligned to original token positions.

#### `merge_paths(px, py, ps, merge_ln, merge="sum")`
**Purpose**
Implement Eq. (8b):
- Merge via sum (default): `px + py + ps`
- Normalize: `LayerNorm(...)`
Return `Sigma` of shape `(B, L, D)`.

#### `gated_output(Sigma, g, out_linear)`
**Purpose**
Implement Eq. (8d):
- `y = Linear(Sigma ⊙ g)`
Return `(B, L, D)`.

#### `assert_mdssm_io(x, y, L, d_model)`
**Purpose**
Validation helper:
- input/output shapes match `(B, 3L, D)`
- outputs finite
- no NaNs / inf

---

## File: `md_mamba_block.py`

### Architectural role
Implements the **MD-Mamba block (Eq. 7)**: a stable pre-norm residual wrapper around MD-SSM plus a standard FFN.

### Required names / responsibilities

#### `class FFN(nn.Module)`
**Purpose**
Implements the FFN used in Eq. (7b):
- `Linear → activation (SiLU/GELU) → Dropout → Linear → Dropout`
- Shape-preserving: `(B, N, D) → (B, N, D)`.

#### `class MDMambaBlock(nn.Module)`
**Purpose**
Implement Eq. (7a–7b):
- `x1 = x + MDSSM(LN(x))`
- `y  = x1 + FFN(LN(x1))`

**Input / Output**
- Input: `(B, N, D)` where `N=3L`
- Output: `(B, N, D)`

#### `residual_add(x, delta)`
**Purpose**
Helper for residual connections: `x + delta`.

#### `assert_block_io(x, y, d_model)`
**Purpose**
Validation helper:
- input/output shapes equal
- last dim equals `d_model`
- outputs finite

---

## Composition: how these files build Stage 2

1) `SequenceCore` (`mamba_core.py`) defines the scan-path sequence operator: `(B, L, D) -> (B, L, D)`.
2) `MDSSM` (`md_ssm.py`) implements Eq. (8) using:
   - p-branch (8a)
   - three scan paths Ψx/Ψy/Ψθ using `SequenceCore`
   - merge + LN (8b)
   - g-branch (8c)
   - gated output projection (8d)
3) `MDMambaBlock` (`md_mamba_block.py`) implements Eq. (7) with pre-norm residual + FFN around `MDSSM`.
4) The Stage 2 stack applies `MDMambaBlock` K times (K=3 in the paper) to output `Tout` for Stage 3.

---
