# Stage 3 — Weighted Fusion and Classification (Paper §2.3)

Stage 3 aggregates the refined token sequence produced by Stage 2 and produces final logits. It implements the paper's **Weighted Fusion and Classification** module.

---

## 2.3 Paper Summary

**Token Split and Weighted Fusion (Eq. 9):** The objective is to aggregate complementary information across token scales while mitigating dominance from any single representation. We reverse the enhanced sequence T_out into scale-specific components (T′_s, T′_c, T′_f). Dual pooling (AP + MP) per group; weighted fusion with learnable ω_i:

**Eq. (9):** F_fuse = Σ_{i∈{s,c,f}} ω_i · concat(AP(T′_i), MP(T′_i))

**Classification (Eq. 10, 11):** Normalization and regularization, then linear projection with SiLU:

**Eq. (10):** F̄_fuse = Drop(LayerNorm(F_fuse))

**Eq. (11):** F̂_fuse = SiLU(Linear(F̄_fuse))

A final linear classifier maps F̂_fuse to logits over C disease categories.

---

## Input / Output

**Input:** `T_out` shaped `(B, 3L, D)`
- `B`: batch size
- `L`: tokens per scale group
- `D`: embedding dim
- Tokens are ordered as `[T_s || T_c || T_f]` (structural, contextual, fine-grained)

**Output:** `logits` shaped `(B, C)`
- `C`: number of classes

## Steps

### 1) Token split
`T_out` is partitioned into:
- `T_s`: `(B, L, D)`
- `T_c`: `(B, L, D)`
- `T_f`: `(B, L, D)`

### 2) Dual pooling per scale
For each `T_i ∈ {T_s, T_c, T_f}`:
- `AP_i = mean(T_i, dim=1)` -> `(B, D)`
- `MP_i = max(T_i, dim=1)` -> `(B, D)`
- `P_i = concat([AP_i, MP_i])` -> `(B, 2D)`

### 3) Weighted fusion
Learn weights `ω = [ω_s, ω_c, ω_f]` (normalized via softmax) and compute:
`F_fuse = Σ_i ω_i * P_i` -> `(B, 2D)`

### 4) Classification head
- `F = Dropout(LayerNorm(F_fuse))`
- `H = SiLU(Linear(F))`
- `logits = Linear(H)` -> `(B, C)`

## Code Mapping

| Paper | Code |
|-------|------|
| Eq. (9) F_fuse = Σ ω_i · concat(AP, MP) | `weighted_fusion.py` WeightedFusion |
| Eq. (10) F̄ = Drop(LayerNorm(F_fuse)) | `head.py` Stage3Head.norm, dropout |
| Eq. (11) F̂ = SiLU(Linear(F̄)), logits | `head.py` Stage3Head.linear1, classifier |

---

## Module Layout

| File | Contents |
|------|----------|
| `split.py` | `split_scales(T, L)`, `infer_L(T)` |
| `pooling.py` | `dual_pool_tokens(Ti)` -> `(B, 2D)` |
| `weighted_fusion.py` | `WeightedFusion.forward(Ts, Tc, Tf)` -> `F_fuse` |
| `head.py` | `Stage3Head` (LN + Dropout, Linear + SiLU, classifier) |
| `wrapper.py` | `Stage3.forward(T_out)` -> logits |

## Usage

```python
from mcmd_mamba.models.stage3 import Stage3

stage3 = Stage3(embed_dim=D, num_classes=C, tokens_per_scale=L, dropout=0.1)
logits = stage3(T_out)   # T_out: (B, 3L, D)
```
