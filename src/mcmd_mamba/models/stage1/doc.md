# Stage 1 — Multi-Channel Enhancement (Paper §2.1)

Stage 1 implements **multi-channel feature extraction**, **pyramidal self-attention**, and **sequential token concatenation** to produce a scale-aware token sequence for Stage 2.

---

## 2.1 Multi-Channel Feature Extraction

**Paper:** The objective is to capture retinal anatomical details that cannot be represented by a single input transformation. This design alleviates scale bias and information loss from uniform receptive fields. Three parallel branches model **structural**, **contextual**, and **fine-grained** information using scale-specific inputs and convolutional kernels.

Given fundus image I ∈ ℝ^{H×W×3}:

| Branch | Input | Purpose |
|--------|-------|---------|
| **Structural** | Gray(I) | Global anatomical layout |
| **Contextual** | CLAHE(I) | Local intensity variations |
| **Fine-grained** | Sobel(I) | Vessels and micro-lesions |

Each branch uses an adapted backbone with kernel sizes 7×7, 5×5, and 3×3, producing feature maps Fs, Fc, Ff ∈ ℝ^{hi×wi×d}.

**Code:** `mce.py`

- `rgb_to_gray(x)` — Grayscale from RGB
- `clahe_1ch_from_rgb(x_rgb)` — CLAHE in LAB space, return 1-channel
- `sobel_edges(x_gray)` — Sobel magnitude
- `SimpleBranchCNN(in_ch, out_ch, stem_kernel)` — Per-branch CNN (placeholder for ConvNeXt adapter)
- `MultiChannelEnhancement(feat_ch)` — Three branches: structural (Gray), contextual (CLAHE), fine (Sobel)

---

## Pyramidal Self-Attention (Eq. 1)

**Paper:** The multi-scale feature maps make it difficult to localize regions relevant to diagnosis. Direct aggregation amplifies background noise. We introduce pyramidal self-attention that selectively highlights diagnostically informative regions. The fine-grained map Ff constructs a spatial mask M; masks for Fs and Fc are obtained by downsampling M. Each feature map is reweighted:

**Eq. (1):** F′_i = Fi ⊗ Mi,  i ∈ {s, c, f}

**Code:** `psa.py`

- `PyramidalSelfAttention(patch_size, sharpen)` — Builds M from Ff: patch-max, normalize, interpolate, sigmoid sharpen
- `apply_psa(M, feat)` — Resizes M to feat spatial dims and multiplies: feat * M_resized

---

## Sequential Token Concatenation (Eq. 2, 3)

**Paper:** To unify multi-scale representations into a single ordered sequence, each reweighted feature map F′_i is spatially downsampled, flattened, and linearly projected into a shared embedding space, producing Ti ∈ ℝ^{Ni×d}. Tokens are concatenated:

**Eq. (2):** Tconcat = [Ts ∥ Tc ∥ Tf] ∈ ℝ^{N×d},  N = Σ_i Ni

To retain scale-specific information, learnable scale embeddings Escale ∈ ℝ^{3L×D} are introduced:

**Eq. (3):** T = Tconcat ⊕ Escale

**Code:** `tokenization.py`

- `Tokenizer(cfg, in_channels, d_model, token_hw)` — Resize to token_hw, flatten, linear project → (B, L, d_model)
- `ScaleEmbedding(d_model, num_tokens)` — Learnable embeddings; forward adds embed to tokens
- `concat_with_scale_embeddings(tokens, scale_embed)` — Tconcat + Escale

---

## Wrapper: Full Stage 1 Pipeline

**Code:** `wrapper.py`

- `Stage1(feat_ch, d_model, token_hw, ...)` — Wires:
  1. MCE(x_rgb, x_clahe_1ch) → Fs, Fc, Ff
  2. M = PSA(Ff); Fs, Fc, Ff = apply_psa(M, ...)
  3. Ts, Tc, Tf = Tokenizer(...) for each branch
  4. Tconcat = concat([Ts, Tc, Tf]); T = concat_with_scale_embeddings(Tconcat, scale_embed)
- `Stage1.split_tokens(T)` — Splits T back into (Ts, Tc, Tf) for Stage 3

---

## Module Layout

| File | Contents |
|------|----------|
| `mce.py` | rgb_to_gray, clahe_1ch_from_rgb, sobel_edges, SimpleBranchCNN, MultiChannelEnhancement |
| `psa.py` | PyramidalSelfAttention, apply_psa (Eq. 1) |
| `tokenization.py` | Tokenizer, ScaleEmbedding, concat_with_scale_embeddings (Eq. 2, 3) |
| `wrapper.py` | Stage1 — full pipeline MCE → PSA → Tokenize → concat + scale_embed |

---

## Input / Output

**Input:** x_rgb (B, 3, H, W), x_clahe_1ch (B, 1, H, W) optional

**Output:** T (B, 3L, D) where L = token_hw[0] × token_hw[1]
