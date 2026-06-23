"""
Stage 1 wrapper: MCE → PSA (mask into Fs,Fc,Ff) → Tokenize → concat + scale_embed.
Split bookkeeping (L, indices) for Stage 3 to split back into {Ts, Tc, Tf}.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .mce import MultiChannelEnhancement
from .psa import PyramidalSelfAttention, apply_psa
from .tokenization import ScaleEmbedding, Tokenizer, concat_with_scale_embeddings


class Stage1(nn.Module):
    """
    Wires: Fs,Fc,Ff = MCE(...) → M = PSA(Ff) → Fs,Fc,Ff *= M → Ts,Tc,Tf = tokenizer(...) → T = concat + scale_embed.
    Stores L (tokens per branch) and split indices for Stage 3.
    """

    def __init__(
        self,
        feat_ch: int = 256,
        d_model: int = 256,
        token_hw: Tuple[int, int] = (12, 12),
        patch_size: int = 8,
        sharpen: float = 4.0,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.feat_ch = cfg.get("feat_ch", feat_ch)
        self.d_model = cfg.get("d_model", d_model)
        self.token_hw = tuple(cfg.get("token_hw", token_hw))
        self.L = self.token_hw[0] * self.token_hw[1]  # tokens per branch (fixed for easy split)
        self.num_branches = 3  # Fs, Fc, Ff

        # MCE: Gray / CLAHE / Sobel → feature maps
        self.mce = MultiChannelEnhancement(feat_ch=self.feat_ch, cfg=cfg.get("mce", {}))

        # PSA: mask from Ff, then apply to all three
        self.psa = PyramidalSelfAttention(
            patch_size=cfg.get("psa", {}).get("patch_size", patch_size),
            sharpen=cfg.get("psa", {}).get("sharpen", sharpen),
            cfg=cfg.get("psa"),
        )

        # Tokenizer: one per branch (same in_ch=feat_ch, same token_hw → same L)
        tokenizer_cfg = {**cfg, "token_hw": self.token_hw}
        self.tokenizer = Tokenizer(
            tokenizer_cfg,
            in_channels=self.feat_ch,
            d_model=self.d_model,
            token_hw=self.token_hw,
        )

        # Scale embedding for full sequence length 3*L
        self.scale_embed = ScaleEmbedding(self.d_model, num_tokens=self.num_branches * self.L)

        # Split bookkeeping for Stage 3: indices [0:L], [L:2*L], [2*L:3*L]
        self.split_lengths: List[int] = [self.L] * self.num_branches
        self._split_indices: Optional[List[Tuple[int, int]]] = None

    def _get_split_indices(self) -> List[Tuple[int, int]]:
        """Return [(start, end), ...] for Ts, Tc, Tf in concatenated T."""
        if self._split_indices is None:
            start = 0
            self._split_indices = []
            for length in self.split_lengths:
                self._split_indices.append((start, start + length))
                start += length
        return self._split_indices

    def split_tokens(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split concatenated T (B, 3*L, D) back into Ts, Tc, Tf (each B, L, D)."""
        indices = self._get_split_indices()
        assert T.shape[1] == sum(self.split_lengths), (
            f"Token length {T.shape[1]} does not match expected 3*L = {3 * self.L}"
        )
        Ts = T[:, indices[0][0] : indices[0][1], :]
        Tc = T[:, indices[1][0] : indices[1][1], :]
        Tf = T[:, indices[2][0] : indices[2][1], :]
        return Ts, Tc, Tf

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe_1ch: Optional[torch.Tensor] = None,
        *,
        clahe_is_precomputed: bool = True,
        clahe_fallback_from_rgb: bool = True,
        check_shapes: bool = True,
        **mce_kwargs: Any,
    ) -> torch.Tensor:
        """
        x_rgb: (B, 3, H, W) normalized [0,1]
        x_clahe_1ch: (B, 1, H, W) optional; precomputed from dataset recommended.
        check_shapes: if True, assert mask and token shapes.
        Returns: T (B, 3*L, d_model) with scale embedding applied.
        """
        # 1) MCE
        Fs, Fc, Ff = self.mce(
            x_rgb,
            x_clahe_1ch,
            clahe_is_precomputed=clahe_is_precomputed,
            clahe_fallback_from_rgb=clahe_fallback_from_rgb,
            **mce_kwargs,
        )

        # 2) PSA mask from Ff, apply to all three
        M = self.psa(Ff)  # (B, 1, Hf, Wf)
        if check_shapes:
            assert M.shape[-2:] == Ff.shape[-2:], (
                f"PSA mask spatial shape {M.shape[-2:]} should match Ff {Ff.shape[-2:]}"
            )
        Fs = apply_psa(M, Fs)
        Fc = apply_psa(M, Fc)
        Ff = apply_psa(M, Ff)
        if check_shapes:
            assert Fs.shape[-2:] == Fc.shape[-2:] == Ff.shape[-2:], (
                "After apply_psa resize, Fs/Fc/Ff spatial shapes should match"
            )

        # 3) Tokenize (resize to token_hw inside tokenizer → same L for all)
        Ts = self.tokenizer(Fs)  # (B, L, D)
        Tc = self.tokenizer(Fc)
        Tf = self.tokenizer(Ff)
        if check_shapes:
            assert Ts.shape[1] == Tc.shape[1] == Tf.shape[1] == self.L, (
                f"Token length per branch should be L={self.L}, got Ts={Ts.shape[1]}, Tc={Tc.shape[1]}, Tf={Tf.shape[1]}"
            )
            assert Ts.shape[2] == Tc.shape[2] == Tf.shape[2] == self.d_model

        # 4) Concat + scale embed
        T = torch.cat([Ts, Tc, Tf], dim=1)  # (B, 3*L, D)
        T = concat_with_scale_embeddings(T, self.scale_embed)

        return T
