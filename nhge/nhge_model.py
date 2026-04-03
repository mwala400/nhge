"""
Neuro-Harmonic Graph Engine (NHGE)
====================================
A novel ML architecture that replaces transformer parallelism with
iterative harmonic resonance over a dynamic graph structure.

Core idea:
- Tokens become graph nodes with phase (θ) and amplitude (h) states
- Edges carry harmonic weights w_e based on semantic similarity
- Information propagates iteratively via resonance, not parallel attention
- Convergence is detected dynamically — no fixed depth needed

Author: NHGE Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Harmonic Edge Layer
#    Computes edge weights from pairwise node similarity, modulated by phase
# ---------------------------------------------------------------------------

class HarmonicEdgeLayer(nn.Module):
    """
    Builds a dynamic adjacency matrix where each edge weight is:
        w_ij = softmax( (Q_i · K_j) / sqrt(d) ) * cos(θ_i - θ_j)

    The cosine term encodes phase alignment — nodes in phase reinforce,
    out-of-phase nodes suppress. This is the harmonic mechanism.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.phase_proj = nn.Linear(d_model, n_heads)   # produces per-head phase θ

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h   : (B, N, D) node hidden states
            mask: (B, N) boolean, True = padding

        Returns:
            edge_weights : (B, H, N, N) — harmonic adjacency
            phases       : (B, N, H)    — per-node phase angles
        """
        B, N, D = h.shape

        # Query/Key projections, split into heads
        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # (B, H, N, d_head)

        # Scaled dot-product similarity
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # (B, H, N, N)

        # Phase angles for each node per head
        phases = self.phase_proj(h)  # (B, N, H)

        # Phase difference matrix: θ_i - θ_j  →  cos(Δθ)
        phi_i = phases.permute(0, 2, 1).unsqueeze(-1)    # (B, H, N, 1)
        phi_j = phases.permute(0, 2, 1).unsqueeze(-2)    # (B, H, 1, N)
        harmonic_mod = torch.cos(phi_i - phi_j)          # (B, H, N, N)

        # Modulate similarity with harmonic term
        scores = scores * (1.0 + harmonic_mod) * 0.5

        if mask is not None:
            # Mask padding nodes
            col_mask = mask.unsqueeze(1).unsqueeze(2)    # (B, 1, 1, N)
            scores = scores.masked_fill(col_mask, float('-inf'))

        edge_weights = F.softmax(scores, dim=-1)
        return edge_weights, phases


# ---------------------------------------------------------------------------
# 2. Harmonic Node Update
#    Aggregates neighbour messages weighted by edge harmonics
# ---------------------------------------------------------------------------

class HarmonicNodeUpdate(nn.Module):
    """
    For each node v:
        m_v  = Σ_u  w_{vu} * h_u                  (neighbour aggregation)
        h_v' = LayerNorm(h_v + FFN(h_v + m_v))    (residual + feed-forward)
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, h: torch.Tensor, edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h            : (B, N, D)
            edge_weights : (B, H, N, N)

        Returns:
            h_new : (B, N, D)
        """
        B, N, D = h.shape

        # Value projections, split into heads
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # (B, H, N, d_head)

        # Harmonic message aggregation per head
        messages = torch.matmul(edge_weights, V)          # (B, H, N, d_head)
        messages = messages.transpose(1, 2).contiguous().view(B, N, D)
        messages = self.W_o(messages)

        # Residual 1: add aggregated messages
        h = self.norm1(h + messages)

        # Residual 2: feed-forward
        h = self.norm2(h + self.ffn(h))

        return h


# ---------------------------------------------------------------------------
# 3. Phase Update
#    Phases evolve based on node state changes (harmonic "tuning")
# ---------------------------------------------------------------------------

class PhaseUpdate(nn.Module):
    """
    After each node update, phases are re-tuned:
        Δθ = tanh(W_phase · h) * π
        θ  ← θ + α * Δθ   (damped update)
    """

    def __init__(self, d_model: int, n_heads: int, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.W_phase = nn.Linear(d_model, n_heads)

    def forward(
        self, h: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h      : (B, N, D)
            phases : (B, N, H)

        Returns:
            phases_new : (B, N, H)
        """
        delta = torch.tanh(self.W_phase(h)) * math.pi    # (B, N, H)
        return phases + self.alpha * delta


# ---------------------------------------------------------------------------
# 4. NHGE Block — one complete harmonic iteration step
# ---------------------------------------------------------------------------

class NHGEBlock(nn.Module):
    """
    Single harmonic iteration step:
        1. Compute edge weights + phases   (HarmonicEdgeLayer)
        2. Update node states              (HarmonicNodeUpdate)
        3. Update phases                   (PhaseUpdate)
        4. Return new state + convergence delta
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        phase_alpha: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.edge_layer  = HarmonicEdgeLayer(d_model, n_heads)
        self.node_update = HarmonicNodeUpdate(d_model, n_heads, ffn_dim, dropout)
        self.phase_upd   = PhaseUpdate(d_model, n_heads, phase_alpha)

    def forward(
        self,
        h: torch.Tensor,
        phases: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new      : (B, N, D)
            phases_new : (B, N, H)
            delta      : scalar — mean L2 change in node states (convergence metric)
        """
        edge_weights, _ = self.edge_layer(h, mask)
        h_new = self.node_update(h, edge_weights)
        phases_new = self.phase_upd(h_new, phases)

        # Convergence metric
        delta = (h_new - h).norm(dim=-1).mean()

        return h_new, phases_new, delta


# ---------------------------------------------------------------------------
# 5. Full NHGE Model
# ---------------------------------------------------------------------------

class NHGE(nn.Module):
    """
    Neuro-Harmonic Graph Engine

    Architecture:
        Embedding → (N shared NHGE blocks, iterated T times) → Readout → Head

    Key properties:
        - Shared weights across iterations (like a universal transformer)
        - Early stopping when convergence threshold ε is met
        - Max iterations T caps computation
        - Graph readout via mean or learned attention pooling
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        n_layers: int = 4,          # distinct NHGE blocks stacked per iteration
        max_iter: int = 8,          # maximum harmonic iterations
        epsilon: float = 1e-3,      # convergence threshold
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,   # None = language model (vocab_size logits)
        phase_alpha: float = 0.1,
        readout: str = "mean",       # "mean" | "cls" | "attention"
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.max_iter  = max_iter
        self.epsilon   = epsilon
        self.readout   = readout
        ffn_dim = ffn_dim or d_model * 4

        # --- Embedding ---
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_seq_len, d_model)
        self.embed_drop  = nn.Dropout(dropout)
        self.embed_norm  = nn.LayerNorm(d_model)

        # --- Phase initialiser (maps embedding to initial phases) ---
        self.phase_init = nn.Linear(d_model, n_heads)

        # --- Harmonic iteration blocks ---
        self.blocks = nn.ModuleList([
            NHGEBlock(d_model, n_heads, ffn_dim, dropout, phase_alpha)
            for _ in range(n_layers)
        ])

        # --- Readout (attention pooling) ---
        if readout == "attention":
            self.pool_query = nn.Linear(d_model, 1)

        # --- Output head ---
        out_dim = num_classes if num_classes is not None else vocab_size
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = input_ids.shape
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)

        h = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        h = self.embed_norm(self.embed_drop(h))
        phases = self.phase_init(h)   # (B, N, H) — initial phases from content
        return h, phases

    def _readout(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Collapse (B, N, D) → (B, D) for classification tasks.
        """
        if self.readout == "cls":
            return h[:, 0, :]

        if self.readout == "attention":
            scores = self.pool_query(h).squeeze(-1)   # (B, N)
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, 1)
            return (weights * h).sum(dim=1)                    # (B, D)

        # Default: mean pooling (ignore padding)
        if mask is not None:
            lengths = (~mask).float().sum(dim=1, keepdim=True).unsqueeze(-1)
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            return h.sum(dim=1) / lengths.squeeze(1)
        return h.mean(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_iterations: bool = False,
        lm_mode: bool = False,
    ) -> dict:
        """
        Args:
            input_ids        : (B, N) token ids
            mask             : (B, N) True = padding
            return_iterations: include per-iteration deltas in output
            lm_mode          : if True, output per-token logits (B, N, V)
                               if False, output pooled logits (B, C)

        Returns dict with keys:
            logits      : final predictions
            iterations  : number of harmonic iterations performed
            deltas      : list of convergence deltas (if return_iterations)
            h_final     : final node states (B, N, D)
        """
        h, phases = self._embed(input_ids)
        deltas = []
        iters = 0

        for t in range(self.max_iter):
            h_prev = h
            for block in self.blocks:
                h, phases, delta = block(h, phases, mask)
            deltas.append(delta.item())
            iters = t + 1

            # Early stopping on convergence
            if delta < self.epsilon:
                break

        h = self.output_norm(h)

        if lm_mode:
            logits = self.output_head(h)          # (B, N, vocab)
        else:
            pooled = self._readout(h, mask)
            logits = self.output_head(pooled)     # (B, C)

        out = {"logits": logits, "iterations": iters, "h_final": h}
        if return_iterations:
            out["deltas"] = deltas
        return out


# ---------------------------------------------------------------------------
# 6. Convenience constructors
# ---------------------------------------------------------------------------

def nhge_small(vocab_size: int, **kwargs) -> NHGE:
    """~15M params — fast experiments"""
    defaults = dict(d_model=128, n_heads=4, n_layers=2, max_iter=6)
    defaults.update(kwargs)   # caller kwargs always win, no duplicate error
    return NHGE(vocab_size, **defaults)


def nhge_base(vocab_size: int, **kwargs) -> NHGE:
    """~85M params — comparable to BERT-base"""
    defaults = dict(d_model=512, n_heads=8, n_layers=4, max_iter=8)
    defaults.update(kwargs)
    return NHGE(vocab_size, **defaults)


def nhge_large(vocab_size: int, **kwargs) -> NHGE:
    """~340M params — comparable to GPT-2 medium"""
    defaults = dict(d_model=1024, n_heads=16, n_layers=6, max_iter=10)
    defaults.update(kwargs)
    return NHGE(vocab_size, **defaults)
