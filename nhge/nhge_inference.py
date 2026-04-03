"""
NHGE Inference Engine
======================
Ready-to-use inference for:
  - Text generation (greedy, top-k, nucleus sampling)
  - Text classification
  - Embedding extraction
  - Harmonic state visualisation
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

try:
    from nhge.nhge_model import NHGE
    from nhge.nhge_tokenizer import NHGETokenizer
except ImportError:
    from nhge_model import NHGE
    from nhge_tokenizer import NHGETokenizer


class NHGEInference:
    """
    High-level inference wrapper for NHGE.

    Usage:
        inf = NHGEInference(model, tokenizer, device="cuda")
        text = inf.generate("Once upon a time", max_new_tokens=100)
        label = inf.classify("I love this movie!")
    """

    def __init__(
        self,
        model: NHGE,
        tokenizer: NHGETokenizer,
        device: str = "cpu",
    ):
        self.model     = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device    = device

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """
        Autoregressive text generation with NHGE.

        NHGE generates one token at a time. After each token, the full
        graph is re-iterated from the updated sequence. The harmonic
        resonance adapts dynamically to the growing context.
        """
        ids = self.tokenizer.encode(
            prompt, add_bos=True, max_length=self.model.pos_embed.num_embeddings - max_new_tokens
        )
        generated = list(ids)

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
            out = self.model(input_ids, lm_mode=True)
            logits = out["logits"][0, -1, :]       # (V,) — last position

            # Repetition penalty
            if repetition_penalty != 1.0:
                for prev_id in set(generated):
                    if logits[prev_id] < 0:
                        logits[prev_id] *= repetition_penalty
                    else:
                        logits[prev_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            if do_sample:
                # Top-k
                if top_k > 0:
                    kth_val = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                    logits = logits.masked_fill(logits < kth_val, float('-inf'))

                # Top-p (nucleus)
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float('-inf')
                    logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = logits.argmax(dim=-1).item()

            if next_id == self.tokenizer.EOS_ID:
                break

            generated.append(next_id)

        return self.tokenizer.decode(generated, skip_special=True)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def classify(
        self,
        texts: List[str],
        max_length: int = 128,
        label_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of texts.

        Returns list of dicts: {"label": int, "name": str, "scores": list}
        """
        enc = self.tokenizer.batch_encode(texts, max_length=max_length, add_cls=True)
        ids  = torch.tensor(enc["input_ids"],  dtype=torch.long,  device=self.device)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.bool, device=self.device)

        out = self.model(ids, mask, lm_mode=False)
        probs = F.softmax(out["logits"], dim=-1)

        results = []
        for i in range(len(texts)):
            p = probs[i].tolist()
            label_id = int(torch.argmax(out["logits"][i]).item())
            results.append({
                "label":  label_id,
                "name":   label_names[label_id] if label_names else str(label_id),
                "scores": p,
                "iters":  out["iterations"],
            })
        return results

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        texts: List[str],
        max_length: int = 128,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Extract sentence embeddings after harmonic convergence.

        Returns: (B, D) tensor
        """
        enc = self.tokenizer.batch_encode(texts, max_length=max_length)
        ids  = torch.tensor(enc["input_ids"],  dtype=torch.long,  device=self.device)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.bool, device=self.device)

        out = self.model(ids, mask, lm_mode=False)
        h   = out["h_final"]   # (B, N, D)

        if pool == "mean":
            pad = mask.unsqueeze(-1)
            h_masked = h.masked_fill(pad, 0.0)
            lengths = (~mask).float().sum(1, keepdim=True)
            return (h_masked.sum(1) / lengths)
        elif pool == "cls":
            return h[:, 0, :]
        else:
            return h.mean(1)

    # ------------------------------------------------------------------
    # Visualise harmonic state (returns dict for plotting)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def harmonic_state(self, text: str, max_length: int = 64) -> Dict:
        """
        Returns internal phase angles and convergence deltas
        for a given input — useful for visualising NHGE's dynamics.
        """
        ids = self.tokenizer.encode(text, max_length=max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids, return_iterations=True, lm_mode=False)

        tokens = [self.tokenizer.id2tok.get(i, "[UNK]") for i in ids]
        return {
            "tokens":     tokens,
            "iterations": out["iterations"],
            "deltas":     out.get("deltas", []),
            "h_norm":     out["h_final"][0].norm(dim=-1).tolist(),
        }
