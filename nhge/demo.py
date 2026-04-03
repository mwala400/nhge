"""
NHGE Quick-Start Demo
======================
Run this file to see NHGE working end-to-end:
  1. Build a tiny vocabulary from sample text
  2. Construct an NHGE-small model
  3. Do a forward pass and inspect convergence behaviour
  4. Run a mock classification task
  5. Print harmonic state diagnostics

No GPU required — runs on CPU in seconds.
"""

import torch
import sys
import os

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

try:
    from nhge.nhge_model import NHGE, nhge_small
    from nhge.nhge_tokenizer import NHGETokenizer
    from nhge.nhge_inference import NHGEInference
except ImportError:
    from nhge_model import NHGE, nhge_small
    from nhge_tokenizer import NHGETokenizer
    from nhge_inference import NHGEInference


# ------------------------------------------------------------------
# 1. Sample corpus
# ------------------------------------------------------------------

CORPUS = [
    "the neuro harmonic graph engine processes tokens as graph nodes",
    "harmonic resonance allows information to propagate iteratively",
    "unlike transformers nhge does not require parallel attention",
    "each node updates its state based on neighbouring node phases",
    "convergence is detected dynamically reducing unnecessary computation",
    "the model adapts the number of iterations to input complexity",
    "simple inputs converge quickly complex ones require more iterations",
    "phase alignment between nodes encodes semantic similarity",
    "the harmonic edge weight combines similarity and phase coherence",
    "this architecture is an advancement over the transformer model",
]

LABELS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]    # 0=architecture, 1=comparison
LABEL_NAMES = ["architecture", "comparison"]


# ------------------------------------------------------------------
# 2. Build tokenizer
# ------------------------------------------------------------------

print("=" * 60)
print("NEURO-HARMONIC GRAPH ENGINE — Demo")
print("=" * 60)
print()

tok = NHGETokenizer(mode="word")
tok.build_vocab(CORPUS, min_freq=1, max_vocab=500)
print(f"Vocab size: {tok.vocab_size}")
print()


# ------------------------------------------------------------------
# 3. Build NHGE-small
# ------------------------------------------------------------------

model = nhge_small(
    vocab_size=tok.vocab_size,
    num_classes=2,      # binary classification demo
    readout="attention",
    epsilon=1e-4,
    dropout=0.0,        # no dropout for inference demo
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")
print()


# ------------------------------------------------------------------
# 4. Single forward pass — inspect convergence
# ------------------------------------------------------------------

print("--- Single forward pass ---")
enc = tok.batch_encode(CORPUS[:3], max_length=24, add_cls=True)
ids  = torch.tensor(enc["input_ids"],  dtype=torch.long)
mask = torch.tensor(enc["attention_mask"], dtype=torch.bool)

with torch.no_grad():
    out = model(ids, mask, lm_mode=False, return_iterations=True)

print(f"Logits shape  : {out['logits'].shape}")
print(f"Iterations ran: {out['iterations']}")
print(f"Deltas per iter: {[f'{d:.5f}' for d in out.get('deltas', [])]}")
print()


# ------------------------------------------------------------------
# 5. Classification demo
# ------------------------------------------------------------------

print("--- Classification demo ---")
inf = NHGEInference(model, tok, device="cpu")

results = inf.classify(
    ["harmonic resonance drives the graph update",
     "nhge replaces the transformer in language models"],
    max_length=24,
    label_names=LABEL_NAMES,
)
for i, r in enumerate(results):
    probs = [f"{p:.3f}" for p in r["scores"]]
    print(f"  [{i}] pred={r['name']:<14} scores={probs}  iters={r['iters']}")
print()


# ------------------------------------------------------------------
# 6. Harmonic state introspection
# ------------------------------------------------------------------

print("--- Harmonic state introspection ---")
state = inf.harmonic_state(
    "phase alignment between nodes encodes semantic similarity",
    max_length=16,
)
print(f"  Tokens    : {state['tokens']}")
print(f"  Iterations: {state['iterations']}")
print(f"  Deltas    : {[f'{d:.5f}' for d in state['deltas']]}")
print(f"  Node norms: {[f'{n:.3f}' for n in state['h_norm']]}")
print()


# ------------------------------------------------------------------
# 7. Embedding similarity
# ------------------------------------------------------------------

print("--- Embedding similarity ---")
embs = inf.embed(
    ["harmonic graph iteration",
     "neuro harmonic resonance",
     "unrelated random words here"],
    max_length=12,
)
def cosine(a, b):
    return (a @ b / (a.norm() * b.norm())).item()

print(f"  sim(0,1) = {cosine(embs[0], embs[1]):.4f}  ← expect higher (related)")
print(f"  sim(0,2) = {cosine(embs[0], embs[2]):.4f}  ← expect lower  (unrelated)")
print()

print("=" * 60)
print("NHGE demo complete — model is working correctly.")
print("Next steps:")
print("  1. Build a real tokenizer with nhge_tokenizer.NHGETokenizer.build_vocab()")
print("  2. Prepare DataLoaders using nhge_trainer.TokenDataset")
print("  3. Train with nhge_trainer.NHGETrainer(model, train_loader, ...)")
print("  4. Generate text with nhge_inference.NHGEInference.generate()")
print("=" * 60)
