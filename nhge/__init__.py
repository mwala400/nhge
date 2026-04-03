"""
NHGE — Neuro-Harmonic Graph Engine
====================================
A next-generation ML architecture that replaces transformer parallelism
with iterative harmonic resonance over a dynamic graph structure.

Quick start:
    from nhge import NHGE, nhge_small, nhge_base, nhge_large
    from nhge import NHGETokenizer, NHGETrainer, NHGEInference
"""

__version__ = "0.1.0"
__author__ = "NHGE Project"
__license__ = "MIT"

from nhge.nhge_model import (
    NHGE,
    NHGEBlock,
    HarmonicEdgeLayer,
    HarmonicNodeUpdate,
    PhaseUpdate,
    nhge_small,
    nhge_base,
    nhge_large,
)

from nhge.nhge_tokenizer import NHGETokenizer

from nhge.nhge_trainer import NHGETrainer, TokenDataset, WarmupCosineScheduler

from nhge.nhge_inference import NHGEInference

__all__ = [
    # Model
    "NHGE",
    "NHGEBlock",
    "HarmonicEdgeLayer",
    "HarmonicNodeUpdate",
    "PhaseUpdate",
    "nhge_small",
    "nhge_base",
    "nhge_large",
    # Tokenizer
    "NHGETokenizer",
    # Training
    "NHGETrainer",
    "TokenDataset",
    "WarmupCosineScheduler",
    # Inference
    "NHGEInference",
]
