"""
NHGE — Neuro-Harmonic Graph Engine
====================================
A next-generation ML architecture that replaces transformer parallelism
with iterative harmonic resonance over a dynamic graph structure.

Quick start:
    from nhge import NHGE, nhge_small, nhge_base, nhge_large
    from nhge import NHGETokenizer, NHGETrainer, NHGEInference
"""

# ====================== AUTO VERSION SUPPORT ======================
# This works together with setuptools_scm in pyproject.toml
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.3"   # fallback when _version.py is not generated yet

__author__ = "NHGE Project Team , Hekima A. Mwala & MWALA_LEARN TEAM"
__license__ = "MIT"

# ====================== IMPORTS ======================
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

# ====================== PUBLIC API ======================
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

# Optional helper function
def help():
    """Print quick help for the NHGE package."""
    print(__doc__)
    print(f"\nVersion : {__version__}")
    print(f"Author  : {__author__}")
    print(f"License : {__license__}")
    print("\nCLI commands:")
    print("   python -m nhge --version")
    print("   python -m nhge --info")