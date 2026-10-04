"""Metis: a local, task-specific decision model post-training prototype.

Importing the package does not import torch, Transformers or download anything.
"""
__version__ = "0.1.0.dev0"

from .api import Predictor

__all__ = ["Predictor", "__version__"]
