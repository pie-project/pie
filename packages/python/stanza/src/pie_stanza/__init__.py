"""Stanza by pie: Poem-programmable post-training.

A stanza is a few lines of Poem that say what is trainable in a model and
where it attaches. This package mounts stanzas on a pie model program,
trains one by differentiating that very program, and serves the result with
pie next to other adapters in the same batch.
"""

from .adapter import Adapter, from_peft
from .check import Report
from .data import Dataset, chat, pairs
from .evaluate import evaluate, json_exact
from .model import Model, Trainer
from .serve import Pie
from .skew import Skew, skew

__all__ = ["Adapter", "Dataset", "Model", "Pie", "Report", "Skew", "Trainer", "chat", "evaluate", "from_peft",
           "json_exact", "pairs", "skew"]
