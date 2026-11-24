"""pie_stanza.train: the trainer. It replays the trace of the Poem program pie
serves, trains one stanza's banks on it, and writes adapter bundles.

    program = Program.load("run")                 # what `pie model trace` wrote
    t = Trainer(program, "Qwen3-0.6B", stanza="ia3")   # a Hugging Face checkpoint directory
    t.forward_backward(Batch(tokens, weights))
    t.optim_step(lr=3e-3)
    t.save("adapters/larkspur-ia3", name="larkspur-ia3")
"""

from .batch import Batch
from .bundle import read_bundle, write_bundle
from .loss import LOSSES
from .peft import from_peft, to_peft
from .program import Bank, Program
from .trainer import Trainer

__all__ = ["LOSSES", "Bank", "Batch", "Program", "Trainer", "from_peft", "read_bundle", "to_peft", "write_bundle"]
