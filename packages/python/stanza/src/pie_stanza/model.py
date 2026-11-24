"""The model a customer post-trains, and its trainer.

A `Model` is one pie model id, the Hugging Face checkpoint it reads, and the
PIE_HOME it is mounted and served from. Its methods mount stanzas, check the
mounted program, make a trainer for one stanza, and serve the program with
pie. The trainer and the server read the same `<pie_home>/models` files.
"""

from __future__ import annotations

import os
import subprocess
from functools import cached_property
from pathlib import Path

from . import check, mount, serve
from .adapter import Adapter


class Model:
    """One pie model id, its checkpoint, and the PIE_HOME it lives in."""

    def __init__(self, id: str, weights: str | os.PathLike | None = None, *, pie_home: str = "~/.pie-stanza",
                 models_dir: str | None = None, pie_bin: str = "target/release/pie"):
        self.id = id
        """The pie model id, such as `qwen3-0.6b`."""
        self.weights = Path(weights).resolve() if weights else None
        """The Hugging Face checkpoint directory; None for a model that is only mounted and checked."""
        self.home = Path(pie_home).expanduser().resolve()
        """The PIE_HOME every pie command of this model runs under."""
        self.models = Path(models_dir).resolve() if models_dir else self.home / "models"
        """The models directory mounts write and traces read; pie serves `<home>/models`."""
        self.pie_bin = Path(pie_bin).absolute()
        """The pie binary."""

    def pie(self, *args: str) -> str:
        """Run one pie command under this model's PIE_HOME and return its stdout;
        a failure raises with pie's own message."""
        done = subprocess.run([str(self.pie_bin), *args], env=self.env(), capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"pie {' '.join(args)} failed ({done.returncode}):\n{done.stderr.strip()}")
        return done.stdout

    def env(self) -> dict[str, str]:
        """The environment pie runs in: this PIE_HOME, and its config file."""
        env = {k: v for k, v in os.environ.items() if k != "PIE_CONFIG"}
        return env | {"PIE_HOME": str(self.home)}

    def mount(self, *stanzas: str) -> list[str]:
        """Mount these stanzas, in order: standard-library names (`"lora"`,
        `"ia3"`) or `.poem` paths. Returns the mounted stanza names."""
        return mount.mount(self, stanzas)

    def run_dir(self) -> Path:
        """The run directory of the program as mounted now, traced on first use."""
        mount.seed(self)
        out = self.home / "stanza" / "runs" / mount.mount_hash(self.models) / self.id
        return out if (out / "trace.json").exists() else mount.trace(self, out)

    def program(self):
        """The mounted program, as the trainer reads it."""
        from .train import Program

        return Program.load(self.run_dir())

    def check(self, *, weights: bool = False) -> "check.Report":
        """Where each mounted stanza attaches, what it costs per tenant, and
        whether lowering keeps its banks; with `weights`, also four checks
        against the checkpoint."""
        return check.report(self, weights=weights)

    def trainer(self, stanza: str | None, *, trainable: list[str] | None = None, device: str = "mps",
                dtype: str = "bf16", seed: int = 0) -> "Trainer":
        """A trainer for one mounted stanza (slot 0 of its banks), or with
        `stanza=None` for the checkpoint params `trainable` names by glob."""
        from . import train

        if self.weights is None:
            raise ValueError(f"{self.id}: training needs weights, a Hugging Face checkpoint directory")
        run = self.run_dir()
        inner = train.Trainer(train.Program.load(run), self.weights, stanza=stanza, trainable=trainable,
                              device=device, dtype=dtype, seed=seed)
        return Trainer(inner, run)

    def serve(self, *, port: int = 18431):
        """Start `pie serve` on this model's program and checkpoint; a context
        manager that gives a `Pie` and stops the server on exit."""
        return serve.serve(self, port)

    @cached_property
    def tokenizer(self):
        """The checkpoint's Hugging Face tokenizer, which renders every token id."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.weights)


class Trainer:
    """A trainer for one stanza of a mounted model.

    It runs `pie_stanza.train.Trainer` and keeps what a bundle's provenance
    needs: the data its batches came from, the learning rate, and the run
    directory it was made from. `save` returns an `Adapter`.
    """

    def __init__(self, inner, run: Path):
        self.inner = inner
        """The `pie_stanza.train.Trainer` doing the work."""
        self.run = run
        """The run directory of the program it trains."""
        self.adapter: Adapter | None = None
        """The bundle it saved or loaded last; what `skew` compares pie against."""
        self.data: list[str] = []
        """The sha256 of each data file its batches came from, in first-seen order."""
        self.lr: float | None = None
        """The learning rate of its last optimizer step."""

    @property
    def log(self) -> list[dict]:
        """One `{"step", "loss", "tokens", "seconds"}` entry per optimizer step."""
        return self.inner.log

    def forward_backward(self, batch, loss="cross_entropy", *, reference: bool = False) -> float:
        """Accumulate the gradient of `loss` over `batch`; returns the loss."""
        sha = getattr(batch, "data_sha256", None)
        if sha and sha not in self.data:
            self.data.append(sha)
        return self.inner.forward_backward(batch, loss, reference=reference)

    def optim_step(self, lr: float, **options) -> None:
        """One AdamW step at `lr`; `options` go to `pie_stanza.train.Trainer.optim_step`."""
        self.inner.optim_step(lr, **options)
        self.lr = lr

    def logprobs(self, tokens: list[list[int]], *, adapted: bool = True, dtype: str | None = None):
        """Per row: the log-prob of each next token, the argmax, and the top-2 gap."""
        return self.inner.logprobs(tokens, adapted=adapted, dtype=dtype)

    def save(self, out_dir: str | os.PathLike, *, name: str, meta: dict | None = None) -> Adapter:
        """Write the bundle to `out_dir` and return it as an `Adapter`."""
        meta = {"data_sha256": ",".join(self.data), "lr": self.lr, "run": str(self.run)} | (meta or {})
        self.adapter = Adapter.load(self.inner.save(out_dir, name=name, meta=meta))
        return self.adapter

    def load(self, adapter: Adapter) -> None:
        """Make the bundle `adapter` the trained slot, to measure it."""
        from .train import read_bundle

        self.inner.load(read_bundle(adapter.path)[0])
        self.adapter = adapter
