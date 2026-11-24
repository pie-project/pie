"""An adapter: one trained stanza, as a bundle directory on disk.

The bundle holds `adapter.safetensors` (one bf16 tensor per bank, keyed by
the Poem bank name) and `stanza.json` (identity, shape and provenance). The
trainer writes it; pie serves its `adapter.safetensors` bytes unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Adapter:
    """A bundle read from disk."""

    path: Path
    """The bundle directory."""
    name: str
    """The name it was saved under."""
    id: str
    """`ada_` and the first 16 hex digits of the sha256 of `adapter.safetensors`."""
    stanza: str
    """The stanza whose banks it fills."""
    bytes: bytes
    """The `adapter.safetensors` file, as pie receives it."""
    meta: dict
    """The whole of `stanza.json`."""

    @staticmethod
    def load(dir: str | Path) -> "Adapter":
        """Read the bundle in `dir`; refuse it if its bytes no longer match its id."""
        path = Path(dir)
        meta = json.loads((path / "stanza.json").read_text())
        data = (path / "adapter.safetensors").read_bytes()
        id = "ada_" + hashlib.sha256(data).hexdigest()[:16]
        if meta["id"] != id:
            raise ValueError(f"{path}: adapter.safetensors hashes to {id}, but stanza.json says {meta['id']}")
        return Adapter(path=path, name=meta["name"], id=id, stanza=meta["stanza"], bytes=data, meta=meta)

    def to_peft(self, out_dir: str | Path) -> Path:
        """Write this adapter as a Hugging Face PEFT directory (LoRA and IA3 only)."""
        from .train import Program, to_peft

        to_peft(self.path, out_dir, Program.load(self.meta["run"]))
        return Path(out_dir)


def from_peft(peft_dir: str | Path, model, name: str) -> Adapter:
    """Convert a Hugging Face PEFT LoRA or IA3 directory into a bundle for
    `model`'s mounted program, saved under `<pie_home>/stanza/adapters/<name>`."""
    from . import train

    peft_type = json.loads((Path(peft_dir) / "adapter_config.json").read_text())["peft_type"]
    stanza = {"LORA": "lora", "IA3": "ia3"}.get(peft_type)
    if stanza is None:
        raise ValueError(f"{peft_dir}: PEFT type {peft_type} has no standard-library stanza")
    trainer = model.trainer(stanza, device="cpu")
    trainer.inner.load(train.from_peft(peft_dir, model.program(), stanza=stanza))
    return trainer.save(model.home / "stanza" / "adapters" / name, name=name, meta={"source": str(peft_dir)})
