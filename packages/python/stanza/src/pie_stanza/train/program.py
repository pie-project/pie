"""The program: one traced model with its mounted stanzas, as `pie model trace`
writes it into a run directory.

A run directory holds four files. `trace.json` is the `poem_ir::Trace` the
trainer replays. `banks.json` lists every registered bank, the weights stanzas
train. `import.json` says which checkpoint tensor fills each frozen param.
`mount.json` records which stanzas were mounted and the sha256 of every Poem
file that was read. `Program.load` reads all four and checks that they agree.
A bank whose name is not `layers.{l}.{stanza}.{site}.{bank}` belongs to no
stanza: it stays zero, and no trainer trains it.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

BANKS_FORMAT = "pie-stanza.banks/1"
IMPORT_FORMAT = "pie-stanza.import/1"
INITS = ("zeros", "kaiming", "normal")


@dataclass(frozen=True)
class Bank:
    """One registered bank: a weight a stanza declares, `slots` adapters deep."""

    name: str
    """The Poem struct path, `layers.{l}.{stanza}.{site}.{bank}`."""
    shape: tuple[int, ...]
    """The bank's shape; axis 0 is the slot."""
    dtype: str
    """The dtype pie serves the bank at, such as `bf16`."""
    stanza: str | None
    """The stanza that declares the bank, or None for a bank no stanza names."""
    site: str | None
    """The tap it attaches at, such as q, k, v, o, gate, up or down."""
    layer: int | None
    """The layer index, or None for a bank outside the layers."""
    init: str
    """How the trainer starts it: zeros, kaiming or normal."""


@dataclass
class Program:
    """A traced model and its mounted stanzas, read from one run directory."""

    model: str
    """The model id, such as `qwen3-0.6b`."""
    package: str
    """The Poem package that traced it, such as `qwen3_dense`."""
    slots: int
    """Adapters resident at once: axis 0 of every bank."""
    stanzas: list[str]
    """The mounted stanzas, in mount order."""
    banks: list[Bank]
    """Every registered bank, in trace order."""
    trace: dict
    """The `poem_ir::Trace` as parsed JSON."""
    imports: list[dict]
    """`import.json`'s params: which checkpoint tensors fill each frozen param."""
    mount: dict
    """`mount.json` as parsed JSON."""
    path: Path
    """The run directory."""

    @staticmethod
    def load(run_dir: str | os.PathLike) -> Program:
        """Read a run directory and check that its four files agree."""
        path = Path(run_dir)
        trace = _read(path / "trace.json")
        listed = _read(path / "banks.json")
        imported = _read(path / "import.json")
        mount = _read(path / "mount.json")
        for name, doc, want in (("banks.json", listed, BANKS_FORMAT), ("import.json", imported, IMPORT_FORMAT)):
            if doc.get("format") != want:
                raise ValueError(f"{path / name}: format {doc.get('format')!r}, expected {want!r}")

        registered = {p["name"]: p for p in trace["params"] if p["source"] == "Registered"}
        names = [b["name"] for b in listed["banks"]]
        if sorted(names) != sorted(registered):
            odd = sorted(set(names) ^ set(registered))
            raise ValueError(f"{path}: banks.json and trace.json disagree on bank `{odd[0]}`")

        stated = any("init" in p for p in [*listed["banks"], *registered.values()])
        if not stated:
            warnings.warn(
                f"{path}: neither banks.json nor trace.json states an init for any bank; guessing from names "
                "(`lora_a` and `a` start kaiming, every other bank zeros)",
                stacklevel=2,
            )
        banks = []
        for b in listed["banks"]:
            init = b.get("init", registered[b["name"]].get("init", "zeros")) if stated else _guess(b["name"])
            if init not in INITS:
                raise ValueError(f"bank `{b['name']}`: init {init!r} is not one of {INITS}")
            shape = tuple(b["shape"])
            if shape[0] != listed["slots"] or list(shape) != registered[b["name"]]["shape"]:
                raise ValueError(f"bank `{b['name']}`: shape {list(shape)} does not match the trace or {listed['slots']} slots")
            layer = None if b.get("layer") is None else int(b["layer"])
            banks.append(Bank(b["name"], shape, b["dtype"], b.get("stanza"), b.get("site"), layer, init))

        stanzas = list(mount.get("stanzas") or dict.fromkeys(b.stanza for b in banks if b.stanza is not None))
        for b in banks:
            if b.stanza is not None and b.stanza not in stanzas:
                raise ValueError(f"bank `{b.name}` belongs to stanza `{b.stanza}`, which mount.json does not mount")
        return Program(
            model=listed["model"],
            package=listed["package"],
            slots=int(listed["slots"]),
            stanzas=stanzas,
            banks=banks,
            trace=trace,
            imports=imported["params"],
            mount=mount,
            path=path,
        )

    def banks_of(self, stanza: str) -> list[Bank]:
        """The banks `stanza` declares; refuses a stanza this program does not mount."""
        if stanza not in self.stanzas:
            raise ValueError(f"stanza `{stanza}` is not mounted; this program mounts {self.stanzas}")
        return [b for b in self.banks if b.stanza == stanza]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _guess(name: str) -> str:
    return "kaiming" if name.rsplit(".", 1)[-1] in ("lora_a", "a") else "zeros"
