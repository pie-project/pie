"""Mounting: choosing which stanzas the model program carries.

A mount copies each stanza's `.poem` file into `<models>/lib/stanzas/`,
rewrites `<models>/lib/form.poem` to load them in order, and traces the
resulting program with `pie model trace` into a run directory named by the
mount's hash. The trainer and the server then read the same files.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

STANZAS = Path("lib/stanzas")
"""Where stanza files live, under the models directory."""
FORM = Path("lib/form.poem")
"""The mount file, under the models directory."""
PLATFORMS = ("metal", "cuda", "xla", "gpu", "vulkan")
"""The platforms a trace is lowered for, to prove every bank survives compilation."""


def seed(model) -> None:
    """Fill an empty models directory with the packages the pie binary ships
    (any one-shot pie command does this; `pie model list` is the cheapest)."""
    if not model.models.is_dir() or not any(model.models.iterdir()):
        if model.models != model.home / "models":
            raise FileNotFoundError(f"{model.models} is empty, and pie seeds only {model.home / 'models'}")
        model.pie("model", "list")


def source(arg: str, models: Path) -> tuple[str, Path]:
    """The stanza name and file an argument of `mount` names: a standard-library
    name (`lora`) or a path to a `.poem` file, named by its stem."""
    path = Path(arg) if arg.endswith(".poem") or "/" in arg else models / STANZAS / f"{arg}.poem"
    name = path.stem
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"stanza name `{name}` is not a lowercase identifier")
    if not path.is_file():
        raise FileNotFoundError(f"no stanza `{name}`: {path} does not exist")
    return name, path


def form(names: list[str], command: str) -> str:
    """The text of `lib/form.poem` for these stanzas, in mount order."""
    lines = [f"# Written by `{command}`. The next mount rewrites it; do not edit it by hand."]
    lines += [f'load("//lib/stanzas/{n}.poem", {n}_banks = "banks", {n}_apply = "apply")' for n in names]
    lines.append("STANZAS = [")
    lines += [f'    ("{n}", struct(banks = {n}_banks, apply = {n}_apply)),' for n in names]
    lines.append("]")
    return "\n".join(lines) + "\n"


def mount_hash(models: Path) -> str:
    """The first 8 hex digits of the sha256 over every `.poem` file under `models`."""
    h = hashlib.sha256()
    for f in sorted(models.rglob("*.poem")):
        h.update(f.relative_to(models).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:8]


def layer_fields(run_dir: Path) -> set[str]:
    """The names of a layer's own fields (its weights and caches) in a trace,
    which a stanza name must not take."""
    trace = json.loads((run_dir / "trace.json").read_text())
    names = [p["name"] for p in trace["params"] if p["source"] != "Registered"]
    names += [next(iter(c.values()))["name"] for c in trace["caches"]]
    return {n.split(".")[2] for n in names if n.startswith("layers.")}


def trace(model, out: Path) -> Path:
    """Run `pie model trace` for `model` into `out`, written whole or not at all."""
    part = out.with_name(out.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.parent.mkdir(parents=True, exist_ok=True)
    args = ["model", "trace", model.id, "--out", str(part), "--models-dir", str(model.models),
            "--lowered", ",".join(PLATFORMS)]
    if model.weights is not None:
        # import.json, how the trainer reads the frozen weights, is written against the checkpoint.
        args += ["--checkpoint", str(model.weights)]
    model.pie(*args)
    shutil.rmtree(out, ignore_errors=True)
    part.rename(out)
    return out


def mount(model, args: tuple[str, ...]) -> list[str]:
    """Mount the stanzas `args` name on `model` and trace the result; on a
    refusal or a failed trace the previous mount stays in place."""
    seed(model)
    picked = [source(a, model.models) for a in args]
    names = [n for n, _ in picked]
    if len(set(names)) != len(names):
        raise ValueError(f"a stanza is named twice: {names}")
    taken = layer_fields(model.run_dir()) & set(names)
    if taken:
        raise ValueError(f"stanza `{sorted(taken)[0]}` takes the name of a layer field; rename its file")
    before = {model.models / FORM: (model.models / FORM).read_bytes()}
    for name, path in picked:
        dest = model.models / STANZAS / f"{name}.poem"
        if path.resolve() != dest.resolve():
            before[dest] = dest.read_bytes() if dest.exists() else None
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    (model.models / FORM).write_text(form(names, " ".join(["stanza mount", *args])))
    try:
        model.run_dir()
    except Exception:
        for f, data in before.items():
            f.write_bytes(data) if data is not None else f.unlink()
        raise
    return names
