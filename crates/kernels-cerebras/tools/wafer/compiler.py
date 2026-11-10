"""compiler.py: the plan the Rust `wafer-compile` writes, for the Python preps (the tiny fixtures' and the
families the engine does not cut itself yet).

  plan(out, model=None, shape=None, preset=None, set="", positions=None)   wafer-compile plan: <out>/plan.json, tiling.json
  tiling(plan_dir)                                                        <plan_dir>/tiling.json, as attributes

The binary: $WAFER_COMPILE, else `wafer-compile` on PATH, else the crate's cargo target."""
import json, os, shutil, subprocess

CRATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def _cmd():
    if os.environ.get("WAFER_COMPILE"):
        return [os.environ["WAFER_COMPILE"]]
    if shutil.which("wafer-compile"):
        return ["wafer-compile"]
    return ["cargo", "run", "-q", "--release", "--manifest-path", os.path.join(CRATE, "Cargo.toml"),
            "--bin", "wafer-compile", "--"]


def plan(out, model=None, shape=None, preset=None, set="", positions=None):
    """wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset ..] [--set ..] [--positions ..]"""
    a = _cmd() + ["plan", "--out", os.path.abspath(out)]
    a += ["--model", model] if model else ["--shape", os.path.abspath(shape)]
    if preset:
        a += ["--preset", preset]
    if set:
        a += ["--set", set]
    if positions:
        a += ["--positions", str(positions)]
    subprocess.run(a, check=True)
    return out


class Named:
    """A JSON object's fields as attributes."""

    def __init__(self, d):
        self.__dict__.update(d)


def tiling(plan_dir):
    """What a prep tiles the checkpoint by (wafer-compile plan's tiling.json): the plan's cut in the prep's terms (the
    unit shape, the head, Gemma's PLE storage columns, Gemma 12B's row groups and every row's dims) and the preset."""
    return Named(json.load(open(os.path.join(plan_dir, "tiling.json"))))
