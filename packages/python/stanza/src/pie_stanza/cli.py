"""The `stanza` command: mount, check, train, serve, evaluate and measure skew from a shell.

Every subcommand takes `--pie-home` (default `$PIE_HOME`, else
`~/.pie-stanza`), `--model` (default `qwen3-0.6b`) and `--weights`, the
Hugging Face checkpoint (default `$STANZA_WEIGHTS`). `stanza check` runs its
weight checks only when `--weights` is given. `stanza train` streams one JSON
line per optimizer step to stdout; with `--loss dpo` it reads preference
pairs (OpenAI's preference JSONL) and trains against the plain arm. `eval` and `skew` talk to a server that
`stanza serve` started, and take the model from the adapter's `stanza.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

from .adapter import Adapter
from .data import chat, pairs
from .evaluate import evaluate, json_exact
from .model import Model
from .serve import Pie
from .skew import skew

ROWS = 8
"""Rows per training batch."""


def parser() -> argparse.ArgumentParser:
    """The command line `stanza` reads."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pie-home", default=os.environ.get("PIE_HOME") or "~/.pie-stanza")
    common.add_argument("--model", default="qwen3-0.6b")
    common.add_argument("--weights")
    top = argparse.ArgumentParser(prog="stanza", description="Poem-programmable post-training on pie.")
    sub = top.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mount", parents=[common], help="mount stanzas on the model program")
    p.add_argument("stanzas", nargs="*", help="standard-library names or .poem files, in order")
    sub.add_parser("check", parents=[common], help="print where each mounted stanza attaches")
    p = sub.add_parser("train", parents=[common], help="train one mounted stanza on chat or preference JSONL")
    p.add_argument("--stanza", required=True)
    p.add_argument("--loss", choices=["cross_entropy", "dpo"], default="cross_entropy")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", required=True)
    for name, text in [("serve", "serve the mounted program until stopped"),
                       ("eval", "score an adapter's replies on chat JSONL"),
                       ("skew", "compare trained and served log-probs")]:
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("--port", type=int, default=18431)
        if name != "serve":
            p.add_argument("--adapter", required=True)
            p.add_argument("--data", required=True)
    sub.choices["skew"].add_argument("--n", type=int, default=32)
    return top


def main(argv: list[str] | None = None) -> int:
    """Run one `stanza` subcommand; returns the exit code."""
    args = parser().parse_args(argv)
    weights = args.weights or os.environ.get("STANZA_WEIGHTS")
    try:
        if args.cmd in ("eval", "skew"):
            adapter = Adapter.load(args.adapter)
            args.model = adapter.meta["model"]
        model = Model(args.model, weights, pie_home=args.pie_home)
        if args.cmd == "mount":
            print("mounted " + " ".join(model.mount(*args.stanzas)))
        elif args.cmd == "check":
            report = model.check(weights=args.weights is not None)
            print(report)
            return 0 if report.ok else 1
        elif args.cmd == "train":
            trainer = model.trainer(args.stanza)
            data = (pairs if args.loss == "dpo" else chat)(args.data, model)
            for batch in data.batches(rows=ROWS, epochs=args.epochs, seed=0):
                trainer.forward_backward(batch, args.loss)
                trainer.optim_step(lr=args.lr)
                print(json.dumps(trainer.log[-1]), flush=True)
            saved = trainer.save(args.out, name=Path(args.out).name)
            print(f"saved {saved.path} ({saved.id})", file=sys.stderr)
        elif args.cmd == "serve":
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            with model.serve(port=args.port):
                print(f"serving {model.id} at ws://127.0.0.1:{args.port}; Ctrl-C stops it", flush=True)
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    pass
        elif args.cmd == "eval":
            pie = Pie(f"ws://127.0.0.1:{args.port}", model)
            print(json.dumps(evaluate(pie, adapter, chat(args.data, model), json_exact)))
        elif args.cmd == "skew":
            trainer = model.trainer(adapter.stanza)
            trainer.load(adapter)
            pie = Pie(f"ws://127.0.0.1:{args.port}", model)
            measured = skew(trainer, pie, chat(args.data, model).sample(args.n)).summary()
            meta = adapter.meta | {"skew": measured}
            (adapter.path / "stanza.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(json.dumps(measured))
    except (ValueError, FileNotFoundError, RuntimeError) as why:
        print(f"stanza: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
