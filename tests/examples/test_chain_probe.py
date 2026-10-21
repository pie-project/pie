"""A pass fed its tokens on the device by the pass ahead of it, end to end.

`chain-probe` runs rounds of two passes: one picks a token from the host's
anchor, the next embeds that pick `width` times. With `--chain` the two are
one frame and the pick never reaches the host before the second pass runs;
without, the host reads it and hands it back. The claim is that the two write
the same tokens, and that the chained run submits half the frames.

    uv run python tests/examples/test_chain_probe.py --config ~/.pie/config.toml

Like `test_frames.py` this drives the `pie` binary, so the engine boots with
the model the config names; any text model serves.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERLETS_DIR = REPO_ROOT / "examples"
NAME = "chain-probe"


def build_guest() -> Path:
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        subprocess.run(
            ["cargo", "build", "-p", NAME, "--target", "wasm32-wasip3", "--release"],
            cwd=INFERLETS_DIR,
            check=True,
        )
    wasm = INFERLETS_DIR / "target" / "wasm32-wasip3" / "release" / f"{NAME.replace('-', '_')}.wasm"
    if not wasm.exists():
        raise FileNotFoundError(f"no guest at {wasm}; build it or unset PIE_INFERLETS_NO_BUILD")
    return wasm


def run(binary: Path, config: Path | None, wasm: Path, chain: bool, rounds: int, width: int) -> dict:
    cmd = [str(binary)]
    if config is not None:
        cmd += ["--config", str(config)]
    cmd += ["run", "-p", str(wasm), "--", "--rounds", str(rounds), "--width", str(width)]
    if chain:
        cmd.append("--chain")
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` exited {done.returncode}:\n{done.stderr[-2000:]}")
    line = next(
        (line for line in done.stdout.splitlines() if line.startswith("{")),
        None,
    )
    if line is None:
        raise RuntimeError(f"no answer in the output:\n{done.stdout[-2000:]}")
    return json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pie", type=Path, default=REPO_ROOT / "target" / "release" / "pie")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--width", type=int, default=3)
    args = parser.parse_args()

    wasm = build_guest()
    host = run(args.pie, args.config, wasm, False, args.rounds, args.width)
    chained = run(args.pie, args.config, wasm, True, args.rounds, args.width)

    failures = []
    if chained["tokens"] != host["tokens"]:
        failures.append(f"tokens differ:\n  host    {host['tokens']}\n  chained {chained['tokens']}")
    if len(host["tokens"]) != args.rounds * (args.width + 1):
        failures.append(f"{len(host['tokens'])} tokens for {args.rounds} rounds of {args.width + 1}")
    if chained["frames"] != args.rounds or host["frames"] != 2 * args.rounds:
        failures.append(f"frames: chained {chained['frames']}, host {host['frames']}")
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print(f"ok: {args.rounds} chained rounds wrote what the host-fed ones wrote")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
