"""Every device verification, as one command with one exit code.

These are the checks that have caught every kernel bug in this project, and
they are not unit tests: each walks real documents through the device and
compares the mask word for word against the reference matcher, which is the
only thing that can see a mask that narrowed. A unit test asserts a property
someone thought of; these assert agreement with an independent implementation.

    python -m gpu_lr1.verify

`--quick` runs the smaller configuration of each, which is what CI does on
every push; the full run is what a release needs.

The six, and what each is the only one to see:

    device      the mask and the configuration set, over corpus documents
    walk        random walks, which reach states no corpus document does
    conflicts   the same, on the schemas that need GLR-lite forking
    mixed       many grammars in one batch, where the arena rebasing lives
    draft       speculative verification of a draft tree
    verdicts    the compile-time refusal table, re-derived from ACTION

`verdicts` is the one no other check could catch: the kernels trust that table,
so every device verification agrees with a wrong entry by construction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Module, the arguments for a full run, and the arguments for a quick one.
CHECKS = [
    ("gpu_lr1.verify.verdicts", [], []),
    ("gpu_lr1.verify.device", ["12"], ["4"]),
    ("gpu_lr1.verify.walk", ["6", "20", "40"], ["3", "5", "20"]),
    ("gpu_lr1.verify.conflicts", ["8", "200"], ["3", "60"]),
    ("gpu_lr1.verify.mixed", ["6", "24"], ["3", "8"]),
    ("gpu_lr1.verify.draft", ["6", "4", "8"], ["3", "2", "4"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="the smaller configuration of each, for CI")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these, by name")
    arguments = parser.parse_args()

    failed = []
    for module, full, quick in CHECKS:
        name = module.rsplit(".", 1)[1]
        if arguments.only and name not in arguments.only:
            continue
        argv = quick if arguments.quick else full
        print(f"=== {name} {' '.join(argv)}", flush=True)
        start = time.perf_counter()
        outcome = subprocess.run(
            [sys.executable, "-m", module, *argv], check=False
        )
        elapsed = time.perf_counter() - start
        status = "ok" if outcome.returncode == 0 else f"FAILED ({outcome.returncode})"
        print(f"--- {name} {status} in {elapsed:.1f}s\n", flush=True)
        if outcome.returncode != 0:
            failed.append(name)

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("all verifications passed")


if __name__ == "__main__":
    main()
