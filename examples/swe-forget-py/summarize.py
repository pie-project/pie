"""Summarize harness results: one row per mode, over the tasks every mode finished.

    python summarize.py results.jsonl [more.jsonl ...]
"""

import json
import sys
from collections import defaultdict


def main() -> None:
    rows = [json.loads(line) for path in sys.argv[1:] for line in open(path) if line.strip()]
    by_mode: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_mode[r["mode"]][r["task"]] = r
    common = set.intersection(*[set(t) for t in by_mode.values()]) if by_mode else set()
    print(f"{len(common)} tasks finished by every mode\n")
    header = ("mode", "solved", "gen tok/task", "prefill tok/task", "gen tok/solve", "s/task", "errors")
    print("{:<8} {:>7} {:>13} {:>16} {:>14} {:>8} {:>7}".format(*header))
    for mode, tasks in sorted(by_mode.items()):
        rs = [tasks[t] for t in common]
        n = len(rs) or 1
        solved = sum(1 for r in rs if r.get("solved"))
        gen = sum(r.get("decoded_tokens", 0) or 0 for r in rs)
        pre = sum(r.get("prefilled_tokens", 0) or 0 for r in rs)
        wall = sum(r.get("wall_seconds", 0) or 0 for r in rs)
        errors = sum(1 for r in rs if r.get("error"))
        print("{:<8} {:>7} {:>13.0f} {:>16.0f} {:>14.0f} {:>8.1f} {:>7}".format(
            mode, f"{solved}/{len(rs)}", gen / n, pre / n, gen / max(solved, 1), wall / n, errors))


if __name__ == "__main__":
    main()
