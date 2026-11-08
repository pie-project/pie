"""Tasks for the swarm: small local repositories with a bug and a hidden test,
and SWE-bench instances. A task is a repository to copy, an issue to show the
agent, and a command whose exit code says whether a patch fixed it."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    id: str
    issue: str
    setup: callable  # (dir) -> None: lays the repository out in an empty directory
    test: str  # run in a patched copy; exit 0 means solved
    hidden: dict  # file -> text, written only for the test


def git_init(path: Path) -> None:
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=path, check=True)


def files(tree: dict):
    def setup(path: Path) -> None:
        for name, text in tree.items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(text)
        git_init(path)
    return setup


TOY = [
    Task(
        "toy-duration",
        "parse_duration in timefmt/duration.py is wrong for some inputs: parse_duration('45m') "
        "returns 2700 as expected, but parse_duration('2h') returns 2 and parse_duration('1h30m') "
        "returns 1830. It should return seconds for any mix of h, m and s units, like '1h2m3s'.",
        files({
            "timefmt/__init__.py": "from .duration import parse_duration\n",
            "timefmt/duration.py": (
                "import re\n\n"
                "UNITS = {'h': 1, 'm': 60, 's': 1}\n\n\n"
                "def parse_duration(text):\n"
                "    \"\"\"'1h30m' -> 5400 seconds.\"\"\"\n"
                "    total = 0\n"
                "    for amount, unit in re.findall(r'(\\d+)([hms])', text):\n"
                "        total += int(amount) * UNITS[unit]\n"
                "    return total\n"
            ),
            "README.md": "timefmt: small time helpers.\n",
        }),
        "python3 hidden_test.py",
        {"hidden_test.py": (
            "from timefmt import parse_duration as p\n"
            "assert p('2h') == 7200, p('2h')\nassert p('1h30m') == 5400\nassert p('45m') == 2700\n"
            "assert p('1h2m3s') == 3723\nassert p('90s') == 90\nprint('ok')\n"
        )},
    ),
    Task(
        "toy-slugify",
        "textkit.slugify leaves repeated and trailing dashes: slugify('Hello,  World!') gives "
        "'hello---world-' but should give 'hello-world'. Punctuation and whitespace runs should "
        "become a single dash, with none at the ends.",
        files({
            "textkit/__init__.py": "from .slug import slugify\n",
            "textkit/slug.py": (
                "def slugify(text):\n"
                "    out = []\n"
                "    for ch in text.lower():\n"
                "        out.append(ch if ch.isalnum() else '-')\n"
                "    return ''.join(out)\n"
            ),
        }),
        "python3 hidden_test.py",
        {"hidden_test.py": (
            "from textkit import slugify as s\n"
            "assert s('Hello,  World!') == 'hello-world', s('Hello,  World!')\n"
            "assert s('  a b ') == 'a-b'\nassert s('Already-good') == 'already-good'\n"
            "assert s('x--y') == 'x-y'\nprint('ok')\n"
        )},
    ),
    Task(
        "toy-paging",
        "store.paging.page(items, number, size) skips an item between pages: with items 0..9 and "
        "size 3, page 1 is [0, 1, 2] but page 2 is [4, 5, 6] instead of [3, 4, 5]. The last page "
        "should hold whatever is left, and a page past the end should be empty. Pages count from 1.",
        files({
            "store/__init__.py": "",
            "store/paging.py": (
                "def page(items, number, size):\n"
                "    start = (number - 1) * (size + 1)\n"
                "    return items[start:start + size]\n\n\n"
                "def pages(items, size):\n"
                "    return (len(items) + size - 1) // size\n"
            ),
        }),
        "python3 hidden_test.py",
        {"hidden_test.py": (
            "from store.paging import page, pages\n"
            "xs = list(range(10))\nassert page(xs, 2, 3) == [3, 4, 5], page(xs, 2, 3)\n"
            "assert page(xs, 4, 3) == [9]\nassert page(xs, 5, 3) == []\nassert pages(xs, 3) == 4\n"
            "print('ok')\n"
        )},
    ),
]


def swe_bench(path: str, ids: list[str] | None = None) -> list[dict]:
    """SWE-bench rows from a JSONL export (instance_id, problem_statement, ...)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if not ids or r["instance_id"] in ids]
