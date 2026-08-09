"""The gate-coverage audit: a hand-maintained crate list rots, quietly.

The `workspace-check` job lints and formats a LIST of crates rather than
the workspace, and a list is a claim about the workspace that nothing
checks. This tree has already paid for that claim twice.

The rename of 2026-08-08 collapsed six crate names onto three. It left
`-p tensor-compiler` in one clippy step FOUR TIMES, which is harmless in
itself -- but the reason it took months to notice is the same reason a
crate DROPPING OUT would have taken months to notice: nobody reads a
fifteen-entry `-p` list. When the duplication was finally found, the
first question was whether a crate had fallen out rather than merely
doubled, and answering it meant diffing the list against
`cargo metadata` by hand.

The second time was worse and silent. Sixteen of the twenty ungated
crates had drifted out of rustfmt -- `engine` by 146 hunks,
`kernels-cuda` by 144 -- because they had never been in the fmt list at
all. There was no signal. A crate that is not in the list is
indistinguishable, from CI's output, from a crate that passes.

So this is the check that makes the list say what it means: **every
workspace member is either gated or excluded on purpose.**

## Why exclusions are named here and not merely absent

An absent crate and an excluded crate look identical in `ci.yml` -- both
are "not in the `-p` list". The difference is that one is a decision and
the other is an oversight, and only one of them should survive a reader.
So an exclusion has to be written down WITH ITS REASON, and a crate that
is neither gated nor listed below is an error.

That also gives the exclusions an expiry: `EXCLUSIONS` says what would
have to change for the crate to join, so the entry is falsifiable rather
than permanent. The ones left are waiting on a toolkit, a GPU, a Mac, or
a rewrite that is actively in flight.

## Gates on other jobs count as gates

`STEPS` reads steps by name from the whole workflow, not from one job, so
a crate gated on the macOS runner is GATED and not merely excused.
`driver-metal` used to be an exclusion reading "gated on the macOS job
instead, in BOTH feature halves" -- a true sentence that nothing checked.
Deleting those two steps would have left the audit green while the crate
went unlinted, which is precisely the failure this file exists to catch,
one level up. Naming them makes the claim a check: remove either step and
the audit fails saying so.

Duplicates are therefore counted WITHIN a step rather than across them. A
crate named twice in one command is the bug that happened here; a crate
named in two steps is a crate linted in two feature configurations, and
`driver-metal` is deliberately in both halves because a lint that fires
in only one half is a lint nobody sees.

## What it does not check

That the gate PASSES -- CI runs the gate itself for that. This only
checks that the gate is asked about every crate. A gate that is asked
and fails is a red build; a gate that is never asked is a green one, and
the second is what this is for.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"

# Crates deliberately outside a gate, and what would have to change.
#
# The value is the reason. It is printed when the audit fails on a crate
# that IS listed here, which cannot happen -- but it is also the thing a
# reader comes here for, so it lives in the data rather than a comment.
EXCLUSIONS = {
    "fmt": {
        "driver-cuda": (
            "being rewritten wholesale (108 commits in three days); "
            "reformatting it would collide with in-flight work for no "
            "benefit to files that are being replaced. Add it when the "
            "rewrite lands."
        ),
        "driver-vulkan": (
            "same, and it is already rustfmt-clean -- so the only thing "
            "adding it buys today is a merge conflict. Add it when the "
            "rewrite lands."
        ),
        "kernels-cuda": (
            "144 drifted hunks against 57 commits in three days, five of "
            "them in the last day and none of them mine. `engine` carried "
            "this same reason until its churn turned out to have been my "
            "own and stopped; this one has not stopped. Re-measure with "
            "`git log --since='24 hours ago' -- crates/kernels-cuda`."
        ),
    },
    "clippy": {
        # Everything not yet at zero warnings. A gate is worth nothing
        # until the crate is clean, so the entry to remove here is the
        # last warning rather than the line. Counts are unique warning
        # SITES from a cold `cargo clean -p <crate>` -- a warm clippy run
        # replays nothing and reports zero, which is how several of these
        # looked clean for months.
        "driver-cuda": "needs nvcc, and is being rewritten",
        "driver-vulkan": "needs glslc, and is being rewritten",
        "kernels-cuda": "needs nvcc",
        "kernels-metal": "needs a Mac",
        "kernels-vulkan": "needs glslc",
        "pie-gpu-tests": "needs a GPU to be worth compiling",
        "pie-server-py": "a pyo3 extension; built by maturin, not by this job",
    },
}

# The step names carrying each gate's `-p` list, in ci.yml.
STEPS = {
    "fmt": ["cargo fmt (compiler crates)"],
    "clippy": [
        "cargo clippy (deny warnings)",
        "cargo clippy (model, deny warnings)",
        # On the macOS job, because a lint can depend on `cfg(target_os)`
        # and only a Mac can ask the question this crate needs asked. Both
        # halves, because a lint that fires in one half is a lint nobody
        # sees. Named here so the claim is CHECKED: while `driver-metal`
        # was an exclusion saying "gated on the macOS job instead", nothing
        # verified that those steps still existed.
        "cargo clippy (driver-metal, portable half)",
        "cargo clippy (driver-metal, metal-4)",
    ],
}


def members():
    """Every crate Cargo considers part of this workspace."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {p["name"] for p in json.loads(out.stdout)["packages"]}


def steps_by_name():
    """`ci.yml`'s `run:` bodies, keyed by step name.

    Parsed by indentation rather than with `yaml`, so that the audit needs
    nothing installed to answer -- refusing to answer for want of a
    dependency is the failure mode it exists to prevent, and an audit that
    takes a different code path depending on an optional import is two
    audits with one name.

    A step ends at the next non-blank line indented no further than its own
    `- name:`, which is where the next step or the next job key begins. The
    comment blocks in this workflow sit at exactly that indentation, so a
    step's body stops before the prose introducing the next one -- which
    matters, because that prose quotes `-p` lists.
    """
    steps = {}
    lines = CI.read_text().split("\n")
    index = 0
    while index < len(lines):
        head = re.match(r"^(\s*)- name:\s*(.+?)\s*$", lines[index])
        if not head:
            index += 1
            continue
        indent = len(head.group(1))
        body = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            index += 1
        block = "\n".join(body)
        # Everything after the `run:` key, whether the command is inline
        # on that line or a `|` block beneath it -- slicing past the whole
        # LINE would drop a one-line command, which is how the `model`
        # clippy step first read as ungated.
        run = re.search(r"^[ \t]*run:[ \t]*\|?[ \t]*", block, re.M)
        steps[head.group(2)] = block[run.end() :] if run else ""
    return steps


def listed(body):
    """The crates a `-p`-style command names, and how often."""
    counts = {}
    for crate in re.findall(r"-p\s+([A-Za-z0-9_-]+)", body):
        counts[crate] = counts.get(crate, 0) + 1
    return counts


def main():
    all_members = members()
    bodies = steps_by_name()
    problems = []

    for gate, step_names in STEPS.items():
        gated = {}
        for step in step_names:
            if step not in bodies:
                problems.append(
                    f"{gate}: ci.yml has no step named {step!r}. This audit "
                    f"reads the step by name, so a rename silently stops it "
                    f"checking anything -- update STEPS in this file."
                )
                continue
            for crate, count in listed(bodies[step]).items():
                # Counted WITHIN a step, not across them. A crate named
                # twice in one command is the bug that happened here; a
                # crate named in two steps is a crate gated in two feature
                # configurations, which is the point of having two.
                if count > 1:
                    problems.append(
                        f"{gate}: `-p {crate}` appears {count} times in "
                        f"{step!r}. Harmless to run and a sign nobody is "
                        f"reading the list -- which is how a crate goes "
                        f"missing from it."
                    )
                gated[crate] = gated.get(crate, 0) + count

        for crate in sorted(gated):
            if crate not in all_members:
                problems.append(
                    f"{gate}: `-p {crate}` is not a workspace member. It was "
                    f"renamed or removed, and the gate has been silently "
                    f"narrower ever since."
                )

        excluded = EXCLUSIONS.get(gate, {})
        for crate in sorted(excluded):
            if crate not in all_members:
                problems.append(
                    f"{gate}: {crate} is excluded but is not a workspace "
                    f"member. Drop the entry."
                )
            if crate in gated:
                problems.append(
                    f"{gate}: {crate} is both gated and excluded. The "
                    f"exclusion is stale -- drop it."
                )

        for crate in sorted(all_members - set(gated) - set(excluded)):
            problems.append(
                f"{gate}: {crate} is neither gated nor excluded. Add it to "
                f"the {gate} step in ci.yml, or to EXCLUSIONS['{gate}'] in "
                f"this file WITH THE REASON. A crate that is merely absent "
                f"looks exactly like a crate that passes."
            )

    if problems:
        print("ci-gate-audit: the gate lists do not describe the workspace\n")
        for problem in problems:
            print(f"  * {problem}")
        print(f"\n{len(problems)} problem(s).")
        return 1

    for gate in STEPS:
        covered = len(all_members) - len(EXCLUSIONS.get(gate, {}))
        print(
            f"ci-gate-audit: {gate} covers {covered}/{len(all_members)} "
            f"crates, {len(EXCLUSIONS.get(gate, {}))} excluded on purpose."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
