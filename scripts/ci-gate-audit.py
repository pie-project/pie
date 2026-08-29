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
crates had drifted out of rustfmt -- `runtime` by 146 hunks, and by 144 the
ahead-of-time CUDA archive crate that has since been deleted -- because
they had never been in the fmt list at
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

`engine-metal` was the case that shaped this and it has now been through
the whole cycle. It began as an exclusion reading "gated on the macOS job
instead, in BOTH feature halves" -- a true sentence that nothing checked,
so deleting those two steps would have left this audit green while the
crate went unlinted. Naming them in `STEPS` turned the sentence into a
check. Then R3 put the crate out of the workspace and deleted the macOS
job, and the check fired from the other direction: the step names were
gone and the audit said so by name. Both failure modes it was built for
have now happened to the same crate, and it reported each.

Duplicates are therefore counted WITHIN a step rather than across them. A
crate named twice in one command is the bug that happened here; a crate
named in two steps is a crate linted in two feature configurations, which
is why two steps are allowed to name one -- a lint that fires in only one
feature half is a lint nobody sees.

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

# The engine-flavor and probe chain: an inner library, and the package that
# must re-declare its features for them to be reachable from a build command.
# `pie` is where the chain has to end, because it is the only member with a
# `[[bin]]`.
FORWARDS = [("runtime", "worker"), ("worker", "pie")]

# Features deliberately not forwarded, and why. Same contract as EXCLUSIONS:
# the reason lives in the data, because the next reader's question is "was
# this decided or forgotten?".
FORWARD_EXCLUSIONS = {
    ("worker", "pie"): {
        "nixl": (
            "`transport`'s NIXL engine is a stub -- `crates/transport/src/"
            "lib.rs` says so and `Error` has a variant for being asked for "
            "it in a build that lacks it. Forwarding it would put a "
            "selectable flag on the CLI that turns on nothing a user can "
            "reach. Forward it when the engine exists."
        ),
    },
}

# Crates deliberately outside a gate, and what would have to change.
#
# The value is the reason. It is printed when the audit fails on a crate
# that IS listed here, which cannot happen -- but it is also the thing a
# reader comes here for, so it lives in the data rather than a comment.
EXCLUSIONS = {
    "fmt": {
        "engine-cuda": (
            "being rewritten wholesale (108 commits in three days); "
            "reformatting it would collide with in-flight work for no "
            "benefit to files that are being replaced. Add it when the "
            "rewrite lands."
        ),
        # The ahead-of-time archive crate had an entry of its own here --
        # "144 drifted hunks against 57 commits in three days" -- and it was
        # dropped when the crate was deleted at `85c6c674b`, because an
        # exclusion naming a non-member is what the check below refuses. The
        # JIT crate that has its name now is the entry that remains.
        "kernels-cuda": (
            "1,320 drifted hunks against 47 commits in three days, three "
            "of them in the last day: the crate the CUDA rewrite moved "
            "INTO, so it is the churn itself rather than a crate that "
            "drifted once. Re-measure with `cargo fmt --check -p "
            "kernels-cuda | grep -c '^Diff'` and `git log "
            "--since='24 hours ago' -- crates/kernels-cuda`."
        ),
    },
    "clippy": {
        # Everything not yet at zero warnings. A gate is worth nothing
        # until the crate is clean, so the entry to remove here is the
        # last warning rather than the line. Counts are unique warning
        # SITES from a cold `cargo clean -p <crate>` -- a warm clippy run
        # replays nothing and reports zero, which is how several of these
        # looked clean for months.
        "engine-cuda": "needs nvcc, and is being rewritten",
        # `engine-vulkan` STOOD HERE reading "needs slangc on the runner;
        # zero warnings otherwise", and the entry outlived its subject the
        # way an exclusion always can: R3 put the crate in the root
        # manifest's `exclude`, and an exclusion naming a NON-MEMBER is what
        # the check below refuses -- correctly, because it reads as a crate
        # somebody decided not to lint rather than as a crate cargo can no
        # longer see. `engine-metal` and `engine-wgpu` went the same way and
        # were never entries here at all: they were named in the gate lists,
        # which is why they surfaced as `-p ... is not a workspace member`
        # instead. Both halves of this file caught the same deletion from
        # opposite sides, which is the point of having both.
        #
        # None of the three comes back as an exclusion. They come back as
        # members, at P5, and then the question is whether they are gated.
        # `kernels-cuda` STOOD HERE reading "52 warnings, and the rewrite is
        # landing in it". Both halves expired. The crate is named by the
        # clippy step in `ci.yml` AND was excluded here, which is the one
        # combination this audit calls out by itself -- an exclusion that
        # excludes nothing still reads as a crate nobody lints. And the 52
        # warnings are gone: the crate passes this gate's exact flags at
        # zero, verified by running the step. The rewrite landed, and the
        # exclusion outlived it.
        # `kernels-metal` and `kernels-vulkan` STOOD HERE, reading "needs a
        # Mac" and "needs slangc on the runner". Neither claim survives:
        # this gate names both crates, and neither toolchain is needed to
        # CHECK one -- `kernels-vulkan` only shells out to `slangc` under
        # `native`, which is off by default. `kernels-vulkan` is clean under
        # this gate's exact flags.
        #
        # `kernels-metal` is not, and that is a real failure rather than a
        # missing exclusion: its LIB is clean, but `--all-targets` does not
        # compile on a non-Mac toolchain (6 errors, E0061/E0308, in the lib
        # test and `tests/entrypoints.rs`). Re-excluding it here would turn
        # a red gate into a silent one, which is the trade this whole file
        # exists to refuse.
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
        # `cargo clippy (engine-metal, portable half)` and `(engine-metal,
        # metal-4)` STOOD HERE, on the macOS job, because a lint can depend
        # on `cfg(target_os)` and only a Mac can ask that question in both
        # feature halves. Naming them here is what turned "gated on the macOS
        # job instead" from a sentence into a check, and the check then did
        # its job in the direction nobody expected: R3 deleted the macOS job
        # with the crate, and this list said so by name rather than going
        # quietly narrower.
        #
        # They return at P5 with the job. A step named here that ci.yml does
        # not have is an error, so this list cannot be restored ahead of the
        # steps it claims.
    ],
}


def members():
    """Every crate Cargo considers part of this workspace."""
    return {name for name in packages()}


def packages():
    """`name -> declared features` for every workspace member."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {p["name"]: p["features"] for p in json.loads(out.stdout)["packages"]}


def forward_problems(pkgs):
    """Features that stop at a library instead of reaching the binary.

    A feature is only real if the package that produces a BINARY can select
    it. `runtime` and `worker` are libraries, so every flavor and probe they
    offer has to be re-declared twice on the way out -- by hand, in three
    manifests, with nothing checking that the copies agree.

    They have not agreed. The root package lost its `engine-metal` forward
    when the C++ Metal driver was retired and never got it back, leaving
    five dead `#[cfg]` blocks in `pie init`'s config template (see
    `scripts/cfg-feature-audit.py`, which catches the OTHER half of that
    bug: the `cfg` naming a feature nobody declares). `profile-hot-path`
    and `profile-all` reached `worker` and stopped. `nixl` still does, and
    is named below so that it stops being an accident.

    Read out of `cargo metadata`, so this sees what Cargo resolved rather
    than what a manifest appears to say.
    """
    problems = []
    for inner, outer in FORWARDS:
        forwarded = {
            edge.split("/", 1)[1]
            for values in pkgs[outer].values()
            for edge in values
            if edge.startswith(f"{inner}/")
        }
        allowed = FORWARD_EXCLUSIONS.get((inner, outer), {})
        for feature in sorted(pkgs[inner]):
            if feature == "default" or feature in forwarded:
                continue
            if feature in allowed:
                continue
            problems.append(
                f"forward: `{inner}/{feature}` is not forwarded by "
                f"`{outer}`, so no build of `{outer}` can select it. Add "
                f"the forward, or add it to FORWARD_EXCLUSIONS[({inner!r}, "
                f"{outer!r})] in this file WITH THE REASON."
            )
        for feature in sorted(allowed):
            if feature not in pkgs[inner]:
                problems.append(
                    f"forward: `{inner}/{feature}` is excused from being "
                    f"forwarded but `{inner}` no longer declares it. Drop "
                    f"the entry."
                )
            elif feature in forwarded:
                problems.append(
                    f"forward: `{inner}/{feature}` is excused from being "
                    f"forwarded and IS forwarded by `{outer}`. The "
                    f"exclusion is stale -- drop it."
                )
    return problems


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
    pkgs = packages()
    all_members = set(pkgs)
    bodies = steps_by_name()
    problems = forward_problems(pkgs)

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
