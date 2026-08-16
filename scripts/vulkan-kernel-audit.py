#!/usr/bin/env python3
"""The Vulkan shader tree's entrypoint census.

`crates/kernels-vulkan/src/` says what the table DECLARES; this says what the
tree INSTANTIATES, and `tests/entrypoints.rs` is where the two are compared.
Invariant (1) of the Metal refactor note, restated for this backend:

    every entrypoint in `kernels/` resolves to exactly one (row, axis point),
    and every (row, axis point) to exactly one entrypoint

The census is cheaper here than on the Metal side and the reason is worth
stating. A Metal entrypoint is a `[[host_name(...)]]` produced by expanding an
`instantiate_*` macro, so counting them needs a C preprocessor -- which is why
`metal-kernel-audit.py` runs one. A Slang compute shader has exactly one entry
point and it is always `main`; what distinguishes two variants is the `-D` set
and the `.spv` they are written to. So the variant list cannot be implicit in
the source: it has to be DECLARED, and this reads the declarations.

    // pie:instantiate rms_single_row_bfloat16 N_READS=4

`build.rs` reads the same lines and compiles exactly them. That is deliberate --
an audit that derived the set differently from the build could pass while the
build produced something else -- and it is why `--compile` exists here: running
slangc over every declared variant is what proves a declaration is a variant that
builds, which is the half a set comparison cannot see.

There was a third mode, `--bindings`, and what it covered is now uncovered. A
shader's `binding = N` is a FUNCTION of its row -- `kernels_vulkan::bindings`
numbers buffer-kinded operands densely from zero in operand order and sends
scalars to a push block -- and a wrong one is silent, because Vulkan does not
report a descriptor that was declared and never written. Sixty entrypoints were
reading unwritten descriptors when that check was first run. It got the row's
half by running `crates/kernels-vulkan/examples/dump_layout.rs`, which is
deleted with the rest of `examples/`, so the mode is deleted rather than
repointed. Binding drift and push-field reordering are now caught by a device
test producing wrong numbers, or not at all.

Usage:
    vulkan-kernel-audit.py            # print the census
    vulkan-kernel-audit.py --compile  # additionally run slangc over every variant
    vulkan-kernel-audit.py --defines  # fail on an instantiate key no body reads
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "kernels-vulkan"
SHADERS = CRATE / "kernels"

MARKER = "pie:instantiate"

# The capability tiers, best first. This list and `Capability` in
# `src/capability.rs` are the same vocabulary; `build.rs` reads the Rust one
# through `#[path]` and this reads its own copy, so a tier added there and not
# here shows up as a parse failure rather than as a module the build writes and
# the census cannot see.
TIERS = ("coopmat", "fp16", "baseline")
BASELINE = "baseline"


@dataclass(frozen=True)
class Variant:
    """One `(shader, tier, define set)` triple -- one SPIR-V module.

    A tier is an ADDITIONAL module for an entrypoint that already exists, never
    a new entrypoint: the census counts entrypoints, and adding a `@coopmat`
    body must leave that count alone. See `src/capability.rs`.
    """

    entrypoint: str
    file: Path
    tier: str
    defines: tuple[tuple[str, str], ...]
    line: int

    @property
    def where(self) -> str:
        return f"{self.file.relative_to(SHADERS)}:{self.line}"

    @property
    def module(self) -> str:
        """The `.spv` name a `native` build writes.

        Baseline is unsuffixed so a driver that has never heard of tiers reads
        the right file knowing only the entrypoint.
        """
        return f"{self.entrypoint}.spv" if self.tier == BASELINE \
            else f"{self.entrypoint}.{self.tier}.spv"


def directives(path: Path) -> list[Variant]:
    out: list[Variant] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped.startswith("//"):
            continue
        rest = stripped[2:].lstrip()
        if not rest.startswith(MARKER):
            continue
        words = rest[len(MARKER):].split()
        if not words:
            die(f"{path}:{lineno}: a `{MARKER}` names an entrypoint first")

        # The tier is optional and follows the name. Absent means baseline, so
        # every directive written before tiers existed keeps its meaning.
        rest_words = words[1:]
        tier = BASELINE
        if rest_words and rest_words[0].startswith("@"):
            tier = rest_words[0][1:]
            if tier not in TIERS:
                die(f"{path}:{lineno}: `@{tier}` is not a capability tier; "
                    f"expected one of {', '.join(sorted(TIERS))}")
            rest_words = rest_words[1:]

        defines = []
        for word in rest_words:
            if "=" not in word:
                die(f"{path}:{lineno}: `{word}` is not a KEY=VALUE define")
            key, value = word.split("=", 1)
            defines.append((key, value))
        out.append(Variant(words[0], path, tier, tuple(defines), lineno))
    return out


def census() -> dict[tuple[str, str], Variant]:
    """Every variant the tree declares, keyed by `(entrypoint, tier)`.

    A duplicate is fatal rather than absorbed: two directives claiming one name
    AT ONE TIER would have the second silently overwrite the first's `.spv`,
    which is this backend's spelling of the duplicate `host_name` the Metal
    tree's `no_two_rows_claim_the_same_entrypoint` exists to catch. Across
    different tiers the same name is expected -- that is what a tier IS.
    """
    seen: dict[tuple[str, str], Variant] = {}
    sources = sorted(SHADERS.rglob("*.slang"), key=lambda p: str(p))
    for path in sources:
        for variant in directives(path):
            key = (variant.entrypoint, variant.tier)
            prior = seen.get(key)
            if prior is not None:
                die(
                    f"`{variant.entrypoint}` is instantiated twice at tier "
                    f"`{variant.tier}`: {prior.where} and {variant.where}"
                )
            seen[key] = variant
    if not seen:
        die(f"no `{MARKER}` directive under {SHADERS}")

    # A tier is a faster answer to a question the baseline already answers. A
    # tiered module with no baseline would be an entrypoint that exists only on
    # some devices -- exactly what the mechanism is built to prevent.
    for entrypoint, tier in seen:
        if tier != BASELINE and (entrypoint, BASELINE) not in seen:
            die(
                f"`{entrypoint}` is instantiated at tier `{tier}` with no "
                f"baseline ({seen[(entrypoint, tier)].where}); every entrypoint "
                f"must resolve on a device with no optional features"
            )
    return seen


def compile_all(variants: dict[tuple[str, str], Variant]) -> int:
    """slangc over every declared variant, every tier. Returns the failure count."""
    slangc = os.environ.get("PIE_SLANGC") or shutil.which("slangc")
    if slangc is None:
        die("no slangc on PATH; set PIE_SLANGC or install shader-slang/slang")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for (name, tier), variant in sorted(variants.items()):
            cmd = [
                slangc,
                "-target", "spirv",
                "-stage", "compute",
                "-entry", "main",
                "-fvk-use-entrypoint-name",
                "-emit-spirv-directly",
                "-allow-glsl",
            ]
            # spirv-opt is skipped for coopmat, matching `build.rs` and the
            # llama.cpp finding it borrows (ggml #15344): spirv-opt miscompiles
            # cooperative-matrix shaders. Auditing with flags the build does not
            # use would prove the wrong thing.
            cmd.append("-O0" if tier == "coopmat" else "-O2")
            cmd += [
                "-I",
                str(SHADERS),
                f"-DPIE_ENTRYPOINT={name}",
            ]
            cmd += [f"-D{k}={v}" for k, v in variant.defines]
            cmd += ["-o", str(Path(tmp) / variant.module), str(variant.file)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                failures += 1
                print(f"FAIL {name} @{tier} ({variant.where})", file=sys.stderr)
                print(done.stderr.rstrip(), file=sys.stderr)
    return failures


def die(message: str) -> None:
    print(f"vulkan-kernel-audit: {message}", file=sys.stderr)
    raise SystemExit(1)


# A define an `instantiate` line sets and no body reads produces a module that
# is byte-identical to its sibling under a name that promises otherwise. The
# ones below are known and each has a reason; anything NEW is a defect.
INERT_DEFINES = {
    # Named for the fp16 activation path `quant/qmm_t.slang` really has (a
    # separate pre-cast buffer at binding 7). The routed file never grew one,
    # so these nine modules are copies of their non-fp16 siblings -- measured,
    # not argued: `kernels-vulkan`'s
    # `the_routed_fp16_modules_are_their_bf16_siblings_under_another_name`
    # compares the compiled bytes and they are identical. Harmless
    # only because `affine_qmm_t_routed_fp16`'s row is UNSTATED and so cannot
    # be bound at all; if it is ever stated, the body has to exist first or a
    # driver will hand fp16 bytes to a shader reading bf16.
    ("moe/qmm_t_routed.slang", "PIE_FP16"),
    # The encode path is selected by `PIE_MXFP4`, not by this.
    ("quant/transcode.slang", "PIE_ENCODE"),
    # ABI-name-only points: the entrypoint name carries the axis, the body
    # does not vary on it.
    ("quant/qmv.slang", "PIE_K_LANES"),
    ("quant/qmm_t.slang", "PIE_PROBE_SHAPE"),
    # `_p32_sg8` is a byte-for-byte duplicate of `_p32`. On METAL this same
    # point is a real 256-thread threadgroup whose shared arrays are sized for
    # eight simdgroups; here it is a name only, and the two directions of that
    # divergence are measured by `kernels-vulkan`'s
    # `the_page_shape_tails_are_one_real_variant_and_one_bare_name`. Survivable
    # because no head dim in `deployment::ATTN_HEAD_DIMS` spells a page tail,
    # so nothing can select either one.
    ("attn/sdpa_paged.slang", "PIE_SHORT_GROUP"),
}


def check_defines(variants) -> int:
    """Every key an `instantiate` line sets should change the module it makes.

    A define nothing reads is not merely tidy-up: it is a name that claims a
    behaviour the module does not have, and the two entrypoints it separates
    compile to the same bytes. `PIE_FP16` is exactly that, over nine routed
    GEMM names.

    Text is enough here and preprocessing is not wanted -- the question is
    whether the FILE mentions the macro anywhere, not whether one particular
    variant took that branch.
    """
    seen = {}
    for variant in variants.values():
        for key, _ in variant.defines:
            seen.setdefault(variant.file, set()).add(key)

    failures = 0
    for file, keys in sorted(seen.items()):
        text = file.read_text()
        where = str(file.relative_to(SHADERS))
        body = text.split("// pie:instantiate")[0]
        for key in sorted(keys):
            if key in body:
                continue
            if (where, key) in INERT_DEFINES:
                continue
            failures += 1
            print(
                f"INERT {where}: `{key}` is set by an instantiate line and no "
                f"body reads it, so the modules it names are byte-identical to "
                f"their siblings. Implement it or drop it from the name; if it "
                f"is deliberately name-only, say so in INERT_DEFINES.",
                file=sys.stderr,
            )
    print(f"defines: {sum(len(k) for k in seen.values())} instantiate keys checked")
    return failures



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile", action="store_true", help="slangc every variant")
    ap.add_argument("--defines", action="store_true",
                    help="fail on an instantiate key no shader body reads")
    args = ap.parse_args()

    variants = census()
    # The census counts ENTRYPOINTS, not modules. A tier must never move this
    # set -- that invariant is why a tier is a second compile of a name that
    # already exists rather than a name of its own.
    names = sorted({e for e, _ in variants})
    files = sorted({v.file for v in variants.values()})
    tiered = [v for v in variants.values() if v.tier != BASELINE]

    print(
        f"{len(names)} entrypoints over {len(files)} shader files "
        f"({len(variants)} SPIR-V modules)"
    )
    for path in files:
        n = sum(1 for v in variants.values() if v.file == path)
        print(f"  {n:4d}  {path.relative_to(SHADERS)}")
    for tier in TIERS:
        if tier == BASELINE:
            continue
        n = sum(1 for v in tiered if v.tier == tier)
        if n:
            print(f"  @{tier}: {n} additional modules")

    if args.compile:
        failures = compile_all(variants)
        if failures:
            die(f"{failures} of {len(variants)} modules failed to compile")
        print(f"slangc: {len(variants)} modules compile")

    if args.defines:
        failures = check_defines(variants)
        if failures:
            die(f"{failures} instantiate keys name a behaviour no body has")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
