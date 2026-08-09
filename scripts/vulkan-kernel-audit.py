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
`metal-kernel-audit.py` runs one. A GLSL compute shader has exactly one entry
point and it is always `main`; what distinguishes two variants is the `-D` set
and the `.spv` they are written to. So the variant list cannot be implicit in
the source: it has to be DECLARED, and this reads the declarations.

    // pie:instantiate rms_single_row_bfloat16 N_READS=4

`build.rs` reads the same lines and compiles exactly them. That is deliberate --
an audit that derived the set differently from the build could pass while the
build produced something else -- and it is why `--compile` exists here: running
glslc over every declared variant is what proves a declaration is a variant that
builds, which is the half a set comparison cannot see.

Usage:
    vulkan-kernel-audit.py            # print the census
    vulkan-kernel-audit.py --check    # fail if entrypoints.generated.txt drifts
    vulkan-kernel-audit.py --write    # regenerate entrypoints.generated.txt
    vulkan-kernel-audit.py --compile  # additionally run glslc over every variant
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "kernels-vulkan"
SHADERS = CRATE / "kernels"
ARTIFACT = CRATE / "entrypoints.generated.txt"

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
    for path in sorted(SHADERS.rglob("*.comp")):
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
    """glslc over every declared variant, every tier. Returns the failure count."""
    glslc = os.environ.get("PIE_GLSLC") or shutil.which("glslc")
    if glslc is None:
        die("no glslc on PATH; set PIE_GLSLC or install the Vulkan SDK / shaderc")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for (name, tier), variant in sorted(variants.items()):
            cmd = [
                glslc,
                "-fshader-stage=compute",
                "--target-env=vulkan1.3",
            ]
            # spirv-opt is skipped for coopmat, matching `build.rs` and the
            # llama.cpp finding it borrows (ggml #15344): spirv-opt miscompiles
            # cooperative-matrix shaders. Auditing with flags the build does not
            # use would prove the wrong thing.
            if tier != "coopmat":
                cmd.append("-O")
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


def check_bindings(variants: dict[tuple[str, str], Variant]) -> int:
    """Compare each shader's declared bindings against the table's launch ABI.

    The numbers in a `.comp` are not free to choose. `kernels_vulkan::bindings`
    numbers buffer-kinded operands densely from zero in the row's operand order
    and sends scalars to a push block, so a shader's `binding = N` is a
    FUNCTION of the row -- and a wrong one is invisible: Vulkan does not report
    a descriptor that was declared and never written, it just reads whatever is
    there.

    Sixty entrypoints were doing exactly that when this check was written. Two
    causes, both of them a shader author transcribing Metal's buffer indices:
    `residual` bound at 7 where the row puts it at 5, and the paged KV write
    page/offset at 9/10 where the scalars moving to push constants had pulled
    them down to 10/11.

    The push block is checked three ways: by NAME positionally, by WIDTH, and
    by SIZE. Names and widths are checked over the first N fields only, because
    a block may legitimately declare more than a variant READS -- a shared
    block serving several variants naturally does. What it may not do is
    declare more than the ROW states, because the block must fit the pipeline
    layout's push range; that is the SIZE check, and it is absolute.
    Order is the dangerous one of the three. A shader whose field 0 is
    `row_stride` where the row's first scalar is `out_vec_size` reads a
    plausible number from the wrong slot, and nothing -- not validation, not a
    crash -- will say so. Matching the first N names against the row's scalar
    operands catches that.

    The active preprocessor branch is what matters -- a `#if` picks which
    bindings a variant really declares -- so this preprocesses each variant with
    its own defines rather than reading the source text.
    """
    glslc = os.environ.get("PIE_GLSLC") or shutil.which("glslc")
    if glslc is None:
        die("no glslc on PATH; set PIE_GLSLC or install the Vulkan SDK / shaderc")

    table = subprocess.run(
        ["cargo", "run", "-q", "-p", "kernels-vulkan", "--example", "dump_layout"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if table.returncode != 0:
        die(f"cannot read the table:\n{table.stderr.rstrip()}")
    expect = {}
    for line in table.stdout.splitlines():
        name, buffers, pushes = line.split("\t")
        # Each field is `name:offset:size`; see `examples/dump_layout.rs`.
        fields = []
        for f in pushes.split(","):
            if not f:
                continue
            fname, off, size = f.rsplit(":", 2)
            fields.append((fname, int(off), int(size)))
        expect[name] = (int(buffers), fields)

    failures = 0
    for (name, tier), variant in sorted(variants.items()):
        buffers, pushes = expect.get(name, (0, []))
        # An UNSTATED row names no operands, so there is no ABI to check
        # against. Saying nothing is the row's own defect, not the shader's.
        if buffers == 0:
            continue
        cmd = [glslc, "-fshader-stage=compute", "--target-env=vulkan1.3", "-E",
               "-I", str(SHADERS), f"-DPIE_ENTRYPOINT={name}"]
        cmd += [f"-D{k}={v}" for k, v in variant.defines]
        cmd += [str(variant.file)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            continue  # --compile is what reports a shader that will not build
        declared = {int(n) for n in re.findall(r"binding\s*=\s*(\d+)", done.stdout)}
        over = sorted(b for b in declared if b >= buffers)
        if over:
            failures += 1
            print(
                f"OVERRUN {name} @{tier} ({variant.where}): declares binding(s) "
                f"{over} but the row has {buffers} buffer operands (0..{buffers - 1}). "
                f"Ask `cargo run -p kernels-vulkan --example dump_layout -- <symbol>`.",
                file=sys.stderr,
            )

        # Set EQUALITY against `range(buffers)` is the check this obviously
        # wants, and it is not available here. It was written and withdrawn:
        # it found two modules and both were correct.
        #
        # A shader legitimately declares a SUBSET for two different reasons.
        # A macro-gated body omits a binding it does not read --
        # `moe/qmv_routed.comp` under `PIE_MXFP4` skips `biases` at 2, because
        # the MXFP4 codec has no separate bias plane. And several rows carry
        # Metal ring-ABI PLACEHOLDER operands that no shader ever reads:
        # `kv_append_paged` has thirteen buffers of which the shader declares
        # 0/1/2/3 and then 10/11, so bindings 4..9 are a hole by design.
        #
        # Nothing in the row distinguishes a placeholder from an operand a
        # shader forgot, so an equality check reports both and a reader learns
        # to ignore it. The overrun direction is the one that is unambiguous --
        # a binding past the end is wrong however the row is shaped -- and it
        # is the direction that catches the transcription bug this script
        # exists for, since Metal's combined indices are always HIGHER than
        # Vulkan's dense ones.

        block = re.search(
            r"push_constant\s*\)\s*uniform\s+\w+\s*\{(.*?)\}", done.stdout, re.S
        )
        # `int k` / `uint width[4]` -- drop any array suffix, then the
        # declarator is the last token and the one before it is the type.
        #
        # The type used to be discarded on the grounds that it was the row's
        # business. It is not: a push block is std430, so a field's byte offset
        # is a function of every WIDTH before it, and a shader that declares
        # `int` where the row says `Usize` keeps every name in the right order
        # while shifting all of them four bytes. `attn/kv_write.comp` is the
        # shape where that bites -- `int` then two `uint64_t`, so the block is
        # 4 + 4 pad + 8 + 8 = 24 and not the 20 that adding widths gives.
        GLSL_WIDTH = {"int": 4, "uint": 4, "float": 4, "bool": 4,
                      "int64_t": 8, "uint64_t": 8, "uvec2": 8, "ivec2": 8,
                      "vec2": 8, "uvec4": 16, "ivec4": 16, "vec4": 16}
        fields = []
        if block is not None:
            for member in block.group(1).split(";"):
                member = member.strip()
                if member:
                    toks = re.sub(r"\[.*", "", member).split()
                    ty = toks[-2] if len(toks) >= 2 else ""
                    fields.append((toks[-1], GLSL_WIDTH.get(ty)))

        # And the block must FIT. This is the one direction that is not a
        # matter of taste, and the comment above about descriptor sets used to
        # sit here too, wrongly: "Vulkan only requires the layout to cover what
        # is statically used". It does not.
        # `VUID-VkComputePipelineCreateInfo-layout-10069` requires the DECLARED
        # block to be contained in the range, whether the module reads the tail
        # of it or not, and the range is built from the row's scalars. A
        # validation layer reported it on 120 entrypoints across four families;
        # this driver had been creating those pipelines for weeks.
        #
        # So a shared block has to be shaped by the same `#if`s that shape the
        # bodies reading it, and a field no variant reads at all has to go.
        #
        # The size is std430: each member starts at the next multiple of its
        # own width, and the block ends after the last one. A member whose type
        # is not in the table above makes the size unknowable, and an unknown
        # size is not reported -- `--check` is where an unrecognised shader
        # belongs.
        size, known = 0, block is not None
        for member in (block.group(1).split(";") if block is not None else []):
            member = member.strip()
            if not member:
                continue
            arr = re.search(r"\[(\d+)\]", member)
            count = int(arr.group(1)) if arr else 1
            toks = re.sub(r"\[.*", "", member).split()
            width = GLSL_WIDTH.get(toks[-2] if len(toks) >= 2 else "")
            if width is None:
                known = False
                break
            size = (size + width - 1) // width * width + width * count
        want = pushes[-1][1] + pushes[-1][2] if pushes else 0
        if known and size > want:
            failures += 1
            print(
                f"RANGE {name} @{tier} ({variant.where}): push block is {size} "
                f"bytes but the row's scalars end at {want}. The block must fit "
                f"inside the pipeline layout's push range even where the module "
                f"never reads the tail. Gate the extra fields on the macro that "
                f"reads them, or state them in the row.",
                file=sys.stderr,
            )

        widths = [w for _, w in fields[: len(pushes)]]
        want_widths = [size for _, _, size in pushes]
        if any(w is not None and w != s for w, s in zip(widths, want_widths)):
            failures += 1
            print(
                f"WIDTH {name} @{tier} ({variant.where}): push block declares "
                f"{[(n, w) for n, w in fields[: len(pushes)]]} but the row's "
                f"scalars are {[(n, s) for n, _, s in pushes]}. A width that "
                f"disagrees moves every field after it, and the names still "
                f"line up. Ask `cargo run -p kernels-vulkan --example "
                f"dump_layout -- <symbol>`.",
                file=sys.stderr,
            )

        pushes = [n for n, _, _ in pushes]
        fields = [n for n, _ in fields]
        if fields[: len(pushes)] != pushes:
            failures += 1
            print(
                f"PUSH {name} @{tier} ({variant.where}): push block begins "
                f"{fields[: len(pushes) + 1]} but the row's scalars are {pushes}. "
                f"Ask `cargo run -p kernels-vulkan --example dump_layout -- <symbol>`.",
                file=sys.stderr,
            )
    return failures


def die(message: str) -> None:
    print(f"vulkan-kernel-audit: {message}", file=sys.stderr)
    raise SystemExit(1)


# A define an `instantiate` line sets and no body reads produces a module that
# is byte-identical to its sibling under a name that promises otherwise. The
# ones below are known and each has a reason; anything NEW is a defect.
INERT_DEFINES = {
    # Named for the fp16 activation path `quant/qmm_t.comp` really has (a
    # separate pre-cast buffer at binding 7). The routed file never grew one,
    # so these nine modules are copies of their non-fp16 siblings. Harmless
    # only because `affine_qmm_t_routed_fp16`'s row is UNSTATED and so cannot
    # be bound at all; if it is ever stated, the body has to exist first or a
    # driver will hand fp16 bytes to a shader reading bf16.
    ("moe/qmm_t_routed.comp", "PIE_FP16"),
    # The encode path is selected by `PIE_MXFP4`, not by this.
    ("quant/transcode.comp", "PIE_ENCODE"),
    # ABI-name-only points: the entrypoint name carries the axis, the body
    # does not vary on it.
    ("quant/qmv.comp", "PIE_K_LANES"),
    ("quant/qmm_t.comp", "PIE_PROBE_SHAPE"),
    ("attn/sdpa_paged.comp", "PIE_SHORT_GROUP"),
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
    ap.add_argument("--check", action="store_true", help="fail on drift")
    ap.add_argument("--write", action="store_true", help="regenerate the artifact")
    ap.add_argument("--compile", action="store_true", help="glslc every variant")
    ap.add_argument("--defines", action="store_true",
                    help="fail on an instantiate key no shader body reads")
    ap.add_argument("--bindings", action="store_true",
                    help="check declared bindings against the table's launch ABI")
    args = ap.parse_args()

    variants = census()
    # The artifact pins ENTRYPOINTS, not modules. A tier must never move this
    # set -- that invariant is the reason the artifact stayed a flat name list
    # when tiers arrived.
    names = sorted({e for e, _ in variants})
    files = sorted({v.file for v in variants.values()})
    tiered = [v for v in variants.values() if v.tier != BASELINE]

    if args.write:
        ARTIFACT.write_text("\n".join(names) + "\n")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}: {len(names)} entrypoints")
    elif args.check:
        if not ARTIFACT.exists():
            die(f"{ARTIFACT} does not exist; run with --write")
        have = ARTIFACT.read_text().split()
        if have != names:
            appeared = sorted(set(names) - set(have))
            vanished = sorted(set(have) - set(names))
            for name in appeared:
                where = variants[(name, BASELINE)].where
                print(f"+ {name} ({where})", file=sys.stderr)
            for name in vanished:
                print(f"- {name}", file=sys.stderr)
            die("entrypoints.generated.txt is stale; run with --write")
        print(f"{ARTIFACT.relative_to(ROOT)} is current: {len(names)} entrypoints")
    else:
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
        print(f"glslc: {len(variants)} modules compile")

    if args.defines:
        failures = check_defines(variants)
        if failures:
            die(f"{failures} instantiate keys name a behaviour no body has")

    if args.bindings:
        failures = check_bindings(variants)
        if failures:
            die(f"{failures} modules declare a binding the row does not have")
        print(f"bindings: {len(variants)} modules agree with the table")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
