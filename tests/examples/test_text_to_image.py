"""A prompt goes in and an image comes out, with nothing about the family in the guest.

`text-to-image` is the generic sampler of imagegen design D1/D4/D11/D12: it finds
its text and denoise readings by ROLE (`takes_tokens && Hidden`,
`!takes_tokens && Velocity`), sizes the latent grid from `model.latent()`, steps
`model.schedule()`, places every lane's rotary coordinates from the reading's own
`positions` convention, and exits either through a `vae.decode` reading or as the
raw final latent. This gate drives it through the CLI a person actually types
(`pie --config C run -o DIR`), because the latent leaves as a FILE and the file is
half the claim.

The half in FRONT of this -- that `pie model list` reports the row's
generative facts, that `pie run text-to-image` resolves the guest by name with
no `--path`, and that what arrives is a named file -- is
`test_generating_images.py`. This suite is about the sampler.

Two claims, one per row:
  * on a generative row with a text encoder (FLUX.2 klein-4B), four steps at 1024²
    land a finite 64x64x128 latent whose sigmas are the model's own, and
    `scripts/imagegen/decode_latent.py` turns it into a PNG of the stated size;
  * on a row with NO text reading (`mini-dit`, whose caption rows are random
    embeddings), the guest refuses BY NAME -- "this model has no text reading" --
    rather than drawing something nobody asked for.

**WANTS A GENERATIVE MODEL.** There is no text-model fallback: a row with no
denoise reading is reported as a skip, not a failure. The runs that mean to
exercise the door name their config:
    CUDA_VISIBLE_DEVICES=3 uv run python tests/examples/test_text_to_image.py \\
        --config ~/.pie/config.flux2.toml \\
        --model-dir ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/

    CUDA_VISIBLE_DEVICES=3 uv run python tests/examples/test_text_to_image.py \\
        --config ~/.pie/config.minidit.toml --expect-refusal

where the config's `[model] model` is the imported `.zt` (`pie model import
<diffusers folder> --sku flux2-klein-4b-bf16-kv-bf16 --out ...`). Unlike the
curated suites this one does not use the embedded `pie.server` wheel: what is
under test includes `pie run -o`, so it runs the binary.

**GIVE THIS RUN ITS OWN CONFIG FILE.** `~/.pie/config.flux2.toml` is shared,
and a config another agent repoints at a different artifact (or a different
port) is indistinguishable from a numerics regression: the run still succeeds
and hands back a plausible latent that decodes to the wrong picture. Copy one
to a name only this suite uses.

**TWO CONFIG KEYS THIS RUN NEEDS, AND WHY.** `[engine] max_model_len` is the
per-slot token ceiling, and a 1024² job is 4096 latent rows a lane; at the
default 4096 the engine refuses the second fire by name ("this fire wants 8192
kv tokens in one slot and the load reserved 4096" -- the frame carries the
depth, not one lane). Set it at or above the rows the job carries times the
submit depth: `max_model_len = 32768` covers 1024² comfortably. `[engine]
graphs = "on"` is the recording policy the denoise arm is meant to run under;
`off` works and is slower.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

INFERLETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERLETS_DIR.parent.parent

NAME = "text-to-image"

# The refusal that means "this load has no text encoder", not "the sampler
# broke". The guest spells it exactly this way, on purpose.
NO_TEXT = "this model has no text reading"

# Refusals that mean "this load has no latent at all" -- a text row. Reported
# as a skip.
NO_LATENT = (
    "declares no readings",
    "no token-less reading",
    "pass kind is not attention",
    "states no latent space",
    "states no schedule",
)


def build_guest() -> Path:
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        subprocess.run(
            ["cargo", "build", "-p", NAME, "--release", "--target", "wasm32-wasip2"],
            cwd=INFERLETS_DIR,
            check=True,
        )
    wasm = (INFERLETS_DIR / "target" / "wasm32-wasip2" / "release"
            / f"{NAME.replace('-', '_')}.wasm")
    if not wasm.exists():
        raise FileNotFoundError(f"no guest at {wasm}; build it or unset PIE_INFERLETS_NO_BUILD")
    return wasm


def find_cli(binary: str | None) -> Path:
    """The `pie` this drives. Built only when it is not already there: a
    generative row's load is minutes and a rebuild on top of that is the
    difference between a gate and a coffee break."""
    if binary:
        return Path(binary)
    for profile in ("release", "debug"):
        path = REPO_ROOT / "target" / profile / "pie"
        if path.exists():
            return path
    raise FileNotFoundError(
        "no `pie` binary under target/{release,debug}; "
        "`cargo build -p pie --bin pie --features cuda` or pass --pie"
    )


def resolve(path: str) -> str:
    path = os.path.expanduser(path)
    if any(ch in path for ch in "*?["):
        hits = sorted(glob.glob(path))
        if not hits:
            raise SystemExit(f"{path}: matches nothing")
        return hits[0]
    return path


def run_guest(args, out_dir: Path, extra: list[str]) -> tuple[int, str, str]:
    wasm = build_guest()
    cmd = [str(find_cli(args.pie))]
    if args.config:
        cmd += ["--config", resolve(args.config)]
    cmd += ["run", "--path", str(wasm), "-o", str(out_dir), "--"] + extra
    print("Run:   ", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=args.timeout)
    return proc.returncode, proc.stdout, proc.stderr


def report_of(stdout: str) -> dict:
    """`pie run` prints a human header before the document; take the JSON."""
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    assert lines, f"no JSON document in the inferlet's output:\n{stdout[-2000:]}"
    doc = json.loads(lines[-1])
    if isinstance(doc, dict) and "result" in doc:
        doc = doc["result"]
    if isinstance(doc, str):
        doc = json.loads(doc)
    return doc


def check_png(path: Path, width: int, height: int) -> str:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (width, height), f"{path.name} is {w}x{h}, expected {width}x{height}"
    try:
        from PIL import Image
    except ImportError:
        return "magic + IHDR"
    with Image.open(path) as img:
        img.load()
        rgb = img.convert("RGB")
        pixels = list(rgb.getdata())
        # A decode that produced a flat field is not an image; a real one
        # carries at least a few hundred distinct colours over a megapixel.
        assert len(set(pixels)) > 256, (
            f"{path.name} has {len(set(pixels))} distinct colours: the decode "
            f"produced a flat field, not a picture"
        )
    return "Pillow"


# ---------------------------------------------------------------------------
# the two claims
# ---------------------------------------------------------------------------

def a_prompt_becomes_a_latent_a_vae_can_decode(args) -> None:
    """Four steps at 1024² land a finite latent, and its PNG is a picture."""
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pie-t2i-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out:    {out_dir}")

    code, stdout, stderr = run_guest(args, out_dir, [
        "--prompt", args.prompt,
        "--width", str(args.size), "--height", str(args.size),
        "--steps", str(args.steps), "--seed", str(args.seed),
        "--out", "image",
    ])
    sys.stdout.write(stdout)
    if code != 0 or args.verbose:
        sys.stderr.write(stderr)
    if code != 0:
        blob = stdout + stderr
        if NO_TEXT in blob or any(mark in blob for mark in NO_LATENT):
            raise FileNotFoundError(
                f"the loaded row has no text-to-image path, so there is nothing to "
                f"drive here: {blob[-400:]}"
            )
        raise AssertionError(f"`pie run` exited {code}:\n{stderr[-2000:]}")

    report = report_of(stdout)
    print(f"Report: { {k: v for k, v in report.items() if k != 'sigmas'} }")
    print(f"Sigmas: {[round(s, 4) for s in report['sigmas']]}")

    assert report["width"] == args.size and report["height"] == args.size, report
    assert report["rows"] == report["grid_h"] * report["grid_w"], report
    assert report["steps"] == args.steps, report
    assert len(report["sigmas"]) == args.steps + 1 and report["sigmas"][-1] == 0.0, report
    assert report["context_rows"] > 0, "the prompt encoded to no rows"
    assert report["non_finite"] == 0, f"the latent carries non-finite values: {report}"
    # A four-step flow lands a latent of roughly unit scale; an all-zero or
    # exploded one is the loop having gone wrong in a way the shape hides.
    assert 0.05 < report["std"] < 20.0, f"the final latent's scale is {report['std']}"

    # THE DECODED EXIT (design D8/D11). A model that declares a `vae.decode`
    # reading fires it in-guest and sends a PNG the runtime encoded: there is
    # no latent blob, no sidecar and nothing for `decode_latent.py` to do,
    # because the picture is already a picture.
    if report.get("decoded"):
        png = out_dir / report["file"]
        assert png.exists(), f"the report says it sent {report['file']} and {out_dir} has it not"
        how = check_png(png, args.size, args.size)
        print(f"[OK] {png}  {png.stat().st_size:,} bytes  (checked by {how}) "
              f"-- decoded in-guest by reading `{report['decode_reading']}`")
        return

    # The other exit goes out through `session.send-file-as`, so the latent
    # arrives NAMED too and `pie run -o` writes it under that name -- no
    # `file-0000.bin` to rename, and no need to read the report to find out
    # which file is which.
    latent = out_dir / report["file"]
    assert latent.exists(), (
        f"the report names {report['file']}, which is not in {out_dir}: "
        f"{sorted(p.name for p in out_dir.iterdir())}"
    )
    assert not sorted(out_dir.glob("file-*.bin")), (
        "a numbered blob arrived beside the named one: something is still "
        "sending through the unnamed `session.send-file`"
    )
    assert latent.stat().st_size == report["bytes"], (latent.stat().st_size, report["bytes"])
    sidecar = latent.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2))
    print(f"Latent: {latent}  ({latent.stat().st_size:,} bytes)")

    if not args.model_dir:
        print("⚠️  no --model-dir: the latent is not decoded, so this run proved the "
              "loop and not the picture")
        return
    png = out_dir / "image.png"
    decode = [args.python, str(REPO_ROOT / "scripts" / "imagegen" / "decode_latent.py"),
              "--latent", str(latent), "--sidecar", str(sidecar),
              "--model-dir", resolve(args.model_dir), "--out", str(png),
              "--family", args.family]
    print("Decode:", " ".join(decode))
    done = subprocess.run(decode, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=args.timeout)
    sys.stdout.write(done.stdout)
    if done.returncode != 0:
        sys.stderr.write(done.stderr)
        raise AssertionError(f"decode_latent.py exited {done.returncode}")
    how = check_png(png, args.size, args.size)
    print(f"✅ {png}  {png.stat().st_size:,} bytes  (checked by {how})")


def a_model_with_no_text_reading_refuses_by_name(args) -> None:
    """`mini-dit` cannot be told what to draw, and says so instead of guessing."""
    out_dir = Path(tempfile.mkdtemp(prefix="pie-t2i-refuse-"))
    code, stdout, stderr = run_guest(args, out_dir, [
        "--prompt", args.prompt, "--steps", "1",
    ])
    shutil.rmtree(out_dir, ignore_errors=True)
    blob = stdout + stderr
    assert code != 0, f"the guest ran on a model with no text encoder:\n{stdout[-1000:]}"
    assert NO_TEXT in blob, (
        f"expected the refusal {NO_TEXT!r}; got:\n{blob[-2000:]}"
    )
    print(f"✅ refused by name: {NO_TEXT!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="the serving config; its `[model] model` is the imported artifact")
    ap.add_argument("--pie", default=None, help="the `pie` binary (default: target/{release,debug})")
    ap.add_argument("--model-dir", default=None,
                    help="the diffusers folder to decode with; without it the latent is "
                         "checked but not decoded")
    ap.add_argument("--family", default="flux2", help="decode_latent.py's --family")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that has torch + diffusers (the imagegen venv)")
    ap.add_argument("--prompt", default="a red bicycle leaning on a blue wall")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write files here instead of a temp dir")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--expect-refusal", action="store_true",
                    help="run the mini-dit claim instead: the guest must refuse by name")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    claim = (a_model_with_no_text_reading_refuses_by_name if args.expect_refusal
             else a_prompt_becomes_a_latent_a_vae_can_decode)
    print(f"🔄 {claim.__name__}")
    try:
        claim(args)
    except FileNotFoundError as why:
        print(f"⏭️  SKIPPED: {why}")
        return 0
    except AssertionError as why:
        print(f"❌ FAILED: {why}")
        return 1
    print("\n✅ the claim holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
