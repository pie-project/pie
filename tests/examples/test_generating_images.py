"""The commands a person types to get a picture out of pie, in the order they type them.

This is the USER PATH, not the sampler. `test_text_to_image.py` is the gate on
the guest -- that it finds its readings by role, steps the family's own sigmas,
refuses a model with no text encoder. This one asserts the half in front of
that: that the facts are visible before the run, that the run needs no path,
and that what comes back is a named file a decoder turns into a picture of the
size that was asked for.

Three claims, in the order the guide teaches them:
  * `pie model list` / `pie model info` report the served row's GENERATIVE
    facts -- its readings, its latent space, its schedule -- off the catalog,
    without opening a tensor;
  * `pie run text-to-image -o DIR -- --prompt "..."` runs the guest BY NAME.
    No `--path`: the name resolves against the
    `examples` of the checkout the command runs in;
  * the file that arrives is NAMED -- `image.png` where the row drives its own
    `vae.decode` reading, `image.latent.f32` where it does not, never
    `file-0000.bin` -- and it is a PNG of the size that was asked for, either
    straight out of the run or through `scripts/imagegen/decode_latent.py`.

**WHAT THE PICTURE CHECK DOES NOT PROVE.** It proves the file decodes, that it
is the size that was asked for, and that the decode was not a flat field. It
does NOT prove the trajectory was right, and that limit was measured rather
than assumed: a run whose denoise loop went wrong decodes to a textured field
whose colour count, local gradient and contrast are indistinguishable from a
photograph's (11.9 vs 11.9 mean horizontal gradient over the two observed on
2026-09-06). Separating those needs a reference, which this suite does not
carry. A person looking at the PNG is still the check on fidelity.

**WANTS A GENERATIVE MODEL.** A config bound to a text row has nothing to
drive here and is reported as a skip, not a failure:
    CUDA_VISIBLE_DEVICES=3 uv run python tests/examples/test_generating_images.py \\
        --config ~/.pie/config.t2i-client.toml \\
        --model-dir ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/

**GIVE THIS RUN ITS OWN CONFIG FILE**, with its own `[server] port` and
`[engine] max_model_len = 32768`. A shared config another process repoints at
a different artifact mid-run is indistinguishable from a numerics regression:
the run still succeeds and hands back a plausible latent for the wrong model.
The `max_model_len` default of 4096 kills a 1024**2 job on its second fire
("this fire wants 8192 kv tokens in one slot").

**THE SUBMIT DEADLINE IS NOT A KNOB THIS SUITE TOUCHES**, and it used to be.
A denoise frame composes an attention group out of the image lane and the
context lane, and the group forms only when both are members of ONE step; that
wait was once leashed by `[runtime] submit_deadline` (50 ms), and a 1024**2
job's first submit carries megabytes of seeded `context` and `latents` across
the guest boundary, so under load the leash fired first and the frame sealed
with the IMAGE LANE ALONE -- an unconditioned denoise, reported as a success,
decoding to exactly the textured field the note above says this suite cannot
tell from a photograph. Measured on FLUX.2-klein-4B on 2026-09-06, 4 steps at
1024**2, one prompt and seed, one GPU with no neighbours:
    50 ms, before the fix   1 of 3 correct eager, 4 of 5 bodied, and the wrong
                            ones bit-identical to each other per trajectory --
                            a race, which reads exactly like a numerics
                            non-determinism and is not one. `graphs = "off"`
                            made it WORSE, not better, because the eager text
                            fire is slower and misses the leash more often.
    50 ms, after the fix    4 of 4 bit-identical and correct.

The runtime now keeps the promise instead: a stated cohort fires whole or not
at all (`crates/runtime/src/scheduler/frame.rs`, `group_short`). So this suite
runs at the default -- if a partial group is ever sealed again, a run of this
that comes back as a textured field is one of the things that says so.

The PNG needs an interpreter with torch + diffusers (`--python`); without
`--model-dir` the run proves the loop and the file name and says so.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERLETS_DIR = REPO_ROOT / "examples"

NAME = "text-to-image"

# The guest's own refusals, by the words it spells them with. Each one means
# "this load cannot do text-to-image", which is a skip and not a failure.
NO_PATH = (
    "this model has no text reading",
    "declares no readings",
    "no token-less reading",
    "pass kind is not attention",
    "states no latent space",
    "states no schedule",
)


# ---------------------------------------------------------------------------
# the tools this drives
# ---------------------------------------------------------------------------

def find_cli(binary: str | None) -> Path:
    """The `pie` this drives, already built: a generative row's load is
    minutes and a rebuild on top of that is the difference between a gate and
    a coffee break."""
    if binary:
        return Path(binary)
    for profile in ("release", "debug"):
        path = REPO_ROOT / "target" / profile / "pie"
        if path.exists():
            return path
    raise FileNotFoundError(
        "no `pie` binary under target/{release,debug}; "
        "`cargo build -p pie --bin pie --features cuda --release` or pass --pie"
    )


def build_guest() -> None:
    """Build the curated guest, which is what `pie run <name>` will find."""
    if os.environ.get("PIE_INFERLETS_NO_BUILD"):
        return
    subprocess.run(
        ["cargo", "build", "-p", NAME, "--release", "--target", "wasm32-wasip2"],
        cwd=INFERLETS_DIR,
        check=True,
    )


def resolve(path: str) -> str:
    path = os.path.expanduser(path)
    if any(ch in path for ch in "*?["):
        hits = sorted(glob.glob(path))
        if not hits:
            raise SystemExit(f"{path}: matches nothing")
        return hits[0]
    return path


def served_artifact(config: str) -> Path:
    """The `.zt` the config binds, so the fact claims can be made about the
    same file the run claims are made about."""
    for line in Path(resolve(config)).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("model") and "=" in stripped and ".zt" in stripped:
            return Path(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
    raise FileNotFoundError(f"{config} has no `[model] model` naming a `.zt`")


# ---------------------------------------------------------------------------
# claim 1 -- the facts are visible before anything is run
# ---------------------------------------------------------------------------

def the_store_says_what_the_row_can_do(args) -> dict:
    """`pie model list` and `pie model info` report the generative facts.

    Run against a PRIVATE `$PIE_HOME` holding one symlink to the served
    artifact. That is not a shortcut around the store: it IS the store layout
    (`models/<name>/archive.zt`), and building it here means the claim is
    about the artifact this suite is also about to run, on a machine whose
    real store may hold anything.
    """
    home = Path(tempfile.mkdtemp(prefix="pie-facts-home-"))
    artifact = served_artifact(args.config)
    if not artifact.exists():
        raise FileNotFoundError(f"the config binds {artifact}, which is not there")
    model_dir = home / "models" / artifact.stem
    model_dir.mkdir(parents=True)
    (model_dir / "archive.zt").symlink_to(artifact)

    env = dict(os.environ, PIE_HOME=str(home), HF_HOME=str(home / "hf"))
    cli = str(find_cli(args.pie))

    listing = subprocess.run([cli, "model", "list", "--json"], env=env,
                             capture_output=True, text=True, check=True)
    entries = json.loads(listing.stdout)["artifacts"]
    assert len(entries) == 1, f"expected the one symlinked artifact, got {entries}"
    entry = entries[0]

    if entry.get("generative") is None:
        raise FileNotFoundError(
            f"{entry['address']} was imported as `{entry.get('sku')}`, which is not a "
            f"generative row: there is nothing to draw with"
        )
    facts = entry["generative"]

    # The three D12 answers, each present and each self-consistent.
    names = [r["name"] for r in facts["readings"]]
    text = [r for r in facts["readings"] if r["tokens"] and r["readout"] == "hidden"]
    denoise = [r for r in facts["readings"]
               if not r["tokens"] and r["readout"] == "velocity"]
    # A row with no text reading (the `mini-dit` fixture, whose caption rows
    # are random embeddings) cannot be told what to draw. That is a skip and
    # not a failure -- and it is also the claim the `--expect-refusal` half of
    # `test_text_to_image.py` makes, so there is nothing left for this suite
    # to say about it.
    if not text or not denoise:
        raise FileNotFoundError(
            f"{entry['address']} declares {names}, which is not a text-to-image path: "
            f"the listing agrees with the guest's refusal"
        )
    # The human listing must agree with the document, or the marker it prints
    # is decoration.
    human = subprocess.run([cli, "model", "list"], env=env,
                           capture_output=True, text=True, check=True)
    assert "generative" in human.stdout and "text-to-image" in human.stdout, (
        f"the readings are a text-to-image pair but `pie model list` did not say so:\n"
        f"{human.stdout}"
    )
    latent = facts["latent"]
    assert latent and latent["channels"] > 0, facts
    schedule = facts["schedule"]
    assert schedule and schedule["kind"] in ("flow", "epsilon", "v-prediction"), facts
    assert facts["max_latent_rows"] > 0, facts

    # One latent row covers this many pixels, which is what a size rounds to.
    step_w = latent["patch_w"] * latent["spatial_compression"]
    step_h = latent["patch_h"] * latent["spatial_compression"]
    assert args.size % step_w == 0 and args.size % step_h == 0, (
        f"{args.size} is not a multiple of this row's {step_w}x{step_h} pixel step"
    )

    # `info` is the same facts rendered for a person, and the readings are the
    # part a listing has no room for.
    info = subprocess.run([cli, "model", "info", entry["address"]], env=env,
                          capture_output=True, text=True, check=True)
    assert "Generative" in info.stdout and "Readings" in info.stdout, info.stdout
    for name in names:
        assert name in info.stdout, f"`model info` never mentions the {name!r} reading"

    print(f"✅ {entry['address']}: {len(names)} readings ({', '.join(names)}), "
          f"latent {latent['channels']}ch /{latent['spatial_compression']}, "
          f"{schedule['kind']} schedule, one row is {step_w}x{step_h} px")
    return facts


# ---------------------------------------------------------------------------
# claims 2 and 3 -- the run, and the picture
# ---------------------------------------------------------------------------

def a_bare_name_draws_a_named_file(args, out_dir: Path) -> dict:
    cmd = [str(find_cli(args.pie)), "--config", resolve(args.config), "run", NAME,
           "-o", str(out_dir), "--",
           "--prompt", args.prompt,
           "--width", str(args.size), "--height", str(args.size),
           "--steps", str(args.steps), "--seed", str(args.seed),
           "--out", "image"]
    print("Run:   ", " ".join(cmd))
    # From REPO_ROOT, because curated resolution walks up from the working
    # directory to find `examples` -- which is the behaviour under test.
    done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=args.timeout)
    blob = done.stdout + done.stderr
    if done.returncode != 0:
        if any(mark in blob for mark in NO_PATH):
            raise FileNotFoundError(
                f"the loaded row has no text-to-image path: {blob[-400:]}"
            )
        raise AssertionError(f"`pie run {NAME}` exited {done.returncode}:\n{blob[-3000:]}")

    # The name is half the claim. `session.send-file-as` carries it, `pie run
    # -o` writes under it, and the report says which file it was talking about.
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, f"no JSON report in the output:\n{done.stdout[-2000:]}"
    report = json.loads(lines[-1])

    written = sorted(p.name for p in out_dir.iterdir())
    assert written == [report["file"]], (
        f"expected exactly the file the report names ({report['file']}); "
        f"{out_dir} holds {written}"
    )
    assert not any(name.startswith("file-") for name in written), (
        "the file arrived unnamed, so `pie run -o` numbered it: "
        "`session.send-file-as` is not carrying the name"
    )

    assert report["width"] == args.size and report["height"] == args.size, report
    assert report["rows"] == report["grid_h"] * report["grid_w"], report
    assert report["steps"] == args.steps, report
    assert len(report["sigmas"]) == args.steps + 1, report
    assert report["context_rows"] > 0, "the prompt encoded to no rows"
    assert report["non_finite"] == 0, f"the latent carries non-finite values: {report}"
    assert 0.05 < report["std"] < 20.0, f"the final latent's scale is {report['std']}"
    assert (out_dir / report["file"]).stat().st_size == report["bytes"], report

    print(f"✅ {out_dir / report['file']}  ({report['bytes']:,} bytes), "
          f"{report['steps']} steps at {report['width']}x{report['height']}")
    return report


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
        pixels = list(img.convert("RGB").getdata())
        # A decode that produced a flat field is not an image; a real one
        # carries thousands of distinct colours over a megapixel.
        assert len(set(pixels)) > 256, (
            f"{path.name} has {len(set(pixels))} distinct colours: the decode "
            f"produced a flat field, not a picture"
        )
    return "Pillow"


def the_file_decodes_to_a_picture(args, out_dir: Path, report: dict) -> None:
    """The second half of the raw-latent exit: the sidecar plus the family's
    own VAE."""
    latent = out_dir / report["file"]
    sidecar = out_dir / "image.json"
    sidecar.write_text(json.dumps(report, indent=2))
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
    print(f"✅ {png}  {png.stat().st_size:,} bytes  ({args.size}x{args.size}, checked by {how})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="the serving config; give this run its own")
    ap.add_argument("--pie", default=None, help="the `pie` binary (default: target/{release,debug})")
    ap.add_argument("--model-dir", default=None,
                    help="the diffusers folder to decode with; without it the file is "
                         "checked but not decoded")
    ap.add_argument("--family", default="flux2", help="decode_latent.py's --family")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter that has torch + diffusers (the imagegen venv)")
    ap.add_argument("--prompt", default="a red bicycle leaning on a blue wall")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write files here instead of a temp dir")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pie-t2i-user-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out:    {out_dir}")

    try:
        print("🔄 the_store_says_what_the_row_can_do")
        the_store_says_what_the_row_can_do(args)
        build_guest()
        print("🔄 a_bare_name_draws_a_named_file")
        report = a_bare_name_draws_a_named_file(args, out_dir)
        if report.get("decoded"):
            # The row drove its own `vae.decode` reading, so what arrived IS
            # the picture (design D8/D11) and there is nothing left to decode.
            png = out_dir / report["file"]
            how = check_png(png, args.size, args.size)
            print(f"✅ {png}  {png.stat().st_size:,} bytes  ({args.size}x{args.size}, "
                  f"checked by {how}) -- decoded in-guest by reading "
                  f"`{report['decode_reading']}`")
            print("\n✅ every claim holds")
            return 0
        if not args.model_dir:
            print("⚠️  no --model-dir: the latent is not decoded, so this run proved "
                  "the commands and not the picture")
            return 0
        print("🔄 the_file_decodes_to_a_picture")
        the_file_decodes_to_a_picture(args, out_dir, report)
    except FileNotFoundError as why:
        print(f"⏭️  SKIPPED: {why}")
        return 0
    except AssertionError as why:
        print(f"❌ FAILED: {why}")
        return 1
    print("\n✅ every claim holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
