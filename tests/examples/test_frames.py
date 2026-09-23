"""The binary-output path, end to end, through the CLI a person actually types.

`frames-probe` builds pixels with `frames.from-rgb8` and hands the handle to
`session.send-frames`; `pie run -o DIR` writes what arrives. The claim is that
three named files land in DIR and that each one is really the format its name
says -- which is the only way, from outside, to tell "the runtime encoded and
streamed it" from "something returned an empty success".

Unlike the other suites here this one does NOT use the embedded `pie.server`
wheel: the thing under test includes `pie run -o`, so it runs the binary.

    uv run python tests/examples/test_frames.py
    uv run python tests/examples/test_frames.py --video y4m   # no NVENC needed
    uv run python tests/examples/test_frames.py --keep        # leave DIR behind

The engine boots with whatever `~/.pie/config.toml` says, because that is what
`pie run` does; there is no model in the loop, but the standalone still loads
one before it will run anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERLETS_DIR = REPO_ROOT / "examples"

NAME = "frames-probe"


def build_guest() -> Path:
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        subprocess.run(
            ["cargo", "build", "-p", NAME, "--target", "wasm32-wasip2"],
            cwd=INFERLETS_DIR,
            check=True,
        )
    wasm = INFERLETS_DIR / "target" / "wasm32-wasip2" / "debug" / f"{NAME.replace('-', '_')}.wasm"
    if not wasm.exists():
        raise FileNotFoundError(f"no guest at {wasm}; build it or unset PIE_INFERLETS_NO_BUILD")
    return wasm


def build_cli(features: str, release: bool) -> Path:
    """Build (or find) the `pie` binary the test drives.

    `--release` is not an optimisation of the test, it is what makes it
    finish: a debug `pie` spends minutes loading a checkpoint before it will
    run anything, and this suite is not about the loader.
    """
    profile = "release" if release else "debug"
    if not os.environ.get("PIE_INFERLETS_NO_BUILD"):
        cmd = ["cargo", "build", "-p", "pie", "--bin", "pie"]
        if release:
            cmd.append("--release")
        if features:
            cmd += ["--features", features]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    binary = REPO_ROOT / "target" / profile / "pie"
    if not binary.exists():
        raise FileNotFoundError(f"no `pie` binary at {binary}")
    return binary


# ---------------------------------------------------------------------------
# Format checks. Pillow when it is installed, magic bytes when it is not --
# the magic-byte path is not a weaker test of "did a file arrive", only of
# "does it decode", and it says which one it ran.
# ---------------------------------------------------------------------------

def check_png(path: Path, width: int, height: int) -> str:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    try:
        from PIL import Image
    except ImportError:
        # IHDR is the first chunk and carries the size, so even without
        # Pillow the dimensions are checkable.
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (width, height), f"{path.name} is {w}x{h}, expected {width}x{height}"
        return "magic + IHDR"
    with Image.open(path) as img:
        img.load()
        assert img.size == (width, height), f"{path.name} is {img.size}"
        # The fixture's gradient: x drives red, y drives green. A transposed
        # or mis-strided encoder fails exactly here and nowhere else.
        rgb = img.convert("RGB")
        assert rgb.getpixel((0, 0))[0] < rgb.getpixel((width - 1, 0))[0], "red must rise with x"
        assert rgb.getpixel((0, 0))[1] < rgb.getpixel((0, height - 1))[1], "green must rise with y"
    return "Pillow"


def check_mp4(path: Path, width: int, height: int, frames: int) -> str:
    data = path.read_bytes()
    assert data[4:8] == b"ftyp", f"{path.name} does not begin with an ftyp box"
    assert b"avcC" in data, f"{path.name} has no avcC (H.264 sample entry)"
    assert b"moov" in data and b"mdat" in data, f"{path.name} is missing moov/mdat"
    if not shutil.which("ffprobe"):
        return "magic + box names"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    assert stream["codec_name"] == "h264", stream
    assert (stream["width"], stream["height"]) == (width, height), stream
    assert int(stream["nb_frames"]) == frames, stream
    return "ffprobe"


def check_y4m(path: Path, width: int, height: int, frames: int) -> str:
    data = path.read_bytes()
    assert data.startswith(b"YUV4MPEG2 "), f"{path.name} is not a y4m"
    header = data.split(b"\n", 1)[0].decode()
    assert f"W{width}" in header and f"H{height}" in header, header
    assert data.count(b"FRAME\n") == frames, f"{path.name} has the wrong frame count"
    return "header"


def check_wav(path: Path) -> str:
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", f"{path.name} is not a wav"
    rate = struct.unpack("<I", data[24:28])[0]
    channels = struct.unpack("<H", data[22:24])[0]
    assert (rate, channels) == (24_000, 1), f"{path.name} is {rate} Hz x{channels}"
    declared = struct.unpack("<I", data[40:44])[0]
    assert len(data) == 44 + declared, "the data chunk length disagrees with the file"
    return "header"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="mp4", choices=["mp4", "y4m"],
                        help="clip format the fixture asks for (default: mp4, needs NVENC)")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--features", default="cuda",
                        help="cargo features for the `pie` binary (default: cuda)")
    parser.add_argument("--out", default=None, help="write files here instead of a temp dir")
    parser.add_argument("--keep", action="store_true", help="do not delete the temp dir")
    parser.add_argument("--debug-cli", action="store_true",
                        help="drive the debug `pie` (slow: minutes just to load the model)")
    parser.add_argument("--verbose", action="store_true",
                        help="mirror the engine's own log, not just the inferlet's output")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    wasm = build_guest()
    binary = build_cli(args.features, release=not args.debug_cli)

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pie-frames-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Out:    {out_dir}")

    cmd = [
        str(binary), "run",
        "--path", str(wasm),
        "-o", str(out_dir),
        "--",
        "--width", str(args.width),
        "--height", str(args.height),
        "--frames", str(args.frames),
        "--video", args.video,
    ]
    print("Run:   ", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=args.timeout)
    sys.stdout.write(proc.stdout)
    # stderr only when something went wrong, or when asked: `pie run` boots a
    # whole engine and its INFO log buries the three lines that matter.
    if proc.returncode != 0 or args.verbose:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"❌ `pie run` exited {proc.returncode}")
        return 1

    # The inferlet returns a JSON summary naming the files it sent.
    start = proc.stdout.find('{"width"')
    assert start >= 0, f"no summary in the inferlet's output:\n{proc.stdout[-2000:]}"
    report = json.loads(proc.stdout[start:proc.stdout.find("}", start) + 1])
    print(f"Report: {report}")
    if report.get("video_fallback"):
        print(f"⚠️  the clip fell back to y4m: {report['video_fallback']}")

    failures = []
    checks = [
        ("gradient.png", lambda p: check_png(p, args.width, args.height)),
        ("tone.wav", check_wav),
    ]
    video_name = report["files"][1]
    if video_name.endswith(".mp4"):
        checks.append((video_name, lambda p: check_mp4(p, args.width, args.height, args.frames)))
    else:
        checks.append((video_name, lambda p: check_y4m(p, args.width, args.height, args.frames)))

    for name, check in checks:
        path = out_dir / name
        try:
            assert path.exists(), f"`pie run -o` did not write {name}"
            how = check(path)
            print(f"✅ {name:16s} {path.stat().st_size:>9,} bytes  (checked by {how})")
        except AssertionError as error:
            failures.append(f"{name}: {error}")
            print(f"❌ {name:16s} {error}")

    if not args.keep and not args.out:
        shutil.rmtree(out_dir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nAll files arrived and decode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
