#!/usr/bin/env python3
"""chk_stack.py <cslc out dir> [min bytes, default 768]: fails (exit 1) when any PE image leaves less
than `min` bytes between its used memory (cs_readelf -m: code, data and the task table) and 0xC000,
where the stack, which the linker does not count, grows down from. $CSL_SDK/cs_readelf reads the
images (CSL_SDK: the SDK directory)."""
import glob, os, re, subprocess, sys
out = sys.argv[1]
need = int(sys.argv[2]) if len(sys.argv) > 2 else 768
RE = os.path.join(os.environ["CSL_SDK"], "cs_readelf")
imgs = {}
for f in glob.glob(os.path.join(out, "bin", "out_*_*.elf")):
    m = re.match(r"out_(\d+)_(\d+)\.elf", os.path.basename(f))
    if m:
        imgs.setdefault(m.group(1), []).append(f)
worst, bad = None, []
for k, fs in sorted(imgs.items(), key=lambda kv: int(kv[0])):
    f = sorted(fs)[0]   # every chunk of an image holds the same program
    txt = subprocess.run([RE, "-m", f], capture_output=True, text=True).stdout
    used = max((int(v) for v in re.findall(r"\):\s*(\d+) bytes", txt)), default=0)
    room = 0xC000 - used
    print(f"image {k}: {used} B used, {room} B stack room ({len(fs)} files)")
    if worst is None or room < worst[1]:
        worst = (k, room)
    if room < need:
        bad.append(k)
if worst is None:
    sys.exit(f"{out}: no PE images")
print(f"STACK_ROOM min {worst[1]} B (image {worst[0]}); need >= {need}: {'FAIL ' + ','.join(bad) if bad else 'ok'}")
sys.exit(1 if bad else 0)
