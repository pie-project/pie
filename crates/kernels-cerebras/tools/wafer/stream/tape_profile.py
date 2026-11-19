"""tape_profile.py <program dir> <parts file> <tsc dir>: a tiled launch's profile by the stream's ops.
Each tape PE's stamp 2 is when its own words left it: the array has read the stream to the end of that
PE's words. An op's cycles: the time the stream crossed its last word less the time it crossed its first
(interpolated between tape PEs); ops with no stream words fall in the gap at their offset (merged into
the next op's first crossing). Prints a table, then the totals by kind and resolution."""
import glob, json, os, re, sys
import numpy as np

prog, parts, tdir = sys.argv[1:4]
meta = json.load(open(f"{prog}/stream.json"))
W, H = meta["fabric"]
gx, gy = meta["array"]
ntape = meta["tape"]
tw = int(re.search(r"const TW: u16 = (\d+);", open(f"{prog}/ttape.csl").read()).group(1))
files = sorted(glob.glob(f"{tdir}/tsc0_*.tsc"), key=lambda f: int(re.search(r"_(\d+)\.tsc", f).group(1)))
ts = np.concatenate([np.fromfile(f, "<u4") for f in files]).astype(np.int64).reshape(H, W, 3)
# image PEs at fabric (X + 1, Y)
img = ts[0:gy, 1:gx + 1]
t0 = img[..., 0].min()
rel = lambda v: (v - t0) % (1 << 32)
end_img = rel(img[..., 1]).max()
n = gx + 1
tape = []
for k in range(ntape):
    r, c = divmod(k, n)
    x = c if r % 2 == 0 else gx - c
    tape.append(rel(ts[gy + r, x, 2]))
tape0_raw = float(tape[0])
tape = np.array(tape, dtype=np.float64)
# Each PE's stamps are its own time since its launch arrived, and the launch reaches parts of the
# fabric at different times: the tape's timeline jumps where its snake enters a part launched
# later (or earlier). The stream through the tape is continuous, so a jump at a row's turn is
# spliced to the median step of the row before it; the first PE's own words (the array waiting for
# its last PE's launch) likewise.
step = np.diff(tape)
raw_jumps = []
for k in range(1, ntape):
    if k % n == 0:
        prev = step[max(0, k - n):k - 1]
        med = float(np.median(prev)) if len(prev) else float(np.median(step))
        if abs(step[k - 1] - med) > 0.5 * med:
            raw_jumps.append((k, step[k - 1] - med))
            step[k - 1] = med
first = float(np.median(step[:n]))
tape = np.concatenate([[first], first + np.cumsum(step)])
if raw_jumps:
    print("spliced launch jumps at tape PEs (k, cycles):", [(k, round(j)) for k, j in raw_jumps])
print(f"the first tape PE's own words left at {tape0_raw:.0f} of its own clock (the launch's wait), taken as {first:.0f}")
# tape k's own words end at stream offset (k + 1) * tw
off = np.concatenate([[0.0], (np.arange(ntape) + 1.0) * tw])
tim = np.concatenate([[0.0], tape])
tim = np.maximum.accumulate(tim)
ops = []
for ln in open(parts):
    m = re.match(r"\s+op (\d+) (\w+) \((\d+), (\d+)\): (\d+) cycles, (\d+) stream words", ln)
    if m:
        ops.append(dict(op=int(m[1]), what=m[2], at=f"{m[3]}x{m[4]}", priced=int(m[5]), words=int(m[6])))
pos = 0
prev_t = 0.0
rows = []
for o in ops:
    a, b = pos, pos + o["words"]
    pos = b
    if o["words"] == 0:
        o["chip"] = None
        continue
    tb = float(np.interp(b, off, tim))
    o["chip"] = tb - prev_t  # (since the last op with words: the word-less ops before it too)
    prev_t = tb
print(f"launch: image PEs {end_img} cycles; stream {pos} words over {ntape} tape PEs of {tw}; last tape stamp {tape.max():.0f}")
print(f"{'op':>4} {'what':8} {'at':8} {'words':>9} {'priced':>10} {'chip':>11} {'ratio':>6}")
pend = 0
for o in ops:
    pend += o["priced"]
    if o["chip"] is None:
        continue
    print(f"{o['op']:>4} {o['what']:8} {o['at']:8} {o['words']:>9} {pend:>10} {o['chip']:>11.0f} {o['chip'] / max(pend, 1):>6.2f}")
    o["pend"] = pend
    pend = 0
print(f"stream end (spliced) {prev_t:.0f}; the longest PE's own clock {end_img}; the image PEs' ends: {np.percentile(rel(img[..., 1]), [0, 50, 100]).round()}")
by = {}
for o in ops:
    if o.get("chip") is not None:
        k = (o["what"], o["at"])
        p, c = by.get(k, (0, 0))
        by[k] = (p + o["pend"], c + o["chip"])
for k, (p, c) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k[0]:6} {k[1]:8}: priced {p / 1e6:7.2f}M chip {c / 1e6:7.2f}M ({c / max(p, 1):.2f}x)")
