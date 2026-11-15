"""Inside one wave of a traced serve: every gap between trace events, with the log lines that fall inside it.
Usage: python3 wave_gaps.py <server log> <wave, from 0> <until ms> <least gap ms>   (PIE_WAVE_TRACE=1, debug logging)
Reads the waves as wave_tl.py does; prints the gaps of at least <least gap ms> in the wave's first <until ms>."""
import re
import sys

path, wave, until, least = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
pat = re.compile(r'\[(step-lanes|wave-trace)\] t=(\d+)us (.*)')
ansi = re.compile(r'\x1b\[[0-9;]*m')
lines = [ansi.sub('', l.rstrip('\n')) for l in open(path, errors='replace')]
ev = []
for n, line in enumerate(lines):
    m = pat.search(line)
    if m:
        ev.append((int(m.group(2)) / 1000, m.group(1), m.group(3).strip(), n))
waves, last = [], -1e18
for i, (t, k, x, n) in enumerate(ev):
    m = re.search(r'prefill_lanes=(\d+)', x)
    r = re.search(r'tokens=(\d+)', x)
    if k == 'step-lanes' and m and int(m.group(1)) > 0 and int(r.group(1)) > 64:
        if t - last > 1000:
            waves.append(i)
        last = t
start = waves[wave]
t0 = ev[start][0]
prev = ev[start]
counts = {}
for e in ev[start + 1:]:
    if e[0] - t0 > until:
        break
    gap = e[0] - prev[0]
    if gap >= least:
        print(f'+{prev[0] - t0:7.1f} ms  gap {gap:6.1f} ms  after [{prev[2][:90]}]  until [{e[2][:90]}]  (events before it: {counts})')
        counts = {}
        between = [l for l in lines[prev[3] + 1:e[3]] if not pat.search(l)]
        for l in between[:12]:
            print('        | ' + l[:230])
        if len(between) > 12:
            print(f'        | ... {len(between) - 12} more lines')
    key = e[2].split()[0] if e[1] == 'wave-trace' else 'step'
    counts[key] = counts.get(key, 0) + 1
    prev = e
