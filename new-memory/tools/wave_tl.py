"""Per wave of a traced serve: the prefill fires, when their frames retire, and the first decode intervals.
Usage: python3 wave_tl.py <server log> [lanes of a full decode fire, 64]   (server run with PIE_WAVE_TRACE=1)
A wave is the prefill fires of one batch of prompts (a gap of over 1 s starts the next); times are ms from its first fire.
The last line counts the decode fires of a single row per wave ("one-row fires per 64-lane request"): a wave cut into
frames of one prompt each fires every lane's first row alone."""
import re
import statistics
import sys

path = sys.argv[1]
full = int(sys.argv[2]) if len(sys.argv) > 2 else 64
pat = re.compile(r'\[(step-lanes|wave-trace)\] t=(\d+)us (.*)')
ev = []
for line in open(path, errors='replace'):
    m = pat.search(line)
    if m:
        ev.append((int(m.group(2)) / 1000, m.group(1), m.group(3).strip()))


def f(text, key):
    m = re.search(r'(?<![a-z_])' + key + r'=(\d+)', text)
    return int(m.group(1)) if m else 0


# A wave starts at a prefill fire more than 1 s after the last prefill fire.
waves, last = [], -1e18
for i, (t, k, x) in enumerate(ev):
    if k == 'step-lanes' and f(x, 'prefill_lanes') > 0 and f(x, 'tokens') > 64:
        if t - last > 1000:
            waves.append(i)
        last = t
def one_row(k, x):
    """A decode fire of a single lane: no prompt in it, one row."""
    return k == 'step-lanes' and f(x, 'prefill_lanes') == 0 and f(x, 'lanes') == 1


ones = []
for w, start in enumerate(waves):
    t0 = ev[start][0]
    end = waves[w + 1] if w + 1 < len(waves) else len(ev)
    pre, dec, ret, head = [], [], [], []
    for t, k, x in ev[start:end]:
        if k == 'step-lanes':
            if f(x, 'prefill_lanes') > 0:
                pre.append((t - t0, f(x, 'lanes'), f(x, 'tokens')))
            elif f(x, 'lanes') == full:
                dec.append(t - t0)
            elif not dec:
                head.append(f(x, 'lanes'))
        elif x.startswith('retire'):
            ret.append((t - t0, f(x, 'lanes')))
    rows = sum(p[2] for p in pre)
    print(f'wave {w}: {len(pre)} prefill step-lanes lines, rows {rows}, lanes/line {sorted(set(p[1] for p in pre))}, first +0.0 last +{pre[-1][0]:.1f} ms')
    if len(pre) <= 8:
        print('   prefill lines: ' + ' '.join(f'+{t:.1f}({l}l,{n}r)' for t, l, n in pre))
    # The decode fires ahead of the first full one: each lane's first row
    # rides its prompt's frame, so a frame of one prompt fires one row alone.
    ones.append(sum(one_row(k, x) for _, k, x in ev[start:end]))
    widths = ' '.join(f'{n}x{l}' for l in sorted(set(head), reverse=True) for n in [head.count(l)])
    print(f'   one-row decode fires {ones[-1]}; decode fires ahead of the first {full}-lane one, fires x lanes: {widths or "none"}')
    if dec:
        firsts = [dec[0]] + [b for a, b in zip(dec, dec[1:]) if b - a > 1.0]
        print(f'   first {full}-lane decode fire +{dec[0]:.1f} ms; distinct decode frames {len(firsts)}')
        iv = [b - a for a, b in zip(firsts, firsts[1:])]
        print('   decode frame intervals ms: ' + ' '.join(f'{x:.1f}' for x in iv[:20]))
        if len(iv) > 12:
            med = statistics.median(iv[6:])
            print(f'   interval median after the 6th: {med:.2f} ms; the first 6 over 6 medians: {sum(iv[:6]) - 6 * med:.1f} ms')
        r = [x for x in ret if x[0] <= firsts[min(6, len(firsts) - 1)]]
        if len(r) <= 14:
            print('   retires to the 6th decode frame: ' + ' '.join(f'+{t:.1f}({l})' for t, l in r))
        else:
            print(f'   retires to the 6th decode frame: {len(r)}; first +{r[0][0]:.1f}({r[0][1]}) ... ' + ' '.join(f'+{t:.1f}({l})' for t, l in r[-6:]))
total = sum(one_row(k, x) for _, k, x in ev)
if ones:
    print(f'one-row fires per {full}-lane request: {" ".join(map(str, ones))} (mean {sum(ones) / len(ones):.1f}); in the run {total}, outside the waves {total - sum(ones)}')
else:
    print(f'one-row fires in the run {total}; no wave of prompts found')
