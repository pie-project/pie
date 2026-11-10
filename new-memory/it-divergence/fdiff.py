import json, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
b0 = json.load(open(f'{D}/base-degenerate-frame0-wsA.json'))['steps']
b1 = json.load(open(f'{D}/base-degenerate-frame1-wsB.json'))['steps']
n = json.load(open(f'{D}/merged-shared-frame0-wsA-wsB.json'))['steps']
print('frames: base = 2 frames x', [len(s['lanes']) for s in b0 + b1], 'lane(s); merged = 1 frame x',
      [len(s['lanes']) for s in n], 'lanes')
print('steps per frame: base', len(b0), len(b1), 'merged', len(n),
      '; attachments', [s['attachments'] for s in b0 + b1 + n])
nl = n[0]['lanes']
for i, l in enumerate(nl):
    print(f"merged lane {i}: slot={l['slot']} first_token={l['tokens'][0]} rs_slot={l['rs_slot']} "
          f"kv.pages={l['kv']['pages']} tables={l['tables']} fixed={l['fixed']}")
ID = {'slot', 'kv', 'rs_slot', 'tables', 'fixed'}
for name, bl, first in [('wsA (prompt 9710..)', b0[0]['lanes'][0], 9710), ('wsB (prompt 10210..)', b1[0]['lanes'][0], 10210)]:
    ml = [l for l in nl if l['tokens'][0] == first][0]
    diffs = {k: (bl[k], ml.get(k)) for k in bl if bl[k] != ml.get(k)}
    other = {k: v for k, v in diffs.items() if k not in ID}
    print(f"{name}: lane index base 0 (own frame) vs merged {nl.index(ml)}; "
          f"id-only diffs: {sorted(set(diffs) & ID)}; other diffs: {other or 'none'}")
