import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    j = json.load(open(f))
    desc = []
    for s in j['steps']:
        lanes = []
        for l in s['lanes']:
            lanes.append(f"slot{l['slot']} t{len(l['tokens'])} pos{(l['positions'][:1] + l['positions'][-1:]) if l['positions'] else ''} "
                         f"grp={l['group']} rs={l['rs'] if isinstance(l['rs'], str) else list(l['rs'])[0]} rsr={l['rs_reset']} "
                         f"fixed={[(sh, p[:2]) for sh, p in l['fixed']]} tables={[(sh, len(r)) for sh, r in l['tables']]} "
                         f"mask={'y' if l['mask'] else 'n'} ro={l['readout'] if isinstance(l['readout'], str) else list(l['readout'])[0]}")
        desc.append(lanes)
    print(f[-11:], len(j['steps']), 'steps')
    for st in desc:
        for ln in st:
            print('   ', ln)
