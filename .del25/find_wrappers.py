import re, subprocess, sys, json
src = open('crates/model-compiler/src/dsl.rs').read().split('\n')
syms = [l.strip() for l in open('.del25/list25.txt') if l.strip()]
# find all fn declarations with their line, and brace-depth ranges (naive but adequate: we
# just track the nearest preceding `fn NAME` at a lower indent)
fn_re = re.compile(r'^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+([A-Za-z0-9_]+)')
fns = []
for i,l in enumerate(src):
    m = fn_re.match(l)
    if m:
        fns.append((i, len(m.group(1)), m.group(2)))
def enclosing(lineno):
    best=None
    for (i,ind,name) in fns:
        if i < lineno:
            best=(i,ind,name)
        else:
            break
    return best
out={}
for s in syms:
    hits=[i for i,l in enumerate(src) if '"%s"'%s in l]
    entry=[]
    for h in hits:
        e=enclosing(h)
        entry.append({'line':h+1,'fn':e[2] if e else None,'fnline':e[0]+1 if e else None,'text':src[h].strip()})
    out[s]=entry
print(json.dumps(out, indent=1))
