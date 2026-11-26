"""pie-engine-0 OS-runtime waits by kind, per gap class. Reads KERNEL, RUNTIME, OSRT_API (no arguments), OSRT_CALLCHAINS (symbols), ThreadNames, StringIds."""
import sqlite3,sys,collections,re
db,out=sys.argv[1],sys.argv[2]
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
S=dict(c.execute('select id,value from StringIds'))
TN={g:S.get(n,'?') for n,p,g in c.execute('select nameId,priority,globalTid from ThreadNames')}
eng=[g for g,n in TN.items() if n=='pie-engine-0'][0]
K=c.execute('select start,end,correlationId,shortName,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
G=collections.OrderedDict()
for k in K: G.setdefault(k[2],[]).append(k)
nm=lambda k:S.get(k[4],'') or S.get(k[3],'')
PRE=re.compile(r'gemm_relu_bf16_256x128|128x128_ldg8_f2f_stages_64x3')
steps=sorted([dict(start=min(k[0] for k in ks),end=max(k[1] for k in ks),pre=any(PRE.search(nm(k)) for k in ks)) for ks in G.values() if any('BatchDecodeWithPagedKVCache' in nm(k) for k in ks)],key=lambda s:s['start'])
gaps=[(b['start']-a['end'],a['end'],b['start'],b['pre']) for a,b in zip(steps,steps[1:])]
chain_kind={}
for i,sym,d in c.execute('select id,symbol,stackDepth from OSRT_CALLCHAINS order by id,stackDepth'):
    f=S.get(sym,'')
    if i not in chain_kind or chain_kind[i]=='other':
        chain_kind[i]='mutex lock_slow (parking_lot)' if 'RawMutex9lock_slow' in f else 'channel park (crossbeam select)' if 'crossbeam_channel' in f else 'condvar wait (parking_lot)' if 'Condvar' in f else chain_kind.get(i,'other')
names=collections.defaultdict(str); names.update({k:v for k,v in S.items()})
rows=c.execute('select start,end,nameId,callchainId from OSRT_API where globalTid=?',(eng,)).fetchall()
def cls(lo,hi):
    tot=collections.Counter(); n=collections.Counter()
    for s,e,nmid,ch in rows:
        if e<lo or s>hi: continue
        d=min(e,hi)-max(s,lo); name=S.get(nmid,'?')
        k=name if name!='futex' else ('futex: '+(chain_kind.get(ch,'other') if ch else 'short (<80 us, no backtrace)'))
        tot[k]+=d; n[k]+=1
    return tot,n
o=open(out,'w'); w=lambda *x:print(*x,file=o)
win=(K[0][0],K[-1][1]); span=win[1]-win[0]
w(f'# {db.split("/")[-1]}: pie-engine-0 (the thread that issues every step launch); window {span/1e9:.2f} s; {len(steps)} steps')
def report(tag,ivs):
    T=collections.Counter(); N=collections.Counter(); L=sum(hi-lo for lo,hi in ivs)
    for lo,hi in ivs:
        t,n=cls(lo,hi); T.update(t); N.update(n)
    w(f'{tag}: {len(ivs)} interval(s), {L/1e6:.1f} ms; engine thread in OS-runtime waits {sum(T.values())/1e6:.1f} ms ({100*sum(T.values())/max(L,1):.0f}%)')
    for k,v in T.most_common(6): w(f'    {k:42} n{N[k]:6}  {v/1e6:8.1f} ms  {100*v/max(L,1):5.1f}%')
report('whole window',[win])
report('long pre-prefill stretches (>60 ms)',[(lo,hi) for g,lo,hi,p in gaps if g>60e6])
report('ordinary gaps 3-50 ms',[(lo,hi) for g,lo,hi,p in gaps if 3e6<=g<=50e6])
report('steps themselves (GPU busy)',[(s['start'],s['end']) for s in steps])
