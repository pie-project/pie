"""Top mutex-wait stacks by total wait. Reads only OSRT_API (start, end, thread, name, callchain id; no arguments),
OSRT_CALLCHAINS (symbol, depth), ThreadNames, StringIds."""
import sqlite3,sys,collections,re,subprocess
db,out=sys.argv[1],sys.argv[2]
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
S=dict(c.execute('select id,value from StringIds'))
TN={g:S.get(n,'?') for n,p,g in c.execute('select nameId,priority,globalTid from ThreadNames')}
CH=collections.defaultdict(list)
for i,sym,d in c.execute('select id,symbol,stackDepth from OSRT_CALLCHAINS order by id,stackDepth'): CH[i].append(S.get(sym,'?'))
fut={k for k,v in S.items() if v in('futex','pthread_mutex_lock','pthread_rwlock_wrlock','pthread_rwlock_rdlock')}
rows=c.execute('select start,end,globalTid,nameId,callchainId from OSRT_API where callchainId is not null').fetchall()
# demangle all symbols once
syms=sorted({f for v in CH.values() for f in v})
dm={}
p=subprocess.run(['rustfilt'],input='\n'.join(syms),capture_output=True,text=True) if False else None
def demangle(s):
    s=re.sub(r'::h[0-9a-f]{16}$','',s)
    return s
try:
    import cxxfilt
except Exception: cxxfilt=None
r=subprocess.run(['c++filt','-n'],input='\n'.join(syms),capture_output=True,text=True)
for a,b in zip(syms,r.stdout.split('\n')): dm[a]=demangle(b)
def is_lock(fr): return any(re.search(r'lock_slow|RawMutex|raw_mutex|RwLock|rwlock|Mutex.*lock',f) for f in fr)
o=open(out,'w'); w=lambda *x:print(*x,file=o)
w(f'# {db.split("/")[-1]}: OS-runtime waits >= 80 us that carry a backtrace (frame pointers), lock waits only; function names only')
for who,sel in (('pie-engine-0',lambda n:n=='pie-engine-0'),('tokio-rt-worker (all 27)',lambda n:n.startswith('tokio-rt-worker'))):
    agg=collections.Counter(); cnt=collections.Counter(); tot=0
    for s,e,g,nmid,ch in rows:
        if not sel(TN.get(g,'')) or nmid not in fut or ch not in CH: continue
        fr=[dm.get(f,f) for f in CH[ch]]
        if not is_lock(fr): continue
        key=tuple(f[:150] for f in fr[1:13]); agg[key]+=e-s; cnt[key]+=1; tot+=e-s
    w(f'\n## {who}: lock waits with a backtrace: {sum(cnt.values())} waits, {tot/1e6:.1f} ms; distinct stacks {len(agg)}')
    for i,(k,v) in enumerate(agg.most_common(12)):
        w(f'\n[{i+1}] {v/1e6:.1f} ms in {cnt[k]} waits')
        for f in k: w('     '+f)
