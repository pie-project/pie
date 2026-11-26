"""Host waits in step gaps. Reads only: CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_RUNTIME, OSRT_API (no arguments), OSRT_CALLCHAINS (symbol only), ThreadNames, StringIds."""
import sqlite3,sys,collections,re,statistics as st
db,out=sys.argv[1],sys.argv[2]
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
S=dict(c.execute('select id,value from StringIds'))
TN={g:S.get(n,'?') for n,p,g in c.execute('select nameId,priority,globalTid from ThreadNames')}
K=c.execute('select start,end,correlationId,shortName,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
R=c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
rt={r[3]:r for r in R}; rth=collections.defaultdict(list)
for r in R: rth[r[2]].append(r)
G=collections.OrderedDict()
for k in K: G.setdefault(k[2],[]).append(k)
nm=lambda k:S.get(k[4],'') or S.get(k[3],'')
PRE=re.compile(r'gemm_relu_bf16_256x128|128x128_ldg8_f2f_stages_64x3')
steps=[]
for corr,ks in G.items():
    ns=[nm(k) for k in ks]
    if any('BatchDecodeWithPagedKVCache' in n for n in ns): steps.append(dict(start=min(k[0] for k in ks),end=max(k[1] for k in ks),pre=any(PRE.search(n) for n in ns),launch=rt.get(corr)))
steps.sort(key=lambda s:s['start'])
gaps=[(b['start']-a['end'],a,b) for a,b in zip(steps,steps[1:])]
O=c.execute('select start,end,globalTid,nameId,callchainId from OSRT_API').fetchall()
CC=collections.defaultdict(list)
for i,sym,d in c.execute('select id,symbol,stackDepth from OSRT_CALLCHAINS order by id,stackDepth'): CC[i].append(S.get(sym,'?'))
o=open(out,'w'); w=lambda *x:print(*x,file=o)
t0=K[0][0]
w(f'# {db.split("/")[-1]}: {len(steps)} steps, prefill {sum(s["pre"] for s in steps)}; window {(K[-1][1]-t0)/1e9:.2f} s; gaps sum {sum(g for g,_,_ in gaps)/1e6:.1f} ms')
long=[x for x in gaps if x[0]>60e6]; ordin=sorted([x for x in gaps if 3e6<=x[0]<=50e6],key=lambda x:-x[0])[:6]
def clean(s):
    s=re.sub(r'::h[0-9a-f]{16}$','',s); return s[:110]
def show(tag,gp,a,b):
    lo,hi=a['end'],b['start']; lt=b['launch'][2] if b['launch'] else None
    w(f'\n## {tag}: gap {gp/1e6:.1f} ms at +{(lo-t0)/1e9:.3f} s, next step prefill={int(b["pre"])}; launching thread: {TN.get(lt,"?")} ({lt})')
    cu=collections.Counter(); cun=collections.Counter()
    for r in rth.get(lt,[]):
        if r[1]>=lo and r[0]<=hi: cu[S.get(r[4],'?')]+=min(r[1],hi)-max(r[0],lo); cun[S.get(r[4],'?')]+=1
    w(f'   launching thread CUDA calls: {sum(cun.values())} calls, {sum(cu.values())/1e6:.1f} ms; '+'; '.join(f'{k.split("_v")[0]} n{cun[k]} {v/1e6:.1f}ms' for k,v in cu.most_common(6)))
    per=collections.defaultdict(lambda:collections.Counter()); pern=collections.defaultdict(lambda:collections.Counter()); stk=collections.defaultdict(collections.Counter)
    for s,e,g,n,cc in O:
        if e>=lo and s<=hi:
            d=min(e,hi)-max(s,lo); name=S.get(n,'?'); per[g][name]+=d; pern[g][name]+=1
            if cc in CC:
                fr=[clean(f) for f in CC[cc] if f and not re.match(r'^(syscall|__|pthread_|futex|epoll|poll|nanosleep|sem_|clock_|<unknown>|0x)',f)]
                fr=[f for f in fr if not re.search(r'parking_lot|lock_api|tokio..runtime|core..ops..function|std..sys|std..thread|futex|_ZN4core|RNvNtCs.*core',f)] or fr
                stk[(g,name)][' < '.join(fr[:5])]+=d
    span=hi-lo
    rows=sorted(((sum(v.values()),g) for g,v in per.items()),reverse=True)[:10]
    lv=per.get(lt,{}); w(f'   launching thread OS-runtime calls >=10 us: {sum(pern[lt].values()) if lt in pern else 0} calls, {sum(lv.values())/1e6 if lv else 0:.1f} ms ({100*(sum(lv.values()) if lv else 0)/span:.1f}% of the gap): '+'; '.join(f'{k} n{pern[lt][k]} {v/1e6:.2f}ms' for k,v in (lv.most_common(5) if lv else [])))
    tw=[g for g in per if TN.get(g,'').startswith('tokio-rt-worker')]; w(f'   tokio-rt-worker threads with calls in the gap: {len(tw)}; their futex time {sum(per[g]["futex"] for g in tw)/1e6:.0f} ms over {sum(pern[g]["futex"] for g in tw)} calls')
    w(f'   OS-runtime calls >=10 us overlapping the gap, per thread (time clipped to the gap; % of gap):')
    for tot,g in rows:
        w(f'     {TN.get(g,"?")[:28]:28} {"LAUNCH " if g==lt else "       "}{100*tot/span:5.1f}%  '+'; '.join(f'{k} n{pern[g][k]} {v/1e6:.1f}ms' for k,v in per[g].most_common(4)))
    w('   top backtraces (blocking calls, >=80 us), by time in the gap:')
    allst=collections.Counter()
    for (g,name),cnt in stk.items():
        for s,v in cnt.items(): allst[(TN.get(g,'?')[:20],name,s)]+=v
    for (tn,name,s),v in allst.most_common(8): w(f'     {v/1e6:6.1f} ms  {tn} {name}: {s}')
for i,(gp,a,b) in enumerate(sorted(long,key=lambda x:x[1]['end'])): show(f'long pre-prefill stretch {i+1}',gp,a,b)
for i,(gp,a,b) in enumerate(ordin): show(f'ordinary gap {i+1}',gp,a,b)
