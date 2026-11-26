"""Step gaps from an nsys sqlite: reads only CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_RUNTIME, StringIds."""
import sqlite3,sys,collections,statistics as st,re
db,out=sys.argv[1],sys.argv[2]
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
S=dict(c.execute('select id,value from StringIds'))
K=c.execute('select start,end,correlationId,shortName,demangledName,streamId,graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start').fetchall()
R=c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start').fetchall()
rt_by_corr={r[3]:r for r in R}
byth=collections.defaultdict(list)
for r in R: byth[r[2]].append(r)
groups=collections.OrderedDict()
for k in K: groups.setdefault(k[2],[]).append(k)
def nm(k): return S.get(k[4],'') or S.get(k[3],'')
PRE=re.compile(r'gemm_relu_bf16_256x128|128x128_ldg8_f2f_stages_64x3')
steps=[]
for corr,ks in groups.items():
    names=[nm(k) for k in ks]
    if not any('BatchDecodeWithPagedKVCache' in n for n in names): continue
    r=rt_by_corr.get(corr); api=S.get(r[4],'?') if r else '?'
    steps.append(dict(corr=corr,start=min(k[0] for k in ks),end=max(k[1] for k in ks),prefill=any(PRE.search(n) for n in names),api=api,launch=r,nk=len(ks)))
steps.sort(key=lambda s:s['start'])
gaps=[]
for a,b in zip(steps,steps[1:]): gaps.append((b['start']-a['end'],a,b))
o=open(out,'w')
w=lambda *x: print(*x,file=o)
w(f'# {db.split("/")[-1]}: steps (graph/launch groups holding the decode kernel) {len(steps)}; window {(K[-1][1]-K[0][0])/1e9:.2f} s')
w(f'# step launch APIs: {dict(collections.Counter(s["api"] for s in steps))}')
w(f'# prefill-carrying steps {sum(s["prefill"] for s in steps)}, decode-only {sum(not s["prefill"] for s in steps)}')
dur=[(s['end']-s['start'])/1e3 for s in steps]
w(f'# step GPU span us: p50 {st.median(dur):.0f}, prefill p50 {st.median([(s["end"]-s["start"])/1e3 for s in steps if s["prefill"]] or [0]):.0f}, decode-only p50 {st.median([(s["end"]-s["start"])/1e3 for s in steps if not s["prefill"]] or [0]):.0f}')
g=[x[0]/1e3 for x in gaps]
w(f'# gaps (us) between step N end and step N+1 first kernel: n {len(g)} sum {sum(g)/1e3:.1f} ms p50 {st.median(g):.1f} p90 {sorted(g)[int(.9*len(g))]:.1f} max {max(g):.1f}')
edges=[-1e9,0,10,25,50,100,250,500,1000,2500,5000,1e12]
w('# histogram gap_us: '+', '.join(f'[{lo:g},{hi:g}): {sum(lo<=x<hi for x in g)}' for lo,hi in zip(edges,edges[1:])))
w('\n# 20 largest gaps')
w('rank\tgap_us\tstepN_prefill\tstepN1_prefill\tstepN_span_us\tN1_launch_api\tlaunch_api_start_minus_stepN_end_us\tlaunch_api_dur_us\tdevice_start_minus_launch_api_end_us')
big=sorted(gaps,key=lambda x:-x[0])[:20]
for i,(gp,a,b) in enumerate(big):
    l=b['launch']
    w(f"{i+1}\t{gp/1e3:.1f}\t{int(a['prefill'])}\t{int(b['prefill'])}\t{(a['end']-a['start'])/1e3:.0f}\t{b['api']}\t{(l[0]-a['end'])/1e3:.1f}\t{(l[1]-l[0])/1e3:.1f}\t{(b['start']-l[1])/1e3:.1f}" if l else f"{i+1}\t{gp/1e3:.1f}\t-")
w('\n# host side for the 20 largest gaps: CUDA runtime calls on the thread that launched step N+1, from step N end to step N+1 first kernel (name, start offset from step N end us, duration us)')
for i,(gp,a,b) in enumerate(big):
    l=b['launch']
    if not l: continue
    rows=[r for r in byth[l[2]] if r[1]>=a['end'] and r[0]<=b['start']]
    tot=collections.Counter(); 
    for r in rows: tot[S.get(r[4],'?')]+= (min(r[1],b['start'])-max(r[0],a['end']))
    busy=sum(tot.values())
    w(f"gap {i+1} {gp/1e3:.1f} us: {len(rows)} calls covering {busy/1e3:.1f} us; by name: "+"; ".join(f"{k} {v/1e3:.1f}" for k,v in tot.most_common(6)))
    for r in rows[:12]: w(f"   {S.get(r[4],'?')}\t+{(r[0]-a['end'])/1e3:.1f}\t{(r[1]-r[0])/1e3:.1f}")
