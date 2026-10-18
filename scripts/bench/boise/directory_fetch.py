import json, urllib.request, time
def ask(p, worker):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":12,"temperature":0}).encode(), headers={"content-type":"application/json","x-plex-prop-worker":str(worker)})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300))
    return r["usage"]["prompt_tokens_details"]["cached_tokens"], r["usage"]["prompt_tokens"], round(time.time()-t,2), r["choices"][0]["text"]
for tag in ["x", "y"]:
    p = " ".join(f"[{tag}] Ledger {i}: owl {i} carried {i*3%17} acorns to tree {i%7}." for i in range(140)) + " Total acorns:"
    c0,n,t0,a = ask(p, 0); time.sleep(1.5)   # computed on worker 0; reported to the directory
    c1,_,t1,b = ask(p, 1)                   # worker 1 holds none of it: fetches from worker 0
    print(f"{tag}: w0 computed cached {c0}/{n} {t0}s | w1 via directory cached {c1}/{n} {t1}s | same={a==b}")
