import json, urllib.request, time
def ask(p, worker, n=8):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":n,"temperature":0}).encode(), headers={"content-type":"application/json","x-plex-prop-worker":str(worker)})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300))
    return r["usage"]["prompt_tokens_details"]["cached_tokens"], r["usage"]["prompt_tokens"], round(time.time()-t,2), r["choices"][0]["text"]
mk = lambda tag: " ".join(f"[{tag}] Ledger {i}: owl {i} carried {i*3%17} acorns to tree {i%7}." for i in range(140)) + " Total acorns:"
W0, W1 = 12, 13
ask("warm up", W0, 2); ask("warm up", W1, 2)
for tag in ["u", "v", "w", "x"]:
    c0,n,t0,x = ask(mk(tag), W0); time.sleep(1.5)
    c1,_,t1,y = ask(mk(tag), W1)                 # fetched from worker 0
    c2,_,t2,z = ask(mk(tag+"-own"), W1)          # worker 1 computes a same-size prompt itself
    print(f"{tag}: w0 computes {t0}s | w1 fetches {t1}s (cached {c1}/{n}) | w1 computes {t2}s | same={x==y}")
