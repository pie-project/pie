import json, urllib.request, time, sys
def ask(p, worker, n=12):
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":n,"temperature":0}).encode(), headers={"content-type":"application/json","x-plex-prop-worker":str(worker)})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300))
    return r["usage"]["prompt_tokens_details"]["cached_tokens"], r["usage"]["prompt_tokens"], round(time.time()-t,2), r["choices"][0]["text"]
step, worker = sys.argv[1], int(sys.argv[2])
p = " ".join(f"[pool2] Ledger {i}: owl {i} carried {i*3%17} acorns to tree {i%7}." for i in range(140)) + " Total acorns:"
print(step, ask(p, worker))
