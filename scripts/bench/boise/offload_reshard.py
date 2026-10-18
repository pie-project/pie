import json, urllib.request, time
def ask(p, headers={}, n=24):
    h={"content-type":"application/json"}; h.update(headers)
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":n,"temperature":0}).encode(), headers=h)
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300))
    return r["usage"]["prompt_tokens_details"]["cached_tokens"], round(time.time()-t,2), r["choices"][0]["text"]
for tag in ["t1","t2","t3"]:
    p = " ".join(f"[{tag}] Entry {i}: the heron {i} caught {i*5%23} fish at pond {i%6}." for i in range(130)) + " Summary of fish caught:"
    c1,t1,a = ask(p)                                   # prefilled on the TP=2 peer, resharded here
    c2,t2,b = ask(p.replace(f"[{tag}]", f"[{tag}]"), {"x-plex-prop-offload":"never"})  # cached now; recompute fresh below
    q = p + " "                                        # a distinct prompt computed here alone
    c3,t3,c = ask(p.replace("Summary", "Summary"), {"x-plex-prop-offload":"never"})
    print(f"{tag}: offloaded cached={c1} {t1}s -> {a!r}")
print("--- reference, computed on A alone ---")
for tag in ["t1","t2","t3"]:
    p = " ".join(f"[{tag}] Entry {i}: the heron {i} caught {i*5%23} fish at pond {i%6}." for i in range(130)) + " Summary of fish caught:"
    p = "​" + p  # same text, different first token: nothing cached
    print(tag, ask(p, {"x-plex-prop-offload":"never"})[2].__repr__())
print("--- B (TP=2) computing it itself ---")
def askb(p, n=24):
    req = urllib.request.Request("http://127.0.0.1:8090/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":n,"temperature":0}).encode(), headers={"content-type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["text"]
for tag in ["t1","t2","t3"]:
    p = "​​" + " ".join(f"[{tag}] Entry {i}: the heron {i} caught {i*5%23} fish at pond {i%6}." for i in range(130)) + " Summary of fish caught:"
    print(tag, repr(askb(p)))
