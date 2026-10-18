import json, urllib.request, time
def ask(p, headers={}):
    h={"content-type":"application/json"}; h.update(headers)
    req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":8,"temperature":0}).encode(), headers=h)
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300)); return r["usage"]["prompt_tokens_details"]["cached_tokens"], round(time.time()-t,2)
long = lambda tag: " ".join(f"[{tag}] fox {i} jumps over dog {i%11}." for i in range(260)) + " Answer:"
print("default (cost >=1024):", ask(long("d")))
print("prop never           :", ask(long("n"), {"x-plex-prop-offload":"never"}))
print("short, default       :", ask("Short question about foxes? " * 20))
print("short, always        :", ask("Short question about dogs, cats and birds? " * 40, {"x-plex-prop-offload":"always"}))
