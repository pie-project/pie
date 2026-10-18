import json, urllib.request, time
def ask(port, p, n=16):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":n,"temperature":0}).encode(), headers={"content-type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300)); return r, time.time()-t
prefix = "System: you are a careful assistant who counts animals. " * 12
suffix = " ".join(f"Note {i}: a fox number {i} met a dog {i%9}." for i in range(150))
r1, t1 = ask(8080, prefix, 2)
print("turn1 A local", round(t1,2), r1["usage"])
p2 = prefix + suffix + "\nHow many foxes? Answer:"
r2, t2 = ask(8080, p2)
print("turn2 A offloaded", round(t2,2), r2["usage"], repr(r2["choices"][0]["text"]))
r3, t3 = ask(8090, p2)
print("same on B", round(t3,2), r3["choices"][0]["text"] == r2["choices"][0]["text"])
