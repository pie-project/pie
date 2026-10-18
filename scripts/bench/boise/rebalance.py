import json, http.client, time, threading, sys
A, N, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
done = []
def stream(i):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=900)
    p = f"Stream {i}: " + " ".join(f"Entry {k}: fox {k} met owl {k%5}." for k in range(60)) + " Story:"
    c.request("POST", "/v1/completions", json.dumps({"model":"default","prompt":p,"max_tokens":T,"temperature":0,"stream":True,"ignore_eos":True}), {"content-type":"application/json","x-plex-prop-worker":str(A)})
    r = c.getresponse(); t0 = time.time(); n = 0
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line: n += 1
    done.append((time.time() - t0, n))
t0 = time.time()
ts = [threading.Thread(target=stream, args=(i,)) for i in range(N)]
for t in ts: t.start(); time.sleep(0.05)
for t in ts: t.join()
wall = time.time() - t0
tokens = sum(n for _, n in done)
print(f"{N} streams of {T} tokens on worker {A}: wall {wall:.1f}s, {tokens/wall:.0f} tok/s, per-stream {min(d for d,_ in done):.1f}-{max(d for d,_ in done):.1f}s")
