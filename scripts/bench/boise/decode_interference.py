"""Decode streams on the home, long prompts arriving beside them: the
inter-token gaps of the streams, with the long prompts' prefill offloaded
or computed at home (`offload` prop)."""
import json, threading, time, http.client, sys, statistics
MODE = sys.argv[1]  # "cost" (offload) or "never"
STREAMS, DECODE, LONGS, EVERY = 8, 200, 6, 2.0
def stream(i, gaps):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
    body = json.dumps({"model":"default","prompt":f"Stream {i}: write a long story about owls and acorns.","max_tokens":DECODE,"temperature":0,"stream":True,"ignore_eos":True})
    c.request("POST", "/v1/completions", body, {"content-type":"application/json","x-plex-prop-offload":"never"})
    r = c.getresponse(); last = None
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            now = time.time()
            if last is not None: gaps.append(now - last)
            last = now
def long_prompt(k, ttfts):
    p = " ".join(f"[{MODE}{k}] Ledger {i}: owl {i} carried {i*3%17} acorns to tree {i%7}." for i in range(140)) + " Total acorns:"
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
    t = time.time()
    c.request("POST", "/v1/completions", json.dumps({"model":"default","prompt":p,"max_tokens":1,"temperature":0}), {"content-type":"application/json","x-plex-prop-offload":MODE})
    c.getresponse().read(); ttfts.append(time.time()-t)
gaps, ttfts = [], []
threads = [threading.Thread(target=stream, args=(i, gaps)) for i in range(STREAMS)]
for t in threads: t.start()
time.sleep(1.0)
longs = []
for k in range(LONGS):
    th = threading.Thread(target=long_prompt, args=(k, ttfts)); th.start(); longs.append(th); time.sleep(EVERY)
for t in threads + longs: t.join()
g = sorted(gaps)
q = lambda f: round(g[int(f*(len(g)-1))]*1000, 1)
print(f"{MODE}: decode gaps n={len(g)} p50={q(.5)}ms p90={q(.9)}ms p99={q(.99)}ms max={round(g[-1]*1000,1)}ms | long-prompt latency mean={round(statistics.mean(ttfts),2)}s")
