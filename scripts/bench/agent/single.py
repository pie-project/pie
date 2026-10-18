"""Unloaded engine speed through the same API: a cold 3K-token prefill
(TTFT) and single-stream decode (TPOT), then N concurrent decodes."""
import json, sys, time, http.client, random, threading
host, port = sys.argv[1], int(sys.argv[2])
def run(prompt, n):
    c = http.client.HTTPConnection(host, port, timeout=600); t = time.perf_counter()
    c.request("POST", "/v1/completions", json.dumps({"model": "default", "prompt": prompt, "max_tokens": n,
              "temperature": 0, "ignore_eos": True, "stream": True}), {"content-type": "application/json"})
    first, last, k = None, None, 0
    for line in c.getresponse():
        if line.startswith(b"data:") and b"[DONE]" not in line and b'"text"' in line:
            now = time.perf_counter(); first = first or now; last = now; k += 1
    return first - t, (last - first) / max(1, k - 1)
rng = random.Random(int(time.time()))
words = lambda n: " ".join("".join(rng.choice("abcdefghij") for _ in range(5)) for _ in range(n))
cold = [run(words(1500) + "\nSummary:", 8)[0] for _ in range(3)]
print(f"cold prefill ~3K tokens: TTFT {[round(x*1000) for x in cold]} ms")
_, tpot = run("Write a long story about a fox.", 256)
print(f"single-stream TPOT {tpot*1000:.1f} ms")
for n in (8, 32):
    out = [None] * n
    ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, run(f"Story {i} {words(20)}:", 256))) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"{n} concurrent: TPOT mean {sum(o[1] for o in out)/n*1000:.1f} ms")
