#!/usr/bin/env python3
"""Mixed prefill/decode on one server: N decode streams run while cold
prefills of P tokens arrive every I seconds. What it isolates is how a
server shares the device between the two: the decodes' inter-token gaps
while prefills come in, and the prefills' TTFT while decodes run.

    mixed.py --url http://127.0.0.1:8080 --streams 32 --prefill-words 700 --every 1.0 --seconds 40
"""
import argparse, json, random, statistics, threading, time, http.client
from urllib.parse import urlparse


def stream(url, prompt, n, on_token=None):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t = time.perf_counter()
    c.request("POST", "/v1/completions", json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": n, "temperature": 0,
        "ignore_eos": True, "stream": True}), {"content-type": "application/json"})
    first = None
    for line in c.getresponse():
        if line.startswith(b"data:") and b"[DONE]" not in line and b'"text"' in line:
            now = time.perf_counter()
            first = first or now
            if on_token:
                on_token(now)
    return (first - t) if first else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--decode-tokens", type=int, default=2000)
    ap.add_argument("--prefill-words", type=int, default=700)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=40)
    a = ap.parse_args()
    rng = random.Random(time.time_ns())
    words = lambda n: " ".join("".join(rng.choice("abcdefghijklmnop") for _ in range(5)) for _ in range(n))

    stamps = [[] for _ in range(a.streams)]
    def decoder(i):
        stream(a.url, f"Story {i} {words(8)}:", a.decode_tokens, stamps[i].append)
    ds = [threading.Thread(target=decoder, args=(i,), daemon=True) for i in range(a.streams)]
    [d.start() for d in ds]
    time.sleep(8)  # every stream decoding
    t0 = time.perf_counter()
    ttfts, ps = [], []
    while time.perf_counter() - t0 < a.seconds:
        p = threading.Thread(target=lambda: ttfts.append(stream(a.url, words(a.prefill_words) + "\nSummary:", 1)), daemon=True)
        p.start(); ps.append(p)
        time.sleep(max(0.0, min(a.every, a.seconds - (time.perf_counter() - t0))))
    [p.join() for p in ps]
    t1 = time.perf_counter()
    gaps = sorted(1000 * (y - x) for s in stamps for x, y in zip(s, s[1:]) if t0 <= x and y <= t1)
    tokens = sum(1 for s in stamps for x in s if t0 <= x <= t1)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))]
    tt = sorted(1000 * x for x in ttfts if x)
    print(json.dumps({
        "url": a.url, "streams": a.streams, "prefills": len(tt),
        "decode_tok_per_s": round(tokens / (t1 - t0), 1),
        "itl_ms": {p: round(q(gaps, p), 1) for p in (50, 90, 99)},
        "itl_mean_ms": round(statistics.mean(gaps), 1),
        "prefill_ttft_ms": {p: round(q(tt, p)) for p in (50, 90, 99)},
    }))


if __name__ == "__main__":
    main()
