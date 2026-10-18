"""A fixed load to profile an engine under: `prefill` sends N cold prompts
of W words one after another (one token each); `decode` runs S streams of
T tokens at once. Prints wall time per phase."""
import sys, time, threading, random
sys.path.insert(0, __import__("os").path.dirname(__file__))
from mixed import stream
url, mode = sys.argv[1], sys.argv[2]
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else time.time_ns())
words = lambda n: " ".join("".join(rng.choice("abcdefghijklmnop") for _ in range(5)) for _ in range(n))
t = time.perf_counter()
if mode == "prefill":
    ttfts = [stream(url, words(700) + "\nSummary:", 1) for _ in range(5)]
    print("prefill TTFTs ms", [round(x * 1000) for x in ttfts])
else:
    ts = [threading.Thread(target=stream, args=(url, f"Story {i} {words(8)}:", 200)) for i in range(32)]
    [x.start() for x in ts]; [x.join() for x in ts]
print(f"{mode}: {time.perf_counter() - t:.2f}s")
