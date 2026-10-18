import json, urllib.request, time, sys
def prompt(tag, n=120):
    return " ".join(f"[{tag}] Item {i}: the quick brown fox number {i} jumps over lazy dog {i*7%13}." for i in range(n)) + "\nQuestion: what is the color of the fox? Answer:"
def ask(port, p, max_tokens=16):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps({"model":"default","prompt":p,"max_tokens":max_tokens,"temperature":0}).encode(), headers={"content-type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req, timeout=300)); return r, time.time()-t
# warm both (program installs, JIT)
for port in (8080, 8090): ask(port, "hello", 2)
for tag in sys.argv[1:]:
    p = prompt(tag)
    a, ta = ask(8080, p)          # home A, offloads prefill to B
    b, tb = ask(8090, prompt(tag+"-local"))  # B computes a same-size prompt itself, cold
    a2, ta2 = ask(8080, prompt(tag+"-local"))  # A: B already holds it -> offload hits B's cache
    print(f"{tag}: A offloaded {ta:.2f}s cached={a['usage']['prompt_tokens_details']['cached_tokens']}/{a['usage']['prompt_tokens']} | B local cold {tb:.2f}s | A offload, B warm {ta2:.2f}s same={a2['choices'][0]['text']==b['choices'][0]['text']}")
