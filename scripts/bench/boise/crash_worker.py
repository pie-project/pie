"""A long turn runs; the worker running it is killed (SIGKILL), with no
word to anyone. The gateway sees its link close and launches the turn again
on the other worker after what the client was sent. The client's stream
against the same turn run while nothing crashed."""
import json, http.client, time, threading, subprocess, re
PROMPT = " ".join(f"[crash1] Entry {i}: the fox {i} met the owl {i%5}." for i in range(80)) + "\nNow write a long story about the fox and the owl:"
def stream(n=2000, on_start=None):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
    c.request("POST", "/v1/completions", json.dumps({"model":"default","prompt":PROMPT,"max_tokens":n,"temperature":0,"stream":True,"ignore_eos":True}), {"content-type":"application/json"})
    r = c.getresponse(); pieces, ids, last, gap, gap_at = [], set(), None, 0, None
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
        d = json.loads(line[5:]); ids.add(d.get("id")); now = time.time()
        if last and now - last > gap: gap, gap_at = now - last, len(pieces)
        last = now; pieces.append(d["choices"][0].get("text", ""))
        if on_start and len(pieces) == 50: on_start(); on_start = None
    return pieces, ids, gap_at, round(gap*1000)
def where():
    # The worker the last launch was dispatched to.
    log = open("gw.log", errors="ignore").read()
    hits = re.findall(r'Dispatch \{ process: "\d+", worker: "(\d+)" \}', log)
    return hits[-1]
stream(1)                                   # the prompt cached
ref, _, ref_at, ref_gap = stream()
print(f"reference: longest pause {ref_gap} ms at frame {ref_at}")
def stop():
    w = where()
    for line in subprocess.run(["pgrep","-a","pie-worker"], capture_output=True, text=True).stdout.splitlines():
        pid, cmd = line.split(" ", 1)
        conf = re.search(r"-c (\S+)", cmd).group(1)
        log = re.sub(r"\d*\.toml$", ".log", conf)
        text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="ignore").read())
        if f"worker_id=worker#{w}" in text:
            print(f"SIGKILL to worker {w} ({conf}) after 50 frames"); subprocess.run(["kill", "-9", pid])
moved, ids, at, gap = stream(on_start=stop)
print(f"crashed: {len(moved)} frames, one id: {len(ids)==1}, longest pause {gap} ms at frame {at}")
print("identical to the run nothing crashed:", moved == ref)
# The relaunch prefills what was sent in one pass where the reference
# decoded it token by token: the KV differs in its last bits, and a greedy
# near-tie far on may go the other way. The frames up to it are the test.
if moved != ref:
    at = next(i for i, (a, b) in enumerate(zip(moved, ref)) if a != b)
    print(f"first difference at frame {at}:")
    print("  crashed:", repr("".join(moved[max(0, at - 3):at + 4])))
    print("  nothing:", repr("".join(ref[max(0, at - 3):at + 4])))
