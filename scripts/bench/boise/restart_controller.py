"""The controller restarts while a long turn streams. Nothing else does:
the gateway and the workers dial it again and register as who they were,
the stream goes on, and turns launched after route as before."""
import json, http.client, time, subprocess, re, threading
R = "/root/Workspace/pie/target/release"
PROMPT = " ".join(f"[ctl1] Entry {i}: the fox {i} met the owl {i%5}." for i in range(80)) + "\nNow write a long story:"
def stream(n=2000, on=None):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
    c.request("POST", "/v1/completions", json.dumps({"model":"default","prompt":PROMPT,"max_tokens":n,"temperature":0,"stream":True,"ignore_eos":True}), {"content-type":"application/json"})
    r = c.getresponse(); pieces, last, gap, at = [], None, 0, None
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
        now = time.time()
        if last and now - last > gap: gap, at = now - last, len(pieces)
        last = now
        pieces.append(json.loads(line[5:])["choices"][0].get("text", ""))
        if on and len(pieces) == 50: on(); on = None
    return pieces, (round(gap * 1000), at)
def text(log): return re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="ignore").read())
def ids(log): return sorted(set(re.findall(r"worker_id=worker#(\d+)", text(log))))
before = {log: ids(log) for log in ("wa.log", "wb.log")}
# A first full run caches the prompt and what it decodes; later runs read
# those pages and agree with each other, not with the first (frame 51 on).
stream()
ref, ref_gap = stream()
print(f"reference: longest pause {ref_gap[0]} ms at frame {ref_gap[1]}")
def restart():
    global killed_at
    subprocess.run(["pkill", "-x", "pie-controller"]); killed_at = time.time(); time.sleep(1)
    subprocess.Popen([f"{R}/pie-controller", "--listen", "127.0.0.1:7000", "--metrics-addr", "127.0.0.1:9300"],
                     stdout=open("ctl2.log", "w"), stderr=subprocess.STDOUT)
    print("controller restarted after 50 frames")
got, gap = stream(on=lambda: threading.Thread(target=restart).start())
print(f"stream: {len(got)} frames, longest pause {gap[0]} ms at frame {gap[1]}, equal to before: {got == ref}")
time.sleep(8)
for log in ("wa.log", "wb.log"):
    again = "registered with the controller again" in text(log)
    print(f"{log}: ids {before[log]} -> {ids(log)}, registered again: {again}, aborted: {'restarting worker' in text(log)}")
print("gateway registered again:", "gateway re-registered" in text("gw.log"))
print("controller sees:", re.findall(r"routing table updated epoch=\d+ workers=\d+", text("gw.log"))[-1])
for i in range(4):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=120)
    c.request("POST", "/v1/completions", json.dumps({"model":"default","prompt":f"after {i}: hello","max_tokens":8}), {"content-type":"application/json"})
    r = c.getresponse(); print("turn after:", r.status, json.loads(r.read()).get("choices", [{}])[0].get("text", "")[:30].__repr__())
