import json, http.client, time, sys
A, B, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PROMPT = " ".join(f"[{TAG}] Entry {i}: the fox {i} met the owl {i%5}." for i in range(80)) + "\nNow write a long story about the fox and the owl:"
def stream(headers, n=2000):
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=600)
    body = {"model":"default","prompt":PROMPT,"max_tokens":n,"temperature":0,"stream":True,"ignore_eos":True}
    h = {"content-type":"application/json"}; h.update(headers)
    c.request("POST", "/v1/completions", json.dumps(body), h)
    r = c.getresponse(); pieces, ids, last, gap_at, gap = [], set(), None, None, 0
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]": continue
        d = json.loads(line[5:]); ids.add(d.get("id")); now = time.time()
        if last and now - last > gap: gap, gap_at = now - last, len(pieces)
        last = now; pieces.append(d["choices"][0].get("text", ""))
    return "".join(pieces), pieces, ids, gap_at, round(gap*1000)
stream({"x-plex-prop-worker": str(A)}, 1)                    # the prompt cached at A
still, sp, _, _, _ = stream({"x-plex-prop-worker": str(A)})
moved, mp, ids, at, gap = stream({"x-plex-prop-worker": str(A), "x-plex-prop-move-to": str(B), "x-plex-prop-move-after-ms": "1000"})
print(f"moved: {len(mp)} frames, one id: {len(ids)==1}, the move after frame {at} ({gap} ms)")
same_before = "".join(sp[:at]) == "".join(mp[:at])
print("identical before the move:", same_before)
seam = "".join(mp[at-8:at]) + " ‖ " + "".join(mp[at:at+8])
print("around the move:", repr(seam))
print("reference there:", repr("".join(sp[at-8:at]) + " ‖ " + "".join(sp[at:at+8])))
k = next((i for i,(x,y) in enumerate(zip(sp, mp)) if x != y), None)
print("first differing frame:", k, "of", len(mp))
