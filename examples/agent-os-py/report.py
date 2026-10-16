"""Build one HTML report from results/*.json: quality against compute, with error bars.

    python3 report.py results/e1-*.json results/e3-*.json > report.html

Each point is one run: x = decode tokens per problem (the compute the engine actually
spent), y = accuracy, bars = one binomial standard error. Points are grouped by the
strategy that produced them, so a strategy that is better at the same compute sits
above the fixed-width line, and one that is cheaper at the same accuracy sits left.
"""

import json
import math
import sys

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent OS results</title><style>
:root{--bg:#fbfbfd;--fg:#1d1d1f;--mut:#6e6e73;--card:#fff;--line:#e5e5ea}
@media(prefers-color-scheme:dark){:root{--bg:#000;--fg:#f5f5f7;--mut:#a1a1a6;--card:#1c1c1e;--line:#2c2c2e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0;overflow-x:auto}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}th{color:var(--mut)}
svg{width:100%;height:auto}.ax{stroke:var(--line)}.lb{fill:var(--mut);font-size:11px}.leg span{margin-right:14px;color:var(--mut);font-size:12px}.leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
</style></head><body><main>
<h1>Quality at fixed compute</h1><div class="sub" id="sub"></div>
<div class="card"><div class="leg" id="leg"></div><svg id="ch" viewBox="0 0 960 460"></svg></div>
<div class="card"><table id="tb"></table></div>
<script>
const R = __DATA__;
const COL = {fixed:"#8e8e93", adaptive:"#0071e3", council:"#bf5af2"};
document.getElementById("sub").textContent = `${R.length} runs · accuracy is exact-match on MATH-500 · bars are one binomial standard error`;
document.getElementById("leg").innerHTML = Object.entries(COL).map(([k,c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("");
const W=960,H=460,L=60,B=40,T=14,Rr=20;
const xs=R.map(r=>r.tok), xm=Math.max(...xs)*1.08, ym=Math.min(100, Math.max(...R.map(r=>r.acc+r.se))+6);
const X=v=>L+(W-L-Rr)*v/xm, Y=v=>T+(H-T-B)*(1-v/ym);
let s="";
for(let v=0;v<=ym;v+=10) s+=`<line class="ax" x1="${L}" x2="${W-Rr}" y1="${Y(v)}" y2="${Y(v)}"/><text class="lb" x="${L-8}" y="${Y(v)+4}" text-anchor="end">${v}%</text>`;
const step=Math.pow(10,Math.floor(Math.log10(xm/4)))*(xm/4/Math.pow(10,Math.floor(Math.log10(xm/4)))>5?5:xm/4/Math.pow(10,Math.floor(Math.log10(xm/4)))>2?2:1);
for(let v=0;v<=xm;v+=step) s+=`<line class="ax" x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H-B}"/><text class="lb" x="${X(v)}" y="${H-B+16}" text-anchor="middle">${Math.round(v)}</text>`;
s+=`<text class="lb" x="${(L+W)/2}" y="${H-6}" text-anchor="middle">decode tokens per problem (compute)</text>`;
const fixed=R.filter(r=>r.kind==="fixed").sort((a,b)=>a.tok-b.tok);
s+=`<polyline fill="none" stroke="${COL.fixed}" stroke-width="1.5" stroke-dasharray="4 3" points="${fixed.map(r=>X(r.tok)+","+Y(r.acc)).join(" ")}"/>`;
R.forEach(r=>{const c=COL[r.kind]||"#888";
 s+=`<line stroke="${c}" x1="${X(r.tok)}" x2="${X(r.tok)}" y1="${Y(r.acc-r.se)}" y2="${Y(r.acc+r.se)}"/><circle cx="${X(r.tok)}" cy="${Y(r.acc)}" r="5.5" fill="${c}"><title>${r.name}: ${r.acc.toFixed(1)}% at ${r.tok} tok/problem (n=${r.n})</title></circle><text class="lb" x="${X(r.tok)+8}" y="${Y(r.acc)-8}">${r.name}</text>`;});
document.getElementById("ch").innerHTML=s;
document.getElementById("tb").innerHTML=`<tr><th>strategy</th><th>problems</th><th>accuracy</th><th>tokens/problem</th><th>acc per 1k tokens</th><th>agg tok/s</th><th>prefill saved</th><th>max mem</th></tr>`+
 R.slice().sort((a,b)=>a.tok-b.tok).map(r=>`<tr><td>${r.name}</td><td>${r.n}</td><td>${r.acc.toFixed(1)}% ± ${r.se.toFixed(1)}</td><td>${r.tok}</td><td>${(r.acc/(r.tok/1000)).toFixed(1)}</td><td>${r.agg}</td><td>${r.saved.toLocaleString()}</td><td>${r.mem}%</td></tr>`).join("");
</script></main></body></html>"""


def point(path):
    r = json.load(open(path))
    s = r["summary"]
    name = s["policy"]
    kind = "fixed" if name.startswith("fixed") else "council" if name.startswith("council") else "adaptive"
    n, acc = s["problems"], s["accuracy"]
    se = math.sqrt(acc * (1 - acc) / n) * 100
    label = path.split("/")[-1].rsplit(".", 1)[0]
    return {"name": label, "kind": kind, "n": n, "acc": acc * 100, "se": se, "tok": round(s["decode_tokens_per_problem"]),
            "agg": s.get("agg_tok_s"), "saved": s["prefill_saved_by_sharing"], "mem": r.get("memory", {}).get("max_used_pct", "?")}


if __name__ == "__main__":
    pts = [point(p) for p in sys.argv[1:]]
    sys.stdout.write(PAGE.replace("__DATA__", json.dumps(pts)))
