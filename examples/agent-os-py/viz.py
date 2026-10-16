"""Turn a results/<label>.json (from bench.py) into one self-contained HTML page.

    python3 viz.py results/exp-adaptive.json > agents.html

One timeline per problem: every lane is a bar from its spawn wave to the wave it
ended, coloured by how it ended; the strategist's decisions (spawn, kill, finish,
wait, deny) sit on the wave axis with the reason it gave. The ledger and the vote
tally are beside it. Nothing here is estimated except fields marked 'est'.
"""

import json
import sys

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent OS run</title>
<style>
:root{--bg:#fbfbfd;--fg:#1d1d1f;--mut:#6e6e73;--card:#fff;--line:#e5e5ea;--ok:#1a7f37;--bad:#cf222e;--amb:#bf8700;--blue:#0071e3;--gray:#8e8e93}
@media(prefers-color-scheme:dark){:root{--bg:#000;--fg:#f5f5f7;--mut:#a1a1a6;--card:#1c1c1e;--line:#2c2c2e;--ok:#3fb950;--bad:#ff6b6b;--amb:#e3b341;--blue:#2997ff;--gray:#636366}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--mut);margin-bottom:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}.stat b{display:block;font-size:20px}.stat span{color:var(--mut);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:12px 0}
select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);max-width:100%}
.q{color:var(--mut);margin:6px 0 10px;white-space:pre-wrap;max-height:4.4em;overflow:hidden}
svg{width:100%;height:auto;display:block}.lbl{fill:var(--mut);font-size:11px}.tick{stroke:var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0;color:var(--mut);font-size:12px}.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}th{color:var(--mut);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600;color:#fff}
.right{color:var(--ok)}.wrong{color:var(--bad)}code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
</style></head><body><main>
<h1>Agent OS run</h1><div class="sub" id="sub"></div>
<div class="grid" id="stats"></div>
<div class="card"><label>Problem <select id="pick"></select></label><div class="q" id="q"></div>
<div class="legend" id="legend"></div><svg id="tl"></svg></div>
<div class="card"><h2>Decisions and why</h2><table id="ev"></table></div>
<script>
const R = __DATA__;
const COL = {answered:"var(--ok)", eos:"var(--blue)", looping:"var(--amb)", "max length":"var(--gray)", killed:"var(--bad)", running:"var(--blue)"};
const S = R.summary, P = R.problems, T = R.trace;
document.getElementById("sub").textContent = `${S.policy} · ${S.problems} problems · accuracy ${(S.accuracy*100).toFixed(1)}% · ${S.decode_tokens_per_problem} decode tokens per problem`;
const stat = (v,l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;
document.getElementById("stats").innerHTML =
  stat((S.accuracy*100).toFixed(1)+"%","accuracy") + stat(S.decode_tokens_per_problem,"decode tokens / problem") +
  stat(S.agg_tok_s+" tok/s","aggregate decode") + stat(S.peak_rows,"peak concurrent agents") +
  stat(S.spawned,"agents spawned") + stat(S.killed,"agents killed") +
  stat(S.prefill_saved_by_sharing.toLocaleString(),"prefill tokens saved by sharing") +
  stat((S.est_tokens_saved_by_kills||0).toLocaleString(),"decode tokens saved by kills (est)") +
  stat(S.waves,"waves");
const pick = document.getElementById("pick");
P.forEach(p => { const o = document.createElement("option"); o.value = p.id; o.textContent = `#${p.id} ${p.correct ? "✓" : "✗"} gold ${p.gold} → ${p.final} (${p.lanes} agents, ${p.tokens} tok)`; pick.appendChild(o); });
document.getElementById("legend").innerHTML = [["answered","answered"],["eos","finished"],["looping","looping (killed)"],["max length","hit length cap"],["killed","killed by strategist"]].map(([k,t]) => `<span><i style="background:${COL[k]}"></i>${t}</span>`).join("");
function draw() {
  const id = +pick.value, p = P.find(x => x.id === id);
  document.getElementById("q").textContent = (p.question || "").slice(0, 400);
  const ev = T.filter(e => e.problem === id);
  const lanes = {};
  ev.forEach(e => {
    if (e.action === "SPAWN") e.lanes.forEach(l => lanes[l] = {id: l, kind: e.kind, from: e.wave, to: null, why: "running", ans: null});
  });
  ev.forEach(e => {
    if ((e.action === "DONE" || e.action === "KILL") && lanes[e.lane]) { const L = lanes[e.lane]; if (L.to === null) { L.to = e.wave; L.why = e.action === "KILL" ? "killed" : e.why; L.ans = e.answer; L.tok = e.tokens || e.tokens_so_far; } }
  });
  const list = Object.values(lanes), maxW = Math.max(1, ...ev.map(e => e.wave), ...list.map(l => l.to ?? 0)) + 1;
  const W = 1000, rowH = 18, top = 22, left = 54, H = top + list.length * rowH + 40;
  const x = w => left + (W - left - 10) * w / maxW;
  let s = "";
  for (let w = 0; w <= maxW; w += Math.max(1, Math.ceil(maxW / 12))) s += `<line class="tick" x1="${x(w)}" x2="${x(w)}" y1="${top-6}" y2="${H-24}"/><text class="lbl" x="${x(w)}" y="${H-8}" text-anchor="middle">${w}</text>`;
  s += `<text class="lbl" x="${left}" y="12">wave (32 tokens each)</text>`;
  list.forEach((L, i) => {
    const y = top + i * rowH, end = L.to ?? maxW, c = COL[L.why] || COL.running;
    s += `<text class="lbl" x="${left-6}" y="${y+12}" text-anchor="end">${L.kind === "judge" ? "J" : "A"}${L.id}</text>`;
    s += `<rect x="${x(L.from)}" y="${y+2}" width="${Math.max(3, x(end)-x(L.from))}" height="${rowH-6}" rx="4" fill="${c}"><title>agent ${L.id} (${L.kind}) · ${L.why}${L.ans ? " · answer " + L.ans : ""} · ${L.tok || "?"} tokens</title></rect>`;
    if (L.ans) s += `<text class="lbl" x="${x(end)+4}" y="${y+12}">${L.ans}</text>`;
  });
  ev.filter(e => e.action === "FINISH").forEach(e => { s += `<line x1="${x(e.wave)}" x2="${x(e.wave)}" y1="${top-6}" y2="${H-24}" stroke="var(--blue)" stroke-dasharray="4 3"><title>${e.why}</title></line>`; });
  const tl = document.getElementById("tl");
  tl.setAttribute("viewBox", `0 0 ${W} ${H}`);
  tl.innerHTML = s;
  const rows = ev.filter(e => ["SPAWN","KILL","FINISH","WAIT","DENY"].includes(e.action));
  const tag = a => `<span class="pill" style="background:${{SPAWN:"var(--blue)",KILL:"var(--bad)",FINISH:"var(--ok)",WAIT:"var(--gray)",DENY:"var(--amb)"}[a]}">${a}</span>`;
  document.getElementById("ev").innerHTML = `<tr><th>wave</th><th></th><th>why</th><th>detail</th></tr>` + rows.map(e => {
    let d = "";
    if (e.action === "SPAWN") d = `${e.lanes.length} × ${e.kind} · shared prefix ${e.shared_prefix_tokens} tok → ${e.prefill_saved} prefill tokens saved`;
    if (e.action === "KILL") d = `agent ${e.lane} at ${e.tokens_so_far} tok · saved ≈ ${e.est_saved_tokens} (est)`;
    if (e.action === "FINISH") d = `final <code>${e.final}</code> · votes ${JSON.stringify(e.votes)} · saved ≈ ${e.est_saved_tokens} (est)`;
    return `<tr><td>${e.wave}</td><td>${tag(e.action)}</td><td>${e.why}</td><td>${d}</td></tr>`;
  }).join("");
}
pick.onchange = draw; draw();
</script></main></body></html>"""


def main(path):
    r = json.load(open(path))
    sys.stdout.write(PAGE.replace("__DATA__", json.dumps(r)))


if __name__ == "__main__":
    main(sys.argv[1])
