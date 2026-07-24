#!/usr/bin/env python3

import base64
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPLICATION = ROOT / "tests/policies/replication-report.json"
ROADMAP = ROOT / "plex_policy_reproducibility_roadmap.json"
PERFORMANCE = ROOT / "plex_policy_performance_report.json"
OUTPUT = ROOT / "plex_current_model_presentation.html"

GROUPS = {
    "Fairness & workflow": ["agentix", "vtc", "fairserve", "dlpm", "justitia"],
    "Routing": ["preble", "lmetric", "smetric", "routebalance"],
    "Cache & residency": ["continuum", "kvflow", "marconi", "ragcache", "peek", "saga"],
    "Pipeline": ["helium", "pard"],
}

MILESTONES = [
    ("435e8c23b", "Fairness", "PLAS/ATLAS, VTC, OIT/WSC, D²LPM, GPS"),
    ("f456ccf6f", "Routing", "E2, hotspot detection, session guards, fused routing"),
    ("3c67395bc", "Cache", "TTL, STE, hybrid state, PGDSF, cLPM, WA-LRU"),
    ("8f448e37f", "Pipeline", "Whole-DAG/TRT execution and proactive dropping"),
]

ASSETS = {
    "architecture": ROOT / "website/docs/overview/pie-arch.svg",
    "programmable": ROOT / "website/static/img/programmable-serving.svg",
    "tax": ROOT / "website/blog/_figures/programmability-tax.svg",
    "agent_benchmark": ROOT / "website/docs/overview/agent_latency_tput.svg",
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/svg+xml;base64,{payload}"


def evidence_label(level: str) -> str:
    return {
        "policy-kernel-reproduction": "Policy-kernel faithful",
        "decision-trace-parity-with-deferred-mechanics": "Decision-trace parity",
        "inspired-adaptation": "Inspired adaptation",
    }.get(level, level)


def status_class(level: str) -> str:
    if level == "policy-kernel-reproduction":
        return "faithful"
    if level == "decision-trace-parity-with-deferred-mechanics":
        return "parity"
    return "open"


def fmt_ratio(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:.2f}×"


def fmt_delta(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:+.2f}%"


def bar_chart(entries: list[dict[str, Any]]) -> str:
    rows = []
    selected = [entry for entry in entries if entry["id"] in sum(GROUPS.values(), [])]
    values = [
        max(float(entry.get("offline", {}).get("proxy_improvement_ratio", 1.0)), 1e-9)
        for entry in selected
    ]
    scale = max([math.log2(value) for value in values] + [1.0])
    for entry, value in zip(selected, values):
        width = max(2.0, 100 * math.log2(max(value, 1.0)) / scale)
        rows.append(
            f"""
            <div class="bar-row">
              <span class="bar-name">{esc(entry['id'])}</span>
              <span class="bar-track"><i style="width:{width:.2f}%"></i></span>
              <strong>{fmt_ratio(value)}</strong>
            </div>"""
        )
    return "".join(rows)


def live_chart(entries: list[dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        if entry["id"] not in sum(GROUPS.values(), []):
            continue
        live = entry.get("live")
        if not live:
            continue
        delta = float(live.get("throughput_delta_percent", 0.0))
        width = min(abs(delta) * 4, 100)
        rows.append(
            f"""
            <div class="delta-row">
              <span>{esc(entry['id'])}</span>
              <span class="delta-track"><i class="{'positive' if delta >= 0 else 'negative'}"
                style="width:{width:.2f}%"></i></span>
              <strong>{fmt_delta(delta)}</strong>
            </div>"""
        )
    return "".join(rows) or '<p class="muted">Live matrix not available.</p>'


def policy_card(
    entry: dict[str, Any],
    perf: dict[str, Any] | None,
    closure: dict[str, Any] | None,
) -> str:
    level = entry["evidence_level"]
    deferred = entry.get("deferred_mechanics", [])
    offline = (perf or {}).get("offline", {})
    live = (perf or {}).get("live")
    mechanics = (
        " · ".join(esc(item) for item in deferred)
        if deferred
        else "No deferred policy mechanic"
    )
    kernel = "".join(f"<li>{esc(item)}</li>" for item in entry["policy_kernel"])
    return f"""
    <article class="policy-card {status_class(level)}" data-policy="{esc(entry['id'])}">
      <header>
        <div>
          <span class="eyebrow">{esc(entry['id'])}</span>
          <h3>{esc(entry['title'])}</h3>
        </div>
        <span class="badge">{esc(evidence_label(level))}</span>
      </header>
      <ul class="kernel">{kernel}</ul>
      <div class="policy-metrics">
        <span><b>{fmt_ratio(offline.get('proxy_improvement_ratio'))}</b> offline trend</span>
        <span><b>{fmt_delta((live or {}).get('throughput_delta_percent'))}</b> live throughput</span>
        <span><b>{esc((closure or {}).get('closure_commit', '—'))}</b> closure commit</span>
      </div>
      <p class="deferred"><b>Boundary:</b> {mechanics}</p>
      <a href="{esc(entry['source_url'])}" target="_blank" rel="noreferrer">Primary paper ↗</a>
    </article>"""


def matrix(entries: list[dict[str, Any]]) -> str:
    cells = []
    closed = set(sum(GROUPS.values(), []))
    for entry in entries:
        level = entry["evidence_level"]
        cells.append(
            f"""<div class="matrix-cell {status_class(level)} {'closed' if entry['id'] in closed else ''}"
              title="{esc(entry['title'])}">
              <b>{esc(entry['id'])}</b><span>{esc(evidence_label(level))}</span>
            </div>"""
        )
    return "".join(cells)


def primitive_chart(roadmap: dict[str, Any]) -> str:
    catalog = roadmap["primitive_catalog"][:11]
    maximum = max((item["policy_count"] for item in catalog), default=1)
    return "".join(
        f"""
        <div class="primitive-row">
          <span>{esc(item['id'])}</span>
          <i style="width:{100 * item['policy_count'] / maximum:.1f}%"></i>
          <b>{item['policy_count']}</b>
        </div>"""
        for item in catalog
    )


def main() -> None:
    replication = read(REPLICATION)
    roadmap = read(ROADMAP)
    performance = read(PERFORMANCE)
    replication_entries = replication["entries"]
    perf_entries = performance["entries"]
    perf_by_id = {entry["id"]: entry for entry in perf_entries}
    closure_by_id = {entry["id"]: entry for entry in roadmap["entries"]}
    current_ids = sum(GROUPS.values(), [])
    current_entries = [
        entry for entry in replication_entries if entry["id"] in current_ids
    ]
    policy_cards = "".join(
        policy_card(entry, perf_by_id.get(entry["id"]), closure_by_id.get(entry["id"]))
        for entry in current_entries
    )
    group_sections = "".join(
        f"""
        <div class="group-chip">
          <b>{esc(group)}</b>
          <span>{len(policies)} policies</span>
          <small>{esc(' · '.join(policies))}</small>
        </div>"""
        for group, policies in GROUPS.items()
    )
    timeline = "".join(
        f"""
        <div class="milestone">
          <span>{esc(commit)}</span><b>{esc(name)}</b><p>{esc(detail)}</p>
        </div>"""
        for commit, name, detail in MILESTONES
    )
    policy_kernel = replication["evidence_counts"].get(
        "policy-kernel-reproduction", 0
    )
    decision_parity = replication["evidence_counts"].get(
        "decision-trace-parity-with-deferred-mechanics", 0
    )
    trends = sum(
        bool(entry.get("offline", {}).get("trend_reproduced")) for entry in perf_entries
    )
    robust = sum(
        bool(entry.get("offline", {}).get("robustness", {}).get("trend_stable"))
        for entry in perf_entries
    )
    live_entries = [entry for entry in perf_entries if entry.get("live")]
    token_equal = sum(
        bool(entry["live"].get("token_outputs_equal")) for entry in live_entries
    )
    assets = {name: data_uri(path) for name, path in ASSETS.items()}

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLEX Current-Model Policy Reproduction</title>
<style>
:root {{
  --ink:#182128; --muted:#64727d; --paper:#f7f4ed; --card:#fffefa;
  --line:#d9d4c7; --accent:#dd5f37; --blue:#276b8d; --green:#237a57;
  --gold:#b8871d; --shadow:0 18px 55px rgba(31,43,52,.10);
}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
a{{color:var(--blue)}} h1,h2,h3,p{{margin-top:0}} h2{{font-size:clamp(2rem,4vw,4.5rem);line-height:1.02;letter-spacing:-.045em}}
.wrap{{width:min(1220px,calc(100% - 40px));margin:auto}}
nav{{position:sticky;top:0;z-index:20;background:rgba(247,244,237,.9);backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}}
nav .wrap{{display:flex;align-items:center;gap:18px;min-height:58px}} nav b{{margin-right:auto}}
nav a{{font-size:.84rem;text-decoration:none;color:var(--muted)}} nav a:hover{{color:var(--ink)}}
section{{padding:90px 0;border-bottom:1px solid var(--line)}} .hero{{padding:110px 0 80px;overflow:hidden}}
.eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:800;color:var(--accent)}}
.hero h1{{font-size:clamp(3.8rem,10vw,8.6rem);line-height:.82;letter-spacing:-.075em;margin:.2em 0}}
.hero h1 em{{display:block;color:var(--accent);font-style:normal}} .lead{{max-width:780px;font-size:1.28rem;color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:42px}}
.metric{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:var(--shadow)}}
.metric b{{display:block;font-size:2.4rem;line-height:1}} .metric span{{color:var(--muted);font-size:.82rem}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:34px;align-items:center}} .visual{{background:#fff;border:1px solid var(--line);border-radius:24px;padding:18px;box-shadow:var(--shadow)}}
.visual img{{display:block;width:100%;max-height:540px;object-fit:contain}} .caption{{font-size:.8rem;color:var(--muted);margin:10px 4px 0}}
.group-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .group-chip{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}}
.group-chip b,.group-chip span,.group-chip small{{display:block}} .group-chip span,.group-chip small{{color:var(--muted)}} .group-chip small{{margin-top:8px}}
.matrix{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}} .matrix-cell{{min-height:82px;border:1px solid var(--line);border-radius:12px;padding:12px;background:#ece8df;opacity:.58}}
.matrix-cell.closed{{opacity:1;box-shadow:0 8px 25px rgba(0,0,0,.06)}} .matrix-cell b,.matrix-cell span{{display:block}} .matrix-cell span{{font-size:.68rem;color:var(--muted)}}
.matrix-cell.faithful{{border-color:#89b8a4;background:#e9f5ef}} .matrix-cell.parity{{border-color:#d7b765;background:#fbf4de}}
.chart-card{{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:var(--shadow)}}
.bar-row,.delta-row{{display:grid;grid-template-columns:110px 1fr 70px;gap:10px;align-items:center;margin:8px 0;font-size:.82rem}}
.bar-track,.delta-track{{height:10px;background:#e7e2d8;border-radius:99px;overflow:hidden}} .bar-track i,.delta-track i{{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#66a9c3);border-radius:99px}}
.delta-track i.negative{{background:linear-gradient(90deg,#e89b79,var(--accent))}} .delta-track i.positive{{background:var(--green)}}
.timeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .milestone{{border-top:4px solid var(--accent);background:var(--card);padding:16px;border-radius:0 0 14px 14px}}
.milestone span{{font:700 .72rem ui-monospace,monospace;color:var(--muted)}} .milestone p{{font-size:.8rem;color:var(--muted);margin:7px 0 0}}
.policy-toolbar{{display:flex;gap:10px;margin-bottom:18px}} .policy-toolbar input{{width:100%;padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:white;font:inherit}}
.policy-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .policy-card{{background:var(--card);border:1px solid var(--line);border-top:5px solid var(--green);border-radius:16px;padding:20px;box-shadow:0 8px 25px rgba(0,0,0,.045)}}
.policy-card.parity{{border-top-color:var(--gold)}} .policy-card header{{display:flex;gap:15px;justify-content:space-between}} .policy-card h3{{font-size:1.08rem;line-height:1.25;max-width:520px}}
.badge{{font-size:.67rem;white-space:nowrap;border-radius:99px;padding:6px 9px;background:#edf4ef;height:max-content}} .parity .badge{{background:#fbf1d4}}
.kernel{{padding-left:18px;color:var(--muted);font-size:.84rem}} .policy-metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.policy-metrics span{{background:#f2efe7;border-radius:10px;padding:9px;font-size:.7rem}} .policy-metrics b{{display:block;font-size:.85rem;color:var(--ink)}}
.deferred{{font-size:.75rem;color:var(--muted);margin:12px 0 6px}} .policy-card>a{{font-size:.78rem}}
.primitive-row{{display:grid;grid-template-columns:210px 1fr 28px;gap:10px;align-items:center;margin:10px 0}} .primitive-row i{{height:13px;background:linear-gradient(90deg,var(--accent),#f1ae78);border-radius:99px}}
.claim{{border-left:5px solid var(--accent);padding:15px 20px;background:var(--card);margin:12px 0}} .muted{{color:var(--muted)}}
footer{{padding:55px 0;color:var(--muted)}} @media(max-width:850px){{.metrics,.group-grid,.timeline{{grid-template-columns:1fr 1fr}}.split,.policy-grid{{grid-template-columns:1fr}}.matrix{{grid-template-columns:repeat(3,1fr)}}nav a{{display:none}}}}
@media print{{nav{{display:none}}section{{break-inside:avoid;padding:40px 0}}body{{background:white}}.policy-card,.visual,.chart-card,.metric{{box-shadow:none}}}}
</style>
</head>
<body>
<nav><div class="wrap"><b>PLEX · current-model closure</b><a href="#architecture">Architecture</a><a href="#evidence">Evidence</a><a href="#policies">Policies</a><a href="#roadmap">Roadmap</a></div></nav>

<main>
<section class="hero"><div class="wrap">
  <span class="eyebrow">Programmable LLM serving · evidence, not imitation</span>
  <h1>One runtime.<em>Seventeen policies.</em></h1>
  <p class="lead">PLEX expresses serving algorithms as typed, stateful programs. The current programming model now closes every policy that required no new primitive: thirteen policy-kernel reproductions and four decision-trace reproductions with explicit physical-mechanism boundaries.</p>
  <div class="metrics">
    <div class="metric"><b>17 / 17</b><span>current-model policies implemented</span></div>
    <div class="metric"><b>{policy_kernel} + {decision_parity}</b><span>policy-kernel / decision-trace reproductions</span></div>
    <div class="metric"><b>{trends} / 31</b><span>offline trends reproduced</span></div>
    <div class="metric"><b>{robust} / 31</b><span>stable across robustness seeds</span></div>
  </div>
</div></section>

<section id="architecture"><div class="wrap">
  <span class="eyebrow">Architecture</span><h2>Policies move out of the engine,<br>without entering the hot path.</h2>
  <div class="split">
    <div class="visual"><img src="{assets['programmable']}" alt="Programmable serving architecture"><p class="caption">Existing PIE visualization: programmable serving separates application intent, runtime policy, and optimized execution.</p></div>
    <div><div class="claim"><b>Typed decisions.</b><br>Set-oriented admit, route, schedule, cache, and feedback operations with versioned validation.</div>
    <div class="claim"><b>Async enactment.</b><br>vLLM reads cached immutable plans; queue pressure, stale snapshots, or unsafe output immediately fall back to native scheduling.</div>
    <div class="claim"><b>State after reality.</b><br>Input, service, movement, and action accounting commit from enacted feedback—not optimistic plan construction.</div></div>
  </div>
  <div class="visual" style="margin-top:28px"><img src="{assets['architecture']}" alt="PIE architecture"></div>
</div></section>

<section><div class="wrap">
  <span class="eyebrow">Closure map</span><h2>Four families, one contract.</h2>
  <div class="group-grid">{group_sections}</div>
  <div class="matrix" style="margin-top:26px">{matrix(replication_entries)}</div>
</div></section>

<section id="evidence"><div class="wrap">
  <span class="eyebrow">Executable evidence</span><h2>Every claim has a trace,<br>a state assertion, and a boundary.</h2>
  <div class="split">
    <div class="chart-card"><h3>Offline policy-kernel trend</h3><p class="muted">Log-scaled improvement over each proxy baseline; ratios are not presented as paper end-to-end speedups.</p>{bar_chart(perf_entries)}</div>
    <div class="chart-card"><h3>Live vLLM throughput delta</h3><p class="muted">{token_equal}/{len(live_entries)} live policies preserved token outputs in the recorded matrix.</p>{live_chart(perf_entries)}</div>
  </div>
  <div class="timeline" style="margin-top:26px">{timeline}</div>
</div></section>

<section><div class="wrap">
  <span class="eyebrow">Existing performance visual</span><h2>Programmability stays a control-plane cost.</h2>
  <div class="split">
    <div class="visual"><img src="{assets['tax']}" alt="Programmability tax visualization"></div>
    <div class="visual"><img src="{assets['agent_benchmark']}" alt="Agent benchmark visualization"></div>
  </div>
</div></section>

<section id="policies"><div class="wrap">
  <span class="eyebrow">Policy atlas</span><h2>Seventeen mechanisms,<br>shown at the claim boundary.</h2>
  <div class="policy-toolbar"><input id="policy-search" placeholder="Filter policy, paper, or mechanism…"></div>
  <div class="policy-grid" id="policy-grid">{policy_cards}</div>
</div></section>

<section id="roadmap"><div class="wrap">
  <span class="eyebrow">Beyond the current model</span><h2>The remaining fourteen need authority,<br>not another heuristic.</h2>
  <div class="split">
    <div class="chart-card"><h3>Primitive demand</h3>{primitive_chart(roadmap)}</div>
    <div>
      <div class="claim"><b>Current-model closure:</b> {roadmap.get('current_model_closed_count', 17)} policies.</div>
      <div class="claim"><b>New-primitive frontier:</b> 14 policies across {roadmap['new_primitive_count']} deduplicated primitive families.</div>
      <div class="claim"><b>Non-claim:</b> a stable proxy trend is not an exact paper ratio; decision parity is not physical mechanism parity.</div>
      <p class="muted">Examples include pause/resume, co-execution, cache movement, migration transactions, tool resources, stable-output completion, and workflow graph authority.</p>
    </div>
  </div>
</div></section>

<section><div class="wrap">
  <span class="eyebrow">Takeaway</span><h2>A serving paper can become<br>a loadable, replayable policy.</h2>
  <p class="lead">PLEX does not claim that one hook reproduces an entire serving system. It makes the boundary testable: algorithmic state and decisions live in the policy; trusted facts come from the host; physical actions are negotiated and acknowledged; unsupported semantics remain named roadmap items.</p>
</div></section>
</main>
<footer><div class="wrap">Generated from <code>replication-report.json</code>, <code>plex_policy_performance_report.json</code>, and <code>plex_policy_reproducibility_roadmap.json</code>. Self-contained HTML; print-friendly.</div></footer>
<script>
const input=document.getElementById('policy-search');
input.addEventListener('input',()=>{{
  const q=input.value.toLowerCase();
  document.querySelectorAll('.policy-card').forEach(card=>{{
    card.style.display=card.textContent.toLowerCase().includes(q)?'block':'none';
  }});
}});
</script>
</body></html>
"""
    OUTPUT.write_text(page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
