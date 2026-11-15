#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from baseline import DEFAULT_BUDGET, episode_specs, run_http, summarize  # noqa: E402

MATCH_TOL = 0.15


async def _run(driver: str, args, specs: list[dict]) -> list[dict]:
    if driver == "pie":
        from run import run_pie  # noqa: PLC0415 -- needs main's pie_client (venv-main)

        return await run_pie(args.pie_uri, specs, args.concurrency)
    return await run_http(args.base_url, args.model, specs, args.concurrency)


def _warm_specs(specs: list[dict], n: int) -> list[dict]:
    out = []
    for i in range(n):
        w = dict(specs[i % len(specs)])
        w["seed"] = 100_000 + i
        w["budget_tokens"] = min(int(w.get("budget_tokens", 2048)), 256)
        w["max_wall_s"] = 120.0
        out.append(w)
    return out


def _fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def _row(label: str, s: dict, driver: str) -> list[str]:
    pie_like = driver == "pie"
    return [
        label,
        _fmt(s.get("episodes_per_hour")),
        _fmt(s.get("gen_tokens_per_s")),
        _fmt(s["gen_tokens_per_episode"], 0),
        f"{s['success_rate']:.2f}",
        f"{s['mean_reward']:.3f}",
        _fmt(s["mean_wall_s"], 2),
        _fmt(s["requests_per_episode"], 1),
        _fmt(s["forward_passes_per_episode"], 0) if s["forward_passes_per_episode"] else "n/a",
        "0 (never re-read)" if pie_like else _fmt(s["resent_prompt_tokens_per_episode"], 0),
        f"{s['errors']}/{s['timeouts']}",
    ]


HEADER = ["driver", "team episodes/h", "gen tokens/s", "gen tokens/episode", "success rate", "mean reward", "mean wall/episode (s)",
          "requests (chunks)/episode", "forward passes/episode", "re-sent prompt tokens/episode", "errors/timeouts"]


def _table(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "|".join("---" for _ in HEADER) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pairs(a: list[dict], b: list[dict]) -> dict:
    sa = {r["input"]["seed"]: bool(r["result"]["success"]) for r in a if "error" not in r and "input" in r}
    sb = {r["input"]["seed"]: bool(r["result"]["success"]) for r in b if "error" not in r and "input" in r}
    seeds = sorted(set(sa) & set(sb))
    return {"seeds": len(seeds), "both": sum(sa[s] and sb[s] for s in seeds),
            "first_only": sum(sa[s] and not sb[s] for s in seeds), "second_only": sum(sb[s] and not sa[s] for s in seeds),
            "neither": sum(not sa[s] and not sb[s] for s in seeds)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="aggregate", choices=["aggregate", "maze"])
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8, help="K episodes per driver, seeds seed..seed+K-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8, help="episodes in flight at once (both drivers)")
    ap.add_argument("--budget-tokens", type=int, default=None, help="per agent (default 2048 aggregate, 8192 maze)")
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-wall-s", type=float, default=900.0, help="per-episode cap (both drivers)")
    ap.add_argument("--pie-uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--pie-driver", default="pie", choices=["pie", "http"],
                    help="'http' puts run_http in the pie slot: a bench self-test, not a comparison")
    ap.add_argument("--order", default="pie,http", help="which driver runs first")
    ap.add_argument("--warmup", type=int, default=1, help="short untimed episodes per driver first")
    ap.add_argument("--go-threshold", type=float, default=2.0, help="GO when the ratio is at least this (the plan's gate 1: 2x)")
    ap.add_argument("--out-dir", default=None, help="writes pie.jsonl, http.jsonl, bench.md, bench.json here")
    args = ap.parse_args(argv)

    budget = args.budget_tokens or DEFAULT_BUDGET.get(args.task, 2048)
    specs = episode_specs(args.task, args.n_agents, args.episodes, args.seed, budget=budget, chunk=args.chunk_tokens,
                          depth=args.depth, lam=args.lam, temperature=args.temperature, top_p=args.top_p,
                          max_wall_s=args.max_wall_s)
    slots = {"pie": args.pie_driver, "http": "http"}
    labels = {"pie": "pie (one inferlet)" if args.pie_driver == "pie" else "http (in pie slot: self-test)",
              "http": "http (/v1/completions)"}
    order = [d.strip() for d in args.order.split(",") if d.strip() in slots] or ["pie", "http"]
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    results, summaries = {}, {}
    for slot in order:
        driver = slots[slot]
        if args.warmup > 0:
            t = time.perf_counter()
            warm = asyncio.run(_run(driver, args, _warm_specs(specs, args.warmup)))
            bad = [w.get("error") for w in warm if "error" in w]
            print(f"[{slot}] warm-up: {len(warm)} episode(s) in {time.perf_counter() - t:.1f}s"
                  + (f", errors: {bad[:2]}" if bad else ""), file=sys.stderr)
        print(f"[{slot}] running {len(specs)} {args.task} episodes (N={args.n_agents}, concurrency "
              f"{args.concurrency}) on {driver} ...", file=sys.stderr)
        t = time.perf_counter()
        res = asyncio.run(_run(driver, args, specs))
        wall = time.perf_counter() - t
        results[slot], summaries[slot] = res, summarize(res, wall)
        s = summaries[slot]
        print(f"[{slot}] {wall:.1f}s: {s['episodes_per_hour']:.1f} episodes/h, success {s['success_rate']:.2f}, "
              f"{s['gen_tokens_per_episode']:.0f} gen tokens/episode, errors {s['errors']}", file=sys.stderr)
        for r in res:
            if "error" in r:
                print(f"[{slot}] seed {r.get('input', {}).get('seed')}: ERROR {r['error'][:300]}", file=sys.stderr)
        if args.out_dir:
            with open(os.path.join(args.out_dir, f"{slot}.jsonl"), "w") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")

    sp, sh = summaries["pie"], summaries["http"]
    eph_p, eph_h = sp.get("episodes_per_hour") or 0.0, sh.get("episodes_per_hour") or 0.0
    ratio = eph_p / eph_h if eph_h > 0 else float("inf")
    gp, gh = sp["gen_tokens_per_episode"], sh["gen_tokens_per_episode"]
    diff = (gp - gh) / gh if gh > 0 else float("inf")
    matched = abs(diff) <= MATCH_TOL
    pairs = _pairs(results["pie"], results["http"])
    clean = not (sp["errors"] or sp["timeouts"] or sh["errors"] or sh["timeouts"])
    verdict = ("GO" if ratio >= args.go_threshold else "NO-GO") if matched and clean else "NOT COMPARABLE"

    settings = (f"task **{args.task}**, N={args.n_agents} agents, K={args.episodes} episodes (seeds {args.seed}.."
                f"{args.seed + args.episodes - 1}), chunk {args.chunk_tokens}, budget {budget}/agent, depth "
                f"{args.depth}, temperature {args.temperature}, top_p {args.top_p}, concurrency {args.concurrency} "
                f"episodes, warm-up {args.warmup}/driver, order {','.join(order)}")
    servers = (f"pie: `{args.pie_uri}`" if args.pie_driver == "pie" else f"pie slot: run_http on `{args.base_url}`")
    servers += f"; http: `{args.base_url}` (model `{args.model}`)"
    md = [
        f"# team-episode bench ({_dt.datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        settings + ".",
        "",
        servers + ".",
        "",
        _table([_row(labels["pie"], sp, slots["pie"]), _row(labels["http"], sh, "http")]),
        "",
        f"**GO/NO-GO ratio** (pie episodes/h / http episodes/h): **{ratio:.2f}x** -> **{verdict}** "
        f"(threshold {args.go_threshold:.2f}x)."
        + ("" if clean else " Errors or wall-capped episodes on a side: episodes/h counts only episodes that "
                            "ended on their own, so the ratio is not a like-for-like comparison."),
        "",
        f"Matched output: mean generated tokens/episode pie {gp:.0f} vs http {gh:.0f} ({diff:+.1%}; tolerance "
        f"+-{MATCH_TOL:.0%}): " + ("matched." if matched else "**WARNING: not matched, the ratio is not comparable.**"),
        "",
        f"Same seeds ({pairs['seeds']}): both succeeded {pairs['both']}, pie only {pairs['first_only']}, http only "
        f"{pairs['second_only']}, neither {pairs['neither']}.",
        "",
        f"http prompt tokens sent/episode {sh['prompt_tokens_sent_per_episode']:,.0f}, of which re-sent context "
        f"{sh['resent_prompt_tokens_per_episode']:,.0f} (server prefix cache hit {sh['cached_prompt_tokens_per_episode']:,.0f}).",
        "",
        "Notes: team episodes/h = episodes that ended on their own (no error, no wall cap) over the batch wall time; "
        "gen tokens/s = all generated tokens over the batch wall time. requests = generation chunks (one HTTP request each for http; pie counts its chunks the same way). "
        "Wall per episode is from each driver's own clock. "
        "With vLLM as the http server, launch it with `--logprobs-mode processed_logprobs` (its default reports "
        "pre-temperature logprobs).",
    ]
    if not matched:
        print(f"WARNING: mean generated tokens per episode differ by {diff:+.1%} (pie {gp:.0f}, http {gh:.0f}); "
              f"the episodes/hour ratio is not comparable.", file=sys.stderr)
    if args.pie_driver != "pie":
        md.append("")
        md.append("**Self-test**: both rows ran the http driver; the ratio only checks the bench.")
    text = "\n".join(md)
    print(text)
    if args.out_dir:
        with open(os.path.join(args.out_dir, "bench.md"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(args.out_dir, "bench.json"), "w") as f:
            json.dump({"settings": vars(args), "budget_tokens": budget, "summaries": summaries, "ratio": ratio,
                       "gen_tokens_diff": diff, "matched": matched, "verdict": verdict, "pairs": pairs}, f, indent=1)


if __name__ == "__main__":
    main()
