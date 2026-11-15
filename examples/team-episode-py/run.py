
import argparse
import asyncio
import hashlib
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def build_program(episode_path: str | pathlib.Path | None = None, main_path: str | pathlib.Path | None = None) -> bytes:
    ep = pathlib.Path(episode_path) if episode_path else HERE / "episode.py"
    mn = pathlib.Path(main_path) if main_path else HERE / "main.py"
    return ep.read_bytes() + b"\n" + mn.read_bytes()


def flatten(traj: list[dict]) -> dict:
    tokens, mask, logprobs = [], [], []
    for seg in traj:
        toks = seg.get("tokens") or []
        tokens += toks
        mask += [seg["mask"]] * len(toks)
        logprobs += seg["logprobs"] if seg["mask"] and seg.get("logprobs") is not None else [None] * len(toks)
    return {"tokens": tokens, "mask": mask, "logprobs": logprobs}


async def install(client, program: bytes) -> str:
    name = f"team-episode-{hashlib.sha256(program).hexdigest()[:10]}.py"
    return await client.install_program_bytes(program, name)


async def _run_one(client, prog: str, inp: dict, sem: asyncio.Semaphore, flat: bool) -> dict:
    async with sem:
        t = time.monotonic()
        try:
            proc = await client.launch_process(prog, inp)
            out = await proc.result()
        except Exception as e:  # noqa: BLE001 -- one failed episode must not sink the batch
            return {"driver": "pie", "error": f"{type(e).__name__}: {e}", "input": inp,
                    "timing": {"client_wall_s": round(time.monotonic() - t, 3)}}
        if not isinstance(out, dict):
            return {"driver": "pie", "error": f"non-JSON return: {str(out)[:300]}", "input": inp,
                    "timing": {"client_wall_s": round(time.monotonic() - t, 3)}}
        out.setdefault("timing", {})["client_wall_s"] = round(time.monotonic() - t, 3)
        out["input"] = inp
        if flat and "trajectories" in out:
            out["flat"] = [flatten(traj) for traj in out["trajectories"]]
        return out


async def run_pie(uri: str, episodes: list[dict], concurrency: int, program: bytes | None = None,
                  flat: bool = True) -> list[dict]:
    from pie_client import PieClient

    client = PieClient(uri)
    await client.connect()
    try:
        prog = await install(client, program if program is not None else build_program())
        sem = asyncio.Semaphore(max(1, concurrency))
        return list(await asyncio.gather(*(_run_one(client, prog, dict(e), sem, flat) for e in episodes)))
    finally:
        await client.close()


def summarize(results: list[dict], wall_s: float) -> dict:
    done = [r for r in results if "error" not in r]
    ok = [r for r in done if not r.get("aborted")]  # aborted: the engine ended it early (e.g. KV full)
    n = len(ok)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    gen = sum(r["timing"].get("gen_tokens", 0) for r in done)  # work done, aborted episodes included
    return {
        "episodes": len(results),
        "errors": len(results) - len(done),
        "aborted": len(done) - n,
        "success_rate": mean(1.0 if r["result"].get("success") else 0.0 for r in ok),
        "mean_reward": mean(r["result"].get("reward", 0.0) for r in ok),
        "episodes_per_hour": n / wall_s * 3600 if wall_s > 0 else 0.0,
        "gen_tokens": gen,
        "gen_tokens_per_s": gen / wall_s if wall_s > 0 else 0.0,
        "mean_wall_s": mean(r["timing"].get("wall_s", 0.0) for r in ok),
        "mean_client_wall_s": mean(r["timing"].get("client_wall_s", 0.0) for r in ok),
        "mean_forward_passes": mean(r["timing"].get("forward_passes", 0) for r in ok),
        "mean_lanes": mean(r["timing"].get("mean_lanes", 0.0) for r in ok),
        "timeouts": sum(1 for r in ok if r["timing"].get("timeout")),
        "batch_wall_s": wall_s,
    }


def check(r: dict) -> list[str]:
    bad = []
    if "error" in r:
        return [r["error"]]
    for a, traj in enumerate(r["trajectories"]):
        if not traj or traj[0]["kind"] != "prompt":
            bad.append(f"agent {a}: trajectory does not start with its prompt")
        for i, seg in enumerate(traj):
            toks, lps = seg.get("tokens"), seg.get("logprobs")
            if seg["kind"] == "gen":
                if seg["mask"] != 1 or toks is None or lps is None or len(toks) != len(lps):
                    bad.append(f"agent {a} seg {i}: gen segment mask/tokens/logprobs mismatch")
                elif any(lp > 1e-4 for lp in lps):
                    bad.append(f"agent {a} seg {i}: a logprob above 0")
            elif seg["mask"] != 0 or lps is not None:
                bad.append(f"agent {a} seg {i}: {seg['kind']} segment must have mask 0, logprobs null")
    for a, f in enumerate(r.get("flat") or []):
        if not (len(f["tokens"]) == len(f["mask"]) == len(f["logprobs"])):
            bad.append(f"agent {a}: flat tokens/mask/logprobs lengths differ")
        if any((m == 1) != (lp is not None) for m, lp in zip(f["mask"], f["logprobs"])):
            bad.append(f"agent {a}: flat logprobs must be set exactly where mask is 1")
    for v in r["timing"].get("verify") or []:
        if v["max_abs_diff"] > 1.0:
            bad.append(f"agent {v['agent']}: re-scored logprobs differ from live ones by {v['max_abs_diff']}")
    if r["timing"].get("idle_decode_violations"):
        bad.append(f"{r['timing']['idle_decode_violations']} decode rows for agents that were not running")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--task", default="aggregate", choices=["aggregate", "maze"])
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="episode i runs seed + i")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget-tokens", type=int, default=2048)
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-wall-s", type=float, default=600.0)
    ap.add_argument("--verify", action="store_true",
                    help="re-score every trajectory teacher-forced in a fresh working set (timing.verify)")
    ap.add_argument("--episode-file", default=None, help="episode.py to concatenate (default: beside run.py)")
    ap.add_argument("--out", default=None, help="JSONL, one output object per episode")
    args = ap.parse_args()

    episodes = []
    for i in range(args.episodes):
        e = {"task": args.task, "n_agents": args.n_agents, "seed": args.seed + i,
             "budget_tokens": args.budget_tokens, "chunk_tokens": args.chunk_tokens, "depth": args.depth,
             "temperature": args.temperature, "top_p": args.top_p, "max_wall_s": args.max_wall_s}
        if args.lam is not None:
            e["lam"] = args.lam
        if args.verify:
            e["verify"] = True
        episodes.append(e)

    program = build_program(args.episode_file)
    t = time.monotonic()
    results = asyncio.run(run_pie(args.uri, episodes, args.concurrency, program))
    wall = time.monotonic() - t
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    problems = 0
    for r in results:
        bad = check(r)
        problems += len(bad)
        seed = r.get("input", {}).get("seed")
        for b in bad[:5]:
            print(f"[seed {seed}] {b}", file=sys.stderr)
    s = summarize(results, wall)
    print(f"{s['episodes']} episodes ({s['errors']} errors, {s['aborted']} aborted, {s['timeouts']} timeouts, "
          f"{problems} check failures) "
          f"in {wall:.1f}s at concurrency {args.concurrency}")
    print(f"success rate {s['success_rate']:.3f}  mean reward {s['mean_reward']:.3f}")
    print(f"team episodes/hour {s['episodes_per_hour']:.1f}  generated tokens/s {s['gen_tokens_per_s']:.1f} "
          f"({s['gen_tokens']} tokens)")
    print(f"mean wall/episode {s['mean_wall_s']:.2f}s (client {s['mean_client_wall_s']:.2f}s)  "
          f"mean forward passes {s['mean_forward_passes']:.0f}  mean lanes/decode pass {s['mean_lanes']:.2f}")


if __name__ == "__main__":
    main()
