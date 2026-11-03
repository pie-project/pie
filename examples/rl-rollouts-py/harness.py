"""GRPO rollout groups for math prompts: flat sampling against tree rollouts, on pie or vLLM.

  python harness.py pie  --model DIR --mode tree --out pie_tree.jsonl
  python harness.py vllm --model DIR --mode flat --out vllm_flat.jsonl

Every arm samples `--group` answers per prompt at temperature 1 with the same
prompt tokens (rendered here from the model's template, thinking off) and the
same answer budget; rewards come from math-verify against the gold answer.

- flat: `group` answers from the prompt, as plain GRPO sampling does.
- tree: `--roots` answers from the prompt; every `--segment` tokens the least
  confident unfinished answers fork, up to `--per-round` a round, until the
  group is full (answers that end first leave room that fresh answers fill).

On pie the tree runs in the engine (main.py): a fork reads its parent's KV as
it stands. On vLLM it runs here on AsyncLLM: each segment is a request that
sends the answer so far as its prompt, served from the prefix cache when the
cache still holds it, with 20 log-probabilities a token for the confidence.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_problems(limit: int, levels: set[str], dataset: str = "math") -> list[dict]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"id": f"gsm8k-{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(ds)]
        return rows[:limit] if limit else rows
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
            for r in ds if str(r["level"]) in levels]
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def reward(text: str, gold: str) -> float:
    from math_verify import parse, verify

    try:
        return float(verify(parse(f"${gold}$"), parse(text)))
    except Exception:  # noqa: BLE001
        return 0.0


# ── vLLM: the same rollouts on AsyncLLM ────────────────────────────────────────


class VChain:
    ids = 0

    def __init__(self, tokens: list[int], parent=None) -> None:
        self.id = VChain.ids
        VChain.ids += 1
        self.tokens, self.done, self.conf = list(tokens), False, 0.0
        self.shared, self.parent = len(tokens), (parent.id if parent else None)


async def vllm_group(engine, cfg: dict, pid: str, prompt: list[int]) -> dict:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    t0 = time.monotonic()
    stats = {"decoded": 0, "prompt": 0, "cached": 0}

    async def write(c: VChain, budget: int, conf: bool) -> None:
        sp = SamplingParams(n=1, temperature=cfg["temperature"], max_tokens=budget, detokenize=False,
                            seed=cfg["seed"] + 7919 * c.id + len(c.tokens),
                            logprobs=cfg["k_conf"] if conf else None, output_kind=RequestOutputKind.FINAL_ONLY,
                            stop_token_ids=cfg["stop_tokens"])
        out = None
        async for out in engine.generate({"prompt_token_ids": prompt + c.tokens}, sp, f"{pid}-{c.id}-{len(c.tokens)}"):
            pass
        o = out.outputs[0]
        new = [t for t in o.token_ids if t not in cfg["stop_tokens"]]
        stats["decoded"] += len(o.token_ids)
        stats["prompt"] += len(prompt) + len(c.tokens)
        stats["cached"] += out.num_cached_tokens or 0
        c.tokens += new
        if conf and o.logprobs:
            confs = []
            for lp in o.logprobs:
                top = [x.logprob for x in lp.values() if x.rank is not None and x.rank <= cfg["k_conf"]]
                confs.append(-sum(top) / len(top) if top else 0.0)
            c.conf = sum(confs) / len(confs)
        c.done = o.finish_reason == "stop" or len(c.tokens) >= cfg["max_tokens"]

    group = cfg["group"]
    if cfg["mode"] == "flat":
        chains = [VChain([]) for _ in range(group)]
        await asyncio.gather(*[write(c, cfg["max_tokens"], False) for c in chains])
    else:
        chains = [VChain([]) for _ in range(cfg["roots"])]
        while True:
            live = [c for c in chains if not c.done]
            if not live:
                break
            await asyncio.gather(*[write(c, min(cfg["segment"], cfg["max_tokens"] - len(c.tokens)), True) for c in live])
            room = min(group - len(chains), cfg["per_round"])
            unsure = sorted((c for c in live if not c.done), key=lambda c: c.conf)[:room]
            chains += [VChain(c.tokens, parent=c) for c in unsure]
        if len(chains) < group:
            extra = [VChain([]) for _ in range(group - len(chains))]
            chains += extra
            await asyncio.gather(*[write(c, cfg["max_tokens"], False) for c in extra])
    return {"mode": cfg["mode"], "tokens": [c.tokens for c in chains], "lengths": [len(c.tokens) for c in chains],
            "shared": [c.shared for c in chains], "parents": [c.parent for c in chains],
            "decoded_tokens": stats["decoded"], "prompt_tokens": stats["prompt"], "cached_tokens": stats["cached"],
            "seconds": round(time.monotonic() - t0, 2)}


async def run_vllm(rows: list[dict], cfg: dict, args, tokenizer) -> list[dict]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
        enable_prefix_caching=True, disable_log_stats=True, kv_cache_dtype="fp8" if args.kv_fp8 else "auto"))
    sem = asyncio.Semaphore(args.concurrent)

    async def one(row: dict) -> dict:
        async with sem:
            try:
                res = await asyncio.wait_for(vllm_group(engine, cfg, row["id"].replace("/", "_"), row["ids"]), args.timeout)
                res["answers"] = [tokenizer.decode(t) for t in res.pop("tokens")]
                return res
            except Exception as e:  # noqa: BLE001
                import traceback
                return {"error": "".join(traceback.format_exception(e.__cause__ or e))[-1500:]}

    try:
        return await asyncio.gather(*[one(r) for r in rows])
    finally:
        engine.shutdown()


async def run_pie(rows: list[dict], cfg: dict, args) -> list[dict]:
    from pie_client import Event, PieClient

    program: dict = {}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrent)

    async def one(row: dict) -> dict:
        async with sem, PieClient(args.uri) as client:
            await client.authenticate("rl-rollouts")
            async with lock:
                if "id" not in program:
                    program["id"] = await client.install_program_bytes(
                        (HERE / "main.py").read_bytes(), "rl-rollouts.py", "0.1.0", force_overwrite=True)
            proc = await client.launch_process(
                program["id"], {"prompt_tokens": row["ids"], "stop_tokens": cfg["stop_tokens"], "config": cfg})

            async def result() -> dict:
                while True:
                    event, value = await proc.recv()
                    if event is Event.Return:
                        return json.loads(value)
                    if event is Event.Error:
                        return {"error": str(value)[-800:]}

            try:
                res = await asyncio.wait_for(result(), args.timeout)
            except asyncio.TimeoutError:
                res = {"error": f"no answer after {args.timeout} s"}
            print(json.dumps({"done": row["id"], "seconds": res.get("seconds"), "error": res.get("error")}), flush=True)
            return res

    return await asyncio.gather(*[one(r) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["pie", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["flat", "tree"], default="tree")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--roots", type=int, default=4)
    ap.add_argument("--segment", type=int, default=256)
    ap.add_argument("--per-round", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--problems", type=int, default=64)
    ap.add_argument("--levels", default="4,5", help="MATH levels to draw prompts from")
    ap.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    ap.add_argument("--concurrent", type=int, default=8, help="prompts in flight")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--kv-fp8", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    rows = load_problems(args.problems, set(args.levels.split(",")), args.dataset)
    for r in rows:
        text = tokenizer.apply_chat_template([{"role": "user", "content": r["problem"] + SUFFIX}],
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False)
        r["ids"] = tokenizer.encode(text, add_special_tokens=False)
    cfg = {"mode": args.mode, "group": args.group, "roots": args.roots, "segment": args.segment,
           "per_round": args.per_round, "k_conf": 20, "temperature": 1.0, "max_tokens": args.max_tokens,
           "seed": 1, "stop_tokens": stops}

    t0 = time.monotonic()
    results = asyncio.run(run_pie(rows, cfg, args) if args.engine == "pie" else run_vllm(rows, cfg, args, tokenizer))
    wall = time.monotonic() - t0

    leaves = correct = mixed = decoded = errors = 0
    with open(args.out, "w") as f:
        for r, res in zip(rows, results):
            if "error" in res:
                errors += 1
                f.write(json.dumps({"id": r["id"], **res}) + "\n")
                continue
            rewards = [reward(a, r["answer"]) for a in res["answers"]]
            leaves += len(rewards)
            correct += sum(rewards)
            mixed += 0 < sum(rewards) < len(rewards)
            decoded += res["decoded_tokens"]
            res.pop("answers")
            f.write(json.dumps({"id": r["id"], "gold": r["answer"], "rewards": rewards, **res}) + "\n")
    summary = {"engine": args.engine, "mode": args.mode, "prompts": len(rows), "group": args.group,
               "leaves": leaves, "accuracy": round(correct / max(leaves, 1), 4),
               "mixed_groups": mixed, "decoded_tokens": decoded, "wall_s": round(wall, 1), "errors": errors}
    print(json.dumps(summary), flush=True)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
