"""AIME 2025 by parallel reasoning: DeepConf as a pie program, or the same algorithm on vLLM.

  python harness.py pie  --model DIR --mode low --budget 64 --out pie_low.jsonl
  python harness.py vllm --model DIR --mode low --budget 64 --out vllm_low.jsonl

`--mode cons` is plain majority voting over `budget` traces; `low` and `high`
are DeepConf-low (keep the top 10%) and DeepConf-high (the top 90%).

Both engines read the same prompt tokens, rendered here from the model's own
chat template, and sample with the paper's settings for Qwen3: temperature
0.6, top-p 0.95, top-k 20, up to 32,768 tokens. On pie the algorithm runs in
the engine (main.py): the GPU computes each token's confidence, and the
program stops a trace on the spot. On vLLM it runs here, beside AsyncLLM in
the same process: every token comes back with its top-20 log-probabilities,
this file keeps the window mean, and a trace below the threshold is aborted.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).parent
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MODES = {"cons": None, "low": 10, "high": 90}


def load_problems(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("MathArena/aime_2025", split="train")
    rows = [{"id": str(r["problem_idx"]), "problem": r["problem"], "answer": str(r["answer"])} for r in ds]
    return rows[:limit] if limit else rows


def norm(answer: str | None) -> str | None:
    """An AIME answer as an integer string where it is one."""
    if answer is None:
        return None
    a = answer.strip().strip("$").replace("\\!", "").replace(",", "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\^\{?\\circ\}?", "", a).strip()
    m = re.fullmatch(r"0*(\d+)(?:\.0*)?", a)
    return str(int(m.group(1))) if m else a


# ── the algorithm, as main.py runs it in pie, for the vLLM arm ─────────────────


def boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return "".join(out).strip()
        out.append(c)
        i += 1
    return None


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class Trace:
    def __init__(self, tid: int) -> None:
        self.id, self.tokens = tid, []
        self.window, self.head, self.window_sum, self.total = [], 0, 0.0, 0.0
        self.lowest, self.state, self.answer = None, "running", None

    def score(self) -> float:
        if self.lowest is not None:
            return self.lowest
        return self.total / len(self.tokens) if self.tokens else 0.0

    def add(self, conf: float, window: int) -> float | None:
        """Add a token's confidence; the window mean once the window is full."""
        self.total += conf
        if len(self.window) < window:
            self.window.append(conf)
            self.window_sum += conf
        else:
            self.window_sum += conf - self.window[self.head]
            self.window[self.head] = conf
            self.head = (self.head + 1) % window
        if len(self.window) < window:
            return None
        mean = self.window_sum / window
        self.lowest = mean if self.lowest is None else min(self.lowest, mean)
        return mean


class VllmDeepConf:
    """main.py's DeepConf, one question, against an AsyncLLM engine."""

    def __init__(self, engine, tokenizer, cfg: dict, pid: str, prompt: list[int]) -> None:
        self.engine, self.tokenizer, self.cfg, self.pid, self.prompt = engine, tokenizer, cfg, pid, prompt
        self.threshold, self.traces, self.done, self.decoded = None, [], False, 0

    async def run_trace(self, t: Trace, stop_early: bool) -> None:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        cfg = self.cfg
        online = cfg["eta"] is not None
        sp = SamplingParams(
            n=1, temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
            max_tokens=cfg["max_tokens"], seed=cfg["seed"] + 7919 * t.id, detokenize=False,
            logprobs=cfg["k_conf"] if online else None,
            output_kind=RequestOutputKind.DELTA if online else RequestOutputKind.FINAL_ONLY,
        )
        rid = f"{self.pid}-{t.id}"
        tokens: list[int] = []
        async for out in self.engine.generate({"prompt_token_ids": self.prompt}, sp, rid):
            o = out.outputs[0]
            new = list(o.token_ids)
            self.decoded += len(new)
            if not online:
                tokens += new
                continue
            for tok, lp in zip(new, o.logprobs or []):
                tokens.append(tok)
                top = sorted((x.logprob for x in lp.values() if x.rank is not None and x.rank <= cfg["k_conf"]), reverse=True)
                conf = -sum(top) / len(top) if top else 0.0
                mean = t.add(conf, cfg["window"])
                if self.done:
                    t.state = "cancelled"
                elif stop_early and mean is not None and mean < self.threshold:
                    t.state = "stopped"
                if t.state != "running":
                    await self.engine.abort(rid)
                    t.tokens = tokens
                    return
        t.tokens = [x for x in tokens if x not in self.tokenizer.all_special_ids]
        t.state = "finished"
        t.answer = boxed(self.tokenizer.decode(t.tokens))

    def votes(self, filtered: bool) -> dict:
        done = [t for t in self.traces if t.state == "finished" and t.answer is not None]
        eta = self.cfg["eta"]
        out: dict = {}
        if eta is None:
            for t in done:
                out[t.answer] = out.get(t.answer, 0) + 1
            return out
        if filtered and done:
            done = sorted(done, key=lambda t: t.score(), reverse=True)[: max(1, round(len(done) * eta / 100))]
        for t in done:
            out[t.answer] = out.get(t.answer, 0.0) + t.score()
        return out

    def consensus(self) -> float:
        v = self.votes(filtered=False)
        total = sum(v.values())
        return max(v.values()) / total if total else 0.0

    async def run(self) -> dict:
        cfg = self.cfg
        t0 = time.monotonic()
        budget, online = cfg["budget"], cfg["eta"] is not None
        started = 0

        def new() -> Trace:
            nonlocal started
            t = Trace(started)
            started += 1
            self.traces.append(t)
            return t

        if online:
            warm = [new() for _ in range(min(cfg["n_init"], budget))]
            await asyncio.gather(*[self.run_trace(t, False) for t in warm])
            scores = [t.score() for t in warm if t.tokens]
            self.threshold = percentile(scores, 100 - cfg["eta"]) if scores else 0.0
            if self.consensus() >= cfg["tau"]:
                self.done = True

        async def worker() -> None:
            while not self.done and started < budget:
                await self.run_trace(new(), online)
                if online and self.consensus() >= cfg["tau"]:
                    self.done = True

        await asyncio.gather(*[worker() for _ in range(cfg["parallel"])])
        v = self.votes(filtered=True)
        count = lambda s: sum(t.state == s for t in self.traces)
        return {
            "answer": max(v, key=v.get) if v else None, "votes": v, "threshold": self.threshold,
            "traces": len(self.traces), "finished": count("finished"), "stopped": count("stopped"),
            "cancelled": count("cancelled"), "decoded_tokens": self.decoded,
            "trace_tokens": [len(t.tokens) for t in self.traces], "seconds": round(time.monotonic() - t0, 2),
        }


# ── the engines ────────────────────────────────────────────────────────────────


async def run_pie(rows: list[dict], cfg: dict, args) -> list[dict]:
    from pie_client import Event, PieClient

    program: dict = {}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrent)

    async def one(row: dict) -> dict:
        async with sem, PieClient(args.uri) as client:
            await client.authenticate("deepconf")
            async with lock:
                if "id" not in program:
                    program["id"] = await client.install_program_bytes(
                        (HERE / "main.py").read_bytes(), "deepconf.py", "0.1.0", force_overwrite=True)
            proc = await client.launch_process(
                program["id"], {"prompt_tokens": row["ids"], "stop_tokens": row["stops"], "config": cfg})

            async def result() -> dict:
                while True:
                    event, value = await proc.recv()
                    if event is Event.Return:
                        return json.loads(value)
                    if event is Event.Error:
                        return {"error": str(value)[-500:]}

            try:
                return await asyncio.wait_for(result(), args.timeout)
            except asyncio.TimeoutError:
                return {"error": f"no answer after {args.timeout} s"}

    async def logged(row: dict) -> dict:
        res = await one(row)
        print(json.dumps({"done": row["id"], "answer": res.get("answer"), "gold": row["answer"],
                          "tokens": res.get("decoded_tokens"), "error": bool(res.get("error"))}), flush=True)
        return res

    return await asyncio.gather(*[logged(r) for r in rows])


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
                return await asyncio.wait_for(VllmDeepConf(engine, tokenizer, cfg, row["id"], row["ids"]).run(), args.timeout)
            except Exception as e:  # noqa: BLE001
                import traceback
                cause = e.__cause__ or e
                return {"error": "".join(traceback.format_exception(cause))[-1500:]}

    async def logged(row: dict) -> dict:
        res = await one(row)
        print(json.dumps({"done": row["id"], "answer": res.get("answer"), "gold": row["answer"],
                          "tokens": res.get("decoded_tokens"), "error": bool(res.get("error"))}), flush=True)
        return res

    try:
        return await asyncio.gather(*[logged(r) for r in rows])
    finally:
        engine.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["pie", "vllm"])
    ap.add_argument("--model", required=True, help="the model's directory: its tokenizer and template")
    ap.add_argument("--mode", choices=list(MODES), default="low")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=16, help="traces in flight per question after the warmup")
    ap.add_argument("--n-init", type=int, default=16, help="warmup traces that set the threshold")
    ap.add_argument("--concurrent", type=int, default=30, help="questions at once")
    ap.add_argument("--problems", type=int, default=0, help="the first N questions; 0 is all 30")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th question from the i-th, one engine per GPU")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=34816)
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--kv-fp8", action="store_true", help="vLLM: FP8 KV, as pie's e4m3 deployment")
    ap.add_argument("--timeout", type=float, default=2400, help="seconds a question may take")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    i, n = map(int, args.shard.split("/"))
    rows = load_problems(args.problems)[i::n]
    for r in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["problem"] + SUFFIX}], add_generation_prompt=True, tokenize=False,
            enable_thinking=True)
        r["ids"] = tokenizer.encode(text, add_special_tokens=False)
        r["stops"] = stops
    eta = MODES[args.mode]
    cfg = {"budget": args.budget, "eta": eta, "max_tokens": args.max_tokens,
           "parallel": args.parallel if eta is not None else args.budget,
           "n_init": args.n_init, "tau": 0.95, "window": 2048, "k_conf": 20,
           "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1}

    t0 = time.monotonic()
    if args.engine == "pie":
        results = asyncio.run(run_pie(rows, cfg, args))
    else:
        results = asyncio.run(run_vllm(rows, cfg, args, tokenizer))
    wall = time.monotonic() - t0

    correct = tokens = 0
    with open(args.out, "w") as f:
        for r, res in zip(rows, results):
            ok = "error" not in res and norm(res.get("answer")) == norm(r["answer"])
            correct += ok
            tokens += res.get("decoded_tokens") or 0
            f.write(json.dumps({"id": r["id"], "gold": r["answer"], "correct": ok, "engine": args.engine,
                                "mode": args.mode, "budget": args.budget, **res}) + "\n")
    summary = {"engine": args.engine, "mode": args.mode, "budget": args.budget, "problems": len(rows),
               "correct": correct, "decoded_tokens": tokens, "wall_s": round(wall, 1),
               "errors": sum("error" in r for r in results)}
    print(json.dumps(summary), flush=True)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
