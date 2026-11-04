"""LatentMAS (Zou et al., ICML 2026) on pie, against vLLM: a question's team of four agents (planner,
critic, refiner, judger) as the paper runs it, with the paper's prompts and answer checks.

  python harness.py pie  --model Qwen/Qwen3-4B --method latent --task gsm8k --out pie_latent.jsonl
  python harness.py vllm --model Qwen/Qwen3-4B --method text   --task gsm8k --out vllm_text.jsonl

Prompts, datasets and scoring are the authors' (`--ref`, a checkout of github.com/Gen-Verse/LatentMAS:
prompts.py, data.py, utils.py), so only the engine differs. Methods:

- single: one agent answers.
- text: TextMAS, the paper's text baseline: each agent writes, and every later agent reads all the notes.
- latent: LatentMAS: planner, critic and refiner each read their prompt into one shared context and take
  `--steps` latent steps there; the judger answers from that context. pie only: an engine that takes
  text and returns text has no latent step. With `--exit`, an agent stops its latent steps early once a
  step's likeliest token ends a turn or a thought (LaTER's exit rule).
- anchor: a text anchor: the planner writes a short plan without thinking (`--plan-tokens`), and the judger
  answers from the plan, as TextMAS's judger does.

`--think-budget` caps the answering agent's thinking (single agent, judger): if the thought has not closed
after that many tokens, the close phrase (`--close`, Qwen's own) is read and the agent answers in what is
left of its token budget (budget forcing, Muennighoff et al. 2025).

On pie a question's team is one process (main.py). On vLLM an agent's turn is one AsyncLLM request,
with prefix caching. Either way `--concurrent` teams run at once, each going on as its agents finish.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
CODE = ("humanevalplus", "mbppplus")


def load(args):
    sys.path.insert(0, str(Path(args.ref).resolve()))
    import data

    loader = {"gsm8k": data.load_gsm8k, "arc_challenge": data.load_arc_challenge, "arc_easy": data.load_arc_easy,
              "humanevalplus": data.load_humanevalplus, "mbppplus": data.load_mbppplus}[args.task]
    items = list(loader(split="test"))
    return items[: args.n] if args.n else items


def prompts(args):
    """The authors' message builders for this task and method."""
    from methods import default_agents
    from prompts import (build_agent_message_sequential_latent_mas, build_agent_messages_sequential_text_mas,
                         build_agent_messages_single_agent)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ns = SimpleNamespace(task=args.task, think=False, prompt="sequential", model_name=args.model,
                         text_mas_context_length=-1, method=None)
    agents = default_agents()
    return SimpleNamespace(
        agents=agents, tok=tok,
        nothink=tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False),
        close=tok.encode(args.close, add_special_tokens=False),
        think_end=tok.convert_tokens_to_ids("</think>"),
        stops={tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")},
        single=lambda q: build_agent_messages_single_agent(question=q, args=SimpleNamespace(**{**vars(ns), "method": "baseline"})),
        text=lambda role, q, ctx: build_agent_messages_sequential_text_mas(role=role, question=q, context=ctx, method="text_mas", args=ns),
        latent=lambda role, q: build_agent_message_sequential_latent_mas(role=role, question=q, context="", method="latent_mas", args=ns),
    )


def score(args, item, text):
    """As the authors' methods do: the last python block run against the tests (10 s), else the last
    \\boxed{} (or the last number), lower-cased."""
    from utils import extract_gsm8k_answer, extract_markdown_python_block, normalize_answer, run_with_timeout

    if args.task in CODE:
        code = extract_markdown_python_block(text)
        return None, bool(code) and run_with_timeout(code + "\n" + item["gold"], timeout=10)[0]
    pred = normalize_answer(extract_gsm8k_answer(text))
    return pred, bool(pred and item["gold"] and pred == item["gold"])


def embedding_scale(model: str) -> float:
    """The mean norm of the model's input embeddings: the size a latent input is scaled to."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(model) if Path(model).exists() else Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            if "model.embed_tokens.weight" in st.keys():
                return st.get_tensor("model.embed_tokens.weight").float().norm(dim=1).mean().item()
    raise SystemExit(f"no model.embed_tokens.weight under {path}")


# ── a team's turns, on either engine ──────────────────────────────────────────────


async def answer(args, p, ask, request):
    """The answering agent's turn, its thinking capped at `--think-budget` tokens: if the thought is still open
    there, the close phrase is read after it and the agent answers in the rest of its budget."""
    budget = args.think_budget
    if not budget or budget >= request["max_tokens"]:
        return (await ask(request))["text"]
    first = await ask({**request, "max_tokens": budget})
    made = first["tokens"]
    if len(made) < budget or made[-1] in p.stops:
        return first["text"]
    closed = p.think_end in made
    tail = [made[-1]] + ([] if closed else p.close)
    rest = await ask({**request, "messages": None, "tokens": tail, "after": None, "fresh": False,
                      "seed": request["seed"] + 7919, "max_tokens": max(request["max_tokens"] - len(made) - len(tail) + 1, 1)})
    return first["text"] + ("" if closed else args.close) + rest["text"]


async def solve(args, p, item, k, ask):
    """One question: the method's agents in turn; `ask(request)` runs one turn and returns its reply."""
    q = item["question"]
    sampling = {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    if args.method == "single":
        return await answer(args, p, ask, {"op": "say", "messages": p.single(q), "fresh": True, "seed": k, **sampling})
    if args.method == "anchor":
        plan = (await ask({"op": "say", "messages": p.text("planner", q, ""), "after": p.nothink, "fresh": True,
                           "seed": k * 8, **{**sampling, "max_tokens": args.plan_tokens}}))["text"].strip()
        return await answer(args, p, ask, {"op": "say", "messages": p.text("judger", q, f"[Planner]:\n{plan}\n\n"),
                                           "fresh": True, "seed": k * 8 + 3, **sampling})
    if args.method == "text":
        context = ""
        for i, agent in enumerate(p.agents):
            out = (await ask({"op": "say", "messages": p.text(agent.role, q, context), "fresh": True,
                              "seed": k * 8 + i, **sampling}))["text"].strip()
            if agent.role == "judger":
                return out
            context += f"[{agent.name}]:\n{out}\n\n"
    for agent in p.agents:
        messages = p.latent(agent.role, q)
        if agent.role != "judger":
            await ask({"op": "think", "messages": messages, "steps": args.steps, "scale": args.scale,
                       "exits": sorted(p.stops | {p.think_end}) if args.exit else None})
        else:
            # as the authors' code: with no latent steps the judger answers on a fresh context
            return await answer(args, p, ask, {"op": "say", "messages": messages, "fresh": args.steps == 0, "seed": k,
                                               **sampling})


async def run_pie(args, p, items, tally):
    from pie_client import Event, PieClient

    client = PieClient(args.uri)
    await client.connect()
    await client.authenticate("latentmas")
    program = await client.install_program_bytes((HERE / "main.py").read_bytes(), "latentmas.py", "0.1.0", force_overwrite=True)
    gate = asyncio.Semaphore(args.concurrent)

    async def team(k, item):
        async with gate:
            process = None

            async def ask(request):
                nonlocal process
                if process is None:
                    process = await client.launch_process(program, request)
                else:
                    await process.signal(json.dumps(request))
                event, value = await process.recv()
                if event != Event.Message:
                    raise RuntimeError(f"the team's process ended: {str(value)[-500:]}")
                reply = json.loads(value)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                tally["made"] += len(reply.get("tokens", ()))
                tally["read"] += reply.get("read", 0)
                tally["steps"] += reply.get("taken", 0)
                return reply

            try:
                return await solve(args, p, item, k, ask)
            finally:
                if process is not None:
                    await process.signal("quit")

    try:
        t0 = time.monotonic()
        texts = await asyncio.gather(*[team(k, item) for k, item in enumerate(items)], return_exceptions=True)
        return texts, time.monotonic() - t0
    finally:
        await client.close()


async def run_vllm(args, p, items, tally):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if args.method == "latent":
        raise SystemExit("vLLM takes text and returns text: it has no latent step")
    tok = p.tok
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model, max_model_len=args.max_model_len,
                                                       gpu_memory_utilization=0.9, enable_prefix_caching=True,
                                                       disable_log_stats=True))
    gate = asyncio.Semaphore(args.concurrent)
    ids = iter(range(1 << 62))

    async def team(k, item):
        async with gate:
            # the team's context as pie's working set holds it: every prompt read, and every token written but
            # the last (which pie samples and does not read); a fresh turn starts it over
            context = []

            async def ask(request):
                nonlocal context
                if request.get("messages"):
                    prompt = tok.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
                    new = tok.encode(prompt, add_special_tokens=False)
                else:
                    new = [int(t) for t in request["tokens"]]
                new += list(request.get("after") or ())
                ids_ = (new if request.get("fresh", True) else context + new)
                sp = SamplingParams(temperature=request["temperature"], top_p=request["top_p"], max_tokens=request["max_tokens"],
                                    seed=request["seed"], output_kind=RequestOutputKind.FINAL_ONLY,
                                    **({"top_k": request["top_k"]} if request["top_k"] else {}))
                out = None
                async for out in engine.generate({"prompt_token_ids": ids_}, sp, f"r{next(ids)}"):
                    pass
                o = out.outputs[0]
                context = ids_ + list(o.token_ids)[:-1]
                tally["made"] += len(o.token_ids)
                tally["read"] += len(new)
                return {"text": o.text, "tokens": list(o.token_ids)}

            return await solve(args, p, item, k, ask)

    try:
        t0 = time.monotonic()
        texts = await asyncio.gather(*[team(k, item) for k, item in enumerate(items)], return_exceptions=True)
        return texts, time.monotonic() - t0
    finally:
        engine.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["pie", "vllm"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--method", choices=["single", "text", "latent", "anchor"], default="latent")
    ap.add_argument("--task", choices=["gsm8k", "arc_challenge", "arc_easy", "humanevalplus", "mbppplus"], default="gsm8k")
    ap.add_argument("--n", type=int, default=0, help="questions, from the first (0: all)")
    ap.add_argument("--steps", type=int, default=40, help="latent steps per agent (the paper: accuracy peaks at 40-80)")
    ap.add_argument("--scale", type=float, default=0.0, help="latent input norm (0: the model's mean embedding norm)")
    ap.add_argument("--exit", action="store_true", help="latent: stop an agent's steps once one reads as finished")
    ap.add_argument("--think-budget", type=int, default=0, help="thinking tokens of the answering agent (0: no cap)")
    ap.add_argument("--close", default="\n\nConsidering the limited time by the user, I have to give the solution based on the "
                    "thinking directly now.\n</think>\n\n", help="read when the thinking budget runs out")
    ap.add_argument("--plan-tokens", type=int, default=512, help="anchor: the plan's tokens")
    ap.add_argument("--max-tokens", type=int, default=0, help="per agent (0: the paper's, 2048, or 4096 for code)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0: off; Hugging Face's generate takes the model's own (Qwen3: 20)")
    ap.add_argument("--concurrent", type=int, default=64, help="teams at once")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--ref", default="LatentMAS", help="a checkout of github.com/Gen-Verse/LatentMAS")
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.max_tokens = args.max_tokens or (4096 if args.task in CODE else 2048)

    items = load(args)
    p = prompts(args)
    if args.method == "latent" and not args.scale:
        args.scale = embedding_scale(args.model)
    tally = {"made": 0, "read": 0, "steps": 0}
    # timed from the first team to the last answer: neither engine's start-up counts
    texts, wall = asyncio.run((run_pie if args.engine == "pie" else run_vllm)(args, p, items, tally))

    correct = errors = 0
    with open(args.out, "w") as f:
        for k, (item, text) in enumerate(zip(items, texts)):
            if isinstance(text, BaseException):
                errors += 1
                f.write(json.dumps({"k": k, "error": repr(text)[-800:]}) + "\n")
                continue
            pred, ok = score(args, item, text)
            correct += ok
            f.write(json.dumps({"k": k, "pred": pred, "correct": ok, "text": text}) + "\n")
    summary = {"engine": args.engine, "model": args.model, "method": args.method, "task": args.task, "n": len(items),
               "steps": args.steps if args.method == "latent" else None, "top_k": args.top_k,
               "exit": args.exit, "think_budget": args.think_budget, "latent_steps_taken": tally["steps"],
               "accuracy": round(correct / max(len(items), 1), 4),
               "wall_s": round(wall, 1), "tokens_written": tally["made"], "tokens_read": tally["read"],
               "concurrent": args.concurrent, "errors": errors}
    print(json.dumps(summary), flush=True)
    Path(args.out + ".summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
