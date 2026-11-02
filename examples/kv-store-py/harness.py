"""Questions about a codebase read once: pie's KV store against vLLM.

  python harness.py pie  --arm store --tokens 131072 --out pie_store.json
  python harness.py vllm --arm rag   --tokens 131072 --out vllm_rag.json

The corpus is real Python source, one file a chunk, with a `RETRY_LIMIT = <n>`
line planted in `--questions` files; each question asks a file's value. BM25
over the chunks picks `--k` per question, the same picks on every arm.

- pie store: the corpus read once; each question's fork forgets the chunks it
  did not pick.
- pie full: the same forks reading the whole store (a control).
- vLLM full: the whole corpus before every question, with prefix caching.
- vLLM rag: the picked chunks before each question, read for that question.

Every arm reads the same token ids, built here from the model's tokenizer.
"""

import argparse
import asyncio
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SYSTEM = "You read a codebase and answer questions about it."
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def corpus(root: Path, tokenizer, budget: int, questions: int, max_file: int, seed: int):
    """Files under `root` as chunks until `budget` tokens; `questions` of them get a planted value."""
    rng = random.Random(seed)
    files = sorted(p for p in root.rglob("*.py") if p.stat().st_size > 2000)
    rng.shuffle(files)
    chunks, total = [], 0
    for p in files:
        text = p.read_text(errors="replace")
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_file]
        body = tokenizer.decode(ids)
        rel = str(p.relative_to(root))
        chunks.append({"path": rel, "body": body})
        total += len(ids) + 16
        if total >= budget:
            break
    planted = rng.sample(range(len(chunks)), min(questions, len(chunks)))
    for i in planted:
        lines = chunks[i]["body"].split("\n")
        value = rng.randrange(1000, 9999)
        lines.insert(rng.randrange(1, max(2, len(lines) - 1)), f"RETRY_LIMIT = {value}")
        chunks[i]["body"], chunks[i]["value"] = "\n".join(lines), str(value)
    for c in chunks:
        c["text"] = f"\n### file: {c['path']}\n{c['body']}\n"
        c["ids"] = tokenizer.encode(c["text"], add_special_tokens=False)
    asks = [{"chunk": i, "path": chunks[i]["path"], "answer": chunks[i]["value"]} for i in planted]
    return chunks, asks


def bm25(chunks: list[dict], query: str, k: int, k1: float = 1.2, b: float = 0.75) -> list[int]:
    docs = [Counter(w.lower() for w in WORD.findall(c["text"])) for c in chunks]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / len(lengths)
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    terms = [w.lower() for w in WORD.findall(query)]
    scores = []
    for d, length in zip(docs, lengths):
        s = 0.0
        for w in terms:
            if w in d:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * d[w] * (k1 + 1) / (d[w] + k1 * (1 - b + b * length / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: -scores[i])[:k]


def build(args, tokenizer):
    chunks, asks = corpus(Path(args.source), tokenizer, args.tokens, args.questions, args.max_file, args.seed)
    head = tokenizer.encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nHere is the codebase.\n",
                            add_special_tokens=False)
    for a in asks:
        q = f"\n\nQuestion: In {a['path']}, what is RETRY_LIMIT set to? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n"
        if args.no_think:  # a hybrid Qwen3 answers at once behind an empty thinking block
            q += "<think>\n\n</think>\n\n"
        a["tokens"] = tokenizer.encode(q, add_special_tokens=False)
        a["keep"] = sorted(bm25(chunks, f"{a['path']} RETRY_LIMIT", args.k))
        a["hit"] = a["chunk"] in a["keep"]
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return head, chunks, asks, stops


def score(asks: list[dict], answers: list[str]) -> int:
    return sum(a["answer"] in (text or "") for a, text in zip(asks, answers))


async def run_pie(args, head, chunks, asks, stops) -> dict:
    from pie_client import Event, PieClient

    async with PieClient(args.uri) as client:
        await client.authenticate("kv-store")
        program = await client.install_program_bytes((HERE / "main.py").read_bytes(), "kv-store.py", "0.1.0",
                                                     force_overwrite=True)
        t0 = time.monotonic()
        proc = await client.launch_process(program, {
            "prefix": head, "chunks": [c["ids"] for c in chunks], "stop_tokens": stops,
            "questions": [{"tokens": a["tokens"], "keep": a["keep"]} for a in asks],
            "select": args.arm == "store", "decode": args.decode})
        while True:
            event, value = await proc.recv()
            if event is Event.Return:
                out = json.loads(value)
                out["wall_s"] = round(time.monotonic() - t0, 3)
                return out
            if event is Event.Error:
                return {"error": str(value)[-800:]}


def run_vllm(args, head, chunks, asks, stops) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0, max_tokens=args.decode, stop_token_ids=stops)
    prompts = []
    for a in asks:
        picked = range(len(chunks)) if args.arm == "full" else a["keep"]
        ids = head + [t for i in picked for t in chunks[i]["ids"]] + a["tokens"]
        prompts.append({"prompt_token_ids": ids})
    t0 = time.monotonic()
    if args.arm == "full":
        # read the corpus once into the prefix cache, as pie reads its store, so the questions
        # that follow hit it instead of each reading it at the same time
        store = head + [t for c in chunks for t in c["ids"]]
        llm.generate([{"prompt_token_ids": store}], SamplingParams(max_tokens=1), use_tqdm=False)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.monotonic() - t0
    answers = [o.outputs[0].text for o in outs]
    return {"answers": answers, "answer_s": round(wall, 3), "wall_s": round(wall, 3),
            "prompt_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),
            "store_tokens": len(head) + sum(len(c["ids"]) for c in chunks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["pie", "vllm"])
    ap.add_argument("--arm", choices=["store", "full", "rag"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="a directory of Python source to read as the codebase")
    ap.add_argument("--tokens", type=int, default=131072, help="corpus size")
    ap.add_argument("--questions", type=int, default=64)
    ap.add_argument("--k", type=int, default=4, help="chunks a question keeps")
    ap.add_argument("--max-file", type=int, default=1536, help="tokens a file keeps")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=139264)
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    head, chunks, asks, stops = build(args, tokenizer)
    if args.engine == "pie":
        result = asyncio.run(run_pie(args, head, chunks, asks, stops))
    else:
        result = run_vllm(args, head, chunks, asks, stops)
    answers = result.get("answers") or []
    summary = {"engine": args.engine, "arm": args.arm, "corpus_tokens": result.get("store_tokens"),
               "questions": len(asks), "k": args.k, "retrieval_hits": sum(a["hit"] for a in asks),
               "right": score(asks, answers), "answer_s": result.get("answer_s"), "wall_s": result.get("wall_s"),
               "read_s": result.get("read_s"), "error": result.get("error")}
    print(json.dumps(summary), flush=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "result": result,
                   "asks": [{k: a[k] for k in ("path", "answer", "keep", "hit")} for a in asks]}, f)


if __name__ == "__main__":
    main()
