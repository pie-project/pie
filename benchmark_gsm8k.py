import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from pie_client import PieClient

SERVER_URL = "ws://127.0.0.1:8080"
GSM8K_PATH = "/home/dhruv/pie/gsm8k_test.jsonl"
RESULTS_PATH = "/home/dhruv/pie/benchmark_results.json"
CONFIG_PATH = "/home/dhruv/.pie/config.toml"
SERVE_LOG = "/home/dhruv/pie/serve.log"
PIE_BIN = "/home/dhruv/pie/target/release/pie"
SC_WASM = "/home/dhruv/pie/build-out/self_consistency.wasm"
SC_MANIFEST = "/home/dhruv/pie/inferlets/self-consistency/Pie.toml"
GOT_WASM = "/home/dhruv/pie/build-out/graph_of_thought.wasm"
GOT_MANIFEST = "/home/dhruv/pie/inferlets/graph-of-thought/Pie.toml"

GOT_NUM_PROPOSALS = 8
GOT_PROPOSAL_TOKENS = 256
GOT_AGGREGATION_TOKENS = 256
GOT_TOTAL_BUDGET = GOT_NUM_PROPOSALS * GOT_PROPOSAL_TOKENS + 6 * GOT_AGGREGATION_TOKENS  # 3584

SC_NUM_PROPOSALS = 8
SC_PROPOSAL_TOKENS = GOT_TOTAL_BUDGET // SC_NUM_PROPOSALS  # 448

# Baseline: single-shot generation at the FULL matched token budget.
# Uses the self-consistency inferlet with exactly one proposal, so it shares
# the same prompt template/extraction logic as SC and GoT.
BASELINE_TOKENS = GOT_TOTAL_BUDGET  # 3584

MAX_RETRIES_PER_QUESTION = 2  # if server crashes mid-question, retry this many times after restart


def load_gsm8k(path, n, seed=42):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    random.Random(seed).shuffle(lines)
    return lines[:n]


def extract_ground_truth(answer_field: str) -> float:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not m:
        raise ValueError(f"No #### marker in: {answer_field!r}")
    return float(m.group(1).replace(",", ""))


def parse_result(raw: str):
    """Parse the {"answer": "...", "tokens": N} JSON returned by inferlets.
    Falls back to legacy bare-string parsing for backward compatibility."""
    tokens = None
    answer_str = raw
    try:
        obj = json.loads(raw)
        answer_str = obj.get("answer", "NO_CONSENSUS")
        tokens = obj.get("tokens")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    if answer_str == "NO_CONSENSUS" or answer_str == "SERVER_ERROR":
        pred = None
    else:
        try:
            pred = float(answer_str)
        except (ValueError, TypeError):
            pred = None
    return pred, tokens


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(rows):
    with open(RESULTS_PATH, "w") as f:
        json.dump(rows, f, indent=2)


def start_server():
    """Start pie serve in the background, wait for it to be ready."""
    log_f = open(SERVE_LOG, "a")
    env = os.environ.copy()
    env["GGML_CUDA_GRAPHS"] = "0"
    proc = subprocess.Popen(
        [PIE_BIN, "serve", "--config", CONFIG_PATH, "--no-auth"],
        stdout=log_f, stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def is_server_alive():
    result = subprocess.run(["pgrep", "-f", "pie serve"], capture_output=True)
    return result.returncode == 0


def kill_server():
    subprocess.run(["pkill", "-9", "-f", "pie serve"], capture_output=True)
    time.sleep(2)


async def wait_for_server(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with PieClient(SERVER_URL) as client:
                await client.authenticate("local-dev")
                await client.ping()
                return True
        except Exception:
            await asyncio.sleep(2)
    return False


async def install_inferlets():
    async with PieClient(SERVER_URL) as client:
        await client.authenticate("local-dev")
        await client.install_program(SC_WASM, SC_MANIFEST, force_overwrite=True)
        await client.install_program(GOT_WASM, GOT_MANIFEST, force_overwrite=True)


async def restart_server_and_wait():
    print("  [restarting server...]", flush=True)
    kill_server()
    start_server()
    ok = await wait_for_server()
    if not ok:
        raise RuntimeError("Server did not come back up after restart")
    await install_inferlets()
    print("  [server restarted and inferlets reinstalled]", flush=True)


async def run_one_question(client, q):
    baseline_input = {
        "question": q,
        "proposal_tokens": [BASELINE_TOKENS],
        "aggregation_tokens": 128,
    }
    baseline_result = await client.run_processes("self-consistency@0.1.0", [baseline_input])

    sc_input = {
        "question": q,
        "proposal_tokens": [SC_PROPOSAL_TOKENS] * SC_NUM_PROPOSALS,
        "aggregation_tokens": 128,
    }
    sc_result = await client.run_processes("self-consistency@0.1.0", [sc_input])

    got_input = {
        "question": q,
        "proposal_tokens": [GOT_PROPOSAL_TOKENS] * GOT_NUM_PROPOSALS,
        "aggregation_tokens": GOT_AGGREGATION_TOKENS,
    }
    got_result = await client.run_processes("graph-of-thought@0.1.0", [got_input])

    return baseline_result[0], sc_result[0], got_result[0]


async def run_benchmark(n_questions: int):
    problems = load_gsm8k(GSM8K_PATH, n_questions)
    ground_truths = [extract_ground_truth(p["answer"]) for p in problems]
    questions = [p["question"] for p in problems]

    rows = load_existing_results()
    already_done = len(rows)
    if already_done > 0:
        print(f"Resuming: {already_done}/{n_questions} already completed, continuing from question {already_done + 1}", flush=True)

    if not await wait_for_server(timeout=5):
        print("Server not responding, starting it...", flush=True)
        await restart_server_and_wait()

    i = already_done
    while i < n_questions:
        q = questions[i]
        gt = ground_truths[i]
        print(f"[{i+1}/{n_questions}] running...", flush=True)

        attempt = 0
        while True:
            try:
                async with PieClient(SERVER_URL) as client:
                    await client.authenticate("local-dev")
                    baseline_raw, sc_raw, got_raw = await run_one_question(client, q)
                break
            except Exception as e:
                attempt += 1
                print(f"  [error: {e}] (attempt {attempt}/{MAX_RETRIES_PER_QUESTION + 1})", flush=True)
                if attempt > MAX_RETRIES_PER_QUESTION:
                    print(f"  [giving up on question {i+1} after {attempt} attempts, recording as failed]", flush=True)
                    baseline_raw, sc_raw, got_raw = "SERVER_ERROR", "SERVER_ERROR", "SERVER_ERROR"
                    break
                await restart_server_and_wait()

        baseline_pred, baseline_tokens = parse_result(baseline_raw)
        sc_pred, sc_tokens = parse_result(sc_raw)
        got_pred, got_tokens = parse_result(got_raw)
        baseline_ok = baseline_pred is not None and abs(baseline_pred - gt) < 1e-6
        sc_ok = sc_pred is not None and abs(sc_pred - gt) < 1e-6
        got_ok = got_pred is not None and abs(got_pred - gt) < 1e-6

        rows.append({
            "question": q, "ground_truth": gt,
            "baseline_pred": baseline_pred, "baseline_correct": baseline_ok, "baseline_tokens": baseline_tokens,
            "sc_pred": sc_pred, "sc_correct": sc_ok, "sc_tokens": sc_tokens,
            "got_pred": got_pred, "got_correct": got_ok, "got_tokens": got_tokens,
        })
        save_results(rows)  # incremental write — never lose progress again
        i += 1

    def summarize(key, label):
        preds = [r for r in rows if r[f"{key}_pred"] is not None or r[f"{key}_tokens"] is not None or True]
        correct = sum(r[f"{key}_correct"] for r in rows)
        no_consensus = sum(r[f"{key}_pred"] is None for r in rows)
        tok_vals = [r[f"{key}_tokens"] for r in rows if r[f"{key}_tokens"] is not None]
        avg_tokens = sum(tok_vals) / len(tok_vals) if tok_vals else float("nan")
        print(f"{label}: {correct}/{n_questions} correct ({100*correct/n_questions:.1f}%), "
              f"{no_consensus} NO_CONSENSUS, avg {avg_tokens:.0f} tokens/question")

    print("\n=== Results ===")
    print(f"N = {n_questions}")
    summarize("baseline", "Baseline (single-shot)")
    summarize("sc", "Self-consistency")
    summarize("got", "Graph-of-Thought")
    print(f"\nFull per-question results in {RESULTS_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(run_benchmark(n))
