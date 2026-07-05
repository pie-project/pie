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


def parse_prediction(raw: str):
    if raw == "NO_CONSENSUS":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


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

    return sc_result[0], got_result[0]


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
                    sc_raw, got_raw = await run_one_question(client, q)
                break
            except Exception as e:
                attempt += 1
                print(f"  [error: {e}] (attempt {attempt}/{MAX_RETRIES_PER_QUESTION + 1})", flush=True)
                if attempt > MAX_RETRIES_PER_QUESTION:
                    print(f"  [giving up on question {i+1} after {attempt} attempts, recording as failed]", flush=True)
                    sc_raw, got_raw = "SERVER_ERROR", "SERVER_ERROR"
                    break
                await restart_server_and_wait()

        sc_pred = parse_prediction(sc_raw)
        got_pred = parse_prediction(got_raw)
        sc_ok = sc_pred is not None and abs(sc_pred - gt) < 1e-6
        got_ok = got_pred is not None and abs(got_pred - gt) < 1e-6

        rows.append({
            "question": q, "ground_truth": gt,
            "sc_pred": sc_pred, "sc_correct": sc_ok,
            "got_pred": got_pred, "got_correct": got_ok,
        })
        save_results(rows)  # incremental write — never lose progress again
        i += 1

    sc_correct = sum(r["sc_correct"] for r in rows)
    got_correct = sum(r["got_correct"] for r in rows)
    sc_no_consensus = sum(r["sc_pred"] is None for r in rows)
    got_no_consensus = sum(r["got_pred"] is None for r in rows)

    print("\n=== Results ===")
    print(f"N = {n_questions}")
    print(f"Self-consistency: {sc_correct}/{n_questions} correct ({100*sc_correct/n_questions:.1f}%), {sc_no_consensus} NO_CONSENSUS")
    print(f"Graph-of-Thought: {got_correct}/{n_questions} correct ({100*got_correct/n_questions:.1f}%), {got_no_consensus} NO_CONSENSUS")
    print(f"\nFull per-question results in {RESULTS_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(run_benchmark(n))
