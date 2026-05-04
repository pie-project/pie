"""Simplified ES Training Script.

Takes advantage of the runtime's contention management — we simply
submit all requests upfront and let the runtime handle queuing,
suspension, and restoration internally.  No per-worker scheduling,
capacity tracking, or retry logic needed.

Supports two modes:
  - **Local** (``--local``): Spins up an in-process Pie server.
    Inferlets are fetched from the registry.
  - **Distributed** (default): Connects to remote server(s) via
    ``--servers`` URIs.  Inferlets are fetched from the registry.

All TrainingConfig fields can be passed as CLI flags (uses python-fire).

Usage::

    # Local mode (single server, in-process)
    python train.py --local --model meta-llama/Llama-3.2-1B-Instruct

    # Resume from checkpoint
    python train.py --local --resume_from evo-math-v2-step-255

    # Override ES hyperparameters
    python train.py --local --population_size 256 --tasks_per_seed 8
"""

import fire
import asyncio
import csv
import json
import re
import os
import time
import sys
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from blake3 import blake3
from tqdm.auto import tqdm

from pie_client import PieClient, Event

from countdown import CountdownDataset
from openr1math import OpenR1MathDataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for the ES training run."""

    # --- Mode ---
    local: bool = False
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda:0"
    # `dev` = pie_driver_dev (flashinfer); `sglang` = pie_driver_sglang.
    # When `sglang`, the driver is configured with `enable_adapter=True`
    # so the QKV adapter wrappers get installed at engine load.
    driver: str = "dev"
    gpu_mem_util: float = 0.8
    cpu_mem_budget: int = 12
    max_concurrent_processes: int = 256
    default_token_budget: int = 4096
    max_batch_size: int = 512

    # --- Server (distributed mode) ---
    servers: list[str] = field(default_factory=lambda: ["ws://127.0.0.1:8080"])

    # --- Inferlet names (from registry) ---
    inferlet_names: dict[str, str] = field(
        default_factory=lambda: {
            "es-init": "es-init@0.1.5",
            "es-rollout": "es-rollout@0.1.6",
            "es-update": "es-update@0.1.5",
        }
    )

    # When False, rollouts skip the adapter entirely (no ZO perturbation).
    # Used for A/B latency comparisons; training-meaningful runs leave it True.
    use_adapter: bool = True

    # --- Dataset ---
    dataset: str = "math"  # "countdown" or "math"
    data_path: str = "./Countdown-Tasks-3to4"

    # --- ES Hyperparameters ---
    adapter_name: str = ""
    training_steps: int = 10000
    population_size: int = 2048
    tasks_per_seed: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    initial_sigma: float = 0.005
    max_sigma: float = 0.014
    mu_fraction: float = 0.5
    max_tokens_gen: int = 1024
    system_prompt: str = (
        "You are a helpful AI Assistant that provides well-reasoned and "
        "detailed responses. You first think about the reasoning process as "
        "an internal monologue and then provide the user with the answer. "
        "Respond in the following format: <think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )

    # --- Evaluation ---
    test_size: int = 100
    eval_every: int = 2

    # --- Checkpointing ---
    checkpoint_every: int = 5
    resume_from: str = ""  # checkpoint name to resume from, e.g. "evo-math-v2-step-255"

    def __post_init__(self):
        if not self.adapter_name:
            self.adapter_name = f"evo-{self.dataset}-v2"


# =============================================================================
# Helpers
# =============================================================================


async def launch_and_collect(
    client: PieClient,
    inferlet: str,
    input_dict: dict,
) -> Optional[str]:
    """Launch an inferlet and block until it returns or errors."""
    process = await client.launch_process(inferlet, input=input_dict)
    result = None
    while True:
        event, message = await process.recv()
        if event in (Event.Return, Event.Message):
            result = message
            if event == Event.Return:
                break
        elif event == Event.Error:
            tqdm.write(f"⚠️  Process {process.process_id} failed: {message}")
            break
    return result


async def run_on_all_clients(
    clients: list[PieClient],
    inferlet: str,
    input_dict: dict,
) -> list[Optional[str]]:
    """Run the same inferlet call on every client in parallel."""
    return await asyncio.gather(
        *(launch_and_collect(c, inferlet, input_dict) for c in clients)
    )


# =============================================================================
# Rollout Distribution
# =============================================================================


async def run_rollouts(
    clients: list[PieClient],
    inferlet: str,
    seeds: np.ndarray,
    tasks: list[dict],
    config: TrainingConfig,
    desc: str = "rollout",
) -> dict:
    """Submit all rollout requests across clients, wait for all results.

    Work is partitioned evenly across the clients.  Each client
    receives all of its requests upfront — the runtime's contention
    management handles queuing and scheduling internally.
    """
    texts_out, seeds_out, tasks_out = [], [], []
    latencies = []
    output_lens = []
    n = len(seeds)
    pbar = tqdm(total=n, desc=desc, dynamic_ncols=True, leave=False, file=sys.stdout)

    async def _tracked(client, seed, task):
        """Build input, launch one process, parse result, update progress bar."""
        problem = task.get("problem", str(task))
        uid = blake3(f"{seed}:{problem}".encode()).hexdigest()
        inp = {
            "name": config.adapter_name,
            "rollouts": [{"uid": uid, "task": problem, "seed": int(seed)}],
            "max_num_outputs": config.max_tokens_gen,
            "system_prompt": config.system_prompt,
            "use_adapter": config.use_adapter,
        }
        t0 = time.time()
        res_json = await launch_and_collect(client, inferlet, inp)
        latencies.append(time.time() - t0)

        # Parse once — return texts directly
        texts = None
        if res_json:
            try:
                parsed = json.loads(res_json)
                if isinstance(parsed, list):
                    texts = parsed
                    for t in texts:
                        output_lens.append(len(t))
            except (json.JSONDecodeError, TypeError):
                pass
        pbar.update(1)
        return seed, task, texts

    coros = [
        _tracked(clients[i % len(clients)], seed, task)
        for i, (seed, task) in enumerate(zip(seeds, tasks))
    ]
    results = await asyncio.gather(*coros)
    pbar.close()

    # Diagnostic summary
    if latencies:
        arr = np.array(latencies)
        diag = (
            f"[{desc}] n={len(arr)} | "
            f"lat min={np.min(arr):.1f} P50={np.percentile(arr,50):.1f} "
            f"P90={np.percentile(arr,90):.1f} P99={np.percentile(arr,99):.1f} "
            f"max={np.max(arr):.1f}s"
        )
        if output_lens:
            olen = np.array(output_lens)
            diag += (
                f" | out_chars mean={np.mean(olen):.0f} "
                f"P90={np.percentile(olen,90):.0f} max={np.max(olen)}"
            )
        tqdm.write(diag)

    for seed, task, texts in results:
        if texts:
            texts_out.extend(texts)
            seeds_out.extend([seed] * len(texts))
            tasks_out.extend([task] * len(texts))

    return {"texts": texts_out, "seeds": seeds_out, "tasks": tasks_out}


# =============================================================================
# ES Trainer
# =============================================================================


class ESTrainer:
    """Manages ES training: adapter init, rollouts, scoring, updates."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.clients: list[PieClient] = []
        self.train_dataset = None
        self.eval_dataset = None
        self._exit_stack = AsyncExitStack()
        self._server = None  # Only used in local mode

    # -- Lifecycle ------------------------------------------------------------

    async def setup(self):
        """Connect to server(s), initialize adapter, load datasets."""
        if self.config.local:
            await self._setup_local()
        else:
            await self._setup_distributed()

        await self._install_local_inferlets()
        await self._init_adapter()
        self._load_datasets()

    async def _install_local_inferlets(self):
        """Build-then-install the locally-built es-* inferlets on every client.
        Bypasses the registry — uses the .wasm artifacts in
        sdk/demo/zo-training/inferlets/target/wasm32-wasip2/release/."""
        import os as _os
        here = _os.path.dirname(_os.path.abspath(__file__))
        target_dir = _os.path.join(
            here, "inferlets", "target", "wasm32-wasip2", "release"
        )
        manifests = _os.path.join(here, "inferlets")
        artifacts = {
            "es-init":    (_os.path.join(target_dir, "es_init.wasm"),    _os.path.join(manifests, "es-init", "Pie.toml")),
            "es-rollout": (_os.path.join(target_dir, "es_rollout.wasm"), _os.path.join(manifests, "es-rollout", "Pie.toml")),
            "es-update":  (_os.path.join(target_dir, "es_update.wasm"),  _os.path.join(manifests, "es-update", "Pie.toml")),
        }
        for name, (wasm, mf) in artifacts.items():
            if not _os.path.exists(wasm) or not _os.path.exists(mf):
                raise RuntimeError(
                    f"Local inferlet artifact missing: {wasm} or {mf}. "
                    f"Build with: cd inferlets && cargo build --release --target wasm32-wasip2"
                )
        tqdm.write("📦 Installing locally-built inferlets on all clients...")
        for client in self.clients:
            for name, (wasm, mf) in artifacts.items():
                await client.install_program(wasm, mf, force_overwrite=True)
        tqdm.write(f"✅ Installed {len(artifacts)} local inferlets on {len(self.clients)} client(s).")

    async def teardown(self):
        """Clean up all resources."""
        await self._exit_stack.aclose()
        tqdm.write("Resources cleaned up.")

    async def _setup_local(self):
        """Spin up an in-process Pie server and connect."""
        from pie.server import Server
        from pie.config import (
            Config, ModelConfig, ServerConfig, AuthConfig,
            TelemetryConfig, DriverConfig,
        )

        device = (
            [d.strip() for d in self.config.device.split(",")]
            if "," in self.config.device
            else [self.config.device]
        )

        # Driver-specific subsection (`[model.X.driver.<type>]`).
        driver_subsection: dict = {}
        if self.config.driver == "sglang":
            # ZO training requires the QKV adapter wrappers; without
            # enable_adapter the four adapter RPCs would still be wired
            # but spec_propose etc. wouldn't fire and adapter math would
            # be a no-op. Setting it here keeps the script ergonomic.
            driver_subsection["enable_adapter"] = True
            driver_subsection["mem_fraction_static"] = self.config.gpu_mem_util
            if self.config.cpu_mem_budget > 0:
                driver_subsection["cpu_mem_budget_in_gb"] = self.config.cpu_mem_budget
        elif self.config.cpu_mem_budget > 0:
            # Native / dummy honour cpu_mem_budget_in_gb directly.
            driver_subsection["cpu_mem_budget_in_gb"] = self.config.cpu_mem_budget

        cfg = Config(
            server=ServerConfig(port=0),
            auth=AuthConfig(enabled=False),
            telemetry=TelemetryConfig(),
            models={
                "default": ModelConfig(
                    name="default",
                    hf_repo=self.config.model,
                    driver=DriverConfig(
                        type=self.config.driver,
                        device=device,
                        options=driver_subsection,
                    ),
                    default_token_budget=self.config.default_token_budget,
                ),
            },
        )

        tqdm.write(
            f"🚀 Starting local Pie server "
            f"(model={self.config.model}, device={device}, driver={self.config.driver})..."
        )
        server = await self._exit_stack.enter_async_context(Server(cfg))
        self._server = server
        client = await server.connect()
        self.clients = [client]
        tqdm.write("✅ Local server ready.")

    async def _setup_distributed(self):
        """Connect to remote Pie server(s)."""
        tqdm.write("🔌 Connecting to Pie servers...")
        for uri in self.config.servers:
            client = await self._exit_stack.enter_async_context(PieClient(uri))
            await client.authenticate("main-user")
            self.clients.append(client)
        tqdm.write(f"✅ Connected to {len(self.clients)} server(s).")

    def _load_datasets(self):
        """Load training and evaluation datasets."""
        tqdm.write(f"💿 Loading dataset: {self.config.dataset}")
        if self.config.dataset == "countdown":
            self.train_dataset = CountdownDataset(
                self.config.data_path, "train", self.config.test_size
            )
            self.eval_dataset = CountdownDataset(
                self.config.data_path, "test", self.config.test_size
            )
        elif self.config.dataset == "math":
            self.train_dataset = OpenR1MathDataset("train", self.config.test_size)
            self.eval_dataset = OpenR1MathDataset("test", self.config.test_size)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        tqdm.write("✅ Datasets loaded.")

    async def _init_adapter(self):
        """Initialize ES adapter on all clients."""
        tqdm.write("⚙️  Initializing adapter on all clients...")
        init_input = {
            "name": self.config.adapter_name,
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "population_size": self.config.population_size,
            "mu_fraction": self.config.mu_fraction,
            "initial_sigma": self.config.initial_sigma,
            "upload": self.config.resume_from,
        }
        results = await run_on_all_clients(
            self.clients, self.config.inferlet_names["es-init"], init_input,
        )
        if any(r is None for r in results):
            failed = [i for i, r in enumerate(results) if r is None]
            raise RuntimeError(f"Adapter init failed on client(s): {failed}")
        tqdm.write("✅ Adapter initialized.")

    # -- Training loop --------------------------------------------------------

    async def train(self):
        """Main training loop."""
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"🚀 Starting ES Training ({len(self.clients)} client(s))")
        tqdm.write(f"{'='*50}")

        consecutive_failures = 0
        MAX_FAILURES = 3

        start_step = 1
        if self.config.resume_from:
            m = re.search(r'step-(\d+)', self.config.resume_from)
            if m:
                start_step = int(m.group(1)) + 1
                tqdm.write(f"🔄 Resuming from step {start_step} (checkpoint: {self.config.resume_from})")
        for step in range(start_step, self.config.training_steps + 1):
            t0 = time.time()
            tqdm.write(f"\n--- Step {step}/{self.config.training_steps} ---")

            # Generate seeds and sample tasks
            base_seeds = np.random.randint(
                -(2**63), 2**63 - 1,
                size=self.config.population_size, dtype=np.int64,
            )
            seeds = np.repeat(base_seeds, self.config.tasks_per_seed)
            task_indices = np.random.choice(
                len(self.train_dataset),
                size=self.config.population_size * self.config.tasks_per_seed,
            )
            tasks = [self.train_dataset[i] for i in task_indices]

            # Rollout phase
            results = await run_rollouts(
                self.clients,
                self.config.inferlet_names["es-rollout"],
                seeds, tasks, self.config,
                desc=f"Step {step}",
            )

            # Score and aggregate
            scores, metrics = self._score(base_seeds, results, step)

            # Update phase
            await self._update(base_seeds, scores, step)

            dt = time.time() - t0
            metrics["perf/step_duration_sec"] = dt
            n = metrics["num_finished_episodes"]
            tqdm.write(
                f"Step {step}: reward={metrics['mean_reward']:.4f} | "
                f"episodes={n} | {dt:.1f}s"
            )

            if n == 0:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    tqdm.write(f"\n❌ Aborting: {MAX_FAILURES} steps with 0 episodes.")
                    return
            else:
                consecutive_failures = 0

            # Evaluation
            if step % self.config.eval_every == 0 or step == self.config.training_steps:
                await self._evaluate(step)

        tqdm.write("\n🎉 Training finished!")

    # -- Scoring --------------------------------------------------------------

    def _score(self, base_seeds, rollout_results, step: int = 0) -> tuple[list[float], dict]:
        """Score generations and aggregate by seed."""
        reward_infos = [
            task["verifier"](text)
            for text, task in zip(rollout_results["texts"], rollout_results["tasks"])
        ]
        scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
        format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]
        answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

        # Save input-output pairs to CSV on checkpoint steps
        if step > 0 and step % self.config.checkpoint_every == 0:
            self._save_rollout_csv(
                step, rollout_results, scores, format_rewards, answer_rewards,
            )

        by_seed = defaultdict(list)
        for s, sc in zip(rollout_results["seeds"], scores):
            by_seed[int(s)].append(sc)

        aggregated = []
        missing = 0
        for s in base_seeds:
            vals = by_seed.get(int(s))
            if vals:
                aggregated.append(float(np.mean(vals)))
            else:
                aggregated.append(0.0)
                missing += 1

        mu_k = max(1, int(np.ceil(self.config.mu_fraction * self.config.population_size)))
        out_lens = [len(t.split()) for t in rollout_results["texts"]]

        metrics = {
            "mean_reward": float(np.mean(scores)) if scores else 0.0,
            "mean_format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "mean_answer_reward": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
            "std_reward": float(np.std(scores)) if scores else 0.0,
            "num_finished_episodes": len(rollout_results["texts"]),
            "mean_response_len": float(np.mean(out_lens)) if out_lens else 0.0,
            "es/mean_population_score": float(np.mean(aggregated)),
            "es/mean_fittest_score": float(np.mean(sorted(aggregated, reverse=True)[:mu_k])),
            "rollout/missing_seeds": missing,
        }
        return aggregated, metrics

    def _save_rollout_csv(
        self, step: int, rollout_results: dict,
        scores: list[float], format_rewards: list[float], answer_rewards: list[float],
    ) -> None:
        """Save input-output response pairs to a CSV for debugging."""
        log_dir = "rollout_logs"
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"rollouts_step_{step}.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "problem", "response", "reward",
                "format_reward", "answer_reward", "response_len",
            ])
            for i, (text, task, seed) in enumerate(
                zip(rollout_results["texts"], rollout_results["tasks"], rollout_results["seeds"])
            ):
                problem = task.get("problem", str(task))
                writer.writerow([
                    int(seed), problem, text, scores[i],
                    format_rewards[i], answer_rewards[i], len(text.split()),
                ])

        tqdm.write(f"📝 Saved {len(scores)} rollouts → {path}")

    # -- Update ---------------------------------------------------------------

    async def _update(self, base_seeds, scores, step: int):
        """Broadcast update to all clients."""
        tqdm.write("Phase: Update")
        update_input = {
            "name": self.config.adapter_name,
            "seeds": [int(s) for s in base_seeds],
            "scores": [float(s) for s in scores],
            "max_sigma": self.config.max_sigma,
        }
        if step > 0 and step % self.config.checkpoint_every == 0:
            ckpt = f"{self.config.adapter_name}-step-{step}"
            tqdm.write(f"💾 Checkpoint: {ckpt}")
            update_input["download"] = ckpt

        await run_on_all_clients(
            self.clients, self.config.inferlet_names["es-update"], update_input,
        )

    # -- Evaluation -----------------------------------------------------------

    async def _evaluate(self, step: int):
        """Run evaluation on the central model (seed=0)."""
        tqdm.write(f"\n{'─'*20} Eval @ Step {step} {'─'*20}")
        t0 = time.time()

        num_eval = len(self.eval_dataset)
        eval_seeds = np.zeros(num_eval, dtype=np.int64)
        eval_tasks = [self.eval_dataset[i] for i in range(num_eval)]

        results = await run_rollouts(
            self.clients,
            self.config.inferlet_names["es-rollout"],
            eval_seeds, eval_tasks, self.config,
            desc="eval",
        )

        _, metrics = self._score(eval_seeds, results, step)
        tqdm.write(f"✅ Eval: mean_reward={metrics['mean_reward']:.4f} ({time.time()-t0:.1f}s)")


# =============================================================================
# CLI & Main
# =============================================================================


def train(**kwargs):
    """Launch ES training. All TrainingConfig fields can be passed as CLI flags.

    Examples::

        python train.py --local --training_steps 1000
        python train.py --local --resume_from evo-math-v2-step-255
        python train.py --servers ws://gpu1:8080 ws://gpu2:8080
    """
    config = TrainingConfig(**kwargs)

    async def _run():
        trainer = ESTrainer(config)
        try:
            await trainer.setup()
            await trainer.train()
        except Exception as e:
            tqdm.write(f"\n❌ Error: {e}")
            raise
        finally:
            await trainer.teardown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted.")


if __name__ == "__main__":
    fire.Fire(train)
