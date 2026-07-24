from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch

from gpu_lr1.benchmark import RotatingLogits, Timing, machine_metadata, measure
from gpu_lr1.lr1 import CanonicalLR1Compiler, pack_lr1_tables
from gpu_lr1.lr1_tokens import (
    BoundedLR1TokenAutomaton,
    LR1ConfigurationLimitError,
    LR1TokenCompileTimeoutError,
    LR1TokenVocabulary,
    PackedLR1TokenTables,
    compile_bounded_lr1_token_automaton,
    pack_bounded_lr1_token_automata,
    select_and_advance_bounded_lr1_cpu,
    triton_bounded_lr1_step,
)
from gpu_lr1.lr1_workloads import balanced_grammar, byte_arithmetic_grammar
from gpu_lr1.vocab import Vocabulary


@dataclass(frozen=True)
class TokenBenchmarkProfile:
    vocab_sizes: tuple[int, ...]
    arithmetic_depths: tuple[int, ...]
    balanced_depths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    mixes: tuple[str, ...]
    warmup: int
    iterations: int


class CyclingLogits:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def next(self) -> torch.Tensor:
        value = self.values[self.index % self.values.shape[0]]
        self.index += 1
        return value


PROFILES = {
    "quick": TokenBenchmarkProfile(
        vocab_sizes=(256, 1024),
        arithmetic_depths=(4, 6),
        balanced_depths=(8, 12),
        batch_sizes=(1, 32, 512),
        mixes=("hot", "heterogeneous", "sparse", "dense"),
        warmup=3,
        iterations=10,
    ),
    "full": TokenBenchmarkProfile(
        vocab_sizes=(256, 1024, 4096),
        arithmetic_depths=(4, 6, 8),
        balanced_depths=(8, 12, 16),
        batch_sizes=(1, 8, 32, 128, 512, 2048),
        mixes=("hot", "heterogeneous", "sparse", "dense"),
        warmup=8,
        iterations=30,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark bounded tokenizer-token LR(1) configuration expansion"
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--vocab-sizes", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--runtime-vocab-size", type=int, default=1024)
    parser.add_argument("--synthetic-logit-columns", type=int, default=32768)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--qwen-buffer-count", type=int, default=4)
    parser.add_argument("--compile-timeout", type=float, default=10.0)
    parser.add_argument("--max-configurations", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lr1-token-benchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the LR(1) token benchmark requires a CUDA GPU")
    base = PROFILES[args.profile]
    profile = TokenBenchmarkProfile(
        vocab_sizes=(
            tuple(args.vocab_sizes)
            if args.vocab_sizes is not None
            else base.vocab_sizes
        ),
        arithmetic_depths=base.arithmetic_depths,
        balanced_depths=base.balanced_depths,
        batch_sizes=(
            tuple(args.batch_sizes)
            if args.batch_sizes is not None
            else base.batch_sizes
        ),
        mixes=base.mixes,
        warmup=args.warmup if args.warmup is not None else base.warmup,
        iterations=(
            args.iterations if args.iterations is not None else base.iterations
        ),
    )
    if args.runtime_vocab_size not in profile.vocab_sizes:
        raise ValueError("runtime-vocab-size must be included in vocab-sizes")
    if args.synthetic_logit_columns < args.runtime_vocab_size:
        raise ValueError(
            "synthetic-logit-columns must cover the runtime vocabulary"
        )
    if args.qwen_buffer_count <= 0:
        raise ValueError("qwen-buffer-count must be positive")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    compile_records, cache = compile_scaling_matrix(
        profile,
        compile_timeout=args.compile_timeout,
        max_configurations=args.max_configurations,
    )
    runtime_keys = (
        ("arithmetic", 4, args.runtime_vocab_size),
        ("arithmetic", 6, args.runtime_vocab_size),
        ("balanced", 8, args.runtime_vocab_size),
        ("balanced", 12, args.runtime_vocab_size),
    )
    missing = [key for key in runtime_keys if key not in cache]
    if missing:
        raise RuntimeError(f"runtime automata failed to compile: {missing}")
    runtime_automata = [cache[key] for key in runtime_keys]
    tables = pack_bounded_lr1_token_automata(runtime_automata)
    tensors = tables.torch_tensors(device)

    runtime_records = []
    for mix in profile.mixes:
        for batch_size in profile.batch_sizes:
            states = select_states(
                tables,
                mix=mix,
                batch_size=batch_size,
                seed=args.seed,
            )
            runtime_records.extend(
                benchmark_runtime(
                    tables,
                    tensors,
                    states,
                    mix=mix,
                    logit_columns=args.synthetic_logit_columns,
                    warmup=profile.warmup,
                    iterations=profile.iterations,
                    device=device,
                )
            )

    qwen_records, qwen_tables = compile_qwen_probe(
        args.qwen_model,
        compile_timeout=args.compile_timeout,
        max_configurations=args.max_configurations,
    )
    qwen_tensors = qwen_tables.torch_tensors(device)
    qwen_runtime_records = []
    for batch_size in profile.batch_sizes:
        states = select_states(
            qwen_tables,
            mix="heterogeneous",
            batch_size=batch_size,
            seed=args.seed,
        )
        qwen_runtime_records.append(
            benchmark_large_vocab_runtime(
                qwen_tables,
                qwen_tensors,
                states,
                mix="qwen3_heterogeneous",
                buffer_count=args.qwen_buffer_count,
                warmup=profile.warmup,
                iterations=profile.iterations,
                device=device,
            )
        )

    direct_tables = pack_lr1_tables(
        [
            CanonicalLR1Compiler(
                byte_arithmetic_grammar("direct-byte-arithmetic-d4")
            ).compile(),
            CanonicalLR1Compiler(
                byte_arithmetic_grammar("direct-byte-arithmetic-d6")
            ).compile(),
            CanonicalLR1Compiler(
                balanced_grammar("direct-balanced-d8")
            ).compile(),
            CanonicalLR1Compiler(
                balanced_grammar("direct-balanced-d12")
            ).compile(),
        ]
    )
    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "profile_details": asdict(profile),
            "token_alphabet": "0123456789+*()",
            "bounded_semantics": (
                "exact for reachable LR state stacks up to max_stack_depth"
            ),
        },
        "compile_scaling": compile_records,
        "runtime_tables": token_table_summary(tables),
        "qwen3_probe": {
            "model": args.qwen_model,
            "grammars": qwen_records,
            "runtime_tables": token_table_summary(qwen_tables),
            "runtime_benchmarks": qwen_runtime_records,
        },
        "direct_lr_tables": {
            "states": direct_tables.num_states,
            "productions": direct_tables.num_productions,
            "memory_bytes": direct_tables.memory_bytes(),
            "memory_total_bytes": int(
                sum(direct_tables.memory_bytes().values())
            ),
        },
        "runtime_benchmarks": runtime_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload)


def compile_scaling_matrix(
    profile: TokenBenchmarkProfile,
    *,
    compile_timeout: float,
    max_configurations: int,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, int, int], BoundedLR1TokenAutomaton],
]:
    records = []
    cache: dict[tuple[str, int, int], BoundedLR1TokenAutomaton] = {}
    for vocab_size in profile.vocab_sizes:
        vocabulary = bpe_like_vocabulary(vocab_size)
        for family, depths in (
            ("arithmetic", profile.arithmetic_depths),
            ("balanced", profile.balanced_depths),
        ):
            for depth in depths:
                grammar = (
                    byte_arithmetic_grammar(
                        f"byte-arithmetic-d{depth}-v{vocab_size}"
                    )
                    if family == "arithmetic"
                    else balanced_grammar(
                        f"balanced-d{depth}-v{vocab_size}"
                    )
                )
                compiled = CanonicalLR1Compiler(grammar).compile()
                byte_terminals = (
                    {byte: chr(byte) for byte in b"0123456789+*()"}
                    if family == "arithmetic"
                    else {ord("("): "(", ord(")"): ")"}
                )
                token_vocabulary = LR1TokenVocabulary.from_byte_vocabulary(
                    compiled,
                    vocabulary,
                    byte_terminals,
                )
                key = (family, depth, vocab_size)
                started = time.perf_counter()
                try:
                    automaton = compile_bounded_lr1_token_automaton(
                        compiled,
                        token_vocabulary,
                        max_stack_depth=depth,
                        max_configurations=max_configurations,
                        max_compile_seconds=compile_timeout,
                    )
                except (
                    LR1ConfigurationLimitError,
                    LR1TokenCompileTimeoutError,
                ) as error:
                    records.append(
                        {
                            "family": family,
                            "vocab_size": vocab_size,
                            "representable_tokens": (
                                token_vocabulary.representable_tokens
                            ),
                            "max_stack_depth": depth,
                            "status": type(error).__name__,
                            "seconds": time.perf_counter() - started,
                            "message": str(error),
                        }
                    )
                    continue
                cache[key] = automaton
                records.append(
                    {
                        "family": family,
                        "vocab_size": vocab_size,
                        "representable_tokens": (
                            token_vocabulary.representable_tokens
                        ),
                        "max_terminals_per_token": (
                            token_vocabulary.max_terminals_per_token
                        ),
                        "max_stack_depth": depth,
                        "status": "ok",
                        "seconds": automaton.compile_seconds,
                        "states": automaton.num_states,
                        "edges": int(automaton.csr_indices.size),
                        "max_row_nnz": int(automaton.row_nnz.max()),
                        "mean_row_nnz": float(automaton.row_nnz.mean()),
                        "max_reductions_per_edge": int(
                            automaton.csr_reductions.max(initial=0)
                        ),
                        "overflow_edges": automaton.overflow_edges,
                        "runtime_bytes": automaton_runtime_bytes(automaton),
                        "stack_signature_entries": int(
                            automaton.config_depths.sum()
                        ),
                    }
                )
    return records, cache


def compile_qwen_probe(
    model_name: str,
    *,
    compile_timeout: float,
    max_configurations: int,
) -> tuple[list[dict[str, object]], PackedLR1TokenTables]:
    vocabulary = Vocabulary.huggingface(model_name)
    records = []
    automata = []
    for family, depth in (("arithmetic", 4), ("balanced", 16)):
        grammar = (
            byte_arithmetic_grammar("qwen3-byte-arithmetic")
            if family == "arithmetic"
            else balanced_grammar("qwen3-balanced")
        )
        compiled = CanonicalLR1Compiler(grammar).compile()
        byte_terminals = (
            {byte: chr(byte) for byte in b"0123456789+*()"}
            if family == "arithmetic"
            else {ord("("): "(", ord(")"): ")"}
        )
        token_vocabulary = LR1TokenVocabulary.from_byte_vocabulary(
            compiled,
            vocabulary,
            byte_terminals,
        )
        automaton = compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=depth,
            max_configurations=max_configurations,
            max_compile_seconds=compile_timeout,
        )
        automata.append(automaton)
        records.append(
            {
                "family": family,
                "vocab_size": vocabulary.size,
                "representable_tokens": (
                    token_vocabulary.representable_tokens
                ),
                "max_terminals_per_token": (
                    token_vocabulary.max_terminals_per_token
                ),
                "max_stack_depth": depth,
                "seconds": automaton.compile_seconds,
                "states": automaton.num_states,
                "edges": int(automaton.csr_indices.size),
                "max_row_nnz": int(automaton.row_nnz.max()),
                "runtime_bytes": automaton_runtime_bytes(automaton),
            }
        )
    return records, pack_bounded_lr1_token_automata(automata)


def benchmark_large_vocab_runtime(
    tables: PackedLR1TokenTables,
    tensors,
    states_np: np.ndarray,
    *,
    mix: str,
    buffer_count: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    batch_size = states_np.size
    states = torch.from_numpy(states_np).to(device)
    logits = torch.randn(
        (buffer_count, batch_size, tables.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    starts = tables.csr_indptr[states_np]
    desired = tables.csr_indices[starts]
    expected_states = tables.csr_next_state[starts]
    buffer_ids = torch.arange(buffer_count, device=device)[:, None]
    batch_ids = torch.arange(batch_size, device=device)[None, :]
    desired_device = torch.from_numpy(desired).to(device)[None, :]
    logits[buffer_ids, batch_ids, desired_device] = 20.0
    source = CyclingLogits(logits)
    output_tokens = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=device,
    )
    output_states = torch.empty_like(states)

    def function() -> torch.Tensor:
        return triton_bounded_lr1_step(
            source.next(),
            tensors,
            states,
            output_tokens=output_tokens,
            output_states=output_states,
        )[1]

    source.reset()
    function()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(output_tokens.cpu().numpy(), desired)
    np.testing.assert_array_equal(
        output_states.cpu().numpy(),
        expected_states,
    )
    source.reset()
    timing = measure(
        function,
        warmup=warmup,
        iterations=iterations,
        measure_cuda=True,
    )
    row_nnz = tables.row_nnz[states_np]
    depths = tables.config_depths[states_np]
    return {
        "mix": mix,
        "batch_size": batch_size,
        "unique_grammars": int(
            np.unique(
                np.searchsorted(
                    tables.state_offsets[1:],
                    states_np,
                    side="right",
                )
            ).size
        ),
        "row_nnz_min": int(row_nnz.min()),
        "row_nnz_mean": float(row_nnz.mean()),
        "row_nnz_max": int(row_nnz.max()),
        "model_vocab_density_mean": float(
            row_nnz.mean() / tables.vocab_size
        ),
        "config_depth_min": int(depths.min()),
        "config_depth_mean": float(depths.mean()),
        "config_depth_max": int(depths.max()),
        "strategy": "bounded_lr1_qwen3_csr_fused",
        "logits_buffer_count": buffer_count,
        **asdict(timing),
        "sequences_per_second": float(
            batch_size / (timing.cuda_mean_us * 1e-6)
        ),
    }


def benchmark_runtime(
    tables: PackedLR1TokenTables,
    tensors,
    states_np: np.ndarray,
    *,
    mix: str,
    logit_columns: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, object]]:
    batch_size = states_np.size
    states = torch.from_numpy(states_np).to(device)
    pool_count = 1 + warmup + 2 * iterations
    logits_pool = RotatingLogits(
        torch.randn(
            (pool_count, batch_size, logit_columns),
            dtype=torch.float16,
            device=device,
        )
    )
    validation_logits = logits_pool.values[0]
    cpu_logits = validation_logits[:, : tables.vocab_size].float().cpu().numpy()
    expected_tokens, expected_states = select_and_advance_bounded_lr1_cpu(
        cpu_logits,
        tables,
        states_np,
    )
    output_tokens = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=device,
    )
    output_states = torch.empty_like(states)

    def function() -> torch.Tensor:
        return triton_bounded_lr1_step(
            logits_pool.next(),
            tensors,
            states,
            output_tokens=output_tokens,
            output_states=output_states,
        )[1]

    logits_pool.reset(0)
    function()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(
        output_tokens.cpu().numpy(),
        expected_tokens,
    )
    np.testing.assert_array_equal(
        output_states.cpu().numpy(),
        expected_states,
    )
    logits_pool.reset(1)
    timing = measure(
        function,
        warmup=warmup,
        iterations=iterations,
        measure_cuda=True,
    )
    row_nnz = tables.row_nnz[states_np]
    depths = tables.config_depths[states_np]
    common = {
        "mix": mix,
        "batch_size": batch_size,
        "unique_grammars": int(
            np.unique(
                np.searchsorted(
                    tables.state_offsets[1:],
                    states_np,
                    side="right",
                )
            ).size
        ),
        "row_nnz_min": int(row_nnz.min()),
        "row_nnz_mean": float(row_nnz.mean()),
        "row_nnz_max": int(row_nnz.max()),
        "model_vocab_density_mean": float(row_nnz.mean() / logit_columns),
        "config_depth_min": int(depths.min()),
        "config_depth_mean": float(depths.mean()),
        "config_depth_max": int(depths.max()),
    }
    records = [
        {
            **common,
            "strategy": "bounded_lr1_csr_fused",
            **asdict(timing),
            "sequences_per_second": float(
                batch_size / (timing.cuda_mean_us * 1e-6)
            ),
        }
    ]
    cpu_timing = measure_cpu(
        cpu_logits,
        tables,
        states_np,
        warmup=warmup,
        iterations=iterations,
    )
    records.append(
        {
            **common,
            "strategy": "bounded_lr1_cpu_python_reference",
            **asdict(cpu_timing),
            "sequences_per_second": float(
                batch_size / (cpu_timing.wall_mean_us * 1e-6)
            ),
        }
    )
    return records


def select_states(
    tables: PackedLR1TokenTables,
    *,
    mix: str,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + batch_size * 31 + len(mix))
    nnz = tables.row_nnz
    nonempty = np.flatnonzero(nnz > 0)
    if mix == "hot":
        median = np.median(nnz[nonempty])
        state = nonempty[np.argmin(np.abs(nnz[nonempty] - median))]
        return np.full(batch_size, state, dtype=np.int32)
    if mix == "sparse":
        cutoff = np.percentile(nnz[nonempty], 25)
        pool = nonempty[nnz[nonempty] <= cutoff]
        return rng.choice(pool, batch_size, replace=True).astype(np.int32)
    if mix == "dense":
        cutoff = np.percentile(nnz[nonempty], 90)
        pool = nonempty[nnz[nonempty] >= cutoff]
        return rng.choice(pool, batch_size, replace=True).astype(np.int32)
    if mix == "heterogeneous":
        selected = np.empty(batch_size, dtype=np.int32)
        for index in range(batch_size):
            grammar_id = index % tables.num_grammars
            start = int(tables.state_offsets[grammar_id])
            end = int(tables.state_offsets[grammar_id + 1])
            pool = np.arange(start, end, dtype=np.int32)
            pool = pool[nnz[start:end] > 0]
            selected[index] = rng.choice(pool)
        return selected
    raise ValueError(f"unknown mix: {mix}")


def bpe_like_vocabulary(
    size: int,
    alphabet: bytes = b"0123456789+*()",
) -> Vocabulary:
    if size < 2:
        raise ValueError("token vocabulary requires EOS and one token")
    tokens = [b""]
    for length in range(1, 16):
        for values in product(alphabet, repeat=length):
            tokens.append(bytes(values))
            if len(tokens) >= size:
                return Vocabulary(
                    tuple(tokens),
                    name=f"grammar-bpe-{size}",
                )
    raise ValueError("requested token vocabulary is too large")


def automaton_runtime_bytes(automaton: BoundedLR1TokenAutomaton) -> int:
    return int(
        automaton.csr_indptr.nbytes
        + automaton.csr_indices.nbytes
        + automaton.csr_next_state.nbytes
        + automaton.accepting.nbytes
    )


def token_table_summary(tables: PackedLR1TokenTables) -> dict[str, object]:
    row_nnz = tables.row_nnz
    memory = tables.memory_bytes()
    runtime_without_diagnostics = (
        memory["csr_tokens"]
        + memory["csr_next_state"]
        + memory["state_metadata"]
    )
    return {
        "grammars": tables.num_grammars,
        "states": tables.num_states,
        "vocab_size": tables.vocab_size,
        "edges": int(tables.csr_indices.size),
        "compile_seconds": tables.compile_seconds,
        "overflow_edges": tables.overflow_edges,
        "row_nnz": {
            "min": int(row_nnz.min()),
            "median": float(np.median(row_nnz)),
            "p95": float(np.percentile(row_nnz, 95)),
            "max": int(row_nnz.max()),
            "mean": float(row_nnz.mean()),
        },
        "config_depth": {
            "min": int(tables.config_depths.min()),
            "median": float(np.median(tables.config_depths)),
            "p95": float(np.percentile(tables.config_depths, 95)),
            "max": int(tables.config_depths.max()),
        },
        "memory_bytes": memory,
        "runtime_without_diagnostics_bytes": int(runtime_without_diagnostics),
        "memory_total_bytes": int(sum(memory.values())),
    }


def measure_cpu(
    logits: np.ndarray,
    tables: PackedLR1TokenTables,
    states: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> Timing:
    def function() -> None:
        select_and_advance_bounded_lr1_cpu(logits, tables, states)

    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return Timing(
        wall_p50_us=float(statistics.median(ordered)),
        wall_p95_us=float(ordered[p95_index]),
        wall_mean_us=float(statistics.fmean(ordered)),
        cuda_mean_us=None,
    )


def print_summary(payload: dict[str, object]) -> None:
    metadata = payload["metadata"]
    runtime = payload["runtime_tables"]
    failures = sum(
        record["status"] != "ok" for record in payload["compile_scaling"]
    )
    print(
        f"GPU: {metadata['gpu']} | bounded grammars={runtime['grammars']} "
        f"states={runtime['states']} edges={runtime['edges']}"
    )
    print(
        f"compile={runtime['compile_seconds']:.3f}s "
        f"runtime_bytes={runtime['runtime_without_diagnostics_bytes']} "
        f"compile_failures={failures}"
    )
    qwen = payload["qwen3_probe"]["runtime_tables"]
    print(
        f"qwen3_states={qwen['states']} qwen3_edges={qwen['edges']} "
        f"qwen3_runtime_bytes={qwen['runtime_without_diagnostics_bytes']}"
    )
    print("strategy,mix,batch,row_nnz_mean,depth_mean,cuda_us,wall_p50_us")
    for record in payload["runtime_benchmarks"]:
        cuda = record["cuda_mean_us"]
        cuda_text = "" if cuda is None else f"{cuda:.3f}"
        print(
            f"{record['strategy']},{record['mix']},{record['batch_size']},"
            f"{record['row_nnz_mean']:.2f},"
            f"{record['config_depth_mean']:.2f},"
            f"{cuda_text},{record['wall_p50_us']:.3f}"
        )
    for record in payload["qwen3_probe"]["runtime_benchmarks"]:
        cuda = record["cuda_mean_us"]
        cuda_text = "" if cuda is None else f"{cuda:.3f}"
        print(
            f"{record['strategy']},{record['mix']},{record['batch_size']},"
            f"{record['row_nnz_mean']:.2f},"
            f"{record['config_depth_mean']:.2f},"
            f"{cuda_text},{record['wall_p50_us']:.3f}"
        )


if __name__ == "__main__":
    main()
