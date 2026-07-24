from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gpu_lr1.benchmark import (
    Timing,
    machine_metadata,
    measure,
)
from gpu_lr1.kernels import make_argmax_workspace, triton_bitset_argmax
from gpu_lr1.lr1 import CanonicalLR1Compiler
from gpu_lr1.lr1_token_benchmark import CyclingGraphs, CyclingLogits
from gpu_lr1.lr1_tokens import (
    LR1TokenVocabulary,
    compile_bounded_lr1_token_automaton,
    make_bounded_lr1_step_plan,
    pack_bounded_lr1_token_automata,
)
from gpu_lr1.lr1_workloads import (
    bounded_arithmetic_ebnf,
    bounded_balanced_ebnf,
    bounded_balanced_grammar,
    bounded_byte_arithmetic_grammar,
)
from gpu_lr1.vocab import Vocabulary


class _CapturedTensorGraph:
    def __init__(self, graph: torch.cuda.CUDAGraph, output: torch.Tensor) -> None:
        self.graph = graph
        self.output = output

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


class _CyclingTensorGraphs:
    def __init__(self, graphs: list[_CapturedTensorGraph]) -> None:
        self.graphs = graphs
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def next(self) -> torch.Tensor:
        graph = self.graphs[self.index % len(self.graphs)]
        self.index += 1
        return graph.replay()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair Qwen3 gpu-lr1 versus XGrammar benchmark"
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048],
    )
    parser.add_argument(
        "--xgrammar-threads",
        nargs="+",
        default=["1", "2", "4", "8", "auto"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--logits-buffer-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fair-xgrammar-qwen3.json"),
    )
    args = parser.parse_args()

    try:
        import xgrammar as xgr
    except ImportError as exc:
        raise RuntimeError("install gpu-lr1[baselines] first") from exc

    if args.logits_buffer_count <= 0:
        raise ValueError("logits-buffer-count must be positive")
    thread_candidates: list[int | str] = []
    for value in args.xgrammar_threads:
        thread_candidates.append(
            "auto" if value == "auto" else int(value)
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    vocabulary = Vocabulary.huggingface(args.qwen_model)

    ours_started = time.perf_counter()
    grammar_specs = _grammar_specs()
    compiled_lr = [
        CanonicalLR1Compiler(spec["grammar"]).compile()
        for spec in grammar_specs
    ]
    token_vocabularies = [
        LR1TokenVocabulary.from_byte_vocabulary(
            compiled,
            vocabulary,
            spec["byte_terminals"],
        )
        for compiled, spec in zip(compiled_lr, grammar_specs)
    ]
    automata = [
        compile_bounded_lr1_token_automaton(
            compiled,
            token_vocabulary,
            max_stack_depth=spec["stack_depth"],
            max_configurations=50_000,
            max_compile_seconds=30.0,
        )
        for compiled, token_vocabulary, spec in zip(
            compiled_lr,
            token_vocabularies,
            grammar_specs,
        )
    ]
    if any(automaton.overflow_edges for automaton in automata):
        raise AssertionError("fair grammars must compile without bounded overflow")
    tables = pack_bounded_lr1_token_automata(automata)
    ours_compile_seconds = time.perf_counter() - ours_started
    tensors = tables.torch_tensors(device)

    tokenizer_info = xgr.TokenizerInfo(
        list(vocabulary.tokens),
        vocab_type=xgr.VocabType.RAW,
        vocab_size=vocabulary.size,
        stop_token_ids=[vocabulary.eos_token_id],
    )
    xcompiler = xgr.GrammarCompiler(tokenizer_info)
    xcompile_started = time.perf_counter()
    xcompiled = [
        xcompiler.compile_grammar(xgr.Grammar.from_ebnf(spec["ebnf"]))
        for spec in grammar_specs
    ]
    xgrammar_compile_seconds = time.perf_counter() - xcompile_started
    xgrammar_cache_size_bytes = int(xcompiler.get_cache_size_bytes())

    records = []
    mask_checks = []
    for batch_size in args.batch_sizes:
        states, grammar_ids, local_states = _select_fair_states(
            automata,
            tables.state_offsets,
            batch_size=batch_size,
            seed=args.seed,
        )
        matcher_started = time.perf_counter()
        matchers = []
        for grammar_id, local_state in zip(grammar_ids, local_states):
            matcher = xgr.GrammarMatcher(xcompiled[grammar_id])
            for token_id in automata[
                grammar_id
            ].config_witness_tokens[local_state]:
                if not matcher.accept_token(token_id):
                    raise AssertionError(
                        "XGrammar rejected a gpu-lr1 configuration witness"
                    )
            matchers.append(matcher)
        matcher_setup_seconds = time.perf_counter() - matcher_started

        host_mask = torch.empty(
            xgr.get_bitmask_shape(batch_size, vocabulary.size),
            dtype=xgr.bitmask_dtype,
            pin_memory=True,
        )
        one_thread_matcher = xgr.BatchGrammarMatcher(max_threads=1)
        one_thread_matcher.batch_fill_next_token_bitmask(matchers, host_mask)
        expected_mask = _expected_bitmask(tables, states)
        actual_mask = host_mask.numpy().view(np.uint32)
        _mask_unused_bits(actual_mask, vocabulary.size)
        if not np.array_equal(actual_mask, expected_mask):
            different = int(np.count_nonzero(actual_mask != expected_mask))
            raise AssertionError(
                f"XGrammar and gpu-lr1 masks differ in {different} words"
            )
        mask_checks.append(
            {
                "batch_size": batch_size,
                "rows": batch_size,
                "words_compared": int(actual_mask.size),
                "equal": True,
            }
        )

        fill_timings = {}
        batch_matchers = {}
        for threads in thread_candidates:
            batch_matcher = xgr.BatchGrammarMatcher(max_threads=threads)

            def fill(
                batch_matcher=batch_matcher,
            ) -> None:
                batch_matcher.batch_fill_next_token_bitmask(
                    matchers,
                    host_mask,
                )

            fill_timing = _measure_cpu(
                fill,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            fill_timings[str(threads)] = fill_timing
            batch_matchers[str(threads)] = batch_matcher
        best_threads = min(
            fill_timings,
            key=lambda key: fill_timings[key]["p50_us"],
        )
        best_batch_matcher = batch_matchers[best_threads]
        fill_timing = fill_timings[best_threads]

        states_device = torch.from_numpy(states).to(device)
        starts = tables.csr_indptr[states]
        desired = tables.csr_indices[starts]
        desired_next = tables.csr_next_state[starts]
        desired_device = torch.from_numpy(desired).to(device)
        desired_next_device = torch.from_numpy(desired_next).to(device)
        batch_ids = torch.arange(batch_size, device=device)
        buffer_ids = torch.arange(
            args.logits_buffer_count,
            device=device,
        )[:, None]
        logits = torch.randn(
            (
                args.logits_buffer_count,
                batch_size,
                vocabulary.size,
            ),
            dtype=torch.float16,
            device=device,
        )
        logits[buffer_ids, batch_ids[None, :], desired_device[None, :]] = 20.0

        gpu_tokens = torch.empty(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        gpu_states = torch.empty_like(states_device)
        gpu_plan_started = time.perf_counter()
        gpu_plan = make_bounded_lr1_step_plan(
            logits[0],
            tensors,
            states_device,
            output_tokens=gpu_tokens,
            output_states=gpu_states,
            autotune=True,
        )
        gpu_plan_setup_seconds = time.perf_counter() - gpu_plan_started
        gpu_source = CyclingLogits(logits)
        gpu_source.reset()
        gpu_plan_timing = measure(
            lambda: gpu_plan(gpu_source.next())[1],
            warmup=args.warmup,
            iterations=args.iterations,
            measure_cuda=True,
        )
        gpu_graph_started = time.perf_counter()
        gpu_graph_source = CyclingGraphs(
            [
                gpu_plan.capture(logits[index])
                for index in range(args.logits_buffer_count)
            ]
        )
        gpu_graph_setup_seconds = time.perf_counter() - gpu_graph_started
        gpu_graph_source.reset()
        gpu_graph_timing = measure(
            gpu_graph_source.next,
            warmup=args.warmup,
            iterations=args.iterations,
            measure_cuda=True,
        )
        if not torch.equal(gpu_tokens, desired_device):
            raise AssertionError("gpu-lr1 selected an unexpected token")
        if not torch.equal(gpu_states, desired_next_device):
            raise AssertionError("gpu-lr1 produced an unexpected next state")

        device_mask = torch.empty_like(host_mask, device=device)
        local_rows = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        bitset_workspace = make_argmax_workspace(
            batch_size,
            vocabulary.size,
            device=device,
        )
        fused_tokens = torch.empty_like(gpu_tokens)
        x_source = CyclingLogits(logits.clone())

        def fill_mask() -> None:
            best_batch_matcher.batch_fill_next_token_bitmask(
                matchers,
                host_mask,
            )

        fill_mask()
        device_mask.copy_(host_mask, non_blocking=True)
        torch.cuda.synchronize()
        x_source.reset()
        x_fused_gpu_timing = measure(
            lambda: triton_bitset_argmax(
                x_source.next(),
                device_mask,
                local_rows,
                workspace=bitset_workspace,
                output=fused_tokens,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            measure_cuda=True,
        )
        if not torch.equal(fused_tokens, desired_device):
            raise AssertionError("fused XGrammar mask selected wrong token")

        x_fused_graphs = []
        for index in range(args.logits_buffer_count):
            triton_bitset_argmax(
                logits[index],
                device_mask,
                local_rows,
                workspace=bitset_workspace,
                output=fused_tokens,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                triton_bitset_argmax(
                    logits[index],
                    device_mask,
                    local_rows,
                    workspace=bitset_workspace,
                    output=fused_tokens,
                )
            x_fused_graphs.append(
                _CapturedTensorGraph(graph, fused_tokens)
            )
        x_fused_graph_source = _CyclingTensorGraphs(x_fused_graphs)
        x_fused_graph_source.reset()
        x_fused_graph_timing = measure(
            x_fused_graph_source.next,
            warmup=args.warmup,
            iterations=args.iterations,
            measure_cuda=True,
        )
        x_copy_graphs = []
        for index in range(args.logits_buffer_count):
            device_mask.copy_(host_mask, non_blocking=True)
            triton_bitset_argmax(
                logits[index],
                device_mask,
                local_rows,
                workspace=bitset_workspace,
                output=fused_tokens,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                device_mask.copy_(host_mask, non_blocking=True)
                triton_bitset_argmax(
                    logits[index],
                    device_mask,
                    local_rows,
                    workspace=bitset_workspace,
                    output=fused_tokens,
                )
            x_copy_graphs.append(
                _CapturedTensorGraph(graph, fused_tokens)
            )
        x_copy_graph_source = _CyclingTensorGraphs(x_copy_graphs)

        x_source.reset()

        def xgrammar_fused_serial() -> torch.Tensor:
            fill_mask()
            device_mask.copy_(host_mask, non_blocking=True)
            return triton_bitset_argmax(
                x_source.next(),
                device_mask,
                local_rows,
                workspace=bitset_workspace,
                output=fused_tokens,
            )

        x_fused_serial_timing = _measure_cuda_wall(
            xgrammar_fused_serial,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        x_fused_graph_source.reset()

        def xgrammar_fused_graph_serial() -> torch.Tensor:
            fill_mask()
            device_mask.copy_(host_mask, non_blocking=True)
            return x_fused_graph_source.next()

        x_fused_graph_serial_timing = _measure_cuda_wall(
            xgrammar_fused_graph_serial,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        x_copy_graph_source.reset()

        def xgrammar_copy_graph_serial() -> torch.Tensor:
            fill_mask()
            return x_copy_graph_source.next()

        x_copy_graph_serial_timing = _measure_cuda_wall(
            xgrammar_copy_graph_serial,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        x_source.reset()

        def xgrammar_fused_stateful() -> torch.Tensor:
            output = xgrammar_fused_serial()
            token_list = output.cpu().tolist()
            accepted = best_batch_matcher.batch_accept_token(
                matchers,
                token_list,
            )
            if not all(accepted):
                raise AssertionError("XGrammar rejected its selected token")
            best_batch_matcher.batch_rollback(matchers, 1)
            return output

        x_fused_stateful_timing = _measure_cuda_wall(
            xgrammar_fused_stateful,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        x_fused_graph_source.reset()

        def xgrammar_fused_graph_stateful() -> torch.Tensor:
            output = xgrammar_fused_graph_serial()
            token_list = output.cpu().tolist()
            accepted = best_batch_matcher.batch_accept_token(
                matchers,
                token_list,
            )
            if not all(accepted):
                raise AssertionError("XGrammar rejected its selected token")
            best_batch_matcher.batch_rollback(matchers, 1)
            return output

        x_fused_graph_stateful_timing = _measure_cuda_wall(
            xgrammar_fused_graph_stateful,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        x_copy_graph_source.reset()

        def xgrammar_copy_graph_stateful() -> torch.Tensor:
            output = xgrammar_copy_graph_serial()
            token_list = output.cpu().tolist()
            accepted = best_batch_matcher.batch_accept_token(
                matchers,
                token_list,
            )
            if not all(accepted):
                raise AssertionError("XGrammar rejected its selected token")
            best_batch_matcher.batch_rollback(matchers, 1)
            return output

        x_copy_graph_stateful_timing = _measure_cuda_wall(
            xgrammar_copy_graph_stateful,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        native_logits = logits.clone()
        native_source = CyclingLogits(native_logits)
        native_tokens = torch.empty(
            batch_size,
            dtype=torch.int64,
            device=device,
        )
        native_source.reset()

        def xgrammar_native_serial() -> torch.Tensor:
            fill_mask()
            device_mask.copy_(host_mask, non_blocking=True)
            working_logits = native_source.next()
            xgr.apply_token_bitmask_inplace(
                working_logits,
                device_mask,
                vocab_size=vocabulary.size,
                backend="triton",
            )
            return torch.argmax(
                working_logits,
                dim=1,
                out=native_tokens,
            )

        x_native_serial_timing = _measure_cuda_wall(
            xgrammar_native_serial,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if not torch.equal(native_tokens, desired_device.to(torch.int64)):
            raise AssertionError("native XGrammar path selected wrong token")

        common = {
            "batch_size": batch_size,
            "unique_grammars": len(set(grammar_ids)),
            "vocab_size": vocabulary.size,
            "allowed_tokens_min": int(tables.row_nnz[states].min()),
            "allowed_tokens_mean": float(tables.row_nnz[states].mean()),
            "allowed_tokens_max": int(tables.row_nnz[states].max()),
            "matcher_setup_seconds": matcher_setup_seconds,
            "gpu_plan_setup_seconds": gpu_plan_setup_seconds,
            "gpu_graph_setup_seconds": gpu_graph_setup_seconds,
            "xgrammar_best_threads": best_threads,
            "xgrammar_fill_candidates": fill_timings,
            "masks_equal": True,
        }
        records.extend(
            [
                _gpu_record(
                    common,
                    "gpu_lr1_plan",
                    gpu_plan_timing,
                    gpu_plan.strategy,
                ),
                _gpu_record(
                    common,
                    "gpu_lr1_cuda_graph",
                    gpu_graph_timing,
                    gpu_plan.strategy,
                ),
                _wall_record(
                    common,
                    "xgrammar_mask_fill_cpu",
                    fill_timing,
                ),
                _gpu_record(
                    common,
                    "xgrammar_fused_gpu_only",
                    x_fused_gpu_timing,
                    "gpu_bitset_argmax",
                ),
                _gpu_record(
                    common,
                    "xgrammar_fused_gpu_cuda_graph",
                    x_fused_graph_timing,
                    "gpu_bitset_argmax_graph",
                ),
                _wall_record(
                    common,
                    "xgrammar_fused_serial_no_accept",
                    x_fused_serial_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_fused_graph_serial_no_accept",
                    x_fused_graph_serial_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_copy_graph_serial_no_accept",
                    x_copy_graph_serial_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_fused_serial_accept_rollback",
                    x_fused_stateful_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_fused_graph_accept_rollback",
                    x_fused_graph_stateful_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_copy_graph_accept_rollback",
                    x_copy_graph_stateful_timing,
                ),
                _wall_record(
                    common,
                    "xgrammar_native_apply_argmax_no_accept",
                    x_native_serial_timing,
                ),
            ]
        )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "xgrammar_threads": [
                str(value) for value in thread_candidates
            ],
            "logits": (
                "identical full-width Qwen3 FP16 buffers with a shared "
                "grammar-allowed winner"
            ),
            "measurement_boundary": (
                "isolated constraint selection; model execution excluded"
            ),
        },
        "language": {
            "arithmetic_parenthesis_depth": 2,
            "balanced_nesting_depth": 3,
            "whitespace": "not allowed",
        },
        "gpu_lr1_compile_seconds": ours_compile_seconds,
        "xgrammar_compile_seconds": xgrammar_compile_seconds,
        "xgrammar_cache_size_bytes": xgrammar_cache_size_bytes,
        "gpu_lr1_tables": {
            "states": tables.num_states,
            "edges": int(tables.csr_indices.size),
            "runtime_bytes": tables.memory_bytes(),
        },
        "mask_checks": mask_checks,
        "benchmarks": records,
        "comparisons": _build_comparisons(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)


def _grammar_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "bounded-arithmetic",
            "grammar": bounded_byte_arithmetic_grammar(2),
            "ebnf": bounded_arithmetic_ebnf(2),
            "byte_terminals": {
                ord(symbol): symbol for symbol in "0123456789+*()"
            },
            "stack_depth": 32,
        },
        {
            "name": "bounded-balanced",
            "grammar": bounded_balanced_grammar(3),
            "ebnf": bounded_balanced_ebnf(3),
            "byte_terminals": {ord("("): "(", ord(")"): ")"},
            "stack_depth": 16,
        },
    ]


def _select_fair_states(
    automata,
    state_offsets: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    rng = np.random.default_rng(seed + batch_size * 43)
    states = np.empty(batch_size, dtype=np.int32)
    grammar_ids = []
    local_states = []
    for index in range(batch_size):
        grammar_id = index % len(automata)
        automaton = automata[grammar_id]
        candidates = np.flatnonzero(automaton.row_nnz > 0)
        local_state = int(rng.choice(candidates))
        states[index] = int(state_offsets[grammar_id]) + local_state
        grammar_ids.append(grammar_id)
        local_states.append(local_state)
    return states, grammar_ids, local_states


def _expected_bitmask(tables, states: np.ndarray) -> np.ndarray:
    words = (tables.vocab_size + 31) // 32
    result = np.zeros((states.size, words), dtype=np.uint32)
    for batch_id, state in enumerate(states):
        start = int(tables.csr_indptr[state])
        end = int(tables.csr_indptr[state + 1])
        tokens = tables.csr_indices[start:end]
        np.bitwise_or.at(
            result[batch_id],
            tokens // 32,
            np.left_shift(
                np.uint32(1),
                (tokens & 31).astype(np.uint32),
            ),
        )
    return result


def _mask_unused_bits(mask: np.ndarray, vocab_size: int) -> None:
    remainder = vocab_size & 31
    if remainder:
        mask[:, -1] &= np.uint32((1 << remainder) - 1)


def _gpu_record(
    common: dict[str, object],
    strategy: str,
    timing: Timing,
    launch_strategy: str,
) -> dict[str, object]:
    return {
        **common,
        "strategy": strategy,
        "launch_strategy": launch_strategy,
        **asdict(timing),
        "sequences_per_second": float(
            common["batch_size"] / (timing.cuda_mean_us * 1e-6)
        ),
    }


def _wall_record(
    common: dict[str, object],
    strategy: str,
    timing: dict[str, float],
) -> dict[str, object]:
    return {
        **common,
        "strategy": strategy,
        "wall_p50_us": timing["p50_us"],
        "wall_p95_us": timing["p95_us"],
        "wall_mean_us": timing["mean_us"],
        "cuda_mean_us": None,
        "sequences_per_second": float(
            common["batch_size"] / (timing["mean_us"] * 1e-6)
        ),
    }


def _measure_cpu(
    function: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return _summarize(samples)


def _measure_cuda_wall(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return _summarize(samples)


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "p50_us": float(statistics.median(ordered)),
        "p95_us": float(ordered[p95_index]),
        "mean_us": float(statistics.fmean(ordered)),
    }


def _print_summary(payload: dict[str, object]) -> None:
    print(
        "batch,strategy,threads,allowed_mean,cuda_us,wall_p50_us,"
        "sequences_per_second"
    )
    for item in payload["benchmarks"]:
        cuda = item["cuda_mean_us"]
        cuda_text = "" if cuda is None else f"{cuda:.3f}"
        print(
            f"{item['batch_size']},{item['strategy']},"
            f"{item['xgrammar_best_threads']},"
            f"{item['allowed_tokens_mean']:.2f},{cuda_text},"
            f"{item['wall_p50_us']:.3f},"
            f"{item['sequences_per_second']:.1f}"
        )


def _build_comparisons(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    batch_sizes = sorted({int(item["batch_size"]) for item in records})
    for batch_size in batch_sizes:
        by_strategy = {
            item["strategy"]: item
            for item in records
            if item["batch_size"] == batch_size
        }
        gpu_plan = by_strategy["gpu_lr1_plan"]
        gpu_graph = by_strategy["gpu_lr1_cuda_graph"]
        x_optimistic = by_strategy[
            "xgrammar_copy_graph_serial_no_accept"
        ]
        x_stateful = by_strategy[
            "xgrammar_copy_graph_accept_rollback"
        ]
        x_gpu_graph = by_strategy["xgrammar_fused_gpu_cuda_graph"]
        output.append(
            {
                "batch_size": batch_size,
                "gpu_plan_vs_xgrammar_optimistic_wall": (
                    x_optimistic["wall_p50_us"]
                    / gpu_plan["wall_p50_us"]
                ),
                "gpu_graph_vs_xgrammar_optimistic_wall": (
                    x_optimistic["wall_p50_us"]
                    / gpu_graph["wall_p50_us"]
                ),
                "gpu_graph_vs_xgrammar_stateful_wall": (
                    x_stateful["wall_p50_us"]
                    / gpu_graph["wall_p50_us"]
                ),
                "gpu_graph_vs_xgrammar_gpu_graph_cuda": (
                    x_gpu_graph["cuda_mean_us"]
                    / gpu_graph["cuda_mean_us"]
                ),
            }
        )
    return output


if __name__ == "__main__":
    main()
