"""Benchmark ragged fused constrained sampling against the deployed path.

Rows come from XGrammar's own builtin JSON grammar over the real Qwen3
vocabulary, so the narrow/wide widths are ground truth rather than synthetic.
The mixture profile matches the measured composition of a realistic JSON
document, where about half of the tokens are emitted from a string body.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gpu_lr1.ragged_sampler import (
    RaggedSamplerTables,
    capture_ragged_sample,
    ragged_sample,
)


TYPED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[A-Za-z ]{1,40}$"},
        "age": {"type": "integer"},
        "role": {"enum": ["admin", "editor", "viewer"]},
        "active": {"type": "boolean"},
        "score": {"type": "number"},
        "tags": {
            "type": "array",
            "items": {"enum": ["alpha", "beta", "gamma"]},
        },
    },
    "required": ["name", "age", "role", "active", "score", "tags"],
    "additionalProperties": False,
}
TYPED_INSTANCE = {
    "name": "Ada Lovelace",
    "age": 36,
    "role": "admin",
    "active": True,
    "score": 9.5,
    "tags": ["alpha", "beta"],
}

NARROW_PREFIX = '{"a": 123'
WIDE_PREFIX = '{"a": "hello wor'
WIDE_SHARE = 0.514


@dataclass
class Rows:
    indptr: np.ndarray
    indices: np.ndarray
    next_state: np.ndarray
    widths: list[int] = field(default_factory=list)


def build_real_rows(path: Path) -> tuple[Rows, int, np.ndarray]:
    """Rebuild rows recorded by replaying a real model's output through a tokenizer.

    Rows arrive in whichever form was smaller, so wide ones are stored as the
    tokens they forbid and are expanded here.
    """
    data = np.load(path)
    indptr = data["indptr"].astype(np.int64)
    payload = data["payload"].astype(np.int32)
    kinds = data["kinds"]
    vocab_size = int(data["vocab_size"][0])
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    _XGRAMMAR_CACHE["row_schema"] = data["row_schema"]
    _XGRAMMAR_CACHE["row_step"] = data["row_step"]
    _XGRAMMAR_CACHE["real_schemas"] = sidecar["schemas"]
    _XGRAMMAR_CACHE["real_tokens"] = sidecar["tokens"]
    _XGRAMMAR_CACHE["real_tokenizer"] = sidecar["tokenizer"]

    rows_list: list[np.ndarray] = []
    for index in range(kinds.size):
        stored = payload[int(indptr[index]) : int(indptr[index + 1])]
        if kinds[index] == 0:
            rows_list.append(stored)
        else:
            allowed = np.ones(vocab_size, dtype=bool)
            allowed[stored.astype(np.int64)] = False
            rows_list.append(np.flatnonzero(allowed).astype(np.int32))
    row_widths = np.asarray([row.size for row in rows_list], dtype=np.int64)
    new_indptr = np.zeros(len(rows_list) + 1, dtype=np.int32)
    for index, row in enumerate(rows_list):
        new_indptr[index + 1] = new_indptr[index] + row.size
    rows = Rows(
        new_indptr,
        np.concatenate(rows_list).astype(np.int32),
        np.zeros(int(new_indptr[-1]), dtype=np.int32),
        row_widths.tolist(),
    )
    return rows, vocab_size, data["widths"], row_widths


def build_schema_rows(model: str) -> tuple[Rows, int, np.ndarray]:
    """Rows taken from every step of a real schema-guided generation.

    The free-form JSON grammar is the worst case for a width-sensitive engine
    and the best case for nobody: a typed schema spends most steps at a
    structural position where only a handful of tokens are legal.
    """
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiled = xgr.GrammarCompiler(info).compile_json_schema(
        json.dumps(TYPED_SCHEMA)
    )
    vocab_size = info.vocab_size
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, vocab_size)

    sets: list[np.ndarray] = []
    for token in tokenizer.encode(
        json.dumps(TYPED_INSTANCE, separators=(",", ":"))
    ):
        matcher.fill_next_token_bitmask(mask)
        bits = np.unpackbits(mask.numpy().view(np.uint8), bitorder="little")
        sets.append(np.flatnonzero(bits[:vocab_size]).astype(np.int32))
        if not matcher.accept_token(token):
            break

    indptr = np.zeros(len(sets) + 1, dtype=np.int32)
    for index, row in enumerate(sets):
        indptr[index + 1] = indptr[index] + row.size
    indices = np.concatenate(sets).astype(np.int32)
    widths = np.asarray([row.size for row in sets])
    _XGRAMMAR_CACHE["schema_compiled"] = compiled
    _XGRAMMAR_CACHE["schema_tokens"] = tokenizer.encode(
        json.dumps(TYPED_INSTANCE, separators=(",", ":"))
    )
    rows = Rows(
        indptr,
        indices,
        np.zeros(indices.size, dtype=np.int32),
        widths.tolist(),
    )
    return rows, vocab_size, widths


def build_rows(model: str) -> tuple[Rows, int]:
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiled = xgr.GrammarCompiler(info).compile_builtin_json_grammar()
    vocab_size = info.vocab_size

    def allowed(prefix: str) -> np.ndarray:
        matcher = xgr.GrammarMatcher(compiled)
        if not matcher.accept_string(prefix):
            raise RuntimeError(f"grammar rejected prefix {prefix!r}")
        mask = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(mask)
        bits = np.unpackbits(mask.numpy().view(np.uint8), bitorder="little")
        return np.flatnonzero(bits[:vocab_size]).astype(np.int32)

    narrow = allowed(NARROW_PREFIX)
    wide = allowed(WIDE_PREFIX)
    indices = np.concatenate([narrow, wide])
    indptr = np.asarray([0, narrow.size, narrow.size + wide.size], np.int32)
    next_state = np.zeros(indices.size, dtype=np.int32)
    return Rows(indptr, indices, next_state, [narrow.size, wide.size]), vocab_size


def select_rows(
    profile: str, batch: int, seed: int, num_rows: int = 2
) -> np.ndarray:
    if profile in ("schema", "real"):
        rng = np.random.default_rng(seed)
        return rng.integers(0, num_rows, size=batch).astype(np.int32)
    if profile == "narrow":
        return np.zeros(batch, dtype=np.int32)
    if profile == "wide":
        return np.ones(batch, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return (rng.random(batch) < WIDE_SHARE).astype(np.int32)


def measure(function, *, warmup: int, iterations: int) -> float:
    return measure_distribution(
        function, warmup=warmup, iterations=iterations
    )["p50"]


def measure_distribution(function, *, warmup: int, iterations: int) -> dict:
    """Percentiles, because a median hides the tail that MaskBench targets."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    ordered = np.sort(np.asarray(samples))
    return {
        "p50": float(np.percentile(ordered, 50)),
        "p90": float(np.percentile(ordered, 90)),
        "p99": float(np.percentile(ordered, 99)),
        "mean": float(ordered.mean()),
        "iterations": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ragged fused constrained sampling"
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512, 2048]
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["narrow", "wide", "mixed"],
        choices=("narrow", "wide", "mixed", "schema", "real"),
    )
    parser.add_argument(
        "--real-rows",
        type=Path,
        default=Path("results/jsonschemabench-rows.npz"),
        help="rows recorded by gpu_lr1.collect_real_rows",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wide-bitsets", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("results/ragged-sampler.json")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.profiles == ["real"]:
        rows, vocab_size, widths, row_widths = build_real_rows(args.real_rows)
        print(
            f"replay of {_XGRAMMAR_CACHE['real_tokenizer']}: "
            f"{len(widths)} recorded steps, vocab {vocab_size}, "
            f"median width {int(np.median(widths))}, "
            f"{(widths > 8192).mean() * 100:.1f}% wide, "
            f"{(widths == 1).mean() * 100:.1f}% forced; "
            f"{len(row_widths)} rows replayed"
        )
    elif args.profiles == ["schema"]:
        rows, vocab_size, widths = build_schema_rows(args.qwen_model)
        print(
            f"typed schema: {len(widths)} steps, median width "
            f"{int(np.median(widths))}, "
            f"{(widths > 8192).mean() * 100:.1f}% wide"
        )
    else:
        rows, vocab_size = build_rows(args.qwen_model)
    if args.profiles not in (["schema"], ["real"]):
        print(
            f"vocab={vocab_size} narrow_row={rows.widths[0]} "
            f"wide_row={rows.widths[1]} "
            f"exceptions={vocab_size - rows.widths[1]}"
        )

    tables = RaggedSamplerTables(
        torch.from_numpy(rows.indptr).to(device),
        torch.from_numpy(rows.indices).to(device),
        torch.from_numpy(rows.next_state).to(device),
    )
    if not args.no_wide_bitsets:
        tables.build_wide_bitsets(vocab_size)
        if tables.bitset is not None:
            tables.drop_wide_token_lists()
            usage = tables.memory_bytes()
            resident = sum(
                value
                for key, value in usage.items()
                if key != "csr_only_equivalent"
            )
            print(
                "table memory: "
                f"{resident / 2**20:.1f} MiB resident vs "
                f"{usage['csr_only_equivalent'] / 2**20:.1f} MiB as plain CSR "
                f"({usage['csr_only_equivalent'] / max(resident, 1):.1f}x smaller); "
                + ", ".join(
                    f"{key}={value / 2**20:.1f}MiB"
                    for key, value in usage.items()
                    if key != "csr_only_equivalent"
                )
            )

    sampling = _load_flashinfer()
    xgr = _load_xgrammar()
    records: list[dict] = []

    print()
    header = (
        f"{'profile':>8} {'batch':>6} {'graph':>10} {'eager':>10} "
        f"{'fi_unconstr':>13} {'xgr_gpu+fi':>12} {'xgr_full+fi':>13} "
        f"{'vs_unconstr':>12} {'xgr_thr':>8}"
    )
    print(header)
    for profile in args.profiles:
        for batch in args.batch_sizes:
            selected = select_rows(
                profile, batch, args.seed + batch, tables.num_rows
            )
            logits = torch.randn(
                batch, vocab_size, dtype=torch.float32, device=device
            )
            row_tensor = torch.from_numpy(selected).to(device)
            temperature = torch.full(
                (batch,), args.temperature, dtype=torch.float32, device=device
            )
            top_k = torch.full(
                (batch,), args.top_k, dtype=torch.int32, device=device
            )
            top_p = torch.full(
                (batch,), args.top_p, dtype=torch.float32, device=device
            )
            uniform = torch.rand(batch, device=device)
            out_tokens = torch.empty(batch, dtype=torch.int32, device=device)
            out_states = torch.empty(batch, dtype=torch.int32, device=device)

            # A serving scheduler already tracks each sequence's grammar
            # state, so whether the batch holds a wide row is known for free.
            wide_present = profile != "narrow"

            def fused() -> None:
                ragged_sample(
                    logits,
                    tables,
                    row_tensor,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    uniform=uniform,
                    out_tokens=out_tokens,
                    out_states=out_states,
                    wide_present=wide_present,
                )

            eager_us = measure(
                fused, warmup=args.warmup, iterations=args.iterations
            )
            captured = capture_ragged_sample(
                logits,
                tables,
                row_tensor,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                uniform=uniform,
                out_tokens=out_tokens,
                out_states=out_states,
                wide_present=wide_present,
            )
            fused_us = measure(
                captured.replay,
                warmup=args.warmup,
                iterations=args.iterations,
            )

            record = {
                "profile": profile,
                "batch_size": batch,
                "gpugrammar_eager_us": eager_us,
                "gpugrammar_ragged_us": fused_us,
            }
            baseline = xgrammar_gpu = xgrammar_full = None
            if sampling is not None:
                fi_k = top_k.clone()
                fi_p = top_p.clone()

                def unconstrained() -> None:
                    sampling.top_k_top_p_sampling_from_logits(
                        logits, fi_k, fi_p
                    )

                baseline = measure(
                    unconstrained,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                record["flashinfer_unconstrained_us"] = baseline

                if xgr is not None:
                    mask, matchers, batch_matcher = _xgrammar_state(
                        xgr, args.qwen_model, selected, vocab_size, profile
                    )
                    device_mask = mask.to(device)
                    pinned = mask.pin_memory()

                    def masked() -> None:
                        work = logits.clone()
                        xgr.apply_token_bitmask_inplace(work, device_mask)
                        sampling.top_k_top_p_sampling_from_logits(
                            work, fi_k, fi_p
                        )

                    masked_graph = _capture(masked, device)
                    xgrammar_gpu = measure(
                        masked_graph or masked,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )

                    # Both engines must pay for advancing their state: our
                    # next-state write is inside the fused kernel, so XGrammar
                    # is charged accept_token plus the rollback that restores
                    # the batch for the next iteration.
                    advance_tokens = [
                        int(t) for t in _first_allowed(mask, vocab_size)
                    ]

                    def full_path() -> None:
                        batch_matcher.batch_fill_next_token_bitmask(
                            matchers, mask
                        )
                        pinned.copy_(mask)
                        moved = pinned.to(device, non_blocking=True)
                        work = logits.clone()
                        xgr.apply_token_bitmask_inplace(work, moved)
                        sampling.top_k_top_p_sampling_from_logits(
                            work, fi_k, fi_p
                        )
                        batch_matcher.batch_accept_token(
                            matchers, advance_tokens
                        )
                        batch_matcher.batch_rollback(matchers, 1)

                    xgrammar_full = measure(
                        full_path,
                        warmup=max(1, args.warmup // 2),
                        iterations=max(5, args.iterations // 2),
                    )
                    record["xgrammar_gpu_mask_us"] = xgrammar_gpu
                    record["xgrammar_full_path_us"] = xgrammar_full
                    record["xgrammar_threads"] = _XGRAMMAR_CACHE.get("threads")
                    record["xgrammar_fill_us"] = _XGRAMMAR_CACHE.get("fill_us")

            speedup = baseline / fused_us if baseline else float("nan")
            record["speedup_vs_unconstrained"] = speedup
            records.append(record)
            print(
                f"{profile:>8} {batch:6d} {fused_us:9.1f}u {eager_us:9.1f}u "
                f"{_fmt(baseline):>13} {_fmt(xgrammar_gpu):>12} "
                f"{_fmt(xgrammar_full):>13} {speedup:11.1f}x "
                f"{str(_XGRAMMAR_CACHE.get('threads')):>8}"
            )

    payload = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "vocab_size": vocab_size,
            "narrow_row_nnz": rows.widths[0],
            "wide_row_nnz": rows.widths[1],
            "wide_share": WIDE_SHARE,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


_XGRAMMAR_CACHE: dict = {}
_XGRAMMAR_THREAD_CANDIDATES = (1, 2, 4, 8, 16, "auto")


def _best_batch_matcher(xgr, matchers, mask, candidates):
    """Pick the fastest XGrammar thread count, as the repo's rules require.

    `max_threads="auto"` is not optimal: on a 24-thread host it ran 2.4-2.9x
    slower than the best fixed count, so using it would understate XGrammar.
    """
    best = None
    best_us = float("inf")
    for threads in candidates:
        matcher = xgr.BatchGrammarMatcher(max_threads=threads)
        for _ in range(2):
            matcher.batch_fill_next_token_bitmask(matchers, mask)
        samples = []
        for _ in range(5):
            started = time.perf_counter_ns()
            matcher.batch_fill_next_token_bitmask(matchers, mask)
            samples.append((time.perf_counter_ns() - started) / 1_000)
        median = statistics.median(samples)
        if median < best_us:
            best_us, best = median, (matcher, threads)
    return best[0], best[1], best_us


def _xgrammar_state(
    xgr, model: str, selected: np.ndarray, vocab_size: int, profile: str
):
    if profile == "real":
        # Rebuild each matcher at the exact schema and step the row came from.
        compiler = _XGRAMMAR_CACHE.get("real_compiler")
        if compiler is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                _XGRAMMAR_CACHE["real_tokenizer"]
            )
            compiler = xgr.GrammarCompiler(
                xgr.TokenizerInfo.from_huggingface(tokenizer),
                cache_enabled=True,
            )
            _XGRAMMAR_CACHE["real_compiler"] = compiler
            _XGRAMMAR_CACHE["real_compiled"] = {}
        cache = _XGRAMMAR_CACHE["real_compiled"]
        schemas = _XGRAMMAR_CACHE["real_schemas"]
        sequences = _XGRAMMAR_CACHE["real_tokens"]
        row_schema = _XGRAMMAR_CACHE["row_schema"]
        row_step = _XGRAMMAR_CACHE["row_step"]
        matchers = []
        for row in selected:
            schema_id = int(row_schema[int(row)])
            if schema_id not in cache:
                cache[schema_id] = compiler.compile_json_schema(
                    schemas[schema_id]
                )
            matcher = xgr.GrammarMatcher(cache[schema_id])
            for token in sequences[schema_id][: int(row_step[int(row)])]:
                matcher.accept_token(token)
            matchers.append(matcher)
        mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
        batch_matcher, threads, fill_us = _best_batch_matcher(
            xgr, matchers, mask, _XGRAMMAR_THREAD_CANDIDATES
        )
        _XGRAMMAR_CACHE["threads"] = threads
        _XGRAMMAR_CACHE["fill_us"] = fill_us
        return mask, matchers, batch_matcher
    if profile == "schema":
        # Put every XGrammar matcher at the same schema position gpugrammar
        # is at, so its adaptive mask cache gets the same opportunity.
        compiled = _XGRAMMAR_CACHE["schema_compiled"]
        tokens = _XGRAMMAR_CACHE["schema_tokens"]
        matchers = []
        for row in selected:
            matcher = xgr.GrammarMatcher(compiled)
            for token in tokens[: int(row)]:
                matcher.accept_token(token)
            matchers.append(matcher)
        mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
        batch_matcher, threads, fill_us = _best_batch_matcher(
            xgr, matchers, mask, _XGRAMMAR_THREAD_CANDIDATES
        )
        _XGRAMMAR_CACHE["threads"] = threads
        _XGRAMMAR_CACHE["fill_us"] = fill_us
        return mask, matchers, batch_matcher
    if "compiled" not in _XGRAMMAR_CACHE:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        _XGRAMMAR_CACHE["compiled"] = xgr.GrammarCompiler(
            info
        ).compile_builtin_json_grammar()
    compiled = _XGRAMMAR_CACHE["compiled"]
    matchers = []
    for row in selected:
        matcher = xgr.GrammarMatcher(compiled)
        matcher.accept_string(WIDE_PREFIX if row % 2 else NARROW_PREFIX)
        matchers.append(matcher)
    mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
    batch_matcher, threads, fill_us = _best_batch_matcher(
        xgr, matchers, mask, _XGRAMMAR_THREAD_CANDIDATES
    )
    _XGRAMMAR_CACHE["threads"] = threads
    _XGRAMMAR_CACHE["fill_us"] = fill_us
    return mask, matchers, batch_matcher


def _first_allowed(mask, vocab_size: int) -> np.ndarray:
    """One legal token per sequence, used only to advance XGrammar's state."""
    bits = np.unpackbits(mask.numpy().view(np.uint8), axis=-1, bitorder="little")
    return bits[:, :vocab_size].argmax(axis=-1)


def _capture(function, device):
    """Capture device work in a CUDA graph, or return None if not capturable."""
    try:
        for _ in range(3):
            function()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        torch.cuda.synchronize(device)
        return graph.replay
    except Exception:  # noqa: BLE001
        return None


def _load_flashinfer():
    try:
        from flashinfer import sampling

        return sampling
    except Exception as error:  # noqa: BLE001
        print(f"flashinfer unavailable, skipping baselines: {error}")
        return None


def _load_xgrammar():
    try:
        import xgrammar

        return xgrammar
    except Exception as error:  # noqa: BLE001
        print(f"xgrammar unavailable, skipping baselines: {error}")
        return None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}u"


if __name__ == "__main__":
    main()
