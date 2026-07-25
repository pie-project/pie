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


NARROW_PREFIX = '{"a": 123'
WIDE_PREFIX = '{"a": "hello wor'
WIDE_SHARE = 0.514


@dataclass
class Rows:
    indptr: np.ndarray
    indices: np.ndarray
    next_state: np.ndarray
    widths: list[int] = field(default_factory=list)


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


def select_rows(profile: str, batch: int, seed: int) -> np.ndarray:
    if profile == "narrow":
        return np.zeros(batch, dtype=np.int32)
    if profile == "wide":
        return np.ones(batch, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return (rng.random(batch) < WIDE_SHARE).astype(np.int32)


def measure(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return float(statistics.median(samples))


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
        choices=("narrow", "wide", "mixed"),
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
    rows, vocab_size = build_rows(args.qwen_model)
    print(
        f"vocab={vocab_size} narrow_row={rows.widths[0]} wide_row={rows.widths[1]} "
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
            print(
                f"wide complement bitsets: {tuple(tables.bitset.shape)} "
                f"({tables.bitset.numel() * 4 / 1024:.1f} KiB total)"
            )

    sampling = _load_flashinfer()
    xgr = _load_xgrammar()
    records: list[dict] = []

    print()
    header = (
        f"{'profile':>8} {'batch':>6} {'graph':>10} {'eager':>10} "
        f"{'fi_unconstr':>13} {'xgr_gpu+fi':>12} {'xgr_full+fi':>13} "
        f"{'vs_unconstr':>12}"
    )
    print(header)
    for profile in args.profiles:
        for batch in args.batch_sizes:
            selected = select_rows(profile, batch, args.seed + batch)
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
                        xgr, args.qwen_model, selected, vocab_size
                    )
                    device_mask = mask.to(device)
                    pinned = mask.pin_memory()

                    def masked() -> None:
                        work = logits.clone()
                        xgr.apply_token_bitmask_inplace(work, device_mask)
                        sampling.top_k_top_p_sampling_from_logits(
                            work, fi_k, fi_p
                        )

                    xgrammar_gpu = measure(
                        masked,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )

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

                    xgrammar_full = measure(
                        full_path,
                        warmup=max(1, args.warmup // 2),
                        iterations=max(5, args.iterations // 2),
                    )
                    record["xgrammar_gpu_mask_us"] = xgrammar_gpu
                    record["xgrammar_full_path_us"] = xgrammar_full

            speedup = baseline / fused_us if baseline else float("nan")
            record["speedup_vs_unconstrained"] = speedup
            records.append(record)
            print(
                f"{profile:>8} {batch:6d} {fused_us:9.1f}u {eager_us:9.1f}u "
                f"{_fmt(baseline):>13} {_fmt(xgrammar_gpu):>12} "
                f"{_fmt(xgrammar_full):>13} {speedup:11.1f}x"
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


def _xgrammar_state(xgr, model: str, selected: np.ndarray, vocab_size: int):
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
        matcher.accept_string(WIDE_PREFIX if row else NARROW_PREFIX)
        matchers.append(matcher)
    batch_matcher = xgr.BatchGrammarMatcher(max_threads="auto")
    mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
    batch_matcher.batch_fill_next_token_bitmask(matchers, mask)
    return mask, matchers, batch_matcher


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
