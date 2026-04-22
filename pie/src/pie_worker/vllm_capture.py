"""Capture vLLM requests and responses for debugging continuous batching.

Enabled by PIE_VLLM_CAPTURE=1 (or a path to output dir).
Writes JSONL files that can be diffed between c=1 (non-continuous, correct)
and c>1 (continuous batching, possibly garbled) to find where they diverge.

Each line captures one prepare_step + execute_step pair:
  - batch_id, step_in_fire_batch, timestamp
  - Per-request: req_id, classification (NEW/CONTINUING/FINISHED),
    token_ids (input), block_ids, num_computed_tokens, seq_len
  - Per-request output: sampled_token_ids
  - SequenceTracker state snapshot (active_requests count, token_history keys)

Usage:
  PIE_VLLM_CAPTURE=1 pie http ...        # writes to /tmp/pie-vllm-capture/
  PIE_VLLM_CAPTURE=/path/to/dir pie http ...  # writes to specified dir
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


class VllmCapture:
    """Captures vLLM SchedulerOutput inputs and ModelRunnerOutput results."""

    def __init__(self) -> None:
        raw = os.environ.get("PIE_VLLM_CAPTURE", "")
        if not raw or raw == "0":
            self._enabled = False
            self._fd = None
            return

        self._enabled = True
        if raw == "1":
            capture_dir = Path("/tmp/pie-vllm-capture")
        else:
            capture_dir = Path(raw)
        capture_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        pid = os.getpid()
        path = capture_dir / f"capture-{ts}-{pid}.jsonl"
        self._fd = open(path, "w")
        print(f"[CAPTURE] Writing to {path}", file=sys.stderr, flush=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def capture_step(
        self,
        *,
        batch_id: int,
        step_index: int,
        scheduler_output: Any,
        tracker: Any,
        arrays: Any,
        is_merged: bool = False,
        merged_req_count: int = 0,
    ) -> None:
        """Capture pre-execution state (SchedulerOutput + tracker snapshot)."""
        if not self._enabled:
            return
        try:
            record = self._build_input_record(
                batch_id=batch_id,
                step_index=step_index,
                scheduler_output=scheduler_output,
                tracker=tracker,
                arrays=arrays,
                is_merged=is_merged,
                merged_req_count=merged_req_count,
            )
            self._fd.write(json.dumps(record, default=str) + "\n")
            self._fd.flush()
        except Exception as e:
            print(f"[CAPTURE] Error capturing step: {e}",
                  file=sys.stderr, flush=True)

    def capture_output(
        self,
        *,
        batch_id: int,
        step_index: int,
        results: list[dict],
        model_output: Any = None,
        batch_req_ids: list[str] | None = None,
    ) -> None:
        """Capture post-execution output (sampled tokens)."""
        if not self._enabled:
            return
        try:
            record = {
                "type": "output",
                "ts_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                "batch_id": batch_id,
                "step_index": step_index,
                "results": [
                    {
                        "tokens": r.get("tokens", []),
                        "dists_len": len(r.get("dists", [])),
                    }
                    for r in results
                ],
            }
            if batch_req_ids:
                record["batch_req_ids"] = batch_req_ids
            # Per-request token mapping from vLLM's model output
            if model_output is not None:
                if hasattr(model_output, "sampled_token_ids"):
                    st = model_output.sampled_token_ids
                    if hasattr(st, "tolist"):
                        record["raw_sampled_token_ids"] = st.tolist()
                if hasattr(model_output, "req_id_to_index"):
                    record["req_id_to_index"] = dict(model_output.req_id_to_index)
            self._fd.write(json.dumps(record, default=str) + "\n")
            self._fd.flush()
        except Exception as e:
            print(f"[CAPTURE] Error capturing output: {e}",
                  file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._fd:
            self._fd.close()
            self._fd = None

    # -- internal ----------------------------------------------------------

    def _build_input_record(
        self,
        *,
        batch_id: int,
        step_index: int,
        scheduler_output: Any,
        tracker: Any,
        arrays: Any,
        is_merged: bool,
        merged_req_count: int,
    ) -> dict:
        record: dict[str, Any] = {
            "type": "input",
            "ts_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC),
            "batch_id": batch_id,
            "step_index": step_index,
            "is_merged": is_merged,
            "merged_req_count": merged_req_count,
        }

        # Rust-provided identity (correlate with WASM instances)
        if hasattr(arrays, "request_ids") and arrays.request_ids:
            record["request_ids"] = arrays.request_ids
        if hasattr(arrays, "is_new") and arrays.is_new:
            record["is_new"] = arrays.is_new

        # New requests
        new_reqs = []
        for nr in scheduler_output.scheduled_new_reqs:
            ptids = nr.prompt_token_ids
            all_block_ids = (
                [list(g) for g in nr.block_ids]
                if isinstance(nr.block_ids, tuple)
                else [list(nr.block_ids)]
            )
            new_reqs.append({
                "req_id": nr.req_id,
                "prompt_token_ids_len": len(ptids),
                "prompt_first_8": ptids[:8],
                "prompt_last_8": ptids[-8:],
                "num_computed_tokens": nr.num_computed_tokens,
                "block_ids": all_block_ids,
                "block_ids_len": sum(len(g) for g in all_block_ids),
            })
        record["new_reqs"] = new_reqs

        # Cached (continuing) requests
        cached = scheduler_output.scheduled_cached_reqs
        if hasattr(cached, "req_ids") and cached.req_ids:
            cached_reqs = []
            for idx, rid in enumerate(cached.req_ids):
                entry: dict[str, Any] = {"req_id": rid}
                if hasattr(cached, "num_computed_tokens") and idx < len(cached.num_computed_tokens):
                    entry["num_computed_tokens"] = cached.num_computed_tokens[idx]
                if hasattr(cached, "num_output_tokens") and idx < len(cached.num_output_tokens):
                    entry["num_output_tokens"] = cached.num_output_tokens[idx]
                if hasattr(cached, "new_block_ids") and idx < len(cached.new_block_ids):
                    nbi = cached.new_block_ids[idx]
                    entry["new_block_ids"] = [list(g) for g in nbi] if nbi else None
                if hasattr(cached, "all_token_ids") and rid in cached.all_token_ids:
                    tids = cached.all_token_ids[rid]
                    entry["all_token_ids_len"] = len(tids)
                    entry["all_token_ids_first_8"] = tids[:8]
                    entry["all_token_ids_last_8"] = tids[-8:]
                cached_reqs.append(entry)
            record["cached_reqs"] = cached_reqs
        else:
            record["cached_reqs"] = []

        # Finished
        record["finished_req_ids"] = list(scheduler_output.finished_req_ids)

        # Num scheduled tokens
        record["num_scheduled_tokens"] = dict(scheduler_output.num_scheduled_tokens)

        # Arrays summary
        record["arrays"] = {
            "num_requests": arrays.num_requests,
            "tokens_per_req": arrays.tokens_per_req,
            "seq_lens": arrays.seq_lens.tolist() if hasattr(arrays.seq_lens, "tolist") else list(arrays.seq_lens),
            "blocks_per_req": [list(b) for b in arrays.blocks_per_req],
        }

        # Tracker state snapshot
        record["tracker"] = {
            "active_count": len(tracker.active_requests),
            "active_keys": sorted(list(tracker.active_requests.keys()))[:20],
            "history_count": len(tracker.token_history),
            "issued_count": len(tracker.all_issued_req_ids),
            "freed_block_ids_pending": len(tracker._freed_block_ids),
        }

        return record


# Singleton — created on first import if env var is set
_instance: VllmCapture | None = None


def get_capture() -> VllmCapture:
    global _instance
    if _instance is None:
        _instance = VllmCapture()
    return _instance
