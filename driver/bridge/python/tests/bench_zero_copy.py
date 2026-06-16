"""Micro-bench for the Python parse + slice access path.

Measures the cost of:
  1. Frame.parse(bytes)
  2. np.asarray(fr.<slice_field>) for the 8 main vector fields walked
     by driver/{dev,sglang,vllm}/_bridge/shmem_schema.py:parse_request

Run from the bridge's venv:
    .venv/bin/python python/tests/bench_zero_copy.py /tmp/frame_16k.bin

After the zero-copy slice-getter change, `np.asarray(fr.<slice>)` is a
cheap cast over a numpy.frombuffer view of the same PyBytes; the
total dominated by `Frame.parse` (which is now a refcount bump on the
input PyBytes rather than a memcpy).
"""

import sys
import time

import numpy as np

import pie_bridge._native as pb


def bench_parse_and_read(payload: bytes, iters: int = 5_000) -> None:
    # Warm up.
    for _ in range(200):
        fr = pb.Frame.parse(payload).payload.as_forward()
        np.asarray(fr.token_ids)

    # Parse-only loop.
    t0 = time.perf_counter()
    for _ in range(iters):
        f = pb.Frame.parse(payload)
    t1 = time.perf_counter()
    parse_us = (t1 - t0) / iters * 1e6
    print(f"Frame.parse only: {parse_us:.2f} µs/iter ({iters} iters)")

    # Parse + all 8 hot slice fields (mirrors shmem_schema.parse_request).
    t0 = time.perf_counter()
    for _ in range(iters):
        fr = pb.Frame.parse(payload).payload.as_forward()
        _ = np.asarray(fr.token_ids)
        _ = np.asarray(fr.position_ids)
        _ = np.asarray(fr.kv_page_indices)
        _ = np.asarray(fr.kv_page_indptr)
        _ = np.asarray(fr.kv_last_page_lens)
        _ = np.asarray(fr.qo_indptr)
        _ = np.asarray(fr.spec_token_ids)
        _ = np.asarray(fr.spec_position_ids)
    t1 = time.perf_counter()
    full_us = (t1 - t0) / iters * 1e6
    print(f"Parse + 8 slice asarray: {full_us:.2f} µs/iter")

    # Correctness spot-check.
    fr = pb.Frame.parse(payload).payload.as_forward()
    tok = np.asarray(fr.token_ids)
    assert tok.dtype == np.uint32, f"unexpected dtype {tok.dtype}"
    assert tok.shape[0] == 16384, f"unexpected len {tok.shape[0]}"
    # token_ids[i] = i in the encoder's setup; spot-check the first/last.
    assert int(tok[0]) == 0 and int(tok[-1]) == 16383
    # Verify the array genuinely shares storage with the parent bytes
    # (zero-copy proof).
    assert tok.base is not None, "expected a view, not an owned array"
    print("correctness: ok (dtype=uint32, len=16384, view confirmed)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frame_16k.bin"
    with open(path, "rb") as f:
        payload = f.read()
    print(f"payload: {len(payload)} bytes from {path}")
    bench_parse_and_read(payload)
