"""Round-trip the BPIS responses produced by the C++ encoder.

The C++ unit-test target (`test_response_writer`) writes two fixture
files: one BPIS flat-mode response and one msgpack-fallback response.
This test parses them with `msgpack` (the library the runtime uses)
and asserts the structure matches what the encoder claims to emit.

The fixtures themselves are committed-by-build-artifact: we look in
the standard CMake build dir. If they're missing, the test is skipped
so it doesn't block environments without a CUDA build.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

# Layout: pie/tests/pie_driver_cuda_native/test_response_writer.py
#         driver/cuda/build/_response_writer_{flat,msgpack}.bin
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_DIR = _REPO_ROOT / "driver" / "cuda" / "build"

# Header layout: magic, mode, num_requests, total_tokens (16 bytes,
# little-endian). Mirrors `response::HEADER_SIZE` / `MAGIC`.
_HEADER = struct.Struct("<IIII")
_MAGIC = 0x42504953  # 'BPIS'
_MODE_FLAT = 0
_MODE_MSGPACK = 1


def _read(path: Path) -> bytes:
    if not path.exists():
        pytest.skip(f"fixture {path} missing — run "
                    f"`cmake --build driver/cuda/build --target test_response_writer` "
                    f"and `ctest` first")
    return path.read_bytes()


def test_flat_response():
    blob = _read(_BUILD_DIR / "_response_writer_flat.bin")
    magic, mode, n_req, n_tokens = _HEADER.unpack_from(blob, 0)
    assert magic == _MAGIC
    assert mode == _MODE_FLAT
    assert n_req == 3
    assert n_tokens == 3

    counts = list(struct.unpack_from(f"<{n_req}I", blob, _HEADER.size))
    assert counts == [2, 0, 1]

    tokens_off = _HEADER.size + 4 * n_req
    tokens = list(struct.unpack_from(f"<{n_tokens}I", blob, tokens_off))
    assert tokens == [11, 12, 99]


def test_msgpack_response():
    msgpack = pytest.importorskip("msgpack")

    blob = _read(_BUILD_DIR / "_response_writer_msgpack.bin")
    magic, mode, n_req, _ = _HEADER.unpack_from(blob, 0)
    assert magic == _MAGIC
    assert mode == _MODE_MSGPACK
    assert n_req == 3

    body = msgpack.unpackb(blob[_HEADER.size:], raw=False)
    assert isinstance(body, dict)
    assert "results" in body
    results = body["results"]
    assert len(results) == 3

    r0, r1, r2 = results

    # req 0: tokens, dists, logits all populated.
    assert r0["tokens"] == [7]
    assert len(r0["dists"]) == 1
    ids, probs = r0["dists"][0]
    assert ids == [42, 13, 7]
    assert probs == pytest.approx([0.5, 0.3, 0.2])
    assert len(r0["logits"]) == 1
    # 8 f32 little-endian: 1.0, 2.0, 0, 0, 0, 0, 0, 0.
    f32 = struct.unpack("<8f", r0["logits"][0])
    assert f32[0] == pytest.approx(1.0)
    assert f32[1] == pytest.approx(2.0)
    assert all(v == 0.0 for v in f32[2:])
    assert r0["logprobs"] == []
    assert r0["entropies"] == []
    assert r0["spec_tokens"] == []
    assert r0["spec_positions"] == []

    # req 1: logprobs (lengths 1 and 3) + 1 entropy.
    assert r1["tokens"] == []
    assert r1["logprobs"] == [
        pytest.approx([-1.5]),
        pytest.approx([-0.1, -2.2, -3.3]),
    ]
    assert r1["entropies"] == pytest.approx([2.5])
    assert r1["dists"] == []
    assert r1["logits"] == []

    # req 2: every field empty.
    for key in ("tokens", "dists", "logits", "logprobs",
                "entropies", "spec_tokens", "spec_positions"):
        assert r2[key] == [], (key, r2[key])
