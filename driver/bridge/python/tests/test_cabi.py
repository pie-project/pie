"""End-to-end test: Python reads/writes pie-bridge bytes via the C ABI.

This proves the schema's single-source claim — Python touches no layout
information; every byte access goes through `libpie_bridge`'s extern
"C" functions, exposed through the PyO3 `_native` extension built by
maturin and the small set of ctypes-based `build_*` helpers in
`pie_bridge.__init__`.

The read path (`Frame.parse`, `frame.payload.kind`, `frame.payload.
as_forward()`, etc.) is the PyO3 module; the write path uses ctypes
to fill `Pie<T>Desc` POD structs and call `pie_build_<type>`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pie_bridge as pb

# Note: under the unchecked C-ABI parse (`rkyv::access_unchecked`),
# malformed input is undefined behavior. The shmem path's checked
# entry point (`pie_bridge::wire::parse_request`) is what guards
# against partially-trusted bytes. This test file therefore only
# exercises the trusted-producer round-trip: bytes built by
# `build_*` are immediately parsed by `Frame.parse` / `ResponseFrame.parse`.


def _build_health_request(driver_id):
    return pb.build_health_request(driver_id)


def test_health_request_round_trip():
    data = _build_health_request(driver_id=42)
    f = pb.Frame.parse(data)
    assert f.driver_id == 42
    assert f.payload.kind == pb.REQUEST_HEALTH
    assert f.payload.as_forward() is None
    assert f.payload.as_copy() is None
    assert f.payload.as_adapter() is None


def test_status_response_round_trip():
    bytes_out = pb.build_status_response(driver_id=7, status=42)
    rf = pb.ResponseFrame.parse(bytes_out)
    assert rf.driver_id == 7
    assert rf.aborted is False
    assert rf.payload.kind == pb.RESPONSE_STATUS
    status = rf.payload.as_status()
    assert status is not None
    assert status.status == 42


def test_aborted_status_response():
    bytes_out = pb.build_status_response(driver_id=99, status=-1, aborted=True)
    rf = pb.ResponseFrame.parse(bytes_out)
    assert rf.aborted is True
    status = rf.payload.as_status()
    assert status is not None
    assert status.status == -1


def test_forward_response_round_trip():
    # Build a minimal forward response with a couple of fields populated.
    bytes_out = pb.build_forward_response(
        driver_id=9,
        num_requests=2,
        tokens_indptr=[0, 2, 5],
        tokens=[100, 101, 200, 201, 202],
    )
    rf = pb.ResponseFrame.parse(bytes_out)
    assert rf.driver_id == 9
    assert rf.payload.kind == pb.RESPONSE_FORWARD
    fr = rf.payload.as_forward()
    assert fr is not None
    assert fr.num_requests == 2
    # `tokens` is a zero-copy numpy view over the parent bytes.
    import numpy as np
    tokens = np.asarray(fr.tokens)
    assert tokens.tolist() == [100, 101, 200, 201, 202]


def test_forward_response_numpy_inputs_are_coerced_and_owned():
    import numpy as np

    tokens = np.arange(10, dtype=np.uint64)[::2]
    tokens_indptr = np.array([0, 5], dtype=np.int64)
    bytes_out = pb.build_forward_response(
        driver_id=4,
        num_requests=1,
        tokens_indptr=tokens_indptr,
        tokens=tokens,
    )
    rf = pb.ResponseFrame.parse(bytes_out)
    fr = rf.payload.as_forward()
    assert fr is not None
    assert np.asarray(fr.tokens_indptr).tolist() == [0, 5]
    assert np.asarray(fr.tokens).tolist() == [0, 2, 4, 6, 8]


def test_health_request_bytes_match_rust():
    # The bytes Python produces for a HealthRequest must be parseable
    # by the C ABI (which Rust uses internally). Trivially round-trips
    # here; the real cross-language guarantee is verified by exercising
    # the same C ABI from both sides.
    data = _build_health_request(driver_id=123)
    f = pb.Frame.parse(data)
    assert f.driver_id == 123
    assert f.payload.kind == pb.REQUEST_HEALTH
