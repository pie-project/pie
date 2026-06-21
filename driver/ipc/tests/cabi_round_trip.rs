//! In-proc descriptor round-trip (Python-free protocol layer, #13).
//!
//! The C-ABI rkyv accessors (`pie_parse_*`/`pie_build_*`/readers) are gone —
//! C++ reads the `repr(C) Pie<T>Desc` directly. This exercises the surviving
//! surface: `ToDesc` (`T::as_desc`, the zero-copy view the runtime hands to
//! C++), `FromDesc` (`T::from_desc`, the response path C++→runtime), and the
//! rkyv wire (`encode_request`/`parse_request`, the out-of-proc Rust↔Rust path).

use pie_ipc::wire::{encode_request, parse_request};
use pie_ipc::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, ForwardRequest,
    Frame, PIE_RESPONSE_PAYLOAD_STATUS, PieForwardResponseDesc, PieResponseFrameDesc,
    PieResponsePayloadDesc, RequestPayload, ResponseFrame, ResponsePayload, Sampler, StatusResponse,
};

// ---------------------------------------------------------------------------
// ToDesc → FromDesc round-trip (the in-proc Rust↔C++ pivot)
// ---------------------------------------------------------------------------

#[test]
fn forward_request_desc_round_trip() {
    let mut req = ForwardRequest {
        token_ids: vec![10, 20, 30],
        position_ids: vec![0, 1, 2],
        sampler_indptr: vec![0, 1],
        adapter_bindings: vec![AdapterBinding {
            adapter_id: 99,
            seed: -1,
        }],
        ..Default::default()
    };
    req.set_samplers(&[Sampler::TopK {
        temperature: 0.5,
        k: 40,
    }]);
    let frame = Frame {
        driver_id: 7,
        payload: RequestPayload::Forward(req),
    };

    // ToDesc: the view's Desc aliases `frame`'s heap (zero copy).
    let view = frame.as_desc();
    assert_eq!(view.desc.driver_id, 7);
    let fwd = &view.desc.payload.forward;
    assert_eq!(fwd.token_ids_len, 3);
    let toks = unsafe { std::slice::from_raw_parts(fwd.token_ids_ptr, fwd.token_ids_len) };
    assert_eq!(toks, &[10u32, 20, 30]);
    // adapter_bindings is a flat-POD slice (no nested Desc).
    assert_eq!(fwd.adapter_bindings_len, 1);
    let binds =
        unsafe { std::slice::from_raw_parts(fwd.adapter_bindings_ptr, fwd.adapter_bindings_len) };
    assert_eq!(binds[0].adapter_id, 99);
    assert_eq!(binds[0].seed, -1);

    // FromDesc: rebuild the native value from the Desc and check it matches.
    let back = Frame::from_desc(&view.desc);
    assert_eq!(back.driver_id, 7);
    match back.payload {
        RequestPayload::Forward(fr) => {
            assert_eq!(fr.token_ids, vec![10, 20, 30]);
            assert_eq!(fr.position_ids, vec![0, 1, 2]);
            assert_eq!(
                fr.adapter_bindings,
                vec![AdapterBinding {
                    adapter_id: 99,
                    seed: -1
                }]
            );
            match fr.sampler_at(0) {
                Some(Sampler::TopK { k, .. }) => assert_eq!(k, 40),
                other => panic!("expected TopK, got {other:?}"),
            }
        }
        other => panic!("expected Forward, got {other:?}"),
    }
}

#[test]
fn copy_request_desc_pod_enums_by_value() {
    let frame = Frame {
        driver_id: 3,
        payload: RequestPayload::Copy(CopyRequest {
            dir: CopyDir::D2H,
            srcs: vec![1, 2, 3],
            dsts: vec![10, 20, 30],
            resource: CopyResource::Kv,
        }),
    };
    let view = frame.as_desc();
    // Flat-POD enums are embedded in the Desc by value.
    assert_eq!(view.desc.payload.copy.dir, CopyDir::D2H);
    assert_eq!(view.desc.payload.copy.resource, CopyResource::Kv);

    let back = Frame::from_desc(&view.desc);
    match back.payload {
        RequestPayload::Copy(cr) => {
            assert_eq!(cr.dir, CopyDir::D2H);
            assert_eq!(cr.resource, CopyResource::Kv);
            assert_eq!(cr.srcs, vec![1, 2, 3]);
            assert_eq!(cr.dsts, vec![10, 20, 30]);
        }
        other => panic!("expected Copy, got {other:?}"),
    }
}

#[test]
fn adapter_request_desc_round_trip() {
    let frame = Frame {
        driver_id: 4,
        payload: RequestPayload::Adapter(AdapterRequest {
            op: AdapterOp::Load,
            adapter_id: 0xCAFE,
            path: "/tmp/x.bin".to_string(),
        }),
    };
    let view = frame.as_desc();
    assert_eq!(view.desc.payload.adapter.op, AdapterOp::Load);

    let back = Frame::from_desc(&view.desc);
    match back.payload {
        RequestPayload::Adapter(ar) => {
            assert_eq!(ar.op, AdapterOp::Load);
            assert_eq!(ar.adapter_id, 0xCAFE);
            assert_eq!(ar.path, "/tmp/x.bin");
        }
        other => panic!("expected Adapter, got {other:?}"),
    }
}

#[test]
fn status_response_from_desc() {
    // The response path: C++ fills a `PieResponseFrameDesc`, the runtime
    // reconstructs the native `ResponseFrame` via `from_desc`.
    let desc = PieResponseFrameDesc {
        driver_id: 7,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_STATUS,
            forward: PieForwardResponseDesc::default(),
            status: StatusResponse { status: 42 },
        },
    };
    let rf = ResponseFrame::from_desc(&desc);
    assert_eq!(rf.driver_id, 7);
    assert!(!rf.aborted);
    match rf.payload {
        ResponsePayload::Status(s) => assert_eq!(s.status, 42),
        other => panic!("expected Status, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// rkyv wire (out-of-proc Rust↔Rust)
// ---------------------------------------------------------------------------

#[test]
fn frame_rkyv_wire_round_trip() {
    let frame = Frame {
        driver_id: 42,
        payload: RequestPayload::Health,
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = parse_request(&bytes).unwrap();
    assert_eq!(archived.driver_id.to_native(), 42);
}
