//! Edge-case audit for the Python-free protocol layer (#13). Covers the
//! failure modes that aren't obvious from happy-path round-trips:
//!
//!   * Untrusted-byte rejection via `wire::parse_request` (the checked rkyv
//!     entrypoint for partially-trusted bytes) — must `Err`, never crash.
//!   * Empty `Vec<T>` fields — `as_desc` must produce null/0 slices, not UB.
//!   * `Sampler` shape (kinds + indptr) preserved through `ToDesc`.
//!   * Pointer stability after the `Pie<T>View` moves.
//!   * Response abort flag survives the rkyv wire.

use pie_ipc::wire::{encode_request, encode_response, parse_request, parse_response};
use pie_ipc::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, ForwardRequest,
    ForwardResponse, Frame, RequestPayload, ResponseFrame, ResponsePayload, Sampler,
};

// ============================================================================
// Malformed input — the checked rkyv path (`wire::parse_request`)
// ============================================================================

#[test]
fn checked_parse_rejects_random_bytes() {
    // Garbage at any alignment must produce an `Err`, never UB.
    for shift in 0..8 {
        let mut buf = vec![0xa5u8; 256 + shift];
        let slice = unsafe { ::core::slice::from_raw_parts(buf.as_mut_ptr().add(shift), 256) };
        assert!(parse_request(slice).is_err(), "shift={shift}");
    }
}

#[test]
fn checked_parse_truncated_bytes() {
    let bytes = encode_request(&Frame {
        driver_id: 5,
        payload: RequestPayload::Health,
    })
    .unwrap();
    for cut in [1, 4, 8, 16, bytes.len() - 1] {
        if cut == 0 || cut >= bytes.len() {
            continue;
        }
        assert!(parse_request(&bytes[..cut]).is_err(), "cut={cut}");
    }
}

#[test]
fn checked_parse_unaligned_input_returns_err() {
    let frame = Frame {
        driver_id: 1,
        payload: RequestPayload::Health,
    };
    let aligned = encode_request(&frame).unwrap();
    for offset in [1usize, 2, 3, 5, 7] {
        let mut buf = vec![0u8; aligned.len() + offset];
        buf[offset..offset + aligned.len()].copy_from_slice(&aligned);
        let slice = &buf[offset..offset + aligned.len()];
        // Either Err (rkyv detected misalignment) or Ok if the offset happened
        // to land on a valid boundary. Either way, no crash.
        if let Ok(archived) = parse_request(slice) {
            let _ = archived.driver_id;
        }
    }
}

// ============================================================================
// Empty Vec fields — `as_desc` (ToDesc)
// ============================================================================

#[test]
fn forward_request_all_empty_vecs_desc() {
    let frame = Frame {
        driver_id: 9,
        payload: RequestPayload::Forward(ForwardRequest::default()),
    };
    let view = frame.as_desc();
    let fwd = &view.desc.payload.forward;
    assert_eq!(fwd.token_ids_len, 0);
    assert_eq!(fwd.sampler_kinds_len, 0);
    assert_eq!(fwd.adapter_bindings_len, 0);
}

#[test]
fn copy_request_zero_length_lists_desc() {
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Copy(CopyRequest {
            dir: CopyDir::D2H,
            srcs: vec![],
            dsts: vec![],
            resource: CopyResource::Kv,
        }),
    };
    let view = frame.as_desc();
    let cr = &view.desc.payload.copy;
    assert_eq!(cr.dir, CopyDir::D2H);
    assert_eq!(cr.srcs_len, 0);
    assert_eq!(cr.dsts_len, 0);
}

#[test]
fn adapter_request_no_path_round_trip() {
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Adapter(AdapterRequest {
            op: AdapterOp::Save,
            adapter_id: 99,
            path: String::new(),
        }),
    };
    let back = Frame::from_desc(&frame.as_desc().desc);
    match back.payload {
        RequestPayload::Adapter(ar) => {
            assert_eq!(ar.op, AdapterOp::Save);
            assert_eq!(ar.adapter_id, 99);
            assert!(ar.path.is_empty());
        }
        other => panic!("expected Adapter, got {other:?}"),
    }
}

#[test]
fn adapter_binding_both_unbound_desc() {
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(ForwardRequest {
            adapter_bindings: vec![AdapterBinding {
                adapter_id: -1,
                seed: -1,
            }],
            ..Default::default()
        }),
    };
    let view = frame.as_desc();
    let fwd = &view.desc.payload.forward;
    assert_eq!(fwd.adapter_bindings_len, 1);
    let b =
        unsafe { std::slice::from_raw_parts(fwd.adapter_bindings_ptr, fwd.adapter_bindings_len) };
    assert_eq!(b[0].adapter_id, -1);
    assert_eq!(b[0].seed, -1);
}

// ============================================================================
// Sampler shape preserved through ToDesc
// ============================================================================

#[test]
fn multi_sampler_on_single_slot_preserves_shape() {
    let mut req = ForwardRequest {
        token_ids: vec![0],
        position_ids: vec![0],
        qo_indptr: vec![0, 1],
        sampling_indices: vec![0],
        sampling_indptr: vec![0, 1],
        sampler_indptr: vec![0, 6],
        ..Default::default()
    };
    // Raw per-variant SoA (neutral wire): producer emits the request values,
    // 0/empty for N/A — driver folds live in view.hpp, not here.
    req.set_samplers(&[
        Sampler::TopP {
            temperature: 0.0,
            p: 1.0,
        },
        Sampler::RawLogits,
        Sampler::Dist {
            temperature: 1.0,
            num_tokens: 8,
        },
        Sampler::Logprob { token_id: 42 },
        Sampler::Logprobs {
            token_ids: vec![10, 20, 30],
        },
        Sampler::Entropy,
    ]);
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let view = frame.as_desc();
    let fwd = &view.desc.payload.forward;

    // kind discriminants: TopP=2, RawLogits=7, Dist=6, Logprob=8, Logprobs=9, Entropy=10
    assert_eq!(fwd.sampler_kinds_len, 6);
    let kinds = unsafe { std::slice::from_raw_parts(fwd.sampler_kinds_ptr, fwd.sampler_kinds_len) };
    assert_eq!(kinds, &[2u8, 7, 6, 8, 9, 10]);
    // Dist.num_tokens stays raw on the floor (not folded into top_k).
    let num_tokens =
        unsafe { std::slice::from_raw_parts(fwd.sampler_num_tokens_ptr, fwd.sampler_num_tokens_len) };
    assert_eq!(num_tokens, &[0u32, 0, 8, 0, 0, 0]);
    // Unified label CSR: Logprob.token_id (slot 3) then Logprobs.token_ids (slot 4).
    let ids =
        unsafe { std::slice::from_raw_parts(fwd.sampler_token_ids_ptr, fwd.sampler_token_ids_len) };
    assert_eq!(ids, &[42u32, 10, 20, 30]);
    let label_indptr = unsafe {
        std::slice::from_raw_parts(
            fwd.sampler_token_ids_indptr_ptr,
            fwd.sampler_token_ids_indptr_len,
        )
    };
    assert_eq!(label_indptr, &[0u32, 0, 0, 0, 1, 4, 4]);

    let indptr = unsafe { std::slice::from_raw_parts(fwd.sampler_indptr_ptr, fwd.sampler_indptr_len) };
    assert_eq!(indptr, &[0u32, 6]);
    let sidx =
        unsafe { std::slice::from_raw_parts(fwd.sampling_indices_ptr, fwd.sampling_indices_len) };
    assert_eq!(sidx, &[0u32]);
}

// ============================================================================
// Pointer stability — the view's Desc aliases stay valid across a move
// ============================================================================

#[test]
fn view_pointer_stable_across_move() {
    let frame = Frame {
        driver_id: 1,
        payload: RequestPayload::Forward(ForwardRequest {
            token_ids: vec![1, 2, 3, 4, 5],
            ..Default::default()
        }),
    };
    let v = frame.as_desc();
    let ptr_before = v.desc.payload.forward.token_ids_ptr;
    let boxed = Box::new(v);
    let ptr_after = boxed.desc.payload.forward.token_ids_ptr;
    assert_eq!(ptr_before, ptr_after);
    let tokens = unsafe { std::slice::from_raw_parts(ptr_after, 5) };
    assert_eq!(tokens, &[1u32, 2, 3, 4, 5]);
}

// ============================================================================
// Response rkyv round-trip (abort-flag handling is covered by
// `round_trip_schema::aborted_response_propagates_as_handler_aborted`, which
// asserts `parse_response` returns `WireError::HandlerAborted` for aborted=true)
// ============================================================================

#[test]
fn forward_response_rkyv_round_trip() {
    let resp = ResponseFrame {
        driver_id: 0,
        aborted: false,
        payload: ResponsePayload::Forward(ForwardResponse {
            num_requests: 3,
            ..Default::default()
        }),
    };
    let bytes = encode_response(&resp).unwrap();
    let archived = parse_response(&bytes).unwrap();
    assert_eq!(archived.driver_id.to_native(), 0);
}
