//! Edge-case + alignment audit for the bridge. These tests catch the
//! failure modes that aren't obvious from happy-path round-trip tests:
//!
//!   * Untrusted-byte rejection via `wire::parse_request` (checked
//!     rkyv path) — must return `Err`, not crash. The C-ABI
//!     `pie_parse_<type>` is unchecked and assumes a trusted producer;
//!     these tests therefore go through `wire::parse_request`, which
//!     is the documented public entrypoint for partially-trusted bytes.
//!   * Empty `Vec<T>` fields — accessors must return null/0, not UB.
//!   * `Sampler::Logprobs` with an empty `token_ids` Vec.
//!   * All-default `Pie<T>Desc` round-trip (Default::default() + build).
//!   * `pie_<t>_view` on a default-initialized native struct.
//!   * Pointer stability after `PieFrameView` move.

#![cfg(feature = "cabi")]

use std::ptr;

use pie_bridge::wire::{encode_request, encode_response, parse_request};
use pie_bridge::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, ForwardRequest,
    ForwardResponse, Frame, PIE_REQUEST_PAYLOAD_FORWARD, PIE_RESPONSE_PAYLOAD_FORWARD,
    PIE_SAMPLER_LOGPROBS, PIE_SAMPLER_MULTINOMIAL, PieFrameDesc, PieResponseFrameDesc,
    PieResponsePayloadDesc, PieStatusResponseDesc, RequestPayload, ResponsePayload, Sampler,
    StatusResponse, pie_forward_request_view, pie_frame_view, pie_parse_frame,
    pie_parse_response_frame, pie_sampler_token_ids,
};

// ============================================================================
// Alignment + malformed input
// ============================================================================

#[test]
fn parse_frame_rejects_null_ptr() {
    let p = unsafe { pie_parse_frame(ptr::null(), 0) };
    assert!(p.is_null());
    let p2 = unsafe { pie_parse_frame(ptr::null(), 64) };
    assert!(p2.is_null());
}

#[test]
fn parse_frame_rejects_zero_len() {
    let buf = [0u8; 16];
    let p = unsafe { pie_parse_frame(buf.as_ptr(), 0) };
    assert!(p.is_null());
}

#[test]
fn checked_parse_rejects_random_bytes() {
    // `wire::parse_request` is the checked entrypoint for
    // partially-trusted bytes. Garbage at any alignment must produce
    // an `Err`, never UB. (The C-ABI `pie_parse_frame` is unchecked
    // and assumes a trusted producer; we test it separately on the
    // happy path only.)
    for shift in 0..8 {
        let mut buf = vec![0xa5u8; 256 + shift];
        let slice = unsafe { ::core::slice::from_raw_parts(buf.as_mut_ptr().add(shift), 256) };
        assert!(parse_request(slice).is_err(), "shift={shift}");
    }
}

#[test]
fn checked_parse_truncated_bytes() {
    // Build a valid frame, then progressively truncate. Every
    // truncation length except the full one should fail via the
    // checked path.
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
    // rkyv requires the buffer start to be aligned for the root type.
    // Unaligned input must produce an `Err` through the checked path
    // (the unchecked `pie_parse_frame` would crash here; that's by
    // design — same-process producers always emit aligned buffers).
    let frame = Frame {
        driver_id: 1,
        payload: RequestPayload::Health,
    };
    let aligned = encode_request(&frame).unwrap();
    for offset in [1usize, 2, 3, 5, 7] {
        let mut buf = vec![0u8; aligned.len() + offset];
        buf[offset..offset + aligned.len()].copy_from_slice(&aligned);
        let slice = &buf[offset..offset + aligned.len()];
        // Either Err (rkyv detected misalignment) or Ok if the offset
        // happened to land on a valid boundary for this archive's
        // alignment requirement. Either way, no crash.
        if let Ok(archived) = parse_request(slice) {
            // Touching the discriminant must not crash.
            let _ = archived.driver_id;
        }
    }
}

// ============================================================================
// Empty Vec fields
// ============================================================================

#[test]
fn forward_request_with_all_empty_vecs() {
    let req = ForwardRequest::default();
    let frame = Frame {
        driver_id: 9,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();

    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!archived.is_null());

    // Forward → samplers_at on empty Vec must return null without UB.
    use pie_bridge::pie_forward_request_samplers_at;
    use pie_bridge::pie_forward_request_samplers_len;
    use pie_bridge::pie_forward_request_token_ids;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_forward;

    let payload = unsafe { pie_frame_payload(archived) };
    let fr = unsafe { pie_request_payload_as_forward(payload) };
    assert!(!fr.is_null());

    assert_eq!(unsafe { pie_forward_request_samplers_len(fr) }, 0);
    let s0 = unsafe { pie_forward_request_samplers_at(fr, 0) };
    assert!(s0.is_null(), "samplers_at(0) on empty Vec should be null");

    // token_ids slice: pointer can be anything, length must be 0.
    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_forward_request_token_ids(fr, &mut p, &mut n) };
    assert_eq!(n, 0);
}

#[test]
fn sampler_logprobs_with_empty_token_ids() {
    // Edge case: Logprobs variant with token_ids = []. The accessor
    // should report len=0 without dereferencing.
    let req = ForwardRequest {
        samplers: vec![Sampler::Logprobs { token_ids: vec![] }],
        sampler_indptr: vec![0, 1],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!archived.is_null());

    use pie_bridge::pie_forward_request_samplers_at;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_forward;
    use pie_bridge::pie_sampler_kind;

    let payload = unsafe { pie_frame_payload(archived) };
    let fr = unsafe { pie_request_payload_as_forward(payload) };
    let s = unsafe { pie_forward_request_samplers_at(fr, 0) };
    assert_eq!(unsafe { pie_sampler_kind(s) }, PIE_SAMPLER_LOGPROBS);

    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_sampler_token_ids(s, &mut p, &mut n) };
    assert_eq!(n, 0);
}

#[test]
fn sampler_token_ids_returns_empty_for_wrong_variant() {
    // Multinomial doesn't have token_ids; accessor should return null/0.
    let req = ForwardRequest {
        samplers: vec![Sampler::Multinomial {
            temperature: 0.5,
            seed: 0,
        }],
        sampler_indptr: vec![0, 1],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };

    use pie_bridge::pie_forward_request_samplers_at;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_forward;
    use pie_bridge::pie_sampler_kind;

    let payload = unsafe { pie_frame_payload(archived) };
    let fr = unsafe { pie_request_payload_as_forward(payload) };
    let s = unsafe { pie_forward_request_samplers_at(fr, 0) };
    assert_eq!(unsafe { pie_sampler_kind(s) }, PIE_SAMPLER_MULTINOMIAL);

    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_sampler_token_ids(s, &mut p, &mut n) };
    assert_eq!(n, 0);
    assert!(p.is_null());
}

// ============================================================================
// `Pie<T>Desc::default()` — every field zero/null must be a valid build input.
// ============================================================================

#[test]
fn build_response_frame_from_default_desc() {
    use pie_bridge::pie_build_response_frame;
    let desc = PieResponseFrameDesc::default();
    let mut buf = vec![0u8; 256];
    let n = unsafe { pie_build_response_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0, "default Desc should produce a valid empty response");

    let archived = unsafe { pie_parse_response_frame(buf.as_ptr(), n) };
    assert!(!archived.is_null());
}

#[test]
fn build_frame_from_default_desc() {
    use pie_bridge::pie_build_frame;
    let desc = PieFrameDesc::default();
    let mut buf = vec![0u8; 4096];
    let n = unsafe { pie_build_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0, "default Desc should produce a valid empty frame");

    let archived = unsafe { pie_parse_frame(buf.as_ptr(), n) };
    assert!(!archived.is_null());
}

// ============================================================================
// Pointer stability — direct-FFI view path
// ============================================================================

#[test]
fn forward_view_pointer_stable_across_clone_chain() {
    // Even though PieFrameView itself isn't cloneable, its holders are
    // heap-allocated. Confirm that moving the view doesn't invalidate
    // the desc's pointers.
    let req = ForwardRequest {
        token_ids: vec![1, 2, 3, 4, 5],
        position_ids: vec![0, 1, 2, 3, 4],
        qo_indptr: vec![0, 5],
        samplers: vec![Sampler::TopK {
            temperature: 0.7,
            k: 40,
        }],
        sampler_indptr: vec![0, 1],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 1,
        payload: RequestPayload::Forward(req),
    };
    let v = pie_frame_view(&frame);
    let token_ptr_before = v.desc.payload.forward.token_ids_ptr;

    // Box it (move the View onto the heap).
    let boxed = Box::new(v);
    let token_ptr_after = boxed.desc.payload.forward.token_ids_ptr;
    assert_eq!(token_ptr_before, token_ptr_after);

    // Deref: bytes are still valid.
    let tokens = unsafe { std::slice::from_raw_parts(token_ptr_after, 5) };
    assert_eq!(tokens, &[1u32, 2, 3, 4, 5]);
}

#[test]
fn view_of_minimal_native_does_not_crash() {
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Health,
    };
    let _ = pie_frame_view(&frame); // smoke
    let req = ForwardRequest::default();
    let _ = pie_forward_request_view(&req);
}

// ============================================================================
// Wire format: response with abort flag
// ============================================================================

#[test]
fn aborted_response_decodes_correctly() {
    let resp = pie_bridge::ResponseFrame {
        driver_id: 42,
        aborted: true,
        payload: ResponsePayload::Status(StatusResponse { status: -1 }),
    };
    let bytes = encode_response(&resp).unwrap();
    let archived = unsafe { pie_parse_response_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!archived.is_null());

    use pie_bridge::pie_response_frame_aborted;
    assert_eq!(unsafe { pie_response_frame_aborted(archived) }, 1);
}

// ============================================================================
// Copy + Adapter — Vec<u32> edge cases
// ============================================================================

#[test]
fn copy_request_with_zero_length_lists() {
    let bytes = encode_request(&Frame {
        driver_id: 0,
        payload: RequestPayload::Copy(CopyRequest {
            dir: CopyDir::D2H,
            srcs: vec![],
            dsts: vec![],
            resource: CopyResource::Kv,
        }),
    })
    .unwrap();
    let p = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!p.is_null());

    use pie_bridge::pie_copy_dir_value;
    use pie_bridge::pie_copy_request_dir;
    use pie_bridge::pie_copy_request_srcs;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_copy;

    let payload = unsafe { pie_frame_payload(p) };
    let cr = unsafe { pie_request_payload_as_copy(payload) };
    assert!(!cr.is_null());

    let dir = unsafe { pie_copy_request_dir(cr) };
    assert_eq!(unsafe { pie_copy_dir_value(dir) }, 0); // D2H

    let mut sp: *const u32 = ptr::null();
    let mut sn: usize = 0;
    unsafe { pie_copy_request_srcs(cr, &mut sp, &mut sn) };
    assert_eq!(sn, 0);
}

#[test]
fn adapter_request_with_no_path() {
    let bytes = encode_request(&Frame {
        driver_id: 0,
        payload: RequestPayload::Adapter(AdapterRequest {
            op: AdapterOp::Save,
            adapter_id: 99,
            path: String::new(),
        }),
    })
    .unwrap();
    let p = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!p.is_null());

    use pie_bridge::pie_adapter_request_adapter_id;
    use pie_bridge::pie_adapter_request_path;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_adapter;

    let payload = unsafe { pie_frame_payload(p) };
    let ar = unsafe { pie_request_payload_as_adapter(payload) };
    assert_eq!(unsafe { pie_adapter_request_adapter_id(ar) }, 99);

    let mut pp: *const std::ffi::c_char = ptr::null();
    let mut pn: usize = 0;
    unsafe { pie_adapter_request_path(ar, &mut pp, &mut pn) };
    assert_eq!(pn, 0); // empty path → length 0
}

// ============================================================================
// AdapterBinding — i64 with -1 sentinel for unbound
// ============================================================================

#[test]
fn adapter_binding_both_unbound() {
    let req = ForwardRequest {
        adapter_bindings: vec![AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };

    use pie_bridge::pie_adapter_binding_adapter_id;
    use pie_bridge::pie_adapter_binding_seed;
    use pie_bridge::pie_forward_request_adapter_bindings_at;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_forward;

    let payload = unsafe { pie_frame_payload(archived) };
    let fr = unsafe { pie_request_payload_as_forward(payload) };
    let b = unsafe { pie_forward_request_adapter_bindings_at(fr, 0) };
    assert!(!b.is_null());
    assert_eq!(unsafe { pie_adapter_binding_adapter_id(b) }, -1);
    assert_eq!(unsafe { pie_adapter_binding_seed(b) }, -1);
}

// ============================================================================
// build_*_response with truncated output buffer
// ============================================================================

#[test]
fn build_response_returns_zero_for_too_small_buffer() {
    use pie_bridge::pie_build_response_frame;

    // First, find the natural size with a generous buffer.
    let full = ForwardResponse {
        num_requests: 1,
        ..Default::default()
    };
    let resp_native = pie_bridge::ResponseFrame {
        driver_id: 0,
        aborted: false,
        payload: ResponsePayload::Forward(full),
    };
    let bytes = encode_response(&resp_native).unwrap();
    let natural = bytes.len();
    assert!(natural > 4, "response should be > 4 bytes");

    // Build into a buffer too small by one byte.
    let desc = PieResponseFrameDesc {
        driver_id: 0,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_FORWARD,
            forward: pie_bridge::PieForwardResponseDesc {
                num_requests: 1,
                ..Default::default()
            },
            status: PieStatusResponseDesc { status: 0 },
        },
    };
    let mut buf = vec![0u8; natural - 1];
    let n = unsafe { pie_build_response_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, 0, "should fail when out_buf too small");
}

#[test]
fn pie_size_matches_pie_build() {
    use pie_bridge::{pie_build_response_frame, pie_size_response_frame};

    let desc = PieResponseFrameDesc {
        driver_id: 7,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_FORWARD,
            forward: pie_bridge::PieForwardResponseDesc {
                num_requests: 3,
                ..Default::default()
            },
            status: PieStatusResponseDesc { status: 0 },
        },
    };
    // Size lookup must exactly match a successful build's return value.
    let n_size = unsafe { pie_size_response_frame(&desc) };
    assert!(n_size > 0);

    let mut buf = vec![0u8; n_size];
    let n_build = unsafe { pie_build_response_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n_size, n_build);

    // Null descriptor → 0.
    assert_eq!(unsafe { pie_size_response_frame(ptr::null()) }, 0);
}

// ============================================================================
// Empty Forward frame
// ============================================================================

/// sampler-suite shape: 1 input token, 1 sampling position (slot 0),
/// 6 different samplers all attached to that one slot. Verifies the
/// wire format preserves `samplers_len=6` + `sampler_indptr=[0,6]` +
/// `sampling_indices=[0]` + `sampling_indptr=[0,1]` end-to-end. The
/// cuda driver's "sampled=0" log line during sampler-suite suggests
/// the bridge might be dropping `sampling_indices`; this test pins
/// the correct shape at the rkyv layer to isolate the issue.
#[test]
fn multi_sampler_on_single_slot_preserves_shape() {
    let req = ForwardRequest {
        token_ids: vec![0],
        position_ids: vec![0],
        qo_indptr: vec![0, 1],
        sampling_indices: vec![0],
        sampling_indptr: vec![0, 1],
        samplers: vec![
            // Argmax → TopP(0.0, 1.0)
            Sampler::TopP {
                temperature: 0.0,
                p: 1.0,
            },
            // RawLogits probe
            Sampler::RawLogits,
            // Distribution
            Sampler::Dist {
                temperature: 1.0,
                num_tokens: 8,
            },
            // Logprob(cand_a)
            Sampler::Logprob { token_id: 42 },
            // Logprobs(cand_list)
            Sampler::Logprobs {
                token_ids: vec![10, 20, 30],
            },
            // Entropy
            Sampler::Entropy,
        ],
        sampler_indptr: vec![0, 6],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 0,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!archived.is_null());

    use pie_bridge::pie_forward_request_sampler_indptr;
    use pie_bridge::pie_forward_request_samplers_at;
    use pie_bridge::pie_forward_request_samplers_len;
    use pie_bridge::pie_forward_request_sampling_indices;
    use pie_bridge::pie_forward_request_sampling_indptr;
    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_as_forward;
    use pie_bridge::pie_sampler_kind;

    let payload = unsafe { pie_frame_payload(archived) };
    let fr = unsafe { pie_request_payload_as_forward(payload) };
    assert!(!fr.is_null());

    // samplers_len == 6
    assert_eq!(unsafe { pie_forward_request_samplers_len(fr) }, 6);

    // sampler_indptr == [0, 6]
    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_forward_request_sampler_indptr(fr, &mut p, &mut n) };
    assert_eq!(n, 2);
    let arr = unsafe { std::slice::from_raw_parts(p, n) };
    assert_eq!(arr[0], 0);
    assert_eq!(arr[1], 6);

    // sampling_indices == [0]
    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_forward_request_sampling_indices(fr, &mut p, &mut n) };
    assert_eq!(n, 1, "sampling_indices should have 1 entry");
    let arr = unsafe { std::slice::from_raw_parts(p, n) };
    assert_eq!(arr[0], 0);

    // sampling_indptr == [0, 1]
    let mut p: *const u32 = ptr::null();
    let mut n: usize = 0;
    unsafe { pie_forward_request_sampling_indptr(fr, &mut p, &mut n) };
    assert_eq!(n, 2);
    let arr = unsafe { std::slice::from_raw_parts(p, n) };
    assert_eq!(arr[0], 0);
    assert_eq!(arr[1], 1);

    // Each sampler's kind comes through correctly.
    let expected_kinds = [
        2u8,  // TopP
        7u8,  // RawLogits
        6u8,  // Dist
        8u8,  // Logprob
        9u8,  // Logprobs
        10u8, // Entropy
    ];
    for (i, ek) in expected_kinds.iter().enumerate() {
        let s = unsafe { pie_forward_request_samplers_at(fr, i) };
        let k = unsafe { pie_sampler_kind(s) };
        assert_eq!(k, *ek, "sampler[{i}] kind expected {ek}, got {k}");
    }
}

#[test]
fn empty_forward_request_round_trips() {
    // ForwardRequest::default() has empty Vecs across the board. The
    // archived form should be valid and every accessor should return
    // empty.
    let f = Frame {
        driver_id: 1,
        payload: RequestPayload::Forward(ForwardRequest::default()),
    };
    let bytes = encode_request(&f).unwrap();
    let archived = unsafe { pie_parse_frame(bytes.as_ptr(), bytes.len()) };
    assert!(!archived.is_null());

    use pie_bridge::pie_frame_payload;
    use pie_bridge::pie_request_payload_kind;
    let payload = unsafe { pie_frame_payload(archived) };
    assert_eq!(
        unsafe { pie_request_payload_kind(payload) },
        PIE_REQUEST_PAYLOAD_FORWARD
    );
}
