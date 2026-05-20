//! Round-trip via the C ABI surface: build a request in Rust, encode
//! to bytes, then exercise the C-ABI accessors as a C++/Python caller
//! would. Every `pie_*` symbol used here is emitted by `#[schema]` on
//! the corresponding type — no hand-written C-ABI accessors exist.

#![cfg(feature = "cabi")]

use std::ptr;

use pie_bridge::wire::encode_request;
use pie_bridge::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, ForwardRequest,
    Frame, PIE_ADAPTER_OP_LOAD, PIE_COPY_DIR_D2H, PIE_REQUEST_PAYLOAD_FORWARD,
    PIE_REQUEST_PAYLOAD_HEALTH, PIE_RESPONSE_PAYLOAD_FORWARD, PIE_RESPONSE_PAYLOAD_STATUS,
    PIE_SAMPLER_LOGPROBS, PIE_SAMPLER_MULTINOMIAL, PIE_SAMPLER_TOP_K, PieAdapterBindingDesc,
    PieAdapterRequestDesc, PieCopyRequestDesc, PieForwardRequestDesc, PieForwardResponseDesc,
    PieFrameDesc, PieRequestPayloadDesc, PieResponseFrameDesc, PieResponsePayloadDesc,
    PieSamplerDesc, PieStatusResponseDesc, RequestPayload, Sampler, pie_adapter_binding_adapter_id,
    pie_adapter_binding_seed, pie_adapter_op_value, pie_adapter_request_adapter_id,
    pie_adapter_request_op, pie_adapter_request_path, pie_build_frame, pie_build_response_frame,
    pie_copy_dir_value, pie_copy_request_dir, pie_copy_request_dsts, pie_copy_request_resource,
    pie_copy_request_srcs, pie_copy_resource_value, pie_forward_request_adapter_bindings_at,
    pie_forward_request_adapter_bindings_len, pie_forward_request_context_ids,
    pie_forward_request_has_user_mask, pie_forward_request_position_ids,
    pie_forward_request_samplers_at, pie_forward_request_samplers_len,
    pie_forward_request_single_token_mode, pie_forward_request_token_ids, pie_frame_driver_id,
    pie_frame_payload, pie_parse_frame, pie_parse_response_frame, pie_request_payload_as_adapter,
    pie_request_payload_as_copy, pie_request_payload_as_forward, pie_request_payload_kind,
    pie_response_frame_aborted, pie_response_frame_driver_id, pie_response_frame_payload,
    pie_response_payload_as_forward, pie_response_payload_as_status, pie_response_payload_kind,
    pie_sampler_k, pie_sampler_kind, pie_sampler_seed, pie_sampler_temperature,
    pie_sampler_token_ids, pie_status_response_status,
};

#[test]
fn frame_health_round_trip_through_cabi() {
    let bytes = encode_request(&Frame {
        driver_id: 42,
        payload: RequestPayload::Health,
    })
    .unwrap();

    unsafe {
        let frame = pie_parse_frame(bytes.as_ptr(), bytes.len());
        assert!(!frame.is_null());
        assert_eq!(pie_frame_driver_id(frame), 42);

        let payload = pie_frame_payload(frame);
        assert_eq!(
            pie_request_payload_kind(payload),
            PIE_REQUEST_PAYLOAD_HEALTH
        );
        assert!(pie_request_payload_as_forward(payload).is_null());
        assert!(pie_request_payload_as_copy(payload).is_null());
        assert!(pie_request_payload_as_adapter(payload).is_null());
    }
}

#[test]
fn forward_frame_through_cabi() {
    let req = ForwardRequest {
        token_ids: vec![10, 20, 30, 40, 50],
        position_ids: vec![0, 1, 2, 3, 4],
        context_ids: vec![0xCAFE, 0xBABE],
        single_token_mode: false,
        has_user_mask: true,
        samplers: vec![
            Sampler::Multinomial {
                temperature: 0.7,
                seed: 42,
            },
            Sampler::TopK {
                temperature: 0.5,
                k: 40,
            },
            Sampler::Logprobs {
                token_ids: vec![100, 200, 300],
            },
        ],
        sampler_indptr: vec![0, 3],
        adapter_bindings: vec![AdapterBinding {
            adapter_id: 99,
            seed: -1,
        }],
        ..Default::default()
    };
    let bytes = encode_request(&Frame {
        driver_id: 7,
        payload: RequestPayload::Forward(req),
    })
    .unwrap();

    unsafe {
        let frame = pie_parse_frame(bytes.as_ptr(), bytes.len());
        assert!(!frame.is_null());
        assert_eq!(pie_frame_driver_id(frame), 7);

        let payload = pie_frame_payload(frame);
        assert_eq!(
            pie_request_payload_kind(payload),
            PIE_REQUEST_PAYLOAD_FORWARD
        );

        let fr = pie_request_payload_as_forward(payload);
        assert!(!fr.is_null());

        let mut p: *const u32 = ptr::null();
        let mut n: usize = 0;
        pie_forward_request_token_ids(fr, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[10u32, 20, 30, 40, 50]);

        let mut p: *const u32 = ptr::null();
        let mut n: usize = 0;
        pie_forward_request_position_ids(fr, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[0u32, 1, 2, 3, 4]);

        let mut p: *const u64 = ptr::null();
        let mut n: usize = 0;
        pie_forward_request_context_ids(fr, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[0xCAFEu64, 0xBABE]);

        assert_eq!(pie_forward_request_single_token_mode(fr), 0);
        assert_eq!(pie_forward_request_has_user_mask(fr), 1);

        assert_eq!(pie_forward_request_samplers_len(fr), 3);

        let s0 = pie_forward_request_samplers_at(fr, 0);
        assert_eq!(pie_sampler_kind(s0), PIE_SAMPLER_MULTINOMIAL);
        assert!((pie_sampler_temperature(s0) - 0.7).abs() < 1e-6);
        assert_eq!(pie_sampler_seed(s0), 42);

        let s1 = pie_forward_request_samplers_at(fr, 1);
        assert_eq!(pie_sampler_kind(s1), PIE_SAMPLER_TOP_K);
        assert_eq!(pie_sampler_k(s1), 40);

        let s2 = pie_forward_request_samplers_at(fr, 2);
        assert_eq!(pie_sampler_kind(s2), PIE_SAMPLER_LOGPROBS);
        let mut p: *const u32 = ptr::null();
        let mut n: usize = 0;
        pie_sampler_token_ids(s2, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[100u32, 200, 300]);

        assert_eq!(pie_forward_request_adapter_bindings_len(fr), 1);
        let b = pie_forward_request_adapter_bindings_at(fr, 0);
        assert!(!b.is_null());
        assert_eq!(pie_adapter_binding_adapter_id(b), 99);
        assert_eq!(pie_adapter_binding_seed(b), -1);
    }
}

#[test]
fn copy_frame_through_cabi() {
    let bytes = encode_request(&Frame {
        driver_id: 3,
        payload: RequestPayload::Copy(CopyRequest {
            dir: CopyDir::D2H,
            srcs: vec![1, 2, 3],
            dsts: vec![10, 20, 30],
            resource: CopyResource::Kv,
        }),
    })
    .unwrap();

    unsafe {
        let frame = pie_parse_frame(bytes.as_ptr(), bytes.len());
        let payload = pie_frame_payload(frame);
        let cr = pie_request_payload_as_copy(payload);
        assert!(!cr.is_null());

        let dir = pie_copy_request_dir(cr);
        assert_eq!(pie_copy_dir_value(dir), PIE_COPY_DIR_D2H);
        let resource = pie_copy_request_resource(cr);
        assert_eq!(pie_copy_resource_value(resource), 0);

        let mut p: *const u32 = ptr::null();
        let mut n: usize = 0;
        pie_copy_request_srcs(cr, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[1u32, 2, 3]);
        pie_copy_request_dsts(cr, &mut p, &mut n);
        assert_eq!(std::slice::from_raw_parts(p, n), &[10u32, 20, 30]);
    }
}

#[test]
fn adapter_frame_through_cabi() {
    let bytes = encode_request(&Frame {
        driver_id: 4,
        payload: RequestPayload::Adapter(AdapterRequest {
            op: AdapterOp::Load,
            adapter_id: 0xCAFE,
            path: "/tmp/x.bin".to_string(),
        }),
    })
    .unwrap();

    unsafe {
        let frame = pie_parse_frame(bytes.as_ptr(), bytes.len());
        let payload = pie_frame_payload(frame);
        let ar = pie_request_payload_as_adapter(payload);
        assert!(!ar.is_null());

        let op = pie_adapter_request_op(ar);
        assert_eq!(pie_adapter_op_value(op), PIE_ADAPTER_OP_LOAD);
        assert_eq!(pie_adapter_request_adapter_id(ar), 0xCAFE);

        let mut p: *const std::ffi::c_char = ptr::null();
        let mut n: usize = 0;
        pie_adapter_request_path(ar, &mut p, &mut n);
        let path = std::slice::from_raw_parts(p as *const u8, n);
        assert_eq!(std::str::from_utf8(path).unwrap(), "/tmp/x.bin");
    }
}

/// Helper: a zeroed `PieForwardResponseDesc` (all Vec fields empty,
/// `num_requests = 0`). Tests fill in just the fields they care about.
fn empty_forward_response_desc() -> PieForwardResponseDesc {
    PieForwardResponseDesc {
        num_requests: 0,
        tokens_indptr_ptr: ptr::null(),
        tokens_indptr_len: 0,
        tokens_ptr: ptr::null(),
        tokens_len: 0,
        dists_req_indptr_ptr: ptr::null(),
        dists_req_indptr_len: 0,
        dists_kv_indptr_ptr: ptr::null(),
        dists_kv_indptr_len: 0,
        dists_ids_ptr: ptr::null(),
        dists_ids_len: 0,
        dists_probs_ptr: ptr::null(),
        dists_probs_len: 0,
        logits_req_indptr_ptr: ptr::null(),
        logits_req_indptr_len: 0,
        logits_byte_indptr_ptr: ptr::null(),
        logits_byte_indptr_len: 0,
        logits_bytes_ptr: ptr::null(),
        logits_bytes_len: 0,
        logprobs_req_indptr_ptr: ptr::null(),
        logprobs_req_indptr_len: 0,
        logprobs_val_indptr_ptr: ptr::null(),
        logprobs_val_indptr_len: 0,
        logprobs_values_ptr: ptr::null(),
        logprobs_values_len: 0,
        entropies_indptr_ptr: ptr::null(),
        entropies_indptr_len: 0,
        entropies_ptr: ptr::null(),
        entropies_len: 0,
    }
}

#[test]
fn build_status_response_round_trip() {
    let desc = PieResponseFrameDesc {
        driver_id: 7,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_STATUS,
            forward: empty_forward_response_desc(),
            status: PieStatusResponseDesc { status: 42 },
        },
    };

    let mut buf = vec![0u8; 256];
    let n = unsafe { pie_build_response_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);

    unsafe {
        let rf = pie_parse_response_frame(buf.as_ptr(), n);
        assert!(!rf.is_null());
        assert_eq!(pie_response_frame_driver_id(rf), 7);
        assert_eq!(pie_response_frame_aborted(rf), 0);

        let payload = pie_response_frame_payload(rf);
        assert_eq!(
            pie_response_payload_kind(payload),
            PIE_RESPONSE_PAYLOAD_STATUS
        );

        let sr = pie_response_payload_as_status(payload);
        assert!(!sr.is_null());
        assert_eq!(pie_status_response_status(sr), 42);
        assert!(pie_response_payload_as_forward(payload).is_null());
    }
}

#[test]
fn build_forward_response_round_trip() {
    let tokens_indptr = [0u32, 2, 5];
    let tokens = [100u32, 101, 200, 201, 202];
    let mut fwd = empty_forward_response_desc();
    fwd.num_requests = 2;
    fwd.tokens_indptr_ptr = tokens_indptr.as_ptr();
    fwd.tokens_indptr_len = tokens_indptr.len();
    fwd.tokens_ptr = tokens.as_ptr();
    fwd.tokens_len = tokens.len();

    let desc = PieResponseFrameDesc {
        driver_id: 9,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_FORWARD,
            forward: fwd,
            status: PieStatusResponseDesc { status: 0 },
        },
    };

    let mut buf = vec![0u8; 4096];
    let n = unsafe { pie_build_response_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);

    unsafe {
        let rf = pie_parse_response_frame(buf.as_ptr(), n);
        assert!(!rf.is_null());
        assert_eq!(pie_response_frame_driver_id(rf), 9);

        let payload = pie_response_frame_payload(rf);
        assert_eq!(
            pie_response_payload_kind(payload),
            PIE_RESPONSE_PAYLOAD_FORWARD
        );
        assert!(!pie_response_payload_as_forward(payload).is_null());
    }
}

#[test]
fn build_health_request_round_trip() {
    // PieRequestPayloadDesc::kind=HEALTH; embedded sub-descs need to be
    // present (their fields are unused for the Health variant).
    let fr = empty_forward_request_desc();
    let cr = PieCopyRequestDesc {
        dir: 0,
        srcs_ptr: ptr::null(),
        srcs_len: 0,
        dsts_ptr: ptr::null(),
        dsts_len: 0,
        resource: 0,
    };
    let ar = PieAdapterRequestDesc {
        op: 0,
        adapter_id: 0,
        path_ptr: ptr::null(),
        path_len: 0,
    };
    let desc = PieFrameDesc {
        driver_id: 99,
        payload: PieRequestPayloadDesc {
            kind: PIE_REQUEST_PAYLOAD_HEALTH,
            forward: fr,
            copy: cr,
            adapter: ar,
        },
    };

    let mut buf = vec![0u8; 256];
    let n = unsafe { pie_build_frame(&desc, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);

    unsafe {
        let frame = pie_parse_frame(buf.as_ptr(), n);
        assert!(!frame.is_null());
        assert_eq!(pie_frame_driver_id(frame), 99);
        let payload = pie_frame_payload(frame);
        assert_eq!(
            pie_request_payload_kind(payload),
            PIE_REQUEST_PAYLOAD_HEALTH
        );
    }
}

fn empty_forward_request_desc() -> PieForwardRequestDesc {
    PieForwardRequestDesc {
        token_ids_ptr: ptr::null(),
        token_ids_len: 0,
        position_ids_ptr: ptr::null(),
        position_ids_len: 0,
        kv_page_indices_ptr: ptr::null(),
        kv_page_indices_len: 0,
        kv_page_indptr_ptr: ptr::null(),
        kv_page_indptr_len: 0,
        kv_last_page_lens_ptr: ptr::null(),
        kv_last_page_lens_len: 0,
        qo_indptr_ptr: ptr::null(),
        qo_indptr_len: 0,
        rs_slot_ids_ptr: ptr::null(),
        rs_slot_ids_len: 0,
        rs_slot_flags_ptr: ptr::null(),
        rs_slot_flags_len: 0,
        masks_ptr: ptr::null(),
        masks_len: 0,
        mask_indptr_ptr: ptr::null(),
        mask_indptr_len: 0,
        logit_masks_ptr: ptr::null(),
        logit_masks_len: 0,
        logit_mask_indptr_ptr: ptr::null(),
        logit_mask_indptr_len: 0,
        sampling_indices_ptr: ptr::null(),
        sampling_indices_len: 0,
        sampling_indptr_ptr: ptr::null(),
        sampling_indptr_len: 0,
        samplers_ptr: ptr::null(),
        samplers_len: 0,
        sampler_indptr_ptr: ptr::null(),
        sampler_indptr_len: 0,
        adapter_bindings_ptr: ptr::null(),
        adapter_bindings_len: 0,
        spec_token_ids_ptr: ptr::null(),
        spec_token_ids_len: 0,
        spec_position_ids_ptr: ptr::null(),
        spec_position_ids_len: 0,
        spec_indptr_ptr: ptr::null(),
        spec_indptr_len: 0,
        output_spec_flags_ptr: ptr::null(),
        output_spec_flags_len: 0,
        context_ids_ptr: ptr::null(),
        context_ids_len: 0,
        single_token_mode: 0,
        has_user_mask: 0,
    }
}

// silence unused-import lint until a test exercises these.
#[allow(dead_code)]
fn _unused() {
    let _ = PieSamplerDesc {
        kind: 0,
        temperature: 0.0,
        seed: 0,
        k: 0,
        p: 0.0,
        num_tokens: 0,
        token_id: 0,
        token_ids_ptr: ptr::null(),
        token_ids_len: 0,
    };
    let _ = PieAdapterBindingDesc {
        adapter_id: -1,
        seed: -1,
    };
}
