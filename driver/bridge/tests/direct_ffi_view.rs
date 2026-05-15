//! Direct-FFI view path: build a native Rust value, take its
//! `Pie<T>View<'_>`, and confirm the resulting `Pie<T>Desc` aliases
//! the source data with no rkyv encode/decode. Pair with
//! `pie_<t>_from_desc` (the build-fn's internal helper) to round-trip
//! back to native — exercising the same conversion the in-process C++
//! handoff would use.

#![cfg(feature = "cabi")]

use std::ptr;

use pie_bridge::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, ForwardRequest, Frame,
    PIE_ADAPTER_OP_LOAD, PIE_COPY_DIR_D2H, PIE_REQUEST_PAYLOAD_FORWARD, PIE_REQUEST_PAYLOAD_HEALTH,
    PIE_SAMPLER_LOGPROBS, PIE_SAMPLER_MULTINOMIAL, PIE_SAMPLER_TOP_K, RequestPayload, Sampler,
    pie_adapter_request_view, pie_copy_request_view, pie_forward_request_view, pie_frame_view,
    pie_request_payload_view, pie_sampler_view,
};

#[test]
fn frame_health_view_no_rkyv() {
    let f = Frame {
        driver_id: 42,
        payload: RequestPayload::Health,
    };
    let view = pie_frame_view(&f);
    assert_eq!(view.desc.driver_id, 42);
    assert_eq!(view.desc.payload.kind, PIE_REQUEST_PAYLOAD_HEALTH);
}

#[test]
fn forward_request_view_aliases_native_slices() {
    // Allocate the data once on the Rust heap.
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
    let view = pie_forward_request_view(&req);

    // Slices alias the native Vec's heap allocation (same data ptr).
    assert_eq!(view.desc.token_ids_ptr, req.token_ids.as_ptr());
    assert_eq!(view.desc.token_ids_len, 5);
    assert_eq!(view.desc.position_ids_ptr, req.position_ids.as_ptr());
    assert_eq!(view.desc.context_ids_ptr, req.context_ids.as_ptr());

    // Bool primitives copied by value.
    assert_eq!(view.desc.single_token_mode, 0);
    assert_eq!(view.desc.has_user_mask, 1);

    // Samplers: a fresh Vec<PieSamplerDesc> lives inside the view.
    assert_eq!(view.desc.samplers_len, 3);
    let samplers = unsafe { std::slice::from_raw_parts(view.desc.samplers_ptr, 3) };
    assert_eq!(samplers[0].kind, PIE_SAMPLER_MULTINOMIAL);
    assert!((samplers[0].temperature - 0.7).abs() < 1e-6);
    assert_eq!(samplers[0].seed, 42);
    assert_eq!(samplers[1].kind, PIE_SAMPLER_TOP_K);
    assert_eq!(samplers[1].k, 40);
    assert_eq!(samplers[2].kind, PIE_SAMPLER_LOGPROBS);
    // Logprobs.token_ids alias the native Vec<u32>'s heap.
    let native_logprobs = match &req.samplers[2] {
        Sampler::Logprobs { token_ids } => token_ids,
        _ => unreachable!(),
    };
    assert_eq!(samplers[2].token_ids_ptr, native_logprobs.as_ptr());
    assert_eq!(samplers[2].token_ids_len, 3);

    // Adapter bindings: i64 sentinels (-1 = unbound).
    assert_eq!(view.desc.adapter_bindings_len, 1);
    let bindings = unsafe { std::slice::from_raw_parts(view.desc.adapter_bindings_ptr, 1) };
    assert_eq!(bindings[0].adapter_id, 99);
    assert_eq!(bindings[0].seed, -1);
}

#[test]
fn copy_request_view_unit_enum_field() {
    let cr = CopyRequest {
        dir: CopyDir::D2H,
        srcs: vec![1, 2, 3],
        dsts: vec![10, 20, 30],
    };
    let view = pie_copy_request_view(&cr);
    // Unit enum field is u8 (matches PieCopyDirDesc = u8).
    assert_eq!(view.desc.dir, PIE_COPY_DIR_D2H);
    assert_eq!(view.desc.srcs_ptr, cr.srcs.as_ptr());
    assert_eq!(view.desc.srcs_len, 3);
    assert_eq!(view.desc.dsts_len, 3);
}

#[test]
fn adapter_request_view_with_path_sentinel() {
    let ar = AdapterRequest {
        op: AdapterOp::Load,
        adapter_id: 0xCAFE,
        path: "/tmp/x.bin".to_string(),
    };
    let view = pie_adapter_request_view(&ar);
    assert_eq!(view.desc.op, PIE_ADAPTER_OP_LOAD);
    assert_eq!(view.desc.adapter_id, 0xCAFE);
    assert_eq!(view.desc.path_len, 10);
    // path_ptr aliases the native String's heap.
    assert_eq!(view.desc.path_ptr, ar.path.as_ptr());

    // Empty-path case: ptr aliases an empty string (may or may not be null
    // depending on allocator), len is 0.
    let ar2 = AdapterRequest {
        op: AdapterOp::Save,
        adapter_id: 0,
        path: String::new(),
    };
    let view2 = pie_adapter_request_view(&ar2);
    assert_eq!(view2.desc.path_len, 0);
}

#[test]
fn frame_forward_view_nested_through_enum_dispatch() {
    let req = ForwardRequest {
        token_ids: vec![1, 2, 3],
        ..Default::default()
    };
    let frame = Frame {
        driver_id: 7,
        payload: RequestPayload::Forward(req),
    };
    let view = pie_frame_view(&frame);
    assert_eq!(view.desc.driver_id, 7);
    assert_eq!(view.desc.payload.kind, PIE_REQUEST_PAYLOAD_FORWARD);
    // The active variant's embedded sub-desc has the request's slice ptrs.
    let fwd_desc = &view.desc.payload.forward;
    assert_eq!(fwd_desc.token_ids_len, 3);
    let inner = match &frame.payload {
        RequestPayload::Forward(r) => r,
        _ => unreachable!(),
    };
    assert_eq!(fwd_desc.token_ids_ptr, inner.token_ids.as_ptr());
}

#[test]
fn request_payload_view_health_variant_zeros_inactive() {
    let p = RequestPayload::Health;
    let view = pie_request_payload_view(&p);
    assert_eq!(view.desc.kind, PIE_REQUEST_PAYLOAD_HEALTH);
    // Inactive variant sub-descs are zero-initialized.
    assert!(view.desc.forward.token_ids_ptr.is_null());
    assert!(view.desc.copy.srcs_ptr.is_null());
    assert!(view.desc.adapter.path_ptr.is_null());
}

#[test]
fn sampler_view_inline_struct_variant() {
    let s = Sampler::TopKTopP {
        temperature: 0.9,
        k: 50,
        p: 0.95,
    };
    let view = pie_sampler_view(&s);
    assert_eq!(view.desc.kind, 4); // TopKTopP
    assert!((view.desc.temperature - 0.9).abs() < 1e-6);
    assert_eq!(view.desc.k, 50);
    assert!((view.desc.p - 0.95).abs() < 1e-6);

    // Logprobs variant: token_ids ptr aliases native Vec<u32>.
    let token_ids = vec![100u32, 200, 300];
    let token_ids_ptr_native = token_ids.as_ptr();
    let s2 = Sampler::Logprobs { token_ids };
    let view2 = pie_sampler_view(&s2);
    assert_eq!(view2.desc.kind, PIE_SAMPLER_LOGPROBS);
    assert_eq!(view2.desc.token_ids_ptr, token_ids_ptr_native);
    assert_eq!(view2.desc.token_ids_len, 3);
}

/// The view holders are dropped only when the `PieXView` itself drops.
/// Smoke-test that the desc's pointers stay valid for the entire view
/// lifetime, even after the view is moved.
#[test]
fn view_pointers_stable_across_move() {
    let req = ForwardRequest {
        token_ids: vec![1, 2, 3, 4, 5],
        samplers: vec![Sampler::TopK {
            temperature: 1.0,
            k: 5,
        }],
        sampler_indptr: vec![0, 1],
        ..Default::default()
    };
    let v1 = pie_forward_request_view(&req);
    let token_ptr = v1.desc.token_ids_ptr;
    let samplers_ptr = v1.desc.samplers_ptr;

    // Move the view; pointers in `desc` must still resolve.
    let v2 = v1;
    assert_eq!(v2.desc.token_ids_ptr, token_ptr);
    assert_eq!(v2.desc.samplers_ptr, samplers_ptr);
    // Deref the samplers ptr — heap allocation owned by v2's holder.
    let s0 = unsafe { &*v2.desc.samplers_ptr };
    assert_eq!(s0.kind, PIE_SAMPLER_TOP_K);
    assert_eq!(s0.k, 5);
    // Use req to keep it alive for the lifetime check.
    let _ = &req;
}

/// A used variable to silence unused-import lints when types defined
/// via the macro aren't otherwise referenced here.
#[allow(dead_code)]
fn _unused() {
    let _p: *const u8 = ptr::null();
}
