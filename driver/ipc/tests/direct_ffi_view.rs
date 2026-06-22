//! Direct-FFI view path: build a native Rust value, take its
//! `Pie<T>View<'_>`, and confirm the resulting `Pie<T>Desc` aliases
//! the source data with no rkyv encode/decode. Pair with
//! `pie_<t>_from_desc` (the build-fn's internal helper) to round-trip
//! back to native — exercising the same conversion the in-process C++
//! handoff would use.

use std::ptr;

use pie_ipc::{
    AdapterBinding, AdapterOp, AdapterRequest, CopyDir, CopyRequest, CopyResource, ForwardRequest,
    Frame, PIE_REQUEST_PAYLOAD_FORWARD, PIE_REQUEST_PAYLOAD_HEALTH, PIE_SAMPLER_LOGPROBS,
    PIE_SAMPLER_MULTINOMIAL, PIE_SAMPLER_TOP_K, RequestPayload, Sampler, pie_adapter_request_view,
    pie_copy_request_view, pie_forward_request_view, pie_frame_view, pie_request_payload_view,
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
    let mut req = ForwardRequest {
        token_ids: vec![10, 20, 30, 40, 50],
        position_ids: vec![0, 1, 2, 3, 4],
        context_ids: vec![0xCAFE, 0xBABE],
        single_token_mode: false,
        has_user_mask: true,
        sampler_indptr: vec![0, 3],
        adapter_bindings: vec![AdapterBinding {
            adapter_id: 99,
            seed: -1,
        }],
        ..Default::default()
    };
    req.set_samplers(&[
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
    ]);
    let view = pie_forward_request_view(&req);

    // Slices alias the native Vec's heap allocation (same data ptr).
    assert_eq!(view.desc.token_ids_ptr, req.token_ids.as_ptr());
    assert_eq!(view.desc.token_ids_len, 5);
    assert_eq!(view.desc.position_ids_ptr, req.position_ids.as_ptr());
    assert_eq!(view.desc.context_ids_ptr, req.context_ids.as_ptr());

    // Bool primitives copied by value.
    assert_eq!(view.desc.single_token_mode, 0);
    assert_eq!(view.desc.has_user_mask, 1);

    // Sampler SoA: every array is a primitive slice aliasing the native Vec's
    // heap (zero-copy — no per-sampler holder Vec).
    assert_eq!(view.desc.sampler_kinds_len, 3);
    assert_eq!(view.desc.sampler_kinds_ptr, req.sampler_kinds.as_ptr());
    let kinds = unsafe { std::slice::from_raw_parts(view.desc.sampler_kinds_ptr, 3) };
    assert_eq!(
        kinds,
        &[PIE_SAMPLER_MULTINOMIAL, PIE_SAMPLER_TOP_K, PIE_SAMPLER_LOGPROBS]
    );
    let temps = unsafe { std::slice::from_raw_parts(view.desc.sampler_temperatures_ptr, 3) };
    assert!((temps[0] - 0.7).abs() < 1e-6);
    let seeds = unsafe { std::slice::from_raw_parts(view.desc.sampler_seeds_ptr, 3) };
    assert_eq!(seeds[0], 42);
    let top_k = unsafe { std::slice::from_raw_parts(view.desc.sampler_top_k_ptr, 3) };
    assert_eq!(top_k[1], 40);
    // Logprobs.token_ids land in the unified label CSR, aliasing the native Vec.
    assert_eq!(view.desc.sampler_token_ids_ptr, req.sampler_token_ids.as_ptr());
    assert_eq!(view.desc.sampler_token_ids_len, 3);
    let labels = unsafe { std::slice::from_raw_parts(view.desc.sampler_token_ids_ptr, 3) };
    assert_eq!(labels, &[100u32, 200, 300]);

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
        resource: CopyResource::Kv,
    };
    let view = pie_copy_request_view(&cr);
    // Flat-POD enum field embedded by value (the desc holds a `CopyDir`).
    assert_eq!(view.desc.dir, CopyDir::D2H);
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
    assert_eq!(view.desc.op, AdapterOp::Load);
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

/// The view's Desc slices alias the source `ForwardRequest`'s heap and must
/// stay valid for the view's whole lifetime, even after the view is moved.
#[test]
fn view_pointers_stable_across_move() {
    let mut req = ForwardRequest {
        token_ids: vec![1, 2, 3, 4, 5],
        sampler_indptr: vec![0, 1],
        ..Default::default()
    };
    req.set_samplers(&[Sampler::TopK {
        temperature: 1.0,
        k: 5,
    }]);
    let v1 = pie_forward_request_view(&req);
    let token_ptr = v1.desc.token_ids_ptr;
    let kinds_ptr = v1.desc.sampler_kinds_ptr;

    // Move the view; pointers in `desc` must still resolve.
    let v2 = v1;
    assert_eq!(v2.desc.token_ids_ptr, token_ptr);
    assert_eq!(v2.desc.sampler_kinds_ptr, kinds_ptr);
    // Deref the SoA ptrs — they alias req's heap.
    let kind = unsafe { *v2.desc.sampler_kinds_ptr };
    let k = unsafe { *v2.desc.sampler_top_k_ptr };
    assert_eq!(kind, PIE_SAMPLER_TOP_K);
    assert_eq!(k, 5);
    // Use req to keep it alive for the lifetime check.
    let _ = &req;
}

/// A used variable to silence unused-import lints when types defined
/// via the macro aren't otherwise referenced here.
#[allow(dead_code)]
fn _unused() {
    let _p: *const u8 = ptr::null();
}
