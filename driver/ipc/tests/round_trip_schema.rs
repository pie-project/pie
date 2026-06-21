//! rkyv round-trip tests: encode an owned `Frame` / `ResponseFrame`,
//! parse the archived bytes, verify zero-copy field access matches.

use pie_ipc::wire::{WireError, encode_request, encode_response, parse_request, parse_response};
use pie_ipc::{
    AdapterBinding, AdapterOp, AdapterRequest, ArchivedRequestPayload, ArchivedResponsePayload,
    CopyDir, CopyRequest, CopyResource, ForwardRequest, ForwardResponse, Frame, RequestPayload,
    ResponseFrame, ResponsePayload, Sampler, StatusResponse,
};

#[test]
fn frame_health_round_trip() {
    let f = Frame {
        driver_id: 42,
        payload: RequestPayload::Health,
    };
    let bytes = encode_request(&f).unwrap();
    let archived = parse_request(&bytes).unwrap();
    assert_eq!(archived.driver_id, 42);
    assert!(matches!(archived.payload, ArchivedRequestPayload::Health));
}

#[test]
fn frame_forward_round_trip() {
    let mut req = ForwardRequest {
        token_ids: vec![10, 20, 30],
        position_ids: vec![0, 1, 2],
        qo_indptr: vec![0, 3],
        masks: vec![pie_ipc::Brle {
            buffer: vec![0xAAAAu32, 0xBBBB],
            total_size: 0xAAAA + 0xBBBB,
        }],
        context_ids: vec![0xCAFE, 0xBABE],
        output_spec_flags: vec![true, false, true],
        single_token_mode: true,
        has_user_mask: true,
        sampler_indptr: vec![0, 2],
        adapter_bindings: vec![
            AdapterBinding {
                adapter_id: 42,
                seed: -1,
            },
            AdapterBinding {
                adapter_id: -1,
                seed: -7,
            },
        ],
        // Multimodal side-channel: two visual spans (an M-RoPE image with
        // mrope positions, then a 1-D-RoPE image without). Exercises the
        // appended ForwardRequest fields over the real rkyv wire.
        image_indptr: vec![0, 2],
        image_grids: vec![1, 4, 4, 1, 2, 2],
        image_anchor_positions: vec![10, 100],
        image_pixels: vec![1, 2, 3, 9],
        image_pixel_indptr: vec![0, 3, 4],
        image_mrope_positions: vec![10, 10, 10, 10, 10, 11],
        image_mrope_indptr: vec![0, 6, 6],
        // Multimodal audio side-channel: one clip's log-mel features + anchor.
        audio_features: vec![1, 2, 3, 4, 5, 6, 7, 8],
        audio_feature_indptr: vec![0, 8],
        audio_anchor_rows: vec![3],
        audio_indptr: vec![0, 1],
        ..Default::default()
    };
    req.set_samplers(&[
        Sampler::TopK {
            temperature: 0.7,
            k: 50,
        },
        Sampler::Logprobs {
            token_ids: vec![1, 2, 3],
        },
    ]);
    let f = Frame {
        driver_id: 99,
        payload: RequestPayload::Forward(req.clone()),
    };

    let bytes = encode_request(&f).unwrap();
    let archived = parse_request(&bytes).unwrap();
    assert_eq!(archived.driver_id, 99);
    let ArchivedRequestPayload::Forward(arch_req) = &archived.payload else {
        panic!("expected Forward variant");
    };
    assert_eq!(arch_req.token_ids.as_slice(), &[10u32, 20, 30]);
    assert_eq!(arch_req.position_ids.as_slice(), &[0u32, 1, 2]);
    assert!(arch_req.single_token_mode);
    assert!(arch_req.has_user_mask);
    // SoA sampler wire: TopK(kind 1) + Logprobs(kind 9), labels in the CSR.
    assert_eq!(arch_req.sampler_kinds.as_slice(), &[1u8, 9]);
    assert_eq!(arch_req.sampler_token_ids.as_slice(), &[1u32, 2, 3]);
    assert_eq!(arch_req.sampler_token_ids_indptr.as_slice(), &[0u32, 0, 3]);
    // Multimodal side-channel survives the rkyv round-trip intact.
    assert_eq!(arch_req.image_indptr.as_slice(), &[0u32, 2]);
    assert_eq!(arch_req.image_grids.as_slice(), &[1u32, 4, 4, 1, 2, 2]);
    assert_eq!(arch_req.image_anchor_positions.as_slice(), &[10u32, 100]);
    assert_eq!(arch_req.image_pixels.as_slice(), &[1u8, 2, 3, 9]);
    assert_eq!(arch_req.image_pixel_indptr.as_slice(), &[0u32, 3, 4]);
    assert_eq!(
        arch_req.image_mrope_positions.as_slice(),
        &[10u32, 10, 10, 10, 10, 11]
    );
    assert_eq!(arch_req.image_mrope_indptr.as_slice(), &[0u32, 6, 6]);
    // Audio side-channel survives the rkyv round-trip intact.
    assert_eq!(arch_req.audio_features.as_slice(), &[1u8, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(arch_req.audio_feature_indptr.as_slice(), &[0u32, 8]);
    assert_eq!(arch_req.audio_anchor_rows.as_slice(), &[3u32]);
    assert_eq!(arch_req.audio_indptr.as_slice(), &[0u32, 1]);
}

#[test]
fn sampler_variants_archive() {
    let samplers = vec![
        Sampler::Multinomial {
            temperature: 0.5,
            seed: 42,
        },
        Sampler::TopK {
            temperature: 0.6,
            k: 40,
        },
        Sampler::TopP {
            temperature: 0.7,
            p: 0.9,
        },
        Sampler::MinP {
            temperature: 0.8,
            p: 0.05,
        },
        Sampler::TopKTopP {
            temperature: 0.9,
            k: 30,
            p: 0.95,
        },
        Sampler::Embedding,
        Sampler::Dist {
            temperature: 1.0,
            num_tokens: 8,
        },
        Sampler::RawLogits,
        Sampler::Logprob { token_id: 123 },
        Sampler::Logprobs {
            token_ids: vec![1, 2, 3],
        },
        Sampler::Entropy,
    ];
    let mut req = ForwardRequest {
        sampler_indptr: vec![0, samplers.len() as u32],
        ..Default::default()
    };
    req.set_samplers(&samplers);
    let f = Frame {
        driver_id: 1,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&f).unwrap();
    let archived = parse_request(&bytes).unwrap();
    let ArchivedRequestPayload::Forward(arch_req) = &archived.payload else {
        panic!()
    };

    // The flattened wire: every variant lands as its kind discriminant (0..10)
    // plus its raw per-variant value in the matching SoA array.
    assert_eq!(
        arch_req.sampler_kinds.as_slice(),
        &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    // TopK.k (slot 1) and TopKTopP.k (slot 4) — raw, 0 elsewhere.
    assert_eq!(
        arch_req.sampler_top_k.as_slice(),
        &[0u32, 40, 0, 0, 30, 0, 0, 0, 0, 0, 0]
    );
    // Dist.num_tokens (slot 6) stays separate (not folded into top_k).
    assert_eq!(
        arch_req.sampler_num_tokens.as_slice(),
        &[0u32, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0]
    );
    assert_eq!(arch_req.sampler_seeds.as_slice(), &[42u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // Unified label CSR: Logprob.token_id (slot 8) then Logprobs.token_ids (slot 9).
    assert_eq!(arch_req.sampler_token_ids.as_slice(), &[123u32, 1, 2, 3]);
    assert_eq!(
        arch_req.sampler_token_ids_indptr.as_slice(),
        &[0u32, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4]
    );
}

#[test]
fn copy_request_round_trip() {
    let f = Frame {
        driver_id: 3,
        payload: RequestPayload::Copy(CopyRequest {
            dir: CopyDir::D2H,
            srcs: vec![1, 2, 3],
            dsts: vec![10, 20, 30],
            resource: CopyResource::Kv,
        }),
    };
    let bytes = encode_request(&f).unwrap();
    let archived = parse_request(&bytes).unwrap();
    let ArchivedRequestPayload::Copy(cr) = &archived.payload else {
        panic!()
    };
    // CopyDir archives as its enum discriminant.
    assert!(matches!(cr.dir, pie_ipc::ArchivedCopyDir::D2H));
    assert_eq!(cr.srcs.as_slice(), &[1u32, 2, 3]);
    assert_eq!(cr.dsts.as_slice(), &[10u32, 20, 30]);
}

#[test]
fn adapter_request_round_trip() {
    let f = Frame {
        driver_id: 4,
        payload: RequestPayload::Adapter(AdapterRequest {
            op: AdapterOp::Load,
            adapter_id: 0xCAFE,
            path: "/tmp/x.bin".to_string(),
        }),
    };
    let bytes = encode_request(&f).unwrap();
    let archived = parse_request(&bytes).unwrap();
    let ArchivedRequestPayload::Adapter(ar) = &archived.payload else {
        panic!()
    };
    assert!(matches!(ar.op, pie_ipc::ArchivedAdapterOp::Load));
    assert_eq!(ar.adapter_id, 0xCAFEu64);
    assert_eq!(ar.path.as_str(), "/tmp/x.bin");
}

#[test]
fn forward_response_round_trip() {
    let resp = ForwardResponse {
        num_requests: 2,
        tokens_indptr: vec![0, 2, 5],
        tokens: vec![100, 101, 200, 201, 202],
        entropies_indptr: vec![0, 1, 2],
        entropies: vec![0.42, 1.5],
        spec_indptr: vec![0, 2, 3],
        spec_tokens: vec![11, 12, 21],
        spec_positions: vec![5, 6, 9],
        ..Default::default()
    };
    let f = ResponseFrame {
        driver_id: 7,
        aborted: false,
        payload: ResponsePayload::Forward(resp),
    };
    let bytes = encode_response(&f).unwrap();
    let archived = parse_response(&bytes).unwrap();
    assert_eq!(archived.driver_id, 7);
    let ArchivedResponsePayload::Forward(fr) = &archived.payload else {
        panic!()
    };
    assert_eq!(fr.num_requests, 2u32);
    assert_eq!(fr.tokens.as_slice(), &[100u32, 101, 200, 201, 202]);
    assert_eq!(fr.spec_indptr.as_slice(), &[0u32, 2, 3]);
    assert_eq!(fr.spec_tokens.as_slice(), &[11u32, 12, 21]);
    assert_eq!(fr.spec_positions.as_slice(), &[5u32, 6, 9]);
}

#[test]
fn status_response_round_trip() {
    let f = ResponseFrame {
        driver_id: 1,
        aborted: false,
        payload: ResponsePayload::Status(StatusResponse { status: 0 }),
    };
    let bytes = encode_response(&f).unwrap();
    let archived = parse_response(&bytes).unwrap();
    let ArchivedResponsePayload::Status(sr) = &archived.payload else {
        panic!()
    };
    assert_eq!(sr.status, 0i32);
}

#[test]
fn aborted_response_propagates_as_handler_aborted() {
    let f = ResponseFrame {
        driver_id: 99,
        aborted: true,
        payload: ResponsePayload::Status(StatusResponse { status: -1 }),
    };
    let bytes = encode_response(&f).unwrap();
    let result = parse_response(&bytes);
    assert!(matches!(result.err(), Some(WireError::HandlerAborted)));
}

#[test]
fn corrupt_buffer_rejected() {
    let buf = vec![0xFFu8; 64];
    let result = parse_request(&buf);
    assert!(matches!(result.err(), Some(WireError::Verify(_))));
}
