//! Sampling-program carrier (lane L1, programmable sampler): exercises the
//! `ForwardRequest::sampling_program_*`/`sampling_input_*`/`sampling_late_*` SoA
//! carrier across every layer it crosses — the AoS push/at helpers, the batch
//! merge, the rkyv out-of-proc wire, and the in-proc `ToDesc`/`FromDesc` pivot
//! the C++ driver reads. The bytecode is opaque to the bridge, so these bytes
//! are arbitrary stand-ins for real `pie-sampling-ir` output.

use pie_ipc::wire::{encode_request, parse_request};
use pie_ipc::{
    ArchivedRequestPayload, ForwardRequest, Frame, RequestPayload, SamplingBinding, SamplingInput,
    SamplingProgramSubmission,
};

fn sample_program() -> SamplingProgramSubmission {
    SamplingProgramSubmission {
        bytecode: vec![0xDE, 0xAD, 0xBE, 0xEF],
        inputs: vec![
            SamplingInput {
                key: 7,
                bytes: vec![1, 2, 3],
            },
            SamplingInput {
                key: 9,
                bytes: vec![4, 5],
            },
        ],
        bindings: vec![
            SamplingBinding::Logits,
            SamplingBinding::Tensor { key: 7 },
            SamplingBinding::Tensor { key: 9 },
        ],
        late_keys: vec![100, 200],
        late_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// AoS helpers: push_sampling_program ↔ sampling_program_at
// ---------------------------------------------------------------------------

#[test]
fn push_then_read_single_program() {
    let prog = sample_program();
    let mut req = ForwardRequest::default();
    req.push_sampling_program(&prog);

    assert_eq!(req.n_sampling_programs(), 1);
    // Nested CSRs carry their leading 0 + one boundary for the single program.
    assert_eq!(req.sampling_program_bytes_indptr, vec![0, 4]);
    assert_eq!(req.sampling_input_indptr, vec![0, 2]);
    assert_eq!(req.sampling_late_indptr, vec![0, 2]);
    // Index table: keys, offsets into the blob, and per-entry lengths.
    assert_eq!(req.sampling_input_keys, vec![7, 9]);
    assert_eq!(req.sampling_input_offsets, vec![0, 3]);
    assert_eq!(req.sampling_input_lens, vec![3, 2]);
    assert_eq!(req.sampling_input_blob, vec![1, 2, 3, 4, 5]);
    assert_eq!(req.sampling_late_keys, vec![100, 200]);

    assert_eq!(req.sampling_program_at(0), Some(prog));
    assert_eq!(req.sampling_program_at(1), None);
}

#[test]
fn push_then_read_multiple_programs() {
    let mut req = ForwardRequest::default();
    let p0 = SamplingProgramSubmission {
        bytecode: vec![1, 1, 1],
        inputs: vec![SamplingInput {
            key: 1,
            bytes: vec![10, 11],
        }],
        bindings: vec![SamplingBinding::Logits, SamplingBinding::Tensor { key: 1 }],
        late_keys: vec![],
        late_inputs: vec![],
    };
    let p1 = SamplingProgramSubmission {
        bytecode: vec![2, 2],
        inputs: vec![],
        bindings: vec![SamplingBinding::Logits],
        late_keys: vec![42],
        late_inputs: vec![],
    };
    req.push_sampling_program(&p0);
    req.push_sampling_program(&p1);

    assert_eq!(req.n_sampling_programs(), 2);
    assert_eq!(req.sampling_program_at(0), Some(p0));
    assert_eq!(req.sampling_program_at(1), Some(p1));
    assert_eq!(req.sampling_program_at(2), None);
    // Bytecode CSR concatenates 3 + 2 bytes.
    assert_eq!(req.sampling_program_bytes_indptr, vec![0, 3, 5]);
    // Late CSR: p0 has 0 keys, p1 has 1.
    assert_eq!(req.sampling_late_indptr, vec![0, 0, 1]);
    // Binding CSR: p0 has 2 slots, p1 has 1.
    assert_eq!(req.sampling_binding_indptr, vec![0, 2, 3]);
}

#[test]
fn empty_request_has_no_programs() {
    let req = ForwardRequest::default();
    assert_eq!(req.n_sampling_programs(), 0);
    assert_eq!(req.sampling_program_at(0), None);
}

// ---------------------------------------------------------------------------
// Batch merge: extend_sampling_programs_from
// ---------------------------------------------------------------------------

#[test]
fn batch_merge_offsets_every_nested_csr() {
    // Two single-request programs merged into a batch whose nested CSRs are
    // rooted with a leading 0 (mirrors `new_batched_forward_request_with_capacity`).
    let p0 = SamplingProgramSubmission {
        bytecode: vec![0xAA, 0xBB],
        inputs: vec![SamplingInput {
            key: 1,
            bytes: vec![9, 9, 9],
        }],
        bindings: vec![SamplingBinding::Tensor { key: 1 }],
        late_keys: vec![5],
        late_inputs: vec![],
    };
    let p1 = sample_program();

    let mut req0 = ForwardRequest::default();
    req0.push_sampling_program(&p0);
    let mut req1 = ForwardRequest::default();
    req1.push_sampling_program(&p1);

    let mut batch = ForwardRequest {
        sampling_program_indptr: vec![0],
        sampling_program_bytes_indptr: vec![0],
        sampling_input_indptr: vec![0],
        sampling_late_indptr: vec![0],
        sampling_binding_indptr: vec![0],
        ..Default::default()
    };

    batch.extend_sampling_programs_from(&req0);
    batch
        .sampling_program_indptr
        .push(batch.n_sampling_programs() as u32);
    batch.extend_sampling_programs_from(&req1);
    batch
        .sampling_program_indptr
        .push(batch.n_sampling_programs() as u32);

    // Per-request count CSR: one boundary per request, cumulative program count.
    assert_eq!(batch.sampling_program_indptr, vec![0, 1, 2]);
    assert_eq!(batch.n_sampling_programs(), 2);

    // Byte ranges: p0 (2 bytes) then p1 (4 bytes).
    assert_eq!(batch.sampling_program_bytes_indptr, vec![0, 2, 6]);
    // Input index table rebased: p0's 1 entry at blob offset 0, p1's 2 entries
    // at blob offsets 3 and 6 (p0 contributed 3 bytes to the shared blob).
    assert_eq!(batch.sampling_input_indptr, vec![0, 1, 3]);
    assert_eq!(batch.sampling_input_offsets, vec![0, 3, 6]);
    assert_eq!(batch.sampling_input_blob, vec![9, 9, 9, 1, 2, 3, 4, 5]);
    // Late keys concatenated, CSR offset.
    assert_eq!(batch.sampling_late_indptr, vec![0, 1, 3]);
    assert_eq!(batch.sampling_late_keys, vec![5, 100, 200]);

    // The merged programs read back identically.
    assert_eq!(batch.sampling_program_at(0), Some(p0));
    assert_eq!(batch.sampling_program_at(1), Some(p1));
}

#[test]
fn batch_merge_skips_requests_without_programs() {
    // A program-less request must advance the per-request count CSR without
    // adding a program (parallels the image/audio side-channel merge).
    let mut batch = ForwardRequest {
        sampling_program_indptr: vec![0],
        sampling_program_bytes_indptr: vec![0],
        sampling_input_indptr: vec![0],
        sampling_late_indptr: vec![0],
        sampling_binding_indptr: vec![0],
        ..Default::default()
    };
    let plain = ForwardRequest::default();

    batch.extend_sampling_programs_from(&plain);
    batch
        .sampling_program_indptr
        .push(batch.n_sampling_programs() as u32);
    batch.extend_sampling_programs_from(&plain);
    batch
        .sampling_program_indptr
        .push(batch.n_sampling_programs() as u32);

    assert_eq!(batch.n_sampling_programs(), 0);
    assert_eq!(batch.sampling_program_indptr, vec![0, 0, 0]);
    assert!(batch.sampling_program_bytes.is_empty());
    assert_eq!(batch.sampling_program_bytes_indptr, vec![0]);
}

// ---------------------------------------------------------------------------
// rkyv wire (out-of-proc Rust↔Rust)
// ---------------------------------------------------------------------------

#[test]
fn forward_program_rkyv_round_trip() {
    let prog = sample_program();
    let mut req = ForwardRequest {
        token_ids: vec![5],
        position_ids: vec![0],
        ..Default::default()
    };
    req.push_sampling_program(&prog);
    req.sampling_program_indptr = vec![0, req.n_sampling_programs() as u32];

    let frame = Frame {
        driver_id: 3,
        payload: RequestPayload::Forward(req),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = parse_request(&bytes).unwrap();
    let ArchivedRequestPayload::Forward(arch) = &archived.payload else {
        panic!("expected Forward variant");
    };

    assert_eq!(arch.sampling_program_indptr.as_slice(), &[0u32, 1]);
    assert_eq!(
        arch.sampling_program_bytes.as_slice(),
        &[0xDEu8, 0xAD, 0xBE, 0xEF]
    );
    assert_eq!(arch.sampling_program_bytes_indptr.as_slice(), &[0u32, 4]);
    assert_eq!(arch.sampling_input_blob.as_slice(), &[1u8, 2, 3, 4, 5]);
    assert_eq!(arch.sampling_input_keys.as_slice(), &[7u32, 9]);
    assert_eq!(arch.sampling_input_offsets.as_slice(), &[0u32, 3]);
    assert_eq!(arch.sampling_input_lens.as_slice(), &[3u32, 2]);
    assert_eq!(arch.sampling_input_indptr.as_slice(), &[0u32, 2]);
    assert_eq!(arch.sampling_late_keys.as_slice(), &[100u32, 200]);
    assert_eq!(arch.sampling_late_indptr.as_slice(), &[0u32, 2]);
}

#[test]
fn forward_without_program_carries_empty_carrier() {
    // The legacy sampler path leaves every carrier field empty over the wire.
    let frame = Frame {
        driver_id: 1,
        payload: RequestPayload::Forward(ForwardRequest {
            token_ids: vec![1, 2],
            ..Default::default()
        }),
    };
    let bytes = encode_request(&frame).unwrap();
    let archived = parse_request(&bytes).unwrap();
    let ArchivedRequestPayload::Forward(arch) = &archived.payload else {
        panic!("expected Forward variant");
    };
    assert!(arch.sampling_program_bytes.as_slice().is_empty());
    assert!(arch.sampling_program_indptr.as_slice().is_empty());
    assert!(arch.sampling_input_blob.as_slice().is_empty());
    assert!(arch.sampling_late_keys.as_slice().is_empty());
}

// ---------------------------------------------------------------------------
// In-proc ToDesc → FromDesc (the Rust↔C++ pivot the driver reads)
// ---------------------------------------------------------------------------

#[test]
fn forward_program_desc_round_trip() {
    let prog = sample_program();
    let mut req = ForwardRequest::default();
    req.push_sampling_program(&prog);
    req.sampling_program_indptr = vec![0, 1];

    let frame = Frame {
        driver_id: 11,
        payload: RequestPayload::Forward(req),
    };

    // ToDesc: the Desc aliases the frame's heap (zero copy) — read the new
    // carrier slices exactly as the C++ driver would off `PieForwardRequestDesc`.
    let view = frame.as_desc();
    let fwd = &view.desc.payload.forward;
    assert_eq!(fwd.sampling_program_bytes_len, 4);
    let bytecode = unsafe {
        std::slice::from_raw_parts(fwd.sampling_program_bytes_ptr, fwd.sampling_program_bytes_len)
    };
    assert_eq!(bytecode, &[0xDEu8, 0xAD, 0xBE, 0xEF]);
    let keys = unsafe {
        std::slice::from_raw_parts(fwd.sampling_input_keys_ptr, fwd.sampling_input_keys_len)
    };
    assert_eq!(keys, &[7u32, 9]);
    let late = unsafe {
        std::slice::from_raw_parts(fwd.sampling_late_keys_ptr, fwd.sampling_late_keys_len)
    };
    assert_eq!(late, &[100u32, 200]);

    // FromDesc: rebuild the native value and confirm the program survives.
    let back = Frame::from_desc(&view.desc);
    match back.payload {
        RequestPayload::Forward(fr) => {
            assert_eq!(fr.n_sampling_programs(), 1);
            assert_eq!(fr.sampling_program_at(0), Some(prog));
        }
        other => panic!("expected Forward, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// WS8 P2 device-resident next-input link (per-row)
// ---------------------------------------------------------------------------

#[test]
fn next_input_link_helpers_and_roundtrip() {
    let mut req = ForwardRequest::default();
    // This forward is a pipeline source, retained under global link id 5.
    req.set_pipeline_source_link(5);
    // Two per-row links naming DIFFERENT producers (the re-formed-batch case):
    // row 0 ← producer 5 row 2 → input slot 0; row 1 ← producer 9 row 7 → slot 1;
    // a skipped lane (u32::MAX dest).
    req.push_next_input_link(5, 2, 0);
    req.push_next_input_link(9, 7, 1);
    req.push_next_input_link(9, 3, u32::MAX);

    assert_eq!(req.n_next_input_links(), 3);
    assert_eq!(req.pipeline_source_link, 5);
    assert_eq!(req.next_input_producer_links, vec![5, 9, 9]);
    assert_eq!(req.next_input_src_rows, vec![2, 7, 3]);
    assert_eq!(req.next_input_dest_slots, vec![0, 1, u32::MAX]);

    // The directive survives the desc (C-ABI) round-trip — the form the C++
    // executor reads to resolve producer link ids → retained `pi.sampled`.
    let frame = Frame {
        driver_id: 7,
        payload: RequestPayload::Forward(req),
    };
    let view = frame.as_desc();
    let fwd = &view.desc.payload.forward;
    assert_eq!(fwd.pipeline_source_link, 5);
    let producer_links = unsafe {
        std::slice::from_raw_parts(
            fwd.next_input_producer_links_ptr,
            fwd.next_input_producer_links_len,
        )
    };
    assert_eq!(producer_links, &[5u32, 9, 9]);
    let back = Frame::from_desc(&view.desc);
    match back.payload {
        RequestPayload::Forward(fr) => {
            assert_eq!(fr.pipeline_source_link, 5);
            assert_eq!(fr.next_input_producer_links, vec![5, 9, 9]);
            assert_eq!(fr.next_input_src_rows, vec![2, 7, 3]);
            assert_eq!(fr.next_input_dest_slots, vec![0, 1, u32::MAX]);
        }
        other => panic!("expected Forward, got {other:?}"),
    }
}
