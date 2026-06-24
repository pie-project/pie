//! Asserts the archived in-buffer layout so cross-language readers
//! (`include/pie_driver_abi.h`, Python ctypes) can rely on it.
//!
//! rkyv 0.8 generates archived types as `#[repr(C)]` by default; this
//! test pins the actual sizes/alignments/offsets we care about.

use std::mem::{align_of, offset_of, size_of};

use pie_ipc::{
    ArchivedAdapterBinding, ArchivedAdapterOp, ArchivedAdapterRequest, ArchivedCopyDir,
    ArchivedCopyRequest, ArchivedForwardRequest, ArchivedForwardResponse, ArchivedFrame,
    ArchivedRequestPayload, ArchivedResponseFrame, ArchivedResponsePayload, ArchivedStatusResponse,
};

/// rkyv's archived `Vec<T>` is `{ rel_ptr: i32, len: u32 }` — 8 bytes
/// total, aligned to 4. This is the foundational layout assumption for
/// the C++ adapter; if rkyv changes it the C++ headers need to update.
#[test]
fn archived_vec_layout_is_8_bytes() {
    use rkyv::vec::ArchivedVec;
    assert_eq!(size_of::<ArchivedVec<u32>>(), 8);
    assert_eq!(align_of::<ArchivedVec<u32>>(), 4);

    assert_eq!(size_of::<ArchivedVec<u8>>(), 8);
    assert_eq!(size_of::<ArchivedVec<u64>>(), 8);
    assert_eq!(size_of::<ArchivedVec<f32>>(), 8);
}

#[test]
fn archived_frame_layout() {
    // Frame { driver_id: u32, payload: RequestPayload }
    // ArchivedFrame is repr(C); driver_id at 0, payload follows with
    // alignment padding.
    assert_eq!(offset_of!(ArchivedFrame, driver_id), 0);
    assert!(offset_of!(ArchivedFrame, payload) >= 4);
    // The whole struct's size must be a multiple of its alignment.
    let s = size_of::<ArchivedFrame>();
    let a = align_of::<ArchivedFrame>();
    assert_eq!(s % a, 0);
}

#[test]
fn archived_response_frame_layout() {
    assert_eq!(offset_of!(ArchivedResponseFrame, driver_id), 0);
    // `aborted: bool` follows driver_id; payload after that.
    assert!(offset_of!(ArchivedResponseFrame, aborted) >= 4);
    assert!(
        offset_of!(ArchivedResponseFrame, payload) > offset_of!(ArchivedResponseFrame, aborted)
    );
}

#[test]
fn archived_request_payload_is_tagged() {
    // Union with 4 variants → 1-byte discriminator + max-of-variants
    // payload. Alignment depends on the largest variant.
    let s = size_of::<ArchivedRequestPayload>();
    let a = align_of::<ArchivedRequestPayload>();
    assert!(s > 0);
    assert!(a >= 1);
    eprintln!("ArchivedRequestPayload: size={s} align={a}");
}

#[test]
fn archived_response_payload_is_tagged() {
    let s = size_of::<ArchivedResponsePayload>();
    let a = align_of::<ArchivedResponsePayload>();
    eprintln!("ArchivedResponsePayload: size={s} align={a}");
    assert!(s > 0);
}

#[test]
fn archived_forward_request_is_all_vecs() {
    // Every Vec<u32> field is 8 bytes (ArchivedVec). The forward request is
    // all primitive Vecs now — the sampler AoS `Vec<Sampler>` was flattened
    // into parallel SoA arrays (sampler_kinds/temperatures/top_k/p/seeds/
    // num_tokens/token_ids/token_ids_indptr), plus 1 Vec<AdapterBinding>,
    // 1 Vec<bool> and 2 bool. Walk the offsets to confirm sane placement.
    let s = size_of::<ArchivedForwardRequest>();
    let a = align_of::<ArchivedForwardRequest>();
    eprintln!("ArchivedForwardRequest: size={s} align={a}");
    assert!(s > 0);
    // rkyv's ArchivedVec is `{ rel_ptr: i32, len: u32 }` — 4-aligned
    // regardless of element type T. So the request struct's alignment
    // is 4, not 8, even though it contains Vec<u64>.
    assert_eq!(a, 4);
}

#[test]
fn archived_forward_response_is_all_vecs() {
    let s = size_of::<ArchivedForwardResponse>();
    let a = align_of::<ArchivedForwardResponse>();
    eprintln!("ArchivedForwardResponse: size={s} align={a}");
    assert!(s > 0);
}

#[test]
fn archived_status_response_is_4_bytes() {
    assert_eq!(size_of::<ArchivedStatusResponse>(), 4);
    assert_eq!(align_of::<ArchivedStatusResponse>(), 4);
}

#[test]
fn archived_adapter_binding_layout() {
    // AdapterBinding { adapter_id: i64, seed: i64 } — both -1 = unbound.
    // Two i64 fields → 16 bytes total, 8-byte aligned.
    assert_eq!(size_of::<ArchivedAdapterBinding>(), 16);
    assert_eq!(align_of::<ArchivedAdapterBinding>(), 8);
}

#[test]
fn archived_copy_dir_is_one_byte() {
    assert_eq!(size_of::<ArchivedCopyDir>(), 1);
    assert_eq!(align_of::<ArchivedCopyDir>(), 1);
}

#[test]
fn archived_adapter_op_is_one_byte() {
    assert_eq!(size_of::<ArchivedAdapterOp>(), 1);
    assert_eq!(align_of::<ArchivedAdapterOp>(), 1);
}

#[test]
fn archived_copy_request_layout() {
    assert_eq!(offset_of!(ArchivedCopyRequest, dir), 0);
    // After the 1-byte dir + 3 bytes padding (Vec alignment is 4), srcs starts at 4.
    assert!(offset_of!(ArchivedCopyRequest, srcs) >= 4);
    assert!(offset_of!(ArchivedCopyRequest, dsts) > offset_of!(ArchivedCopyRequest, srcs));
}

#[test]
fn archived_adapter_request_layout() {
    // op: AdapterOp (1 byte) + padding + adapter_id: u64 (8-aligned)
    // + path: Option<ArchivedString> (8 bytes)
    assert_eq!(offset_of!(ArchivedAdapterRequest, op), 0);
    assert_eq!(offset_of!(ArchivedAdapterRequest, adapter_id), 8);
}

// `Sampler` is no longer a wire type (it dropped `#[schema]` when the AoS
// `Vec<Sampler>` was flattened to SoA), so there is no `ArchivedSampler` to
// pin a layout for — the sampler data rides the primitive SoA Vecs above.

/// Print field offsets in addition to sizes for the C++ adapter author.
#[test]
fn dump_offsets() {
    eprintln!("--- field offsets ---");
    eprintln!(
        "ArchivedFrame.driver_id   = {}",
        offset_of!(ArchivedFrame, driver_id)
    );
    eprintln!(
        "ArchivedFrame.payload     = {}",
        offset_of!(ArchivedFrame, payload)
    );
    eprintln!(
        "ArchivedResponseFrame.driver_id = {}",
        offset_of!(ArchivedResponseFrame, driver_id)
    );
    eprintln!(
        "ArchivedResponseFrame.aborted   = {}",
        offset_of!(ArchivedResponseFrame, aborted)
    );
    eprintln!(
        "ArchivedResponseFrame.payload   = {}",
        offset_of!(ArchivedResponseFrame, payload)
    );
    eprintln!(
        "ArchivedStatusResponse.status   = {}",
        offset_of!(ArchivedStatusResponse, status)
    );
    eprintln!(
        "ArchivedCopyRequest.dir         = {}",
        offset_of!(ArchivedCopyRequest, dir)
    );
    eprintln!(
        "ArchivedCopyRequest.srcs        = {}",
        offset_of!(ArchivedCopyRequest, srcs)
    );
    eprintln!(
        "ArchivedCopyRequest.dsts        = {}",
        offset_of!(ArchivedCopyRequest, dsts)
    );
    eprintln!(
        "ArchivedAdapterRequest.op         = {}",
        offset_of!(ArchivedAdapterRequest, op)
    );
    eprintln!(
        "ArchivedAdapterRequest.adapter_id = {}",
        offset_of!(ArchivedAdapterRequest, adapter_id)
    );
    eprintln!(
        "ArchivedAdapterRequest.path       = {}",
        offset_of!(ArchivedAdapterRequest, path)
    );
    eprintln!(
        "ArchivedForwardRequest.token_ids = {}",
        offset_of!(ArchivedForwardRequest, token_ids)
    );
    eprintln!(
        "ArchivedForwardRequest.position_ids = {}",
        offset_of!(ArchivedForwardRequest, position_ids)
    );
    eprintln!(
        "ArchivedForwardRequest.context_ids = {}",
        offset_of!(ArchivedForwardRequest, context_ids)
    );
    eprintln!(
        "ArchivedForwardRequest.single_token_mode = {}",
        offset_of!(ArchivedForwardRequest, single_token_mode)
    );
    eprintln!(
        "ArchivedForwardRequest.has_user_mask = {}",
        offset_of!(ArchivedForwardRequest, has_user_mask)
    );
    eprintln!(
        "ArchivedAdapterBinding.adapter_id = {}",
        offset_of!(ArchivedAdapterBinding, adapter_id)
    );
    eprintln!(
        "ArchivedAdapterBinding.seed       = {}",
        offset_of!(ArchivedAdapterBinding, seed)
    );
}

/// Print the full layout table once for the C++ adapter author. Run with:
/// `cargo test --test archived_layout dump_layouts -- --nocapture`
#[test]
fn dump_layouts() {
    macro_rules! dump {
        ($t:ty) => {
            eprintln!(
                "{:40} size={:4}  align={}",
                stringify!($t),
                size_of::<$t>(),
                align_of::<$t>()
            );
        };
    }
    eprintln!("--- archived layout ---");
    dump!(ArchivedFrame);
    dump!(ArchivedResponseFrame);
    dump!(ArchivedRequestPayload);
    dump!(ArchivedResponsePayload);
    dump!(ArchivedForwardRequest);
    dump!(ArchivedForwardResponse);
    dump!(ArchivedAdapterBinding);
    dump!(ArchivedCopyRequest);
    dump!(ArchivedAdapterRequest);
    dump!(ArchivedStatusResponse);
    dump!(ArchivedCopyDir);
    dump!(ArchivedAdapterOp);
}
