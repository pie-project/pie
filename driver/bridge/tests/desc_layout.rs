//! Layout regression tests for the macro-emitted `Pie<T>Desc` POD types.
//!
//! The F1 compatibility shim in `driver/{portable,cuda}/src/_bridge/`
//! relies on the Rust `#[repr(C)] Pie<T>Desc` field order, sizes, and
//! offsets matching the C struct declarations in `include/pie_bridge.h`
//! byte-for-byte. If the macro changes how it emits Desc fields
//! (rename, reorder, add padding), both sides silently corrupt the
//! wire — there's no compile-time check.
//!
//! These tests pin the layout. Any future macro change that breaks
//! ABI compatibility will fail one of these assertions before reaching
//! the C++ build, which is much cheaper than diagnosing a wire-format
//! corruption later.

#![cfg(feature = "cabi")]

use std::mem::{align_of, offset_of, size_of};

use pie_bridge::{
    PieAdapterBindingDesc, PieAdapterRequestDesc, PieCopyRequestDesc, PieForwardRequestDesc,
    PieForwardResponseDesc, PieFrameDesc, PieRequestPayloadDesc, PieResponseFrameDesc,
    PieResponsePayloadDesc, PieSamplerDesc, PieStatusResponseDesc,
};

// ============================================================================
// Foundational shapes
// ============================================================================

/// All Desc types must be POD (zero-init valid, trivially copyable).
/// Using `mem::zeroed` in `Default::default()` requires this.
#[test]
fn descs_are_copy_clone_default() {
    fn assert_copy<T: Copy>() {}
    fn assert_clone<T: Clone>() {}
    fn assert_default<T: Default>() {}

    assert_copy::<PieFrameDesc>();
    assert_copy::<PieResponseFrameDesc>();
    assert_copy::<PieRequestPayloadDesc>();
    assert_copy::<PieResponsePayloadDesc>();
    assert_copy::<PieForwardRequestDesc>();
    assert_copy::<PieForwardResponseDesc>();
    assert_copy::<PieCopyRequestDesc>();
    assert_copy::<PieAdapterRequestDesc>();
    assert_copy::<PieStatusResponseDesc>();
    assert_copy::<PieAdapterBindingDesc>();
    assert_copy::<PieSamplerDesc>();

    assert_clone::<PieFrameDesc>();
    assert_default::<PieFrameDesc>();
    assert_default::<PieResponseFrameDesc>();
}

// ============================================================================
// Top-level frames
// ============================================================================

#[test]
fn pie_frame_desc_layout() {
    // C declaration:
    //   typedef struct PieFrameDesc {
    //     uint32_t              driver_id;
    //     PieRequestPayloadDesc payload;
    //   } PieFrameDesc;
    assert_eq!(offset_of!(PieFrameDesc, driver_id), 0);
    // payload alignment may bump the offset; allow ≥4 (after u32).
    assert!(offset_of!(PieFrameDesc, payload) >= 4);
    // Total size = u32 padded to alignment + sub-desc size.
    let s = size_of::<PieFrameDesc>();
    let a = align_of::<PieFrameDesc>();
    assert_eq!(s % a, 0, "PieFrameDesc must be a multiple of its alignment");
}

#[test]
fn pie_response_frame_desc_layout() {
    // C: { uint32_t driver_id; uint8_t aborted; PieResponsePayloadDesc payload; }
    assert_eq!(offset_of!(PieResponseFrameDesc, driver_id), 0);
    assert_eq!(offset_of!(PieResponseFrameDesc, aborted), 4);
    assert!(offset_of!(PieResponseFrameDesc, payload) >= 5);
}

// ============================================================================
// Tagged-union payload descs
// ============================================================================

#[test]
fn pie_request_payload_desc_layout() {
    // C: { uint8_t kind; PieForwardRequestDesc forward; PieCopyRequestDesc copy;
    //      PieAdapterRequestDesc adapter; }
    // kind at 0; sub-descs follow in declared order.
    assert_eq!(offset_of!(PieRequestPayloadDesc, kind), 0);
    let off_fwd = offset_of!(PieRequestPayloadDesc, forward);
    let off_copy = offset_of!(PieRequestPayloadDesc, copy);
    let off_adapter = offset_of!(PieRequestPayloadDesc, adapter);
    assert!(off_fwd > 0);
    assert!(off_copy > off_fwd);
    assert!(off_adapter > off_copy);
}

#[test]
fn pie_response_payload_desc_layout() {
    assert_eq!(offset_of!(PieResponsePayloadDesc, kind), 0);
    let off_fwd = offset_of!(PieResponsePayloadDesc, forward);
    let off_status = offset_of!(PieResponsePayloadDesc, status);
    assert!(off_fwd > 0);
    assert!(off_status > off_fwd);
}

// ============================================================================
// Slice fields: ptr + len pairs
// ============================================================================

#[test]
fn pie_forward_request_desc_ptr_len_pairing() {
    // Each Vec<T> field in ForwardRequest must produce a (ptr, len)
    // pair in PieForwardRequestDesc. Verify they're adjacent and the
    // len follows the ptr.
    let off_tok_ptr = offset_of!(PieForwardRequestDesc, token_ids_ptr);
    let off_tok_len = offset_of!(PieForwardRequestDesc, token_ids_len);
    assert!(off_tok_len > off_tok_ptr);
    // u64 pointer + usize on 64-bit systems → 16 bytes per pair.
    assert_eq!(
        off_tok_len - off_tok_ptr,
        size_of::<*const u32>(),
        "len must follow ptr by exactly one pointer width",
    );

    // Spot-check the last pair too (output_spec_flags).
    let off_osf_ptr = offset_of!(PieForwardRequestDesc, output_spec_flags_ptr);
    let off_osf_len = offset_of!(PieForwardRequestDesc, output_spec_flags_len);
    assert!(off_osf_len > off_osf_ptr);
    assert_eq!(off_osf_len - off_osf_ptr, size_of::<*const u8>());

    // single_token_mode / has_user_mask trail the slice fields.
    let off_stm = offset_of!(PieForwardRequestDesc, single_token_mode);
    let off_hum = offset_of!(PieForwardRequestDesc, has_user_mask);
    assert!(off_stm > off_osf_len);
    assert_eq!(
        off_hum - off_stm,
        1,
        "bools should be 1 byte each, adjacent"
    );
}

#[test]
fn pie_forward_response_desc_starts_with_num_requests() {
    // C: { uint32_t num_requests; ... (slice pairs) ... }
    assert_eq!(offset_of!(PieForwardResponseDesc, num_requests), 0);
    assert!(offset_of!(PieForwardResponseDesc, tokens_indptr_ptr) >= 4);
}

#[test]
fn pie_copy_request_desc_layout() {
    // C: { uint8_t dir; const uint32_t* srcs_ptr; size_t srcs_len; ... }
    assert_eq!(offset_of!(PieCopyRequestDesc, dir), 0);
    let off_srcs_ptr = offset_of!(PieCopyRequestDesc, srcs_ptr);
    let off_dsts_ptr = offset_of!(PieCopyRequestDesc, dsts_ptr);
    assert!(off_srcs_ptr > 0);
    assert!(off_dsts_ptr > off_srcs_ptr);
}

#[test]
fn pie_adapter_request_desc_layout() {
    // C: { uint8_t op; uint64_t adapter_id; const uint8_t* path_ptr; size_t path_len; }
    assert_eq!(offset_of!(PieAdapterRequestDesc, op), 0);
    let off_id = offset_of!(PieAdapterRequestDesc, adapter_id);
    let off_path_ptr = offset_of!(PieAdapterRequestDesc, path_ptr);
    let off_path_len = offset_of!(PieAdapterRequestDesc, path_len);
    assert_eq!(off_id, 8, "u64 adapter_id should be 8-aligned");
    assert!(off_path_ptr > off_id);
    assert!(off_path_len > off_path_ptr);
}

#[test]
fn pie_status_response_desc_is_four_bytes() {
    // C: { int32_t status; }
    assert_eq!(size_of::<PieStatusResponseDesc>(), 4);
    assert_eq!(align_of::<PieStatusResponseDesc>(), 4);
    assert_eq!(offset_of!(PieStatusResponseDesc, status), 0);
}

// ============================================================================
// Sampler tagged-union desc
// ============================================================================

#[test]
fn pie_sampler_desc_layout() {
    // C: { uint8_t kind; float temperature; uint32_t seed; uint32_t k; float p;
    //      uint32_t num_tokens; uint32_t token_id;
    //      const uint32_t* token_ids_ptr; size_t token_ids_len; }
    //
    // Field order is the macro's natural source order — variants are
    // walked in declaration order and each new field is appended once.
    // `Multinomial` declares `{ temperature, seed }` first, so `seed`
    // lands at offset 8 in both Rust and the hand-written C header.
    // (Historical: an earlier draft used `Option<u32>` for `seed`,
    // which forced the macro to emit `seed_has`/`seed` in-place at the
    // first-appearance slot while the C header parked them at the end
    // — a silent ABI mismatch that corrupted `top_p` reads on the C++
    // side and produced token-0 spam from the cuda sampler. The fix
    // was to drop `Option` from the schema in favor of sentinels.)
    assert_eq!(offset_of!(PieSamplerDesc, kind), 0);
    assert_eq!(offset_of!(PieSamplerDesc, temperature), 4);
    assert_eq!(offset_of!(PieSamplerDesc, seed), 8);
    assert_eq!(offset_of!(PieSamplerDesc, k), 12);
    assert_eq!(offset_of!(PieSamplerDesc, p), 16);
    assert_eq!(offset_of!(PieSamplerDesc, num_tokens), 20);
    assert_eq!(offset_of!(PieSamplerDesc, token_id), 24);
    assert_eq!(offset_of!(PieSamplerDesc, token_ids_ptr), 32);
    assert_eq!(offset_of!(PieSamplerDesc, token_ids_len), 40);
    assert_eq!(size_of::<PieSamplerDesc>(), 48);

    // token_ids ptr+len pair must be adjacent.
    let p = offset_of!(PieSamplerDesc, token_ids_ptr);
    let l = offset_of!(PieSamplerDesc, token_ids_len);
    assert_eq!(l - p, size_of::<*const u32>());
}

#[test]
fn pie_adapter_binding_desc_layout() {
    // C: { int64_t adapter_id; int64_t seed; }  — both -1 = unbound.
    assert_eq!(offset_of!(PieAdapterBindingDesc, adapter_id), 0);
    assert_eq!(offset_of!(PieAdapterBindingDesc, seed), 8);
    assert_eq!(size_of::<PieAdapterBindingDesc>(), 16);
}

// ============================================================================
// Zero-init validity
// ============================================================================

/// `Default::default()` uses `mem::zeroed`. Confirm that for every Desc,
/// all-zero is a valid POD state (no enum discriminants out of range,
/// no Option-niche conflicts, etc.).
#[test]
fn descs_default_to_all_zero() {
    let f = PieFrameDesc::default();
    assert_eq!(f.driver_id, 0);
    assert_eq!(f.payload.kind, 0);

    let rf = PieResponseFrameDesc::default();
    assert_eq!(rf.driver_id, 0);
    assert_eq!(rf.aborted, 0);
    assert_eq!(rf.payload.kind, 0);

    let s = PieSamplerDesc::default();
    assert_eq!(s.kind, 0);
    assert_eq!(s.temperature, 0.0);
    assert!(s.token_ids_ptr.is_null());
    assert_eq!(s.token_ids_len, 0);

    // AdapterBindingDesc default is "both fields zero" — but the
    // *runtime* meaning of zero for adapter_id is "adapter 0" (a valid
    // ID), not "unbound". Producers must explicitly write -1 to mark
    // unbound. Default-construction is for memory hygiene only here.
    let b = PieAdapterBindingDesc::default();
    assert_eq!(b.adapter_id, 0);
    assert_eq!(b.seed, 0);
}

// ============================================================================
// Reasonable size bounds — catches accidental bloat
// ============================================================================

#[test]
fn desc_sizes_are_within_reasonable_bounds() {
    // PieFrameDesc embeds all three request variants → biggest top-level
    // type. Should be well under 2 KiB on a 64-bit system.
    assert!(
        size_of::<PieFrameDesc>() < 2048,
        "PieFrameDesc unexpectedly large: {}",
        size_of::<PieFrameDesc>()
    );
    // PieResponseFrameDesc embeds ForwardResponse + Status.
    assert!(size_of::<PieResponseFrameDesc>() < 1024);
    // Single-variant descs.
    assert!(size_of::<PieStatusResponseDesc>() == 4);
    assert!(size_of::<PieAdapterBindingDesc>() <= 32);
    assert!(size_of::<PieSamplerDesc>() <= 64);
}
