//! Flat-POD wire types — the schema's fixed-layout descriptors.
//!
//! These five types ride ON the rkyv ring (they appear inside the rich
//! `#[schema]` types in [`crate::schema`]), so they are part of
//! [`crate::SCHEMA_HASH`] and `build.rs` hashes this file. But unlike the rich
//! types they are *flat POD*: every field is a fixed-size scalar, so the rkyv
//! archived form is byte-identical to the native `#[repr(C)]` / `#[repr(u8)]`
//! layout on a little-endian target. A C++/Python consumer therefore reads them
//! by a plain cast — no generated reader / descriptor / builder / PyO3 surface
//! is needed, so they skip the `#[schema]` derive macro entirely. cbindgen
//! emits them 1:1 from this source (no `parse.expand` needed for these).
//!
//! Rich parents reference these with the `#[schema(pod)]` field attribute, which
//! tells `pie-driver-abi-derive` to embed the type by value in the parent
//! descriptor and read it by cast instead of chasing an rkyv relative pointer.

use rkyv::{Archive, Deserialize, Serialize};

/// Copy direction for [`crate::CopyRequest`].
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[repr(u8)]
pub enum CopyDir {
    D2H,
    H2D,
    D2D,
    H2H,
}

/// Which cache a [`crate::CopyRequest`] moves.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[repr(u8)]
pub enum CopyResource {
    Kv,
    Rs,
}

/// Adapter management op for [`crate::AdapterRequest`].
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[repr(u8)]
pub enum AdapterOp {
    Load,
    Save,
    ZoInit,
    ZoUpdate,
    // CSM native audio output (pie:core/audio-out). Appended per the enum
    // evolution rule (append-only — the discriminant byte is on the wire). The
    // `AdapterRequest.path` carries a JSON request
    // `{"prompt":[u32,...],"max_frames":u32,"out_path":"..."}`; the CSM driver
    // runs the full generation (backbone prefill + per-frame depth loop + Mimi
    // decode), writes the raw little-endian f32 PCM to `out_path`, and returns
    // `StatusResponse.status` = number of Mimi frames produced (negative on
    // error). Reuses the Adapter cold-path transport so no new wire payload
    // variant is needed. See AUDIO_OUTPUT.md.
    GenerateAudio,
}

/// Per-slot adapter binding. `-1` sentinels mean "unbound" — both fields
/// are signed so the wire form matches what cuda's legacy SoA path
/// already consumes (`.as<int64_t>()`), no shim conversion needed.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[repr(C)]
pub struct AdapterBinding {
    /// `-1` means no adapter bound for this slot.
    pub adapter_id: i64,
    /// `-1` means no caller-provided adapter seed.
    pub seed: i64,
}

/// Generic status response. Convention:
///   * 0 = success
///   * negative = error
///   * -1 reserved for abort sentinel (the server writes one of these
///     when a lease drops without commit; see [`crate::ResponseFrame::aborted`]).
///   * positive = method-specific
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy)]
#[rkyv(derive(Debug))]
#[repr(C)]
pub struct StatusResponse {
    pub status: i32,
}
