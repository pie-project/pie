//! XQA's by-value aggregate, pinned field-for-field against a layout
//! **measured out of NVRTC's PTX**.
//!
//! # Why this exists
//!
//! `kernel_mha` (`csrc/vendor/xqa/mha.cuh:2801`) takes
//! `KVCacheList<usePagedKVCache> const cacheList` **by value** — forty bytes
//! of pointers and a count, in the middle of its parameter list. Every other
//! XQA parameter is a scalar or a pointer and crosses as one `u64` cell;
//! this one does not, and until it could cross, the six `attention_xqa*.cu`
//! host programs could not be retired.
//!
//! `northstar.md` §5.1 predicted this file's arrival and named the failure
//! mode it has to defend against:
//!
//! > `ArgValue::Bytes` is written and unused. Its first caller … is where
//! > the tag bypass gets tested — and note the failure mode: **a wrong
//! > bypass is a launch with a garbage struct, not a type error.**
//!
//! It was half right about the caller: XQA's `KVCacheList` got here before
//! `attn`'s `MLAParams`. It was entirely right about the failure mode, which
//! is why nothing below was read off the header.
//!
//! # Where the numbers came from
//!
//! **Not from reading the header.** `mhaUtils.cuh:241-257` gives the field
//! ORDER, and the order is all a human can safely read off — the struct's
//! shape depends on `ENABLE_4BIT_KV_CACHE`, which inserts two more pointers
//! in the middle, and on `beamWidth`, and neither is visible at the
//! declaration site.
//!
//! The offsets were measured by `nvrtc-probes/xqa_kvcachelist.py`, which
//! compiles the real struct under the real carried header set and the real
//! `XQA_COMMON_OPTIONS` and reads the offsets back out of the emitted PTX.
//! **Re-running that script is how these assertions get checked against a new
//! upstream.** Its method is `nvrtc-probes/params_layout.py`'s, whose banner
//! records which constant-expression routes NVRTC rejects and why:
//! `offsetof` and `__builtin_offsetof` are both unavailable, the cast of a
//! null-member address is rejected, and only
//! `(unsigned)((char*)&((S*)0)->b - (char*)(S*)0)` — the *difference* of two
//! pointers — folds.
//!
//! Measured, `rc=0`, NVRTC 13.0, `compute_89`, `-std=c++17 -default-device`:
//!
//! ```text
//! ENABLE_4BIT_KV_CACHE=0  beamWidth=1  tokensPerPage=32
//! sizeof(KVCachePageIndex)=4  sizeof(SeqLenDataType)=4  sizeof(ptr)=8
//!
//! KVCacheList<true>: sizeof=40 alignof=8
//!      0  kCacheVLLM         GMemCacheHead*
//!      8  vCacheVLLM         GMemCacheHead*
//!     16  kvCachePageList    KVCachePageIndex const*
//!     24  seqLenList         SeqLenDataType const*
//!     32  maxNbPagesPerSeq   uint32_t
//! ```
//!
//! This confirms the transcription `families/attn.rs` made by eye — *"four
//! pointers plus a `uint32_t`: 40 bytes, 8-aligned"* — which is the first
//! time in this migration that reading and measuring have agreed. The reason
//! they agree is the next section, and it is the thing to check before
//! trusting any future XQA mirror.
//!
//! # The trap that cannot reach this struct, and why that is not luck
//!
//! `params_layout.txt` records three transcription traps the FA2 mirrors hit,
//! all of them the same kind:
//!
//! | what the declaration suggests | what it is |
//! |---|---|
//! | `uint_fastdiv` — one `u32` | **24 bytes**, shifting everything after it by twenty |
//! | CuTe's `dA` — three integers | **8 bytes**; the static modes are compressed out |
//! | two `paged_kv_t`s with equal `sizeof` | **different interiors** — `num_heads` at +24 vs +20 |
//!
//! Every one of them is a **nested aggregate** whose own size or interior
//! disagrees between `csrc/shim`'s headers and CCCL's. The probe measures the
//! *shim's* layout, because the shim is what NVRTC is handed; a mirror built
//! from CCCL's numbers would be wrong in the interior while agreeing on
//! `sizeof`, which is the worst possible way to be wrong.
//!
//! `KVCacheList<true>` has **no nested aggregate**. It is four pointers and a
//! `uint32_t`, and both header sets agree that a pointer is eight bytes and a
//! `uint32_t` is four. So these five offsets are correct under the shim's
//! headers *and* under CCCL's, which makes this the only XQA-side mirror for
//! which the shim/CCCL split is not a live hazard — worth writing down
//! precisely because the next XQA aggregate will not have that property.
//!
//! # What this mirror claims, and its precondition
//!
//! It claims to be `KVCacheList<true>` **under `ENABLE_4BIT_KV_CACHE == 0`**.
//! That is not an assumption: `XQA_COMMON_OPTIONS` (`families/attn.rs`) sets
//! `CACHE_ELEM_ENUM=0` for all six lattice members, and
//! `ENABLE_4BIT_KV_CACHE` is derived from it. Should a member ever compile
//! with `CACHE_ELEM_ENUM=4`, `kSfCacheVLLM`/`vSfCacheVLLM` appear at 16 and
//! 24, everything after shifts by sixteen, and `sizeof` becomes 56 — so the
//! `static_assert`s [`typecheck_tu`](crate::x::abi::typecheck_tu) emits from
//! [`LAYOUTS`] fail on that unit and name the struct. **That is the check.**
//! The Rust-side assertions cannot see a `-D`; the C++ ones compile under the
//! member's own options and can.
//!
//! `KVCacheList<false>` — the contiguous cache — measured `sizeof=24
//! alignof=8`, `data@0`, `seqLenList@8`, `capacity@16`. It is recorded here
//! and **not mirrored**, because no lattice member instantiates it: all six
//! set `USE_PAGED_KV_CACHE=1`. Data only for what has a reading consumer;
//! this paragraph is the reading consumer, and a `#[repr(C)]` struct nothing
//! passes would not be.
//!
//! # For `xqa-nvrtc`
//!
//! This module is the floor half only: the mirror, its layout, and its
//! [`Abi`](crate::x::Abi) impl. The `unit!`, `contract!` and `bind!` for the
//! six-member lattice belong here too and are yours to add — the `Abi` impl
//! is what makes `cache_list: KvCacheList` an ordinary `unit!` parameter, so
//! nothing about declaring it differs from declaring an `i32`.
//!
//! **Write the `unit!` entry before the bind.** The by-value path bypasses
//! the runtime's per-operand tag check by construction, so the typecheck TU
//! is the only thing standing between a transposed parameter and a launch
//! reading a garbage struct — and a garbage struct is not a type error, it is
//! a fault or, worse, an answer.

use crate::by_value;

/// A device address held as an opaque word.
///
/// Re-exported so a mirror's fields read as one vocabulary. See
/// [`x::abi::DevicePtr`](crate::x::abi::DevicePtr) for why these are `u64`
/// and not `*mut T`.
pub use crate::x::abi::DevicePtr;

/// `KVCacheList<true>` — XQA's paged KV cache descriptor, passed by value.
///
/// Mirrors `csrc/vendor/xqa/mhaUtils.cuh:241-257` under
/// `ENABLE_4BIT_KV_CACHE == 0`. Nothing here may be dereferenced on the host:
/// every field is a device address or a device-side count.
///
/// Field names are Rust's, not C++'s; the C++ spelling of each travels in
/// [`LAYOUTS`] and is what the `static_assert`s name, so a reader chasing a
/// failed assertion never has to guess which Rust field moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvCacheList {
    /// `kCacheVLLM` — `GMemCacheHead*`, the K pages.
    pub k_cache: DevicePtr,
    /// `vCacheVLLM` — `GMemCacheHead*`, the V pages.
    pub v_cache: DevicePtr,
    /// `kvCachePageList` — `KVCachePageIndex const*`, shape
    /// `[batchSize][beamWidth][2][maxNbPagesPerSeq]`.
    pub page_list: DevicePtr,
    /// `seqLenList` — `SeqLenDataType const*`, shape `[batchSize][beamWidth]`.
    pub seq_len_list: DevicePtr,
    /// `maxNbPagesPerSeq` — the third stride of `page_list`, in pages.
    pub max_pages_per_seq: u32,
}

by_value! {
    KvCacheList as "KVCacheList<true>",
    tag = KvCacheLayerView,
    probe = "nvrtc-probes/xqa_kvcachelist.py",
    size = 40, align = 8,
    {
        k_cache           @ 0  as "kCacheVLLM",
        v_cache           @ 8  as "vCacheVLLM",
        page_list         @ 16 as "kvCachePageList",
        seq_len_list      @ 24 as "seqLenList",
        max_pages_per_seq @ 32 as "maxNbPagesPerSeq",
    }
}

impl KvCacheList {
    /// The descriptor for one paged KV cache.
    ///
    /// Every argument is a device address the caller already holds; this
    /// function only assembles them, which is why it is not `unsafe` — no
    /// pointer is read here, and the launch that reads them is.
    #[must_use]
    pub const fn paged(
        k_cache: DevicePtr,
        v_cache: DevicePtr,
        page_list: DevicePtr,
        seq_len_list: DevicePtr,
        max_pages_per_seq: u32,
    ) -> Self {
        Self { k_cache, v_cache, page_list, seq_len_list, max_pages_per_seq }
    }
}

/// The by-value aggregates the XQA units pass, for the typecheck TU.
///
/// `unit!` cannot derive this: `PARAMS` holds C++ spellings and a spelling
/// cannot be turned back into a type. So it is written once, here, and handed
/// to [`typecheck_tu`](crate::x::abi::typecheck_tu) with `PARAMS`.
///
/// A `KVCacheList<true>` parameter that appears in a unit whose `LAYOUTS`
/// omits it still typechecks by name — `is_same_v` catches a wrong type — but
/// loses the `sizeof`/`offsetof` assertions, which are the half that catches
/// a header whose FIELDS moved under a type whose NAME did not.
pub static LAYOUTS: &[crate::x::Layout] = &[<KvCacheList as crate::x::ByValue>::LAYOUT];
