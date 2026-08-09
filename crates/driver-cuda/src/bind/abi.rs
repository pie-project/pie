//! The types a launcher takes that are neither scalars nor pointers.
//!
//! One `#[repr(C)]` mirror per C++ record, and nothing else. In particular
//! **no driver state lives here**: a record earns a place in this module only
//! by being kernel vocabulary — something a launcher reads and forgets.
//! `AttentionWorkspace` and the plan caches look like they belong and do not;
//! they are scratch pools and caches the DRIVER owns, they are C++ objects
//! only because they are currently defined on the kernel side of the line,
//! and the answer for them is to move rather than to be mirrored here.
//!
//! [`KvCacheLayerView`] is the opposite case and the reason this module
//! exists. Its C++ header says why it was written: *"a neutral per-layer KV
//! descriptor, so a cache owner in the driver can expose its storage to a
//! kernel without this crate importing `store/`"*. It carries no ownership,
//! outlives no call, and is pure description — which is exactly what makes a
//! `#[repr(C)]` mirror the whole of the port rather than a wrapper over one.
//!
//! ## Why a mirror is safe to write here at all
//!
//! Because it is checked. `tests/launch_abi.rs` reads each mirror's own
//! layout with `offset_of!`, bakes those numbers into a generated C++
//! translation unit as `static_assert`s, and compiles it against the real
//! header. Size, alignment, every field offset and the member COUNT all have
//! to agree or the file does not build. No number below was written by hand
//! and none is checked by inspection.

use core::ffi::{c_int, c_void};

use crate::dtype::DType;

/// How a KV cache stores its pages.
///
/// Discriminants are the C++ enum's and are load-bearing for the same reason
/// [`DType`]'s are: the value crosses the boundary as the one byte the C++
/// declares (`enum class KvCacheScheme : std::uint8_t`), so agreeing on the
/// numbers is what makes it a cast instead of a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvCacheScheme {
    /// Stored as the model's own dtype; no scales.
    Native = 0,
    /// FP8 with one scale for the whole tensor.
    Fp8PerTensor = 1,
    /// INT8 with a scale per (token, head).
    Int8PerTokenHead = 2,
    /// FP8 with a scale per (token, head).
    Fp8PerTokenHead = 3,
    /// FP4 with a scale per block.
    Fp4Block = 4,
}

/// One layer's KV storage, as a kernel sees it.
///
/// Field order is the C++'s and may not be rearranged — the mirror is checked
/// positionally, so a reordering is a build failure rather than a subtle one,
/// but it is still a reordering of a shared ABI and not a local choice.
///
/// The two envelope pointers are null unless envelopes were explicitly
/// enabled on the cache; [`Self::has_envelopes`] is the C++'s own predicate.
/// They are `*mut u16` rather than opaque because the C++ types them that way
/// — `std::uint16_t*` — and dropping that would make the mirror describe less
/// than the header does.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct KvCacheLayerView {
    /// Which layer this describes.
    pub layer: c_int,
    /// Where its pages actually live, for a KV-shared layer.
    pub source_layer: c_int,
    /// Pages in this layer's pool.
    pub num_pages: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// KV heads, after any GQA grouping.
    pub num_kv_heads: c_int,
    /// Channels per head.
    pub head_dim: c_int,
    /// How the pages are stored, and therefore whether the scale planes
    /// below are meaningful.
    pub scheme: KvCacheScheme,
    /// The element type the pages actually hold, which is the model's dtype
    /// only under [`KvCacheScheme::Native`].
    pub storage_dtype: DType,
    /// The quantisation block, for the schemes that have one.
    pub block_size: c_int,
    /// The K pages.
    pub k_pages: *mut c_void,
    /// The V pages.
    pub v_pages: *mut c_void,
    /// K's scale plane; null under [`KvCacheScheme::Native`].
    pub k_scales: *mut c_void,
    /// V's scale plane; null under [`KvCacheScheme::Native`].
    pub v_scales: *mut c_void,
    /// A bf16 shadow of K, for the kernels that cannot dequantise inline.
    pub k_bf16_pages: *mut c_void,
    /// A bf16 shadow of V, for the same reason.
    pub v_bf16_pages: *mut c_void,
    /// Quest per-page key envelopes, `[num_pages, num_kv_heads, head_dim]`
    /// bf16 each. Null unless envelopes were enabled on the cache.
    pub k_env_min: *mut u16,
    /// The other envelope plane; see [`Self::has_envelopes`].
    pub k_env_max: *mut u16,
    /// Pages are `[..., num_kv_heads, page_size, head_dim]` rather than
    /// `[..., page_size, num_kv_heads, head_dim]`.
    pub hnd_layout: bool,
    /// Storage is the model's own bf16; [`Self::is_native_bf16`] reads it.
    pub native_bf16: bool,
}

impl KvCacheLayerView {
    /// Both envelope planes are present.
    ///
    /// The C++ predicate, kept rather than left to the caller: it tests BOTH
    /// pointers, and a caller that checked one would be right on every cache
    /// that exists and wrong on the one that half-allocates.
    pub fn has_envelopes(&self) -> bool {
        !self.k_env_min.is_null() && !self.k_env_max.is_null()
    }

    /// Storage is the model's own bf16, so no dequantisation step applies.
    pub fn is_native_bf16(&self) -> bool {
        self.native_bf16
    }
}

/// The attention scratch, as a launcher sees it.
///
/// The `#[repr(C)]` half of the split step 2b made: the pool itself is the
/// driver's (`AttentionWorkspace` in `driver-cuda/csrc/src/`), and this is
/// the five values the kernels actually read out of it. A census of `attn/`
/// is what fixed the field list at five — the kernels call `float_buffer`,
/// `int_buffer`, `float_bytes`, `int_bytes` and `page_locked_int`, and
/// nothing else — so this mirror is not a subset anyone chose, it is the
/// whole of what crosses.
///
/// Passed BY VALUE, unlike [`KvCacheLayerView`]'s C++ original which is too.
/// Five words is cheaper to copy than to chase, and by-value is the one
/// passing mode where a layout proof is the entire proof: there is no
/// lifetime to get wrong.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AttentionWorkspaceView {
    /// Device scratch FlashInfer accumulates split-KV partials into.
    pub float_buffer: *mut c_void,
    /// How much of it there is. Kernels check their budget against this.
    pub float_bytes: usize,
    /// Device scratch holding per-request scheduling metadata.
    pub int_buffer: *mut c_void,
    /// How much of it there is.
    pub int_bytes: usize,
    /// Pinned host mirror of `int_buffer`, staged by a plan and uploaded by
    /// the driver. Which slot this is rotates per step, and the rotation is
    /// not visible from here.
    pub page_locked_int: *mut c_void,
}

/// One layer's paged MLA cache.
///
/// MLA stores a compressed latent and a decoupled rope plane instead of K and
/// V, which is why this is its own descriptor and not [`KvCacheLayerView`]
/// with two of its pointers left null: the two caches have different page
/// SHAPES, and a mirror that pretended otherwise would have to carry a flag
/// nothing reads.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MlaCacheLayerView {
    /// Which layer this describes.
    pub layer: c_int,
    /// Pages in this layer's pool.
    pub num_pages: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// Width of the compressed latent.
    pub kv_lora_rank: c_int,
    /// Width of the decoupled rope plane.
    pub qk_rope_head_dim: c_int,
    /// The latent pages.
    pub ckv_pages: *mut c_void,
    /// The rope-plane pages.
    pub kpe_pages: *mut c_void,
}

/// FlashInfer's sm90 prefill schedule.
///
/// Unlike the plan CACHES — `DecodePlanCache` and friends, which the header
/// leaves incomplete on purpose — this one is defined, so it is a layout the
/// driver can build rather than a handle it can only hold. The offsets are
/// into the workspace's `int_buffer`, which is why they are `i64` and not
/// pointers: the buffer moves, the schedule does not.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HopperPrefillPlan {
    /// Offset of the qo tile index array.
    pub qo_tile_indices_offset: i64,
    /// Offset of the qo indptr.
    pub qo_indptr_offset: i64,
    /// Offset of the kv indptr.
    pub kv_indptr_offset: i64,
    /// Offset of the per-tile qo length array.
    pub qo_len_offset: i64,
    /// Offset of the per-tile kv length array.
    pub kv_len_offset: i64,
    /// Offset of the head index array.
    pub head_indices_offset: i64,
    /// Offset of the work indptr.
    pub work_indptr_offset: i64,
    /// Offset of the batch index array.
    pub batch_indices_offset: i64,
    /// Every head runs the same schedule, so the head arrays are shared.
    pub same_schedule_for_all_heads: bool,
    /// Tokens the schedule covers.
    pub total_tokens: c_int,
    /// Requests it covers.
    pub num_requests: c_int,
    /// Query heads.
    pub num_q_heads: c_int,
    /// KV heads.
    pub num_kv_heads: c_int,
    /// Head width.
    pub head_dim: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// Sliding-window extent, or `-1` for none.
    pub window_left: c_int,
    /// The schedule is causal.
    pub causal: bool,
    /// The schedule was built. A default-constructed plan is not.
    pub valid: bool,
}

/// Original-YaRN scaling, for the MLA rope.
///
/// Passed by `const*` rather than `const&` because it is OPTIONAL — a
/// deployment without original-YaRN passes null — which is a fact the row
/// carries as `nullable` and no C++ signature can.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct YarnOriginalParams {
    /// Interpolation factor.
    pub factor: f32,
    /// Fast-rotating dimension cutoff.
    pub beta_fast: f32,
    /// Slow-rotating dimension cutoff.
    pub beta_slow: f32,
    /// Post-scaling applied to attention logits.
    pub attention_factor: f32,
    /// The context length the checkpoint was trained at.
    pub original_max_position: c_int,
}

/// One lane's structured-mask descriptor, for `attn::pack_structured_mask`
/// — a driver-internal launcher, so the mirror serves the bridge rather
/// than any DSL row. Three `u32`s; the C++ defaults them to zero and so
/// does [`Default`] here.
///
/// **RE-EXPORTED, no longer defined here.** The one definition is
/// `kernels_cuda_new::x::attn::params::StructuredMaskParams`, in the crate
/// that owns `attn`'s device text, for the reason `x/xqa.rs` gives for
/// `KvCacheList`: `unit!` has to NAME the type to declare
/// `attn::device::pack_structured_mask`, and `driver-cuda` depends on
/// `kernels-cuda-new` rather than the other way round. A mirror in two
/// crates is a layout assertion that can drift.
///
/// It nearly did. `pack_dense_mask.cuh:29-50` argues that two definitions of
/// this POD are acceptable *because* `pack_dense_mask.cu` `static_assert`ed
/// `sizeof`, `alignof` and all three offsets across them — and that `.cu` and
/// its `.hpp` are deleted, so from that day the two agreed by luck. They did
/// agree: `kind@0 window@4 sink@8`, `sizeof=12 alignof=4`, MEASURED out of
/// NVRTC's PTX by `nvrtc-probes/attn_structured_mask.py` against the surviving
/// `attn::device::StructuredMaskParams`. Kept as a re-export and not deleted
/// because [`kernels::Ty::needs_mirror`] answers true for
/// `Ty::StructuredMasks` — `Ty::rust()` spells it as the unqualified `*const
/// StructuredMaskParams`, so a generated binding only compiles in a module
/// that has this name in scope, and [`ffi`] below is that module.
pub use kernels_cuda_new::x::attn::params::StructuredMaskParams;

/// The activation the fused CUTLASS MoE runs between its two grouped GEMMs.
///
/// A mirror of `moe::MoeActivation` — an `enum class` with the default `int`
/// underlying type, passed BY VALUE, so `#[repr(i32)]` with the C++'s own
/// discriminants. `tests/launch_abi.rs` pins each value against the header;
/// a reordered C++ enum fails there rather than routing gemma-4 through
/// qwen's activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MoeActivation {
    /// nemotron_h.
    Relu2 = 0,
    /// qwen3.5 / qwen3.6 MoE, glm5 / kimi / deepseek_v4.
    Swiglu = 1,
    /// gemma-4 26B-A4B routed experts (GELU-tanh gate).
    Geglu = 2,
}

/// Which rows of a gpt-oss packed MXFP4 scale table the Marlin repack
/// selects. Mirror of `quant::Mxfp4RowSelect` (`enum class : int`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Mxfp4RowSelect {
    /// Every row, in order.
    Identity = 0,
    /// Even rows — the gate half of an interleaved bank.
    Even = 1,
    /// Odd rows — the up half.
    Odd = 2,
}

/// THE ARCHIVE'S DOOR STOOD HERE — `pub mod ffi`, and it is DELETED.
///
/// Its whole body was one line:
///
/// ```ignore
/// include!(concat!(env!("OUT_DIR"), "/launch_bindings.rs"));
/// ```
///
/// `build.rs` wrote that file from every family table, one
/// `extern "C" pie_k_<symbol>` per STATED row, and the generated dispatch
/// called them. **It was the only item in `driver-cuda/src` that reached the
/// C++ archive** — north star §6's census found 76 `bridge` gate attributes
/// and exactly one whose body needed a symbol `launch_bindings.rs` declares.
/// The other 75 were fossil or row-world (BRIDGE_RETIREMENT.md §2.2, and
/// `e88f1ffff` for the 68 retractions).
///
/// It goes now because it has nothing to declare. `emit_rust_bindings` emits
/// a declaration per row `abi::stated()` keeps, `stated()` drops a row with
/// no operands, `table::ROW_TABLES` is `&[]`, and every row `table::KERNELS`
/// still holds is a `Contract::sig` — which states none. The generated file
/// is empty, so this module declared nothing and its only caller, the
/// generated `match` in `dispatch_generated`, is deleted in the same index.
///
/// The seven hand-written `pie_x_*` extras that also stood here went earlier
/// and for a different reason, recorded because the two are easy to
/// conflate: the decode/prefill plan-cache factories, their deleters,
/// `set_decode_plan_int_base` and the two planners were C++ lifetimes a C ABI
/// could not express, and they are `fire::flashinfer_fa2::{DecodePlanCache,
/// PrefillPlanCache}` now — plain Rust whose deleter is `Drop`.

#[cfg(feature = "_cuda")]
/// Seed a KV cache's envelope tiers as EMPTY, safely.
///
/// The `unsafe` for this launch used to sit in `pools/kv_cache_live.rs`,
/// which is a pool reaching past `device`/`bind` for a CUDA symbol —
/// §7's *"an answer that does not need a card does not need `unsafe`
/// either"* read from the other side: a module that only says WHAT to
/// launch should not have to spell HOW.
///
/// Safe because the planes are ones the cache's own `alloc_tensor`
/// produced and the extents are the geometry it allocated them against;
/// a null plane is a cache with no envelopes, which the launcher reads
/// as nothing to do.
///
/// # This was the tree's `ffi::pie_k_layout_launch_envelope_seed_empty_bf16`
///
/// It called `layout/envelope.cu`'s `launch_envelope_seed_empty_bf16` through
/// a shim entry `abi::emit_c_shim` generated from
/// `table::driver_internal`'s `envelope_seed` row. **Both are gone**: the
/// `.cu` is deleted, the row is deleted, and the launch is
/// [`kernels_cuda_new::x::layout::envelope_seed_empty`] — a driver-owned
/// `kernels::Launch` over a `LaunchRule::Unstated` JIT row. The signature
/// here does not change, so no caller of this function does.
///
/// **The two `.cast()`s are the port's one caller-visible change.** The
/// planes are `*mut u16` in this driver — a `TensorSpec` of raw halfwords —
/// and `*mut bf16` in the declaration, because the declaration carries the
/// `.cuh`'s own type and the typecheck translation unit compares whole
/// parameter types. The cast is where those two spellings meet, and it is
/// here rather than inside the host program so that the program's parameter
/// list still reads as the kernel's.
///
/// The routine's `Result` is dropped rather than returned, and that is this
/// function's own decision rather than an oversight: the C++ launcher returned
/// `void` and its guard was silent, the callers in `pools/kv_cache_live.rs`
/// pass extents the cache allocated against and cannot be empty, and a
/// `Result` here would be a new refusal channel for a condition no caller can
/// act on. `#[must_use]` on `Result` is what makes dropping it a decision
/// spelled out loud.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn seed_envelopes_empty(
    env_min: *mut u16,
    env_max: *mut u16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: crate::device::StreamRef<'_>,
) {
    // SAFETY: `stream` outlives the launch, and the two envelope planes are
    // the cache's own allocation for `num_pages * num_kv_heads * head_dim`.
    let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(stream.as_raw().cast()) };
    let _ = kernels_cuda_new::x::layout::envelope_seed_empty(
        &ctx,
        env_min.cast(),
        env_max.cast(),
        num_pages,
        num_kv_heads,
        head_dim,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pointer-carrying mirror is neither `Send` nor `Sync` by accident, so
    /// say what it is: a borrowed description of device memory the caller
    /// owns. Nothing here keeps the pages alive.
    #[test]
    fn a_view_is_a_borrow_and_owns_nothing() {
        let v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 4,
            page_size: 16,
            num_kv_heads: 2,
            head_dim: 64,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        let copy = v;
        assert_eq!(copy.num_pages, 4);
        assert!(copy.is_native_bf16());
    }

    /// `has_envelopes` needs BOTH, which is the C++'s rule and not an
    /// obvious one — a half-allocated cache is exactly what it guards.
    #[test]
    fn one_envelope_plane_is_not_envelopes() {
        let mut v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 1,
            page_size: 1,
            num_kv_heads: 1,
            head_dim: 1,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        assert!(!v.has_envelopes());
        let mut cell: u16 = 0;
        v.k_env_min = &mut cell;
        assert!(!v.has_envelopes(), "one plane is not enough");
        v.k_env_max = &mut cell;
        assert!(v.has_envelopes());
    }
}
