//! `attn/kv_paged.cu`'s device-window append, in Rust.
//!
//! One launcher of the eight that file held. It is the one that came first
//! because it is the one whose C++ was **entirely** host code that no rule
//! and no `Source` can carry: two refusals that THREW, a `Term::Is` over a
//! layout flag, and a grid that spans lanes the launch does not serve.
//!
//! # THIS FILE IS NOW FIVE LAUNCHERS, AND THREE OF THEM ARE REACHED
//!
//! `write_kv_to_pages`, `write_kv_to_pages_bf16` and
//! `write_kv_explicit_bf16` landed later and their C++ IS GONE from
//! `attn/kv_paged.cu`; they are reached from `bind::service`, because
//! `execution::RUST_SERVED` names the two that have `table::attn` rows.
//! `write_kv_explicit_bf16` needed §60.6's symbol split to get there — its
//! DEVICE rows are `attn::write_kv_explicit_bf16_dev` and arms — and
//! `write_kv_to_pages_bf16` needed nothing, because it never had a row at
//! all.
//!
//! # `write_kv_explicit_bf16_devwin` IS STILL NOT REACHED, AND THAT IS §58
//!
//! Read this before looking for its caller: there is none, and its C++
//! launcher is still in `attn/kv_paged.cu`. The port is finished. **The
//! routing is not blocked** — it was thought to be, for one pass, and the
//! account of why it is not is worth keeping because the mistake is a
//! recurring one.
//!
//! What looked like a wall was two invariants that seemed to contradict:
//!
//! * `execution::tests::a_walk_is_only_a_walk` asserts every `WALKED` symbol
//!   satisfies `unit::unit_of(sym).is_none()` — §52.11, *"a walk may drive a
//!   JIT'd kernel; it may not be one"*.
//! * `device::Specialisation::agrees` requires the opposite: it resolves
//!   `unit_of(self.base)` and then `unit.row(self.base)`, and
//!   `SPECIALISATIONS`' `WRITE_KV_EXPLICIT_DEVWIN` names
//!   `attn::write_kv_explicit_bf16_devwin` as its base. The symbol MUST have
//!   a unit row.
//!
//! Both assertions are right. They contradict only if this symbol needs a
//! `Walk`, and **it does not**. §58 is the resolution: a `Specialisation` IS
//! a walk, for the one shape it covers — `Specialisation::choose ->
//! Option<&Arm>` is a `Control::Switch { on }` with the subject restricted to
//! a `Fact` and the arms restricted to instantiations of one symbol. The two
//! are the same idea at two granularities, not competing classifications.
//!
//! > A symbol is `Specialisation`-selected **or** `Walk`-driven, never both.
//! > If a host program needs a `Walk` *and* an instantiation choice, the walk
//! > drives a symbol whose specialisation resolves underneath it — two
//! > symbols, which is what §52.11 already requires for its own reason.
//!
//! So this file wants no `Walk`, no `RUST_SERVED` entry and no
//! `bind::service` shim. It wants what it already has: a device row, a
//! specialisation over `hnd_layout`, and `fire::hand::fire`. **A `pub mod`
//! with no caller is a staging state, not an error.**
//!
//! The framing that cost the pass, kept because it is the fifth instance of
//! one shape in this session: *"may this symbol be walked"* was the wrong
//! question; *"must it also be rowed"* was the question. A true statement one
//! hop short of the one that decides.
//!
//! What survives here is what was expensive to derive: the geometry, cited
//! line by line; the refusal set, with which exits panic and which decline
//! and why; and the argument order, checked against the device row. When the symbol is split, this file is the body — add the
//! `WALKED` entry, the `RUST_SERVED` name and a `bind::service` shim and it
//! is reached, with nothing in it to rewrite.
//!
//! The kernel is unmoved. `attn::device::write_kv_explicit_devwin` lives in
//! `kernels-cuda-new/csrc/src/attn/kv_paged.cuh` and NVRTC compiles it as the
//! `attn/kv_paged` unit; `families::attn` rows both instantiations as
//! `attn::write_kv_explicit_bf16_devwin#hnd` and `#nhd`, and
//! `device::SPECIALISATIONS`' `WRITE_KV_EXPLICIT_DEVWIN` is the `Term::Is`
//! that chooses between them. What was left in the archive was the program
//! around the launch, and this module is that program.
//!
//! # The two refusals, and why they are not declines
//!
//! The C++ threw for both:
//!
//! ```text
//! kv_paged.cu:252   if (!layer.is_native_bf16()) {
//! kv_paged.cu:253       throw std::runtime_error(
//! kv_paged.cu:254           "write_kv_explicit_bf16_devwin requires native bf16 KV cache");
//! kv_paged.cu:262   if (layer.has_envelopes()) {
//! kv_paged.cu:263       throw std::runtime_error(
//! kv_paged.cu:264           "write_kv_explicit_bf16_devwin: envelope maintenance not yet "
//! kv_paged.cu:265           "windowed -- use the host-window form");
//! ```
//!
//! A throw is not a decline. `fire/gemv.rs` draws that line and this module
//! keeps it: a decline is a launch that had nothing to do, and both of these
//! are a launch that had something to do and cannot do it correctly. There is
//! no second kernel to fall back to — the quantised append is a different
//! kernel with a different argument list, and the envelope form is the
//! host-window `write_kv_explicit_bf16`, which takes no `win` at all — so
//! answering either with a substitute would be the silent-fallback failure
//! `Walk::refuses` exists to make unspellable. They panic, naming the
//! condition in the launcher's own words.
//!
//! The third exit IS a decline. `if (n_max <= 0) return;` (`kv_paged.cu:256`)
//! is an empty fire, which is not an error anywhere in this driver.
//!
//! # The grid
//!
//! `<<<n_max, 256, 0, stream>>>` at `kv_paged.cu:284` and `:292`. `n_max` is
//! the fire's FULL lane count and not this launch's region: the grid spans
//! every lane and out-of-window rows early out on `win[0]`/`win[1]`, which is
//! what lets a captured launch replay across row splits. `families::attn`
//! states the same rectangle as `LaunchRule::PerRow` and its header explains
//! at length why `Dims::rows` and `n_max` are the same number — the twin row
//! is `whole = true`, so no windowed statement can reach it. This module does
//! not use the rule: it is handed `n_max` by its caller, which is the same
//! `DispatchCtx::rows_total` the row would have recovered, and it builds the
//! `Launch` from it directly so that the number in the grid is the number the
//! `.cu` put there.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::bind::abi::{KvCacheLayerView, KvCacheScheme};
use crate::dtype::DType;

/// The block width both instantiations are launched at.
///
/// `constexpr int BLOCK = 256` at `kv_paged.cu:250`, and the same 256
/// `LaunchRule::PerRow` fixes. Not read from the row: this module states the
/// geometry the launcher stated, and a block width taken from somewhere else
/// is a second place for it to drift.
const BLOCK: u32 = 256;

/// Whether the append ran.
///
/// `#[must_use]` for the reason `fire/gemv.rs` gives: a launcher that can say
/// no must not be callable in a way that spells "it declined" the same as "it
/// ran".
#[must_use]
pub enum WriteKvExplicitDevwin {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and the reason.
    Declined(Decline),
}

/// Every way [`write_kv_explicit_bf16_devwin`] declines, in the launcher's
/// own words.
///
/// One variant, because the launcher had one silent exit. The other two exits
/// panic — see the module header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `n_max <= 0` — `kv_paged.cu:256`. An empty fire.
    NoLanes,
}

/// The device-window explicit KV append.
///
/// `layer` is the destination cache, `k_curr`/`v_curr` the rows to append,
/// `w_page`/`w_off` the per-row destination page and offset the fire
/// published, `win_d` the device window `{start, len}` the kernel reads
/// before any guard, and `n_max` the fire's full lane count. `row_valid` is
/// nullable and the row says so.
///
/// # Panics
///
/// If the cache is not native bf16, or if it carries envelopes. Both were
/// `throw std::runtime_error` in the C++ and both are conditions on the
/// CALLER, not on the launch; see the module header for why neither may be
/// answered with a decline. Also if the kernel table and this driver
/// disagree — `fire::hand::fire` panics with the symbol named.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window. That
/// is the assertion the caller already made when it handed the same pointers
/// to a `<<<>>>`.
pub unsafe fn write_kv_explicit_bf16_devwin(
    layer: KvCacheLayerView,
    k_curr: *const std::ffi::c_void,
    v_curr: *const std::ffi::c_void,
    w_page: *const u32,
    w_off: *const u32,
    win_d: *const u32,
    n_max: i32,
    stream: *mut std::ffi::c_void,
    row_valid: *const u8,
) -> WriteKvExplicitDevwin {
    // `kv_paged.cu:252` — the first `throw`, before the empty-extent test,
    // and in that order here too: a caller that passes an fp8 cache is wrong
    // whether or not it also passes zero rows.
    assert!(
        layer.is_native_bf16(),
        "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
    );
    // `kv_paged.cu:256`.
    if n_max <= 0 {
        return WriteKvExplicitDevwin::Declined(Decline::NoLanes);
    }
    // `kv_paged.cu:262` — envelope (quest) maintenance is not wired on this
    // variant. The C++ comment is the specification and it is reproduced
    // verbatim rather than paraphrased: "Envelope maintenance (quest) is NOT
    // wired on this variant yet — the campaign converts it when a windowed
    // producer needs it; until then a caller with envelopes must stay on the
    // host-window form." The host-window form is
    // `attn::write_kv_explicit_bf16`, which calls
    // `kernels::layout::launch_envelope_merge_written_bf16` after its append
    // and which `x::layout::envelope` refuses a row for.
    assert!(
        !layer.has_envelopes(),
        "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
         windowed — use the host-window form"
    );

    // `kv_paged.cu:283` — `if (layer.hnd_layout)`. The two arms differ in the
    // INSTANTIATION and in nothing else, which is why
    // `device::SPECIALISATIONS`' `WRITE_KV_EXPLICIT_DEVWIN` can state them as
    // one `Term::Is` over one operand, and why the argument list below is
    // built once.
    let symbol = if layer.hnd_layout {
        "attn::write_kv_explicit_bf16_devwin_dev#hnd"
    } else {
        "attn::write_kv_explicit_bf16_devwin_dev#nhd"
    };

    // `kv_paged.cu:284` / `:292`: `<<<n_max, BLOCK, 0, stream>>>`.
    let launch = Launch {
        grid: [n_max.unsigned_abs(), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };

    // The operand order is `families::attn`'s
    // `write_kv_explicit_devwin_hnd`/`_nhd` row, which is the `__global__`'s
    // parameter list — not the launcher's, which took the view whole and
    // carried a stream.
    let values = [
        ArgValue::Ptr(k_curr.cast_mut()),
        ArgValue::Ptr(v_curr.cast_mut()),
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(w_page.cast_mut().cast()),
        ArgValue::Ptr(w_off.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::Ptr(win_d.cast_mut().cast()),
        ArgValue::I32(n_max),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.num_kv_heads),
        ArgValue::I32(layer.head_dim),
    ];

    super::hand::fire(symbol, launch, &values, stream);
    WriteKvExplicitDevwin::Launched
}

// ===========================================================================
// THE QUANTISED APPEND, `kv_paged.cu:114-218`
// ===========================================================================

/// `__nv_fp8_interpretation_t`, on the host side of the launch.
///
/// The C++ never declared this: it computed the value inline at the call site
/// and passed it straight through — `kv_paged.cu:159-161` and again at
/// `:394-396`, the same ternary twice. Two copies of a two-armed decision
/// whose wrong answer is a numerically plausible page is one copy too many,
/// so the Rust names it once and both fires read it.
///
/// The discriminants are `cuda_fp8.h:185-188`'s declaration order, which is
/// what an unscoped enum with no explicit values means: `__NV_E4M3` is 0 and
/// `__NV_E5M2` is 1. It is `u32` because `kernels::Ty::Fp8Kind` crosses as
/// four bytes, and `abi::emit_device_typecheck` asserts that width in the TU
/// that instantiates the kernels rather than leaving it to this file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Fp8Kind {
    /// `__NV_E4M3` — four exponent bits, three mantissa.
    E4M3 = 0,
    /// `__NV_E5M2` — five exponent bits, two mantissa.
    E5M2 = 1,
}

impl Fp8Kind {
    /// The host ternary at `kv_paged.cu:159-161` and `:394-396`, verbatim:
    ///
    /// ```text
    /// const auto fp8_kind = layer.storage_dtype == DType::FP8_E5M2
    ///     ? __NV_E5M2
    ///     : __NV_E4M3;
    /// ```
    ///
    /// **E4M3 is the fallthrough and not a default.** Every dtype that is not
    /// `Fp8E5M2` lands on E4M3, including dtypes that are not fp8 at all —
    /// that is the C++'s behaviour and it is reproduced rather than tightened,
    /// because the call sites reach here only under
    /// `KvCacheScheme::Fp8PerTensor`, where the cache's own scheme has already
    /// said the pages are fp8. Adding a refusal for the other dtypes would be
    /// a new refusal, and this pass does not invent refusals.
    pub fn of(storage_dtype: DType) -> Self {
        if storage_dtype == DType::Fp8E5M2 {
            Self::E5M2
        } else {
            Self::E4M3
        }
    }
}

/// The fp4 block width when the cache does not state one.
///
/// `kv_paged.cu:196-201` and `:429-431` — `layer.block_size > 0 ?
/// layer.block_size : 16`, written out twice in the C++, once in the writer
/// and once in the reader. **They must agree**: a page written in blocks of
/// one width and read in blocks of another is a wrong answer at every
/// element, so the substitution lives here once and both fires call it.
///
/// 16 is not a tuning constant. It is the arena's layout — an fp4 block scale
/// covers sixteen values — which is why the C++ hard-coded it in both places
/// rather than plumbing it.
fn fp4_block_size(layer: &KvCacheLayerView) -> i32 {
    if layer.block_size > 0 { layer.block_size } else { 16 }
}

/// Whether the quantised append ran, and on which scheme.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum WriteKvQuantised {
    /// One of the four quantised appenders was launched on the caller's
    /// stream.
    Launched,
    /// Nothing was launched, and the reason.
    Declined(QuantisedDecline),
}

/// Every way [`write_kv_to_pages_quantised`] declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantisedDecline {
    /// `layer.scheme` is [`KvCacheScheme::Native`] — `kv_paged.cu:212-213`,
    /// `case KvCacheScheme::Native: break;`.
    ///
    /// **This is a decline and NOT a fallback.** The C++ reached that `break`
    /// only as dead code: `:135` returns on `layer.is_native_bf16()` long
    /// before the switch, and the `break` covers the case where a cache
    /// declares `Native` storage in a dtype that is not bf16. Nothing is
    /// launched in either language. A caller that wants the native path calls
    /// the bf16 appender; this function will not choose it for them.
    NativeScheme,
    /// `total_tokens <= 0`. The C++ had no such test here — it launched
    /// `<<<0, 256>>>`, which CUDA accepts as a no-op — but the two-axis and
    /// three-axis grids below multiply `total_tokens` by other extents, and a
    /// zero grid stated three different ways is three chances to state it
    /// wrong. One test, before any of them.
    NoTokens,
}

/// `attn/kv_paged.cu:155-215` — the quantised half of `write_kv_to_pages`.
///
/// Four launches behind a `switch (layer.scheme)`, three symbols and two
/// instantiations of one of them. Every one of them was unstatable until
/// `kernels::Ty::Fp8Kind` existed, and they moved together because they are
/// arms of one decision: a set of rows covering three of the four schemes is
/// a dispatch that writes a page in the wrong format on the fourth.
///
/// # What this is NOT
///
/// It is not all of `write_kv_to_pages`. That function's first act is
/// `kv_paged.cu:135`, `if (layer.is_native_bf16())`, which delegates to
/// `write_kv_to_pages_bf16` and then — only when `has_envelopes()` and not
/// `hnd_layout` and `total_tokens > 0` — fires
/// `layout::launch_envelope_update_appended_bf16` on the same stream. That
/// second fire has **no device row**: `families/layout.rs:88-141` refuses one
/// because its grid has two meaningful axes. So the native arm stays in C++
/// and this function covers the switch below it, exactly.
///
/// The `first_token` refusal at `kv_paged.cu:130-134` belongs to the native
/// arm too — it throws when `first_token != 0 && !is_native_bf16()`, which is
/// a statement about the delegation and not about any launch here — so it is
/// the caller's and this function does not take a `first_token`.
///
/// # Panics
///
/// If the kernel table and this driver disagree — `fire::hand::fire` panics
/// with the symbol named. A broken JIT is not a decline; that is
/// `fire/gemv.rs`' rule.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_to_pages_quantised(
    layer: KvCacheLayerView,
    k_curr: *const std::ffi::c_void,
    v_curr: *const std::ffi::c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut std::ffi::c_void,
) -> WriteKvQuantised {
    if total_tokens <= 0 {
        return WriteKvQuantised::Declined(QuantisedDecline::NoTokens);
    }

    // `kv_paged.cu:123-125` — the three geometry fields, read once. The C++
    // hoisted them for the same reason and this keeps the read count at one
    // per field, so no arm below can pick up a different `page_size`.
    let page_size = layer.page_size;
    let h_kv = layer.num_kv_heads;
    let d = layer.head_dim;

    // The four CSR arrays every arm passes, in the order every arm passes
    // them. Built once because they are the same four in the same places in
    // all four signatures — `kv_paged.cuh:379-382`, `:415-418`, `:553-556` —
    // and a transposition here would be a silent gather from the wrong array.
    let qo = ArgValue::Ptr(qo_indptr.cast_mut().cast());
    let pi = ArgValue::Ptr(kv_page_indices.cast_mut().cast());
    let pp = ArgValue::Ptr(kv_page_indptr.cast_mut().cast());
    let lp = ArgValue::Ptr(kv_last_page_lens.cast_mut().cast());

    let tokens = total_tokens.unsigned_abs();
    let heads = h_kv.unsigned_abs();

    match layer.scheme {
        // `kv_paged.cu:158-169`:
        //
        // ```text
        // :162   device::write_kv_fp8_per_tensor<<<total_tokens, BLOCK, 0, stream>>>(
        // ```
        //
        // One block per token at 256 — the same `BLOCK` this module already
        // states, and the same grid `LaunchRule::PerRow` fixes.
        KvCacheScheme::Fp8PerTensor => {
            let launch = Launch { grid: [tokens, 1, 1], block: [BLOCK, 1, 1], smem: 0 };
            let values = [
                ArgValue::Ptr(k_curr.cast_mut()),
                ArgValue::Ptr(v_curr.cast_mut()),
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                qo, pi, pp, lp,
                ArgValue::I32(num_requests),
                ArgValue::I32(page_size),
                ArgValue::I32(h_kv),
                ArgValue::I32(d),
                // `kv_paged.cu:159-161`, now stated once. See [`Fp8Kind::of`].
                ArgValue::U32(Fp8Kind::of(layer.storage_dtype) as u32),
            ];
            super::hand::fire("attn::write_kv_fp8_per_tensor", launch, &values, stream);
        }
        // `kv_paged.cu:170-195` — two arms that differ ONLY in the template
        // argument and the scheme that reaches them:
        //
        // ```text
        // :172   const dim3 grid(total_tokens, num_kv_heads);
        // :173   const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
        // :174   device::write_kv_per_token_head<false><<<grid, BLOCK, shmem, stream>>>(
        // :186   const dim3 grid(total_tokens, num_kv_heads);   // :185, identical
        // :187   device::write_kv_per_token_head<true><<<grid, BLOCK, shmem, stream>>>(
        // ```
        //
        // Two symbols and not a `Specialisation`: `UseFp8` is read off the
        // CACHE's scheme and appears nowhere in the kernel's parameter list,
        // so there is no operand for `Term::Is` to test. `families/attn.rs`
        // argues it beside the rows.
        KvCacheScheme::Int8PerTokenHead | KvCacheScheme::Fp8PerTokenHead => {
            let symbol = if layer.scheme == KvCacheScheme::Fp8PerTokenHead {
                "attn::write_kv_fp8_per_token_head"
            } else {
                "attn::write_kv_int8_per_token_head"
            };
            // `kv_paged.cu:173` — `2 * (BLOCK / 32) * sizeof(float)`. Two
            // floats per warp, for the K and V absmax reductions at
            // `kv_paged.cuh:428`. Written as the C++ arithmetic and not as
            // `64` so that a reader can check it against the block width
            // three lines up rather than against a number.
            let smem = 2 * (BLOCK / 32) * (std::mem::size_of::<f32>() as u32);
            let launch = Launch { grid: [tokens, heads, 1], block: [BLOCK, 1, 1], smem };
            let values = [
                ArgValue::Ptr(k_curr.cast_mut()),
                ArgValue::Ptr(v_curr.cast_mut()),
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                ArgValue::Ptr(layer.k_scales),
                ArgValue::Ptr(layer.v_scales),
                qo, pi, pp, lp,
                ArgValue::I32(num_requests),
                ArgValue::I32(page_size),
                ArgValue::I32(h_kv),
                ArgValue::I32(d),
            ];
            super::hand::fire(symbol, launch, &values, stream);
        }
        // `kv_paged.cu:196-211`:
        //
        // ```text
        // :199   const int block_size = layer.block_size > 0 ? layer.block_size : 16;
        // :201   const int blocks = (head_dim + block_size - 1) / block_size;
        // :202   const dim3 grid(total_tokens, num_kv_heads, blocks);
        // :203   device::write_kv_fp4_block<<<grid, 32, 0, stream>>>(
        // ```
        //
        // 32 threads, and the kernel reads all three grid axes
        // (`kv_paged.cuh:563-565`).
        KvCacheScheme::Fp4Block => {
            let block_size = fp4_block_size(&layer);
            let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
            let launch = Launch {
                grid: [tokens, heads, blocks.unsigned_abs()],
                block: [32, 1, 1],
                smem: 0,
            };
            let values = [
                ArgValue::Ptr(k_curr.cast_mut()),
                ArgValue::Ptr(v_curr.cast_mut()),
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                ArgValue::Ptr(layer.k_scales),
                ArgValue::Ptr(layer.v_scales),
                qo, pi, pp, lp,
                ArgValue::I32(num_requests),
                ArgValue::I32(page_size),
                ArgValue::I32(h_kv),
                ArgValue::I32(d),
                ArgValue::I32(block_size),
            ];
            super::hand::fire("attn::write_kv_fp4_block", launch, &values, stream);
        }
        // `kv_paged.cu:212-213`. See [`QuantisedDecline::NativeScheme`] for
        // why this is a decline and not a delegation.
        KvCacheScheme::Native => {
            return WriteKvQuantised::Declined(QuantisedDecline::NativeScheme);
        }
    }

    WriteKvQuantised::Launched
}

/// `attn/kv_paged.cu:393-402` — the per-tensor arm of
/// `dequant_kv_cache_layer_to_bf16_active`.
///
/// ```text
/// :388   const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);
/// :397   device::dequant_fp8_pages_active<<<blocks, BLOCK, 0, stream>>>(
/// ```
///
/// The one arm of that four-arm switch whose row could not be stated before
/// `kernels::Ty::Fp8Kind`; the other three have had rows since the split
/// (`KV_PAGED_SIGS[0..=2]`).
///
/// # Why this arm is separate, and where the rest is
///
/// It was written alone, ahead of its three siblings, because its row was the
/// new thing: `attn::dequant_fp8_pages_active_bf16` could not be stated at all
/// until `kernels::Ty::Fp8Kind` existed, and the `fp8_kind` ternary is what
/// that type is for. The note that stood here said the siblings were *"a
/// transcription away and deliberately not written, because writing them would
/// put a second copy of a live switch in the tree, and the copy that is NOT
/// called is the one that drifts."*
///
/// **They are written now**, in
/// [`dequant_kv_cache_layer_to_bf16_active`] below, and the reason they could
/// not be has been discharged rather than overruled: the live C++ consumer was
/// `driver-cuda/csrc/attn/attention_flashinfer.cu:648`, `:675`, `:1098` and
/// `:1244`, and FA2's host program moves to Rust in the same pass that writes
/// them. This function is now called BY that switch rather than instead of it,
/// so there is one copy again.
///
/// # Panics
///
/// If the kernel table and this driver disagree.
///
/// # Safety
///
/// As [`write_kv_to_pages_quantised`].
pub unsafe fn dequant_fp8_per_tensor_pages_active(
    layer: KvCacheLayerView,
    kv_page_indices: *const u32,
    num_pages_in_batch: i32,
    stream: *mut std::ffi::c_void,
) -> WriteKvQuantised {
    // `kv_paged.cu:384` — `if (layer.is_native_bf16() || num_pages_in_batch <= 0) return;`
    if layer.is_native_bf16() || num_pages_in_batch <= 0 {
        return WriteKvQuantised::Declined(QuantisedDecline::NoTokens);
    }
    if layer.scheme != KvCacheScheme::Fp8PerTensor {
        return WriteKvQuantised::Declined(QuantisedDecline::NativeScheme);
    }

    // `kv_paged.cu:385-388`. `page_elems` is an `int` in the C++ and the
    // product that feeds `logical_n` is widened FIRST — `static_cast<long
    // long>(num_pages_in_batch) * page_elems` — which is the whole reason the
    // kernel's `n` is `long long`: a batch of pages times a page's elements
    // overflows 32 bits at production page counts. The widening is kept in
    // the same place.
    let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
    let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
    let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);

    let launch = Launch {
        grid: [blocks as u32, 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(layer.k_bf16_pages),
        ArgValue::Ptr(layer.v_bf16_pages),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::I64(logical_n),
        ArgValue::I32(page_elems),
        // `kv_paged.cu:394-396` — the second copy of the ternary, now the
        // same call as the first.
        ArgValue::U32(Fp8Kind::of(layer.storage_dtype) as u32),
    ];

    super::hand::fire("attn::dequant_fp8_pages_active_bf16", launch, &values, stream);
    WriteKvQuantised::Launched
}

// ===========================================================================
// THE BEAM-REPAIR CELL MOVE, `kv_paged.cu:352-378` — PORTED, AND ITS C++ IS GONE
// ===========================================================================

/// Whether the cell move ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum CopyKvCells {
    /// `copy_kv_cells<HND>` was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and the reason.
    Declined(CopyDecline),
}

/// The one way [`copy_kv_cells_bf16`] declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyDecline {
    /// `N <= 0` — `kv_paged.cu:364`. An empty move.
    NoCells,
}

/// Beam-repair cell moves, per layer, disjoint spans by contract.
///
/// The contract paragraph is `kv_paged.hpp`'s, carried here verbatim when its
/// declaration was deleted rather than paraphrased, because every clause of
/// it is load-bearing on the CALLER:
///
/// > Compaction primitive (Design-B lazy GC): move N token KV cells (single
/// > layer) from explicit (src physical page, src offset) → (dst physical
/// > page, dst offset) targets, for both K and V. Raw element copy — correct
/// > because the KV cache is stored POST-RoPE (slot = pure storage; positions
/// > live in the per-beam mask). Caller guarantees DISJOINT src/dst spans
/// > (in-place two-pointer) so one pass needs no scratch. Invoke per layer to
/// > move all layers. Native-bf16 KV.
///
/// The disjointness is the one the kernel cannot check and the driver cannot
/// either: `dst_page`/`dst_off` and `src_page`/`src_off` are device arrays,
/// and a launch that read them to prove the spans apart would cost the round
/// trip the primitive exists to avoid.
///
/// **This one is finished, not staged.** Its whole consumer set was one Rust
/// call — `serve::transfer.rs` through the generated
/// `ffi::pie_k_attn_copy_kv_cells_bf16` — so the move was Rust-to-Rust and
/// the C++ launcher, its `.hpp` declaration and its `table/driver_internal.rs`
/// row went in the same edit. The row had to go WITH the launcher: a
/// `driver_internal` row states `operands` and is in neither
/// `device::JIT_DISPATCHED` nor `execution::RUST_SERVED`, so `emit_c_shim`
/// would keep writing a `pie_k_attn_copy_kv_cells_bf16` forwarder onto a
/// definition that no longer exists. It cannot be routed instead: a
/// `driver_internal` row is not in `table::TABLES`, so `table::sig` cannot
/// resolve it and `every_taken_over_row_is_stated` refuses `RUST_SERVED`;
/// and its operands are all `Source::Unbound`, so `emit_dispatch` would skip
/// it whole and drop the arm too. Deletion is the only honest close, and the
/// consumer set makes it a true one.
///
/// The two DEVICE rows stay — `attn::copy_kv_cells_bf16#hnd` and `#nhd`,
/// `families/attn.rs:3293`/`:3301` — because they are what this fires. So is
/// `SPECIALISATIONS`' `COPY_KV_CELLS`, whose base `attn::copy_kv_cells_bf16`
/// still resolves through `unit_of`.
///
/// `layer` is the cache; `dst_page`/`dst_off` and `src_page`/`src_off` are
/// the per-cell physical page and offset arrays, `N` cells each.
///
/// # Panics
///
/// If the cache is not native bf16 — `kv_paged.cu:360-363` threw, and it is a
/// condition on the CALLER rather than on the launch, so it may not be
/// answered with a decline. Also if the kernel table and this driver
/// disagree.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
pub unsafe fn copy_kv_cells_bf16(
    layer: KvCacheLayerView,
    dst_page: *const u32,
    dst_off: *const u32,
    src_page: *const u32,
    src_off: *const u32,
    n: i32,
    stream: *mut std::ffi::c_void,
) -> CopyKvCells {
    // `kv_paged.cu:360-363`, and in that order: the scheme is checked before
    // the extent, so a caller that passes a quantised cache is wrong whether
    // or not it also passes zero cells.
    assert!(
        layer.is_native_bf16(),
        "attn::copy_kv_cells_bf16 requires native bf16 KV cache"
    );
    // `kv_paged.cu:364`.
    if n <= 0 {
        return CopyKvCells::Declined(CopyDecline::NoCells);
    }

    // `kv_paged.cu:366` — `if (layer.hnd_layout)`. One `Term::Is` over one
    // operand, which is what `SPECIALISATIONS`' `COPY_KV_CELLS` states, so
    // the argument list below is built once.
    let symbol = if layer.hnd_layout {
        "attn::copy_kv_cells_bf16#hnd"
    } else {
        "attn::copy_kv_cells_bf16#nhd"
    };

    // `kv_paged.cu:367` / `:373`: `<<<N, BLOCK, 0, stream>>>`, with
    // `constexpr int BLOCK = 256` at `:365` — the same 256 this module
    // already states and the same `LaunchRule::PerRow` fixes.
    let launch = Launch { grid: [n.unsigned_abs(), 1, 1], block: [BLOCK, 1, 1], smem: 0 };

    // The operand order is the `__global__`'s, not the launcher's: the
    // launcher took the view whole and carried a stream, and the row takes
    // the two page pointers out of it.
    let values = [
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(dst_page.cast_mut().cast()),
        ArgValue::Ptr(dst_off.cast_mut().cast()),
        ArgValue::Ptr(src_page.cast_mut().cast()),
        ArgValue::Ptr(src_off.cast_mut().cast()),
        ArgValue::I32(n),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.num_kv_heads),
        ArgValue::I32(layer.head_dim),
    ];

    super::hand::fire(symbol, launch, &values, stream);
    CopyKvCells::Launched
}

// ===========================================================================
// THE NATIVE bf16 APPEND, `kv_paged.cu:66-107` AND `:109-153`
// ===========================================================================

/// `kv_paged.cu:151` — the `max_touched` bound the append's envelope refresh
/// is launched over.
///
/// ```text
/// (total_tokens + page_size - 1) / page_size + num_requests
/// ```
///
/// The ceiling of the token count in pages, plus one straddle per request:
/// every request can start mid-page, so the pages a fire touched is at most
/// the pages its tokens fill plus one more per request. **It is a BOUND and
/// nothing measures it** — blocks past a request's real page span early out
/// in the kernel — which is exactly why no `LaunchRule` states the grid it
/// feeds. It is public because it is the whole reason
/// [`kernels_cuda_new::x::layout::envelope_update_appended`] takes a
/// `max_touched` it cannot derive.
///
/// Returns zero when `page_size <= 0`, which is a layer the callers below
/// have already declined on.
#[must_use]
pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
    if page_size <= 0 {
        return 0;
    }
    (total_tokens + page_size - 1) / page_size + num_requests
}

/// Whether the native bf16 append ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum WriteKvNative {
    /// The append was launched on the caller's stream, and — when the layer
    /// asked for it — the envelope refresh behind it.
    Launched,
    /// Nothing was launched, and the reason.
    Declined(NativeDecline),
}

/// Every way the native appenders decline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeDecline {
    /// `kv_paged.cu:84` — `total_tokens - first_token <= 0`. Note the C++
    /// tests the DIFFERENCE, not `total_tokens`: a fire whose leading rows
    /// were already written by a fused kernel and which has nothing left is
    /// this, not an empty fire.
    NoTokensLeft,
    /// `kv_paged.cu:320` — `B <= 0`, the explicit form's empty extent.
    NoRows,
}

/// `attn/kv_paged.cu:66` — `write_kv_to_pages_bf16`, and `:109`'s native arm.
///
/// The CSR append for a native-bf16 cache: one block per token, scattering
/// `k_curr`/`v_curr` rows into the pages `kv_page_indices`/`kv_page_indptr`
/// name.
///
/// ```text
/// :87   device::write_kv<true> <<<launch_tokens, BLOCK, 0, stream>>>(...)
/// :97   device::write_kv<false><<<launch_tokens, BLOCK, 0, stream>>>(...)
/// ```
///
/// with `launch_tokens = total_tokens - first_token` (`:83`) and
/// `BLOCK = 256` (`:82`).
///
/// **The grid is not `total_tokens` and that is the whole point of
/// `first_token`.** A partial append writes the TAIL: the leading rows were
/// written by a fused QKV kernel, the grid covers what is left, and the
/// kernel offsets its own token index by `first_token`. `LaunchRule::PerRow`
/// would state `total_tokens`, which is why this launch is driver-owned even
/// though its rows carry a rule.
///
/// # Envelope maintenance rides the append
///
/// `kv_paged.cu:144-152`, reproduced here rather than left to the caller
/// because it is stream-ordered against the write above it and a caller that
/// forgot it would leave stale envelopes rather than fail:
///
/// ```text
/// if (layer.has_envelopes() && !layer.hnd_layout && total_tokens > 0)
///     launch_envelope_update_appended_bf16(...)
/// ```
///
/// Opt-in: `has_envelopes()` is false unless a program declared it needs
/// them. `!hnd_layout` because the envelope kernels index `[page, kv_head,
/// head_dim]` pages and the head-major layout is a different stride.
///
/// # `first_token` and the refusal that belongs to it
///
/// `kv_paged.cu:130-134` throws when `first_token != 0` on a non-native
/// cache. That test lives in the DISPATCHER — [`write_kv_to_pages`] — and
/// not here, because this function is only ever reached on a native cache
/// and the throw is a statement about the delegation.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_to_pages_bf16(
    layer: KvCacheLayerView,
    k_curr: *const std::ffi::c_void,
    v_curr: *const std::ffi::c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut std::ffi::c_void,
    row_valid: *const u8,
    first_token: i32,
) -> WriteKvNative {
    // `kv_paged.cu:83-84`.
    let launch_tokens = total_tokens - first_token;
    if launch_tokens <= 0 {
        return WriteKvNative::Declined(NativeDecline::NoTokensLeft);
    }

    // `kv_paged.cu:85` — `if (hnd_layout)`. Two instantiations, one argument
    // list; `device::SPECIALISATIONS`' `WRITE_KV` states the same `Term::Is`
    // over the base row's sixteenth operand.
    let symbol = if layer.hnd_layout {
        "attn::write_kv_bf16#hnd"
    } else {
        "attn::write_kv_bf16#nhd"
    };

    let launch = Launch {
        grid: [launch_tokens.unsigned_abs(), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };

    // `families::attn`'s `write_kv_hnd`/`_nhd` row, which is the
    // `__global__`'s parameter list. `win` is the tenth and is NULL here —
    // the C++ passes `/*win=*/nullptr` at `:92` and `:102`, because a CSR
    // append has no device window; the windowed form is
    // [`write_kv_explicit_bf16_devwin`].
    let values = [
        ArgValue::Ptr(k_curr.cast_mut()),
        ArgValue::Ptr(v_curr.cast_mut()),
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::Ptr(std::ptr::null_mut()),
        ArgValue::I32(num_requests),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.num_kv_heads),
        ArgValue::I32(layer.head_dim),
        ArgValue::I32(first_token),
    ];
    super::hand::fire(symbol, launch, &values, stream);

    // `kv_paged.cu:144-152`. Same stream, after the write, so the refresh is
    // ordered behind the pages it describes.
    if layer.has_envelopes() && !layer.hnd_layout && total_tokens > 0 {
        let _ = unsafe {
            kernels_cuda_new::x::layout::envelope_update_appended(
                layer.k_pages.cast(),
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                layer.k_env_min.cast(),
                layer.k_env_max.cast(),
                num_requests,
                max_touched_pages(total_tokens, num_requests, layer.page_size),
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            )
        };
    }
    WriteKvNative::Launched
}

/// `attn/kv_paged.cu:109` — `write_kv_to_pages`, the whole dispatcher.
///
/// One `if (layer.is_native_bf16())` over two ports that already exist:
/// [`write_kv_to_pages_bf16`] above and [`write_kv_to_pages_quantised`]. It
/// is here rather than at the call site because the REFUSAL between them is
/// the dispatcher's, not either arm's.
///
/// # Panics
///
/// `kv_paged.cu:130-134`, `if (first_token != 0 && !layer.is_native_bf16())`
/// — a `throw std::runtime_error`, so a panic naming the symbol. A non-zero
/// `first_token` means the leading rows were written by a fused kernel that
/// only exists for the native bf16 cache; on any other scheme a partial write
/// here leaves the prefix rows holding garbage from `k_curr` rows nobody
/// filled. **This is not a decline** — it is a caller that asked for a thing
/// that does not exist, and there is no second answer to give.
///
/// Also if the kernel table and this driver disagree; see
/// [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_to_pages(
    layer: KvCacheLayerView,
    k_curr: *const std::ffi::c_void,
    v_curr: *const std::ffi::c_void,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut std::ffi::c_void,
    row_valid: *const u8,
    first_token: i32,
) -> WriteKvNative {
    assert!(
        first_token == 0 || layer.is_native_bf16(),
        "attn::write_kv_to_pages: partial (first_token) writes require the \
         native bf16 cache"
    );
    if layer.is_native_bf16() {
        return unsafe {
            write_kv_to_pages_bf16(
                layer,
                k_curr,
                v_curr,
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                total_tokens,
                num_requests,
                stream,
                row_valid,
                first_token,
            )
        };
    }
    // The quantised switch, already ported. Its `Declined` answers are its
    // own and are NOT translated into this function's — a caller that wants
    // to know which scheme declined calls it directly.
    match unsafe {
        write_kv_to_pages_quantised(
            layer,
            k_curr,
            v_curr,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            total_tokens,
            num_requests,
            stream,
        )
    } {
        WriteKvQuantised::Launched => WriteKvNative::Launched,
        WriteKvQuantised::Declined(QuantisedDecline::NoTokens) => {
            WriteKvNative::Declined(NativeDecline::NoTokensLeft)
        }
        // `kv_paged.cu:212-213`, `case KvCacheScheme::Native: break;` — dead
        // code in the C++, reached only by a cache declaring `Native` storage
        // in a dtype that is not bf16. Nothing was launched there either.
        WriteKvQuantised::Declined(QuantisedDecline::NativeScheme) => {
            WriteKvNative::Declined(NativeDecline::NoRows)
        }
    }
}

/// `attn/kv_paged.cu:304` — `write_kv_explicit_bf16`, the host-window form.
///
/// The explicit append: no CSR, only the per-row `w_page`/`w_off` descriptor
/// the program wrote. `B` is the row count and the grid.
///
/// ```text
/// :321  device::write_kv_explicit<true> <<<B, BLOCK, 0, stream>>>(...)
/// :329  device::write_kv_explicit<false><<<B, BLOCK, 0, stream>>>(...)
/// ```
///
/// # Envelope maintenance rides this append too
///
/// `kv_paged.cu:339-347`. **The CSR-derived path cannot be reused** — there
/// is no page list here — so this one calls
/// [`kernels_cuda_new::x::layout::envelope_merge_written`] with the per-row
/// descriptor instead of
/// [`kernels_cuda_new::x::layout::envelope_update_appended`] with the page
/// CSR. That is the whole
/// reason the envelope tier has two merge points and not one.
///
/// # Panics
///
/// If the cache is not native bf16 — `kv_paged.cu:314-317`, a
/// `throw std::runtime_error`, so a panic naming the symbol rather than a
/// decline. Also if the kernel table and this driver disagree; see
/// [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_explicit_bf16(
    layer: KvCacheLayerView,
    k_curr: *const std::ffi::c_void,
    v_curr: *const std::ffi::c_void,
    w_page: *const u32,
    w_off: *const u32,
    b: i32,
    stream: *mut std::ffi::c_void,
    row_valid: *const u8,
) -> WriteKvNative {
    // `kv_paged.cu:314`, before the empty-extent test and in that order.
    assert!(
        layer.is_native_bf16(),
        "attn::write_kv_explicit_bf16 requires native bf16 KV cache"
    );
    // `kv_paged.cu:320`.
    if b <= 0 {
        return WriteKvNative::Declined(NativeDecline::NoRows);
    }

    let symbol = if layer.hnd_layout {
        "attn::write_kv_explicit_bf16_dev#hnd"
    } else {
        "attn::write_kv_explicit_bf16_dev#nhd"
    };
    let launch = Launch {
        grid: [b.unsigned_abs(), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(k_curr.cast_mut()),
        ArgValue::Ptr(v_curr.cast_mut()),
        ArgValue::Ptr(layer.k_pages),
        ArgValue::Ptr(layer.v_pages),
        ArgValue::Ptr(w_page.cast_mut().cast()),
        ArgValue::Ptr(w_off.cast_mut().cast()),
        ArgValue::Ptr(row_valid.cast_mut().cast()),
        ArgValue::I32(b),
        ArgValue::I32(layer.page_size),
        ArgValue::I32(layer.num_kv_heads),
        ArgValue::I32(layer.head_dim),
    ];
    super::hand::fire(symbol, launch, &values, stream);

    // `kv_paged.cu:344-347`. `k_curr` and not `layer.k_pages`: the merge
    // reads the ROWS just written, which it can address directly because
    // `w_page`/`w_off` say where each one landed.
    if layer.has_envelopes() && !layer.hnd_layout {
        let _ = unsafe {
            kernels_cuda_new::x::layout::envelope_merge_written(
                k_curr.cast(),
                w_page,
                w_off,
                kernels_cuda_new::x::abi::MaybeConst::new(row_valid),
                layer.k_env_min.cast(),
                layer.k_env_max.cast(),
                b,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            )
        };
    }
    WriteKvNative::Launched
}

// ===========================================================================
// THE TWO PAGE-VIEW BUILDERS, `kv_paged.cu:309` AND `:324`
// ===========================================================================
//
// **Neither carries an `Execution` classification, and that is §58.**
//
// A single launch with no choice and no loop needs none: `fire/attn_score.rs`
// fires a row and carries none either. §59.2 declined to transcribe these two
// because it could not see which classification they wanted, and the answer
// is that the question does not apply. What they wanted was for their
// `table::attn` rows to go — both were UNSOURCED (`Source::Unbound` on every
// operand, so `crate::abi` skipped them whole and no dispatch was ever
// generated from either) and their two `dsl::cuda` wrappers had no caller in
// `crates/model/src`. Row and wrapper deleted together, which is §54's rule.
//
// The DEVICE rows stay and are what these fire: `families/attn.rs`'
// `build_window_page_view` on `LaunchRule::Single` and `build_full_split_view`
// on `SingleWarp`. Fired through `super::hand::fire` with a driver-owned
// `Launch` rather than through `bind::jit::fire`, because there is no `Dims`
// here — a caller planning a windowed read has a batch count and a page CSR,
// not a fire's rectangle.

/// Whether a page-view build ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum PageView {
    /// The builder was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and which extent was empty.
    Declined(PageViewDecline),
}

/// Every way the two builders decline. Each is a `return` in the C++.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageViewDecline {
    /// `kv_paged.cu:318` — `R <= 0`, an empty batch.
    NoRequests,
    /// `kv_paged.cu:318` — `keep_pages <= 0`. A window that keeps no pages is
    /// not a window; the C++ declined rather than writing an empty CSR, and
    /// so does this.
    NoKeptPages,
    /// `kv_paged.cu:335` — `splits <= 0`.
    NoSplits,
    /// `kv_paged.cu:335` — `page_size <= 0`.
    NoPageSize,
}

/// `attn/kv_paged.cu:309` — `build_window_page_view`.
///
/// Rewrites a page CSR to keep only the last `keep_pages` pages of each
/// request, which is how a sliding-window layer reads a full-length cache
/// without copying it.
///
/// ```text
/// :319   device::build_window_page_view<<<1, 256, 0, stream>>>(
/// :320       src_indices, src_indptr, keep_pages, dst_indptr, dst_indices, R);
/// ```
///
/// One block of 256, which is `LaunchRule::Single` to the digit. Stated here
/// rather than taken from the rule because the rule needs a
/// `kernels_cuda_new::Dims` and this caller has none.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_window_page_view(
    src_indices: *const u32,
    src_indptr: *const u32,
    keep_pages: i32,
    dst_indptr: *mut u32,
    dst_indices: *mut u32,
    r: i32,
    stream: *mut std::ffi::c_void,
) -> PageView {
    // `kv_paged.cu:318`, split so the caller learns which extent was empty.
    if r <= 0 {
        return PageView::Declined(PageViewDecline::NoRequests);
    }
    if keep_pages <= 0 {
        return PageView::Declined(PageViewDecline::NoKeptPages);
    }
    let launch = Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0 };
    let values = [
        ArgValue::Ptr(src_indices.cast_mut().cast()),
        ArgValue::Ptr(src_indptr.cast_mut().cast()),
        ArgValue::I32(keep_pages),
        ArgValue::Ptr(dst_indptr.cast()),
        ArgValue::Ptr(dst_indices.cast()),
        ArgValue::I32(r),
    ];
    super::hand::fire("attn::build_window_page_view", launch, &values, stream);
    PageView::Launched
}

/// `attn/kv_paged.cu:324` — `build_full_split_view`.
///
/// Describes one request's page span as `splits` consecutive sub-requests, so
/// a long prefill can be attended in pieces against one page table.
///
/// ```text
/// :335   device::build_full_split_view<<<1, 32, 0, stream>>>(
/// :336       src_indptr, src_last_page_len, splits, page_size,
/// :337       dst_indptr, dst_indices, dst_last, src_indices);
/// ```
///
/// **32 and not 256, and the kernel says why** — the measurement is carried
/// here rather than consumed by the port: `kv_paged.cuh:842` is
/// `if (threadIdx.x != 0) return;` and the whole body is a serial walk over
/// `splits`. Every thread but one exits immediately, so the launch is one
/// warp because a warp is the smallest thing the hardware schedules. That is
/// a fact about the DEVICE, which is why `LaunchRule::SingleWarp` fixes 32
/// rather than taking it from a `Dims` field, and why this constant is not a
/// tuning knob.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_full_split_view(
    src_indptr: *const u32,
    src_last_page_len: *const u32,
    splits: i32,
    page_size: i32,
    dst_indptr: *mut u32,
    dst_indices: *mut u32,
    dst_last: *mut u32,
    src_indices: *const u32,
    stream: *mut std::ffi::c_void,
) -> PageView {
    // `kv_paged.cu:335`.
    if splits <= 0 {
        return PageView::Declined(PageViewDecline::NoSplits);
    }
    if page_size <= 0 {
        return PageView::Declined(PageViewDecline::NoPageSize);
    }
    let launch = Launch { grid: [1, 1, 1], block: [32, 1, 1], smem: 0 };
    // The operand order is the `__global__`'s, which puts `src_indices` LAST
    // — after three outputs — and not beside `src_indptr` where a reader
    // expects it. Transcribed rather than tidied: the row states the same
    // order and `Args::bind` checks it.
    let values = [
        ArgValue::Ptr(src_indptr.cast_mut().cast()),
        ArgValue::Ptr(src_last_page_len.cast_mut().cast()),
        ArgValue::I32(splits),
        ArgValue::I32(page_size),
        ArgValue::Ptr(dst_indptr.cast()),
        ArgValue::Ptr(dst_indices.cast()),
        ArgValue::Ptr(dst_last.cast()),
        ArgValue::Ptr(src_indices.cast_mut().cast()),
    ];
    super::hand::fire("attn::build_full_split_view", launch, &values, stream);
    PageView::Launched
}

// ===========================================================================
// `dequant_kv_cache_layer_to_bf16_active`, `kv_paged.cu:191-261` — THE WHOLE
// SWITCH, AND THE LAST THING HOLDING THE ARCHIVE CENSUS ABOVE ZERO
// ===========================================================================
//
// The three arms below were deliberately absent until now, and the note on
// `dequant_fp8_per_tensor_pages_active` above says why: the C++ switch had a
// live C++ consumer — `driver-cuda/csrc/attn/attention_flashinfer.cu:648`,
// `:675`, `:1098`, `:1244`, four call sites of C++ calling C++ by symbol with
// no shim between — so a Rust copy would have been a second copy of a live
// switch, and *"the copy that is NOT called is the one that drifts."*
//
// That is no longer true. FA2's host program moves to Rust in this pass, the
// four call sites become [`dequant_kv_cache_layer_to_bf16_active`] below, and
// the C++ switch's consumer set empties. The prohibition is discharged by the
// thing it was waiting for rather than overridden.
//
// The paragraph is kept because its RULE is right and outlives this case: do
// not transcribe a live switch into a second language until the first copy is
// dead or dies in the same change.

/// Every arm of `dequant_kv_cache_layer_to_bf16_active`, `kv_paged.cu:204-259`.
///
/// One `match` where the C++ had one `switch`, on the same discriminant, in
/// the same order, launching the same four kernels at the same geometry.
///
/// # The geometry, once, for all four
///
/// ```text
/// :197   if (layer.is_native_bf16() || num_pages_in_batch <= 0) return;
/// :198   constexpr int BLOCK = 256;
/// :199   const int page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
/// :200   const long long logical_n =
/// :201       static_cast<long long>(num_pages_in_batch) * page_elems;
/// :202   const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);
/// ```
///
/// All four launches are `<<<blocks, BLOCK, 0, stream>>>` — `:209`, `:219`,
/// `:231`, `:246` — so the grid is computed once here and not per arm. The
/// widening on `:201` is load-bearing and is kept in the same place: a page
/// count times a page's elements overflows 32 bits at production page counts,
/// which is why the kernels take `long long`.
///
/// # What the C++ did at the end and this does not
///
/// `:260` is `CUDA_CHECK(cudaGetLastError())`. There is no equivalent here and
/// that is deliberate: `hand::fire` reports its own launch failures against
/// the symbol it fired, which is strictly more information than a
/// `cudaGetLastError` attributing the fault to whichever of four kernels ran
/// last. A synchronous check after every dequant would also serialise a path
/// that runs once per layer per step.
///
/// # Panics
///
/// If the kernel table and this driver disagree — [`super::hand::fire`]'s
/// contract, and the reason a broken JIT panics with the symbol named rather
/// than declining.
///
/// # Safety
///
/// As [`write_kv_to_pages_quantised`]: every pointer in `layer` must be a
/// device address of the extent the layer describes, `kv_page_indices` must
/// hold `num_pages_in_batch` entries, and `stream` must outlive the launch.
pub unsafe fn dequant_kv_cache_layer_to_bf16_active(
    layer: KvCacheLayerView,
    kv_page_indices: *const u32,
    num_pages_in_batch: i32,
    stream: *mut std::ffi::c_void,
) -> WriteKvQuantised {
    // `:197`. Both halves, in the C++'s order: a native-bf16 layer has nothing
    // to dequantise and an empty batch has nothing to dequantise it from.
    if layer.is_native_bf16() || num_pages_in_batch <= 0 {
        return WriteKvQuantised::Declined(QuantisedDecline::NoTokens);
    }

    let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
    let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
    let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    let launch = Launch { grid: [blocks as u32, 1, 1], block: [BLOCK, 1, 1], smem: 0 };

    // The four arms share their first six operands — the two quantised page
    // planes, the two scale planes and the two bf16 outputs — and diverge only
    // in what follows. Named once so a mis-ordered arm is visible as a
    // difference rather than as six lines that look like the others.
    let indices = ArgValue::Ptr(kv_page_indices.cast_mut().cast());

    match layer.scheme {
        // `:205-216`. Already written, and called rather than repeated: this
        // is the arm whose row needed `kernels::Ty::Fp8Kind`, and its `fp8_kind`
        // ternary is the argument that row exists to carry.
        KvCacheScheme::Fp8PerTensor => unsafe {
            dequant_fp8_per_tensor_pages_active(layer, kv_page_indices, num_pages_in_batch, stream)
        },

        // `:217-228`. Per-token-per-head fp8: a scale plane per (token, head)
        // instead of one per tensor, so the kernel needs the page geometry to
        // find a scale and takes `page_size`, `h_kv` and `d` where the
        // per-tensor arm took a flat `page_elems`.
        KvCacheScheme::Fp8PerTokenHead => {
            let values = [
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                ArgValue::Ptr(layer.k_scales),
                ArgValue::Ptr(layer.v_scales),
                ArgValue::Ptr(layer.k_bf16_pages),
                ArgValue::Ptr(layer.v_bf16_pages),
                indices,
                ArgValue::I64(logical_n),
                ArgValue::I32(layer.page_size),
                ArgValue::I32(layer.num_kv_heads),
                ArgValue::I32(layer.head_dim),
            ];
            super::hand::fire(
                "attn::dequant_fp8_per_token_head_pages_active_bf16",
                launch,
                &values,
                stream,
            );
            WriteKvQuantised::Launched
        }

        // `:229-240`. Byte-for-byte the arm above with a different element
        // type on the page planes — `std::int8_t` at `:232-233` where fp8 has
        // `__nv_fp8_storage_t`. Two symbols and not one template because the
        // rows are `I8s` and `U8s` and a single row could not say which
        // (`families/attn.rs:3705-3706` argues it).
        KvCacheScheme::Int8PerTokenHead => {
            let values = [
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                ArgValue::Ptr(layer.k_scales),
                ArgValue::Ptr(layer.v_scales),
                ArgValue::Ptr(layer.k_bf16_pages),
                ArgValue::Ptr(layer.v_bf16_pages),
                indices,
                ArgValue::I64(logical_n),
                ArgValue::I32(layer.page_size),
                ArgValue::I32(layer.num_kv_heads),
                ArgValue::I32(layer.head_dim),
            ];
            super::hand::fire(
                "attn::dequant_int8_per_token_head_pages_active_bf16",
                launch,
                &values,
                stream,
            );
            WriteKvQuantised::Launched
        }

        // `:241-256`. The only arm with a twelfth operand, and the only one
        // whose `n` is LOGICAL rather than physical: an fp4 page holds two
        // values per byte, so the grid covers twice the bytes it reads and
        // every address inside the kernel is derived by halving. The row keeps
        // the kernel's name for it (`families/attn.rs:3314-3318`).
        KvCacheScheme::Fp4Block => {
            // `:242-244`. The default is the kernel's, not a policy: a layer
            // that never stated a block size is an fp4 layer at NVFP4's own
            // 16-element block.
            let block_size = if layer.block_size > 0 { layer.block_size } else { 16 };
            let values = [
                ArgValue::Ptr(layer.k_pages),
                ArgValue::Ptr(layer.v_pages),
                ArgValue::Ptr(layer.k_scales),
                ArgValue::Ptr(layer.v_scales),
                ArgValue::Ptr(layer.k_bf16_pages),
                ArgValue::Ptr(layer.v_bf16_pages),
                indices,
                ArgValue::I64(logical_n),
                ArgValue::I32(layer.page_size),
                ArgValue::I32(layer.num_kv_heads),
                ArgValue::I32(layer.head_dim),
                ArgValue::I32(block_size),
            ];
            super::hand::fire("attn::dequant_fp4_pages_active_bf16", launch, &values, stream);
            WriteKvQuantised::Launched
        }

        // `:257-258`, `case KvCacheScheme::Native: break;`.
        //
        // Unreachable in the C++ — `:197` returns on `is_native_bf16()` first
        // — and reachable here only for a cache declaring `Native` storage in
        // a dtype that is not bf16. It declines, with the same reasoning
        // `QuantisedDecline::NativeScheme` already carries: nothing is
        // launched in either language, and a caller that wants the native path
        // asks for it by name.
        KvCacheScheme::Native => WriteKvQuantised::Declined(QuantisedDecline::NativeScheme),
    }
}
