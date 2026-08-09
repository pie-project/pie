//! DeepSeek-V4's hash router, in Rust. **`moe/dsv4_routing.cu` is deleted.**
//!
//! One launcher, and it was the last one that file had. Its `__global__` is
//! NVRTC's, out of `kernels-cuda-new/csrc/src/moe/dsv4_routing.cuh`, which
//! also still carries the sibling router `topk_sqrtsoftplus` as device text —
//! §43.9 deleted THAT launcher because its row is in
//! `device::JIT_DISPATCHED`, the shim emits no entry and nothing reached it.
//!
//! The deleted file's own header said what it was for, and it is the sentence
//! this module inherits:
//!
//! > Both `__global__`s moved to `dsv4_routing.cuh`, which this file includes
//! > and NVRTC compiles from a string — one definition, two compilers. What
//! > stays is what a JIT has no use for: the emptiness guards and the two
//! > `<<<>>>`s that name a grid.
//!
//! There is now one compiler and no `<<<>>>`. The guard and the grid are
//! below.
//!
//! # It is a `Walk`, and `tid2eid` is why
//!
//! `execution::WALKED` carries `moe::hash_route_lookup` with
//! `Control::Supplies { what: "tid2eid and vocab_size" }`. The geometry was
//! never the blocker: `LaunchRule::RowsFlat` was ported FROM this launcher's
//! `<<<>>>` and states it digit for digit, and the device row
//! `moe::hash_route_lookup_dev` still carries it. What no `Source` can name
//! is a `[vocab, K]` table keyed by TOKEN ID — the fire's rectangle does not
//! carry the vocabulary, so neither the table nor its first extent is an
//! extent of any value the statement mentions. `families/moe.rs` says the
//! same at that row: *"a guessed `Weight(0)` for a table no declaration has
//! named yet would bind the wrong buffer with nothing to report it."*
//!
//! # Two names for one kernel
//!
//! `a_walk_is_only_a_walk` asserts a walked symbol has no unit, so the ABI
//! symbol and the device row are different strings:
//!
//! ```text
//!   hash_route_lookup              -> hash_route_lookup_dev
//! ```
//!
//! `fire/moe_dispatch.rs` carries the same table for its eight, and
//! `families/moe.rs` is where all twelve mappings are written out once.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `dsv4_routing.cu:20`'s `kDsv4Block`, and the deleted file called it *"a
/// HOST constant because that is all it is now"*.
///
/// It was read by both of that file's launchers and means two different
/// things in them. In `topk_sqrtsoftplus_bf16`, deleted by §43.9, it was a
/// block PER TOKEN — the router stages `num_experts` logits and reduces them.
/// Here it is a flat tile width over tokens, one THREAD each. The constant
/// survives because this launch survives; the other reading went with the
/// launcher that held it.
const DSV4_BLOCK: u32 = 256;

/// Whether the router fired or refused, and on which term.
///
/// `#[must_use]` for `fire/gemv.rs`'s reason: *"it declined"* must not be
/// spellable the way *"it ran"* is. A [`Route::Declined`] is never a
/// fallback — no other router runs in its place and `topk_idx` and `topk_w`
/// are left exactly as the caller had them, which for a first forward means
/// uninitialised.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The kernel was submitted to the stream.
    Launched,
    /// No kernel was submitted, on the named term.
    Declined(Decline),
}

/// One variant per term of the launcher's single `return`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `tokens <= 0` — `dsv4_routing.cu:36`. An empty batch.
    NoTokens,
    /// `top_k <= 0` — the same line, and NOT redundant with it.
    ///
    /// The kernel's own guard is `if (n >= tokens) return;` and tests nothing
    /// about `K`. A batch of real tokens routed to zero experts would launch
    /// a full grid that writes no `topk_idx` entry and no `topk_w` entry, and
    /// the mixture downstream would read whatever was in those buffers. The
    /// C++ tested it on the host for that reason and so does this.
    NoExperts,
}

/// Routes each token by a checkpoint hash table, weighting by the router
/// logits at the hashed experts.
///
/// `dsv4_routing.cu:22-46`, the whole body:
///
/// ```text
/// if (tokens <= 0 || top_k <= 0) return;
/// // One thread per token, not one block: the kernel's whole body is a table
/// // read and a K-long gather.
/// const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
/// device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(
///     token_ids, tid2eid,
///     static_cast<const device::bf16*>(logits),
///     topk_idx, topk_w,
///     tokens, vocab_size, num_experts, top_k,
///     renormalize, routed_scaling_factor);
/// ```
///
/// # The indices come from the table; the WEIGHTS still come from the logits
///
/// This is the half of the kernel that a name like "hash routing" hides, and
/// it is the part a reimplementation gets wrong. DeepSeek-V4 fixes each token's
/// expert SET by a hash of its id, so `topk_idx` is a gather out of
/// `tid2eid` and the router logits never choose anything. But `topk_w` is
/// still `sqrt(softplus(logits))` READ AT THOSE HASHED INDICES, renormalised
/// across `K` when `renormalize` is set and scaled by
/// `routed_scaling_factor`. Substituting a uniform `1 / K` — which looks
/// harmless once the indices are fixed — is a different model, not a faster
/// path to the same one.
///
/// # One thread per token, and the launcher said why in its own words
///
/// > One thread per token, not one block: the kernel's whole body is a table
/// > read and a K-long gather.
///
/// The sibling `topk_sqrtsoftplus` gives a token a whole block because it
/// stages `num_experts` logits in shared memory and reduces them; nothing
/// here is reduced. `families/moe.rs` works the alternatives through at the
/// device row: `Rms` would launch 256 times the blocks and idle 255 lanes of
/// each, `Elementwise` would launch `top_k` times too many. **Both would
/// produce the right output**, because `if (n >= tokens) return;` is the
/// kernel's first line — which is exactly why the row waited for a rule that
/// divides the rows and stops, and `RowsFlat` is that rule and has this row
/// alone.
///
/// `tokens` is spent on the grid AND passed as an operand, because the last
/// block is partial and the guard is the kernel's own.
///
/// # `kDsv4MaxExperts` is NOT a precondition here
///
/// The deleted file carried one host precondition — *"`num_experts >
/// device::kDsv4MaxExperts` returns without launching, because a wider router
/// would overrun the kernel's static shared arrays"* — and it belonged to
/// `topk_sqrtsoftplus_bf16`, which stages logits in a `[kDsv4MaxExperts]`
/// array. This kernel stages nothing, so the 512-expert ceiling does not
/// apply to it and reproducing the check here would refuse a launch that is
/// fine. It is recorded because the two routers sit in one `.cuh` and the
/// next reader will find the constant.
///
/// # Safety
///
/// `token_ids` is `[tokens]` i32, each entry in `[0, vocab_size)`; `tid2eid`
/// is `[vocab_size, top_k]` i64; `logits` is `[tokens, num_experts]` bf16;
/// `topk_idx` is writable for `[tokens, top_k]` i32 and `topk_w` for
/// `[tokens, top_k]` f32. A token id past `vocab_size` reads the table out of
/// bounds — the kernel bounds `n` against `tokens` and nothing else.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hash_route_lookup(
    token_ids: *const i32,
    tid2eid: *const i64,
    logits: *const std::ffi::c_void,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    vocab_size: i32,
    num_experts: i32,
    top_k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut std::ffi::c_void,
) -> Route {
    // `:36`, both terms, kept apart so the caller learns which it hit.
    if tokens <= 0 {
        return Route::Declined(Decline::NoTokens);
    }
    if top_k <= 0 {
        return Route::Declined(Decline::NoExperts);
    }
    let launch = Launch {
        // `:39-40` — `grid = (tokens + kDsv4Block - 1) / kDsv4Block`, then
        // `<<<grid, kDsv4Block, 0, stream>>>`.
        grid: [tokens.unsigned_abs().div_ceil(DSV4_BLOCK), 1, 1],
        block: [DSV4_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(token_ids.cast_mut().cast()),
        ArgValue::Ptr(tid2eid.cast_mut().cast()),
        ArgValue::Ptr(logits.cast_mut()),
        ArgValue::Ptr(topk_idx.cast()),
        ArgValue::Ptr(topk_w.cast()),
        ArgValue::I32(tokens),
        ArgValue::I32(vocab_size),
        ArgValue::I32(num_experts),
        ArgValue::I32(top_k),
        ArgValue::Bool(renormalize),
        ArgValue::F32(routed_scaling_factor),
    ];
    fire("moe::hash_route_lookup_dev", launch, &values, stream);
    Route::Launched
}
