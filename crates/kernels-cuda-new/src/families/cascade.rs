//! FlashInfer's cascade merge — the split-KV path's other half, as one unit
//! of ten rows.
//!
//! # What this family is for
//!
//! An FA2 fire that SPLITS its KV does not write the answer. It writes partial
//! outputs into `tmp_v` and partial log-sum-exps into `tmp_s`, one set per
//! chunk, and something has to fold them into `o` before `o` means anything.
//! Upstream folds them by calling back into `attention/cascade.cuh` from
//! inside `BatchPrefillWithPagedKVCacheDispatched` (`prefill.cuh:4351`) and
//! `BatchDecodeWithPagedKVCacheDispatched` (`decode.cuh:823`). Those two
//! functions are host code, this crate compiles device text, and the split
//! path therefore arrived here in two pieces: the FA2 kernel is
//! [`super::fa2`]'s 460 rows, and the fold is the ten below.
//!
//! Between the two, `driver-cuda` had neither. `new-horizon.md` §38 deleted
//! `attn::merge_attention_states_bf16`'s table row because its whole consumer
//! set was `dsl::cuda::merge_attention_states`, which nothing called; the
//! C++ launcher that still held the device text went with
//! `driver-cuda/csrc/` when nvcc left that crate. The visible consequence was
//! `fire/flashinfer_fa2.rs` setting `disable_split_kv: true` — upstream's own
//! predicate answered `false` for every head dim, because there was nothing
//! to merge with. This family is what lets that answer go back to upstream's.
//!
//! # Three kernels, and which of them the FA2 seam actually needs
//!
//! `cascade.cuh` holds three merge `__global__`s that matter here, reached by
//! two host launchers:
//!
//! | launcher | `cascade.cuh` | kernels |
//! |---|---|---|
//! | `MergeStates` | `:637-668` | `MergeStatesKernel`, `MergeStatesLargeNumIndexSetsKernel` |
//! | `VariableLengthMergeStates` | `:686-736` | `PersistentVariableLengthMergeStatesKernel` |
//!
//! **The batched FA2 split path calls the second launcher, not the first.**
//! `prefill.cuh:4351` and `decode.cuh:823` both name
//! `VariableLengthMergeStates`, and they have to: a batch's rows are split
//! into different numbers of chunks, so the chunk count per row is an
//! `indptr` and not a scalar. `MergeStates` takes one `num_index_sets` for
//! every row in the launch and is what the SINGLE-request paths use
//! (`prefill.cuh:2559`, `decode.cuh:739`), where the whole launch is one
//! sequence and the chunk count is uniform.
//!
//! Both are rowed here. `table/attn.rs`' record of the deleted row specified
//! the first pair and named `cascade.cuh:644-664` by line; that specification
//! is transcribed exactly and is `MERGE_STATES_SIGS[0..6]`. It is not
//! sufficient on its own, and the difference is not a matter of taste: a
//! `MergeStates` fired at a variable-length batch would fold each row against
//! the same chunk count and write plausible, finite, wrong numbers into `o`.
//! The four `varlen` rows exist so that the seam can be closed with the
//! kernel upstream closes it with.
//!
//! # The lattice, read off `cascade.cuh:641-651` and not guessed
//!
//! `DISPATCH_HEAD_DIM` (`flashinfer/utils.cuh:216-236`) instantiates exactly
//! `{64, 128, 256, 512}` and throws for anything else. For bf16
//! (`sizeof(DTypeIn) == 2`):
//!
//! ```text
//! vec_size = max(16 / sizeof(DTypeIn), HEAD_DIM / 32)     :642
//! bdx      = HEAD_DIM / vec_size                          :643
//! bdy      = 128 / bdx        (large-index-set arm only)  :646
//! ```
//!
//! | head dim | `vec_size` | `bdx` | `bdy` | smem (`:653`) |
//! |---|---|---|---|---|
//! | 64  | 8  | 8  | 16 | 4·16·64·2  + 128·4 = **8,704** |
//! | 128 | 8  | 16 | 8  | 4·8·128·2  + 128·4 = **8,704** |
//! | 256 | 8  | 32 | 4  | 4·4·256·2  + 128·4 = **8,704** |
//! | 512 | 16 | 32 | 4  | 4·4·512·2  + 128·4 = **16,896** |
//!
//! Every figure is under 48 KB, so the `cudaFuncSetAttribute` at `:656` and
//! `:715` raises a limit that is already high enough and there is nothing for
//! the Rust to express. `fire/merge_states.rs` says the same thing from the
//! other side and cites the same two lines.
//!
//! # Six rows for eight points, and why that is not a gap
//!
//! `MergeStatesKernel` takes ONE template argument that varies here —
//! `vec_size` — because its `bdx`/`bdy` come from `blockDim` and its
//! `head_dim` is a runtime parameter (`cascade.cuh:213-216`). `vec_size` is 8
//! at head dims 64, 128 and 256 and 16 at 512, so four head dims name **two**
//! instantiations. They are rowed as two rows named by `vec_size` rather than
//! as four named by head dim, so that no two rows in this crate carry the
//! same `elem`; `fire/merge_states.rs` maps the head dim onto them.
//!
//! `MergeStatesLargeNumIndexSetsKernel` and
//! `PersistentVariableLengthMergeStatesKernel` take `bdx` and `bdy` as
//! template arguments (`:275-281`, `:366-371`) — `head_dim` is
//! `vec_size * bdx` as a `constexpr` inside them — so all four head dims are
//! distinct instantiations and each gets a row.
//!
//! # Which shim mechanism carries these rows
//!
//! **The first: a stated operand list, bound by `Args::bind`, fired through
//! `KernelModule::fire`.** Not [`super::fa2`]'s third mechanism.
//!
//! `fa2`'s 460 rows state no `operands` because each of its `__global__`s
//! takes exactly one argument — a params struct by value — and
//! `kernels_cuda_new::Ty` has no variant for a struct and must not grow one.
//! Nothing here has that shape. The widest of the three kernels takes eight
//! parameters and every one of them is a pointer or a `uint32_t`:
//! `ArgValue::Ptr` and `ArgValue::U32` bind all of them today, which is
//! precisely the property `table/attn.rs`' record called *"the cheapest
//! available proof that the whole shape works"*.
//!
//! Taking the weaker mechanism when the stronger one applies would have been
//! a real loss. `Args::bind` compares this list against the values the
//! composer passes, so a transposition between `fire/merge_states.rs` and
//! `cascade.cuh` is refused at the bind with both names in the message. Under
//! `fire_raw` it would be a `void*` in the wrong slot: `s` and `s_merged` are
//! both `float*`, `v` and `v_merged` are both bf16 pointers, and swapping
//! either pair type-checks in C++ and in Rust and produces a merge that reads
//! its own output.
//!
//! Every operand states its `Ty` and NO `kernels::Source`. That is the
//! deliberate half: a `Source` says where a TABLE-DRIVEN binder finds the
//! value, and nothing binds these from a table. They are fired by a Rust host
//! program that reads a plan, and `Source::Unbound` is the row saying so —
//! the same statement `LaunchRule::Unstated` makes about the geometry, which
//! `fire/merge_states.rs` states instead because the geometry depends on a
//! runtime comparison no rule can make.

use kernels::{KernelSig, kernel, operands};

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The head dims `DISPATCH_HEAD_DIM` instantiates, in its order.
///
/// `flashinfer/utils.cuh:216-236`, and the same four
/// [`super::fa2::HEAD_DIMS`] carries — necessarily the same, since the buffers
/// these kernels merge are the ones an FA2 split fire wrote. `tests` checks
/// the two lists against each other rather than trusting the coincidence.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `num_smem_stages`, `cascade.cuh:649` and `:702`.
///
/// A template argument of both staged kernels and a term in the shared-memory
/// figure, so it is named once here and read by [`smem_bytes`].
pub const NUM_SMEM_STAGES: u32 = 4;

/// `num_threads`, `cascade.cuh:645` and `:700`.
///
/// The block is `bdx * bdy` and this is that product, held fixed across the
/// head dims: `bdy` is *defined* as `num_threads / bdx` at `:646` and `:701`.
pub const NUM_THREADS: u32 = 128;

/// `(vec_size, bdx, bdy)` for a head dim, `cascade.cuh:642-646`.
///
/// The `bdy` returned is the STAGED arm's — `num_threads / bdx`.
/// `MergeStatesKernel`'s arm computes its own `bdy` as `num_heads`
/// (`:659`), which is a runtime value and not part of this lattice.
///
/// `None` for a head dim `DISPATCH_HEAD_DIM` does not instantiate. That
/// `default:` is a `throw std::invalid_argument` upstream; here it is an
/// absent row, which `fire/merge_states.rs` turns into a typed refusal. The
/// two are not the same thing and the difference is the point: an exception
/// crossing the C ABI is undefined behaviour, and in this tree it used to
/// unwind to `SIGABRT` with no message.
#[must_use]
pub const fn geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    // `vec_size = max(16 / sizeof(DTypeIn), HEAD_DIM / 32)` with
    // `sizeof(DTypeIn) == 2`, so the left term is 8.
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

/// The staged arms' dynamic shared memory, `cascade.cuh:653` and `:703-704`.
///
/// ```text
/// num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float)
/// ```
///
/// 8,704 B at head dims 64, 128 and 256; 16,896 B at 512. Both are under the
/// 48 KB every architecture gives a block without asking, which is why
/// `:656`'s and `:715`'s `cudaFuncSetAttribute` has nothing to do and why
/// `fire/merge_states.rs` does not make a driver call to match it.
///
/// `None` propagates [`geometry`]'s refusal rather than answering zero: a
/// zero here would be a legal `Launch::smem` and the staged kernels index
/// `extern __shared__` unconditionally.
#[must_use]
pub const fn smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

/// The one unit: `csrc/src/cascade/merge_states.cuh`, ten rows.
pub static UNITS: &[Unit] = &[MERGE_STATES];

/// `cascade.cuh`'s three merge kernels at the four head dims
/// `DISPATCH_HEAD_DIM` instantiates.
pub const MERGE_STATES: Unit = Unit {
    name: "cascade/merge_states",
    root: ROOT,
    rows: MERGE_STATES_ROWS,
    options: OPTIONS,
};

/// The root, bound once.
///
/// `include_str!` is expanded per call site, so one `const` rather than one
/// per unit — the same reason [`super::fa2`]'s header gives, with one unit
/// instead of 56.
const ROOT: &str = include_str!("../../csrc/src/cascade/merge_states.cuh");

/// `--device-as-default-execution-space`, and it is load-bearing here for the
/// same reason it is in [`super::fa2`].
///
/// `cascade.cuh` carries host launcher TEMPLATES beside its kernels —
/// `MergeStates`, `VariableLengthMergeStates`, `AttentionSum` and three more,
/// each with `cudaLaunchKernel` and `cudaFuncSetAttribute` in its body — and
/// NVRTC reports *"A function without execution space annotations is
/// considered a `__host__` function"* against them without this flag. None of
/// them is ever instantiated by this unit; they still have to parse.
///
/// **Measured, not assumed.** `examples/vendor_probe.rs`' `MERGE` candidate
/// compiles exactly this header and asks for eight name expressions — the
/// four `MergeStatesLargeNumIndexSetsKernel` points and
/// `MergeStatesKernel` at each head dim's `vec_size` — and gets 8 of 8. Its
/// base option list (`vendor_probe.rs:955-963`) contains this flag and its
/// `MERGE.options` is empty, so the measurement was made with this list and
/// no other.
///
/// **No `-I`.** This crate's NVRTC passes none and reads nothing from disk;
/// the vendored closure arrives as `includeNames[]` from `carried.rs`. See
/// [`super::fa2`]'s `OPTIONS` for the whole of that argument, which is not
/// repeated.
const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

/// The `__global__` this unit's first six rows instantiate.
const MERGE_PATH: &str = "::flashinfer::MergeStatesKernel";

/// `cascade.cuh:275-281`.
const LARGE_PATH: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel";

/// `cascade.cuh:366-371`.
const VARLEN_PATH: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel";

/// The ten contracts.
///
/// Read against `cascade.cuh` by eye: each operand list is the `__global__`'s
/// parameter list in its own order, with `| null` on exactly the pointers the
/// kernel tests. All three test `s_merged` (`:298`, `:337`, `:461`) and only
/// the variable-length one tests `seq_len_ptr` (`:375`).
#[rustfmt::skip]
static MERGE_STATES_SIGS: [KernelSig; 10] = [
    // ── `MergeStatesKernel`, `cascade.cuh:213-216` ──────────────────────
    //
    // Named by `vec_size` and not by head dim: `bdx` and `bdy` reach this
    // kernel through `blockDim` and `head_dim` through a parameter, so head
    // dims 64, 128 and 256 are one instantiation. See this module's header.
    //
    // `head_dim` is an operand HERE and a `constexpr` in the other two, which
    // is the visible half of the same fact.
    kernel!(merge_states_v8 "attn::cascade::merge_states_v8",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf,
            s: F32s,
            v_merged: BufMut,
            s_merged: F32sMut | null,
            num_index_sets: U32,
            num_heads: U32,
            head_dim: U32,
        ]),
    kernel!(merge_states_v16 "attn::cascade::merge_states_v16",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf,
            s: F32s,
            v_merged: BufMut,
            s_merged: F32sMut | null,
            num_index_sets: U32,
            num_heads: U32,
            head_dim: U32,
        ]),
    // ── `MergeStatesLargeNumIndexSetsKernel`, `cascade.cuh:275-281` ─────
    //
    // Six parameters, not seven: `head_dim` is `vec_size * bdx` as a
    // `constexpr` at `:288`, so it is in the row's `elem` rather than in its
    // operand list.
    kernel!(merge_states_large_hd64 "attn::cascade::merge_states_large_hd64",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd128 "attn::cascade::merge_states_large_hd128",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd256 "attn::cascade::merge_states_large_hd256",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd512 "attn::cascade::merge_states_large_hd512",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    // ── `PersistentVariableLengthMergeStatesKernel`, `:366-371` ─────────
    //
    // `indptr` is `IdType*` — `int32_t`, so `I32s` and not `U32s`, which is
    // the one operand in this file where the two would both compile. It is
    // `[max_seq_len + 1]` entries and `num_index_sets` for row `pos` is
    // `indptr[pos + 1] - indptr[pos]` (`:401`), which is why this kernel and
    // not `MergeStatesKernel` is what a batched split needs.
    //
    // `seq_len` is a DEVICE pointer to one `uint32_t` and is nullable:
    // `:375` reads `seq_len_ptr ? *seq_len_ptr : max_seq_len`. The prefill
    // seam passes `params.total_num_rows`, which the plan writes on the
    // device; the decode seam passes null and lets `max_seq_len` stand
    // (`decode.cuh:823-824`).
    kernel!(merge_states_varlen_hd64 "attn::cascade::merge_states_varlen_hd64",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd128 "attn::cascade::merge_states_varlen_hd128",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd256 "attn::cascade::merge_states_varlen_hd256",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd512 "attn::cascade::merge_states_varlen_hd512",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
];

/// The ten instantiations, in [`MERGE_STATES_SIGS`]' order.
///
/// The template argument lists are `cascade.cuh`'s own, in its own order:
///
/// ```text
/// MergeStatesKernel                        <vec_size, DTypeIn, DTypeO>
/// MergeStatesLargeNumIndexSetsKernel       <vec_size, bdx, bdy, num_smem_stages, DTypeIn, DTypeO>
/// PersistentVariableLengthMergeStatesKernel<vec_size, bdx, bdy, num_smem_stages, DTypeIn, DTypeO, IdType>
/// ```
///
/// The numbers are [`geometry`]'s and the type names are
/// `csrc/src/cascade/merge_states.cuh`'s three aliases. Both are written out
/// as literals rather than composed, because `elem` is a `&'static str` — the
/// only `const` string concatenation available here is `concat!`, which takes
/// literals — and because a row a reader cannot diff against
/// `cascade.cuh:641-651` by eye is a row that stops being checked.
/// `tests::the_rows_match_the_derivation` is what makes the numbers and
/// [`geometry`] agree.
#[rustfmt::skip]
static MERGE_STATES_ROWS: &[DeviceKernel] = &[
    // head dims 64, 128, 256 — `vec_size = max(8, HEAD_DIM / 32) = 8`.
    DeviceKernel { sig: &MERGE_STATES_SIGS[0], template_path: MERGE_PATH, elem: concat!(
        "8, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    // head dim 512 — `max(8, 16) = 16`.
    DeviceKernel { sig: &MERGE_STATES_SIGS[1], template_path: MERGE_PATH, elem: concat!(
        "16, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },

    DeviceKernel { sig: &MERGE_STATES_SIGS[2], template_path: LARGE_PATH, elem: concat!(
        "8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[3], template_path: LARGE_PATH, elem: concat!(
        "8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[4], template_path: LARGE_PATH, elem: concat!(
        "8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[5], template_path: LARGE_PATH, elem: concat!(
        "16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },

    DeviceKernel { sig: &MERGE_STATES_SIGS[6], template_path: VARLEN_PATH, elem: concat!(
        "8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[7], template_path: VARLEN_PATH, elem: concat!(
        "8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[8], template_path: VARLEN_PATH, elem: concat!(
        "8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[9], template_path: VARLEN_PATH, elem: concat!(
        "16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
];

/// The symbol that merges a uniform-chunk-count batch at `head_dim`.
///
/// `MergeStatesKernel`'s two rows, selected by `vec_size`. `None` is
/// `DISPATCH_HEAD_DIM`'s `default:`.
#[must_use]
pub fn merge_states_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 | 128 | 256 => Some(MERGE_STATES_SIGS[0].symbol),
        512 => Some(MERGE_STATES_SIGS[1].symbol),
        _ => None,
    }
}

/// The symbol for the staged arm at `head_dim`, `cascade.cuh:651`.
#[must_use]
pub fn merge_states_large_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(MERGE_STATES_SIGS[2].symbol),
        128 => Some(MERGE_STATES_SIGS[3].symbol),
        256 => Some(MERGE_STATES_SIGS[4].symbol),
        512 => Some(MERGE_STATES_SIGS[5].symbol),
        _ => None,
    }
}

/// The symbol for the variable-length arm at `head_dim`, `cascade.cuh:705`.
#[must_use]
pub fn merge_states_varlen_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(MERGE_STATES_SIGS[6].symbol),
        128 => Some(MERGE_STATES_SIGS[7].symbol),
        256 => Some(MERGE_STATES_SIGS[8].symbol),
        512 => Some(MERGE_STATES_SIGS[9].symbol),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HEAD_DIMS, MERGE_STATES, NUM_SMEM_STAGES, NUM_THREADS, geometry,
        merge_states_large_symbol, merge_states_symbol, merge_states_varlen_symbol, smem_bytes,
    };

    /// The four head dims this lattice covers are FA2's four.
    ///
    /// Not a coincidence to be trusted: the buffers these kernels merge are
    /// the ones an FA2 split fire wrote, so a head dim FA2 instantiates and
    /// this does not is a split plan with no merge — which is the exact hole
    /// `disable_split_kv` was turned on to avoid.
    #[test]
    fn the_lattice_covers_flashinfer_fa2s_head_dims() {
        assert_eq!(HEAD_DIMS, crate::families::fa2::HEAD_DIMS);
    }

    /// Every head dim resolves all three symbols, and nothing else resolves
    /// any.
    #[test]
    fn every_head_dim_has_all_three_arms() {
        for &hd in HEAD_DIMS {
            assert!(merge_states_symbol(hd).is_some(), "{hd}");
            assert!(merge_states_large_symbol(hd).is_some(), "{hd}");
            assert!(merge_states_varlen_symbol(hd).is_some(), "{hd}");
            assert!(geometry(hd).is_some(), "{hd}");
            assert!(smem_bytes(hd).is_some(), "{hd}");
        }
        // `DISPATCH_HEAD_DIM`'s `default:`, which upstream `throw`s from.
        for hd in [0u32, 32, 96, 120, 1024] {
            assert!(merge_states_symbol(hd).is_none(), "{hd}");
            assert!(merge_states_large_symbol(hd).is_none(), "{hd}");
            assert!(merge_states_varlen_symbol(hd).is_none(), "{hd}");
            assert!(geometry(hd).is_none(), "{hd}");
            assert!(smem_bytes(hd).is_none(), "{hd}");
        }
    }

    /// Every symbol this family answers is a row of its unit.
    #[test]
    fn every_symbol_is_a_row() {
        for &hd in HEAD_DIMS {
            for symbol in [
                merge_states_symbol(hd).unwrap(),
                merge_states_large_symbol(hd).unwrap(),
                merge_states_varlen_symbol(hd).unwrap(),
            ] {
                assert!(MERGE_STATES.hosts(symbol), "{symbol}");
            }
        }
    }

    /// [`geometry`] and the rows' `elem` strings agree.
    ///
    /// The rows spell their template arguments as literals so a reader can
    /// diff them against `cascade.cuh:641-651`; this is what stops the two
    /// from drifting. The staged rows carry `<vec_size, bdx, bdy, 4, …>`, so
    /// the check is a prefix match on the `elem`.
    #[test]
    fn the_rows_match_the_derivation() {
        for &hd in HEAD_DIMS {
            let (vec_size, bdx, bdy) = geometry(hd).unwrap();
            assert_eq!(bdx * vec_size, hd, "bdx * vec_size is the head dim at {hd}");
            assert_eq!(bdx * bdy, NUM_THREADS, "the staged block is 128 threads at {hd}");

            let want = format!("{vec_size}, {bdx}, {bdy}, {NUM_SMEM_STAGES}, ");
            for symbol in
                [merge_states_large_symbol(hd).unwrap(), merge_states_varlen_symbol(hd).unwrap()]
            {
                let row = MERGE_STATES.row(symbol).expect("the symbol is a row");
                assert!(row.elem.starts_with(&want), "{symbol}: {:?} vs {want:?}", row.elem);
            }

            let row = MERGE_STATES.row(merge_states_symbol(hd).unwrap()).unwrap();
            assert!(row.elem.starts_with(&format!("{vec_size}, ")), "{}", row.elem);
        }
    }

    /// The two shared-memory figures, `cascade.cuh:653`.
    ///
    /// Written out rather than recomputed from the same expression: a test
    /// that repeats the implementation checks that `+` still works. These are
    /// the numbers `table/attn.rs`' record has carried since before the row
    /// came back, and they are what `fire/merge_states.rs` puts in
    /// `Launch::smem`.
    #[test]
    fn the_shared_memory_is_the_figure_the_record_carries() {
        assert_eq!(smem_bytes(64), Some(8_704));
        assert_eq!(smem_bytes(128), Some(8_704));
        assert_eq!(smem_bytes(256), Some(8_704));
        assert_eq!(smem_bytes(512), Some(16_896));
        // Under 48 KB at every point, which is why `cascade.cuh:656` and
        // `:715` are no-ops nothing here has to express.
        for &hd in HEAD_DIMS {
            assert!(smem_bytes(hd).unwrap() < 48 * 1024, "{hd}");
        }
    }
}
