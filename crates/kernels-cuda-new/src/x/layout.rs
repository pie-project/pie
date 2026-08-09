//! `layout` — pure addressing, as two truths.
//!
//! Gather, scatter, split, concat, transpose, embed, and the quest envelope
//! tier. Five roots of device text under `csrc/src/layout/`, sixteen
//! `__global__` instantiations over them, thirteen host programs, seven
//! contracts and five binds. §5 step 5's first family after the pilot.
//!
//! # What this replaces
//!
//! ```text
//!   before                                              lines
//!   kernels-cuda-new/src/families/layout.rs  16 rows       885
//!   kernels-cuda-new/src/table/layout.rs      7 rows       160
//!   driver-cuda/src/fire/embed.rs             1 program    185
//!   driver-cuda/src/fire/envelope.rs          3 programs   351
//!   driver-cuda/src/bind/service.rs           1 wrapper    ~32
//!                                                       ------
//!                                                        1,613
//!   after
//!   kernels-cuda-new/src/x/layout.rs   13 host programs, 16 device
//!                                       rows, 7 contracts, 5 binds
//! ```
//!
//! **Six of the thirteen host programs are NEW**, and none of their geometry
//! is invented. `split_bf16_rows`, `split_qwen_gdn_ba_bf16`,
//! `gather_bf16_rows` and `transpose_bf16_nld_to_lnd` were in
//! `device::JIT_DISPATCHED`: a `LaunchRule` opened their grid and a generated
//! arm bound their operands, so the `.cu` launchers went with the files
//! rather than being ported. `deinterleave_rows_bf16`, `deinterleave_vec_bf16`
//! and `concat_bf16_rows` never had a Rust caller at all. Every one of them
//! cites the rule function in `runtime/launch.rs` that fired it, and where
//! the rule was itself checked against a `<<<>>>` the citation carries both
//! numbers. [`copy_if_valid_slot`] is the one that cites a `<<<>>>` directly,
//! because `layout/slot_ops.cu:59-62` is quoted verbatim on the row this file
//! deletes.
//!
//! # Five units, five modules — the idiom `x/mlp.rs` found first
//!
//! `unit!` emits `UNITS`, `ROWS`, `PARAMS` and `mod raw` at its invocation
//! site, so two invocations in one scope collide on four names. Each root
//! therefore gets a module of its own and the family re-exports the five as
//! [`UNITS`], which `families::ALL` reads. `rope` had one root and never met
//! this.
//!
//! # THE FIRST TWO-KERNEL BODY IN FN-WORLD — read this before copying it
//!
//! §2.3's `Composed`/`Walk` shape — *two different kernels in one body* —
//! was written and unproven when `rope` landed, because every one of rope's
//! twelve host programs is a single launch preceded by host arithmetic.
//! [`envelope_merge_written`] is the first real one: **one symbol, one
//! declaration, and either one launch of `merge_written_fused` or an ORDERED
//! pair of `reset_started_pages` then `merge_written`**, chosen on
//! `num_tokens <= 128`. `families/layout.rs` called that fork *"one symbol,
//! two launches"* and named it the second half of its refusal to row the
//! tier; §5 step 5 says `Composed`'s two ops become two-call bodies, and this
//! is what that looks like. Nothing in the floor was needed for it — a body
//! is a `fn` and a `fn` may call twice — which is itself the finding.
//!
//! The pair may not be fused or reordered. The reset writes the identity into
//! pages this batch is the first writer of and the merge then READS what it
//! wrote; they are two launches on one stream for that reason and not for a
//! register budget.
//!
//! # bf16 throughout, and one deliberate `u16`
//!
//! §3.2's two-formats-one-width hazard is not exercised here — no kernel in
//! this family has an f16 twin, and `families/layout.rs` says why: *"A second
//! format costs one row and no C++ ... none is declared here because no fire
//! asks for one, and a row for a kernel nothing states is a claim about a
//! caller that does not exist."*
//!
//! What IS exercised is the neighbouring distinction. [`gather_rows`]'s two
//! kernels instantiate at `device::u16` and not at `device::bf16`, and the
//! deleted row's reason travels with them:
//!
//! > `device::u16` and not `device::bf16` on both, because both are pure
//! > copies: neither ever converts to float, and the ahead-of-time launchers
//! > take `u16*` for exactly that reason. A tag type that promises arithmetic
//! > nobody performs is a tag type that invites it.
//!
//! **That is the floor gap this port reports.** `x/abi.rs` has no
//! `ptr_abi!(u16, …)`, so `*const u16` has no `Abi` impl and this file does
//! not compile until one lands. The two alternatives are the bypass
//! `x/abi.rs`'s own `i8` note names — `*const c_void` puts `const void*` in
//! the typecheck TU where the kernel says `const device::u16*` — and
//! `*const bf16`, which would put `device::bf16*` there and retract the
//! sentence above. Both spellings the impl needs are
//! `kernels::Ty::U16s.cpp()` and `U16sMut.cpp()` verbatim
//! (`crates/kernels/src/lib.rs:1045-1046`), which is what the deleted row
//! already stated as `U16s`/`U16sMut`. Reported to `kernelx-floor` rather
//! than worked around.
//!
//! # Two symbols here are the DRIVER's, and get no `Entry`
//!
//! `qwen35_verify_stash_store` and `qwen35_verify_stash_load` are
//! PSEUDO-SYMBOLS: they name an operation of the declared executor, not a
//! `__global__` the linker resolves. [`x::route`](crate::x::route)'s "THE ONE
//! OVERLAP" is the rule — a driver op contributes to [`SIGS`] and not to
//! `FAMILIES`, because an `Entry` shadows the `DriverOp` arm and a `none:`
//! arm would refuse a live model at load. So the two have contracts and no
//! `bind!` arm, `execution::SERVED`'s `DriverOp` tuples for them stay, and
//! `x/adapter.rs` is the worked example.
//!
//! # `layout::split_q_gate_bf16` is declared here and hosted elsewhere
//!
//! Its device text is `layout/deinterleave.cuh`, so its row is in
//! [`deinterleave`]'s `unit!` — *"the rows stay where the device text is"*,
//! `x/driver_internal.rs`'s rule, because a second `unit!` naming the same
//! root would compile it twice and `unit_of` would answer with whichever
//! won. Its host program is
//! [`crate::x::driver_internal::split_q_gate_bf16`], which fires it by symbol
//! through `x::fire::fire`, and its table row is `table::driver_internal`'s.
//! This file therefore declares the kernel and states no contract for it.

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::bf16;
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// FIVE `unit!` INVOCATIONS CANNOT SHARE A SCOPE. Each root gets a module and
// the family re-exports the five below. `x/mlp.rs` found this first with two
// roots; the note is repeated per family because it is the first thing a
// reader of a multi-root family hits.
// ---------------------------------------------------------------------------

/// `layout/deinterleave.cuh` — the packed-bank splits and the row concat.
pub mod deinterleave {
    use super::bf16;

    unit! {
        /// gpt-oss's parity deinterleave, Qwen's GDN halves, the concat/split
        /// pair, and full attention's per-head query/gate cut.
        ///
        /// Six of the header's seven kernels, all six at `device::bf16` and
        /// all six written over `T`. The seventh,
        /// `repeat_interleave_heads`, is a template for one reason only —
        /// *"a plain `__global__` answers `nvrtcGetLoweredName` with
        /// nothing"* — and carries no instantiation, because nothing in the
        /// tree calls it: no model text, no driver-internal row, no sibling
        /// `.cu`. A declaration for it would name a caller that does not
        /// exist.
        ///
        /// # What `T` means here
        ///
        /// The element type and nothing else. Every one of these is a pure
        /// copy or a pure interleave, so nothing converts to float and
        /// `Elem<T>` is not needed — `deinterleave.cuh:11-15`.
        unit DEINTERLEAVE = "layout/deinterleave",
            text = include_str!("../../csrc/src/layout/deinterleave.cuh"),
            file = "layout/deinterleave.cuh";

        /// `deinterleave.cuh:85` — gpt-oss packs gate and up ROW BY ROW, so
        /// splitting them is a parity deinterleave and not a slice.
        ///
        /// Weight-shaped: `H` is a WEIGHT's row count, which is why the
        /// deleted row sourced neither extent, and `LaunchRule::RouteRows`
        /// recovered the row count from the fire's rectangle regardless.
        fn deinterleave_rows = "layout::device::deinterleave_rows" <T> (
            fused: *const T,
            gate_out: *mut T,
            up_out: *mut T,
            h: i32,
        ) where *const T, *mut T {
            "layout::deinterleave_rows_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:109` — the flat form of the same split, one
        /// thread per output element.
        ///
        /// **`I` SURVIVES where the row form's extent did not.**
        /// `LaunchRule::Elementwise` rounds the element count up to a whole
        /// block, so the tail threads of the last block have to be told to
        /// stop — an extent a rule RECOVERS is not an operand, and an extent
        /// a rule ROUNDS is.
        fn deinterleave_vec = "layout::device::deinterleave_vec" <T> (
            fused: *const T,
            gate_out: *mut T,
            up_out: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "layout::deinterleave_vec_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:152` — `[N, left] ++ [N, right] -> [N,
        /// left+right]`, one block per row.
        ///
        /// `layout::split_bf16_rows`, the inverse, is alive on both sides;
        /// being half of a pair is not a consumer, and this one's table row
        /// and DSL wrapper went together in §54.
        fn concat_rows = "layout::device::concat_rows" <T> (
            left: *const T,
            right: *const T,
            out: *mut T,
            left_dim: i32,
            right_dim: i32,
        ) where *const T, *mut T {
            "layout::concat_bf16_rows" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:188` — the inverse: one packed row out to two.
        ///
        /// Two results, so both widths come off the results and the source
        /// needs no extent of its own.
        fn split_rows = "layout::device::split_rows" <T> (
            src: *const T,
            left: *mut T,
            right: *mut T,
            left_dim: i32,
            right_dim: i32,
        ) where *const T, *mut T {
            "layout::split_bf16_rows" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:170` — Qwen's GDN bank, split by HALVES where
        /// [`deinterleave_rows`](raw::deinterleave_rows) splits by parity.
        ///
        /// Same shape, different layout, checkpoint decides — which is why
        /// they are two kernels and not one with a flag.
        fn split_qwen_gdn_ba = "layout::device::split_qwen_gdn_ba" <T> (
            ba: *const T,
            b_out: *mut T,
            a_out: *mut T,
            v_h: i32,
        ) where *const T, *mut T {
            "layout::split_qwen_gdn_ba_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:130` — full attention's per-head query/gate cut.
        ///
        /// **Declared here, hosted in `x/driver_internal.rs`.** The device
        /// text is this root's, so the row is this unit's; the host program
        /// is [`crate::x::driver_internal::split_q_gate_bf16`], which fires
        /// it by symbol, and the table row is `table::driver_internal`'s.
        /// See this file's header.
        ///
        /// `N` and `num_heads` SURVIVE as operands where a `RouteRows` row
        /// count would not: the kernel guards `if (n >= N || h >= num_heads)
        /// return;` and multiplies both back into every address it forms —
        /// `(n * num_heads + h) * 2 * head_dim` — so they are addressing
        /// arithmetic the grid happens to agree with rather than an extent
        /// the grid recovers.
        fn split_q_gate = "layout::device::split_q_gate" <T> (
            packed: *const T,
            q_out: *mut T,
            gate_out: *mut T,
            n: i32,
            num_heads: i32,
            head_dim: i32,
        ) where *const T, *mut T {
            "layout::split_q_gate_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `layout/gather_rows.cuh` — the epilogue's gather and the PLE relay.
pub mod gather_rows {
    unit! {
        /// Two of the header's four kernels, both at `device::u16`.
        ///
        /// The other two cannot be declared and the reasons are the deleted
        /// family's, unchanged: `transpose_nld_to_lnd_vec4` is selected on
        /// the HOST from pointer alignment and takes `uint4*`, and
        /// `embed_scaled_concat` has three parameters — `vocab`, `scale`,
        /// `hidden_first` — that no caller in the tree supplies.
        ///
        /// # `device::u16`, and the floor gap it reports
        ///
        /// Both are pure copies; neither ever converts to float. See this
        /// file's header for why `*const u16` is written rather than
        /// `*const c_void` or `*const bf16`, and for the `ptr_abi!(u16, …)`
        /// this unit needs from `x/abi.rs`.
        unit GATHER_ROWS = "layout/gather_rows",
            text = include_str!("../../csrc/src/layout/gather_rows.cuh"),
            file = "layout/gather_rows.cuh";

        /// `gather_rows.cuh:78` — THE EPILOGUE'S GATHER.
        ///
        /// A prefill streams one row per token and reads one distribution
        /// per request, so the rows actually sampled have to be collected
        /// before the final norm and the head — and they are not a
        /// contiguous run, which is why this is a gather rather than a
        /// slice.
        ///
        /// **The last parameter is the row WIDTH, not a vocabulary.** The
        /// header used to name it `vocab` while the caller passed `H`,
        /// because this gathers hidden rows on their way INTO the head; the
        /// parameter is named `width` now, since a template over `T` reading
        /// `vocab` would be a comment that is wrong at every call site.
        ///
        /// The kernel strides by `blockDim.x` deliberately: the block is
        /// sized `min(1024, ceil(width/32)*32)` by the rule this row was
        /// fired under, and the file-scope `constexpr int BLOCK = 256` it
        /// used to stride by would have dropped every element past the 256th
        /// the moment the rule disagreed.
        fn gather_rows = "layout::device::gather_rows" <T> (
            src: *const T,
            row_indices: *const i32,
            dst: *mut T,
            width: i32,
        ) where *const T, *mut T {
            "layout::gather_bf16_rows" => where [T = u16] "device::u16",
        }

        /// `gather_rows.cuh:132` — the PLE relay: `[N, L, D] -> [L, N, D]`,
        /// so a layer reads a contiguous slice. Addressing, not arithmetic.
        ///
        /// `Elementwise` recovered NOTHING this kernel needs — it flattens
        /// the rectangle to an element count and rounds it up to a block —
        /// so `n`, `layers` and `dim` all stay, because the kernel divides
        /// the flat index by them, and `total` joins them because the tail
        /// threads of the last block have to be told to stop.
        fn transpose_nld_to_lnd = "layout::device::transpose_nld_to_lnd" <T> (
            src: *const T,
            dst: *mut T,
            n: i32,
            layers: i32,
            dim: i32,
            total: usize,
        ) where *const T, *mut T {
            "layout::transpose_bf16_nld_to_lnd" => where [T = u16] "device::u16",
        }
    }
}

/// `layout/slot_ops.cuh` — the slot-conditional byte copy.
pub mod slot_ops {
    unit! {
        /// One of the header's two kernels.
        ///
        /// The other, `zero_slots_if_fresh`, launches `dim3(request_count,
        /// layer_count)` at 256 — a second grid axis over LAYERS, which no
        /// `LaunchRule` spells and no `Dims` field carries. Under fn-world
        /// that refusal is retired in principle (a `fn` states any grid it
        /// likes) and stands in practice for the reason that outlived the
        /// rule: **nothing calls it.** `slot_ops.cu` is deleted and this
        /// header is the only text either kernel has, so a declaration for
        /// the second would name a caller that does not exist. It compiles
        /// anyway — a unit compiles its root, not its rows.
        unit SLOT_OPS = "layout/slot_ops",
            text = include_str!("../../csrc/src/layout/slot_ops.cuh"),
            file = "layout/slot_ops.cuh";

        /// `slot_ops.cuh:64` — copy a slot's bytes if the slot is valid.
        ///
        /// A plain `__global__` over `u8`, so its instantiation is its bare
        /// path: [`DeviceKernel::PLAIN`](crate::device::DeviceKernel::PLAIN).
        ///
        /// # This declaration is the tree's only witness for `<<<1, 256>>>`
        ///
        /// The deleted row was the only `kernels::LaunchRule::Single` in the
        /// table, and its doc said why it survived §54's sweep when its own
        /// table row and DSL wrapper did not:
        ///
        /// > a table row is a claim about who CALLS a symbol, and a family
        /// > row is a claim about what the kernel IS. Nothing stopped calling
        /// > this kernel — it is the tree's only witness for
        /// > `LaunchRule::Single`, fired three times by
        /// > `tests/launch_rules.rs` through `crate::runtime::fire`, which
        /// > resolves through `crate::unit::unit_of` and has never looked at
        /// > `table::layout`. Deleting it would delete the only evidence a
        /// > `LaunchRule` variant has.
        ///
        /// **In fn-world the witness moves from the rule to the `fn`.** A
        /// declaration's row is `LaunchRule::Unstated` by construction, so
        /// `runtime::fire` can no longer evaluate a geometry for this symbol
        /// and [`copy_if_valid_slot`](super::copy_if_valid_slot) states the
        /// literal instead. That is a predicted breakage, written down in
        /// this port's report: `tests/launch_rules.rs` asserts the row states
        /// `Rule::Single` and then fires it three times through the rule.
        fn copy_if_valid_slot = "layout::device::copy_if_valid_slot" (
            src: *const u8,
            dst: *mut u8,
            bytes: usize,
            slot_ids: *const i32,
            request: usize,
        ) {
            "layout::copy_if_valid_slot" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `layout/envelope.cuh` — the quest per-page key envelope tier.
pub mod envelope {
    use super::bf16;
    use crate::x::abi::MaybeConst;

    unit! {
        /// Five of the header's seven kernels.
        ///
        /// For every `(page, kv_head)` a min and a max over the page's live
        /// keys, `[num_pages, num_kv_heads, head_dim]` bf16 in each of two
        /// planes.
        ///
        /// # `Tu = 0`, and why `elem` spells a value here
        ///
        /// Four of the five carry `template <int Tu = 0>` — a parameter that
        /// parameterises nothing, added because a plain `__global__` in a
        /// header caps that header at one includer (`envelope.cuh:100-119`
        /// measured the link error, twice per kernel). Every template
        /// argument is rendered, so a defaulted one rendered as ABSENT is a
        /// different instantiation and the declaration must spell it.
        /// `device::i32(0)` is the spelling that resolves under NVRTC: a bare
        /// `0` fails the name-map pragma with *expected an identifier*,
        /// because a literal cannot take a namespace.
        ///
        /// **The cost is the offline typecheck, and it is stated rather than
        /// discovered.** `abi::emit_device_typecheck` refuses a row whose
        /// `elem` head is a value — it spells every buffer operand as a
        /// pointer to that head — so the four `Tu` rows are not checkable
        /// from `elem` alone. Only `update_appended`, whose `elem` is
        /// `device::bf16`, is. That is the same trade
        /// `attn::write_kv_per_token_head`'s two `device::*_type::value` rows
        /// already make.
        ///
        /// # The two that are not declared
        ///
        /// `recompute` and `dot<BLOCK>`. `launch_envelope_recompute_bf16` was
        /// deleted by §54 — the incremental
        /// [`update_appended`](raw::update_appended) gives the answer a full
        /// recompute gives, at the cost of the touched set instead of the
        /// cache — and `dot` has no caller in the tree. Both are still
        /// compiled, because a unit compiles its root.
        unit ENVELOPE = "layout/envelope",
            text = include_str!("../../csrc/src/layout/envelope.cuh"),
            file = "layout/envelope.cuh";

        /// `envelope.cuh:377` — the whole maintenance step for a SHORT
        /// append, in one kernel.
        ///
        /// It resets a page it is the first writer of and merges into one it
        /// is not, deciding per token from `w_off`. Two kernels' work, and
        /// the reason [`envelope_merge_written`](super::envelope_merge_written)
        /// forks at all is that the decision is only sound while the launch
        /// is small enough that no two blocks race on the same page.
        ///
        /// `row_valid` may be null — the kernel tests it — so a null caller
        /// means *"every row is valid"* and not *"no rows are"*. That is why
        /// it is [`MaybeConst`] and not `*const u8`: the absence is in the
        /// type rather than in a comment.
        fn merge_written_fused = "layout::device::merge_written_fused" (
            k_curr: *const bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_merge_written_fused_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:492` — the FIRST of the two launches taken when
        /// `num_tokens > 128`.
        ///
        /// It only resets, and it must complete before
        /// [`merge_written`](raw::merge_written) reads what it wrote — which
        /// is why these are two launches on one stream and not one kernel.
        fn reset_started_pages = "layout::device::reset_started_pages" (
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_reset_started_pages_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:535` — the SECOND of the two.
        ///
        /// No `w_off`: the reset already used it, and this kernel merges
        /// every written row unconditionally. Dropping it is the C++'s
        /// argument list at `envelope.cu:54-57`, not an omission.
        fn merge_written = "layout::device::merge_written" (
            k_curr: *const bf16,
            w_page: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_merge_written_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:337` — the `+inf`/`-inf` identity across a whole
        /// fresh pool.
        ///
        /// `n` is `usize` in both languages and that is load-bearing: a real
        /// pool is `num_pages` in the tens of thousands, and `num_pages *
        /// num_kv_heads * head_dim` overflows `i32` before it overflows
        /// anything the kernel indexes with.
        fn seed_empty = "layout::device::seed_empty" (
            env_min: *mut bf16,
            env_max: *mut bf16,
            n: usize,
        ) {
            "layout::envelope_seed_empty_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:238` — the incremental fold of the pages an append
        /// touched.
        ///
        /// The one of the five that is `template <class T>` rather than
        /// `template <int Tu>`, so its `elem` is `device::bf16` and it is the
        /// only one of the five the offline typecheck can check.
        ///
        /// **`max_touched` is a launch extent and is NOT in this list.** The
        /// kernel takes `num_requests`, `page_size`, `num_kv_heads`,
        /// `head_dim` and reads its own `blockIdx.x`; the bound that sizes
        /// the grid is the host's and reaches the kernel as a rectangle.
        fn update_appended = "layout::device::update_appended" <T> (
            k_pages: *const T,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_requests: i32,
            page_size: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) where *const T {
            "layout::envelope_update_appended_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `layout/embed.cuh` — the flat embedding gather.
pub mod embed {
    use super::bf16;

    unit! {
        /// One template, two instantiations — the first launch of every fire.
        ///
        /// # Two rows for one `__global__`
        ///
        /// `embed<bool VEC>` is one template and a declaration names one
        /// instantiation, so the symbols are suffixed `#vec`/`#scalar` — the
        /// same device-side disambiguation
        /// `attn::write_kv_explicit_bf16_dev#hnd` uses — and `elem` spells
        /// the non-type argument as `device::true_type::value` and
        /// `device::false_type::value`, which is the spelling that resolves
        /// under NVRTC's name-map pragma.
        ///
        /// **The cost is the offline typecheck, stated rather than
        /// discovered.** `abi::emit_device_typecheck` refuses a row whose
        /// `elem` head is a value, so neither of these two is checkable from
        /// `elem` alone — the same trade the four `Tu` rows in
        /// [`super::envelope`] make.
        ///
        /// `embed_vocab_shard`, the header's other `__global__`, is not
        /// declared: `new-horizon.md` §28.4 measured it as a second name for
        /// a job a reached row already does — `lower.rs` sends every `Embed`,
        /// sharded or not, to `layout::embed_bf16` — and no
        /// `vocab_offset`/`local_vocab` appears in `crates/model`,
        /// `model-loader` or `driver-cuda/src` at all.
        unit EMBED = "layout/embed",
            text = include_str!("../../csrc/src/layout/embed.cuh"),
            file = "layout/embed.cuh";

        /// `embed.cuh:60` — gather one row of the vocabulary table per token.
        ///
        /// Out-of-range token ids read row 0 rather than faulting —
        /// `embed.cuh:72`, `(tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0`
        /// — which is the kernel's decision and not the host's.
        ///
        /// `per_row` is `vec ? hidden / 8 : hidden` and is therefore NOT
        /// derivable from anything a `Dims` carries: it is the host's answer
        /// to the alignment test, handed to the kernel so the flat index can
        /// be split back into `(token, lane)`.
        fn embed = "layout::device::embed" (
            token_ids: *const i32,
            weight: *const bf16,
            y: *mut bf16,
            hidden: i32,
            vocab: i32,
            num_tokens: i32,
            per_row: i32,
        ) {
            "layout::embed#vec" => "device::true_type::value",
            "layout::embed#scalar" => "device::false_type::value",
        }
    }
}

/// The units `layout` compiles.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above gives. `families::ALL` reads this.
pub static UNITS: &[Unit] = &[
    deinterleave::DEINTERLEAVE,
    embed::EMBED,
    envelope::ENVELOPE,
    gather_rows::GATHER_ROWS,
    slot_ops::SLOT_OPS,
];

// ---------------------------------------------------------------------------
// The numbers, once each.
// ---------------------------------------------------------------------------

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
///
/// The block every pointwise rule in this tree uses, and the block the three
/// launches below that came from `Elementwise` take.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — `const MAX_BLOCK: u32 = 1024;`, the cap
/// `route_rows` clamps a row width to.
const MAX_BLOCK: u32 = 1024;

/// `LaunchRule::RouteRows`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1028-1034` — `grid [rows, 1, 1]`, `block [min(1024,
/// max(1, ceil(width / 32)) * 32), 1, 1]`, no shared memory. One block per
/// row, as wide as the row, rounded up to a warp and capped.
///
/// **The cap is safe only because the kernels stride.** Every caller below
/// walks `i += blockDim.x` and bounds on its own width, so a block narrower
/// than the row computes all of it in several passes. That is the rule's own
/// warning and it is why the four row-shaped launchers here can be written as
/// one expression.
///
/// # Which width, and why it is not always the obvious one
///
/// The rule read `Dims::width`, and `bind/mod.rs`'s `jit_dims` filled that
/// from `width_of(b, n_in + 0)` — **the FIRST OUTPUT's row width**
/// (`abi.rs:2028`, the emitted `jit_dims(b, spec, ctx, attn, rows,
/// width_of(b, n_in + 0), width_of(b, 0))`). For a split that is the LEFT
/// half and not the packed total. Each caller states which width it passes.
///
/// Not a `const fn`, where the rest of this file's helpers are: `Ord::min`
/// and `Ord::max` are not callable in a `const` context, and the rule's
/// expression is transcribed rather than rearranged.
#[must_use]
fn route_rows(rows: i32, width: i32) -> Launch {
    Launch::per_row(
        rows.unsigned_abs(),
        width
            .unsigned_abs()
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:828-834` and its `eval` arm at `:2450-2456` —
/// `n = dims.rows * dims.width`, then `grid [ceil(n / 256), 1, 1]`,
/// `block [256, 1, 1]`, no shared memory. The grid rounds UP, which is why
/// every kernel fired through it keeps its own element count as an operand.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning
// `Fired` so that "it declined" cannot be spelled like "it ran".
// ---------------------------------------------------------------------------

/// `layout::split_bf16_rows` — one packed row out to two.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was in `device::JIT_DISPATCHED` with `LaunchRule::RouteRows`, so
/// `deinterleave.cu`'s launcher went with the file and there was nothing to
/// port. [`route_rows`] is that rule, and the width it is given is
/// `left_dim`, because `jit_dims` filled `Dims::width` from the FIRST
/// output's row width and `Source::OutWidth(0)` is what the deleted row bound
/// `left_dim` from — the same number twice.
///
/// The kernel's loop bound is `left_dim + right_dim`, so a block sized on the
/// left half alone still reaches every element: `for (i = threadIdx.x; i <
/// total; i += blockDim.x)`, `deinterleave.cuh:199`. Reproduced rather than
/// improved, which is the port's first duty.
///
/// # Safety
///
/// `src` must address `n * (left_dim + right_dim)` live bf16 elements, `left`
/// and `right` `n * left_dim` and `n * right_dim` writable ones, and `stream`
/// must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn split_bf16_rows(
    src: *const bf16,
    left: *mut bf16,
    right: *mut bf16,
    n: i32,
    left_dim: i32,
    right_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    // `eval`'s own guard, `runtime/launch.rs:2464` — `RouteRows` refuses a
    // zero width rather than launching a zero-thread block.
    if left_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "left_dim" });
    }
    unsafe {
        deinterleave::raw::split_rows(
            "layout::split_bf16_rows",
            route_rows(n, left_dim),
            src,
            left,
            right,
            left_dim,
            right_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::split_qwen_gdn_ba_bf16` — Qwen's GDN bank, split by halves.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::RouteRows` over `Dims { rows, width: out_width(0) }`, and
/// `out_width(0)` is `v_h` — which is exactly what the deleted row bound
/// `v_h` from (`Source::OutWidth(0)`), so the block width and the operand are
/// the same number and cannot disagree.
///
/// # Safety
///
/// `ba` must address `n * 2 * v_h` live bf16 elements, `b_out` and `a_out`
/// `n * v_h` writable ones each, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn split_qwen_gdn_ba_bf16(
    ba: *const bf16,
    b_out: *mut bf16,
    a_out: *mut bf16,
    n: i32,
    v_h: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    unsafe {
        deinterleave::raw::split_qwen_gdn_ba(
            "layout::split_qwen_gdn_ba_bf16",
            route_rows(n, v_h),
            ba,
            b_out,
            a_out,
            v_h,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::deinterleave_rows_bf16` — gpt-oss's parity split, row-shaped.
///
/// # This launcher is NEW and has no trace-facing contract
///
/// §54 deleted this symbol's table row and its `dsl::cuda` wrapper together,
/// on a five-door consumer sweep that found nothing in any language calling
/// either. **What survived was the claim that the KERNEL is real**, and the
/// deleted family row put the distinction plainly:
///
/// > These three device rows stay because the KERNELS are real and their
/// > device text is in `layout/deinterleave.cuh`; what went is the claim that
/// > something asks for them. If a model wants one back, it comes back as a
/// > table row and a wrapper TOGETHER, with a caller.
///
/// So there is no `contract!` for it and this `fn` is how a caller would
/// arrive. Its geometry is the rule the deleted row stated,
/// `LaunchRule::RouteRows` ([`route_rows`]), over `H` — the width both output
/// rectangles have.
///
/// # Safety
///
/// `fused` must address `2 * rows * h` live bf16 elements, `gate_out` and
/// `up_out` `rows * h` writable ones each, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn deinterleave_rows_bf16(
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "h" });
    }
    unsafe {
        deinterleave::raw::deinterleave_rows(
            "layout::deinterleave_rows_bf16",
            route_rows(rows, h),
            fused,
            gate_out,
            up_out,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::deinterleave_vec_bf16` — the same split, one thread per element.
///
/// # This launcher is NEW and has no trace-facing contract
///
/// [`deinterleave_rows_bf16`]'s note applies verbatim. The geometry is
/// `LaunchRule::Elementwise` ([`elementwise`]) over `i` elements, and `i`
/// stays an operand because that rule rounds the grid UP to a whole block —
/// the tail threads of the last block are stopped by the kernel's own `if (i
/// >= I) return;` at `deinterleave.cuh:116`.
///
/// # Safety
///
/// `fused` must address `2 * i` live bf16 elements and `gate_out`/`up_out`
/// `i` writable ones each; `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn deinterleave_vec_bf16(
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    i: i32,
    stream: *mut c_void,
) -> Fired {
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        deinterleave::raw::deinterleave_vec(
            "layout::deinterleave_vec_bf16",
            elementwise(i.unsigned_abs()),
            fused,
            gate_out,
            up_out,
            i,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::concat_bf16_rows` — `[N, left] ++ [N, right]`.
///
/// # This launcher is NEW and has no trace-facing contract
///
/// [`deinterleave_rows_bf16`]'s note applies verbatim. `LaunchRule::RouteRows`
/// ([`route_rows`]) over `left_dim`, which is where `jit_dims` read
/// `Dims::width` from — the first output's width — and the kernel's loop
/// bounds on `left_dim + right_dim` and strides, so the narrower block
/// reaches every element in more passes.
///
/// # Safety
///
/// `left` and `right` must address `rows * left_dim` and `rows * right_dim`
/// live bf16 elements, `out` `rows * (left_dim + right_dim)` writable ones,
/// and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn concat_bf16_rows(
    left: *const bf16,
    right: *const bf16,
    out: *mut bf16,
    rows: i32,
    left_dim: i32,
    right_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if left_dim + right_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "left_dim + right_dim" });
    }
    unsafe {
        deinterleave::raw::concat_rows(
            "layout::concat_bf16_rows",
            route_rows(rows, left_dim),
            left,
            right,
            out,
            left_dim,
            right_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::gather_bf16_rows` — the epilogue's gather.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was in `device::JIT_DISPATCHED` with `LaunchRule::RouteRows`;
/// `gather_rows.cu` was deleted whole by §43 and held no launcher for it by
/// then. [`route_rows`] is the rule, over `width` — which the deleted row
/// bound from `Source::OutWidth(0)`, the same number `jit_dims` put in
/// `Dims::width`.
///
/// **`num_dst_rows` is the grid and not an operand.** The rule launches one
/// block per DESTINATION row and the kernel reads `blockIdx.x` as its slot,
/// so the count reaches it as a rectangle; that is why the `__global__` has
/// four parameters where the deleted table row had six.
///
/// # Why this had no row and no arm for so long
///
/// The table row's own note, which is a fact about the driver and not about
/// the kernel:
///
/// > `driver-cuda`'s shell built every fire row as `samples: true`, so
/// > `sampled < window.len()` was false on every fire and `lower::epilogue`
/// > never stated the gather. The moment the shell read the step's real
/// > readout list, every prefill asked for this and got `NoArm`.
///
/// # Safety
///
/// `src` must address the rows `row_indices` names at `width` u16 elements
/// each, `row_indices` `num_dst_rows` live `i32`s, `dst` `num_dst_rows *
/// width` writable u16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn gather_bf16_rows(
    src: *const u16,
    row_indices: *const i32,
    dst: *mut u16,
    num_dst_rows: i32,
    width: i32,
    stream: *mut c_void,
) -> Fired {
    if num_dst_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if width <= 0 {
        return Fired::Declined(Refusal::Empty { what: "width" });
    }
    unsafe {
        gather_rows::raw::gather_rows(
            "layout::gather_bf16_rows",
            route_rows(num_dst_rows, width),
            src,
            row_indices,
            dst,
            width,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::transpose_bf16_nld_to_lnd` — the PLE relay.
///
/// `[N, L, D] -> [L, N, D]`, so a layer reads a contiguous slice.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::Elementwise` ([`elementwise`]), whose `eval` arm computes
/// `n = dims.rows * dims.width` before flattening — `runtime/launch.rs:2451`.
/// `total` is the same number from the other side: the deleted row bound it
/// `Source::OutElements(0)`, which is `rows * out_width(0)`, and the result
/// rectangle is `[layers, n, dim]`, so `n * layers * dim` is that product.
/// **Computed here from the three extents the kernel divides by**, because
/// those are the three this `fn` is given and a `total` that disagreed with
/// them would round the grid to a different size than the bound it hands the
/// kernel.
///
/// `usize` and not `i32`: the kernel's index is `usize` and gemma-4's flat
/// `[N, layers*dim]` row is the widest thing in the fire.
///
/// # Safety
///
/// `src` and `dst` must address `n * layers * dim` live u16 elements, `dst`
/// writable, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn transpose_bf16_nld_to_lnd(
    src: *const u16,
    dst: *mut u16,
    n: i32,
    layers: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if layers <= 0 {
        return Fired::Declined(Refusal::Empty { what: "layers" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "ple_dim" });
    }
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    unsafe {
        gather_rows::raw::transpose_nld_to_lnd(
            "layout::transpose_bf16_nld_to_lnd",
            elementwise(u32::try_from(total).unwrap_or(u32::MAX)),
            src,
            dst,
            n,
            layers,
            dim,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::copy_if_valid_slot` — copy a slot's bytes if the slot is valid.
///
/// # The geometry is a LITERAL, and it is the one `<<<>>>` this file quotes
///
/// `layout/slot_ops.cu:59-62`, as the deleted row carried it:
///
/// ```text
/// :58   if (bytes == 0) return;
/// :59   constexpr int kThreads = 256;
/// :60   device::copy_if_valid_slot<<<1, kThreads, 0, stream>>>(
/// :61       src, dst, bytes, slot_ids, request);
/// ```
///
/// One block, whatever the rectangle, because `request` selects a single slot
/// and the loop strides the whole byte span. `kernels::LaunchRule::Single` was
/// written from this launcher and `attn/kv_paged.cu:516` together, and the
/// row's doc argued at length that a quotient reading — `RouteRows`' one
/// block per row, or `RowsFlat`'s `ceil(rows / 256)` — would launch
/// `dims.rows` blocks racing on one CSR or repeat the same copy `dims.rows`
/// times. That argument is unchanged; what changes is that the `1` is written
/// here rather than derived there. §5.1: **a kernel that fits neither
/// convenience writes the literal.**
///
/// `bytes == 0` is the launcher's own guard and it is a decline, not a
/// panic: a slot with nothing in it is a real thing a batch produces.
///
/// # Safety
///
/// `src` and `dst` must address `bytes` live bytes, `dst` writable,
/// `slot_ids` must be indexable at `request`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn copy_if_valid_slot(
    src: *const u8,
    dst: *mut u8,
    bytes: usize,
    slot_ids: *const i32,
    request: usize,
    stream: *mut c_void,
) -> Fired {
    // `slot_ops.cu:58`.
    if bytes == 0 {
        return Fired::Declined(Refusal::Empty { what: "bytes" });
    }
    unsafe {
        slot_ops::raw::copy_if_valid_slot(
            "layout::copy_if_valid_slot",
            Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0, smem_opt_in: false },
            src,
            dst,
            bytes,
            slot_ids,
            request,
            stream,
        );
    }
    Fired::Launched
}

/// `envelope.cu:37` and `:134` — `head_dim < 256 ? head_dim : 256`.
///
/// A block width off a cache extent, capped. Not a `LaunchRule`'s to state —
/// §21.14 refuses a rule whose block comes off a `Dims` field, because a
/// block width is the launcher's property and a fire can make no statement
/// about it true or false.
const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 {
        head_dim.unsigned_abs()
    } else {
        256
    }
}

/// `envelope.cu:71` — the seed's own block, which is fixed rather than
/// derived.
const SEED_BLOCK: u32 = 256;

/// `envelope.cuh:374`, `kEnvelopeFuseMaxTokens`.
///
/// Above this the fused kernel is unsound, not merely slower: it resets a
/// page it is the first writer of and merges into one it is not, deciding per
/// token, and that is only race-free while the launch is small enough that no
/// two blocks reach the same page. **This is a measurement and it survives
/// the port** — the constant is the C++'s, read from the header rather than
/// re-derived here.
const FUSE_MAX_TOKENS: i32 = 128;

/// `layout::envelope_merge_written_bf16` — fold explicitly-written KV rows
/// into the envelope planes. **One launch or two.**
///
/// `layout/envelope.cu:24`, `launch_envelope_merge_written_bf16`:
///
/// ```text
/// :41   if (num_tokens <= kEnvelopeFuseMaxTokens)
///           device::merge_written_fused<<<grid, threads, 0, stream>>>(...)
/// :49   else
///           device::reset_started_pages<<<grid, threads, 0, stream>>>(...)
/// :54       device::merge_written  <<<grid, threads, 0, stream>>>(...)
/// ```
///
/// with `grid = dim3(num_tokens, num_kv_heads)` (`:36`) and
/// `threads = head_dim < 256 ? head_dim : 256` (`:37`).
///
/// # THE FIRST §2.3 TWO-KERNEL BODY, and the ordering it carries
///
/// The two-launch arm is ORDERED: the reset writes the identity into pages
/// this batch is the first writer of, and the merge then reads what it wrote.
/// They are two launches on one stream for that reason and not for a register
/// budget, **so nothing here may reorder or fuse them.** A table row could
/// not say this — `families/layout.rs` refused the whole tier over it, in the
/// words *"A row is one symbol firing one launch; a row for either half
/// states half a contract"* — and a `fn` says it by being a `fn`.
///
/// `row_valid` may be null — the kernel tests it — so a null caller means
/// *"every row is valid"* and not *"no rows are"*.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_merge_written(
    k_curr: *const bf16,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: crate::x::abi::MaybeConst<u8>,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `envelope.cu:35`, split so the caller learns which extent was empty.
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // The cache-shape half of the same guard. One refusal for the two,
    // because a layer whose heads or channels are empty has no envelope to
    // maintain and no caller can act differently on which of them was zero.
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let launch = Launch {
        grid: [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        block: [threads_for(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    };

    if num_tokens <= FUSE_MAX_TOKENS {
        unsafe {
            envelope::raw::merge_written_fused(
                "layout::envelope_merge_written_fused_bf16",
                launch,
                k_curr,
                w_page,
                w_off,
                row_valid,
                env_min,
                env_max,
                num_tokens,
                num_kv_heads,
                head_dim,
                stream,
            );
        }
        return Fired::Launched;
    }

    unsafe {
        envelope::raw::reset_started_pages(
            "layout::envelope_reset_started_pages_bf16",
            launch,
            w_page,
            w_off,
            row_valid,
            env_min,
            env_max,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        );
        // `merge_written` takes no `w_off`: the reset above consumed it, and
        // this kernel folds every written row unconditionally. Dropping it
        // here is the C++'s argument list at `:54-57`, not an omission.
        envelope::raw::merge_written(
            "layout::envelope_merge_written_bf16",
            launch,
            k_curr,
            w_page,
            row_valid,
            env_min,
            env_max,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::envelope_seed_empty_bf16` — write the `+inf`/`-inf` identity
/// across a whole envelope pool.
///
/// So that a page no one has written yet reduces to *"nothing here"* rather
/// than to whatever the allocation held.
///
/// `layout/envelope.cu:62`, `launch_envelope_seed_empty_bf16`:
///
/// ```text
/// :76   device::seed_empty<<<blocks, 256, 0, stream>>>(env_min, env_max, n)
/// ```
///
/// with `n = num_pages * num_kv_heads * head_dim` in `usize` (`:69`) and
/// `blocks = (n + 255) / 256` (`:73`).
///
/// **The product is `usize` in both languages and that is load-bearing.** A
/// 64-page pool at 128 heads is nothing; a real one is `num_pages` in the
/// tens of thousands, and `num_pages * num_kv_heads * head_dim` overflows
/// `i32` before it overflows anything the kernel indexes with.
///
/// This is the one geometry in the tier a rule COULD have stated — it is a
/// flat elementwise grid — and the deleted row was `Unstated` anyway, because
/// `n` is a product of three cache extents and no fire's row count.
/// `Elementwise` would take `dims.rows`, and a page pool's size is not a
/// rectangle's height.
///
/// # Safety
///
/// Both planes are device addresses the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_seed_empty(
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `envelope.cu:68`.
    if num_pages <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_pages" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    unsafe {
        envelope::raw::seed_empty(
            "layout::envelope_seed_empty_bf16",
            Launch {
                grid: [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
                block: [SEED_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            env_min,
            env_max,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::envelope_update_appended_bf16` — fold the pages an append touched
/// back in.
///
/// The incremental form of the whole-cache rebuild: pages are append-only, so
/// re-reducing only the pages this append touched gives the answer a full
/// recompute gives, at the cost of the touched set instead of the cache.
/// (`launch_envelope_recompute_bf16`, the full form, was deleted by §54 for
/// exactly that reason; `device::recompute` is still in the header.)
///
/// `layout/envelope.cu:115`:
///
/// ```text
/// :137  device::update_appended<device::bf16><<<grid, threads, 0, stream>>>(
/// ```
///
/// with `grid = dim3(max_touched, num_kv_heads)` (`:133`) and
/// `threads = head_dim < 256 ? head_dim : 256` (`:134`).
///
/// **`max_touched` is a BOUND, not a count.** The caller computes it —
/// `kv_paged.cu:216` computed `(total_tokens + page_size - 1) / page_size +
/// num_requests`, the ceiling plus one straddle per request — and blocks past
/// a request's real page span early out. Nothing measures it, which is
/// precisely why no `LaunchRule` can state this grid: it is host arithmetic
/// over two fire extents and a cache extent.
///
/// Note that `max_touched` is a launch extent and is NOT in the kernel's
/// argument list; the `__global__` takes `num_requests`, `page_size`,
/// `num_kv_heads`, `head_dim` and reads its own `blockIdx.x`.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_update_appended(
    k_pages: *const bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_requests: i32,
    max_touched: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `envelope.cu:131-134`, split into the three answers a caller can act
    // on.
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_requests" });
    }
    if max_touched <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the touched-page bound" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0 {
        return Fired::Declined(Refusal::Empty {
            what: "the layer's kv heads, head_dim or page_size",
        });
    }

    unsafe {
        envelope::raw::update_appended(
            "layout::envelope_update_appended_bf16",
            Launch {
                grid: [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
                block: [threads_for(head_dim), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            k_pages,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            env_min,
            env_max,
            num_requests,
            page_size,
            num_kv_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `embed.cu:31` — `constexpr int BLOCK = 256;`.
///
/// The embed's own block. Equal to [`BLOCK`] and written separately because
/// the two numbers have different origins: this one is the launcher's, and
/// [`BLOCK`] is `runtime/launch.rs`'s rule constant.
const EMBED_BLOCK: u32 = 256;

/// `embed.cu:35` — the vector width, in `bf16` elements.
///
/// Eight `bf16` is sixteen bytes, which is why the alignment test is 16 and
/// the divisibility test is 8. One constant, two tests.
const VEC_WIDTH: i32 = 8;

/// `(uintptr_t)p % 16 == 0`, which is what `fire::hand::aligned16` was.
///
/// The predicate the C++ spelled inline at `embed.cu:34-35`. It moves with
/// the one caller that is left in this crate; `driver-cuda`'s copy stays
/// where its other callers are.
#[cfg(feature = "_cuda")]
#[must_use]
fn aligned16(p: *const c_void) -> bool {
    (p as usize) % 16 == 0
}

/// `embed.cu:33-35` — the host test that picks `VEC`.
///
/// ```text
/// :33   const bool vec = (hidden % 8) == 0 &&
/// :34                    (reinterpret_cast<std::uintptr_t>(weight) % 16) == 0 &&
/// :35                    (reinterpret_cast<std::uintptr_t>(y) % 16) == 0;
/// ```
///
/// Transcribed with `%` kept as `%`: `hidden % 8` is an `int` remainder and
/// the two pointer tests are `& 15` in every sense that matters.
///
/// Public because it is the whole of what the row world could not express —
/// `layout/embed.cuh:18-25` refused a row over it, in the words *"No `Source`
/// in `kernels/src/lib.rs` produces 'is this pointer 16-byte aligned'"* — and
/// a caller staging its own buffers can ask the same question this `fn` asks.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn vectorisable(hidden: i32, weight: *const bf16, y: *const bf16) -> bool {
    hidden % VEC_WIDTH == 0 && aligned16(weight.cast()) && aligned16(y.cast())
}

/// `layout::embed_bf16` — the first launch of every fire.
///
/// Gathers one row of the vocabulary table per token. Out-of-range token ids
/// read row 0 rather than faulting — `embed.cuh:72`, `(tid_raw >= 0 &&
/// tid_raw < vocab) ? tid_raw : 0` — which is the kernel's decision and not
/// this file's.
///
/// # The measurement, carried rather than consumed
///
/// `layout/embed.cuh:27-31`:
///
/// > The vectorised form is not an optimisation to drop: at decode the
/// > token-per-block form issued 24 dependent 2-byte loads from 8 blocks and
/// > ran at 8 GB/s — the row it reads is a random offset into the largest
/// > tensor in the model, so the access is a cold TLB miss whose latency only
/// > a wide grid hides.
///
/// That is why this takes the scalar arm rather than always taking it, and
/// why the arm is not "the slow path we could delete".
///
/// # The `VEC` choice is the whole file
///
/// `layout/embed.cuh:18-25` is the argument that kept this launcher in C++
/// through four passes:
///
/// > `embed` is NOT a row and is not templated. Its `VEC` parameter is chosen
/// > on the HOST from a run-time test the device cannot make — `hidden % 8 ==
/// > 0` and both `weight` and `y` 16-byte aligned — and the element count it
/// > launches over is `num_tokens * (vec ? hidden/8 : hidden)`, an extent
/// > that depends on the answer. No `Source` in `kernels/src/lib.rs` produces
/// > "is this pointer 16-byte aligned", and `new-horizon.md` §10.5 refuses an
/// > invented one.
///
/// Every clause of that is still true, and none of it is an argument for C++
/// or for a row. It is an argument that the choice is a HOST PROGRAM, which
/// is what this is — and under fn-world the choice is one `if` rather than a
/// `Control::Switch` classification beside a row.
///
/// # Geometry
///
/// ```text
/// :36   const int per_row = vec ? hidden / 8 : hidden;
/// :37   const long long total = static_cast<long long>(num_tokens) * per_row;
/// :38   dim3 grid(static_cast<unsigned>((total + BLOCK - 1) / BLOCK));
/// :39   dim3 block(BLOCK);
/// :41   device::embed<true><<<grid, block, 0, stream>>>(...)
/// :47   device::embed<false><<<grid, block, 0, stream>>>(...)
/// ```
///
/// The `long long` is transcribed as `i64` and matters: `num_tokens * hidden`
/// for a 128k-token prefill against an 8192-wide model overflows `i32`, and
/// the C++ widened before multiplying for exactly that reason. The grid is
/// then narrowed to `u32`, which is the cast the C++ spells as
/// `static_cast<unsigned>`.
///
/// # Safety
///
/// `token_ids` must address `num_tokens` live `i32`s, `weight` `vocab *
/// hidden` live bf16 elements, `y` `num_tokens * hidden` writable ones, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn embed_bf16(
    token_ids: *const i32,
    weight: *const bf16,
    y: *mut bf16,
    num_tokens: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    // `embed.cu:32`, split so the caller learns which extent was empty.
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // A zero-width model, which is a bug upstream rather than a shape; the
    // C++ returned rather than launching an empty grid and so does this.
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    let vec = vectorisable(hidden, weight, y.cast_const());
    let per_row = if vec { hidden / VEC_WIDTH } else { hidden };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let launch = Launch {
        grid: [blocks, 1, 1],
        block: [EMBED_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    let symbol = if vec { "layout::embed#vec" } else { "layout::embed#scalar" };
    unsafe {
        embed::raw::embed(
            symbol, launch, token_ids, weight, y, hidden, vocab, num_tokens, per_row, stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// The declaration the readers that cannot call read.
//
// Seven contracts, carrying `table/layout.rs`'s seven rows minus everything
// that described a launcher. Five of the seven get a bind below; the two
// stash pseudo-symbols do not, and that is `route`'s rule rather than an
// omission — see this file's header.
//
// `Contract::DEFAULT` supplies the other nine fields of each. `needs`,
// `lacks`, `depth_prefix_plan`, `publishes_aux` and `lowered_as` are stated
// by nothing here, as they were by nothing in the rows these replace.
// ---------------------------------------------------------------------------

contract! {
    /// One packed row out to two — `[N, left+right] -> [N, left], [N, right]`.
    ///
    /// Two results, so both widths come off the results and the source needs
    /// no extent of its own. `model/src/qwen_3_5/forward/mod.rs:547` states
    /// it through `dsl::cuda::split_rows`.
    SPLIT_ROWS = "layout::split_bf16_rows" as split_rows

    /// Qwen's GDN bank, split by HALVES where the parity deinterleave splits
    /// by parity. Same shape, different layout, checkpoint decides — which is
    /// why they are two kernels and not one with a flag.
    /// `model/src/qwen_3_5/forward/mod.rs:553` states it through
    /// `dsl::cuda::split_qwen_gdn_ba`.
    SPLIT_QWEN_GDN_BA = "layout::split_qwen_gdn_ba_bf16" as split_qwen_gdn_ba

    /// The embedding gather — the first launch of every fire.
    ///
    /// **`weight` is a NAMED weight and never a slot in the argument run**: a
    /// vocab table is not something a trace produces, so the embedding's
    /// weight is only ever the statement's own name. That was
    /// `Source::WeightNamed` on the deleted row and is `Cx::weight_named(0)`
    /// in the bind.
    ///
    /// No `dsl::cuda` wrapper names this symbol and it is stated all the
    /// same: `model-compiler/src/lower.rs:1505` is `Embed { .. } =>
    /// Semantic::Kernels(&["layout::embed_bf16"])`, which is the door a
    /// wrapper sweep alone would miss.
    EMBED = "layout::embed_bf16" as embed

    /// The epilogue's gather: collect the rows a prefill actually samples,
    /// before the final norm and the head.
    ///
    /// Like [`EMBED`] it has no `dsl::cuda` wrapper and is reached from
    /// `model-compiler/src/lower.rs:1321`, inside `epilogue()`.
    GATHER_ROWS = "layout::gather_bf16_rows" as gather_rows

    /// The PLE relay: `[N, L, D] -> [L, N, D]`, so a layer reads a contiguous
    /// slice. Addressing, not arithmetic. `model/src/gemma_4/forward/mod.rs:173`
    /// states it through `dsl::cuda::transpose_nld_to_lnd`.
    TRANSPOSE_NLD_TO_LND = "layout::transpose_bf16_nld_to_lnd" as transpose_nld_to_lnd

    /// PSEUDO-SYMBOL, deliberately unstated — and one of the three
    /// `launch_abi::the_pseudo_symbol_rows_are_exactly_the_known_three`
    /// enforces.
    ///
    /// It names an OPERATION of the declared executor, not a C++ launcher:
    /// the driver stashes the speculative verify pass's state as one dispatch
    /// case, so there is no single function whose signature a declaration
    /// could state. **No `Entry`**, because `execution::SERVICE` already
    /// answers `DriverOp` for it and an `Entry` would shadow that arm — see
    /// [`crate::x::route`]'s "THE ONE OVERLAP".
    VERIFY_STASH_STORE = "qwen35_verify_stash_store" as verify_stash_store

    /// [`VERIFY_STASH_STORE`]'s other half, on the same terms.
    VERIFY_STASH_LOAD = "qwen35_verify_stash_load" as verify_stash_load
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Five binds, no `none:` arms, and two contracts with no arm at all.
//
// The absence of a `none:` here is worth a sentence, because `rope` had four
// and the family shape suggested more: every operand `table/layout.rs` left
// unsourced was unsourced for a reason that a `Cx` query now answers.
// `token_ids`, `vocab`, `sampling_indices` and `ple_dim` were the four, they
// were `Source::Ctx`/`Source::SamplingIndices` on the rows, and they are four
// methods on `Cx` — added by the floor for this port, which is why this
// family declines nothing at load.
//
// The two contracts with NO arm are not refusals either. They are driver ops,
// and `route`'s "THE ONE OVERLAP" is explicit that a `none:` for one would be
// wrong: it would make `route` answer `Unbound` and refuse at load a symbol
// that fires correctly today.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    SPLIT_ROWS => { cx, stream => {
        unsafe {
            split_bf16_rows(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.out_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    SPLIT_QWEN_GDN_BA => { cx, stream => {
        // `v_h` off `Source::OutWidth(0)`, which is also the width the
        // `RouteRows` block was sized on. One number, read once.
        unsafe {
            split_qwen_gdn_ba_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    EMBED => { cx, stream => {
        // The deleted row's seven sources, in order: `Ctx("token_ids")`,
        // `WeightNamed`, `Out(0)`, `Rows`, `OutWidth(0)`, `Ctx("vocab")`,
        // and a stream that was never an operand.
        unsafe {
            embed_bf16(
                cx.token_ids()?,
                cx.weight_named(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.vocab()?,
                stream,
            )
        }
        .ok()
    }},

    GATHER_ROWS => { cx, stream => {
        // `row_indices` is `Cx::sampling_indices`, which the row spelled
        // `Source::SamplingIndices`. It is null when the fire samples every
        // row — the decode case, where the lowering states no gather at all
        // — so the query answers `Refusal::Unstated` rather than handing the
        // kernel a null it would dereference on its first line.
        unsafe {
            gather_bf16_rows(
                cx.arg_in(0)?.cast_const().cast::<u16>(),
                cx.sampling_indices()?,
                cx.arg_out(0)?.cast::<u16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    TRANSPOSE_NLD_TO_LND => { cx, stream => {
        // The row read `layers` as
        // `Div(Width(In(0)), CtxNonZero("ple_dim"))` and `dim` as
        // `Ctx("ple_dim")`, and its note said the division is "exactly the
        // arithmetic the hand-written arm did, refusal on an unset `ple_dim`
        // included". `Cx::ple_dim` IS that refusal — the `NonZero` was the
        // row world saying `Option` in the only vocabulary it had — so the
        // guard below is the one thing the `Div` could not spell: a zero
        // divisor that reached the kernel as a layer count.
        let ple_dim = cx.ple_dim()?;
        if ple_dim <= 0 {
            return Err(Refusal::Empty { what: "ple_dim" });
        }
        unsafe {
            transpose_bf16_nld_to_lnd(
                cx.arg_in(0)?.cast_const().cast::<u16>(),
                cx.arg_out(0)?.cast::<u16>(),
                cx.rows().count,
                cx.in_width(0)? / ple_dim,
                ple_dim,
                stream,
            )
        }
        .ok()
    }},
}
