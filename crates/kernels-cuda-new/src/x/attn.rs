//! `attn` in fn-world — §5 step 5's last family, and the largest by a wide
//! margin.
//!
//! Twenty-three `.cuh` roots, forty-one table rows and fifteen host programs
//! in `driver-cuda/src/fire/`. This file takes the self-contained leaves
//! first: **five roots, six rows**. The rest of the
//! family stays in row-world, which is what [`crate::x::route`]'s
//! [`Route::Rows`](crate::x::Route::Rows) fallthrough is for — `table::attn`
//! keeps its place in `table::ROW_TABLES` and `families::attn` keeps its
//! place in `families::ALL`, and every row this pass did not take fires
//! exactly as it did before.
//!
//! # What crossed, and the arrangement each root got
//!
//! | root | rows declared | contracts | binds | table rows deleted |
//! |---|---|---|---|---|
//! | [`attn_sink`] | 2 | 2 | 2 | `lse_log2_to_ln`, `attention_sink_rescale` |
//! | [`attn_res`] | 1 | 1 | 1 | `attn_res_blend` |
//! | [`head_dim_pad`] | 2 | 2 | 2 | `pad_head_dim`, `strip_head_dim` |
//! | [`softcap`] | 2 | 1 | 1 | `logit_softcap` |
//! | [`split_packed`] | 2 | 0 | 0 | — |
//! | [`pack_dense_mask`] | 2 | 0 | 0 | — |
//! | [`dsa_indexer`] | 3 | 0 | 0 | — |
//! | [`mla_naive`] | 2 | 0 | 0 | — |
//! | [`kimi_mla`] | 2 | 2 | 2 | `kimi_split_kv_a_norm`, `kimi_split_q_b` |
//! | [`mla_paged`] | 2 | 0 | 0 | — |
//! | [`mla_fa2`] | 6 | 1 | 1 | `attention_mla` (unbound) |
//! | [`qkv_fused`] | 11 | 1 | 1 | `qkv_packed_post` |
//! | [`dsv4_compress`] | 10 | 3 | 1 | `dsv4_compress_gather_paged`, `dsv4_store_comp_entries` (both unbound), `combine_attn_outputs` |
//! | [`kv_paged`] | 20 | 0 | 0 | 7 host programs moved; 4 rows on 4 queries |
//!
//! Fourteen of forty-one. Twenty-seven rows remain in `table/attn.rs` —
//! thirty-five five passes ago, `0dc8e9e9b` took
//! `attn::attention_xqa_decode_bf16_prepared` since, [`kimi_mla`] took two
//! here, [`mla_fa2`] took `attn::dispatch_attention_mla_bf16`, [`qkv_fused`]
//! took `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` and
//! [`dsv4_compress`] took three more. **Six roots moved no row at all**,
//! all six unit-only: [`split_packed`], [`pack_dense_mask`], [`dsa_indexer`],
//! [`mla_naive`], [`mla_paged`] and now [`kv_paged`].
//!
//! [`kv_paged`] is the biggest of the six by a wide margin — twenty rows,
//! more than the next two roots together — and it is unit-only because its
//! four `table/attn.rs` rows are a second named half. **That half is THE
//! MOVE and the move is DONE**: seven host programs came out of
//! `driver-cuda/src/fire/kv_paged.rs` and are `pub unsafe fn`s in
//! [`kv_paged`], called from the driver's four shim entries and four
//! preludes. None of them needed a driver RESOURCE, which is the whole
//! discriminator.
//!
//! **The four rows still stand, and deliberately.** They are blocked on four
//! facts `AttnCtx` carries and `Cx` does not state — `first_token`,
//! `num_pages_in_batch`, `w_page_d`, `w_off_d` — which is four `query!`
//! lines and four field reads. A `contract!` written before them would mint
//! an `Entry` with no arm, and `write_kv_to_pages` fires once per layer of
//! every fire. A row that still fires is worth more than a `Route::Unbound`.
//! The table at [`kv_paged`] is the ask.
//!
//! (The first statement of this paragraph said they were blocked because
//! `kernels-cuda-new` cannot call `driver-cuda`. **That is true and it is not
//! the reason** — the dependency runs the other way, and this file's
//! `x::layout` neighbours are already called from that very module. The
//! correction is at [`kv_paged`], in place.)
//!
//! **[`mla_fa2`] IS NO LONGER ONE OF THEM, and its row is the one that
//! mattered.** Crossing it deleted
//! `crates/kernels-cuda/csrc/src/attn/attention_mla.cu` — the last two
//! nvcc-compiled `<<<>>>` in the workspace — and took
//! `kernels-cuda/tests/sources.rs`' `EXPECTED` to **0**.
//!
//! It crossed **unbound**, and that is a real crossing rather than a
//! deferral: a contract with no `operands` is what `abi::emit_c_shim` reads
//! to mean "no ahead-of-time entry", so the shim entry the `.cu` was the
//! definition of stopped being emitted and the file stopped having a
//! declaration to satisfy. What the `none:` arm withholds is the FIRE, not
//! the crossing. Both host programs exist — [`mla_fa2`] here and
//! `driver-cuda/src/fire/mla_naive.rs` for sm_100 — and `Cx` states neither
//! the MLA cache layer, the plan handle, the attention workspace nor
//! `sm_scale`, which is verbatim the reason `executor_bind.rs:1519` gives for
//! the row never having been armed in the first place. **It never fired, and
//! a row that never fires while holding an nvcc translation unit hostage is
//! worth strictly less than the contract that replaces it.**
//!
//! **Eleven roots, and the eleventh is a NEW `.cuh`.**
//! `csrc/src/attn/attention_mla_fa2.cuh` is not one of the twenty-three: it
//! is an NVRTC root in `csrc/src/attn/fa2.cuh`'s idiom — includes, four
//! `using`s, one alias template, one `__device__` echo, no `__global__` of
//! ours — written so the six rows can name upstream's kernel. The root it
//! replaced is `crates/kernels-cuda/csrc/src/attn/attention_mla.cu`, which
//! was host program throughout and is now deleted.
//!
//! **Unit-only is a real arrangement and five roots now use it**, which is
//! enough to state the rule: a root whose host programs are already Rust and
//! already outside a `bind!` crosses as a `unit!` and nothing else. The device
//! text belongs where the family is; the program belongs where it already
//! runs. `x/driver_internal.rs` says the first half — *"the rows stay where
//! the device text is"* — and these five are the second.
//!
//! **The rule's consequence is a schedule — with one correction the owner
//! asked for.** The framing offered was that the rows left in
//! `attn::KERNELS` are exactly the roots with `bind!` consumers.
//! **`dsa_indexer` disagrees, and so do `mla_naive` and `mla_paged`**: all
//! three crossed unit-only, their host programs are Rust in `fire/`, and
//! every one of their table rows STAYED. A row is the trace-facing dispatch
//! entry; it survives until a CONTRACT replaces it, whatever world the host
//! program lives in. **A `unit!` moves device text; only a contract retires a
//! row** — and [`mla_fa2`] is the correction to the earlier version of that
//! sentence, which said `bind!`. A contract with no `operands` retires the
//! shim entry whether or not it has a bind, which is precisely how the
//! unbindable row left.
//!
//! # §66 — the row count is the whole of what is left in the CUDA lane
//!
//! `kernels-cuda/native` is the ONLY switch over the entire nvcc and `.cpp`
//! surface in the workspace, and the only thing that turns it on is
//! `driver-cuda/bridge`, which is deletable when `ROW_TABLES` empties. So
//! nvcc-zero, `.cpp`-zero and step 6 half B are not three goals — they are
//! three consequences of `attn`'s twenty-eight, and `moe`'s four have gone.
//!
//! **THE LAST TWO `<<<>>>` IN THE TREE ARE GONE.** They were
//! `mla_naive_paged_kernel` and `mla_mma_paged_kernel` in `attention_mla.cu`,
//! behind `attn::dispatch_attention_mla_bf16`, and that row crossed here.
//! Their device text is [`mla_naive`]'s `.cuh` and their host program is
//! `driver-cuda/src/fire/mla_naive.rs`; neither moved. What moved is the FA2
//! arm, which had no Rust at all, and the row, which is what the file's
//! shim entry hung on. `kernels-cuda/csrc/CMakeLists.txt`'s
//! `PIE_CUDA_GRAPH_KERNEL_SOURCES` is now an empty list, and the only file
//! this repository still asks nvcc to compile is `moe/flashinfer_moe.cu` in
//! a different target.
//!
//! # `ArgValue::Bytes` — §5.1's standing question, ANSWERED and CLOSED, and
//! # not the way the warning predicted
//!
//! §5.1 warned eleven families that *a wrong bypass is a launch with a
//! garbage struct, not a type error*, and asked each of them to be the first
//! family-level caller. `attn` is that family, via `MLAParams`, and the
//! answer is that **the bypass was never what blocked it.**
//!
//! `by_value!`'s grammar required `tag = $tag:ident` and asserted
//! `Ty::$tag.needs_mirror()` — a closed list of six kinds that does not
//! include `MLAParams`. So **an open set of `Abi` impls was gated behind a
//! closed set of `Ty` tags**, and the obvious fix, a seventh variant, is the
//! one `x/abi.rs:415-417` argues against in its own words. The floor took the
//! other patch instead: an UNTAGGED arm, keeping every assertion and dropping
//! only the permission. [`mla_params`] is its first caller and the full
//! account is in that module's doc.
//!
//! `x/xqa.rs`'s `KvCacheList` was the tree's only `by_value!` and it worked
//! because `Ty::KvCacheLayerView` happened to already exist and happened to
//! mean roughly the right thing. That is why eleven families produced no
//! second one.
//!
//! What the measurement itself said, because it is worth the header:
//! `sizeof(MLAParams) = 288`, and a transcription would have written 248 —
//! **forty bytes short**, because `uint_fastdiv` measures twenty-four bytes
//! and not four, twice. And `PROFILER_PARAMS_DECL` sits in the middle of the
//! struct, expanding to a pointer or to nothing depending on a macro defined
//! in another file: `-DFLASHINFER_ENABLE_PROFILER=1` moves `sizeof` to 296
//! and every field after `work_indptr` by eight. That is `x/xqa.rs`'s
//! `ENABLE_4BIT_KV_CACHE` hazard in a second family, which makes it a pattern.
//!
//! `softcap` declares two device rows against one table row on purpose, and
//! §3.2's hazard is the reason — see [`softcap`].
//!
//! # What did NOT cross, and why — the useful half of this header
//!
//! **`attn/softcap.cuh` was blocked on the floor and is no longer.** Its cap
//! came from `Source::CtxNonZero("final_logit_softcap")` — `DispatchCtx`'s
//! field at `driver-cuda/src/bind/mod.rs:1161`, *"gemma's FINAL logit softcap
//! (`cap * tanh(x / cap)` over the logits)"* — and [`Facts`] had no method
//! that reached it. `Facts::final_logit_softcap() -> Option<f32>` landed in
//! `a46bdbbe0` with the driver impl `(self.ctx.final_logit_softcap > 0.0)
//! .then_some(…)`, which is `CtxNonZero`'s reading moved into the type: zero
//! is ABSENCE, not a cap of zero, and a cap of zero would scale every logit
//! to nothing. The root crossed in the same pass that asked for it.
//!
//! **A `none:` arm would have been wrong here and it is worth saying why**,
//! because the shape recurs: `none:` surfaces as
//! [`Route::Unbound`](crate::x::Route::Unbound) at model LOAD, so a `none:`
//! for a symbol that fires correctly today would have refused every Gemma-2,
//! Gemma-3 and Gemma-3n deployment at load. The row world is not a fallback
//! for a bind that cannot be written; it is where a kernel legitimately
//! stays until one can be, and asking for the accessor is what the floor's
//! *"say so with the exact patch you want"* is for.
//!
//! **`attn/split_packed.cuh` crossed as a UNIT AND NOTHING ELSE**, which is
//! a fifth thing a root can do and is the honest shape for it: both of its
//! kernels already have host programs and neither host is a bind. The
//! non-devwin half, `attn::split_qkv_bf16`, is
//! [`crate::x::driver_internal::split_qkv_bf16`] — the fourth arrangement,
//! no contract because no trace can state it. The devwin half **cannot** be
//! bound and it is not a floor gap: it takes BASE pointers — the window
//! lives in device memory so a captured graph can replay across row splits
//! without re-recording — and [`Cx::arg_in`]/[`Cx::arg_out`] read
//! `BoundLaunch::args`, which `resolve_arg_windowed` has already offset by
//! the region's first row, so a bind would hand the kernel pointers it
//! windows a second time. Its host program stays
//! `driver-cuda/src/fire/split_packed.rs`, its row stays a row, and only the
//! device text moved — which is what `x/driver_internal.rs` says the
//! destination is.
//!
//! **`attn/pack_dense_mask.cuh` crossed the same way, and its blocker was
//! answered by `x/xqa.rs`'s precedent rather than by a floor patch.** The
//! mirror lives in [`params`], in THIS crate, for `x/xqa.rs`'s reason:
//! `unit!` has to name the type and `driver-cuda` depends on
//! `kernels-cuda-new` rather than the other way round. `driver-cuda`'s
//! `bind/abi.rs` now re-exports it, so there is one definition and one
//! measured layout. The `Abi` impl is hand-written beside the mirror and not
//! a `ptr_abi!` in `x/abi.rs`, because `ptr_abi!` is private to that module
//! and `x/abi.rs`'s own header asks for exactly this: *"adding a crossing
//! type means writing one impl, next to the kernel that needed it, and
//! nothing else in the tree changes."* **No floor patch was needed.**
//!
//! Both kernels are unit-only, like `split_packed`'s two: their
//! `table::driver_internal` rows are already gone, deleted with
//! `attn/pack_dense_mask.cu` and its `.hpp`, and
//! `driver-cuda/tests/launch_abi.rs:651-654` records why — *"Empty consumer
//! set on all five channels; not ported, per §60.1."* Nothing in the tree
//! launches either one today. So: `unit!` yes, `contract!` no, `bind!` no,
//! and a `none:` arm would be wrong for the reason §5.1 gives — it would
//! surface at model load as a refusal for a symbol no trace can state.
//!
//! # THE DEFECT THE MIRROR SURFACED: `Ty::StructuredMasks` names a type
//! # that no longer exists
//!
//! `kernels/src/lib.rs:1079` spells `Ty::StructuredMasks` as `"const
//! ::pie_cuda_driver::kernels::attn::StructuredMaskParams*"` — namespace
//! `attn`, which was `pack_dense_mask.hpp`'s. **That header is deleted**, and
//! a sweep of `*.hpp`, `*.cuh`, `*.cu` and `*.rs` finds exactly one
//! definition left: `attn::device::StructuredMaskParams`, at
//! `pack_dense_mask.cuh:136`. The host spelling names nothing.
//!
//! It is not a live break, because `emit::crossing` refuses the operand
//! before `Ty::cpp()` is ever asked for it, and the two rows that could have
//! asked are gone. It is a spelling waiting to be wrong. [`params`] states
//! the DEVICE spelling, which is the one NVRTC resolves, and records the
//! divergence rather than editing `crates/kernels` mid-sweep.
//!
//! `pack_dense_mask.cuh`'s own header claims a check that also died with the
//! `.cu`: *"`pack_dense_mask.cu` includes BOTH definitions and
//! `static_assert`s size, alignment and all three field offsets against each
//! other."* Those five `static_assert`s were the entire argument for a second
//! definition of a three-`u32` POD. The re-export removes the second
//! definition instead of restoring the check, which is the stronger answer.
//!
//! **`attn/page_compact.cuh` is a row that can cross and a program that
//! cannot yet.** Every one of its eleven operands is unsourced —
//! `scratch_counts` is a driver-owned scratch buffer and `keep_stride` comes
//! off a host CSR, and no `Source` spells either — so no `Cx` query reaches
//! them and the bind that would exist is the empty one. It is left for the
//! pass that moves `FirePageMask` with it; splitting a program from the only
//! caller that can supply its buffers would leave both halves half-done.
//!
//! **The FA2 and XQA lattices are deliberately untouched**, per §5.1: 56
//! units on one root and six on another, both already NVRTC-native, neither
//! blocking anything.
//!
//! # `Facts::plan()` and `Facts::slab()` were not exercised
//!
//! §5.1 names `attn` as where they are first used and most likely to be
//! wrong. They are not used HERE: all three roots this pass took are
//! pointwise or per-head corrections applied AFTER an attention kernel has
//! written its output, and none of them reads a page CSR or a state slab.
//! [`Cx::plan`] IS NOW EXERCISED, by [`kv_paged`], and it is right: its six
//! fields are exactly the four CSR arrays, `row_valid` and `num_requests`
//! that the two append rows source from `AttnCtx` one at a time, and
//! `bind/facts.rs:505` fills all six by direct field copy. It states no
//! `first_token`, which is the one thing those rows also need and the first
//! entry in [`kv_paged`]'s ask — a plan is what the fire's geometry is, not
//! where a partial write resumes from, so that is a separate query and not a
//! seventh field.
//! Agent `sweep-ssm` is exercising [`Cx::slab`] concurrently.
//!
//! # `ArgValue::Bytes` was not exercised either
//!
//! §5.1 names `attn` (`MLAParams`/`HopperParams`) as its first family-level
//! caller. Not in this pass: every parameter of these five kernels is a
//! pointer or a scalar, and `unit!` needs no new grammar for either. The
//! by-value aggregate is still owed a first caller — `x/xqa.rs`'s
//! `KvCacheList` is the only `by_value!` in the tree and it is agent
//! `xqa-nvrtc`'s. **The failure mode §5.1 warns about therefore remains
//! untested**: a wrong bypass is a launch with a garbage struct rather than
//! a type error, and the typecheck TU is the only thing that would catch it.
//!
//! # §3.2's two-formats-one-width hazard IS exercised, and this is the first
//!
//! `softcap` is the only place in `attn` where one template is instantiated
//! at two sixteen-bit formats, and §5.1 named it as live for this family.
//! What the hazard is, concretely: `bf16` and `f16` are both sixteen bits and
//! both `unsigned short` to any C ABI, so **the two rows the row world wrote
//! for them are byte-identical apart from the symbol string** — same
//! `LaunchRule::Elementwise`, same `in_place = &[(0, 0)]`, same three
//! operands at the same `Ty`s from the same `Source`s. The only thing that
//! told them apart was `DeviceKernel::elem`, which lives on the UNIT row and
//! not on the table row, and `Ty::BufMut` is a `void*` whose element type is
//! whatever that `elem` said. Feed a bf16 buffer to the f16 symbol and it
//! binds, launches, and computes `cap * tanh(x / cap)` over a reinterpretation
//! of the same bits — neither format has a trap representation, so the answer
//! is finite, plausible and wrong.
//!
//! In fn-world `x/abi.rs` makes them distinct unit structs, so
//! [`logit_softcap_bf16`] takes `*mut bf16`, [`logit_softcap_f16`] takes
//! `*mut f16`, and no caller can pass one for the other without writing a
//! cast. **The residue, stated rather than glossed:** what pairs a symbol
//! STRING with a type parameter is still a human decision, made in the two
//! host `fn`s below and nowhere else. The port does not eliminate it; it
//! reduces it from every call site to two adjacent lines, and it puts a
//! check on the other end — `unit!`'s `where [T = f16] "device::f16"` feeds
//! `abi::emit_device_typecheck`, which spells the parameter
//! `::pie_cuda_driver::kernels::device::f16*` against the `__global__`'s own,
//! so a row whose type disagrees with the DEVICE TEXT is a C++ compile error
//! naming the symbol. A row whose type disagrees with the host `fn`'s cast is
//! the one gap, and both are in this file, twenty lines apart.
//!
//!
//! Every launch below is a [`Launch`] literal or one of its two conveniences,
//! and every one cites the `<<<>>>` or the `LaunchRule` it came from. Four of
//! the five had **no host program at all** before this file — they were rows,
//! and the generated dispatch arm built their grid from the rule — so the
//! citation is the rule function in `runtime/launch.rs` plus the `<<<>>>` the
//! rule's own doc was checked against. Nothing here is invented.
//!
//! [`Facts`]: crate::x::Facts
//! [`Cx::arg_in`]: crate::x::Cx::arg_in
//! [`Cx::arg_out`]: crate::x::Cx::arg_out
//! [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
//! [`Cx::plan`]: crate::x::Cx::plan
//! [`Cx::slab`]: crate::x::Cx::slab

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;
use kernels::{Cap, Prepare};

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// THREE `unit!` INVOCATIONS CANNOT SHARE A SCOPE — each emits `UNITS`, `ROWS`,
// `PARAMS` and `raw` at module scope. Each root gets a module and the family
// re-exports the three below, which is `x/layout.rs`' arrangement for five and
// `x/mlp.rs`' for two.
// ---------------------------------------------------------------------------

/// `attn`'s `#[repr(C)]` mirrors of C++ aggregates, and their measured
/// layouts.
///
/// One struct today. It is a module rather than three items at file scope
/// because `x/xqa.rs` is the precedent and its shape is worth matching
/// exactly: the mirror, its [`Abi`](crate::x::Abi) impl, and a [`Layout`]
/// carrying numbers **measured out of NVRTC's PTX** with the probe that
/// measured them named beside them. `MLAParams` and `HopperParams` land here
/// when `mla_*` crosses; nothing is written for them in advance, because
/// §0's rule is data only for what has a reading consumer.
///
/// [`Layout`]: crate::x::Layout
pub mod params {
    use core::ffi::c_void;

    use kernels::Ty;

    /// One lane's structured-mask descriptor, as
    /// `attn::pack_structured_mask` reads it.
    ///
    /// Mirrors `csrc/src/attn/pack_dense_mask.cuh:136`.
    ///
    /// # Where the numbers came from
    ///
    /// **Not from reading the header, and not from the driver's mirror.**
    /// Measured by `nvrtc-probes/attn_structured_mask.py`, whose method is
    /// `nvrtc-probes/params_layout.py`'s: emit `(unsigned)((char*)&((P*)0)->f
    /// - (char*)(P*)0)` into a `__constant__` array and read the initialiser
    /// back out of the PTX. `offsetof` and `__builtin_offsetof` are both
    /// unavailable under NVRTC; only the pointer DIFFERENCE folds.
    ///
    /// Measured, `rc=0`, NVRTC 13.0, `compute_89`, `-std=c++17
    /// -default-device`, against `csrc/src/attn/pack_dense_mask.cuh` under
    /// `-I csrc/{shim,vendor,src}`:
    ///
    /// ```text
    /// attn::device::StructuredMaskParams: sizeof=12  alignof=4
    ///     0  kind
    ///     4  window
    ///     8  sink
    /// ```
    ///
    /// This is the second time in the migration that reading and measuring
    /// agreed, and the reason is `x/xqa.rs`'s: **no nested aggregate.** Three
    /// `unsigned int`s, and every header set in the tree agrees that an
    /// `unsigned int` is four bytes. The traps `params_layout.txt` records —
    /// `uint_fastdiv` at 24 bytes not 4, CuTe's `dA` at 8 not 24, two
    /// `paged_kv_t`s with equal `sizeof` and different interiors — are all
    /// nested aggregates whose size disagrees between `csrc/shim` and CCCL,
    /// and none of them can reach a struct with no members but scalars. Worth
    /// writing down for the same reason `xqa` wrote it down: the property is
    /// this struct's, not the technique's.
    ///
    /// # There was supposed to be an oracle, and it is gone
    ///
    /// `pack_dense_mask.cuh:29-50` still argues that two definitions of this
    /// POD are acceptable because *"`pack_dense_mask.cu` includes BOTH
    /// definitions and `static_assert`s size, alignment and all three field
    /// offsets against each other. A field added, reordered or widened on
    /// either side fails the ahead-of-time build with the two spellings named
    /// in the message."* **Those five `static_assert`s no longer exist.**
    /// `attn/pack_dense_mask.cu` and `attn/pack_dense_mask.hpp` are both
    /// deleted — `driver-cuda/tests/launch_abi.rs:651-654` records the
    /// deletion — so from that day the two mirrors agreed by luck.
    ///
    /// They did agree: `driver-cuda/src/bind/abi.rs`'s three `u32`s in this
    /// order match the measurement field for field. That is the answer to
    /// "check yours against the oracle" — the oracle was right and unchecked,
    /// which is the worse of the two ways to be right. It is now a re-export
    /// of this type, so there is one definition and the question cannot
    /// recur.
    ///
    /// # `kind` is a number, and there are two numberings
    ///
    /// The `__global__` reads it as a literal — `descriptor.kind == 1` causal,
    /// `== 2` sliding window, `== 3` sink (`pack_dense_mask.cuh:236-240`). The
    /// host enum it mirrored, `attn::StructuredMaskKind`, went with the
    /// `.hpp`; this doc and the kernel body are the only surviving record of
    /// the numbering.
    ///
    /// **`ptir/tier0.cuh:613`'s `Tier0StructuredMaskKind` is a DIFFERENT
    /// numbering** — `Causal = 0`, `SlidingWindow = 1`, `SinkWindow = 2`, the
    /// default discriminants of an `enum class : uint8_t`. It describes the
    /// same three kinds one lower. Filling this field from that enum yields
    /// sliding-window where causal was meant, on every lane, with no
    /// diagnostic anywhere: `kind == 0` matches none of the three arms, so
    /// every bit falls to the `causal &&` conjunction alone and the mask is
    /// silently a plain causal one. No enum is minted here to close that,
    /// because nothing on this side reads one — §0's placement rule — and an
    /// enum whose only consumer is its own definition is the data this
    /// migration is removing.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct StructuredMaskParams {
        /// `kind` — 1 causal, 2 sliding window, 3 sink. See the type's doc
        /// for the other numbering and what it costs.
        pub kind: u32,
        /// `window` — the sliding window's extent in keys, for kinds 2 and 3.
        ///
        /// The kernel saturates `key + window` at `0xffffffff` rather than
        /// letting it wrap, because a wrapped sum reads as a CLOSED window
        /// and masks out exactly the tokens the open one admits
        /// (`pack_dense_mask.cuh:227-233`).
        pub window: u32,
        /// `sink` — the attention-sink width in keys, for kind 3: every key
        /// below it is admitted regardless of the window.
        pub sink: u32,
    }

    /// How C++ spells the struct itself, for the `static_assert`s
    /// [`typecheck_tu`](crate::x::abi::typecheck_tu) emits from [`LAYOUTS`].
    ///
    /// **Not [`Abi::CPP`](crate::x::Abi::CPP), and the difference is the
    /// point.** `Layout::cpp`'s doc says "the same string as `Abi::CPP`",
    /// which holds for a by-value aggregate like `x::xqa::KvCacheList` and
    /// cannot hold here: this struct crosses as a POINTER, so `Abi::CPP` is
    /// `const T*` while `sizeof`, `alignof` and `offsetof` all need the bare
    /// `T`. Two constants, one suffix apart, so neither can be used where the
    /// other belongs.
    const STRUCTURED_MASK_PARAMS: &str =
        "::pie_cuda_driver::kernels::attn::device::StructuredMaskParams";

    /// The array of descriptors, one per lane, as `pack_structured_mask`
    /// takes it.
    ///
    /// # Why this impl is written by hand
    ///
    /// [`ptr_abi!`] is a plain `macro_rules!` private to `x/abi.rs`, and
    /// `x/abi.rs` is not this sweep's to edit. It does not need to be:
    /// `x/abi.rs`'s own header states the rule — *"adding a crossing type
    /// means writing one impl, next to the kernel that needed it, and nothing
    /// else in the tree changes"* — and that is exactly what this is. Only
    /// the `*const` half is written, because every use is `const`; a `*mut`
    /// impl would be a spelling with no `__global__` behind it.
    ///
    /// # The namespace is `attn::device` and NOT `attn`
    ///
    /// `kernels/src/lib.rs:1079` spells `Ty::StructuredMasks` in the `attn`
    /// namespace, which was `pack_dense_mask.hpp`'s and is now nothing's.
    /// This string is the one NVRTC resolves, which is what the typecheck TU
    /// compiles. See this module's file header for the full account; the
    /// divergence is deliberate and recorded rather than papered over by
    /// matching a spelling that names no type.
    ///
    /// [`ptr_abi!`]: crate::x::abi
    impl crate::x::Abi for *const StructuredMaskParams {
        const CPP: &'static str =
            "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*";
        const TY: Ty = Ty::StructuredMasks;
        #[cfg(feature = "_cuda")]
        fn arg(&self) -> crate::runtime::ArgValue {
            crate::runtime::ArgValue::Ptr(*self as *mut c_void)
        }
    }

    // The same three numbers `by_value!` would have asserted, written out
    // because `by_value!` is the wrong macro for a pointer crossing: its
    // `Abi::arg` is `ArgValue::Bytes`, which would pass twelve bytes of
    // descriptor where the kernel wants an eight-byte address to an ARRAY of
    // them. The assertions are the half that does apply, so they are kept.
    const _: () = assert!(
        ::core::mem::size_of::<StructuredMaskParams>() == 12,
        "StructuredMaskParams: sizeof disagrees with the measured \
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::align_of::<StructuredMaskParams>() == 4,
        "StructuredMaskParams: alignof disagrees with the measured \
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, kind) == 0,
        "StructuredMaskParams.kind: offset disagrees with the measured \
         attn::device::StructuredMaskParams::kind",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, window) == 4,
        "StructuredMaskParams.window: offset disagrees with the measured \
         attn::device::StructuredMaskParams::window",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, sink) == 8,
        "StructuredMaskParams.sink: offset disagrees with the measured \
         attn::device::StructuredMaskParams::sink",
    );

    /// The measured layout, as C++ `static_assert`s.
    ///
    /// Handed to [`typecheck_tu`](crate::x::abi::typecheck_tu) with
    /// [`pack_dense_mask::PARAMS`](super::pack_dense_mask::PARAMS). The Rust
    /// assertions above catch a drifted mirror; these catch a drifted header,
    /// and only both together catch a field that moved in the header while
    /// someone updated the mirror to the wrong numbers.
    ///
    /// A `LAYOUTS` for a POINTER parameter is not what `x/xqa.rs`'s is for —
    /// a pointer crossing is checked by `is_same_v` on the whole parameter
    /// list whether or not its pointee's layout is asserted. It is here
    /// because this pointee has a mirror, and a mirror with no assertion is
    /// the shape that was already wrong once in this file's history: see the
    /// deleted `.cu` above.
    pub static LAYOUTS: &[crate::x::Layout] = &[crate::x::Layout {
        cpp: STRUCTURED_MASK_PARAMS,
        size: 12,
        align: 4,
        fields: &[("kind", 0), ("window", 4), ("sink", 8)],
        probe: "nvrtc-probes/attn_structured_mask.py",
    }];
}

/// `attn/attn_sink.cuh` — gpt-oss's sink correction and the LSE rebase it
/// reads.
pub mod attn_sink {
    use super::bf16;

    unit! {
        /// The attention-sink pair, both rows: the log2→ln rebase and the
        /// per-head rescale that reads it.
        ///
        /// Both are corrections applied AFTER an attention kernel has
        /// already written its output, which is why they are separate
        /// launches and not a fused epilogue — the attention kernel is
        /// flashinfer's and cannot be edited.
        unit ATTN_SINK = "attn/attn_sink",
            text = include_str!("../../csrc/src/attn/attn_sink.cuh"),
            file = "attn/attn_sink.cuh";

        /// `attn_sink.cuh:74` — flashinfer publishes its LSE in log2 and the
        /// sink correction works in ln.
        ///
        /// A unit conversion, stated so a reader never has to guess which
        /// base an LSE is in — and the drift it prevents is measured:
        /// without it the sigmoid argument was off by 0.693, which matched
        /// HF's top-1 on most prompts and then degenerated greedy decoding
        /// after a few steps.
        ///
        /// **`elem` is `attn::device::f32` and not `device::bf16`'s
        /// sibling.** The prelude has no `device::f32` alias and `Elem` has
        /// no `float` specialisation to hang one on, so the alias is declared
        /// in the `.cuh` beside the kernel that is the only thing asking for
        /// it.
        ///
        /// `n` is `usize` where the sink twin's `N` is `i32`, because the
        /// kernel's parameter is `device::usize` — the twin's `int` was the
        /// launcher's signature, not the kernel's.
        fn lse_log2_to_ln = "attn::device::lse_log2_to_ln" <T> (
            lse: *mut T,
            n: usize,
        ) where *mut T {
            "attn::lse_log2_to_ln" => where [T = f32] "attn::device::f32",
        }

        /// `attn_sink.cuh:93` — `o[t, h, :] *= sigmoid(ln_lse[t, h] -
        /// sink[h])`, in place.
        ///
        /// GPT-OSS learns a per-head sink scalar and extends the softmax
        /// denominator with `exp(sink)`; this is that correction, applied to
        /// the attention OUTPUT.
        ///
        /// **`N` and `num_q_heads` stay operands though the rule recovers
        /// both.** They are the kernel's own `if (t >= N || h >=
        /// num_q_heads) return;` and its row stride `num_q_heads *
        /// head_dim`; an operand list shorter than the `__global__`'s
        /// parameter list is a `void**` the driver reads past. What left is
        /// the stream, which was never one.
        fn attn_sink_rescale = "attn::device::attn_sink_rescale" <T> (
            o: *mut T,
            lse: *const f32,
            sinks: *const T,
            n: i32,
            num_q_heads: i32,
            head_dim: i32,
        ) where *const T, *mut T {
            "attn::attention_sink_rescale_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/attn_res.cuh` — K3's residual-block blend.
pub mod attn_res {
    use super::bf16;

    unit! {
        /// K3's residual-block blend, at bf16.
        ///
        /// K3's residual stream is not a single vector: a prefix and `B`
        /// candidate blocks compete, and the layer picks a convex
        /// combination of them. The score of each candidate is an
        /// RMS-normalised projection — normalise the row, dot it against
        /// `norm_weight * proj_weight`, softmax the `B + 1` scores, blend.
        /// Fusing it is not an optimisation but a memory decision.
        unit ATTN_RES = "attn/attn_res",
            text = include_str!("../../csrc/src/attn/attn_res.cuh"),
            file = "attn/attn_res.cuh";

        /// `attn_res.cuh:99` — one block per token, 256 threads.
        ///
        /// **`T` is gone from the operand list where the deleted twin stated
        /// it.** It did two jobs: a bound check, which is now the grid's
        /// promise, and a block stride, which survives as `block_rows`. The
        /// launcher's `block_rows > 0 ? block_rows : T` default is the row
        /// count, which is the value that ternary produced on every call
        /// site that existed.
        fn attn_res_blend = "attn::device::attn_res_blend" <T> (
            prefix: *const T,
            blocks: *const T,
            norm_weight: *const T,
            proj_weight: *const T,
            out: *mut T,
            b: i32,
            h: i32,
            block_rows: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::attn_res_blend_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/head_dim_pad.cuh` — flashinfer's supported head widths, reached by
/// padding and reached back out of by stripping.
pub mod head_dim_pad {
    use super::bf16;

    unit! {
        /// The pad and the strip, at bf16.
        ///
        /// flashinfer compiles its attention kernels for a fixed set of head
        /// widths — 64, 128, 256, 512 — and a model whose `head_dim` is none
        /// of them cannot use them at all. Phi-3-mini ships 96. These two
        /// kernels buy that model the dense path: pad every head out to the
        /// next supported width on the way in, strip the padding on the way
        /// out. The zero pad is not arbitrary: `q_e . k_e = q[:d] . k[:d]`
        /// when both are zero above `d`.
        ///
        /// A unit that could not exist until [`LaunchRule::PerHead`] did.
        /// Both kernels were extracted, proved NVRTC-clean and left rowless,
        /// and a rowless unit is refused rather than compiled:
        /// `every_unit_compiles_and_every_row_resolves` asserts
        /// `!unit.rows.is_empty()`, because a cubin nothing can fire is one
        /// cached per architecture for nobody.
        ///
        /// [`LaunchRule::PerHead`]: crate::runtime::LaunchRule::PerHead
        unit HEAD_DIM_PAD = "attn/head_dim_pad",
            text = include_str!("../../csrc/src/attn/head_dim_pad.cuh"),
            file = "attn/head_dim_pad.cuh";

        /// `head_dim_pad.cuh:73` — copy `head_dim` values per (token, head)
        /// and zero the trailing columns.
        ///
        /// Threads stride over the PADDED extent so every thread executes
        /// exactly one store — a copy or a zero — rather than one branch
        /// executing and the other stalling. Same instruction count either
        /// side of the boundary.
        ///
        /// `num_tokens` and the stream are not operands: the first is
        /// `grid.y` and the second never was one. Everything the
        /// `__global__` declares stays, including `num_heads` — the geometry
        /// puts the count on an axis the kernel does not read it back from,
        /// so a row without it is a `void**` one entry short.
        fn pad_head_dim = "attn::device::pad_head_dim" <T> (
            packed: *const T,
            padded: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::pad_head_dim_bf16" => where [T = bf16] "device::bf16",
        }

        /// `head_dim_pad.cuh:92` — the inverse, and the same five operands
        /// with the two buffers swapped.
        fn strip_head_dim = "attn::device::strip_head_dim" <T> (
            padded: *const T,
            packed: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::strip_head_dim_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/softcap.cuh` — the logit cap, at both numeric formats.
///
/// **The two-formats-one-width root**, and the only one in `attn`. The header
/// argues the hazard; this module is where it is answered, by two rows on one
/// `fn` declaration whose type binding is the whole of the difference between
/// them.
pub mod softcap {
    use super::{bf16, f16};

    unit! {
        /// One `__global__` template and nothing else — no host function, no
        /// `<<<>>>`, no entry point, which is what `softcap.cuh`'s own header
        /// says about itself.
        ///
        /// # Why it is a template when the original was not
        ///
        /// The original was `_bf16` and only `_bf16`, because an
        /// ahead-of-time build has to choose its instantiations and nobody
        /// spends a translation unit on a second one. Under a JIT the element
        /// type is the row's, so a capped fp16 logit row costs **the line
        /// below** rather than a `cicc` invocation. That is the measurement
        /// `norm/elementwise.cuh` made first with its fp16 `residual_add`,
        /// and it is the reason this migration was worth making.
        unit SOFTCAP = "attn/softcap",
            text = include_str!("../../csrc/src/attn/softcap.cuh"),
            file = "attn/softcap.cuh";

        /// `softcap.cuh:67` — `x = cap * tanh(x / cap)`, elementwise and in
        /// place.
        ///
        /// The cap is a SATURATION and not a clamp: the tail is squashed
        /// smoothly, which is what gemma-2/3 and grok train against, and a
        /// hard `min`/`max` here changes the distribution the sampler then
        /// reads.
        ///
        /// **`x` is the only buffer and there is no destination.** That is
        /// what the deleted row's `in_place = &[(0, 0)]` said and what
        /// `Buffers::assign` was already relying on — *"the logit softcap
        /// accumulates into the logits it was handed"*, where it widens a
        /// seam's pin over an alias set. While the row said nothing the set
        /// had one member and the widening reached nothing: the head wrote
        /// the logits into the arena, the cap ran over `ws.logits`, and the
        /// sampler read an uncapped previous fire. [`super::LOGIT_SOFTCAP`]
        /// carries the `in_place` forward.
        ///
        /// **`n` is `usize` and not `i32`**, because the kernel's parameter
        /// is `device::usize`. There is no bound check against a row count
        /// and no `rows` argument: the grid covers `n` and `if (i >= n)` at
        /// `softcap.cuh:69` is the only guard there has ever been.
        ///
        /// **The reciprocal is not a parameter.** `attn_softcap.cu`'s
        /// launcher passed BOTH `1.f / cap` and `cap`, because a `<<<>>>` is
        /// the only place a host can do arithmetic on the way to a kernel;
        /// `softcap.cuh:70` does the division on the device instead and it is
        /// the same bits. This crate compiles every unit with
        /// `--prec-div=true`, so `1.f / cap` is the correctly-rounded fp32
        /// quotient on the device exactly as it was on the host, and
        /// `--fmad=false` keeps the multiply from being contracted into
        /// something else — two IEEE-754 operations either way, same
        /// rounding, same result. **A `fn` does not restore it**: doing the
        /// division here would put a second implementation of the same
        /// expression on the host, and §0's *"nothing is written twice"* is
        /// exactly that.
        ///
        /// # Two rows, and the second one has no consumer on purpose
        ///
        /// `attn::logit_softcap_f16` is *"the row the ahead-of-time build
        /// never had"* — `families/attn.rs`' header names it that. It has no
        /// table row, no [`contract`](super::LOGIT_SOFTCAP) and no trace
        /// spelling, and it keeps all three of those absences here: a
        /// contract is what a trace MAY say, and no trace says this. What it
        /// has is device text that compiles and a typechecked signature, so
        /// the day a head lands in fp16 the cost is a contract rather than a
        /// port.
        fn logit_softcap = "attn::device::logit_softcap" <T> (
            x: *mut T,
            cap: f32,
            n: usize,
        ) where *mut T {
            "attn::logit_softcap_bf16" => where [T = bf16] "device::bf16",
            "attn::logit_softcap_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `attn/split_packed.cuh` — the fused QKV product cut into three operands.
///
/// **A ROOT WHOSE HOST PROGRAMS ARE BOTH SOMEWHERE ELSE, and that is the
/// arrangement rather than an omission.** Neither of its two kernels gets a
/// `fn` here, a `contract!` or a `bind!`, because each already has a host and
/// neither host is a bind:
///
/// * `attn::split_qkv_bf16` — the non-devwin half — **is already in
///   fn-world**, as [`crate::x::driver_internal::split_qkv_bf16`. It is the
///   fourth arrangement: no unit, no contract, no bind, absent from
///   `FAMILIES` and from `SIGS`, because no trace can state it.
///   `model-compiler`'s `lower.rs::semantic()` DOES name a symbol for
///   `SplitQkv` at `:1545-1548` — but it names this DEVICE row, and
///   `lowered.kernels` (`lower.rs:1095-1096`) is written only from the
///   launch-emitting path, which a driver-internal symbol was never on. So
///   `x::route` never sees it, and `driver_internal`'s six functions fire it
///   BY SYMBOL through [`crate::x::fire::fire`]. What it does: one pass over
///   packed memory, pure copy, no compute — the fused matmul writes one
///   row-major `[N, q_dim + 2 * kv_dim]` tensor and everything downstream
///   addresses Q, K and V as its own `[N, dim]` buffer, because their widths
///   differ under GQA and a single stride cannot describe all three.
/// * `attn::split_qkv_bf16_devwin` — the table symbol, whose device row is
///   spelled `attn::split_qkv_devwin` — keeps its table row and its host program
///   in `driver-cuda/src/fire/split_packed.rs`, reached through
///   `bind::service`. It **cannot** be bound, and that is not a floor gap —
///   see the module header.
///
/// So this module is the unit and only the unit, which is what
/// `x/driver_internal.rs` says the destination is: *"the rows stay where the
/// device text is — in `families::{attn,norm,layout,mlp}` today, and in
/// `x::{attn,norm,layout,mlp}` when those families land"*. The root moves; the
/// two programs do not move with it, because they are already where they go.
///
/// [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
pub mod split_packed {
    use super::bf16;

    unit! {
        /// Two `__global__` templates, no host code.
        ///
        /// # `LaunchRule::SplitPacked` is not why either row exists here
        ///
        /// Both launchers were `<<<dim3(ceil(max(q_dim, kv_dim) / 256), n),
        /// 256>>>`. `SplitPacked` is the rule with that shape and its grid
        /// over the INPUT width (`q_dim + 2 * kv_dim`) is WIDER than the
        /// launcher's over `max(q_dim, kv_dim)` — the outputs are identical
        /// either way, because *"every loop strides by `blockDim.x *
        /// gridDim.x` and bounds itself on its own output width, so extra
        /// blocks contribute nothing but a shorter loop"*. Wider is safe in
        /// this direction and only this one; a grid narrower than an output
        /// leaves the tail of every row unwritten. In fn-world neither row
        /// carries a rule at all: a `Launch` is written by whoever fires it,
        /// and both firers write it from the numbers they were handed.
        unit SPLIT_PACKED = "attn/split_packed",
            text = include_str!("../../csrc/src/attn/split_packed.cuh"),
            file = "attn/split_packed.cuh";

        /// `split_packed.cuh:74` — the host-window form, over
        /// already-offset pointers.
        ///
        /// Six parameters where the launcher had eight: `n_tokens` is the
        /// grid's second axis and never reaches the kernel, and the stream
        /// was never an operand. The two widths come off what is WRITTEN and
        /// not off the packed operand — a `[N, q + 2 * kv]` row cannot say
        /// where the cut falls, and both results can.
        ///
        /// Fired by [`crate::x::driver_internal::split_qkv_bf16`], whose
        /// `Launch` is the launcher's own literal: `grid [ceil(max(q_dim,
        /// kv_dim) / 256), n_tokens, 1]`, `block [256, 1, 1]`, no shared
        /// memory.
        ///
        /// [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
        fn split_qkv = "attn::device::split_qkv" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_bf16" => where [T = bf16] "device::bf16",
        }

        /// `split_packed.cuh:111` — the device-window form, over BASE
        /// pointers.
        ///
        /// **A second kernel and not a flag**, because the pointers the two
        /// are handed mean different things: base pointers here,
        /// already-offset pointers there, and a null check cannot reconcile
        /// that. The grid spans the full lane count and rows outside
        /// `[win[0], win[0] + win[1])` return before touching anything, which
        /// is what lets a captured graph replay across different row splits
        /// without re-recording — the window changes in a buffer, not in a
        /// launch.
        ///
        /// **`win` is the seventh parameter and `n_max` is not a parameter
        /// at all**: it is the grid's second axis. `split_packed.cu:45-46`
        /// — `dim3 grid(xblocks, n_max)` then `device::split_qkv_devwin<bf16>
        /// <<<grid, BLOCK, 0, stream>>>(packed, q_out, k_out, v_out, win_d,
        /// q_dim, kv_dim)`.
        ///
        /// Fired by `driver-cuda/src/fire/split_packed.rs` and NOT by a
        /// bind, for the two reasons `families/attn.rs` stated at length and
        /// this port confirms: `grid.y` is the FIRE's lane count
        /// (`Ctx("rows_total")`) and not the statement's rectangle, so under
        /// a peel a rule's `Dims::rows` would be the tail's length while the
        /// kernel compares an ABSOLUTE `blockIdx.y` against the device
        /// window — the rows past that length never visited, Q, K and V
        /// keeping the previous fire's bytes there; and `Cx::arg_in` /
        /// `Cx::arg_out` return pointers `resolve_arg_windowed` has already
        /// offset by the region's first row, which this kernel would window
        /// a second time.
        ///
        /// # TWO SYMBOLS FOR ONE KERNEL, and the row is the device one
        ///
        /// The string here is `attn::split_qkv_devwin`. The string
        /// `table/attn.rs` states is `attn::split_qkv_bf16_devwin`, and they
        /// are deliberately different: the table symbol is what a DISPATCH
        /// names and the device symbol is what NVRTC resolves.
        /// `driver-cuda/src/fire/split_packed.rs` holds both as constants
        /// side by side — `SPLIT_DEVWIN_SYMBOL` at `:49` and
        /// `SPLIT_DEVWIN_DEVICE` at `:52` — and is the only thing that
        /// bridges them. A `unit!` row states the DEVICE symbol, so getting
        /// this wrong would resolve nothing and fail at the fire rather than
        /// at a type.
        fn split_qkv_devwin = "attn::device::split_qkv_devwin" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            win: *const u32,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_devwin" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/pack_dense_mask.cuh` — the two custom-mask packers, both plain
/// `__global__`s, both unit-only.
///
/// # Unit and nothing else, and this is the *emptiest* of the five arrangements
///
/// `split_packed` crossed as a unit with its host programs left in place.
/// These two cross as a unit with **no host program anywhere**. Their
/// `table::driver_internal` rows are already deleted, and
/// `driver-cuda/tests/launch_abi.rs:651-654` holds the evidence and the
/// verdict: *"`pack_dense_mask` and `pack_structured_mask` stood here and are
/// GONE with `attn/pack_dense_mask.cu`, its `.hpp` and their two
/// `table::driver_internal` rows. Empty consumer set on all five channels;
/// not ported, per §60.1."* `driver-cuda/src/fire/page_mask.rs` — the module
/// whose name suggests otherwise — plans the sideband arena, carves the mask
/// slots and compacts the page CSR, and launches neither kernel.
///
/// So there is no `contract!` and no `bind!`, and a `none:` arm would be
/// wrong: §5.1's rule is that `none:` surfaces as `Route::Unbound` at model
/// LOAD, and no trace can state a symbol the DSL has no statement for.
///
/// **The device text is kept and that is a decision, not an oversight.**
/// `tests/specialise.rs:2743-3298` is its reading consumer and a substantial
/// one: it compiles this unit through NVRTC, loads both plain rows, fires
/// `attn::pack_dense_mask` with `pack_dense_mask.cu:94`'s literal geometry
/// and compares every byte of the packed bitmap against a hand
/// transliteration of `pack_dense_mask.cuh:127-160`. It finds the unit
/// through `unit::UNITS`, which concatenates `families::ALL`, which lists
/// `x::attn::UNITS` beside `families::attn::UNITS` — so the move is
/// transparent to it. It also asserts `instantiation()` is
/// `::pie_cuda_driver::kernels::attn::device::pack_dense_mask` and that the
/// mangled name carries no `I...E` template bracket, both of which the rows
/// below keep.
///
/// # The geometry, preserved
///
/// Both launched `<<<B, 128, 0, stream>>>` — one block per lane at a fixed
/// 128 threads, with a stride loop over that lane's output bytes.
/// `pack_dense_mask.cu:93-94` and `:109-110`, before that file's deletion;
/// `families/attn.rs` recorded it as [`LaunchRule::PerRowNarrow`] *"to the
/// digit"* and added the caveat that survives the move: the 128 *"is not a
/// preference here the way it is for the audio tower — nothing folds warp
/// partials, so the width is not a numerics contract — but it is still the
/// launcher's, and a rule that widened it to 256 would state a launch this
/// tree does not make."* No `fn` writes that `Launch` today because nothing
/// fires these; whoever writes one writes `Launch { grid: [b, 1, 1], block:
/// [128, 1, 1], smem: 0, .. }` and cites this paragraph.
///
/// # `b` is an operand and the grid does not recover it
///
/// Both kernels READ it — `if (b >= B) return;` and `if (request >= B)
/// return;` are the first lines of each — so a declaration that dropped it on
/// the grounds that the grid's `x` extent already carries it would hand the
/// kernel whatever the previous launch left in that slot.
/// `PAGE_COMPACT_SIGS` keeps `num_requests` for the same reason.
///
/// # `DeviceKernel::PLAIN` and not `""`
///
/// The constant is the row's STATEMENT that this `__global__` has no template
/// parameter list; the empty string is what an unfilled field looks like. The
/// distinction is checked by NVRTC in both directions, and
/// `examples/argform_probe.rs` holds the measurement: `plain<device::bf16>`
/// is *"type name is not allowed"*, and a bare template path is *"cannot
/// determine which instance of function template … is intended"*. So a row
/// that states the wrong one of the two fails `tests/units.rs`, with NVRTC's
/// own sentence. **No device text changed for either row**, then or now.
///
/// # THE ONE THING THE FLOOR STILL OWES THIS ROOT, and it is not blocking
///
/// `runtime::args::is_pointer` (`src/runtime/args.rs:396`) does not list
/// `Ty::StructuredMasks`, so `Args::bind` falls through to its catch-all and
/// answers `ArgError::Unsupported`. That is the same refusal `emit::crossing`
/// made in the row world, from the same predicate. **It means the typed stub
/// `raw::pack_structured_mask` would panic at the bind** — `x::fire::fire`
/// binds through `Args::bind` exactly as `fire::hand::fire` does.
///
/// Nothing hits it today: no host program fires this kernel, which is why the
/// crossing is unit-only. It is stated here so that whoever writes one finds
/// the answer rather than the panic. The patch is one token, and the row
/// world already wrote the argument for it — *"the descriptor array IS a
/// device pointer, and saying so is a change to the `Ty` vocabulary rather
/// than to this row"*:
///
/// ```text
/// src/runtime/args.rs:396, in `is_pointer`'s `matches!` list:
///     | Ty::I32Array
/// +   | Ty::StructuredMasks
/// ```
///
/// It is NOT taken in this pass — **and the premise it was declined on has
/// since moved, so here is the re-derivation.** `src/emit.rs` was retired
/// (§6 half A, `0a03f064c`) and its `one()` was a reader: it asked
/// `is_pointer` whether every operand marshalled before emitting a typed
/// `fn`. Swept again after the retirement:
///
/// ```text
/// crates/kernels-cuda-new/src/runtime/args.rs:405  const fn is_pointer   (the definition)
/// crates/kernels-cuda-new/src/runtime/args.rs:507  Args::bind            (the ONLY caller)
/// ```
///
/// **One caller.** Everything else that names the string is prose — the
/// mentions in `abi.rs:996`, `device.rs:2072` and this module — or a
/// DIFFERENT function: `driver-cuda/src/bind/device.rs:226` is a second,
/// independent `const fn is_pointer` with its own single caller at `:293`.
/// The two lists are byte-identical, twenty-four entries in the same order,
/// **and both omit `Ty::StructuredMasks`**, so the patch is one token in two
/// files and taking only one of them makes the crates disagree about a type.
///
/// My own argument therefore flips: *"`is_pointer` is read by more than
/// `Args::bind`"* is no longer true. What is still true is the second half —
/// no host program fires this kernel — so the patch remains reviewable rather
/// than urgent. `device.rs::scalar`'s doc bounds the risk exactly: it is the
/// complement list, written as the closed SCALAR set on purpose, and *"the
/// two lists drifting apart costs a refusal, never a launch."* A `Ty` on
/// neither list is refused before a launch can happen; moving one onto the
/// pointer list can only remove a refusal, never mis-marshal a cell.
///
/// [`LaunchRule::PerRowNarrow`]: kernels::LaunchRule::PerRowNarrow
pub mod pack_dense_mask {
    use super::params::StructuredMaskParams;

    unit! {
        /// Two `__global__`s and no host code at all.
        ///
        /// The unit `instantiation()` could not spell while it could only
        /// write `path<...>`: neither kernel has a type or a compile-time
        /// value to abstract over — every buffer is `u8`/`u32`/`i32` mask
        /// metadata and the block width reaches them as `blockDim.x` — and
        /// `pack_dense_mask.cuh` refused to invent one on `mxfp4_marlin.cuh`'s
        /// precedent: *"a width parameter would be a lie that compiles."*
        unit PACK_DENSE_MASK = "attn/pack_dense_mask",
            text = include_str!("../../csrc/src/attn/pack_dense_mask.cuh"),
            file = "attn/pack_dense_mask.cuh";

        /// `pack_dense_mask.cuh:149` — a dense byte-per-cell mask packed to
        /// FlashInfer's bitmap ABI.
        ///
        /// `kvm_dense` is `[TOTAL_Q, STRIDE]` with one byte per cell (0/1);
        /// `mask_indptr` is the per-lane BYTE offset into `packed`
        /// (`[LANES+1]`, prefix-summed on the host from `ceil(qo_len[l] *
        /// klen[l] / 8)`); `qo_indptr` (`[LANES+1]`) gives each lane's
        /// query-row range; `packed` is pre-zeroed.
        ///
        /// `p_page` is `STRIDE`, the dense mask's logical row stride, and
        /// `b` is the lane count — see the module doc for why `b` is not
        /// recovered from the grid.
        ///
        /// Every buffer is unsourced and stays that way: `mask_indptr` is a
        /// host-built prefix sum the driver owns, `packed` is a pre-zeroed
        /// driver allocation, and `p_page` is the dense mask's row stride. No
        /// `Source` spells any of the three, which is why the row world's two
        /// rows were `table::driver_internal`'s and not `table::attn`'s.
        fn pack_dense_mask = "attn::device::pack_dense_mask" (
            kvm_dense: *const u8,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            packed: *mut u8,
            b: i32,
            p_page: i32,
        ) {
            "attn::pack_dense_mask" => crate::device::DeviceKernel::PLAIN,
        }

        /// `pack_dense_mask.cuh:189` — the same bitmap ABI materialised
        /// straight out of a causal / sliding-window / sink descriptor, with
        /// no dense tensor in between.
        ///
        /// `masks` is one [`StructuredMaskParams`] per lane, read as
        /// `masks[request]` at `pack_dense_mask.cuh:204`. Its mirror,
        /// its measured layout and the `Abi` impl behind this parameter are
        /// [`super::params`]; that module also records the two numberings of
        /// `kind` and which one this kernel reads.
        ///
        /// **This is the parameter that blocked the crossing**, and the row
        /// world was blocked on the same operand from the other side:
        /// `Ty::StructuredMasks` is a `Ty` that `runtime::args`' `is_pointer`
        /// does not admit, so `emit::crossing` refused it and the row had no
        /// generated entry point. A `unit!` declaration has no such refusal —
        /// `Abi` is an open set of impls and `is_pointer` is not consulted —
        /// so the declaration below is the first statement of this kernel's
        /// full signature that anything checks.
        fn pack_structured_mask = "attn::device::pack_structured_mask" (
            positions: *const u32,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            masks: *const StructuredMaskParams,
            packed: *mut u8,
            b: i32,
        ) {
            "attn::pack_structured_mask" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/dsa_indexer.cuh` — glm5's sparse-attention index network, three
/// `__global__` templates, unit-only.
///
/// # The third unit-only crossing, and the reason is `split_packed`'s
///
/// All three host programs are already Rust and already where they belong:
/// `driver-cuda/src/fire/dsa_indexer.rs` holds `knorm_rope`, `q_rope` and
/// `topk_mask`, each firing its DEVICE symbol through `fire::hand::fire`.
/// None of them is a bind and none can become one here — two of the three
/// rows are unsourced in `table::attn` and the third's three integers arrive
/// on `Source::Param`, which is the statement's parameter channel and not a
/// `Cx` query. So the device text moves and nothing else does.
///
/// # THE SYMBOL SPLIT IS LIVE ON ALL THREE, and it is §60.6's
///
/// | table symbol (`table::attn`) | device symbol (declared below) |
/// |---|---|
/// | `attn::dsa_index_knorm_rope_bf16` | `attn::dsa_index_knorm_rope_dev` |
/// | `attn::dsa_index_q_rope_bf16` | `attn::dsa_index_q_rope_dev` |
/// | `attn::dsa_index_topk_mask` | `attn::dsa_index_topk_mask_dev` |
///
/// `fire/dsa_indexer.rs:45-61` holds both halves of each pair as constants
/// side by side and is the only thing that bridges them. A `unit!` row states
/// the DEVICE symbol; getting it wrong resolves nothing and fails at the
/// fire, not at a type. Note also that `_bf16` is DROPPED and not merely
/// suffixed: these are `template <class T>` and the ROW picks `T`, so the
/// format lives in the binding group and not in the string.
///
/// # The geometry, preserved — all three, and two of them state a shape no
/// # rule states
///
/// From `dsa_indexer.cu` before its deletion, and now from
/// `driver-cuda/src/fire/dsa_indexer.rs`, which holds the same numbers:
///
/// * `index_knorm_rope` — `<<<tokens, kBlock = 256, 0, stream>>>`, one block
///   per token. `LaunchRule::PerRow` and **not `Rms`**: `Rms` requests
///   thirty-two bytes of dynamic shared memory that no launcher passes and no
///   kernel here reads — `block_sum`'s warp buffer, which this shape has no
///   reduction to need, because its reduction is a static `__shared__ float
///   red[256]`. Harmless in effect and wrong as a contract: a rule is meant to
///   REPRODUCE its launcher, and one that asks for memory the launcher did not
///   is a rule nobody can check against the `<<<>>>` it came from.
/// * `index_q_rope` — `<<<tokens, round_up(n_heads, 32), 0, stream>>>` with a
///   one-warp floor (`dsa_indexer.cu:34-35`, now
///   `fire::dsa_indexer::q_rope_block`). ONE THREAD PER HEAD. No rule states
///   it and none can: every rule that sizes a block on a row sizes it on the
///   row's WIDTH, and `idx_q`'s row is `n_heads * head_dim` — the two differ
///   by a factor of 64 or 128. `LaunchRule::RouteRows` would open 128× the
///   block. The block is a statement PARAMETER, not a rectangle. **In fn-world
///   that objection evaporates**: a `Launch` is a `fn`'s literal, and the
///   `fn` already exists in the driver.
/// * `index_topk_mask` — `<<<tokens, kBlock, tokens * sizeof(float),
///   stream>>>`, guarded by `if (tokens <= 0) return;`.
///   `LaunchRule::RowScores` was ported FROM this launcher and states grid,
///   block AND the dynamic allocation exactly: `rows * 4` bytes is `tokens *
///   sizeof(float)` written twice. **The shared allocation is why it is
///   neither `Rms` nor `PerRow`**: the kernel declares `extern __shared__
///   float logit[]` and fills `logit[0..nkeys)` where `nkeys = blockIdx.x +
///   1` — one float per KEY, and every key of this fire is a row of it. At
///   `Rms`' thirty-two bytes the last row of a 4,096-token prefill would
///   select its top-k from eight floats it wrote and 4,088 it did not; at
///   `PerRow`'s zero, from none. Neither faults. `dsa_indexer.cuh`'s own
///   words: *"a launch that under-sizes shared memory does not fail, it reads
///   another block's floats"* — a wrong mask, a wrong attention, and nothing
///   downstream checks it.
///
/// # The extents that stay operands
///
/// `tokens` is gone from `knorm_rope` — one block per row IS `tokens` and the
/// kernel never addresses with it. `head_dim` stays, because the kernel
/// strides over it. `N` stays on `topk_mask` although the grid opens over it,
/// because the kernel needs it a second time as the pitch of `mask` (`mrow =
/// mask + i * N`) and as the bound of its causal zero-fill. **An extent a rule
/// recovers is not an operand — an extent a kernel ADDRESSES with is.**
///
/// # `kMaxRopeDim` is a bound this declaration cannot state
///
/// Both RoPE kernels stage `rope_dim` floats in a per-thread `float buf[256]`
/// before rotating them. `rope_dim` is a run-time value, so the array cannot
/// be sized on it, and **a model with `rope_dim > 256` overruns it**. Nothing
/// in a `unit!`, a `Launch` or a `Refusal` can see that: it is a device-side
/// local. `dsa_indexer.cuh` states the bound in its own header and this
/// paragraph is the second place it is written, because a host program that
/// eventually binds these must refuse above 256 and no type will remind it.
pub mod dsa_indexer {
    use super::bf16;

    unit! {
        /// Three `__global__` templates and the RoPE helper they share. No
        /// host code.
        ///
        /// The rotation is INTERLEAVED and not split-half — pairs are `(2i,
        /// 2i+1)`, which is what glm5's index network trains against. A
        /// split-half rotation on the same buffer is a different function and
        /// the two agree only when `rope_dim` is 2, so getting it wrong is
        /// invisible in a unit test with tiny dims.
        unit DSA_INDEXER = "attn/dsa_indexer",
            text = include_str!("../../csrc/src/attn/dsa_indexer.cuh"),
            file = "attn/dsa_indexer.cuh";

        /// `dsa_indexer.cuh:106` — LayerNorm over `head_dim` then interleaved
        /// RoPE, in place on `idx_k`.
        ///
        /// **LayerNorm and not RMSNorm**: the mean is subtracted and there is
        /// a bias. That is why `w` and `b` are two operands where a `norm`
        /// kernel would have one, and why this file cannot borrow `norm`'s
        /// reduction — the prelude's `block_sum` folds in a different order,
        /// and the last bit of this LayerNorm feeds a top-k RANKING.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::knorm_rope` with
        /// `Launch { grid: [tokens, 1, 1], block: [256, 1, 1], smem: 0 }`.
        fn index_knorm_rope = "attn::device::index_knorm_rope" <T> (
            idx_k: *mut T,
            w: *const T,
            b: *const T,
            positions: *const i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::dsa_index_knorm_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:151` — interleaved RoPE on the first `rope_dim`
        /// of each index head of `idx_q`.
        ///
        /// One block per token, one thread per head, `if (h >= n_heads)
        /// return;`. `n_heads` is passed AND sizes the block, which is
        /// `Control::Supplies` exactly — see the module doc for why no rule
        /// can state that shape.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::q_rope`.
        fn index_q_rope = "attn::device::index_q_rope" <T> (
            idx_q: *mut T,
            positions: *const i32,
            n_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
        ) where *mut T {
            "attn::dsa_index_q_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:187` — causal top-k mask over the index scores,
        /// one block per query token.
        ///
        /// ```text
        /// logit[i, j] = sum_h relu(q[i, h] . k[j]) * w[i, h]
        /// ```
        ///
        /// The softmax scale is monotonic and therefore irrelevant to a
        /// RANKING, so it is omitted rather than computed and divided out.
        /// The threshold is forty rounds of bisection on the logit range and
        /// not a sort: a sort of `nkeys` floats per block costs shared memory
        /// proportional to the sequence and a partial sort still has to be
        /// exact at the boundary. Forty halvings of an fp32 interval reach the
        /// representable neighbourhood of the true k-th value, and the tie
        /// behaviour (`>= thr` admits every equal logit) is the original's —
        /// so a row of equal scores admits more than `topk` keys, exactly as
        /// it did.
        ///
        /// `template <class T>` and nothing else: `kBlock` is a file-scope
        /// `constexpr int` the kernel strides by, not a template argument, so
        /// there is no non-type argument to cite and the 256 a launcher opens
        /// has to agree with `dsa_indexer.cuh`'s `kBlock` instead. It does.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::topk_mask`, whose
        /// `Launch` carries the `rows * 4` shared allocation the module doc
        /// argues for.
        fn index_topk_mask = "attn::device::index_topk_mask" <T> (
            idx_q: *const T,
            idx_k: *const T,
            idx_w: *const T,
            mask: *mut u8,
            n: i32,
            n_heads: i32,
            head_dim: i32,
            topk: i32,
        ) where *const T {
            "attn::dsa_index_topk_mask_dev" => where [T = bf16] "device::bf16",
        }
    }
}

/// `flashinfer::MLAParams` — measured, mirrored, and pinned with
/// [`by_value!`](crate::by_value)'s untagged arm.
///
/// # The measurement
///
/// `nvrtc-probes/attn_mla_params.py`, NVRTC 13.0, `compute_89`,
/// `-std=c++17 -default-device -diag-suppress=1105`, `-I csrc/{shim,vendor,src}`,
/// over `flashinfer/attention/mla_params.cuh:26` instantiated as
/// `MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>` — the
/// instantiation `attention_mla.cu:264` uses. Every member is a pointer or a
/// scalar, so the four type arguments change no offset; they only have to
/// compile.
///
/// ```text
/// sizeof = 288   alignof = 8
///   0..176  the twenty-three pointers, eight apart
///     184   block_size          (uint_fastdiv)
///     208   num_heads           (uint_fastdiv)
///     232   q_nope_stride_n ... 268  o_stride_h   (ten uint32_t)
///     272   sm_scale     276  ckv_scale     280  kpe_scale
///     284   return_lse_base_on_e (bool, then three bytes of tail padding)
/// ```
///
/// # WHY TRANSCRIBING IT WOULD HAVE BEEN WRONG BY FORTY BYTES
///
/// This is the strongest case in the tree for §5.1's *measure, do not
/// transcribe*, and it fails on two counts a reader cannot see at the
/// declaration site.
///
/// **1. `uint_fastdiv` is twenty-four bytes, not four.** Measured directly:
/// `sizeof(::flashinfer::uint_fastdiv) = 24, alignof = 8`. `mla_params.cuh`
/// spells `uint_fastdiv block_size; uint_fastdiv num_heads;` and a reader
/// sees two divisors, which read as `uint32_t`s — that is what the NAME says.
/// `params_layout.py` caught this exact trap once before and it recurs here
/// with two instances instead of one. A transcription would put
/// `q_nope_stride_n` at 192; it is at **232**. Every field from `num_heads`
/// on is displaced, and `sizeof` would have been written as 248.
///
/// **2. `PROFILER_PARAMS_DECL` is a conditional field IN THE MIDDLE.**
/// `mla_params.cuh:56` expands to `uint64_t* profiler_buffer;` at
/// `profiler.cuh:87` and to NOTHING at `:139`, chosen by
/// `FLASHINFER_ENABLE_PROFILER` — defined in a different file, invisible
/// where the struct is declared. Both sides were measured:
///
/// ```text
/// JIT default (what a unit! text is compiled with)   sizeof = 288
/// -DFLASHINFER_ENABLE_PROFILER=1                     sizeof = 296   (+8)
/// ```
///
/// Every field from `block_size` on moves by eight. **This is `x/xqa.rs`'s
/// `ENABLE_4BIT_KV_CACHE` again** — *"the struct's shape depends on
/// [a macro], which inserts [a pointer] in the middle, and it is not visible
/// at the declaration site"* — in a second family, which is the second
/// instance and therefore the pattern rather than the anecdote. The mirror
/// below pins the JIT-default shape, and if a build ever defines the profiler
/// macro the `size_of` assertion fires and names the struct.
///
/// # THE FLOOR GAP THIS ROOT FOUND — CLOSED, as `by_value!`'s untagged arm
///
/// Recorded because the gap is the finding and the patch is its answer.
/// `by_value!`'s original grammar required a tag:
///
/// ```text
/// by_value! { $rust as $cpp, tag = $tag:ident, … }
///     const _: () = assert!(::kernels::Ty::$tag.needs_mirror(), …);
/// ```
///
/// and `Ty::needs_mirror()` (`kernels/src/lib.rs:1174`) is a CLOSED list of
/// six: `AttentionWorkspaceView`, `KvCacheLayerView`, `MlaCacheLayerView`,
/// `HopperPrefillPlan`, `YarnOriginalParams`, `StructuredMasks`. None is
/// `MLAParams`, and none may be borrowed for it — `runtime/args.rs:487`
/// already records why: *"the check would pass on a `MLAParams` bound where a
/// `HopperParams` is declared and catch nothing."*
///
/// **So an open set of `Abi` impls was gated behind a closed set of `Ty`
/// tags**, in a crate three portable backends share. `x/xqa.rs`'s
/// `KvCacheList` was the tree's only `by_value!` because
/// `Ty::KvCacheLayerView` happened to already exist and happened to mean
/// roughly the right thing; eleven families produced no second one.
///
/// The obvious patch — a seventh variant — was the one `x/abi.rs:415-417`
/// argues against in its own words: *"a `Ty` variant per aggregate would have
/// been the forty-variant `LaunchRule` mistake one level down."* So the tag
/// became optional instead. `Abi::TY` for an untagged aggregate is
/// `Ty::MlaPlanCache`, which is on **neither** `is_pointer`'s list nor
/// `bind::device::scalar`'s in either crate — a walker that consulted it gets
/// a named `ArgError::Unsupported`, never a silent accept of eight bytes
/// where two hundred and eighty-eight were meant. Every assertion the tagged
/// arm makes is kept; only the permission is dropped. `x/abi.rs`'s new arm
/// states it best: **the field was never carrying a fact, it was carrying a
/// permission.**
///
/// One thing fell out of the sibling patch and is worth carrying here:
/// `Ty::StructuredMasks` is the only entry on `is_pointer`'s list whose
/// `needs_mirror()` is ALSO true, and **the pair proves the two properties
/// are independent.** `needs_mirror` asks *is there a struct pair to keep in
/// sync*; `is_pointer` asks *does a launch marshal eight bytes of address*.
/// The old tag assertion conflated them.
///
/// # What there is NOT to compare against, and what that makes this
///
/// §5.1 hoped a `by_value!` pin might disagree with what the host packs
/// today. It cannot: swept `driver-cuda/src` and `kernels-cuda-new/src` for
/// `MLAParams`/`MlaParams`/`mla_params` and **every hit is prose.** The
/// struct is packed only in `attention_mla.cu`, which is C++ and
/// `xqa-finish`'s.
///
/// **So this is the reference, not a copy of one.** It is written to be the
/// thing the archive is checked against: every offset is a measurement with a
/// named probe, and the two fields a reader gets wrong carry the number they
/// would have got wrong in the assertion message.
pub mod mla_params {
    use super::bf16;
    use crate::by_value;
    use crate::x::{ByValue, Layout};

    /// `flashinfer::uint_fastdiv` — twenty-four bytes, and that is the whole
    /// point of it being a type here.
    ///
    /// Opaque on purpose. `fastdiv.cuh:26-48` makes `impl_` and `d_` PRIVATE,
    /// so the probe cannot reach their offsets — `nvrtcCompileProgram`
    /// answers *"member `flashinfer::uint_fastdiv::d_` is inaccessible"* —
    /// and a mirror that named them would be transcribing exactly what this
    /// module refuses to transcribe. What IS measurable is the size and the
    /// alignment, and those are what a by-value crossing needs.
    ///
    /// A host that eventually fills one must compute the magic-number pair
    /// the same way `fastdiv.cuh:36`'s `__host__` constructor does; that
    /// constructor is `#ifndef __CUDACC_RTC__` precisely because NVRTC
    /// refuses an explicitly `__host__` function, so **the device never
    /// constructs one and the Rust caller must**. That is a second host
    /// program's problem and it is recorded here because nothing else will
    /// say it.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct UintFastdiv {
        /// `impl_` and `d_` together, unreachable individually.
        pub opaque: [u64; 3],
    }

    impl UintFastdiv {
        /// Build the pair the device halves read, by the shim's algorithm.
        ///
        /// # This is a mirror of `csrc/shim/cuda/cmath`, and it says so
        ///
        /// The vendored `flashinfer/fastdiv.cuh` is **not** the classical
        /// magic-number `uint_fastdiv` its name suggests. It is a wrapper
        /// around `cuda::fast_mod_div<uint32_t>` with members `impl_` then
        /// `d_`, and `cuda::fast_mod_div` is **implemented by this repository**
        /// — the toolkit's CCCL 3.0.1 has no such class, measured, *"namespace
        /// cuda has no member fast_mod_div"*. So the algorithm below is read
        /// off a file this tree owns:
        ///
        /// ```text
        /// csrc/shim/cuda/cmath:196-206
        ///     all_ones = ~0ull;  q = all_ones / d;  r = all_ones % d
        ///     __magic_ = q + (r + 1 == d ? 1 : 0) + 1     // = floor(2^64 / d)
        ///     __divisor_ = d
        /// layout: { u32 __divisor_ @ 0, u64 __magic_ @ 8 }  size 16, align 8
        /// ```
        ///
        /// and `uint_fastdiv` wraps it as `{ fast_mod_div impl_ @ 0, u32 d_ @
        /// 16 }` — **24 bytes, align 8**, which is what the assertions at the
        /// foot of this module pin.
        ///
        /// # The rule the shim states, obeyed here
        ///
        /// `cmath:265-273` says it verbatim: *"a Rust mirror of anything
        /// containing a `uint_fastdiv` must be pinned against THIS layout, and
        /// must say so."* It also names the hazard it exists to prevent — *"a
        /// params block filled by the ahead-of-time path and fired by the JIT
        /// one"* — because **CCCL's `fast_mod_div` is `{divisor@0,
        /// multiplier@4, add@8, shift@12}`, the same `sizeof` and a different
        /// interior.** `paged_kv_t::num_heads` measured at +24 under the shim
        /// and +20 under CCCL, and an earlier check of `sizeof` alone reported
        /// agreement.
        ///
        /// **The consequence for this family, stated once and here:**
        /// `MlaParams` is pinned at 288 bytes against the SHIM, which is what
        /// NVRTC compiles. Under nvcc and CCCL each `uint_fastdiv` is 20 bytes
        /// and the struct is 40 short. `attention_mla.cu` therefore packs a
        /// differently-shaped `MLAParams` than this module does, and the two
        /// were never going to agree. That is fine while each side is
        /// internally consistent — it is exactly the shim's stated rule — but
        /// it means **nothing built here may fill a struct the ahead-of-time
        /// path launches with, in either direction.**
        ///
        /// # `d == 0`
        ///
        /// The shim's device path special-cases `d == 1`; it has no answer for
        /// zero and neither does this. A zero divisor yields a zero magic,
        /// which is a wrong answer rather than a fault, so **every host
        /// program in this module refuses a zero extent before it packs** and
        /// this is a `const fn` that cannot refuse for them.
        #[must_use]
        pub const fn new(d: u32) -> Self {
            let d64 = d as u64;
            let magic = if d == 0 {
                0
            } else {
                let q = u64::MAX / d64;
                let r = u64::MAX % d64;
                q + if r + 1 == d64 { 1 } else { 0 } + 1
            };
            // Word 0 is `impl_.__divisor_` in its low half and four bytes of
            // padding in its high half; word 1 is `impl_.__magic_`; word 2 is
            // `d_` and the struct's own tail padding. Written as three `u64`
            // because the members are unreachable individually — the pin
            // above is on the whole 24 bytes. Little-endian is assumed and is
            // not a portability gap: the struct exists to be memcpy'd to a
            // CUDA device, and there is no big-endian CUDA host.
            Self { opaque: [d64, magic, d64] }
        }
    }

    /// `flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>`, at
    /// `<bf16, bf16, bf16, i32>`.
    ///
    /// Field order and spelling are `mla_params.cuh:31-77`'s; every offset is
    /// the module doc's measurement, asserted below.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct MlaParams {
        /// @0 — the non-positional half of Q.
        pub q_nope: *mut bf16,
        /// @8 — the rotary half of Q.
        pub q_pe: *mut bf16,
        /// @16 — the compressed KV cache.
        pub ckv: *mut bf16,
        /// @24 — the K positional cache.
        pub kpe: *mut bf16,
        /// @32 — split-K partial output.
        pub partial_o: *mut bf16,
        /// @40 — split-K partial log-sum-exp, always f32.
        pub partial_lse: *mut f32,
        /// @48 — the merged output.
        pub final_o: *mut bf16,
        /// @56 — the merged log-sum-exp, always f32.
        pub final_lse: *mut f32,
        /// @64
        pub q_indptr: *mut i32,
        /// @72
        pub kv_indptr: *mut i32,
        /// @80
        pub partial_indptr: *mut i32,
        /// @88
        pub merge_packed_offset_start: *mut i32,
        /// @96
        pub merge_packed_offset_end: *mut i32,
        /// @104
        pub merge_partial_packed_offset_start: *mut i32,
        /// @112
        pub merge_partial_packed_offset_end: *mut i32,
        /// @120
        pub merge_partial_stride: *mut i32,
        /// @128
        pub kv_indices: *mut i32,
        /// @136
        pub q_len: *mut i32,
        /// @144
        pub kv_len: *mut i32,
        /// @152
        pub q_start: *mut i32,
        /// @160
        pub kv_start: *mut i32,
        /// @168
        pub kv_end: *mut i32,
        /// @176 — the persistent-kernel work queue.
        pub work_indptr: *mut i32,
        /// @184 — **twenty-four bytes**, not four. See the module doc.
        pub block_size: UintFastdiv,
        /// @208 — twenty-four bytes.
        pub num_heads: UintFastdiv,
        /// @232
        pub q_nope_stride_n: u32,
        /// @236
        pub q_nope_stride_h: u32,
        /// @240
        pub q_pe_stride_n: u32,
        /// @244
        pub q_pe_stride_h: u32,
        /// @248
        pub ckv_stride_page: u32,
        /// @252
        pub ckv_stride_n: u32,
        /// @256
        pub kpe_stride_page: u32,
        /// @260
        pub kpe_stride_n: u32,
        /// @264
        pub o_stride_n: u32,
        /// @268
        pub o_stride_h: u32,
        /// @272
        pub sm_scale: f32,
        /// @276 — per-tensor symmetric dequant scale for an fp8 `ckv`.
        /// Defaults to `1.0` in C++ and has no effect on the bf16/f16 path;
        /// a Rust packer must write the 1.0 itself, because a zeroed struct
        /// would scale every value to zero and that is a silent wrong answer
        /// rather than a fault.
        pub ckv_scale: f32,
        /// @280 — the same for `kpe`, and the same warning.
        pub kpe_scale: f32,
        /// @284 — one byte, then three of tail padding to reach 288.
        pub return_lse_base_on_e: bool,
    }

    // THE PIN. Every offset is `nvrtc-probes/attn_mla_params.py`'s, and the
    // assertions are what make this a mirror rather than a transcription: a
    // field inserted, widened or reordered on either side fails the build
    // here and names the field, instead of launching over a struct whose tail
    // is forty bytes out of place.
    //
    // WHICH VARIANT IS PINNED: the JIT default, `sizeof = 288`, WITHOUT
    // `profiler_buffer`. That is the one NVRTC compiles from a `unit!`'s text
    // — no `-DFLASHINFER_ENABLE_PROFILER` is passed anywhere in this tree —
    // and it is stated here because the declaration site does not show it. If
    // a build ever defines the macro, `sizeof` becomes 296 and the first
    // assertion below fires with that number in its message.
    //
    // The named fields are the ones a reader would get WRONG, plus the ends:
    // the first pointer, the last pointer, both `uint_fastdiv`s, the first
    // and last `uint32_t`, and all four tail scalars. The twenty-one interior
    // pointers are eight apart with nothing between them and are checked by
    // `work_indptr @ 176` closing the run.
    by_value! {
        MlaParams as "::flashinfer::MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>",
        untagged,
        probe = "nvrtc-probes/attn_mla_params.py",
        size = 288, align = 8,
        {
            q_nope               @ 0   as "q_nope",
            work_indptr          @ 176 as "work_indptr",
            block_size           @ 184 as "block_size",
            num_heads            @ 208 as "num_heads",
            q_nope_stride_n      @ 232 as "q_nope_stride_n",
            o_stride_h           @ 268 as "o_stride_h",
            sm_scale             @ 272 as "sm_scale",
            ckv_scale            @ 276 as "ckv_scale",
            kpe_scale            @ 280 as "kpe_scale",
            return_lse_base_on_e @ 284 as "return_lse_base_on_e",
        }
    }

    /// The layouts this module pins.
    ///
    /// `typecheck_tu` has no callers yet — `xqa::LAYOUTS` is unconsumed for
    /// the same reason — but when it gets one, this is the entry that turns
    /// the probe's numbers into a compiled `static_assert` against the real
    /// `mla_params.cuh`, which is the only check that survives a vendor bump.
    pub static LAYOUTS: &[Layout] = &[<MlaParams as ByValue>::LAYOUT];

    // `uint_fastdiv` is asserted separately because `by_value!` above cannot
    // reach it: the macro asserts the fields it is GIVEN, and the two
    // `uint_fastdiv` members are named there by offset, not by size. If this
    // type were 4 bytes as its name suggests, `block_size @ 184` would still
    // hold and `num_heads @ 208` would fail with no explanation of why. These
    // two assertions are that explanation, and they fire first.
    const _: () = assert!(
        ::core::mem::size_of::<UintFastdiv>() == 24,
        "UintFastdiv: sizeof disagrees with the measured ::flashinfer::uint_fastdiv \
         (24, NOT 4 — see nvrtc-probes/attn_mla_params.py)",
    );
    const _: () = assert!(
        ::core::mem::align_of::<UintFastdiv>() == 8,
        "UintFastdiv: alignof disagrees with the measured ::flashinfer::uint_fastdiv",
    );
}

/// `attn/attention_mla_naive.cuh` — the Blackwell MLA fallback pair,
/// unit-only, and the root that stands between the tree and nvcc-zero.
///
/// # Why this root and not `kv_paged`
///
/// §66: `kernels-cuda/native` is the only switch over the whole nvcc and
/// `.cpp` surface in the workspace, it is turned on by exactly one thing
/// (`driver-cuda/bridge`), and that is deletable when `ROW_TABLES` empties.
/// The last two `<<<>>>` in the tree are `mla_naive_paged_kernel` and
/// `mla_mma_paged_kernel` in `attention_mla.cu` — the device text below.
/// **Crossing this root's row is retiring that file**; they were never two
/// tasks.
///
/// The row is not crossed yet and this is the fourth unit-only crossing.
/// `driver-cuda/src/fire/mla_naive.rs` already holds both host programs in
/// Rust and fires both symbols by name through `hand::fire`, so by the rule
/// the last three roots established, the device text moves and nothing else
/// does. What remains for the ROW is stated in that module's own header and
/// is not this crossing's business: `attn::dispatch_attention_mla_bf16` has
/// TWO arms — this pair, and `flashinfer::mla::BatchMLAPagedAttention<MASK,
/// 512, 64>` for everything below sm_100 — and *"a row loses its shim entry
/// whole or not at all, so both arms must be Rust before either can be."*
/// The FA2 arm is the one that passes [`super::mla_params::MlaParams`] by
/// value, which is now unblocked.
///
/// # Why the pair exists — `attention_mla.cu:150-157`, the only place it is
/// # argued, and it must not be lost with the file
///
/// > FlashInfer's FA2 `BatchMLAPagedAttention` (a cooperative kernel)
/// > produces zero output on sm_100; the ecosystem (sglang/vllm) routes
/// > Blackwell MLA to trtllm/cutlass/ragged kernels instead. This is a
/// > correctness-first, arch-agnostic latent-space MLA: one block per (token,
/// > head), flash-style online softmax over the paged ckv/kpe cache. Output
/// > is in the kv_lora latent space (same as the FA2 path), so the rest of
/// > the MLA forward (latent_to_v, o_proj) is unchanged.
///
/// The selector is a device query — `cudaDevAttrComputeCapabilityMajor >= 10`
/// at `attention_mla.cu:56-62` — and it is not these rows' business: it
/// chooses between this pair and FlashInfer's MLA, a different symbol in a
/// different unit.
///
/// **The two kernels are ALTERNATIVES, not a sequence.** The C++ launcher
/// tries the tensor-core kernel first (`attention_mla_naive.cuh:218`) and
/// falls through to the scalar one; `fire::mla_naive` plans one or the other
/// and fires exactly once. Nothing composes them, so there is no `Composed`
/// here and no intermediate buffer.
///
/// # THE GEOMETRY, and why neither kernel can have a `LaunchRule`
///
/// ```text
/// attention_mla_naive.cuh:265   dim3 grid(total_tokens, num_heads / G);      block 256
/// attention_mla_naive.cuh:725   dim3 grid(num_heads / kBM, total_tokens);    block 256
/// ```
///
/// Same block, and the grids are **TRANSPOSES** of each other — tokens on `x`
/// for the scalar kernel, tokens on `y` for the tensor-core one. A rule
/// stating one would be actively wrong for the other while looking right:
/// `grid.y` is capped at 65 535 and `grid.x` is not, **so the transpose
/// decides which of tokens and head blocks may exceed 65 535.** That is a
/// correctness fact wearing the clothes of a layout preference.
///
/// The scalar kernel's `G` is `execution::Control::Supplies`' own example —
/// *"passed to the kernel AND divides the head axis of the grid"* — and it is
/// not merely unstated but **UNSTATEABLE by a formula**: `:241-249` SEARCHES
/// for it, halving from 8 until the grid reaches `kMlaWaveTarget = 296`
/// blocks. A rule computes; this looks. In fn-world both objections dissolve,
/// because a `Launch` is a `fn`'s literal and `fire::mla_naive::plan` is
/// already that `fn`.
///
/// # The shared-memory opt-in, and the measurement that closed it
///
/// The old refusal — *"`attention_mla_naive.cu` keeps its
/// `cudaFuncSetAttribute` opt-in to 200 KB of shared memory behind a
/// `std::call_once`, host state no `LaunchRule` can carry"* — was wrong on
/// both halves and both corrections must survive:
///
/// * **It is not host state a rule has to carry.** `runtime::module`'s
///   `raise_dynamic_smem_cap` performs the opt-in inside `KernelModule::fire`,
///   once per `(CUdevice, CUfunction)` above a 48 KiB high-water mark, driven
///   by `Launch::smem` and nothing else. `x::launch::Launch`'s `smem_opt_in`
///   is the author's side of the same fact.
/// * **The 200 KiB was never needed.** `attention_mla_naive.cuh:251`'s
///   allocation is `(8 * CKV + 16) * 4` and `:228` refuses `CKV > 512`, so
///   the largest request the SCALAR kernel can make is **16 448 bytes** — a
///   third of the 48 KiB default. The arithmetic is preserved in
///   `fire::mla_naive::NAIVE_OPT_IN_BYTES_UNREACHED`. The TENSOR-CORE
///   kernel's **100 032** is above the default and IS raised.
///
/// The real blocker was neither: the file was MIXED — two `__global__`s and
/// four host functions in one header, opening `<mutex>`, `<stdexcept>`,
/// `<string>` and `<cuda_runtime.h>`, so it could not be a unit root at all.
/// The host half now lives in `attention_mla.cu` and in Rust in
/// `fire/mla_naive.rs`, and what is left compiles.
///
/// # PROBED — NVRTC 13.0, `sm_89`, carried headers only
///
/// Under this crate's numerics contract (`--fmad=false --prec-div=true
/// --prec-sqrt=true`) and `-I csrc/{src,shim,vendor}` with **no toolkit
/// include path**:
///
/// ```text
///   rc = 0, 0 errors
///   117 621 bytes of PTX, 2 .entry
///     _ZN15pie_cuda_driver7kernels4attn9mla_naive22mla_naive_paged_kernelE...
///     _ZN15pie_cuda_driver7kernels4attn9mla_naive10mma_detail20mla_mma_paged_kernelE...
/// ```
///
/// It needed three new shim headers and they are **measured, not assumed**:
/// the same text compiled with `/usr/local/cuda/include` answering
/// `cuda_pipeline.h`, `math_constants.h` and `cstring` produced
/// **byte-identical PTX, register allocation included.** See
/// `csrc/shim/cuda_pipeline.h`, which carries the comparison and the one PTX
/// operand it turned on.
///
/// **A fourth finding from the same probe, and it is the shape this sweep
/// keeps meeting:** the file called `std::memcpy` and never included
/// `<cstring>`. Under nvcc `<cuda_runtime.h>` supplied it transitively; under
/// NVRTC it is an error no include path can fix, because the include was
/// never written. **The set nvcc accepted was not the set the file
/// declared.**
///
/// # No options, and the `#ifndef` defaults are why
///
/// `PIE_MLA_MMA_BK`, `_WARPS`, `_STAGES` and `_MINBLK` are all `#ifndef`
/// guarded with their defaults at `:302-322`, so the unit needs no `-D` to
/// compile at the shape everything currently runs. Putting them in
/// `Unit::options` would be the hook `unit.rs`'s own doc warns against: they
/// are tuning constants with one live value, and `Unit::cache_key` spanning
/// them would make a cubin cache key out of a number nobody varies. A second
/// tile is a second unit with a second root, the way `XQA_LATTICE` spells its
/// six.
pub mod mla_naive {
    use super::bf16;

    unit! {
        /// Two `__global__`s and the `mma_detail` helpers the second one
        /// needs. No host code — that left for `attention_mla.cu` and
        /// `fire/mla_naive.rs` before this crossing.
        unit MLA_NAIVE = "attn/attention_mla_naive",
            text = include_str!("../../csrc/src/attn/attention_mla_naive.cuh"),
            file = "attn/attention_mla_naive.cuh";

        /// `attention_mla_naive.cuh:92` — the scalar flash-softmax kernel,
        /// nineteen parameters ending in `G`.
        ///
        /// **`DeviceKernel::PLAIN` and no `_bf16` suffix**, both for the same
        /// reason: there is no template parameter list, so there is nothing
        /// for `elem` to pick and nothing a format suffix could claim a
        /// choice about. Every buffer is `__nv_bfloat16` in the kernel's own
        /// declaration. A suffix here would assert a specialisation that does
        /// not exist.
        ///
        /// The path is two levels deep and that is the header's own nesting:
        /// `pie_cuda_driver::kernels::attn::mla_naive`.
        ///
        /// **`index_mask` is nullable and null is not merely "no mask".**
        /// `attn/attention_mla.hpp:36-38`, which went with the file:
        ///
        /// > DSA top-k mask for the naive path: `[num_query_tokens,
        /// > mask_stride]` uint8 (1=attend). Applied to in-batch keys
        /// > (`j < mask_stride`). Null = dense. **Only valid for
        /// > single-request pure prefill (key `j` == batch token `j`).**
        ///
        /// That last sentence is a correctness precondition no type states
        /// and no refusal can check — the kernel indexes `mask + t *
        /// index_mask_stride` and a multi-request batch makes `j` mean two
        /// different things. It travels here and in
        /// `fire::mla_naive::NaivePtrs::index_mask`, and those are now the
        /// only two places it is written.
        ///
        /// Scores are reduced INSIDE A WARP, not across the block: a
        /// block-wide tree reduction per key costs seven `__syncthreads()`
        /// per KV entry, which at decode dwarfs the arithmetic. Each warp
        /// keeps its own running max/sum/accumulator in registers and the
        /// partial softmax states merge once at the end — flash-decoding's
        /// structure, and the reason `G` exists at all.
        fn mla_naive_paged_kernel = "attn::mla_naive::mla_naive_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            ckv: i32,
            kpe: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
            g: i32,
        ) {
            "attn::mla_naive_paged" => crate::device::DeviceKernel::PLAIN,
        }

        /// `attention_mla_naive.cuh:371` — the tensor-core kernel, sixteen
        /// parameters.
        ///
        /// **THE THREE MISSING PARAMETERS ARE THE MOST INFORMATIVE THING
        /// ABOUT THIS DECLARATION.** There is no `ckv`, no `kpe` and no `G`.
        /// The kernel is compiled AGAINST `kCkv = 512` and `kKpe = 64`
        /// (`:281-282`) because the `mma.sync` fragment shapes are written
        /// for them, and its head group is fixed at `kBM = 16` (`:275`). That
        /// is exactly why `mla_mma_supported` (`:698`) COMPARES those three
        /// rather than forwarding them: the predicate is the only place the
        /// shape is checked, and passing them would imply a generality the
        /// `ld_b_v`/`ld_a` offsets do not have.
        ///
        /// So a host program that fires this kernel must check the three
        /// against 512, 64 and 16 BEFORE it launches — there is no operand to
        /// carry them and therefore no chance of the kernel disagreeing.
        /// `fire::mla_naive` does; that check is the refusal, and it belongs
        /// above the fire for §5.1's hoisting reason.
        ///
        /// Path is three levels deep — `attn::mla_naive::mma_detail` — which
        /// is where `ld_a`, `ld_b_v` and `mma_m16n8k16` live too.
        ///
        /// **100 032 bytes of dynamic shared memory**, which is above the
        /// 48 KiB default and IS raised by `raise_dynamic_smem_cap`. Its
        /// `__launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK)` is in the device
        /// text and needs no host statement.
        fn mla_mma_paged_kernel = "attn::mla_naive::mma_detail::mla_mma_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
        ) {
            "attn::mla_mma_paged" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/kimi_mla.cuh` — kimi_k3's two latent-attention preparation kernels.
///
/// **The only FULL crossing in this pass**, and the reason it was taken
/// before the larger roots: both of its table rows are sourced on every
/// operand, both are stated by `model-compiler/src/dsl.rs`
/// (`kimi_split_kv_a_norm` at `:5452`, `kimi_split_q_b` at `:5483`), and
/// `crates/model/src/kimi_k3/forward/mod.rs:150-156` states them in that
/// order. A unit, two contracts and two binds retire two of the family's
/// thirty-four rows, leaving thirty-two — thirty-one once [`mla_fa2`]'s
/// contract took `attn::dispatch_attention_mla_bf16`.
///
/// # The crossing CLOSES a measured wrong-answer defect
///
/// `device.rs:991-1006` records it in the past tense and the record is worth
/// repeating here, because the port is what ends it rather than what found
/// it: `kimi_split_q_b`'s row describes the LAUNCHER, whose `tokens`,
/// `heads`, `nope` and `rope` are `Rows` and `Param(0..2)` and which formed
/// the device kernel's `total` from them. **The JIT has no launcher**, so it
/// sizes the grid from `LaunchRule::Elementwise` instead — `rows *
/// width_of(b, n_in + 0)`, the FIRST OUTPUT's width. This kernel splits `q_b`
/// into `q_nope` and `q_pe`, so the extent it must cover is wider than out 0
/// *by construction*, and the generated launch under-covers by exactly
/// `nope / (nope + rope)`. Measured at 6 rows of 8 heads, nope 128, rope 64:
/// it wrote four of six rows and left **4,082 of 12,544 bytes** of `q_nope`
/// and **2,041 of 6,400** of `q_pe` holding the harness's poison fill.
///
/// The near-miss is the part to keep: the row's third harness shape — one
/// row, one head — agrees in every byte, because 200 elements round up into a
/// 256-thread block that covers all 255 the kernel wanted. **One shape would
/// have certified it.**
///
/// A `bind!` body computes its own extent, so the arm below reads
/// `rows * in_width(0)` — the kernel's own `total`, which is what
/// `kimi_mla.cuh:13` says the archive launcher computed — and there is no
/// rule left to disagree with it. That is the same shape as
/// [`per_head`]'s head-count defect, which `head_dim_pad`'s crossing closed
/// two passes ago, and it is the second time in this family that **a
/// launch-rule input and a kernel's own addressing were two different
/// numbers wearing one name**.
///
/// # No twin to check
///
/// §60.6's symbol split reached neither kernel: `families::attn`'s
/// `KIMI_MLA_SIGS` and `table::attn`'s two rows name the same two strings,
/// `attn::kimi_split_q_b_bf16` and `attn::kimi_split_kv_a_norm_bf16`, and
/// `driver-cuda/src/fire/` has no `kimi` module to bridge anything. The
/// device symbol IS the table symbol, so the `unit!` rows below are spelled
/// exactly as the deleted rows were.
pub mod kimi_mla {
    use super::bf16;

    unit! {
        /// Two `__global__` templates and nothing else, which is what the
        /// header says about itself in its first line.
        ///
        /// The `<<<>>>`s were in `kimi_mla.cu`, which `#include`d this
        /// header rather than defining what it launched — so the
        /// ahead-of-time build and the JIT compiled ONE text and the
        /// crossing costs no reconciliation. The header records why that
        /// split exists: `norm/altup_aux` once shipped **two definitions of
        /// six kernels**, each correct for whichever half of the tests
        /// exercised it, and no test could see the disagreement because no
        /// test ran both.
        unit KIMI_MLA = "attn/kimi_mla",
            text = include_str!("../../csrc/src/attn/kimi_mla.cuh"),
            file = "attn/kimi_mla.cuh";

        /// `kimi_mla.cuh:67` — split a fused `q_b` projection into its nope
        /// and rope halves.
        ///
        /// `q_b` is `[tokens, heads, nope + rope]`; the results are
        /// `[tokens, heads, nope]` and `[tokens, heads, rope]`. One thread
        /// per SOURCE element, which is why `total` is an argument and not a
        /// grid read: `Elementwise` rounds the grid up and `if (i >= total)`
        /// at `:77` is the only guard there has ever been.
        ///
        /// **`total` is the input's element count and nothing else.** The
        /// module header above measures what happens when something computes
        /// it from an output instead.
        ///
        /// The `long long` casts on the destination indices at `:84` and
        /// `:86` are not decoration — `tokens * heads * nope` overflows
        /// `int` on a long prefill at kimi_k3's head count, and the product
        /// is formed before it is used as an index.
        fn split_q_b = "attn::device::split_q_b" <T> (
            q_b: *const T,
            q_nope: *mut T,
            q_pe: *mut T,
            total: i32,
            heads: i32,
            nope: i32,
            rope: i32,
        ) where *const T, *mut T {
            "attn::kimi_split_q_b_bf16" => where [T = bf16] "device::bf16",
        }

        /// `kimi_mla.cuh:101` — split `kv_a` into a normalised latent and
        /// its rope-carrying companion.
        ///
        /// One kernel rather than a split followed by an RMSNorm, because
        /// the latent half is read twice by the norm and would otherwise
        /// make a round trip through global memory in between. The `k_pe`
        /// copy is **unnormalised on purpose**: rope is applied to it later
        /// and normalising a value that is about to be rotated changes the
        /// angle.
        ///
        /// `src_row_stride` is the SOURCE row width, which is
        /// `kv_lora + rope` unless a caller hands a wider buffer — the fused
        /// MLA prepare does, which is why the stride is an operand and not a
        /// sum.
        ///
        /// # `256` IS THIS ROW'S TO STATE
        ///
        /// `split_kv_a_norm` is `template <class T, int BLOCK_DIM = 256>`,
        /// and until the argument LIST was statable this row could only
        /// spell `<device::bf16>` and let the default supply the rest. That
        /// worked and was fragile in a way nothing here would have caught:
        /// the kernel declares `__shared__ float buf[BLOCK_DIM]` at `:107`
        /// and reduces by halving from `BLOCK_DIM / 2` at `:127`, so the
        /// width **sizes an array and fixes a tree**. Had someone moved the
        /// default to 512, `kimi_mla.cu` would have kept working — it spelled
        /// `<device::bf16, BS>` with `constexpr int BS = 256` — while this
        /// row started instantiating a 512-wide reduction under a 256-wide
        /// launch, where the upper half of `buf` is never written and the
        /// first halving step reads it. **That is a plausible number, not a
        /// crash.**
        ///
        /// Both halves are cited, as a non-type argument requires: the
        /// launcher was `<<<tokens, BS>>>` with `BS = 256`, the template's
        /// default is 256, they agree today, and this row no longer depends
        /// on their continuing to. [`super::rms`] is the other end of the
        /// same 256.
        fn split_kv_a_norm = "attn::device::split_kv_a_norm" <T> (
            kv_a: *const T,
            norm_weight: *const T,
            kv_c: *mut T,
            k_pe: *mut T,
            kv_lora: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::kimi_split_kv_a_norm_bf16" => where [T = bf16] "device::bf16, 256",
        }
    }
}

/// `attn/mla_paged.cuh` — the MLA cache's append and its preparation pass.
///
/// **Unit-only, the fifth**, and by the rule the last four established: *a
/// root whose host programs are already Rust and already outside a `bind!`
/// crosses as a `unit!` and nothing else.* Both host programs are
/// `driver-cuda/src/fire/mla_paged.rs`, whole, with every grid and block
/// figure already cited to a line there. **Both table rows stay**, because a
/// `unit!` moves device text and only a `bind!` retires a row.
///
/// # Why they cannot be bound, which is not a floor gap
///
/// `table::attn`'s two rows are UNSOURCED on every operand and `whole =
/// true`. That is §60.7's case and it is legitimate: `crate::abi` skips a row
/// with any `Source::Unbound` operand whole, so no dispatch arm was ever
/// generated for either and neither was reachable before the crossing or
/// after it. What the rows buy is the shim entry. What a `bind!` would need
/// is a `MlaCacheLayerView` — one dispatch argument whose FIVE fields the
/// kernels take unpacked — and `Cx::kv_layer()` answers the KV cache's
/// layout, not the MLA cache's. That is a real gap and it is **not asked for
/// here**, because the consumer that would exercise it is
/// `dispatch_attention_mla_bf16`, whose own arm is blocked on three other
/// things (see [`mla_fa2`]) and which would state the view itself.
///
/// # The two symbols are BOTH twinned, and this is the family's worst pair
///
/// §60.6's split reached both kernels and `fire/mla_paged.rs:66-70` is the
/// only bridge:
///
/// ```text
/// device  attn::write_mla        table  attn::write_mla_to_pages
/// device  attn::mla_prepare      table  attn::mla_prepare_bf16
/// ```
///
/// The `unit!` below states the DEVICE names, as `families::attn`'s deleted
/// `MLA_PAGED_SIGS` did and for its stated reason: the ahead-of-time symbol
/// takes the `MlaCacheLayerView` by value and unpacks it, so a row claiming
/// the launcher's name would claim a view the `__global__` has never seen.
///
/// **`mla_prepare_bf16` carries a format suffix its kernel cannot justify and
/// `write_mla_to_pages` does not** — two twins of one root spelled by two
/// different conventions. Neither kernel has an element-type parameter; both
/// are `bf16` in their own declarations. The rows are what they are and this
/// pass does not rename them, but a reader deriving a device name from a
/// table name would get one of the two wrong.
pub mod mla_paged {
    use super::bf16;

    unit! {
        /// Two `__global__`s, one of them a template over its BLOCK WIDTH
        /// and neither over an element type.
        ///
        /// `mla_paged.cuh:87-95` argues that `write_mla` stays a non-template:
        /// it has no honest parameter, and §21.6's measurement — a plain
        /// `__global__` is nameable by its bare qualified path, which NVRTC
        /// lowers and `cuModuleGetFunction` resolves — is what lets it be a
        /// row at all.
        unit MLA_PAGED = "attn/mla_paged",
            text = include_str!("../../csrc/src/attn/mla_paged.cuh"),
            file = "attn/mla_paged.cuh";

        /// `mla_paged.cuh:174` — append one token's latent KV to its page.
        ///
        /// `<<<total_tokens, 256, 0, stream>>>`, one block per row of
        /// `ckv_curr`, which is `[Tokens, kv_lora_rank]`.
        ///
        /// `row_valid` is NULLABLE and the kernel says so at `:190`:
        /// `if (row_valid != nullptr && row_valid[t] == 0) return;`. A fire
        /// that published no validity mask hands a null pointer, which is
        /// why the parameter is `*const u8` and not an operand the caller may
        /// omit.
        ///
        /// `R` is `num_requests` — the CSR's request count, which
        /// `mla_resolve_dst` walks — and NOT the token count the grid opens
        /// over. Two extents, one launch, and only one of them is
        /// recoverable from a rule.
        fn write_mla = "attn::device::write_mla" (
            ckv_curr: *const bf16,
            kpe_curr: *const bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            kv_lora_rank: i32,
            qk_rope_head_dim: i32,
        ) {
            "attn::write_mla" => crate::device::DeviceKernel::PLAIN,
        }

        /// `mla_paged.cuh:223` — the fused MLA prepare, at `BLOCK_DIM = 256`.
        ///
        /// # The grid's leading lane is not a head
        ///
        /// ```text
        /// mla_paged.cu:67    dim3 grid(total_tokens, 1 + q_blocks);
        /// mla_paged.cu:68    device::mla_prepare<BS><<<grid, BS, 0, stream>>>(...)
        /// ```
        ///
        /// `mla_paged.cuh:236` reads `const int qb = blockIdx.y - 1;` and
        /// takes the KV path when `qb < 0`, so lane `y = 0` owns the `kv_a`
        /// norm, the `k_pe` rotation and the paged write for its token, and
        /// lanes `1..=q_blocks` are the query heads. A rule that folded the
        /// `1` into the head axis would open the right number of blocks,
        /// shift every head down by one, drop the last, and **never write the
        /// cache** — while `q_nope`/`q_pe` still filled, so the fire would
        /// produce a plausible query against an unwritten page.
        ///
        /// # `256` is a block width AND a comparison
        ///
        /// `mla_paged.cu:64` computes `heads_per_block = half >= BS ? 1 : BS
        /// / half` from the same `BS`, where `half` is `qk_rope_head_dim / 2`
        /// — so the block width and the second grid axis are **one number
        /// stated twice**, and the row states it because
        /// `__shared__ float buf[BLOCK_DIM]` is sized by it and reduced by
        /// halving over it. A row at `<512>` under a 256-wide launch would
        /// leave the upper half of `buf` unwritten, read it on the first
        /// halving step, AND compute half the query blocks: two wrong
        /// answers from one changed literal.
        ///
        /// `device::i32(256)` and not `256`: `DeviceKernel::instantiation`
        /// qualifies an `elem` that does not begin `::` with
        /// `::pie_cuda_driver::kernels::`, so the functional-cast spelling is
        /// what survives that prefix as a non-type argument.
        fn mla_prepare = "attn::device::mla_prepare" (
            kv_a: *const bf16,
            kv_a_norm_w: *const bf16,
            q_b: *const bf16,
            kv_c: *mut bf16,
            k_pe: *mut bf16,
            q_nope: *mut bf16,
            q_pe: *mut bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            positions: *const i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            heads: i32,
            kv_lora: i32,
            nope: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
            theta: f32,
            interleaved: bool,
            heads_per_block: i32,
            yarn_factor: f32,
            yarn_low_dim: f32,
            yarn_high_dim: f32,
            yarn_mscale: f32,
        ) {
            "attn::mla_prepare" => "device::i32(256)",
        }
    }
}

/// The FlashInfer FA2 MLA host program — **compiled, and not yet fireable.**
///
/// This is the host half of `table::attn`'s `dispatch_attention_mla_bf16`
/// FA2 arm: `flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>`, whose
/// device text is `attention_mla.cu`'s `mla_mma_paged_kernel` — **two of the
/// last `<<<>>>` nvcc compiles in the workspace.** §66 measured what that row
/// is worth: `kernels-cuda/native` is the only switch over the entire nvcc
/// and `.cpp` surface, it is turned on by `driver-cuda/bridge` alone, and
/// `bridge` is deletable when `ROW_TABLES` empties.
///
/// # State, in one paragraph
///
/// The root is on disk, the unit is enrolled with its one NVRTC option, all
/// six instantiations and all three shared-memory echoes lower, the launch is
/// declared `cooperative = true`, and [`arm_for`], [`pack`] and [`grid`]
/// produce everything [`raw::attention`] takes. The FA2 arm of
/// `attn::dispatch_attention_mla_bf16` is Rust end to end, the row is gone
/// from `table/attn.rs`, and `attention_mla.cu` is deleted.
///
/// **What has no caller is the ARM CHOICE**, not this arm. The contract
/// `crate::x::attn::ATTENTION_MLA` is `none:`, because a bind body would
/// have to pick between this and `driver-cuda/src/fire/mla_naive.rs` on
/// `cudaDevAttrComputeCapabilityMajor >= 10` and nothing in `Cx` or the
/// runtime states a compute capability — nor the MLA cache layer, the plan
/// handle, the workspace or `sm_scale`. Four `Cx` queries and one device
/// attribute; the exact patch is beside the `none:` arm.
///
/// # The `unit!` exists now, and the reason it did not is worth keeping
///
/// **The vendored `mla.cuh` did not compile under NVRTC and never had.**
/// `csrc/vendor/flashinfer/attention/mla.cuh:806` and `:847` write
///
/// ```text
/// o_smem.template store_128b(o_smem_offset_w, o_partial_ptr);
/// ```
///
/// `store_128b` IS a member template (`permuted_smem.cuh:184`,
/// `template <typename T>`), but `T` is deducible, so the `template`
/// disambiguator is followed by a name rather than a template-id. **NVRTC
/// 13.0 rejects this at both `-std=c++17` and `-std=c++20`** —
/// `error: argument list for template "S<N>::f [with N=N]" is missing` —
/// isolated away from FlashInfer entirely in `nvrtc-probes/mla_template_kw.py`
/// so that the finding is about the construct and not about the closure. Both
/// alternative spellings compile: dropping the keyword, which is what
/// `prefill.cuh:1922` writes for the identical call, and adding an explicit
/// `<DTypeO>`.
///
/// **This is the `<cstring>` shape a third time, and worse.** That one was a
/// dependency satisfied by accident; this is a header CARRIED but never
/// compiled. `source.rs` names no `mla.cuh` entry, but `carried.rs` generates
/// the set by WALKING `csrc/`, so the file has been shipped to every NVRTC
/// compile for months while nothing instantiated anything that reached those
/// two lines. `csrc/shim/cooperative_groups.h`'s own banner says as much:
/// `mla.cuh` is *"the third file to open this door and which nothing in the
/// tree includes"*. A `unit!` naming this root would have been a compilation
/// that fails at the JIT, which is worse than no unit at all — so the device
/// text stayed where it was until the one-token repair landed, and the repair
/// was a vendoring decision and therefore an ask.
///
/// # The three asks, all landed as `f622dcf8d`
///
/// 1. **`csrc/vendor/flashinfer/attention/mla.cuh:806,847`** — the `template`
///    keyword, now under `#ifndef __CUDACC_RTC__` with upstream's spelling
///    kept and ours in the `#else`. `MODIFICATIONS`' recovery transform
///    gained one clause — *discard the `#else` branch* — so FlashInfer
///    v0.6.15 is still recoverable byte for byte.
/// 2. **`csrc/shim/cooperative_groups.h`** — `grid_group` and `this_grid()`.
///    The shim had omitted them ON PURPOSE and its banner was right to: *"a
///    grid-wide barrier is a LAUNCH MODE"*, and a faked `sync()` either
///    deadlocks or lets `mla.cuh:1061`'s stage two read stage one's partials.
///    It was asked for only ALONGSIDE (3), which is what the banner's own
///    last sentence had said to do. `sync()` calls NVIDIA's own pair,
///    `cudaCGGetIntrinsicHandle(scopeGrid)` then `cudaCGSynchronize(handle, 0)`,
///    rather than anything invented.
/// 3. **`runtime/module.rs::fire_ex`** — a third `CUlaunchAttribute` slot for
///    `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` and a `cooperative: bool` parameter
///    in fifth position. `fire_ex` was already the precedent AND the
///    argument: `fimoe-rust` added it rather than growing `Launch`, because
///    *"`Launch` is `eval`'s return type and every family builds one, and a
///    cluster is a property of the kernel at instantiation."*
///
/// **One thing is UNMEASURED and the shim marks it `CG_SYNC_RESOLUTION_UNMEASURED`.**
/// The two `cudaCG*` intrinsics lower to `.extern .func`, which is a promise
/// and not a link, and whether `cuModuleLoadData` resolves them needs a CUDA
/// context no probe here takes. `libcudadevrt.a` carries no `cudaCG*` — and
/// no `cudaGraphSetConditional` either, which is §62's case, where the driver
/// resolved it anyway because NVRTC and nvcc share `cicc`. If MLA's second
/// stage fails at module load, that marker is the line.
///
/// # The two clauses this root asked `unit!` for, both landed as `a9a633d38`
///
/// **`options =` on the unit line.** `unit!` hard-coded `options: &[]` on the
/// reasoning that a compile-option list is a property of the recipe rather
/// than of a unit — true for eleven families and false for the twelfth. See
/// [`OPTIONS`] and the sixteen errors it was measured against; the default
/// stays `&[]`.
///
/// **`cooperative =` on the `fn` line.** On the DECLARATION and not at the
/// call site, because it is a property of the kernel: `mla.cuh:1061`'s two
/// stages are separated by a `this_grid().sync()` and every other kernel in
/// this tree synchronises no further than its own block. At a call site a
/// caller can forget it and get a hang; on the declaration a mismatch is a
/// compile error.
///
/// `x::fire::fire_ex(symbol, launch, cooperative, values, stream)` is what
/// `raw::attention` now reaches, and `fire` delegates to it with `false` — so
/// the resolution order is still stated once, which is what that file is for.
/// **Nothing here ever held a private copy of it**, and the four days that
/// choice cost were the right four days.
///
/// **With (1) applied the kernel LOWERS, and now does so from the tree.**
/// `nvrtc-probes/attn_mla_fa2_root.py` compiles the candidate root and
/// `attn_mla_fa2_ondisk.py` compiles `csrc/src/attn/attention_mla_fa2.cuh`
/// as written, against `csrc/{shim,vendor,src}` with nothing patched out of
/// tree: rc=0, 2.3 MB of PTX, six `.entry`, and all nine name expressions —
/// six kernels and three `&`-prefixed echoes — lowered.
///
/// # Three things that were believed to be blockers and are NOT
///
/// **The grid is not an occupancy query.** `flashinfer_decode.rs:1860-1885`
/// says *"the GRID must come from an occupancy query rather than from a
/// rectangle"*. It does not: `scheduler.cuh:1607-1608` sets
/// `num_blks_x = cluster_size` (1 or 2) and `num_blks_y = num_sm /
/// cluster_size`, so the grid is **exactly `num_sm` blocks and resident by
/// construction**, from `cudaDevAttrMultiProcessorCount`. That is why
/// [`grid`] below is two reads and no query. The planner is already Rust —
/// `crate::plan::mla`, `Schedule { cluster_size, num_clusters, … }` — and
/// [`MlaPlanInfo`] is already mirrored and offset-asserted.
///
/// **The shared-memory size needs no runtime measurement.** `mla.cuh:1128`
/// computes `smem_size = sizeof(KTraits::SharedStorage)` in C++, and
/// `nvrtc-probes/attn_mla_fa2_smem.py` measured all three
/// `DISPATCH_SMEM_CONFIG` arms: **`sizeof(SharedStorage)` is EXACTLY the
/// arm's own threshold literal**, 221 696 / 147 968 / 92 672, align 16, with
/// `causal` changing nothing. So the selection rule is literally *"the
/// biggest tile whose shared storage fits an SM"* and a host needs one number
/// per arm. [`ARMS`] is that table.
///
/// **The plan cache exists.** `driver-cuda/src/fire/flashinfer_fa2.rs:534`'s
/// `MlaPlanCache` is documented as *"plans; nothing in this crate yet
/// launches from it"*.
///
/// # The instantiation name is a §3.2 hazard in a new dress
///
/// There are TWO `KernelTraits` in the closure — `prefill.cuh:159`, whose
/// first parameter is a `MaskMode`, and `mla.cuh:81`, whose first parameter
/// is a `bool CAUSAL_`. `mla.cuh:1124` spells it unqualified and
/// enclosing-namespace lookup picks the right one; **a transcription that
/// writes the qualified name has to know which, and both are nameable.** The
/// root's `Traits` alias writes `::flashinfer::mla::KernelTraits` once, so
/// the six rows never spell it at all — which is the third reason the alias
/// exists, after the two `fa2.cuh` gives. Fifteen parameters against eleven
/// is a substitution failure, not a diagnostic that names the confusion.
pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use crate::plan::MlaPlanInfo;
    use crate::x::launch::Launch;

    /// One `DISPATCH_SMEM_CONFIG` arm of `mla.cuh:1079`.
    ///
    /// `sizeof(KTraits::SharedStorage)` and the threshold that selects the arm
    /// are the SAME NUMBER, which is a measurement and not a coincidence of
    /// this table's construction — see the module doc.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Arm {
        /// `KTraits::NUM_STAGES`.
        pub stages: u32,
        /// `KTraits::CTA_TILE_KV`.
        pub cta_tile_kv: u32,
        /// `KTraits::QK_SHARD`.
        pub qk_shard: bool,
        /// `sizeof(KTraits::SharedStorage)`, in bytes, and the smallest
        /// `smem_limit_per_sm` this arm may be chosen for.
        pub smem: u32,
    }

    /// The three arms, widest first, which is `DISPATCH_SMEM_CONFIG`'s order.
    ///
    /// `NUM_THREADS` is 256 on all three (`nthrs(32, 4, 2)`) and `CTA_TILE_Q`
    /// is 64; `NUM_MMA_KV` is `CTA_TILE_KV / 16`. `causal` selects nothing
    /// here — it is a template argument of the kernel, not of the storage.
    pub const ARMS: [Arm; 3] = [
        Arm { stages: 2, cta_tile_kv: 64, qk_shard: true, smem: 221_696 },
        Arm { stages: 2, cta_tile_kv: 32, qk_shard: true, smem: 147_968 },
        Arm { stages: 1, cta_tile_kv: 16, qk_shard: false, smem: 92_672 },
    ];

    /// The widest arm this device's shared-memory budget admits.
    ///
    /// `smem_limit_per_sm` is `cudaDevAttrMaxSharedMemoryPerMultiprocessor`.
    /// Returns `None` when even the narrowest arm does not fit, which is a
    /// device this kernel was never compiled for and is a refusal the caller
    /// must make **before the first launch**.
    #[must_use]
    pub const fn arm_for(smem_limit_per_sm: u32) -> Option<Arm> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(ARMS[i]);
            }
            i += 1;
        }
        None
    }

    unit! {
        /// `flashinfer::mla::BatchMLAPagedAttentionKernel`, six ways.
        ///
        /// The root holds no `__global__` of ours: it is the `#include` list,
        /// four `using`s, one alias template and one `__device__` echo, in
        /// `csrc/src/attn/fa2.cuh`'s idiom and for its reasons. Read that
        /// file's header for the two `KernelTraits` hazard, the three
        /// residency facts and the `&`-prefix rule.
        unit MLA_FA2 = "attn/attention_mla_fa2",
            text = include_str!("../../csrc/src/attn/attention_mla_fa2.cuh"),
            file = "attn/attention_mla_fa2.cuh",
            options = OPTIONS;

        /// `mla.cuh:879` — the whole of paged MLA, in two stages separated by
        /// a grid-wide barrier.
        ///
        /// **ONE parameter, by value.** `mla.cuh:1130` is
        /// `void* args[] = {(void*)&params};` and the kernel takes
        /// `const __grid_constant__ Params params` — so there is no operand
        /// list to speak of and every pointer, extent and stride the kernel
        /// reads is a field of [`MlaParams`], packed by [`pack`]. That is
        /// what made `ArgValue::Bytes` this family's dependency, and it is
        /// the failure mode §5.1 named: a wrong bypass is a launch with a
        /// garbage struct, not a type error.
        ///
        /// # This launch MUST be cooperative, and the declaration says so
        ///
        /// `mla.cuh:1061` calls `grid.sync()` between the two stages and
        /// `:1132` launches through `cudaLaunchCooperativeKernel`. A
        /// non-cooperative launch is not an error at any layer — it is a
        /// deadlock in stage two — so `cooperative = true` is on the `fn`
        /// line and `raw::attention` reaches `x::fire::fire_ex`. **On the
        /// declaration and not at the call site**, because it is a property
        /// of this kernel and of no other in the tree: every other
        /// `__global__` here synchronises no further than its own block. A
        /// caller cannot forget it, and a caller that disagreed with it would
        /// not compile.
        ///
        /// # The six rows are three shared-memory arms times the mask
        ///
        /// [`ARMS`] is `DISPATCH_SMEM_CONFIG` (`mla.cuh:1100-1120`) and the
        /// mask is a `bool`: `MaskMode::kCustom` is refused with
        /// `cudaErrorNotSupported` at `:1123`, before a traits type is
        /// formed, so nothing here is three-valued.
        ///
        /// Each `elem` starts with `::` so `DeviceKernel::qualify` leaves it
        /// alone — it has to, because the kernel takes TWO template arguments
        /// and `qualify` prefixes a field once rather than per argument.
        fn attention = "::flashinfer::mla::BatchMLAPagedAttentionKernel" (
            params: MlaParams,
        ), cooperative = true {
            "attn::mla_fa2_kv64_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv64_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
        }
    }

    /// The one NVRTC option this root needs, and the sixteen errors that say
    /// so.
    ///
    /// **Measured, not assumed.** Without
    /// `--device-as-default-execution-space` the root is rejected sixteen
    /// times — `csrc/shim/type_traits:253`, seven sites in `cascade.cuh` and
    /// eight in `prefill.cuh` — all *"A function without execution space
    /// annotations ... is considered a host function"*. `mla.cuh:33` includes
    /// `prefill.cuh`, so this is the same closure `families::fa2` compiles
    /// and the same flag it passes (`families/fa2.rs:302`): a third instance
    /// of one entry rather than a new class.
    ///
    /// Per-unit and never global, for the reason `unit::Unit::options` gives
    /// at its own field: turning it on for everyone would silently compile
    /// OUR unannotated host helpers onto the device, and the shared options
    /// are a contract the cache key spans. It is the first `options =` clause
    /// in the tree and the reason the clause exists — the default stays
    /// `&[]`, because an option repeated per unit would be a recipe wearing a
    /// declaration.
    pub const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

    /// The six row symbols, indexed by `[arm][causal]`, parallel to [`ARMS`].
    ///
    /// `HEAD_DIM_CKV = 512`, `HEAD_DIM_KPE = 64`, `CTA_TILE_Q = 64` and the
    /// four types are fixed in the root's `Traits` alias, so a row states the
    /// four numbers that vary and nothing else — the `<MASK, 512, 64>` the
    /// dispatch arm names, spread over the traits the kernel actually takes.
    ///
    /// **`__nv_bfloat16` mangles as `pie_cuda_driver::kernels::device::bf16`**
    /// in the lowered PTX, because the shim's `cuda_bf16.h` typedefs it into
    /// the prelude namespace. That is a property of this tree's shim and not
    /// of FlashInfer, and it is the reason a symbol lookup against a
    /// vendor-built cubin would not find these.
    pub const SYMBOLS: [[&str; 2]; 3] = [
        ["attn::mla_fa2_kv64_full", "attn::mla_fa2_kv64_causal"],
        ["attn::mla_fa2_kv32_full", "attn::mla_fa2_kv32_causal"],
        ["attn::mla_fa2_kv16_full", "attn::mla_fa2_kv16_causal"],
    ];

    /// The compiler's own `sizeof(KTraits::SharedStorage)` per arm, as name
    /// expressions, parallel to [`ARMS`].
    ///
    /// `fa2::PrefillGeometry::ECHO_TEMPLATE` is the precedent and the
    /// `&` prefix is not decoration: `nvrtcAddNameExpression` refuses
    /// `smem_bytes_mla<KT>` and accepts `&smem_bytes_mla<KT>`, because a
    /// function's name is its address and a variable's is not. All three
    /// lowered in `nvrtc-probes/attn_mla_fa2_ondisk.py`.
    ///
    /// **Nothing reads these yet**, and what they would catch is narrow and
    /// serious: `ARMS[i].smem` is a LITERAL copied out of upstream's own
    /// threshold comparison, and upstream changing `SharedStorage` without
    /// changing the threshold leaves the launch correctly sized by its
    /// `sizeof` and the ARM chosen wrong, silently, on a device whose shared
    /// memory falls between the old literal and the new size. Whoever wires
    /// `cuModuleGetGlobal` compares and refuses rather than trusting either
    /// side.
    pub const SMEM_ECHO: [&str; 3] = [
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>>",
    ];

    /// The MLA cache's shape, as `dispatch_mla_512_64` reads it off
    /// `MlaPlanCache`.
    ///
    /// Nine numbers and a scale, and the reason they are a struct rather than
    /// ten parameters is that eight of the ten `*_stride_*` fields below are
    /// PRODUCTS of two of them — writing the products at the call site is how
    /// a packer gets one wrong.
    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        /// Tokens per page.
        pub page_size: u32,
        /// Query heads.
        pub num_heads: u32,
        /// The compressed KV width — `HEAD_DIM_CKV`, 512.
        pub kv_lora_rank: u32,
        /// The positional width — `HEAD_DIM_KPE`, 64.
        pub qk_rope_head_dim: u32,
        /// `1 / sqrt(head_dim)`, or whatever the deployment states.
        pub sm_scale: f32,
    }

    /// Where the two workspace arenas start, and the addresses the fire
    /// carries.
    #[derive(Clone, Copy, Debug)]
    pub struct Buffers {
        /// `AttentionWorkspaceView::int_buffer`.
        pub int_buffer: *mut u8,
        /// `AttentionWorkspaceView::float_buffer`.
        pub float_buffer: *mut u8,
        /// `[tokens, num_heads, kv_lora_rank]`.
        pub q_nope: *mut bf16,
        /// `[tokens, num_heads, qk_rope_head_dim]`.
        pub q_pe: *mut bf16,
        /// The layer's compressed KV pages.
        pub ckv_pages: *mut bf16,
        /// The layer's positional pages.
        pub kpe_pages: *mut bf16,
        /// The result, in the latent space.
        pub out: *mut bf16,
        /// The uploaded page-index array. NOT workspace-relative — it is a
        /// device pointer the fire already holds.
        pub kv_page_indices: *mut i32,
        /// The LSE, or null when the statement does not ask for one.
        pub lse: *mut f32,
    }

    /// `int_buf + offset`, in ELEMENTS of `T`.
    ///
    /// `attention_mla.cu`'s `offset_ptr<T>` to the byte: every `*_offset` in
    /// [`MlaPlanInfo`] is an index into the arena's element array and not a
    /// byte count, which is why the multiply is here and not at the call
    /// sites. Getting this wrong scales every plan pointer by four and is a
    /// fault rather than a wrong answer, which is the only reason it is safe
    /// to state once.
    unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
        unsafe { base.cast::<T>().offset(offset as isize) }
    }

    /// Fill an [`MlaParams`] the way `attention_mla.cu:264-320` does.
    ///
    /// # This is the FIRST Rust packer of this struct and is the reference
    ///
    /// There is no second side to compare against: `MLAParams` is packed only
    /// in `attention_mla.cu`, which is the ahead-of-time path, and
    /// [`UintFastdiv::new`] records why the two shapes could never have
    /// agreed anyway — the shim's `fast_mod_div` is 16 bytes where CCCL's
    /// interior differs at the same `sizeof`, so `MLAParams` is 288 here and
    /// 248 there. **This is not a transcription of that file; it is the thing
    /// that file would have to be checked against.**
    ///
    /// # `ckv_scale` and `kpe_scale` are written HERE and are not in the C++
    ///
    /// `attention_mla.cu` never assigns them: it relies on
    /// `mla_params.cuh`'s default member initialiser of `1.f`. A Rust `struct`
    /// literal has no such thing, so a packer that omitted them would leave
    /// whatever the caller's memory held, and a ZEROED struct would scale
    /// every value to zero — **a silent wrong answer, not a fault.** The
    /// struct's own field doc says this and this function is where it is
    /// obeyed.
    ///
    /// # Safety
    ///
    /// Every pointer in `buffers` must be a device address valid for the
    /// fire, and `plan` must be the plan those arenas were uploaded from.
    /// Nothing is dereferenced here; the requirement is the kernel's.
    #[must_use]
    pub unsafe fn pack(
        plan: &MlaPlanInfo,
        shape: Shape,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        let int_buf = buffers.int_buffer;
        let float_buf = buffers.float_buffer;
        MlaParams {
            q_nope: buffers.q_nope,
            q_pe: buffers.q_pe,
            ckv: buffers.ckv_pages,
            kpe: buffers.kpe_pages,
            partial_o: unsafe { offset_ptr(float_buf, plan.partial_o_offset) },
            partial_lse: unsafe { offset_ptr(float_buf, plan.partial_lse_offset) },
            final_o: buffers.out,
            // `want_lse` is the statement's, not the plan's: FlashInfer reads
            // a null `final_lse` as "do not write one", so the decision is a
            // pointer and not a flag. A caller that passed a live pointer for
            // a statement with one result would write past the end of the
            // fire's allocation.
            final_lse: if want_lse { buffers.lse } else { ::core::ptr::null_mut() },
            q_indptr: unsafe { offset_ptr(int_buf, plan.q_indptr_offset) },
            kv_indptr: unsafe { offset_ptr(int_buf, plan.kv_indptr_offset) },
            partial_indptr: unsafe { offset_ptr(int_buf, plan.partial_indptr_offset) },
            merge_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_start_offset)
            },
            merge_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_end_offset)
            },
            merge_partial_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_start_offset)
            },
            merge_partial_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_end_offset)
            },
            merge_partial_stride: unsafe {
                offset_ptr(int_buf, plan.merge_partial_stride_offset)
            },
            // NOT workspace-relative. The page indices are uploaded by the
            // caller and handed in as a device address, which is what
            // `attention_mla.cu:293` does with its `kv_page_indices_d`.
            kv_indices: buffers.kv_page_indices,
            q_len: unsafe { offset_ptr(int_buf, plan.q_len_offset) },
            kv_len: unsafe { offset_ptr(int_buf, plan.kv_len_offset) },
            q_start: unsafe { offset_ptr(int_buf, plan.q_start_offset) },
            kv_start: unsafe { offset_ptr(int_buf, plan.kv_start_offset) },
            kv_end: unsafe { offset_ptr(int_buf, plan.kv_end_offset) },
            work_indptr: unsafe { offset_ptr(int_buf, plan.work_indptr_offset) },
            block_size: UintFastdiv::new(shape.page_size),
            num_heads: UintFastdiv::new(shape.num_heads),
            q_nope_stride_n: shape.num_heads * shape.kv_lora_rank,
            q_nope_stride_h: shape.kv_lora_rank,
            q_pe_stride_n: shape.num_heads * shape.qk_rope_head_dim,
            q_pe_stride_h: shape.qk_rope_head_dim,
            ckv_stride_page: shape.page_size * shape.kv_lora_rank,
            ckv_stride_n: shape.kv_lora_rank,
            kpe_stride_page: shape.page_size * shape.qk_rope_head_dim,
            kpe_stride_n: shape.qk_rope_head_dim,
            // The output is in the LATENT space, so its strides are
            // `kv_lora_rank`'s and not the model's head dim. That is the same
            // reading `attention_mla_naive.cuh` takes and the reason the rest
            // of the MLA forward is unchanged between the two backends.
            o_stride_n: shape.num_heads * shape.kv_lora_rank,
            o_stride_h: shape.kv_lora_rank,
            sm_scale: shape.sm_scale,
            ckv_scale: 1.0,
            kpe_scale: 1.0,
            // `attention_mla.cu:320` writes `true` unconditionally: the
            // kernel emits its LSE in natural log rather than log2, which is
            // the opposite convention to the rest of FlashInfer and is why
            // `attn::lse_log2_to_ln` exists for the OTHER backends and must
            // NOT be applied to this one.
            return_lse_base_on_e: true,
        }
    }

    /// The grid, from the plan and nothing else.
    ///
    /// `[num_blks_x, num_blks_y, 1]` — `cluster_size` by `num_sm /
    /// cluster_size`, so the product is `num_sm` and **every block is
    /// resident by construction.** That is what makes a cooperative launch
    /// legal here without an occupancy query, and it is the correction to
    /// `flashinfer_decode.rs:1873`.
    ///
    /// The block is `[256, 1, 1]` on all three arms — `nthrs(32, 4, 2)` — and
    /// the lowered PTX agrees: `.maxntid 256, 1, 1`.
    #[must_use]
    pub const fn grid(plan: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch {
            grid: [plan.num_blks_x as u32, plan.num_blks_y as u32, 1],
            block: [256, 1, 1],
            smem: arm.smem,
            // 221 696 and 147 968 are both far above the 48 KiB static cap,
            // so two of the three arms REQUIRE the opt-in.
            // `runtime::module::raise_dynamic_smem_cap` performs it inside
            // the fire, once per `(CUdevice, CUfunction)`, driven by
            // `Launch::smem` — which is why this is a flag and not a call.
            smem_opt_in: true,
        }
    }
}

/// The units `attn` compiles in fn-world.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above gives. `families::ALL` reads this **beside**
/// `families::attn::UNITS`, which still holds the NINE roots these passes did
/// not take. A root appears in exactly one of the two lists: a second `unit!`
/// naming the same text would be a second compilation of it under a second
/// unit name, and `unit_of` would answer with whichever won.
/// `attn/qkv_fused.cuh` — the fused QKV epilogues.
///
/// Three `__global__` templates and no host code, which is the whole reason
/// this root crosses cleanly: every decision the deleted `qkv_fused.cu` made
/// is already Rust in `driver-cuda/src/fire/qkv_fused.rs`.
///
/// # ELEVEN ROWS OVER NINE INSTANTIATIONS, and the `#` names are carried
/// # VERBATIM
///
/// `families::attn::QKV_FUSED` stated eleven rows, and two of them —
/// `attn::qkv_decode_qk_norm_rope_write_kv` and
/// `..._warp_d128` — are BASE rows that name the same instantiation as their
/// own `#norope` arm. That was a `Specialisation` mechanism: a base row plus
/// arms, with `flags_are_covered` proving the base unreachable.
///
/// **fn-world has no `Specialisation` and the rows are carried anyway.** The
/// reason is not symmetry, it is that `driver-cuda/src/fire/qkv_fused.rs`
/// fires by NAME: `warp_symbol(head_dim, rope_table)` and
/// `block_symbol(rope_table)` return exactly these strings, `#rope` and
/// `#norope` suffixes included. Renaming them here would be a
/// `NoLoweredName` at the first decode fire on a machine with a GPU, and the
/// suffix is legal in fn-world for a checkable reason: `x/abi.rs:824`'s
/// `mangle` already lists `'#'` among the characters it replaces when it
/// writes the typecheck TU. **The row world's arm spelling is a symbol here,
/// not a mechanism**, and that is the cheapest possible crossing of a
/// `Specialisation` — cheaper than `x/norm.rs:1033`'s *"`Specialisation`s
/// become `if`s"*, because the `if` already exists and is in the driver.
///
/// A duplicate name expression costs nothing: NVRTC accepts it, and
/// `QKV_FUSED_ROWS` said so first.
///
/// # `128` means two different things and the two are spelled the same
///
/// `qkv_decode_qk_norm_rope_write_kv<128, …>`'s first argument is a BLOCK
/// width — it sizes `__shared__ float buf[BLOCK]`.
/// `qkv_decode_qk_norm_rope_write_kv_warp<128, …>`'s is a HEAD width: it
/// fixes `ELEMS_PER_THREAD = HEAD_DIM / 32` and every `#pragma unroll` under
/// it, while the block width is the launcher's `WARP_BLOCK = 256`, read at
/// run time from `blockDim.x`. The warp form is compiled at 64, 128 and 256
/// because `head_dim` decides which, and the block form at 128 only because
/// its argument is not a head width at all.
///
/// # `win` and `row_valid` are both nullable and mean different things
///
/// `row_valid` is a validity mask a fire either published or did not, tested
/// `row_valid != nullptr && row_valid[row] == 0`: absent means *every row is
/// valid*. `win` is the Peel device window's PREFIX form, and the non-devwin
/// entry point hands it `nullptr` outright: absent means *the split is not
/// device-decided*. Both are `| null` and both are real.
///
/// # `rope_table` is a HOST NULL TEST, not a launch parameter
///
/// It is a `const float*` PARAMETER of both instantiations — the host passes
/// it to `USE_ROPE_TABLE = false` too, which reads it never. So every arm
/// forwards all twenty-two arguments and the base binds exactly what a
/// fall-through kernel declares; there is no cell to leave unread. That is
/// why the pair is two instantiations rather than a runtime branch: the
/// unrolled table read is compiled out, not skipped.
pub mod qkv_fused {
    use super::bf16;
    // Gated with the host `fn` below, because `raw` is and because
    // `Fired` and `Refusal` are `x::attn`'s own gated imports: a plain
    // `use super::Fired` here does not resolve in the toolkit-free build.
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        /// Three `__global__` templates, no host code.
        ///
        /// The root moved from `crates/kernels-cuda/csrc/src/attn/` long ago;
        /// what moves here is the DECLARATION, out of `families::attn`'s
        /// `DeviceKernel` rows and into the grammar that states a kernel's
        /// parameters beside its instantiations.
        unit QKV_FUSED = "attn/qkv_fused",
            text = include_str!("../../csrc/src/attn/qkv_fused.cuh"),
            file = "attn/qkv_fused.cuh";

        /// `qkv_fused.cuh:412` — the PACKED prefill epilogue.
        ///
        /// Six statements in one launch — q norm, q rope, k norm, k rope, v
        /// norm and the paged KV write — and the only value that survives to
        /// a result is q. Everything else lands in the cache, which is what
        /// the contract's `sink: Some("kv.pages")` says.
        ///
        /// `<<<dim3(num_rows, num_q_heads + num_kv_heads), 256>>>`: one block
        /// per (row, head), q heads first and kv heads after, which is what
        /// `head_idx < num_q_heads` reads. `BLOCK` is 256 and IS the block
        /// width here, unlike the warp form below.
        ///
        /// Eighteen parameters and no `rope_table`, no `w_page`, no `w_off`
        /// and no `win`: the prefill form derives its destination from the
        /// CSR and takes its angles from `theta`.
        fn packed = "attn::device::qkv_packed_qk_norm_rope_vnorm_write_kv" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" => "device::i32(256)",
        }

        /// `qkv_fused.cuh:115` — the BLOCK decode form.
        ///
        /// One block per request row, `BLOCK = 128` threads, a shared
        /// reduction per head. The fall-through when the warp form has no
        /// instantiation for the fire's head width.
        ///
        /// Three rows for two instantiations: the base and `#norope` are the
        /// same text. See the module doc.
        fn block = "attn::device::qkv_decode_qk_norm_rope_write_kv" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            rope_table: *const f32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: *const u8,
            win: *const u32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv#norope" => "device::i32(128), false",
        }

        /// `qkv_fused.cuh:252` — the WARP decode form.
        ///
        /// One warp per head instead of one block per row, which is why it
        /// takes `num_requests` where the block form takes `head_dim`: the
        /// head width is the TEMPLATE argument and the request count is what
        /// the grid stride bounds itself on.
        ///
        /// Seven rows for six instantiations, at three head widths. The two
        /// `d64` and two `d256` pairs are the arms
        /// `families::attn::QKV_FUSED_ROWS` called *"the two expansions this
        /// unit was missing"* — before them, `head_dim == 64` and
        /// `head_dim == 256` reached a launch no row named.
        fn warp = "attn::device::qkv_decode_qk_norm_rope_write_kv_warp" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            rope_table: *const f32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: *const u8,
            win: *const u32,
            num_requests: i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope" => "device::i32(64), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope" => "device::i32(64), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope" => "device::i32(256), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope" => "device::i32(256), false",
        }
    }

    /// `BLOCK` for the packed form, and it IS the block width.
    ///
    /// The template argument and `blockDim.x` are the same number here, which
    /// is not true of [`warp`] one `fn` up — so this constant may be read as
    /// either and the warp form's may not.
    pub const PACKED_BLOCK: u32 = 256;

    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` — the fused
    /// prefill epilogue, over a packed `[N, q + 2·kv]` row.
    ///
    /// # The grid is two-dimensional and the second axis is the HEAD, not a
    /// # tile
    ///
    /// `dim3(num_rows, num_q_heads + num_kv_heads)`. The kernel reads
    /// `head_idx < num_q_heads` to decide whether it is a q head or a kv one,
    /// so the two banks are ONE axis and their order is load-bearing. A grid
    /// that put kv heads first would norm the wrong weights and write the
    /// wrong pages, with every pointer valid.
    ///
    /// # Every refusal is hoisted, and there is only one launch to hoist past
    ///
    /// One kernel, so the `layout` rule is satisfied trivially — but the
    /// refusals are still stated before it rather than folded into the grid,
    /// because a zero extent here is a silently empty launch and not a fault.
    ///
    /// # Safety
    ///
    /// Every pointer must be a device address valid for the fire, `packed`
    /// must hold `num_rows` rows of `(num_q_heads + 2·num_kv_heads)·head_dim`
    /// elements, and the page arrays must describe the layer the cache
    /// pointers came from.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        row_valid: *const u8,
        num_rows: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_rows <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if num_q_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "kv heads" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page size" });
        }
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        unsafe {
            raw::packed(
                "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
                Launch {
                    grid: [num_rows.unsigned_abs(), heads, 1],
                    block: [PACKED_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                packed,
                q_out,
                k_pages,
                v_pages,
                q_weight,
                k_weight,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                row_valid,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            );
        }
        Fired::Launched
    }
}

/// `attn/dsv4_compress.cuh` — deepseek_v4's SECOND KV cache, and the eleven
/// `__global__`s that build, attend and merge it.
///
/// The model attends a fine-grained cache and a compressed one holding one
/// entry per `ratio` tokens, and merges the two outputs by their
/// log-sum-exps. That merge is exact algebra and not an approximation — the
/// same one FlashInfer's KV-split uses — which is why [`combine_attn_outputs`]
/// is a kernel here rather than a fixup.
///
/// # Ten rows, and the six that no longer need a `Specialisation` argument
///
/// Transcribed one for one from `families::attn::DSV4_COMPRESS`, which stated
/// all ten before this pass and stated them completely. Six carry `bf16`, two
/// carry `device::i32` — the DEFAULT the `.cuh` gives `T` in
/// `template <class T = device::i32>`, spelled out because
/// `DeviceKernel::instantiation()` always emits an argument list and naming
/// the default is what keeps the JIT's object and the archive's the same one
/// rather than two that happen to agree — and one is `PLAIN`.
///
/// **`compressed_attn_paged` is the `PLAIN` one and it is the only unadorned
/// `__global__` in this crate's fn-world.** It takes no template parameter at
/// all, so there is no `<T>` to write; `DeviceKernel::PLAIN` is the `elem`
/// that says so, and `qualify` leaves it alone.
///
/// # THE SYMBOL SPLIT, and why fn-world does not need it
///
/// Four of these rows are spelled `…_dev`: `dsv4_boundary_meta_decode_dev`,
/// `dsv4_boundary_meta_paged_dev`, `compressed_attn_paged_dev` and
/// `combine_attn_outputs_dev`. §60.6 split them because a symbol that was
/// both a table row and a unit row was unit-hosted and therefore unwalkable
/// (§52.11), and the launcher could not be taken over while the two names
/// were one.
///
/// **The suffixes are CARRIED VERBATIM and must be**, for the same reason
/// `qkv_fused`'s `#rope` is: `driver-cuda/src/fire/dsv4_compress.rs` fires
/// them by name — `COMBINE_DEVICE`, `META_DECODE_DEVICE`,
/// `META_PAGED_DEVICE` and `COMPRESSED_PAGED_DEVICE` at `:59`-`:68` — through
/// `hand::fire`, which resolves `unit_of(symbol)` and binds against that
/// row's own `sig`. Renaming one is a panic at the first deepseek_v4 fire on
/// a machine with a GPU, and nothing on a machine without one would say so.
///
/// That the split is no longer NEEDED is a separate statement from that it is
/// no longer THERE. Collapsing it is four edits in a file this pass does not
/// own, for no gain, and the four table rows it would collide with are still
/// in `table/attn.rs`.
///
/// # What crosses and what stays
///
/// Two of the six table rows this root serves cross here:
/// `attn::dsv4_compress_gather_paged_bf16` and
/// `attn::dsv4_store_comp_entries_bf16`, both UNBOUND — see their contracts
/// below for what a bind would need and why the trace cannot supply it.
///
/// The other four stay, and they stay for `qkv_decode_fused`'s reason
/// exactly: their host programs are written, in Rust, in
/// `driver-cuda/src/fire/dsv4_compress.rs`, and served through
/// `bind::service`. A row served by a driver host program is not a row
/// waiting to cross; it is a row whose fn-world half lives one crate up.
/// **This unit is what those four fire**, and moving it here changed nothing
/// for them: `families/mod.rs` concatenates `crate::x::attn::UNITS`, so
/// `unit_of` answers the same four symbols it always did.
///
/// # The two stale sentences in the root, still stale
///
/// `dsv4_compress.cuh:50-52` says *"No ported rule computes a shared-memory
/// size from an operand width"*; `PagedScores` and `PagedScoresDecode` both
/// do. `:76-78` says *"`compressed_attn` and `compressed_attn_paged` are
/// blocked by their HOST half"*; true of the first, whose launcher builds a
/// `CompressedAttnParams[R]` on the host and `cudaMallocAsync`s it, and false
/// of the second, whose host half is a null guard, a grid, a smem and one
/// launch. `families/attn.rs` corrected both and
/// `driver-cuda/src/fire/dsv4_compress.rs` restates them; this is the third
/// place, and the reason there are three is that none of them is the file.
///
/// **The header's whole "which launchers became rows, and which did not"
/// section is now answered by its own text being here.** Every reason it
/// gives is a reason a `LaunchRule` could not state a geometry, and a host
/// `fn` states geometries. The two it named as structurally blocked — the
/// boundary-meta pair, "blocked TWICE", once for having no element type and
/// once for a 128-wide block where `Elementwise` is 256 — are two of the four
/// the driver already launches with a `Launch` it wrote itself.
pub mod dsv4_compress {
    use super::bf16;
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        unit DSV4_COMPRESS = "attn/dsv4_compress",
            text = include_str!("../../csrc/src/attn/dsv4_compress.cuh"),
            file = "attn/dsv4_compress.cuh";

        /// `:105` — the mean over each window of `ratio` input tokens.
        ///
        /// `n` is the INPUT token count and the grid covers
        /// `n / ratio * dim`, so the extent a caller sizes the launch from
        /// and the extent the kernel is told differ by the ratio. Both
        /// survive: the launch is sized off the result, the kernel divides
        /// its own index by `dim`.
        fn average_pool = "attn::device::average_pool"(
            input: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::average_pool_bf16" => "device::bf16",
        }

        /// `:130` — the absolute position table, added in place.
        ///
        /// `_f32` in the symbol names the TABLE's format and not the data's:
        /// `ape` is fp32 and `data` is the row type's. The launcher was named
        /// for the table and the symbol keeps that name, because a symbol
        /// that changes spelling during a migration is a symbol two tables
        /// disagree about.
        fn add_ape = "attn::device::add_ape"(
            data: *mut bf16,
            ape: *const f32,
            n_compressed: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::add_ape_f32" => "device::bf16",
        }

        /// `:154` — a per-dimension softmax over `ratio` gate scores, then
        /// the weighted sum of the values under it.
        fn gated_softmax_pool = "attn::device::gated_softmax_pool"(
            kv: *const bf16,
            score: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::gated_softmax_pool_bf16" => "device::bf16",
        }

        /// The unpaged gather — one block per compressed entry, striding its
        /// own row.
        ///
        /// Carried, and it has no live caller: `fire/dsv4_compress.rs`'
        /// header records the unpaged five as *"a closed cycle of dead
        /// callers"*. It stays because the family declared it and a
        /// transcription that drops a row is a transcription nobody can
        /// check against the thing it came from. Its cost is one NVRTC
        /// instantiation.
        fn dsv4_compress_gather = "attn::device::dsv4_compress_gather"(
            kv_proj: *const bf16,
            score_proj: *const bf16,
            ape: *const f32,
            boundary_tok: *const i32,
            boundary_pos: *const i32,
            window_lo: *const i32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
        ) {
            "attn::dsv4_compress_gather_bf16" => "device::bf16",
        }

        /// `:578` — the paged gather, and the first of the two the planner
        /// actually names.
        ///
        /// `ape` is nullable: the kernel tests `ape != nullptr` twice, once
        /// per pass over the window.
        fn dsv4_compress_gather_paged = "attn::device::dsv4_compress_gather_paged"(
            state_kv: *const bf16,
            state_score: *const bf16,
            ape: *const f32,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
            page_size: i32,
        ) {
            "attn::dsv4_compress_gather_paged_bf16" => "device::bf16",
        }

        /// `:648` — commit those entries to the compressed cache, each at its
        /// own boundary token's slot.
        ///
        /// `boundary_pos[c] < 0` marks a padding row and the kernel returns
        /// on it, which is what makes a CUDA-graph-safe decode able to launch
        /// a fixed number of blocks whatever the batch does.
        fn dsv4_store_comp_entries = "attn::device::dsv4_store_comp_entries"(
            entries: *const bf16,
            comp_kv_pages: *mut bf16,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            head_dim: i32,
            page_size: i32,
        ) {
            "attn::dsv4_store_comp_entries_bf16" => "device::bf16",
        }

        /// `:530` — which decode rows close a compression window.
        ///
        /// `row_valid` is nullable and absence means *every row is valid*.
        /// Fired by `driver-cuda/src/fire/dsv4_compress.rs:220`.
        fn dsv4_boundary_meta_decode = "attn::device::dsv4_boundary_meta_decode"(
            positions: *const i32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_decode_dev" => "device::i32",
        }

        /// `:544` — the prefill form.
        ///
        /// One line different from its decode twin: the request index comes
        /// from a binary search over `qo_indptr` instead of being shortcut to
        /// the token index. A SECOND kernel rather than a wider first one,
        /// because the decode form is what a CUDA-graph capture calls and
        /// giving it two more parameters would make every capture carry a
        /// `qo_indptr` it does not read.
        fn dsv4_boundary_meta_paged = "attn::device::dsv4_boundary_meta_paged"(
            positions: *const i32,
            qo_indptr: *const u32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            num_requests: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_paged_dev" => "device::i32",
        }

        /// `:666` — the attention itself, over the compressed cache.
        ///
        /// The `PLAIN` row. `grid(total_tokens, num_q_heads)` at 128 with
        /// `(head_dim + 128) * sizeof(float)` of dynamic shared memory —
        /// eleven lines of host half, which is why this one was never
        /// blocked by anything but the sentence its sibling earned.
        fn compressed_attn_paged = "attn::device::compressed_attn_paged"(
            q: *const bf16,
            comp_kv_pages: *const bf16,
            o: *mut bf16,
            lse_out: *mut f32,
            positions: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            req_of_token: *const i32,
            num_q_heads: i32,
            head_dim: i32,
            ratio: i32,
            page_size: i32,
            scale: f32,
        ) {
            "attn::compressed_attn_paged_dev" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:216` — the merge, by log-sum-exp.
        ///
        /// If `lse2` is `-inf` the compressed half had no entries and `o1`
        /// passes through unchanged, which is the empty case handled ON THE
        /// DEVICE rather than by a host refusal — §5.1's rule for a geometry
        /// that depends on a device-side output.
        fn combine_attn_outputs = "attn::device::combine_attn_outputs"(
            o1: *const bf16,
            lse1: *const f32,
            o2: *const bf16,
            lse2: *const f32,
            o_out: *mut bf16,
            lse_out: *mut f32,
            num_heads: i32,
            head_dim: i32,
        ) {
            "attn::combine_attn_outputs_dev" => "device::bf16",
        }
    }

    /// `route_rows`' warp rounding, and the clamp that makes it legal at any
    /// width.
    ///
    /// `runtime/launch.rs:1044`. One block per compressed entry, the block as
    /// wide as the row rounded up to a warp and capped at 1024; above the cap
    /// the kernel's `for (int d = threadIdx.x; d < head_dim; d += blockDim.x)`
    /// covers the row in several passes. **The cap is safe here only because
    /// of that stride** — before it, this rounding would have silently
    /// computed a prefix.
    #[cfg(feature = "_cuda")]
    #[expect(clippy::cast_sign_loss, reason = "both are guarded positive by every caller")]
    fn route_rows(rows: i32, width: i32) -> Launch {
        let (rows, width) = (rows as u32, width as u32);
        Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
    }

    /// Build one compressed entry per boundary token.
    ///
    /// # Safety
    ///
    /// Every pointer addresses a live allocation of the extent the kernel
    /// reads, `ape` and nothing else may be null, and the stream outlives the
    /// launch.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_compress_gather_paged_bf16(
        state_kv: *const bf16,
        state_score: *const bf16,
        ape: *const f32,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        out: *mut bf16,
        num_entries: i32,
        head_dim: i32,
        ratio: i32,
        coff: i32,
        page_size: i32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        // Every refusal before the one launch, which for a single launch is
        // free and is written this way anyway: the day a second statement
        // joins this body, the order is already right.
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if ratio <= 0 {
            return Fired::Declined(Refusal::Empty { what: "ratio" });
        }
        if coff <= 0 {
            return Fired::Declined(Refusal::Empty { what: "coff" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_compress_gather_paged(
                "attn::dsv4_compress_gather_paged_bf16",
                route_rows(num_entries, head_dim),
                state_kv,
                state_score,
                ape,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                out,
                head_dim,
                ratio,
                coff,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }

    /// Commit those entries to the compressed cache.
    ///
    /// # Safety
    ///
    /// As above; no operand of this one is nullable.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_store_comp_entries_bf16(
        entries: *const bf16,
        comp_kv_pages: *mut bf16,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        num_entries: i32,
        head_dim: i32,
        page_size: i32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_store_comp_entries(
                "attn::dsv4_store_comp_entries_bf16",
                route_rows(num_entries, head_dim),
                entries,
                comp_kv_pages,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                head_dim,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }
}

/// `attn/kv_paged.cuh` — the paged KV cache's appenders, its quantised
/// writers and its dequantisers.
///
/// # The five `Specialisation`s are gone and this is why
///
/// `families::attn` carried fifteen rows for five `template <bool
/// HND_LAYOUT>` kernels — a BASE row per kernel and two arms — plus a
/// [`crate::device::Specialisation`] each, five `Take` prefix arrays, and an
/// entry in `families::attn::SPECIALISATIONS`. Twenty artefacts to express
/// `if (hnd_layout)`.
///
/// **`driver-cuda/src/fire/kv_paged.rs` already wrote that `if` in Rust**
/// and already fired the chosen arm BY NAME through `fire::hand::fire` —
/// `"attn::write_kv_bf16#hnd"`/`"…#nhd"`, and the same pair for
/// `copy_kv_cells_bf16`, `write_kv_explicit_bf16_dev` and
/// `write_kv_explicit_bf16_devwin_dev`. So `x/norm.rs:1033`'s
/// *"`Specialisation`s become `if`s"* is not work this crossing does;
/// **it is work the driver port had already done**, and the declaration was
/// the last thing still saying otherwise. Four of those five `if`s are in
/// this module now, unchanged, above `raw::` calls instead of above
/// `hand::fire`; the fifth is `copy_kv_cells_bf16`, which stayed in the
/// driver. This unit hosts the ten arms; `runtime::fire::selects` is asked
/// about none of them.
///
/// # The five base rows went with them, and could not have gone alone
///
/// `attn::write_kv_bf16` and its four siblings are the specialisations'
/// BASES: a sixteenth operand `hnd_layout: Bool` that no kernel takes, so
/// that a fire could hand the flag to `selects` and `TAKE_15` could drop it
/// again. Nothing else ever named them — the driver names arms, and a base
/// has no instantiation to lower to.
///
/// This is the second time in this family that two artefacts were each
/// other's only reason to exist (the first was `QKV_DECODE_BLOCK` and six
/// `quoted()` pins). Deleting the `Specialisation`s alone would have left
/// five rows with a bogus operand; deleting the base rows alone would have
/// broken `Specialisation::agrees`. **They are one edit, and the shape is
/// worth recognising: an artefact whose only citation is the artefact that
/// only exists to cite it.**
///
/// # `write_kv_at_positions` has no caller in the workspace
///
/// Its two arms are carried anyway, exactly as `dsv4_compress_gather` is:
/// they were being instantiated before this crossing and a transcription
/// that silently drops a kernel is a transcription nobody can check against
/// the thing it came from. Cost is two NVRTC instantiations.
///
/// # Half B: the bodies are HERE; the four rows still stand
///
/// The seven host programs moved out of `driver-cuda/src/fire/kv_paged.rs`
/// and are the `pub unsafe fn`s below. The four `table::attn` rows over this
/// root — `attn::write_kv_to_pages`, `attn::write_kv_explicit_bf16`,
/// `attn::write_kv_explicit_bf16_devwin` and
/// `attn::dequant_kv_cache_layer_to_bf16_active` — **did not cross with
/// them**, and the reason is one sentence long: four facts the rows source
/// from `AttnCtx` have no `Cx` query.
///
/// ```text
/// row                        needs, and Cx cannot state
/// write_kv_to_pages          first_token
/// dequant                    num_pages_in_batch
/// write_kv_explicit_bf16     w_page_d   w_off_d
/// write_kv_explicit_devwin   w_page_d   w_off_d   win_d
/// ```
///
/// Everything else each row needs is stated: the layer is `cx.kv_layer()`,
/// the four CSR arrays and `row_valid` and `num_requests` are `cx.plan()`,
/// `k_curr`/`v_curr` are `arg_in(0)`/`arg_in(1)`, and `total_tokens`,
/// `B` and `n_max` are all `cx.rows().count` — the last of those because
/// the devwin row is `whole = true`, so `n_max` is the fire's full lane
/// count and no windowed statement can reach it.
///
/// **`AttnCtx` carries all four already** — `first_token` at
/// `bind/mod.rs:1401`, `num_pages_in_batch` at `:1399`, `w_page_d` at
/// `:1403`, `w_off_d` at `:1405`. So this is four `query!` lines and four
/// field reads, not a feature; it is the same shape as the eleven that
/// landed in `d391f583c`, and unlike the MLA layer list there is something
/// to fill them from. `win_d` is the exception and is a different question,
/// answered below.
///
/// Until those land the four stay rows, and that is the safe state rather
/// than a compromise: **a contract retires a row, a bind does not**, so
/// writing no `contract!` leaves all four firing exactly as they fire
/// today. A `contract!` with a `none:` arm would mint an `Entry`, and an
/// `Entry` with no arm is `Route::Unbound`, which refuses the model at
/// load — `x/gemm.rs:1145` paid for that once and `write_kv_to_pages` fires
/// once per layer of every fire.
///
/// # Why the move, in its final form
///
/// **A driver op is a symbol whose body needs a driver RESOURCE** — a
/// cuBLAS handle, an NCCL communicator, a pool, an allocator. `x::gemm`'s
/// twelve are driver ops because `cublasLtMatmul` is on the far side of a
/// seam no `Cx` query can cross. These seven need no resource: they need a
/// KV layer's seventeen facts, and `Cx::kv_layer()` states all seventeen.
///
/// Half A said instead that they were blocked because `kernels-cuda-new`
/// cannot call `driver-cuda`. **That sentence is true and it is not the
/// reason** — the dependency runs the other way, and two of these bodies
/// were already calling `x::layout::envelope_*` from the middle of
/// themselves. The correction is kept in place rather than replaced,
/// because a true statement standing in for the reason, in a document
/// written to record the reason, is the failure worth naming.
///
/// # `WriteKvNative` needed no floor change, and the measurement is why
///
/// The four `Launched`/`Declined` enums did not move. All ten call sites —
/// four shim entries and four preludes in `bind/service.rs`, plus two
/// internal — consumed the return with `let _ =`. **No reader distinguished
/// `Launched` from `Declined`, and none inspected a payload.** `Fired` is
/// `#[must_use]` and says strictly more than anything read, so the rich
/// enums were a distinction with no consumer. No third `Fired` variant, no
/// floor edit.
///
/// The declines gained information rather than losing it. `kv_paged.cu:384`
/// folded two reasons into one `if (a || b) return;` because `void` could
/// carry neither; `Refusal` carries both, so the dequant now says
/// `Absent { "quantised pages on a bf16 layer" }` where it used to say the
/// same thing as `Empty { "active pages" }`.
///
/// # Two things the move surfaced
///
/// **`dequant_kv_cache_layer_to_bf16_active` is a subroutine before it is a
/// trace symbol.** Four *other* host programs call it as a prelude, at
/// `bind/service.rs`'s two FA2 decode entries and two prefill entries. A
/// `bind!` arm is reachable only from a trace, so the moved body is a
/// `pub fn` those four call and the arm will be a call to it — which is
/// `x::layout::envelope_*`'s arrangement again.
///
/// **`execution.rs`'s `Walk` for `attn::write_kv_explicit_bf16_devwin`
/// states that its row is fully sourced. Its row states `Source::Unbound`
/// on all nine operands.** `abi.rs:810` skips such a row whole, so
/// `emit_rust_dispatch` has never written a dispatch arm for it, so its
/// `RUST_SERVED` entry and its `bind::service` shim have never been
/// reachable. The claim is true of the *sibling* `write_kv_explicit` row,
/// which is fully sourced, and was written for this one. That is the fifth
/// artefact in this family that nothing re-derives, after `device.rs:991`'s
/// hold, `DSV4_COMPRESS_SIGS[4]`'s sources, `assert_eq!(checked, 14)` and
/// `RUST_SERVED`'s "all four unsourced". It is corrected in place at the
/// `Walk`, not deleted.
///
/// The consequence for the ask: `win_d` is **not** a fourth query of the
/// same kind. `AttnCtx` has no window array at all, and no trace has ever
/// reached this symbol, so the honest crossing for the devwin row is a
/// `contract!` and a `none:` arm naming that — which is safe here and only
/// here, precisely because no dispatch arm exists to shadow.
///
/// # §52.11
///
/// All four are `execution::WALKED` symbols and
/// `execution::tests::a_walk_is_only_a_walk` requires `unit_of(sym)` to be
/// `None` for every one. **No symbol in this unit is any of those four** —
/// that is what §60.6's `_dev` suffix bought, and it is why this unit can
/// exist while they are still walks. It is also what the crossing must
/// undo, in the same commit that adds the contracts.
pub mod kv_paged {
    use super::bf16;
    use crate::x::abi::MaybeConst;
    use crate::x::fp8_kind;
    // `core::ffi::c_void` and not `super::c_void`: `x::attn`'s import of it
    // is `#[cfg(feature = "_cuda")]` and a `use super::c_void` here would
    // inherit that obligation without saying so, which is exactly the defect
    // `qkv_fused` shipped and had to have fixed. A `unit!` is compiled in
    // every configuration.
    use core::ffi::c_void;

    // `Launch` is ungated at the top of `x::attn`; `Fired` and `Refusal` are
    // not. So these three name their gate EXPLICITLY and reach the canonical
    // path rather than `super::`, because `use super::X` inherits the
    // parent's cfg obligations and says nothing about it — the defect
    // `qkv_fused` shipped, invisible in any build with the feature on.
    #[cfg(feature = "_cuda")]
    use crate::x::contract::{Fired, Refusal};
    #[cfg(feature = "_cuda")]
    use crate::x::{KvDType, KvLayer, KvScheme};
    #[cfg(feature = "_cuda")]
    use super::Launch;

    // `fp8_kind` STOOD HERE — a local newtype and an `Abi` impl, carried
    // under `X_ABI_FP8_KIND_LOCAL` because `x/abi.rs` had none and is the
    // owner's file. **It is the floor now** (`x::fp8_kind`, `63d8aaebe`) and
    // the workaround is deleted; the two `fn` lines below name the real one.

    unit! {
        unit KV_PAGED = "attn/kv_paged",
            text = include_str!("../../csrc/src/attn/kv_paged.cuh"),
            file = "attn/kv_paged.cuh";

        /// `:153` — the batched append, one block per token.
        ///
        /// `first_token` is ADDED to `blockIdx.x` inside the kernel, so a
        /// launch covers `[first_token, first_token + rows)` and the grid is
        /// the count rather than the end. `row_valid` and `win` are both
        /// nullable and both tested per block.
        ///
        /// `r` and not `R`: the row spelled it `r` and the kernel spells it
        /// `R`, and a Rust parameter cannot be `R` without reading as a type
        /// parameter at every call site. The typecheck translation unit
        /// compares TYPES, not names.
        fn write_kv = "attn::device::write_kv"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: MaybeConst<u8>,
            win: MaybeConst<u32>,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            first_token: i32,
        ) {
            "attn::write_kv_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_bf16#nhd" => "device::false_type::value",
        }

        /// `:223` — the same append with each token's absolute KV position
        /// taken as data rather than derived from the page tables.
        ///
        /// **Measured: no caller.** `write_kv_at_positions` appears in this
        /// workspace exactly four times — the `__global__` at
        /// `kv_paged.cuh:223`, the two arm names below, and this line. No
        /// `<<<>>>`, no `hand::fire`, no `bind::service` entry, no
        /// `model-compiler` builder, no `table::attn` row. It is a kernel
        /// with device text and no host program on either side of the
        /// migration, and it was one before this crossing too.
        ///
        /// Carried rather than dropped, for the reason the module header
        /// gives: a transcription that silently drops a kernel is a
        /// transcription nobody can check against the thing it came from.
        /// Cost is two NVRTC instantiations in a unit that already compiles
        /// eighteen. It goes when someone deletes the `__global__`, and that
        /// deletion is a `.cuh` edit rather than a row.
        fn write_kv_at_positions = "attn::device::write_kv_at_positions"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            positions: *const i32,
            position_delta: i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_at_positions_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_at_positions_bf16#nhd" => "device::false_type::value",
        }

        /// `:279` — the append that is told each lane's physical page and
        /// offset outright, one block per lane.
        ///
        /// **The `_dev` in the symbol is §60.6's split and not a decoration.**
        /// `table::attn`'s `attn::write_kv_explicit_bf16` is a WALK — a
        /// throw, an empty-extent decline, an instantiation choice and a
        /// conditional second launch — and §52.11 requires a walked symbol to
        /// be hosted by no unit. The kernel and the walk therefore spend two
        /// spellings, and this is the kernel's.
        fn write_kv_explicit = "attn::device::write_kv_explicit"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            b: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_dev#nhd" => "device::false_type::value",
        }

        /// `:781` — the explicit append under a device-resident window.
        ///
        /// `win` is a two-word `{start, len}` the DEVICE wrote, and `n_max`
        /// is the launch's upper bound rather than the count: the kernel
        /// reads the real length out of `win` and returns on the blocks past
        /// it. That is the shape §5.1 names — a refusal that depends on a
        /// device-side value is a device-side branch, and this kernel is
        /// already written as one.
        fn write_kv_explicit_devwin = "attn::device::write_kv_explicit_devwin"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            win: *const u32,
            n_max: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_devwin_dev#nhd" => "device::false_type::value",
        }

        /// `:326` — cell-to-cell copy inside the page arena, one block per
        /// cell. The only kernel here that reads and writes the same buffer.
        fn copy_kv_cells = "attn::device::copy_kv_cells"(
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            dst_page: *const u32,
            dst_off: *const u32,
            src_page: *const u32,
            src_off: *const u32,
            n: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::copy_kv_cells_bf16#hnd" => "device::true_type::value",
            "attn::copy_kv_cells_bf16#nhd" => "device::false_type::value",
        }

        /// `:390` — the fp8 append with one scale for the whole tensor.
        ///
        /// `k_pages` is `*mut u8` and the kernel says `__nv_fp8_storage_t*`,
        /// which IS `unsigned char`; `static_assert(is_same_v<>)` resolves
        /// the typedef and the row already said `U8sMut`. The BYTE is not the
        /// format — the format is `fp8_kind`, and it is an operand rather
        /// than a template default because defaulting it would decode an
        /// E5M2 page as E4M3 and produce a numerically plausible wrong
        /// answer.
        fn write_kv_fp8_per_tensor = "attn::device::write_kv_fp8_per_tensor"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            fp8_kind: fp8_kind,
        ) {
            "attn::write_kv_fp8_per_tensor" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:425` — the per-token-per-head quantised append, `template <bool
        /// UseFp8>`.
        ///
        /// **This is the one where an enum is not a flag.** The two
        /// instantiations are int8 and fp8, and the symbols say so; the
        /// pages are `void*` because the kernel casts to the storage type
        /// under the template. `false_type` is INT8 and `true_type` is FP8 —
        /// the opposite reading of the pair to the four `HND_LAYOUT`
        /// kernels above, where `true` is the layout named second.
        fn write_kv_per_token_head = "attn::device::write_kv_per_token_head"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages_raw: *mut c_void,
            v_pages_raw: *mut c_void,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_int8_per_token_head" => "device::false_type::value",
            "attn::write_kv_fp8_per_token_head" => "device::true_type::value",
        }

        /// `:562` — the fp4 append, two values to the byte, blocked scales.
        fn write_kv_fp4_block = "attn::device::write_kv_fp4_block"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::write_kv_fp4_block" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:655` — the per-tensor fp8 dequantiser over the active pages.
        ///
        /// `n` is `i64` because it indexes a page arena that is multiple
        /// gigabytes at production page counts, and `page_elems` is the whole
        /// page's element count rather than a head dim: this form needs no
        /// per-head geometry because its scale is the tensor's.
        fn dequant_fp8_pages_active = "attn::device::dequant_fp8_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_elems: i32,
            fp8_kind: fp8_kind,
        ) {
            "attn::dequant_fp8_pages_active_bf16" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:678` — the per-token-per-head fp8 dequantiser.
        fn dequant_fp8_per_token_head = "attn::device::dequant_fp8_per_token_head_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_fp8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:708` — the same, for int8 pages.
        fn dequant_int8_per_token_head = "attn::device::dequant_int8_per_token_head_pages_active"(
            k_pages: *const i8,
            v_pages: *const i8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_int8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:736` — the fp4 dequantiser.
        ///
        /// **`logical_n` and not `n`.** An fp4 page holds two values per
        /// byte, so the count the grid covers is the LOGICAL element count
        /// and every address inside the kernel halves it. The kernel's name
        /// is kept because an operand spelled `n` here would be the one
        /// number in the signature that means something else.
        fn dequant_fp4 = "attn::device::dequant_fp4_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            logical_n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::dequant_fp4_pages_active_bf16" => "device::bf16",
        }

        /// `:826` — the window page view, one thread walking every request.
        ///
        /// `LaunchRule::Single` in the family it left: one block, one
        /// thread. `fire/kv_paged.rs` writes that `Launch` by hand and this
        /// row states no rule, which is the same arrangement every unit row
        /// is in.
        fn build_window_page_view = "attn::device::build_window_page_view"(
            src_indices: *const u32,
            src_indptr: *const u32,
            keep_pages: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            r: i32,
        ) {
            "attn::build_window_page_view" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:860` — the full split view, one warp.
        ///
        /// Note the parameter order: `src_indices` is LAST, after three
        /// outputs. That is the kernel's and it is kept, because the only
        /// thing a reordering would buy is a signature that reads better and
        /// binds wrong.
        fn build_full_split_view = "attn::device::build_full_split_view"(
            src_indptr: *const u32,
            src_last_page_len: *const u32,
            splits: i32,
            page_size: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            dst_last: *mut u32,
            src_indices: *const u32,
        ) {
            "attn::build_full_split_view" => crate::device::DeviceKernel::PLAIN,
        }
    }

    // =====================================================================
    // HALF B — the four host programs, MOVED (not made driver ops).
    //
    // The discriminator, which Half A got wrong and stated the correction
    // for: **a driver op is a symbol whose body needs a driver RESOURCE** —
    // a cuBLAS handle, an NCCL comm, a pool, an allocator. `x::gemm`'s
    // twelve are driver ops because `cublasLtMatmul` lives across a seam no
    // `Cx` can reach. These four need no resource: they need a KV layer's
    // seventeen facts, and `Cx::kv_layer()` states all seventeen.
    //
    // What each body needed and did NOT have before `d391f583c`: eleven of
    // those seventeen. `Cx::KvLayer` carries them now, including the two
    // predicates ANSWERED — `has_envelopes` and `is_native_bf16` — so no
    // body below re-derives either.
    // =====================================================================

    /// `kv_paged.cu`'s `constexpr int BLOCK = 256`, which every launch in
    /// that file used and only one of which was ever written down.
    #[cfg(feature = "_cuda")]
    const BLOCK: u32 = 256;

    /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
    /// gives them.
    ///
    /// Spelled as constants and not as literals at the two call sites
    /// because they are the same two values in both, and the C++ wrote the
    /// ternary twice (`kv_paged.cu:394-396` is the second copy).
    #[cfg(feature = "_cuda")]
    const NV_E4M3: u32 = 0;
    #[cfg(feature = "_cuda")]
    const NV_E5M2: u32 = 1;

    /// The interpretation an fp8 page is written and read under.
    ///
    /// **E4M3 is the fallthrough and not a default.** The C++ tested for
    /// E5M2 and took E4M3 for everything else, including dtypes that are
    /// not fp8 at all — which is unreachable, because only an fp8 scheme
    /// reaches either caller. Reproduced rather than tightened: tightening
    /// it would be a behaviour change dressed as a cleanup.
    #[cfg(feature = "_cuda")]
    fn fp8_kind_of(storage_dtype: KvDType) -> fp8_kind {
        fp8_kind(if matches!(storage_dtype, KvDType::Fp8E5M2) { NV_E5M2 } else { NV_E4M3 })
    }

    /// NVFP4's block, when the layer states none.
    ///
    /// `kv_paged.cu:242-244`. 16 is the arena's layout and not a tuning
    /// knob — the writer and the reader both spelled it, and a cache
    /// written at one block and read at another is silently wrong rather
    /// than slow. One `fn` so the two cannot drift.
    #[cfg(feature = "_cuda")]
    fn fp4_block_size(layer: &KvLayer) -> i32 {
        if layer.block_size > 0 { layer.block_size } else { 16 }
    }

    /// An upper bound on the pages an append can touch.
    ///
    /// `pub` because it is not this module's number: `envelope_update_appended`
    /// takes a `max_touched` it cannot derive — it holds the page CSR but not
    /// the token count that will be written into it — so the caller that
    /// knows both supplies it.
    ///
    /// The bound is the token span rounded up to whole pages, plus one page
    /// per request for the partially-filled tail each request appends into.
    /// Returns `0` for a non-positive page size rather than dividing by it.
    #[must_use]
    pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
        if page_size <= 0 {
            return 0;
        }
        (total_tokens + page_size - 1) / page_size + num_requests
    }

    // ---------------------------------------------------------------------
    // 1. The explicit-slot writes — a (page, offset) pair per row, already
    //    resolved on the device, so neither of these two reads a CSR.
    // ---------------------------------------------------------------------

    /// `attn::write_kv_explicit_bf16` — write B rows to B explicit slots.
    ///
    /// The parameter order is the driver's, unchanged, including `stream`
    /// arriving before `row_valid`. Reordering it would read better and is
    /// exactly the edit that turns a mechanical move into a silent
    /// mis-binding, which is the class this port keeps finding.
    ///
    /// # Panics
    ///
    /// If the layer is not native bf16. `kv_paged.cu:314` threw, and a
    /// throw is not a refusal: a quantised cache reaching an unquantised
    /// writer is a caller that computed the wrong thing, not a shape this
    /// kernel declines to handle.
    ///
    /// # Safety
    ///
    /// Every pointer must be a device allocation of the stated extent, and
    /// `stream` a live stream.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        b: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16 requires native bf16 KV cache"
        );
        if b <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit(
                symbol,
                Launch::per_row(b.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                b,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }

        // `k_curr` and NOT `k_pages`: the merge reads the ROWS just written,
        // which are contiguous and are the only thing whose envelope changed.
        // Reading them back through the page indirection would be the same
        // values at a worse stride, and would need the write to have landed.
        if layer.has_envelopes && !layer.hnd {
            let _ = unsafe {
                crate::x::layout::envelope_merge_written(
                    k_curr,
                    w_page,
                    w_off,
                    MaybeConst::new(row_valid),
                    layer.k_env_min.cast(),
                    layer.k_env_max.cast(),
                    b,
                    layer.num_kv_heads,
                    layer.head_dim,
                    stream,
                )
            };
        }
        Fired::Launched
    }

    /// `attn::write_kv_explicit_bf16_devwin` — the same write with a
    /// device-side window, so the row count is a ceiling rather than a
    /// count and the kernel reads `win[]` to find the real one.
    ///
    /// # Panics
    ///
    /// If the layer is not native bf16 (`kv_paged.cu:252`), or if it
    /// carries envelopes (`:262`) — envelope maintenance was never
    /// windowed, and merging a window's rows against a full-row envelope
    /// would widen it with rows the window excluded.
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16_devwin(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        win_d: *const u32,
        n_max: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        if n_max <= 0 {
            return Fired::Declined(Refusal::Empty { what: "lanes" });
        }
        // `:262`, and it stays a throw for the reason the C++ gave it one:
        // a windowed write into an enveloped layer would leave the envelope
        // describing rows the window never wrote.
        assert!(
            !layer.has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_devwin_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit_devwin(
                symbol,
                Launch::per_row(n_max.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                win_d,
                n_max,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }
        Fired::Launched
    }

    // ---------------------------------------------------------------------
    // 2. The CSR append — one entry point, two halves, four quantised arms.
    // ---------------------------------------------------------------------

    /// The native-bf16 append, `kv_paged.cu:60-120`.
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`]; the four CSR arrays must describe
    /// `num_requests` requests over `total_tokens` tokens.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        let launch_tokens = total_tokens - first_token;
        if launch_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens after first_token" });
        }

        let symbol =
            if layer.hnd { "attn::write_kv_bf16#hnd" } else { "attn::write_kv_bf16#nhd" };
        unsafe {
            raw::write_kv(
                symbol,
                Launch::per_row(launch_tokens.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                MaybeConst::new(row_valid),
                // The window this kernel can take and this caller never has:
                // the CSR append writes every token it was given.
                MaybeConst::none(),
                num_requests,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                first_token,
                stream,
            );
        }

        // Two statements in a `fn`, in order, on one stream — `Composed`'s
        // shape without `Composed`, because the second reads what the first
        // wrote and nothing between them is a decision.
        if layer.has_envelopes && !layer.hnd && total_tokens > 0 {
            let _ = unsafe {
                crate::x::layout::envelope_update_appended(
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
        Fired::Launched
    }

    /// The quantised append, `kv_paged.cu:130-190` — four schemes, three
    /// kernels, and a fifth arm that declines.
    ///
    /// The per-token-head case fires **two symbols and not a
    /// `Specialisation`**: `UseFp8` is read off the cache's scheme and
    /// appears in no parameter of either kernel, so the choice is the host's
    /// and the registry had no way to make it. That was true of all five of
    /// this file's `Specialisation`s and is why `device::SPECIALISED` is
    /// empty.
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`]; the layer's scale planes must be
    /// sized for its scheme.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_quantised(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
    ) -> Fired {
        if total_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens" });
        }
        let page_size = layer.page_size;
        let h_kv = layer.num_kv_heads;
        let d = layer.head_dim;
        let tokens = total_tokens.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match layer.scheme {
            KvScheme::Fp8PerTensor => unsafe {
                raw::write_kv_fp8_per_tensor(
                    "attn::write_kv_fp8_per_tensor",
                    Launch::per_row(tokens, BLOCK),
                    k_curr,
                    v_curr,
                    layer.k_pages.cast::<u8>(),
                    layer.v_pages.cast::<u8>(),
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    num_requests,
                    page_size,
                    h_kv,
                    d,
                    fp8_kind_of(layer.storage_dtype),
                    stream,
                );
            },

            KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead => {
                let symbol = if matches!(layer.scheme, KvScheme::Fp8PerTokenHead) {
                    "attn::write_kv_fp8_per_token_head"
                } else {
                    "attn::write_kv_int8_per_token_head"
                };
                // Two `float`s per warp: the block reduces an absmax for K
                // and one for V, and a warp contributes one of each.
                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                let launch = Launch {
                    grid: [tokens, heads, 1],
                    block: [BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(smem);
                unsafe {
                    raw::write_kv_per_token_head(
                        symbol,
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages,
                        layer.v_pages,
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        stream,
                    );
                }
            }

            KvScheme::Fp4Block => {
                let block_size = fp4_block_size(layer);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                // A warp per block and not a block per token: an fp4 block is
                // 16 values, and 32 lanes cover one with room for the pair
                // packing.
                let launch = Launch {
                    grid: [tokens, heads, blocks.unsigned_abs()],
                    block: [32, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                };
                unsafe {
                    raw::write_kv_fp4_block(
                        "attn::write_kv_fp4_block",
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages.cast::<u8>(),
                        layer.v_pages.cast::<u8>(),
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        block_size,
                        stream,
                    );
                }
            }

            KvScheme::Native => {
                return Fired::Declined(Refusal::Absent {
                    what: "a quantised writer for Native storage",
                });
            }
        }
        Fired::Launched
    }

    /// `attn::write_kv_to_pages` — the entry point, which chooses.
    ///
    /// # Panics
    ///
    /// If `first_token != 0` on a cache that is not native bf16
    /// (`kv_paged.cu:130-134`). A partial write resumes into a page the
    /// quantised writers cannot address mid-block, and the C++ threw rather
    /// than write a wrong scale.
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        assert!(
            first_token == 0 || layer.is_native_bf16,
            "attn::write_kv_to_pages: partial (first_token) writes require the \
             native bf16 cache"
        );
        if layer.is_native_bf16 {
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
        // The driver translated the quantised decline into a native one by
        // hand, through two enums that agreed on nothing. One `Fired` needs
        // no translation, and the `Refusal` the callee stated is the one the
        // caller returns — which is the reason `Fired` says more here than
        // the pair it replaced, not less.
        unsafe {
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
        }
    }

    // ---------------------------------------------------------------------
    // 3. The active-page dequant — the beam-repair cell move,
    //    `kv_paged.cu:352-378`.
    // ---------------------------------------------------------------------

    /// The fp8-per-tensor arm, called by name from
    /// [`dequant_kv_cache_layer_to_bf16_active`] and directly by the four
    /// host programs that only ever hold that scheme.
    ///
    /// # Safety
    ///
    /// `kv_page_indices` must list `num_pages_in_batch` valid page indices,
    /// and the layer's bf16 mirror planes must be sized for them.
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_fp8_per_tensor_pages_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        // `kv_paged.cu:384` folded these two into one `if (a || b) return;`
        // because `void` could carry neither reason. `Refusal` can carry
        // both, so it does — a native layer has nothing to dequantise and an
        // empty batch has nothing to dequantise it from, and those are not
        // the same answer to the caller that asked.
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        if !matches!(layer.scheme, KvScheme::Fp8PerTensor) {
            return Fired::Declined(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) = active_geometry(layer, num_pages_in_batch);
        unsafe {
            raw::dequant_fp8_pages_active(
                "attn::dequant_fp8_pages_active_bf16",
                launch,
                layer.k_pages.cast::<u8>().cast_const(),
                layer.v_pages.cast::<u8>().cast_const(),
                layer.k_bf16_pages.cast::<bf16>(),
                layer.v_bf16_pages.cast::<bf16>(),
                kv_page_indices,
                logical_n,
                page_elems,
                fp8_kind_of(layer.storage_dtype),
                stream,
            );
        }
        Fired::Launched
    }

    /// The element count an active-page pass covers, and the grid that
    /// covers it.
    ///
    /// **The widening is load-bearing** and is why it is a `fn`:
    /// `page_elems` is an `int`, and `num_pages_in_batch * page_elems`
    /// overflows 32 bits at production page counts — which is the whole
    /// reason every one of these kernels takes `n` as a `long long`. The C++
    /// widened before multiplying (`kv_paged.cu:385-388`); so does this, in
    /// one place, so the four arms cannot each get it right separately.
    #[cfg(feature = "_cuda")]
    fn active_geometry(layer: &KvLayer, num_pages_in_batch: i32) -> (i64, i32, Launch) {
        let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        let launch = Launch {
            grid: [blocks as u32, 1, 1],
            block: [BLOCK, 1, 1],
            smem: 0,
            smem_opt_in: false,
        };
        (logical_n, page_elems, launch)
    }

    /// `attn::dequant_kv_cache_layer_to_bf16_active` — dequantise the pages
    /// this batch touches into the layer's bf16 mirror.
    ///
    /// `pub` and called from four other host programs as well as from its
    /// own arm: it is a prelude, not a leaf. Those four called it through
    /// the shim and now call it here.
    ///
    /// There is no `cudaGetLastError` after the launch, deliberately: the
    /// C++ checked because a shim entry that fired one of four names could
    /// not say which had failed, and `hand::fire` names the symbol it fired.
    ///
    /// # Safety
    ///
    /// As [`dequant_fp8_per_tensor_pages_active`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_kv_cache_layer_to_bf16_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        let (logical_n, _page_elems, launch) = active_geometry(layer, num_pages_in_batch);

        match layer.scheme {
            // Called rather than repeated. Its own guards re-run and pass —
            // three predicates over facts nothing between the two calls can
            // change.
            KvScheme::Fp8PerTensor => unsafe {
                dequant_fp8_per_tensor_pages_active(
                    layer,
                    kv_page_indices,
                    num_pages_in_batch,
                    stream,
                )
            },

            // A scale plane per (token, head) rather than one per tensor, so
            // the kernel needs the page geometry to find a scale and takes
            // `page_size`, `h_kv` and `d` where the per-tensor arm took a
            // flat `page_elems`.
            KvScheme::Fp8PerTokenHead => {
                unsafe {
                    raw::dequant_fp8_per_token_head(
                        "attn::dequant_fp8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            // Byte-for-byte the arm above with a different element type on
            // the page planes. Two symbols and not one template because the
            // pages are `i8` in one and `u8` in the other, and a single
            // declaration could not say which.
            KvScheme::Int8PerTokenHead => {
                unsafe {
                    raw::dequant_int8_per_token_head(
                        "attn::dequant_int8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<i8>().cast_const(),
                        layer.v_pages.cast::<i8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            // The only arm with a twelfth operand, and the only one whose
            // `n` is LOGICAL rather than physical: an fp4 page holds two
            // values per byte, so the grid covers twice the bytes it reads
            // and every address inside the kernel is derived by halving.
            KvScheme::Fp4Block => {
                unsafe {
                    raw::dequant_fp4(
                        "attn::dequant_fp4_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        fp4_block_size(layer),
                        stream,
                    );
                }
                Fired::Launched
            }

            // `case KvCacheScheme::Native: break;`. Unreachable in the C++,
            // which returned on `is_native_bf16()` first, and reachable here
            // only for a cache declaring `Native` storage in a dtype that is
            // not bf16.
            KvScheme::Native => {
                Fired::Declined(Refusal::Absent { what: "a quantised dequant for Native storage" })
            }
        }
    }
}

pub static UNITS: &[Unit] = &[
    attn_res::ATTN_RES,
    attn_sink::ATTN_SINK,
    dsa_indexer::DSA_INDEXER,
    dsv4_compress::DSV4_COMPRESS,
    head_dim_pad::HEAD_DIM_PAD,
    kimi_mla::KIMI_MLA,
    kv_paged::KV_PAGED,
    mla_fa2::MLA_FA2,
    mla_naive::MLA_NAIVE,
    mla_paged::MLA_PAGED,
    pack_dense_mask::PACK_DENSE_MASK,
    qkv_fused::QKV_FUSED,
    softcap::SOFTCAP,
    split_packed::SPLIT_PACKED,
];

// ---------------------------------------------------------------------------
// The numbers, once each.
// ---------------------------------------------------------------------------

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
///
/// The block every pointwise rule in this tree uses, and the block
/// [`elementwise`] and [`attn_res_blend_bf16`] take.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:599` — `const PAD_BLOCK: u32 = 128;`.
///
/// **A kernel requirement and not a tuning number.** Both head-dim kernels
/// stride `d += kPadBlock`, the compile-time constant at
/// `head_dim_pad.cuh:65`, so a narrower block never visits the columns above
/// it — which for `pad_head_dim` is padding that was never zeroed, and for
/// `strip_head_dim` a head whose tail keeps whatever the destination held.
/// Neither fails; both answer.
const PAD_BLOCK: u32 = 128;

/// `runtime/launch.rs:608` — `const SINK_BLOCK_MIN: u32 = WARP;`.
const SINK_BLOCK_MIN: u32 = 32;

/// `runtime/launch.rs:610` — `const SINK_BLOCK_MAX: u32 = 128;`.
const SINK_BLOCK_MAX: u32 = 128;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:828-834` and its `eval` arm — `n = dims.rows *
/// dims.width`, then `grid [ceil(n / 256), 1, 1]`, `block [256, 1, 1]`, no
/// shared memory. The grid rounds UP, which is why every kernel fired through
/// it keeps its own element count as an operand.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
///
/// `runtime/launch.rs:737-746` — grid `[rows, 1, 1]`, block `[256, 1, 1]`,
/// shared `(256 / 32) * 4` bytes: one `float` per warp, which is what a
/// two-stage block reduction needs and all it needs.
///
/// One block per ROW and not per element, because the reduction is over the
/// row: `kimi_mla.cuh:127` halves from `BLOCK_DIM / 2`, so the block width is
/// the tree's width and the same 256 the row's `elem` states. **They are one
/// number and only their agreement makes the shared array the right size.**
///
/// `smem` is 32 bytes at 256 threads. Stated as an expression rather than a
/// literal so that a changed [`BLOCK`] changes both ends together — the thing
/// that could not be said while the width lived in a template default.
#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1417-1423`, and **this launcher is the one the rule was
/// derived from** — `attn/attn_sink.cu`, `attention_sink_rescale_bf16`:
///
/// ```text
/// const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_q_heads));
/// const int block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim);
/// device::attn_sink_rescale<bf16><<<grid, block, 0, stream>>>(...);
/// ```
///
/// which is `[rows, q_heads, 1]` and `clamp(head_dim, 32, 128)` to the digit.
/// **The ROW is `grid.x` and the head is `grid.y` here, the transpose of
/// [`per_head`]'s** — the two axis orders are the kernels' and not a
/// convention, and a rule read off the wrong one runs the same block count
/// over the wrong cells.
///
/// `q_heads` and not `kv_heads`, because the tensor this rescales is the
/// attention OUTPUT: one row per query head. A grouped-query fire has two
/// head counts to pick the wrong one from.
#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `u32::clamp` is not `const`, and the rule's expression is transcribed
/// rather than rearranged.
#[must_use]
const fn head_dim_block(head_dim: u32) -> u32 {
    if head_dim < SINK_BLOCK_MIN {
        SINK_BLOCK_MIN
    } else if head_dim > SINK_BLOCK_MAX {
        SINK_BLOCK_MAX
    } else {
        head_dim
    }
}

/// `LaunchRule::PerHead`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1381-1383` — `grid [heads, rows, 1]`, `block [128, 1,
/// 1]`, no shared memory — and `head_dim_pad.cu` is the launcher it cites:
/// `dim3 grid(num_heads, num_tokens)`, `dim3 block(kPadBlock)`. **The head is
/// `grid.x` and the row is `grid.y`**, the transpose of every other
/// head-shaped rule in the vocabulary, because that is the axis order these
/// two kernels read: `blockIdx.y` is the token at `head_dim_pad.cuh:78` and
/// `:97`, `blockIdx.x` is the head at `:79` and `:98`.
///
/// # THE DEFECT THIS FUNCTION CLOSES, measured
///
/// The rule evaluates `per_head(dims.rows, dims.kv_heads)` — it reads
/// `ctx.num_kv_heads`, **a field no part of either row mentions**. The head
/// count these kernels address with is the packed side's width over
/// `head_dim`, which is a QUERY head count wherever a q-projection is what
/// was padded. At Phi-3-mini's 12 heads of 64 with `num_kv_heads = 6` the two
/// arms differ in **6,100 of 12,544 bytes for the pad** and **4,588 of 9,472
/// for the strip**: the JIT writes half the rectangle and reports success.
///
/// Both symbols are in `device::JIT_DISPATCHED` all the same, which is how
/// the defect was reachable. **The port closes it by construction**: the `fn`
/// below is handed `num_heads`, uses that same number for `grid.x` and for
/// the kernel's addressing, and has no way to reach a KV head count. That is
/// the whole of the argument for taking this root early.
#[must_use]
const fn per_head(rows: u32, heads: u32) -> Launch {
    Launch { grid: [heads, rows, 1], block: [PAD_BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// The merge's geometry — the grid of [`per_head_elementwise`] and a
/// **different block**, which is why no `LaunchRule` states it.
///
/// `dsv4_compress.cu:65` and `:87`:
///
/// ```text
/// dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_heads));
/// const int block = (head_dim < 32) ? 32 : ((head_dim > 256) ? 256 : head_dim);
/// ```
///
/// The grid is `PerHeadElementwise` to the digit. The block is not: this
/// clamps into `[32, 256]` and the rule clamps into `[32, 128]`, so on a head
/// wider than 128 the rule answers with half these threads. The kernel
/// strides `d += blockDim.x` and reduces nothing, so the narrow block
/// computes the same bytes in two passes — **a slower kernel and never a
/// wrong answer**, which is precisely why rowing it was refused: the row
/// would agree at deepseek_v4's 128-wide heads, stop agreeing at the first
/// config that widened one, and nothing would fail and nothing would report.
///
/// `driver-cuda/src/fire/dsv4_compress.rs` carried that argument at length
/// and closed it *"Reconciling it is a decision about `SINK_BLOCK_MAX` in
/// `runtime/launch.rs`, which is not this file's to make."* **It is still not
/// this file's to make, and fn-world is why it never has to be**: a `fn`
/// states its own geometry, so the two clamps can differ in the open instead
/// of one of them being wrong. What the crossing changes is that the
/// divergence is now four lines from the launch it belongs to rather than in
/// another crate.
#[must_use]
const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [combine_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `[32, 256]`, transcribed rather than rearranged — `u32::clamp` is not
/// `const`, and [`head_dim_block`]'s twin is deliberately not reused: these
/// are two clamps that agree on the floor and disagree on the ceiling, and a
/// shared helper would be the place someone later unifies them.
#[must_use]
const fn combine_block(head_dim: u32) -> u32 {
    if head_dim < COMBINE_BLOCK_MIN {
        COMBINE_BLOCK_MIN
    } else if head_dim > COMBINE_BLOCK_MAX {
        COMBINE_BLOCK_MAX
    } else {
        head_dim
    }
}

/// A warp. `dsv4_compress.cu:87`'s `(head_dim < 32) ? 32`.
const COMBINE_BLOCK_MIN: u32 = 32;

/// `dsv4_compress.cu:87`'s `(head_dim > 256) ? 256`. **Not
/// `SINK_BLOCK_MAX`** — see [`combine_attn`].
const COMBINE_BLOCK_MAX: u32 = 256;

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning
// `Fired` so that "it declined" cannot be spelled like "it ran".
// ---------------------------------------------------------------------------

/// `attn::lse_log2_to_ln` — rebase flashinfer's LSE from log2 to ln, in place.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was `LaunchRule::Elementwise` in `device::JIT_DISPATCHED`, so
/// `attn/attn_sink.cu` went with the file and there was nothing to port.
/// [`elementwise`] is that rule, and `n` is the element count the deleted row
/// sourced `Source::OutElements(0)` — `rows * out_width(0)`, which the bind
/// spells with the same two queries.
///
/// The rebase is in place on the value it names: the statement's result and
/// its first operand are the same buffer, so the element count is the
/// result's own extent.
///
/// # Safety
///
/// `lse` must address `n` live, writable `f32`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn lse_log2_to_ln(lse: *mut f32, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "lse elements" });
    }
    let Ok(elems) = u32::try_from(n) else {
        // `Elementwise`'s grid is `ceil(n / 256)` in `u32`. An LSE with more
        // than 4.29e9 entries is not a fire this tree can make, and a silent
        // truncation here would launch a grid over a prefix of it.
        //
        // WIDE AND NOT NARROW, which this arm had backwards until
        // `Refusal::Wide` existed to say it: the count is ABOVE what the
        // grid can express, not below the kernel's smallest unit of work.
        // Both fields saturate — the ceiling is `u32::MAX` and the value is
        // larger still, and neither fits the `i32` the refusal carries — so
        // the sentence reads with `at` equal to `max`. What it gets right is
        // the direction, which is the whole reason the variant exists.
        return Fired::Declined(Refusal::Wide {
            what: "lse elements",
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    unsafe {
        attn_sink::raw::lse_log2_to_ln(
            "attn::lse_log2_to_ln",
            elementwise(elems),
            lse,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attention_sink_rescale_bf16` — gpt-oss's per-head sink correction,
/// in place on the attention output.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`per_head_elementwise`] transcribes `attn/attn_sink.cu`'s own `<<<>>>`,
/// which is the launcher `runtime::launch::per_head_elementwise` was derived
/// from. Row on `grid.x`, head on `grid.y`, block `clamp(head_dim, 32, 128)`.
///
/// In place on the output it corrects, which is what lets the o_proj GEMM and
/// the residual add downstream read rescaled activations without a copy.
/// `lse` is the dispatch's SECOND result — a value only a sink layer declares
/// — and the sinks are the layer's learned weight.
///
/// # Safety
///
/// `o` addresses `n * num_q_heads * head_dim` live, writable bf16 elements;
/// `lse` addresses `n * num_q_heads` live `f32`s; `sinks` addresses
/// `num_q_heads` live bf16 elements. All three live on `stream`, which must
/// outlive the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attention_sink_rescale_bf16(
    o: *mut bf16,
    lse: *const f32,
    sinks: *const bf16,
    n: i32,
    num_q_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads" });
    }
    // `runtime/launch.rs`' `Ungeometric::Empty`: a head of no channels makes
    // the loop execute zero times, so the launch would report success having
    // written nothing.
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        attn_sink::raw::attn_sink_rescale(
            "attn::attention_sink_rescale_bf16",
            per_head_elementwise(n.unsigned_abs(), num_q_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o,
            lse,
            sinks,
            n,
            num_q_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attn_res_blend_bf16` — K3's residual-block blend.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::PerRow` — `runtime/launch.rs:1103`, `grid [rows, 1, 1]`,
/// `block [256, 1, 1]`, no shared memory — and the launcher it reproduces is
/// `<<<T, kThreads, 0>>>` in `attn/attn_res.cu` with `constexpr int kThreads
/// = 256` at `attn_res.cuh:69`.
///
/// **`PerRow`, not `Rms`.** `Rms` requests thirty-two bytes of dynamic shared
/// memory that no launcher here passes and no kernel here reads —
/// `block_sum`'s warp buffer, which this shape has no reduction to need: its
/// reduction is the static `__shared__ float scratch[kThreads / 32]` at
/// `attn_res.cuh:109`. Harmless in effect and wrong as a contract: a rule is
/// meant to REPRODUCE its launcher, and one that asks for memory the launcher
/// did not is a rule nobody can check against the `<<<>>>` it came from.
///
/// # `b` is an operand over an operand
///
/// How many candidate blocks the packed input holds is the BLOCKS operand's
/// row width over the RESULT's — an operand-over-operand ratio, where every
/// `*WidthOver` variant the row grammar had divides by a CONTEXT field. A
/// caller that guessed a param would launch the right kernel over the wrong
/// rectangle. The bind reads `in_width(1) / out_width(0)`, which is the two
/// widths of one statement.
///
/// # Safety
///
/// `prefix` and `out` address `t * h` live bf16 elements, `blocks` addresses
/// `t * b * h`, and `norm_weight` and `proj_weight` address `h` each. `out`
/// is writable and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attn_res_blend_bf16(
    prefix: *const bf16,
    blocks: *const bf16,
    norm_weight: *const bf16,
    proj_weight: *const bf16,
    out: *mut bf16,
    t: i32,
    b: i32,
    h: i32,
    block_rows: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if b <= 0 {
        return Fired::Declined(Refusal::Empty { what: "blocks" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        attn_res::raw::attn_res_blend(
            "attn::attn_res_blend_bf16",
            Launch::per_row(t.unsigned_abs(), BLOCK),
            prefix,
            blocks,
            norm_weight,
            proj_weight,
            out,
            b,
            h,
            block_rows,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::pad_head_dim_bf16` — pad every head out to a width flashinfer
/// compiles for.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`per_head`], which is `head_dim_pad.cu`'s own `dim3 grid(num_heads,
/// num_tokens)` / `dim3 block(kPadBlock)` — and see that function's doc for
/// the `dims.kv_heads` defect this signature closes.
///
/// # Which side is PACKED
///
/// Whichever end is `head_dim` wide — the input on the way in, the output on
/// the way out. So the head count divides out of the packed side and the
/// padded width is the other side over that count. Both readings are the
/// ahead-of-time rows', kept verbatim; the bind below is where they are
/// spelled.
///
/// # Safety
///
/// `packed` addresses `num_tokens * num_heads * head_dim` live bf16 elements
/// and `padded` addresses `num_tokens * num_heads * head_dim_padded`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn pad_head_dim_bf16(
    packed: *const bf16,
    padded: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::pad_head_dim(
            "attn::pad_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            packed,
            padded,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim_bf16`].
///
/// Same geometry, same refusals, the two buffers swapped: `padded` is read
/// and `packed` is written, so the head count comes off the PACKED side,
/// which is the output here.
///
/// # Safety
///
/// `padded` addresses `num_tokens * num_heads * head_dim_padded` live bf16
/// elements and `packed` addresses `num_tokens * num_heads * head_dim`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn strip_head_dim_bf16(
    padded: *const bf16,
    packed: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::strip_head_dim(
            "attn::strip_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            padded,
            packed,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// The four preconditions both head-dim launchers share, resolved BEFORE
/// either of them launches anything.
///
/// One function rather than two copies, because the pad and the strip are the
/// same rectangle read from opposite ends and a copy that drifted would
/// refuse one direction and not the other.
///
/// The last is not a geometry check: `head_dim_padded < head_dim` makes
/// `pad_head_dim` copy `head_dim` values into a `head_dim_padded`-wide row,
/// which is a write past the destination's head and not an empty launch.
/// `head_dim_pad.cuh`'s loop bounds itself on the PADDED extent, so nothing
/// on the device stops it.
#[cfg(feature = "_cuda")]
#[must_use]
fn head_dim_refusal(
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Option<Refusal> {
    if num_tokens <= 0 {
        return Some(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Some(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Some(Refusal::Empty { what: "head_dim" });
    }
    if head_dim_padded < head_dim {
        return Some(Refusal::Narrow { what: "head_dim_padded", at: head_dim_padded });
    }
    None
}

// ---------------------------------------------------------------------------
/// The guard `attn_softcap.cu`'s launcher opened with, as a refusal.
///
/// `if (n == 0 || !(cap > 0.f)) return;` — a bare `return` inside a `<<<>>>`
/// wrapper, which the caller could not tell apart from a launch. Here it has
/// two names, because the two halves are different facts: an empty extent is
/// [`Refusal::Empty`], and an absent cap is [`Refusal::Unstated`], which is
/// the same sentence `Source::CtxNonZero("final_logit_softcap")` was making
/// in the row.
///
/// **`cap.is_nan() || cap <= 0.0` is `!(cap > 0.f)` written without a negated
/// comparison**, and the equality is exact: for NaN the original's `>` is
/// false so it returned, and for every other value `cap <= 0.0` is the
/// complement of `cap > 0.0`. Positive infinity passes both, as it did in the
/// archive — `1/inf` is 0, `tanh(0)` is 0 and `inf * 0` is NaN, so an
/// infinite cap poisons the logits. That is the launcher's behaviour and it
/// is kept: a port is not the place to fix a case no `Facts` can produce,
/// and inventing a refusal here would make this `fn` and the row it replaces
/// answer differently.
#[cfg(feature = "_cuda")]
fn softcap_launch(cap: f32, n: usize) -> Result<Launch, Refusal> {
    if cap.is_nan() || cap <= 0.0 {
        return Err(Refusal::Unstated { what: "a logit soft cap" });
    }
    if n == 0 {
        return Err(Refusal::Empty { what: "logit elements" });
    }
    let Ok(elems) = u32::try_from(n) else {
        // As in [`lse_log2_to_ln`], including the saturation: `Elementwise`'s
        // grid is `ceil(n / 256)` in `u32`, a silent truncation would launch
        // over a prefix, and the count is ABOVE the ceiling rather than below
        // a floor.
        return Err(Refusal::Wide {
            what: "logit elements",
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    Ok(elementwise(elems))
}

/// `attn::logit_softcap_bf16` — gemma's final logit cap, in place.
///
/// # Geometry
///
/// `attn_softcap.cu`, quoted whole by `softcap.cuh`'s header: `const auto
/// blocks = (n + 255) / 256; logit_softcap_bf16_kernel<<<blocks, 256, 0,
/// stream>>>(x, 1.f / cap, cap, n);`. That is [`elementwise`], which is
/// `LaunchRule::Elementwise`, which is what the deleted row said — three
/// spellings of one grid and they agree.
///
/// # The pairing this line is responsible for
///
/// `T` is inferred from `x`, and the symbol is the literal beside it. Nothing
/// in Rust ties the two together; the header says so at length. This is one
/// of the two places in the family where that pairing is written, the other
/// being [`logit_softcap_f16`] directly below, and they are adjacent so that
/// a reader checking one checks both.
///
/// # Safety
///
/// `x` must address `n` live, writable `bf16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_bf16(
    x: *mut bf16,
    cap: f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_bf16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// **This program has no contract and no caller**, and that is the state
/// `families/attn.rs` left the row in: *"the row the ahead-of-time build
/// never had"*, a second instantiation of a template that was already there.
/// It exists because the device row exists and truth two is a `fn` — a row
/// that can be fired needs a program, whether or not a trace says it yet.
///
/// Everything else is [`logit_softcap_bf16`]'s, including the geometry: the
/// element count is elements and not bytes, so the same `(n + 255) / 256`
/// covers a buffer of the same length in either format.
///
/// # Safety
///
/// `x` must address `n` live, writable `f16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_f16(x: *mut f16, cap: f32, n: usize, stream: *mut c_void) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_f16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
/// nope and rope halves.
///
/// # The extent is computed HERE, and that is the point of the crossing
///
/// `total` is `tokens * heads * (nope + rope)` — the INPUT's element count,
/// which is what [`kimi_mla`]'s kernel guards on and what the archive
/// launcher passed. The row this replaces stated the same thing
/// (`total <- InElements(0)`) and then had its grid sized by
/// `LaunchRule::Elementwise` from the first OUTPUT's width, which for a
/// kernel whose job is to make two narrower tensors out of one wider one is
/// short by exactly the split ratio. Measured at 6 rows of 8 heads, nope 128,
/// rope 64: **4,082 of 12,544 bytes of `q_nope` and 2,041 of 6,400 of `q_pe`
/// still held the poison fill.**
///
/// The parameters below are the four EXTENTS and not the product, so no
/// caller can hand a `total` that disagrees with the shape it also hands. A
/// launcher that takes a count someone else computed can be given a wrong
/// one; this one cannot.
///
/// # The product is formed in `i64` because the kernel's is not
///
/// `total` is an `int` on the device and `kimi_mla.cuh:84` casts to
/// `long long` only for the destination INDEX. At kimi_k3's head count a long
/// prefill reaches 2^31 elements before it reaches anything else, so the
/// product is formed wide here and the row count is refused when it will not
/// fit. That refusal is hoisted above the single launch, as every refusal in
/// this file is.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_q_b_bf16(
    q_b: *const bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    tokens: i32,
    heads: i32,
    nope: i32,
    rope: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if nope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_nope_head_dim" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    let total = i64::from(tokens) * width;
    if total > i64::from(i32::MAX) {
        return Fired::Declined(Refusal::Wide {
            what: "rows",
            at: tokens,
            max: i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX),
        });
    }
    let total = total as i32;
    unsafe {
        kimi_mla::raw::split_q_b(
            "attn::kimi_split_q_b_bf16",
            elementwise(total.unsigned_abs()),
            q_b,
            q_nope,
            q_pe,
            total,
            heads,
            nope,
            rope,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::kimi_split_kv_a_norm_bf16` — split `kv_a`, RMS-normalise the latent
/// half, copy the rope half unnormalised.
///
/// `<<<tokens, 256>>>` with `(256 / 32) * 4` bytes of shared memory — one
/// block per row, which is what [`rms`] says and why the row's `elem` states
/// `256` rather than letting the template default supply it. The block width
/// sizes `__shared__ float buf[BLOCK_DIM]` and fixes the halving tree; the
/// launch width and the instantiation width are one number and this file is
/// where they meet.
///
/// # `src_row_stride` is checked, which the row world could not do
///
/// The source row is `kv_lora + rope` wide unless a caller hands a wider
/// buffer — the fused prepare does. A stride NARROWER than the two halves it
/// is being asked to read out of is a read past the row into the next one,
/// which produces a plausible normalised vector built from the wrong token.
/// It is refused here because a `fn` can see all three numbers at once; a row
/// carrying three independent `Source`s cannot compare them.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_kv_a_norm_bf16(
    kv_a: *const bf16,
    norm_weight: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    tokens: i32,
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if kv_lora <= 0 {
        return Fired::Declined(Refusal::Empty { what: "kv_lora_rank" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    if src_row_stride < kv_lora + rope {
        return Fired::Declined(Refusal::Narrow { what: "src_row_stride", at: src_row_stride });
    }
    unsafe {
        kimi_mla::raw::split_kv_a_norm(
            "attn::kimi_split_kv_a_norm_bf16",
            rms(tokens.unsigned_abs()),
            kv_a,
            norm_weight,
            kv_c,
            k_pe,
            kv_lora,
            rope,
            src_row_stride,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::combine_attn_outputs_bf16` — merge two attention halves and their
/// log-sum-exps into one.
///
/// `dsv4_compress.cu:57-96`. How a sparse branch and a dense branch are
/// recombined: deepseek_v4 attends over a compressed cache and over the
/// selected fine blocks, and this is the single launch that folds the two
/// results into the one the rest of the layer reads.
///
/// The geometry is [`combine_attn`] and the paragraph there is the reason
/// this launcher was a `Walk` rather than a row for as long as the row world
/// lasted.
///
/// # The empty case is on the device, and that is not a gap
///
/// `lse2 == -inf` means the second half had no entries, and the kernel
/// passes `o1` through unchanged. That is a device-side branch on a
/// device-side value, which is §5.1's rule: no host can read it without a
/// synchronise, and a fire is a straight line. The three refusals below are
/// the ones a host CAN see, and every one of them is resolved before the
/// launch.
///
/// # `num_heads` and `head_dim` are the statement's, not the fire's
///
/// They are `Param(0)` and `Param(1)` on the row this replaces — not widths
/// and not `Cx::head_dim()`. The merged tensor is `[N, num_heads, head_dim]`,
/// so `out_width(0)` is their PRODUCT and no division of it recovers two
/// numbers. The statement carries both because it has to.
///
/// # Safety
///
/// Every pointer must address the extents these three numbers describe, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn combine_attn_outputs_bf16(
    o1: *const bf16,
    lse1: *const f32,
    o2: *const bf16,
    lse2: *const f32,
    o_out: *mut bf16,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `dsv4_compress.cu:64` — `if (N <= 0) return;`, and it is `grid.x`.
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    // Neither of these was tested by the C++, which formed `grid.y` from
    // `num_heads` and would have launched a zero-extent grid. A `fn` can see
    // all three numbers at once and a launch of no blocks that reports
    // success is the thing `Fired` exists to distinguish.
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        dsv4_compress::raw::combine_attn_outputs(
            "attn::combine_attn_outputs_dev",
            combine_attn(n.unsigned_abs(), num_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o1,
            lse1,
            o2,
            lse2,
            o_out,
            lse_out,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

// Truth three, declared: what a trace may say.
//
// Thirteen contracts, carrying thirteen of `table/attn.rs`' forty-one rows
// minus everything that described a launcher. `softcap`
// declares two device rows and only one of them is a thing a trace may say —
// and [`qkv_fused`] declares ELEVEN and states one, which is the same fact at
// eleven times the scale: ten of its rows are a launcher's arms.
//
// [`dsv4_compress`] declares TEN and states three, and its other seven are
// not arms — three of them are fired by
// `driver-cuda/src/fire/dsv4_compress.rs` through `hand::fire`, which is a
// host program in another crate rather than a contract here. **A unit row
// does not need a contract to have a caller.** The fourth of those was
// `combine_attn_outputs_dev`, and it now has both: a contract here and a
// host `fn` above, which is what a row becomes when nothing outside the
// statement is needed to fire it.
// `Contract::DEFAULT` supplies the other fields of each; `needs`, `lacks`,
// `depth_prefix_plan`, `publishes_aux` and `lowered_as` are stated by
// nothing here, as they were by nothing in the rows these replace.
//
// The twenty-seven rows these passes did not take keep their contracts in
// `table/attn.rs`. `table::TABLES` concatenates both lists, so
// `model-compiler` reads one vocabulary and cannot tell which world serves a
// symbol — which is the property the split is allowed to have.
// ---------------------------------------------------------------------------

contract! {
    /// FlashInfer publishes its LSE in log2 and the sink correction works in
    /// ln. A unit conversion, stated so a reader never has to guess which
    /// base an LSE is in.
    ///
    /// In place on the value it names: the statement's result and its first
    /// operand are the same buffer, which is what `in_place` says.
    /// `model-compiler/src/dsl.rs` states it through
    /// `dsl::cuda::lse_log2_to_ln`.
    LSE_LOG2_TO_LN = "attn::lse_log2_to_ln" as lse_log2_to_ln {
        in_place: &[(0, 0)],
    }

    /// Rescales the attention output IN PLACE against the per-head sink
    /// logit; the LSE is read-only.
    ///
    /// gpt-oss's sink layers state it right after the dispatch, so
    /// `attn.out` observes the RESCALED result. The LSE is the dispatch's
    /// second RESULT, which only a sink layer declares — so it is operand 1
    /// and traced, not a scratch the executor remembers handing the
    /// dispatch.
    ATTENTION_SINK_RESCALE = "attn::attention_sink_rescale_bf16" as attention_sink_rescale {
        in_place: &[(0, 0)],
    }

    /// K3's residual-block blend: a prefix and `B` candidate blocks, scored
    /// by an RMS-normalised projection and combined.
    ATTN_RES_BLEND = "attn::attn_res_blend_bf16" as attn_res_blend

    /// The pad half of what `head_dim_padded` COSTS.
    ///
    /// Stating the pair turns `if (c.head_dim_padded)` in the model body
    /// into a fact the trace carries. Row-shaped — each token's heads pad
    /// independently.
    PAD_HEAD_DIM = "attn::pad_head_dim_bf16" as pad_head_dim

    /// The strip half. See [`PAD_HEAD_DIM`].
    STRIP_HEAD_DIM = "attn::strip_head_dim_bf16" as strip_head_dim

    /// Gemma's final logit cap — `cap * tanh(x / cap)` over the logits,
    /// where they lie.
    ///
    /// One buffer and no destination, which `Buffers::assign` was already
    /// relying on while the row said nothing: it widens a seam's pin over an
    /// alias set, the set had one member, and the widening reached nothing.
    /// The head wrote the logits into the arena, the cap ran over
    /// `ws.logits`, and the sampler read an uncapped previous fire. `in_place`
    /// is that alias, stated.
    ///
    /// A dispatch parameter and not a launch parameter: gemma-2's
    /// `attn_logit_softcapping` is a different fact and is not this. Only
    /// Gemma-2, Gemma-3 and Gemma-3n state a final cap, so the bind's
    /// [`Cx::final_logit_softcap`] refuses on every other deployment — which
    /// is a per-fire refusal and NOT a `none:` arm, because a `none:` arm
    /// refuses at model load and would take Gemma with it.
    ///
    /// `model-compiler/src/dsl.rs:6379` states it through
    /// `dsl::cuda::logit_softcap`, gated on `final_logit_softcapping` being
    /// present in the config — so its presence is a trace-time decision and
    /// the trace either carries the statement or does not.
    ///
    /// [`Cx::final_logit_softcap`]: crate::x::Cx::final_logit_softcap
    LOGIT_SOFTCAP = "attn::logit_softcap_bf16" as logit_softcap {
        in_place: &[(0, 0)],
    }

    /// kimi_k3's fused query projection, split into the halves attention
    /// wants: `[tokens, heads, nope]` and `[tokens, heads, rope]` out of
    /// `[tokens, heads, nope + rope]`.
    ///
    /// `model-compiler/src/dsl.rs:5483` states it; `crates/model`'s
    /// `kimi_k3/forward/mod.rs:156` is the caller. Two results and no alias,
    /// so no `in_place`: the source stays live, because the rope half is
    /// rotated afterwards and the nope half is not.
    ///
    /// Row-shaped — each token's heads split independently — which is why
    /// the deleted row was not `whole` and this contract states nothing
    /// about windows.
    KIMI_SPLIT_Q_B = "attn::kimi_split_q_b_bf16" as kimi_split_q_b

    /// The key/value half of the same split, with an RMS norm fused into it.
    ///
    /// `model-compiler/src/dsl.rs:5452` states it; `kimi_k3/forward/mod.rs:150`
    /// is the caller, one line before [`KIMI_SPLIT_Q_B`]'s.
    ///
    /// One kernel and not a split followed by a norm, because the latent
    /// half is read twice by the reduction and would otherwise make a round
    /// trip through global memory in between. The rope half is copied
    /// **unnormalised**, which is a property of the statement and not of the
    /// launch: normalising a value that is about to be rotated changes the
    /// angle.
    KIMI_SPLIT_KV_A_NORM = "attn::kimi_split_kv_a_norm_bf16" as kimi_split_kv_a_norm

    /// deepseek_v4's compressed-cache gather — one entry per boundary token,
    /// by a per-dimension softmax over the gate scores of the window ending
    /// there.
    ///
    /// `model-compiler/src/dsl.rs:4684` states it as
    /// `dsv4_compress_gather_paged`, and `crates/model/src/deepseek_v4` is
    /// its only caller. `whole` on the row it replaces; that is not a
    /// geometry fact and does not survive into a contract.
    DSV4_COMPRESS_GATHER_PAGED = "attn::dsv4_compress_gather_paged_bf16"
        as dsv4_compress_gather_paged {
        sink: Some("kv.compressed"),
    }

    /// The commit half — those entries into the compressed cache, each at its
    /// own boundary token's slot.
    ///
    /// `dsl.rs:4702`. **It produces no value at all** — `record` is called
    /// with `None` for the result — so its whole effect is a store the
    /// contract vocabulary cannot name, and `whole: true` with no `sink` is
    /// what the row said about that and what this says.
    DSV4_STORE_COMP_ENTRIES = "attn::dsv4_store_comp_entries_bf16"
        as dsv4_store_comp_entries {
        whole: true,
    }

    /// The merge that puts deepseek_v4's two attention branches back
    /// together — two outputs and two log-sum-exps in, one of each out.
    ///
    /// `model-compiler/src/dsl.rs:4749` states it and
    /// `model/src/deepseek_v4/forward/mod.rs:146` is its only caller: the
    /// compressed branch and the fine-block branch each attend over their own
    /// cache, and neither result means anything until the LSEs have weighted
    /// them against each other.
    ///
    /// TWO RESULTS, and both are traced. `lse_out` is `Out(1)` and not a
    /// scratch the executor remembers handing the launcher — the caller
    /// destructures `(o, lse)` and the LSE is read again by the layer above.
    ///
    /// `num_heads` and `head_dim` are `Param`s. See the `fn`: the merged
    /// tensor's width is their product and no division of a width recovers
    /// two numbers.
    COMBINE_ATTN_OUTPUTS = "attn::combine_attn_outputs_bf16" as combine_attn_outputs

    /// The fused QKV prefill epilogue — six statements in one launch, and
    /// the only value that survives is q.
    ///
    /// q norm, q rope, k norm, k rope, v norm and the paged KV write. The
    /// `sink` is what says the rest of it lands in the cache: everything but
    /// q is written to `k_pages`/`v_pages` and observed by a later dispatch
    /// rather than by a result.
    ///
    /// `model-compiler/src/dsl.rs` states it as `qkv_packed_post`. Its
    /// decode sibling, `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, is NOT
    /// here: its host program is `driver-cuda/src/fire/qkv_fused.rs` and it
    /// is served through `bind::service`, so its row stays where
    /// `attn::split_qkv_bf16_devwin`'s does and for the same reason.
    QKV_PACKED_POST = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" as qkv_packed_post {
        sink: Some("kv.pages"),
    }

    /// Attention over the latent cache — DeepSeek/Kimi MLA, the row that
    /// retires `attention_mla.cu` and with it the last nvcc-compiled
    /// `<<<>>>` in the workspace.
    ///
    /// # ONE SYMBOL, TWO HOST PROGRAMS, AND THAT IS WHY IT IS NOT BOUND
    ///
    /// `attention_mla.cu:150` chooses between them at run time on the
    /// compute capability, because FA2 MLA **produces zero output on
    /// sm_100** — a wrong answer and not a failure. Below sm_100 it is
    /// FlashInfer's cooperative `BatchMLAPagedAttentionKernel`, which is
    /// [`mla_fa2`] in this file, unit and all; on sm_100 it is the scalar
    /// and mma pair in [`mla_naive`], whose host program is
    /// `driver-cuda/src/fire/mla_naive.rs`, one crate up.
    ///
    /// A row loses its ahead-of-time shim entry whole or not at all, so both
    /// arms had to be Rust before either could be. **They both now are.**
    /// What is left is the seam between them, and the seam is `Cx` — see the
    /// `none:` arm below, which names the four queries and nothing else.
    ///
    /// # `needs: Prepare::MlaPlan`
    ///
    /// Its own kind of plan, built from the latent geometry (`kv_lora_rank`,
    /// `qk_rope_head_dim`) that no other prepare has a field for and cached
    /// in an `MlaPlanCache` rather than in the shared attention workspace.
    /// `crate::plan::mla` is already the Rust half of it.
    ///
    /// # `lacks: &[Cap::Scores]`
    ///
    /// There is no capture variant of this dispatch, so a program whose
    /// `attn.out` seam wants the score matrix cannot be served over the rows
    /// it covers. It publishes an LSE, which is a different thing and not
    /// what the capability names — and, per [`mla_fa2`]'s `pack`, an LSE in
    /// **natural log**, so [`LSE_LOG2_TO_LN`] must never be applied to it.
    ATTENTION_MLA = "attn::dispatch_attention_mla_bf16" as attention_mla {
        needs: Prepare::MlaPlan,
        lacks: &[Cap::Scores],
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Six binds and one `none:` arm. Every operand `table/attn.rs` sourced for
// those six rows is a `Cx` query that exists — four of them because they
// always did, and `softcap`'s cap because `Facts::final_logit_softcap` was
// asked for and landed. That is not true of `attention_mla`, whose four
// unsourced operands are four queries that do not exist, nor of the roots
// that remain: `page_compact` and the devwin split want buffers no `Source`
// ever spelled, and the first of those is a floor gap while the second is
// not.
//
// `//` and never `///` inside this invocation: the arms are array elements
// and an attribute cannot precede one.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    LSE_LOG2_TO_LN => { cx, stream => {
        // `Source::OutElements(0)` is `rows * out_width(0)`, and both halves
        // are `Cx` queries. The result and the operand are the same buffer,
        // so reading the extent off the RESULT is reading it off the thing
        // that will be written.
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Empty { what: "lse elements" });
        };
        unsafe { lse_log2_to_ln(cx.arg_out(0)?.cast::<f32>(), n, stream) }.ok()
    }},

    ATTENTION_SINK_RESCALE => { cx, stream => {
        // The deleted row's six sources, in order: `Out(0)`, `In(1)`,
        // `Weight(0)`, `Rows`, `Ctx("num_q_heads")`, `Ctx("head_dim")`.
        //
        // `In(1)` and not `In(0)`: the LSE is the dispatch's SECOND result,
        // and the statement that declares it is the only one that can.
        unsafe {
            attention_sink_rescale_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.num_q_heads()?,
                cx.head_dim()?,
                stream,
            )
        }
        .ok()
    }},

    ATTN_RES_BLEND => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0..3)`, `Out(0)`,
        // `Rows`, `Div(Width(In(1)), Width(Out(0)))`, `OutWidth(0)`, `Rows`
        // again for `block_rows`, and `Ctx("eps")`.
        //
        // `Cx::rms_eps` IS `Ctx("eps")`: `bind/facts.rs:284` reads
        // `self.ctx.eps`, the same field the row named. The method's name
        // says which fire-wide epsilon it is, not which kernel may read it.
        //
        // `norm_weight` and `proj_weight` are `In(2)` and `In(3)` and not
        // weights: K3 states them as operands, so they are the statement's
        // and the binder resolves them like any other.
        let h = cx.out_width(0)?;
        let b = cx.in_width(1)? / h;
        unsafe {
            attn_res_blend_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                b,
                h,
                cx.rows().count,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    PAD_HEAD_DIM => { cx, stream => {
        // THE PACKED SIDE IS THE INPUT ON THE WAY IN. The head count is the
        // packed side's width over `head_dim`, and the padded width is the
        // other side over that count — `PACKED_HEADS_IN` and the `Div` the
        // deleted row wrapped around it, as two divisions of numbers this
        // statement already carries.
        //
        // The head count is NOT `cx.num_kv_heads()`. See `per_head`'s doc:
        // the rule read one and the kernels address with the other, and at
        // 12 heads of 64 against 6 KV heads the difference is 6,100 of
        // 12,544 bytes.
        let head_dim = cx.head_dim()?;
        let packed_width = cx.in_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "in_width(0)", at: packed_width });
        }
        unsafe {
            pad_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.out_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    STRIP_HEAD_DIM => { cx, stream => {
        // THE PACKED SIDE IS THE OUTPUT ON THE WAY OUT, which is the whole
        // difference from the arm above and the reason the deleted rows kept
        // two constants rather than one expression written twice: a copy
        // that drifted would count heads on the PADDED side, where the
        // divisor is `head_dim_padded`, so the count comes out short and the
        // launch covers a prefix of the heads.
        let head_dim = cx.head_dim()?;
        let packed_width = cx.out_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "out_width(0)", at: packed_width });
        }
        unsafe {
            strip_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.in_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    LOGIT_SOFTCAP => { cx, stream => {
        // The deleted row's three sources, in order: `Out(0)`,
        // `CtxNonZero("final_logit_softcap")`, `OutElements(0)`. The stream
        // was a fourth and is a parameter now.
        //
        // `final_logit_softcap()` IS the `NonZero` half: the driver's impl
        // returns `None` for a cap of zero, so a deployment that states none
        // declines here with `nothing states a logit soft cap` rather than
        // scaling every logit to nothing. The `fn` re-checks it, because a
        // `fire/` caller can reach the `fn` without passing through a `Cx`.
        let cap = cx.final_logit_softcap()?;
        // In place on the value it names, so the extent is read off the
        // RESULT — the same buffer the first operand is.
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Narrow { what: "logit elements", at: elems });
        };
        unsafe { logit_softcap_bf16(cx.arg_out(0)?.cast::<bf16>(), cap, n, stream) }.ok()
    }},

    KIMI_SPLIT_Q_B => { cx, stream => {
        // The deleted row's seven sources, in order: `In(0)`, `Out(0)`,
        // `Out(1)`, `Rows`, `Param(0)`, `Param(1)`, `Param(2)`. The stream
        // was an eighth and is a parameter now.
        //
        // THE EXTENT IS NOT AMONG THEM, and that is the whole of it. The
        // row's four numbers describe the LAUNCHER, which formed the device
        // kernel's `total` from them; the JIT has no launcher and so formed
        // its grid from `LaunchRule::Elementwise` instead, which is `rows *
        // out_width(0)` and is short by `rope / (nope + rope)`. The `fn`
        // takes the same four numbers the row states and forms the product
        // itself, which is the arrangement the launcher had.
        //
        // `heads`, `nope` and `rope` are `Param`s and not widths: `q_b` is
        // ONE operand of width `heads * (nope + rope)`, and no division of
        // its width recovers three numbers. The statement carries them.
        //
        // `unwrap_or(0)` on the narrowing is `x::moe`'s and `x::quant`'s
        // idiom and is safe for the same reason: a `u32` above `i32::MAX`
        // becomes zero, and the `fn` refuses zero with the extent's own
        // name. A silent narrowing would be the alternative.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let heads = param(0)?;
        let nope = param(1)?;
        let rope = param(2)?;
        unsafe {
            kimi_split_q_b_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                heads,
                nope,
                rope,
                stream,
            )
        }
        .ok()
    }},

    KIMI_SPLIT_KV_A_NORM => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0)`, `Weight(0)`,
        // `Out(0)`, `Out(1)`, `Rows`, `OutWidth(0)`, `OutWidth(1)`,
        // `Ctx("eps")`, `InWidth(0)`. The stream was a tenth.
        //
        // Every extent here IS a width of the statement's own tensors, which
        // is why this half never had `kimi_split_q_b`'s problem: the two
        // results' widths ARE `kv_lora_rank` and `qk_rope_head_dim`, and the
        // operand's width IS the source stride. `Rows` is the grid.
        //
        // `norm_weight` is `Weight(0)` and not `In(1)`: kimi states the
        // latent norm's scale as a weight, unlike `attn_res_blend` one
        // contract up, which states both of its as operands. The two are
        // different statements and each is read as it was written.
        unsafe {
            kimi_split_kv_a_norm_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.out_width(1)?,
                cx.in_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // NOT BINDS, AND THE MEASUREMENT IS IN THE STATEMENT RATHER THAN IN `Cx`.
    //
    // Both host programs are written and both are twelve lines: the geometry
    // is `route_rows(num_entries, head_dim)` and every refusal is hoisted.
    // What is missing is not a launch and not a query — it is that **the
    // trace does not name most of what these kernels read.**
    //
    // `dsl.rs:4684` records `dsv4_compress_gather_paged` with
    // `vec![boundary_pos.id]` for its inputs and `vec![]` for its params, and
    // `:4702` records `dsv4_store_comp_entries` with `vec![entries.id,
    // boundary_pos.id]` and `vec![]`. The kernels take twelve and eight
    // operands. So of the gather's five buffers a statement names ONE, and of
    // its three integers it names NONE.
    //
    // **This is the same defect `families::attn` has, one level up, and it is
    // worth naming because that file looks like it disagrees.**
    // `DSV4_COMPRESS_SIGS[4]` annotates the gather `state_kv <- Source::In(0)`
    // through `boundary_req <- Source::In(4)`, `ratio <- Source::Param(0)`,
    // `coff <- Source::Param(1)`. Those `Source`s describe a statement that
    // does not exist: `In(1)`..`In(4)` and both `Param`s have nothing behind
    // them, because `record` was handed one input and no parameters. The
    // TABLE row states no `Source` at all and is right to — and the two rows
    // have disagreed for as long as both have existed, with nothing
    // comparing them, because a device row's sources are read by nobody.
    // `driver-cuda/tests/executor_bind.rs`' UNARMED list holds both symbols,
    // which is the only place the truth was recorded.
    //
    // WHAT A BIND WOULD NEED, and it is a `dsl` change before it is a `Cx`
    // one: `state_kv` and `state_score` are deepseek_v4's compression state,
    // a slab no `StateStore` names; `ape` is a weight; `boundary_req` is
    // `dsv4_boundary_meta_*`'s SECOND output, which the statement drops on
    // the floor at `dsl.rs:4676`; `ratio` and `coff` are model config that
    // `record`'s empty parameter list refuses to carry. Every one of those is
    // a fact the STATEMENT would have to state before a query could answer
    // it, which is why neither of these is *"a fact `Cx` does not carry"* —
    // it is a statement that names one of its own operands.
    //
    // Crossing them UNBOUND costs nothing measurable and is the same trade
    // `ATTENTION_MLA` took: both symbols are on `executor_bind.rs`' UNARMED
    // list, both are in `device::JIT_DISPATCHED` so no shim entry was ever
    // emitted for either, and both table rows were unsourced so
    // `abi::emit_rust_dispatch` skipped them WHOLE and never generated an arm.
    // A row that no path fires, while holding a row table open, is worth less
    // than the contract that replaces it.
    DSV4_COMPRESS_GATHER_PAGED => { none:
        "deepseek_v4's compression state is not a value this trace names: \
         `dsl::cuda::dsv4_compress_gather_paged` records ONE input \
         (`boundary_pos`) and NO parameters for a kernel that reads five \
         buffers and three integers, so `state_kv`, `state_score`, `ape`, \
         `boundary_req`, `ratio` and `coff` have no operand to come from — \
         the host program is written, in `x::attn::dsv4_compress::\
         dsv4_compress_gather_paged_bf16`, and what it is waiting for is a \
         statement rather than a query"
    },

    DSV4_STORE_COMP_ENTRIES => { none:
        "the commit half is blocked by the same statement as the gather: \
         `dsl::cuda::dsv4_store_comp_entries` names `entries` and \
         `boundary_pos`, and the kernel also reads `boundary_req` — \
         `dsv4_boundary_meta_*`'s second output, which the trace discards — \
         and needs `head_dim` and `page_size` besides; the host program is \
         `x::attn::dsv4_compress::dsv4_store_comp_entries_bf16`"
    },

    COMBINE_ATTN_OUTPUTS => { cx, stream => {
        // The deleted row's ten sources, in order: `In(0)`, `In(1)`, `In(2)`,
        // `In(3)`, `Out(0)`, `Out(1)`, `Rows`, `Param(0)`, `Param(1)`. The
        // stream was a tenth and is a parameter now.
        //
        // EVERY ONE OF THEM IS SOURCED FROM THE STATEMENT, which is what
        // makes this the cheapest bind in the family: no layer, no
        // workspace, no plan, no device attribute. It was a `Walk` for one
        // reason and one only — `execution.rs`' `Control::Supplies`, the
        // BLOCK width — and a `fn` supplies its own geometry, so the
        // classification had nothing left to describe.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let num_heads = param(0)?;
        let head_dim = param(1)?;
        unsafe {
            combine_attn_outputs_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                num_heads,
                head_dim,
                stream,
            )
        }
        .ok()
    }},

    QKV_PACKED_POST => { cx, stream => {
        // The deleted row's twenty-one sources, in order: `In(0)`, `Out(0)`,
        // `KvLayerField(k_pages)`, `KvLayerField(v_pages)`, `Weight(0)`,
        // `Weight(1)`, `Positions`, `Attn(kv_page_indices_d)`,
        // `Attn(kv_page_indptr_d)`, `Attn(kv_last_page_lens_d)`,
        // `Attn(row_valid_d)`, `Rows`, `Div(Width(Out(0)),
        // KvLayerField(head_dim))`, `KvLayerField(num_kv_heads)`,
        // `KvLayerField(head_dim)`, `KvLayerField(page_size)`,
        // `KvLayerField(hnd_layout)`, `CtxByLayer(theta)`, `Ctx(eps)`, and
        // the stream.
        //
        // SIX OF THEM ARE ONE QUERY. `Cx::kv_layer` returns the same five
        // fields `Source::KvLayerField` spelled one string at a time, plus
        // `hnd` — so a family that wanted six lookups pays for one, which is
        // the argument `Cx::gdn` makes for `ssm`'s eleven.
        //
        // FOUR MORE ARE ANOTHER. `Cx::plan` carries the three CSR arrays and
        // `row_valid` together, and they ARE together: the mask indexes the
        // same rows the CSR describes, and a fire that published one
        // published all four.
        //
        // `num_q_heads` off the RESULT and not the operand, exactly as the
        // row had it: `packed` is `[N, q + 2·kv]` and cannot say where the
        // cut falls; `q_out` is `[N, q]` and can. Dividing by the LAYER's
        // head_dim rather than the context's is the row's choice too — the
        // pages decide the width, because the pages are what is written.
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        if layer.head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let num_q_heads = cx.out_width(0)? / layer.head_dim;
        unsafe {
            qkv_fused::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                plan.row_valid,
                cx.rows().count,
                num_q_heads,
                layer.num_kv_heads,
                layer.head_dim,
                layer.page_size,
                layer.hnd,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // NOT A BIND, AND THE MISSING THING IS VOCABULARY RATHER THAN A HOST
    // PROGRAM. Both arms of this symbol are now Rust and both are reachable
    // from a bind body's crate or one above it:
    //
    //   FA2, everything below sm_100 — `mla_fa2` in THIS FILE. Unit, six
    //   instantiations, `MlaParams` packer, `arm_for`, `grid`, and a
    //   cooperative `fire_ex`. Nothing about it is pending.
    //
    //   NAIVE, sm_100 only — `driver-cuda/src/fire/mla_naive.rs`, a
    //   driver-side host program in the crate ABOVE this one, the same
    //   shape as `fire/xqa.rs`. `attention_mla.cu:150` picks it on
    //   `cudaDevAttrComputeCapabilityMajor >= 10`, because FA2 MLA writes
    //   ZERO OUTPUT on sm_100 — a wrong answer, not a fault.
    //
    // THE FOUR `Cx` QUERIES THIS ARM ASKED FOR ALL LANDED, and two of them
    // answer. `Cx::attn_workspace` and `Cx::sm_scale` are implemented from
    // `AttnCtx::workspace` and `AttnCtx::sm_scale` — and they were `x/xqa.rs`'
    // ask as much as this one, two families naming the same two lines because
    // both rows carried `Source::Attn(..)`.
    //
    // `Cx::mla_layer` and `Cx::mla_plan` exist and REFUSE, and the reason is
    // one absence rather than two gaps. MEASURED:
    //
    //   * `fire/launch.rs:1521`'s `kv_pools_for` matches on `dep.kv` and
    //     returns `PIE_STATUS_UNSUPPORTED` for `KvStyle::Mla` before any pool
    //     is built. Its own comment is why the match is there: a missing MLA
    //     branch was *"not a `todo!()`, not a refusal, just an absence, which
    //     is how the MLA lineage loaded, reported itself healthy and would
    //     have died at its first fire."*
    //   * `pools::mla_cache::MlaCachePool` has ZERO callers in `driver-cuda`.
    //   * `serve/load.rs:397` refuses every MLA checkpoint AT MODEL LOAD:
    //     *"this checkpoint attends through a latent ckv/kpe pair, which this
    //     driver does not build — `pools::mla_cache` is ported and has no
    //     forward path to serve."*
    //
    // So `AttnCtx` carrying an `mla_layers: Vec<MlaCacheLayerView>` beside its
    // `layers` is three edits and an afternoon — and there would be NOTHING
    // TO FILL IT WITH. `crate::plan::mla` is in the same position one level
    // up: it builds a plan against a cache that is never materialised, which
    // is why `mla_plan` refuses too rather than being the easier of the two.
    // The task is provisioning the MLA cache — a `KvState` shape, a growth
    // path, capture-stable bases, a `views()` equivalent — and that is the
    // MLA LINEAGE, not this seam.
    //
    // WHICH IS WHY THIS ARM COSTS NOTHING TODAY, and the check is one line:
    // `load.rs:397` refuses the checkpoint before any trace exists to name
    // this symbol. Same shape as `x/xqa.rs`' decode arm, which refuses
    // nothing because every deployment states `xqa_decode: false`.
    //
    // The FIFTH fact is the arm predicate and it is NOT a `Cx` query: the
    // compute capability is a property of the device, not of the fire, and no
    // other fn-world body has needed one. It belongs beside `num_sm` in
    // whatever the runtime grows for device attributes. It is not on the
    // critical path — with no MLA cache there is no fire to choose an arm for
    // — but it is the one of the three that is genuinely small, and stating
    // it here keeps it from being rediscovered.
    //
    // ONE THING TO SETTLE WHEN THE CACHE LANDS: `Cx::attn_workspace` hands
    // back `AttnCtx::workspace`, the DECODE plan's, and deliberately does not
    // guess between that and `prefill_workspace`. `AttnCtx::prefill_workspace`
    // states the rule it is protecting — *"a launcher must take the workspace
    // its own plan was raised in"* — and MLA raises its own plan through
    // `MlaPlanCache`, so which of the two it must take is a question for
    // whoever wires `mla_plan`, and the answer may be a second query rather
    // than a smarter first one.
    ATTENTION_MLA => { none: "attention over the latent cache cannot be bound         because this driver does not build one: `fire/launch.rs`' `kv_pools_for`         refuses `KvStyle::Mla` and `serve/load.rs` refuses the checkpoint at         model load, so `Cx::mla_layer` and `Cx::mla_plan` have nothing to         answer with — both host programs are written, in `x::attn::mla_fa2` and         `driver-cuda/src/fire/mla_naive.rs`, and choosing between them needs a         device compute capability besides" },
}
