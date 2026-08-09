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
//!
//! Six of forty-one. Thirty-five rows remain in `table/attn.rs`, and the last
//! three roots moved no row at all: [`split_packed`]'s and [`dsa_indexer`]'s
//! device text crossed with their host programs already where they belong,
//! and [`pack_dense_mask`]'s two `table::driver_internal` rows were deleted
//! before this sweep began.
//!
//! **Unit-only is a real arrangement and three roots now use it**, which is
//! enough to state the rule: a root whose host programs are already Rust and
//! already outside a `bind!` crosses as a `unit!` and nothing else. The device
//! text belongs where the family is; the program belongs where it already
//! runs. `x/driver_internal.rs` says the first half — *"the rows stay where
//! the device text is"* — and these three are the second.
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
//! The first `attn` caller of [`Cx::plan`] will be `kv_paged` or
//! `attention_naive_paged`, both of which state `qo_indptr`,
//! `kv_page_indices`, `kv_page_indptr` and `kv_last_page_lens` as operands.
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
/// It is NOT taken in this pass. `is_pointer` is read by more than
/// `Args::bind`, `Ty::StructuredMasks` reaches no surviving row, and a
/// vocabulary change made for a kernel nothing launches is a change nothing
/// checks. Ask when a caller exists.
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

/// The units `attn` compiles in fn-world.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above gives. `families::ALL` reads this **beside**
/// `families::attn::UNITS`, which still holds the twenty roots this pass did
/// not take. A root appears in exactly one of the two lists: a second `unit!`
/// naming the same text would be a second compilation of it under a second
/// unit name, and `unit_of` would answer with whichever won.
pub static UNITS: &[Unit] = &[
    attn_res::ATTN_RES,
    attn_sink::ATTN_SINK,
    dsa_indexer::DSA_INDEXER,
    head_dim_pad::HEAD_DIM_PAD,
    pack_dense_mask::PACK_DENSE_MASK,
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

// Truth three, declared: what a trace may say.
//
// Six contracts, carrying six of `table/attn.rs`' forty-one rows minus
// everything that described a launcher. Six and not seven: `softcap`
// declares two device rows and only one of them is a thing a trace may say. `Contract::DEFAULT` supplies the
// other fields of each; `needs`, `lacks`, `depth_prefix_plan`,
// `publishes_aux` and `lowered_as` are stated by nothing here, as they were
// by nothing in the rows these replace.
//
// The thirty-five rows this pass did not take keep their contracts in
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
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Six binds and no `none:` arms. Every operand `table/attn.rs` sourced for
// these six rows is a `Cx` query that exists — four of them because they
// always did, and `softcap`'s cap because `Facts::final_logit_softcap` was
// asked for and landed. That is not true of the nineteen roots that remain:
// `page_compact` and the devwin split want buffers no `Source` ever spelled,
// and the first of those is a floor gap while the second is not.
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
}
