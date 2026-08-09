//! Linear attention and state-space recurrences: GDN, KDA, mamba, and the
//! causal convolution that feeds them.
//!
//! §5 step 5 for `ssm`. `families/ssm.rs` (2 367 lines, 38 device rows) and
//! `table/ssm.rs` (616 lines, 27 rows) are deleted; the host programs move in
//! from `driver-cuda/src/fire/{causal_conv1d,gated_delta_net,kda,nemotron_h}
//! .rs` (201 + 948 + 287 + 592 lines) and the thirteen `ssm_*` wrappers in
//! `driver-cuda/src/bind/service.rs`. Five roots, five units, one file.
//!
//! # FIVE `unit!` INVOCATIONS CANNOT SHARE A SCOPE
//!
//! `unit!` emits `UNITS`, `ROWS`, `PARAMS` and `mod raw` at module scope, so
//! two invocations in one file collide on four names. Each root gets an
//! inline `pub mod` and the family re-exports the five in [`UNITS`] below.
//! `x/mlp.rs` found this with two roots and `x/layout.rs` is the five-root
//! worked example; the note is repeated per family because it is the first
//! thing a reader of a multi-root family hits. The wrapping also gives
//! `raw::` a natural qualifier, which matters more here than in `layout`:
//! `gated_delta_net::raw::recurrent_step_batched` and
//! `gated_delta_net_prep::raw::widen` would otherwise be one namespace.
//!
//! # The five units, and what each is for
//!
//! | module | root | device rows | host programs |
//! |---|---|---|---|
//! | [`causal_conv1d`] | `ssm/causal_conv1d.cuh` (426) | 4 | 2 |
//! | [`gated_delta_net`] | `ssm/gated_delta_net.cuh` (1 689) | 13 | 10 |
//! | [`gated_delta_net_prep`] | `ssm/gated_delta_net_prep.cuh` (326) | 8 | 4 |
//! | [`kda`] | `ssm/kda.cuh` (315) | 4 | 4 |
//! | [`nemotron_h`] | `ssm/nemotron_h.cuh` (632) | 9 | 7 |
//!
//! Twenty-seven host programs for twenty-seven contracts. **Fifteen of them
//! are NEW**, in the sense `layout`'s were: their rows were in
//! `device::JIT_DISPATCHED` and fired through a `LaunchRule`, so the `.cu`
//! launcher went with the file and there was nothing to port. Their geometry
//! is the rule's, transcribed below with a citation each, never invented.
//!
//! **`ssm/gated_delta_net` is a unit now and was not one before.** The old
//! family kept it out, and the reason is worth carrying because it was
//! measured twice. It was first a caution, then a MEASUREMENT: a unit for it
//! was declared and run, and NVRTC answered `gated_delta_net.cuh(73):
//! catastrophic error: could not open source file "cstdint"`, with the
//! `<cuda_bf16.h>` at line 66 having resolved — so the worry about
//! `__nv_bfloat162` and `__floats2bfloat162_rn` not shimming was wrong. The
//! include is gone: the header takes its integer names from the prelude, and
//! the same probe answered `OK` — three recurrence kernels instantiated,
//! **133 808 bytes of cubin in 1 564 ms at sm_89**. What kept the unit out
//! after that was `Unit::compile_with` refusing a unit with no
//! instantiations, and that in turn was because no MODEL TEXT states any of
//! the fourteen recurrence kernels. In fn-world a row is a declared
//! instantiation and not a trace's statement, so the thirteen below are
//! exactly the instantiations the four host programs fire, and the refusal
//! has nothing left to refuse.
//!
//! [`kda`]'s own cost was measured on the same terms and stands: **41 392
//! bytes of cubin in 140 ms on an L40S**, for a header whose other two
//! kernels are plain `__global__`s that compiling it pulls in as text either
//! way.
//!
//! # THE FLOOR GAP, and this file does not compile without it
//!
//! §5.1 says `Facts::plan()` and `Facts::slab()` *"are called by nothing"*
//! and that `attn` and `ssm` are *"where they are first exercised and
//! therefore where they are most likely to be wrong"*. This is that first
//! call, and the finding is:
//!
//! * **`plan()` is right.** `ssm` reads two of its six fields —
//!   `qo_indptr` and `requests` — which are exactly `Source::Attn
//!   ("qo_indptr_d")` and `Source::Attn("num_requests")`, the only two the 27
//!   deleted rows ever named. `requests: 0` meaning "a rectangle with nothing
//!   in it" rather than absence is the right call for a batched recurrence:
//!   every launcher here already refuses `r <= 0`, and a `None` there would
//!   have turned an empty batch into `Refusal::Unstated`, which is a
//!   different sentence about a different problem.
//! * **`slab()` is incomplete, not wrong.** It answers an ADDRESS with no
//!   ADDRESSING. Every kernel in this file that takes `state_base` also takes
//!   `slot_ids` and `slot_stride_elems` in the next two slots, because the
//!   slab is `[num_slots, ...]` and a request's own slice is
//!   `base + slot_ids[r] * stride`. A base alone cannot be used by anything.
//!   The two-variant `Slab` enum is right and the `spec.state.layer` indexing
//!   is right; what is missing is the rest of the GDN context.
//!
//! `Cx` reaches none of it. Eighteen of the 27 deleted rows source at least
//! one operand from `Source::Gdn(...)` — eleven distinct keys, all of them
//! fields of `driver-cuda/src/bind/mod.rs`'s `GdnCtx`, which `Fire` already
//! borrows as `self.gdn`. Writing `none:` arms for those eighteen would
//! convert live rows into a load-time refusal, which is the regression §5.1's
//! `TEMP-REVIEW` paragraph is about. So this file is written against the API
//! it needs, and the exact patch is:
//!
//! ```text
//! // crates/kernels-cuda-new/src/x/cx.rs
//!
//! /// A gated-delta-net or mamba layer's state addressing and head geometry.
//! ///
//! /// The eleven `Source::Gdn` keys `table/ssm.rs` named, and no others.
//! /// `slot_ids` is the addressing half of `Facts::slab` — a slab base is
//! /// unusable without it — and the two strides are `i64` because a slot
//! /// stride in elements overflows `i32` at 2^31 and lands in another
//! /// request's state rather than out of bounds.
//! #[derive(Clone, Copy, Debug)]
//! pub struct Gdn {
//!     pub k_h: i32,
//!     pub v_h: i32,
//!     pub k_d: i32,
//!     pub v_d: i32,
//!     pub conv_dim: i32,
//!     pub conv_k: i32,
//!     pub n_groups: i32,
//!     pub conv_stride_elems: i64,
//!     pub state_stride_elems: i64,
//!     pub slot_ids: *const i32,
//!     pub write_state: bool,
//! }
//!
//! // on `Facts`:
//! /// This layer's gated-delta-net context.
//! fn gdn(&self) -> Option<Gdn> { None }
//! /// The join's FOREIGN values — `Source::Aux`'s reach.
//! fn aux(&self, i: usize) -> Option<*mut c_void> { let _ = i; None }
//! /// The launch's `i`th OUT — `Source::ResultOrRegion`'s reach, which is
//! /// `arg_out(i)` for a statement that produces a result and the owning
//! /// guard's value for one that writes into a REGION of a larger one.
//! fn result(&self, i: usize) -> Option<*mut c_void> { let _ = i; None }
//! /// The statement's own weight, resolved with a `"_bias"` suffix.
//! /// Null is absence: a checkpoint that ships no conv bias is a fact
//! /// about the checkpoint and not drift, and both kernels test it.
//! fn weight_bias(&self) -> Option<*mut c_void> { None }
//!
//! // on `Cx`:
//! query!(gdn -> Gdn, "the gated-delta-net context");
//! query!(aux(i: usize) -> *mut c_void, "a joined auxiliary value");
//! query!(result(i: usize) -> *mut c_void, "the launch's result");
//! // NOT a `query!`: absence is legal, so this one answers `Option`
//! // the way `rows()` and `layer()` answer plainly.
//! pub fn weight_bias(&self) -> Option<*mut c_void> { self.facts.weight_bias() }
//! ```
//!
//! and on the driver side, in `driver-cuda/src/bind/facts.rs`, the four are
//! one field read each — `self.gdn` is already there, and `aux`, `result` and
//! `weight_bias` want pre-resolved slices on `Fire` for the reason `w_named`
//! is pre-resolved and stated in that file: *"a `Resolver` is `&mut` and
//! `Facts` is not"*. `join_aux(spec, i, ..)` and `join_out(spec, i, ..)` in
//! `bind/mod.rs:1927` and `:1969` are the resolves; doing them at the
//! dispatch site is the same move `w_named` already makes.
//!
//! `quant` met the `Source::WeightSuffix` half of this and answered it with
//! `none:` arms, correctly — no `dsl` constructor emits its dequantisers with
//! two weight operands, so its trace genuinely cannot reach them. `ssm`'s two
//! conv rows are the opposite case: `dsl::cuda::causal_conv1d` states them and
//! `execution::WALKED` walks them today. The four other suffixes `quant` and
//! `moe` want (`_scales`, `_gate_bias`, `_up_bias`) generalise `weight_bias`
//! into a `(suffix, ptr)` slice whenever a second family needs it; one method
//! is the smallest thing that closes `ssm` without inventing a seam.
//!
//! **And two `ptr_abi!` lines in `x/abi.rs`**, for the nemotron MoE pointer
//! builders, whose fourteen pointer-array parameters have no `Abi` impl
//! because no ported family has taken a `void**` yet:
//!
//! ```text
//! ptr_abi!(*const c_void, "const void* const*", BufArray, "const void**", BufArrayOut);
//! ptr_abi!(*mut c_void,   "void* const*",       BufArrayMut, "void**",    BufArrayOutMut);
//! ```
//!
//! The four `cpp()` strings are `kernels::Ty`'s own, at
//! `crates/kernels/src/lib.rs:1021-1024`, so a row and a declaration
//! describing the same parameter still produce the same typecheck line. There
//! is no way to write these two rows honestly without them: spelling a
//! `void**` as `*mut c_void` would put `Ty::BufMut` where the row said
//! `Ty::BufArrayOut` and `void*` in the typecheck TU where the kernel says
//! `void**` — a bypass with no type error anywhere, which is exactly what the
//! `i8` note in `x/abi.rs` exists to prevent.
//!
//! # `Cx::out_elements` is NOT asked for
//!
//! `Source::OutElements(i)` is `rows * out_width(i)`, and both halves are
//! already `Cx` queries. Two of the deleted rows used it and both binds below
//! compute the product on the host. A query for it would be a third spelling
//! of a number `Cx` already answers twice.
//!
//! # The audit this file inherits, unchanged
//!
//! `gated_delta_net.cu` held seventeen `<<<>>>` and **eight were
//! unreachable**. `qwen_gdn_gqa_ilp2_enabled()` (`:59`),
//! `qwen_gdn_k_last_state_enabled()` (`:61`) and `qwen_gdn_fused_step_enabled()`
//! (`:68`) were `constexpr false`; `qwen_gdn_fla_prefill_enabled()` (`:110`)
//! was `constexpr true`. So `_fused` and `_ilp2` get no row, every `KLast=true`
//! instantiation is unported, and the fused step's `shmem = (2 * K_d +
//! (fused ? 1 : 0)) * 4` loses its conditional term — which is not a
//! simplification, it is the dead branch's arithmetic not being carried.
//! `PIE_QWEN35_GDN_SMEM_STEP`, an environment variable that chose between the
//! two step arms at runtime, was deleted rather than ported; the choice is the
//! `v_d == 128 && k_d == 128` test in
//! [`recurrent_step_batched_gqa_state_bf16`] and nothing reads the variable.
//!
//! `nemotron_h.cu` held two dead `<<<>>>` of its own — `:143` behind an
//! `if constexpr (false)` and `:182` after an unconditional return — and
//! neither is ported.
//!
//! # What every kernel here shares, and the three ways to get it wrong
//!
//! * **`V_d` must not come from the launch rule.** `RecurrentScan` took
//!   `Dims::head_dim`, which the dispatch scaffolding filled from the
//!   ATTENTION head dim. A GDN layer's value width is its own and the two
//!   differ on every checkpoint that has both.
//! * **`V_h` is `kv_heads` and not `q_heads`.** A 4:1 GQA fire sized on
//!   `q_heads` would open a quarter of the blocks it needs and silently
//!   freeze three of four value heads — no error, no wrong shape, a stale
//!   state.
//! * **`slot_stride_elems` is `i64`.** An `i32` overflows at 2^31 elements
//!   and wraps into ANOTHER REQUEST's state rather than off the end of the
//!   allocation, so nothing faults and one request reads another's history.
//!
//! # `Contract`, and the fields this family does state
//!
//! `publishes_aux` on three (`nemotron_mamba_split`,
//! `nemotron_prepare_mamba_params`, `nemotron_prepare_mamba_dt_da`) and
//! `whole` on five. Everything else is [`Contract::DEFAULT`], as it was on
//! the rows these replace.

use crate::unit::Unit;
use crate::x::abi::{MaybeConst, bf16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use crate::x::cx::Slab;
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// Five roots, five modules, for the reason the header gives.
// ---------------------------------------------------------------------------

/// `ssm/causal_conv1d.cuh` — the depthwise causal convolution, batched.
pub mod causal_conv1d {
    use super::{MaybeConst, bf16};
    use core::ptr::NonNull;

    unit! {
        /// Four of the header's five `__global__` templates, all at
        /// `device::bf16`.
        ///
        /// The fifth is `causal_conv1d_update` — the single-request update,
        /// which the batched one at `:380` supersedes: nothing in the tree
        /// fires it, and the batched form takes the same convolution over a
        /// slot table. A declaration for it would name a caller that does not
        /// exist.
        ///
        /// The header holds no host function and no `<<<>>>` at all: it is
        /// five namespace-qualified `__global__`s, which is why the fires
        /// below state every rectangle themselves.
        unit CAUSAL_CONV1D = "ssm/causal_conv1d",
            text = include_str!("../../csrc/src/ssm/causal_conv1d.cuh"),
            file = "ssm/causal_conv1d.cuh";

        /// `causal_conv1d.cuh:380` — one step of the convolution for each of
        /// `R` requests, reading and advancing each one's `[K, C]` ring.
        ///
        /// `R` crosses as an operand AND sizes `grid.y`: the kernel guards
        /// `if (r >= R || c >= C) return;` at `:392` with the same number the
        /// grid was opened on. Both, because `SplitPacked` rounds `C` up to a
        /// whole block and the tail threads have to be told to stop.
        fn causal_conv1d_update_batched =
            "ssm::device::causal_conv1d_update_batched" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            state_base: *mut T,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            y: *mut T,
            r: i32,
            c: i32,
            k: i32,
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_update_batched_bf16" =>
                where [T = bf16] "device::bf16",
        }

        /// `causal_conv1d.cuh:95` — the single-request prefill, one channel
        /// per block, `SILU` off.
        ///
        /// **Nothing in `ssm` fires this and it must survive.** Gemma-4's
        /// audio tower states it by symbol at
        /// `driver-cuda/src/tower/gemma4_audio.rs:873`, with `grid [C, 1, 1]`,
        /// `block [64, 1, 1]` and a null `bias` and `state_out`, transcribed
        /// there from the tower's own C++. That tower is not a trace, so this
        /// row carries no contract and no bind — the declaration exists so the
        /// symbol resolves, which is all the tower asks of it.
        ///
        /// `families/ssm.rs` recorded this row as an INVERSION worth keeping:
        /// it was once reported as having an empty consumer set, on a sweep
        /// that looked for trace statements and found none. A symbol fired by
        /// string from a hand-written tower has a consumer that no sweep for
        /// `dsl::` wrappers can see.
        fn causal_conv1d_prefill = "ssm::device::causal_conv1d_prefill" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out: Option<NonNull<T>>,
            n: i32,
            c: i32,
            k: i32,
        ) where *const T, *mut T, MaybeConst<T>, Option<NonNull<T>> {
            "ssm::causal_conv1d_prefill_noact_bf16" =>
                where [T = bf16] "device::bf16, false",
        }

        /// `causal_conv1d.cuh:297` — the batched prefill, CHANNELS TILED
        /// across `blockDim.x`.
        ///
        /// The arm [`super::causal_conv1d_prefill_batched_bf16`] takes from
        /// eight requests up. One block covers `TILE` channels of one request,
        /// so a wide batch opens `ceil(C / 128) * R` blocks where the
        /// per-channel form would open `C * R` narrow ones.
        fn causal_conv1d_prefill_batched_channel_tile =
            "ssm::device::causal_conv1d_prefill_batched_channel_tile" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out_base: *mut T,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            c: i32,
            k: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
            commit_len: MaybeConst<i32>,
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_prefill_batched_bf16#channel_tile" =>
                where [T = bf16] "device::bf16",
        }

        /// `causal_conv1d.cuh:212` — the batched prefill, ONE BLOCK PER
        /// CHANNEL.
        ///
        /// The arm below eight requests, where `C` blocks of 64 is more
        /// parallelism than `ceil(C / 128)` blocks of 128.
        ///
        /// **The parameter order is not the launcher's.** `R` never crosses —
        /// it is `grid.y` and the kernel reads it as `blockIdx.y` — and the
        /// mask precedes the commit lengths here where a reader of the fire's
        /// argument list would expect the reverse. Thirteen values, in this
        /// order, and the fire below is written against this list and not
        /// against its own parameter names.
        fn causal_conv1d_prefill_batched =
            "ssm::device::causal_conv1d_prefill_batched" <T> (
            x: *const T,
            weight: *const T,
            bias: MaybeConst<T>,
            y: *mut T,
            state_out_base: *mut T,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            c: i32,
            k: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
            commit_len: MaybeConst<i32>,
        ) where *const T, *mut T, MaybeConst<T> {
            "ssm::causal_conv1d_prefill_batched_bf16#per_channel" =>
                where [T = bf16] "device::bf16",
        }
    }
}

/// `ssm/gated_delta_net.cuh` — the recurrence itself.
pub mod gated_delta_net {
    use super::MaybeConst;
    use core::ffi::c_void;

    unit! {
        /// Thirteen instantiations of five templates — the four the host
        /// programs below fire, and the two the deleted `RecurrentScan` and
        /// `WarpTiledScan` rows fired through a rule.
        ///
        /// # THE THREE `elem` SPELLINGS, AND NONE OF THEM IS `device::bf16`
        ///
        /// * **`"ssm::device::f32"`** and not `device::f32`. The prelude names
        ///   no fp32 alias; this header declares its own at `:151`, beside a
        ///   comment on why a leaf may typedef but must not specialise `Elem`.
        ///   The unqualified spelling fails at the name-map pragma with
        ///   `namespace ... has no member "f32"`, before any launch.
        /// * **`"ssm::device::state_bf16"`** and not `device::bf16`, which the
        ///   header calls out at `:143` as *"a different type:
        ///   `state_load`/`state_store` below are template-specialised on
        ///   `__nv_bfloat16` and a `device::bf16` state would fall into the
        ///   generic `static_cast` primary template instead. Same two bytes, a
        ///   different rounding path, and nothing reports the substitution."*
        ///   **This is why the bf16-state parameter is `*mut c_void` and not
        ///   `*mut bf16`**: `Abi for *mut bf16` spells `CPP` as
        ///   `device::bf16*`, which is the type the header says this is NOT.
        ///   `Ty::BufMut` is what the deleted rows said and `void*` is what
        ///   the typecheck TU can honestly compare. Transcribed, not improved.
        /// * **`"ssm::device::gqa_smem_bv"`**, a named `constexpr int = 128`
        ///   at `:178`, because `elem` is the whole template argument list with
        ///   `::pie_cuda_driver::kernels::` glued to its FIRST token: a row
        ///   spelling `"128"` emits
        ///   `recurrent_step_batched_gqa_smem<::pie_cuda_driver::kernels::128>`,
        ///   which is not a C++ token sequence. `DeviceKernel::PLAIN` is not
        ///   the way out either — it emits no angle brackets, and
        ///   `tests/layers.rs`'s `every_row_spells_a_qualified_instantiation`
        ///   asserts a plain row's instantiation carries neither `<` nor `>`.
        ///   `"ssm::device::f32, 128, 128"` escapes the problem only because
        ///   its first argument is a type.
        unit GATED_DELTA_NET = "ssm/gated_delta_net",
            text = include_str!("../../csrc/src/ssm/gated_delta_net.cuh"),
            file = "ssm/gated_delta_net.cuh";

        /// The non-GQA decode step: one block per (request, value head), two
        /// key heads' worth of floats shared.
        ///
        /// `S` is the STATE element and nothing else — every other pointer is
        /// `float*` on both instantiations. That is what makes one `fn` serve
        /// the fp32 and bf16 state slabs.
        fn recurrent_step_batched =
            "ssm::device::recurrent_step_batched" <S> (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::recurrent_gated_delta_step_batched" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::recurrent_gated_delta_step_batched_state_bf16" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The GQA decode step, HBM state: `k_h` joins the list and the value
        /// heads fan out over it.
        ///
        /// `#hbm` on the bf16 arm and no suffix on the fp32 one, because the
        /// bf16 launcher has two arms and the fp32 launcher has one. The
        /// suffix is the LAUNCHER's, not the kernel's — same
        /// `recurrent_step_batched_gqa` template underneath both.
        fn recurrent_step_batched_gqa =
            "ssm::device::recurrent_step_batched_gqa" <S> (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::recurrent_gated_delta_step_batched_gqa" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#hbm" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The GQA decode step with the VALUE TILE in shared memory.
        ///
        /// One instantiation and one non-type argument, `BV = 128`. Same
        /// parameter list as the HBM form; a different kernel because the
        /// state tile lives somewhere else.
        ///
        /// **This is the arm §30's measurement was taken on**, and the
        /// measurement is the reason both arms exist rather than one:
        /// 2 406 µs to 1 579 µs at R=511, a 34% cut on the step, +32%
        /// end-to-end on Qwen3.5-4B — 6 924 to 9 166 tok/s. It was checked
        /// byte-identical against the HBM arm across eight shapes,
        /// 535 822 336 bytes compared, so the two arms are one function with
        /// two memories and not two answers.
        fn recurrent_step_batched_gqa_smem =
            "ssm::device::recurrent_step_batched_gqa_smem" (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut c_void,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) {
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#smem" =>
                "ssm::device::gqa_smem_bv",
        }

        /// The chunked prefill, warp-tiled over the value width, GQA-aware.
        ///
        /// Sixteen parameters: the step's thirteen plus `qo_indptr`,
        /// `write_state` and the mask. No `commit_len` — this kernel commits
        /// the whole region or none of it.
        fn chunk_gated_delta_prefill_batched_warp_tiled_gqa =
            "ssm::device::chunk_gated_delta_prefill_batched_warp_tiled_gqa" <S> (
            q_norm_kh: *const f32,
            k_norm_kh: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            write_state_mask: *const u8,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The FLA chunked prefill: `<StateT, BV, BK_MAX>`, `BV = BK_MAX =
        /// 128`.
        ///
        /// **The 9x, and it is bit-identical**: 47.5 ms to 5.3 ms per layer
        /// against the per-token form below, on the same inputs, with the same
        /// outputs to the bit. Seventeen parameters — the warp-tiled form's
        /// sixteen plus `commit_len`, and the mask moves to last.
        fn chunk_gated_delta_prefill_batched_fla =
            "ssm::device::chunk_gated_delta_prefill_batched_fla" <S> (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            commit_len: MaybeConst<i32>,
            write_state_mask: MaybeConst<u8>,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched#fla" =>
                where [S = f32] "ssm::device::f32, 128, 128",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#fla" =>
                where [S = c_void] "ssm::device::state_bf16, 128, 128",
        }

        /// The legacy per-token chunked prefill — the fallback arm.
        ///
        /// **Thirteen parameters, five fewer than the FLA form**, and the five
        /// are the whole difference: no `k_h`, so it is not GQA-aware and
        /// treats every value head as its own key head; no `write_state`, no
        /// `commit_len`, no mask, so it always commits. A caller that reaches
        /// this arm on a GQA layer gets a different answer, not a slower one,
        /// which is why [`super::chunk_prefill`] states the FLA test rather
        /// than trying both.
        fn chunk_gated_delta_prefill_batched =
            "ssm::device::chunk_gated_delta_prefill_batched" <S> (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched#per_token" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#per_token" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }

        /// The chunked prefill with the WHOLE `[K_d, V_d]` state in shared
        /// memory.
        ///
        /// `k_d * v_d * 4` bytes, which is **65 536 at 128x128 — over the
        /// 48 KiB default cap**, and the only launch in `ssm` that needs
        /// `smem_opt_in`. Fifteen parameters: the per-token form's thirteen
        /// plus `write_state` and the mask, and no `commit_len`.
        fn chunk_gated_delta_prefill_batched_cached =
            "ssm::device::chunk_gated_delta_prefill_batched_cached" <S> (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            g_log: *const f32,
            beta: *const f32,
            state_base: *mut S,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            write_state: bool,
            write_state_mask: MaybeConst<u8>,
        ) where *mut S {
            "ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem" =>
                where [S = f32] "ssm::device::f32, false",
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem" =>
                where [S = c_void] "ssm::device::state_bf16, false",
        }
    }
}

/// `ssm/gated_delta_net_prep.cuh` — the casts, the fan-out, and the two
/// post-convolution preparations.
pub mod gated_delta_net_prep {
    use core::ffi::c_void;

    unit! {
        /// Eight instantiations: the two casts at bf16 AND f16, the GQA head
        /// fan-out, the L2 norm, and the two halves of the Qwen post-conv
        /// preparation.
        ///
        /// **The f16 twins have no contract and no bind, and that is not an
        /// omission.** They were declared instantiations in the deleted family
        /// too, with no table row: the templates are written over `Elem<T>`
        /// and declaring both proves the text compiles at both, which is what
        /// a unit is for. Nothing in the tree fires them.
        unit GATED_DELTA_NET_PREP = "ssm/gated_delta_net_prep",
            text = include_str!(
                "../../csrc/src/ssm/gated_delta_net_prep.cuh"
            ),
            file = "ssm/gated_delta_net_prep.cuh";

        /// `T -> float`, one thread per element.
        ///
        /// `Buf` and not `*const T`, because one `fn` carries both
        /// instantiations and neither the deleted row nor this declaration
        /// needs to say which: the cast is selected by the SYMBOL, and the
        /// buffer is opaque to the host either way. `n` is `usize` because the
        /// kernel's loop bound is `device::usize`.
        fn widen = "ssm::device::widen" (
            x: *const c_void,
            y: *mut f32,
            n: usize,
        ) {
            "ssm::bf16_to_fp32" => "device::bf16",
            "ssm::f16_to_fp32" => "device::f16",
        }

        /// `float -> T`, the inverse of [`widen`](raw::widen).
        fn narrow = "ssm::device::narrow" (
            x: *const f32,
            y: *mut c_void,
            n: usize,
        ) {
            "ssm::fp32_to_bf16" => "device::bf16",
            "ssm::fp32_to_f16" => "device::f16",
        }

        /// GQA head fan-out: `[N, K_h, D] -> [N, V_h, D]`, each key head
        /// repeated `V_h / K_h` times.
        ///
        /// **The device takes `repeat` where the deleted launcher took `n`.**
        /// `N` is `grid.x` and never crosses; `repeat` is `V_h / K_h`, which
        /// the row spelled `Source::Div(&Gdn("v_h"), &Gdn("k_h"))` and the
        /// bind below computes from the same two numbers. Getting these two
        /// the wrong way round writes `N` copies of head 0.
        ///
        /// `families/ssm.rs` checked the body against a wider block than the
        /// rule opens and found it safe: every thread bounds on `d < D` and
        /// strides by `blockDim.x`, so a block wider than `D` costs lanes and
        /// not correctness.
        fn repeat_interleave_heads =
            "ssm::device::repeat_interleave_heads_fp32" (
            in_: *const f32,
            out: *mut f32,
            k_h: i32,
            v_h: i32,
            d: i32,
            repeat: i32,
        ) {
            "ssm::repeat_interleave_heads_fp32" => "ssm::device::f32",
        }

        /// Row-wise L2 norm with a scale, `T -> float`.
        ///
        /// **TWO template arguments, and the second is the block width.**
        /// `l2norm_scale<device::bf16, 128>`, with `constexpr int BLOCK = 128`
        /// in the launcher this replaces. The bare `128` sits in slot 2 and is
        /// not prefixed, because `instantiation()` pastes
        /// `::pie_cuda_driver::kernels::` ONCE at the front of the string.
        ///
        /// `families/ssm.rs` named this the family's fold hazard and the
        /// warning survives with the number: the kernel folds a `__shared__
        /// float buf[BLOCK]` with `for (off = blockDim.x / 2; off > 0; off >>=
        /// 1)`. A 256-wide launch of a 128-wide instantiation indexes past a
        /// static array; a 64-wide one normalises by a sum missing the terms
        /// the unlaunched lanes held. Finite, plausible, wrong. It is
        /// unreachable HERE because [`super::PER_ROW_NARROW_BLOCK`] and this
        /// `elem`'s `128` are the same number, stated once each and cited to
        /// each other.
        ///
        /// `n` is not a parameter: the row count is `grid.x`.
        fn l2norm_scale = "ssm::device::l2norm_scale" (
            x: *const c_void,
            y: *mut f32,
            hidden: i32,
            scale: f32,
            eps: f32,
        ) {
            "ssm::l2norm_scale_bf16_to_fp32" => "device::bf16, 128",
        }

        /// The first half of Qwen's post-convolution preparation: the Q/K
        /// norms.
        ///
        /// **Declared here, hosted in `x/driver_internal.rs`.** The device text
        /// is this root's, so the row is this unit's; the host program is
        /// [`crate::x::driver_internal::qwen_gdn_post_conv_prep_bf16`], which
        /// fires this and its sibling as an ordered pair, and the contract is
        /// `table::driver_internal`'s. Both rows stay where the device text is.
        ///
        /// `template <class T, int BLOCK>` with `BLOCK = 128`, the width the
        /// launcher launches at — one number in both places, as
        /// [`l2norm_scale`](raw::l2norm_scale) above.
        fn qwen_gdn_qk_norm = "ssm::device::qwen_gdn_qk_norm" (
            qkv_post: *const c_void,
            q_out: *mut f32,
            k_out: *mut f32,
            k_h: i32,
            k_d: i32,
            conv_dim: i32,
            q_scale: f32,
        ) {
            "ssm::qwen_gdn_post_conv_prep_bf16#qk_norm" => "device::bf16, 128",
        }

        /// The second half: V, the gate log, and beta.
        ///
        /// [`qwen_gdn_qk_norm`](raw::qwen_gdn_qk_norm)'s sibling, and the two
        /// are not two arms — they are two launches in a fixed order.
        fn qwen_gdn_v_g_beta = "ssm::device::qwen_gdn_v_g_beta" (
            qkv_post: *const c_void,
            a: *const c_void,
            b: *const c_void,
            a_log: *const f32,
            dt_bias: *const c_void,
            v_out: *mut f32,
            g_log_out: *mut f32,
            beta_out: *mut f32,
            k_h: i32,
            v_h: i32,
            k_d: i32,
            v_d: i32,
            conv_dim: i32,
        ) {
            "ssm::qwen_gdn_post_conv_prep_bf16#v_g_beta" => "device::bf16, 128",
        }
    }
}

/// `ssm/kda.cuh` — Kimi Delta Attention.
pub mod kda {
    use super::bf16;

    unit! {
        /// All four of the header's kernels: two single-argument templates on
        /// a `(tokens, heads)` grid, and the two plain `__global__`s that
        /// carry the recurrence.
        ///
        /// Compiling the header pulls the recurrence in as text whether or not
        /// it is declared, so declaring it costs nothing and buys the two
        /// launches below a resolvable symbol. That cost was measured:
        /// **41 392 bytes of cubin in 140 ms on an L40S.**
        ///
        /// `families/ssm.rs` corrected a misreading here that is worth keeping,
        /// because it is the shape of the mistake and not the mistake: the two
        /// recurrence kernels were once reported as unrepresentable BECAUSE
        /// they are `PLAIN`. They are not — `DeviceKernel::PLAIN` is exactly
        /// how a non-template `__global__` is declared, and what actually
        /// blocked them was `Args::bind` having no `I64` and no rule saying
        /// `(rows, heads)`. Both landed; the `PLAIN` was never the problem.
        unit KDA = "ssm/kda",
            text = include_str!("../../csrc/src/ssm/kda.cuh"),
            file = "ssm/kda.cuh";

        /// `A`'s exponential gate and the beta activation, per (token, head).
        fn kda_gate_beta = "ssm::device::kda_gate_beta" <T> (
            raw_g: *const T,
            raw_beta: *const T,
            a_log: *const f32,
            dt_bias: *const f32,
            gate_out: *mut f32,
            beta_out: *mut f32,
            t: i32,
            h: i32,
            d: i32,
            lower_bound: f32,
        ) where *const T {
            "ssm::kda_gate_beta_bf16" => where [T = bf16] "device::bf16",
        }

        /// The output RMSNorm, gated by `g` — the epilogue of a KDA layer.
        ///
        /// **No `t`.** The token count is `grid.x` and the kernel reads it as
        /// `blockIdx.x`, where [`kda_gate_beta`](raw::kda_gate_beta) one row up
        /// takes it as an operand. Two kernels in one header disagreeing about
        /// that is the kind of thing a rule hid and a parameter list states.
        fn kda_o_norm_gated = "ssm::device::kda_o_norm_gated" <T> (
            o: *const f32,
            g: *const T,
            weight: *const f32,
            out: *mut T,
            h: i32,
            d: i32,
            eps: f32,
        ) where *const T, *mut T {
            "ssm::kda_o_norm_gated_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode step: one block per (request, head), the delta rule
        /// applied once.
        fn kda_recurrent_step_batched =
            "ssm::device::kda_recurrent_step_batched" (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            gate: *const f32,
            beta: *const f32,
            state_base: *mut f32,
            slot_ids: *const i32,
            slot_stride_elems: i64,
            out: *mut f32,
            h: i32,
            d: i32,
        ) {
            "ssm::kda_recurrent_step_batched#step" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The prefill: the same recurrence over a whole region, one warp per
        /// state row.
        ///
        /// [`kda_recurrent_step_batched`](raw::kda_recurrent_step_batched)'s
        /// list plus `qo_indptr`, which is what makes it a region and not a
        /// step.
        fn kda_prefill_batched = "ssm::device::kda_prefill_batched" (
            q_norm: *const f32,
            k_norm: *const f32,
            v: *const f32,
            gate: *const f32,
            beta: *const f32,
            state_base: *mut f32,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            slot_stride_elems: i64,
            out: *mut f32,
            h: i32,
            d: i32,
        ) {
            "ssm::kda_prefill_batched#prefill" =>
                crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `ssm/nemotron_h.cuh` — the mamba scan, the three-way split, Zamba's gated
/// norm, and the two MoE pointer builders.
pub mod nemotron_h {
    use super::{MaybeConst, bf16};
    use core::ffi::c_void;

    unit! {
        /// Nine instantiations: three templates at `device::bf16` and six
        /// plain `__global__`s.
        ///
        /// **The third linear-attention shape here, and not a variant of the
        /// other two.** Mamba carries a `[head_dim, state_size]` slab per head
        /// and advances it with a scalar `dA` from a per-token `dt` — a
        /// selective scan, not a delta rule. A different state SHAPE, which is
        /// why none of the GDN or KDA rows stands in for it.
        ///
        /// # `_dev` on the two pointer builders
        ///
        /// `table::ssm` states `ssm::build_nemotron_moe_ptrs_aligned_bf16` and
        /// `..._decode_batched_bf16` — the LAUNCHER names — and
        /// `execution::WALKED` claims them there, so the device rows must be
        /// different strings or `a_walk_is_only_a_walk` fails. The suffix is
        /// `families::moe`'s convention and this family follows it.
        unit NEMOTRON_H = "ssm/nemotron_h",
            text = include_str!("../../csrc/src/ssm/nemotron_h.cuh"),
            file = "ssm/nemotron_h.cuh";

        /// Three bf16 tables widened to fp32, once per layer.
        fn prepare_mamba_params = "ssm::device::prepare_mamba_params" <T> (
            a_log: *const T,
            d: *const T,
            dt_bias: *const T,
            a: *mut f32,
            d_f32: *mut f32,
            dt_bias_f32: *mut f32,
            num_heads: i32,
        ) where *const T {
            "ssm::nemotron_prepare_mamba_params" =>
                where [T = bf16] "device::bf16",
        }

        /// `dt` softplussed and `dA = exp(dt * A)` precomputed, per (token,
        /// head).
        ///
        /// **The kernel's first extent is `total` and not `n`.** The deleted
        /// launcher took `N` and `num_heads` and multiplied them on the host;
        /// the bind below does the same multiplication, because `dt_out` is
        /// `[N, num_heads]` and that product is what the guard `i >= total`
        /// compares against.
        fn prepare_mamba_dt_da = "ssm::device::prepare_mamba_dt_da" <T> (
            dt: *const T,
            a: *const f32,
            dt_bias: *const f32,
            dt_out: *mut f32,
            da_out: *mut f32,
            total: i32,
            num_heads: i32,
            time_step_min: f32,
        ) where *const T {
            "ssm::nemotron_prepare_mamba_dt_da" =>
                where [T = bf16] "device::bf16",
        }

        /// Zamba's gated output RMSNorm: each norm GROUP of a row scaled by
        /// `silu(gate)`, normalised over the group, multiplied by `weight`.
        ///
        /// **`GatedRms` was named after this launcher**, so the transcription
        /// in [`super::gated_rms`] and this kernel's geometry are the same
        /// three lines read from two ends.
        ///
        /// **The 256 is not negotiable in either direction.** The kernel
        /// declares `__shared__ float buf[256]` STATICALLY and folds it with
        /// `for (off = blockDim.x / 2; off > 0; off >>= 1)`. Wider indexes past
        /// a static array, which the hardware does not report; narrower
        /// normalises by a sum missing the terms the unlaunched lanes held.
        ///
        /// `n` is not a parameter — the row count is `grid.x` — and
        /// `gate_stride` is NOT `hidden`: the gate may be a window into a wider
        /// fused projection, which is why the launcher passed it rather than
        /// assuming it.
        fn zamba_rmsnorm_gated = "ssm::device::zamba_rmsnorm_gated" <T> (
            x: *const T,
            gate: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            gate_stride: i32,
            group_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "ssm::zamba_rmsnorm_gated_bf16" => where [T = bf16] "device::bf16",
        }

        /// The three-way cut of the fused projection, GATED arm.
        ///
        /// `gate` is not nullable here, and that is the whole arm: the sibling
        /// exists BECAUSE `gate` can be null, so a null in this slot is the
        /// other kernel's job and not a spare cell.
        ///
        /// `total` crosses as an operand and also sized the grid, and the two
        /// must be equal: a caller supplying a `total` smaller than
        /// `n * projection_dim` leaves the tail of the rectangle unwritten;
        /// larger, and it reads past `projected`.
        fn mamba_split = "ssm::device::mamba_split" (
            projected: *const c_void,
            gate: *mut c_void,
            conv_in: *mut c_void,
            dt: *mut c_void,
            projection_dim: i32,
            intermediate: i32,
            conv_dim: i32,
            num_heads: i32,
            total: i32,
        ) {
            "ssm::nemotron_mamba_split_bf16#split" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The same cut, UNGATED: no `gate` parameter at all.
        ///
        /// The projection is read at the same `[N, projection_dim]` stride and
        /// the `intermediate` span is SKIPPED (`nemotron_h.cuh:153-154`), which
        /// is why this arm's extent is `n * (conv_dim + num_heads)` and not
        /// `n * projection_dim`.
        ///
        /// **`ElementwiseIn` would over-launch this by `n * intermediate`
        /// elements' worth of blocks**, and the output would be byte-identical
        /// because those blocks return on `i >= total`. That is the near miss
        /// which is invisible at one shape and costs occupancy at every shape,
        /// and it is why the deleted row was `LaunchRule::Unstated` while its
        /// gated sibling was not.
        fn mamba_split_conv_dt = "ssm::device::mamba_split_conv_dt" (
            projected: *const c_void,
            conv_in: *mut c_void,
            dt: *mut c_void,
            projection_dim: i32,
            intermediate: i32,
            conv_dim: i32,
            num_heads: i32,
            total: i32,
        ) {
            "ssm::nemotron_mamba_split_bf16#conv_dt" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The selective scan, PREFILL: one warp per `head_dim` row.
        ///
        /// `dt_precomputed` and `da_precomputed` are nullable on purpose. The
        /// kernel tests both and recomputes from `dt_in`, `A` and `dt_bias`
        /// when absent (`nemotron_h.cuh:257-263`). Nemotron-H fires
        /// `ssm::nemotron_prepare_mamba_dt_da` to fill them and Zamba does not;
        /// an absent pair is a fact about a model, not drift.
        fn mamba_ssm_batched_prefill_reg =
            "ssm::device::mamba_ssm_batched_prefill_reg" (
            conv_out: *const c_void,
            dt_in: *const c_void,
            a: *const f32,
            d: *const f32,
            dt_bias: *const f32,
            dt_precomputed: MaybeConst<f32>,
            da_precomputed: MaybeConst<f32>,
            state_base: *mut c_void,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            y: *mut c_void,
            num_heads: i32,
            head_dim: i32,
            state_size: i32,
            n_groups: i32,
            conv_dim: i32,
            intermediate: i32,
            time_step_min: f32,
        ) {
            "ssm::nemotron_mamba_ssm_batched_bf16#prefill_reg" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The selective scan, DECODE: the same parameter list, one block per
        /// (request, head).
        ///
        /// `nemotron_h.cuh:378-384` is this kernel's own null test for the two
        /// precomputed tables.
        fn mamba_ssm_batched_warp = "ssm::device::mamba_ssm_batched_warp" (
            conv_out: *const c_void,
            dt_in: *const c_void,
            a: *const f32,
            d: *const f32,
            dt_bias: *const f32,
            dt_precomputed: MaybeConst<f32>,
            da_precomputed: MaybeConst<f32>,
            state_base: *mut c_void,
            slot_ids: *const i32,
            qo_indptr: *const u32,
            y: *mut c_void,
            num_heads: i32,
            head_dim: i32,
            state_size: i32,
            n_groups: i32,
            conv_dim: i32,
            intermediate: i32,
            time_step_min: f32,
        ) {
            "ssm::nemotron_mamba_ssm_batched_bf16#warp" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The decode MoE pointer build: one thread per ROUTE.
        ///
        /// `total` is `n * top_k` and not `n`, and it is the same number the
        /// grid is opened on. The `.cuh` records the extent as *"rows *
        /// top_k"*, and it is the single easiest thing in this family to get
        /// wrong: passing `n` builds a `top_k`-th of the pointer table and
        /// leaves the rest of the batched GEMM reading whatever the allocation
        /// held.
        fn build_nemotron_moe_ptrs_decode_batched =
            "ssm::device::build_nemotron_moe_ptrs_decode_batched" (
            topk_idx: *const i32,
            topk_w: *const f32,
            up_weight_ptrs: *const *const c_void,
            down_weight_ptrs: *const *const c_void,
            norm_x: *const c_void,
            expert_up: *mut c_void,
            expert_act: *mut c_void,
            expert_out: *mut c_void,
            a_up_ptrs: *mut *const c_void,
            b_up_ptrs: *mut *const c_void,
            c_up_ptrs: *mut *mut c_void,
            a_down_ptrs: *mut *const c_void,
            b_down_ptrs: *mut *const c_void,
            c_down_ptrs: *mut *mut c_void,
            weights_out: *mut f32,
            total: i32,
            top_k: i32,
            hidden: i32,
            intermediate: i32,
        ) {
            "ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16" =>
                crate::device::DeviceKernel::PLAIN,
        }

        /// The aligned MoE pointer build: one thread per padded BLOCK.
        ///
        /// `max_blocks` is a HOST SCALAR — the padded block count the counting
        /// sort produced — and not the extent of anything the fire allocated,
        /// which is why no launch rule could ever have sized this. The `.cuh`
        /// says *"extent is a host scalar"* in the same list.
        fn build_nemotron_moe_ptrs_aligned =
            "ssm::device::build_nemotron_moe_ptrs_aligned" (
            expert_ids: *const i32,
            up_weight_ptrs: *const *const c_void,
            down_weight_ptrs: *const *const c_void,
            aligned_in: *const c_void,
            aligned_up: *mut c_void,
            aligned_act: *mut c_void,
            aligned_out: *mut c_void,
            a_up_ptrs: *mut *const c_void,
            b_up_ptrs: *mut *const c_void,
            c_up_ptrs: *mut *mut c_void,
            a_down_ptrs: *mut *const c_void,
            b_down_ptrs: *mut *const c_void,
            c_down_ptrs: *mut *mut c_void,
            max_blocks: i32,
            block_size: i32,
            hidden: i32,
            intermediate: i32,
        ) {
            "ssm::build_nemotron_moe_ptrs_aligned_dev_bf16" =>
                crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// The units `ssm` compiles.
///
/// Hand-written where a one-root family's is generated, for the reason this
/// file's header gives. `families::ALL` reads this.
pub static UNITS: &[Unit] = &[
    causal_conv1d::CAUSAL_CONV1D,
    gated_delta_net::GATED_DELTA_NET,
    gated_delta_net_prep::GATED_DELTA_NET_PREP,
    kda::KDA,
    nemotron_h::NEMOTRON_H,
];

// ---------------------------------------------------------------------------
// The numbers, once each.
//
// Two groups, and they are not interchangeable. The first is the LAUNCH
// RULES the fifteen `device::JIT_DISPATCHED` rows were fired through, as the
// expressions they evaluate to; the second is the `<<<>>>` constants the four
// `fire/` modules held beside their launches. A rule was a claim about a
// whole class of kernels and a constant is a claim about one, so they are
// cited differently: a rule to `runtime/launch.rs`, a constant to the `.cu`
// line the deleted `fire/` module quoted.
// ---------------------------------------------------------------------------

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
///
/// The block every pointwise rule in this tree uses.
const RULE_BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:698` — `const LAYERNORM_BLOCK: u32 = 128;`.
///
/// **The same 128 as [`gated_delta_net_prep`]'s `l2norm_scale` `elem`.** The
/// kernel is `l2norm_scale<device::bf16, BLOCK>` with a `__shared__ float
/// buf[BLOCK]`, so the launch width and the template argument are one number;
/// this constant and that `elem` cite each other so they cannot drift apart
/// silently. See the row's own doc for what drift would cost.
const PER_ROW_NARROW_BLOCK: u32 = 128;

/// `runtime/launch.rs:608-610` — `SINK_BLOCK_MIN = WARP`, `SINK_BLOCK_MAX =
/// 128`, the clamp `PerHeadElementwise` sizes a head's block by.
const SINK_BLOCK_MIN: u32 = WARP;
/// `runtime/launch.rs:610`.
const SINK_BLOCK_MAX: u32 = 128;

/// `runtime/launch.rs:640` — `const SCAN_BLOCK: u32 = 128;`.
const SCAN_BLOCK: u32 = 128;

/// `runtime/launch.rs:686` — `const SCAN_WARPS: u32 = 4;`.
const SCAN_WARPS: u32 = 4;

/// `runtime/launch.rs:589` — `const FLOAT: u32 = 4;`, `sizeof(float)` as the
/// shared-memory rules spell it.
const FLOAT: u32 = 4;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:828-834` — `n = dims.rows * dims.width`, then `grid
/// [ceil(n / 256), 1, 1]`, `block [256, 1, 1]`, no shared memory. The grid
/// rounds UP, which is why every kernel fired through it keeps its own
/// element count as an operand.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}

/// `LaunchRule::PerRowNarrow`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1137-1139` — `grid [rows, 1, 1]`, `block [128, 1, 1]`,
/// no shared memory. One block per row, 128 wide regardless of the row's
/// width, because the kernel strides.
#[must_use]
const fn per_row_narrow(rows: u32) -> Launch {
    Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1417-1423` — `grid [rows, heads, 1]`, `block
/// [clamp(head_dim, 32, 128), 1, 1]`, no shared memory.
///
/// **The row is `grid.x` and the head is `grid.y`**, which is the TRANSPOSE
/// of what an attention reader expects. `families/ssm.rs` names this the
/// axis-transpose hazard, and it is the reason `causal_conv1d_update_batched`
/// below is fired through `split_packed` and not through this: the two rules
/// differ in nothing but which axis the row is on, and a kernel reading
/// `blockIdx.y` as its request would silently process request 0 `C` times.
///
/// Not a `const fn`: `Ord::clamp` is not callable in a `const` context, and
/// the rule's expression is transcribed rather than rearranged.
#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `LaunchRule::GatedRms`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1455-1457` — `grid [rows, heads, 1]`, `block [256, 1,
/// 1]`, no shared memory. Same grid as [`per_head_elementwise`], a fixed
/// block where that one clamps.
#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch { grid: [rows, heads, 1], block: [RULE_BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// `LaunchRule::RecurrentScan`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1533-1539` — `grid [rows, heads, 1]`, `block [128, 1,
/// 1]`, `smem = 2 * head_dim * sizeof(float)`.
///
/// **`head_dim` here is `K_d` and not `V_d`.** The shared allocation holds
/// two KEY-width rows of floats; the rule's parameter is named for the
/// attention shape it was first written against, and every caller below
/// passes the key width. The rule's own overflow check returned
/// `Ungeometric::Empty`; here the multiplication is `saturating` and the
/// callers guard `k_d > 0` before reaching it.
#[must_use]
const fn recurrent_scan(rows: u32, heads: u32, k_d: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [SCAN_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
    .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

/// `LaunchRule::WarpTiledScan`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1594-1600` — `grid [rows, heads, ceil(value_width /
/// 4)]`, `block [4 * 32, 1, 1]`, no shared memory. The third axis is the
/// value width divided by the block's WARP COUNT, so the 4 appears twice and
/// must move together.
#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    Launch {
        grid: [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        block: [SCAN_WARPS * WARP, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `LaunchRule::SplitPacked`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1674-1676` — `grid [ceil(in_width / 256), rows, 1]`,
/// `block [256, 1, 1]`, no shared memory. **Rows on `grid.y`**, which is
/// [`per_head_elementwise`]'s transpose and this rule's whole point.
#[must_use]
const fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch {
        grid: [in_width.div_ceil(RULE_BLOCK), rows, 1],
        block: [RULE_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

// ── the `<<<>>>` constants, one group per deleted `fire/` module ──────────

/// `causal_conv1d.cu:64` — `constexpr int TILE = 128;`.
const CONV_TILE: u32 = 128;

/// `causal_conv1d.cu:78` — `constexpr int BLOCK = 64;` on the per-channel
/// arm.
const CONV_PER_CHANNEL_BLOCK: u32 = 64;

/// `causal_conv1d.cu:65` — the request count from which the channel-tiled arm
/// is taken.
///
/// Below eight requests, `C` blocks of 64 is more parallelism than
/// `ceil(C / 128)` blocks of 128; at and above it the tile wins. One
/// threshold, one comparison, and it is the launcher's own.
const CONV_CHANNEL_TILE_FROM: i32 = 8;

/// `kda.cu:50` — `constexpr int BLOCK = 256;` on the decode step.
const KDA_STEP_BLOCK: u32 = 256;

/// `kda.cu:73` — `constexpr int MAX_WARPS = 32;`, the prefill's warp cap.
const KDA_PREFILL_MAX_WARPS: i32 = 32;

/// `kda.cu:51` and `:75` — `3 * D * sizeof(float)`, the prefill's and the
/// step's shared request.
///
/// Three `D`-wide float rows. `shmem(128)` is **1 536 bytes**, comfortably
/// under the 48 KiB default cap, which is why neither KDA launch opts in.
#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

/// `nemotron_h.cu:36` — `constexpr int BLOCK = 256;` on both split arms.
const SPLIT_BLOCK: u32 = 256;

/// `nemotron_h.cu:120` — `constexpr int BLOCK = 256;` on the decode scan.
const SSM_DECODE_BLOCK: u32 = 256;

/// `nemotron_h.cu:123` — `constexpr int BLOCK = 512;` on the prefill scan.
///
/// **The 512 appears twice and must move together**: the prefill's third grid
/// axis is `ceil(head_dim / (512 / 32))` — one warp per `head_dim` row, 16 of
/// them. `SSM_PREFILL_BLOCK / WARP == 16` is the identity that ties them.
const SSM_PREFILL_BLOCK: u32 = 512;

/// `nemotron_h.cu:77` and `:120` — `constexpr int BLOCK = 256;` on both
/// pointer builders.
const PTRS_BLOCK: u32 = 256;

/// `gated_delta_net.cu:249` — `constexpr int BV = 128;` on the shared-memory
/// step.
///
/// One of FOUR names for the number 128 in this family, cutting four
/// different axes: this one is the VALUE TILE the shared step covers per
/// block, [`GDN_BLOCK`] is a thread count, [`BV_FLA`] is the FLA prefill's
/// value tile, and [`BK_MAX_FLA`] is a KEY-width bound. They agree today and
/// each has its own line so that they can stop agreeing.
const SMEM_BV: u32 = 128;

/// `gated_delta_net.cu:253` — `constexpr int BLOCK = 128;`, a THREAD COUNT.
const GDN_BLOCK: u32 = 128;

/// `gated_delta_net.cu:322` — `constexpr int BV = 128;` on the FLA prefill.
const BV_FLA: u32 = 128;

/// `gated_delta_net.cu:321` — `constexpr int BK_MAX = 128;`, the FLA
/// prefill's KEY-width BOUND.
///
/// **The FLA arm's shared request is `2 * BK_MAX * 4` and not `2 * K_d * 4`**
/// — 1 024 bytes at every shape, because the kernel is instantiated at the
/// bound and allocates for it. Substituting `K_d` there would under-allocate
/// for every layer narrower than 128 and the kernel would read past its own
/// tile.
const BK_MAX_FLA: i32 = 128;

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning `Fired`
// so that "it declined" cannot be spelled like "it ran".
//
// EVERY REFUSAL IS ABOVE EVERY LAUNCH. §5.1's rule for a multi-launch body —
// *"a `Declined` returned after something has already gone to the device is a
// lie of exactly the kind `Fired` exists to prevent"* — is kept here by
// construction rather than by care: the four bodies that can fire two
// different kernels ([`causal_conv1d_prefill_batched_bf16`],
// [`recurrent_step_batched_gqa_state_bf16`], [`mamba_split_bf16`],
// [`mamba_ssm_batched_bf16`]) each resolve their whole refusal set, then pick
// a symbol and a rectangle, then fire once. None of them fires twice, so none
// of them can return after a launch. See [`mamba_split_bf16`] for the one
// place the archive had a refusal INSIDE an arm and where it went.
// ---------------------------------------------------------------------------

// ── ssm/causal_conv1d ────────────────────────────────────────────────────

/// `ssm::causal_conv1d_update_batched_bf16` — one convolution step per
/// request, advancing each one's `[K, C]` ring.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was in `device::JIT_DISPATCHED` with `LaunchRule::SplitPacked`, so
/// `causal_conv1d.cu`'s launcher went with the file. [`split_packed`] is that
/// rule, and the width it is given is `c` — the channel count, which the
/// deleted row bound from `Source::Gdn("conv_dim")` and which is also the
/// kernel's own `C`. The same number twice, by construction.
///
/// **`SplitPacked` and not `PerHeadElementwise`, and the difference is the
/// axis.** This kernel reads `blockIdx.y` as the REQUEST and
/// `blockIdx.x * blockDim.x + threadIdx.x` as the channel
/// (`causal_conv1d.cuh:390-391`). `PerHeadElementwise` puts the row on
/// `grid.x`. `families/ssm.rs` recorded that transposition as this row's
/// hazard and it is why the two rules are cited separately in this file.
///
/// # Safety
///
/// `x` and `y` must address `r * c` live bf16 elements, `weight` `c * k`,
/// `state_base` at least `slot_ids[r] * slot_stride_elems + k * c` writable
/// ones for every `r`, `slot_ids` `r` live `i32`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_update_batched_bf16(
    x: *const bf16,
    weight: *const bf16,
    bias: MaybeConst<bf16>,
    state_base: *mut bf16,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    y: *mut bf16,
    r: i32,
    c: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_k" });
    }
    unsafe {
        causal_conv1d::raw::causal_conv1d_update_batched(
            "ssm::causal_conv1d_update_batched_bf16",
            split_packed(r.unsigned_abs(), c.unsigned_abs()),
            x,
            weight,
            bias,
            state_base,
            slot_ids,
            slot_stride_elems,
            y,
            r,
            c,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::causal_conv1d_prefill_batched_bf16` — the batched prefill, in
/// whichever of its two shapes the request count asks for.
///
/// `causal_conv1d.cu:63-82`, both arms:
///
/// ```text
/// :63   if (R <= 0 || C <= 0 || K <= 0) return;
/// :64   constexpr int TILE = 128;
/// :65   if (R >= 8) {
/// :66     dim3 grid((C + TILE - 1) / TILE, R);
/// :67     device::causal_conv1d_prefill_batched_channel_tile<bf16>
/// :68         <<<grid, dim3(TILE), 0, stream>>>(...);
/// :78   constexpr int BLOCK = 64;
/// :79     dim3 grid(C, R);
/// :80     device::causal_conv1d_prefill_batched<bf16>
/// :81         <<<grid, dim3(BLOCK), 0, stream>>>(...);
/// ```
///
/// **The argument list is the kernel's and not this function's.** `r` never
/// crosses — it is `grid.y` — and `write_state_mask` precedes `commit_len`,
/// which is the reverse of this signature's tail. That inversion was in the
/// deleted `fire/causal_conv1d.rs` too and it is transcribed, not tidied: the
/// parameter order here is the one every caller in the tree already writes.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `qo_indptr` addresses `r + 1` live `u32`, and `stream` is live for the
/// same window.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn causal_conv1d_prefill_batched_bf16(
    x: *const bf16,
    weight: *const bf16,
    bias: MaybeConst<bf16>,
    y: *mut bf16,
    state_out_base: *mut bf16,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    r: i32,
    c: i32,
    k: i32,
    stream: *mut c_void,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
) -> Fired {
    // `causal_conv1d.cu:63`, split so the caller learns which extent was
    // empty. Every one of them is resolved here, above both arms.
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if c <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_dim" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "conv_k" });
    }
    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());
    if r >= CONV_CHANNEL_TILE_FROM {
        // One block covers `TILE` channels of one request. At eight requests
        // and up this is more work per block and fewer of them.
        unsafe {
            causal_conv1d::raw::causal_conv1d_prefill_batched_channel_tile(
                "ssm::causal_conv1d_prefill_batched_bf16#channel_tile",
                Launch {
                    grid: [chans.div_ceil(CONV_TILE), rows, 1],
                    block: [CONV_TILE, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                x,
                weight,
                bias,
                y,
                state_out_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                c,
                k,
                write_state,
                write_state_mask,
                commit_len,
                stream,
            );
        }
        return Fired::Launched;
    }
    // Below eight requests, `C` narrow blocks beat `ceil(C / 128)` wide ones.
    unsafe {
        causal_conv1d::raw::causal_conv1d_prefill_batched(
            "ssm::causal_conv1d_prefill_batched_bf16#per_channel",
            Launch {
                grid: [chans, rows, 1],
                block: [CONV_PER_CHANNEL_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            x,
            weight,
            bias,
            y,
            state_out_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            c,
            k,
            write_state,
            write_state_mask,
            commit_len,
            stream,
        );
    }
    Fired::Launched
}

// ── ssm/gated_delta_net_prep ─────────────────────────────────────────────

/// `ssm::bf16_to_fp32` — widen a whole buffer.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::Elementwise` over `Dims { rows, width }`, which is
/// [`elementwise`]. The element count the rule divides and the `n` the kernel
/// bounds on are the same number here — `Source::OutElements(0)` on the
/// deleted row — which is what makes the rounded-up tail safe.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `y` `n` writable floats, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn bf16_to_fp32(
    x: *const c_void,
    y: *mut f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let Ok(count) = u32::try_from(n) else {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    };
    if count == 0 {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    }
    unsafe {
        gated_delta_net_prep::raw::widen(
            "ssm::bf16_to_fp32",
            elementwise(count),
            x,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::fp32_to_bf16` — [`bf16_to_fp32`]'s inverse, on the same rule.
///
/// # Safety
///
/// `x` must address `n` live floats and `y` `n` writable bf16 elements, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn fp32_to_bf16(
    x: *const f32,
    y: *mut c_void,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let Ok(count) = u32::try_from(n) else {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    };
    if count == 0 {
        return Fired::Declined(Refusal::Empty { what: "element count" });
    }
    unsafe {
        gated_delta_net_prep::raw::narrow(
            "ssm::fp32_to_bf16",
            elementwise(count),
            x,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::repeat_interleave_heads_fp32` — fan `K_h` key heads out to `V_h`
/// value heads.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::GatedRms` over `Dims { rows, kv_heads }`, which is
/// [`gated_rms`] — `grid [n, v_h, 1]`, `block [256, 1, 1]`. The block is
/// wider than the rule's model needs and that is safe here, which
/// `families/ssm.rs` checked in the body rather than assuming: every thread
/// bounds on `d < D` and strides by `blockDim.x`.
///
/// **`repeat` is `v_h / k_h` and the kernel's last parameter; `n` is `grid.x`
/// and crosses nowhere.** The deleted launcher took `n` in that slot. Getting
/// them the wrong way round writes `n` copies of head 0 into every head.
///
/// # Safety
///
/// `in_` must address `n * k_h * d` live floats and `out` `n * v_h * d`
/// writable ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn repeat_interleave_heads_fp32(
    in_: *const f32,
    out: *mut f32,
    n: i32,
    k_h: i32,
    v_h: i32,
    d: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net_prep::raw::repeat_interleave_heads(
            "ssm::repeat_interleave_heads_fp32",
            gated_rms(n.unsigned_abs(), v_h.unsigned_abs()),
            in_,
            out,
            k_h,
            v_h,
            d,
            v_h / k_h,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::l2norm_scale_bf16_to_fp32` — row-wise L2 norm with a scale, widening
/// as it goes.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::PerRowNarrow`, which is [`per_row_narrow`] — one block per
/// row, [`PER_ROW_NARROW_BLOCK`] wide. **That width is also the kernel's
/// template argument**, and the row's whole reason for existing: see the
/// declaration's doc for what a 256-wide launch of a 128-wide instantiation
/// does to a static `__shared__ float buf[BLOCK]`.
///
/// # Safety
///
/// `x` must address `n * hidden` live bf16 elements and `y` the same count of
/// writable floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn l2norm_scale_bf16_to_fp32(
    x: *const c_void,
    y: *mut f32,
    n: i32,
    hidden: i32,
    scale: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        gated_delta_net_prep::raw::l2norm_scale(
            "ssm::l2norm_scale_bf16_to_fp32",
            per_row_narrow(n.unsigned_abs()),
            x,
            y,
            hidden,
            scale,
            eps,
            stream,
        );
    }
    Fired::Launched
}

// ── ssm/kda ──────────────────────────────────────────────────────────────

/// `ssm::kda_gate_beta_bf16` — the gate and beta activations, per (token,
/// head).
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::PerHeadElementwise`, which is [`per_head_elementwise`] —
/// `grid [t, h, 1]`, `block [clamp(d, 32, 128), 1, 1]`. The head dim it
/// clamps is `d`, which the deleted row bound from `Source::Param(0)`: a
/// statement parameter and not a context value, so the launch width and the
/// kernel's `d` are one number the statement states once.
///
/// # Safety
///
/// `raw_g` and `raw_beta` must address `t * h * d` and `t * h` live bf16
/// elements, `a_log` and `dt_bias` `h` live floats, `gate_out` and `beta_out`
/// `t * h * d` and `t * h` writable ones, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_gate_beta_bf16(
    raw_g: *const bf16,
    raw_beta: *const bf16,
    a_log: *const f32,
    dt_bias: *const f32,
    gate_out: *mut f32,
    beta_out: *mut f32,
    t: i32,
    h: i32,
    d: i32,
    lower_bound: f32,
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_gate_beta(
            "ssm::kda_gate_beta_bf16",
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            raw_g,
            raw_beta,
            a_log,
            dt_bias,
            gate_out,
            beta_out,
            t,
            h,
            d,
            lower_bound,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_o_norm_gated_bf16` — the gated output RMSNorm that closes a KDA
/// layer.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::PerHeadElementwise` again, and the same clamp on the same
/// `Source::Param(0)`/`Param(1)` pair. **`t` is not an operand here** where it
/// is one on [`kda_gate_beta_bf16`]: this kernel reads the token from
/// `blockIdx.x` and the other bounds on it, so the token count is a
/// rectangle for one and a rectangle AND a guard for the other. Two kernels,
/// one header, and the parameter lists say which is which.
///
/// # Safety
///
/// `o` must address `t * h * d` live floats, `g` the same count of bf16
/// elements, `weight` `h * d` live floats, `out` `t * h * d` writable bf16
/// elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_o_norm_gated_bf16(
    o: *const f32,
    g: *const bf16,
    weight: *const f32,
    out: *mut bf16,
    t: i32,
    h: i32,
    d: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_o_norm_gated(
            "ssm::kda_o_norm_gated_bf16",
            per_head_elementwise(t.unsigned_abs(), h.unsigned_abs(), d.unsigned_abs()),
            o,
            g,
            weight,
            out,
            h,
            d,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_recurrent_step_batched` — one delta-rule step per (request,
/// head).
///
/// `kda.cu:47-53`:
///
/// ```text
/// :47   if (R <= 0 || H <= 0 || D <= 0) return;
/// :50   constexpr int BLOCK = 256;
/// :51   const size_t shmem = 3 * D * sizeof(float);
/// :52   device::kda_recurrent_step_batched<<<dim3(R, H), dim3(BLOCK), shmem, stream>>>(
/// :53       q_norm, k_norm, v, gate, beta, state_base, slot_ids,
/// :53       slot_stride_elems, out, H, D);
/// ```
///
/// **`r` does not cross.** It is `grid.x` and the kernel reads it as
/// `blockIdx.x`; the eleven values below omit it, where the twelve parameters
/// of this function include it.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `state_base` addresses `slot_ids[r] * slot_stride_elems + h * d * d`
/// writable floats for every `r`, and `stream` is live for the same window.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_recurrent_step_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_recurrent_step_batched(
            "ssm::kda_recurrent_step_batched#step",
            Launch {
                grid: [r.unsigned_abs(), h.unsigned_abs(), 1],
                block: [KDA_STEP_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(kda_shmem(d.unsigned_abs())),
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            h,
            d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::kda_prefill_batched` — the same recurrence over a whole region, ONE
/// WARP PER STATE ROW.
///
/// `kda.cu:64-77`:
///
/// ```text
/// :64   if (R <= 0 || H <= 0 || D <= 0) return;
/// :73   constexpr int MAX_WARPS = 32;
/// :74   const int block = std::min(D, MAX_WARPS) * 32;
/// :75   const size_t shmem = 3 * D * sizeof(float);
/// :76   device::kda_prefill_batched<<<dim3(R, H), dim3(block), shmem, stream>>>(
/// :77       ..., qo_indptr, ..., H, D);
/// ```
///
/// # The 2.2x, and the block width it was measured at
///
/// This shape replaced a one-warp-per-block prefill and the measurement is
/// the reason it exists: **26.2 ms to 12.0 ms per layer at T=2048**, at K3's
/// widths. At `D = 128` the `min` saturates and the block is **1 024 threads**
/// — the widest launch in this family — because 32 warps is the cap and 128
/// state rows want more. Below the cap it is exactly one warp per row:
/// `D = 16` gives 512.
///
/// The shared request is [`kda_shmem`]'s and is the step's: 1 536 bytes at
/// `D = 128`, under the 48 KiB default cap, so no opt-in.
///
/// # Safety
///
/// As [`kda_recurrent_step_batched`], plus `qo_indptr` addressing `r + 1`
/// live `u32`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn kda_prefill_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    if d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        kda::raw::kda_prefill_batched(
            "ssm::kda_prefill_batched#prefill",
            Launch {
                grid: [r.unsigned_abs(), h.unsigned_abs(), 1],
                block: [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(kda_shmem(d.unsigned_abs())),
            q_norm,
            k_norm,
            v,
            gate,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            h,
            d,
            stream,
        );
    }
    Fired::Launched
}

// ── ssm/nemotron_h ───────────────────────────────────────────────────────

/// `ssm::nemotron_prepare_mamba_params` — widen `A_log`, `D` and `dt_bias`
/// once per layer.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::Elementwise`, which is [`elementwise`], over
/// `Source::Param(0)` — `num_heads`. Three tables of `num_heads` entries
/// widened by one thread each.
///
/// # Safety
///
/// The three inputs must address `num_heads` live bf16 elements each and the
/// three outputs `num_heads` writable floats each, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn nemotron_prepare_mamba_params(
    a_log: *const bf16,
    d: *const bf16,
    dt_bias: *const bf16,
    a: *mut f32,
    d_f32: *mut f32,
    dt_bias_f32: *mut f32,
    num_heads: i32,
    stream: *mut c_void,
) -> Fired {
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    unsafe {
        nemotron_h::raw::prepare_mamba_params(
            "ssm::nemotron_prepare_mamba_params",
            elementwise(num_heads.unsigned_abs()),
            a_log,
            d,
            dt_bias,
            a,
            d_f32,
            dt_bias_f32,
            num_heads,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_prepare_mamba_dt_da` — softplus `dt` and precompute
/// `dA = exp(dt * A)`.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::Elementwise` over `Source::OutElements(0)`, and `dt_out` is
/// `[n, num_heads]`, so the rule's element count and the kernel's `total`
/// are the same product. **`n` does not cross; `total` does.** This function
/// takes `n` and multiplies, exactly as the deleted launcher did, so that
/// the product exists in one place.
///
/// # Safety
///
/// `dt` must address `n * num_heads` live bf16 elements, `a` and `dt_bias`
/// `num_heads` live floats, `dt_out` and `da_out` `n * num_heads` writable
/// floats each, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn nemotron_prepare_mamba_dt_da(
    dt: *const bf16,
    a: *const f32,
    dt_bias: *const f32,
    dt_out: *mut f32,
    da_out: *mut f32,
    n: i32,
    num_heads: i32,
    time_step_min: f32,
    stream: *mut c_void,
) -> Fired {
    let total = n.saturating_mul(num_heads);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * num_heads" });
    }
    unsafe {
        nemotron_h::raw::prepare_mamba_dt_da(
            "ssm::nemotron_prepare_mamba_dt_da",
            elementwise(total.unsigned_abs()),
            dt,
            a,
            dt_bias,
            dt_out,
            da_out,
            total,
            num_heads,
            time_step_min,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::zamba_rmsnorm_gated_bf16` — the gated output RMSNorm Zamba closes a
/// Mamba block with.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::GatedRms`, which is [`gated_rms`] and which was NAMED after
/// this kernel: `grid [rows, n_groups, 1]`, `block [256, 1, 1]`. The second
/// axis is the group count, which the deleted row bound from
/// `Source::Param(2)`, and the group WIDTH the kernel divides by is
/// `hidden / n_groups`.
///
/// **The 256 is a static `__shared__ float buf[256]` and moves in neither
/// direction.** See the declaration's doc.
///
/// # Safety
///
/// `x` and `y` must address `rows * hidden` live/writable bf16 elements,
/// `gate` `rows * gate_stride`, `weight` `hidden`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn zamba_rmsnorm_gated_bf16(
    x: *const bf16,
    gate: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    hidden: i32,
    gate_stride: i32,
    n_groups: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    if n_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_groups" });
    }
    unsafe {
        nemotron_h::raw::zamba_rmsnorm_gated(
            "ssm::zamba_rmsnorm_gated_bf16",
            gated_rms(rows.unsigned_abs(), n_groups.unsigned_abs()),
            x,
            gate,
            weight,
            y,
            hidden,
            gate_stride,
            hidden / n_groups,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_mamba_split_bf16` — the three-way cut of the fused
/// projection, in whichever of its two shapes `gate`'s presence asks for.
///
/// `nemotron_h.cu:34-54`, both arms:
///
/// ```text
/// :34   const int total = N * projection_dim;
/// :35   if (total <= 0) return;
/// :38   if (gate == nullptr) {
/// :39     const int conv_dt_total = N * (conv_dim + num_heads);
/// :40     const int conv_dt_grid = (conv_dt_total + BLOCK - 1) / BLOCK;
/// :41     device::mamba_split_conv_dt<<<conv_dt_grid, BLOCK, 0, stream>>>(...)
/// :48   const int grid = (total + BLOCK - 1) / BLOCK;
/// :49   device::mamba_split<<<grid, BLOCK, 0, stream>>>(...)
/// ```
///
/// **The product is tested in both arms even though only one launches on
/// it.** That is the launcher's own reading and it is kept: a fire with no
/// rows or no projection has nothing to cut either way.
///
/// # The one refusal in this family that was under a branch
///
/// `conv_dt_total <= 0` sat INSIDE the `gate == nullptr` arm. §5.1's rule for
/// multi-launch bodies — *"a multi-launch body must resolve every refusal
/// condition before its first launch"* — does not strictly bind here, because
/// this body fires once whichever arm it takes and so cannot return after a
/// launch. It is hoisted anyway, and hoisted CONDITIONALLY:
///
/// ```text
/// let ungated = gate.is_null();          // the discriminant, resolved first
/// if total <= 0                { ... }   // both arms
/// if ungated && conv_dt <= 0   { ... }   // the ungated arm's own
/// ```
///
/// An UNCONDITIONAL hoist would have been wrong and is the interesting part:
/// `conv_dim + num_heads` can be zero or negative with `total > 0`, and the
/// gated arm launches in that case today. Hoisting a refusal out of an arm
/// it does not belong to invents one, which §5.1 forbids in the same breath
/// as it demands the hoist. The `&&` is how both hold.
///
/// # Safety
///
/// `projected` is `[n, projection_dim]` bf16; `conv_in` and `dt` are writable
/// for `[n, conv_dim]` and `[n, num_heads]`; `gate` is writable for
/// `[n, intermediate]` or null. All live on `stream`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_split_bf16(
    projected: *const c_void,
    gate: *mut c_void,
    conv_in: *mut c_void,
    dt: *mut c_void,
    n: i32,
    projection_dim: i32,
    intermediate: i32,
    conv_dim: i32,
    num_heads: i32,
    stream: *mut c_void,
) -> Fired {
    let ungated = gate.is_null();
    // `nemotron_h.cu:34-35` — the product, both arms.
    let total = n.saturating_mul(projection_dim);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * projection_dim" });
    }
    // `:39` — the ungated arm's own extent, hoisted but still that arm's.
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * (conv_dim + num_heads)" });
    }
    if ungated {
        unsafe {
            nemotron_h::raw::mamba_split_conv_dt(
                "ssm::nemotron_mamba_split_bf16#conv_dt",
                Launch {
                    grid: [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                    block: [SPLIT_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                projected,
                conv_in,
                dt,
                projection_dim,
                intermediate,
                conv_dim,
                num_heads,
                conv_dt_total,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        nemotron_h::raw::mamba_split(
            "ssm::nemotron_mamba_split_bf16#split",
            Launch {
                grid: [total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                block: [SPLIT_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            projected,
            gate,
            conv_in,
            dt,
            projection_dim,
            intermediate,
            conv_dim,
            num_heads,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::nemotron_mamba_ssm_batched_bf16` — the selective scan, over `r`
/// requests' token runs found through `qo_indptr`.
///
/// `nemotron_h.cu:119-172`:
///
/// ```text
/// :119  if (R <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0)
/// :119      return;
/// :123  constexpr int PREFILL_BLOCK = 512;
/// :124  const int num_warps = PREFILL_BLOCK / 32;
/// :125  dim3 grid(R, num_heads, (head_dim + num_warps - 1) / num_warps);
/// :126  const size_t shared = 2ull * state_size * sizeof(float);
/// :164  dim3 grid(R, num_heads);            // decode, block 256
/// :166  const size_t shared = 2ull * state_size * sizeof(float);
/// ```
///
/// # Two live arms and the discriminant is the rectangle
///
/// `sequence_prefill` is not a mode flag a caller invents: the deleted row
/// bound it from `Source::Ne(&Source::Rows, &Source::Attn("num_requests"))`
/// — **a fire carrying more rows than requests IS a prefill.** The bind
/// below re-derives it from the same two facts.
///
/// The 16 in the third grid axis is `PREFILL_BLOCK / 32`, the block's warp
/// count, spelled as that division rather than as a literal because it is one
/// warp per `head_dim` row: change the block and the axis must follow. The
/// deleted `fire/nemotron_h.rs` had a test asserting exactly that identity
/// and it survives as this sentence, because the two numbers are now written
/// once each in one expression.
///
/// **`state_size` is not covered by any rectangle check.** It sizes the
/// shared allocation and a zero there is a legal launch of a kernel with no
/// state, so it is refused explicitly.
///
/// # Safety
///
/// `conv_out` and `dt` are bf16 over the token run; `a`, `d` and `dt_bias`
/// are `[num_heads]` fp32; `ssm_state_base` is a slot arena; `slot_ids` is
/// `[r]`; `qo_indptr` is `[r + 1]`; `y` is writable for the token run. All
/// live on `stream`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba_ssm_batched_bf16(
    conv_out: *const c_void,
    dt: *const c_void,
    a: *const f32,
    d: *const f32,
    dt_bias: *const f32,
    dt_precomputed: MaybeConst<f32>,
    da_precomputed: MaybeConst<f32>,
    ssm_state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    y: *mut c_void,
    r: i32,
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    n_groups: i32,
    conv_dim: i32,
    intermediate: i32,
    time_step_min: f32,
    sequence_prefill: bool,
    stream: *mut c_void,
) -> Fired {
    // `nemotron_h.cu:119`, split so the caller learns which extent was empty.
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    if state_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "state_size" });
    }
    // `:126-127` and `:166-167` — the same expression in both arms.
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());
    if sequence_prefill {
        unsafe {
            nemotron_h::raw::mamba_ssm_batched_prefill_reg(
                "ssm::nemotron_mamba_ssm_batched_bf16#prefill_reg",
                Launch {
                    grid: [
                        rows,
                        heads,
                        head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP),
                    ],
                    block: [SSM_PREFILL_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(smem),
                conv_out,
                dt,
                a,
                d,
                dt_bias,
                dt_precomputed,
                da_precomputed,
                ssm_state_base,
                slot_ids,
                qo_indptr,
                y,
                num_heads,
                head_dim,
                state_size,
                n_groups,
                conv_dim,
                intermediate,
                time_step_min,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        nemotron_h::raw::mamba_ssm_batched_warp(
            "ssm::nemotron_mamba_ssm_batched_bf16#warp",
            Launch {
                grid: [rows, heads, 1],
                block: [SSM_DECODE_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(smem),
            conv_out,
            dt,
            a,
            d,
            dt_bias,
            dt_precomputed,
            da_precomputed,
            ssm_state_base,
            slot_ids,
            qo_indptr,
            y,
            num_heads,
            head_dim,
            state_size,
            n_groups,
            conv_dim,
            intermediate,
            time_step_min,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16` — one thread per
/// route, filling six device pointer arrays for a pair of batched GEMMs.
///
/// `nemotron_h.cu:75-94`:
///
/// ```text
/// :75   const int routes = N * top_k;
/// :76   if (routes <= 0) return;
/// :77   constexpr int BLOCK = 256;
/// :78   const int blocks = (routes + BLOCK - 1) / BLOCK;
/// :79   device::build_nemotron_moe_ptrs_decode_batched<<<blocks, BLOCK, 0,
/// :79       stream>>>(..., routes, top_k, hidden, intermediate);
/// ```
///
/// **`routes` is computed once and used twice** — it opens the grid AND it is
/// the kernel's bound, whose parameter is named `total`. Forwarding `n` there
/// would build a `top_k`-th of the pointer arrays and leave the rest whatever
/// the arena held. The device row's operand is called `total` for that
/// reason and this function keeps the product in one expression.
///
/// # Safety
///
/// `topk_idx` is `[n, top_k]` i32 and `topk_w` `[n, top_k]` f32;
/// `up_weight_ptrs`/`down_weight_ptrs` are host-filled device arrays of at
/// least `num_experts` pointers; the six output arrays hold at least
/// `n * top_k` pointers each; `weights_out` is writable for `n * top_k` f32;
/// `expert_up`, `expert_act` and `expert_out` are the decode intermediates.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_decode_batched_bf16(
    topk_idx: *const i32,
    topk_w: *const f32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    norm_x: *const c_void,
    expert_up: *mut c_void,
    expert_act: *mut c_void,
    expert_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    weights_out: *mut f32,
    n: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) -> Fired {
    // `:75-76`.
    let routes = n.saturating_mul(top_k);
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows * top_k" });
    }
    unsafe {
        nemotron_h::raw::build_nemotron_moe_ptrs_decode_batched(
            "ssm::build_nemotron_moe_ptrs_decode_batched_dev_bf16",
            Launch {
                // `:78` — `(routes + 256 - 1) / 256`.
                grid: [routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                block: [PTRS_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            topk_idx,
            topk_w,
            up_weight_ptrs,
            down_weight_ptrs,
            norm_x,
            expert_up,
            expert_act,
            expert_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            weights_out,
            // `routes`, not `n`. See the paragraph above.
            routes,
            top_k,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::build_nemotron_moe_ptrs_aligned_dev_bf16` — one thread per padded
/// block of the sorted MoE layout.
///
/// `nemotron_h.cu:117-137`:
///
/// ```text
/// :117  if (max_blocks <= 0 || block_size <= 0 || hidden <= 0 ||
/// :118      intermediate <= 0) {
/// :119    return;
/// :120  }
/// :120  constexpr int BLOCK = 256;
/// :121  const int blocks = (max_blocks + BLOCK - 1) / BLOCK;
/// :122  device::build_nemotron_moe_ptrs_aligned<<<blocks, BLOCK, 0, stream>>>
/// ```
///
/// **Four guard terms and only the first is about the grid.** `block_size`,
/// `hidden` and `intermediate` are MULTIPLIERS inside the kernel's address
/// arithmetic — a zero in any of them collapses a stride and aliases every
/// block's pointer onto the same row — so the launcher refused all four and
/// so does this. The `moe/moe_dispatch.cu` twin guards only `max_blocks`; the
/// difference is transcribed rather than reconciled, because reconciling it
/// would be inventing a refusal in one of the two.
///
/// Unlike that twin there is no shared-expert branch here: Nemotron-H's MoE
/// has no shared expert, so no operand is rewritten and this is a single
/// launch behind a guard.
///
/// # Safety
///
/// `expert_ids` is `[max_blocks]` i32; the two weight-pointer arrays are
/// device arrays of at least `num_experts` pointers; the six output arrays
/// hold at least `max_blocks` pointers each; the three aligned buffers are
/// the padded rectangles at `block_size * max_blocks` rows.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn build_nemotron_moe_ptrs_aligned_bf16(
    expert_ids: *const i32,
    up_weight_ptrs: *const *const c_void,
    down_weight_ptrs: *const *const c_void,
    aligned_in: *const c_void,
    aligned_up: *mut c_void,
    aligned_act: *mut c_void,
    aligned_out: *mut c_void,
    a_up_ptrs: *mut *const c_void,
    b_up_ptrs: *mut *const c_void,
    c_up_ptrs: *mut *mut c_void,
    a_down_ptrs: *mut *const c_void,
    b_down_ptrs: *mut *const c_void,
    c_down_ptrs: *mut *mut c_void,
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) -> Fired {
    // `:117-120`, all four terms, split so the caller learns which.
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "max_blocks" });
    }
    if block_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "block_size" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    if intermediate <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    unsafe {
        nemotron_h::raw::build_nemotron_moe_ptrs_aligned(
            "ssm::build_nemotron_moe_ptrs_aligned_dev_bf16",
            Launch {
                // `:121` — `(max_blocks + 256 - 1) / 256`.
                grid: [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
                block: [PTRS_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            expert_ids,
            up_weight_ptrs,
            down_weight_ptrs,
            aligned_in,
            aligned_up,
            aligned_act,
            aligned_out,
            a_up_ptrs,
            b_up_ptrs,
            c_up_ptrs,
            a_down_ptrs,
            b_down_ptrs,
            c_down_ptrs,
            max_blocks,
            block_size,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

// ── ssm/gated_delta_net ──────────────────────────────────────────────────

/// The head width at which [`recurrent_step_batched_gqa_state_bf16`] takes
/// its shared-memory arm.
///
/// `gated_delta_net.cu:246` — `if (V_d == 128 && K_d == 128)`. It is the same
/// number as [`SMEM_BV`] and as [`GDN_BLOCK`] and as [`BV_FLA`] and they are
/// four constants and not one: this is a PREDICATE on the caller's shape,
/// `SMEM_BV` is a tile width, `GDN_BLOCK` is a block width and `BV_FLA` is a
/// different tile width in a different kernel. `families/ssm.rs` recorded
/// that all four spell 128 today and that collapsing them would couple four
/// independent decisions; the four names are that finding, kept.
const GDN_SMEM_ARM_WIDTH: i32 = 128;

/// The five extents the prefill entry points and their two bodies share.
///
/// A struct and not five positional `i32`s because they are all `i32` and all
/// adjacent: transposing a pair stops being a compile error and starts being
/// a wrong grid. `k_h` is `0` for the cached pair, which do not take one.
///
/// Carried over verbatim from the deleted `fire/gated_delta_net.rs`, which is
/// where that reasoning was written down.
#[cfg(feature = "_cuda")]
#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

/// The operands the four prefill entry points share.
///
/// **Generic in the state's element type**, which is the one thing that
/// changed in the move. The archive spelled `state_base` as `*mut c_void` for
/// both dtypes and let the row say which; here the `unit!` stub is generic
/// over `S` and the SYMBOL says which, so the shared body must be generic too
/// or the two dtypes could not share it. `S` is `f32` or `c_void`, and
/// `c_void` is `state_bf16` — NOT the prelude's `device::bf16`, which
/// `gated_delta_net.cuh:143` is explicit is a different type.
#[cfg(feature = "_cuda")]
struct Operands<S> {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut S,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
    write_state: bool,
}

/// The body of both `chunk_prefill_batched*` entry points.
///
/// `gated_delta_net.cu:303-356` and `:358-408` are the same forty lines twice
/// with one template argument changed; they are one function here, taking the
/// two symbols. The arms' **operand lists** are identical between the fp32
/// and bf16 forms — only the state's element type differs, and that is in the
/// symbol, not in the call.
///
/// # The nine-fold, and where it lives
///
/// The FLA arm is the reason this family was worth porting: **47.5 ms to
/// 5.3 ms per layer**, bit-identical output, by giving each `BV_FLA`-wide
/// slice of `V_d` its own block instead of walking the value dimension inside
/// one. The per-token arm is the fallback for shapes that do not divide.
///
/// # Every refusal is above both launches
///
/// This is the body §5.1 named — *"`gated_delta_net`'s loop is where this is
/// most likely to bite"* — and the answer is that it does not bite: the arm
/// test `k_d <= BK_MAX_FLA && v_d % BV_FLA == 0` reads two HOST scalars that
/// the caller already had before either kernel existed. Nothing here depends
/// on a device-side output, so there is no refusal that cannot be hoisted and
/// none is left below a `fire`.
///
/// # `k_h` is deliberately NOT guarded
///
/// The four-term refusal is `r`, `v_h`, `k_d`, `v_d`. `k_h` is absent from it
/// and reaches the FLA kernel, which divides by it. A zero divides by zero on
/// the device. That is `gated_delta_net.cu:305`'s own reading and it is
/// reproduced rather than repaired: adding the fifth term would change which
/// fires are refused, and this port is not the place to decide that. It is
/// written down here so the next reader knows it is a transcription and not
/// an oversight.
#[cfg(feature = "_cuda")]
unsafe fn chunk_prefill<S>(
    fla: &'static str,
    per_token: &'static str,
    ops: Operands<S>,
    shape: Shape,
    stream: *mut c_void,
) -> Fired
where
    *mut S: crate::x::Abi,
{
    let Shape { r, k_h, v_h, k_d, v_d } = shape;
    // `gated_delta_net.cu:305`, split so the caller learns which extent was
    // empty. All four, above both arms.
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        // `gated_delta_net.cu:326-336`:
        //
        //     const int NV = V_d / BV_FLA;
        //     dim3 grid_fla(NV, R, V_h);
        //     dim3 block_fla(BV_FLA);
        //     const int shmem_bytes_fla = 2 * BK_MAX_FLA * sizeof(float);
        //
        // The shared size is fixed at `2 * 128 * 4 = 1024` — it is `BK_MAX`,
        // the BOUND on `K_d`, not `K_d` itself, so it does not shrink for a
        // narrow head.
        unsafe {
            gated_delta_net::raw::chunk_gated_delta_prefill_batched_fla(
                fla,
                Launch {
                    grid: [v_d.unsigned_abs() / BV_FLA, rows, heads],
                    block: [BV_FLA, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
                ops.q_norm,
                ops.k_norm,
                ops.v,
                ops.g_log,
                ops.beta,
                ops.state_base,
                ops.slot_ids,
                ops.qo_indptr,
                ops.slot_stride_elems,
                ops.out,
                k_h,
                v_h,
                k_d,
                v_d,
                ops.write_state,
                ops.commit_len,
                ops.write_state_mask,
                stream,
            );
        }
        return Fired::Launched;
    }
    // `gated_delta_net.cu:337-354`, the `KLast = false` instantiation:
    //
    //     constexpr int BLOCK = 128;
    //     dim3 grid(R, V_h);
    //     dim3 block(BLOCK);
    //     const int shmem_bytes = 2 * K_d * sizeof(float);
    //
    // FOUR fewer operands than the FLA arm, and no `K_h`: this kernel is not
    // GQA-aware and does not express a partial state commit.
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched(
            per_token,
            Launch {
                grid: [rows, heads, 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(2 * k_d.unsigned_abs() * FLOAT),
            ops.q_norm,
            ops.k_norm,
            ops.v,
            ops.g_log,
            ops.beta,
            ops.state_base,
            ops.slot_ids,
            ops.qo_indptr,
            ops.slot_stride_elems,
            ops.out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// The body of both `chunk_prefill_batched_cached*` entry points.
///
/// One kernel, no switch: the prefill that holds the **whole** `K_d × V_d`
/// state tile in shared memory for the length of the token run.
///
/// # `K_d * V_d * sizeof(float)` is 64 KiB at production shapes
///
/// `128 * 128 * 4 = 65_536`, against a **48 KiB** default per-block dynamic
/// limit. A launch asking for more without opting in fails with
/// `CUDA_ERROR_INVALID_VALUE` — not a wrong answer, a hard failure at the
/// first fire. [`Launch::smem`] is what sets `smem_opt_in` here, and it is
/// the reason this body writes `.smem(...)` where a zero-shared body writes
/// the field: the opt-in flag is derived from the number, so it cannot be
/// forgotten and cannot disagree with it.
///
/// The C++ opted in through a file-local `gdn_raise_shmem_cap`
/// (`gated_delta_net.cu:80-108`): a `cudaFuncSetAttribute` with
/// `cudaFuncAttributeMaxDynamicSharedMemorySize`, guarded by a
/// function-local `static int high_water` so the driver call happened once
/// per growth rather than once per fire. That helper is not reproduced: it
/// lives in `runtime::module::raise_dynamic_smem_cap`, called from the fire
/// whenever the request exceeds the default, because a JIT'd kernel has a
/// `CUfunction` and not a `__global__` address, because the high-water mark
/// must be keyed on `(device, function)` and no launcher sees the device, and
/// because **every** kernel above 48 KiB needs it rather than just this one.
///
/// # No `commit_len`
///
/// This kernel takes `write_state` and `write_state_mask` and **not**
/// `commit_len` — the state it writes is the one it has been holding, so
/// there is no partial commit to express.
///
/// # The refusal, and it is load-bearing twice
///
/// `if (R <= 0 || V_h <= 0 || K_d <= 0 || V_d <= 0) return;` —
/// `gated_delta_net.cu:416`. `K_d` and `V_d` size the shared request, and a
/// zero there would ask the driver to raise a cap to nothing.
#[cfg(feature = "_cuda")]
unsafe fn cached<S>(
    symbol: &'static str,
    ops: Operands<S>,
    shape: Shape,
    stream: *mut c_void,
) -> Fired
where
    *mut S: crate::x::Abi,
{
    let Shape { r, v_h, k_d, v_d, .. } = shape;
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_cached(
            symbol,
            // `gated_delta_net.cu:419-422`:
            //
            //     constexpr int BLOCK = 128;
            //     dim3 grid(R, V_h);
            //     dim3 block(BLOCK);
            //     const int shmem_bytes = K_d * V_d * sizeof(float);
            Launch {
                grid: [r.unsigned_abs(), v_h.unsigned_abs(), 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
            ops.q_norm,
            ops.k_norm,
            ops.v,
            ops.g_log,
            ops.beta,
            ops.state_base,
            ops.slot_ids,
            ops.qo_indptr,
            ops.slot_stride_elems,
            ops.out,
            v_h,
            k_d,
            v_d,
            ops.write_state,
            ops.write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched#{fla,per_token}` — fp32 state.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `state_base` addresses
/// `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d` writable floats for
/// every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
    stream: *mut c_void,
) -> Fired {
    unsafe {
        chunk_prefill(
            "ssm::chunk_gated_delta_prefill_batched#fla",
            "ssm::chunk_gated_delta_prefill_batched#per_token",
            Operands {
                q_norm,
                k_norm,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                out,
                commit_len,
                write_state_mask,
                write_state,
            },
            Shape { r, k_h, v_h, k_d, v_d },
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_state_bf16#{fla,per_token}` — the
/// same two kernels over a bf16 state slab.
///
/// # Safety
///
/// As [`chunk_prefill_batched`], with `state_base` addressing that many
/// writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_state_bf16(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    commit_len: MaybeConst<i32>,
    write_state_mask: MaybeConst<u8>,
    stream: *mut c_void,
) -> Fired {
    unsafe {
        chunk_prefill(
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#fla",
            "ssm::chunk_gated_delta_prefill_batched_state_bf16#per_token",
            Operands {
                q_norm,
                k_norm,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                out,
                commit_len,
                write_state_mask,
                write_state,
            },
            Shape { r, k_h, v_h, k_d, v_d },
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem` — fp32
/// state, whole tile resident.
///
/// **Takes no `K_h`**: the kernel is not GQA-aware and requires the expanded
/// layout. [`Shape::k_h`] is `0` here and unused.
///
/// # Safety
///
/// As [`chunk_prefill_batched`], minus `commit_len`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_cached(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: MaybeConst<u8>,
    stream: *mut c_void,
) -> Fired {
    unsafe {
        cached(
            "ssm::chunk_gated_delta_prefill_batched_cached#state_in_smem",
            Operands {
                q_norm,
                k_norm,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                out,
                commit_len: MaybeConst::none(),
                write_state_mask,
                write_state,
            },
            Shape { r, k_h: 0, v_h, k_d, v_d },
            stream,
        )
    }
}

/// `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem` —
/// the same kernel over a bf16 state slab. **This is the row that asks for
/// 64 KiB.**
///
/// # Safety
///
/// As [`chunk_prefill_batched_cached`], with a bf16 state slab.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_prefill_batched_cached_state_bf16(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: MaybeConst<u8>,
    stream: *mut c_void,
) -> Fired {
    unsafe {
        cached(
            "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16#state_in_smem",
            Operands {
                q_norm,
                k_norm,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                qo_indptr,
                slot_stride_elems,
                out,
                commit_len: MaybeConst::none(),
                write_state_mask,
                write_state,
            },
            Shape { r, k_h: 0, v_h, k_d, v_d },
            stream,
        )
    }
}

/// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#{smem,hbm}` — one
/// delta-rule step per request, GQA layout, bf16 state.
///
/// # The 34 %, and the shape it was measured at
///
/// This is the row §30 of the archive was written about. The `#smem` arm
/// stages the `K_d × BV` state tile in shared memory and reads each key row
/// once instead of `V_d / BLOCK` times:
///
/// ```text
/// R = 511, K_d = V_d = 128, V_h = 16, K_h = 2
/// hbm   2 406 us
/// smem  1 579 us          -34 %
/// end to end on Qwen3.5-4B: 6 924 -> 9 166 tok/s, +32 %
/// ```
///
/// Output was **byte-identical across all eight shapes tested**, 535 822 336
/// bytes compared. That is why the arm is taken whenever it fits rather than
/// behind a flag: `PIE_QWEN35_GDN_SMEM_STEP` was the env var that gated it
/// during the bring-up and it is **deleted, not ported** — a knob whose only
/// correct setting is on is not a knob.
///
/// # Both arms, both cited
///
/// ```text
/// gated_delta_net.cu:248-252   smem
///     constexpr int BV = 128;
///     dim3 grid_smem((V_d + BV - 1) / BV, R, V_h);
///     dim3 block_smem(BV);
///     shmem = K_d * BV * sizeof(__nv_bfloat16) + 2 * K_d * sizeof(float)
///
/// gated_delta_net.cu:284-287   hbm
///     constexpr int BLOCK = 128;
///     dim3 grid(R, V_h);
///     dim3 block(BLOCK);
///     shmem = (2 * K_d + (fused ? 1 : 0)) * sizeof(float)
/// ```
///
/// with `fused` constant-false, so the `+ 1` never appears. At `K_d = 128`
/// the `#smem` request is `128 * 128 * 2 + 2 * 128 * 4 = 33 792` bytes —
/// under the 48 KiB default, so this arm needs no opt-in even though the
/// cached prefill above it does.
///
/// # The discriminant is a pair of host scalars
///
/// `v_d == 128 && k_d == 128`, not a device value, so both the arm choice and
/// every refusal are settled before anything is launched.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable `__nv_bfloat16` elements for every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_step_batched_gqa_state_bf16(
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    // Not an extent but a ratio: the GQA fan-out must be whole or the head
    // map is not a map. `Narrow` carries the numerator that failed.
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    if v_d == GDN_SMEM_ARM_WIDTH && k_d == GDN_SMEM_ARM_WIDTH {
        unsafe {
            gated_delta_net::raw::recurrent_step_batched_gqa_smem(
                "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#smem",
                Launch {
                    grid: [v_d.unsigned_abs().div_ceil(SMEM_BV), r.unsigned_abs(), v_h.unsigned_abs()],
                    block: [SMEM_BV, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
                q_norm_kh,
                k_norm_kh,
                v,
                g_log,
                beta,
                state_base,
                slot_ids,
                slot_stride_elems,
                out,
                k_h,
                v_h,
                k_d,
                v_d,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched_gqa(
            "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16#hbm",
            Launch {
                grid: [r.unsigned_abs(), v_h.unsigned_abs(), 1],
                block: [GDN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(2 * k_d.unsigned_abs() * FLOAT),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched` — one delta-rule step per
/// (request, value head), fp32 state, EXPANDED layout.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was in `device::JIT_DISPATCHED` with `LaunchRule::RecurrentScan`,
/// so no `.cu` launcher existed to move. [`recurrent_scan`] is that rule and
/// it is `grid [r, v_h, 1]`, `block [128, 1, 1]`, `smem = 2 * k_d * 4` —
/// the same rectangle and the same shared expression the `#hbm` arm of
/// [`recurrent_step_batched_gqa_state_bf16`] reaches by hand, which is the
/// check that the rule was a description of this kernel and not a guess.
///
/// **No `k_h`.** This kernel is not GQA-aware; the caller must have expanded
/// the key heads already, which is what `ssm::repeat_interleave_heads_fp32`
/// is for.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `state_base` addresses `slot_ids[i] * slot_stride_elems + v_h * k_d * v_d`
/// writable floats for every `i < r`.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched(
            "ssm::recurrent_gated_delta_step_batched",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched_state_bf16` — the same kernel
/// over a bf16 state slab.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`recurrent_scan`], as [`recurrent_gated_delta_step_batched`]. The two
/// rows differ only in the state's element type, which is in the symbol and
/// therefore in the instantiation, and in nothing the host computes.
///
/// # Safety
///
/// As [`recurrent_gated_delta_step_batched`], with `state_base` addressing
/// that many writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched_state_bf16(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched(
            "ssm::recurrent_gated_delta_step_batched_state_bf16",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm,
            k_norm,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::recurrent_gated_delta_step_batched_gqa` — the GQA step, fp32 state.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`recurrent_scan`] again. **This is the fp32 twin of the `#hbm` arm of
/// [`recurrent_step_batched_gqa_state_bf16`]** and has no `#smem` sibling:
/// only the bf16 state was given one, because the shared tile it stages is
/// `K_d * BV * sizeof(__nv_bfloat16)` and the fp32 form of the same tile is
/// twice that, `65 536` at production widths, which is the opt-in cliff.
/// That asymmetry is in the archive and is reproduced, not repaired.
///
/// # Safety
///
/// As [`recurrent_gated_delta_step_batched`], plus `q_norm_kh` and
/// `k_norm_kh` addressing `k_h`-head rather than `v_h`-head rectangles.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_step_batched_gqa(
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::recurrent_step_batched_gqa(
            "ssm::recurrent_gated_delta_step_batched_gqa",
            recurrent_scan(r.unsigned_abs(), v_h.unsigned_abs(), k_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa` — the warp-tiled
/// GQA prefill, fp32 state.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::WarpTiledScan`, which is [`warp_tiled_scan`] — `grid [r, v_h,
/// ceil(v_d / 4)]`, `block [128, 1, 1]`, no shared memory. **The 4 is the
/// block's warp count and appears twice**, once as the third grid axis's
/// divisor and once inside the block width, so the two must move together.
/// The rule spells `128` as `SCAN_WARPS * WARP` for that reason and so does
/// this file.
///
/// **The value width the third axis divides is `v_d` and NOT `k_d`.**
/// `families/ssm.rs` recorded this as the row's hazard: the rule's parameter
/// is called a value width, the two are equal at every shape shipped today,
/// and a `k_d` there would launch a `k_d/v_d` fraction of the value tiles and
/// silently leave the rest of `out` unwritten.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch;
/// `qo_indptr` addresses `r + 1` live `u32`; `write_state_mask` addresses `r`
/// live bytes or is null.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: *const u8,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_warp_tiled_gqa(
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            write_state,
            write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` — the
/// same kernel over a bf16 state slab.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`warp_tiled_scan`], as
/// [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`].
///
/// # Safety
///
/// As [`chunk_gated_delta_prefill_batched_warp_tiled_gqa`], with `state_base`
/// addressing writable `__nv_bfloat16` elements instead of floats.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    q_norm_kh: *const f32,
    k_norm_kh: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    write_state: bool,
    write_state_mask: *const u8,
    stream: *mut c_void,
) -> Fired {
    if r <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if k_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_h" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    if k_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_d" });
    }
    if v_d <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_d" });
    }
    if v_h % k_h != 0 {
        return Fired::Declined(Refusal::Narrow { what: "v_h per k_h", at: v_h });
    }
    unsafe {
        gated_delta_net::raw::chunk_gated_delta_prefill_batched_warp_tiled_gqa(
            "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
            warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs()),
            q_norm_kh,
            k_norm_kh,
            v,
            g_log,
            beta,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out,
            k_h,
            v_h,
            k_d,
            v_d,
            write_state,
            write_state_mask,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// One small declaration serves the readers that cannot call.
//
// Twenty-seven contracts, one per deleted `table/ssm.rs` row, and the only
// thing a `Contract` says that a `fn` cannot: the symbol a trace names, the
// `dsl::cuda` wrapper that names it, and the four dispatch facts —
// `whole`, `publishes_aux`, `needs`, `lacks` — that the compiler and the
// join read before any fire exists.
//
// **`Contract::DEFAULT` plus `..`**, so a row states only what it changes.
// Across twenty-seven rows that is five `whole`s and three `publishes_aux`
// lists; `table/ssm.rs` stated nothing else beyond the defaults, which is
// itself the measurement that made this block short.
// ---------------------------------------------------------------------------

contract! {
    /// The three-way cut of Nemotron-H's fused in-projection:
    /// `[N, projection_dim] -> gate, conv_in, dt`.
    ///
    /// **Every extent comes off an operand**, which is unusual for this
    /// module: a three-way cut states all three destinations, so the widths
    /// that say where the cuts fall are the results' own. Nothing here needs
    /// the GDN context that blocks the rest of `ssm`.
    ///
    /// `model-compiler/src/dsl.rs` names it `nemotron_mamba_split`.
    NEMOTRON_MAMBA_SPLIT = "ssm::nemotron_mamba_split_bf16" as nemotron_mamba_split {
        // The raw `dt` this cut produces is the join's slot 0 — the `dt_raw`
        // that `nemotron_mamba_ssm` reads as `Source::Aux(0)`. Stated here so
        // the join never has to know this kernel's name.
        publishes_aux: &[(0, 2)],
    }

    /// `A_log`, `D` and `dt_bias` widened to fp32, once per layer.
    ///
    /// **The only row in this module that needs nothing from the GDN context
    /// BUT a scalar** — no slab, no aux operand, no attention context. Which
    /// is why it is the one `Source::Gdn` got on its own: the rest of `ssm`
    /// wants per-layer state slabs and operands the trace does not state, and
    /// naming a field does not reach those.
    NEMOTRON_PREPARE_MAMBA_PARAMS = "ssm::nemotron_prepare_mamba_params"
        as nemotron_prepare_mamba_params
    {
        // The three fp32 tables, at the join's slots 1..3 — `a`, `d` and
        // `dt_bias` in the order `Source::Aux(1)`, `Aux(2)` and `Aux(3)` read
        // them.
        publishes_aux: &[(1, 0), (2, 1), (3, 2)],
    }

    /// `dt` softplussed and `dA = exp(dt * A)` precomputed, per (token, head).
    ///
    /// Reads `Source::Aux(3)` — the `dt_bias_f32` that
    /// [`NEMOTRON_PREPARE_MAMBA_PARAMS`] published — and publishes two of its
    /// own, which is the whole reason this family needs an aux channel at
    /// all: three statements in a row, each reading the last one's fp32
    /// tables, none of which the trace names.
    NEMOTRON_PREPARE_MAMBA_DT_DA = "ssm::nemotron_prepare_mamba_dt_da"
        as nemotron_prepare_mamba_dt_da
    {
        publishes_aux: &[(4, 0), (5, 1)],
    }

    /// The selective scan: a `[head_dim, state_size]` slab per head, advanced
    /// by a scalar `dA` from a per-token `dt`.
    ///
    /// **The third linear-attention shape here, and not a variant of the
    /// other two.** GDN and KDA carry a delta rule; mamba carries a different
    /// state SHAPE, which is why none of the GDN or KDA rows stands in for
    /// it.
    ///
    /// `whole` because it walks token runs out of `qo_indptr` and the
    /// recurrence has a strict per-token state dependency: a row window would
    /// start the scan from the wrong state, which is a different answer
    /// rather than a misaddressed one.
    NEMOTRON_MAMBA_SSM = "ssm::nemotron_mamba_ssm_batched_bf16" as nemotron_mamba_ssm {
        whole: true,
    }

    /// KDA's gate and beta activations, per (token, head).
    KDA_GATE_BETA = "ssm::kda_gate_beta_bf16" as kda_gate_beta

    /// KDA's decode step: one delta-rule step per (request, head).
    ///
    /// `whole` because the recurrence is sequential in the token index.
    KDA_RECURRENT_STEP = "ssm::kda_recurrent_step_batched" as kda_recurrent_step {
        whole: true,
    }

    /// KDA's prefill: the same recurrence over a whole region.
    ///
    /// **`whole` twice over**: it walks windows out of `qo_indptr`, AND the
    /// recurrence has a strict per-token state dependency — a row window
    /// would start the scan from the wrong state, which is a different answer
    /// rather than a misaddressed one.
    KDA_PREFILL = "ssm::kda_prefill_batched" as kda_prefill {
        whole: true,
    }

    /// KDA's gated output norm: the recurrence's fp32 output, the gate, one
    /// weight, one bf16 result.
    ///
    /// `h` and `d` ride the PARAM channel — the result is `[Tokens, h * d]`
    /// and only their product is a shape, so neither factor can be recovered
    /// from a width and the statement must state both.
    KDA_O_NORM_GATED = "ssm::kda_o_norm_gated_bf16" as kda_o_norm_gated

    /// The decode short convolution: one step per request, advancing each
    /// one's `[K, C]` ring in the conv slab.
    GDN_CONV_UPDATE = "ssm::causal_conv1d_update_batched_bf16" as gdn_conv_update

    /// The prefill short convolution, over `qo_indptr`'s token runs.
    GDN_CONV_PREFILL = "ssm::causal_conv1d_prefill_batched_bf16" as gdn_conv_prefill

    /// The GDN decode step, fp32 state, expanded head layout.
    GDN_STEP = "ssm::recurrent_gated_delta_step_batched" as gdn_step

    /// The GDN decode step, fp32 state, GQA layout.
    GDN_STEP_GQA = "ssm::recurrent_gated_delta_step_batched_gqa" as gdn_step_gqa

    /// The GDN decode step, bf16 state, expanded head layout.
    ///
    /// **`r` comes off `Source::Rows` here and off
    /// `Source::Attn("num_requests")` on its three siblings.** The two are the
    /// same number for a decode step — a decode's rectangle IS its requests —
    /// and the archive spelled them differently anyway. Transcribed as it
    /// stood: the bind below reads `cx.rows().count` for this one contract
    /// and `cx.plan()?.requests` for the others, and if that ever diverges it
    /// is a fact about the trace and not about this file.
    GDN_STEP_STATE_BF16 = "ssm::recurrent_gated_delta_step_batched_state_bf16"
        as gdn_step_state_bf16

    /// The GDN decode step, bf16 state, GQA layout. **The 34 % row** — see
    /// [`recurrent_step_batched_gqa_state_bf16`] for the measurement and for
    /// which of its two arms carries it.
    GDN_STEP_GQA_STATE_BF16 = "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"
        as gdn_step_gqa_state_bf16

    /// The GDN chunked prefill, fp32 state. **The nine-fold row** — see
    /// [`chunk_prefill`].
    GDN_PREFILL_FLA = "ssm::chunk_gated_delta_prefill_batched" as gdn_prefill_fla

    /// The GDN chunked prefill, bf16 state.
    GDN_PREFILL_FLA_STATE_BF16 = "ssm::chunk_gated_delta_prefill_batched_state_bf16"
        as gdn_prefill_fla_state_bf16

    /// The GDN prefill that holds the whole state tile in shared memory, fp32
    /// state.
    GDN_PREFILL_CACHED = "ssm::chunk_gated_delta_prefill_batched_cached"
        as gdn_prefill_cached

    /// The same, bf16 state. **This is the row that asks for 64 KiB** and so
    /// the row that made `KernelModule::fire` raise the dynamic shared cap —
    /// see [`cached`].
    GDN_PREFILL_CACHED_STATE_BF16 =
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
        as gdn_prefill_cached_state_bf16

    /// The warp-tiled GQA prefill, fp32 state.
    GDN_PREFILL_WARP_TILED_GQA = "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
        as gdn_prefill_warp_tiled_gqa

    /// The warp-tiled GQA prefill, bf16 state.
    GDN_PREFILL_WARP_TILED_GQA_STATE_BF16 =
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
        as gdn_prefill_warp_tiled_gqa_state_bf16

    /// Fan `K_h` key heads out to `V_h` value heads, fp32.
    ///
    /// The operand is `[Tokens, key_heads, key_dim]` and the repeated result
    /// is `[Tokens, value_heads, key_dim]`, so all three counts are dims the
    /// statement already carries. It states its result since the repeat
    /// stopped being output-less.
    REPEAT_INTERLEAVE_HEADS = "ssm::repeat_interleave_heads_fp32" as repeat_interleave_heads

    /// Row-wise L2 norm with a scale, widening bf16 to fp32.
    ///
    /// KDA's arithmetic is fp32 throughout, so operands living in bf16 in the
    /// workspace cross explicitly. It launches, so the trace records it.
    /// The two scalars are the KDA convention — an unscaled L2 norm at the
    /// context's epsilon.
    L2NORM_SCALE_TO_F32 = "ssm::l2norm_scale_bf16_to_fp32" as l2norm_scale_to_f32

    /// bf16 to fp32, whole buffer. **The first row whose every argument the
    /// statement already carries**: one operand, one result, and an element
    /// count that is the result's own extent.
    BF16_TO_F32 = "ssm::bf16_to_fp32" as bf16_to_f32

    /// fp32 to bf16, on the same terms.
    F32_TO_BF16 = "ssm::fp32_to_bf16" as f32_to_bf16

    /// Zamba's gated output RMSNorm.
    ///
    /// **`hidden` and `gate_stride` are two widths and not one.** The gate may
    /// be a window into a wider fused projection, so the row read
    /// `Source::Width(&Source::In(0))` and `Source::Width(&Source::In(1))`
    /// separately, and the bind reads two `in_width`s.
    ///
    /// `group_size` is `hidden / n_groups` and `n_groups` is the only thing
    /// this row needs from the GDN context.
    ZAMBA_RMSNORM_GATED = "ssm::zamba_rmsnorm_gated_bf16" as zamba_rmsnorm_gated

    /// The aligned-batch MoE pointer build for Nemotron-H's expert GEMMs.
    ///
    /// `whole` — it is one thread per padded BLOCK of a counting sort's
    /// output, and a row window of a sort's output is not a sort's output.
    BUILD_NEMOTRON_MOE_PTRS_ALIGNED = "ssm::build_nemotron_moe_ptrs_aligned_bf16"
        as build_nemotron_moe_ptrs_aligned
    {
        whole: true,
    }

    /// The decode MoE pointer build: one thread per ROUTE.
    ///
    /// `whole`, and `total` is `n * top_k` — see
    /// [`build_nemotron_moe_ptrs_decode_batched_bf16`] for why forwarding `n`
    /// there is the easiest mistake in this family.
    BUILD_NEMOTRON_MOE_PTRS_DECODE = "ssm::build_nemotron_moe_ptrs_decode_batched_bf16"
        as build_nemotron_moe_ptrs_decode
    {
        whole: true,
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Twenty-three binds and FOUR `none:` arms, which is the highest ratio of
// refusals to binds in the tree so far and is a fact about this family rather
// than about this port: `table/ssm.rs` left four rows with EVERY operand
// unsourced, and an unsourced operand is a promise no `Source` can keep.
//
// The four are `kda_recurrent_step`, `kda_prefill`,
// `build_nemotron_moe_ptrs_aligned` and `build_nemotron_moe_ptrs_decode`, and
// they fail for the same reason twice over — §52.3's missing
// `Source::Scratch(name, extent)`. Their sentences say so in the words the
// deleted rows used, because §5.1 is right that the prose beside an unsourced
// operand is already the user-facing sentence.
//
// A `none:` arm here is NOT the `adapter` shape: nothing else fires these
// symbols. `ssm::causal_conv1d_prefill_noact_bf16` IS the adapter shape and
// therefore has no contract at all in this block — `tower/gemma4_audio.rs`
// fires it by string and a `Route::Unbound` on it would be a lie.
//
// EVERY BIND BELOW HOISTS. §5.1's multi-launch rule is satisfied twice: once
// because the host programs resolve their own refusals above their own
// launches, and once because a bind body's `?`s all run before it calls one.
// The two conv binds and the two split binds are the only bodies here whose
// `fn` can pick between two kernels, and every one of them picks after every
// refusal.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    NEMOTRON_MAMBA_SPLIT => { cx, stream => {
        // The deleted row's ten sources, in order: `In(0)`, `Out(0..2)`,
        // `Rows`, `InWidth(0)`, `OutWidth(0..2)`. The only row in this family
        // that reaches no context at all.
        unsafe {
            mamba_split_bf16(
                cx.arg_in(0)?.cast_const(),
                cx.arg_out(0)?,
                cx.arg_out(1)?,
                cx.arg_out(2)?,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.out_width(1)?,
                cx.out_width(2)?,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_PREPARE_MAMBA_PARAMS => { cx, stream => {
        // `num_heads` off `Source::Gdn("v_h")` — mamba's head count lives in
        // the GDN context beside the delta-rule families' because the driver
        // computes them in one place, not because it is a GDN kernel.
        let gdn = cx.gdn()?;
        unsafe {
            nemotron_prepare_mamba_params(
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.weight(2)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.arg_out(2)?.cast::<f32>(),
                gdn.v_h,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_PREPARE_MAMBA_DT_DA => { cx, stream => {
        // `dt_bias` is `Source::Aux(3)` — the third table
        // `NEMOTRON_PREPARE_MAMBA_PARAMS` published, which no trace names.
        // `time_step_min` was `Source::Lit(Lit::F32(0.0))`.
        unsafe {
            nemotron_prepare_mamba_dt_da(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.aux(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                0.0,
                stream,
            )
        }
        .ok()
    }},

    NEMOTRON_MAMBA_SSM => { cx, stream => {
        // The widest bind in this family: five aux slots, a state slab, the
        // attention plan and eight GDN scalars. `sequence_prefill` was
        // `Source::Ne(&Source::Rows, &Source::Attn("num_requests"))` — a fire
        // carrying more rows than requests IS a prefill — and it is re-derived
        // from the same two facts here, not passed in.
        //
        // `intermediate` was `Source::Mul(&Gdn("v_h"), &Gdn("v_d"))`.
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        let rows = cx.rows().count;
        unsafe {
            mamba_ssm_batched_bf16(
                cx.arg_in(0)?.cast_const(),
                cx.aux(0)?.cast_const(),
                cx.aux(1)?.cast_const().cast::<f32>(),
                cx.aux(2)?.cast_const().cast::<f32>(),
                cx.aux(3)?.cast_const().cast::<f32>(),
                MaybeConst::new(cx.arg_in(1)?.cast_const().cast::<f32>()),
                MaybeConst::new(cx.aux(5)?.cast_const().cast::<f32>()),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                plan.qo_indptr,
                cx.arg_out(0)?,
                plan.requests,
                gdn.v_h,
                gdn.v_d,
                gdn.k_d,
                gdn.n_groups,
                gdn.conv_dim,
                gdn.v_h.saturating_mul(gdn.v_d),
                0.0,
                rows != plan.requests,
                stream,
            )
        }
        .ok()
    }},

    KDA_GATE_BETA => { cx, stream => {
        // `h` off `Source::OutWidth(1)` — `beta_out` is `[Tokens, h]`, so the
        // head count IS that result's width — and `d` off `Source::Param(0)`,
        // because `gate_out` is `[Tokens, h * d]` and only the product is a
        // shape. `lower_bound` was `Source::Lit(Lit::F32(0.0))`.
        let d = i32::try_from(cx.param(0)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head dim",
        })?;
        unsafe {
            kda_gate_beta_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.weight(1)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(1)?,
                d,
                0.0,
                stream,
            )
        }
        .ok()
    }},

    KDA_RECURRENT_STEP => { none: "kda_recurrent_step needs a KDA state \
        arena and the per-request slot ids that index it; a trace states \
        neither, and no operand source names a driver-allocated slab" },

    KDA_PREFILL => { none: "kda_prefill needs a KDA state arena, the \
        per-request slot ids that index it, and the query-offset plan the \
        driver assembles between statements; a trace states none of them" },

    KDA_O_NORM_GATED => { cx, stream => {
        // `h` and `d` both ride the param channel: the result is
        // `[Tokens, h * d]` and only their product is a shape.
        let h = i32::try_from(cx.param(0)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head count",
        })?;
        let d = i32::try_from(cx.param(1)?).map_err(|_| Refusal::Unstated {
            what: "the KDA head dim",
        })?;
        unsafe {
            kda_o_norm_gated_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                h,
                d,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    GDN_CONV_UPDATE => { cx, stream => {
        // `bias` was `Source::WeightSuffix("_bias")` and is NULLABLE: the
        // device text says `// [C] nullable` at `causal_conv1d.cuh:383` and
        // the kernel tests it at `:397`. `Cx::weight_bias()` answers `None`
        // rather than refusing, because an absent bias is a fact about a
        // checkpoint and not a missing operand.
        let gdn = cx.gdn()?;
        let bias = cx.weight_bias().map_or_else(
            MaybeConst::none,
            |p| MaybeConst::new(p.cast_const().cast::<bf16>()),
        );
        unsafe {
            causal_conv1d_update_batched_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                bias,
                cx.slab(Slab::Conv)?.cast::<bf16>(),
                gdn.slot_ids,
                gdn.conv_stride_elems,
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                gdn.conv_dim,
                gdn.conv_k,
                stream,
            )
        }
        .ok()
    }},

    GDN_CONV_PREFILL => { cx, stream => {
        // `r` off the plan and not off `Rows`, which is the prefill's whole
        // point: the rectangle is tokens and the grid axis is requests.
        // `commit_len` and `write_state_mask` were both `Lit::Null` — the
        // partial-commit path exists in the kernel and no statement reaches
        // it, so they cross absent rather than being dropped.
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        let bias = cx.weight_bias().map_or_else(
            MaybeConst::none,
            |p| MaybeConst::new(p.cast_const().cast::<bf16>()),
        );
        unsafe {
            causal_conv1d_prefill_batched_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                bias,
                cx.arg_out(0)?.cast::<bf16>(),
                cx.slab(Slab::Conv)?.cast::<bf16>(),
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.conv_stride_elems,
                plan.requests,
                gdn.conv_dim,
                gdn.conv_k,
                stream,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
            )
        }
        .ok()
    }},

    GDN_STEP => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_GQA => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched_gqa(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_STATE_BF16 => { cx, stream => {
        // `cx.rows().count` and NOT `cx.plan()?.requests` — see the
        // contract's doc. The archive spelled this one row's `r` as
        // `Source::Rows` and its three siblings' as
        // `Source::Attn("num_requests")`; the two agree for a decode step and
        // the difference is transcribed rather than reconciled.
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_gated_delta_step_batched_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.rows().count,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_STEP_GQA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        unsafe {
            recurrent_step_batched_gqa_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                cx.plan()?.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_FLA => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_FLA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_CACHED => { cx, stream => {
        // No `k_h`: the kernel is not GQA-aware and requires the expanded
        // layout. No `commit_len`: the state it writes is the one it has
        // been holding.
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_cached(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_CACHED_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_prefill_batched_cached_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                MaybeConst::none(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_WARP_TILED_GQA => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?.cast::<f32>(),
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                core::ptr::null(),
                stream,
            )
        }
        .ok()
    }},

    GDN_PREFILL_WARP_TILED_GQA_STATE_BF16 => { cx, stream => {
        let gdn = cx.gdn()?;
        let plan = cx.plan()?;
        unsafe {
            chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_in(4)?.cast_const().cast::<f32>(),
                cx.slab(Slab::Recurrent)?,
                gdn.slot_ids,
                plan.qo_indptr,
                gdn.state_stride_elems,
                cx.result(0)?.cast::<f32>(),
                plan.requests,
                gdn.k_h,
                gdn.v_h,
                gdn.k_d,
                gdn.v_d,
                gdn.write_state,
                core::ptr::null(),
                stream,
            )
        }
        .ok()
    }},

    REPEAT_INTERLEAVE_HEADS => { cx, stream => {
        // `k_h`, `v_h` and `d` are `Gdn("k_h")`, `Gdn("v_h")` and
        // `Gdn("v_d")` — on Metal they were `OutDim(0, 1)` and `OutDim(0, 2)`,
        // where a value's dims are the binder's to read. Here they are the
        // same two numbers from the place that computes them.
        let gdn = cx.gdn()?;
        unsafe {
            repeat_interleave_heads_fp32(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.result(0)?.cast::<f32>(),
                cx.rows().count,
                gdn.k_h,
                gdn.v_h,
                gdn.v_d,
                stream,
            )
        }
        .ok()
    }},

    L2NORM_SCALE_TO_F32 => { cx, stream => {
        // `scale` was `Source::Lit(Lit::F32(1.0))` — an UNSCALED L2 norm, the
        // KDA convention — and `eps` the context's.
        unsafe {
            l2norm_scale_bf16_to_fp32(
                cx.arg_in(0)?.cast_const(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                1.0,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    BF16_TO_F32 => { cx, stream => {
        // `n` was `Source::OutElements(0)`, which is `rows * out_width(0)`.
        // Both halves are already queries, so no fifth `Cx` method was asked
        // for: the product is written here, once, in the two binds that need
        // it.
        let n = usize::try_from(cx.rows().count)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
        unsafe {
            bf16_to_fp32(cx.arg_in(0)?.cast_const(), cx.arg_out(0)?.cast::<f32>(), n, stream)
        }
        .ok()
    }},

    F32_TO_BF16 => { cx, stream => {
        let n = usize::try_from(cx.rows().count)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
        unsafe {
            fp32_to_bf16(cx.arg_in(0)?.cast_const().cast::<f32>(), cx.arg_out(0)?, n, stream)
        }
        .ok()
    }},

    ZAMBA_RMSNORM_GATED => { cx, stream => {
        // TWO widths and not one: `hidden` is `Width(&In(0))` and
        // `gate_stride` is `Width(&In(1))`, because the gate may be a window
        // into a wider fused projection. `group_size` is
        // `Div(&Width(&In(0)), &Gdn("n_groups"))` and the host program does
        // that division, so `n_groups` crosses here and the quotient there.
        let gdn = cx.gdn()?;
        unsafe {
            zamba_rmsnorm_gated_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.in_width(1)?,
                gdn.n_groups,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    BUILD_NEMOTRON_MOE_PTRS_ALIGNED => { none: "build_nemotron_moe_ptrs_aligned \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the padded block layout a counting sort produced; a trace states \
        none of them and no operand source names a scratch slab" },

    BUILD_NEMOTRON_MOE_PTRS_DECODE => { none: "build_nemotron_moe_ptrs_decode \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the decode intermediates the MoE path allocates between \
        statements; a trace states none of them" },
}
