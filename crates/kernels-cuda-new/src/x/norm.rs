//! `norm` — RMSNorm, the fused landings, AltUp's rank-K residual and
//! deepseek-v4's hyper-connections, as two truths.
//!
//! The device text is `csrc/src/norm/{rmsnorm,add_bias,altup,altup_aux,
//! elementwise,dsv4_hc}.cuh`, unchanged. This file is the other truth:
//! thirty-five `__global__` instantiations over six roots, thirty-two host
//! programs, twenty-eight contracts and the binds that connect them. §5 step 5,
//! and the family §5.1 named as the first proof of `Composed`/`Walk`.
//!
//! # What this replaces
//!
//! ```text
//!   before                                               lines
//!   kernels-cuda-new/src/families/norm.rs  6 units, 35 rows  1756
//!   kernels-cuda-new/src/table/norm.rs            26 rows     421
//!   driver-cuda/src/fire/rmsnorm.rs         4 host programs   465
//!   driver-cuda/src/fire/dsv4_hc.rs         4 host programs   287
//!   driver-cuda/src/bind/service.rs          6 wrappers      ~270
//!                                                          ------
//!                                                            3199
//!   after
//!   kernels-cuda-new/src/x/norm.rs   32 host programs, 35 device
//!                                     rows, 28 contracts, 15 binds,
//!                                     13 refusals
//! ```
//!
//! Eight of the host programs are a MOVE: `fire/rmsnorm.rs` and
//! `fire/dsv4_hc.rs` were already Rust, already fired NVRTC text, and already
//! cited the `.cu` line beside every launch. What changed is their root —
//! they now live beside the `.cuh` they launch, take typed pointers instead
//! of `*mut c_void`, and return [`Fired`] instead of nothing. The other
//! twenty-four are the rule-driven rows' launchers, written from the
//! `LaunchRule` each row stated and cited to `runtime/launch.rs`, which is
//! the arithmetic those rules evaluate to and whose tests pin each one
//! against the `<<<>>>` it was ported from.
//!
//! # THE UNIT ARRANGEMENT: SIX ROOTS, SIX INVOCATIONS, AND THE MACRO TOOK IT
//!
//! §5.1 flagged `unit!`'s one-unit-per-invocation shape as *"the most likely
//! place the macros need real work"*, and `norm` is the family that tests it:
//! `csrc/src/norm/` holds EIGHT `.cuh` roots where `rope` had one. It needed
//! no grammar change. Six of the eight are units, each in an inline `pub mod`
//! whose name is the root's file stem, and [`UNITS`] is written by hand —
//! exactly the shape `unit!`'s own doc describes and `x::mlp` established at
//! two roots.
//!
//! **The other two are not units and never were.** `rmsnorm_tile.cuh` and
//! `rmsnorm_rasr_tile.cuh` are `#include`d only by
//! `csrc/src/tile_alternatives.cuh:366,368`, which is reached only from
//! `tests/upstream_manifest.rs`. No row names a symbol in either, no
//! `include_str!` reads them, and nothing in `families/norm.rs` did. A root
//! is a unit when something launches out of it; these are a comparison the
//! vendor manifest compiles.
//!
//! The qualifier the module wrapper adds is a gain and not a tax:
//! `rmsnorm::raw::rmsnorm_vec8` and `dsv4_hc::raw::hc_post` say which header
//! they came out of at every call site, which a flat file would have had to
//! spell into thirty-five stub names.
//!
//! # `Composed`/`Walk`, PROVEN — AND IT CROSSES A FAMILY BOUNDARY
//!
//! §2.3's two-different-kernels-in-one-body shape is
//! [`rmsnorm_bf16_with_fp16`]'s middle arm, and it is the first one in the
//! tree that is not two of ITS OWN kernels:
//!
//! ```text
//!   unstrided_bf16(...)                    // norm::rmsnorm_strided_bf16
//!   cast_to_fp16(y, y_fp16, n * hidden)    // fires: quant::bf16_to_fp16
//! ```
//!
//! The second launch belongs to `quant`, and `norm` must not declare
//! `quant::bf16_to_fp16` — a `raw::` fn exists only for a `__global__` this
//! family DECLARES, and declaring it here would mint a second definition of a
//! row `x::quant` owns, which is the one thing §0 forbids.
//!
//! **That does not cost the typed path, and the first draft of this file said
//! it did.** [`cast_to_fp16`] calls `quant`'s own stub by its full path:
//!
//! ```text
//! crate::x::quant::dequant_wna16::raw::bf16_to_narrow::<f16>(..)
//! ```
//!
//! A `raw::` stub is not bound to the unit it was declared beside. The
//! expansion takes `symbol`, `launch`, its typed parameters and `stream` and
//! calls `x::fire::fire`, which resolves `unit_of(symbol)` GLOBALLY —
//! `$UNIT` appears nowhere in a stub body, and the module path is Rust
//! namespacing and only that. So a family boundary costs nothing here, and
//! what §2.3 does not cover is smaller and sharper than "the typed path
//! degrades":
//!
//! > **A cross-family call makes the callee's unit a dependency of the
//! > caller's host program, and nothing in the type system says so**, because
//! > `symbol` is a `&'static str`. A missing unit panics at the fire naming
//! > the symbol — right behaviour, wrong time. The remedy is a comment:
//! > `// fires: quant::bf16_to_fp16` beside the call, so the caller is
//! > greppable from the callee.
//!
//! What the floor got RIGHT, and it is the subtle half: [`Fired`]'s rule that
//! a multi-launch body resolves every refusal before its first launch is
//! exactly what the archive's C++ did not do — `rmsnorm.cu:58-68` tested
//! `y_fp16 == nullptr` and `rmsnorm_vec8_ok` before either launch already,
//! and the port's `num_rows <= 0` check is hoisted above both for the first
//! time. The C++ launched a zero grid and let `cudaGetLastError` report it at
//! whatever synchronisation came next.
//!
//! The weaker second instance is [`rmsnorm_residual_add_scale_rmsnorm_bf16`]:
//! three arms, one launch each, chosen by five addresses and a width. It is
//! `Walk`'s other half — a `Specialisation` that no `Term` list can hold,
//! because `hidden_size >= 2560` is a COMPARISON and every `Term` in that
//! vocabulary is unary (`new-horizon.md` §44.6).
//!
//! # A `Launch` FINDING: `LaunchRule::Rms`'s 32 BYTES ARE READ BY NOBODY HERE
//!
//! `Rms` allocates `(BLOCK / WARP) * 4` bytes of dynamic shared memory
//! (`runtime/launch.rs:737-745`) and five `norm` rows stated it. **Not one of
//! `rmsnorm.cuh`'s kernels declares `extern __shared__`**: every reduction in
//! that file folds through a STATIC `__shared__ float buf[BLOCK]`
//! (`rmsnorm.cuh:205, 297, 371, 425, 473, 510, 577, 603, 653, 672, 704, 739,
//! 784`), and so does every one of `dsv4_hc.cuh`'s (`:123-125, 298, 377`).
//! The C++ launchers passed `0`. Only `altup_aux.cuh`'s `compute_rms` and
//! `magnitude_rescale` declare `extern __shared__ float smem[]` (`:97`,
//! `:117`), and those two are the only launches below that carry
//! [`RMS_SMEM`].
//!
//! This is `LaunchRule::Rope`'s generous-allocation bug in miniature and it
//! was harmless — CUDA accepts an unread dynamic allocation — but it is the
//! same class of defect and the same cause: **a rule sized shared memory for
//! a kernel it could not read.** fn-world's structural answer is
//! `x::launch`'s: the author who writes the byte count is the author who
//! reads it in the kernel.
//!
//! # WHAT THE FLOOR COULD NOT EXPRESS, AND WHAT IS REFUSED BECAUSE OF IT
//!
//! Thirteen of the twenty-eight contracts bind facts [`Cx`](crate::x::Cx) has no
//! query for. Each is a `none:` arm carrying the exact `Facts` method it
//! wants, each surfaces at model load through `Route::Unbound` with the
//! family, the symbol and its sentence, and — except for the last three —
//! each is one defaulted method away from being three lines of bind, because
//! the host program above it is complete and takes the value as an ordinary
//! argument.
//!
//! | contract | the row's `Source` | the `Facts` method it needs |
//! |---|---|---|
//! | `norm::rmsnorm_bf16` | `IfPresent(PerHeadDim, …)` | `per_head_dim() -> Option<i32>` |
//! | `norm::rmsnorm_gemma_bf16` | the same | `per_head_dim()` |
//! | `norm::rmsnorm_no_scale_bf16` | the same | `per_head_dim()` |
//! | `norm::rmsnorm_gated_bf16` | the same | `per_head_dim()` |
//! | `norm::rmsnorm_gated_fp32_in_bf16` | `Gdn("v_d")` | `per_head_dim()`, set from `gdn.v_d` |
//! | `norm::altup_predict_bf16` | `Ctx("altup_streams")` | `altup_streams() -> Option<i32>` |
//! | `norm::mean_streams_bf16` | `CtxNonZero("altup_streams")` | `altup_streams()` |
//! | `norm::altup_correct_bf16` | `Ctx("altup_active")` | `altup_active() -> Option<i32>` |
//! | `norm::rmsnorm_residual_add_scale_rmsnorm_bf16` | `LayerScale` | `layer_scale() -> Option<f32>` |
//! | `norm::scalar_mul_bf16` | `Or(ParamF32(0), NamedScale)` | `named_scale() -> Option<f32>` |
//! | `norm::hc_pre_postprocess_bf16` | nothing, on any operand | five slabs nothing carries |
//! | `norm::hc_head_postprocess_bf16` | nothing, on any operand | the same |
//! | `norm::rmsnorm_bf16_with_fp16` | nothing, on any operand | the fp16 destination |
//!
//! `per_head_dim` is `spec.per_head_dim`, filled at `driver-cuda/src/bind/mod.rs:1798`
//! and read by `Source::PerHeadDim`; `altup_streams` and `altup_active` are
//! `DispatchCtx` fields at `bind/mod.rs:1244` and `:1246`; `layer_scale` is
//! `bind/mod.rs:1643` (`spec.weight` ending `ple_norm` indexes `ctx.scales`,
//! else 1.0); `named_scale` is `bind/mod.rs:1613` (`spec.weight`'s
//! `scale.` prefix). Five reads and five defaulted methods.
//!
//! **`x/cx.rs` is the floor's file, so this port declares the refusals rather
//! than editing it** — the convention `x::mlp` set for the same six-of-twelve
//! reason.
//!
//! ## `scalar_mul` is the one refusal that would have been WORSE half-bound
//!
//! Its row is `Source::Or(&Source::ParamF32(0), &Source::NamedScale)`, and
//! `Cx` answers the first arm — `cx.param_f32(0)`. Binding it would work for
//! every model that states the number and silently mis-scale gemma-3n and
//! gemma-2, which state a NAME and no param, and whose fires would then take
//! `Refusal::Absent` at the fire instead of `Route::Unbound` at load. A
//! refusal at load names the model; a refusal at fire names a token.
//!
//! ## HC's two unsourced rows are NOT one method away, and their prose is the
//! deleted file's
//!
//! `fire/dsv4_hc.rs`'s header is the argument, carried verbatim:
//!
//! > HC's mixing matrices are not values a statement names. `mixes`, `scale`
//! > and `base` are three `float` slabs the layer carries, `post_mix` and
//! > `comb_mix` are scratch the launcher hands from one kernel to the next,
//! > and `sinkhorn_iters` and `hc_post_alpha` are model constants. A `Source`
//! > for any of them would be a guess about where a lowering puts a buffer,
//! > and **a half-bound row is worse than an unbound one**.
//!
//! # TWO OF THE FOUR HC ROWS WERE UNSOURCED BY ASSOCIATION, NOT BY NECESSITY
//!
//! `hc_rmsnorm_to_f32` and `hc_post` sat in the same unsourced block as
//! `hc_pre` and `hc_head` and are not in the same position. Their DSL
//! statements carry every operand their kernels need —
//! `dsl.rs:4486` states `[residual]` in and a `[Tokens, width]` fp32 out;
//! `dsl.rs:4552` states `[x, residual, post_mix, comb_mix]` in and a
//! `[Tokens, hc_mult, hidden]` out, from which `M = out_width / in_width(0)`
//! and `H = in_width(0)` — so fn-world binds two of the four rows the row
//! world skipped whole. Nothing was added to the lowering to make that true;
//! the rows were written together and refused together.
//!
//! # `rmsnorm_gated`'s `weight` IS `const float*`
//!
//! `rmsnorm.cuh:721` — the gated norm's weight is fp32 where its activations
//! are bf16. The row could only say `Ty::Buf`, which is documented as an
//! OPAQUE `void*`; the declaration below says `*const f32` and is correct by
//! construction. This is `x::mlp`'s `y_fp16` finding from a second direction,
//! and the third row of the same shape (`rmsnorm_gated_f32_in`, whose `x` is
//! fp32 as well) is why it is worth stating twice.
//!
//! # WHAT `Contract` NEEDED: ONE FIELD OF TEN
//!
//! Ten contracts state `in_place` and no contract states anything else.
//! `whole`, `needs`, `lacks`, `sink`, `depth_prefix_plan`, `publishes_aux`
//! and `lowered_as` stay unexercised after this family, because
//! `table/norm.rs` stated none of them: every kernel here is ROW-SHAPED —
//! token `t`'s output reads only token `t`'s inputs — so a peel may split any
//! of them, none obligates a host plan, and there is no seam capability for
//! one to refuse. The row world already wrote that as a claim rather than an
//! omission, beside the AltUp block, and this port carries it forward
//! unchanged.
//!
//! `Facts::plan()` and `Facts::slab()` get no first caller here either.
//! Nothing in `norm` reads a per-request index array or a gated-delta-net
//! state slab; `attn` and `ssm` remain where they are first exercised.
//!
//! # A `Refusal` VARIANT THAT WAS MISSING, AND IS NOT ANY MORE
//!
//! `Refusal::Wide { what, at, max }` landed with this family and [`hc_mult_ok`]
//! returns it. The argument below is kept because it is why the variant
//! exists, and because the shape it names -- a refusal written backwards
//! because the vocabulary had no word for it -- is the thing to watch for.
//!
//! [`MAX_HC_MULT`] is a precondition the compiled kernel cannot exceed, and
//! `Refusal` has no word for it. `Narrow { what, at }` means BELOW the
//! smallest unit of work; this is ABOVE the largest the instantiation can
//! hold. `fire/dsv4_hc.rs` spelled it `assert!` and documented a `# Panics`;
//! a `fn` reached from a `bind!` body must not panic, because a declined
//! launch is a VALUE in this floor and a panic crosses a boundary that has no
//! word for it. So [`hc_mult_ok`] returns `Narrow { what: "…the compiled
//! ceiling on hc_mult", at: MAX_HC_MULT }`, which reads the sentence
//! backwards — the KERNEL's register array is the narrow thing and eight is
//! the width at which it is narrow. It is true, it is diagnosable, and it is
//! not what the field means.
//!
//! **The variant that is wanted is `Refusal::Wide { what, at, max }`**, and it
//! is not a one-family need: any kernel with a compile-time bound on a runtime
//! extent — a head dim, a stream count, a tile — reaches for it. It is the
//! floor's to add.

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
use core::ptr::NonNull;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// SIX ROOTS, SIX INVOCATIONS. `unit!` emits `UNITS`, `ROWS`, `PARAMS` and
// `mod raw` at the invocation site, so each root gets a module of its own and
// this file re-exports the six as [`UNITS`]. `rope` had one root and never met
// this; `mlp` has two and recorded it; `norm` is the family §5.1 said would
// decide whether the macro needed real work, and it did not.
// ---------------------------------------------------------------------------

/// `norm/rmsnorm.cuh` — twelve `__global__` templates, fifteen
/// instantiations, and the only root in this family whose host programs were
/// already Rust.
pub mod rmsnorm {
    use super::{bf16, f16};
    use core::ptr::NonNull;

    unit! {
        /// The RMSNorm family proper: the scalar kernels whose launcher was
        /// `<<<rows, 256>>>` and nothing else, the vectorised twins that read
        /// eight bf16 per thread, and the three-pass gemma-4 landing.
        ///
        /// Twelve `__global__` templates and one `__device__` reduction, and
        /// nothing else: no host function, no `<<<>>>`, no entry point. The
        /// host program is the `fn`s below.
        unit RMSNORM = "norm/rmsnorm",
            text = include_str!("../../csrc/src/norm/rmsnorm.cuh"),
            file = "norm/rmsnorm.cuh";

        /// `rmsnorm.cuh:220` — `y = x * rsqrt(mean(x^2) + eps) * w`, with
        /// each side's row stride stated.
        ///
        /// TWO SYMBOLS, ONE INSTANTIATION, and they are not redundant.
        /// `norm::rmsnorm_strided_bf16` is what `rmsnorm.cu`'s launcher fired
        /// and what a `dsl::cuda::rmsnorm` statement lowers to;
        /// `norm::rmsnorm_bf16` is the symbol the SEMANTIC `OpKind::Rmsnorm`
        /// fans to, whose weight may arrive by name rather than in a slot and
        /// whose row count may be per-head. One instantiation serves both
        /// because the difference is entirely in the host program, which is
        /// the whole of what "the strides are the two values' own widths"
        /// meant in `table/norm.rs`.
        fn rmsnorm = "norm::device::rmsnorm" <T> (
            x: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_strided_bf16" => where [T = bf16] "device::bf16, 256",
            "norm::rmsnorm_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:236` — the same with `(1 + w)` folded instead of `w`.
        ///
        /// Different arithmetic, same signature, same row space: gemma stores
        /// its norm weights centred on zero.
        fn rmsnorm_gemma = "norm::device::rmsnorm_gemma" <T> (
            x: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gemma_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:262` — eight contiguous bf16 per thread, one 16-byte
        /// load, with an optional fp16 copy of the same result.
        ///
        /// **The measurement, from `rmsnorm.cuh:249-255`:** *"At decode
        /// `num_rows` is 1, so the kernel is a single block on a 148-SM GPU
        /// and its cost is entirely the length of the per-thread dependent
        /// load chain: at hidden=7168 the scalar form walked 28 loads per
        /// thread, twice. Vectorized it is 4 (BLOCK=512), and measured device
        /// time dropped ~7x (3.48 -> 2.38 us against a 2.20 us empty-launch
        /// floor)."*
        ///
        /// **`y_fp16` is `f16*` and this declaration says so.** The row world
        /// could only spell `Ty::BufMut`, an opaque `void*` whose element
        /// comes from the ROW's `elem` — which is `bf16` on every one of
        /// these four rows and is not true of this parameter.
        /// `families/norm.rs` guarded it with a `Take::Null` and a paragraph
        /// explaining that an edit sourcing this slot from `y` "would
        /// type-check in Rust, bind in the driver, and write half-width data
        /// into a bf16 buffer at legal addresses". Here it does not
        /// type-check.
        ///
        /// `Option<NonNull<f16>>` and not `*mut f16` because the two
        /// `EMIT_FP16=false` instantiations are passed `nullptr` by their
        /// launcher and read the parameter only inside a dead
        /// `if constexpr` — nullable on two rows, required on the other two,
        /// which is a distinction the row world drew with `BufMut | null` on
        /// one row and `BufMut` on its twin.
        ///
        /// # THE KNOWN DEFECT, WHICH THIS PORT DOES NOT FIX
        ///
        /// `rmsnorm.cuh:318` writes `y_fp16[i * 8 + j]` with NO row offset, so
        /// `EMIT_FP16=true` is wrong above one row.
        /// `tests/launch_rules.rs::the_emit_fp16_kernel_is_wrong_above_one_row`
        /// pins the signature so it cannot be renamed away, and
        /// `tests/specialise.rs` names `norm::rmsnorm_strided_bf16#vec8`. A
        /// port that quietly corrected it would make these symbols compute
        /// something the archive did not, on a path a golden was recorded
        /// over. The defect is the kernel's and stays the kernel's.
        ///
        /// # Why four rows and not two
        ///
        /// 256 and 512 are the SAME template at two block widths, and both
        /// are load-bearing. `BLOCK` sizes the `__shared__ float[BLOCK]` that
        /// `block_reduce_sum_exact` folds through, so an instantiation
        /// compiled at 512 and launched at 256 folds 256 floats no thread
        /// wrote — finite, plausible and wrong. The 256 pair exists because
        /// `LaunchRule::Rms` launched 256 threads and `RMSNORM_STRIDED_VEC8`
        /// chose between the scalar row and the `#vec8` row at that width;
        /// the 512 pair is the width the LAUNCHER fired
        /// (`rmsnorm.cu:88-94`, `:69-77` — `constexpr int VBLOCK = 512`).
        ///
        /// **fn-world removes the reason the 256 pair had to exist** — a host
        /// program states its own block width, so [`super::strided_bf16`]
        /// fires the 512 rows and the `Rms` constraint binds nothing. The 256
        /// rows stay because two tests fire them directly:
        /// `tests/launch_rules.rs:8231-8372` on
        /// `norm::rmsnorm_bf16_with_fp16#vec8` and `tests/specialise.rs:53,419`
        /// on `norm::rmsnorm_strided_bf16#vec8`. Deleting an instantiation
        /// two tests name is not a port.
        ///
        /// `device::i32(256)` and not `256`: `DeviceKernel::instantiation`
        /// prefixes the FIRST template argument with
        /// `::pie_cuda_driver::kernels::`, and this template's first
        /// parameter is `int BLOCK` — so a bare literal lands as
        /// `::pie_cuda_driver::kernels::256` and NVRTC answers *expected an
        /// identifier*. The prelude's `i32` alias (`pie_device.cuh:463`)
        /// makes it a qualified constant expression that resolves where the
        /// prefix puts it. `crate::device::args` records the eight forms this
        /// was measured over.
        fn rmsnorm_vec8 = "norm::device::rmsnorm_vec8" (
            x: *const bf16,
            weight: *const bf16,
            y: *mut bf16,
            y_fp16: Option<NonNull<f16>>,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) {
            "norm::rmsnorm_strided_bf16#vec8" => "device::i32(256), false, false",
            "norm::rmsnorm_bf16_with_fp16#vec8" => "device::i32(256), false, true",
            "norm::rmsnorm_strided_bf16#vec8_512" => "device::i32(512), false, false",
            "norm::rmsnorm_bf16_with_fp16#vec8_512" => "device::i32(512), false, true",
        }

        /// `rmsnorm.cuh:401` — the residual add and the NEXT block's pre-norm,
        /// fused.
        ///
        /// Numerically the two-kernel sequence: the kernel matches
        /// `residual_add`'s bf16 rounding before norming, which is what makes
        /// it a binding a declaration may state rather than a different
        /// computation. `hidden` is read AND written.
        fn residual_add_rmsnorm = "norm::device::residual_add_rmsnorm" <T> (
            hidden: *mut T,
            residual: *const T,
            weight: *const T,
            norm_out: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::residual_add_rmsnorm_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:492` — norm `x`, then add the result into the
        /// residual stream in place.
        fn rmsnorm_residual_add = "norm::device::rmsnorm_residual_add" <T> (
            x: *const T,
            weight: *const T,
            hidden: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_residual_add_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:548` — gemma-4's landing: residual add, scale, and
        /// the next block's norm, vectorised.
        ///
        /// `template <int BLOCK>` with `bf16` hard-coded in its own
        /// parameters, exactly as `rmsnorm_vec8` is — so `elem` is a VALUE on
        /// both rows and `device::i32(...)` for the reason
        /// [`rmsnorm_vec8`](fn@rmsnorm_vec8) gives.
        ///
        /// **The two widths are a MEASUREMENT and not a convention.** Swept
        /// under graph replay at the shapes these models use, in us:
        ///
        /// ```text
        ///   hidden   scalar256  scalar512   vec256  vec512  vec1024
        ///     2048        4.38       3.68     2.72    2.93     3.31
        ///     2816        6.17       4.83     3.46    3.12     3.51
        ///     5376        8.48       6.55     4.44    4.07     4.02
        /// ```
        ///
        /// vec512 above hidden 2560 and vec256 below: best at 2816, within
        /// 1.5% of best at 5376, and 2048 prefers the narrower block. Both are
        /// bit-identical to the scalar form at all three sizes — only the two
        /// sum reductions reassociate, and at these lengths that rounds to the
        /// same bf16. The scalar form measured **10.79 us/call in
        /// gemma-4-26B's decode, 8% of the step**; against the shipping
        /// scalar/256 the vectorised arms are −38%, −49% and −53%.
        fn rmsnorm_rasr_vec8 = "norm::device::rmsnorm_rasr_vec8" (
            x: *const bf16,
            weight: *const bf16,
            hidden: *mut bf16,
            scale: f32,
            next_weight: *const bf16,
            norm_out: *mut bf16,
            hidden_size: i32,
            eps: f32,
        ) {
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_512" => "device::i32(512)",
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_256" => "device::i32(256)",
        }

        /// `rmsnorm.cuh:631` — the same three passes, scalar.
        ///
        /// The arm taken when the five addresses do not all land on 16 bytes,
        /// and it is 512 wide at EVERY width because the unaligned path is
        /// the one the sweep above could not improve: `scalar512` beats
        /// `scalar256` at all three sizes. `template <class T, int BLOCK>`, so
        /// `elem` reads like the scalar rows.
        fn rmsnorm_residual_add_scale_rmsnorm =
            "norm::device::rmsnorm_residual_add_scale_rmsnorm" <T> (
            x: *const T,
            weight: *const T,
            hidden: *mut T,
            scale: f32,
            next_weight: *const T,
            norm_out: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#scalar_512" =>
                where [T = bf16] "device::bf16, 512",
        }

        /// `rmsnorm.cuh:686` — the weightless per-head norm, the V-norm.
        ///
        /// No gamma, so no `WEIGHT_PLUS_ONE` variant.
        fn rmsnorm_no_scale = "norm::device::rmsnorm_no_scale" <T> (
            x: *const T,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_no_scale_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:718` — norm gated by a second activation, with an
        /// **fp32** weight.
        ///
        /// `weight` is `const float* __restrict__` (`rmsnorm.cuh:721`) where
        /// `x`, `gate` and `y` are `T`. See this module's header: the row
        /// could only say `Ty::Buf` and the declaration says `*const f32`.
        fn rmsnorm_gated = "norm::device::rmsnorm_gated" <T> (
            x: *const T,
            gate: *const T,
            weight: *const f32,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gated_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:763` — the same with an fp32 INPUT as well.
        ///
        /// Two of the four pointers are fp32 and two are `T`, which is the
        /// signature the row world had no way to write down.
        fn rmsnorm_gated_f32_in = "norm::device::rmsnorm_gated_f32_in" <T> (
            x: *const f32,
            gate: *const T,
            weight: *const f32,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gated_fp32_in_bf16" => where [T = bf16] "device::bf16, 256",
        }
    }
}

/// `norm/add_bias.cuh` — one `__device__` row body, two `__global__`s over
/// it, one instantiated.
pub mod add_bias {
    use super::bf16;

    unit! {
        /// Both launchers were the same three lines
        /// (`add_bias.cuh:9-13`): `if (num_rows <= 0 || dim <= 0) return;`
        /// then `kernel<<<num_rows, 256, 0, stream>>>`. Those three lines are
        /// [`super::add_bias_bf16`] now.
        unit ADD_BIAS = "norm/add_bias",
            text = include_str!("../../csrc/src/norm/add_bias.cuh"),
            file = "norm/add_bias.cuh";

        /// `add_bias.cuh:82` — `out[row][i] += bias[i]`, contiguous rows.
        ///
        /// **`add_bias_strided` (`:98`) has no row and keeps none.** The
        /// header's own argument for why both kernels exist is about
        /// `model-compiler` lowering two statements to two symbols; only one
        /// statement exists. A `__global__` with no instantiation costs
        /// nothing — it is a template — and deleting it would delete the
        /// header's paragraph about why the loop is written once as
        /// `add_bias_row`. The unstated half stays device text.
        fn add_bias = "norm::device::add_bias" <T> (
            out: *mut T,
            bias: *const T,
            dim: i32,
        ) where *const T, *mut T {
            "norm::add_bias_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/altup.cuh` — gemma-3n's rank-K residual, predict and correct.
pub mod altup {
    use super::bf16;

    unit! {
        /// Two `__global__` templates. The header records that these rows
        /// were once REFUSED — *"No rows name these kernels, and that is a
        /// measurement, not an oversight"* — because the shape they wanted
        /// was *"one block per (row, group) pair, tiled over the row, which
        /// nobody has written yet"*. Someone wrote it: `LaunchRule::AltUpStreams`,
        /// ported from the launchers, and it is [`super::altup_streams`]
        /// below. The refusal was right and it is now discharged.
        unit ALTUP = "norm/altup",
            text = include_str!("../../csrc/src/norm/altup.cuh"),
            file = "norm/altup.cuh";

        /// `altup.cuh:77` — predict each of `K` streams as a coefficient
        /// combination of all `K`.
        ///
        /// `coefs` is `[T_len, K, K]` fp32, unpacked and TRANSPOSED by
        /// [`super::rmsnorm::raw`]'s sibling
        /// [`altup_aux::raw::unpack_predict_coefs`] before this runs — which
        /// is why the two are separate kernels and separate rows.
        fn altup_predict = "norm::device::altup_predict" <T> (
            streams: *const T,
            coefs: *const f32,
            predictions: *mut T,
            k: i32,
            t_len: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::altup_predict_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup.cuh:106` — correct every stream from the one the layer
        /// actually ran.
        ///
        /// `active_idx` selects that stream; the coefficients arrive already
        /// incremented by one, which is `unpack_correct_coefs`' `v + 1.0f`
        /// (`altup_aux.cuh:183`) and is why the parameter is named
        /// `correction_coefs_plus_one` in the device text.
        fn altup_correct = "norm::device::altup_correct" <T> (
            predictions: *const T,
            activated: *const T,
            correction_coefs_plus_one: *const f32,
            corrected: *mut T,
            k: i32,
            t_len: i32,
            h: i32,
            active_idx: i32,
        ) where *const T, *mut T {
            "norm::altup_correct_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/altup_aux.cuh` — the five auxiliaries AltUp needs around the two
/// above, plus the flat `tanh`.
pub mod altup_aux {
    use super::{bf16, f16};

    unit! {
        /// Six `__global__` templates and one include, and the header is
        /// explicit that it is *"the ONLY C++ these kernels have"*: there is
        /// no `.cu`, no entry point, and the instantiation set is stated by
        /// the rows because `nvrtcAddNameExpression` takes an instantiation
        /// as a STRING. The offline path forces the same set by taking the
        /// address of each instantiation in the generated typecheck TU
        /// (`abi::emit_device_typecheck`) — *"the file that PROVES the rows
        /// is also the file that instantiates them"*.
        ///
        /// This root also has an INDEPENDENT consumer set that this port does
        /// not touch: `device::ALTUP_AUX` (`device.rs:421`) is a Tier A pilot
        /// row read by `kernels-cuda/src/norm_device.rs:45` and
        /// `driver-cuda/src/bind/{nvrtc,launch,device}.rs`. It is not in
        /// `table::TABLES`, so it cannot collide with anything here.
        unit ALTUP_AUX = "norm/altup_aux",
            text = include_str!("../../csrc/src/norm/altup_aux.cuh"),
            file = "norm/altup_aux.cuh";

        /// `altup_aux.cuh:91` — the RMS of each row, to fp32.
        ///
        /// **`ref` is a Rust keyword, so the parameter is `reference`.** The
        /// device text is unchanged and the C++ name is what NVRTC sees;
        /// this rename is visible only in Rust, and it is the second one in
        /// the tree after `x::layout`'s. A `unit!` parameter name is a Rust
        /// binding, not part of the ABI.
        ///
        /// One of exactly TWO kernels in this family that read dynamic
        /// shared memory (`extern __shared__ float smem[]`, `:97`), so one
        /// of exactly two launches that carry [`super::RMS_SMEM`].
        fn compute_rms = "norm::device::compute_rms" <T> (
            reference: *const T,
            out: *mut f32,
            h: i32,
            eps: f32,
        ) where *const T {
            "norm::compute_rms_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:111` — rescale each row to a target RMS, in place.
        ///
        /// The other `extern __shared__` kernel (`:117`).
        fn magnitude_rescale = "norm::device::magnitude_rescale" <T> (
            x: *mut T,
            target_rms: *const f32,
            h: i32,
            eps: f32,
        ) where *mut T {
            "norm::magnitude_rescale_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:141` — the mean of `K` streams at each position.
        ///
        /// `t_stride` is the stream plane's row count and is NOT `T_len`:
        /// the streams live in one `[K, t_stride, H]` buffer that may be
        /// longer than the tokens being read.
        fn mean_streams = "norm::device::mean_streams" <T> (
            streams: *const T,
            out: *mut T,
            k: i32,
            t_stride: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::mean_streams_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:163` — unpack and TRANSPOSE a `[T, K, K]` bf16
        /// coefficient block to fp32.
        ///
        /// `out[t][j][k] = in[t][k][j]` — the transpose is the kernel.
        fn unpack_predict_coefs = "norm::device::unpack_predict_coefs" <T> (
            in_bf16: *const T,
            out: *mut f32,
            k: i32,
        ) where *const T {
            "norm::altup_unpack_predict_coefs" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:178` — unpack a `[T, K]` bf16 block to fp32 and add
        /// one.
        fn unpack_correct_coefs = "norm::device::unpack_correct_coefs" <T> (
            in_bf16: *const T,
            out: *mut f32,
            k: i32,
        ) where *const T {
            "norm::altup_unpack_correct_coefs" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:189` — elementwise `tanh`, in place, over a flat
        /// extent.
        ///
        /// TWO instantiations from one declaration, and the only place in
        /// this family where `f16` is an element rather than a destination
        /// type. Both are named by `table/norm.rs`'s `tanh` contract through
        /// the driver's `dtype` and both are fired by
        /// `execution.rs`'s `RUST_SERVED` path today.
        fn tanh_inplace = "norm::device::tanh_inplace" <T> (
            x: *mut T,
            n: i32,
        ) where *mut T {
            "norm::tanh_bf16" => where [T = bf16] "device::bf16",
            "norm::tanh_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `norm/elementwise.cuh` — the residual add and the scalar multiply.
pub mod elementwise {
    use super::{bf16, f16};

    unit! {
        /// Both launchers were the same four lines (`elementwise.cuh:9-13`):
        /// `if (n == 0) return;`, `blocks = (n + BLOCK - 1) / BLOCK`,
        /// `kernel<<<blocks, BLOCK, 0, stream>>>`. Those four lines are
        /// [`super::residual_add`] and [`super::scalar_mul`] now, and
        /// `Launch::flat(n, BLOCK)` IS `(n + BLOCK - 1) / BLOCK`.
        ///
        /// Like `altup_aux`, this root also backs the Tier A pilot row
        /// `device::ELEMENTWISE` (`device.rs:568`), which this port leaves
        /// alone.
        unit ELEMENTWISE = "norm/elementwise",
            text = include_str!("../../csrc/src/norm/elementwise.cuh"),
            file = "norm/elementwise.cuh";

        /// `elementwise.cuh:56` — `y += x`, elementwise, accumulated in fp32
        /// and rounded once.
        ///
        /// `n` is `usize` in the device text (`pie_device.cuh`'s alias for
        /// `size_t`), so it is `usize` here. The row world spelled this
        /// `Ty::USize` and the two agree; what is new is that the HOST
        /// program cannot pass an `i32` by accident.
        fn residual_add = "norm::device::residual_add" <T> (
            y: *mut T,
            x: *const T,
            n: usize,
        ) where *const T, *mut T {
            "norm::residual_add_bf16" => where [T = bf16] "device::bf16",
            "norm::residual_add_f16" => where [T = f16] "device::f16",
        }

        /// `elementwise.cuh:74` — `x *= s`, with `s` ROUNDED TO `T` FIRST.
        ///
        /// The header is emphatic that *"the rounding is the kernel, not a
        /// detail"*: `s` arrives fp32, is rounded to `T` and back, and only
        /// then multiplies. A host program that pre-rounded, or a kernel that
        /// multiplied in fp32, would give a different answer on the models
        /// that use it.
        fn scalar_mul = "norm::device::scalar_mul" <T> (
            x: *mut T,
            s: f32,
            n: usize,
        ) where *mut T {
            "norm::scalar_mul_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/dsv4_hc.cuh` — deepseek-v4's hyper-connections, the attention sink
/// correction and the per-head norm.
pub mod dsv4_hc {
    use super::bf16;

    unit! {
        /// Seven `__global__` templates, all seven instantiated here.
        ///
        /// The header opens by saying three are named by rows and four are
        /// not, *"because a kernel nobody can fire is worth less than a
        /// kernel nobody can fire for a reason on record"*. That sentence
        /// describes the ROW world's coverage, and it changes with this port:
        /// the four that had no row were reached only from
        /// `driver-cuda/src/fire/dsv4_hc.rs`'s hand-written launchers, and a
        /// hand-written launcher is exactly what a `fn` is. All seven now
        /// have a declaration, a host program and a contract; four of the
        /// contracts still refuse to BIND, for the reasons this file's header
        /// tabulates, and two that the row world refused by association now
        /// bind.
        unit DSV4_HC = "norm/dsv4_hc",
            text = include_str!("../../csrc/src/norm/dsv4_hc.cuh"),
            file = "norm/dsv4_hc.cuh";

        /// `dsv4_hc.cuh:103` — split the mix matrix, Sinkhorn-normalise the
        /// combination block, and collapse `M` streams to the layer's one
        /// `[N, H]` input.
        ///
        /// `_rows` in the symbol and not in the contract: the CONTRACT is
        /// `norm::hc_pre_postprocess_bf16`, which is what a lowering would
        /// name, and the instantiation carries the launcher's geometry in its
        /// name the way `families/norm.rs` did. Both spellings must exist
        /// because `bind/nvrtc.rs` looks the instantiation up and
        /// `table::sig_of` looks the contract up.
        fn hc_pre_postprocess = "norm::device::hc_pre_postprocess" <T> (
            mixes: *const f32,
            scale: *const f32,
            base: *const f32,
            residual: *const T,
            post_mix: *mut f32,
            comb_mix: *mut f32,
            layer_input: *mut T,
            m: i32,
            h: i32,
            hc_eps: f32,
            hc_post_alpha: f32,
            sinkhorn_iters: i32,
        ) where *const T, *mut T {
            "norm::hc_pre_postprocess_rows_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:239` — scatter the layer's output back across the `M`
        /// streams through the mixer.
        ///
        /// `residual` and `out` may ALIAS (`:244`) and the kernel is written
        /// for it. The contract states `in_place: &[(1, 0)]` for that reason,
        /// which is the one `Contract` field this family needed.
        ///
        /// **`if (M > MAX_HC_MULT) return;` at `:249` is a silent no-op on
        /// the device**, which is why the host program asserts instead. See
        /// this file's header on the missing `Refusal::Wide`.
        fn hc_post = "norm::device::hc_post" <T> (
            x: *const T,
            residual: *const T,
            post_mix: *const f32,
            comb_mix: *const f32,
            out: *mut T,
            n: i32,
            m: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::hc_post_elems_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:285` — the same collapse without the Sinkhorn: a
        /// plain gated sum.
        fn hc_head_postprocess = "norm::device::hc_head_postprocess" <T> (
            mixes: *const f32,
            scale: *const f32,
            base: *const f32,
            residual: *const T,
            out: *mut T,
            m: i32,
            h: i32,
            hc_eps: f32,
        ) where *const T, *mut T {
            "norm::hc_head_postprocess_rows_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:327` — the degenerate mixer: broadcast one stream
        /// into `M`.
        fn hc_expand = "norm::device::hc_expand" <T> (
            input: *const T,
            output: *mut T,
            n: i32,
            m: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::hc_expand_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:358` — RMS-normalise `[N, dim]` bf16 into `[N, dim]`
        /// fp32.
        ///
        /// No weight and no gamma: the mixing weights are applied by the
        /// GEMM that follows. The fp32 OUTPUT is the point — the mix matrix
        /// is computed in fp32 throughout.
        fn hc_rmsnorm_to_f32 = "norm::device::hc_rmsnorm_to_f32" <T> (
            input: *const T,
            output: *mut f32,
            dim: i32,
            eps: f32,
        ) where *const T {
            "norm::hc_rmsnorm_to_f32_rows" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:406` — fold an attention sink logit into an already
        /// normalised output.
        ///
        /// `out` is read and written; `lse` is the softmax's log-sum-exp per
        /// (row, head) and `sink` the learned per-head logit. The scale is
        /// `1 / (1 + exp(sink[h] - lse[n][h]))`, i.e. the sink's share of the
        /// softmax mass removed after the fact — which is why this can be a
        /// separate kernel from attention at all.
        fn attn_sink_correction = "norm::device::attn_sink_correction" <T> (
            out: *mut T,
            lse: *const f32,
            sink: *const f32,
            num_heads: i32,
            head_dim: i32,
        ) where *mut T {
            "norm::attn_sink_correction_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:431` — RMS-normalise each attention head of a
        /// `[N, heads, head_dim]` tensor in place.
        ///
        /// **The head count is `gridDim.y` (`:440`) and is not a parameter.**
        /// The geometry is not a convenience here — it is an ARGUMENT, and a
        /// host program that launched `[rows, 1, 1]` would normalise one head
        /// per token and leave the rest untouched with no error anywhere.
        /// [`super::per_head_rmsnorm_bf16`] takes `num_heads` and puts it on
        /// `grid.y`, which is the only place it can go.
        fn per_head_rmsnorm = "norm::device::per_head_rmsnorm" <T> (
            q: *mut T,
            head_dim: i32,
            eps: f32,
        ) where *mut T {
            "norm::per_head_rmsnorm_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// The six roots this family compiles.
///
/// Hand-written because `unit!` emits one `UNITS` per invocation and there
/// are six invocations. `unit!`'s own doc names this as the multi-root shape;
/// `x::mlp` is the two-root precedent and this is the six-root one.
pub static UNITS: &[Unit] = &[
    rmsnorm::RMSNORM,
    add_bias::ADD_BIAS,
    altup::ALTUP,
    altup_aux::ALTUP_AUX,
    elementwise::ELEMENTWISE,
    dsv4_hc::DSV4_HC,
];

// ---------------------------------------------------------------------------
// The constants the launchers were written in.
// ---------------------------------------------------------------------------

/// `rmsnorm.cu:85`, `dsv4_hc.cu:18`, `elementwise.cuh:12`, `add_bias.cuh:12` —
/// `constexpr int BLOCK = 256;`
///
/// One constant for nearly every launcher in this family, which is what four
/// anonymous namespaces independently made it. `runtime/launch.rs:578` spells
/// the same 256 for the rules that replaced them.
const BLOCK: u32 = 256;

/// `rmsnorm.cu:88` — `constexpr int VBLOCK = 512;`
///
/// The vectorised norms' block width. See [`rmsnorm::rmsnorm_vec8`] for why
/// an instantiation compiled at one width cannot be launched at the other.
const VBLOCK: u32 = 512;

/// `runtime/launch.rs:584` — the warp width, for the two kernels that share
/// one float per warp.
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — the largest block CUDA will launch.
const MAX_BLOCK: u32 = 1024;

/// `runtime/launch.rs:743` — one float per warp of a [`BLOCK`]-wide block.
///
/// **Read by exactly two kernels in this family**, `altup_aux.cuh:97` and
/// `:117`, which are the only two that declare `extern __shared__ float
/// smem[]`. `LaunchRule::Rms` allocated it for five more that fold through a
/// static `__shared__ float buf[BLOCK]` instead; see this file's header.
const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

/// `runtime/launch.rs:727` — `altup.cu:18-19` and `:32-33`'s block width.
///
/// AltUp tiles the hidden axis by 128 and puts the tile on `grid.z`, which is
/// why this is not [`BLOCK`].
const ALTUP_BLOCK: u32 = 128;

/// AltUp's epsilon, which is the ALGORITHM's and not the model's.
///
/// Carried whole from `table/norm.rs:13-21`, which is being deleted:
///
/// > Both rows below carried `Source::Ctx("eps")` and both hand arms passed
/// > this constant instead — the arms were right. `ctx.eps` is the
/// > checkpoint's `rms_norm_eps` (1e-6 for gemma-3n), and substituting it
/// > here is a different computation that still runs. A literal is the
/// > honest spelling: nothing about a rank-K residual stream's magnitude
/// > hold reads the model's norm epsilon.
///
/// In fn-world the two binds pass it as an ordinary argument, so there is no
/// `Source::Lit` to disagree with a `Source::Ctx` any more — the value is in
/// the call.
pub const ALTUP_EPS: f32 = 1e-5;

/// The width above which the vectorised fused norm prefers a 512-thread
/// block.
///
/// `rmsnorm.cu:160` — `if (hidden_size >= 2560)`. A COMPARISON, which is why
/// the launcher could not become a row: every `Term` in the `LaunchRule`
/// vocabulary is unary (`new-horizon.md` §44.6). §5 step 5 says the
/// `Specialisation`s become `if`s, and this is the `if`.
pub const RASR_VEC512_ABOVE: i32 = 2560;

/// `dsv4_hc.cuh:91` — `constexpr int MAX_HC_MULT = 8;`
///
/// The width of `hc_post`'s register array (`float r[MAX_HC_MULT]`), and
/// therefore the largest multiplier the kernel can be launched with. Stated
/// here as well as in the device text because the check that reads it is on
/// this side — `hc_post`'s own `if (M > MAX_HC_MULT) return;`
/// (`dsv4_hc.cuh:249`) is a silent no-op, so it is a diagnosis and not the
/// safety.
pub const MAX_HC_MULT: i32 = 8;

// ---------------------------------------------------------------------------
// Geometry. Every one of these is a `LaunchRule` body from
// `crates/kernels-cuda-new/src/runtime/launch.rs`, which is itself the ported
// `<<<>>>` of a deleted `.cu` launcher, cited line by line.
// ---------------------------------------------------------------------------

/// One block per row, [`BLOCK`] wide, **nothing shared**.
///
/// `runtime/launch.rs:1103` (`LaunchRule::PerRow`) and — for every kernel in
/// this family except `altup_aux`'s two — `LaunchRule::Rms` at `:737` with
/// its `smem` corrected to what the device text reads. The C++ launchers all
/// passed `0`: `rmsnorm.cu:97-102`, `dsv4_hc.cu:39-44`, `:82-86`, `:98-100`.
#[must_use]
const fn per_row(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK)
}

/// [`per_row`] with the warp-reduction scratch the two `altup_aux` kernels
/// declare.
///
/// `runtime/launch.rs:737-745` in full, and the ONLY two launches in this
/// family that need it: `compute_rms` (`altup_aux.cuh:97`) and
/// `magnitude_rescale` (`:117`).
#[must_use]
const fn per_row_reducing(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// Flat pointwise over `n` elements, [`BLOCK`] per block, rounded up.
///
/// `runtime/launch.rs:828` (`LaunchRule::Elementwise`), which is
/// `elementwise.cuh:9-13`'s four lines: `blocks = (n + BLOCK - 1) / BLOCK`.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// Flat pointwise over `n` elements given as a 64-bit count, saturating.
///
/// `dsv4_hc.cu:58-60` computes `total` as `long long` and the grid as an
/// `int`; the archive's `(total + BLOCK - 1) / BLOCK` could overflow an `int`
/// on a batch no deployment has. Saturating at `u32::MAX` is what
/// `fire/dsv4_hc.rs:179` did (`u32::try_from(grid).unwrap_or(u32::MAX)`) and
/// is carried unchanged.
#[must_use]
fn elementwise_wide(n: i64) -> Launch {
    let blocks = (n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    Launch {
        grid: [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// Pointwise with the row on its own grid axis.
///
/// `runtime/launch.rs:1014` (`LaunchRule::ElementwiseRows`): `mean_streams`
/// reads `[K, T, H]` and writes `[T, H]`, so a flat index over the output
/// would have to be divided back into a row and a channel by the kernel.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per row, as wide as the row rounded up to a warp and capped at
/// [`MAX_BLOCK`].
///
/// `runtime/launch.rs:1028` (`LaunchRule::RouteRows`). The cap is safe only
/// because the kernels stride — `unpack_predict_coefs` walks
/// `kk += blockDim.x` — so a block narrower than the row computes all of it
/// in several passes. Before the stride loop this cap would have silently
/// computed a prefix; `altup_aux.cuh` says so.
#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), 1, 1],
        block: [
            width
                .unsigned_abs()
                .div_ceil(WARP)
                .max(1)
                .saturating_mul(WARP)
                .min(MAX_BLOCK),
            1,
            1,
        ],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per (row, head), [`BLOCK`] wide.
///
/// `runtime/launch.rs:1455` (`LaunchRule::GatedRms`). The head count is a
/// GRID AXIS and not an argument for `per_head_rmsnorm`, which reads
/// `gridDim.y` at `dsv4_hc.cuh:440`.
#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), heads.unsigned_abs(), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `dim3(T, K, ceil(H / 128))` at [`ALTUP_BLOCK`] threads.
///
/// `runtime/launch.rs:2405` (`LaunchRule::AltUpStreams`), transcribed from
/// the deleted `norm/altup.cu:18-19` and `:32-33`; the axis order is
/// witnessed by `altup.cuh:83-85`. The `t >= T || k >= K` half of the
/// kernels' guard is dead under exactly this grid and the `h >= H` half is
/// not — `H` is tiled by 128 and the last tile is ragged.
#[must_use]
const fn altup_streams(rows: i32, streams: i32, hidden: i32) -> Launch {
    Launch {
        grid: [
            rows.unsigned_abs(),
            streams.unsigned_abs(),
            hidden.unsigned_abs().div_ceil(ALTUP_BLOCK),
        ],
        block: [ALTUP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per row PER HEAD, [`BLOCK`] wide, nothing shared.
///
/// `runtime/launch.rs:815` (`LaunchRule::RowsPerHead`) verbatim, INCLUDING
/// its refusal: a width that does not divide by the stated head width is
/// refused rather than floored.
///
/// **The measurement this rule exists for, from `runtime/launch.rs:806-814`:**
/// at hidden 2048 over 128-wide heads a stated 128 gives `rows · 16` blocks
/// of a 128-channel norm and an absent one gives `rows` blocks of a
/// 2048-channel norm. Under the old reading — the fire's `head_dim` standing
/// in for a statement that named none — the plain norm took the per-head arm
/// and opened **sixteen times the blocks**, each norming a whole row's width
/// from a sixteenth of a row's offset.
///
/// `stated_head_dim == 0` is the ABSENT arm and not an error, which is the
/// whole content of the distinction: a row that said `Rows`/`InWidth(0)`
/// would norm gemma-4's q/k heads as one row each.
///
/// # Errors
///
/// [`Refusal::Narrow`] when a stated head width does not divide the row.
#[cfg(feature = "_cuda")]
fn rows_per_head(rows: i32, width: i32, stated_head_dim: i32) -> Result<Launch, Refusal> {
    if stated_head_dim == 0 {
        return Ok(per_row(rows));
    }
    let (w, hd) = (width.unsigned_abs(), stated_head_dim.unsigned_abs());
    if w == 0 || !w.is_multiple_of(hd) {
        return Err(Refusal::Narrow { what: "a row that divides by head_dim", at: width });
    }
    let blocks = rows
        .unsigned_abs()
        .checked_mul(w / hd)
        .ok_or(Refusal::Narrow { what: "a row count that fits a grid", at: rows })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

// ---------------------------------------------------------------------------
// Truth two: the host programs.
//
// Eight of these are `driver-cuda/src/fire/{rmsnorm,dsv4_hc}.rs` MOVED — the
// same arithmetic, the same citations, typed pointers instead of
// `*mut c_void`, and a `Fired` where the C++ returned void. The rest are the
// rule-driven rows' launchers, written from the `LaunchRule` each row stated.
//
// A NOTE ON THE EMPTY EXTENT, WHICH IS A DELIBERATE CHANGE. The C++ returned
// silently on `num_rows <= 0` and `fire/rmsnorm.rs` carried that forward with
// the sentence *"nothing to do is not a refusal"*. fn-world declines it:
// `Refusal::Empty` exists for exactly this and `x::mlp` set the precedent.
// The reason the sentence changed is that a `Fired` is RETURNED — a caller
// that means "nothing to do" can ignore a `Declined(Empty)` where it could
// not ignore a `cudaGetLastError` at the next synchronisation.
// ---------------------------------------------------------------------------

/// `rmsnorm.cu:26` — `rmsnorm_vec8_ok`.
///
/// True when every row of a `[num_rows, hidden]` bf16 view starts on a
/// 16-byte boundary and is a whole number of 8-element vectors.
///
/// The order is the C++'s own so the two read as one list, and — as
/// `families/norm.rs`'s `RMSNORM_STRIDED_VEC8` said of the `Term` list that
/// mirrored it — order is not semantic here: `&&` short-circuits, but every
/// clause is a test on a value the caller already holds, so nothing is
/// deferred by an earlier `false`.
///
/// # THIS IS THE `Specialisation` NOW, AND THAT IS §5 STEP 5
///
/// `RMSNORM_STRIDED_VEC8` stated these six clauses as six `Term`s so that a
/// fire could choose an instantiation without host code. It is deleted with
/// `families/norm.rs`, and its measurements are carried on
/// [`strided_bf16`] because a measurement outlives the mechanism it was
/// taken through.
///
/// `aligned16` is `crate::x::fire`'s and is `(p.addr() & 15) == 0`, which is
/// the C++'s `(uintptr_t(p) & 15u) == 0` exactly.
#[cfg(feature = "_cuda")]
#[must_use]
fn vec8_ok(
    x: *const c_void,
    y: *const c_void,
    weight: *const c_void,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
) -> bool {
    hidden % 8 == 0
        && x_row_stride % 8 == 0
        && y_row_stride % 8 == 0
        && crate::x::fire::aligned16(x)
        && crate::x::fire::aligned16(y)
        && crate::x::fire::aligned16(weight)
}

/// `rmsnorm.cu:80` — `norm::rmsnorm_strided_bf16`, both arms.
///
/// The strides are the two values' OWN widths, which is the whole of what
/// "strided" means here: a row of `x` is `x_row_stride` wide and only
/// `hidden` of it is read.
///
/// # THE ARM, AND EVERY MEASUREMENT THAT CHOSE IT
///
/// `families/norm.rs`'s `RMSNORM_STRIDED_VEC8` is the deleted
/// `Specialisation` this `if` replaces, and its evidence is this function's
/// now.
///
/// **The predicate, clause for clause.** `rmsnorm.cu`'s six against the six
/// `Term`s that mirrored them:
///
/// ```text
///   rmsnorm_vec8_ok          operand           term
///   hidden % 8 == 0          3 hidden          Multiple { of: 8 }
///   x_row_stride % 8 == 0    4 x_row_stride    Multiple { of: 8 }
///   y_row_stride % 8 == 0    5 y_row_stride    Multiple { of: 8 }
///   aligned(x)               0 x               Aligned { bytes: 16 }
///   aligned(y)               2 y               Aligned { bytes: 16 }
///   aligned(weight)          1 weight          Aligned { bytes: 16 }
/// ```
///
/// **`tests/specialise.rs` swept 98 304 cases across those six boundaries —
/// eight byte offsets on each of three pointers (on 16, one byte off, one
/// bf16 element off, half a chunk, two short, on the next chunk, and past
/// it), twelve widths covering every residue of `hidden % 8`, and four
/// offsets on each of two strides — and the two agreed on all 98 304. 128 of
/// those cases took the vectorised arm, which is exactly the 8 × 4 × 4 the
/// six clauses predict, so the sweep is not passing by refusing
/// everything.** Deleting the `weight` clause — the realistic mistake,
/// because `weight` is the one pointer of the three that is not an
/// activation — put 96 of 6 144 cases on the wrong arm, and the sweep caught
/// it.
///
/// **The timing.** `tests/specialise.rs` timed 300 launches of each arm
/// through the same symbol on an L40S, release, the two argument lists
/// differing only in two bytes on the output pointer:
///
/// ```text
///   rows  hidden   scalar us  vector us   ratio
///      1    2048        2.52       2.23    1.13x
///      1    4096        2.97       2.64    1.13x
///      1    8192        4.03       3.14    1.28x
///      8    4096        3.09       2.76    1.12x
///     64    4096        3.38       2.96    1.14x
///    512    4096        6.40       4.54    1.41x
///   1024    2048        6.45       4.75    1.36x
/// ```
///
/// The arm wins at every shape measured — 1.12x at decode, 1.41x at prefill,
/// 0.29 to 1.86 us saved per fire — and the choice that picks it cost 21 ns
/// over 100 000 evaluations, about one per cent of the cheapest launch in the
/// table. Those are minima of five batches of 300 and not means of one,
/// because at decode's shapes the two arms are a few hundred nanoseconds
/// apart and a single batch put the ratio either side of 1.0 on consecutive
/// runs. Repeat runs reproduced every ratio to within 0.02. In a DEBUG build
/// the same choice cost 304 ns and the arm lost below 512 rows, which is
/// recorded rather than dropped: at those shapes the harness is timing
/// `fire`'s own host work and not the kernel. The win is a release-build
/// claim and should be read as one.
///
/// **Both arms compute the same bf16, and that was measured rather than
/// assumed.** Scalar thread `t` sums `x[t], x[t+256], …`; vectorised thread
/// `t` sums four `float2` pairs out of each 16-byte chunk it owns — a
/// reassociation. `tests/specialise.rs` fired both arms on identical bytes
/// at five shapes — hidden 2048/2816/4096 at one row, 5376 at three, 2048 at
/// seven — and **0 of 39 424 bf16 values differed, worst case 0 ulp.** The
/// reassociation is real in fp32 and dies in the round to bf16. The
/// tolerance is zero because it was measured to be.
///
/// **What a wrong choice looks like, measured.** The negative control fires
/// the vectorised arm at `hidden = 4095`, where `rmsnorm_vec8_ok` says it
/// must not go. **7 of 4 095 values moved, and 0 of the 4 088 the kernel
/// actually wrote.** `rmsnorm_vec8` computes `nvec = hidden / 8`, sums 4 088
/// of the 4 095 squares, still divides by 4 095, and the norm is wrong by
/// under a tenth of a per cent — which bf16's eight mantissa bits cannot
/// see. A wrong choice here is not a crash and not a NaN, it is **99.83 per
/// cent of the right answer, feeding sixty more layers**, and no
/// relative-error tolerance loose enough to admit a reassociated reduction
/// would flag it.
///
/// # THE BLOCK WIDTH GOES BACK TO THE LAUNCHER'S, AND THAT IS THE PORT
///
/// The `Specialisation` fired `norm::rmsnorm_strided_bf16#vec8` at **256**,
/// and its own doc explains why it had to: *"`LaunchRule::Rms` launches 256
/// threads and `BLOCK` is the size of the `__shared__ float[BLOCK]` the
/// reduction folds through: compiled at 512 and launched at 256,
/// `block_reduce_sum_exact` folds through 256 floats no thread wrote and the
/// norm is finite and wrong. The alternative is a rule that launches 512,
/// which would be a SECOND decision stacked on the alignment one and a
/// change to `runtime::launch`, which this work does not own."*
///
/// **A host program owns its launch, so the second decision costs nothing
/// and this fires `#vec8_512` at 512 — `rmsnorm.cu:88-94` exactly.** That is
/// not a regression against the 256 measurement, it is the archive: the row
/// world's 256 was the deviation the rule forced, and `fire/rmsnorm.rs`
/// already fired 512 for the two callers it served. The 256 rows stay
/// instantiated because two tests fire them.
///
/// # Safety
///
/// `x`, `weight` and `y` must address live device memory of the extents the
/// strides describe, and `stream` must be a live `cudaStream_t` — for the
/// duration of the launch, which is asynchronous, so that ends at the next
/// synchronisation and not at this call's return.
#[cfg(feature = "_cuda")]
pub unsafe fn strided_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // `rmsnorm.cu:86` — `dim3 grid(num_rows)`. The C++ launched a zero grid
    // here and `cudaGetLastError` reported it at the next synchronisation on
    // whatever call happened to be next.
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if vec8_ok(
        x.cast(),
        y.cast_const().cast(),
        weight.cast(),
        hidden,
        x_row_stride,
        y_row_stride,
    ) {
        // `rmsnorm.cu:88-94` — `constexpr int VBLOCK = 512;`
        // `device::rmsnorm_vec8<VBLOCK, false><<<grid, VBLOCK, 0, stream>>>`.
        // `y_fp16` is `None`: `EMIT_FP16=false` reads it only inside a dead
        // `if constexpr`, which is what the declaration's `Option` says.
        unsafe {
            rmsnorm::raw::rmsnorm_vec8(
                "norm::rmsnorm_strided_bf16#vec8_512",
                Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
                x,
                weight,
                y,
                None,
                hidden,
                x_row_stride,
                y_row_stride,
                eps,
                stream,
            );
        }
        return Fired::Launched;
    }
    // `rmsnorm.cu:85,97-102` — `constexpr int BLOCK = 256;`
    // `device::rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>`.
    unsafe {
        rmsnorm::raw::rmsnorm(
            "norm::rmsnorm_strided_bf16",
            per_row(num_rows),
            x,
            weight,
            y,
            hidden,
            x_row_stride,
            y_row_stride,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:38` — `norm::rmsnorm_bf16`, which is one call and nothing
/// else.
///
/// The unstrided view of [`strided_bf16`]: `hidden` is both strides. Kept as
/// a function rather than inlined at its two call sites because that is what
/// the archive did, and because the identity `rmsnorm_bf16(…) ==
/// rmsnorm_strided_bf16(…, hidden, hidden, …)` is the whole content of the
/// symbol.
///
/// **Not to be confused with [`rmsnorm_bf16`]**, which is the SEMANTIC
/// `OpKind::Rmsnorm`'s launcher and reads a per-head width. Two host programs
/// for one symbol is what `table/norm.rs` recorded as *"the only pair in the
/// tree whose operand contract was written nowhere"*.
///
/// # Safety
///
/// [`strided_bf16`]'s, unchanged.
#[cfg(feature = "_cuda")]
pub unsafe fn unstrided_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // SAFETY: the caller's obligation, forwarded verbatim.
    unsafe { strided_bf16(x, weight, y, num_rows, hidden, hidden, hidden, eps, stream) }
}

/// `rmsnorm.cu:64` — `kernels::quant::bf16_to_fp16(y, y_fp16, n, stream)`.
///
/// # THIS IS THE `Composed` CROSSING A FAMILY BOUNDARY, AND IT STAYS TYPED
///
/// The second launch of [`rmsnorm_bf16_with_fp16`]'s middle arm, and it is
/// `quant`'s kernel and not this family's. `norm` declares no
/// `quant::bf16_to_fp16` and must not: a `raw::` fn exists for a `__global__`
/// the family DECLARES, and declaring it here would mint a second definition
/// of a row `x::quant` owns.
///
/// **All of that is true and the conclusion drawn from it was wrong.** This
/// function called [`crate::x::fire::fire`] with the symbol and a hand-built
/// `ArgValue` list, on the reasoning that a `raw::` stub can only be named
/// through the module it was declared beside. It cannot: the expansion takes
/// `symbol`, `launch`, its typed parameters and `stream` and calls
/// `x::fire::fire` itself, resolving `unit::unit_of(symbol)` GLOBALLY —
/// **`$UNIT` appears nowhere in a stub body.** The module path is Rust
/// namespacing and only that.
///
/// So this calls `quant`'s own stub, spelled in full:
///
/// ```text
/// crate::x::quant::dequant_wna16::raw::bf16_to_narrow::<f16>(..)
/// ```
///
/// Fully typed, with `quant`'s `Abi::CPP` spellings, declaring nothing twice.
/// The middle segment is the callee's inline unit module — `dequant_wna16`,
/// because that is the root `quant::bf16_to_fp16`'s row lives on
/// (`x/quant.rs:870`, the `bf16_to_narrow<T>` stub at `:978`). It is NOT a
/// module named for the operation, and reading it off the callee rather than
/// guessing is the whole of the care this needs.
///
/// # What IS a real consequence, and has no mechanism
///
/// A cross-family call makes `quant`'s unit a dependency of this host program
/// and **nothing in the type system says so**, because `symbol` is a
/// `&'static str`. If `norm/rmsnorm.cuh` compiles and `quant/dequant_wna16.cuh`
/// does not, this arm panics at the fire naming `quant::bf16_to_fp16` —
/// right behaviour, wrong time.
///
/// The remedy is a comment and not a mechanism: **`// fires:
/// quant::bf16_to_fp16`** sits beside the call, so the caller is greppable
/// from the callee. A dependency discoverable only by grep is worth one line
/// of grep bait.
///
/// # Why not call `x::quant::bf16_to_fp16`, the host program
///
/// It exists, it is `pub`, and it would also be typed. It is not called
/// because it states the geometry through `LaunchRule::Slab` and this call
/// site is a PORT OF A LAUNCHER: the numbers below are
/// `dequant_wna16.cu:66-71`'s own, cited line by line, which is what the
/// archive fired. The two agree — `SLAB_VEC` 8, `BLOCK` 256, `SLAB_GRID_MAX`
/// 1024 — and if they ever disagree this arm must follow the launcher it
/// ports, not the rule. A `Composed` body calling the other family's HOST
/// PROGRAM rather than its stub is a real alternative with real advantages
/// (the callee's refusals come free, and its geometry is not restated); it is
/// a design question §2.3 does not settle and this port does not settle it
/// either.
///
/// The geometry is `quant/dequant_wna16.cu:65-72`'s, which
/// `LaunchRule::Slab` also reproduces exactly (`SLAB_VEC` 8, `BLOCK` 256,
/// `SLAB_GRID_MAX` 1024) — stated here rather than taken from the rule
/// because this call site is a port of a launcher and not a fire of a row.
///
/// # Safety
///
/// `src` must address `count` live bf16 elements, `dst` `count` live fp16
/// elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
unsafe fn cast_to_fp16(src: *const bf16, dst: *mut f16, count: i64, stream: *mut c_void) -> Fired {
    // `dequant_wna16.cu:65` — `if (count == 0) return;`
    if count <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    // `dequant_wna16.cu:66-71`:
    //   constexpr int BS = 256;
    //   const long long n_vec8 = n / 8;
    //   const long long units  = n_vec8 > 0 ? n_vec8 : n;
    //   const int blocks = min((units + BS - 1) / BS, 1024);
    const BS: i64 = 256;
    const SLAB_GRID_MAX: i64 = 1024;
    let n_vec8 = count / 8;
    let units = if n_vec8 > 0 { n_vec8 } else { count };
    let blocks = ((units + BS - 1) / BS).clamp(1, SLAB_GRID_MAX);
    // `dequant_wna16.cu:72` —
    // `device::bf16_to_narrow<__half><<<max(blocks,1), BS, 0, stream>>>(in, out, n)`.
    //
    // The block width is `BS` and not this family's `BLOCK`. Both are 256 and
    // they are two different 256s: `BLOCK` is `rmsnorm.cu:85`'s and `BS` is
    // `dequant_wna16.cu:66`'s.
    let launch = Launch {
        grid: [u32::try_from(blocks).unwrap_or(1024), 1, 1],
        block: [u32::try_from(BS).unwrap_or(256), 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // fires: quant::bf16_to_fp16
    //
    // SAFETY: the caller's obligation, above. The stub is `quant`'s and the
    // types are `quant`'s -- `*const bf16`, `*mut f16`, `i64` -- so a wrong
    // pointer here is a compile error and not a bind panic.
    unsafe {
        crate::x::quant::dequant_wna16::raw::bf16_to_narrow::<f16>(
            "quant::bf16_to_fp16",
            launch,
            src,
            dst,
            count,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:54` — `norm::rmsnorm_bf16_with_fp16`, all three arms.
///
/// RMSNorm that also writes an fp16 copy of its output, for the MXFP4 decode
/// GEMV. The archive's comment for why no row named this entry point is the
/// reason it is a `fn`: *"that fallback is a SECOND launch
/// (`quant::bf16_to_fp16`), which is why no row names this entry point: a row
/// is one kernel, and this one is two whenever the rows are unaligned."*
///
/// # §2.3's `Composed`, PROVEN — AND WHAT IT NEEDED
///
/// The three arms, in the order the C++ tests them:
///
/// 1. `:58` — `y_fp16 == nullptr`: no fp16 copy was asked for, so this is
///    [`unstrided_bf16`] and nothing else.
/// 2. `:62` — the rows do not vectorise: [`unstrided_bf16`] writes the bf16
///    result, then [`cast_to_fp16`] reads it back and narrows it. **Two
///    launches over the same buffer, ordered by the stream, and the second
///    is another family's kernel.**
/// 3. `:69` — the fused arm, one launch, `EMIT_FP16=true`.
///
/// **[`Fired`]'s rule — resolve every refusal before the first launch — is
/// what the C++ did not do and what this body does.** `num_rows <= 0` is
/// hoisted above all three arms here; in the archive it lived inside
/// `rmsnorm_bf16`, so arm 2 could launch a zero grid, get nothing back from
/// it, and then launch the cast over a buffer the first launch had not
/// written. The bug was invisible because both launches were no-ops, and the
/// rule that makes it impossible is the floor's.
///
/// The middle arm's second launch is another family's `raw::` stub, called by
/// its full path — see [`cast_to_fp16`], which also records the one thing
/// §2.3 does not cover: the callee's unit becomes a dependency of this host
/// program and no type says so.
///
/// # The known defect this port does NOT fix
///
/// `rmsnorm_vec8` with `EMIT_FP16=true` is wrong above one row —
/// `rmsnorm.cuh:318` writes the fp16 copy without a row offset — and
/// `tests/launch_rules.rs`'s `the_emit_fp16_kernel_is_wrong_above_one_row`
/// pins the signature so the defect cannot be renamed away. A port that
/// quietly corrected it would make this symbol compute something the archive
/// did not, on a path a golden was recorded over.
///
/// # Safety
///
/// `x`, `weight` and `y` must address `num_rows * hidden` live bf16 elements;
/// `y_fp16`, when `Some`, `num_rows * hidden` live fp16 elements. `stream`
/// must be live across every launch this makes.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_bf16_with_fp16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // HOISTED above all three arms; see this function's header.
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    let Some(fp16) = y_fp16 else {
        // `rmsnorm.cu:58-61`.
        // SAFETY: the caller's obligation, forwarded.
        return unsafe { unstrided_bf16(x, weight, y, num_rows, hidden, eps, stream) };
    };
    // `rmsnorm.cu:62-68`. The predicate reads `hidden` for BOTH strides,
    // which is what makes this the unstrided view.
    if !vec8_ok(x.cast(), y.cast_const().cast(), weight.cast(), hidden, hidden, hidden) {
        // SAFETY: as above, and the second launch reads what the first wrote
        // on the same stream, which orders them.
        unsafe {
            unstrided_bf16(x, weight, y, num_rows, hidden, eps, stream);
            cast_to_fp16(
                y.cast_const(),
                fp16.as_ptr(),
                i64::from(num_rows) * i64::from(hidden),
                stream,
            );
        }
        return Fired::Launched;
    }
    // `rmsnorm.cu:69-77`:
    //   constexpr int VBLOCK = 512;
    //   dim3 grid(num_rows);
    //   device::rmsnorm_vec8<VBLOCK, /*WEIGHT_PLUS_ONE=*/false, /*EMIT_FP16=*/true>
    //       <<<grid, VBLOCK, 0, stream>>>(
    //       x, weight, y, y_fp16, hidden, hidden, hidden, eps);
    //
    // THE TWO `/*NAME=*/` COMMENTS ARE RESTORED HERE and are not decoration.
    // `fire/rmsnorm.rs` quoted this launch with them stripped, and
    // `tests/launch_rules.rs` recorded the loss as a finding: *"what is gone
    // is the only text in the tree that said WHICH BOOL IS WHICH, which is
    // exactly the half `RMSNORM_SIGS[10]` was written from"*. The stripped
    // form reads `<VBLOCK, false, true>`, and a reader has no way to tell
    // that the `true` is the fp16 emission and not the `(1 + w)` fold — the
    // two are one position apart and both are `bool`. The text is recovered
    // from that test's own pin, which held the C++ verbatim.
    unsafe {
        rmsnorm::raw::rmsnorm_vec8(
            "norm::rmsnorm_bf16_with_fp16#vec8_512",
            Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
            x,
            weight,
            y,
            Some(fp16),
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The SEMANTIC `OpKind::Rmsnorm`'s launcher — `norm::rmsnorm_bf16`.
///
/// **Not [`unstrided_bf16`], which is the same symbol.** `table/norm.rs`
/// recorded why there are two: *"nothing STATES them: `OpKind::Rmsnorm`
/// carries a variant and each driver picks between these two from it. That
/// makes them the only pair in the tree whose operand contract was written
/// nowhere."* The pair is `rmsnorm` and `rmsnorm_gemma`, and what they add
/// over the strided launcher is the PER-HEAD reading.
///
/// `per_head_dim == 0` is the absent arm: `hidden` is the row's width and
/// there is one block per row. A stated width makes `hidden` the head width,
/// both strides the head width, and the grid `rows · (width / head_dim)` —
/// see [`rows_per_head`], which carries the measurement.
///
/// `families/norm.rs` left this row's `x_row_stride`/`y_row_stride`
/// deliberately equal to `hidden` and said why: *"`rmsnorm_vec8` reads
/// `hidden` from a `hidden`-strided row, so the predicate's three strides are
/// the STATED head width here and not the row's."* That is why this fn never
/// takes the vectorised arm: a per-head norm's rows are `head_dim` apart
/// inside a `width`-wide row, and `vec8_ok`'s stride clauses are about a
/// different rectangle.
///
/// # Safety
///
/// [`strided_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    // `rmsnorm.cu:97-102` — `device::rmsnorm<device::bf16, BLOCK>`.
    unsafe {
        rmsnorm::raw::rmsnorm(
            "norm::rmsnorm_bf16",
            launch,
            x,
            weight,
            y,
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// gemma's `(1 + w)` fold — `norm::rmsnorm_gemma_bf16`.
///
/// `rmsnorm.cu:254-276`, grid at `:259`, the scalar launch at `:271-276`:
///
/// ```text
/// constexpr int BLOCK = 256;
/// dim3 grid(num_rows);
/// dim3 block(BLOCK);
/// device::rmsnorm_gemma<device::bf16, BLOCK><<<grid, block, 0, stream>>>(
///     x, weight, y, hidden, hidden, hidden, eps);
/// ```
///
/// Different arithmetic, same signature, same row space, same grid. The
/// launcher passes `hidden` for both strides inline rather than through a
/// forward, which is the only structural difference from
/// [`rmsnorm_bf16`].
///
/// # Safety
///
/// [`strided_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gemma_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gemma(
            "norm::rmsnorm_gemma_bf16",
            launch,
            x,
            weight,
            y,
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The weightless per-head norm — `norm::rmsnorm_no_scale_bf16`.
///
/// The V-norm: no gamma, so no `(1 + w)` variant. `rmsnorm.cuh:686` takes
/// `hidden` and no strides at all, so the per-head reading reaches it through
/// the block count alone.
///
/// # Safety
///
/// `x` and `y` must address `rows * width` live bf16 elements, and `stream`
/// must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_no_scale_bf16(
    x: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_no_scale("norm::rmsnorm_no_scale_bf16", launch, x, y, hidden, eps, stream);
    }
    Fired::Launched
}

/// The gated norm — `norm::rmsnorm_gated_bf16`.
///
/// qwen3.5's, in its own launch rather than folded into a projection.
/// **`weight` is `*const f32`** (`rmsnorm.cuh:721`) where the activations are
/// bf16; see this file's header on what the row could say.
///
/// # Safety
///
/// `x`, `gate` and `y` must address `rows * width` live bf16 elements,
/// `weight` `hidden` live floats, and `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gated_bf16(
    x: *const bf16,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gated(
            "norm::rmsnorm_gated_bf16",
            launch,
            x,
            gate,
            weight,
            y,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The gated norm with an fp32 INPUT — `norm::rmsnorm_gated_fp32_in_bf16`.
///
/// `rmsnorm.cu:199` launched `device::rmsnorm_gated_f32_in<device::bf16, 256>`
/// at `<<<num_rows, 256>>>`. `fire/rmsnorm.rs` deliberately did NOT port it,
/// and its closing note says why: the symbol was already in
/// `device::JIT_DISPATCHED`, so `emit_c_shim` emitted no entry and nothing in
/// the tree reached that launcher — *"it was dead C++ waiting for its file to
/// go."*
///
/// **It is here because fn-world needs it and the row world had it only by
/// accident.** `lower.rs:1519` lowers `OpKind::RmsnormGated` to this symbol,
/// so a qwen3.5 trace STATES it; `table/norm.rs` had no row for it and the
/// JIT path bound it from `families/norm.rs`'s. Deleting both without a
/// contract here would make a stated symbol reach `Route::Unknown` and refuse
/// the model at load. See this file's header.
///
/// `trace.rs:581` describes what it computes: per (row, head),
/// `out = w * rmsnorm(x) * silu(gate)`, normalising the trailing head dim of
/// the rank-3 fp32 core output and flattening to the gate's
/// `[Tokens, Vh * Vd]` bf16 shape — the fp32→bf16 conversion fused into the
/// same launch.
///
/// # Safety
///
/// `x` must address `rows * width` live floats, `gate` and `y` the same count
/// of bf16, `weight` `hidden` live floats, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gated_fp32_in_bf16(
    x: *const f32,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gated_f32_in(
            "norm::rmsnorm_gated_fp32_in_bf16",
            launch,
            x,
            gate,
            weight,
            y,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The residual add and the NEXT block's pre-norm, fused —
/// `norm::residual_add_rmsnorm_bf16`.
///
/// Numerically the two-kernel sequence: the kernel matches `residual_add`'s
/// bf16 rounding before norming, which is what makes it a binding a
/// declaration may state rather than a different computation. `hidden` is
/// read AND written.
///
/// `LaunchRule::Rms`, with the smem this file's header explains:
/// `rmsnorm.cuh:425` folds through a STATIC `__shared__ float buf[BLOCK]`, so
/// the launch carries none.
///
/// # Safety
///
/// `hidden`, `residual`, `norm_out` must address `num_rows * hidden_size`
/// live bf16 elements and `weight` `hidden_size` of them; `stream` must be
/// live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_rmsnorm_bf16(
    hidden: *mut bf16,
    residual: *const bf16,
    weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::residual_add_rmsnorm(
            "norm::residual_add_rmsnorm_bf16",
            per_row(num_rows),
            hidden,
            residual,
            weight,
            norm_out,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// Norm, then add into the residual stream in place —
/// `norm::rmsnorm_residual_add_bf16`.
///
/// # Safety
///
/// [`residual_add_rmsnorm_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_residual_add_bf16(
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_residual_add(
            "norm::rmsnorm_residual_add_bf16",
            per_row(num_rows),
            x,
            weight,
            hidden,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:119` — `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`, all
/// three arms.
///
/// gemma-4 fires this: four statements and 221 golden lines. It is the one
/// symbol in `fire/rmsnorm.rs` whose ahead-of-time row was FULLY SOURCED, so
/// it had a generated arm calling
/// `ffi::pie_k_norm_rmsnorm_residual_add_scale_rmsnorm_bf16`, then a
/// generated arm calling `bind::service`, and now a bind — except that the
/// bind refuses, because `Cx` cannot answer `Source::LayerScale`. See this
/// file's header.
///
/// # THE SECOND `Composed`, AND WHY IT IS `Walk` AND NOT `Choose`
///
/// Three arms, one launch each, chosen by FIVE addresses and a width. No
/// `Term` list can hold it: `hidden_size >= 2560` is a COMPARISON and every
/// `Term` in that vocabulary is unary (`new-horizon.md` §44.6). §5 step 5
/// says the `Specialisation`s become `if`s; this one could never be a
/// `Specialisation` at all.
///
/// Note the predicate is NOT [`vec8_ok`]: there are no strides here (the
/// kernel is packed by construction) and there are two more buffers.
///
/// # The measurement, which the port carries rather than consumes
///
/// The scalar form walks the row three times, once per pass, and measured
/// **10.79 us/call in gemma-4-26B's decode — 8% of the step** — against 2.51
/// for the vectorized plain norm. Swept under graph replay at the shapes
/// these models use, in us:
///
/// ```text
///   hidden   scalar256  scalar512   vec256  vec512  vec1024
///     2048        4.38       3.68     2.72    2.93     3.31
///     2816        6.17       4.83     3.46    3.12     3.51
///     5376        8.48       6.55     4.44    4.07     4.02
/// ```
///
/// Against the shipping scalar/256 that is **−38%, −49% and −53%**. The
/// vectorized twin is **bit-identical** to the scalar form at all three sizes
/// (0 of 2048/2816/5376 bf16 values differ) — only the two sum reductions
/// reassociate, and at these lengths that rounds to the same bf16.
///
/// vec512 is chosen above hidden 2560 ([`RASR_VEC512_ABOVE`]) and vec256
/// below: it is best at 2816, within 1.5% of best at 5376, and the 2048 case
/// prefers the narrower block. Scalar keeps hidden < 2560's old width only
/// when the rows are unaligned — which is the `BLOCK = 512` at `:179`, and it
/// is 512 for every width because the unaligned path is the one the sweep
/// could not improve.
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight` and `norm_out` must address live
/// device memory of `num_rows * hidden_size` (the two weights,
/// `hidden_size`) bf16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    scale: f32,
    next_weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // `rmsnorm.cu:152` — `dim3 grid(num_rows)`.
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    let rows = num_rows.unsigned_abs();
    // `rmsnorm.cu:153-158` — `vec_ok`.
    let vec_ok = hidden_size % 8 == 0
        && crate::x::fire::aligned16(x.cast())
        && crate::x::fire::aligned16(hidden.cast_const().cast())
        && crate::x::fire::aligned16(norm_out.cast_const().cast())
        && crate::x::fire::aligned16(weight.cast())
        && crate::x::fire::aligned16(next_weight.cast());
    if vec_ok {
        // `rmsnorm.cu:160-168` — `constexpr int kB = 512;` and `:170-177` —
        // `constexpr int kB = 256;`, both
        // `device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(...)`.
        let (symbol, block) = if hidden_size >= RASR_VEC512_ABOVE {
            ("norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_512", VBLOCK)
        } else {
            ("norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_256", BLOCK)
        };
        unsafe {
            rmsnorm::raw::rmsnorm_rasr_vec8(
                symbol,
                Launch::per_row(rows, block),
                x,
                weight,
                hidden,
                scale,
                next_weight,
                norm_out,
                hidden_size,
                eps,
                stream,
            );
        }
        return Fired::Launched;
    }
    // `rmsnorm.cu:179-189` — `constexpr int BLOCK = 512;`
    // `device::rmsnorm_residual_add_scale_rmsnorm<device::bf16, BLOCK>
    //     <<<grid, block, 0, stream>>>(...)`.
    unsafe {
        rmsnorm::raw::rmsnorm_residual_add_scale_rmsnorm(
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#scalar_512",
            Launch::per_row(rows, VBLOCK),
            x,
            weight,
            hidden,
            scale,
            next_weight,
            norm_out,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `out[row][i] += bias[i]` — `norm::add_bias_bf16`.
///
/// `add_bias.cuh:9-13` was three lines:
/// `if (num_rows <= 0 || dim <= 0) return;` then
/// `kernel<<<num_rows, 256, 0, stream>>>(out, bias, dim)`.
///
/// **The block is [`route_rows`]' and not the launcher's 256, and that was a
/// decision the row made before this port.** `ADD_BIAS_SIGS`' note: *"the
/// launcher was `<<<num_rows, 256>>>` with a stride loop over `dim`, so the
/// rule's wider block reaches the same elements in fewer iterations and the
/// arithmetic per element is unchanged."* Carried rather than reverted: the
/// row world shipped it and the loop is the same loop.
///
/// Like [`rmsnorm_gated_fp32_in_bf16`], this symbol is STATED by a lowering
/// (`lower.rs:1506`, `OpKind::AddBias`) and had no `table/norm.rs` row.
///
/// # Safety
///
/// `out` must address `num_rows * dim` live bf16 elements, `bias` `dim` of
/// them, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn add_bias_bf16(
    out: *mut bf16,
    bias: *const bf16,
    num_rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `add_bias.cuh:11` — `if (num_rows <= 0 || dim <= 0) return;`
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the bias width" });
    }
    unsafe {
        add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            route_rows(num_rows, dim),
            out,
            bias,
            dim,
            stream,
        );
    }
    Fired::Launched
}

/// gemma-3n's altup predict — `norm::altup_predict_bf16`.
///
/// `k` streams of `t_len × h`, one predicted stream per (token, stream)
/// pair. [`altup_streams`] carries the grid.
///
/// # Safety
///
/// `streams` and `predictions` must address `k * t_len * h` live bf16
/// elements, `coefs` `t_len * k * k` live floats, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_predict_bf16(
    streams: *const bf16,
    coefs: *const f32,
    predictions: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if t_len <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup::raw::altup_predict(
            "norm::altup_predict_bf16",
            altup_streams(t_len, k, h),
            streams,
            coefs,
            predictions,
            k,
            t_len,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// gemma-3n's altup correct — `norm::altup_correct_bf16`.
///
/// `active_idx` names which of the `k` predicted streams the block actually
/// ran, and the correction is applied relative to it. The grid is
/// [`altup_predict_bf16`]'s.
///
/// # Safety
///
/// [`altup_predict_bf16`]'s, with `activated` addressing `t_len * h` live
/// bf16 elements and `corrected` `k * t_len * h`.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_correct_bf16(
    predictions: *const bf16,
    activated: *const bf16,
    correction_coefs_plus_one: *const f32,
    corrected: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
    active_idx: i32,
    stream: *mut c_void,
) -> Fired {
    if t_len <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup::raw::altup_correct(
            "norm::altup_correct_bf16",
            altup_streams(t_len, k, h),
            predictions,
            activated,
            correction_coefs_plus_one,
            corrected,
            k,
            t_len,
            h,
            active_idx,
            stream,
        );
    }
    Fired::Launched
}

/// The per-row RMS of the reference stream — `norm::compute_rms_bf16`.
///
/// Writes one `float` per row for [`magnitude_rescale_bf16`] to read, which
/// is the pair that makes altup's magnitude preservation two statements
/// rather than one. **This is one of the two symbols whose block-wide
/// reduction folds through `extern __shared__`** (`altup_aux.cuh:97`), so it
/// launches through [`per_row_reducing`] and carries [`RMS_SMEM`] — see this
/// file's header on why the other thirteen do not.
///
/// # Safety
///
/// `reference` must address `rows * h` live bf16 elements, `out` `rows` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn compute_rms_bf16(
    reference: *const bf16,
    out: *mut f32,
    rows: i32,
    h: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        altup_aux::raw::compute_rms(
            "norm::compute_rms_bf16",
            per_row_reducing(rows),
            reference,
            out,
            h,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// Rescale each row to a stated RMS, in place — `norm::magnitude_rescale_bf16`.
///
/// [`compute_rms_bf16`]'s consumer, and the second `extern __shared__` user
/// (`altup_aux.cuh:117`). `target_rms` is the FIRST input's second operand in
/// the deleted row — `Source::In(1)` — which is why the contract states
/// `in_place: &[(0, 0)]` on `x` and reads the target from input one.
///
/// # Safety
///
/// `x` must address `rows * h` live bf16 elements, `target_rms` `rows` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn magnitude_rescale_bf16(
    x: *mut bf16,
    target_rms: *const f32,
    rows: i32,
    h: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        altup_aux::raw::magnitude_rescale(
            "norm::magnitude_rescale_bf16",
            per_row_reducing(rows),
            x,
            target_rms,
            h,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The mean over altup's `k` streams — `norm::mean_streams_bf16`.
///
/// `LaunchRule::ElementwiseRows`: the rectangle is `rows × h` and the `k`
/// axis is the reduction, so the grid does not carry it. `t_stride` is the
/// distance between one stream's rows, which the deleted row bound from
/// `Source::Rows` — the streams are stored contiguously per stream, so the
/// stride IS the row count and this fn takes it once.
///
/// # Safety
///
/// `streams` must address `k * t_stride * h` live bf16 elements, `out`
/// `rows * h` of them, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn mean_streams_bf16(
    streams: *const bf16,
    out: *mut bf16,
    k: i32,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::mean_streams(
            "norm::mean_streams_bf16",
            elementwise_rows(rows, h),
            streams,
            out,
            k,
            rows,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// bf16 predict coefficients widened to `float` —
/// `norm::altup_unpack_predict_coefs`.
///
/// The row is `k * k` wide and the deleted row recovered `k` from it with
/// `Source::InWidthIsqrt`. **A non-square width returns 0 there rather than
/// refusing** (`bind/mod.rs:1955`), which is why this fn takes `k` from its
/// caller and the bind is the place the isqrt happens: a `k` of zero would
/// launch a kernel that reads nothing and writes nothing, silently.
///
/// Note the symbol carries no `_bf16` suffix although the input is bf16 —
/// the deleted row's spelling, kept because `Route` joins on it.
///
/// # Safety
///
/// `in_bf16` must address `rows * k * k` live bf16 elements, `out` the same
/// count of floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_unpack_predict_coefs(
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::unpack_predict_coefs(
            "norm::altup_unpack_predict_coefs",
            route_rows(rows, k.saturating_mul(k)),
            in_bf16,
            out,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// bf16 correct coefficients widened to `float` —
/// `norm::altup_unpack_correct_coefs`.
///
/// [`altup_unpack_predict_coefs`]'s sibling, and the one difference is the
/// rectangle: the correction coefficients are `k` wide and not `k * k`, so
/// the row read `Source::Width(In(0))` straight rather than through an
/// isqrt.
///
/// # Safety
///
/// `in_bf16` must address `rows * k` live bf16 elements, `out` the same count
/// of floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_unpack_correct_coefs(
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::unpack_correct_coefs(
            "norm::altup_unpack_correct_coefs",
            route_rows(rows, k),
            in_bf16,
            out,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `tanh` in place over a bf16 slab — `norm::tanh_bf16`.
///
/// gemma-3n's logit softcap and altup's router both fire it. Elementwise, so
/// the launch is [`elementwise`] and nothing about the rectangle survives.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn tanh_bf16(x: *mut bf16, n: i32, stream: *mut c_void) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        altup_aux::raw::tanh_inplace("norm::tanh_bf16", elementwise(n), x, n, stream);
    }
    Fired::Launched
}

/// [`tanh_bf16`] over fp16 — `norm::tanh_f16`.
///
/// The second row on the same `fn`, which is what the `where [T = f16]` arm
/// in [`altup_aux`]'s `unit!` exists for.
///
/// # Safety
///
/// [`tanh_bf16`]'s, with `x` addressing fp16.
#[cfg(feature = "_cuda")]
pub unsafe fn tanh_f16(x: *mut f16, n: i32, stream: *mut c_void) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        altup_aux::raw::tanh_inplace("norm::tanh_f16", elementwise(n), x, n, stream);
    }
    Fired::Launched
}

/// `y += x` — `norm::residual_add_bf16`.
///
/// The plainest kernel in the family and the one `lower.rs:1507` lowers
/// `OpKind::ResidualAdd` to. `n` is a `usize` because the device parameter is
/// (`elementwise.cuh`), and the launch narrows it: `LaunchRule::Elementwise`
/// took a `u32`, so this saturates rather than wrapping.
///
/// # Safety
///
/// `y` and `x` must address `n` live bf16 elements and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_bf16(y: *mut bf16, x: *const bf16, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::residual_add("norm::residual_add_bf16", launch, y, x, n, stream);
    }
    Fired::Launched
}

/// [`residual_add_bf16`] over fp16 — `norm::residual_add_f16`.
///
/// # Safety
///
/// [`residual_add_bf16`]'s, with both pointers addressing fp16.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_f16(y: *mut f16, x: *const f16, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::residual_add("norm::residual_add_f16", launch, y, x, n, stream);
    }
    Fired::Launched
}

/// `x *= s` — `norm::scalar_mul_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn scalar_mul_bf16(x: *mut bf16, s: f32, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::scalar_mul("norm::scalar_mul_bf16", launch, x, s, n, stream);
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// deepseek-v4's hyper-connection residual
//
// `fire/dsv4_hc.rs`'s four launchers, plus the three §43.9 deleted by naming
// them in `device::JIT_DISPATCHED` — `hc_expand_bf16`,
// `attn_sink_correction_bf16` and `per_head_rmsnorm_bf16`. In row-world those
// three had no launcher at all and their geometry came from a `LaunchRule`;
// here all seven are `fn`s and the distinction disappears, which is the first
// thing this port makes true that was not true before.
//
// `fire/dsv4_hc.rs`'s header on why the four rows were unsourced, carried
// whole because it is the reason four of these get `none:` arms:
//
//   HC's mixing matrices are not values a statement names. `mixes`, `scale`
//   and `base` are three `float` slabs the layer carries, `post_mix` and
//   `comb_mix` are scratch the launcher hands from one kernel to the next,
//   and `sinkhorn_iters` and `hc_post_alpha` are model constants. A `Source`
//   for any of them would be a guess about where a lowering puts a buffer,
//   and **a half-bound row is worse than an unbound one**: `emit_dispatch`
//   skips a row with one unbound operand whole, so a row that sourced five of
//   thirteen generates exactly as much as a row that sources none, while
//   claiming four bindings nobody checked.
//
// Two of the four turn out to be bindable anyway — see the header.
// ---------------------------------------------------------------------------

/// `dsv4_hc.cu:22` — `norm::hc_pre_postprocess_bf16`.
///
/// The per-token mixing matrix: reads the three `float` slabs, runs
/// `sinkhorn_iters` normalisation passes over the `hc_mult × hc_mult` matrix
/// in shared memory, writes `post_mix` and `comb_mix` for the layer's
/// [`hc_post_bf16`] to read, and collapses the `hc_mult` residual streams into
/// the layer's bf16 input.
///
/// One block per token, striding the hidden axis.
///
/// # The symbol is not the row's symbol
///
/// The contract is `norm::hc_pre_postprocess_bf16` and the fire is
/// `norm::hc_pre_postprocess_rows_bf16`. That split is `families/norm.rs`'s
/// and it survives: the `_rows_` spelling names the DEVICE template
/// instantiated at `BLOCK = 256`, the bare one names the statement. `Route`
/// joins on the contract's, `x::fire` on the unit row's, and the `fn` is the
/// only place both are written — which is exactly what §2's "two truths" buys.
///
/// # Safety
///
/// `residual` and `layer_input` must address `n * hc_mult * hidden_size` and
/// `n * hidden_size` live bf16 elements; `mixes`, `scale` and `base` the
/// slabs the layer carries; `post_mix` and `comb_mix` scratch of `n *
/// hc_mult` and `n * hc_mult * hc_mult` floats. `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_pre_postprocess_bf16(
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const bf16,
    post_mix: *mut f32,
    comb_mix: *mut f32,
    layer_input: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
    stream: *mut c_void,
) -> Fired {
    // `dsv4_hc.cu:38` — `if (N <= 0) return;`
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    // `dsv4_hc.cu:40-45` —
    // `device::hc_pre_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    unsafe {
        dsv4_hc::raw::hc_pre_postprocess(
            "norm::hc_pre_postprocess_rows_bf16",
            per_row(n),
            mixes,
            scale,
            base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            hc_mult,
            hidden_size,
            hc_eps,
            hc_post_alpha,
            sinkhorn_iters,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:47` — `norm::hc_post_bf16`.
///
/// The write-back half: takes the layer's output and the `hc_mult` residual
/// streams it was collapsed from, and re-expands with the mixing weights
/// [`hc_pre_postprocess_bf16`] wrote. Elementwise over `n * hidden_size`, so
/// the grid is a slab and not a row count.
///
/// # THE REFUSAL, AND THE VARIANT THAT IS MISSING
///
/// `dsv4_hc.cu:59` — `if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;`
///
/// The C++ returned silently on both. The empty extent stays silent, because
/// nothing to do is not a refusal. **`hc_mult > MAX_HC_MULT` does not**: it
/// is a model whose hyper-connection width the compiled kernel cannot hold,
/// and a silent return there is a layer that reads its own uninitialised
/// residual and produces plausible tokens. The device header says so at
/// `dsv4_hc.cuh:228` — *"the `M > MAX_HC_MULT` refusal moved here from the
/// launcher, and it had to"* — the kernel now checks it as well, so the
/// refusal is the diagnosis rather than the safety.
///
/// `fire/dsv4_hc.rs` spelled it `assert!` and documented a `# Panics`. **A
/// `fn` here must not panic**: it is called from a `bind!` body whose only
/// outcome vocabulary is `Refusal`, and the whole point of `Fired` is that a
/// declined launch is a value. So the assert becomes a refusal — and there is
/// no `Refusal` variant for *above the compiled maximum*. See
/// [`hc_mult_ok`], which spells out the workaround and what the floor should
/// grow.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out_residual` addressing `n * hc_mult
/// * hidden_size` live bf16 elements.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_post_bf16(
    x: *const bf16,
    residual: *const bf16,
    post_mix: *const f32,
    comb_mix: *const f32,
    out_residual: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    // `dsv4_hc.cu:58-60`:
    //   const long long total = (long long)N * hidden_size;
    //   if (total <= 0 || hc_mult > MAX_HC_MULT) return;
    //   const int grid = (total + BLOCK - 1) / BLOCK;
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    // `dsv4_hc.cu:62-67` —
    // `device::hc_post<device::bf16><<<grid, BLOCK, 0, stream>>>(...)`.
    unsafe {
        dsv4_hc::raw::hc_post(
            "norm::hc_post_elems_bf16",
            elementwise_wide(total),
            x,
            residual,
            post_mix,
            comb_mix,
            out_residual,
            n,
            hc_mult,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:69` — `norm::hc_head_postprocess_bf16`.
///
/// The final collapse: the same mixing arithmetic as
/// [`hc_pre_postprocess_bf16`] but writing one bf16 stream rather than
/// scratch, for the LM head to read. No `post_mix`/`comb_mix` outputs,
/// because nothing after it re-expands.
///
/// `fire/dsv4_hc.rs` noted the argument order — *"`hc_eps` comes **after**
/// `stream` in the launcher's C++ signature and therefore in the
/// ahead-of-time row, and before it in the kernel's. The row is the
/// launcher's and the fire is the kernel's, which is exactly the difference
/// `execution::sig_of` documents."* **That difference is now unwritable.**
/// There is no launcher signature left to disagree with the kernel's: the
/// `unit!` row states the kernel's order and this `fn` is the only caller. The
/// observation is kept because it explains a shape a reader will find in the
/// git history, not because anything still has two orders.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out` addressing `n * hidden_size`
/// live bf16 elements.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_head_postprocess_bf16(
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const bf16,
    out: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    stream: *mut c_void,
) -> Fired {
    // `dsv4_hc.cu:81` — `if (N <= 0) return;`
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    // `dsv4_hc.cu:83-87` —
    // `device::hc_head_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    unsafe {
        dsv4_hc::raw::hc_head_postprocess(
            "norm::hc_head_postprocess_rows_bf16",
            per_row(n),
            mixes,
            scale,
            base,
            residual,
            out,
            hc_mult,
            hidden_size,
            hc_eps,
            stream,
        );
    }
    Fired::Launched
}

/// `[n, hidden] -> [n, hc_mult, hidden]` — `norm::hc_expand_bf16`.
///
/// The entry to the hyper-connection block: one hidden state becomes
/// `hc_mult` residual streams. §43.9 deleted its C++ launcher by naming the
/// symbol in `device::JIT_DISPATCHED`, so **there is no `<<<>>>` to cite** and
/// the geometry is the row's `LaunchRule::ElementwiseIn`: one thread per
/// element of the INPUT rectangle, `n * hidden_size`, each writing `hc_mult`
/// outputs.
///
/// The extent is computed in `i64` and saturated like [`hc_post_bf16`]'s
/// rather than in the rule's `u32`. Same answer wherever the `u32` form was
/// correct, and a defined one above that.
///
/// # Safety
///
/// `input` must address `n * hidden_size` live bf16 elements, `output`
/// `n * hc_mult * hidden_size` of them, and `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_expand_bf16(
    input: *const bf16,
    output: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        dsv4_hc::raw::hc_expand(
            "norm::hc_expand_bf16",
            elementwise_wide(total),
            input,
            output,
            n,
            hc_mult,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:89` — `norm::hc_rmsnorm_to_f32`.
///
/// RMSNorm from bf16 into `float`. The `float` result is what the mixing
/// matrices are computed in, which is why this exists as its own symbol
/// rather than as a `norm::rmsnorm_*` with a wider output: the consumer is
/// [`hc_pre_postprocess_bf16`], not a GEMM.
///
/// One block per row.
///
/// # Safety
///
/// `input` must address `n * dim` live bf16 elements, `output` `n * dim` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_rmsnorm_to_f32(
    input: *const bf16,
    output: *mut f32,
    n: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // `dsv4_hc.cu:97` — `if (N <= 0) return;`
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    // `dsv4_hc.cu:98-100` —
    // `device::hc_rmsnorm_to_f32<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    unsafe {
        dsv4_hc::raw::hc_rmsnorm_to_f32(
            "norm::hc_rmsnorm_to_f32_rows",
            per_row(n),
            input,
            output,
            dim,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The attention sink's log-sum-exp correction —
/// `norm::attn_sink_correction_bf16`.
///
/// Rescales each head's output by `exp(-softplus(sink - lse))`, which is what
/// makes a learned sink a per-head attenuation rather than an extra key.
/// §43.9 deleted its launcher; the geometry is `LaunchRule::GatedRms` —
/// [`gated_rms`], one block per (row, head).
///
/// # Safety
///
/// `out` must address `n * num_heads * head_dim` live bf16 elements, `lse`
/// and `sink` `n * num_heads` and `num_heads` live floats, and `stream` must
/// be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn attn_sink_correction_bf16(
    out: *mut bf16,
    lse: *const f32,
    sink: *const f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the head count" });
    }
    unsafe {
        dsv4_hc::raw::attn_sink_correction(
            "norm::attn_sink_correction_bf16",
            gated_rms(n, num_heads),
            out,
            lse,
            sink,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// QK-norm in place over a packed head axis —
/// `norm::per_head_rmsnorm_bf16`.
///
/// One block per (row, head), each norming `head_dim` contiguous values.
/// Distinct from [`rmsnorm_no_scale_bf16`], which reaches the same shape
/// through `LaunchRule::RowsPerHead` and a stated `per_head_dim`: this one
/// takes the head count as an argument because §43.9's row did
/// (`LaunchRule::GatedRms`), and the two rules pick different block widths for
/// the same rectangle — see [`gated_rms`] and [`rows_per_head`].
///
/// # Safety
///
/// `q` must address `n * num_heads * head_dim` live bf16 elements and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn per_head_rmsnorm_bf16(
    q: *mut bf16,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the head count" });
    }
    unsafe {
        dsv4_hc::raw::per_head_rmsnorm(
            "norm::per_head_rmsnorm_bf16",
            gated_rms(n, num_heads),
            q,
            head_dim,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `hc_mult <= MAX_HC_MULT`, as a refusal.
///
/// # THE FLOOR IS MISSING A VARIANT AND THIS IS THE WORKAROUND
///
/// `Refusal` has five variants and none of them means *above the compiled
/// maximum*. `Empty` is nothing to do, `Absent` and `Unstated` are about
/// facts the context does not carry, `Undeclared` is about a symbol, and
/// `Narrow { what, at }` is *too small at N*. What `MAX_HC_MULT` rejects is
/// **too large**, and there is no spelling for it.
///
/// The workaround reads the sentence backwards: the thing that is narrow is
/// the KERNEL — `float r[MAX_HC_MULT]` is a register array eight wide — and
/// `at: MAX_HC_MULT` is the width at which it is narrow. It is true, it is
/// diagnosable, and it is not what the field means.
///
/// **What the floor should grow is `Refusal::Wide { what, at, max }`**, and
/// this is not a one-family need: any kernel with a compile-time bound on a
/// runtime extent — a head dim, a stream count, a tile — hits it. See the
/// report.
#[cfg(feature = "_cuda")]
fn hc_mult_ok(hc_mult: i32) -> Result<(), Refusal> {
    if hc_mult > MAX_HC_MULT {
        return Err(Refusal::Wide {
            what: "hc_mult, which `hc_post` unrolls into a register array",
            at: hc_mult,
            max: MAX_HC_MULT,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The declarations
//
// Twenty-eight contracts, one per statement a lowering can name. `table/norm.rs`
// had twenty-six rows; the two extra are `add_bias` and
// `rmsnorm_gated_fp32_in`, which lived only in `families/norm.rs`'s JIT rows
// and would have reached `Route::Unknown` the moment both files went. See the
// header.
//
// What survives from a row is what a READER needs: the DSL name, the symbol,
// and `in_place`. `operands`, `launch`, `file`, `returns` and `axes` do not,
// because they described a launcher and the launcher is now a `fn` above.
//
// Of `Contract`'s ten fields this family states exactly one beyond `name` and
// `symbol` — `in_place`, on the ten statements whose result IS their operand.
// `whole`, `needs`, `lacks`, `sink`, `depth_prefix_plan`, `publishes_aux` and
// `lowered_as` are stated by no `norm` row and are stated by none here: every
// one of these kernels is row-shaped (token `t`'s output reads only token
// `t`'s inputs), so a peel may split it, it obligates no host plan, and there
// is no seam capability for it to refuse. `table/norm.rs` said so of AltUp in
// as many words and it is true of the whole family.
// ---------------------------------------------------------------------------

contract! {
    /// The strided norm — the one `fire/rmsnorm.rs` vectorised.
    ///
    /// *"The strides are the two values' OWN widths, which is the whole of
    /// what 'strided' means here: a row of `x` is `x_row_stride` wide and only
    /// `hidden` of it is read. So `hidden` comes off the RESULT and the
    /// strides off each side, and the row needs nothing the binder does not
    /// already hold."*
    RMSNORM_STRIDED = "norm::rmsnorm_strided_bf16" as rmsnorm_strided

    /// The plain RMSNorm, one of the two `OpKind::Rmsnorm` fans to.
    ///
    /// *"Rows because they had none, and they had none because nothing STATES
    /// them: `OpKind::Rmsnorm` carries a variant and each driver picks between
    /// these two from it. That makes them the only pair in the tree whose
    /// operand contract was written nowhere — every other kernel a semantic
    /// kind fans to is also stated by something, so it has a row already."*
    RMSNORM = "norm::rmsnorm_bf16" as rmsnorm

    /// gemma's, folding `(1 + w)`.
    ///
    /// *"Gemma folds `(1 + w)` instead of `w` — different arithmetic, same
    /// signature, same row space."*
    RMSNORM_GEMMA = "norm::rmsnorm_gemma_bf16" as rmsnorm_gemma

    /// The norm that also writes an fp16 copy.
    ///
    /// *"The fp16 copy is what the MXFP4 grouped GEMM consumes; producing it
    /// here rather than casting afterwards is the binding."*
    RMSNORM_WITH_FP16 = "norm::rmsnorm_bf16_with_fp16" as rmsnorm_with_fp16

    /// The weightless per-head norm.
    ///
    /// *"Weightless per-head norm (the V-norm) — no gamma, so no variant."*
    RMSNORM_NO_SCALE = "norm::rmsnorm_no_scale_bf16" as rmsnorm_no_scale {
        in_place: &[(0, 0)],
    }

    /// qwen3.5's gated norm in its own launch.
    RMSNORM_GATED_LAUNCH = "norm::rmsnorm_gated_bf16" as rmsnorm_gated_launch

    /// The gated norm reading an fp32 core output.
    ///
    /// `lower.rs:1519` — `OpKind::RmsnormGated`. It had no `table/norm.rs`
    /// row, and `families/norm.rs:1296` says why that was already known to be
    /// wrong: *"`norm::rmsnorm_gated_fp32_in_bf16` is not in..."* — the JIT
    /// row was carrying a statement the ahead-of-time table never described.
    RMSNORM_GATED_FP32_IN = "norm::rmsnorm_gated_fp32_in_bf16" as rmsnorm_gated_fp32_in

    /// Residual add and the next block's pre-norm, fused.
    ///
    /// *"Numerically the two-kernel sequence (the kernel matches
    /// `residual_add`'s bf16 rounding before norming), which is what makes it
    /// a binding a declaration may state rather than a different
    /// computation."*
    RESIDUAL_ADD_RMSNORM = "norm::residual_add_rmsnorm_bf16" as residual_add_rmsnorm

    /// Norm, then add into the residual stream.
    ///
    /// `(landed, mlp_in)` over `(x, y)`: the stream operand is the one it
    /// lands on, and the landed stream is output 0.
    NORM_RESIDUAL_ADD = "norm::rmsnorm_residual_add_bf16" as norm_residual_add {
        in_place: &[(0, 1)],
    }

    /// gemma-4's four-statements-in-one.
    ///
    /// *"Four statements in one launch, and two: gemma-4 fuses the next
    /// block's input norm into the previous block's landing, which is why its
    /// layer body appears to be missing one."*
    NORM_RESIDUAL_SCALE_NORM = "norm::rmsnorm_residual_add_scale_rmsnorm_bf16"
        as norm_residual_scale_norm {
        in_place: &[(0, 1)],
    }

    /// `out[row][i] += bias[i]`.
    ///
    /// `lower.rs:1506` — `OpKind::AddBias`. Like [`RMSNORM_GATED_FP32_IN`],
    /// stated by a lowering and absent from `table/norm.rs`.
    /// `families/norm.rs:397` on why there is one and not two: *"`add_bias_strided`
    /// was the second, and it was the second because `add_bias.hpp` declares
    /// both — `new-horizon.md` §28.4 measured [it] ... the `__global__` stays
    /// in `norm/add_bias.cuh`; only the claim that some fire [needs it] goes."*
    ADD_BIAS = "norm::add_bias_bf16" as add_bias {
        in_place: &[(0, 0)],
    }

    /// HC's RMSNorm into `float`.
    ///
    /// *"The SECOND rank-K residual scheme here, and not AltUp's. gemma-3n
    /// predicts each stream from a learned combination and corrects from one
    /// ACTIVE stream; HC mixes with a per-token, sinkhorn-normalized matrix
    /// and has no active stream -- every layer reads a weighted collapse of
    /// all of them and writes back to all of them. Row-shaped throughout."*
    HC_RMSNORM_TO_F32 = "norm::hc_rmsnorm_to_f32" as hc_rmsnorm_to_f32

    /// Where a rank-K residual begins.
    ///
    /// *"Where a rank-K residual BEGINS: replicate the embedding into K
    /// streams. AltUp's equivalent is implicit in gemma-3n's workspace layout;
    /// HC states it, which is the one a declaration can read. The
    /// hyper-connection expand: one hidden row in, `hc_mult` of them out. Both
    /// extents come off the two values — the multiplier is what the result is
    /// wider BY — so nothing here is the plan's."*
    HC_EXPAND = "norm::hc_expand_bf16" as hc_expand

    /// The per-token sinkhorn mixing matrix.
    HC_PRE = "norm::hc_pre_postprocess_bf16" as hc_pre

    /// The write-back half.
    HC_POST = "norm::hc_post_bf16" as hc_post

    /// The final collapse, for the LM head.
    HC_HEAD = "norm::hc_head_postprocess_bf16" as hc_head

    /// QK-norm where q lies.
    ///
    /// *"Normalizes q WHERE IT LIES: one operand, one result, the same bytes
    /// — so `q` binds from `Out(0)` and the staging comes off the `in_place`
    /// pair."*
    PER_HEAD_RMSNORM = "norm::per_head_rmsnorm_bf16" as per_head_rmsnorm {
        in_place: &[(0, 0)],
    }

    /// The attention sink's log-sum-exp correction.
    ///
    /// *"The head GEOMETRY off the value, not off the context: this
    /// statement's result is rank-3 `[Tokens, heads, head_dim]`, so the two
    /// counts are its own dims. That is the difference between a fully-stated
    /// row and one that needs a context field it would then share with every
    /// other family's idea of 'the head count'."*
    ATTN_SINK_CORRECTION = "norm::attn_sink_correction_bf16" as attn_sink_correction {
        in_place: &[(0, 0)],
    }

    /// AltUp's prediction step.
    ///
    /// *"A rank-K residual stream: K parallel streams predicted from each
    /// other, one of them run through the real layer, the rest corrected from
    /// the difference. See `dsl::cuda`'s AltUp block for the algebra. Not one
    /// of these carries a contract clause, and that is a claim rather than an
    /// omission: every one is row-shaped -- token `t`'s output reads only
    /// token `t`'s inputs -- so a peel may split it, it obligates no host
    /// plan, and there is no seam capability for it to refuse."*
    ALTUP_PREDICT = "norm::altup_predict_bf16" as altup_predict

    /// AltUp's correction step.
    ALTUP_CORRECT = "norm::altup_correct_bf16" as altup_correct

    /// The `K*K` predict coefficients, bf16 to `float`.
    ALTUP_UNPACK_PREDICT_COEFS = "norm::altup_unpack_predict_coefs"
        as altup_unpack_predict_coefs

    /// The `K` correct coefficients, bf16 to `float`.
    ALTUP_UNPACK_CORRECT_COEFS = "norm::altup_unpack_correct_coefs"
        as altup_unpack_correct_coefs

    /// The mean over AltUp's streams.
    ///
    /// *"`k` is a CONTEXT field and not an extent, because the streams arrive
    /// interleaved: `streams` is `[t, k*h]` and only the fire knows how that
    /// row divides. `CtxNonZero` rather than `Ctx` for the same reason the arm
    /// checked it — a fire that states no stream count is not one this kernel
    /// can be run for, and declining is better than dividing by zero."*
    MEAN_STREAMS = "norm::mean_streams_bf16" as mean_streams

    /// The reference stream's per-row RMS.
    COMPUTE_RMS = "norm::compute_rms_bf16" as compute_rms

    /// The magnitude hold.
    ///
    /// *"In place on the tensor it holds to a magnitude: the row states one
    /// operand and one result and they are the same bytes, which is what lets
    /// `x` bind from `Out(0)` and the width come off the value."*
    MAGNITUDE_RESCALE = "norm::magnitude_rescale_bf16" as magnitude_rescale {
        in_place: &[(0, 0)],
    }

    /// `x *= s`.
    ///
    /// *"The SCALE is the statement's, in the bits the param channel has room
    /// for. It was a NAME, and the driver held the arithmetic that turned four
    /// names into four numbers -- all four derived from dims the host already
    /// knew. A family whose facts do not carry the number states no param and
    /// falls through this branch's arity guard, which is what gemma-3n and
    /// gemma-2 do."*
    SCALAR_MUL = "norm::scalar_mul_bf16" as scalar_mul {
        in_place: &[(0, 0)],
    }

    /// `y += x`.
    ///
    /// *"Accumulates into its FIRST argument. Stating it is what lets a text
    /// add into a window (`select`) and have the window keep the result — see
    /// `KernelSig::in_place`."*
    RESIDUAL_ADD_CUDA = "norm::residual_add_bf16" as residual_add_cuda {
        in_place: &[(0, 0)],
    }

    /// `tanh` in place.
    TANH = "norm::tanh_bf16" as tanh {
        in_place: &[(0, 0)],
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it
//
// Fifteen binds and thirteen `none:` arms. Every refusal names the exact
// `Facts` method that would remove it, because `Cx` is the floor's and
// widening it is the floor's author's edit and not this family's. Each
// sentence surfaces at MODEL LOAD through `Route::Unbound` with the family and
// the symbol prepended, so it is written as a sentence a user reads.
//
// The thirteen are not thirteen problems. They are five:
//
//   * `Facts::per_head_dim()`     — five symbols, and the largest single gap
//   * `Facts::altup_streams()`    — two, plus `altup_active()` for a third
//   * `Facts::layer_scale()`      — one, gemma-4's fused landing
//   * `Facts::named_scale()`      — one
//   * HC's float slabs            — two, and these are not a `Facts` method
//
// Four of the five are one defaulted accessor each over a field
// `DispatchCtx`/`LaunchSpec` ALREADY HOLDS. The fifth is a design question.
// ---------------------------------------------------------------------------

bind! {
    RMSNORM_STRIDED => { cx, stream => {
        // `hidden` off the RESULT and the strides off each side — the
        // deleted row's sentence, and the reason nothing here is the plan's.
        unsafe {
            strided_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    RMSNORM => { none: "Cx has no query for the statement's per-head width. \
        The deleted row read Source::IfPresent(PerHeadDim, ..) on both num_rows \
        and hidden, because OpKind::RmsnormPerHead lowers to this same symbol \
        and norms rows x (width / head_dim) rows of head_dim where the plain \
        kind norms rows of width; without the query this fn would norm \
        gemma-4's q/k heads as one row each. Needs `Facts::per_head_dim() -> \
        Option<i32>` over LaunchSpec::per_head_dim (bind/mod.rs:1798), which \
        the driver already holds" },

    RMSNORM_GEMMA => { none: "Cx has no query for the statement's per-head \
        width, exactly as for norm::rmsnorm_bf16 — same operand contract, \
        different arithmetic. Needs `Facts::per_head_dim()`" },

    RMSNORM_WITH_FP16 => { none: "The deleted row stated no Source on any of \
        its eight operands, so there is nothing to read a binding from: it \
        described the launcher's C signature and never said where y_fp16, or \
        anything else, comes from. The host program above is complete and \
        proven; what is missing is a statement that names the fp16 copy. Needs \
        a lowering that produces two results, and then Source::Out(1)" },

    RMSNORM_NO_SCALE => { none: "Cx has no query for the statement's per-head \
        width; this is the V-norm and the per-head reading is the only one it \
        is ever fired with. Needs `Facts::per_head_dim()`" },

    RMSNORM_GATED_LAUNCH => { none: "Cx has no query for the statement's \
        per-head width. Needs `Facts::per_head_dim()`" },

    RMSNORM_GATED_FP32_IN => { none: "Cx has no query for the gated-delta-net \
        head width. The deleted row bound hidden from Source::Gdn(\"v_d\") and \
        families/norm.rs records the correction it wanted -- spec.per_head_dim \
        set from gdn.v_d where the statement is a gated norm -- so this needs \
        `Facts::per_head_dim()` and the driver-side assignment, not a new \
        Source" },

    RESIDUAL_ADD_RMSNORM => { cx, stream => {
        unsafe {
            residual_add_rmsnorm_bf16(
                cx.arg_in(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    NORM_RESIDUAL_ADD => { cx, stream => {
        unsafe {
            rmsnorm_residual_add_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    NORM_RESIDUAL_SCALE_NORM => { none: "Cx has no query for the layer's \
        residual scale. Every other operand of gemma-4's fused landing is \
        available -- two weights, two results, the row count and the width -- \
        and the one that is not is Source::LayerScale, the per-layer constant \
        the binder reads off the model. Needs `Facts::layer_scale() -> \
        Option<f32>`. This is the family's most expensive refusal: the host \
        program above is the three-arm vectorised form measured at -38%, -49% \
        and -53% against the shipping scalar kernel" },

    ADD_BIAS => { cx, stream => {
        // The bias is the statement's NAMED weight, like the embedding's
        // table -- `Source::WeightNamed` and not `Weight(0)`.
        unsafe {
            add_bias_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.weight_named(0)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    HC_RMSNORM_TO_F32 => { cx, stream => {
        // UNSOURCED BY ASSOCIATION. The deleted row stated no Source on any
        // operand and sat in the same block as `hc_pre` and `hc_head`, whose
        // float slabs genuinely have nowhere to come from. This one's do:
        // `dsl.rs:4486` states `hc_rmsnorm_to_f32(residual, weight, width)`
        // and every operand the kernel takes is on it. The row was wrong by
        // proximity, and a `fn` that reads `Cx` directly is what made that
        // visible.
        //
        // The statement's `weight` is NOT a kernel parameter -- this kernel
        // has no gamma. It names the value whose width the result takes.
        unsafe {
            hc_rmsnorm_to_f32(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    HC_EXPAND => { cx, stream => {
        // Both extents off the two values: the multiplier is what the result
        // is wider BY, so nothing here is the plan's.
        let hidden = cx.in_width(0)?;
        if hidden <= 0 {
            return Err(Refusal::Empty { what: "the hidden width" });
        }
        unsafe {
            hc_expand_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / hidden,
                hidden,
                stream,
            )
        }
        .ok()
    }},

    HC_PRE => { none: "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- nor for the \
        two scratch buffers this kernel hands to norm::hc_post_bf16, nor for \
        the model constants sinkhorn_iters and hc_post_alpha. The deleted row \
        stated no Source on any of its thirteen operands and that was the \
        honest spelling: a half-bound row generates exactly as much as an \
        unbound one while claiming bindings nobody checked. Needs a lowering \
        that states the slabs, which is a design question and not an accessor" },

    HC_POST => { cx, stream => {
        // Also unsourced by association, and also bindable: `dsl.rs:4552`
        // states `hc_post(x, residual, post_mix, comb_mix, hc_mult, hidden)`,
        // so the two scratch buffers this reads are ordinary inputs of the
        // statement. It is `hc_pre` that produces them and cannot say so.
        let hidden = cx.in_width(0)?;
        if hidden <= 0 {
            return Err(Refusal::Empty { what: "the hidden width" });
        }
        unsafe {
            hc_post_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / hidden,
                hidden,
                stream,
            )
        }
        .ok()
    }},

    HC_HEAD => { none: "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- or for \
        hc_eps. Same shape as norm::hc_pre_postprocess_bf16 and the same \
        answer" },

    PER_HEAD_RMSNORM => { cx, stream => {
        // The head count off the VALUE and the head width off the context,
        // which is the split the deleted row made and `Cx` can hold.
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Narrow { what: "head_dim", at: head_dim });
        }
        unsafe {
            per_head_rmsnorm_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                head_dim,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ATTN_SINK_CORRECTION => { cx, stream => {
        // `OutWidthOver`, not `OutDim(0, 1)`: this asks the BINDER how many
        // head-dims fit in a row whose width it already holds, rather than
        // asking the PLAN for the second extent of a rank-3 value -- which
        // the join has never carried, and which is why this row sat on the
        // generator's wall.
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Narrow { what: "head_dim", at: head_dim });
        }
        unsafe {
            attn_sink_correction_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                head_dim,
                stream,
            )
        }
        .ok()
    }},

    ALTUP_PREDICT => { none: "Cx has no query for the AltUp stream count. \
        `streams` is [t, k*h] with the streams interleaved, so only the fire \
        knows how that row divides and the deleted row read \
        Source::Ctx(\"altup_streams\"); DispatchCtx::altup_streams \
        (bind/mod.rs:1244) is the accessor. Needs `Facts::altup_streams() -> \
        Option<i32>`" },

    ALTUP_CORRECT => { none: "Cx has no query for which AltUp stream was run \
        through the real layer. Every extent on this statement comes off its \
        own values -- k from input 2's width, h from input 1's -- and the one \
        that does not is active_idx, DispatchCtx::altup_active \
        (bind/mod.rs:1246). Needs `Facts::altup_active() -> Option<i32>`" },

    ALTUP_UNPACK_PREDICT_COEFS => { cx, stream => {
        // K*K packed in one row, so the launcher's K is the width's square
        // root. A width that is not a perfect square gives 0 here, exactly as
        // `Source::InWidthIsqrt` did (bind/mod.rs:1955) -- and where that
        // silently launched a kernel over nothing, the host program refuses.
        unsafe {
            altup_unpack_predict_coefs(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                isqrt_exact(cx.in_width(0)?),
                stream,
            )
        }
        .ok()
    }},

    ALTUP_UNPACK_CORRECT_COEFS => { cx, stream => {
        // K wide and not K*K, so the width is read straight.
        unsafe {
            altup_unpack_correct_coefs(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MEAN_STREAMS => { none: "Cx has no query for the AltUp stream count, and \
        here it is not an extent at all: the streams arrive interleaved and \
        only the fire knows how the row divides, which is why the deleted row \
        said CtxNonZero rather than Ctx -- declining is better than dividing \
        by zero. Needs `Facts::altup_streams()`" },

    COMPUTE_RMS => { cx, stream => {
        // ALTUP_EPS and not `cx.rms_eps()`: the deleted row carried
        // Source::Ctx("eps") and both hand arms passed the constant instead.
        // The arms were right -- see ALTUP_EPS.
        unsafe {
            compute_rms_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                ALTUP_EPS,
                stream,
            )
        }
        .ok()
    }},

    MAGNITUDE_RESCALE => { cx, stream => {
        unsafe {
            magnitude_rescale_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                ALTUP_EPS,
                stream,
            )
        }
        .ok()
    }},

    SCALAR_MUL => { none: "Cx can read a stated scale but not a named one. \
        The deleted row said Source::Or(ParamF32(0), NamedScale): a statement \
        that carries the number binds today through Cx::param_f32(0), and one \
        that carries a NAME -- which is what gemma-3n and gemma-2 state -- has \
        nowhere to read it from. Binding only the first half would make this \
        symbol work for some models and refuse at fire for exactly the two the \
        deleted row named, which is worse than refusing at load. Needs \
        `Facts::named_scale() -> Option<f32>`" },

    RESIDUAL_ADD_CUDA => { cx, stream => {
        // Accumulates into its FIRST argument, which is the RESULT: stating
        // `in_place` is what lets a text add into a window (`select`) and
        // have the window keep the result.
        let n = usize::try_from(elements(cx)?).unwrap_or(0);
        unsafe {
            residual_add_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    TANH => { cx, stream => {
        unsafe { tanh_bf16(cx.arg_out(0)?.cast::<bf16>(), elements(cx)?, stream) }.ok()
    }},
}

/// `Source::OutElements(0)` — the region's rows times the result's width.
///
/// `Rows::count` and not `Rows::total`: a peel fires the statement once per
/// window and each fire owns its own rows, which is the same reading
/// `x::mlp`'s `elements` takes.
/// # Errors
///
/// [`Refusal::Absent`] when the statement states no first output.
#[cfg(feature = "_cuda")]
fn elements(cx: &crate::x::Cx<'_>) -> Result<i32, Refusal> {
    Ok(cx.rows().count.saturating_mul(cx.out_width(0)?))
}

/// `Source::Isqrt` — the exact integer square root, or `0`.
///
/// `bind/mod.rs:1955` returns zero rather than refusing when the width is not
/// a perfect square, and this matches it so the port changes no binding. What
/// it does change is what happens next: [`altup_unpack_predict_coefs`]
/// declines a zero stream count where the row-world launcher ran a kernel
/// over nothing.
#[cfg(feature = "_cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn isqrt_exact(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut r = f64::from(n).sqrt() as i32;
    while r > 0 && r.saturating_mul(r) > n {
        r -= 1;
    }
    while (r + 1).saturating_mul(r + 1) <= n {
        r += 1;
    }
    if r.saturating_mul(r) == n {
        r
    } else {
        0
    }
}
