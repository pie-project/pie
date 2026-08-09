//! `mlp` — the gated activations and AltUp's sparsity, as two truths.
//!
//! The device text is `csrc/src/mlp/swiglu.cuh` and
//! `csrc/src/mlp/gaussian_topk.cuh`, unchanged. This file is the other
//! truth: sixteen `__global__` declarations, twelve host programs, twelve
//! contracts and the binds that connect them. §5 step 5, the third family.
//!
//! The sixteenth `__global__` — `sigmoid_gate_inplace` — is declared here
//! and *only* declared here: its host program is
//! [`crate::x::driver_internal::sigmoid_gate_inplace_bf16`] and it gets no
//! contract, for the reason that module's header gives.
//!
//! # What this replaces
//!
//! ```text
//!   before                                              lines
//!   kernels-cuda-new/src/families/mlp.rs   2 units, 16 rows  416
//!   kernels-cuda-new/src/table/mlp.rs             12 rows    176
//!   driver-cuda/src/tower/gemma4_vision.rs   1 rule-driven fire
//!                                                          -----
//!                                                            592
//!   after
//!   kernels-cuda-new/src/x/mlp.rs   12 host programs, 16 device
//!                                    rows, 12 contracts, 12 binds
//! ```
//!
//! **Every one of the twelve host programs is new.** `mlp/swiglu.cu` and
//! `mlp/gaussian_topk.cu` were deleted by §54 with the whole of
//! `csrc/src/mlp/`, and every row was already in `device::JIT_DISPATCHED`
//! — a `LaunchRule` opened the grid and a generated arm bound the operands,
//! so there was no `fire/mlp.rs` to move in from. The geometry below is
//! therefore cited to the RULE each row stated and to
//! `driver-cuda/src/bind/launch.rs`, which is the arithmetic those rules
//! evaluate to; that file's tests pin each one against the `<<<>>>` it was
//! ported from. Nothing here is invented, and no number is chosen.
//!
//! | rule | `bind/launch.rs` | grid | block | smem |
//! |---|---|---|---|---|
//! | `Elementwise` | `:128` | `[ceil(n / 256), 1, 1]` | `[256, 1, 1]` | 0 |
//! | `ElementwiseRows` | `:143` | `[rows, ceil(width / 256), 1]` | `[256, 1, 1]` | 0 |
//! | `Rms` | `:116` | `[rows, 1, 1]` | `[256, 1, 1]` | `(256 / 32) * 4` |
//!
//! # The `y_fp16` miss, fixed by the type system rather than by a note
//!
//! `families/mlp.rs` carried a measured defect against its `gpt_oss_glu`
//! row and could not repair it:
//!
//! > Measured rather than reasoned: a function POINTER admits no parameter
//! > conversion whatever, and initialising one from
//! > `mlp::device::gpt_oss_glu<device::bf16>` compiled under nvcc 13.0
//! > `-arch=sm_89` against `(const bf16*, const bf16*, bf16*, f16*, i32,
//! > float, float)` and refused the same list with `bf16*` fourth. `BufMut`
//! > takes its element from the row's `elem` and no `Ty` carries one of its
//! > own, so `bf16*` is the only thing a row can say about this parameter
//! > and it is not true.
//! >
//! > … The repair is a `Ty` carrying its own element — `crates/kernels`' to
//! > add, and not this row's to paper over.
//!
//! **The repair is free here and needed no `Ty`.** `y_fp16` is declared
//! `Option<NonNull<f16>>`, which is [`Abi`](crate::x::Abi)'s nullable fp16
//! pointer: `Abi::CPP` spells `::pie_cuda_driver::kernels::device::f16*`,
//! `Abi::TY` is `Ty::BufMut` for the marshaller, and the Rust type refuses a
//! `*mut bf16` at the call site before the typecheck TU is even reached.
//! The row's own paragraph named the mechanism ("`BufMut` takes its element
//! from the row's `elem`") and fn-world removes the mechanism: a parameter's
//! type comes from the declaration, not from the instantiation's element.
//!
//! This is §5.1's *"`quant` and FA2 are the real test of the unit structs"*
//! answered from a third direction — the [`bf16`]/[`f16`] distinction is
//! load-bearing in a family that has exactly one fp16 parameter, and that
//! parameter is the one the row world got wrong.
//!
//! # The three vectorised kernels still have no row, and the reason is the
//! same one
//!
//! `swiglu.cuh` carries `chunked_swiglu_vec2`,
//! `chunked_swiglu_vec2_gate_second` and `chunked_swiglu_strided_vec2`, and
//! nothing launches them. `families/mlp.rs`' refusal stands word for word,
//! and fn-world does not weaken it — a `fn` could express
//! `ceil(((I + 1) / 2) / BLOCK)` in one line, which is exactly why the
//! refusal was never about the grammar:
//!
//! > **What is owed is a measurement, not a restoration**: nobody has
//! > measured what `I > 10000` was worth, the threshold has no citation
//! > anywhere in the tree, and re-landing a two-element-per-thread path on
//! > the strength of a constant whose origin nobody can name is how a
//! > vocabulary grows for nothing.
//!
//! The same holds for `gpt_oss_glu_strided`, `chunked_swiglu_strided` and
//! `sigmoid_scalar_gate_add`: uninstantiated templates in a carried header,
//! which cost nothing and are not a second definition.
//!
//! # What the floor could not express, and what is refused because of it
//!
//! Six of the twelve contracts bind facts [`Cx`](crate::x::Cx) has no query
//! for. Each is a `none:` arm carrying the exact `Facts` method it wants,
//! and each is one defaulted method away from being three lines of bind —
//! the host program above it is complete and takes the value as an ordinary
//! argument.
//!
//! **Those six sentences are user-facing and are written as sentences.**
//! Since the step-4 flip a `none:` arm is not a silence: it surfaces as
//! [`Route::Unbound`](crate::x::Route::Unbound) at model **load**, and
//! `bind/mod.rs` prints it as `"{symbol}: {why}"` to whoever tried to load
//! the trace. So each opens with what the KERNEL needs and why a bind
//! cannot supply it, and only then says `FLOOR:` and names the method. Do
//! not rewrite them back into floor-speak — a model author meets these, and
//! `Cx has no query for DispatchCtx::glu_limit` tells that reader nothing
//! they can act on.
//!
//! | contract | the row's `Source` | the `Facts` method it needs |
//! |---|---|---|
//! | `mlp::swiglu_bf16` | `Or(In(1), Aux(0))` | `aux(i) -> Option<*mut c_void>` |
//! | `mlp::swiglu_clamp_bf16` | `Ctx("glu_limit")`, `Aux(0)` | `glu_limit()`, `aux` |
//! | `mlp::chunked_swiglu_clamp_bf16` | `Ctx("glu_limit")` | `glu_limit()` |
//! | `mlp::situ_bf16` | `Ctx("situ_beta")`, `Ctx("situ_linear_beta")`, `Aux(0)` | `situ()`, `aux` |
//! | `mlp::chunked_situ_bf16` | `Ctx("situ_beta")`, `Ctx("situ_linear_beta")`, `Ctx("gate_second")` | `situ()`, `gate_second()` |
//! | `mlp::gaussian_topk_bf16` | `CtxByLayer("altup_std_mult")` | `altup_std_mult(layer)` |
//!
//! Every one of these is a `DispatchCtx` field the driver already holds —
//! `glu_limit`, `situ_beta`, `situ_linear_beta`, `gate_second` and
//! `altup_std_mult_by_layer` are `bind/mod.rs:1149-1251` — so the impl side
//! is a read and the trait side is a defaulted `fn`. `Aux` is the only one
//! that is not a field: it is `join_aux(spec, i, frame, resolver)`, the op
//! join's foreign operand, and `Fire` holds `spec` already.
//!
//! `x/cx.rs` is the floor's file, so this port declares the refusals rather
//! than editing it. **`bind/mod.rs:598` attaches the `pair_up` aux slot by
//! matching the literal strings `"mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16"
//! | "mlp::situ_bf16"`** — those three symbols must keep those spellings for
//! the aux to arrive at all, which is why none of them is renamed here.

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
// TWO `unit!` INVOCATIONS CANNOT SHARE A SCOPE. The macro emits `UNITS`,
// `ROWS`, `PARAMS` and `mod raw` at the invocation site, so a family with two
// roots collides on four names. Each unit therefore gets a module of its own
// and this file re-exports the pair as [`UNITS`]. `rope` had one root and
// never met this; `quant` has seven. It is recorded in both ports because it
// is the first thing a fourth family will hit.
// ---------------------------------------------------------------------------

/// `mlp/swiglu.cuh` — the gated activations, flat and chunked.
pub mod swiglu {
    use super::{bf16, f16};
    use core::ptr::NonNull;

    unit! {
        /// `mlp`'s gated activations: fifteen instantiations over as many
        /// templates, every one at `device::bf16`.
        ///
        /// Every template here is `template <class T>` over `device::Elem<T>`
        /// and every instantiation states `device::bf16`, because bf16 is
        /// what this tree stores activations in. An fp16 MLP is fifteen
        /// lines, not fifteen translation units — the measurement
        /// `norm/elementwise`'s `residual_add_f16` made first.
        unit SWIGLU = "mlp/swiglu",
            text = include_str!("../../csrc/src/mlp/swiglu.cuh"),
            file = "mlp/swiglu.cuh";

        /// `swiglu.cuh:135` — `y = silu(gate) * up`, flat.
        fn swiglu = "mlp::device::swiglu" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::swiglu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:185` — the same with the gate clamped to `±limit`.
        fn swiglu_clamp = "mlp::device::swiglu_clamp" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
            limit: f32,
        ) where *const T, *mut T {
            "mlp::swiglu_clamp_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:206` — SiTU, which is not a swiglu variant.
        ///
        /// The tanh saturates far enough out that a bf16 intermediate loses
        /// the distinction the gate exists to make, which is why this is its
        /// own kernel and not `swiglu` with two more arguments.
        fn situ = "mlp::device::situ" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::situ_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:230` — GeGLU-tanh, which is not one either:
        /// `gelu_pytorch_tanh` on the gate is a different function.
        fn geglu_tanh = "mlp::device::geglu_tanh" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::geglu_tanh_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:247` — `y = max(x, 0)^2`.
        fn relu2 = "mlp::device::relu2" <T> (
            x: *const T,
            y: *mut T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::relu2_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:161` — gpt-oss's clamped GLU, with an optional fp16
        /// copy of the result.
        ///
        /// **`y_fp16` is `f16*` and the declaration says so.** The row world
        /// could only say `bf16*` — see this module's header for the
        /// measurement and for why the type system is the repair. It is
        /// `Option<NonNull<f16>>` because the kernel tests it (`swiglu.cuh:178`)
        /// and every caller in the tree passes nothing.
        ///
        /// The parameter ORDER is the kernel's, not the deleted launcher's:
        /// `y_fp16` is fourth here and was last there, because a defaulted
        /// C++ argument has to be, and a `void**` is built from the kernel's
        /// list.
        fn gpt_oss_glu = "mlp::device::gpt_oss_glu" <T> (
            gate: *const T,
            up: *const T,
            y: *mut T,
            y_fp16: Option<NonNull<f16>>,
            n: i32,
            limit: f32,
            alpha: f32,
        ) where *const T, *mut T {
            "mlp::gpt_oss_glu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:261` — `x *= sigmoid(gate)`, in place.
        ///
        /// Qwen3.5's per-token output gate. **This family declares the
        /// `__global__` and nothing else about it.** Its host program is
        /// [`crate::x::driver_internal::sigmoid_gate_inplace_bf16`] and its
        /// contract is deliberately absent — that module's header calls
        /// itself "the fourth arrangement", a family that is a `fn` and
        /// nothing else, because `SigmoidGateMul` is a semantic op the
        /// driver's own binder owns rather than a statement a trace records.
        ///
        /// The row still has to live here, because a row lives in the unit
        /// its FILE names and this `__global__` is in `mlp/swiglu.cuh`.
        /// `x::fire::fire` resolves that symbol through `unit_of`, so
        /// deleting `families::mlp` without declaring it here would leave
        /// `driver_internal`'s launcher naming a symbol no unit hosts.
        fn sigmoid_gate_inplace = "mlp::device::sigmoid_gate_inplace" <T> (
            x: *mut T,
            gate: *const T,
            n: i32,
        ) where *const T, *mut T {
            "mlp::sigmoid_gate_inplace_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:343` — the packed gate‖up bank, one row per block row.
        ///
        /// `chunked_swiglu` and `chunked_swiglu_gate_second` are two
        /// `__global__`s for what the body spells
        /// `chunked_swiglu_body<T, GateSecond>`: an instantiation carries
        /// exactly one template argument, so the packed layout's flag is
        /// part of a name. The dispatcher picks a symbol, which is what it
        /// was already doing with a `bool` argument.
        fn chunked_swiglu = "mlp::device::chunked_swiglu" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:349` — the same at `GateSecond = true`.
        fn chunked_swiglu_gate_second = "mlp::device::chunked_swiglu_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:435` — the packed form with the gate clamped.
        fn chunked_swiglu_clamp = "mlp::device::chunked_swiglu_clamp" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            limit: f32,
        ) where *const T, *mut T {
            "mlp::chunked_swiglu_clamp_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:378` — SiTU over a packed bank.
        fn chunked_situ = "mlp::device::chunked_situ" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::chunked_situ_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:387` — the same at `GateSecond = true`.
        fn chunked_situ_gate_second = "mlp::device::chunked_situ_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
            beta: f32,
            linear_beta: f32,
        ) where *const T, *mut T {
            "mlp::chunked_situ_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:417` — GeGLU-tanh over a packed bank.
        fn chunked_geglu_tanh = "mlp::device::chunked_geglu_tanh" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_geglu_tanh_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:425` — the same at `GateSecond = true`.
        fn chunked_geglu_tanh_gate_second = "mlp::device::chunked_geglu_tanh_gate_second" <T> (
            packed: *const T,
            y: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "mlp::chunked_geglu_tanh_gate_second_bf16" => where [T = bf16] "device::bf16",
        }

        /// `swiglu.cuh:607` — `out += y * sigmoid(x · gate_w)`, the shared
        /// expert's landing.
        ///
        /// The kernel's own comment is the geometry's citation and it names
        /// the rule outright: *"`LaunchRule::Rms` is this launcher exactly:
        /// one block per row, 256 threads, and `(256 / 32) * sizeof(float)`
        /// bytes of dynamic shared memory for the cross-warp fold."* The
        /// `extern __shared__ float smem[]` at `:619` is what those bytes
        /// are for.
        fn sigmoid_dot_scalar_gate_add = "mlp::device::sigmoid_dot_scalar_gate_add" <T> (
            x: *const T,
            gate_w: *const T,
            out: *mut T,
            y: *const T,
            h: i32,
        ) where *const T, *mut T {
            "mlp::sigmoid_dot_scalar_gate_add_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `mlp/gaussian_topk.cuh` — AltUp's activation sparsity, alone in its file
/// because it always was.
pub mod gaussian_topk {
    use super::bf16;

    unit! {
        /// gemma-3n's AltUp sparsity, one instantiation.
        unit GAUSSIAN_TOPK = "mlp/gaussian_topk",
            text = include_str!("../../csrc/src/mlp/gaussian_topk.cuh"),
            file = "mlp/gaussian_topk.cuh";

        /// `gaussian_topk.cuh:72` — zero everything below
        /// `mean + std_multiplier * stddev`, per row, in place.
        ///
        /// `n` is NOT a parameter: the grid is one block per row and the
        /// kernel reads `blockIdx.x` with no guard, so the token count is
        /// pure geometry. The `extern __shared__ float smem[]` at `:81` is
        /// the two block-wide reductions' scratch and is why the launch
        /// carries [`RMS_SMEM`](super::RMS_SMEM) bytes.
        fn gaussian_topk = "mlp::device::gaussian_topk" <T> (
            x: *mut T,
            dim: i32,
            std_multiplier: f32,
        ) where *mut T {
            "mlp::gaussian_topk_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// The units `mlp` compiles.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above gives. `families::ALL` reads this.
pub static UNITS: &[Unit] = &[swiglu::SWIGLU, gaussian_topk::GAUSSIAN_TOPK];

// ---------------------------------------------------------------------------
// The numbers, once each.
// ---------------------------------------------------------------------------

/// Threads per block, everywhere in this family.
///
/// `driver-cuda/src/bind/launch.rs:100` — the `BLOCK` every one of the three
/// rules this family used is written over. It is fixed rather than sized on
/// the row because the two reducing kernels' `block_sum` folds warp by warp:
/// a different block width sums the same values in a different order and
/// answers with a different last bit.
const BLOCK: u32 = 256;

/// Threads per warp — the unit the reductions' shared scratch is counted in.
///
/// `driver-cuda/src/bind/launch.rs:106`.
const WARP: u32 = 32;

/// The dynamic shared memory the two reducing kernels fold through.
///
/// `driver-cuda/src/bind/launch.rs:123` — `(BLOCK / WARP) * 4`, one `float`
/// per warp. `sigmoid_dot_scalar_gate_add` and `gaussian_topk` both declare
/// `extern __shared__ float smem[]` and read back `blockDim.x >> 5` entries,
/// so this is sized by the block and not by the row: **the same function
/// writes the `smem` and states the block**, which is the structural reason
/// `LaunchRule::Rope`'s over-allocation cannot recur here (see
/// [`crate::x::launch`]'s header).
const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `bind/launch.rs:128` — `grid [ceil(n / 256), 1, 1]`, `block [256, 1, 1]`,
/// no shared memory. Every flat activation here launched through it, and
/// `n` stays a kernel argument in all of them because the grid rounds UP:
/// the last block runs threads the buffer does not have and the kernel's
/// `if (idx >= n) return;` is what stops them.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// `LaunchRule::ElementwiseRows`, as the expression it evaluates to.
///
/// `bind/launch.rs:143` — `grid [rows, ceil(width / 256), 1]`,
/// `block [256, 1, 1]`, no shared memory. The chunked kernels take the row
/// off `blockIdx.x` and the column off `blockIdx.y * blockDim.x +
/// threadIdx.x`, so the row axis must be the grid's x.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
///
/// `bind/launch.rs:116` — `grid [rows, 1, 1]`, `block [256, 1, 1]`,
/// [`RMS_SMEM`] bytes. Both callers declare `extern __shared__`.
#[must_use]
const fn rms(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// gpt-oss's `alpha`, which was a defaulted argument of a header that no
/// longer exists.
///
/// `table/mlp.rs` spelled it `Source::Lit(Lit::F32(1.702))` and said why,
/// and the sentence survives the move: *"a default in a header is a fact
/// about the launcher that no caller can see it relying on."* It is a named
/// constant here for the same reason and is passed explicitly by the one
/// bind that has no other value for it.
pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

// ---------------------------------------------------------------------------
// Truth two: the host programs.
// ---------------------------------------------------------------------------

/// `y[i] = silu(gate[i]) * up[i]` over `n` elements — `mlp::swiglu_bf16`.
///
/// # Safety
///
/// `gate` and `up` must address `n` live bf16 elements, `y` `n` writable
/// ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn swiglu_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::swiglu("mlp::swiglu_bf16", elementwise(n), gate, up, y, n, stream);
    }
    Fired::Launched
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// `limit` is a config constant of the deployment (gpt-oss's
/// `swiglu_limit`), which is why this is a different kernel rather than
/// `swiglu` with an argument that is usually infinite.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn swiglu_clamp_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    limit: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::swiglu_clamp(
            "mlp::swiglu_clamp_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            limit,
            stream,
        );
    }
    Fired::Launched
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn situ_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    beta: f32,
    linear_beta: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::situ(
            "mlp::situ_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            beta,
            linear_beta,
            stream,
        );
    }
    Fired::Launched
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// In place on the gate: `table/mlp.rs` declared `in_place = &[(0, 0)]` and
/// the contract below keeps it. gemma-4's PLE gate is the same call with the
/// per-layer relay slice as `up`, which is a placement the caller makes —
/// and is what let the row be stated at all, since with the whole table as
/// operand 1 there was no expression for "plus l · N · ple_dim".
///
/// # Safety
///
/// [`swiglu_bf16`]'s. `y` may alias `gate`.
#[cfg(feature = "_cuda")]
pub unsafe fn geglu_tanh_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::geglu_tanh(
            "mlp::geglu_tanh_bf16",
            elementwise(n),
            gate,
            up,
            y,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `y = max(x, 0)^2` — `mlp::relu2_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements, `y` `n` writable ones, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn relu2_bf16(
    x: *const bf16,
    y: *mut bf16,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::relu2("mlp::relu2_bf16", elementwise(n), x, y, n, stream);
    }
    Fired::Launched
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// `y_fp16` is the second output the MXFP4 decode GEMV reads: emitting it
/// from the same fp32 the bf16 rounds from is what deleted a cast launch.
/// It is `Option<NonNull<f16>>` and **not** a `*mut bf16`, which is the
/// repair this port carries — see the module header for the measurement the
/// row world made and could not act on.
///
/// `alpha` has no caller that varies it; [`GPT_OSS_GLU_ALPHA`] is the
/// header default the row world spelled as a `Lit`.
///
/// # Safety
///
/// [`swiglu_bf16`]'s, plus: when `y_fp16` is `Some`, it must address `n`
/// writable **fp16** elements. `y` may alias `gate`.
#[cfg(feature = "_cuda")]
pub unsafe fn gpt_oss_glu_bf16(
    gate: *const bf16,
    up: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    n: i32,
    limit: f32,
    alpha: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        swiglu::raw::gpt_oss_glu(
            "mlp::gpt_oss_glu_bf16",
            elementwise(n),
            gate,
            up,
            y,
            y_fp16,
            n,
            limit,
            alpha,
            stream,
        );
    }
    Fired::Launched
}

/// SwiGLU over a packed gate‖up bank — `mlp::chunked_swiglu_bf16`.
///
/// `rows` is the OUTPUT's row count and not the fire's: **three callers
/// share this kernel and one of them is the routed MoE leg, whose rows are
/// the padded block-major count rather than the fire's tokens.** The row
/// world said the same thing by binding `Source::OutRows(0)`, and binding
/// the fire's rows instead would activate the first N of the padded rows and
/// leave the rest holding whatever the GEMM wrote.
///
/// `gate_second` picks a SYMBOL rather than passing a `bool`: the two
/// `__global__`s are one body at two values of a template parameter, and the
/// branch leaves the inner loop.
///
/// # Safety
///
/// `packed` must address `rows * 2 * i` live bf16 elements and `y`
/// `rows * i` writable ones. `y` may alias the second half of `packed`,
/// which is what `in_place = &[(0, 1)]` declares. `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_swiglu_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_swiglu_gate_second(
                "mlp::chunked_swiglu_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        } else {
            swiglu::raw::chunked_swiglu(
                "mlp::chunked_swiglu_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        }
    }
    Fired::Launched
}

/// The packed form with the gate clamped —
/// `mlp::chunked_swiglu_clamp_bf16`.
///
/// One `__global__` and no `gate_second` twin: the header has never carried
/// one, and inventing a symbol for a kernel that does not exist is what a
/// declaration is for refusing.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_swiglu_clamp_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    limit: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    unsafe {
        swiglu::raw::chunked_swiglu_clamp(
            "mlp::chunked_swiglu_clamp_bf16",
            elementwise_rows(rows, i),
            packed,
            y,
            i,
            limit,
            stream,
        );
    }
    Fired::Launched
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_situ_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    beta: f32,
    linear_beta: f32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_situ_gate_second(
                "mlp::chunked_situ_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                beta,
                linear_beta,
                stream,
            );
        } else {
            swiglu::raw::chunked_situ(
                "mlp::chunked_situ_bf16",
                launch,
                packed,
                y,
                i,
                beta,
                linear_beta,
                stream,
            );
        }
    }
    Fired::Launched
}

/// GeGLU-tanh over a packed bank — `mlp::chunked_geglu_tanh_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn chunked_geglu_tanh_bf16(
    packed: *const bf16,
    y: *mut bf16,
    rows: i32,
    i: i32,
    gate_second: bool,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = elementwise_rows(rows, i);
    unsafe {
        if gate_second {
            swiglu::raw::chunked_geglu_tanh_gate_second(
                "mlp::chunked_geglu_tanh_gate_second_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        } else {
            swiglu::raw::chunked_geglu_tanh(
                "mlp::chunked_geglu_tanh_bf16",
                launch,
                packed,
                y,
                i,
                stream,
            );
        }
    }
    Fired::Launched
}

/// `out += y * sigmoid(x · gate_w)` — `mlp::sigmoid_dot_scalar_gate_add_bf16`.
///
/// The shared expert's landing, with the gate computed in the same launch
/// rather than in one of its own.
///
/// **`y` is the ADDEND and `out` is the accumulator**, and the order is not
/// symmetric: the hand-written arm this replaced carried a warning that
/// reversing the last two lands the gate on the wrong buffer and still
/// compiles. The parameter names are the kernel's, so a call site reads
/// which is which.
///
/// # Safety
///
/// `x`, `y` and `out` must each address `rows * h` live bf16 elements —
/// `out` writable, and it IS the residual stream the statement takes as its
/// second operand, which is what `in_place = &[(0, 1)]` declares. `gate_w`
/// must address `h` live bf16 elements. `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn sigmoid_dot_scalar_gate_add_bf16(
    x: *const bf16,
    gate_w: *const bf16,
    out: *mut bf16,
    y: *const bf16,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        swiglu::raw::sigmoid_dot_scalar_gate_add(
            "mlp::sigmoid_dot_scalar_gate_add_bf16",
            rms(rows),
            x,
            gate_w,
            out,
            y,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// AltUp's activation sparsity, in place — `mlp::gaussian_topk_bf16`.
///
/// `std_multiplier` is the driver's own derivation and not a config field:
/// the config states `activation_sparsity` and the kernel wants
/// `gaussian_inverse_cdf` of it, per layer.
///
/// # Safety
///
/// `x` must address `rows * dim` live and writable bf16 elements and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn gaussian_topk_bf16(
    x: *mut bf16,
    rows: i32,
    dim: i32,
    std_multiplier: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "dim" });
    }
    unsafe {
        gaussian_topk::raw::gaussian_topk(
            "mlp::gaussian_topk_bf16",
            rms(rows),
            x,
            dim,
            std_multiplier,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// What a trace may say.
//
// Twelve contracts, the twelve `table::mlp` rows. Only `in_place` is stated
// — the other nine `Contract` fields are `Contract::DEFAULT` for every one of
// them, which is what a family of pointwise activations should look like:
// nothing here is `whole`, nothing sinks, nothing publishes an aux and
// nothing lowers under another name.
//
// `mlp::sigmoid_gate_inplace_bf16` is a THIRTEENTH `__global__` in this
// family's unit and it gets NO contract, because `x/driver_internal.rs`
// already owns it as a `fn` and nothing else. Declaring one here would put
// an `Entry` in `FAMILIES` for a symbol the driver's own binder calls
// directly, and `x::route`'s "the one overlap" is precisely that hazard.
// ---------------------------------------------------------------------------

contract! {
    /// SwiGLU over a packed gate‖up bank.
    ///
    /// The ALIGNED leg's spelling states a second operand — the staging the
    /// pointer build named — and writes its result over it. The dense and
    /// shared-expert spellings state one operand, and a pair outside a
    /// statement's arity is not an error (`lower::Buffers`), so one contract
    /// serves all three.
    CHUNKED_SWIGLU_BF16 = "mlp::chunked_swiglu_bf16" as chunked_swiglu {
        in_place: &[(0, 1)],
    }

    /// SwiGLU over two narrow buffers.
    ///
    /// **The symbol string is load-bearing beyond this file.**
    /// `driver-cuda/src/bind/mod.rs:598` matches
    /// `"mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16" | "mlp::situ_bf16"`
    /// literally to attach the `pair_up` aux slot. A rename here is a
    /// silently missing `up` projection there.
    SWIGLU_BF16 = "mlp::swiglu_bf16" as swiglu

    /// SwiGLU with the gate clamped — gpt-oss's `swiglu_limit`.
    SWIGLU_CLAMP_BF16 = "mlp::swiglu_clamp_bf16" as swiglu_clamp

    /// The packed form of the same.
    CHUNKED_SWIGLU_CLAMP_BF16 = "mlp::chunked_swiglu_clamp_bf16" as chunked_swiglu_clamp

    /// `y = max(x, 0)^2`.
    RELU2_BF16 = "mlp::relu2_bf16" as relu2

    /// SiTU, which is not a swiglu variant.
    SITU_BF16 = "mlp::situ_bf16" as situ

    /// The packed form of the same.
    CHUNKED_SITU_BF16 = "mlp::chunked_situ_bf16" as chunked_situ

    /// AltUp's activation sparsity, in place.
    GAUSSIAN_TOPK_BF16 = "mlp::gaussian_topk_bf16" as gaussian_topk {
        in_place: &[(0, 0)],
    }

    /// GeGLU-tanh over two narrow buffers, the gate half in place.
    GEGLU_TANH_BF16 = "mlp::geglu_tanh_bf16" as geglu_tanh {
        in_place: &[(0, 0)],
    }

    /// The packed form of the same.
    CHUNKED_GEGLU_TANH_BF16 = "mlp::chunked_geglu_tanh_bf16" as chunked_geglu_tanh

    /// gpt-oss's clamped GLU. `gate = glu(gate, up)`, so the gate half is
    /// the destination.
    GPT_OSS_GLU_BF16 = "mlp::gpt_oss_glu_bf16" as gpt_oss_glu {
        in_place: &[(0, 0)],
    }

    /// The shared expert's landing: `out += sigmoid(x · gate) * y`, where
    /// `out` IS the residual stream the statement takes as operand 1 — the
    /// header calls it "in-place add destination" in as many words.
    MOE_SHARED_GATE_DOT_BF16 = "mlp::sigmoid_dot_scalar_gate_add_bf16" as moe_shared_gate_dot {
        in_place: &[(0, 1)],
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Six binds and six written refusals, and every one of the six is the same
// shape: a fact the driver holds and `Cx` has no query for. The host program
// above each takes the value as an ordinary argument, so each refusal is one
// defaulted `Facts` method from being three lines. The module header tabulates
// which method, per contract.
//
// `//` and not `///`: these are elements of an array expression and Rust has
// no attributes there.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    CHUNKED_SWIGLU_BF16 => { cx, stream => {
        // `Source::OutRows(0)`, which `bind/mod.rs:1596`'s `rows_of` answers
        // with the region's row count — the same number `Cx::rows().count`
        // is. `gate_second` was `Source::Lit(Lit::Bool(false))` on this row
        // and is a literal `false` here for exactly as long as that stays
        // true; the two-symbol `if` is inside the host program.
        unsafe {
            chunked_swiglu_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                false,
                stream,
            )
        }
        .ok()
    }},

    // `up` is `Source::Or(&Source::In(1), &Source::Aux(0))`: a trace that
    // split the packed projection states both halves, and one that did not
    // leaves `up` to the join, which collected it as the statement's foreign
    // operand. `Cx` answers the first half and has no query for the second,
    // and the fallback is not optional — `bind/mod.rs:598` attaches the
    // `pair_up` aux slot to exactly this symbol, so the traces that need it
    // are the ones that reach here.
    SWIGLU_BF16 => { none: "this kernel reads the up projection's second \
        half, and a trace that leaves the projection packed never names it \
        -- the driver's op join supplies it, and a bind cannot ask the join \
        for anything. FLOOR: `up` is Source::Or(In(1), Aux(0)) and `Cx` \
        answers only In(1); needs `Facts::aux(i) -> Option<*mut c_void>`, \
        which is `join_aux(spec, i, frame, resolver)` and one defaulted \
        method" },

    SWIGLU_CLAMP_BF16 => { none: "this kernel needs the model's GLU clamp \
        limit and the up projection's second half, and a bind can ask for \
        neither. FLOOR: DispatchCtx::glu_limit (bind/mod.rs:1193) and the \
        join's foreign operands; needs `Facts::glu_limit()` and \
        `Facts::aux(i)`" },

    CHUNKED_SWIGLU_CLAMP_BF16 => { none: "this kernel needs the model's GLU \
        clamp limit, and a bind cannot ask for it. FLOOR: \
        DispatchCtx::glu_limit (bind/mod.rs:1193); needs \
        `Facts::glu_limit()`, one defaulted method over a field the driver \
        already holds" },

    RELU2_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            relu2_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    SITU_BF16 => { none: "this kernel needs the model's two SITU betas and \
        the up projection's second half, and a bind can ask for neither. \
        FLOOR: DispatchCtx::situ_beta and situ_linear_beta \
        (bind/mod.rs:1200-1202) and the join's foreign operands; needs \
        `Facts::situ() -> Option<(f32, f32)>` and `Facts::aux(i)`" },

    CHUNKED_SITU_BF16 => { none: "this kernel needs the model's two SITU \
        betas and which half of the packed projection is the gate, and a \
        bind can ask for neither. FLOOR: DispatchCtx::situ_beta, \
        situ_linear_beta (bind/mod.rs:1200-1202) and gate_second \
        (bind/mod.rs:1149); needs `Facts::situ()` and \
        `Facts::gate_second() -> bool`" },

    GAUSSIAN_TOPK_BF16 => { none: "this kernel needs the layer's altup \
        standard-deviation multiplier, which is a per-layer model constant, \
        and a bind cannot ask for it. FLOOR: the row bound \
        Source::CtxByLayer(\"altup_std_mult\") and \
        DispatchCtx::altup_std_mult(layer) (bind/mod.rs:1310) is the \
        accessor; needs `Facts::altup_std_mult(layer) -> Option<f32>`, and \
        `Cx::layer()` already answers the index" },

    GEGLU_TANH_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            geglu_tanh_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                // gemma-4's PLE gate states a `select` of the per-layer relay
                // here, so the layer offset an arm used to add is a placement
                // the host makes and this is an ordinary second input.
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    CHUNKED_GEGLU_TANH_BF16 => { cx, stream => {
        // `Source::Rows` on this row where its siblings say `OutRows(0)`;
        // `rows_of` answers both with the region's count, so one reading
        // serves.
        unsafe {
            chunked_geglu_tanh_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                false,
                stream,
            )
        }
        .ok()
    }},

    GPT_OSS_GLU_BF16 => { cx, stream => {
        let n = elements(cx)?;
        unsafe {
            gpt_oss_glu_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                // `Source::Lit(Lit::Null)`: nothing in this tree asks for the
                // fp16 copy through a trace. `None` is the same absence with
                // the format in the type — see the module header.
                None,
                n,
                cx.param_f32(0)?,
                GPT_OSS_GLU_ALPHA,
                stream,
            )
        }
        .ok()
    }},

    MOE_SHARED_GATE_DOT_BF16 => { cx, stream => {
        unsafe {
            sigmoid_dot_scalar_gate_add_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                // Operand 2, the ADDEND — not the accumulator. Reversing the
                // last two lands the gate on the wrong buffer and compiles.
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},
}

/// `Source::OutElements(0)`, in the vocabulary `Cx` has.
///
/// `bind/mod.rs:1600`'s `elems_of(b, n_in + 0, rows)` is
/// `rows * width_of(b, n_in + 0)` — the region's row count times the first
/// output's row width, and neither factor is the whole tensor's. This is
/// that product, with the saturation the row world's `i32::try_from(..)
/// .unwrap_or(0)` had: a rectangle whose element count does not fit an `i32`
/// is a rectangle this kernel's `i32 n` cannot bound, and the host programs
/// above decline a non-positive `n` rather than launch a grid over the low
/// bits.
///
/// # Errors
///
/// [`Refusal::Absent`] when the statement states no first output.
#[cfg(feature = "_cuda")]
fn elements(cx: &crate::x::Cx<'_>) -> Result<i32, Refusal> {
    let rows = cx.rows().count;
    let width = cx.out_width(0)?;
    Ok(rows.saturating_mul(width))
}
