//! What happens when a trace states one of `gemm`'s symbols — and for two of
//! the three, what does not.
//!
//! # Why this file did not exist, and why its absence was the defect
//!
//! Every other family got a registry when its `bind!` block moved out of
//! `kernels-cuda`. `gemm` never had one, because the matmuls were never
//! `bind!`ed: they were `execution::RUST_SERVED` rows reached through a
//! GENERATED dispatch, and when that generator was deleted the arms it wrote
//! went with it and nothing took their place. A missing registry file looks
//! like a family with nothing to say, which is the one thing it was not.
//!
//! The consequence is what `executor_bind.rs`'s
//! `every_lowered_symbol_runs_or_says_why_not` measures. A symbol no registry
//! mentions answers [`Route::Rows`](super::Route::Rows), falls through to
//! `bind::dispatch`'s hand match, misses every arm and refuses `NoArm` — a
//! message naming neither what is missing nor who would supply it. Three
//! symbols live deployments lower to were in that state, and a bridge ROW
//! test could not see it: a row and an arm are different questions, and every
//! one of the three has a row.
//!
//! # The three, and they are three different answers
//!
//! * `gemm::lora_qkv_correction` — **runs**, in `bind::dispatch`'s hand
//!   match, and is [`Bound::driver`] here so the registry says so.
//! * `gemm::act_x_w` and `gemm::act_x_wt_bf16_out_fp32` — **do not run**, and
//!   have not since the generated dispatch was deleted. Their entries carry
//!   the reason. This is a pre-existing gap being written down, not one being
//!   opened or closed.
//!
//! # The falsehood this file exists partly to stop repeating
//!
//! `bind/mod.rs` recorded the gap and recorded what a fire said about it:
//! three `gemm::` contracts *"are in neither `SERVED` nor `table::`, so
//! `route` answers `Route::Unknown` and the arm below prints 'no contract and
//! no row declares it' — **which is false; a contract does declare all
//! three.**"* Those three crossed and are `driver_bound!` rows in
//! `kernels_cuda::gemm` now, so the declaration is not merely a contract
//! any more; it is a `fn`. The message was wrong then and would be wronger
//! now, and none of the reasons below repeats it.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;

use super::super::cx::Cx;
use super::Bound;

/// `gemm::act_x_w` — the dense matmul, under the spelling the lowering emits.
///
/// `y = act @ Wᵀ`, with `act` `[m, k]`, `W` `[n, k]` and `y` `[m, n]`. The
/// body is `kernels_cuda::gemm::act_x_wt_bf16`, a derived routine; what was
/// missing was only this join, which the generated dispatch used to write and
/// which went with its generator.
///
/// # The rename is honoured HERE and not by `RENAMED_AT_THE_ABI`
///
/// `executor_bind.rs` maps this spelling to `gemm::act_x_wt_bf16` for the ROW
/// question — may a lowered symbol be declared. This is the CALL, and the two
/// are answered separately on purpose: sharing one registry entry between two
/// symbols is what would have to be unpicked the day the portable spelling
/// needs different operand handling, which is the only reason a portable
/// spelling exists at all.
///
/// The weight arrives through the OP JOIN rather than the operand list —
/// see the call site — which is the half of this arm that could not be
/// guessed from the signature.
///
/// # `beta`, which is the one thing here that can silently corrupt
///
/// Zero unless the statement carries an accumulator. `lower::walk`'s epilogue
/// says it in its own words — *"beta is 0 here — the accumulate form needs
/// three operands and this op has two"* — so the operand count IS the fact,
/// and reading it off the presence of a second input is reading the same
/// thing the lowering wrote. `y += matmul(..)` is what produces the three-
/// operand form: the tape rewrites it to the `beta_one` accumulate rather
/// than emitting a separate add, and the output id is unchanged, so nothing
/// downstream can tell the two apart and nothing else here can either.
fn act_x_w_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let beta = if cx.arg_in(1).is_ok() { 1.0 } else { 0.0 };
    // SAFETY: `stream` is this fire's and outlives the launch, and
    // `cx.cublas()` is the engine's handle with that same stream bound --
    // `with_cublas`' obligation is exactly that pairing, and `DispatchCtx`
    // carries the two together so that a caller cannot supply half of it.
    let ctx = unsafe { Ctx::on(stream).with_cublas(cx.cublas()) };
    kernels_cuda::gemm::act_x_wt_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        // `weight_named`, not `weight`. The latter is a weight OPERAND —
        // `arg(n_in + n_out + i)`, a position in the statement's own list —
        // and a dense matmul names its weight by NAME: the DSL's `MatW` is a
        // typed handle from a per-layer namespace, so `spec.weight` carries
        // the string and the op join resolves it into `w_named` before the
        // arm runs. Asking for the operand got `the fire does not carry a
        // weight`, which was true and about the wrong question.
        cx.weight_named(0)?.cast_const(),
        cx.arg_out(0)?,
        cx.rows().count,
        cx.out_width(0)?,
        cx.in_width(0)?,
        beta,
    )
}

/// Every symbol this family accounts for.
///
/// **Not one arm**, and the shape of the list says why for each entry rather
/// than leaving a reader to infer it from three `None`s.
pub static ARMS: &[Bound] = &[
    // ── THE ONE THAT RUNS ───────────────────────────────────────────────
    //
    // `bind::dispatch`'s `"gemm::lora_qkv_correction"` arm reaches
    // `kernels_cuda::gemm::lora_qkv_correction` with a `Staged` borrowed
    // out of the fire's `LoraFireState`. It cannot be an [`Arm`] here, and
    // the reason is `northstar.md` §3.3 rather than effort: an `Arm` receives
    // a [`Cx`], which is QUERY-ONLY, and this body needs three things a `Cx`
    // must not offer — `DispatchCtx::cublas` (a device API with a settable
    // stream, a math mode and a workspace), the fire-scoped `LoraFireState`
    // pointer, and an aux slot the `&mut Resolver` owns. A `Cx` that could
    // hand any of those over is precisely the surface §3.3 says must not
    // exist.
    //
    // So the entry buys the ACCOUNT and not the dispatch, which is the whole
    // of what [`Bound::driver`] is for.
    //
    // [`Arm`]: super::Arm
    // [`Cx`]: super::super::cx::Cx
    Bound::driver("gemm::lora_qkv_correction"),
    // ── THE TWO THAT DO NOT, AND ARE NOT THIS CHANGE'S TO FIX ───────────
    //
    // Both are `arm: None` with a reason, which is a LOAD-time refusal:
    // `Route::refusal` turns it into an `Unfireable` and `fire::launch`
    // rejects the model before it fires. That is the designed consequence and
    // the load path argues for it in place -- *"an unfireable symbol fails
    // under `Union` at capture exactly as it fails under `Resolve` at the
    // fire, so moving the failure here costs no working model. What it buys
    // is the difference between a token-time `NoArm` naming a kernel and a
    // load-time refusal naming the kernel, the reason and the family."*
    Bound { symbol: "gemm::act_x_w", arm: Some(act_x_w_arm), unbound: None },
    Bound {
        symbol: "gemm::act_x_wt_bf16_out_fp32",
        arm: None,
        unbound: Some(
            "one `cublasGemmEx`, bf16 in and fp32 out, whose body is the \
             derived routine `kernels_cuda::gemm::act_x_wt_bf16_out_fp32` \
             and whose arm was the generated dispatch's. Same shape as \
             `gemm::act_x_w` above and the same remedy: an arm reading the \
             two inputs, the fp32 output and `M`/`N`/`K`, plus `ctx.cublas`, \
             which is what keeps it out of this file",
        ),
    },
];
