//! What a trace that states one of `norm`'s symbols binds to.
//!
//! Positional bank versus named bank: `Bank<N, T>` reads an `OpKind::Launch`'s
//! `Arg::Weight` operands, `Weight<N, T>` reads `LaunchSpec::weight`. Both are
//! `*const T`, so a swap below index two is caught only at LAUNCH.

use super::Bound;


/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound::derived("norm::rmsnorm_strided_bf16"),

    // `hidden` is `*mut` bound from `arg_in(0)`: the statement declares a
    // written buffer as an input and no result for it.
    Bound::derived("norm::residual_add_rmsnorm_bf16"),
    // Marking the weight takes it out of the input run, so `x` keeps `In(0)`.
    Bound::derived("norm::rmsnorm_residual_add_bf16"),
    // `bias: Weight<*const T>` reads the NAMED bank; a bare `*const T` here
    // would derive `In(0)`.
    Bound::derived("norm::add_bias_bf16"),

    Bound::derived("norm::hc_rmsnorm_to_f32"),

    Bound::derived("norm::hc_expand_bf16"),
    Bound::derived("norm::hc_post_bf16"),

    // `Env<keys::HeadDim>` and deliberately not `cx.kv_layer()?.head_dim`: that
    // is the CACHE's head width, a different fact that agrees on most
    // checkpoints.
    Bound::derived("norm::per_head_rmsnorm_bf16"),
    // `lse` binds `In(1)` because a fact consumes no position counter.
    Bound::derived("norm::attn_sink_correction_bf16"),
    // The POSITIONAL bank. Its sibling `rmsnorm_gated_fp32_in_bf16` takes the
    // same pointer through the NAMED bank.
    Bound::derived("norm::rmsnorm_gated_bf16"),

    Bound::derived("norm::altup_unpack_correct_coefs"),
    Bound::derived("norm::tanh_bf16"),

    // Both gammas take `Weight<0, *const T>`, the named bank: a statement with a
    // positional weight zero always has `spec.weight == Some(weights[0])`.
    Bound::derived("norm::rmsnorm_bf16"),
    Bound::derived("norm::rmsnorm_gemma_bf16"),

    // `x: In<0, T>` is stated: this launcher spends two pointers on the aliased
    // buffer where `residual_add_bf16` below spends one, so the two rows look
    // identical to derivation and need different indices.
    Bound::derived("norm::rmsnorm_no_scale_bf16"),

    // `k` REFUSES a non-square here and not on `altup_predict_bf16`: `k*k` is a
    // BLOCK width, and a zero-width block writes nothing and returns `Ok(())`.
    Bound::derived("norm::altup_unpack_predict_coefs"),
    // `Lit(F32(ALTUP_EPS))` is NOT the deployment's epsilon; they agree by
    // accident.
    Bound::derived("norm::compute_rms_bf16"),
    Bound::derived("norm::magnitude_rescale_bf16"),

    // `x` is `In<1, _>` here and `In<0, _>` on `rmsnorm_no_scale_bf16`. The
    // device count is a `usize`: narrowing to `i32` would hand four bytes to an
    // eight-byte parameter.
    Bound::derived("norm::residual_add_bf16"),

    Bound::derived("norm::scalar_mul_bf16"),

    // ── The rows with no arm: `route` refuses these at LOAD ────────────
    Bound::derived("norm::rmsnorm_bf16_with_fp16"),
    // Its weight is the NAMED bank, unlike `norm::rmsnorm_gated_bf16`. The
    // obvious repair for the blocker below is a trap: the named source resolves
    // to zero rather than refusing, and zero normalises over `k * v_d`.
    // The head width is `keys::GdnVDim`, which this driver has always
    // answered; the row's reason asked for a `Facts` query that was not the
    // one it needed. See the routine.
    Bound::derived("norm::rmsnorm_gated_fp32_in_bf16"),
    // The two weights are `Bank<0, _>`/`Bank<1, _>`; a counted `In(1)` is the
    // residual STREAM and hands a gamma reader a slab of activations.
    // The statement carries the residual scale now, so the reason this row
    // gave -- "needs `Facts::layer_scale()`, and no statement carries it" --
    // is answered from the other side: gemma-4's builder states it, and no
    // deployment ever published a `layer_scale` to be asked for.
    Bound::derived("norm::rmsnorm_residual_add_scale_rmsnorm_bf16"),
    Bound {
        symbol: "norm::hc_pre_postprocess_bf16",
        arm: None,
        unbound: Some(
            "a hyper-connection layer's slabs -- mixes, scale, base -- and its two scratch buffers",
        ),
    },
    Bound {
        symbol: "norm::hc_head_postprocess_bf16",
        arm: None,
        unbound: Some("the same hyper-connection slabs, and `hc_eps`"),
    },
    // The launcher refuses a non-square and an inexact quotient.
    Bound::derived("norm::altup_predict_bf16"),
    Bound {
        symbol: "norm::altup_correct_bf16",
        arm: None,
        unbound: Some(
            "which AltUp stream was run through the real layer; needs `Facts::altup_active()`",
        ),
    },
    // The stream count is an extent: the operand is `k*h` wide, the result `h`.
    Bound::derived("norm::mean_streams_bf16"),
];
