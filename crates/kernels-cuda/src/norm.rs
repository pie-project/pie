//! Normalization and the residual stream — RMSNorm, the fused landings,
//! gemma-3n's AltUp and deepseek-v4's hyper-connections.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(rmsnorm_gated_launch "norm::rmsnorm_gated_bf16",
        operands = operands![
            x: Buf,
            gate: Buf,
            weight: Buf,
            y: BufMut,
            num_rows: I32,
            hidden: I32,
            eps: F32,
            stream: Stream,
        ]),
    kernel!(rmsnorm_strided "norm::rmsnorm_strided_bf16",
        operands = operands![
            x: Buf,
            weight: Buf,
            y: BufMut,
            num_rows: I32,
            hidden: I32,
            x_row_stride: I32,
            y_row_stride: I32,
            eps: F32,
            stream: Stream,
        ]),
    // gemma-4's end-of-layer shape: the scale sits BETWEEN the add and the
    // norm, which is why it is not `residual_add_rmsnorm` with a multiply
    // somewhere.
    kernel!(residual_add_scale_rmsnorm "norm::residual_add_scale_rmsnorm_bf16",
        operands = operands![
            hidden: BufMut,
            residual: Buf,
            scale: F32,
            next_weight: Buf,
            norm_out: BufMut,
            num_rows: I32,
            hidden_size: I32,
            eps: F32,
            stream: Stream,
        ]),
    // gpt-oss ships its experts as MXFP4 -- 4-bit values with an E8M0
    // exponent byte per block of 32 -- and mixtral's shell runs them through
    // Marlin. Several of these operate on WEIGHTS rather than activations
    // (repacking a scale layout, splitting a fused bias) and have no token
    // extent at all; they are declared because they are launches the fire
    // performs.
    kernel!(add_bias_strided "norm::add_bias_bf16_strided",
        operands = operands![
            out: BufMut,
            bias: Buf,
            num_rows: I32,
            dim: I32,
            stride: I32,
            stream: Stream,
        ]),
    // The fp16 copy is what the MXFP4 grouped GEMM consumes; producing it
    // here rather than casting afterwards is the binding.
    // ── the two the SEMANTIC `Rmsnorm` fans to ─────────────────────
    //
    // Rows because they had none, and they had none because nothing
    // STATES them: `OpKind::Rmsnorm` carries a variant and each driver
    // picks between these two from it. That makes them the only pair in
    // the tree whose operand contract was written nowhere — every other
    // kernel a semantic kind fans to is also stated by something, so it
    // has a row already.
    //
    // Adding the rows is what lets a text name them directly, which is
    // step 2 of DSL-DESIGN.md: the fold is a fact of the STATEMENT, and
    // a driver that reads it off a param is a driver choosing a kernel.
    kernel!(rmsnorm "norm::rmsnorm_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            y: BufMut <- Source::Out(0),
            num_rows: I32 <- Source::Rows,
            hidden: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Gemma folds `(1 + w)` instead of `w` — different arithmetic, same
    // signature, same row space.
    kernel!(rmsnorm_gemma "norm::rmsnorm_gemma_bf16",
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            y: BufMut <- Source::Out(0),
            num_rows: I32 <- Source::Rows,
            hidden: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(rmsnorm_with_fp16 "norm::rmsnorm_bf16_with_fp16",
        operands = operands![
            x: Buf,
            weight: Buf,
            y: BufMut,
            y_fp16: BufMut,
            num_rows: I32,
            hidden: I32,
            eps: F32,
            stream: Stream,
        ]),
    // The SECOND rank-K residual scheme here, and not AltUp's. gemma-3n
    // predicts each stream from a learned combination and corrects from one
    // ACTIVE stream; HC mixes with a per-token, sinkhorn-normalized matrix
    // and has no active stream -- every layer reads a weighted collapse of
    // all of them and writes back to all of them. Row-shaped throughout.
    kernel!(hc_rmsnorm_to_f32 "norm::hc_rmsnorm_to_f32",
        operands = operands![
            input: Buf,
            output: F32sMut,
            n: I32,
            dim: I32,
            eps: F32,
            stream: Stream,
        ]),
    // Where a rank-K residual BEGINS: replicate the embedding into K
    // streams. AltUp's equivalent is implicit in gemma-3n's workspace
    // layout; HC states it, which is the one a declaration can read.
    kernel!(hc_expand "norm::hc_expand_bf16",
        operands = operands![
            input: Buf,
            output: BufMut,
            n: I32,
            hc_mult: I32,
            hidden_size: I32,
            stream: Stream,
        ]),
    kernel!(hc_pre "norm::hc_pre_postprocess_bf16",
        operands = operands![
            mixes: F32s,
            scale: F32s,
            base: F32s,
            residual: Buf,
            post_mix: F32sMut,
            comb_mix: F32sMut,
            layer_input: BufMut,
            n: I32,
            hc_mult: I32,
            hidden_size: I32,
            hc_eps: F32,
            hc_post_alpha: F32,
            sinkhorn_iters: I32,
            stream: Stream,
        ]),
    kernel!(hc_post "norm::hc_post_bf16",
        operands = operands![
            x: Buf,
            residual: Buf,
            post_mix: F32s,
            comb_mix: F32s,
            out_residual: BufMut,
            n: I32,
            hc_mult: I32,
            hidden_size: I32,
            stream: Stream,
        ]),
    kernel!(hc_head "norm::hc_head_postprocess_bf16",
        operands = operands![
            mixes: F32s,
            scale: F32s,
            base: F32s,
            residual: Buf,
            out: BufMut,
            n: I32,
            hc_mult: I32,
            hidden_size: I32,
            stream: Stream,
            hc_eps: F32,
        ]),
    kernel!(per_head_rmsnorm "norm::per_head_rmsnorm_bf16",
        operands = operands![
            q: BufMut,
            n: I32,
            num_heads: I32,
            head_dim: I32,
            eps: F32,
            stream: Stream,
        ]),
    // Residual add + the next block's pre-norm, fused. Numerically the
    // two-kernel sequence (the kernel matches `residual_add`'s bf16 rounding
    // before norming), which is what makes it a binding a declaration may
    // state rather than a different computation.
    kernel!(residual_add_rmsnorm "norm::residual_add_rmsnorm_bf16",
        operands = operands![
            hidden: BufMut <- Source::In(0),
            residual: Buf <- Source::In(1),
            weight: Buf <- Source::Weight(0),
            norm_out: BufMut <- Source::Out(0),
            num_rows: I32 <- Source::Rows,
            hidden_size: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // A rank-K residual stream: K parallel streams predicted from each
    // other, one of them run through the real layer, the rest corrected
    // from the difference. See `dsl::cuda`'s AltUp block for the algebra.
    //
    // Not one of these carries a contract clause, and that is a claim
    // rather than an omission: every one is row-shaped -- token `t`'s
    // output reads only token `t`'s inputs -- so a peel may split it, it
    // obligates no host plan, and there is no seam capability for it to
    // refuse.
    kernel!(altup_predict "norm::altup_predict_bf16",
        operands = operands![
            streams: Buf,
            coefs: F32s,
            predictions: BufMut,
            k: I32,
            t: I32,
            h: I32,
            stream: Stream,
        ]),
    kernel!(altup_correct "norm::altup_correct_bf16",
        operands = operands![
            predictions: Buf,
            activated: Buf,
            correction_coefs_plus_one: F32s,
            corrected: BufMut,
            k: I32,
            t: I32,
            h: I32,
            active_idx: I32,
            stream: Stream,
        ]),
    kernel!(altup_unpack_predict_coefs "norm::altup_unpack_predict_coefs",
        operands = operands![
            in_bf16: Buf,
            out_fp32: F32sMut,
            t: I32,
            k: I32,
            stream: Stream,
        ]),
    kernel!(altup_unpack_correct_coefs "norm::altup_unpack_correct_coefs",
        operands = operands![
            in_bf16: Buf,
            out_fp32: F32sMut,
            t: I32,
            k: I32,
            stream: Stream,
        ]),
    kernel!(mean_streams "norm::mean_streams_bf16",
        operands = operands![
            streams: Buf,
            out: BufMut,
            k: I32,
            t: I32,
            h: I32,
            stream: Stream,
        ]),
    kernel!(compute_rms "norm::compute_rms_bf16",
        operands = operands![
            // `reference`, not the header's `ref`: an operand NAME is the
            // row author's (`renaming_an_operand_is_not_a_mistake`), and
            // `ref` is a Rust keyword — `emit_rust_bindings` would emit it
            // verbatim into an `extern "C"` block that does not parse.
            // The C++ side is unaffected; only this table's spelling moves.
            reference: Buf <- Source::In(0),
            target_rms_out: F32sMut <- Source::Out(0),
            t: I32 <- Source::Rows,
            h: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // In place on the tensor it holds to a magnitude: the row states one
    // operand and one result and they are the same bytes, which is what
    // lets `x` bind from `Out(0)` and the width come off the value.
    kernel!(magnitude_rescale "norm::magnitude_rescale_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            target_rms: F32s <- Source::In(1),
            t: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Weightless per-head norm (the V-norm) — no gamma, so no variant.
    kernel!(rmsnorm_no_scale "norm::rmsnorm_no_scale_bf16", in_place = &[(0, 0)],
        operands = operands![
            x: Buf,
            y: BufMut,
            num_rows: I32,
            hidden: I32,
            eps: F32,
            stream: Stream,
        ]),
    // Four statements in one launch, and two: gemma-4 fuses the next
    // block's input norm into the previous block's landing, which is why
    // its layer body appears to be missing one.
    // `(landed, mlp_in)` over `(x, y)`: the stream operand is the one it
    // lands on, and the landed stream is output 0.
    kernel!(norm_residual_scale_norm "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
        in_place = &[(0, 1)],
        operands = operands![
            x: Buf,
            weight: Buf,
            hidden: BufMut,
            scale: F32,
            next_weight: Buf,
            norm_out: BufMut,
            num_rows: I32,
            hidden_size: I32,
            eps: F32,
            stream: Stream,
        ]),
    kernel!(norm_residual_add "norm::rmsnorm_residual_add_bf16", in_place = &[(0, 1)],
        operands = operands![
            x: Buf <- Source::In(0),
            weight: Buf <- Source::Weight(0),
            hidden: BufMut <- Source::Out(0),
            num_rows: I32 <- Source::Rows,
            hidden_size: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Ctx("eps"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The SCALE is the statement's, in the bits the param channel has
    // room for. It was a NAME, and the driver held the arithmetic that
    // turned four names into four numbers -- all four derived from dims
    // the host already knew. A family whose facts do not carry the
    // number states no param and falls through this branch's arity
    // guard, which is what gemma-3n and gemma-2 do.
    kernel!(scalar_mul "norm::scalar_mul_bf16", in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            s: F32 <- Source::ParamF32(0),
            n: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // Accumulates into its FIRST argument. Stating it is what lets a
    // text add into a window (`select`) and have the window keep the
    // result — see `KernelSig::in_place`.
    kernel!(residual_add_cuda "norm::residual_add_bf16", in_place = &[(0, 0)],
        operands = operands![
            y: BufMut <- Source::Out(0),
            x: Buf <- Source::In(1),
            n: Usize <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(tanh "norm::tanh_bf16", in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The head GEOMETRY off the value, not off the context: this
    // statement's result is rank-3 `[Tokens, heads, head_dim]`, so the
    // two counts are its own dims. That is the difference between a
    // fully-stated row and one that needs a context field it would then
    // share with every other family's idea of "the head count".
    kernel!(attn_sink_correction "norm::attn_sink_correction_bf16",
        in_place = &[(0, 0)],
        operands = operands![
            attn_out: BufMut <- Source::Out(0),
            lse: F32s <- Source::In(1),
            sink: F32s <- Source::Weight(0),
            n: I32 <- Source::Rows,
            num_heads: I32 <- Source::OutDim(0, 1),
            head_dim: I32 <- Source::OutDim(0, 2),
            stream: Stream <- Source::Ctx("stream"),
        ]),
];
