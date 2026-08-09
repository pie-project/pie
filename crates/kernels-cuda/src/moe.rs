//! Mixture of experts: routing, the aligned permutation path, the routed
//! GEMMs and the weighted finalize.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Lit;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // `c` is an OPERAND as well as the result: the aligned staging's
    // addresses are baked into the pointer arrays
    // `build_moe_ptrs_aligned` fills, so this GEMM's destination is the
    // buffer that build named, not one the arena may pick freshly.
    kernel!(moe_grouped_gemm "moe::moe_grouped_gemm_bf16",
        in_place = &[(0, 2)],
        operands = operands![
            a: Buf <- Source::In(0),
            weight_base: Buf <- Source::Weight(0),
            c: BufMut <- Source::Out(0),
            expert_ids: I32s <- Source::In(1),
            // The two block numbers come off the param channel because
            // the operands carry only their PRODUCT — the aligned
            // rectangle's leading extent. `n` and `k` need no help: they
            // are the result's and the operand's own row widths.
            max_blocks: I32 <- Source::Param(1),
            m: I32 <- Source::Param(0),
            n: I32 <- Source::OutWidth(0),
            k: I32 <- Source::InWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // `topk_idx` here is `[N, K]` in TOKEN order, not the route-major order
    // the aligned path sorts into, so a row window keeps each token's routing
    // intact and these are not `whole`.
    kernel!(wna16_gate_up_decode "quant::wna16_gate_up_decode_bf16",
        operands = operands![
            act_fp16: Buf <- Source::In(0),
            topk_idx: I32s <- Source::In(1),
            // FOUR weights and the order is the statement's: each bank is
            // a packed half beside its scales, gate before up. The arm
            // read `args[4..8]` positionally and said so nowhere.
            gate_packed: I32Array <- Source::Weight(0),
            gate_scale: BufArray <- Source::Weight(1),
            up_packed: I32Array <- Source::Weight(2),
            up_scale: BufArray <- Source::Weight(3),
            gate_out_bf16: BufMut <- Source::Out(0),
            up_out_bf16: BufMut <- Source::Out(1),
            num_tokens: I32 <- Source::Rows,
            // `topk_idx` IS `[Tokens, top_k]`, so its row width is the
            // route count — the same reading the two weighted-sum rows
            // take, and for the same reason.
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::InWidth(0),
            intermediate: I32 <- Source::OutWidth(0),
            group_size: I32 <- Source::Ctx("wna16_group_size"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The down leg reads the ACTIVATION's width as its intermediate and
    // writes the hidden — the mirror of the gate/up row above, which is
    // why the two extents look swapped beside it.
    kernel!(wna16_down_decode "quant::wna16_down_decode_bf16",
        operands = operands![
            act_fp16: Buf <- Source::In(0),
            topk_idx: I32s <- Source::In(1),
            down_packed: I32Array <- Source::Weight(0),
            down_scale: BufArray <- Source::Weight(1),
            out_bf16: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
            intermediate: I32 <- Source::InWidth(0),
            group_size: I32 <- Source::Ctx("wna16_group_size"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(apply_per_expert_scale "moe::apply_per_expert_scale_bf16",
        operands = operands![
            topk_idx: I32s,
            topk_w: F32sMut,
            per_expert_scale_bf16: Buf,
            n: I32,
            k: I32,
            stream: Stream,
        ]),
    // `topk_idx` is route-global, so a row window would pick the wrong
    // experts' biases.
    kernel!(add_moe_route_bias "moe::add_moe_route_bias_bf16", whole = true,
        operands = operands![
            out: BufMut,
            bias: Buf,
            topk_idx: I32s,
            num_routes: I32,
            cols: I32,
            out_stride: I32,
            stream: Stream,
        ]),
    kernel!(transpose_expert_scales "moe::transpose_expert_scales_u8",
        operands = operands![
            src: Buf,
            dst: BufMut,
            num_experts: I32,
            n: I32,
            k_groups: I32,
            stream: Stream,
        ]),
    kernel!(mxfp4_moe_gate_up_decode_grouped "quant::mxfp4_moe_gate_up_decode_grouped_bf16",
        whole = true,
        operands = operands![
            act_fp16: Buf,
            sorted_route_ids: I32s,
            counts: I32s,
            gate_up_packed: U8Array,
            gate_up_scales: U8Array,
            gate_bias: BufArray,
            up_bias: BufArray,
            gate_out_bf16: BufMut,
            up_out_bf16: BufMut,
            num_experts: I32,
            top_k: I32,
            hidden: I32,
            intermediate: I32,
            stream: Stream,
        ]),
    // Namespaced in the symbol because it lives in the vendored `marlin_moe`
    // tree, the same way the `ops::` entries do.
    kernel!(mxfp4_moe_gemm_w4a16 "marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16", whole = true),
    kernel!(topk_sqrtsoftplus "moe::topk_sqrtsoftplus_bf16",
        operands = operands![
            logits: Buf,
            topk_idx: I32sMut,
            topk_w: F32sMut,
            correction_bias: F32s,
            tokens: I32,
            num_experts: I32,
            top_k: I32,
            renormalize: Bool,
            routed_scaling_factor: F32,
            stream: Stream,
        ]),
    // Expert INDICES from a table keyed by token id -- a route that is a pure
    // function of the token rather than of its activations. The WEIGHTS still
    // come from the router logits, so the logits GEMM above it does not go
    // away.
    kernel!(hash_route_lookup "moe::hash_route_lookup",
        operands = operands![
            token_ids: I32s,
            tid2eid: I64s,
            logits: Buf,
            topk_idx: I32sMut,
            topk_w: F32sMut,
            tokens: I32,
            vocab_size: I32,
            num_experts: I32,
            top_k: I32,
            renormalize: Bool,
            routed_scaling_factor: F32,
            stream: Stream,
        ]),
    kernel!(topk_sigmoid_bias "moe::topk_sigmoid_bias_fp32",
        operands = operands![
            logits: F32s <- Source::In(0),
            correction_bias: F32s <- Source::Weight(0),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            num_experts: I32 <- Source::InWidth(0),
            k: I32 <- Source::OutWidth(0),
            // The deployment's, both of them — `norm_topk_prob` and
            // `routed_scaling_factor` are config values the driver reads
            // at load, which is why they are context and not params.
            normalize: Bool <- Source::Ctx("moe_norm_topk"),
            routed_scaling_factor: F32 <- Source::Ctx("moe_routed_scaling"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The UNPADDED counterpart of `moe_align`: exact per-expert counts the
    // host reads to build cuBLAS grouped shapes. `whole` for the same reason
    // -- the sort is over all routes.
    kernel!(moe_bucket_exact "moe::moe_bucket_exact", whole = true,
        operands = operands![
            topk_idx: I32s,
            sorted_route_ids: I32sMut,
            route_to_sorted_row: I32sMut,
            counts_out: I32sMut,
            num_routes: I32,
            num_experts: I32,
            stream: Stream,
        ]),
    kernel!(token_batched_weighted_sum_aligned "moe::token_batched_weighted_sum_aligned_bf16",
        whole = true,
        operands = operands![
            out: BufMut,
            aligned_out: Buf,
            weights: F32s,
            route_to_aligned_row: I32s,
            num_tokens: I32,
            top_k: I32,
            hidden: I32,
            stream: Stream,
        ]),
    // glm5 and kimi_k3 route through a permutation rather than a loop: every
    // (token, expert) pair is a route, routes are bucketed by expert and
    // padded to fixed blocks so one batched GEMM covers all experts, and the
    // permutation is undone afterwards.
    //
    // Five of six are `whole`, for the same reason each time: the
    // permutation is computed over ALL routes in the fire, so a statement
    // addressed through `sorted_route_ids` cannot take a row window -- the
    // window would name different routes than the sort did.
    // `num_routes` is the OPERAND's element count: `topk_idx` is
    // `[Tokens, top_k]`, so the fire's tokens times `top_k` is exactly
    // what it holds. That product is what kept this row unstated — the
    // table has no arithmetic — and reading it off a value that already
    // is it costs none.
    //
    // `route_to_aligned_row` is BOUND, where the arm passed null. The
    // statement declares three results and the arena places all three;
    // the inverse map is the one this leg's combine does not read, and
    // "declared but not written" is a claim the declaration does not
    // make.
    kernel!(moe_align "moe::moe_align_decode", whole = true,
        operands = operands![
            topk_idx: I32s <- Source::In(0),
            sorted_route_ids: I32sMut <- Source::Out(0),
            expert_ids: I32sMut <- Source::Out(1),
            route_to_aligned_row: I32sMut <- Source::Out(2),
            num_routes: I32 <- Source::InElements(0),
            num_experts: I32 <- Source::Param(0),
            block_size: I32 <- Source::Param(1),
            max_blocks: I32 <- Source::Param(2),
            num_tokens_past_padded: I32sMut <- Source::Lit(Lit::Null),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // THE THREE THAT REMAIN UNSTATED, and the one thing blocking all
    // three: they take the ROUTE count and `top_k` as separate
    // arguments, and neither is reachable here. `moe_align` above got
    // its route count from `Source::InElements` because `topk_idx` is
    // `[Tokens, top_k]` and holds exactly that product — these three do
    // not take `topk_idx`, and giving them a dataflow edge to it that
    // the kernel does not read would be a lie about the trace.
    //
    // Both numbers ARE in the aligned dim's packed word. What is missing
    // is a way to say "the fire's rows times a number packed in a dim",
    // and the table deliberately has no arithmetic: an expression
    // language here is one more place a binding can be wrong, checked by
    // nothing. The next ask decides whether that is worth revisiting.
    // Every number here comes off the statement, and the one that could
    // not is why `top_k` now rides the param channel: `num_routes` is the
    // fire's tokens times k, and neither `[Tokens, hidden]` nor the
    // permutation `[max_blocks * block_size]` nor the aligned result
    // carries the router's width.
    //
    // `shared_row_begin` is `-1` at EVERY call site in the C++ tree (the
    // hybrid spells it `constexpr int shared_row_begin = -1`, and glm5,
    // kimi and deepseek_v4 pass the literal). A row states that once
    // instead of each arm restating it.
    kernel!(gather_moe_aligned_inputs "moe::gather_moe_aligned_inputs_bf16", whole = true,
        operands = operands![
            norm_x: Buf <- Source::In(0),
            sorted_route_ids: I32s <- Source::In(1),
            aligned_in: BufMut <- Source::Out(0),
            num_routes: I32 <- Source::RoutesOfParam(0),
            aligned_rows: I32 <- Source::InRows(1),
            top_k: I32 <- Source::Param(0),
            hidden: I32 <- Source::OutWidth(0),
            shared_row_begin: I32 <- Source::Lit(Lit::I32(-1)),
            num_tokens: I32 <- Source::Rows,
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(build_moe_ptrs_aligned "moe::build_moe_ptrs_aligned_bf16", whole = true,
        operands = operands![
            expert_ids: I32s,
            gate_up_base: Buf,
            down_base: Buf,
            aligned_in: Buf,
            aligned_gate_up: BufMut,
            aligned_act: BufMut,
            aligned_out: BufMut,
            a_gu_ptrs: BufArrayOut,
            b_gu_ptrs: BufArrayOut,
            c_gu_ptrs: BufArrayOutMut,
            a_dn_ptrs: BufArrayOut,
            b_dn_ptrs: BufArrayOut,
            c_dn_ptrs: BufArrayOutMut,
            max_blocks: I32,
            block_size: I32,
            h: I32,
            i_moe: I32,
            routed_blocks: I32,
            shared_gate_up_base: Buf,
            shared_down_base: Buf,
            stream: Stream,
        ]),
    // The gather's other half, read the same way. `shared_out` is the
    // FOLD's destination and this deployment does not fold the shared
    // expert here — the hand path's `constexpr bool fold_shared = false`,
    // which is the same decision `shared_row_begin = -1` states.
    kernel!(reorder_moe_aligned_output "moe::reorder_moe_aligned_output_bf16", whole = true,
        operands = operands![
            aligned_out: Buf <- Source::In(0),
            sorted_route_ids: I32s <- Source::In(1),
            route_out: BufMut <- Source::Out(0),
            num_routes: I32 <- Source::RoutesOfParam(0),
            aligned_rows: I32 <- Source::InRows(1),
            // The RESULT is `[Tokens, top_k, hidden]`, so its row width
            // is `top_k * hidden` and not this. The OPERAND is
            // `[aligned, hidden]` — one row of the aligned rectangle IS
            // the hidden width, which is why this reads off the input
            // where the gather reads off its output.
            hidden: I32 <- Source::InWidth(0),
            shared_row_begin: I32 <- Source::Lit(Lit::I32(-1)),
            num_tokens: I32 <- Source::Rows,
            shared_out: BufMut <- Source::Lit(Lit::Null),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // `out[dst_idx[i]] += src[i]·w[i]`, and `dst_idx` is route-global: a
    // window over output ROWS is not a window over routes.
    kernel!(scatter_add_weighted "moe::scatter_add_weighted_bf16", whole = true,
        operands = operands![
            out: BufMut,
            src: Buf,
            dst_idx: I32s,
            row_weights: F32s,
            num_routed: I32,
            hidden: I32,
            stream: Stream,
        ]),
    // The exception, and it is the router: a token's top-k reads only its own
    // logits row, so this one splits like any elementwise statement.
    kernel!(topk_sigmoid "moe::topk_sigmoid_bf16",
        operands = operands![
            logits: Buf,
            topk_idx: I32sMut,
            topk_w: F32sMut,
            correction_bias: F32s,
            tokens: I32,
            num_experts: I32,
            top_k: I32,
            renormalize: Bool,
            routed_scaling_factor: F32,
            stream: Stream,
        ]),
    // The router's top-k, then the decode GEMV leg's two routed
    // projections and its combine. The expert axis rides INSIDE the
    // value on this leg, so the whole branch stays a list of rectangles;
    // the grouped-GEMM and host-routed legs reach the same numbers by
    // shapes no `Dim` spells, and are named refusals, not entries.
    kernel!(topk_softmax "moe::topk_softmax_bf16",
        operands = operands![
            logits: Buf <- Source::In(0),
            topk_idx: I32sMut <- Source::Out(0),
            topk_w: F32sMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            num_experts: I32 <- Source::InWidth(0),
            k: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The whole routed block as one call — permute, both grouped GEMMs,
    // the activation and the weighted finalize. The leg decode actually
    // takes, and the only one that is a single rectangle.
    // Namespaced because it is not a `kernels::launch_*` at all: it is an
    // `ops::` entry point that installs tactics and runs a CUTLASS
    // pipeline. The symbol says so.
    kernel!(moe_fused_cutlass "moe::flashinfer_cutlass_moe_bf16",
        returns = "bool",
        operands = operands![
            activation: MoeActivation,
            input: U16s,
            token_selected_experts: I32s,
            token_final_scales: F32s,
            fc1_expert_weights: U16s,
            fc2_expert_weights: U16s,
            output: U16sMut,
            workspace: U8sMut,
            workspace_bytes: Usize,
            unpermuted_row_to_permuted_row: I32sMut,
            num_rows: I32,
            hidden_size: I32,
            inter_size: I32,
            num_experts: I32,
            experts_per_token: I32,
            tp_size: I32,
            tp_rank: I32,
            stream: Stream,
        ]),
    kernel!(moe_gate_up_gemv "moe::moe_gate_up_decode_gemv_bf16",
        operands = operands![
            topk_idx: I32s <- Source::In(0),
            norm_x: Buf <- Source::In(1),
            gate_up_base: Buf <- Source::Weight(0),
            expert_gate_up: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            // `InWidth(0)`, not `OutDim(0, 1)`. The third row in this
            // table to have sat on the generator's wall asking the PLAN
            // for a number an operand already states: `topk_idx` IS
            // `[Tokens, top_k]`. The arm read exactly that.
            top_k: I32 <- Source::InWidth(0),
            h: I32 <- Source::InWidth(1),
            // The result is `[Tokens, top_k * i_moe]`, so the
            // intermediate is what is left of a row once the routes are
            // divided out — and the routes are the other operand's width.
            i_moe: I32 <- Source::OutWidthOverIn(0, 0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The down leg: `h` is what it WRITES per route and `i_moe` what it
    // reads, which is the mirror of the gate/up row above.
    kernel!(moe_down_gemv "moe::moe_down_decode_gemv_bf16",
        operands = operands![
            topk_idx: I32s <- Source::In(0),
            expert_act: Buf <- Source::In(1),
            down_base: Buf <- Source::Weight(0),
            expert_out: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            top_k: I32 <- Source::InWidth(0),
            h: I32 <- Source::OutWidthOverIn(0, 0),
            i_moe: I32 <- Source::InWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The combine folds the residual when the MoE output lands straight
    // on the stream (tp=1) — one launch where the semantic text has a
    // WeightedSum and a ResidualAdd.
    kernel!(moe_weighted_sum "moe::token_batched_weighted_sum_bf16",
        operands = operands![
            out: BufMut <- Source::Out(0),
            src: Buf <- Source::In(0),
            weights: F32s <- Source::In(1),
            num_tokens: I32 <- Source::Rows,
            // NOT `InDim(0, 1)` and `InDim(0, 2)`, which is what these
            // said and why both rows sat on the generator's wall. A DIM
            // is the plan's and the join does not carry it — but neither
            // extent needs the plan: `weights` IS `[Tokens, top_k]`, so
            // its row width is the route count, and the result IS
            // `[Tokens, hidden]`. The arms read exactly those two widths.
            // Asking the plan for a number two operands already state is
            // an inference pass replacing a one-line answer.
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The `_add` spelling accumulates into the residual, which the
    // statement carries as its THIRD operand (`weighted_sum_add(x,
    // weights, residual)`); the plain spelling above writes a fresh
    // value and aliases nothing.
    // The route count and the hidden width are the OPERAND's own dims:
    // the reorder above it produces `[Tokens, top_k, hidden]`, so a row
    // that reads them there needs neither a config nor a context field.
    kernel!(moe_weighted_sum_add "moe::token_batched_weighted_sum_add_bf16",
        in_place = &[(0, 2)],
        operands = operands![
            out: BufMut <- Source::Out(0),
            src: Buf <- Source::In(0),
            weights: F32s <- Source::In(1),
            num_tokens: I32 <- Source::Rows,
            // NOT `InDim(0, 1)` and `InDim(0, 2)`, which is what these
            // said and why both rows sat on the generator's wall. A DIM
            // is the plan's and the join does not carry it — but neither
            // extent needs the plan: `weights` IS `[Tokens, top_k]`, so
            // its row width is the route count, and the result IS
            // `[Tokens, hidden]`. The arms read exactly those two widths.
            // Asking the plan for a number two operands already state is
            // an inference pass replacing a one-line answer.
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The routed MXFP4 GEMVs. Like qwen3_5's GEMV leg the expert axis
    // rides INSIDE the value, so each is one rectangle over `N * k`
    // routes; unlike it, the weight slot names a per-expert POINTER
    // BANK, which is a binding question and not a shape one.
    kernel!(mxfp4_moe_gate_up "quant::mxfp4_moe_gate_up_decode_bf16",
        operands = operands![
            act_fp16: Buf,
            topk_idx: I32s,
            gate_up_packed: U8Array,
            gate_up_scales: U8Array,
            gate_bias: BufArray,
            up_bias: BufArray,
            gate_out_bf16: BufMut,
            up_out_bf16: BufMut,
            num_tokens: I32,
            top_k: I32,
            hidden: I32,
            intermediate: I32,
            stream: Stream,
            act_out_fp16: BufMut,
            glu_limit: F32,
            glu_alpha: F32,
        ]),
    kernel!(mxfp4_moe_down "quant::mxfp4_moe_down_decode_bf16",
        operands = operands![
            act_fp16: Buf,
            topk_idx: I32s,
            down_packed: U8Array,
            down_scales: U8Array,
            down_bias: BufArray,
            out_bf16: BufMut,
            num_tokens: I32,
            top_k: I32,
            hidden: I32,
            intermediate: I32,
            stream: Stream,
        ]),
];
