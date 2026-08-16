//! THE MIXTURE PATHS — four of them, and they are four because the
//! QUANTIZATION is what differs: kimi's WNA16, gpt-oss's MXFP4,
//! nemotron_h's own dispatch, and the aligned path. They sat 1,500
//! lines apart in the single file, which is what made them read as
//! unrelated features rather than as one choice made four ways.

use super::*;

// ── kimi: the WNA16 quantized MoE path ─────────────────────────
//
// 4-bit weights with a bf16 scale per group of `group_size` along K.
// Distinct from MXFP4 (whose scale is an E8M0 exponent byte per 32) and
// from fp8 -- three quantizations, three statements, because which one a
// checkpoint ships is a fact the declaration reads.

/// `kernels::quant::dequant_wna16_int4b8_to_bf16`: widen a packed
/// int4-b8 weight to bf16.
///
/// Weight-shaped: `[out_dim, in_dim/8]` packed to `[out_dim, in_dim]`,
/// no token extent.
pub fn dequant_wna16_int4b8(t: &Trace, l: u32, w: &str, out_dim: u32, in_dim: u32) -> Val {
    record(
        t,
        Some(l),
        "quant::dequant_wna16_int4b8_to_bf16",
        vec![w.to_string()],
        None,
        vec![],
        Some((
            Shape(vec![Dim::Const(out_dim), Dim::Const(in_dim)]),
            DType::BF16,
        )),
    )
    .expect("the dequant produces its value")
}

/// `kernels::quant::wna16_gate_up_decode_bf16`: the gate and up
/// projections, decode-shaped, straight off the packed weights.
///
/// `topk_idx` here is `[N, K]` in TOKEN order -- not the route-major
/// order the aligned path sorts into -- so a row window keeps each
/// token's routing intact and this is not `whole`.
/// `bank` names the layer's expert weights; the statement records the
/// FOUR per-expert tables the launcher actually reads
/// (`<bank>.gate_packed` / `.gate_scale` / `.up_packed` /
/// `.up_scale`). They were unnamed once, and a driver whose executor
/// is model-agnostic could not reach them at all: with no name in the
/// trace there is nothing to resolve, and the only way in was a
/// family's private layer struct — the convention this whole
/// direction exists to remove.
pub fn wna16_gate_up_decode(
    act: &Val,
    topk_idx: &Val,
    intermediate: u32,
    bank: &str,
) -> (Val, Val) {
    let outs = record_many(
        &act.t,
        act.layer,
        "quant::wna16_gate_up_decode_bf16",
        vec![
            format!("{bank}.gate_packed"),
            format!("{bank}.gate_scale"),
            format!("{bank}.up_packed"),
            format!("{bank}.up_scale"),
        ],
        vec![act.id, topk_idx.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            ),
            (
                Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
                DType::BF16,
            ),
        ],
    );
    let mut it = outs.into_iter();
    let gate = it.next().expect("the projection states two outputs");
    let up = it.next().expect("the projection states two outputs");
    (gate, up)
}

/// `kernels::quant::wna16_down_decode_bf16`: the down projection, same
/// shape.
pub fn wna16_down_decode(act: &Val, topk_idx: &Val, hidden: u32, bank: &str) -> Val {
    record(
        &act.t,
        act.layer,
        "quant::wna16_down_decode_bf16",
        vec![format!("{bank}.down_packed"), format!("{bank}.down_scale")],
        None,
        vec![act.id, topk_idx.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the projection produces its value")
}

/// `kernels::norm::rmsnorm_strided_bf16`: the norm, reading and writing
/// a prefix of wider rows.
///
/// How a fused projection's halves get normed in place without a copy:
/// the stride says where the row really ends.
pub fn rmsnorm_strided(x: &Val, weight: &str, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::rmsnorm_strided_bf16",
        vec![weight.to_string()],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the norm produces its value")
}


// ── mixtral / gpt-oss: the MXFP4 MoE path ──────────────────────
//
// gpt-oss ships its experts as MXFP4 -- 4-bit values with an E8M0
// exponent byte per block of 32 -- and mixtral's shell runs them through
// Marlin. Several statements here operate on WEIGHTS rather than
// activations (repacking a scale layout, splitting a fused bias) and have
// no token extent at all. They are stated because they are launches the
// fire performs, and a reader tracing where an operand came from should
// find them on the tape.

/// `kernels::moe::add_moe_route_bias_bf16`: add each route's EXPERT
/// bias, indexed by that route's expert.
///
/// `whole`: `topk_idx` is route-global, so a row window would pick the
/// wrong experts' biases.
pub fn add_moe_route_bias(x: &Val, topk_idx: &Val, bias: &str, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "moe::add_moe_route_bias_bf16",
        vec![bias.to_string()],
        None,
        vec![x.id, topk_idx.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the bias add produces its value")
}

// `build_window_page_view` and `build_full_split_view` WERE HERE and both
// wrappers are DELETED, with their `table::attn` rows and the two
// launchers in `kernels-cuda/csrc/src/attn/kv_paged.cu`. Nothing in
// `crates/model/src` named either wrapper or either symbol; both rows had
// `Source::Unbound` on every operand, so no dispatch was ever generated
// from them. The kernels are untouched and still have device rows in
// `kernels-cuda::families::attn`; `driver-cuda/src/fire/kv_paged.rs`
// is the host program now.

// `dsl::cuda::gemv3` WAS HERE, stating `gemm::gemv3_bf16`.
//
// The row it named is gone -- `kernels-cuda/src/table/gemm.rs` carries
// the tombstone and the reason -- and the reason was that its whole
// consumer set was this wrapper, which nothing called. Deleting one
// half left the other stating a symbol no table declares, which is
// not a harmless leftover: `check_plan` refuses an undeclared symbol
// at LOAD, so a text that reached for this would fail late and for a
// reason with no bearing on what it was trying to do.
//
// Re-adding is a row and a wrapper, together. Either alone is a trap.

/// `kernels::norm::rmsnorm_bf16_with_fp16`: the norm, published in both
/// bf16 and fp16.
///
/// The fp16 copy is what the MXFP4 grouped GEMM below consumes; producing
/// it here rather than casting afterwards is the binding, so the
/// declaration states it.
pub fn rmsnorm_with_fp16(x: &Val, weight: &str, hidden: u32) -> (Val, Val) {
    let outs = record_many(
        &x.t,
        x.layer,
        "norm::rmsnorm_bf16_with_fp16",
        vec![weight.to_string()],
        vec![x.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16),
            (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::F16),
        ],
    );
    let mut it = outs.into_iter();
    let bf16 = it.next().expect("the norm states two outputs");
    let fp16 = it.next().expect("the norm states two outputs");
    (bf16, fp16)
}

/// `kernels::rope::rope_write_kv_bf16`: rope q and k, then commit k/v to
/// the pages, in one launch.
pub fn rope_write_kv(q: &Val, k: &Val, v: &Val, l: u32, q_width: u32) -> Val {
    record(
        &q.t,
        Some(l),
        "rope::rope_write_kv_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![q.id, k.id, v.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the fused rope+write produces its value")
}

/// `kernels::quant::mxfp4_scales_to_marlin_e8m0`: repack the checkpoint's
/// E8M0 scale layout into the one Marlin walks.
pub fn mxfp4_scales_to_marlin(t: &Trace, l: u32, w: &str, groups: u32, rows: u32) -> Val {
    record(
        t,
        Some(l),
        "quant::mxfp4_scales_to_marlin_e8m0",
        vec![w.to_string()],
        None,
        vec![],
        Some((
            Shape(vec![Dim::Const(groups), Dim::Const(rows)]),
            DType::I32,
        )),
    )
    .expect("the repack produces its value")
}

/// `kernels::moe::transpose_expert_scales_u8`: the per-expert group
/// scales, `[E, n, k/32]` -> `[E, k/32, n]`.
pub fn transpose_expert_scales(
    t: &Trace,
    l: u32,
    w: &str,
    experts: u32,
    k_groups: u32,
    n: u32,
) -> Val {
    record(
        t,
        Some(l),
        "moe::transpose_expert_scales_u8",
        vec![w.to_string()],
        None,
        vec![],
        Some((
            Shape(vec![
                Dim::Const(experts),
                Dim::Const(k_groups),
                Dim::Const(n),
            ]),
            DType::I32,
        )),
    )
    .expect("the transpose produces its value")
}

// `pub fn mxfp4_moe_gemm_w4a16` WAS HERE, recording the symbol
// `marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16`. Both halves of it are
// gone, and they had to go together.
//
// Its doc said the name was namespaced "because it lives in the vendored
// `marlin_moe` tree, the same way `ops::` entries do". §47 deleted that
// tree — both `csrc/third_party/marlin` and `csrc/third_party/marlin_moe`,
// 656 KB, with their CMake `option()`s, their `target_sources`/
// `target_include_directories`/`target_compile_definitions`, the
// `kernels.def`/`marlin.cu` shape reconciliation and the
// `PIE_CUDA_HAS_MARLIN_MOE` capability in `kernels_manifest.hpp`. So the
// C++ function this named has not existed for some time.
//
// The row went for its own reason: its `KernelSig::operands` was EMPTY,
// so no fire could ever have bound it, and its whole consumer set was
// this wrapper, which nothing called. A grep for `mxfp4_moe_gemm_w4a16`
// over every `.rs` in the workspace now returns this comment and nothing
// else.
//
// Deleting one half alone is what makes this worth a paragraph. A
// builder that records a symbol no table row declares is not a compile
// error — [`model_ir::kernels::check_plan`] refuses an undeclared symbol at
// LOAD, so a text that reached for this would fail late and for a reason
// with no bearing on what it was trying to do. Re-adding is a row, a
// wrapper and a device text, together. Any one alone is a trap.


// ── nemotron_h: its own MoE dispatch ───────────────────────────

/// `kernels::moe::topk_sigmoid_bias_fp32`: the router, over fp32
/// logits and with a per-expert correction bias.
pub fn topk_sigmoid_bias(logits: &Val, bias: &str, top_k: u32) -> (Val, Val) {
    let outs = record_many(
        &logits.t,
        logits.layer,
        "moe::topk_sigmoid_bias_fp32",
        vec![bias.to_string()],
        vec![logits.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let idx = it.next().expect("the router states two outputs");
    let w = it.next().expect("the router states two outputs");
    (idx, w)
}

/// `kernels::moe::moe_bucket_exact`: bucket routes by expert WITHOUT
/// padding to fixed blocks.
///
/// The unpadded counterpart of [`moe_align`](crate::cuda::moe_align), writing exact
/// per-expert counts the host reads to build cuBLAS grouped shapes.
/// `whole` for the same reason: the sort is over all routes.
///
/// Returns `(sorted_route_ids, route_to_sorted_row, counts)` — the
/// permutation, its inverse, and the per-expert totals.
///
/// # The third result is new, and its absence was a wrong-answer bug
/// waiting on a caller
///
/// **This declared two results and the kernel writes three buffers.**
/// `moe_dispatch.cuh:907`'s parameter list is `topk_idx,
/// sorted_route_ids, route_to_sorted_row, counts_out, num_routes,
/// num_experts`, and `route_to_sorted_row` was named nowhere in this
/// crate — not here, not in the row that used to carry the symbol.
///
/// A binding written from a two-result statement has one buffer with no
/// declared home, so it passes a null, and the store at `:952` is
/// `route_to_sorted_row[r] = out;` with **no null guard**. That is the
/// difference from [`moe_align`](crate::cuda::moe_align), whose otherwise identical
/// inverse map is written behind `if (route_to_aligned_row != nullptr)`
/// and whose third result is therefore genuinely optional. Here it is
/// not optional: the declaration is what makes the buffer exist, and
/// without it the fire does not answer wrongly, it writes to null.
///
/// **The order is the KERNEL's and not the convenient one.** `counts` is
/// the interesting output and was the second of two; it is the third of
/// three now, because a declaration list whose order matches the
/// parameter list is one a binding can read straight down. The cost of
/// the other choice is on record in this family: `moe::hash_route_lookup`
/// deleted a row that stated no `Source` on any operand, and the only
/// surviving statement of which input was which was `dsl.rs`'s own
/// argument vector.
pub fn moe_bucket_exact(topk_idx: &Val, num_experts: u32, top_k: u32) -> (Val, Val, Val) {
    let routes = Shape(vec![Dim::Tokens, Dim::Const(top_k)]);
    let outs = record_many(
        &topk_idx.t,
        topk_idx.layer,
        "moe::moe_bucket_exact",
        vec![],
        vec![topk_idx.id],
        vec![
            (routes.clone(), DType::I32),
            (routes, DType::I32),
            (Shape(vec![Dim::Const(num_experts)]), DType::I32),
        ],
    );
    let mut it = outs.into_iter();
    let sorted = it.next().expect("the bucket states three outputs");
    let inverse = it.next().expect("the bucket states three outputs");
    let counts = it.next().expect("the bucket states three outputs");
    (sorted, inverse, counts)
}

/// `kernels::ssm::build_nemotron_moe_ptrs_aligned_bf16`: the pointer
/// arrays for the block-aligned batched GEMM.
pub fn build_nemotron_moe_ptrs_aligned(expert_ids: &Val, aligned_in: &Val, l: u32) {
    record(
        &expert_ids.t,
        Some(l),
        "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        vec![],
        None,
        vec![expert_ids.id, aligned_in.id],
        None,
    );
}

/// `kernels::ssm::build_nemotron_moe_ptrs_decode_batched_bf16`: the
/// same, for the decode path that skips the permutation entirely.
pub fn build_nemotron_moe_ptrs_decode(topk_idx: &Val, topk_w: &Val, x: &Val, l: u32) {
    record(
        &topk_idx.t,
        Some(l),
        "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        vec![],
        None,
        vec![topk_idx.id, topk_w.id, x.id],
        None,
    );
}


// ── MoE: the ALIGNED dispatch path ─────────────────────────────
//
// glm5 and kimi_k3 route through a permutation, not a loop. Every
// (token, expert) pair is a ROUTE; the routes are bucketed by expert and
// padded to fixed-size blocks so one batched GEMM covers every expert at
// once, and the permutation is undone afterwards.
//
// Five of the six are `whole`, and it is the same reason each time: the
// permutation is computed over ALL routes in the fire. `sorted_route_ids`
// is a global order, so a statement addressed through it cannot be handed
// a row window -- the window would name different routes than the sort
// did. This is the `dyn` axis the trace module doc describes, at the one
// point where it stops being expressible as a row range.

/// `kernels::moe::topk_sigmoid_bf16`: the router — each token's top-k
/// experts and their weights, gated by sigmoid rather than softmax.
///
/// Returns `(topk_idx, topk_w)`. The ONE statement here that is not
/// `whole`: a token's routing reads only its own logits row.
pub fn topk_sigmoid(logits: &Val, top_k: u32) -> (Val, Val) {
    let outs = record_many(
        &logits.t,
        logits.layer,
        "moe::topk_sigmoid_bf16",
        vec![],
        vec![logits.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::I32),
            (Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let idx = it.next().expect("the router states two outputs");
    let w = it.next().expect("the router states two outputs");
    (idx, w)
}

/// `kernels::moe::moe_align_decode`: bucket the routes by expert and
/// pad each bucket to a block.
///
/// Returns `(sorted_route_ids, expert_ids, route_to_aligned_row)` — the
/// permutation, which expert each block belongs to, and the inverse map
/// the combine reads.
/// The three load-time numbers ride the param channel. They are the
/// permutation's own shape — how many experts to bucket into, how
/// wide a block is, how many blocks the padding admits — and the
/// executor was reading two of them out of a config struct and one
/// out of its MoE workspace.
pub fn moe_align(
    topk_idx: &Val,
    max_blocks: u32,
    block_size: u32,
    top_k: u32,
    num_experts: u32,
) -> (Val, Val, Val) {
    let routes = Dim::Const(top_k);
    let outs = record_many_with_params(
        &topk_idx.t,
        topk_idx.layer,
        "moe::moe_align_decode",
        vec![],
        vec![num_experts, block_size, max_blocks],
        vec![topk_idx.id],
        vec![
            (Shape(vec![Dim::Const(max_blocks * block_size)]), DType::I32),
            (Shape(vec![Dim::Const(max_blocks)]), DType::I32),
            (Shape(vec![Dim::Tokens, routes]), DType::I32),
        ],
    );
    let mut it = outs.into_iter();
    let sorted = it.next().expect("the align states three outputs");
    let experts = it.next().expect("the align states three outputs");
    let inverse = it.next().expect("the align states three outputs");
    (sorted, experts, inverse)
}

/// `kernels::moe::gather_moe_aligned_inputs_bf16`: the block-major
/// operand, gathered in the sorted order.
/// `top_k` rides the PARAM channel because the kernel wants
/// `num_routes` — the fire's tokens times k — and nothing else in the
/// statement says k. `x` is `[Tokens, hidden]`, `sorted_route_ids` is
/// `[max_blocks * block_size]`, and the result is the aligned
/// rectangle; none of the three carries the router's width. Same
/// reason [`moe_align`] carries its three load-time numbers there,
/// and stating it is what lets the row generate instead of needing a
/// hand-written arm.
pub fn gather_moe_aligned_inputs(
    x: &Val,
    sorted_route_ids: &Val,
    aligned: Dim,
    hidden: u32,
    top_k: u32,
) -> Val {
    record_with_params(
        &x.t,
        x.layer,
        "moe::gather_moe_aligned_inputs_bf16",
        vec![],
        None,
        vec![top_k],
        vec![x.id, sorted_route_ids.id],
        Some((Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the gather produces its value")
}

/// `kernels::moe::build_moe_ptrs_aligned_bf16`: the aligned leg's
/// staging, and the pointer arrays one batched GEMM per projection
/// needs into it.
///
/// It DECLARES the staging, which is the only SSA-valid way to say
/// what this call does. The kernel bakes the three staging buffers'
/// BASE ADDRESSES into device pointer arrays, so it has to know
/// where they are before anything writes them — and a statement
/// cannot take an operand that a later statement produces. So the
/// build is what fixes where the aligned staging lives, and the two
/// grouped GEMMs and the swiglu between them fill buffers it named:
/// each takes its destination as an operand and writes it in place.
/// Before this, all three were `mw.aligned_*` in the executor and
/// the declaration ended at "a pointer build happens here".
///
/// `(gate_up, act, out)` — `[aligned, 2·I]`, `[aligned, I]`,
/// `[aligned, H]`, all bf16, all block-major.
///
/// The six POINTER ARRAYS are still the driver's: an array of device
/// addresses has no dtype in this vocabulary, and inventing one to
/// hold `void*` is a wider change than this statement needs. They
/// are reachable only from the two GEMMs that this call also serves,
/// so the gap is bounded — see the executor's fallback arm.
pub fn build_moe_ptrs_aligned(
    expert_ids: &Val,
    aligned_in: &Val,
    l: u32,
    gate_up_bank: &str,
    down_bank: &str,
    aligned: Dim,
    hidden: u32,
    moe_intermediate: u32,
) -> (Val, Val, Val) {
    let outs = record_many(
        &expert_ids.t,
        Some(l),
        "moe::build_moe_ptrs_aligned_bf16",
        vec![gate_up_bank.to_string(), down_bank.to_string()],
        vec![expert_ids.id, aligned_in.id],
        vec![
            (
                Shape(vec![aligned, Dim::Const(2 * moe_intermediate)]),
                DType::BF16,
            ),
            (
                Shape(vec![aligned, Dim::Const(moe_intermediate)]),
                DType::BF16,
            ),
            (Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let gate_up = it.next().expect("the ptr build states three stages");
    let act = it.next().expect("the ptr build states three stages");
    let out = it.next().expect("the ptr build states three stages");
    (gate_up, act, out)
}

/// `kernels::moe::reorder_moe_aligned_output_bf16`: undo the block
/// permutation, back to route order.
pub fn reorder_moe_aligned_output(
    aligned_out: &Val,
    sorted_route_ids: &Val,
    top_k: u32,
    hidden: u32,
) -> Val {
    record_with_params(
        &aligned_out.t,
        aligned_out.layer,
        "moe::reorder_moe_aligned_output_bf16",
        vec![],
        None,
        // `top_k` on the param channel, for the gather's reason: the
        // kernel wants `num_routes` and no operand of this statement
        // carries the router's width. The result's SECOND dim is
        // `top_k` as well, so this one could be read off `OutDim(0,
        // 1)` — it is stated the same way as the gather's so the two
        // halves of one permutation read alike.
        vec![top_k],
        vec![aligned_out.id, sorted_route_ids.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the reorder produces its value")
}
