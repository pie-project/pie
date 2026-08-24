use crate::points::{Payload, at_bf16};
use crate::routine::{Bind, Ctx, Fire, In, Out};
use kernels::routine::Refusal;

/// The `Moe` family, claimed. ONE of seven points lands, and the six that do
/// not fail for THREE different reasons — which is the useful part of this
/// census, because only one of the three is a missing shader.
///
/// # 1. The router writes bf16 weights where the point declares f32
///
/// `moe.topk_softmax` states `weights: Out<Self::Tensor<f32>>`, and the same
/// for `topk_sigmoid` and `topk_sqrt_softplus`: a routing weight is a
/// probability and every plane's router computes it in float.
/// `moe/route.wgsl` declares `expert_weights: array<atomic<u32>>` and stores
/// through `pie_pack_bf16` — the weights come out BF16, two to a word.
///
/// A claim would therefore hand an f32 rectangle to a shader that writes
/// half as many bytes into it and leaves the top half of the plan's rectangle
/// holding whatever the arena last put there. `moe.weighted_sum` next door
/// then reads that f32 rectangle. Nothing refuses; the model is wrong.
///
/// The mismatch is UNCONDITIONAL, so no body can refuse it selectively and
/// the honest row is the family's default. Nothing fires `route_softmax`
/// any more: the launcher that did sized the rectangle bf16 out of a legacy
/// plan, and it went with the routine layer.
///
/// **SEAM (P5):** a `PIE_F32_WEIGHTS` store in `route.wgsl` — which also
/// removes the `atomic` and the compare-exchange, since an f32 weight owns
/// its whole word. `topk_sigmoid` and `topk_sqrt_softplus` need shaders as
/// well (this plane's router has a `softmax_over_all` flag and no sigmoid or
/// sqrt-softplus arm at all), so only `topk_softmax` is one edit away.
///
/// # 2. The routed matmuls are `Bank<R: Repr>`, like everything else here
///
/// `moe.matmul_select` and `moe.matmul_select_bias` declare
/// `bank: Const<Self::Tensor<T>>` — one dense expert stack. This plane has
/// `moe/qmv_routed.wgsl` and `moe/qmm_t_routed.wgsl`, and both take an
/// affine or mxfp4 bank as two or three separate weights, with the group
/// size, the bit width and the `(bm, bn)` tile pair choosing the entrypoint
/// out of a 54-way cross. Same seam as
/// `layout.embed` and the whole of `Gemm`; see `quant.rs` for the full
/// statement of it. `moe.matmul_select_bias` is already on baker-todo for
/// exactly this reason, from the cuda side.
///
/// # 3. `moe.weighted_sum` needs an operand the point does not declare
///
/// The point is `(routed, weights) -> y`. `moe/route.wgsl`'s
/// `combine_sorted` reads a FOURTH buffer, `inv`, the inverse of the
/// permutation `route_sort` built — this backend's MoE is a
/// sort/gather/grouped-matmul/scatter pipeline, so the combine has to undo
/// the sort as it folds. The permutation is not plane staging a body could
/// hide: it is produced by `route_sort` from `expert_ids` and consumed here,
/// so it is a VALUE in the dataflow that the declaration column would have
/// to carry.
///
/// **SEAM (P5/floor):** either (a) the sorted pipeline becomes tier-2 —
/// inherent methods on this plane with the text gating on `inputs.wgpu()` —
/// or (b) an unsorted `combine` shader is written whose `inv` is the identity
/// and which reads `routes` instead, matching the declaration. (b) costs a
/// shader and keeps the text plane-agnostic, which is what the design wants.
#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    /// `y = routed + sigmoid(gate) * shared`: the shared expert folded back
    /// in beside the routed sum.
    ///
    /// The one `Moe` point whose operands are three dense rectangles and one
    /// dense result, which is why it is the one that lands. The width is the
    /// routed row's and the shader is told it, because its output is an
    /// `atomic<u32>` and it derives word ownership from the absolute offset.
    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<Payload<T>>,
        shared: In<Payload<T>>,
        gate: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("moe.sigmoid_gate_add at an element other than bf16")?;
        let width = routed.width;
        self.fire(
            Fire::at("moe/route.wgsl", "shared_expert_combine")
                .apply(rows_by_width(width, routed.rows)?),
            &[routed.arg(), shared.arg(), gate.arg(), y.arg(), width.arg()],
        )
    }
}

fn rows_by_width(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs(), 1])
}
