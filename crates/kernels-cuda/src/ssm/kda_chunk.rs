//! KDA in sixteen-token chunks on the tensor cores (`ssm/kda_chunk_sm80.cu`).

use super::*;
use crate::ssm::kda::Kda;
use crate::ssm::kda::KdaGates;

const KDA_CHUNK_SM80: &str = "ssm/kda_chunk_sm80.cu";

/// [`kda_chunked`] on the tensor cores in sixteen-token chunks: two warps a
/// (chunk, head) land the chunk's decayed queries and fold keys, its decay,
/// its update directions `w` and `u` and its decayed causal scores; then a
/// block a (request, head, 64 value columns) walks the chunks with the state
/// in f32. 128-wide heads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kda_chunked_mma(
    ctx: &Ctx,
    op: &'static str,
    shape: Kda,
    gates: KdaGates,
    mixed: RaggedTensor,
    norm_eps: f32,
    state: &RecurrentPool,
    lanes: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const CH: u32 = 16;
    const D: u32 = 128;
    const TASKS: u32 = 2;
    // `KDA16_TASK_SMEM`, and the scan's three `Kda16Scan<2>::STAGE`s and
    // their barriers.
    const LD: u32 = 136;
    const LDT: u32 = 24;
    const LDV: u32 = 136;
    const TASK_SMEM: u32 = 3 * CH * LD * 2 + CH * (CH + 1) * FLOAT + 3 * D * FLOAT;
    const STAGE: u32 = 3 * CH * LD * 2 + CH * LDV * 2 + 2 * CH * LDT * 2 + D * FLOAT;
    let heads = shape.heads;
    let slots = shape.n.div_ceil(CH) + lanes;
    let rows = u64::from(shape.n + CH) * u64::from(heads) * u64::from(D);
    let qg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_qg", rows)?;
    let kg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_kg", rows)?;
    let ke = plane_bf16(ctx, op, "attn.ssm_kda_chunk_ke", rows)?;
    let squares = u64::from(slots) * u64::from(heads) * u64::from(CH * CH);
    let tb = plane_bf16(ctx, op, "attn.ssm_kda_chunk_t", squares)?;
    let pm = plane_bf16(ctx, op, "attn.ssm_kda_chunk_p", squares)?;
    let gl = plane(
        ctx,
        op,
        "attn.ssm_kda_chunk_gl",
        u64::from(slots) * u64::from(heads) * u64::from(D),
    )?;
    ctx.fire(
        op,
        Fire::at(KDA_CHUNK_SM80, "::pie::ssm::kda_chunk_prep").apply(
            Launch::grid([heads.div_ceil(TASKS), slots, 1], [TASKS * 2 * WARP, 1, 1])
                .smem(TASKS * TASK_SMEM),
        ),
        &[
            mixed.data.arg(),
            norm_eps.arg(),
            gates.f.arg(),
            gates.b.arg(),
            gates.a_log.arg(),
            gates.dt_bias.arg(),
            gates.floor.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(ke),
            ArgValue::Ptr(tb),
            ArgValue::Ptr(pm),
            ArgValue::Ptr(gl),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        op,
        Fire::at(KDA_CHUNK_SM80, "::pie::ssm::kda_chunk_scan")
            .apply(Launch::grid([lanes * heads * 2, 1, 1], [128, 1, 1]).smem(3 * STAGE + 64)),
        &[
            ArgValue::Ptr(gl),
            mixed.data.arg(),
            ArgValue::Ptr(qg),
            ArgValue::Ptr(kg),
            ArgValue::Ptr(ke),
            ArgValue::Ptr(tb),
            ArgValue::Ptr(pm),
            state.slab.arg(),
            state.slot_ids.arg(),
            mixed.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(op, lanes)?.arg(),
            stated(op, heads)?.arg(),
            stated(op, ctx.multiprocessors().unwrap_or(1).max(1))?.arg(),
            ctx.stage(),
        ],
    )
}
