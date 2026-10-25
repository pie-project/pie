use crate::attn::fa2::abi::{
    PrefillRaggedBiasParams, PrefillRaggedParams, UintFastdiv, sm_scale_or_default,
};
use crate::attn::fa2::{self, RaggedArm, RaggedPoint};
use crate::attn::kv;
use crate::attn::plan::Device;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;
use dtype::Dtype;

const RAGGED: &str = "attn/ragged.cu";

const OP: &str = "attn.ragged";

const SCRATCH: &str = "attn.ragged.schedule";

const SCHEDULE_BLOCK: u32 = 1024;

const HEAD_DIMS: [u32; 3] = [64, 128, 256];

const KV_CHUNK_SENTINEL: i32 = i32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaggedMask {
    None,
    /// Keeps group `g`'s pair `(q, k)` where bit `q * kv_len + k` from byte `indptr[g]` is set.
    Custom {
        bits: Tensor,
        indptr: Tensor,
    },
    RelativeBias {
        table: Tensor,
        max_len: u32,
    },
}

#[must_use]
const fn cta_tile_q(head_dim: u32) -> u32 {
    if head_dim >= 256 { 64 } else { 128 }
}

fn row_heads(what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

#[must_use]
pub fn padded_work_items(rows: u32, groups: u32, group_size: u32, head_dim: u32) -> u64 {
    let tile = u64::from(cta_tile_q(head_dim));
    let packed = u64::from(rows) * u64::from(group_size.max(1));
    (packed + u64::from(groups) * (tile - 1))
        .div_ceil(tile)
        .max(1)
}

#[derive(Clone, Copy, Debug)]
struct Tables {
    request_indices: u64,
    qo_tile_indices: u64,
    kv_tile_indices: u64,
    block_valid_mask: u64,
    kv_chunk_size: u64,
    bytes: u64,
}

impl Tables {
    fn lay(padded: u64) -> Self {
        const fn align16(n: u64) -> u64 {
            n.div_ceil(16) * 16
        }
        let ints = align16(padded * 4);
        let request_indices = 0;
        let qo_tile_indices = request_indices + ints;
        let kv_tile_indices = qo_tile_indices + ints;
        let block_valid_mask = kv_tile_indices + ints;
        let kv_chunk_size = block_valid_mask + align16(padded);
        Self {
            request_indices,
            qo_tile_indices,
            kv_tile_indices,
            block_valid_mask,
            kv_chunk_size,
            bytes: kv_chunk_size + 16,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ragged(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    if !HEAD_DIMS.contains(&head_dim) {
        return Err(refuse(
            OP,
            format!(
                "no ragged fa2 unit is stamped at head width {head_dim}; the arm holds 64/128/256"
            ),
        ));
    }
    let num_q_heads = row_heads("query", q.width, head_dim)?;
    let num_kv_heads = row_heads("key", k.width, head_dim)?;
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if v.width != k.width {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide value row is not the {}-wide key row",
                v.width, k.width
            ),
        ));
    }
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(k.rows, v.rows, "`{OP}` reads one value row per key row");
    let groups = kv::lanes_of(OP, q_indptr)?;
    if kv_indptr.rows != q_indptr.rows || kv_indptr.dtype != q_indptr.dtype {
        return Err(refuse(
            OP,
            format!(
                "the query table spells {} groups and the key table {}; a group is one entry \
                 of each",
                q_indptr.rows.saturating_sub(1),
                kv_indptr.rows.saturating_sub(1)
            ),
        ));
    }
    let rows = count(OP, "the query rows this attention answers", q.rows)?;
    let group_size = num_q_heads / num_kv_heads;

    let padded = padded_work_items(q.rows, groups.unsigned_abs(), group_size, head_dim);
    let padded_u32 = u32::try_from(padded)
        .ok()
        .filter(|&n| n <= i32::MAX as u32)
        .ok_or_else(|| {
            refuse(
                OP,
                format!("{padded} work items do not fit the device's i32"),
            )
        })?;
    let tables = Tables::lay(padded);
    let bytes = usize::try_from(tables.bytes).map_err(|_| {
        refuse(
            OP,
            "the schedule tables do not fit this host's address space",
        )
    })?;
    let slab = ctx.scratch(OP, SCRATCH, bytes)? as u64;

    ctx.fire(
        OP,
        Fire::at(kernel(RAGGED, "::pie::attn::ragged_schedule"))
            .apply(Launch::grid([1, 1, 1], [SCHEDULE_BLOCK, 1, 1]).smem(SCHEDULE_BLOCK * 4)),
        &[
            q_indptr.arg(),
            groups.arg(),
            stated(OP, padded_u32)?.arg(),
            stated(OP, group_size)?.arg(),
            stated(OP, cta_tile_q(head_dim))?.arg(),
            KV_CHUNK_SENTINEL.arg(),
            ArgValue::Ptr(slab + tables.request_indices),
            ArgValue::Ptr(slab + tables.qo_tile_indices),
            ArgValue::Ptr(slab + tables.kv_tile_indices),
            ArgValue::Ptr(slab + tables.block_valid_mask),
            ArgValue::Ptr(slab + tables.kv_chunk_size),
            ArgValue::ABSENT,
        ],
    )?;

    let params = PrefillRaggedParams {
        q: q.ptr,
        k: k.ptr,
        v: v.ptr,
        q_indptr: q_indptr.ptr,
        kv_indptr: kv_indptr.ptr,
        o: o.ptr,
        group_size: UintFastdiv::new(group_size),
        num_qo_heads: num_q_heads,
        num_kv_heads,
        q_stride_n: q.width,
        q_stride_h: head_dim,
        k_stride_n: k.width,
        k_stride_h: head_dim,
        v_stride_n: v.width,
        v_stride_h: head_dim,
        window_left: -1,
        logits_soft_cap: 0.0,
        sm_scale: sm_scale_or_default(sm_scale, head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        request_indices: slab + tables.request_indices,
        qo_tile_indices: slab + tables.qo_tile_indices,
        kv_tile_indices: slab + tables.kv_tile_indices,
        o_indptr: q_indptr.ptr,
        kv_chunk_size_ptr: slab + tables.kv_chunk_size,
        block_valid_mask: slab + tables.block_valid_mask,
        max_total_num_rows: rows.unsigned_abs(),
        padded_batch_size: padded_u32,
        partition_kv: false,
        ..PrefillRaggedParams::default()
    };
    let device = Device::probe(ctx).ok_or_else(|| {
        refuse(
            OP,
            "the device would not say its multiprocessor count, compute capability or shared memory",
        )
    })?;
    let point = |arm: RaggedArm| RaggedPoint {
        head_dim,
        cta_tile_q: cta_tile_q(head_dim),
        arm,
        padded_batch_size: padded_u32,
        num_kv_heads,
        device,
    };
    match mask {
        RaggedMask::None => fa2::prefill_ragged(ctx, OP, point(RaggedArm::Full), &params),
        RaggedMask::Custom { bits, indptr } => {
            if bits.dtype != Dtype::U8 || indptr.dtype != Dtype::I32 || indptr.rows != q_indptr.rows
            {
                return Err(refuse(
                    OP,
                    format!(
                        "a custom mask is u8 bits and an i32 byte offset a group; this one is \
                         {:?} bits and {} {:?} offsets for {} groups",
                        bits.dtype,
                        indptr.rows,
                        indptr.dtype,
                        q_indptr.rows.saturating_sub(1)
                    ),
                ));
            }
            fa2::prefill_ragged(
                ctx,
                OP,
                point(RaggedArm::Custom),
                &PrefillRaggedParams {
                    maybe_custom_mask: bits.ptr,
                    maybe_mask_indptr: indptr.ptr,
                    ..params
                },
            )
        }
        RaggedMask::RelativeBias { table, max_len } => {
            let span = max_len
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
                .filter(|_| max_len > 0)
                .ok_or_else(|| {
                    refuse(
                        OP,
                        format!("a relative bias over {max_len} positions has no table width"),
                    )
                })?;
            if table.dtype != Dtype::F32 || table.rows < num_q_heads || table.width != span {
                return Err(refuse(
                    OP,
                    format!(
                        "the relative bias table is {} x {} {:?}; the arm reads one f32 row of \
                         {span} (2 · {max_len} − 1) per query head, {num_q_heads} of them",
                        table.rows, table.width, table.dtype
                    ),
                ));
            }
            fa2::prefill_ragged(
                ctx,
                OP,
                point(RaggedArm::RelativeBias),
                &PrefillRaggedBiasParams {
                    base: params,
                    bias: table.ptr,
                    max_len,
                },
            )
        }
    }
}
