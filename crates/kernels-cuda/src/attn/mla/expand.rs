//! MLA prefill over expanded heads (`attn/mla/expand.cu`): the latents
//! widened through kv_b, then dense attention.

use crate::error::Error;

#[cfg(feature = "cuda")]
use crate::attn::kv;
use crate::jit::Ctx;
#[cfg(feature = "cuda")]
use crate::jit::{ArgValue, Fire, Launch, kernel, refuse};
use crate::tensor::{KvPool, Tensor};
#[cfg(feature = "cuda")]
use dtype::Dtype;

#[cfg(feature = "cuda")]
const EXPAND: &str = "attn/mla/expand.cu";

#[cfg(feature = "cuda")]
const BLOCK: u32 = 256;

/// The queries an expanded prefill reads: `q_b`'s rows whole (each head's
/// `nope` part then its rope part, as [`super::prep::split_q_b`] reads them), or the two
/// halves apart.
#[derive(Clone, Copy, Debug)]
pub enum ExpandedQ {
    Packed(Tensor),
    Split { q_nope: Tensor, q_pe: Tensor },
}

/// The shape of a latent attention's heads, as its absorbs state it.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub nope: u32,
    pub rope: u32,
    pub v: u32,
}

/// Whether [`prefill_expanded`] serves these heads on this device.
#[must_use]
pub fn expanded_takes(h: &Heads) -> bool {
    #[cfg(feature = "cuda")]
    {
        h.rope > 0
            && h.nope.is_multiple_of(8)
            && h.rope.is_multiple_of(8)
            && h.v.is_multiple_of(8)
            && (h.kv_lora_rank + h.rope).is_multiple_of(8)
            && crate::attn::prefill::planes_take(h.nope + h.rope, h.v)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = h;
        false
    }
}

/// A latent attention's whole prefill, its query absorb, its scoring over
/// every cached latent row and its output absorb, expanded as vLLM's MLA
/// prefill runs it: the fire's latent rows (each request's pages, prefix
/// and new alike) gathered into rows of their own, expanded through `kv_b`
/// into each head's key part and value, the keys packed beside the shared
/// rope part, and a causal prefill of one key head a query head at the
/// key width `nope + rope` and the value width `v` landing `o` (`[rows]
/// [heads * v]`). The same sums the absorbed form takes, in another order.
/// `Ok(false)` where it does not take the fire: the caller runs the three.
#[allow(clippy::too_many_arguments)]
pub fn prefill_expanded(
    ctx: &Ctx,
    q: ExpandedQ,
    qo_indptr: Tensor,
    pool: &KvPool,
    kv_b: Tensor,
    h: Heads,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<bool, Error> {
    #[cfg(feature = "cuda")]
    {
        use crate::attn::prefill::{Planes, prefill_planes};
        const OP: &str = "fused.mla_prefill_expanded";
        const LAT: &str = "attn.mla_expand_latents";
        const KPE: &str = "attn.mla_expand_rope";
        const IDENT: &str = "attn.mla_expand_pages";
        const LIVE: &str = "attn.mla_expand_live";
        const EXP: &str = "attn.mla_expand_rows";
        const KEYS: &str = "attn.mla_expand_keys";
        const QS: &str = "attn.mla_expand_queries";
        let width = h.heads * (h.nope + h.v);
        let dqk = h.nope + h.rope;
        let rows = o.rows;
        let page = u32::try_from(pool.page_size).unwrap_or(0);
        let pages = pool.page_indices.rows;
        let q_rows = match q {
            ExpandedQ::Packed(q) => q.rows,
            ExpandedQ::Split { q_nope, .. } => q_nope.rows,
        };
        if !expanded_takes(&h)
            || kv_b.dtype != Dtype::Bf16
            || kv_b.rows != width
            || kv_b.width != h.kv_lora_rank
            || o.dtype != Dtype::Bf16
            || o.width != h.heads * h.v
            || q_rows != rows
            || rows == 0
            || pages == 0
            || page == 0
            || !kv::native_bf16(pool)
        {
            return Ok(false);
        }
        let requests = kv::lanes_of(OP, qo_indptr)?.unsigned_abs();
        let cells = u64::from(pages) * u64::from(page);
        let cells32 = u32::try_from(cells).map_err(|_| refuse(OP, "the fire's cells overflow"))?;
        let bytes = |n: u64| usize::try_from(n * 2).unwrap_or(usize::MAX);
        let lat = ctx.scratch_stream(OP, LAT, bytes(cells * u64::from(h.kv_lora_rank)))? as u64;
        let kpe = ctx.scratch_stream(OP, KPE, bytes(cells * u64::from(h.rope)))? as u64;
        let ident = ctx.scratch_stream(OP, IDENT, pages as usize * 4)? as u64;
        let live = ctx.scratch_stream(OP, LIVE, 8)? as u64;
        let runs = cells * u64::from((h.kv_lora_rank + h.rope) / 8);
        let threads = u32::try_from(runs.max(u64::from(pages)))
            .map_err(|_| refuse(OP, "the gather overflows a launch"))?;
        let int = |v: u32| ArgValue::I32(i32::try_from(v).unwrap_or(i32::MAX));
        ctx.fire(
            OP,
            Fire::at(kernel(EXPAND, "::pie::attn::mla::gather"))
                .apply(Launch::flat(threads, BLOCK)),
            &[
                ArgValue::Ptr(pool.keys.ptr),
                ArgValue::Ptr(pool.values.ptr),
                pool.page_indices.arg(),
                int(pages),
                int(page),
                int(h.kv_lora_rank),
                int(h.rope),
                ArgValue::Ptr(lat),
                ArgValue::Ptr(kpe),
                ArgValue::Ptr(ident),
                ArgValue::Ptr(live),
                int(requests),
            ],
        )?;
        // Each cell's heads expanded: `[heads][nope + v]` a cell.
        let exp = ctx.scratch_stream(OP, EXP, bytes(cells * u64::from(width)))? as u64;
        let mut expanded = Tensor::new(exp, cells32, width, Dtype::Bf16);
        crate::linear::gemm::matmul(
            ctx,
            Tensor::new(lat, cells32, h.kv_lora_rank, Dtype::Bf16),
            kv_b,
            &mut expanded,
        )?;
        let keys = ctx.scratch_stream(OP, KEYS, bytes(cells * u64::from(h.heads * dqk)))? as u64;
        let key_runs = cells * u64::from(h.heads) * u64::from(dqk / 8);
        ctx.fire(
            OP,
            Fire::at(kernel(EXPAND, "::pie::attn::mla::pack_keys")).apply(Launch::flat(
                u32::try_from(key_runs)
                    .map_err(|_| refuse(OP, "the key pack overflows a launch"))?,
                BLOCK,
            )),
            &[
                ArgValue::Ptr(exp),
                ArgValue::Ptr(kpe),
                ArgValue::Ptr(keys),
                ArgValue::I64(i64::try_from(cells).unwrap_or(i64::MAX)),
                int(h.heads),
                int(h.nope),
                int(h.v),
                int(h.rope),
            ],
        )?;
        let (q_at, q_stride) = match q {
            ExpandedQ::Packed(q) => {
                if q.width != h.heads * dqk || q.dtype != Dtype::Bf16 {
                    return Ok(false);
                }
                (q.ptr, q.width)
            }
            ExpandedQ::Split { q_nope, q_pe } => {
                let qs =
                    ctx.scratch_stream(OP, QS, bytes(u64::from(rows) * u64::from(h.heads * dqk)))?
                        as u64;
                let q_runs = u64::from(rows) * u64::from(h.heads) * u64::from(dqk / 8);
                ctx.fire(
                    OP,
                    Fire::at(kernel(EXPAND, "::pie::attn::mla::pack_queries")).apply(Launch::flat(
                        u32::try_from(q_runs)
                            .map_err(|_| refuse(OP, "the query pack overflows a launch"))?,
                        BLOCK,
                    )),
                    &[
                        q_nope.arg(),
                        q_pe.arg(),
                        ArgValue::Ptr(qs),
                        ArgValue::I64(i64::from(rows)),
                        int(h.heads),
                        int(h.nope),
                        int(h.rope),
                    ],
                )?;
                (qs, h.heads * dqk)
            }
        };
        prefill_planes(
            ctx,
            OP,
            &Planes {
                q: q_at,
                q_stride,
                o: o.ptr,
                o_stride: o.width,
                k: keys,
                k_head: dqk,
                k_cell: h.heads * dqk,
                v: exp + u64::from(h.nope) * 2,
                v_head: h.nope + h.v,
                v_cell: width,
                cells: cells32,
                heads: h.heads,
                dqk,
                dv: h.v,
                qo_indptr: qo_indptr.ptr,
                kv_indptr: pool.page_indptr.ptr,
                kv_indices: ident,
                last_page_len: pool.last_page_lens.ptr,
                page,
                live,
                requests,
                rows,
                causal: true,
                sm_scale,
            },
        )?;
        Ok(true)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, q, qo_indptr, pool, kv_b, h, sm_scale, o);
        Ok(false)
    }
}
