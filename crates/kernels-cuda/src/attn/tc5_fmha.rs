//! The tcgen05 prefill over a paged bf16 KV cache (`attn/tc5_fmha.cuh`), for
//! the shapes it serves: a causal or full prefill, sliding or not, without a
//! mask, a relative bias or a log-sum-exp, at a 64-, 128-, 256- or 512-wide
//! head, an NHD pool of 16-token pages, and rows long enough that a 128-row
//! tile of packed query heads is mostly full. Everything else stays on FA2.

use cudarc::driver::sys as dr;

use crate::attn::fa2_abi::sm_scale_or_default;
use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, symbol};
use crate::linear::tc5::{Map, encode, supported};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/tc5_fmha.cuh";

/// The pool's NHD layout code.
const NHD: i32 = 0;

/// `pie::attn::fmha::Params`, field for field.
#[repr(C)]
struct Params {
    q: u64,
    o: u64,
    qo_indptr: u64,
    kv_indptr: u64,
    kv_indices: u64,
    last_page_len: u64,
    live: u64,
    q_stride: i32,
    o_stride: i32,
    kv_heads: i32,
    group: i32,
    page: i32,
    scale_log2: f32,
    causal: i32,
    window_left: i32,
}

/// The key tile and ring depth a head width runs at: three stages of K and
/// V fill shared memory, Q and P living in TMEM. A prefill `wide_of` names
/// runs `tc5_fmha_prefill_wide` instead.
fn tiling(head_dim: u32, window_left: i32) -> (u32, u32) {
    if let Some(w) = wide_of(head_dim, window_left) {
        return (w.bkv, w.slots);
    }
    match head_dim {
        256 => (64, 3),
        _ => (128, 3),
    }
}

/// `pie::attn::fmha::smem_bytes`.
const fn smem_bytes(d: u32, bkv: u32, stages: u32) -> u32 {
    2 * stages * (d / 64) * bkv * 128 + 512 + 1024
}

/// `tc5_fmha_prefill_wide`'s shape: 128-key tiles; Q whole in shared
/// memory (`qt` of its columns in TMEM: none); K and V through a ring of
/// three slices of two 64-wide atoms.
#[derive(Clone, Copy)]
struct Wide {
    d: u32,
    bkv: u32,
    qt: u32,
    atoms: u32,
    slots: u32,
}

/// The prefills `tc5_fmha_prefill_wide` serves: every 512-wide head (O
/// would fill TMEM in the other kernel), and a 256-wide one without a
/// window, whose 128-key tiles run S at twice the rate of 64. A sliding
/// 256-wide head (Gemma-4's) stays on the other kernel: its blocks read a
/// window of a few tiles, and a 128-key tile's partial ends cost it more
/// than the rate gains.
fn wide_of(head_dim: u32, window_left: i32) -> Option<Wide> {
    match head_dim {
        512 => Some(Wide {
            d: 512,
            bkv: 128,
            qt: 0,
            atoms: 2,
            slots: 3,
        }),
        256 if window_left < 0 => Some(Wide {
            d: 256,
            bkv: 128,
            qt: 0,
            atoms: 2,
            slots: 5,
        }),
        _ => None,
    }
}

/// `pie::attn::fmha::wide::smem_bytes`.
const fn wide_smem_bytes(w: &Wide) -> u32 {
    (w.d - w.qt) / 64 * 128 * 128 + w.slots * w.atoms * w.bkv * 128 + 512 + 1024
}

/// Whether this prefill runs on the tcgen05 kernel, and if so runs it.
/// `Ok(false)` leaves it to FA2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_prefill(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window_left: i32,
    head_dim: u32,
    sm_scale: f32,
    o: &Tensor,
) -> Result<bool, Error> {
    let shape = &plan.shape;
    let group = shape.group_size();
    let rows = q.data.rows;
    let requests = shape.num_requests;
    let page = u32::try_from(pool.page_size).unwrap_or(0);
    let (bkv, stages) = tiling(head_dim, window_left);
    // The profile's decision (`kernel.sdpa.prefill.tc5`, on for sm_100)
    // and the device's capability both: elsewhere FA2 serves as before.
    let fits = ctx.tuning().prefill_tc5
        && supported()
        && matches!(head_dim, 64 | 128 | 256 | 512)
        && requests > 0
        && kv::native_bf16(pool)
        && pool.layout == NHD
        && page >= 16
        && page.is_multiple_of(16)
        && bkv.is_multiple_of(page)
        && pool.head_stride == i64::from(head_dim)
        && [q.data.ptr, o.ptr, pool.keys.ptr, pool.values.ptr]
            .iter()
            .all(|&at| aligned16(at))
        // A tile is 128 packed query rows: short rows (a decode's) would
        // leave most of it empty and their keys unsplit.
        && u64::from(rows) * u64::from(group) >= 64 * u64::from(requests);
    let Some(live_at) = plan.info.live_lanes_offset else {
        return Ok(false);
    };
    if !fits {
        return Ok(false);
    }
    let kv_heads = shape.num_kv_heads;
    let cells = pool.keys.rows;
    // A box a 16-token page slice, and one a whole tile for a tile whose
    // pages lie in a run.
    let map = |ptr: u64, rows: u32| {
        encode(Map {
            ptr,
            ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            dims: [head_dim, kv_heads, cells],
            strides: [
                u64::from(head_dim) * 2,
                u64::try_from(pool.seq_stride).unwrap_or(0) * 2,
            ],
            boxes: [64, 1, rows],
            swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        })
    };
    let map_k = map(pool.keys.ptr, 16)?;
    let map_v = map(pool.values.ptr, 16)?;
    let map_k_run = map(pool.keys.ptr, bkv)?;
    let map_v_run = map(pool.values.ptr, bkv)?;
    let wide = wide_of(head_dim, window_left);
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let params = Params {
        q: q.data.ptr,
        o: o.ptr,
        qo_indptr: q.indptr.ptr,
        kv_indptr: pool.page_indptr.ptr,
        kv_indices: pool.page_indices.ptr,
        last_page_len: pool.last_page_lens.ptr,
        live: plan.workspace.int_ptr + u64::from(live_at),
        q_stride: int(q.data.width),
        o_stride: int(o.width),
        kv_heads: int(kv_heads),
        group: int(group),
        page: int(page),
        scale_log2: sm_scale_or_default(sm_scale, head_dim) * core::f32::consts::LOG2_E,
        causal: i32::from(plan.causal),
        // The op's window, as FA2 reads it: a plan may serve layers of
        // several windows (Gemma's sliding and global layers share one).
        window_left,
    };
    // Every request's tiles, rounded up a tile each.
    let bound = (rows * group).div_ceil(128) + requests;
    let bytes = |m: &[u64; 16]| ArgValue::Bytes {
        ptr: m.as_ptr().cast(),
        len: 128,
    };
    let tail = [
        bytes(&map_k),
        bytes(&map_v),
        bytes(&map_k_run),
        bytes(&map_v_run),
        ArgValue::Bytes {
            ptr: (&raw const params).cast(),
            len: core::mem::size_of::<Params>(),
        },
        ArgValue::I32(int(bound)),
    ];
    let Some(w) = wide else {
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::attn::fmha::tc5_fmha_prefill<{head_dim}, {bkv}, {stages}>"
                )),
            )
            .apply(
                Launch::grid([bound, kv_heads, 1], [192, 1, 1])
                    .smem(smem_bytes(head_dim, bkv, stages)),
            ),
            &tail,
        )?;
        return Ok(true);
    };
    // A 512-wide head's Q: a box an atom of a block's 128 packed rows, whole
    // tokens' `group` heads; when they are not whole tokens the kernel
    // stores Q itself and the map is never read.
    let whole_tokens = 128 % group == 0;
    let map_q = encode(Map {
        ptr: q.data.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [head_dim, shape.num_q_heads, rows.max(1)],
        strides: [u64::from(head_dim) * 2, u64::from(q.data.width) * 2],
        boxes: if whole_tokens {
            [64, group, 128 / group]
        } else {
            [64, 1, 1]
        },
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })?;
    let mut args = vec![bytes(&map_q)];
    args.extend(tail);
    // Two blocks a key head, each landing half of O, side by side in y.
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::fmha::tc5_fmha_prefill_wide<{}, {}, {}, {}, {}>",
                w.d, w.bkv, w.qt, w.atoms, w.slots
            )),
        )
        .apply(
            Launch::grid([bound, kv_heads * (w.d / 256), 1], [192, 1, 1]).smem(wide_smem_bytes(&w)),
        ),
        &args,
    )?;
    Ok(true)
}
