
use kernels::{Bind, Fire};
use core::ptr::NonNull;

use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;

const TILE_M: u32 = 128;

const TILE_M_SMALL: u32 = 64;

const TILE_N: u32 = 128;

const NO_HEAD_DIM: Refusal = Refusal::Unstated {
    what: "an FA4 forward at this head dim -- 64 and 128 are here",
};

const fn forward_instantiation(
    head_dim: u32,
    causal: bool,
    packed: bool,
    small: bool,
) -> Option<&'static str> {
    Some(match (head_dim, causal, packed, small) {
        (64, false, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, false>>"
        }
        (64, false, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, true>>"
        }
        (64, true, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, true, 4u, false>>"
        }
        (64, true, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, true, 4u, true>>"
        }
        (64, false, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, false, 4u, false>>"
        }
        (64, false, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, false, 4u, true>>"
        }
        (64, true, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, true, 4u, false>>"
        }
        (64, true, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, true, 4u, true>>"
        }
        (128, false, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, false>>"
        }
        (128, false, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, true>>"
        }
        (128, true, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, true, 8u, false>>"
        }
        (128, true, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, true, 8u, true>>"
        }
        (128, false, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, false, 4u, false>>"
        }
        (128, false, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, false, 4u, true>>"
        }
        (128, true, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, true, 4u, false>>"
        }
        (128, true, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, true, 4u, true>>"
        }
        _ => return None,
    })
}

const fn combine_instantiation(head_dim: u32) -> Option<&'static str> {
    Some(match head_dim {
        64 => {
            "::pie::attn::fa4::combine<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, false>>"
        }
        128 => {
            "::pie::attn::fa4::combine<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, false>>"
        }
        _ => return None,
    })
}

const fn smem_bytes(head_dim: u32, small: bool) -> u32 {
    (tile_m(small) + 2 * TILE_N) * head_dim * core::mem::size_of::<u16>() as u32
}

const fn tile_m(small: bool) -> u32 {
    if small { TILE_M_SMALL } else { TILE_M }
}

const fn geometry(head_dim: u32, small: bool) -> Option<(u32, u32, u32)> {
    let num_warps = match (head_dim, small) {
        (64, _) | (128, true) => 4,
        (128, false) => 8,
        _ => return None,
    };
    Some((num_warps * 32, smem_bytes(head_dim, small), tile_m(small)))
}

const fn blocks_m(seqlen_q: u32, pack: u32, small: bool) -> u32 {
    (seqlen_q * pack).div_ceil(tile_m(small))
}

pub const MAX_SPLITS: u32 = 32;

const FIXED_TILES: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    packed: bool,
    splits: u32,
    small: bool,
}

const fn plan(
    seqlen_q: u32,
    seqlen_k: u32,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    num_sms: u32,
    may_split: bool,
) -> Plan {
    if num_sms == 0 || batch == 0 || heads_kv == 0 {
        return Plan { packed: false, splits: 1, small: false };
    }
    let group = heads_q / heads_kv;
    let tiles = if seqlen_k.div_ceil(TILE_N) == 0 { 1 } else { seqlen_k.div_ceil(TILE_N) };
    let mut best_cost = u32::MAX;
    let mut best = Plan { packed: false, splits: 1, small: false };

    let mut which = 0;
    while which < 2 {
        let packed = which == 1;
        if packed && group <= 1 {
            break;
        }
        let pack = if packed { group } else { 1 };

        let small = seqlen_q * pack <= TILE_M_SMALL;
        let m_tiles = blocks_m(seqlen_q, pack, small);
        let blocks = m_tiles * (heads_q / pack) * batch;
        let limit = if may_split && m_tiles == 1 {
            if tiles < MAX_SPLITS { tiles } else { MAX_SPLITS }
        } else {
            1
        };
        let mut splits = 1;
        while splits <= limit {
            let cost = (blocks * splits).div_ceil(num_sms) * (tiles.div_ceil(splits) + FIXED_TILES);
            if cost < best_cost {
                best_cost = cost;
                best = Plan { packed, splits, small };
            }
            splits += 1;
        }
        which += 1;
    }
    best
}

pub fn split_scratch_elems(
    ctx: &Ctx<'_>,
    seqlen_q: u32,
    seqlen_k: u32,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    head_dim: u32) -> (usize, usize) {
    let num_sms = ctx.multiprocessors().unwrap_or(0);
    let p = plan(seqlen_q, seqlen_k, batch, heads_q, heads_kv, num_sms, true);
    if p.splits == 1 {
        return (0, 0);
    }
    let rows = batch as usize * heads_q as usize * seqlen_q as usize * p.splits as usize;
    (rows * head_dim as usize, rows)
}

fn null_check(is_null: bool, which: &'static str) -> Result<(), Refusal> {
    if is_null {
        Err(Refusal::Null { what: which })
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fa4 {

    pub q: *const bf16,
    pub k: *const bf16,
    pub v: *const bf16,
    pub o: *mut bf16,
    pub lse: *mut f32,
    pub o_partial: *mut f32,
    pub lse_partial: *mut f32,
    pub q_stride_b: i32,
    pub q_stride_s: i32,
    pub q_stride_h: i32,
    pub k_stride_b: i32,
    pub k_stride_s: i32,
    pub k_stride_h: i32,
    pub v_stride_b: i32,
    pub v_stride_s: i32,
    pub v_stride_h: i32,
    pub o_stride_b: i32,
    pub o_stride_s: i32,
    pub o_stride_h: i32,
    pub lse_stride_b: i32,
    pub lse_stride_h: i32,
    pub batch: u32,
    pub heads_q: u32,
    pub heads_kv: u32,
    pub head_dim: u32,
    pub seqlen_q: u32,
    pub seqlen_k: u32,
    pub causal: bool,
    pub scale_log2: f32,
}

pub unsafe fn forward(ctx: &Ctx<'_>, job: Fa4) -> Result<(), Refusal> {

    geometry(job.head_dim, false).ok_or(NO_HEAD_DIM)?;

    null_check(job.q.is_null(), "q")?;
    null_check(job.k.is_null(), "k")?;
    null_check(job.v.is_null(), "v")?;
    null_check(job.o.is_null(), "o")?;

    if job.heads_kv == 0 {
        return Err(Refusal::Empty { what: "heads_kv" });
    }
    if !job.heads_q.is_multiple_of(job.heads_kv) {
        return Err(Refusal::Unstated {
            what: "a GQA group -- heads_q must be a multiple of heads_kv",
        });
    }
    if job.batch == 0 {
        return Err(Refusal::Empty { what: "batch" });
    }
    if job.heads_q == 0 {
        return Err(Refusal::Empty { what: "heads_q" });
    }
    if job.seqlen_q == 0 {
        return Err(Refusal::Empty { what: "seqlen_q" });
    }
    if job.seqlen_k == 0 {
        return Err(Refusal::Empty { what: "seqlen_k" });
    }

    let group_size = (job.heads_q / job.heads_kv) as i32;

    let num_sms = ctx.multiprocessors().unwrap_or(0);
    if job.o_partial.is_null() != job.lse_partial.is_null() {
        return Err(Refusal::Null {
            what: if job.o_partial.is_null() { "o_partial" } else { "lse_partial" },
        });
    }
    let Plan { packed, splits, small } = plan(
        job.seqlen_q,
        job.seqlen_k,
        job.batch,
        job.heads_q,
        job.heads_kv,
        num_sms,
        !job.o_partial.is_null(),
    );
    let instantiation =
        forward_instantiation(job.head_dim, job.causal, packed, small).ok_or(NO_HEAD_DIM)?;

    let (num_threads, smem, _tile_m) = geometry(job.head_dim, small).ok_or(NO_HEAD_DIM)?;

    let pack = if packed { group_size as u32 } else { 1 };

    let grid = [
        blocks_m(job.seqlen_q, pack, small),
        job.heads_q / pack,
        job.batch * splits,
    ];

    ctx.fire(Fire::at("attn/fa4.cuh", instantiation).apply(Launch::grid(grid, [num_threads, 1, 1]).smem(smem)), &[
                job.q.arg(),
                job.k.arg(),
                job.v.arg(),
                job.o.arg(),
                NonNull::new(job.lse).arg(),
                job.q_stride_b.arg(),
                job.q_stride_s.arg(),
                job.q_stride_h.arg(),
                job.k_stride_b.arg(),
                job.k_stride_s.arg(),
                job.k_stride_h.arg(),
                job.v_stride_b.arg(),
                job.v_stride_s.arg(),
                job.v_stride_h.arg(),
                job.o_stride_b.arg(),
                job.o_stride_s.arg(),
                job.o_stride_h.arg(),
                job.lse_stride_b.arg(),
                job.lse_stride_h.arg(),
                (job.seqlen_q as i32).arg(),
                (job.seqlen_k as i32).arg(),
                group_size.arg(),
                job.o_partial.arg(),
                job.lse_partial.arg(),
                (splits as i32).arg(),
                (job.heads_q as i32).arg(),
                job.scale_log2.arg(),
            ])?;

    if splits == 1 {
        return Ok(());
    }

    let rows = job.batch * job.heads_q * job.seqlen_q;
    let combine = combine_instantiation(job.head_dim).ok_or(NO_HEAD_DIM)?;
    ctx.fire(Fire::at("attn/fa4.cuh", combine).apply(Launch::grid([rows, 1, 1], [job.head_dim, 1, 1])), &[
                job.o_partial.cast_const().arg(),
                job.lse_partial.cast_const().arg(),
                job.o.arg(),
                NonNull::new(job.lse).arg(),
                (splits as i32).arg(),
                (job.heads_q as i32).arg(),
                (job.seqlen_q as i32).arg(),
                job.o_stride_b.arg(),
                job.o_stride_s.arg(),
                job.o_stride_h.arg(),
                job.lse_stride_b.arg(),
                job.lse_stride_h.arg(),
            ])
}
