#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;
use kernels::{Bind, Fire};

use crate::jit::abi::{MaybeConst, bf16};
use crate::jit::{Ctx, Launch};
use kernels::Refusal;

const BLOCK: u32 = 256;

const TILE: u32 = 16;

const LAYERNORM_BLOCK: u32 = 128;

const AXIAL_HEAD_DIM: u32 = 64;

const WARP: u32 = 32;

const fn tile16(rows: u32, width: u32) -> Launch {
    Launch::grid(
        [width.div_ceil(TILE), rows.div_ceil(TILE), 1],
        [TILE, TILE, 1],
    )
}

fn extent(what: &'static str, value: i32) -> Result<u32, Refusal> {
    if value <= 0 {
        return Err(Refusal::Empty { what });
    }
    Ok(value.unsigned_abs())
}

fn axes(what: &'static str, a: u32, b: u32) -> Result<u32, Refusal> {
    a.checked_mul(b).ok_or(Refusal::Wide {
        what,
        at: i64::from(a) * i64::from(b),
        max: i64::from(u32::MAX),
    })
}

fn flat(what: &'static str, t: usize) -> Result<Launch, Refusal> {
    if t == 0 {
        return Err(Refusal::Empty { what });
    }
    let blocks = t.div_ceil(BLOCK as usize);
    let Ok(blocks) = u32::try_from(blocks) else {
        return Err(Refusal::Wide {
            what,
            at: i64::try_from(t).unwrap_or(i64::MAX),
            max: i64::from(u32::MAX) * i64::from(BLOCK),
        });
    };
    Ok(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1]))
}

pub fn k_rms_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    weight: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let blocks = extent("rows", rows)?;

    extent("width", width)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_rms<::pie::bf16>",
        )
        .apply(Launch::per_row(blocks, BLOCK)),
        &[
            x.cast::<bf16>().arg(),
            MaybeConst::new(weight.cast::<bf16>()).arg(),
            o.cast::<bf16>().arg(),
            rows.arg(),
            width.arg(),
            eps.arg(),
        ],
    )
}

pub fn k_add_bf16(
    ctx: &Ctx<'_>,
    a: *mut c_void,
    b: *const c_void,
    n: usize,
) -> Result<(), Refusal> {
    let launch = flat("n", n)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_add<::pie::bf16>",
        )
        .apply(launch),
        &[a.cast::<bf16>().arg(), b.cast::<bf16>().arg(), n.arg()],
    )
}

pub fn k_f32_to_bf16_bf16(
    ctx: &Ctx<'_>,
    a: *const c_void,
    o: *mut c_void,
    n: usize,
) -> Result<(), Refusal> {
    let launch = flat("n", n)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_f32_to_bf16<::pie::bf16>",
        )
        .apply(launch),
        &[a.cast::<f32>().arg(), o.cast::<bf16>().arg(), n.arg()],
    )
}

pub fn k_gelu_erf_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_gelu_erf<::pie::bf16>",
        )
        .apply(launch),
        &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
    )
}

pub fn k_layernorm_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    g: *const c_void,
    beta: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let blocks = extent("rows", rows)?;

    extent("width", width)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_layernorm<::pie::bf16>",
        )
        .apply(Launch::per_row(blocks, BLOCK)),
        &[
            x.cast::<bf16>().arg(),
            MaybeConst::new(g.cast::<bf16>()).arg(),
            MaybeConst::new(beta.cast::<bf16>()).arg(),
            o.cast::<bf16>().arg(),
            rows.arg(),
            width.arg(),
            eps.arg(),
        ],
    )
}

pub fn k_matmul_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    n: i32,
    k: i32,
    o: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    ctx.fire(
        Fire::at(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_matmul<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            x.cast::<bf16>().arg(),
            w.cast::<bf16>().arg(),
            y.cast::<bf16>().arg(),
            n.arg(),
            k.arg(),
            o.arg(),
        ],
    )
}

pub fn k_clamp_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    o: *mut c_void,
    lo: *const c_void,
    hi: *const c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_naive_kernels.cuh",
            "::pie::vision::k_clamp<::pie::bf16>",
        )
        .apply(launch),
        &[
            x.cast::<bf16>().arg(),
            o.cast::<bf16>().arg(),
            MaybeConst::new(lo.cast::<bf16>()).arg(),
            MaybeConst::new(hi.cast::<bf16>()).arg(),
            t.arg(),
        ],
    )
}

pub fn k_scale_bf16(
    ctx: &Ctx<'_>,
    p: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_scale<::pie::bf16>",
        )
        .apply(launch),
        &[p.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
    )
}

pub fn k_softmax_bf16(ctx: &Ctx<'_>, s: *mut c_void, n: i32) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_softmax<::pie::bf16>",
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[s.cast::<f32>().arg(), n.arg()],
    )
}

pub fn k_pool_finish_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    o: *mut c_void,
    s: f32,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_pool_finish<::pie::bf16>",
        )
        .apply(launch),
        &[
            input.cast::<f32>().arg(),
            o.cast::<bf16>().arg(),
            s.arg(),
            t.arg(),
        ],
    )
}

pub fn k_addpos_grid2d_bf16(
    ctx: &Ctx<'_>,
    y: *mut c_void,
    tb: *const c_void,
    pos: *const c_void,
    n: i32,
    o: i32,
    p: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_addpos_grid2d<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            y.cast::<bf16>().arg(),
            tb.cast::<bf16>().arg(),
            pos.cast::<f32>().arg(),
            n.arg(),
            o.arg(),
            p.arg(),
        ],
    )
}

pub fn k_rope_axial2d_bf16(
    ctx: &Ctx<'_>,
    q: *mut c_void,
    pos: *const c_void,
    n: i32,
    h: i32,
    theta: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let heads = extent("h", h)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_rope_axial2d<::pie::bf16>",
        )
        .apply(Launch::grid([1, heads, rows], [WARP, 1, 1])),
        &[
            q.cast::<bf16>().arg(),
            pos.cast::<f32>().arg(),
            n.arg(),
            h.arg(),
            theta.arg(),
        ],
    )
}

pub fn k_qk_bf16(
    ctx: &Ctx<'_>,
    q: *const c_void,
    k: *const c_void,
    s: *mut c_void,
    n: i32,
    h: i32,
    head: i32,
    scale: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_qk<::pie::bf16>",
        )
        .apply(tile16(rows, rows)),
        &[
            q.cast::<bf16>().arg(),
            k.cast::<bf16>().arg(),
            s.cast::<f32>().arg(),
            n.arg(),
            h.arg(),
            head.arg(),
            scale.arg(),
        ],
    )
}

pub fn k_av_bf16(
    ctx: &Ctx<'_>,
    s: *const c_void,
    v: *const c_void,
    o: *mut c_void,
    n: i32,
    h: i32,
    head: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_av<::pie::bf16>",
        )
        .apply(tile16(rows, AXIAL_HEAD_DIM)),
        &[
            s.cast::<f32>().arg(),
            v.cast::<bf16>().arg(),
            o.cast::<bf16>().arg(),
            n.arg(),
            h.arg(),
            head.arg(),
        ],
    )
}

pub fn k_pool_bf16(
    ctx: &Ctx<'_>,
    h: *const c_void,
    grp: *const c_void,
    o: *mut c_void,
    n: i32,
    d: i32,
    k2: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("d", d)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_pool<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            h.cast::<bf16>().arg(),
            grp.cast::<i32>().arg(),
            o.cast::<f32>().arg(),
            n.arg(),
            d.arg(),
            k2.arg(),
        ],
    )
}

pub fn k_silu_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_silu<::pie::bf16>",
        )
        .apply(launch),
        &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
    )
}

pub fn k_axpy_bf16(
    ctx: &Ctx<'_>,
    a: *mut c_void,
    b: *const c_void,
    scale: f32,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_axpy<::pie::bf16>",
        )
        .apply(launch),
        &[
            a.cast::<bf16>().arg(),
            b.cast::<bf16>().arg(),
            scale.arg(),
            t.arg(),
        ],
    )
}

pub fn k_matmul_bias_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    w: *const c_void,
    b: *const c_void,
    y: *mut c_void,
    n: i32,
    k: i32,
    o: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_matmul_bias<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            x.cast::<bf16>().arg(),
            w.cast::<bf16>().arg(),
            MaybeConst::new(b.cast::<bf16>()).arg(),
            y.cast::<bf16>().arg(),
            n.arg(),
            k.arg(),
            o.arg(),
        ],
    )
}

pub fn k_glu_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    o: *mut c_void,
    n: i32,
    d: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("d", d)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_glu<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            x.cast::<bf16>().arg(),
            o.cast::<bf16>().arg(),
            n.arg(),
            d.arg(),
        ],
    )
}

pub fn k_layernorm_relu_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    w: *const c_void,
    o: *mut c_void,
    r: i32,
    c: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let rows = extent("r", r)?;

    extent("c", c)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_layernorm_relu<::pie::bf16>",
        )
        .apply(Launch::per_row(rows, LAYERNORM_BLOCK)),
        &[
            x.cast::<bf16>().arg(),
            MaybeConst::new(w.cast::<bf16>()).arg(),
            o.cast::<bf16>().arg(),
            r.arg(),
            c.arg(),
            eps.arg(),
        ],
    )
}

pub fn k_sscp_flatten_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let rows = extent("t_out", t_out)?;
    let width = axes("f_out * oc", extent("f_out", f_out)?, extent("oc", oc)?)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_sscp_flatten<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            input.cast::<bf16>().arg(),
            out.cast::<bf16>().arg(),
            oc.arg(),
            t_out.arg(),
            f_out.arg(),
        ],
    )
}

pub fn k_qkv_scale_bf16(
    ctx: &Ctx<'_>,
    q: *mut c_void,
    k: *mut c_void,
    pds: *const c_void,
    n: i32,
    h: i32,
    hd: i32,
    q_scale: f32,
    k_scale: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = axes("h * hd", extent("h", h)?, extent("hd", hd)?)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_qkv_scale<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            q.cast::<bf16>().arg(),
            k.cast::<bf16>().arg(),
            pds.cast::<bf16>().arg(),
            n.arg(),
            h.arg(),
            hd.arg(),
            q_scale.arg(),
            k_scale.arg(),
        ],
    )
}

pub fn k_rel_pos_enc_bf16(
    ctx: &Ctx<'_>,
    pe: *mut c_void,
    p: i32,
    hidden: i32,
) -> Result<(), Refusal> {
    let rows = extent("p", p)?;
    let width = extent("hidden", hidden)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_rel_pos_enc<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[pe.cast::<bf16>().arg(), p.arg(), hidden.arg()],
    )
}

pub fn k_conv2d_s2_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    w: *const c_void,
    out: *mut c_void,
    ic: i32,
    t_in: i32,
    f_in: i32,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_conv2d_s2<::pie::bf16>",
        )
        .apply(launch),
        &[
            input.cast::<bf16>().arg(),
            w.cast::<bf16>().arg(),
            out.cast::<bf16>().arg(),
            ic.arg(),
            t_in.arg(),
            f_in.arg(),
            oc.arg(),
            t_out.arg(),
            f_out.arg(),
        ],
    )
}

pub fn k_chlast_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_chlast<::pie::bf16>",
        )
        .apply(launch),
        &[
            input.cast::<bf16>().arg(),
            out.cast::<bf16>().arg(),
            oc.arg(),
            t_out.arg(),
            f_out.arg(),
        ],
    )
}

pub fn k_chfirst_bf16(
    ctx: &Ctx<'_>,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_chfirst<::pie::bf16>",
        )
        .apply(launch),
        &[
            input.cast::<bf16>().arg(),
            out.cast::<bf16>().arg(),
            oc.arg(),
            t_out.arg(),
            f_out.arg(),
        ],
    )
}

fn channelled(oc: i32, t_out: i32, f_out: i32) -> Result<Launch, Refusal> {
    let channels = extent("oc", oc)?;
    let time = extent("t_out", t_out)?;
    let freq = extent("f_out", f_out)?;
    Ok(Launch::grid(
        [freq.div_ceil(TILE), time.div_ceil(TILE), channels],
        [TILE, TILE, 1],
    ))
}

pub fn k_local_attn_bf16(
    ctx: &Ctx<'_>,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    relk: *const c_void,
    out: *mut c_void,
    n: i32,
    h: i32,
    hd: i32,
    p: i32,
    cap: f32,
) -> Result<(), Refusal> {
    const LOCAL_ATTN_BLOCK: u32 = 128;

    let tiles = extent("n", n)?.div_ceil(LOCAL_ATTN_BLOCK);
    let heads = extent("h", h)?;
    ctx.fire(
        Fire::at(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_local_attn<::pie::bf16>",
        )
        .apply(Launch::grid([tiles, heads, 1], [LOCAL_ATTN_BLOCK, 1, 1])),
        &[
            q.cast::<bf16>().arg(),
            k.cast::<bf16>().arg(),
            v.cast::<bf16>().arg(),
            relk.cast::<bf16>().arg(),
            out.cast::<bf16>().arg(),
            n.arg(),
            h.arg(),
            hd.arg(),
            p.arg(),
            cap.arg(),
        ],
    )
}

pub fn k_bias_bf16(
    ctx: &Ctx<'_>,
    y: *mut c_void,
    b: *const c_void,
    m: usize,
    n: i32,
) -> Result<(), Refusal> {
    let width = usize::try_from(extent("n", n)?).unwrap_or(usize::MAX);
    let count = m.checked_mul(width).ok_or(Refusal::Wide {
        what: "m * n",
        at: i64::MAX,
        max: i64::MAX,
    })?;
    let launch = flat("m * n", count)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_bias<::pie::bf16>",
        )
        .apply(launch),
        &[
            y.cast::<bf16>().arg(),
            b.cast::<bf16>().arg(),
            m.arg(),
            n.arg(),
        ],
    )
}

pub fn k_add_pe_bf16(
    ctx: &Ctx<'_>,
    h: *mut c_void,
    pe: *const c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_add_pe<::pie::bf16>",
        )
        .apply(launch),
        &[h.cast::<bf16>().arg(), pe.cast::<bf16>().arg(), t.arg()],
    )
}

pub fn k_gelu_tanh_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_gelu_tanh<::pie::bf16>",
        )
        .apply(launch),
        &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
    )
}

pub fn k_gelu_bias_bf16(
    ctx: &Ctx<'_>,
    x: *mut c_void,
    b: *const c_void,
    n: i32,
    d: i32,
) -> Result<(), Refusal> {
    fn elements(what: &'static str, rows: i32, width: i32) -> Result<usize, Refusal> {
        let rows = usize::try_from(extent(what, rows)?).unwrap_or(usize::MAX);
        let width = usize::try_from(extent(what, width)?).unwrap_or(usize::MAX);
        rows.checked_mul(width).ok_or(Refusal::Wide {
            what,
            at: i64::MAX,
            max: i64::MAX,
        })
    }

    let launch = flat("n * d", elements("n * d", n, d)?)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_gelu_bias<::pie::bf16>",
        )
        .apply(launch),
        &[
            x.cast::<bf16>().arg(),
            MaybeConst::new(b.cast::<bf16>()).arg(),
            n.arg(),
            d.arg(),
        ],
    )
}

pub fn k_merge_gather_bf16(
    ctx: &Ctx<'_>,
    h: *const c_void,
    g: *mut c_void,
    n_token: i32,
    u: i32,
    c: i32,
) -> Result<(), Refusal> {
    let rows = extent("n_token", n_token)?;
    let width = axes("u * c", extent("u", u)?, extent("c", c)?)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_merge_gather<::pie::bf16>",
        )
        .apply(tile16(rows, width)),
        &[
            h.cast::<bf16>().arg(),
            g.cast::<bf16>().arg(),
            n_token.arg(),
            u.arg(),
            c.arg(),
        ],
    )
}

pub fn k_split_rope_qkv_bf16(
    ctx: &Ctx<'_>,
    qkv: *const c_void,
    b: *const c_void,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    pos: *const c_void,
    n: i32,
    nh: i32,
    head: i32,
    theta: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let heads = extent("nh", nh)?;
    let half = extent("head / 2", head / 2)?;
    ctx.fire(
        Fire::at(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_split_rope_qkv<::pie::bf16>",
        )
        .apply(Launch::grid([heads, rows, 1], [half, 1, 1])),
        &[
            qkv.cast::<bf16>().arg(),
            MaybeConst::new(b.cast::<bf16>()).arg(),
            q.cast::<bf16>().arg(),
            k.cast::<bf16>().arg(),
            v.cast::<bf16>().arg(),
            pos.cast::<f32>().arg(),
            n.arg(),
            nh.arg(),
            head.arg(),
            theta.arg(),
        ],
    )
}
