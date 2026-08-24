use kernels::BindMut;
use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;
use kernels_macros::routine;

use crate::plane::{self, Handle};
use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use crate::views::{AttnMask, AttnSplit, KvCache};
use kernels::raises::Struct;

fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|&p| p == head_dim)
        .ok_or(Refusal::Narrow {
            what: "a head width no shader is compiled for",
            at: i64::from(head_dim),
        })
}

fn vector_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(1024).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * 1024,
    })?;
    Ok([x, rows, 1])
}

fn tiled_grid(q_heads: i32, rows: i32, group: u32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(group).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * i64::from(group),
    })?;
    Ok([x, rows.div_ceil(32), 1])
}

pub(crate) fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
    Ok([
        positive(head_dim, "the head width")?,
        positive(heads, "heads")?,
        positive(depth, "tokens")?,
    ])
}

pub(crate) const fn head_group(grid: [u32; 3]) -> [u32; 3] {
    [grid[0], 1, 1]
}

fn positive(v: i32, what: &'static str) -> Result<u32, Refusal> {
    if v <= 0 {
        return Err(Refusal::Empty { what });
    }
    Ok(v.unsigned_abs())
}

// INLINED into impl Layout; dies with the routine layer.
#[routine(canon = "layout.split_qkv", out(q = rows(packed) x const(q_width)), out(k = rows(packed) x const(kv_width)), out(v = rows(packed) x const(kv_width)))]
pub fn split_qkv_bf16(
    ctx: &Ctx<'_>,
    packed: In<Tensor<bf16>>,
    q: Out<Tensor<bf16>>,
    k: Out<Tensor<bf16>>,
    v: Out<Tensor<bf16>>,
    q_width: Const<u32>,
    kv_width: Const<u32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let packed_width = packed.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("attn/split_qkv.metal", "split_qkv_bf16")
            .apply(Grid::of(elementwise_rows(packed_width, rows)?, [256, 1, 1])),
        &[
            packed.arg(),
            q.arg(),
            k.arg(),
            v.arg(),
            q_width.arg(),
            kv_width.arg(),
        ],
    )
}

// INLINED into impl Gate; dies with the routine layer.
#[routine(canon = "gate.sigmoid_mul", out(attn = like(attn)))]
pub fn gate(
    ctx: &Ctx<'_>,
    attn: InOut<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    row_stride: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = attn.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("attn/gate.metal", "gate_bfloat16")
            .apply(Grid::of(elementwise_rows(width, rows)?, [256, 1, 1])),
        &[attn.arg(), gate.arg(), row_stride.arg()],
    )
}

// INLINED into impl Layout; dies with the routine layer.
#[routine(canon = "layout.split_q_gate")]
pub fn q_gate_split(
    ctx: &Ctx<'_>,
    qg: In<Tensor<bf16>>,
    q_out: Out<Tensor<bf16>>,
    gate_out: Out<Tensor<bf16>>,
    head_dim: Const<i32>,
    qg_row_stride: Const<i32>,
    out_row_stride: Const<i32>,
    q_heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let rows = *rows;
    let lanes = head_grid(*head_dim, *q_heads, rows)?;
    ctx.fire(
        Fire::at("attn/gate.metal", "q_gate_split_bfloat16")
            .apply(Grid::of(lanes, head_group(lanes))),
        &[
            qg.arg(),
            q_out.arg(),
            gate_out.arg(),
            head_dim.arg(),
            qg_row_stride.arg(),
            out_row_stride.arg(),
        ],
    )
}

#[routine]
pub fn kv_append(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    head_dim: Const<i32>,
    heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let k_cache = kvc.keys;
    let v_cache = kvc.values;
    let pos = positions.ptr;
    let k_head_stride = kvc.head_stride;
    let k_seq_stride = kvc.seq_stride;
    let lanes = head_grid(*head_dim, *heads, 1)?;
    ctx.fire(
        Fire::at("attn/kv_write.metal", "kv_append_bfloat16")
            .apply(Grid::of(lanes, head_group(lanes))),
        &[
            k_new.arg(),
            v_new.arg(),
            k_cache.arg_mut(),
            v_cache.arg_mut(),
            pos.arg(),
            head_dim.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
        ],
    )
}

// INLINED into impl Attention; dies with the routine layer.
#[routine(canon = "attention.kv_append")]
pub fn kv_append_paged(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    head_dim: Const<i32>,
    n_kv_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let page_size = kvc.page_size;
    // ZERO IS NOT A PAGE SIZE, and here it is not a harmless one either.
    // `lowering::views::kv` builds this view with `pooled(..).unwrap_or(0)`
    // and its doc says a paged access at page size zero "refuses at its
    // grid" -- true of the paged READ, whose grid is built from the page
    // count, and never true of this write, whose grid is heads by tokens and
    // does not consult the number at all. So a store with no pool behind it
    // planned a full write in which every token divides to page zero, offset
    // zero: twenty-eight layers of keys and values landing on one row and
    // overwriting each other, with no refusal anywhere.
    //
    // The refusal `model_dispatch`'s `a_paged_write_with_no_page_size_...`
    // asserts is this one. It had been asserting it against a routine that
    // could not make it since the write was ported, in a test target the
    // no-ask series never built.
    if page_size <= 0 {
        return Err(Refusal::Empty {
            what: "the KV page size",
        });
    }

    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let ring_4 = ctx.absent()?;
    let ring_6 = ctx.absent()?;
    let ring_7 = ctx.absent()?;
    let ring_8 = ctx.absent()?;
    let ring_9 = ctx.absent()?;
    let ring_11 = ctx.absent()?;
    let w_page = kvc.write_page;
    let w_off = kvc.write_offset;
    let tokens = *tokens;
    let lanes = head_grid(*head_dim, *n_kv_heads, tokens)?;
    ctx.fire(
        Fire::at("attn/kv_write.metal", "kv_append_paged_bfloat16")
            .apply(Grid::of(lanes, head_group(lanes))),
        &[
            k_new.arg(),
            v_new.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            ring_4,
            head_dim.arg(),
            ring_6,
            ring_7,
            ring_8,
            ring_9,
            page_size.arg(),
            ring_11,
            n_kv_heads.arg(),
            w_page.arg(),
            w_off.arg(),
            0_i32.arg(),
        ],
    )
}

// INLINED into impl Attention; dies with the routine layer.
#[routine(out(out = like(logits)))]
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    cap: Const<f32>,
) -> Result<(), Refusal> {
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(
        Fire::at("attn/logit_softcap.metal", "logit_softcap_bfloat16")
            .apply(Grid::of(elementwise(n, 1)?, [256, 1, 1])),
        &[logits.arg(), out.arg(), cap.arg()],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_decode(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    rows: Const<i32>,
    split: In<Struct<AttnSplit>>,
) -> Result<(), Refusal> {
    // This plane fires unsplit; the split policy is another driver's.
    let _ = split;
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let sinks = ctx.absent()?;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.metal",
            [
                "sdpa_paged_decode_bfloat16_d_64",
                "sdpa_paged_decode_bfloat16_d_128",
                "sdpa_paged_decode_bfloat16_d_256",
                "sdpa_paged_decode_bfloat16_d_512",
            ][head_point(*head_dim, &[64, 128, 256, 512])?],
        )
        .apply(Grid::of(vector_grid(*q_heads, rows)?, [1024, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks,
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_decode_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    rows: Const<i32>,
    split: In<Struct<AttnSplit>>,
) -> Result<(), Refusal> {
    // This plane fires unsplit; the split policy is another driver's.
    let _ = split;
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.metal",
            ["sdpa_paged_decode_sink_bfloat16_d_64"][head_point(*head_dim, &[64])?],
        )
        .apply(Grid::of(vector_grid(*q_heads, rows)?, [1024, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_tiled(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    n_rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let sinks = ctx.absent()?;
    let n_rows = *n_rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.metal",
            [
                "sdpa_paged_tiled_bfloat16_d_64",
                "sdpa_paged_tiled_bfloat16_d_128",
                "sdpa_paged_tiled_bfloat16_d_256",
                "sdpa_paged_tiled_bfloat16_d_512",
            ][head_point(*head_dim, &[64, 128, 256, 512])?],
        )
        .apply(Grid::of(tiled_grid(*q_heads, n_rows, 1024)?, [1024, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks,
            n_rows.arg(),
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_tiled_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    n_rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let n_rows = *n_rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.metal",
            ["sdpa_paged_tiled_sink_bfloat16_d_64"][head_point(*head_dim, &[64])?],
        )
        .apply(Grid::of(tiled_grid(*q_heads, n_rows, 1024)?, [1024, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
            n_rows.arg(),
        ],
    )
}

#[routine]
pub fn sdpa_paged_tiled_strided(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    n_rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let sinks = ctx.absent()?;
    let n_rows = *n_rows;

    let q_row_pitch = ctx.param(5)?;

    let o_row_pitch = ctx.param(6)?;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.metal",
            ["sdpa_paged_tiled_strided_bfloat16_d_256"][head_point(*head_dim, &[256])?],
        )
        .apply(Grid::of(tiled_grid(*q_heads, n_rows, 1024)?, [1024, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks,
            n_rows.arg(),
            q_row_pitch.arg(),
            o_row_pitch.arg(),
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_mma(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    n_rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let sinks = ctx.absent()?;
    let n_rows = *n_rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged_mma.metal",
            ["sdpa_paged_mma_bfloat16_d_64"][head_point(*head_dim, &[64])?],
        )
        .apply(Grid::of(tiled_grid(*q_heads, n_rows, 128)?, [128, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks,
            n_rows.arg(),
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_paged_mma_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    n_kv_heads: Const<i32>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    n_rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let page_size = kvc.page_size;

    let k_pages = kvc.keys;
    let v_pages = kvc.values;

    let gqa_factor = if *n_kv_heads > 0 {
        *q_heads / *n_kv_heads
    } else {
        0
    };
    let position_ids = positions.ptr;
    let req_of_token = request_of_token.ptr;
    let kv_page_indices = kvc.page_indices;
    let kv_page_indptr = kvc.page_indptr;
    let attention_mask = maskv.mask;
    let attention_mask_stride = maskv.stride;
    let attention_mask_enabled = maskv.enabled;
    let n_rows = *n_rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged_mma.metal",
            ["sdpa_paged_mma_sink_bfloat16_d_64"][head_point(*head_dim, &[64])?],
        )
        .apply(Grid::of(tiled_grid(*q_heads, n_rows, 128)?, [128, 1, 1])),
        &[
            queries.arg(),
            k_pages.arg(),
            v_pages.arg(),
            out.arg(),
            gqa_factor.arg(),
            position_ids.arg(),
            req_of_token.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            attention_mask.arg(),
            attention_mask_stride.arg(),
            attention_mask_enabled.arg(),
            window.arg(),
            sinks.arg(),
            n_rows.arg(),
        ],
    )
}

#[routine(out(out = like(queries)))]
pub fn sdpa_vector_decode(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    kvc: In<Struct<KvCache>>,
    n_kv_heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let keys = kvc.keys;
    let values = kvc.values;

    let n_kv_heads = *n_kv_heads;
    let gqa_factor = if n_kv_heads > 0 {
        *q_heads / n_kv_heads
    } else {
        0
    };
    let n = out.width;
    let k_head_stride = kvc.head_stride;
    let k_seq_stride = kvc.seq_stride;
    let v_head_stride = kvc.head_stride;
    let v_seq_stride = kvc.seq_stride;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_vector.metal",
            [
                "sdpa_vector_decode_bfloat16_d_64",
                "sdpa_vector_decode_bfloat16_d_128",
                "sdpa_vector_decode_bfloat16_d_256",
            ][head_point(*head_dim, &[64, 128, 256])?],
        )
        .apply(Grid::of(vector_grid(*q_heads, rows)?, [1024, 1, 1])),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
        ],
    )
}

#[routine]
pub fn sdpa_vector_decode_swa(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    window: Const<i32>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    q_row_stride: Const<i32>,
    o_row_stride: Const<i32>,
    kvc: In<Struct<KvCache>>,
    n_kv_heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let keys = kvc.keys;
    let values = kvc.values;

    let n_kv_heads = *n_kv_heads;
    let gqa_factor = if n_kv_heads > 0 {
        *q_heads / n_kv_heads
    } else {
        0
    };
    let n = out.width;
    let k_head_stride = kvc.head_stride;
    let k_seq_stride = kvc.seq_stride;
    let v_head_stride = kvc.head_stride;
    let v_seq_stride = kvc.seq_stride;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_sliding.metal",
            [
                "sdpa_vector_decode_swa_bfloat16_d_256",
                "sdpa_vector_decode_swa_bfloat16_d_512",
            ][head_point(*head_dim, &[256, 512])?],
        )
        .apply(Grid::of(vector_grid(*q_heads, rows)?, [1024, 1, 1])),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
            window.arg(),
            q_row_stride.arg(),
            o_row_stride.arg(),
        ],
    )
}

#[routine]
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    scale: Const<f32>,
    window: Const<i32>,
    sinks: Const<Tensor<bf16>>,
    head_dim: Const<i32>,
    q_heads: Const<i32>,
    q_row_stride: Const<i32>,
    o_row_stride: Const<i32>,
    kvc: In<Struct<KvCache>>,
    n_kv_heads: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let keys = kvc.keys;
    let values = kvc.values;

    let n_kv_heads = *n_kv_heads;
    let gqa_factor = if n_kv_heads > 0 {
        *q_heads / n_kv_heads
    } else {
        0
    };
    let n = out.width;
    let k_head_stride = kvc.head_stride;
    let k_seq_stride = kvc.seq_stride;
    let v_head_stride = kvc.head_stride;
    let v_seq_stride = kvc.seq_stride;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "attn/sdpa_sliding.metal",
            ["sdpa_vector_decode_sink_bfloat16_d_64"][head_point(*head_dim, &[64])?],
        )
        .apply(Grid::of(vector_grid(*q_heads, rows)?, [1024, 1, 1])),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            gqa_factor.arg(),
            n.arg(),
            k_head_stride.arg(),
            k_seq_stride.arg(),
            v_head_stride.arg(),
            v_seq_stride.arg(),
            scale.arg(),
            window.arg(),
            q_row_stride.arg(),
            o_row_stride.arg(),
            sinks.arg(),
        ],
    )
}

/// The `Gate` family, claimed. One point, one kernel, and the kernel is
/// filed here rather than beside the experts for the reason the declaration
/// gives: no expert route comes near it, and every plane puts this one
/// beside its attention.
#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    fn sigmoid_mul<T: Scalar>(
        &self,
        x: InOut<Handle<T>>,
        gate: In<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`gate.sigmoid_mul`, at an element this plane does not stamp";
        let x = plane::in_place::<T, bf16>(x, WHAT)?;
        // `row_stride` IS THE ROW WIDTH, because a mark carries a dense
        // rectangle and nothing else — `attn_gate` reads `tid.y *
        // row_stride` when the number is positive and `tid.y * grid.x`
        // when it is zero, and the grid's x IS this width, so the two
        // readings are the same arithmetic. Stating it is what the
        // strideless-mark law says an executor does.
        //
        self.fire(
            Fire::at("attn/gate.metal", "gate_bfloat16")
                .apply(Grid::of(elementwise_rows(x.width, x.rows)?, [256, 1, 1])),
            &[
                x.arg(),
                plane::input::<T, bf16>(gate, WHAT)?.arg(),
                x.width.arg(),
            ],
        )
    }
}

/// A pool row's head geometry, read off the strides it was laid out with.
///
/// AN APPENDED PLANE CARRIES NO HEAD SEAM. A statement hands the appender one
/// rectangle per plane — `[tokens, kv_heads * head_dim]` — and nothing in it
/// says where one head ends. The POOL knows, and `attn/kv_write.metal` reads
/// its answer the one way it is laid out here: `kv_append_paged` computes a
/// page row as `n_kv_heads * head_dim` and steps a head by `head_dim`, which
/// is NHD, so the pool's head stride IS the head width and its sequence
/// stride IS the page row.
///
/// BOTH ARE CHECKED, and the check is the point. `kernels-cuda`'s `head_split`
/// picks between `seq_stride` and `head_stride` on a `layout` flag its view
/// carries; this view carries no flag because this plane's appender has no
/// second layout to pick. So the reading is pinned, and the pool is asked to
/// agree with it: a row that does not divide by the head stride, or a
/// sequence stride that is not the product, is a pool laid out for an
/// arithmetic this kernel does not perform, and firing into it would write
/// every head onto the same row.
fn head_split(view: &crate::views::PagedKvView, row: i32) -> Result<(i32, i32), Refusal> {
    let head_dim = i32::try_from(view.head_stride.0).map_err(|_| Refusal::Wide {
        what: "the head width this pool row's strides spell",
        at: i64::try_from(view.head_stride.0).unwrap_or(i64::MAX),
        max: i64::from(i32::MAX),
    })?;
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width this pool row's strides spell",
        });
    }
    if row <= 0 || row % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the appended row does not divide by the pool's head stride",
            at: i64::from(row),
        });
    }
    let heads = row / head_dim;
    if view.seq_stride.0 != u64::from(heads.unsigned_abs()) * u64::from(head_dim.unsigned_abs()) {
        return Err(Refusal::Narrow {
            what: "the pool's sequence stride is not the page row this appender writes",
            at: i64::try_from(view.seq_stride.0).unwrap_or(i64::MAX),
        });
    }
    Ok((head_dim, heads))
}

/// The paged append, from the pool row and the two planes alone.
///
/// ONE BODY, TWO POINTS: `attention.kv_append` states a key plane and a value
/// plane, `attention.kv_append_shared` states ONE plane that is both, and the
/// second is the first with the same handle twice. The alias is safe for the
/// reason `kernels-cuda`'s `kv_append_shared` gives at length — the kernel
/// reads both source planes and writes two DISTINCT destinations, so two
/// read-only bindings to one buffer is the legal reading of it — and it is
/// the shipped arithmetic rather than a new one: dsv4's text appends one
/// latent plane to both halves of its pool.
fn append_paged(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    view: &crate::views::PagedKvView,
    what: &'static str,
) -> Result<(), Refusal> {
    // ZERO IS NOT A PAGE SIZE, and here it is not a harmless one either. This
    // write's grid is heads by tokens and never consults the number, so a
    // store with no pool behind it would plan a full write in which every
    // token divides to page zero, offset zero — every layer landing on one
    // row, with no refusal anywhere.
    if view.page_size <= 0 {
        return Err(Refusal::Empty {
            what: "the KV page size",
        });
    }
    if v_new.width != k_new.width || v_new.rows != k_new.rows {
        return Err(Refusal::Narrow { what, at: i64::from(v_new.width) });
    }
    let (head_dim, heads) = head_split(view, k_new.width)?;
    let lanes = head_grid(head_dim, heads, k_new.rows)?;
    ctx.fire(
        Fire::at("attn/kv_write.metal", "kv_append_paged_bfloat16")
            .apply(Grid::of(lanes, head_group(lanes))),
        &[
            k_new.arg(),
            v_new.arg(),
            view.keys.arg_mut(),
            view.values.arg_mut(),
            // Buffers 4, 6-9 and 11 belong to the shared ring/read ABI; this
            // kernel names only the physical destination below.
            ctx.absent()?,
            head_dim.arg(),
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            view.page_size.arg(),
            ctx.absent()?,
            heads.arg(),
            view.write_page.arg(),
            view.write_offset.arg(),
            // THE SOURCE ROW IS PACKED, which is what a dense mark means:
            // zero tells the kernel to stride by `n_kv_heads * head_dim`.
            0_i32.arg(),
        ],
    )
}

/// The pool row a `Cache` mark carries, dereferenced once.
fn pages_of<'a>(
    pages: kernels::routine::Cache<Struct<KvCache>>,
) -> Result<&'a crate::views::PagedKvView, Refusal> {
    let row = pages.raised();
    if row.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    Ok(unsafe { &*row.ptr })
}

/// The `Attention` family, claimed. Three of eleven points land, and the
/// eight absences are this migration's most deliberate: THE SDPA CORE IS
/// CLAIM-ONLY BY DESIGN, exactly as it is on cuda and for a seam of this
/// plane's own shape.
///
/// * `attention.decode` / `attention.prefill` / `attention.masked` — SEAM:
///   THREE OPERANDS THE DECLARATION DOES NOT CARRY. Every `sdpa_paged_*` arm
///   takes `positions`, `request_of_token` AND `maskv: In<Struct<AttnMask>>`
///   beside the query and the pool row, because a paged read walks the CSR
///   per token and consults the custom mask per `(q, kv)` pair. A statement
///   carries the query, the page row and three numbers; nothing declared can
///   conjure a position stream, and a body that reached for one would be
///   staging on the operand column's behalf. `sdpa_vector_decode` needs
///   neither — and is not the answer either: it reads the cache as one
///   CONTIGUOUS slab by strides, with no page indirection at all, which is
///   not the pool this point states.
/// * `attention.decode_lse` / `attention.prefill_lse` — the same seam and one
///   more: SEAM: no `.metal` attention writes a log-sum-exp plane. The
///   online-softmax state lives and dies inside `sdpa_online.h`.
/// * `attention.sink` — SEAM: this plane folds sinks INSIDE the attention
///   (`sdpa_paged_decode_sink`, `sdpa_vector_decode_sink` take a `sinks`
///   bank), where the point states the POST-HOC correction against an already
///   written output and its LSE. Cuda's `attn_sink_correction` is the shape
///   this wants, and without an LSE plane there is nothing for it to correct
///   against.
/// * `attention.merge_lse` / `attention.lse_ln` — SEAM: both operate on an
///   LSE plane, and see above.
/// * `attention.kv_append_shared` LANDS, and it lands because its whole input
///   is the statement's: one plane, one pool row, and the head geometry read
///   off the strides the pool was laid out with. See [`append_paged`].
#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    /// `x = cap * tanh(x / cap)`, in place — gemma's final logit squash.
    ///
    /// The shader takes a separate destination and says it may alias; every
    /// thread writes `out[i]` from the same `i` it read, which is what
    /// [`crate::plane::read_half`] is about.
    fn logit_softcap<T: Scalar>(&self, x: InOut<Handle<T>>, cap: f32) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.logit_softcap`, at an element this plane does not stamp";
        let x = plane::in_place::<T, bf16>(x, WHAT)?;
        let n = x.rows.saturating_mul(x.width);
        self.fire(
            Fire::at("attn/logit_softcap.metal", "logit_softcap_bfloat16")
                .apply(Grid::of(elementwise(n, 1)?, [256, 1, 1])),
            &[
                plane::read_half(x).arg(),
                plane::write_half(x).arg(),
                cap.arg(),
            ],
        )
    }

    fn kv_append<T: Scalar>(
        &self,
        k: In<Handle<T>>,
        v: In<Handle<T>>,
        pages: kernels::routine::Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.kv_append`, at an element this plane does not stamp";
        append_paged(
            self,
            plane::input::<T, bf16>(k, WHAT)?,
            plane::input::<T, bf16>(v, WHAT)?,
            pages_of(pages)?,
            "the value plane, against the key plane it is appended beside",
        )
    }

    /// Leave dsv4's ONE plane in the pool row, as both halves of the read.
    fn kv_append_shared<T: Scalar>(
        &self,
        plane_: In<Handle<T>>,
        pages: kernels::routine::Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.kv_append_shared`, at an element this plane does not stamp";
        let shared = plane::input::<T, bf16>(plane_, WHAT)?;
        append_paged(
            self,
            shared,
            shared,
            pages_of(pages)?,
            "the shared plane, against itself",
        )
    }
}

/// The `Mla` family, implemented and claiming nothing.
///
/// SEAM: THIS PLANE HAS NO LATENT ATTENTION AT ALL. Eleven points — the two
/// latent cuts, the query split, the three absorbs, the append and the four
/// attentions — and the `.metal` tree stamps not one entrypoint that touches
/// a `kv_lora_rank`. The absorbs are grouped GEMMs against a `[heads,
/// kv_lora, nope]` bank, `mla.kv_append` writes a compressed row and its rope
/// tail into one pool, and the four attentions are the fa2 seam plus a
/// selection plane. Cuda carries all eleven; this is the family that has to
/// be written rather than crossed.
#[kernels_macros::claims]
impl kernels::points::Mla for Ctx<'_> {}

/// The `Index` family, implemented and claiming nothing.
///
/// SEAM: glm's sparse-attention indexer, and no `.metal` kernel for any of
/// it — the layernorm+rope over the index keys, the index query's own rope,
/// the top-k over a paged key plane, or the index append. `index.topk` is the
/// one the other plane calls unsolved too (its mask rows are a per-request kv
/// extent that sits in no slot); the other three are kernels this tree does
/// not have.
#[kernels_macros::claims]
impl kernels::points::Index for Ctx<'_> {}

/// The `Pool` family, implemented and claiming nothing.
///
/// SEAM: the pooled-attention ladder — two boundary computations, a gather, a
/// pooled append and a pooled attention with its LSE — and no `.metal` kernel
/// for any of the five. The LSE half of it is the same absence
/// `attention.decode_lse` names above.
#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {}
