#![allow(clippy::too_many_arguments)]

use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use crate::views::{AttnMask, AttnSplit, KvCache};
use kernels::BindMut;
use kernels::raises::Struct;
use kernels::routine::Refusal;
use kernels_macros::routine;

fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|d| *d == head_dim)
        .ok_or(Refusal::Narrow {
            what: "the head width",
            at: i64::from(head_dim),
        })
}

fn vector_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(head_dim.unsigned_abs())
        .ok_or(Refusal::Grid {
            what: "query heads * the head width",
            at: i64::from(q_heads) * i64::from(head_dim),
        })?;
    Ok([x, rows.unsigned_abs(), 1])
}

#[must_use]
pub fn decode_splits(history_bucket: i32, q_heads: i32, rows: i32) -> i32 {
    const TARGET_GROUPS: i64 = 2048;

    const KEYS_PER_SPLIT: i64 = 8;

    const MOST: i64 = 32;

    if history_bucket <= 0 || q_heads <= 0 || rows <= 0 {
        return 1;
    }

    static UNSPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *UNSPLIT.get_or_init(|| std::env::var_os("PIE_NO_FLASH_DECODE").is_some()) {
        return 1;
    }
    let base = i64::from(q_heads) * i64::from(rows);
    let want = (TARGET_GROUPS / base)
        .min(i64::from(history_bucket) / KEYS_PER_SPLIT)
        .min(MOST);
    if want < 2 {
        return 1;
    }

    1 << (63 - want.leading_zeros() as i64).min(30)
}

fn split_grid(head_dim: i32, q_heads: i32, rows: i32, splits: i32) -> Result<[u32; 3], Refusal> {
    if splits <= 0 {
        return Err(Refusal::Empty { what: "splits" });
    }
    let [x, y, _] = vector_grid(head_dim, q_heads, rows)?;
    Ok([x, y, splits.unsigned_abs()])
}

fn tiled_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(32)
        .ok_or(Refusal::Grid {
            what: "query heads * the tile's lane count",
            at: i64::from(q_heads) * 32,
        })?;
    let y = rows
        .unsigned_abs()
        .div_ceil(32)
        .checked_mul(32)
        .ok_or(Refusal::Grid {
            what: "rows rounded up to whole tiles",
            at: i64::from(rows),
        })?;
    Ok([x, y, 1])
}

fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width",
        });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if depth <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    Ok([
        head_dim.unsigned_abs(),
        heads.unsigned_abs(),
        depth.unsigned_abs(),
    ])
}

// INLINED into impl Layout; dies with the routine layer. (layout.split_qkv)
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
        Fire::at(
            crate::routine::module_path("split_qkv_bf16", ctx.best()),
            "split_qkv_bf16",
        )
        .apply(elementwise_rows(packed_width, rows)?),
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

// INLINED into impl Gate; dies with the routine layer. (gate.sigmoid_mul)
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
        Fire::at(
            crate::routine::module_path("gate_bfloat16", ctx.best()),
            "gate_bfloat16",
        )
        .apply(elementwise_rows(width, rows)?),
        &[attn.arg(), gate.arg(), row_stride.arg()],
    )
}

// INLINED into impl Layout; dies with the routine layer. (layout.split_q_gate)
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("q_gate_split_bfloat16", ctx.best()),
            "q_gate_split_bfloat16",
        )
        .apply(head_grid(*head_dim, *q_heads, rows)?),
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
    ctx.fire(
        Fire::at(
            crate::routine::module_path("kv_append_bfloat16", ctx.best()),
            "kv_append_bfloat16",
        )
        .apply(head_grid(*head_dim, *heads, 1)?),
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

// INLINED into impl Attention; dies with the routine layer. (attention.kv_append)
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

    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let w_page = kvc.write_page;
    let w_off = kvc.write_offset;
    let tokens = *tokens;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("kv_append_paged_bfloat16", ctx.best()),
            "kv_append_paged_bfloat16",
        )
        .apply(head_grid(*head_dim, *n_kv_heads, tokens)?),
        &[
            k_new.arg(),
            v_new.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            head_dim.arg(),
            page_size.arg(),
            n_kv_heads.arg(),
            w_page.arg(),
            w_off.arg(),
        ],
    )
}

// INLINED into impl Attention; dies with the routine layer. (attention.logit_softcap)
#[routine(out(out = like(logits)))]
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    cap: Const<f32>,
) -> Result<(), Refusal> {
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(
        Fire::at(
            crate::routine::module_path("logit_softcap_bfloat16", ctx.best()),
            "logit_softcap_bfloat16",
        )
        .apply(elementwise(n, 1)?),
        &[logits.arg(), out.arg(), cap.arg()],
    )
}

// INLINED into impl Attention; dies with the routine layer. (attention.decode)
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
    if split.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the split policy this statement names",
        });
    }
    let sv = unsafe { &*split.ptr };
    let partials = sv.partials;
    let rows = *rows;
    let splits = sv.splits;
    let at = head_point(*head_dim, &[64, 128, 256, 512])?;
    if splits <= 1 {
        return ctx.fire(
            Fire::at(
                crate::routine::module_path(
                    [
                        "sdpa_paged_decode_bfloat16_d_64",
                        "sdpa_paged_decode_bfloat16_d_128",
                        "sdpa_paged_decode_bfloat16_d_256",
                        "sdpa_paged_decode_bfloat16_d_512",
                    ][at],
                    ctx.best(),
                ),
                [
                    "sdpa_paged_decode_bfloat16_d_64",
                    "sdpa_paged_decode_bfloat16_d_128",
                    "sdpa_paged_decode_bfloat16_d_256",
                    "sdpa_paged_decode_bfloat16_d_512",
                ][at],
            )
            .apply(vector_grid(*head_dim, *q_heads, rows)?),
            &[
                queries.arg(),
                k_pages.arg_mut(),
                v_pages.arg_mut(),
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
            ],
        );
    }
    flash_decode(
        ctx,
        Flash {
            split: [
                "sdpa_paged_decode_split_bfloat16_d_64",
                "sdpa_paged_decode_split_bfloat16_d_128",
                "sdpa_paged_decode_split_bfloat16_d_256",
                "sdpa_paged_decode_split_bfloat16_d_512",
            ][at],
            combine: [
                "sdpa_paged_decode_combine_bfloat16_d_64",
                "sdpa_paged_decode_combine_bfloat16_d_128",
                "sdpa_paged_decode_combine_bfloat16_d_256",
                "sdpa_paged_decode_combine_bfloat16_d_512",
            ][at],
            sinks: None,
        },
        queries.ptr,
        k_pages,
        v_pages,
        out.ptr,
        gqa_factor,
        position_ids,
        req_of_token,
        kv_page_indices,
        kv_page_indptr,
        page_size,
        *n_kv_heads,
        *scale,
        attention_mask,
        attention_mask_stride,
        attention_mask_enabled,
        *window,
        partials,
        *head_dim,
        *q_heads,
        rows,
        splits,
    )
}

struct Flash {
    split: &'static str,
    combine: &'static str,
    sinks: Option<(Tensor<bf16>, &'static str)>,
}

fn flash_decode(
    ctx: &Ctx<'_>,
    which: Flash,
    queries: Tensor<bf16>,
    k_pages: Tensor<bf16>,
    v_pages: Tensor<bf16>,
    out: Tensor<bf16>,
    gqa_factor: i32,
    position_ids: Tensor<i32>,
    req_of_token: Tensor<i32>,
    kv_page_indices: Tensor<u32>,
    kv_page_indptr: Tensor<u32>,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: Tensor<u8>,
    attention_mask_stride: u32,
    attention_mask_enabled: Tensor<u8>,
    window: i32,
    partials: Tensor<f32>,
    head_dim: i32,
    q_heads: i32,
    rows: i32,
    splits: i32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            crate::routine::module_path(which.split, ctx.best()),
            which.split,
        )
        .apply(split_grid(head_dim, q_heads, rows, splits)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
            partials.arg_mut(),
        ],
    )?;

    let mut args = vec![out.arg_mut()];
    let entrypoint = match which.sinks {
        Some((sinks, module)) => {
            args.push(sinks.arg());
            module
        }
        None => which.combine,
    };
    args.push(partials.arg_mut());
    args.push(splits.arg());
    ctx.fire(
        Fire::at(
            crate::routine::module_path(entrypoint, ctx.best()),
            entrypoint,
        )
        .apply(vector_grid(head_dim, q_heads, rows)?),
        &args,
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
    if split.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the split policy this statement names",
        });
    }
    let sv = unsafe { &*split.ptr };
    let partials = sv.partials;
    let rows = *rows;
    let splits = sv.splits;
    head_point(*head_dim, &[64])?;
    if splits > 1 {
        return flash_decode(
            ctx,
            Flash {
                split: "sdpa_paged_decode_split_bfloat16_d_64",
                combine: "sdpa_paged_decode_combine_bfloat16_d_64",
                sinks: Some((*sinks, "sdpa_paged_decode_combine_sink_bfloat16_d_64")),
            },
            queries.ptr,
            k_pages,
            v_pages,
            out.ptr,
            gqa_factor,
            position_ids,
            req_of_token,
            kv_page_indices,
            kv_page_indptr,
            page_size,
            *n_kv_heads,
            *scale,
            attention_mask,
            attention_mask_stride,
            attention_mask_enabled,
            *window,
            partials,
            *head_dim,
            *q_heads,
            rows,
            splits,
        );
    }
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_paged_decode_sink_bfloat16_d_64", ctx.best()),
            "sdpa_paged_decode_sink_bfloat16_d_64",
        )
        .apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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

// INLINED into impl Attention; dies with the routine layer. (attention.prefill, attention.masked)
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
    let n_rows = *n_rows;
    ctx.fire(
        Fire::at(
            crate::routine::module_path(
                [
                    "sdpa_paged_tiled_bfloat16_d_64",
                    "sdpa_paged_tiled_bfloat16_d_128",
                    "sdpa_paged_tiled_bfloat16_d_256",
                    "sdpa_paged_tiled_bfloat16_d_512",
                ][head_point(*head_dim, &[64, 128, 256, 512])?],
                ctx.best(),
            ),
            [
                "sdpa_paged_tiled_bfloat16_d_64",
                "sdpa_paged_tiled_bfloat16_d_128",
                "sdpa_paged_tiled_bfloat16_d_256",
                "sdpa_paged_tiled_bfloat16_d_512",
            ][head_point(*head_dim, &[64, 128, 256, 512])?],
        )
        .apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_paged_tiled_sink_bfloat16_d_64", ctx.best()),
            "sdpa_paged_tiled_sink_bfloat16_d_64",
        )
        .apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
    let n_rows = *n_rows;

    let q_row_pitch = ctx.param(5)?;

    let o_row_pitch = ctx.param(6)?;
    head_point(*head_dim, &[256])?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_paged_tiled_strided_bfloat16_d_256", ctx.best()),
            "sdpa_paged_tiled_strided_bfloat16_d_256",
        )
        .apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
    let n_rows = *n_rows;
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_paged_mma_bfloat16_d_64", ctx.best()),
            "sdpa_paged_mma_bfloat16_d_64",
        )
        .apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_paged_mma_sink_bfloat16_d_64", ctx.best()),
            "sdpa_paged_mma_sink_bfloat16_d_64",
        )
        .apply(tiled_grid(*q_heads, n_rows)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
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
            crate::routine::module_path(
                [
                    "sdpa_vector_decode_bfloat16_d_64",
                    "sdpa_vector_decode_bfloat16_d_128",
                    "sdpa_vector_decode_bfloat16_d_256",
                ][head_point(*head_dim, &[64, 128, 256])?],
                ctx.best(),
            ),
            [
                "sdpa_vector_decode_bfloat16_d_64",
                "sdpa_vector_decode_bfloat16_d_128",
                "sdpa_vector_decode_bfloat16_d_256",
            ][head_point(*head_dim, &[64, 128, 256])?],
        )
        .apply(vector_grid(*head_dim, *q_heads, rows)?),
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
            crate::routine::module_path(
                [
                    "sdpa_vector_decode_swa_bfloat16_d_256",
                    "sdpa_vector_decode_swa_bfloat16_d_512",
                ][head_point(*head_dim, &[256, 512])?],
                ctx.best(),
            ),
            [
                "sdpa_vector_decode_swa_bfloat16_d_256",
                "sdpa_vector_decode_swa_bfloat16_d_512",
            ][head_point(*head_dim, &[256, 512])?],
        )
        .apply(vector_grid(*head_dim, *q_heads, rows)?),
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
    sinks: Const<Tensor<bf16>>,
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            crate::routine::module_path("sdpa_vector_decode_sink_bfloat16_d_64", ctx.best()),
            "sdpa_vector_decode_sink_bfloat16_d_64",
        )
        .apply(vector_grid(*head_dim, *q_heads, rows)?),
        &[
            queries.arg(),
            keys.arg(),
            values.arg(),
            out.arg(),
            sinks.arg(),
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

/// Everything a paged attention entrypoint on this plane reads that no
/// declaration carries.
///
/// SEAM, GATHERED IN ONE PLACE. `kernels::points`' `Attention` header says
/// what is not in the slots — "the decode and prefill plan caches, the
/// host mirrors of the two CSRs, the mask view. A body pulls those from
/// `self`" — and on cuda `self` is a struct that holds them. Here `self`
/// is `dyn Encode`, whose `resolve` answers a HANDLE BY COLUMN, so each of
/// the five below is a door `driver-vulkan` does not open yet. They are
/// resolved together so that a body reads one line and the seam has one
/// address.
struct Fired {
    positions: crate::points::Handle<i32>,
    request_of_token: crate::points::Handle<i32>,
    mask: crate::views::MaskView,
    split: crate::views::SplitView,
    /// The pool's KEY head count. The head WIDTH comes with it out of
    /// [`crate::points::pool_heads`] and is not kept: every attention
    /// entrypoint here is stamped per head width, so a body picks its
    /// module from the head width the STATEMENT states and checks the row
    /// against it — the pool's copy would be a second place for the same
    /// number.
    kv_heads: i32,
}

impl Fired {
    /// # Errors
    ///
    /// Whatever the missing door refuses with — see
    /// [`crate::points::Staged`] and [`crate::points::pool_heads`].
    fn of(ctx: &Ctx<'_>, kv: &crate::views::PagedKvView) -> Result<Self, Refusal> {
        use crate::points::Staged;

        // SEAM: the two per-fire token streams. A `#[routine]` takes each
        // as an ordinary `In<Tensor<i32>>` that the lowering splices into
        // the statement's input column; a point declares no such column.
        let positions = ctx.stream::<i32>("positions")?;
        let request_of_token = ctx.stream::<i32>("request_of_token")?;
        // SEAM: two residents that are not `Cache` slots. `AttnMask` is
        // this plane's custom-mask triple and `AttnSplit` is its decode
        // split policy — driver-owned, per fire, and named by no statement.
        let mask = unsafe { *ctx.resident::<crate::views::AttnMask>()? };
        let split = unsafe { *ctx.resident::<crate::views::AttnSplit>()? };
        // SEAM: the pool's head geometry.
        let (kv_heads, _head_dim) = crate::points::pool_heads(kv)?;
        Ok(Self {
            positions,
            request_of_token,
            mask,
            split,
            kv_heads,
        })
    }
}

/// The `Attention` family, claimed. Five of eleven points are written as
/// launchers; six are measured backlog rows, and every one of the six is
/// missing a KERNEL rather than plumbing.
///
/// # The five that land are one launch each, under five seams
///
/// `decode`, `prefill`, `masked`, `logit_softcap` and `kv_append` are
/// transcriptions of `sdpa_paged_decode`, `sdpa_paged_tiled`,
/// `logit_softcap` and `kv_append_paged` below, with the `Const<i32>` runs
/// those routines take replaced by what a bound statement carries. Four of
/// the five then need [`Fired`], which is where this plane's staging story
/// is honest: the mask, the split policy, the two token streams and the
/// pool's head geometry are all real objects the driver already builds and
/// none of them is reachable by NAME. `logit_softcap` needs none of it and
/// fires clean.
///
/// # `masked` and `prefill` are the same launch, and that is not a shortcut
///
/// `sdpa_paged_tiled` binds the mask triple unconditionally and the shader
/// tests `attention_mask_enabled` per row. So on this plane the difference
/// between the two points is which mask the DRIVER staged, not which
/// kernel runs — and the declaration agrees in advance: "the mask is the
/// plane's own staging and appears in no slot; what makes this a point of
/// its own is that the text states a different arithmetic". Two points,
/// one entrypoint, and the statement is what picks.
///
/// `prefill` states `kv_heads` and `decode` does not, which is the one
/// place the two bodies differ before the fire — and even there the stated
/// number is CHECKED against the pool rather than trusted, because the
/// pool is what the pages were allocated against.
///
/// # Five points stay on the floor's default body
///
/// * `attention.decode_lse`, `attention.prefill_lse` — no entrypoint here
///   writes a log-sum-exp. The split decode writes PARTIALS
///   (`[splits, ...]` accumulators the combine folds), which is a
///   different object with a different lifetime: nothing outside the pair
///   may read one.
/// * `attention.merge_lse` — the consumer of an lse, absent for the same
///   reason as the producers.
/// * `attention.sink` — this plane's sinks are FUSED
///   (`sdpa_paged_decode_sink`, `sdpa_paged_tiled_sink`,
///   `sdpa_paged_mma_sink`), taking the sink bank as a fifth binding
///   inside the attention. The point is the POST-HOC correction: rescale
///   an output against an lse a previous reading left. With no lse there
///   is nothing to rescale against, so the fused entrypoints are
///   unreachable from the declared points — they are a tier-2 surface this
///   plane has and the floor does not name.
/// * `attention.kv_append_shared` — dsv4's one-plane append. Every
///   `kv_append` instantiation here writes a key plane and a value plane.
#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    fn decode<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        pages: kernels::routine::Cache<kernels::raises::Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "attention.decode, at an element this plane does not instantiate",
        )?;
        if pages.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv pool row this statement names",
            });
        }
        let kv = unsafe { &*pages.ptr };
        let row = q.all("the query row")?;
        let hd = crate::points::stated("the head width this attention states", head_dim)?;
        let q_heads = crate::points::heads("the query heads this row divides into", row.width, hd)?;
        let w = crate::points::stated("the sliding extent this attention states", window)?;
        let fired = Fired::of(self, kv)?;
        let gqa = if fired.kv_heads > 0 {
            q_heads / fired.kv_heads
        } else {
            0
        };
        let at = head_point(hd, &[64, 128, 256, 512])?;

        if fired.split.splits <= 1 {
            let entrypoint = [
                "sdpa_paged_decode_bfloat16_d_64",
                "sdpa_paged_decode_bfloat16_d_128",
                "sdpa_paged_decode_bfloat16_d_256",
                "sdpa_paged_decode_bfloat16_d_512",
            ][at];
            return self.fire(
                Fire::at(
                    crate::routine::module_path(entrypoint, self.best()),
                    entrypoint,
                )
                .apply(vector_grid(hd, q_heads, row.rows)?),
                &[
                    q.arg(),
                    kv.keys.arg_mut(),
                    kv.values.arg_mut(),
                    o.arg(),
                    gqa.arg(),
                    fired.positions.arg(),
                    fired.request_of_token.arg(),
                    kv.page_indices.arg(),
                    kv.page_indptr.arg(),
                    kv.page_size.arg(),
                    fired.kv_heads.arg(),
                    sm_scale.arg(),
                    fired.mask.mask.arg(),
                    fired.mask.stride.arg(),
                    fired.mask.enabled.arg(),
                    w.arg(),
                ],
            );
        }

        // THE SPLIT READING, TRANSCRIBED RATHER THAN DELEGATED. `flash_decode`
        // below takes this plane's shader `Tensor<bf16>` in eleven positions;
        // a claim body holds `Handle<T>` marks, so wrapping it would mean a
        // conversion layer between two spellings of one descriptor index.
        // The two launches are short and the pair is the arithmetic.
        let split = [
            "sdpa_paged_decode_split_bfloat16_d_64",
            "sdpa_paged_decode_split_bfloat16_d_128",
            "sdpa_paged_decode_split_bfloat16_d_256",
            "sdpa_paged_decode_split_bfloat16_d_512",
        ][at];
        let combine = [
            "sdpa_paged_decode_combine_bfloat16_d_64",
            "sdpa_paged_decode_combine_bfloat16_d_128",
            "sdpa_paged_decode_combine_bfloat16_d_256",
            "sdpa_paged_decode_combine_bfloat16_d_512",
        ][at];
        self.fire(
            Fire::at(crate::routine::module_path(split, self.best()), split).apply(split_grid(
                hd,
                q_heads,
                row.rows,
                fired.split.splits,
            )?),
            &[
                q.arg(),
                kv.keys.arg_mut(),
                kv.values.arg_mut(),
                gqa.arg(),
                fired.positions.arg(),
                fired.request_of_token.arg(),
                kv.page_indices.arg(),
                kv.page_indptr.arg(),
                kv.page_size.arg(),
                fired.kv_heads.arg(),
                sm_scale.arg(),
                fired.mask.mask.arg(),
                fired.mask.stride.arg(),
                fired.mask.enabled.arg(),
                w.arg(),
                fired.split.partials.arg_mut(),
            ],
        )?;
        self.fire(
            Fire::at(crate::routine::module_path(combine, self.best()), combine)
                .apply(vector_grid(hd, q_heads, row.rows)?),
            &[
                o.arg(),
                fired.split.partials.arg_mut(),
                fired.split.splits.arg(),
            ],
        )
    }

    /// The prefill window.
    ///
    /// `indptr` IS UNSPENT, and the absence is the plane's rather than an
    /// oversight: `sdpa_paged_tiled` walks its rows through
    /// `request_of_token` and `position_ids` — a per-ROW request tag —
    /// where cuda's fa2 walks a CSR. Both say the same thing about the same
    /// fire; the declaration states the one every plane can be handed.
    fn prefill<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        indptr: In<crate::points::Handle<i32>>,
        pages: kernels::routine::Cache<kernels::raises::Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        let _ = indptr;
        tiled::<T>(
            self,
            q,
            pages,
            window,
            head_dim,
            Some(kv_heads),
            sm_scale,
            o,
            "attention.prefill",
        )
    }

    /// The prefill window under a custom `(q, kv)` mask.
    ///
    /// The same entrypoint as [`Self::prefill`]; see the impl header for
    /// why that is the honest reading on this plane and not a shortcut.
    /// `kv_heads` is not stated on this point, so it comes off the pool.
    fn masked<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        indptr: In<crate::points::Handle<i32>>,
        pages: kernels::routine::Cache<kernels::raises::Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        let _ = indptr;
        tiled::<T>(
            self,
            q,
            pages,
            window,
            head_dim,
            None,
            sm_scale,
            o,
            "attention.masked",
        )
    }

    /// `x = cap * tanh(x / cap)`, in place.
    ///
    /// THE ONE POINT OF THIS FAMILY THAT NEEDS NO SEAM. Every operand is
    /// declared, the geometry is the rectangle's own element count, and the
    /// entrypoint binds `logits` and `out_` — which are one handle here,
    /// this being elementwise 1:1 and an `InOut`.
    fn logit_softcap<T: kernels::points::Scalar>(
        &self,
        x: InOut<crate::points::Handle<T>>,
        cap: f32,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "attention.logit_softcap, at an element this plane does not instantiate",
        )?;
        let row = x.all("the capped rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("logit_softcap_bfloat16", self.best()),
                "logit_softcap_bfloat16",
            )
            .apply(elementwise(row.elements(), 1)?),
            // `logits` read-only, `out_` writable, one handle; see
            // `Norm::residual_add` for the spelling.
            &[x.ptr.arg(), x.arg(), cap.arg()],
        )
    }

    /// Leave this fire's keys and values in the pool row's pages.
    ///
    /// AN EFFECT AND NOT A RESULT — no `Out` slot, and the destination is
    /// the pool's own arithmetic: `write_page` and `write_offset` are per
    /// ROW of this fire and the view already carries them. What the view
    /// does NOT carry is the head geometry the grid needs, which is the
    /// seam [`crate::points::pool_heads`] names.
    fn kv_append<T: kernels::points::Scalar>(
        &self,
        k: In<crate::points::Handle<T>>,
        v: In<crate::points::Handle<T>>,
        pages: kernels::routine::Cache<kernels::raises::Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "attention.kv_append, at an element this plane does not instantiate",
        )?;
        if pages.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv pool row this statement names",
            });
        }
        let kv = unsafe { &*pages.ptr };
        let row = k.all("the key rows this fire appends")?;
        // SEAM: the pool's `(kv_heads, head_dim)`. The key row's width is
        // their product, so either one settles the other — and the
        // statement carries neither.
        let (kv_heads, head_dim) = crate::points::pool_heads(kv)?;
        if row.width != kv_heads.saturating_mul(head_dim) {
            return Err(Refusal::Narrow {
                what: "the appended key row, against the pool's head geometry",
                at: i64::from(row.width),
            });
        }
        self.fire(
            Fire::at(
                crate::routine::module_path("kv_append_paged_bfloat16", self.best()),
                "kv_append_paged_bfloat16",
            )
            .apply(head_grid(head_dim, kv_heads, row.rows)?),
            &[
                k.arg(),
                v.arg(),
                kv.keys.arg_mut(),
                kv.values.arg_mut(),
                head_dim.arg(),
                kv.page_size.arg(),
                kv_heads.arg(),
                kv.write_page.arg(),
                kv.write_offset.arg(),
            ],
        )
    }
}

/// The tiled paged attention, which is `attention.prefill` and
/// `attention.masked` both.
///
/// `stated` is the key-head count when the point carries one. It is
/// CHECKED against the pool rather than used in its place: the pages were
/// allocated against the pool's geometry, so a statement that disagrees is
/// a statement about a different cache.
fn tiled<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    q: In<crate::points::Handle<T>>,
    pages: kernels::routine::Cache<kernels::raises::Struct<KvCache>>,
    window: u32,
    head_dim: u32,
    stated_kv_heads: Option<u32>,
    sm_scale: f32,
    o: Out<crate::points::Handle<T>>,
    point: &'static str,
) -> Result<(), Refusal> {
    crate::points::at_bf16::<T>(match point {
        "attention.masked" => "attention.masked, at an element this plane does not instantiate",
        _ => "attention.prefill, at an element this plane does not instantiate",
    })?;
    if pages.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv pool row this statement names",
        });
    }
    let kv = unsafe { &*pages.ptr };
    let row = q.all("the query rows this window holds")?;
    let hd = crate::points::stated("the head width this attention states", head_dim)?;
    let q_heads = crate::points::heads("the query heads this row divides into", row.width, hd)?;
    let w = crate::points::stated("the sliding extent this attention states", window)?;
    let fired = Fired::of(ctx, kv)?;
    if let Some(stated) = stated_kv_heads {
        let stated = crate::points::stated("the key heads this attention states", stated)?;
        if stated != fired.kv_heads {
            return Err(Refusal::Narrow {
                what: "the key heads this attention states, against the pool it reads",
                at: i64::from(stated),
            });
        }
    }
    let gqa = if fired.kv_heads > 0 {
        q_heads / fired.kv_heads
    } else {
        0
    };
    let entrypoint = [
        "sdpa_paged_tiled_bfloat16_d_64",
        "sdpa_paged_tiled_bfloat16_d_128",
        "sdpa_paged_tiled_bfloat16_d_256",
        "sdpa_paged_tiled_bfloat16_d_512",
    ][head_point(hd, &[64, 128, 256, 512])?];
    ctx.fire(
        Fire::at(
            crate::routine::module_path(entrypoint, ctx.best()),
            entrypoint,
        )
        .apply(tiled_grid(q_heads, row.rows)?),
        &[
            q.arg(),
            kv.keys.arg_mut(),
            kv.values.arg_mut(),
            o.arg(),
            gqa.arg(),
            fired.positions.arg(),
            fired.request_of_token.arg(),
            kv.page_indices.arg(),
            kv.page_indptr.arg(),
            kv.page_size.arg(),
            fired.kv_heads.arg(),
            sm_scale.arg(),
            fired.mask.mask.arg(),
            fired.mask.stride.arg(),
            fired.mask.enabled.arg(),
            w.arg(),
            row.rows.arg(),
        ],
    )
}

/// The `Gate` family, claimed whole — one point, one launch.
///
/// Filed here and not in [`crate::moe`] because `kernels::points` says to:
/// "Not an MoE combine — no expert route comes near it — and its own
/// family for that reason. Every plane files this kernel beside its
/// attention", and `attn/gate.slang` is where this plane files it.
///
/// `row_stride` is the rectangle's own width, which is what an unstrided
/// statement means. The strided reading exists in `gate.slang` for a
/// caller that holds a slice of a wider projection, and no point states
/// one.
#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    fn sigmoid_mul<T: kernels::points::Scalar>(
        &self,
        x: InOut<crate::points::Handle<T>>,
        gate: In<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "gate.sigmoid_mul, at an element this plane does not instantiate",
        )?;
        let row = x.all("the gated rectangle's row width")?;
        self.fire(
            Fire::at(
                crate::routine::module_path("gate_bfloat16", self.best()),
                "gate_bfloat16",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[x.arg(), gate.arg(), row.width.arg()],
        )
    }
}
