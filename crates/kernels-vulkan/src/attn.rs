#![allow(clippy::too_many_arguments)]

use crate::routine::{
    Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use kernels::raises::Struct;
use crate::views::{AttnMask, KvCache, AttnSplit};
use kernels::BindMut;
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

#[routine(canon = split_qkv)]
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

#[routine(canon = sigmoid_gate_mul)]
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

#[routine(canon = split_q_gate)]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
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

#[routine(canon = kv_append)]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
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

#[routine]
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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
        return Err(Refusal::Null { what: "the split policy this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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
        return Err(Refusal::Null { what: "the split policy this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null { what: "the mask view this statement names" });
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

#[routine]
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
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
        return Err(Refusal::Null { what: "the kv view this statement names" });
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

