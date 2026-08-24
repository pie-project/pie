use kernels::BindMut;

use crate::plane::{Bind, Ctx, Fire, In, InOut, Out};
use crate::points::{Payload, absent, at_bf16};
use crate::views::{AttnFire, AttnFireView};
use kernels::plane::{Cache, Refusal};
use kernels::raises::Struct;
use kernels::shader::{elementwise, elementwise_rows};

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

    if head_dim % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "the head width is not a whole number of bf16 pairs",
            at: i64::from(head_dim),
        });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(head_dim.unsigned_abs() / 2)
        .ok_or(Refusal::Grid {
            what: "query heads * the head width in pairs",
            at: i64::from(q_heads) * i64::from(head_dim) / 2,
        })?;
    Ok([x, rows.unsigned_abs(), 1])
}

fn paged_decode_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let g = vector_grid(head_dim, q_heads, rows)?;

    let keys = (512 / head_dim.unsigned_abs()).max(1);
    let y = g[1].checked_mul(keys).ok_or(Refusal::Grid {
        what: "rows * the decode key block",
        at: i64::from(g[1]) * i64::from(keys),
    })?;
    Ok([g[0], y, g[2]])
}

fn paged_split_grid(
    head_dim: i32,
    q_heads: i32,
    rows: i32,
    splits: i32,
) -> Result<[u32; 3], Refusal> {
    let g = paged_decode_grid(head_dim, q_heads, rows)?;
    if splits <= 0 {
        return Err(Refusal::Grid {
            what: "the decode splits",
            at: i64::from(splits),
        });
    }
    Ok([g[0], g[1], splits.unsigned_abs()])
}

fn paged_merge_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    vector_grid(head_dim, q_heads, rows)
}

fn mma_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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
        .checked_mul(8)
        .ok_or(Refusal::Grid {
            what: "rows rounded up to whole tiles",
            at: i64::from(rows),
        })?;
    Ok([x, y, 1])
}

const fn tiled_lanes(head_dim: i32) -> (u32, u32) {
    let pairs = head_dim.unsigned_abs() / 2;
    let tx = if pairs / 32 > 2 { pairs / 32 } else { 2 };
    let ty = if 256 / tx < 32 { 256 / tx } else { 32 };
    (tx, ty)
}

fn tiled_grid(q_heads: i32, head_dim: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let (tx, ty) = tiled_lanes(head_dim);
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
        .checked_mul(tx)
        .ok_or(Refusal::Grid {
            what: "query heads * the tile's lane count",
            at: i64::from(q_heads) * i64::from(tx),
        })?;

    let y = rows
        .unsigned_abs()
        .div_ceil(32)
        .checked_mul(ty)
        .ok_or(Refusal::Grid {
            what: "rows rounded up to whole tiles",
            at: i64::from(rows),
        })?;
    Ok([x, y, 1])
}

pub(crate) fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
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

fn fire_view(pages: &Cache<Struct<AttnFire>>) -> Result<&AttnFireView, Refusal> {
    if pages.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the attention view this statement's pool row names",
        });
    }
    Ok(unsafe { &*pages.ptr })
}

fn width_of(v: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

fn heads_of(width: i32, head_dim: i32, what: &'static str) -> Result<i32, Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width",
        });
    }
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

const fn gqa(q_heads: i32, kv_heads: i32) -> i32 {
    if kv_heads > 0 { q_heads / kv_heads } else { 0 }
}

fn paged_run(
    view: &AttnFireView,
    queries: crate::plane::ArgValue,
    out: crate::plane::ArgValue,
    gqa_factor: i32,
    kv_heads: i32,
    sm_scale: f32,
) -> [crate::plane::ArgValue; 15] {
    [
        queries,
        view.kv.keys.arg_mut(),
        view.kv.values.arg_mut(),
        out,
        gqa_factor.arg(),
        view.positions.arg(),
        view.request_of_token.arg(),
        view.kv.page_indices.arg(),
        view.kv.page_indptr.arg(),
        view.kv.page_size.arg(),
        kv_heads.arg(),
        sm_scale.arg(),
        view.mask.mask.arg(),
        view.mask.stride.arg(),
        view.mask.enabled.arg(),
    ]
}

#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    fn decode<T: kernels::points::Scalar>(
        &self,
        q: In<Payload<T>>,
        pages: Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.decode at an element other than bf16")?;
        let view = fire_view(&pages)?;
        let head_dim = width_of(head_dim, "the head width this attention states")?;
        let window = width_of(window, "the sliding window this attention states")?;
        let q_heads = heads_of(q.width, head_dim, "the query rectangle's row")?;
        let kv_heads = view.kv_heads;
        let rows = q.rows;
        let point = head_point(head_dim, &[64, 128, 256, 512])?;
        let run = paged_run(
            view,
            q.arg(),
            o.arg(),
            gqa(q_heads, kv_heads),
            kv_heads,
            sm_scale,
        );

        let workgroups = rows.saturating_mul(q_heads);
        let scratch = if workgroups < 128 && view.split.splits > 1 {
            Some(view.split.partials)
        } else {
            None
        };

        let Some(scratch) = scratch else {
            return self.fire(
                Fire::at(
                    "attn/sdpa_paged.wgsl",
                    [
                        "sdpa_paged_decode_bfloat16_d_64",
                        "sdpa_paged_decode_bfloat16_d_128",
                        "sdpa_paged_decode_bfloat16_d_256",
                        "sdpa_paged_decode_bfloat16_d_512",
                    ][point],
                )
                .apply(paged_decode_grid(head_dim, q_heads, rows)?),
                &[
                    run[0],
                    run[1],
                    run[2],
                    run[3],
                    run[4],
                    run[5],
                    run[6],
                    run[7],
                    run[8],
                    run[9],
                    run[10],
                    run[11],
                    run[12],
                    run[13],
                    run[14],
                    window.arg(),
                    absent(self)?,
                ],
            );
        };

        let splits = std::env::var("PIE_SPLITS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8);

        self.fire(
            Fire::at(
                "attn/sdpa_paged.wgsl",
                [
                    "sdpa_paged_decode_split_bfloat16_d_64",
                    "sdpa_paged_decode_split_bfloat16_d_128",
                    "sdpa_paged_decode_split_bfloat16_d_256",
                    "sdpa_paged_decode_split_bfloat16_d_512",
                ][point],
            )
            .apply(paged_split_grid(head_dim, q_heads, rows, splits)?),
            &[
                run[0],
                run[1],
                run[2],
                absent(self)?,
                run[4],
                run[5],
                run[6],
                run[7],
                run[8],
                run[9],
                run[10],
                run[11],
                run[12],
                run[13],
                run[14],
                window.arg(),
                absent(self)?,
                scratch.arg_mut(),
                splits.arg(),
            ],
        )?;

        self.fire(
            Fire::at(
                "attn/sdpa_paged.wgsl",
                [
                    "sdpa_paged_decode_merge_bfloat16_d_64",
                    "sdpa_paged_decode_merge_bfloat16_d_128",
                    "sdpa_paged_decode_merge_bfloat16_d_256",
                    "sdpa_paged_decode_merge_bfloat16_d_512",
                ][point],
            )
            .apply(paged_merge_grid(head_dim, q_heads, rows)?),
            &[
                absent(self)?,
                absent(self)?,
                absent(self)?,
                o.arg(),
                gqa(q_heads, kv_heads).arg(),
                absent(self)?,
                absent(self)?,
                absent(self)?,
                absent(self)?,
                view.kv.page_size.arg(),
                kv_heads.arg(),
                sm_scale.arg(),
                absent(self)?,
                view.mask.stride.arg(),
                absent(self)?,
                window.arg(),
                absent(self)?,
                scratch.arg_mut(),
                splits.arg(),
            ],
        )
    }

    fn prefill<T: kernels::points::Scalar>(
        &self,
        q: In<Payload<T>>,
        indptr: In<Payload<i32>>,
        pages: Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let _ = indptr;
        at_bf16::<T>("attention.prefill at an element other than bf16")?;
        let view = fire_view(&pages)?;
        let stated = width_of(kv_heads, "the kv head count this attention states")?;
        if stated != view.kv_heads {
            return Err(Refusal::Narrow {
                what: "the kv head count this statement states, against the pool row's own",
                at: i64::from(stated),
            });
        }
        tiled(self, q, o, view, window, head_dim, sm_scale)
    }

    fn masked<T: kernels::points::Scalar>(
        &self,
        q: In<Payload<T>>,
        indptr: In<Payload<i32>>,
        pages: Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let _ = indptr;
        at_bf16::<T>("attention.masked at an element other than bf16")?;
        let view = fire_view(&pages)?;
        tiled(self, q, o, view, window, head_dim, sm_scale)
    }

    fn logit_softcap<T: kernels::points::Scalar>(
        &self,
        x: InOut<Payload<T>>,
        cap: f32,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.logit_softcap at an element other than bf16")?;
        let n = x.rows.saturating_mul(x.width);
        self.fire(
            Fire::at("attn/logit_softcap.wgsl", "logit_softcap_bfloat16").apply(elementwise(n, 1)?),
            &[x.ptr.arg(), x.arg(), cap.arg()],
        )
    }

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        k: In<Payload<T>>,
        v: In<Payload<T>>,
        pages: Cache<Struct<AttnFire>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.kv_append at an element other than bf16")?;
        let view = fire_view(&pages)?;
        if view.kv.page_size <= 0 {
            return Err(Refusal::Empty {
                what: "the KV page size",
            });
        }
        let kv_heads = view.kv_heads;
        if kv_heads <= 0 {
            return Err(Refusal::Empty {
                what: "the kv head count this pool row was laid out with",
            });
        }
        if k.width <= 0 || k.width % kv_heads != 0 {
            return Err(Refusal::Narrow {
                what: "the appended key row, against the pool row's head count",
                at: i64::from(k.width),
            });
        }
        let head_dim = k.width / kv_heads;
        self.fire(
            Fire::at("attn/kv_write.wgsl", "kv_append_paged_bfloat16")
                .apply(head_grid(head_dim, kv_heads, k.rows)?),
            &[
                k.arg(),
                v.arg(),
                view.kv.keys.arg_mut(),
                view.kv.values.arg_mut(),
                head_dim.arg(),
                view.kv.page_size.arg(),
                kv_heads.arg(),
                view.kv.write_page.arg(),
                view.kv.write_offset.arg(),
            ],
        )
    }
}

fn tiled<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    q: In<Payload<T>>,
    o: Out<Payload<T>>,
    view: &AttnFireView,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    let head_dim = width_of(head_dim, "the head width this attention states")?;
    let window = width_of(window, "the sliding window this attention states")?;
    let q_heads = heads_of(q.width, head_dim, "the query rectangle's row")?;
    let kv_heads = view.kv_heads;
    let rows = q.rows;
    let run = paged_run(
        view,
        q.arg(),
        o.arg(),
        gqa(q_heads, kv_heads),
        kv_heads,
        sm_scale,
    );
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.wgsl",
            [
                "sdpa_paged_tiled_bfloat16_d_64",
                "sdpa_paged_tiled_bfloat16_d_128",
                "sdpa_paged_tiled_bfloat16_d_256",
                "sdpa_paged_tiled_bfloat16_d_512",
            ][head_point(head_dim, &[64, 128, 256, 512])?],
        )
        .apply(tiled_grid(q_heads, head_dim, rows)?),
        &[
            run[0],
            run[1],
            run[2],
            run[3],
            run[4],
            run[5],
            run[6],
            run[7],
            run[8],
            run[9],
            run[10],
            run[11],
            run[12],
            run[13],
            run[14],
            window.arg(),
            absent(ctx)?,
            rows.arg(),
        ],
    )
}

pub fn mma<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    q: In<Payload<T>>,
    o: Out<Payload<T>>,
    view: &AttnFireView,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    let head_dim = width_of(head_dim, "the head width this attention states")?;
    let window = width_of(window, "the sliding window this attention states")?;
    let q_heads = heads_of(q.width, head_dim, "the query rectangle's row")?;
    let kv_heads = view.kv_heads;
    let rows = q.rows;
    head_point(head_dim, &[64])?;
    let run = paged_run(
        view,
        q.arg(),
        o.arg(),
        gqa(q_heads, kv_heads),
        kv_heads,
        sm_scale,
    );
    ctx.fire(
        Fire::at("attn/sdpa_paged_mma.wgsl", "sdpa_paged_mma_bfloat16_d_64")
            .apply(mma_grid(q_heads, rows)?),
        &[
            run[0],
            run[1],
            run[2],
            run[3],
            run[4],
            run[5],
            run[6],
            run[7],
            run[8],
            run[9],
            run[10],
            run[11],
            run[12],
            run[13],
            run[14],
            window.arg(),
            absent(ctx)?,
            rows.arg(),
        ],
    )
}

#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    fn sigmoid_mul<T: kernels::points::Scalar>(
        &self,
        x: InOut<Payload<T>>,
        gate: In<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("gate.sigmoid_mul at an element other than bf16")?;
        let width = x.width;
        self.fire(
            Fire::at("attn/gate.wgsl", "gate_bfloat16").apply(elementwise_rows(width, x.rows)?),
            &[x.arg(), gate.arg(), width.arg()],
        )
    }
}
