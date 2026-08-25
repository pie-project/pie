#![allow(clippy::too_many_arguments)]

use crate::plane::{Bind, Ctx, Fire, In, InOut, Out, elementwise, elementwise_rows};
use crate::views::AttnFire;
use kernels::BindMut;
use kernels::plane::Refusal;

pub fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|d| *d == head_dim)
        .ok_or(Refusal::Narrow {
            what: "the head width",
            at: i64::from(head_dim),
        })
}

pub fn vector_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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

pub fn split_grid(
    head_dim: i32,
    q_heads: i32,
    rows: i32,
    splits: i32,
) -> Result<[u32; 3], Refusal> {
    if splits <= 0 {
        return Err(Refusal::Empty { what: "splits" });
    }
    let [x, y, _] = vector_grid(head_dim, q_heads, rows)?;
    Ok([x, y, splits.unsigned_abs()])
}

pub fn tiled_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
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

pub fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
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

#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    fn decode<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        pages: kernels::plane::Cache<kernels::raises::Struct<AttnFire>>,
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
        let fired = unsafe { &*pages.ptr };
        let kv = &fired.kv;
        let row = q.all("the query row")?;
        let hd = crate::points::stated("the head width this attention states", head_dim)?;
        let q_heads = crate::points::heads("the query heads this row divides into", row.width, hd)?;
        let w = crate::points::stated("the sliding extent this attention states", window)?;
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
                    crate::plane::module_path(entrypoint, self.best()),
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
            Fire::at(crate::plane::module_path(split, self.best()), split).apply(split_grid(
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
            Fire::at(crate::plane::module_path(combine, self.best()), combine)
                .apply(vector_grid(hd, q_heads, row.rows)?),
            &[
                o.arg(),
                fired.split.partials.arg_mut(),
                fired.split.splits.arg(),
            ],
        )
    }

    fn prefill<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        indptr: In<crate::points::Handle<i32>>,
        pages: kernels::plane::Cache<kernels::raises::Struct<AttnFire>>,
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

    fn masked<T: kernels::points::Scalar>(
        &self,
        q: In<crate::points::Handle<T>>,
        indptr: In<crate::points::Handle<i32>>,
        pages: kernels::plane::Cache<kernels::raises::Struct<AttnFire>>,
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
                crate::plane::module_path("logit_softcap_bfloat16", self.best()),
                "logit_softcap_bfloat16",
            )
            .apply(elementwise(row.elements(), 1)?),
            &[x.ptr.arg(), x.arg(), cap.arg()],
        )
    }

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        k: In<crate::points::Handle<T>>,
        v: In<crate::points::Handle<T>>,
        pages: kernels::plane::Cache<kernels::raises::Struct<AttnFire>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "attention.kv_append, at an element this plane does not instantiate",
        )?;
        if pages.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv pool row this statement names",
            });
        }
        let fired = unsafe { &*pages.ptr };
        let kv = &fired.kv;
        let row = k.all("the key rows this fire appends")?;

        let (kv_heads, head_dim) = (fired.kv_heads, fired.head_dim);
        if row.width != kv_heads.saturating_mul(head_dim) {
            return Err(Refusal::Narrow {
                what: "the appended key row, against the pool's head geometry",
                at: i64::from(row.width),
            });
        }
        self.fire(
            Fire::at(
                crate::plane::module_path("kv_append_paged_bfloat16", self.best()),
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

fn tiled<T: kernels::points::Scalar>(
    ctx: &Ctx<'_>,
    q: In<crate::points::Handle<T>>,
    pages: kernels::plane::Cache<kernels::raises::Struct<AttnFire>>,
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
    let fired = unsafe { &*pages.ptr };
    let kv = &fired.kv;
    let row = q.all("the query rows this window holds")?;
    let hd = crate::points::stated("the head width this attention states", head_dim)?;
    let q_heads = crate::points::heads("the query heads this row divides into", row.width, hd)?;
    let w = crate::points::stated("the sliding extent this attention states", window)?;
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
            crate::plane::module_path(entrypoint, ctx.best()),
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
                crate::plane::module_path("gate_bfloat16", self.best()),
                "gate_bfloat16",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[x.arg(), gate.arg(), row.width.arg()],
        )
    }
}
