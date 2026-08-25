use kernels::BindMut;
use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::Scalar;

use crate::plane::{
    Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16, elementwise, elementwise_rows,
};
use crate::points::{self, Handle};
use crate::views::{AttnFire, AttnFireView};
use kernels::raises::Struct;

const SDPA_THREADS: u32 = 1024;

const SDPA_TILE: u32 = 32;

const SDPA_WIDTHS: [i32; 4] = [64, 128, 256, 512];

const SDPA_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

const SDPA_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

const SDPA_LSE_WIDTHS: [i32; 1] = [64];

const SDPA_DECODE_LSE: [&str; 1] = ["sdpa_paged_decode_lse_bfloat16_d_64"];

const SDPA_TILED_LSE: [&str; 1] = ["sdpa_paged_tiled_lse_bfloat16_d_64"];

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
    let x = heads.checked_mul(SDPA_THREADS).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * i64::from(SDPA_THREADS),
    })?;
    Ok([x, rows, 1])
}

fn tiled_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(SDPA_THREADS).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * i64::from(SDPA_THREADS),
    })?;
    Ok([x, rows.div_ceil(SDPA_TILE), 1])
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

#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    fn sigmoid_mul<T: Scalar>(
        &self,
        x: InOut<Handle<T>>,
        gate: In<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`gate.sigmoid_mul`, at an element this plane does not stamp";
        let x = points::in_place::<T, bf16>(x, WHAT)?;

        self.fire(
            Fire::at("attn/gate.metal", "gate_bfloat16")
                .apply(Grid::of(elementwise_rows(x.width, x.rows)?, [256, 1, 1])),
            &[
                x.arg(),
                points::input::<T, bf16>(gate, WHAT)?.arg(),
                x.width.arg(),
            ],
        )
    }
}

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

fn append_paged(
    ctx: &Ctx<'_>,
    k_new: In<Tensor<bf16>>,
    v_new: In<Tensor<bf16>>,
    view: &crate::views::PagedKvView,
    what: &'static str,
) -> Result<(), Refusal> {
    if view.page_size <= 0 {
        return Err(Refusal::Empty {
            what: "the KV page size",
        });
    }
    if v_new.width != k_new.width || v_new.rows != k_new.rows {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(v_new.width),
        });
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
            0_i32.arg(),
        ],
    )
}

fn pages_of<'a>(
    pages: kernels::plane::Cache<Struct<AttnFire>>,
) -> Result<&'a AttnFireView, Refusal> {
    let row = pages.raised();
    if row.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the attention view this statement names",
        });
    }
    Ok(unsafe { &*row.ptr })
}

fn pool_heads(view: &crate::views::PagedKvView, head_dim: i32) -> Result<i32, Refusal> {
    let stated = positive(head_dim, "the head width this attention states")?;
    if view.head_stride.0 != u64::from(stated) {
        return Err(Refusal::Narrow {
            what: "the head width this attention states, against the pool row's head stride",
            at: i64::from(head_dim),
        });
    }
    let seq = view.seq_stride.0;
    if seq == 0 || !seq.is_multiple_of(view.head_stride.0) {
        return Err(Refusal::Narrow {
            what: "the pool's sequence stride is not a whole number of kv heads",
            at: i64::try_from(seq).unwrap_or(i64::MAX),
        });
    }
    let heads = seq / view.head_stride.0;
    i32::try_from(heads).map_err(|_| Refusal::Wide {
        what: "the kv head count this pool row's strides spell",
        at: i64::try_from(heads).unwrap_or(i64::MAX),
        max: i64::from(i32::MAX),
    })
}

fn kv_heads_agree(view: &AttnFireView, head_dim: u32, kv_heads: u32) -> Result<(), Refusal> {
    let width = points::stated(head_dim, "the head width this attention states")?;
    let stated = points::stated(kv_heads, "the kv head count this attention states")?;
    if stated != pool_heads(&view.kv, width)? {
        return Err(Refusal::Narrow {
            what: "the kv head count this attention states, against the pool row's own",
            at: i64::from(stated),
        });
    }
    Ok(())
}

fn row_heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 || width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the query row does not divide by the head width this attention states",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

struct Paged {
    q_heads: i32,

    kv_heads: i32,

    gqa: i32,

    window: i32,

    rows: i32,

    head_dim: i32,

    at: usize,
}

impl Paged {
    fn of(
        q: In<Tensor<bf16>>,
        view: &AttnFireView,
        window: u32,
        head_dim: u32,
    ) -> Result<Self, Refusal> {
        if view.kv.page_size <= 0 {
            return Err(Refusal::Empty {
                what: "the KV page size",
            });
        }
        let head_dim = points::stated(head_dim, "the head width this attention states")?;
        let kv_heads = pool_heads(&view.kv, head_dim)?;
        let q_heads = row_heads(q.width, head_dim)?;
        if q_heads % kv_heads != 0 {
            return Err(Refusal::Narrow {
                what: "the query heads this row divides into, against the pool row's kv heads",
                at: i64::from(q_heads),
            });
        }
        Ok(Self {
            q_heads,
            kv_heads,
            gqa: q_heads / kv_heads,
            window: points::stated(window, "the sliding extent this attention states")?,
            rows: q.rows,
            head_dim,
            at: head_point(head_dim, &SDPA_WIDTHS)?,
        })
    }
}

/// The log-sum-exp plane a `_lse` reading writes beside its output: one f32
/// per query head per row, which is the shape the point declares.
fn lse_plane(
    lse: Out<Tensor<f32>>,
    shape: &Paged,
    what: &'static str,
) -> Result<Out<Tensor<f32>>, Refusal> {
    if lse.rows != shape.rows || lse.width != shape.q_heads {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(lse.width),
        });
    }
    Ok(lse)
}

/// The tiled reading, with or without the log-sum-exp plane beside its output.
///
/// Two entry points and one argument run: the `_lse` twin is the same
/// nineteen-wide binding with the plane appended, because the shader is the
/// same body and only the seat past `n_rows` is new.
#[allow(clippy::too_many_arguments)]
fn tiled(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    view: &AttnFireView,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    let shape = Paged::of(q, view, window, head_dim)?;
    let entry = match lse {
        None => SDPA_TILED[shape.at],
        Some(_) => SDPA_TILED_LSE[head_point(shape.head_dim, &SDPA_LSE_WIDTHS)?],
    };
    let mut args = vec![
        q.arg(),
        view.kv.keys.arg(),
        view.kv.values.arg(),
        o.arg(),
        shape.gqa.arg(),
        view.positions.arg(),
        view.request_of_token.arg(),
        view.kv.page_indices.arg(),
        view.kv.page_indptr.arg(),
        view.kv.page_size.arg(),
        shape.kv_heads.arg(),
        sm_scale.arg(),
        view.mask.mask.arg(),
        view.mask.stride.arg(),
        view.mask.enabled.arg(),
        shape.window.arg(),
        ctx.absent()?,
        shape.rows.arg(),
    ];
    if let Some(lse) = lse {
        args.push(lse_plane(lse, &shape, "`attention.prefill_lse`'s log-sum-exp plane")?.arg());
    }
    ctx.fire(
        Fire::at("attn/sdpa_paged.metal", entry).apply(Grid::of(
            tiled_grid(shape.q_heads, shape.rows)?,
            [SDPA_THREADS, 1, 1],
        )),
        &args,
    )
}

/// The per-row reading, with or without the log-sum-exp plane. See [`tiled`].
#[allow(clippy::too_many_arguments)]
fn vector(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    view: &AttnFireView,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    let shape = Paged::of(q, view, window, head_dim)?;
    let entry = match lse {
        None => SDPA_DECODE[shape.at],
        Some(_) => SDPA_DECODE_LSE[head_point(shape.head_dim, &SDPA_LSE_WIDTHS)?],
    };
    let mut args = vec![
        q.arg(),
        view.kv.keys.arg(),
        view.kv.values.arg(),
        o.arg(),
        shape.gqa.arg(),
        view.positions.arg(),
        view.request_of_token.arg(),
        view.kv.page_indices.arg(),
        view.kv.page_indptr.arg(),
        view.kv.page_size.arg(),
        shape.kv_heads.arg(),
        sm_scale.arg(),
        view.mask.mask.arg(),
        view.mask.stride.arg(),
        view.mask.enabled.arg(),
        shape.window.arg(),
        ctx.absent()?,
    ];
    if let Some(lse) = lse {
        args.push(lse_plane(lse, &shape, "`attention.decode_lse`'s log-sum-exp plane")?.arg());
    }
    ctx.fire(
        Fire::at("attn/sdpa_paged.metal", entry).apply(Grid::of(
            vector_grid(shape.q_heads, shape.rows)?,
            [SDPA_THREADS, 1, 1],
        )),
        &args,
    )
}

#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    fn decode<T: Scalar>(
        &self,
        q: In<Handle<T>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.decode`, at an element this plane does not stamp";
        vector(
            self,
            points::input::<T, bf16>(q, WHAT)?,
            points::result::<T, bf16>(o, WHAT)?,
            pages_of(pages)?,
            window,
            head_dim,
            sm_scale,
            None,
        )
    }

    fn decode_lse<T: Scalar>(
        &self,
        q: In<Handle<T>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Handle<T>>,
        lse: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.decode_lse`, at an element this plane does not stamp";
        vector(
            self,
            points::input::<T, bf16>(q, WHAT)?,
            points::result::<T, bf16>(o, WHAT)?,
            pages_of(pages)?,
            window,
            head_dim,
            sm_scale,
            Some(points::result::<f32, f32>(
                lse,
                "`attention.decode_lse`'s log-sum-exp plane, at an element other than f32",
            )?),
        )
    }

    fn prefill<T: Scalar>(
        &self,
        q: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.prefill`, at an element this plane does not stamp";
        let _ = indptr;
        let view = pages_of(pages)?;
        kv_heads_agree(view, head_dim, kv_heads)?;
        tiled(
            self,
            points::input::<T, bf16>(q, WHAT)?,
            points::result::<T, bf16>(o, WHAT)?,
            view,
            window,
            head_dim,
            sm_scale,
            None,
        )
    }

    fn prefill_lse<T: Scalar>(
        &self,
        q: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Handle<T>>,
        lse: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.prefill_lse`, at an element this plane does not stamp";
        let _ = indptr;
        let view = pages_of(pages)?;
        kv_heads_agree(view, head_dim, kv_heads)?;
        tiled(
            self,
            points::input::<T, bf16>(q, WHAT)?,
            points::result::<T, bf16>(o, WHAT)?,
            view,
            window,
            head_dim,
            sm_scale,
            Some(points::result::<f32, f32>(
                lse,
                "`attention.prefill_lse`'s log-sum-exp plane, at an element other than f32",
            )?),
        )
    }

    /// gpt-oss's learned per-head sink, applied to an output the softmax has
    /// already normalised, against the log-sum-exp that normalised it.
    ///
    /// The point states `lse` in BASE TWO and the sink in the checkpoint's own
    /// natural log, so the shader rebases one to meet the other; see
    /// `attn/attn_sink.metal`, which is the whole argument.
    fn sink<T: Scalar>(
        &self,
        o: InOut<Handle<T>>,
        lse: In<Handle<f32>>,
        sink: Const<Handle<T>>,
        head_dim: u32,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.sink`, at an element this plane does not stamp";
        let o = points::in_place::<T, bf16>(o, WHAT)?;
        let lse = points::input::<f32, f32>(
            lse,
            "`attention.sink`'s log-sum-exp plane, at an element other than f32",
        )?;
        let sinks = points::weight::<T, bf16>(sink, WHAT)?;
        let width = points::stated(head_dim, "the head width this sink states")?;
        let heads = row_heads(o.width, width)?;
        if lse.rows != o.rows || lse.width != heads {
            return Err(Refusal::Narrow {
                what: "the log-sum-exp plane this sink reads is not one f32 per head per row",
                at: i64::from(lse.width),
            });
        }
        let lanes = head_grid(width, heads, o.rows)?;
        self.fire(
            Fire::at("attn/attn_sink.metal", "attn_sink_rescale_bfloat16")
                .apply(Grid::of(lanes, head_group(lanes))),
            &[
                points::read_half(o).arg(),
                points::write_half(o).arg(),
                lse.arg(),
                sinks.arg(),
            ],
        )
    }

    fn masked<T: Scalar>(
        &self,
        q: In<Handle<T>>,
        indptr: In<Handle<i32>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.masked`, at an element this plane does not stamp";
        let _ = indptr;
        tiled(
            self,
            points::input::<T, bf16>(q, WHAT)?,
            points::result::<T, bf16>(o, WHAT)?,
            pages_of(pages)?,
            window,
            head_dim,
            sm_scale,
            None,
        )
    }

    fn logit_softcap<T: Scalar>(&self, x: InOut<Handle<T>>, cap: f32) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.logit_softcap`, at an element this plane does not stamp";
        let x = points::in_place::<T, bf16>(x, WHAT)?;
        let n = x.rows.saturating_mul(x.width);
        self.fire(
            Fire::at("attn/logit_softcap.metal", "logit_softcap_bfloat16")
                .apply(Grid::of(elementwise(n, 1)?, [256, 1, 1])),
            &[
                points::read_half(x).arg(),
                points::write_half(x).arg(),
                cap.arg(),
            ],
        )
    }

    fn kv_append<T: Scalar>(
        &self,
        k: In<Handle<T>>,
        v: In<Handle<T>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.kv_append`, at an element this plane does not stamp";
        append_paged(
            self,
            points::input::<T, bf16>(k, WHAT)?,
            points::input::<T, bf16>(v, WHAT)?,
            &pages_of(pages)?.kv,
            "the value plane, against the key plane it is appended beside",
        )
    }

    fn kv_append_shared<T: Scalar>(
        &self,
        plane_: In<Handle<T>>,
        pages: kernels::plane::Cache<Struct<AttnFire>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`attention.kv_append_shared`, at an element this plane does not stamp";
        let shared = points::input::<T, bf16>(plane_, WHAT)?;
        append_paged(
            self,
            shared,
            shared,
            &pages_of(pages)?.kv,
            "the shared plane, against itself",
        )
    }
}

#[kernels_macros::claims]
impl kernels::points::Mla for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Index for Ctx<'_> {}

#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {}
