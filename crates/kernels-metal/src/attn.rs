use kernels::BindMut;
use kernels::Grid;
use kernels::points::Scalar;
use kernels::routine::Refusal;

use crate::plane::{self, Handle};
use crate::routine::{
    Asks, Bind, Ctx, Fire, In, InOut, Tensor, bf16, elementwise, elementwise_rows,
};
use crate::views::KvCache;
use kernels::raises::Struct;

/// The arm a head width picks, as an index into the schedule's own name list.
///
/// A width off the list is a `Refusal` and not a compile error: the arm is
/// chosen at the fire and the shader is compiled at load, so asking for a
/// width no `.metal` instantiation covers has to fail here or not at all.
/// The lists themselves are in the `Attention` block's header.
pub fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|&p| p == head_dim)
        .ok_or(Refusal::Narrow {
            what: "a head width no shader is compiled for",
            at: i64::from(head_dim),
        })
}

/// The VECTOR schedule's grid: one 1024-wide threadgroup per (head, row).
///
/// Every lane of the group cooperates on one query row's whole reduction, so
/// the x extent is the head count times the group width and the y extent is
/// the rows themselves.
pub fn vector_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    let heads = positive(q_heads, "query heads")?;
    let rows = positive(rows, "rows")?;
    let x = heads.checked_mul(1024).ok_or(Refusal::Grid {
        what: "query heads * the threadgroup width",
        at: i64::from(heads) * 1024,
    })?;
    Ok([x, rows, 1])
}

/// The TILED schedule's grid: one threadgroup per (head, 32-row tile).
///
/// THE ROW AXIS IS TILED BY 32 where the vector schedule takes rows one at a
/// time, which is the whole difference between the two — a tiled arm walks
/// the pool once for a block of queries. `group` is the threadgroup width the
/// arm was compiled at: 1024 for `sdpa_paged_tiled*`, 128 for the simdgroup-
/// matrix arms in `attn/sdpa_paged_mma.metal`.
pub fn tiled_grid(q_heads: i32, rows: i32, group: u32) -> Result<[u32; 3], Refusal> {
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
///
/// # The ten arms the seam is holding back, and the schedule each rides
///
/// This is the capital the routine layer was carrying: which head widths each
/// arm is stamped at, and which grid it takes. The launches themselves were
/// ten transcriptions of one staging (unwrap the pool row and the mask view,
/// divide `q_heads` by `n_kv_heads` for the GQA factor, bind the mask triple
/// and the window) and a claim body will write that staging from the
/// declaration it finally gets; the table is what it could not re-derive.
///
/// | arm (`_bfloat16_d_<w>`) | file | `<w>` | grid |
/// |---|---|---|---|
/// | `sdpa_paged_decode` | `attn/sdpa_paged.metal` | 64, 128, 256, 512 | [`vector_grid`], group 1024 |
/// | `sdpa_paged_decode_sink` | `attn/sdpa_paged.metal` | 64 | [`vector_grid`], group 1024 |
/// | `sdpa_paged_tiled` | `attn/sdpa_paged.metal` | 64, 128, 256, 512 | [`tiled_grid`] at 1024 |
/// | `sdpa_paged_tiled_sink` | `attn/sdpa_paged.metal` | 64 | [`tiled_grid`] at 1024 |
/// | `sdpa_paged_tiled_strided` | `attn/sdpa_paged.metal` | 256 | [`tiled_grid`] at 1024 |
/// | `sdpa_paged_mma` | `attn/sdpa_paged_mma.metal` | 64 | [`tiled_grid`] at 128 |
/// | `sdpa_paged_mma_sink` | `attn/sdpa_paged_mma.metal` | 64 | [`tiled_grid`] at 128 |
/// | `sdpa_vector_decode` | `attn/sdpa_vector.metal` | 64, 128, 256 | [`vector_grid`], group 1024 |
/// | `sdpa_vector_decode_swa` | `attn/sdpa_sliding.metal` | 256, 512 | [`vector_grid`], group 1024 |
/// | `sdpa_vector_decode_sink` | `attn/sdpa_sliding.metal` | 64 | [`vector_grid`], group 1024 |
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
