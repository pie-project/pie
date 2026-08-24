#![allow(clippy::too_many_arguments)]

use crate::routine::{Bind, Ctx, Fire, In, InOut, Out, elementwise, elementwise_rows};
use crate::views::KvCache;
use kernels::BindMut;
use kernels::routine::Refusal;

/// The arm a head width picks, as an index into a schedule's own name list.
///
/// A width off the list is a `Refusal` and not a compile error: the arm is
/// chosen at the fire and the module was stamped at build time, so asking
/// for a width no `// pie:instantiate` line covers has to fail here or not
/// at all. The lists themselves are in the `Attention` block's header.
pub fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|d| *d == head_dim)
        .ok_or(Refusal::Narrow {
            what: "the head width",
            at: i64::from(head_dim),
        })
}

/// The VECTOR schedule's grid: one lane per channel, per (head, row).
///
/// `sdpa_paged.slang`'s unsplit and combine arms declare
/// `[numthreads(PIE_HEAD_DIM, 1, 1)]`, so a group IS the head width and the
/// x extent is that width times the head count. Rows ride y.
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

/// How many KV splits a decode of this shape should run: the flash-decode
/// policy, and the one host CHOICE in this file.
///
/// Every other `pub` fn here derives a grid from numbers a statement or a
/// shader already fixed. This one decides: it wants about
/// `TARGET_GROUPS` workgroups in flight, will not cut the key range finer
/// than `KEYS_PER_SPLIT` keys a split, caps at `MOST`, and floors the
/// answer to a power of two because the combine folds pairs. `1` means
/// unsplit and is the answer whenever the shape is already wide enough,
/// whenever the history is short, or whenever `PIE_NO_FLASH_DECODE` is set
/// — which is how a bisect turns the whole split path off.
///
/// ITS READER IS THE DRIVER, NOT A BODY HERE. `driver-vulkan` calls this
/// when it stages [`crate::views::SplitView`], and the claimed `decode`
/// reads the answer back off `splits`. So the policy is upstream of the
/// launch on purpose: the partials plane has to be allocated before the
/// fire that writes it.
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

/// The SPLIT schedule's grid: [`vector_grid`] with the KV splits on z.
///
/// Each split reads its own slice of the key range and writes one partial;
/// the combine that follows takes [`vector_grid`] because it folds them.
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

/// The TILED schedule's grid: one 32x32 workgroup per (head, 32-row tile).
///
/// THE ROW AXIS IS TILED BY 32 where the vector schedule takes rows one at
/// a time, which is the whole difference between the two — a tiled arm
/// walks the pool once for a block of queries. Both `sdpa_paged.slang`'s
/// tiled arms and `sdpa_paged_mma.slang` declare `[numthreads(32, 32, 1)]`,
/// so the two multiplies below are that group's two extents.
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

/// The per-head cut's grid: `x` walks a head's channels, `y` the heads,
/// `z` the token rows. What `kv_write.slang` and `gate.slang` address.
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

        // SEAM: the two per-fire token streams. The lowered path splices
        // each into the statement's input column and resolves it there; a
        // point declares no such column.
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
/// `decode`, `prefill`, `masked`, `logit_softcap` and `kv_append` fire
/// `sdpa_paged_decode_*`, `sdpa_paged_tiled_*`, `logit_softcap_bfloat16`
/// and `kv_append_paged_bfloat16`. Four of the five then need [`Fired`], which is where this plane's staging story
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
///   (`sdpa_paged_decode_sink_*`, `sdpa_paged_tiled_sink_*`,
///   `sdpa_paged_mma_sink_*`, `sdpa_vector_decode_sink_*`), taking the sink
///   bank as a trailing binding inside the attention. The point is the
///   POST-HOC correction: rescale an output against an lse a previous
///   reading left. With no lse there is nothing to rescale against, so the
///   fused entrypoints are unreachable from the declared points — they are
///   a tier-2 surface this plane has and the floor does not name.
///
///   THE SPLIT PATH FOLDS ITS SINK IN THE COMBINE, and it is a different
///   module rather than the same one with an extra binding: a split decode
///   under sinks fires `sdpa_paged_decode_split_bfloat16_d_64` and then
///   `sdpa_paged_decode_combine_sink_bfloat16_d_64`, because the sink term
///   belongs to the online softmax's final normalisation and each split
///   holds only part of it.
/// * `attention.kv_append_shared` — dsv4's one-plane append. Every
///   `kv_append` instantiation here writes a key plane and a value plane.
///
/// # The arms behind the seam, and the schedule each rides
///
/// This is the capital the routine layer was carrying: which head widths
/// each arm is stamped at, and which grid it takes. The launches themselves
/// were sixteen transcriptions of one staging — unwrap the pool row and the
/// mask view, divide `q_heads` by `n_kv_heads` for the GQA factor, bind the
/// mask triple and the window — and the two bodies below already write that
/// staging from what a statement carries. The table is what they could not
/// re-derive.
///
/// | arm (`_bfloat16_d_<w>`) | file | `<w>` | grid |
/// |---|---|---|---|
/// | `sdpa_paged_decode` | `attn/sdpa_paged.slang` | 64, 128, 256, 512 | [`vector_grid`] |
/// | `sdpa_paged_decode_split` | `attn/sdpa_paged.slang` | 64, 128, 256, 512 | [`split_grid`] |
/// | `sdpa_paged_decode_combine` | `attn/sdpa_paged.slang` | 64, 128, 256, 512 | [`vector_grid`] |
/// | `sdpa_paged_decode_sink` | `attn/sdpa_paged.slang` | 64 | [`vector_grid`] |
/// | `sdpa_paged_decode_combine_sink` | `attn/sdpa_paged.slang` | 64 | [`vector_grid`] |
/// | `sdpa_paged_tiled` | `attn/sdpa_paged.slang` | 64, 128, 256, 512 | [`tiled_grid`] |
/// | `sdpa_paged_tiled_sink` | `attn/sdpa_paged.slang` | 64 | [`tiled_grid`] |
/// | `sdpa_paged_tiled_strided` | `attn/sdpa_paged.slang` | 256 | [`tiled_grid`] |
/// | `sdpa_paged_mma` | `attn/sdpa_paged_mma.slang` | 64 | [`tiled_grid`] |
/// | `sdpa_paged_mma_sink` | `attn/sdpa_paged_mma.slang` | 64 | [`tiled_grid`] |
/// | `sdpa_vector_decode` | `attn/sdpa_vector.slang` | 64, 128, 256 | [`vector_grid`] |
/// | `sdpa_vector_decode_swa` | `attn/sdpa_sliding.slang` | 256, 512 | [`vector_grid`] |
/// | `sdpa_vector_decode_sink` | `attn/sdpa_sliding.slang` | 64 | [`vector_grid`] |
///
/// THE `mma` ARM IS NOT A TIER OF THE TILED ONE. `module_path` walks
/// `Capability::PREFERENCE` for ONE entrypoint name, stepping down to the
/// module actually stamped; `sdpa_paged_mma_bfloat16_d_64` is a SEPARATE
/// name whose shader uses cooperative matrices, so nothing in the tier walk
/// can reach it from `sdpa_paged_tiled_bfloat16_d_64`. Choosing it is a
/// capability test a body makes, and `Capability::Coopmat`'s `requires()`
/// is the four device features that make it legal.
///
/// The three `_p32` decode arms are stamped against a page size of 32 and
/// `PIE_FAST_FULL`; nothing has ever fired one from Rust, and a body that
/// wants them tests `kv.page_size` rather than a head width.
///
/// `sdpa_vector_decode*` reads the cache as one CONTIGUOUS slab through
/// `head_stride` and `seq_stride`, with no page indirection at all. That is
/// not the pool `attention.decode` states, and `driver-vulkan`'s `views::kv`
/// answers both strides as ZERO on a paged fire — so these three are not a
/// fallback for the paged arms, they are a different cache.
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

        // THE SPLIT READING, IN FULL. Each split reads its own slice of the
        // key range into `partials`; the combine folds them at
        // [`vector_grid`]. The two launches are short and the pair is the
        // arithmetic.
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
