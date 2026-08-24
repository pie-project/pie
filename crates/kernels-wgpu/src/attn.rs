use kernels::BindMut;
use kernels_macros::routine;

use crate::points::{Payload, absent, at_bf16};
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, InOut, Out, Tensor, bf16};
use crate::views::{AttnFire, AttnFireView, AttnMask, AttnSplit, KvCache};
use kernels::raises::Struct;
use kernels::routine::{Cache, Refusal};
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

/// `attn/sdpa_paged_mma.wgsl`'s grid, which is the tiled arm's SHAPE -- tiles
/// of 32 query rows on y, heads on x -- at a lane extent that shader cannot
/// move.
///
/// It was one function with `tiled_grid` while both shaders were
/// `@workgroup_size(32, 8)`, and the split is the news. `sdpa_paged.wgsl`'s
/// tiled arm has no workgroup memory and no barrier, so narrowing its x extent
/// is a local edit; the MMA body stages `k_tile`, `v_tile` and the segment's
/// queries, indexes them `ly * 32u + lx`, and measures a segment width against
/// its eight y lanes under barriers in uniform control flow. Its 32 and its 8
/// are load-bearing in a way the tiled arm's never were.
///
/// So this states them, and `geometry::Rule::SdpaMma` needs no copy of either
/// because it reads `module.local`.
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

/// The tiled arm's lane extents, which are `attn/sdpa_paged.wgsl`'s
/// `PIE_TX` and `PIE_TY` and must be restated here because a `Fire` divides
/// what `apply` is given by the module's own `@workgroup_size` and this
/// function has no module to ask.
///
/// `PIE_TX` is how many lanes share one query row on the channel axis, and it
/// is the REDUNDANCY on `dot_page` -- every one of them walks the whole key
/// history computing the same scalar. It floors at 2, the optimum the sweep
/// in the shader's own doc measured, and rises only where a lane would
/// otherwise carry more than 32 `vec2<f32>` accumulators. `PIE_TY` is the row
/// axis, the smaller of 32 (the tile) and what 256 invocations leave.
///
/// **These two lines must say what the shader's `const PIE_TX` and `const
/// PIE_TY` say.** A `//#define` cannot carry them -- the shader's `const` is
/// what the module publishes, and this is a different crate's arithmetic --
/// and the failure when they disagree is silent and catastrophic: `apply`
/// hands LANES, a `Fire` divides by the module's real `@workgroup_size`, so a
/// host saying 2 against a shader saying 8 dispatches a QUARTER of the query
/// heads and leaves the rest of the attention unwritten. That is exactly what
/// happened while this said 2 and the shader said 8, and `arena`'s workgroup
/// census is what said so: 16,332,666 against 16,338,066, which is 32 query
/// heads becoming 8 in every tiled prefill of every plan.
///
/// `driver-wgpu`'s `geometry::Rule::SdpaTiled` needs no copy of either
/// because it reads `module.local`, which is why the driver plane stayed
/// correct throughout and only this one did not.
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

    // 32 is the TILE -- the rows one group covers, which the shader's `rr <
    // 32u` states and which does not move with the lane extents -- and `ty` is
    // how many lanes sweep it.
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
        Fire::at("attn/split_qkv.wgsl", "split_qkv_bf16")
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
        Fire::at("attn/gate.wgsl", "gate_bfloat16").apply(elementwise_rows(width, rows)?),
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
    ctx.fire(
        Fire::at("attn/gate.wgsl", "q_gate_split_bfloat16")
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
        Fire::at("attn/kv_write.wgsl", "kv_append_bfloat16")
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
    // The metal twin's guard, for the metal twin's reason: this grid is heads
    // by tokens and never consults the page size, so a view built over a store
    // with no pool -- where the pooled numbers come back as zero -- plans a
    // full write in which every token divides to page zero, offset zero, and
    // every layer overwrites the one before it on a single row. See
    // `kernels_metal::attn::kv_append_paged`, where the same hole was found by
    // a driver test that had been asserting this refusal against a routine
    // unable to make it.
    if page_size <= 0 {
        return Err(Refusal::Empty {
            what: "the KV page size",
        });
    }

    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let w_page = kvc.write_page;
    let w_off = kvc.write_offset;
    let tokens = *tokens;
    ctx.fire(
        Fire::at("attn/kv_write.wgsl", "kv_append_paged_bfloat16").apply(head_grid(
            *head_dim,
            *n_kv_heads,
            tokens,
        )?),
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
        Fire::at("attn/logit_softcap.wgsl", "logit_softcap_bfloat16").apply(elementwise(n, 1)?),
        &[logits.arg(), out.arg(), cap.arg()],
    )
}

// INLINED into impl Attention; dies with the routine layer.
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

    let sinks = ctx.absent()?;
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
    let point = head_point(*head_dim, &[64, 128, 256, 512])?;

    let workgroups = rows.saturating_mul(*q_heads);
    // The split policy is the DRIVER's answer now: `splits <= 1` (or a
    // saturated device) fires the unsplit form, exactly what the optional
    // `keys::AttnScratch` ask used to decide by presence.
    let scratch = if workgroups < 128 && !split.ptr.is_null() {
        let sv = unsafe { &*split.ptr };
        (sv.splits > 1).then_some(sv.partials)
    } else {
        None
    };

    let Some(scratch) = scratch else {
        return ctx.fire(
            Fire::at(
                "attn/sdpa_paged.wgsl",
                [
                    "sdpa_paged_decode_bfloat16_d_64",
                    "sdpa_paged_decode_bfloat16_d_128",
                    "sdpa_paged_decode_bfloat16_d_256",
                    "sdpa_paged_decode_bfloat16_d_512",
                ][point],
            )
            .apply(paged_decode_grid(*head_dim, *q_heads, rows)?),
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
                sinks,
            ],
        );
    };

    // EIGHT, AND THE NUMBER BARELY MATTERS -- THE BRANCH ABOVE IT DOES.
    //
    // Swept on an M4 Pro against `what_a_decode_costs_at_length`, a
    // 512-key qwen3-0.6b decode:
    //
    // | splits | 4 | 8 | 16 | none (`scratch = None`) |
    // | --- | --- | --- | --- | --- |
    // | ms | 9.859 | 9.828 | 9.784 | **12.348** |
    // | tok/s | 101.4 | 101.8 | 102.2 | **81.0** |
    //
    // Four to sixteen is 0.8%, which is this harness's own repeatability,
    // so the count is on a flat and 8 is not a tuned number.
    //
    // RE-SWEPT AFTER `PIE_KR` WENT 8 -> 2, on the theory that halving the
    // workgroup memory would let more splits stay resident and move the
    // optimum. Six interleaved rounds of 8/16/32, means over the six:
    //
    // | splits | 8 | 16 | 32 |
    // | --- | --- | --- | --- |
    // | ms | **9.311** | 9.421 | 9.324 |
    //
    // Still flat, and still flat by more than the spread WITHIN a setting
    // (8 ran 9.187 to 9.475 across its six). The theory was wrong and the
    // constant stays. What the sweep did find is that the harness had lost
    // its repeatability -- see `what_a_decode_costs_at_length`, which now
    // takes 200 samples instead of 40 for exactly this reason. Taking the
    // OTHER branch -- one dispatch instead of a split and a merge -- is 26%
    // worse, and that is the result worth keeping.
    //
    // It is worth keeping because it contradicts the obvious arithmetic.
    // A decode's rectangles were believed to cost a flat ~19.6 us each
    // (`device.rs` explains why Metal serializes them), so the merge pass
    // looked like a pure 19.6 us a layer, 0.55 ms a token, that split-K has
    // to earn back.
    //
    // Both of those numbers were later measured marginally -- duplicate a
    // fire, read the increase -- and both were wrong. The merge costs 12.9
    // us, not 19.6, so split-K's toll is 0.36 ms a token rather than 0.55.
    // And the split itself costs **92.1 us**, which makes this pair 2.94 ms
    // of a 9.83 ms token: the single largest thing in a decode after the
    // weights themselves, and about 4x its own 0.64 ms of traffic. The
    // conclusion below still stands, but the arithmetic that made it
    // surprising was doubly generous. Reading
    // the attention's bytes -- about 1 MB a layer at 512 keys -- against
    // this part's bandwidth says the work inside is ~6 us, less than the
    // dispatch that carries it, and on that arithmetic splitting cannot pay.
    //
    // The arithmetic is wrong because the unsplit grid is `rows * q_heads` =
    // 16 workgroups, and 16 workgroups on a 20-core part is not a bandwidth
    // problem, it is a LATENCY one: each of the 16 walks all 512 keys in
    // series with nothing to overlap it. Splitting turns one 512-key walk
    // into eight 64-key walks that run at once. The extra dispatch costs
    // 0.55 ms a token and the parallelism returns 3.1 ms of it.
    //
    // So do not price a decode dispatch by its bytes, and do not assume
    // every decode rectangle is floor. Hardly any of the expensive ones
    // are, which is exactly why the `workgroups < 128` test above exists --
    // and why the split kernel, not the dispatch count, is where this
    // backend's remaining decode time sits.
    // # WHAT THE GPU'S OWN CLOCK SAYS ABOUT THIS KERNEL, AND WHAT IT DOES NOT
    //
    // `PIE_WGPU_STAMP` (see `driver-wgpu/src/device.rs`) times each launch with
    // a timestamp query instead of by subtraction. On a 512-key decode it
    // prices this dispatch at **98.4 us and 32.2% of the token's GPU time**,
    // the largest single kernel in a decode by a factor of one and a half over
    // the biggest `qmv`. The 92.1 us below was measured marginally, by
    // duplicating a fire and reading the increase; two instruments with nothing
    // in common agree to 7%, so that number is now as solid as anything here.
    //
    // AND THE LARGEST KERNEL TURNED OUT TO BE A LADDER. A third of that 98 us
    // was `log2(64) = 6` levels of workgroup-memory reduction, once per four
    // keys, and the `@subgroup` tier of `sdpa_paged.wgsl` replaces them with
    // register exchanges. **0.54 ms a token, 5.7%, three interleaved rounds,
    // every one a win**, and this kernel's share falls from 31% to 26%. The
    // file's own note had recorded the opposite -- "the barrier COUNT is not
    // the cost, what the tree costs is its ADDS" -- from a sound measurement
    // on llvmpipe, where a workgroup runs on one thread and a barrier is
    // nearly free. See that note for the correction and the arithmetic that
    // forced it.
    //
    // BOTH KNOBS WERE RE-SWEPT AFTERWARDS, because a kernel that went from
    // eight barriers a block to two is not the kernel either of them was
    // chosen against. Neither moved.
    //
    // `PIE_KR`, the keys a lane stages before reducing, two interleaved rounds
    // and a rebuild between every cell:
    //
    // ```text
    //   PIE_KR       1       2       4
    //   round 1    8.908   9.613   9.787   ms a token
    //   round 2    9.242   9.653   9.792
    // ```
    //
    // Monotone and steep, exactly as `sdpa_paged.wgsl`'s `PIE_KL` table found
    // before the ladder came out. That table's diagnosis was "hold less
    // state", and it is now confirmed to be about RESIDENCY rather than about
    // the barriers it was competing with: taking six barriers a block away did
    // not make deeper staging affordable.
    //
    // `PIE_SPLITS`, four interleaved rounds -- and this one changed SHAPE even
    // though it did not change VALUE:
    //
    // ```text
    //   splits       4       8      16      32
    //   round 1    9.311   9.160   9.060   9.298   ms a token
    //   round 2    9.321   9.120   8.966   9.348
    //   round 3            9.186   9.166
    //   round 4            9.082   9.196
    // ```
    //
    // The old sweep read flat from 4 to 32 and this one has a basin: 4 and 32
    // each cost about 2%, which is outside the ~1.7% repeatability, while 8
    // and 16 differ by 0.4% and TRADE PLACES between rounds 3 and 4. So 8 and
    // 16 are a tie and the shipped 8 stays -- but the flat is gone, which is
    // what a kernel whose per-workgroup serial work just shrank should look
    // like. Quoting 16 as a win off the first two rounds would have been the
    // fourth rule's whole point.
    //
    // THE SWEEP IT CANNOT DO, and this is worth as much as the number. The
    // same table, read across `PIE_SPLITS`, appears to say that four splits
    // costs 178 us where eight costs 112 -- 66 us x 28 layers = 1.85 ms, a
    // fifth of a token. Measured end to end, interleaved, 200 samples:
    //
    // | | 8 | 4 | 8 | 4 |
    // | --- | --- | --- | --- | --- |
    // | p50 ms | 9.626 | 9.685 | 9.622 | 9.716 |
    //
    // **0.7%, or 0.07 ms.** Twenty-six times smaller than the table implied,
    // and in agreement with the old 40-sample sweep below rather than with the
    // new instrument. The reason is composition, not the clock: `[cost]` sums
    // over a ONE-SECOND WINDOW, this bench's context grows from 512 to 717 keys
    // across its 205 decodes, and a slower setting advances fewer tokens per
    // window -- so the two columns are means over different key counts and are
    // not the same measurement.
    //
    // FIXED, AND THE FIX IS THE POINT: `PIE_WGPU_STAMP=<n>` cuts the report on
    // a FIRE COUNT instead of on wall time, so window `k` covers the same
    // tokens under every setting whatever each took to reach them. Re-swept at
    // 50 fires a window, the bench's own three decode windows, total ms of
    // split across 1400 launches apiece:
    //
    // | window | 1 | 2 | 3 |
    // | --- | --- | --- | --- |
    // | 8 splits | 122.0 | 147.3 | 155.9 |
    // | 4 splits | 132.4 | 148.4 | 160.6 |
    // | | +8.5% | +0.7% | +3.0% |
    //
    // Which is the 0.7% the end-to-end run measured, not the 1600% the
    // wall-clock window invented. The instrument now agrees with the harness
    // it contradicted, and that agreement is what makes it usable for the
    // kernels this bench cannot resolve at all.
    //
    // The same windows also show this kernel's share climbing 31.7% -> 33.8%
    // -> 35.2% as the context grows 512 -> 717, which is the growth the broken
    // window was aliasing against.
    //
    // So eight stays, and now on both numbers.
    //
    // RE-SWEPT A THIRD TIME, after a block stopped carrying `PIE_KB` keys and
    // started carrying `PIE_KB * PIE_DP` of them -- 4 becoming 8 at `d_128` --
    // because that is the quantity this knob divides the history into. Two
    // interleaved rounds, 200 samples:
    //
    // ```text
    //   splits       4       8      16
    //   round 1    7.515   7.451   7.645   ms a token
    //   round 2    7.608   7.445   7.596
    // ```
    //
    // Eight wins both rounds and the BASIN HAS SHARPENED: 16 was a tie a
    // sitting ago and now costs 2.2%, while 4 costs 1.6%. That is the shape a
    // doubled block should produce -- sixteen splits of a 512-key history is
    // 32 keys each, which is four blocks, and the tail of a split that rounds
    // up to whole blocks is a bigger fraction of it than it was. The knob has
    // not moved in three sweeps and its optimum is now the sharpest it has
    // been, which is the more useful half of the result.
    //
    // FOURTH SWEEP, on a token a third smaller than any of the three above and
    // with the stamping OFF -- a sweep run under `PIE_WGPU_PROBE` +
    // `PIE_WGPU_STAMP` read 13.03 / 13.56 / 13.98 / 14.07 ms at 4 / 8 / 16 /
    // 32 against a 7.5 ms baseline, monotone the wrong way and ~75% inflated,
    // and was discarded. Plain medians, three interleaved rounds:
    //
    // | PIE_SPLITS | round 1 | round 2 | round 3 | mean | cost |
    // | --- | --- | --- | --- | --- | --- |
    // | 4 | 7.509 | 7.498 | 7.472 | 7.493 | +0.7% |
    // | 8 | 7.453 | 7.442 | 7.415 | 7.437 | |
    // | 16 | 7.644 | 7.688 | 7.642 | 7.658 | +3.0% |
    //
    // Eight wins all three rounds with no overlap. Four sweeps, four times the
    // same answer, on tokens from 9.8 ms to 7.4 ms. This knob is closed.
    //
    // A PROBE KNOB, and eight is the shipped value. Read from the environment
    // rather than compiled in because `splits` is a runtime argument to the
    // kernel and a grid dimension, so nothing has to be rebuilt to move it --
    // which matters now that `PIE_WGPU_STAMP` can price this one kernel to the
    // microsecond instead of asking a 9.8 ms token to show a 0.8% difference.
    let splits = std::env::var("PIE_SPLITS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(8);

    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.wgsl",
            [
                "sdpa_paged_decode_split_bfloat16_d_64",
                "sdpa_paged_decode_split_bfloat16_d_128",
                "sdpa_paged_decode_split_bfloat16_d_256",
                "sdpa_paged_decode_split_bfloat16_d_512",
            ][point],
        )
        .apply(paged_split_grid(*head_dim, *q_heads, rows, splits)?),
        &[
            queries.arg(),
            k_pages.arg_mut(),
            v_pages.arg_mut(),
            ctx.absent()?,
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
            scratch.arg_mut(),
            splits.arg(),
        ],
    )?;

    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.wgsl",
            [
                "sdpa_paged_decode_merge_bfloat16_d_64",
                "sdpa_paged_decode_merge_bfloat16_d_128",
                "sdpa_paged_decode_merge_bfloat16_d_256",
                "sdpa_paged_decode_merge_bfloat16_d_512",
            ][point],
        )
        .apply(paged_merge_grid(*head_dim, *q_heads, rows)?),
        &[
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            out.arg(),
            gqa_factor.arg(),
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            page_size.arg(),
            n_kv_heads.arg(),
            scale.arg(),
            ctx.absent()?,
            attention_mask_stride.arg(),
            ctx.absent()?,
            window.arg(),
            ctx.absent()?,
            scratch.arg_mut(),
            splits.arg(),
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
    // The sink form fires unsplit on this plane; the policy is stated
    // for table equality and read by the decode form alone.
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.wgsl",
            "sdpa_paged_decode_sink_bfloat16_d_64",
        )
        .apply(paged_decode_grid(*head_dim, *q_heads, rows)?),
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

// INLINED into impl Attention; dies with the routine layer.
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

    let sinks = ctx.absent()?;
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
            "attn/sdpa_paged.wgsl",
            [
                "sdpa_paged_tiled_bfloat16_d_64",
                "sdpa_paged_tiled_bfloat16_d_128",
                "sdpa_paged_tiled_bfloat16_d_256",
                "sdpa_paged_tiled_bfloat16_d_512",
            ][head_point(*head_dim, &[64, 128, 256, 512])?],
        )
        .apply(tiled_grid(*q_heads, *head_dim, n_rows)?),
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged.wgsl",
            "sdpa_paged_tiled_sink_bfloat16_d_64",
        )
        .apply(tiled_grid(*q_heads, *head_dim, n_rows)?),
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

    let sinks = ctx.absent()?;
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
            "attn/sdpa_paged.wgsl",
            "sdpa_paged_tiled_strided_bfloat16_d_256",
        )
        .apply(tiled_grid(*q_heads, *head_dim, n_rows)?),
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

    let sinks = ctx.absent()?;
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
        Fire::at("attn/sdpa_paged_mma.wgsl", "sdpa_paged_mma_bfloat16_d_64")
            .apply(mma_grid(*q_heads, n_rows)?),
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
    head_point(*head_dim, &[64])?;
    ctx.fire(
        Fire::at(
            "attn/sdpa_paged_mma.wgsl",
            "sdpa_paged_mma_sink_bfloat16_d_64",
        )
        .apply(mma_grid(*q_heads, n_rows)?),
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
            "attn/sdpa_vector.wgsl",
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
            "attn/sdpa_sliding.wgsl",
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
            "attn/sdpa_sliding.wgsl",
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

/// The fire's attention view, or a refusal naming the pool row.
///
/// The one unsafe read in this file's points layer: `Cache<Struct<AttnFire>>`
/// carries `*const AttnFireView`, the object the driver built for THIS fire.
/// A raise has no shape, so the mark's `rows` and `width` are zero and the
/// pointer is the whole of it.
fn fire_view(pages: &Cache<Struct<AttnFire>>) -> Result<&AttnFireView, Refusal> {
    if pages.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the attention view this statement's pool row names",
        });
    }
    Ok(unsafe { &*pages.ptr })
}

/// A stated width, as the shader's `i32`.
fn width_of(v: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

/// The query heads a rectangle of `head_dim`-wide heads holds.
///
/// READ, NOT STATED. `attention.decode` declares `head_dim` and no head
/// count, so the count is the operand's own width over it — and a rectangle
/// whose width is not a whole number of heads is refused here rather than
/// silently truncated by the division.
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

/// `q_heads / kv_heads`, the GQA fan every sdpa entrypoint here takes.
///
/// Zero when the pool states no heads, which is the routine layer's reading
/// verbatim — the shader divides by it and a view built over a store with no
/// pool comes back all zeros.
const fn gqa(q_heads: i32, kv_heads: i32) -> i32 {
    if kv_heads > 0 { q_heads / kv_heads } else { 0 }
}

/// The paged sdpa arms' shared operand run, up to but not including the
/// window.
///
/// Fifteen values, in the order every `sdpa_paged*` entrypoint declares them.
/// It is written once because five bodies pass it and a body that reordered
/// two of them would read a mask stride as a page size with nothing to say
/// so — the failure `attn/sdpa_paged.wgsl`'s own header warns about, at the
/// one place a claim can still make it.
fn paged_run(
    view: &AttnFireView,
    queries: crate::routine::ArgValue,
    out: crate::routine::ArgValue,
    gqa_factor: i32,
    kv_heads: i32,
    sm_scale: f32,
) -> [crate::routine::ArgValue; 15] {
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

/// The `Attention` family, claimed. Five of eleven points land.
///
/// # What lands, and what a body had to reach for
///
/// `decode`, `prefill`, `masked`, `logit_softcap` and `kv_append`. Four of
/// those five read the fire's staging off [`AttnFireView`] — the positions,
/// the request-of-token map, the custom-mask triple and the decode split
/// policy — because a point declares operands and scalars only and this
/// plane's `Ctx` is a `dyn Encode` with nothing behind it. That view is this
/// migration's largest single ask of P5 and its doc comment states it.
///
/// # The five that do not land
///
/// * `decode_lse`, `prefill_lse`, `merge_lse`, `sink` — NO SHADER
///   ON THIS PLANE WRITES AN LSE. Every `sdpa_paged*` and `sdpa_vector*`
///   entrypoint here normalises inside the kernel and stores the attention
///   output alone; there is no second `Out` for the row's log-sum-exp and no
///   merge that takes two of them. That is why the sliding/full pair is
///   served by `sdpa_sliding.wgsl` as a WHOLE attention rather than by two
///   attentions and a merge, and it is why the sink correction is baked into
///   `sdpa_paged_decode_sink_bfloat16` instead of being applied after the
///   fact.
///
///   The four are ONE seam, not four: an `lse: Out<Tensor<f32>>` on the
///   paged arms would make `decode_lse`/`prefill_lse` fall out, and
///   `merge_lse`/`sink` are two small shaders on top of it. Until
///   then, a text that wants a sink states `attention.decode` and gets no
///   sink at all, which is why NEITHER is claimed: claiming `decode` with the
///   sinks buffer bound absent is what this plane already does, and it is
///   correct for every model without a sink and refuses for the ones with.
///
///   **SEAM (P5):** `sdpa_paged.wgsl` gains an `lse` binding under a
///   `PIE_LSE` define; `attn/sink.wgsl` and `attn/merge_lse.wgsl` are new
///   files, both reading the lse in the base `attention.decode_lse`
///   states. `Attention::sink`'s operands are all dense rectangles, so it
///   is the cheaper of the two to write.
/// * `kv_append_shared` — dsv4's single-plane append, which cuda claims by
///   binding one address into both source slots. Bindable here too (one
///   handle in two read-only bindings is the `residual_add` pattern), but
///   `attn/kv_write.wgsl`'s paged arm reads `n_kv_heads` and a head width
///   the SHARED statement does not carry, and this plane's view has no
///   `layout` flag to read the split off the strides with the way cuda's
///   `head_split` does. So it waits on the same view work `AttnFireView`
///   names, and dsv4 does not run here yet regardless (`Mla` is empty).
///
/// # Which prefill arm, and why the body cannot choose
///
/// This plane has THREE prefill shaders — `sdpa_paged_tiled`,
/// `sdpa_paged_mma` (cooperative matrix, 2.4x on an M4 Pro) and the strided
/// tiled form. `.wiki/baker.md` says per-fire kernel choice "branches inside
/// the body on the operands' dims", and the mma arm's condition is not a
/// dim: it is `Capability::Matrix`, a property of the ADAPTER. A `dyn Encode`
/// cannot be asked. So the claims below fire the tiled arm unconditionally
/// and the fast arm is unreachable through a point.
///
/// **SEAM (P5):** `Encode` grows `fn capability(&self) -> Capability`. Then
/// `prefill` reads it and takes `mma_grid` at `Capability::Matrix` with
/// `head_dim == 64`, exactly as `driver-wgpu`'s program builder picks today.
#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    /// One query row per request against the pool row's pages.
    ///
    /// # The split policy, transcribed with its measurement intact
    ///
    /// Below 128 workgroups a decode is LATENCY-bound rather than
    /// bandwidth-bound — sixteen workgroups each walking 512 keys in series
    /// with nothing to overlap them — so the fire splits the key history
    /// eight ways and merges the partials. Measured four times on an M4 Pro
    /// (`attn::sdpa_paged_decode`'s note carries all four sweeps): taking the
    /// OTHER branch costs 26%, and the split count itself has been flat-then-
    /// basined at eight through a `PIE_KR` change, a `PIE_KL` change and a
    /// doubled key block.
    ///
    /// The policy is the DRIVER's to publish and this body's to read:
    /// `SplitView::splits <= 1` (a saturated device, or a shell that does not
    /// want the scratch) fires the unsplit form, which is what the optional
    /// `keys::AttnScratch` ask used to decide by presence.
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

        // The unsplit reading, and the test is the WORKGROUP COUNT rather
        // than the token count: what makes a decode latency-bound is how few
        // groups its grid has, and `rows * q_heads` is that number.
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
                    // NO SINK. `attention.sink` is a point of its own and
                    // this plane does not claim it; a text that wants one
                    // states it and gets the family's refusal by name rather
                    // than an attention that quietly dropped it.
                    absent(self)?,
                ],
            );
        };

        // A PROBE KNOB, and eight is the shipped value — read from the
        // environment rather than compiled in because `splits` is a runtime
        // argument AND a grid dimension, so nothing has to be rebuilt to move
        // it. Four sweeps on tokens from 9.8 ms to 7.4 ms have answered
        // eight every time; see `attn::sdpa_paged_decode`.
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
                // The split arm writes PARTIALS, not the result: the output
                // slot is unbound and the scratch plane at the tail is where
                // the eight walks land.
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

        // The merge reads the partials and nothing else, so every operand it
        // does not touch is bound absent. The three it keeps — the gqa fan,
        // the mask stride and the window — are the row geometry it folds
        // over, not inputs it reads.
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

    /// The prefill window: tiles of 32 query rows, heads on x.
    ///
    /// # `indptr` is declared and this plane does not read it
    ///
    /// The point states the fire's query CSR because cuda's prefill needs it:
    /// its fa2 schedule is measured on the host from three CSR slices. This
    /// plane's tiled arm reconstructs the same window from `positions` and
    /// `request_of_token` — one position per token and the request it belongs
    /// to — and walks the page CSR per row. The two carry the same
    /// information and the shader was written against the second pair.
    ///
    /// So `indptr` is spent by being unread, and that is stated rather than
    /// hidden: a body that silently ignored a declared operand would be a
    /// statement whose dataflow edge means nothing. If the tiled arm ever
    /// grows a CSR fast path, the operand is already here.
    ///
    /// # `kv_heads` is stated AND read, and they must agree
    ///
    /// The point declares `kv_heads`; [`AttnFireView`] carries the pool's
    /// own. A disagreement means the plan and the pool were built from
    /// different layer facts, which is precisely the class of bug the
    /// stated-geometry law exists to catch, so it refuses here rather than
    /// attending over the wrong page stride.
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

    /// The prefill window under a CUSTOM mask.
    ///
    /// THE SAME ENTRYPOINT AS [`kernels::points::Attention::prefill`], and
    /// the point's own doc says why that is right: "what makes this a point
    /// of its own is that the text states a different arithmetic, not that it
    /// hands over a buffer". On this plane the mask triple is bound on EVERY
    /// sdpa arm — `attention_mask`, its stride and a per-request `enabled`
    /// byte plane — and the shader reads the mask for a row only where
    /// `enabled` says so. So the difference between the two points is which
    /// `enabled` the driver publishes, and the body is the same fire.
    ///
    /// `kv_heads` is NOT stated by this point, so it is the pool's alone —
    /// the one asymmetry between the masked and unmasked prefills, and it
    /// falls out of the declaration rather than out of anything here.
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

    /// `x = cap * tanh(x / cap)`, in place.
    ///
    /// Aliased like `norm.residual_add`: `attn/logit_softcap.wgsl` declares
    /// `logits` and `out_` and reads and writes the same index, so one handle
    /// fills both bindings.
    ///
    /// The grid is `elementwise(rows * width, 1)` and not
    /// `elementwise(width, rows)`, which is the routine's own reading: this
    /// shader has no row structure at all — its guard is
    /// `arrayLength(&out_)` — so the whole rectangle is one flat run.
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

    /// Write this fire's keys and values into the pool row's pages.
    ///
    /// AN EFFECT AND NOT A RESULT: no `Out` slot, and where the rows land is
    /// the pool's arithmetic — `write_page` and `write_offset`, one per
    /// token, published by the driver on the view.
    ///
    /// # The head geometry is the POOL's and the width is the operand's
    ///
    /// The statement carries neither a head count nor a head width, so the
    /// count comes off the view (the pool chose it when the slab was
    /// allocated) and the width is `k.width / kv_heads`. Read rather than
    /// stated for the reason cuda's `head_split` gives at length: the write
    /// has to agree with the layout the pages were allocated in, and a head
    /// count taken from a statement against a pool laid out for another is
    /// exactly the disagreement nothing else would catch.
    ///
    /// # The page size is refused at zero, and that refusal is load-bearing
    ///
    /// This grid is heads by tokens and never consults the page size, so a
    /// view built over a store with no pool — where the pooled numbers come
    /// back zero — would plan a full write in which every token divides to
    /// page zero, offset zero, and every layer overwrites the one before it
    /// on a single row. The metal twin found this by a driver test that had
    /// been asserting a refusal its routine could not make.
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

/// `sdpa_paged_tiled_bfloat16_d_*`, the arm both prefill points take.
///
/// The tile is 32 query rows and it does not move with the lane extents;
/// `tiled_grid` is the one place the host's copy of `PIE_TX`/`PIE_TY` lives,
/// and its doc records what happened the last time that copy disagreed with
/// the shader (32 query heads becoming 8 in every tiled prefill of every
/// plan, found by a workgroup census and by nothing else).
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
            // The tiled arm takes the row count as a SCALAR as well as
            // through the grid: its y extent is rows rounded up to whole
            // tiles, so the tail tile needs to know where the rows stop.
            rows.arg(),
        ],
    )
}

/// The `Gate` family: one point, claimed.
#[kernels_macros::claims]
impl kernels::points::Gate for Ctx<'_> {
    /// `x *= sigmoid(gate)`, qwen3.5's gated attention output.
    ///
    /// `attn/gate.wgsl`'s note is worth keeping beside the claim: its second
    /// buffer is the GATE and not the tensor, because the statement is in
    /// place on operand 0 and the tensor arrives twice — reading the first
    /// input for both would bind one handle at both slots. `driver-metal` and
    /// `driver-vulkan` both did until this backend's crossing compared the
    /// three arms. The ORDER is what has to match, and the declaration's
    /// order is `(x, gate)`.
    ///
    /// The sigmoid is MLX's stable form in f32, narrowed once at the store:
    /// the naive `1/(1+exp(-x))` overflows `exp` on a large negative argument
    /// before the quotient underflows.
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
