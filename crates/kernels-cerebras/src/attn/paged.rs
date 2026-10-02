//! Paged attention on one PE.
//!
//! Every query row carries its lane (`request_of_token`) and its position in
//! that lane's cache (`positions`). Key position `kp` of lane `l` lives in
//! pool row `page_indices[page_indptr[l] + kp / page_size] * page_size + kp %
//! page_size`. A row admits keys `[lo, hi)` with `hi = min(qpos + 1, cap)`
//! (the whole capacity when its enable flag is 2 or the walk is not causal)
//! and `lo = qpos - window + 1` under a window; an enabled custom mask gates
//! each key besides. The walk is an online softmax in f32: scores are `q · k
//! · sm_scale`, probabilities meet values as `acc = acc · α + p · v`. A row
//! that admits no key answers zeros and a `-inf` log-sum-exp (base 2).
//!
//! Reference: kernels-xla `attn::paged` (semantics), kernels-wgpu `attn::`
//! (signatures).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::{DecodePlan, PrefillPlan};
use crate::csl::Arg;
use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::program::{
    BankSplit, BlockPlan, ColPlane, LaneKind, LanePlan, PagePlan, PageSource, Reduce, Segment,
};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

fn tables(
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
) -> Result<(), Error> {
    expect(op, positions, &[Dtype::I32])?;
    expect(op, request_of_token, &[Dtype::I32])?;
    expect(op, mask, &[Dtype::U8])?;
    expect(op, mask_enabled, &[Dtype::U8])?;
    if positions.elements() != request_of_token.elements()
        || mask_enabled.elements() != positions.elements()
    {
        return Err(refuse(
            op,
            format!(
                "{} positions, {} lanes, {} enable flags",
                positions.elements(),
                request_of_token.elements(),
                mask_enabled.elements()
            ),
        ));
    }
    Ok(())
}

/// Validates the fire tables; the plan carries them into the attention.
/// `kv_len` is not read: the pool's CSR and the positions carry the lengths.
pub fn plan_decode(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<DecodePlan, Error> {
    let _ = (ctx, kv_len);
    tables(
        "attention.plan_decode",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(DecodePlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

pub fn plan_prefill(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<PrefillPlan, Error> {
    let _ = (ctx, kv_len);
    tables(
        "attention.plan_prefill",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(PrefillPlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

/// The per-row tables an attention reads, whichever plan they came from.
struct Tables {
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
}

impl From<&DecodePlan> for Tables {
    fn from(p: &DecodePlan) -> Self {
        Tables {
            positions: p.positions,
            request_of_token: p.request_of_token,
            mask: p.mask,
            mask_enabled: p.mask_enabled,
            mask_stride: p.mask_stride,
        }
    }
}

impl From<&PrefillPlan> for Tables {
    fn from(p: &PrefillPlan) -> Self {
        Tables {
            positions: p.positions,
            request_of_token: p.request_of_token,
            mask: p.mask,
            mask_enabled: p.mask_enabled,
            mask_stride: p.mask_stride,
        }
    }
}

fn pool_shape(op: &'static str, pool: &KvPool, head_dim: u32) -> Result<(u32, u32), Error> {
    expect(op, pool.keys, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, pool.values, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, pool.page_indices, &[Dtype::I32])?;
    expect(op, pool.page_indptr, &[Dtype::I32])?;
    if pool.head_stride != u64::from(head_dim)
        || !pool.seq_stride.is_multiple_of(u64::from(head_dim))
    {
        return Err(refuse(
            op,
            format!(
                "pool strides {}/{} do not describe {head_dim}-wide heads",
                pool.seq_stride, pool.head_stride
            ),
        ));
    }
    let kv_heads = (pool.seq_stride / u64::from(head_dim)) as u32;
    let width = kv_heads * head_dim;
    if pool.keys.width != width || pool.values.width != width {
        return Err(refuse(
            op,
            format!(
                "the pool planes are {} and {} wide, not {width}",
                pool.keys.width, pool.values.width
            ),
        ));
    }
    if pool.page_size <= 0 {
        return Err(refuse(op, format!("page size {}", pool.page_size)));
    }
    Ok((kv_heads, width))
}

/// How many PEs a paged pool of `elements` words over `lanes` lanes spreads
/// over, each lane holding up to `max_pages` pages of `page_words`: the
/// fewest that keep a PE's pages within `shard_words()`. `None` on one PE.
/// The most rows one attention phase takes, when capped: by
/// [`cap_rows_per_phase`] or `PIE_CEREBRAS_ATTN_ROWS` (tests run the whole
/// suite over row windows this way).
static ROWS_PER_PHASE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Caps the rows one attention phase takes (0 lifts the cap).
pub fn cap_rows_per_phase(rows: u32) {
    ROWS_PER_PHASE.store(rows, std::sync::atomic::Ordering::Relaxed);
}

fn rows_per_phase_cap() -> Option<u32> {
    let set = ROWS_PER_PHASE.load(std::sync::atomic::Ordering::Relaxed);
    if set > 0 {
        return Some(set);
    }
    std::env::var("PIE_CEREBRAS_ATTN_ROWS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
}

/// How a paged pool spreads over PEs.
#[derive(Clone)]
struct PageSplit {
    pes: u32,
    lanes_per_pe: u32,
    /// A lane's pages over this many PEs.
    page_groups: u32,
    /// The kv heads over this many PEs (a block along the banks' heads).
    head_groups: u32,
    /// A page's rows over this many PEs (a block along the pages' rows):
    /// each PE attends a strided subset of the keys, merged like page groups.
    row_groups: u32,
}

/// How a paged pool of `elements` over `lanes` lanes spreads over PEs, or
/// `None` when one PE holds it all (the pool plus `extra(1)`). A PE holds
/// as many lanes' pages as fit beside `extra(head_groups)` (the phase's
/// words besides pages, given the kv heads split that many ways); a
/// lane's pages past that split over page groups; a page past that splits
/// its kv heads. The fewest PEs win.
fn page_split(
    op: &'static str,
    elements: u64,
    lanes: u32,
    page_pair: u64,
    max_pages: u32,
    kv_heads: u32,
    page_rows: u32,
    extra: impl Fn(u32) -> u64,
) -> Result<Option<PageSplit>, Error> {
    use crate::linear::gemm::{ARRAY_WORDS, pe_words};
    if lanes == 0 || (elements + extra(1) <= pe_words() && elements <= ARRAY_WORDS) {
        return Ok(None);
    }
    let max_pages = max_pages.max(1);
    let mut best: Option<PageSplit> = None;
    let page_rows = page_rows.max(1);
    for hg in (1..=kv_heads.max(1)).filter(|g| kv_heads.max(1).is_multiple_of(*g)) {
        for ag in (1..=page_rows).filter(|g| page_rows.is_multiple_of(*g)) {
            // A bank plane is one array: its pages within ARRAY_WORDS too.
            let room = pe_words().saturating_sub(extra(hg)).min(2 * ARRAY_WORDS);
            let pair = (page_pair / u64::from(hg) / u64::from(ag)).max(1);
            let pages_per_pe = u32::try_from(room / pair).unwrap_or(u32::MAX);
            if pages_per_pe == 0 {
                continue;
            }
            let (lanes_per_pe, page_groups) = if pages_per_pe >= max_pages {
                ((pages_per_pe / max_pages).clamp(1, lanes), 1)
            } else {
                (1, max_pages.div_ceil(pages_per_pe))
            };
            let pes = lanes.div_ceil(lanes_per_pe) * page_groups * hg * ag;
            if best.as_ref().is_none_or(|b| pes < b.pes) {
                best = Some(PageSplit {
                    pes,
                    lanes_per_pe,
                    page_groups,
                    head_groups: hg,
                    row_groups: ag,
                });
            }
            break;
        }
    }
    let budget = pe_words();
    match best {
        Some(b) => Ok(Some(b)),
        None => Err(refuse(
            op,
            format!(
                "one page's {page_pair} words and the phase's other {} words exceed a PE's {budget} even one kv head and one page row at a time",
                extra(kv_heads.max(1))
            ),
        )),
    }
}

/// The most PEs one lane block's partials spread over for the fabric to
/// merge them (the root gathers that many lse copies).
pub const MERGE_GROUPS: u32 = 16;

/// Whether split attention merges its partials on the fabric
/// (`PIE_CEREBRAS_FABRIC_SUM=0` leaves the merge to the host).
pub fn fabric_sum() -> bool {
    crate::linear::gemm::fabric_sum()
}

/// Words a PE spends on the fabric merge of its `o` (`o_words`) and lse
/// (`l_words`) partials: the partial's copy, the gathered lse copies and
/// the merged lse, and the collectives library.
pub fn merge_words(o_words: u64, l_words: u64) -> u64 {
    o_words + l_words * (u64::from(MERGE_GROUPS) + 1) + crate::linear::gemm::FABRIC_SUM_WORDS
}

/// The fabric merge of a lane block's attention partials over `group`
/// consecutive PEs (one rectangle row of the `pes`): every PE's lse plane
/// (`l_words`) gathers on the row's first PE, which merges them as a
/// log-sum-exp and broadcasts the total; every PE then reweights its `o`
/// plane (`o_words`) by `2^(lse - total)` and the row adds those into the
/// first PE's `o`; the total becomes the lse.
pub fn lse_merge(
    cx: &mut Cx<'_>,
    o: &crate::program::Buf,
    lse: &crate::program::Buf,
    o_words: u64,
    l_words: u64,
    group: u32,
    pes: u32,
) {
    lse_merge_with(cx, o, lse, o_words, l_words, group, pes, false);
}

/// [`lse_merge`], then (`everywhere`) the merged `o` broadcast back over the
/// row so every PE holds it (the resident placement's whole activations).
#[allow(clippy::too_many_arguments)]
pub fn lse_merge_with(
    cx: &mut Cx<'_>,
    o: &crate::program::Buf,
    lse: &crate::program::Buf,
    o_words: u64,
    l_words: u64,
    group: u32,
    pes: u32,
    everywhere: bool,
) {
    use crate::program::{FabricOp, FabricStep, Guarded};
    let all = cx.scratch("lse_all", l_words * u64::from(group));
    let total = cx.scratch("lse_total", l_words);
    let partial = cx.scratch("partial", o_words);
    cx.library("k_lse_merge");
    cx.library("k_lse_scale");
    let d = o_words / l_words.max(1);
    let mut steps = vec![
        FabricStep {
            calls: Vec::new(),
            op: FabricOp::Gather {
                send: lse.name.clone(),
                recv: all.clone(),
                count: l_words,
            },
        },
        FabricStep {
            calls: vec![Guarded::root(
                "k_lse_merge",
                vec![
                    Arg::Scratch(all.clone(), "f32"),
                    Arg::Scratch(total.clone(), "f32"),
                    Arg::Int(i64::from(group)),
                    Arg::Int(l_words as i64),
                ],
            )],
            op: FabricOp::Broadcast {
                buf: total.clone(),
                count: l_words,
            },
        },
        FabricStep {
            calls: vec![
                Guarded::all(
                    "k_lse_scale",
                    vec![
                        Arg::Ptr(o.clone()),
                        Arg::Ptr(lse.clone()),
                        Arg::Scratch(total.clone(), "f32"),
                        Arg::Int(l_words as i64),
                        Arg::Int(d as i64),
                    ],
                ),
                Guarded::all(
                    "k_copy",
                    vec![
                        Arg::Scratch(partial.clone(), "f32"),
                        Arg::Int(0),
                        Arg::Ptr(o.clone()),
                        Arg::Int(0),
                        Arg::Int(o_words as i64),
                    ],
                ),
            ],
            op: FabricOp::Reduce {
                send: partial,
                recv: o.name.clone(),
                count: o_words,
            },
        },
    ];
    if everywhere {
        steps.push(FabricStep {
            calls: Vec::new(),
            op: FabricOp::Broadcast {
                buf: o.name.clone(),
                count: o_words,
            },
        });
    }
    let finish = vec![Guarded::all(
        "k_copy",
        vec![
            Arg::Ptr(lse.clone()),
            Arg::Int(0),
            Arg::Scratch(total.clone(), "f32"),
            Arg::Int(0),
            Arg::Int(l_words as i64),
        ],
    )];
    cx.fabric((group, pes / group), steps, finish);
}

/// The resident placement of a pool over a row of `pes` PEs: every PE runs
/// every lane over the pages it holds (page `g` on PE `g % pes`), `None`
/// when the share and the phase's other words do not fit a PE.
fn resident_split(pes: u32, pool_pages: u32, page_pair: u64, lanes: u32, extra: u64) -> Option<(PageSplit, u32)> {
    use crate::linear::gemm::{ARRAY_WORDS, pe_words};
    let share = pool_pages.div_ceil(pes).max(1);
    let words = u64::from(share) * page_pair;
    (words / 2 <= ARRAY_WORDS && words + extra <= pe_words()).then_some((
        PageSplit {
            pes,
            lanes_per_pe: lanes,
            page_groups: 1,
            head_groups: 1,
            row_groups: 1,
        },
        share,
    ))
}

/// Attends `rows` rows of `q` (`[rows, q_heads * head_dim]`) and writes `o`
/// (same shape) and, when given, `lse` (`[rows, q_heads]`, f32, base 2).
fn attend(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    t: &Tables,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    selection: Option<(Tensor, u32)>,
) -> Result<(), Error> {
    expect(op, q, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, o, &[Dtype::Bf16, Dtype::F32])?;
    tables(op, t.positions, t.request_of_token, t.mask, t.mask_enabled)?;
    if let Some((sel, ratio)) = selection {
        expect(op, sel, &[Dtype::I32])?;
        if window.is_some() {
            return Err(refuse(
                op,
                "a selection and a sliding window both answer which keys a row reads",
            ));
        }
        if ratio == 0 || sel.width == 0 || sel.rows < q.rows {
            return Err(refuse(
                op,
                "the selection is one i32 block-id row per query row, over nonzero-wide blocks",
            ));
        }
    }
    expect(op, mask, &[Dtype::U8])?;
    let (kv_heads, _) = pool_shape(op, pool, head_dim)?;
    if head_dim == 0 || !q.width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("q is {} wide for head_dim {head_dim}", q.width),
        ));
    }
    let q_heads = q.width / head_dim;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            op,
            format!("{q_heads} query heads over {kv_heads} kv heads"),
        ));
    }
    if o.rows < q.rows || o.width != q.width {
        return Err(refuse(
            op,
            format!("o is {}x{}, q is {}x{}", o.rows, o.width, q.rows, q.width),
        ));
    }
    if t.positions.elements() < u64::from(q.rows) {
        return Err(refuse(
            op,
            format!("{} positions for {} rows", t.positions.elements(), q.rows),
        ));
    }
    if let Some(l) = lse {
        expect(op, l, &[Dtype::F32])?;
        if l.rows < q.rows || l.width != q_heads {
            return Err(refuse(
                op,
                format!("lse is {}x{}, wants {}x{q_heads}", l.rows, l.width, q.rows),
            ));
        }
    }
    let rows = q.rows;
    let group = q_heads / kv_heads;
    let ps = pool.page_size as u32;
    let span = pool.max_pages * ps;
    let mask_w = t.mask_stride.min(mask.width);
    let d = head_dim;

    let lanes = pool.page_indptr.elements().saturating_sub(1).max(1);
    // A pool past one PE: lanes (their pages, their pages' kv heads) spread
    // over PEs. Besides pages a PE holds its heads' columns of q, o and the
    // lse, the tables, the mask and the scratch, for the rows of one phase:
    // all the rows when that fits, else a window of them, halved until it
    // does, each window its own phase over views of the row-shaped buffers.
    let packed = crate::linear::gemm::half_pool() && pool.keys.dtype == Dtype::Bf16 && pool.keys.width.is_multiple_of(2);
    let page_pair = u64::from(ps) * u64::from(pool.keys.width) * 2 / if packed { 2 } else { 1 };
    let (top_k, ratio) = selection.map_or((0, 0), |(s, r)| (s.width, r));
    let sblk_words = u64::from(span.checked_div(ratio).map_or(1, |b| b + 1));
    let fabric = fabric_sum();
    let resident = crate::linear::gemm::resident_pes();
    let pool_pages = pool.keys.rows / ps.max(1);
    let extra_for = |r: u32| {
        move |hg: u32| -> u64 {
            let q_cols = u64::from(q.width / hg);
            let merge = if fabric {
                merge_words(u64::from(r) * q_cols, u64::from(r) * u64::from(q_heads / hg))
            } else {
                0
            };
            u64::from(r)
                * (2 * q_cols + u64::from(q_heads / hg) + 3 + u64::from(mask_w) + u64::from(top_k))
                + lanes
                + 1
                + u64::from(d)
                + u64::from(mask_w)
                + sblk_words
                + 16
                + merge
        }
    };
    let pool_words = pool.keys.elements() + pool.values.elements();
    let mut chunk = rows
        .max(1)
        .min(rows_per_phase_cap().unwrap_or(u32::MAX))
        .max(1);
    let mut strided_pages = 0;
    let split = loop {
        if let Some(pes) = resident {
            // The resident row: the pool strided over it, all rows at once.
            match resident_split(pes, pool_pages, page_pair, lanes as u32, extra_for(chunk)(1)) {
                Some((split, share)) => {
                    strided_pages = share;
                    break Some(split);
                }
                None if chunk > 1 => {
                    chunk = chunk.div_ceil(2);
                    continue;
                }
                None => {
                    return Err(refuse(
                        op,
                        format!(
                            "the resident row of {pes} PEs does not hold the pool's {pool_pages} pages beside one row's buffers"
                        ),
                    ));
                }
            }
        }
        match page_split(
            op,
            pool_words,
            lanes as u32,
            page_pair,
            pool.max_pages,
            kv_heads,
            ps,
            extra_for(chunk),
        ) {
            Ok(split) => break split,
            Err(_) if chunk > 1 => chunk = chunk.div_ceil(2),
            Err(e) => return Err(e),
        }
    };
    let mut c0 = 0;
    while c0 < rows {
        let n = (rows - c0).min(chunk);
        ctx.emit(&mut |cx| {
            let qb = cx.read(q)?;
            let (kb, vb) = match (cx.pool_half(pool.keys)?, cx.pool_half(pool.values)?) {
                (Some(k), Some(v)) if packed => (k, v),
                _ => (cx.read(pool.keys)?, cx.read(pool.values)?),
            };
            let packed = kb.elem == "u32";
            // Words a cell of `d` halves takes on the device.
            let per: u32 = if packed { 2 } else { 1 };
            let idx = cx.read(pool.page_indices)?;
            let ptr = cx.read(pool.page_indptr)?;
            let pos = cx.read(t.positions)?;
            let req = cx.read(t.request_of_token)?;
            let en = cx.read(t.mask_enabled)?;
            let mb = cx.read(mask)?;
            let ob = cx.write(o)?;
            let lb = match lse {
                Some(l) => Some(cx.write(l)?),
                None => None,
            };
            // This phase's window of rows: views the host cuts and pastes.
            let qb = cx.window_of(&qb, c0, n)?;
            let pos = cx.window_of(&pos, c0, n)?;
            let req = cx.window_of(&req, c0, n)?;
            let en = cx.window_of(&en, c0, n)?;
            let mb = cx.window_of(&mb, c0, n)?;
            let ob = cx.window_of(&ob, c0, n)?;
            let sel_ptr = match selection {
                Some((sel, _)) => {
                    let sb = cx.read(sel)?;
                    Arg::Ptr(cx.window_of(&sb, c0, n)?)
                }
                None => Arg::Dummy("i32"),
            };
            let sblk = cx.scratch_of("sblk", sblk_words, "u32");
            let lb = match &lb {
                Some(l) => Some(cx.window_of(l, c0, n)?),
                None => None,
            };
            let acc = cx.scratch("acc", u64::from(d));
            let mrow = cx.scratch_of("mrow", u64::from(mask_w.max(1)), "u32");
            let mut lse_buf = lb.clone();
            let (rows_range, heads_range) = match split.clone() {
                None => (format!("0, {n}, 0"), format!("0, {kv_heads}, 0, {ps}")),
                Some(PageSplit {
                    pes,
                    lanes_per_pe,
                    page_groups,
                    head_groups,
                    row_groups,
                }) => {
                    let header = cx.unique("lanes");
                    // Pages split over PEs: every PE's output is partial, merged
                    // through a per-head lse, on the fabric (the page and row
                    // groups of a lane block are consecutive PEs, one
                    // rectangle row) or by the host, which then writes the lse.
                    let mut reduce = Vec::new();
                    let strided = resident.is_some();
                    // Strided: every PE holds a part of every lane's pages.
                    let merge_group = if strided { pes } else { page_groups * row_groups };
                    let fabric = merge_group > 1 && (strided || merge_group <= MERGE_GROUPS) && fabric_sum();
                    if page_groups > 1 || row_groups > 1 || strided {
                        if lse_buf.is_none() {
                            let name = cx.unique("lse");
                            lse_buf = Some(cx.program().declare(&name, Dtype::F32, n, q_heads, true)?);
                        }
                        let lse_name = lse_buf.as_ref().map(|b| b.name.clone()).unwrap_or_default();
                        if fabric {
                            reduce.push((ob.name.clone(), Reduce::FabricRoot { group: merge_group }));
                            reduce.push((lse_name, Reduce::FabricRoot { group: merge_group }));
                        } else {
                            reduce.push((ob.name.clone(), Reduce::Weighted { lse: lse_name.clone() }));
                            reduce.push((lse_name, Reduce::LogSumExp));
                        }
                    }
                    // q, o and the lse travel by head block: the columns of the
                    // PE's kv heads' query heads.
                    let plane = |b: &crate::program::Buf, per_head: u32| ColPlane {
                        name: b.name.clone(),
                        rows: b.rows,
                        width: b.width,
                        by_rows: false,
                        segments: vec![Segment {
                            base: 0,
                            a_group: 0,
                            a_stride: 0,
                            b_stride: group * per_head,
                            span: group * per_head,
                        }],
                        row_block: false,
                    };
                    let mut cols = vec![plane(&qb, d), plane(&ob, d)];
                    if let Some(l) = &lse_buf {
                        cols.push(plane(l, 1));
                    }
                    let plan = LanePlan {
                        pes,
                        kind: LaneKind::ByTable {
                            lane_of_row: req.name.clone(),
                            lanes: lanes as u32,
                        },
                        header,
                        slots: req.name.clone(),
                        banks: vec![(kb.name.clone(), ps * kb.width / per, BankSplit::Block), (vb.name.clone(), ps * vb.width / per, BankSplit::Block)],
                        block: Some(BlockPlan {
                            a: ps,
                            b: kv_heads,
                            w: d / per,
                            a_groups: row_groups,
                            b_groups: head_groups,
                        }),
                        lanes_per_pe,
                        a_first: fabric && !strided,
                        row_outputs: Vec::new(),
                        cols,
                        reduce,
                        window: None,
                        pages: Some(PagePlan {
                            source: PageSource::Csr {
                                indptr: ptr.name.clone(),
                                indices: idx.name.clone(),
                            },
                            page_size: ps,
                            max_pages: pool.max_pages,
                            rewritten: Vec::new(),
                            page_groups,
                            strided,
                            strided_pages,
                        }),
                    };
                    let o_words = plan.plane_words(&plan.cols[1]);
                    let l_words = plan.cols.get(2).map_or(0, |c| plan.plane_words(c));
                    let hdr = cx.lanes(plan)?;
                    if let (true, Some(l)) = (fabric, &lse_buf) {
                        lse_merge_with(cx, &ob, l, o_words, l_words, merge_group, pes, strided);
                    }
                    (
                        format!("{0}[7], {0}[8], {0}[9] * {ps}", hdr.name),
                        format!("{0}[5], {0}[6], {0}[3], {0}[4]", hdr.name),
                    )
                }
            };
            let lse_ptr = lse_buf.as_ref().map_or(Arg::Dummy("f32"), |b| Arg::Ptr(b.clone()));
            let (kernel, vrow) = if packed {
                let vrow = cx.scratch("vrow", u64::from(d));
                cx.library("k_half");
                (cx.library("k_attend_packed"), Some(Arg::Scratch(vrow, "f32")))
            } else {
                (cx.library("k_attend"), None)
            };
            let mut args = vec![Arg::Ptr(qb), Arg::Ptr(kb), Arg::Ptr(vb)];
            args.extend(vrow);
            args.extend([
                Arg::Ptr(idx),
                Arg::Ptr(ptr),
                Arg::Ptr(pos),
                Arg::Ptr(req),
                Arg::Ptr(en),
                Arg::Ptr(mb),
                Arg::Ptr(ob.clone()),
                lse_ptr,
                Arg::Scratch(acc.clone(), "f32"),
                Arg::Scratch(mrow.clone(), "u32"),
                sel_ptr,
                Arg::Scratch(sblk.clone(), "u32"),
                Arg::Int(i64::from(top_k)),
                Arg::Int(i64::from(ratio)),
            ]);
            args.extend(Arg::exprs(&rows_range));
            args.extend(Arg::exprs(&heads_range));
            args.extend([
                Arg::Int(i64::from(group)),
                Arg::Int(i64::from(d)),
                Arg::Int(i64::from(ps)),
                Arg::Int(i64::from(span)),
                Arg::Int(i64::from(mask_w)),
                Arg::Int(i64::from(mask.width)),
                Arg::Float(sm_scale),
                Arg::Int(i64::from(window.unwrap_or(0))),
                Arg::Bool(causal),
                Arg::Int(lanes as i64),
                Arg::Bool(lse_buf.is_some()),
            ]);
            cx.call(kernel, args);
            Ok(())
        })?;
        c0 += n;
    }
    Ok(())
}

pub fn decode(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let t = Tables::from(plan);
    attend(
        ctx,
        "attention.decode",
        q,
        &t,
        t.mask,
        pool,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        None,
        None,
    )
}

pub fn decode_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    let t = Tables::from(plan);
    attend(
        ctx,
        "attention.decode_lse",
        q,
        &t,
        t.mask,
        pool,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        None,
    )
}

fn heads_agree(op: &'static str, pool: &KvPool, head_dim: u32, kv_heads: u32) -> Result<(), Error> {
    let (found, _) = pool_shape(op, pool, head_dim)?;
    if found != kv_heads {
        return Err(refuse(
            op,
            format!("the op states {kv_heads} kv heads, the pool holds {found}"),
        ));
    }
    Ok(())
}

pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    heads_agree(OP, pool, head_dim, kv_heads)?;
    let t = Tables::from(plan);
    attend(
        ctx, OP, q.data, &t, t.mask, pool, window, true, head_dim, sm_scale, o, None, None,
    )
}

pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    heads_agree(OP, pool, head_dim, kv_heads)?;
    let t = Tables::from(plan);
    attend(
        ctx,
        OP,
        q.data,
        &t,
        t.mask,
        pool,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        None,
    )
}

/// A prefill whose custom mask is `mask` rather than the plan's (the plan's
/// per-row enable flags still gate it). `causal: false` reads every key the
/// row's lane holds; so does a causal row whose enable flag is 2.
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let t = Tables::from(plan);
    attend(
        ctx,
        "attention.masked",
        q.data,
        &t,
        mask,
        pool,
        window,
        causal,
        head_dim,
        sm_scale,
        o,
        None,
        None,
    )
}

pub fn masked_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    let t = Tables::from(plan);
    attend(
        ctx,
        "attention.masked_lse",
        q.data,
        &t,
        mask,
        pool,
        window,
        causal,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        None,
    )
}

/// Selected-block decode (DeepSeek sparse attention): `selection` names
/// `ratio`-wide key blocks per row among the closed blocks before the
/// query; the open block is read whole.
pub fn decode_selected(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    ratio: u32,
    o: Tensor,
) -> Result<(), Error> {
    let t = Tables::from(plan);
    attend(
        ctx,
        "attention.decode_selected",
        q,
        &t,
        t.mask,
        pool,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        None,
        Some((selection, ratio)),
    )
}

pub fn prefill_selected(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    ratio: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_selected";
    heads_agree(OP, pool, head_dim, kv_heads)?;
    let t = Tables::from(plan);
    attend(
        ctx,
        OP,
        q.data,
        &t,
        t.mask,
        pool,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        None,
        Some((selection, ratio)),
    )
}

fn append(
    ctx: &Ctx<'_>,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    expect(op, k, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, v, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, write_page, &[Dtype::I32])?;
    expect(op, write_offset, &[Dtype::I32])?;
    expect(op, pool.keys, &[Dtype::Bf16, Dtype::F32])?;
    expect(op, pool.values, &[Dtype::Bf16, Dtype::F32])?;
    let w = pool.keys.width;
    if k.width != w || v.width != pool.values.width || k.rows != v.rows {
        return Err(refuse(
            op,
            format!(
                "k is {}x{}, v is {}x{}, the pool is {w} wide",
                k.rows, k.width, v.rows, v.width
            ),
        ));
    }
    if write_page.elements() < u64::from(k.rows) || write_offset.elements() < u64::from(k.rows) {
        return Err(refuse(
            op,
            format!(
                "{} rows, {} pages, {} offsets",
                k.rows,
                write_page.elements(),
                write_offset.elements()
            ),
        ));
    }
    if pool.page_size <= 0 || !pool.keys.rows.is_multiple_of(pool.page_size as u32) {
        return Err(refuse(
            op,
            format!(
                "{} pool rows in pages of {}",
                pool.keys.rows, pool.page_size
            ),
        ));
    }
    let ps = pool.page_size as u32;
    let pages = pool.keys.rows / ps;
    let shared = pool.keys.buf == pool.values.buf;
    ctx.emit(&mut |cx| {
        let kb = cx.read(k)?;
        let vb = cx.read(v)?;
        let pb = cx.read(write_page)?;
        let obf = cx.read(write_offset)?;
        let packed_ok = crate::linear::gemm::half_pool() && pool.keys.dtype == Dtype::Bf16 && w.is_multiple_of(2);
        let keys = match cx.pool_half(pool.keys)? {
            Some(k) if packed_ok => k,
            _ => {
                cx.read(pool.keys)?;
                cx.write(pool.keys)?
            }
        };
        let packed = keys.elem == "u32";
        let per: u32 = if packed { 2 } else { 1 };
        let values = if shared {
            keys.clone()
        } else if packed {
            cx.pool_half(pool.values)?.ok_or_else(|| refuse(op, "the values pool is not packed like the keys"))?
        } else {
            cx.read(pool.values)?;
            cx.write(pool.values)?
        };
        // A pool past one PE's share: rows spread over PEs, each holding the
        // pages its rows write.
        let page_words = u64::from(ps) * u64::from(w) / u64::from(per);
        let planes = if shared {
            pool.keys.elements()
        } else {
            pool.keys.elements() + pool.values.elements()
        };
        // A page past a PE splits its columns (`w` in `hg` blocks): the PE
        // then holds those columns of k and v and of its pages.
        let extra = |hg: u32| {
            (k.elements() + if shared { 0 } else { v.elements() }) / u64::from(hg)
                + 2 * u64::from(k.rows)
                + 16
        };
        let resident = crate::linear::gemm::resident_pes();
        let mut strided_pages = 0;
        let split = match resident {
            Some(pes) => {
                let pair = if shared { page_words } else { page_words * 2 };
                match resident_split(pes, pages, pair, k.rows, extra(1)) {
                    Some((split, share)) => {
                        strided_pages = share;
                        Some(split)
                    }
                    None => {
                        return Err(refuse(
                            op,
                            format!("the resident row of {pes} PEs does not hold the pool's {pages} pages beside the rows"),
                        ));
                    }
                }
            }
            None => page_split(
                op,
                planes,
                k.rows,
                if shared { page_words } else { page_words * 2 },
                1,
                w,
                1,
                extra,
            )?,
        };
        let (rows_range, width) = match split {
            None => (format!("0, {}", k.rows), w.to_string()),
            Some(PageSplit {
                pes,
                lanes_per_pe: rows_per_pe,
                head_groups,
                ..
            }) => {
                let header = cx.unique("lanes");
                let mut banks = vec![(keys.name.clone(), ps * w / per, BankSplit::Block)];
                if !shared {
                    banks.push((values.name.clone(), ps * w / per, BankSplit::Block));
                }
                // The k and v planes' columns follow the block's `b`, which
                // counts words of a packed page: `per` elements a word.
                let plane = |b: &crate::program::Buf| ColPlane {
                    name: b.name.clone(),
                    rows: b.rows,
                    width: b.width,
                    by_rows: false,
                    segments: vec![Segment {
                        base: 0,
                        a_group: 0,
                        a_stride: 0,
                        b_stride: per,
                        span: per,
                    }],
                    row_block: false,
                };
                let (block, cols) = if head_groups > 1 {
                    let mut cols = vec![plane(&kb)];
                    if !shared {
                        cols.push(plane(&vb));
                    }
                    (
                        Some(BlockPlan {
                            a: ps,
                            b: w / per,
                            w: 1,
                            a_groups: 1,
                            b_groups: head_groups,
                        }),
                        cols,
                    )
                } else {
                    (None, Vec::new())
                };
                let hdr = cx.lanes(LanePlan {
                    pes,
                    kind: LaneKind::PerRow { lanes: k.rows },
                    header,
                    slots: pb.name.clone(),
                    banks,
                    block,
                    lanes_per_pe: rows_per_pe,
                    a_first: false,
                    row_outputs: Vec::new(),
                    cols,
                    reduce: Vec::new(),
                    window: None,
                    pages: Some(PagePlan {
                        source: PageSource::Rows {
                            table: pb.name.clone(),
                        },
                        page_size: ps,
                        max_pages: 1,
                        rewritten: vec![pb.name.clone()],
                        page_groups: 1,
                        strided: resident.is_some(),
                        strided_pages,
                    }),
                })?;
                (
                    format!("{0}[7], {0}[8]", hdr.name),
                    if head_groups > 1 {
                        format!("({0}[6] - {0}[5]) * {per}", hdr.name)
                    } else {
                        w.to_string()
                    },
                )
            }
        };
        let kernel = if packed {
            cx.library("k_half");
            cx.library("k_kv_append_packed")
        } else {
            cx.library("k_kv_append")
        };
        let mut args = vec![
            Arg::Ptr(kb),
            Arg::Ptr(vb),
            Arg::Ptr(keys.clone()),
            Arg::Ptr(values.clone()),
            Arg::Ptr(pb.clone()),
            Arg::Ptr(obf),
        ];
        args.extend(Arg::exprs(&rows_range));
        args.extend([
            Arg::Expr(width.clone()),
            Arg::Int(i64::from(ps)),
            Arg::Int(i64::from(pages)),
            Arg::Bool(shared),
        ]);
        cx.call(kernel, args);
        Ok(())
    })
}

/// Lands row `i` of `k` and `v` in pool row `write_page[i] * page_size +
/// write_offset[i]`; a page or offset out of range drops the row.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(
        ctx,
        "attention.kv_append",
        k,
        v,
        pool,
        write_page,
        write_offset,
    )
}

/// [`kv_append`] for a pool whose keys and values alias one plane.
pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(
        ctx,
        "attention.kv_append_shared",
        plane,
        plane,
        pool,
        write_page,
        write_offset,
    )
}
