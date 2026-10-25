//! Bidirectional attention inside segments: every row attends to the key
//! rows of its own segment, `segments` an i32 `[images + 1]` boundary vector
//! over the shared row numbering (vision encoders), or, ragged, the key rows
//! a second CSR names for its segment, under a keep plane or a relative
//! bias (group-packed attention); a row outside every segment answers zeros.
//!
//! Placement: a lane plan with the segments as ragged lanes (no banks), the
//! key rows split over the block's `a` axis and the kv heads over `b`; k and
//! v travel as row-block planes (their key rows, their heads' columns), q,
//! o and the lse as head-column planes of a row window (the query rows of
//! one phase). PEs over different key groups hold partials of the same rows,
//! merged by their per-head lse like page groups.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{ColPlane, LaneKind, Lanes, Reduce, Segment, SlotState};
use crate::plan::{BlockSplit, Blocks, Merge};
use crate::tensor::Tensor;

/// Words of a PE the attention kernel's code and stack take.
const CODE_WORDS: u64 = 2048;

pub fn bidirectional(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.dense";
    if head_dim == 0 || !k.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("k {} wide for heads of {head_dim}", k.width),
        ));
    }
    attend(
        OP,
        ctx,
        q,
        k,
        v,
        segments,
        segments,
        head_dim,
        k.width / head_dim,
        sm_scale,
        RaggedMask::Segments,
        o,
    )
}

/// What a ragged attention applies beside its segments.
#[derive(Clone, Copy, Debug)]
pub enum RaggedMask {
    Segments,
    /// `keep` is `[q rows][k rows]` (f32 or bf16): a pair of one segment is
    /// kept where its cell is nonzero.
    Custom {
        keep: Tensor,
    },
    /// Adds `table[h, clamp(k − q + max_len − 1)]` (f32, `[q_heads, 2·max_len
    /// − 1]`) to the scaled logit, `k`/`q` indexed within the segment.
    RelativeBias {
        table: Tensor,
        max_len: u32,
    },
}

/// A query row of segment `s` (rows `q_indptr[s]..q_indptr[s + 1]`) attends
/// key rows `kv_indptr[s]..kv_indptr[s + 1]`, under `mask`.
pub fn ragged(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: Tensor,
) -> Result<(), Error> {
    attend(
        "attn.ragged",
        ctx,
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        head_dim,
        kv_heads,
        sm_scale,
        mask,
        o,
    )
}

fn attend(
    op: &'static str,
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    kv_segments: Tensor,
    head_dim: u32,
    kvh: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: Tensor,
) -> Result<(), Error> {
    for t in [q, k, v, o] {
        expect(op, t, &[Dtype::Bf16, Dtype::F32])?;
    }
    expect(op, segments, &[Dtype::I32])?;
    expect(op, kv_segments, &[Dtype::I32])?;
    if head_dim == 0 || !q.width.is_multiple_of(head_dim) || k.width != kvh * head_dim {
        return Err(refuse(
            op,
            format!(
                "q {} and k {} wide for {kvh} kv heads of {head_dim}",
                q.width, k.width
            ),
        ));
    }
    let qh = q.width / head_dim;
    if kvh == 0 || !qh.is_multiple_of(kvh) || v.width != k.width || v.rows != k.rows {
        return Err(refuse(
            op,
            format!("{qh} query heads do not group over {kvh} kv heads of one k/v rectangle"),
        ));
    }
    if o.rows != q.rows || o.width != q.width {
        return Err(refuse(op, "the answer is one row per query row"));
    }
    let entries = segments.elements().min(kv_segments.elements());
    if entries < 2 {
        return Err(refuse(
            op,
            "the segment list is an i32 boundary vector naming at least one segment",
        ));
    }
    if q.rows == 0 || k.rows == 0 {
        return Err(refuse(op, "rows are nonzero"));
    }
    let (keep, bias, max_len) = match mask {
        RaggedMask::Segments => (None, None, 0),
        RaggedMask::Custom { keep } => {
            expect(op, keep, &[Dtype::Bf16, Dtype::F32])?;
            if keep.rows != q.rows || keep.width != k.rows {
                return Err(refuse(
                    op,
                    format!(
                        "the keep plane is {} x {}, and it is one cell a (query row, key row) \
                         of {} x {}",
                        keep.rows, keep.width, q.rows, k.rows
                    ),
                ));
            }
            (Some(keep), None, 0)
        }
        RaggedMask::RelativeBias { table, max_len } => {
            expect(op, table, &[Dtype::F32])?;
            if max_len == 0 || table.elements() < u64::from(qh) * u64::from(2 * max_len - 1) {
                return Err(refuse(
                    op,
                    format!(
                        "the bias table holds {} cells, and this attention reads {qh} heads of \
                         2 x {max_len} - 1",
                        table.elements()
                    ),
                ));
            }
            (None, Some(table), max_len)
        }
    };
    let images = (entries - 1) as u32;
    let group = qh / kvh;
    let d = head_dim;
    let (rows, keys) = (q.rows, k.rows);
    // Words a PE holds for a window of `n` query rows with the keys split
    // `ag` ways and the kv heads `hg` ways.
    let words = |n: u32, ag: u32, hg: u32| -> u64 {
        let kv = 2 * u64::from(keys.div_ceil(ag)) * u64::from(kvh / hg) * u64::from(d);
        let qo = 2 * u64::from(n) * u64::from(qh / hg) * u64::from(d);
        let lse = u64::from(n) * u64::from(qh / hg);
        let keep = if keep.is_some() {
            u64::from(n) * u64::from(keys.div_ceil(ag))
        } else {
            0
        };
        let bias = if bias.is_some() {
            u64::from(qh) * u64::from(2 * max_len - 1)
        } else {
            0
        };
        kv + qo + lse + keep + bias + 2 * (u64::from(images) + 1) + u64::from(d) + 16
    };
    // The fabric merge of a key split: the partial's copy, the lse copies
    // and the collectives library.
    let merged = |n: u32, _: u32, hg: u32| -> u64 {
        let o = u64::from(n) * u64::from(qh / hg);
        super::paged::merge_words(o * u64::from(d), o)
    };
    // The keys split `ag` ways and the kv heads `hg` ways, over windows of
    // the rows; the attention kernel's code takes a good part of a PE.
    let blocks = ctx
        .plan()
        .blocks(&Blocks {
            rows,
            a: keys,
            b: kvh,
            code: CODE_WORDS,
            held: &words,
            merge: &merged,
        })
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "one query row and one key row of one head exceed a PE's {}",
                    ctx.plan().pe_words().saturating_sub(CODE_WORDS)
                ),
            )
        })?;
    let BlockSplit {
        window: chunk,
        a_groups: ag,
        b_groups: hg,
    } = blocks;
    let whole = chunk >= rows && ag == 1 && hg == 1;
    let merge = ag > 1
        && ctx.plan().merge(Merge::Lse {
            group: ag,
            row: false,
        });
    let mut c0 = 0;
    while c0 < rows {
        let n = (rows - c0).min(chunk);
        ctx.emit(&mut |cx| {
            let qb = cx.read(q)?;
            let kb = cx.read(k)?;
            let vb = cx.read(v)?;
            let sb = cx.read(segments)?;
            let tb = cx.read(kv_segments)?;
            let ob = cx.write(o)?;
            let acc = cx.scratch("acc", u64::from(d));
            let qb = cx.window_of(&qb, c0, n)?;
            let ob = cx.window_of(&ob, c0, n)?;
            let keep = match keep {
                Some(keep) => {
                    let kb = cx.read(keep)?;
                    Some(cx.window_of(&kb, c0, n)?)
                }
                None => None,
            };
            let bias = match bias {
                Some(table) => Some(cx.read(table)?),
                None => None,
            };
            let ptr_or = |b: &Option<crate::phase::Buf>| {
                b.as_ref()
                    .map_or(Arg::Dummy("f32"), |b| Arg::Ptr(b.clone()))
            };
            let (keep_arg, bias_arg) = (ptr_or(&keep), ptr_or(&bias));
            if whole {
                cx.call(
                    Kernel::AttendDense,
                    vec![
                        Arg::Ptr(qb),
                        Arg::Ptr(kb),
                        Arg::Ptr(vb),
                        Arg::Ptr(ob),
                        Arg::Dummy("f32"),
                        Arg::Ptr(sb),
                        Arg::Ptr(tb),
                        keep_arg,
                        bias_arg,
                        Arg::Scratch(acc.clone(), "f32"),
                        Arg::Int(0),
                        Arg::Int(i64::from(images)),
                        Arg::Int(0),
                        Arg::Int(i64::from(keys)),
                        Arg::Int(0),
                        Arg::Int(i64::from(kvh)),
                        Arg::Int(0),
                        Arg::Int(i64::from(n)),
                        Arg::Int(i64::from(c0)),
                        Arg::Int(i64::from(group)),
                        Arg::Int(i64::from(d)),
                        Arg::Float(sm_scale),
                        Arg::Bool(false),
                        Arg::Bool(keep.is_some()),
                        Arg::Int(i64::from(max_len)),
                    ],
                );
                return Ok(());
            }
            // Partial rows over key groups need a per-head lse to merge by.
            let lse_name = cx.unique("lse");
            let lse = cx.declare(&lse_name, Dtype::F32, n, qh, true)?;
            let heads = |b: &crate::phase::Buf, per_head: u32, row_block: bool| ColPlane {
                name: b.name.clone(),
                rows: b.rows,
                width: b.width,
                by_rows: false,
                row_block,
                segments: vec![Segment {
                    base: 0,
                    a_group: 0,
                    a_stride: 0,
                    b_stride: per_head,
                    span: per_head,
                }],
            };
            let reduce = if merge {
                vec![
                    (ob.name.clone(), Reduce::FabricRoot { group: ag }),
                    (lse.name.clone(), Reduce::FabricRoot { group: ag }),
                ]
            } else if ag > 1 {
                vec![
                    (
                        ob.name.clone(),
                        Reduce::Weighted {
                            lse: lse.name.clone(),
                        },
                    ),
                    (lse.name.clone(), Reduce::LogSumExp),
                ]
            } else {
                Vec::new()
            };
            let mut cols = vec![
                heads(&kb, d, true),
                heads(&vb, d, true),
                heads(&qb, group * d, false),
                heads(&ob, group * d, false),
                heads(&lse, group, false),
            ];
            // The keep plane: every window row, the PE's key columns.
            if let Some(keep) = &keep {
                cols.push(ColPlane {
                    name: keep.name.clone(),
                    rows: keep.rows,
                    width: keep.width,
                    by_rows: false,
                    row_block: false,
                    segments: vec![Segment {
                        base: 0,
                        a_group: 1,
                        a_stride: 1,
                        b_stride: 0,
                        span: 1,
                    }],
                });
            }
            let lanes = Lanes {
                kind: LaneKind::Ragged {
                    indptr: sb.name.clone(),
                    lanes: images,
                },
                slots: String::new(),
                banks: Vec::new(),
                block: Some(SlotState {
                    a: keys,
                    b: kvh,
                    w: d,
                }),
                a_first: merge,
                row_outputs: Vec::new(),
                cols,
                reduce,
                window: Some((c0, n)),
                pages: None,
            };
            let at = cx.lanes(lanes, blocks.lanes(images))?;
            let (o_words, l_words) = (at.planes[3], at.planes[4]);
            let hdr = at.header;
            if merge {
                super::paged::lse_merge(cx, &ob, &lse, o_words, l_words, ag, ag * hg);
            }
            let h = |i| Arg::Word(hdr.name.clone(), i);
            cx.call(
                Kernel::AttendDense,
                vec![
                    Arg::Ptr(qb),
                    Arg::Ptr(kb),
                    Arg::Ptr(vb),
                    Arg::Ptr(ob.clone()),
                    Arg::Ptr(lse.clone()),
                    Arg::Ptr(sb),
                    Arg::Ptr(tb),
                    keep_arg,
                    bias_arg,
                    Arg::Scratch(acc.clone(), "f32"),
                    h(0),
                    h(1),
                    h(3),
                    h(4),
                    h(5),
                    h(6),
                    h(7),
                    h(8),
                    Arg::Int(i64::from(c0)),
                    Arg::Int(i64::from(group)),
                    Arg::Int(i64::from(d)),
                    Arg::Float(sm_scale),
                    Arg::Bool(true),
                    Arg::Bool(keep.is_some()),
                    Arg::Int(i64::from(max_len)),
                ],
            );
            Ok(())
        })?;
        c0 += n;
    }
    Ok(())
}
