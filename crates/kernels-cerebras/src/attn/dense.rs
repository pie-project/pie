//! Bidirectional attention inside image segments (vision encoders): every
//! row attends to the key rows of its own segment, `segments` an i32
//! `[images + 1]` boundary vector over the shared row numbering; a row
//! outside every segment answers zeros.
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
use crate::phase::{BlockPlan, ColPlane, LaneKind, LanePlan, Reduce, Segment};
use crate::plan::{BlockSplit, Blocks};
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
    for t in [q, k, v, o] {
        expect(OP, t, &[Dtype::Bf16, Dtype::F32])?;
    }
    expect(OP, segments, &[Dtype::I32])?;
    if head_dim == 0 || !q.width.is_multiple_of(head_dim) || !k.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "q {} and k {} wide for heads of {head_dim}",
                q.width, k.width
            ),
        ));
    }
    let (qh, kvh) = (q.width / head_dim, k.width / head_dim);
    if kvh == 0 || !qh.is_multiple_of(kvh) || v.width != k.width || v.rows != k.rows {
        return Err(refuse(
            OP,
            format!("{qh} query heads do not group over {kvh} kv heads of one k/v rectangle"),
        ));
    }
    if o.rows != q.rows || o.width != q.width {
        return Err(refuse(OP, "the answer is one row per query row"));
    }
    if segments.elements() < 2 {
        return Err(refuse(
            OP,
            "the segment list is an i32 boundary vector naming at least one image",
        ));
    }
    if q.rows == 0 || k.rows == 0 {
        return Err(refuse(OP, "rows are nonzero"));
    }
    let images = (segments.elements() - 1) as u32;
    let group = qh / kvh;
    let d = head_dim;
    let (rows, keys) = (q.rows, k.rows);
    // Words a PE holds for a window of `n` query rows with the keys split
    // `ag` ways and the kv heads `hg` ways.
    let fabric = ctx.plan().fabric().fabric_sum;
    let words = |n: u32, ag: u32, hg: u32| -> u64 {
        let kv = 2 * u64::from(keys.div_ceil(ag)) * u64::from(kvh / hg) * u64::from(d);
        let qo = 2 * u64::from(n) * u64::from(qh / hg) * u64::from(d);
        let lse = u64::from(n) * u64::from(qh / hg);
        // The fabric merge of a key split: the partial's copy, the lse
        // copies and the collectives library.
        let merge = if ag > 1 && ag <= super::paged::MERGE_GROUPS && fabric {
            super::paged::merge_words(qo / 2, lse)
        } else {
            0
        };
        kv + qo + lse + merge + u64::from(images) + 1 + u64::from(d) + 16
    };
    // The keys split `ag` ways and the kv heads `hg` ways, over windows of
    // the rows; the attention kernel's code takes a good part of a PE.
    let BlockSplit {
        window: chunk,
        a_groups: ag,
        b_groups: hg,
    } = ctx
        .plan()
        .blocks(&Blocks {
            rows,
            a: keys,
            b: kvh,
            code: CODE_WORDS,
            held: &words,
        })
        .ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "one query row and one key row of one head exceed a PE's {}",
                    ctx.plan().fabric().pe_words.saturating_sub(CODE_WORDS)
                ),
            )
        })?;
    let whole = chunk >= rows && ag == 1 && hg == 1;
    let mut c0 = 0;
    while c0 < rows {
        let n = (rows - c0).min(chunk);
        ctx.emit(&mut |cx| {
            let qb = cx.read(q)?;
            let kb = cx.read(k)?;
            let vb = cx.read(v)?;
            let sb = cx.read(segments)?;
            let ob = cx.write(o)?;
            let acc = cx.scratch("acc", u64::from(d));
            let qb = cx.window_of(&qb, c0, n)?;
            let ob = cx.window_of(&ob, c0, n)?;
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
                    ],
                );
                return Ok(());
            }
            // Partial rows over key groups need a per-head lse to merge by.
            let lse_name = cx.unique("lse");
            let lse = cx.declare(&lse_name, Dtype::F32, n, qh, true)?;
            let header = cx.unique("lanes");
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
            let merge = ag > 1 && ag <= super::paged::MERGE_GROUPS && fabric;
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
            let plan = LanePlan {
                pes: ag * hg,
                kind: LaneKind::Ragged {
                    indptr: sb.name.clone(),
                    lanes: images,
                },
                header,
                slots: String::new(),
                banks: Vec::new(),
                block: Some(BlockPlan {
                    a: keys,
                    b: kvh,
                    w: d,
                    a_groups: ag,
                    b_groups: hg,
                }),
                lanes_per_pe: images,
                a_first: merge,
                row_outputs: Vec::new(),
                cols: vec![
                    heads(&kb, d, true),
                    heads(&vb, d, true),
                    heads(&qb, group * d, false),
                    heads(&ob, group * d, false),
                    heads(&lse, group, false),
                ],
                reduce,
                window: Some((c0, n)),
                pages: None,
            };
            let o_words = plan.plane_words(&plan.cols[3]);
            let l_words = plan.plane_words(&plan.cols[4]);
            let hdr = cx.lanes(plan)?;
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
                ],
            );
            Ok(())
        })?;
        c0 += n;
    }
    Ok(())
}
