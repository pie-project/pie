//! Multi-adapter LoRA: `y += B[a] · (A[a] · x)` per row, `a = routes[row]`,
//! a negative (or unseated) adapter id adding nothing.
//!
//! The banks are `bank_a: [adapters, rank · in]` (adapter `a`'s `[rank, in]`
//! down projection, row-major) and `bank_b: [adapters, out · rank]` (its
//! `[out, rank]` up projection), as kernels-wgpu `gemm/lora.wgsl` reads them.
//! One PE walks the rows: the waist `A[a] · x` is dotted into an f32 scratch,
//! then each output lane dots its up row against the waist.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::library::Kernel;
use crate::phase::{BankSplit, BlockPlan, ColPlane, LaneKind, LanePlan, Reduce, Segment};
use crate::plan::{Held, Routed};
use crate::tensor::Tensor;

pub fn correct(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, bank_a, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, bank_b, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, routes, &[Dtype::I32])?;
    let (rows, in_width, out_width) = (x.rows, x.width, y.width);
    if in_width == 0 || out_width == 0 {
        return Err(refuse(
            OP,
            "the correction's input and output widths are nonzero",
        ));
    }
    if y.rows != rows || routes.elements() != u64::from(rows) {
        return Err(refuse(
            OP,
            format!(
                "{rows} input rows land {} rows under {} adapter ids",
                y.rows,
                routes.elements()
            ),
        ));
    }
    if !bank_a.width.is_multiple_of(in_width) || bank_a.width == 0 {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = bank_a.width / in_width;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}",
                bank_b.width,
                out_width.saturating_mul(rank)
            ),
        ));
    }
    if bank_a.rows != bank_b.rows || bank_a.rows == 0 {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
            ),
        ));
    }
    let adapters = bank_a.rows;
    // Everything on one PE when it fits; else two lane-plan phases (the
    // routes as the slot table, rows as lanes): the waist `x · A[a]ᵀ`, each
    // PE holding rows `[k0, k1)` of its adapters' `[rank][in]` down bank
    // over input slice `[i0, i1)` and that slice of x, writing a partial the
    // host adds over the slices; then `y += waist · B[a]ᵀ`, each PE holding
    // outputs `[o0, o1)` of `[out][rank]` over ranks `[k0, k1)`, its slice of
    // the waist, and its columns of y, a partial the host adds over the ranks.
    let (a_words, b_words) = (u64::from(bank_a.width), u64::from(bank_b.width));
    let whole = u64::from(adapters) * (a_words + b_words)
        + u64::from(rows) * u64::from(in_width + out_width)
        + u64::from(rank);
    if ctx.plan().holds(Held {
        words: whole,
        array: u64::from(adapters) * a_words.max(b_words),
    }) {
        return ctx.emit(&mut |cx| {
            let xb = cx.read(x)?;
            let ab = cx.read(bank_a)?;
            let bb = cx.read(bank_b)?;
            let rb = cx.read(routes)?;
            cx.read(y)?;
            let yb = cx.write(y)?;
            let waist = cx.scratch("waist", u64::from(rank));
            cx.call(
                Kernel::Lora,
                vec![
                    Arg::Ptr(xb),
                    Arg::Ptr(ab),
                    Arg::Ptr(bb),
                    Arg::Ptr(rb),
                    Arg::Ptr(yb),
                    Arg::Scratch(waist.clone(), "f32"),
                    Arg::Int(0),
                    Arg::Int(i64::from(rows)),
                    Arg::Int(0),
                    Arg::Int(i64::from(out_width)),
                    Arg::Int(0),
                    Arg::Int(i64::from(rank)),
                    Arg::Int(0),
                    Arg::Int(i64::from(in_width)),
                    Arg::Int(i64::from(out_width)),
                    Arg::Int(i64::from(rank)),
                    Arg::Int(i64::from(adapters)),
                    Arg::Int(i64::from(bank_a.width)),
                    Arg::Int(i64::from(bank_b.width)),
                    Arg::Bool(false),
                ],
            );
            Ok(())
        });
    }
    // Rows are the lanes, a bank block each (lanes per PE dividing the
    // rows), beside the planes; a split of `b` leaves the output partial.
    let refused = |what: &str| {
        refuse(
            OP,
            format!(
                "one row of the {what} and a row's planes exceed a PE's {}",
                ctx.plan().fabric().pe_words
            ),
        )
    };
    // Down: a slot is `[rank][in]` as `[a = rank][b = in][1]`.
    let down = ctx
        .plan()
        .routed(&Routed {
            lanes: rows,
            a: rank,
            b: in_width,
            block: &|ag, bg| u64::from(rank / ag) * u64::from(in_width / bg),
            planes: &|ag, bg| u64::from(rows) * (u64::from(in_width / bg) + u64::from(rank / ag)),
            partial: &|ag, _| u64::from(rows) * u64::from(rank / ag),
        })
        .ok_or_else(|| refused("down bank"))?;
    // Up: a slot is `[out][rank]` as `[a = out][b = rank][1]`.
    let up = ctx
        .plan()
        .routed(&Routed {
            lanes: rows,
            a: out_width,
            b: rank,
            block: &|ag, bg| u64::from(out_width / ag) * u64::from(rank / bg),
            planes: &|ag, bg| u64::from(rows) * (u64::from(rank / bg) + u64::from(out_width / ag)),
            partial: &|ag, _| u64::from(rows) * u64::from(out_width / ag),
        })
        .ok_or_else(|| refused("up bank"))?;
    let waist_name = format!("waist_{}_{}", x.buf, y.buf);
    let plane = |b: &crate::phase::Buf, segments: Vec<Segment>| ColPlane {
        name: b.name.clone(),
        rows: b.rows,
        width: b.width,
        by_rows: false,
        segments,
        row_block: false,
    };
    let by_a = Segment {
        base: 0,
        a_group: 1,
        a_stride: 1,
        b_stride: 0,
        span: 1,
    };
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let ab = cx.read(bank_a)?;
        let rb = cx.read(routes)?;
        let wb = cx.declare(&waist_name, Dtype::F32, rows, rank, true)?;
        let header = cx.unique("lanes");
        let reduce = down
            .sum
            .iter()
            .map(|r| (wb.name.clone(), r.clone()))
            .collect();
        let pes = down.pes;
        let plan = LanePlan {
            pes,
            kind: LaneKind::PerRow { lanes: rows },
            header,
            slots: rb.name.clone(),
            banks: vec![(ab.name.clone(), bank_a.width, BankSplit::Block)],
            block: Some(BlockPlan {
                a: rank,
                b: in_width,
                w: 1,
                a_groups: down.a_groups,
                b_groups: down.b_groups,
            }),
            lanes_per_pe: down.lanes_per_pe,
            a_first: false,
            row_outputs: Vec::new(),
            cols: vec![plane(&xb, vec![Segment::along_b()]), plane(&wb, vec![by_a])],
            reduce,
            window: None,
            pages: None,
        };
        let w_words = plan.plane_words(&plan.cols[1]);
        let hdr = cx.lanes(plan)?;
        cx.call(
            Kernel::LoraDown,
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(ab),
                Arg::Ptr(rb),
                Arg::Ptr(wb.clone()),
                Arg::Word(hdr.name.clone(), 0),
                Arg::Word(hdr.name.clone(), 1),
                Arg::Word(hdr.name.clone(), 3),
                Arg::Word(hdr.name.clone(), 4),
                Arg::Word(hdr.name.clone(), 5),
                Arg::Word(hdr.name.clone(), 6),
                Arg::Int(i64::from(adapters)),
            ],
        );
        if let Some(Reduce::FabricSum { group }) = down.sum {
            let partial = cx.scratch("partial", w_words);
            cx.call(
                Kernel::Copy,
                vec![
                    Arg::Scratch(partial.clone(), "f32"),
                    Arg::Int(0),
                    Arg::Ptr(wb.clone()),
                    Arg::Int(0),
                    Arg::Int(w_words as i64),
                ],
            );
            cx.fabric_sum(&partial, &wb, w_words, (group, pes / group));
        }
        Ok(())
    })?;
    ctx.emit(&mut |cx| {
        let bb = cx.read(bank_b)?;
        let rb = cx.read(routes)?;
        let wb = cx.declare(&waist_name, Dtype::F32, rows, rank, true)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        let header = cx.unique("lanes");
        let reduce = up
            .sum
            .iter()
            .map(|r| (yb.name.clone(), r.clone()))
            .collect();
        let pes = up.pes;
        let plan = LanePlan {
            pes,
            kind: LaneKind::PerRow { lanes: rows },
            header,
            slots: rb.name.clone(),
            banks: vec![(bb.name.clone(), bank_b.width, BankSplit::Block)],
            block: Some(BlockPlan {
                a: out_width,
                b: rank,
                w: 1,
                a_groups: up.a_groups,
                b_groups: up.b_groups,
            }),
            lanes_per_pe: up.lanes_per_pe,
            a_first: false,
            row_outputs: Vec::new(),
            cols: vec![plane(&wb, vec![Segment::along_b()]), plane(&yb, vec![by_a])],
            reduce,
            window: None,
            pages: None,
        };
        let y_words = plan.plane_words(&plan.cols[1]);
        let hdr = cx.lanes(plan)?;
        cx.call(
            Kernel::LoraUp,
            vec![
                Arg::Ptr(wb),
                Arg::Ptr(bb),
                Arg::Ptr(rb),
                Arg::Ptr(yb.clone()),
                Arg::Word(hdr.name.clone(), 0),
                Arg::Word(hdr.name.clone(), 1),
                Arg::Word(hdr.name.clone(), 3),
                Arg::Word(hdr.name.clone(), 4),
                Arg::Word(hdr.name.clone(), 5),
                Arg::Word(hdr.name.clone(), 6),
                Arg::Int(i64::from(adapters)),
            ],
        );
        if let Some(Reduce::FabricSum { group }) = up.sum {
            let partial = cx.scratch("partial", y_words);
            cx.call(
                Kernel::Copy,
                vec![
                    Arg::Scratch(partial.clone(), "f32"),
                    Arg::Int(0),
                    Arg::Ptr(yb.clone()),
                    Arg::Int(0),
                    Arg::Int(y_words as i64),
                ],
            );
            cx.fabric_sum(&partial, &yb, y_words, (group, pes / group));
        }
        Ok(())
    })
}
