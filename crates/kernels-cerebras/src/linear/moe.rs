//! Mixture-of-experts routing: the top-k softmax router (a row a PE), the
//! routed matmul `y[t*k + s] = x[row]
//! · bank[routes[t, s]]ᵀ` (a lane plan over the routed pairs with the
//! routes as its slot table), the weighted sum of the routed rows, and the
//! sigmoid gate that adds the shared expert.

use dtype::Dtype;

use crate::csl::Arg;
use crate::cx::{Ctx, expect, tile_split};
use crate::error::{Error, refuse};
use crate::linear::gemm::{ARRAY_WORDS, pe_words};
use crate::program::{
    BankSplit, BlockPlan, ColPlane, HostOp, LaneKind, LanePlan, Reduce, Segment, Shard,
};
use crate::tensor::Tensor;

/// `routes[t, s]`, `weights[t, s]` for `s < top_k`: the `top_k` largest of
/// the `experts` logits of row `t` (ties to the lowest index, NaN never
/// picked) and the softmax over those; picks past the row's experts hold
/// route `-1` and weight 0.
pub fn topk_softmax(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_softmax";
    expect(OP, logits, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, routes, &[Dtype::I32])?;
    expect(OP, weights, &[Dtype::F32])?;
    if experts == 0 || top_k == 0 || logits.width < experts {
        return Err(refuse(
            OP,
            format!(
                "{} logits a row for {experts} experts, {top_k} picked",
                logits.width
            ),
        ));
    }
    if routes.rows != logits.rows
        || routes.width != top_k
        || weights.rows != logits.rows
        || weights.width != top_k
    {
        return Err(refuse(
            OP,
            format!(
                "routes {}x{} and weights {}x{} for {} rows of {top_k}",
                routes.rows, routes.width, weights.rows, weights.width, logits.rows
            ),
        ));
    }
    // Rows over PEs; a row's logits and its picks on one.
    let (rg, _) = tile_split(
        OP,
        logits.rows,
        1,
        u64::from(logits.width) + 2 * u64::from(top_k),
        0,
        0,
        1,
    )?;
    let rows_local = logits.rows / rg;
    let (width, lrows) = (logits.width, logits.rows);
    ctx.emit(&mut |cx| {
        let lb = cx.read(logits)?;
        let rb = cx.write(routes)?;
        let wb = cx.write(weights)?;
        if rg > 1 {
            cx.over(rg);
            let tile = Shard::Tile {
                rows: rg,
                cols: 1,
                segments: 1,
            };
            cx.shard(&lb, tile);
            cx.shard(&rb, tile);
            cx.shard(&wb, tile);
        }
        let _ = lrows;
        cx.library("k_topk_softmax");
        cx.call(
            "k_topk_softmax",
            vec![
                Arg::Ptr(lb),
                Arg::Ptr(rb),
                Arg::Ptr(wb),
                Arg::Int(i64::from(rows_local)),
                Arg::Int(i64::from(width)),
                Arg::Int(i64::from(experts)),
                Arg::Int(i64::from(top_k)),
            ],
        );
        Ok(())
    })
}

/// `y[t * top_k + s] = x[row] · bank[routes[t, s]]ᵀ` for every routed pair,
/// `x` holding one row per token (`row = t`) or one per pair (`row = t *
/// top_k + s`); a negative route leaves its row of `y` as it was. `bank` is
/// `[experts, n * k]`, each expert `[n][k]` row-major.
/// `routes[row, slot] = slot`: the identity routing a grouped matmul reads.
pub fn group_routes(ctx: &Ctx<'_>, groups: u32, routes: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    expect(OP, routes, &[Dtype::I32])?;
    if groups == 0 || routes.width != groups {
        return Err(refuse(
            OP,
            format!("the routes are {} wide for {groups} groups", routes.width),
        ));
    }
    let rows = routes.rows;
    let (rg, _) = crate::cx::tile_split(OP, rows, groups, u64::from(groups), 0, 0, groups)?;
    ctx.emit(&mut |cx| {
        let rb = cx.write(routes)?;
        if rg > 1 {
            cx.over(rg);
            cx.shard(&rb, Shard::Rows(rg));
        }
        cx.library("k_group_routes");
        cx.call(
            "k_group_routes",
            vec![Arg::Ptr(rb), Arg::Int(i64::from(rows / rg)), Arg::Int(i64::from(groups))],
        );
        Ok(())
    })
}

/// `x [rows, groups·K]` split into `groups` slices, slice `g` of row `r`
/// multiplied by expert `routes[r, g]` of `w` (`experts` blocks of `N × K`
/// rows), into `y [rows, groups·N]`. The expert bank is the size of a
/// projection over every stream, so the host multiplies.
pub fn matmul_grouped(
    ctx: &Ctx<'_>,
    x: Tensor,
    w: Tensor,
    routes: Tensor,
    groups: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_grouped";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, w, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, routes, &[Dtype::I32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if groups == 0
        || !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || routes.width != groups
        || routes.rows != x.rows
        || y.rows != x.rows
    {
        return Err(refuse(
            OP,
            format!(
                "{groups} groups do not divide a {}x{} row into a {}x{} one, or the routes are {}x{}",
                x.rows, x.width, y.rows, y.width, routes.rows, routes.width
            ),
        ));
    }
    let (k, n) = (x.width / groups, y.width / groups);
    let per = u64::from(k) * u64::from(n);
    if k == 0 || n == 0 || !w.elements().is_multiple_of(per) {
        return Err(refuse(
            OP,
            format!(
                "a {}-element bank is not whole {n}x{k} experts",
                w.elements()
            ),
        ));
    }
    let experts = u32::try_from(w.elements() / per)
        .map_err(|_| refuse(OP, "the bank holds more experts than a route can name"))?;
    let rows = x.rows;
    // Row `r`'s slice `g` is routed pair `r · groups + g`: the routed matmul
    // over the pairs (the planes viewed a pair a row, which their row-major
    // words already are) runs it on the fabric; a bank no lane plan holds
    // falls back to the host.
    let pairs = rows * groups;
    let fabric = select_impl(
        ctx,
        OP,
        x,
        w,
        routes,
        y,
        Some([(pairs, k), (experts, n * k), (pairs, 1), (pairs, n)]),
    );
    if fabric.is_ok() {
        return fabric;
    }
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let wb = cx.read(w)?;
        let rb = cx.read(routes)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        cx.host(HostOp::MatmulGrouped {
            x: xb.name,
            w: wb.name,
            routes: rb.name,
            y: yb.name,
            rows,
            groups,
            k,
            n,
            experts,
        });
        Ok(())
    })
}

pub fn matmul_select(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    select_impl(ctx, "linear.moe_matmul_select", x, bank, routes, y, None)
}

/// A plane's extent as a phase sees it: the tensor's, or a view's.
#[derive(Clone, Copy)]
struct Dims {
    rows: u32,
    width: u32,
}

impl Dims {
    fn of(t: Tensor, view: Option<(u32, u32)>) -> Dims {
        match view {
            Some((rows, width)) => Dims { rows, width },
            None => Dims {
                rows: t.rows,
                width: t.width,
            },
        }
    }

    fn elements(self) -> u64 {
        u64::from(self.rows) * u64::from(self.width)
    }
}

/// `matmul_select` over the planes as `shapes` views them (`x`, `bank`,
/// `routes`, `y`; row-major reshapes of the same words), when given.
fn select_impl(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
    shapes: Option<[(u32, u32); 4]>,
) -> Result<(), Error> {
    let op_name = op;
    let (xd, bd, rd, yd) = (
        Dims::of(x, shapes.map(|s| s[0])),
        Dims::of(bank, shapes.map(|s| s[1])),
        Dims::of(routes, shapes.map(|s| s[2])),
        Dims::of(y, shapes.map(|s| s[3])),
    );
    expect(op_name, x, &[Dtype::Bf16, Dtype::F32])?;
    expect(op_name, bank, &[Dtype::Bf16, Dtype::F32])?;
    expect(op_name, routes, &[Dtype::I32])?;
    expect(op_name, y, &[Dtype::Bf16, Dtype::F32])?;
    let (k, n) = (xd.width, yd.width);
    let top_k = rd.width;
    let pairs = rd.elements();
    if top_k == 0 || k == 0 || n == 0 {
        return Err(refuse(op_name, "the fan-out, K and N are nonzero"));
    }
    let per_pair = if u64::from(xd.rows) == pairs {
        true
    } else if xd.rows == rd.rows {
        false
    } else {
        return Err(refuse(
            op_name,
            format!(
                "the activation's {} rows are neither the tokens nor the routed pairs",
                xd.rows
            ),
        ));
    };
    if u64::from(yd.rows) != pairs {
        return Err(refuse(
            op_name,
            format!("the result has {} rows for {pairs} routed pairs", yd.rows),
        ));
    }
    let per = u64::from(n) * u64::from(k);
    if bd.elements() == 0
        || !bd.elements().is_multiple_of(per)
        || bd.width != 0 && u64::from(bd.width) != per && bd.rows != 1
    {
        return Err(refuse(
            op_name,
            format!(
                "a {}x{} bank is not whole {n}x{k} experts",
                bd.rows, bd.width
            ),
        ));
    }
    let experts = (bd.elements() / per) as u32;
    let stride = u32::try_from(per).map_err(|_| refuse(op_name, "an expert past u32 words"))?;
    let xdiv = if per_pair { 1 } else { top_k };
    let lanes = pairs as u32;
    let whole = bd.elements() + xd.elements() + yd.elements();
    if whole <= pe_words() && bd.elements() <= ARRAY_WORDS {
        return ctx.emit(&mut |cx| {
            let xb = cx.read(x)?;
            let xb = cx.view_as(&xb, xd.rows, xd.width)?;
            let bb = cx.read(bank)?;
            let bb = cx.view_as(&bb, bd.rows, bd.width)?;
            let rb = cx.read(routes)?;
            let rb = cx.view_as(&rb, rd.rows, rd.width)?;
            cx.read(y)?;
            let yb = cx.write(y)?;
            let yb = cx.view_as(&yb, yd.rows, yd.width)?;
            cx.library("k_moe_select");
            cx.call(
                "k_moe_select",
                vec![
                    Arg::Ptr(xb),
                    Arg::Ptr(bb),
                    Arg::Ptr(rb),
                    Arg::Ptr(yb),
                    Arg::Int(0),
                    Arg::Int(i64::from(lanes)),
                    Arg::Int(0),
                    Arg::Int(i64::from(n)),
                    Arg::Int(0),
                    Arg::Int(i64::from(k)),
                    Arg::Int(i64::from(xdiv)),
                    Arg::Int(i64::from(n)),
                    Arg::Int(i64::from(k)),
                    Arg::Int(i64::from(experts)),
                    Arg::Bool(false),
                ],
            );
            Ok(())
        });
    }
    // A lane plan over the pairs: a PE holds a block `[o0, o1) × [c0, c1)`
    // of its lanes' experts, that slice of every x row (a column plane)
    // and its columns of y, a partial over the slices the host adds.
    let budget = pe_words();
    let divisors = |v: u32| (1..=v).filter(move |d| v.is_multiple_of(*d));
    let mut best: Option<(u32, u32, u32, u32)> = None;
    let fabric = crate::linear::gemm::fabric_sum();
    for kg in divisors(k) {
        for og in divisors(n) {
            let block = u64::from(n / og) * u64::from(k / kg);
            let y_plane = u64::from(yd.rows) * u64::from(n / og);
            let mut planes = u64::from(xd.rows) * u64::from(k / kg) + y_plane;
            if kg > 1 && fabric {
                // The partial's copy and the collectives library.
                planes += y_plane + crate::linear::gemm::FABRIC_SUM_WORDS;
            }
            if block + planes > budget || block > ARRAY_WORDS {
                continue;
            }
            let most = ((budget - planes) / block.max(1))
                .min(ARRAY_WORDS / block.max(1))
                .clamp(1, u64::from(lanes)) as u32;
            let per_pe = (1..=most)
                .rev()
                .find(|d| lanes.is_multiple_of(*d))
                .unwrap_or(1);
            let pes = (lanes / per_pe) * og * kg;
            if best.is_none_or(|(p, ..)| pes < p) {
                best = Some((pes, per_pe, og, kg));
            }
            break;
        }
    }
    let (pes, lanes_per_pe, og, kg) = best.ok_or_else(|| {
        refuse(
            op_name,
            format!("one row of an expert ({k} words) and a row of x and y exceed a PE's {budget}"),
        )
    })?;
    ctx.emit(&mut |cx| {
        let xb = cx.read(x)?;
        let xb = cx.view_as(&xb, xd.rows, xd.width)?;
        let bb = cx.read(bank)?;
        let bb = cx.view_as(&bb, bd.rows, bd.width)?;
        let rb = cx.read(routes)?;
        let rb = cx.view_as(&rb, rd.rows, rd.width)?;
        cx.read(y)?;
        let yb = cx.write(y)?;
        let yb = cx.view_as(&yb, yd.rows, yd.width)?;
        let header = cx.unique("lanes");
        let reduce = if kg > 1 && fabric {
            vec![(yb.name.clone(), Reduce::FabricSum { group: kg })]
        } else if kg > 1 {
            vec![(yb.name.clone(), Reduce::Sum)]
        } else {
            Vec::new()
        };
        let plan = LanePlan {
            pes,
            kind: LaneKind::PerRow { lanes },
            header,
            slots: rb.name.clone(),
            banks: vec![(bb.name.clone(), stride, BankSplit::Block)],
            block: Some(BlockPlan {
                a: n,
                b: k,
                w: 1,
                a_groups: og,
                b_groups: kg,
            }),
            lanes_per_pe,
            a_first: false,
            row_outputs: Vec::new(),
            cols: vec![
                ColPlane {
                    name: xb.name.clone(),
                    rows: xb.rows,
                    width: xb.width,
                    by_rows: false,
                    segments: vec![Segment::along_b()],
                    row_block: false,
                },
                ColPlane {
                    name: yb.name.clone(),
                    rows: yb.rows,
                    width: yb.width,
                    by_rows: false,
                    segments: vec![Segment {
                        base: 0,
                        a_group: 1,
                        a_stride: 1,
                        b_stride: 0,
                        span: 1,
                    }],
                    row_block: false,
                },
            ],
            reduce,
            window: None,
            pages: None,
        };
        let y_words = plan.plane_words(&plan.cols[1]);
        let hdr = cx.lanes(plan)?;
        cx.library("k_moe_select");
        cx.call(
            "k_moe_select",
            vec![
                Arg::Ptr(xb),
                Arg::Ptr(bb),
                Arg::Ptr(rb),
                Arg::Ptr(yb.clone()),
                Arg::Word(hdr.name.clone(), 0),
                Arg::Word(hdr.name.clone(), 1),
                Arg::Word(hdr.name.clone(), 3),
                Arg::Word(hdr.name.clone(), 4),
                Arg::Word(hdr.name.clone(), 5),
                Arg::Word(hdr.name.clone(), 6),
                Arg::Int(i64::from(xdiv)),
                Arg::Int(i64::from(n)),
                Arg::Int(i64::from(k)),
                Arg::Int(i64::from(lanes_per_pe)),
                Arg::Bool(true),
            ],
        );
        if kg > 1 && fabric {
            // The PE's y is its partial: copied aside, then the depth
            // group's copies are added into the group's first PE's y.
            let partial = cx.scratch("partial", y_words);
            cx.call(
                "k_copy",
                vec![
                    Arg::Scratch(partial.clone(), "f32"),
                    Arg::Int(0),
                    Arg::Ptr(yb.clone()),
                    Arg::Int(0),
                    Arg::Int(y_words as i64),
                ],
            );
            cx.fabric_sum_with(&partial, &yb, y_words, (kg, pes / kg), crate::linear::gemm::resident_pes().is_some());
        }
        Ok(())
    })
}

/// `y[t] = Σ_s weights[t, s] · routed[t * top_k + s]`, in f32.
pub fn weighted_sum(
    ctx: &Ctx<'_>,
    routed: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
    expect(OP, routed, &[Dtype::Bf16, Dtype::F32])?;
    expect(OP, weights, &[Dtype::F32])?;
    expect(OP, y, &[Dtype::Bf16, Dtype::F32])?;
    if y.rows == 0 || !routed.rows.is_multiple_of(y.rows) || routed.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the routed {}x{} rectangle does not fold into {}x{}",
                routed.rows, routed.width, y.rows, y.width
            ),
        ));
    }
    let top_k = routed.rows / y.rows;
    if weights.rows != y.rows || weights.width != top_k {
        return Err(refuse(
            OP,
            format!(
                "the weights are {}x{} for fan-out {top_k}",
                weights.rows, weights.width
            ),
        ));
    }
    let (rows, width) = (y.rows, y.width);
    let (rg, cg) = tile_split(
        OP,
        rows,
        width,
        u64::from(width) * u64::from(top_k + 1) + u64::from(top_k),
        0,
        0,
        1,
    )?;
    let pes = rg * cg;
    ctx.emit(&mut |cx| {
        let rb = cx.read(routed)?;
        let wb = cx.read(weights)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            cx.shard(
                &rb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
            cx.shard(
                &yb,
                Shard::Tile {
                    rows: rg,
                    cols: cg,
                    segments: 1,
                },
            );
            cx.shard(
                &wb,
                Shard::RowsBy {
                    parts: rg,
                    period: cg,
                },
            );
        }
        cx.library("k_moe_weighted_sum");
        cx.call(
            "k_moe_weighted_sum",
            vec![
                Arg::Ptr(rb),
                Arg::Ptr(wb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(top_k)),
                Arg::Int(i64::from(width / cg)),
            ],
        );
        Ok(())
    })
}

/// `y = routed + σ(gate[row]) · shared`.
pub fn sigmoid_gate_add(
    ctx: &Ctx<'_>,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    for t in [routed, shared, gate, y] {
        expect(OP, t, &[Dtype::Bf16, Dtype::F32])?;
    }
    if shared.rows != routed.rows
        || shared.width != routed.width
        || gate.elements() != u64::from(routed.rows)
        || y.rows != routed.rows
        || y.width != routed.width
    {
        return Err(refuse(
            OP,
            "the routed, shared, gate and result planes do not agree",
        ));
    }
    let (rows, width) = (y.rows, y.width);
    let (rg, cg) = tile_split(OP, rows, width, 3 * u64::from(width) + 1, 0, 0, 1)?;
    let pes = rg * cg;
    ctx.emit(&mut |cx| {
        let rb = cx.read(routed)?;
        let sb = cx.read(shared)?;
        let gb = cx.read(gate)?;
        let yb = cx.write(y)?;
        if pes > 1 {
            cx.over(pes);
            for b in [&rb, &sb, &yb] {
                cx.shard(
                    b,
                    Shard::Tile {
                        rows: rg,
                        cols: cg,
                        segments: 1,
                    },
                );
            }
            cx.shard(
                &gb,
                Shard::RowsBy {
                    parts: rg,
                    period: cg,
                },
            );
        }
        cx.library("k_moe_gate_add");
        cx.call(
            "k_moe_gate_add",
            vec![
                Arg::Ptr(rb),
                Arg::Ptr(sb),
                Arg::Ptr(gb),
                Arg::Ptr(yb),
                Arg::Int(i64::from(rows / rg)),
                Arg::Int(i64::from(width / cg)),
            ],
        );
        Ok(())
    })
}
