//! Routed experts in the unit regime (`stream::moe`): the gated sum of every expert, lowered into
//! steps every unit already runs (the router a printed region, the gate spread over each
//! expert's columns by a constant matvec, the banks' gate, up and down one matvec each as a
//! weight view lays them, the gated activation a region). No CSL of its own.

use std::collections::BTreeMap;

use poem_ir::{Dtype, Linear, Operation, ValueId};

use crate::compile::stream::unit::{Cols, Lowering, Side, Step, View, side_of};
use crate::kernel::{Kernel, R, Source, UnitKernel};

/// Routed experts.
pub struct Moe;

/// The registry's routed experts.
pub static MOE: Moe = Moe;

impl Kernel for Moe {
    fn name(&self) -> &'static str {
        "linear.moe"
    }

    fn ops(&self) -> &'static [&'static str] {
        &["linear.moe_route", "linear.moe_select"]
    }

    fn sources(&self) -> &'static [Source] {
        &[]
    }

    fn unit(&self) -> Option<&dyn UnitKernel> {
        Some(self)
    }
}

impl UnitKernel for Moe {
    /// Routed experts: their gated sum over every expert, as matvecs and regions.
    fn lower(&self, cx: &mut Lowering, at: u32) -> R<bool> {
        let name = cx.name;
        let refuse = |at: u32, what: String| format!("{name}: node {at}: {what}");
        let Lowering {
            r,
            nodes,
            steps,
            sides,
            skip,
            routed,
            ..
        } = cx;
        let r = *r;
        let nodes: &[u32] = nodes;
        let trace = r.trace;
        match &trace.nodes[at as usize].op {
            // routed experts: their gated sum over every expert ([`crate::compile::stream::moe`])
            // (lowered where its first select comes: its input made by then)
            Operation::Linear(Linear::MoeRoute { .. }) => {
                let g = crate::compile::stream::moe::group(trace, at, nodes)
                    .map_err(|e| refuse(at, e))?;
                skip.extend(g.nodes[2..].iter().copied());
                routed.insert(g.nodes[1], g);
            }
            Operation::Linear(Linear::MoeSelect { .. }) if routed.contains_key(&at) => {
                use crate::compile::stream::moe::{Kind, synth};
                let g = routed.remove(&at).unwrap_or_else(|| unreachable!());
                let at = g.route;
                let l = r.cols(g.logits);
                let gs = side_of(sides, l).ok_or_else(|| {
                    refuse(at, "linear.moe_route reads logits no root has".into())
                })?;
                let (e, ei, h) = (g.experts, g.experts * g.inter, g.hidden);
                let gate = synth(at, Kind::Gate);
                steps.push(Step::Region {
                    side: gs,
                    chain: crate::compile::stream::moe::router(trace, &g)
                        .map_err(|e| refuse(at, e))?,
                });
                sides.insert(gate, gs);
                // the experts' columns over the column roots (the most PEs), from the rows: the
                // gate and the layer's input on the rows, moved there where they are not
                let (rows, o) = (Side::Row, Side::Col);
                let to_rows = |steps: &mut Vec<Step>,
                               sides: &mut BTreeMap<ValueId, Side>,
                               c: Cols,
                               from: Side,
                               kind: Kind,
                               x32: bool,
                               round: bool| {
                    if from == rows {
                        return c;
                    }
                    let moved = synth(at, kind);
                    steps.push(Step::Matvec {
                        node: at,
                        from,
                        input: c,
                        weight: ValueId(crate::compile::stream::host::UNIT),
                        view: View::Plain,
                        output: moved,
                        k: c.width,
                        n: c.width,
                        x32,
                        round,
                    });
                    sides.insert(moved, rows);
                    Cols {
                        base: moved,
                        from: 0,
                        width: c.width,
                    }
                };
                let gin = to_rows(
                    steps,
                    sides,
                    Cols {
                        base: gate,
                        from: 0,
                        width: e,
                    },
                    gs,
                    Kind::GateRows,
                    true,
                    false,
                );
                let spread = synth(at, Kind::Spread);
                steps.push(Step::Matvec {
                    node: at,
                    from: rows,
                    input: gin,
                    weight: ValueId(crate::compile::stream::host::UNIT),
                    view: View::Spread { rep: g.inter },
                    output: spread,
                    k: e,
                    n: ei,
                    x32: true,
                    round: false,
                });
                sides.insert(spread, o);
                let xc = r.cols(g.x);
                let xs = side_of(sides, xc).ok_or_else(|| {
                    refuse(at, "linear.moe_select reads a value no root has".into())
                })?;
                let xin = to_rows(
                    steps,
                    sides,
                    xc,
                    xs,
                    Kind::Input,
                    r.dtype(g.x) == Some(Dtype::F32),
                    r.round(g.x),
                );
                let narrow = r.dtype(synth(at, Kind::Gates)).unwrap_or(Dtype::Bf16);
                // (a non-gated activation's experts: their ups alone)
                let banks: Vec<(Kind, (ValueId, u32, u32))> = g
                    .gate
                    .map(|gt| (Kind::Gates, gt))
                    .into_iter()
                    .chain([(Kind::Ups, g.up)])
                    .collect();
                for (kind, (bank, stride, off)) in banks {
                    steps.push(Step::Matvec {
                        node: at,
                        from: rows,
                        input: xin,
                        weight: bank,
                        view: View::Rows {
                            block: g.inter,
                            stride,
                            off,
                        },
                        output: synth(at, kind),
                        k: h,
                        n: ei,
                        x32: r.dtype(g.x) == Some(Dtype::F32),
                        round: true,
                    });
                    sides.insert(synth(at, kind), o);
                }
                steps.push(Step::Region {
                    side: o,
                    chain: crate::compile::stream::moe::activation(&g, narrow)
                        .map_err(|e| refuse(at, e))?,
                });
                let acts = synth(at, Kind::Acts);
                sides.insert(acts, o);
                steps.push(Step::Matvec {
                    node: at,
                    from: o,
                    input: Cols {
                        base: acts,
                        from: 0,
                        width: ei,
                    },
                    weight: g.down,
                    view: View::Blocks { block: g.inter },
                    output: g.y,
                    k: ei,
                    n: h,
                    x32: false,
                    round: r.round(g.y),
                });
                sides.insert(g.y, rows);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}
