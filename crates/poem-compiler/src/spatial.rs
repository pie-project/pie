//! The spatial plan: a trace laid out over a device's cores at once, each
//! part of the model resident where it runs, rather than launched over time
//! ([`crate::temporal`]).
//!
//! The compiler's part is target-neutral. It reads the trace into a
//! [`SpatialGraph`]: blocks (runs of nodes of one layer that a target lays out
//! as one unit, of the kinds the target names) and the values that flow
//! between them. The target ([`SpatialTarget`]) says which kind of block each
//! node belongs to, proposes partitions (its own type: how many cores a block
//! takes, how its dims are cut, which layers share a region), refuses those
//! that do not fit, and costs the rest; [`plan`] keeps the cheapest that fits.
//! Coordinates, routes, kernels and memory placement are the target's, after
//! the plan. `SPATIAL.md` beside this crate says how targets map onto it.

use std::fmt::Debug;

use poem_ir::{Def, Operands, Trace, ValueId};

use crate::analysis::{Analysis, analyze};

/// A kind of block a target lays out, by the name the target gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockKind(pub &'static str);

/// One block of the model: its kind, its layer, the trace nodes it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub kind: BlockKind,
    pub layer: Option<u32>,
    pub nodes: Vec<u32>,
}

/// A value one block makes and another reads: a handoff between the cores
/// that hold them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flow {
    pub from: usize,
    pub to: usize,
    pub value: ValueId,
}

/// The trace as blocks and the flows between them.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialGraph {
    pub blocks: Vec<Block>,
    pub flows: Vec<Flow>,
    /// Each node's block.
    pub block_of: Vec<usize>,
    /// The blocks a collective of the trace sits in (a tensor-parallel trace).
    pub collectives: Vec<usize>,
}

/// What a spatial device offers the compiler.
pub trait SpatialTarget {
    /// How the target cuts a graph over its cores.
    type Partition: Clone + Debug;

    /// The kind of block `node` of `trace` belongs to, or `None` where the
    /// target has no block that runs it. Consecutive nodes of one layer and
    /// one kind form one block.
    fn classify(&self, trace: &Trace, node: u32) -> Option<BlockKind>;

    /// The partitions worth trying for `graph`, in the target's order.
    fn candidates(&self, trace: &Trace, graph: &SpatialGraph) -> Vec<Self::Partition>;

    /// Whether `partition` fits the device, or why not.
    fn fit(&self, graph: &SpatialGraph, partition: &Self::Partition) -> Result<(), String>;

    /// The predicted time a step of `partition` takes; lower is better.
    fn cost(&self, graph: &SpatialGraph, partition: &Self::Partition) -> f64;
}

/// The plan: the graph and the cheapest partition of it that fits.
#[derive(Clone, Debug)]
pub struct SpatialPlan<P> {
    pub analysis: Analysis,
    pub graph: SpatialGraph,
    pub partition: P,
    pub cost: f64,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpatialError {
    #[error(transparent)]
    Analysis(#[from] crate::Error),
    #[error("no block of the target runs node {node}")]
    Unplaced { node: u32 },
    #[error("the target proposes no partition")]
    NoCandidate,
    #[error("no partition fits: {}", .0.join("; "))]
    Unfit(Vec<String>),
}

impl SpatialGraph {
    /// `trace`'s blocks under `target`'s kinds and the flows between them.
    pub fn of(
        trace: &Trace,
        analysis: &Analysis,
        target: &impl SpatialTarget,
    ) -> Result<SpatialGraph, SpatialError> {
        let mut blocks: Vec<Block> = Vec::new();
        let mut block_of = Vec::with_capacity(trace.nodes.len());
        for at in 0..trace.nodes.len() as u32 {
            let kind = target
                .classify(trace, at)
                .ok_or(SpatialError::Unplaced { node: at })?;
            let layer = analysis.layers[at as usize];
            match blocks.last_mut() {
                Some(b) if b.kind == kind && b.layer == layer => b.nodes.push(at),
                _ => blocks.push(Block {
                    kind,
                    layer,
                    nodes: vec![at],
                }),
            }
            block_of.push(blocks.len() - 1);
        }
        let mut flows = Vec::new();
        let mut ins = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            ins.clear();
            node.op.inputs(&mut ins);
            let to = block_of[at];
            for &value in &ins {
                let Some(Def::Op(made)) = trace.values.get(value.0 as usize).map(|v| &v.def) else {
                    continue;
                };
                let from = block_of[*made as usize];
                let flow = Flow { from, to, value };
                if from != to && !flows.contains(&flow) {
                    flows.push(flow);
                }
            }
        }
        let collectives = analysis
            .collectives
            .iter()
            .map(|&at| block_of[at as usize])
            .collect();
        Ok(SpatialGraph {
            blocks,
            flows,
            block_of,
            collectives,
        })
    }
}

/// The spatial plan of `trace` on `target`: its analysis, its graph, and the
/// cheapest of the target's candidate partitions that fits (the first of
/// equal cost).
pub fn plan<T: SpatialTarget>(
    trace: &Trace,
    target: &T,
) -> Result<SpatialPlan<T::Partition>, SpatialError> {
    let analysis = analyze(trace)?;
    let graph = SpatialGraph::of(trace, &analysis, target)?;
    let candidates = target.candidates(trace, &graph);
    if candidates.is_empty() {
        return Err(SpatialError::NoCandidate);
    }
    let mut best: Option<(T::Partition, f64)> = None;
    let mut refusals = Vec::new();
    for p in candidates {
        if let Err(why) = target.fit(&graph, &p) {
            refusals.push(why);
            continue;
        }
        let cost = target.cost(&graph, &p);
        if best.as_ref().is_none_or(|(_, c)| cost < *c) {
            best = Some((p, cost));
        }
    }
    let (partition, cost) = best.ok_or(SpatialError::Unfit(refusals))?;
    Ok(SpatialPlan {
        analysis,
        graph,
        partition,
        cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::Build;
    use poem_ir::Guard;

    /// Even nodes are one kind, odd ones another; a partition is a core
    /// count, fitting up to 2 and faster the more cores.
    struct Toy;

    impl SpatialTarget for Toy {
        type Partition = u32;

        fn classify(&self, _trace: &Trace, node: u32) -> Option<BlockKind> {
            Some(BlockKind(if node.is_multiple_of(2) {
                "even"
            } else {
                "odd"
            }))
        }

        fn candidates(&self, _trace: &Trace, _graph: &SpatialGraph) -> Vec<u32> {
            vec![1, 2, 3]
        }

        fn fit(&self, _graph: &SpatialGraph, p: &u32) -> Result<(), String> {
            if *p <= 2 {
                Ok(())
            } else {
                Err(format!("{p} cores past 2"))
            }
        }

        fn cost(&self, graph: &SpatialGraph, p: &u32) -> f64 {
            graph.blocks.len() as f64 / f64::from(*p)
        }
    }

    #[test]
    fn a_chain_is_a_block_a_node_and_the_cheapest_fit_is_kept() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 8, Guard::Always);
        let z = b.op(y, 8, Guard::Always);
        let w = b.op(z, 8, Guard::Always);
        b.out(w);

        let plan = plan(&b.trace, &Toy).expect("plans");
        assert_eq!(plan.graph.blocks.len(), b.trace.nodes.len());
        assert_eq!(plan.graph.flows.len(), b.trace.nodes.len() - 1);
        assert_eq!(plan.partition, 2);
    }
}
