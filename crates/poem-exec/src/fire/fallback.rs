use core::ops::Range;

use crate::error::KernelError;
use poem_compiler::{CompiledModel, Fallback, Phase, PqTree, Region};
use poem_ir::{
    CacheRow, ClassSet, Def, Dim, Dtype, GeomKind, Node, Operands, RowAxis, RuntimeInput, Trace,
    Ty, ValueId,
};

#[must_use]
pub fn answers(compiled: &CompiledModel, axis: RowAxis, nodes: Range<u32>) -> Vec<Fallback> {
    let Some(table) = compiled.fallback_for(axis) else {
        return Vec::new();
    };
    let mut found: Vec<Fallback> = Vec::new();
    for row in &table.rows {
        if !nodes.contains(&row.node) || found.contains(&row.fallback) {
            continue;
        }
        found.push(row.fallback);
    }
    found
}

#[must_use]
fn answer_at(
    compiled: &CompiledModel,
    axis: RowAxis,
    nodes: Range<u32>,
    bucket: u32,
) -> Option<Fallback> {
    compiled
        .fallback_for(axis)?
        .rows
        .iter()
        .find(|row| nodes.contains(&row.node) && row.buckets.contains(&bucket))
        .map(|row| row.fallback)
}

#[must_use]
pub fn copies(compiled: &CompiledModel, axis: RowAxis, mask: &ClassSet, bucket: u32) -> bool {
    compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(at, region)| &region.mask == mask && compiled.axis_of(*at) == axis)
        .any(|(_, region)| {
            answer_at(compiled, axis, region.nodes.clone(), bucket) == Some(Fallback::Copy)
        })
}

pub trait Serve {
    fn copies(&self, _region: &Region) -> bool {
        false
    }

    fn gather(&mut self, _region: &Region) -> Result<(), KernelError> {
        Err(unserved("gather"))
    }

    fn scatter(&mut self, _region: &Region) -> Result<(), KernelError> {
        Err(unserved("scatter"))
    }
}

fn unserved(half: &'static str) -> KernelError {
    KernelError::Backend {
        op: "fallback.copy",
        detail: format!("this backend answered `Serve::copies` but publishes no row {half}"),
    }
}

#[must_use]
pub fn grouped(compiled: &CompiledModel, axis: RowAxis, nodes: Range<u32>) -> bool {
    compiled.fallback_for(axis).is_some_and(|table| {
        table
            .rows
            .iter()
            .any(|row| nodes.contains(&row.node) && row.fallback == Fallback::Grouped)
    })
}

#[must_use]
pub fn bound(compiled: &CompiledModel, axis: RowAxis, mask: &ClassSet) -> u32 {
    let Some(order) = compiled.order_for(axis) else {
        return 1;
    };
    let classes = compiled.classes.classes.len();
    let order = order.class_order(&ClassSet::of(0..classes));
    let mask: Vec<u8> = mask.iter().map(|class| class as u8).collect();
    PqTree::runs(&order, &mask).max(1)
}

#[must_use]
pub fn promised(compiled: &CompiledModel, axis: RowAxis, region: &Region) -> bool {
    region.phase == Phase::Capture && answers(compiled, axis, region.nodes.clone()).is_empty()
}

#[must_use]
pub fn fragmentable(compiled: &CompiledModel) -> usize {
    let mut seen: Vec<&ClassSet> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        if bound(compiled, compiled.axis_of(at), &region.mask) > 1 && !seen.contains(&&region.mask)
        {
            seen.push(&region.mask);
        }
    }
    seen.len()
}

#[must_use]
pub fn max_runs(compiled: &CompiledModel) -> u32 {
    compiled
        .template()
        .iter()
        .enumerate()
        .map(|(at, region)| bound(compiled, compiled.axis_of(at), &region.mask))
        .max()
        .unwrap_or(1)
        .max(1)
}

#[must_use]
pub fn operands(nodes: &[Node], region: &Region) -> Option<(Vec<ValueId>, Vec<ValueId>)> {
    let mut ins: Vec<ValueId> = Vec::new();
    let mut outs: Vec<ValueId> = Vec::new();
    for node in region.nodes.clone() {
        let node = nodes.get(node as usize)?;
        node.op.inputs(&mut ins);
        node.op.outputs(&mut outs);
    }
    Some((ins, outs))
}

/// What an engine's copy of a window carries beside the token rows of bf16
/// and f32 it gathers and scatters back ([`copyable_mask`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Carries {
    /// Each gathered row's mask (`RuntimeInput::Mask`), gathered with its
    /// rows: the fire's masks are per row.
    pub masks: bool,
    /// Token rows of integers a region only reads (an adapter's routes),
    /// gathered and never scattered back.
    pub read_ints: bool,
}

/// Whether `region` can run over a gathered copy of its lanes' rows: every
/// operand a kv row, a per-lane page geometry, a struct, a constant, or a
/// tensor of token rows the copy carries (`carries`).
fn copyable(trace: &Trace, region: &Region, carries: Carries) -> bool {
    let Some((ins, outs)) = operands(&trace.nodes, region) else {
        return false;
    };
    ins.iter()
        .map(|id| (id, true))
        .chain(outs.iter().map(|id| (id, false)))
        .all(|(id, read)| copyable_operand(trace, *id, read, carries))
}

fn copyable_operand(trace: &Trace, id: poem_ir::ValueId, read: bool, carries: Carries) -> bool {
    let Some(decl) = trace.values.get(id.0 as usize) else {
        return false;
    };
    match &decl.def {
        Def::Cache(c) => matches!(trace.caches.get(*c as usize), Some(CacheRow::Kv { .. })),
        Def::Input(RuntimeInput::Geometry { kind }) => matches!(
            kind,
            GeomKind::Indptr | GeomKind::Indices | GeomKind::LastPageLen | GeomKind::KvLen
        ),
        Def::Input(RuntimeInput::Mask) => carries.masks,
        _ => match &decl.ty {
            Ty::Struct(_) => true,
            Ty::Tensor { shape, dtype } => match shape.first() {
                Some(Dim::Tokens) => {
                    matches!(dtype, Dtype::Bf16 | Dtype::F32)
                        || (read && carries.read_ints && matches!(dtype, Dtype::I32))
                }
                Some(Dim::TokensTimes(_)) => false,
                Some(Dim::Const(_)) | None => true,
                Some(Dim::Lanes | Dim::LanesPlus(_) | Dim::Capturing) => false,
                Some(Dim::ContextTimes(_) | Dim::ContextPages) => false,
                Some(Dim::Readouts) => false,
                Some(Dim::Patches | Dim::Images | Dim::ImagesPlus(_)) => false,
                Some(Dim::Voxels | Dim::VoxelsScaled(..) | Dim::Clips | Dim::ClipsPlus(_)) => false,
            },
        },
    }
}

#[must_use]
/// Whether every region of class set `mask` is [`copyable`] over a copy
/// carrying `carries`.
pub fn copyable_mask(
    trace: &Trace,
    compiled: &CompiledModel,
    mask: &ClassSet,
    carries: Carries,
) -> bool {
    compiled
        .template()
        .iter()
        .filter(|region| &region.mask == mask)
        .all(|region| copyable(trace, region, carries))
}

#[must_use]
/// The class sets whose copies gather rows ([`copyable_mask`], `carries` as
/// there).
pub fn gathers(trace: &Trace, compiled: &CompiledModel, carries: Carries) -> usize {
    let mut sets: Vec<&ClassSet> = Vec::new();
    for region in compiled.template() {
        let owed = compiled
            .fallback
            .rows
            .iter()
            .any(|row| region.nodes.contains(&row.node) && row.fallback == Fallback::Copy);
        if !owed || sets.contains(&&region.mask) {
            continue;
        }
        if copyable_mask(trace, compiled, &region.mask, carries) {
            sets.push(&region.mask);
        }
    }
    sets.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::fixture::{Build, fact};
    use poem_compiler::{Budget, DeviceProfile, compile};
    use poem_ir::Guard;

    fn crossing() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let mut v = b.op(x, 4, Guard::Always);
        let xor = Guard::or(Guard::and(fact(0), !fact(1)), Guard::and(!fact(0), fact(1)));
        for axis in [fact(0), fact(1), xor] {
            let taken = b.op(v, 4, axis.clone());
            let other = b.op(v, 4, !axis.clone());
            v = b.merge(&[(taken, axis.clone()), (other, !axis)], 4);
        }
        let y = b.op(v, 4, Guard::Always);
        b.out(y);
        b
    }

    #[test]
    fn the_bound_is_the_run_count_p4_measured_on_the_order_it_shipped() {
        let b = crossing();
        let wide = Budget {
            max_lanes: 8,
            max_tokens: 4096,
            buckets: vec![64, 4096],
            max_adapters: 0,
            readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
        };
        let compiled =
            compile(&b.trace, &wide, &DeviceProfile::default()).expect("the fixture bakes");

        let mut checked = 0;
        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            let stated = answers(&compiled, axis, region.nodes.clone())
                .into_iter()
                .find_map(|answer| match answer {
                    Fallback::Split { r } => Some(r),
                    _ => None,
                });
            if let Some(stated) = stated {
                assert_eq!(
                    bound(&compiled, axis, &region.mask),
                    stated,
                    "{:?}",
                    region.nodes
                );
                assert!(
                    stated > 1,
                    "a withdrawn consumer costs more than one launch"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "the fixture withdraws at least one consumer");

        for (at, region) in compiled.template().iter().enumerate() {
            let axis = compiled.axis_of(at);
            if promised(&compiled, axis, region) {
                assert_eq!(
                    bound(&compiled, axis, &region.mask),
                    1,
                    "{:?}",
                    region.nodes
                );
            }
        }
    }
}
