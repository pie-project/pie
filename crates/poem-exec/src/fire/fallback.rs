use core::ops::Range;

use crate::error::KernelError;
use poem_compiler::{CompiledModel, Fallback, Phase, PqTree, Region};
use poem_ir::{ClassSet, RowAxis};

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
        let xor = Guard::or(
            Guard::and(fact(0), Guard::not(fact(1))),
            Guard::and(Guard::not(fact(0)), fact(1)),
        );
        for axis in [fact(0), fact(1), xor] {
            let taken = b.op(v, 4, axis.clone());
            let other = b.op(v, 4, Guard::not(axis.clone()));
            v = b.merge(&[(taken, axis.clone()), (other, Guard::not(axis))], 4);
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
