use std::ops::Range;

use model_ir::{ClassSet, ClassTable, RowAxis, ValueId};

use crate::arena::{ArenaMap, Concurrency};
use crate::pq::PqTree;
use crate::stream::StreamPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledModel {
    pub classes: ClassTable,
    pub regions: Vec<Region>,
    pub order: ClassOrder,
    pub fallback: FallbackTable,
    pub arena: ArenaMap,
    pub concurrency: Concurrency,
    pub streams: StreamPlan,
    pub units: Vec<RowAxis>,
    pub units_of: Vec<u32>,
    pub patches: Option<AxisPlan>,
    pub voxels: Option<AxisPlan>,
    pub fold_refused: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AxisPlan {
    pub axis: RowAxis,
    pub order: ClassOrder,
    pub fallback: FallbackTable,
}

impl CompiledModel {
    #[must_use]
    pub fn template(&self) -> &[Region] {
        &self.regions
    }

    #[must_use]
    pub fn unit_of(&self, region: usize) -> u32 {
        self.units_of[region]
    }

    #[must_use]
    pub fn axis_of(&self, region: usize) -> RowAxis {
        self.units[self.unit_of(region) as usize]
    }

    #[must_use]
    pub fn order_for(&self, axis: RowAxis) -> Option<&ClassOrder> {
        match axis {
            RowAxis::Tokens => Some(&self.order),
            RowAxis::Patches => self.patches.as_ref().map(|plan| &plan.order),
            RowAxis::Voxels => self.voxels.as_ref().map(|plan| &plan.order),
        }
    }

    #[must_use]
    pub fn fallback_for(&self, axis: RowAxis) -> Option<&FallbackTable> {
        match axis {
            RowAxis::Tokens => Some(&self.fallback),
            RowAxis::Patches => self.patches.as_ref().map(|plan| &plan.fallback),
            RowAxis::Voxels => self.voxels.as_ref().map(|plan| &plan.fallback),
        }
    }

    #[must_use]
    pub fn unit_script(&self, unit: u32) -> Option<core::ops::Range<u32>> {
        let mut span: Option<core::ops::Range<u32>> = None;
        for (r, region) in self.regions.iter().enumerate() {
            if region.phase != Phase::Capture || self.unit_of(r) != unit {
                continue;
            }
            let r = r as u32;
            span = Some(match span {
                None => r..r + 1,
                Some(held) => held.start..r + 1,
            });
        }
        span
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Prepare,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lowering {
    AlwaysLaunch,
    Switch { merge: ValueId, arm: u8, arms: u8 },
    If,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub nodes: Range<u32>,
    pub mask: ClassSet,
    pub phase: Phase,
    pub axis: Option<RowAxis>,
    pub lowering: Lowering,
    pub stream: u32,
    pub wait: Vec<EventId>,
    pub open: Option<EventId>,
    pub close: Option<EventId>,
    pub collective: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ClassOrder {
    #[default]
    Identity,
    Seriated(PqTree),
}

impl ClassOrder {
    #[must_use]
    pub fn class_order(&self, present: &ClassSet) -> Vec<u8> {
        match self {
            ClassOrder::Identity => present.iter().map(|class| class as u8).collect(),
            ClassOrder::Seriated(tree) => tree
                .frontier()
                .iter()
                .copied()
                .filter(|&class| present.contains(class as usize))
                .collect(),
        }
    }

    #[must_use]
    pub fn tree(&self) -> Option<&PqTree> {
        match self {
            ClassOrder::Identity => None,
            ClassOrder::Seriated(tree) => Some(tree),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FallbackTable {
    pub rows: Vec<FallbackRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackRow {
    pub node: u32,
    pub buckets: Range<u32>,
    pub fallback: Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    View,
    Split { r: u32 },
    Grouped,
    Copy,
}

impl Region {
    #[must_use]
    pub fn launches(&self) -> bool {
        self.phase == Phase::Capture && !self.collective && !self.mask.is_empty()
    }

    #[must_use]
    pub fn windowed(&self, all: usize) -> bool {
        self.launches() && self.mask.len() < all
    }
}
