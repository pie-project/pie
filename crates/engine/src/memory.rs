use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryPlan {
    pub working_set: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub scratch: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

static PLAN: RwLock<Option<MemoryPlan>> = RwLock::new(None);

pub fn publish(plan: MemoryPlan) {
    if let Ok(mut slot) = PLAN.write() {
        *slot = Some(plan);
    }
}

#[must_use]
pub fn latest() -> Option<MemoryPlan> {
    PLAN.read().ok().and_then(|slot| *slot)
}
