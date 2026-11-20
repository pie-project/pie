//! The device's account of where each live guest instance's slots lie: the
//! run its tables take and its heap, by segment and byte range. The runtime
//! hands each Instance slot ref to one instance; a step writes its heap
//! (the one write a guest's data steers) and reads tables the host wrote, so
//! were one ref ever handed out twice, one guest's heap writes could land in
//! another's tables. A bind claims its runs here and is refused by name
//! where one overlaps a run another live instance holds; an evacuation or a
//! park moves the runs and claims them again; a closed instance releases
//! them.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Fault, Result};

/// Bytes `[start, end)` of the segment `segment` (its slab id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub segment: u64,
    pub start: u64,
    pub end: u64,
}

impl Run {
    fn overlaps(&self, other: &Run) -> bool {
        self.segment == other.segment && self.start < other.end && other.start < self.end
    }
}

/// Every live instance's runs, by the instance a claim names.
#[derive(Debug, Default)]
pub struct Claims {
    live: Mutex<Vec<(u64, Run)>>,
    next: AtomicU64,
}

impl Claims {
    /// A name for a new instance's claims.
    pub fn instance(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Claims `runs` for `instance`, in place of what it held: refused by
    /// name, and nothing changed, where one overlaps a run another instance
    /// holds.
    pub fn claim(&self, instance: u64, runs: &[Run]) -> Result<()> {
        let mut live = self.live.lock().expect("claims lock poisoned");
        for run in runs {
            if let Some((holder, held)) = live
                .iter()
                .find(|(holder, held)| *holder != instance && held.overlaps(run))
            {
                return Err(Fault::Program {
                    at: "guest::claims",
                    why: format!(
                        "Instance slots at bytes {}..{} of segment {} already hold instance \
                         {holder}'s run at {}..{}: instance {instance}'s would overlap it",
                        run.start, run.end, run.segment, held.start, held.end
                    ),
                });
            }
        }
        live.retain(|(holder, _)| *holder != instance);
        live.extend(runs.iter().map(|run| (instance, *run)));
        Ok(())
    }

    /// Releases what `instance` holds.
    pub fn release(&self, instance: u64) {
        self.live
            .lock()
            .expect("claims lock poisoned")
            .retain(|(holder, _)| *holder != instance);
    }
}

#[cfg(test)]
mod tests {
    use super::{Claims, Run};

    const fn run(segment: u64, start: u64, end: u64) -> Run {
        Run {
            segment,
            start,
            end,
        }
    }

    /// Two instances whose runs a runtime laid over each other: the second
    /// bind is refused by name and claims nothing; the same bytes in another
    /// segment, or after the first is released, are taken.
    #[test]
    fn a_run_another_instance_holds_refuses_the_bind() {
        let claims = Claims::default();
        let (a, b) = (claims.instance(), claims.instance());
        claims
            .claim(a, &[run(1, 0, 4096), run(2, 8192, 16384)])
            .expect("the first instance claims its runs");
        let why = claims
            .claim(b, &[run(3, 0, 64), run(2, 12288, 12544)])
            .expect_err("a heap over the first instance's run");
        assert!(why.to_string().contains("already hold instance"), "{why}");
        claims
            .claim(b, &[run(3, 0, 64)])
            .expect("the refused claim took nothing, and another segment is free");
        claims
            .claim(a, &[run(1, 0, 4096), run(2, 0, 4096)])
            .expect("an instance's own runs move without refusing it");
        claims
            .claim(b, &[run(2, 12288, 12544), run(3, 0, 64)])
            .expect("what the first moved off is free");
        claims.release(a);
        claims
            .claim(b, &[run(1, 0, 4096)])
            .expect("a released instance's runs are free");
        assert!(
            !run(1, 0, 4096).overlaps(&run(1, 4096, 8192)),
            "touching runs do not overlap"
        );
    }
}
