//! Reusable scratch regions, so a fire's addresses are the same as the last
//! fire's.
//!
//! # Two problems, one shape
//!
//! **The leak.** `ring::allocate` adds every buffer to the context's residency
//! set — `addAllocation`, `commit`, `requestResidency` — and nothing removes
//! it. `Handle` cannot: it is a *view*, handed out for heap slots and
//! sub-ranges as well as whole allocations, so a `Drop` that removed the
//! allocation would remove one the caller does not own. Measured: fifty
//! `allocate`-then-drop cycles leave the residency set holding **fifty
//! allocations and 52 MB**. A serving driver allocates three regions per fire
//! — arena, params, fire tables — so it leaks all three, permanently, at
//! every step.
//!
//! **The addresses.** `.wiki/driver/graph-metal.md` §4 measures that those
//! same three regions are the *only* things that vary between two fires of
//! one `(plan, row shape)`. Everything else — the dispatch order, the ten
//! pipelines, the twelve grids, the weight addresses — is already stable. So
//! they are also the only thing standing between this driver and recording
//! its command buffer once instead of re-encoding 424 dispatches and 3 779
//! address binds per fire (47.5 % of a prefill, **76.4 % of a decode**).
//!
//! One pool answers both: a region that is kept and reused is a region that
//! is added to the residency set once and keeps its address.
//!
//! # Why a free list and not a bump
//!
//! Because a fire in flight still owns its regions. `ALLOCATOR_COUNT = 2`
//! means `Stepper::submit` waits for the step two back, so up to two fires'
//! regions are live at once and a third must not be handed either one's. A
//! [`Lease`] returns to the pool on drop, and `InFlight` holds the lease for
//! exactly as long as the GPU holds the region.
//!
//! # What it does not do
//!
//! Shrink. A pool that returns memory to the device would give up the
//! addresses, which is the point of having it. The bound is the number of
//! distinct `(purpose, size)` pairs a deployment fires, and that is a
//! property of its texts rather than of its traffic.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::device::allocation::Allocation;
use crate::device::context::Context;
use crate::device::handle::Handle;
use crate::error::Result;

/// The pool's own state, shared by every outstanding [`Lease`].
#[derive(Default)]
struct Free {
    /// Regions nobody is using, by `(purpose, bytes)`.
    ///
    /// The size is part of the key rather than a "big enough" search: a
    /// region handed out for more bytes than it was asked for is a region
    /// whose `Handle` claims a length the caller did not request, and
    /// `Handle::over`'s own doc records what that costs — *"a view that
    /// includes it is how one slot quietly reaches its neighbour's
    /// rounding"*.
    by_shape: HashMap<(&'static str, u64), Vec<Allocation>>,
    /// How many regions this pool has ever allocated, for the test that asks
    /// whether reuse is happening at all.
    allocated: usize,
}

/// A pool of reusable device regions.
#[derive(Clone, Default)]
pub struct Scratch {
    free: Rc<RefCell<Free>>,
}

impl Scratch {
    /// An empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A region of exactly `len` bytes, reused if one is free.
    ///
    /// The bytes are **not** zeroed: a reused region holds the last fire's
    /// activations. Every caller in this crate zeroes what it takes, and the
    /// reason is measured — three runs of one fire over one checkpoint's
    /// weights gave widest activations of 11.7, 23.1 and 4.5e12 before the
    /// arena was zeroed. Zeroing here as well would double that cost for no
    /// caller's benefit.
    ///
    /// # Errors
    ///
    /// The device declining a fresh allocation, when nothing is free.
    pub fn take(&self, context: &Context, len: u64, what: &'static str) -> Result<Lease> {
        let len = len.max(1);
        let key = (what, len);
        let existing = self
            .free
            .borrow_mut()
            .by_shape
            .get_mut(&key)
            .and_then(Vec::pop);
        let handle = match existing {
            Some(handle) => handle,
            None => {
                let handle = Allocation::new(context, len, what)?;
                self.free.borrow_mut().allocated += 1;
                handle
            }
        };
        Ok(Lease {
            handle: Some(handle),
            pool: Rc::clone(&self.free),
            key,
        })
    }

    /// How many regions this pool has allocated from the device.
    ///
    /// The number that says whether reuse is working: a hundred fires over
    /// one shape should move it by a handful, not by a hundred.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.free.borrow().allocated
    }
}

/// A region borrowed from a [`Scratch`], returned when dropped.
pub struct Lease {
    /// `Some` until dropped. An `Option` because the return has to move the
    /// allocation out of a `&mut self`.
    handle: Option<Allocation>,
    pool: Rc<RefCell<Free>>,
    key: (&'static str, u64),
}

impl Lease {
    /// The region itself.
    #[must_use]
    pub fn region(&self) -> &Handle {
        self.handle
            .as_ref()
            .expect("a lease holds its region until it is dropped")
            .handle()
    }
}

impl std::ops::Deref for Lease {
    type Target = Handle;

    fn deref(&self) -> &Handle {
        self.region()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.pool
                .borrow_mut()
                .by_shape
                .entry(self.key)
                .or_default()
                .push(handle);
        }
    }
}

impl std::fmt::Debug for Scratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let free = self.free.borrow();
        f.debug_struct("Scratch")
            .field("shapes", &free.by_shape.len())
            .field("allocated", &free.allocated)
            .finish()
    }
}

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease").field("key", &self.key).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Scratch;
    use crate::device::{Allocation, Context};
    use objc2_metal::MTLResidencySet;

    /// What a pool buys once the leak is closed: **the address**.
    ///
    /// This used to assert two things, and the first has retired. Bare
    /// `allocate` added to the residency set and nothing removed, so eight
    /// allocate-and-drop cycles left eight entries -- and this test pinned
    /// that number, with a note that *"if this ever fails, `allocate` learned
    /// to remove and this pool's leak argument needs restating"*.
    /// [`Allocation`] learned, so the argument is restated here rather than
    /// deleted: the number is now ZERO, and it is pinned in
    /// `allocation.rs`'s own tests where the type that owns it lives.
    ///
    /// What does not retire is the address. Fifty pooled takes of one shape
    /// produce ONE device allocation and ONE address, which is what an
    /// indirect command buffer recorded for that shape needs in order to be
    /// replayable -- and no ownership fix supplies that, because a correct
    /// allocate-and-free per fire gives a different address every time.
    #[test]
    fn a_pool_allocates_once_and_keeps_the_address() {
        let Ok(context) = Context::new() else {
            return;
        };
        // What the pool replaces, so the comparison is against this tree's
        // own behaviour rather than against a description of it.
        let before = context.residency().allocationCount();
        for _ in 0..8 {
            drop(Allocation::new(&context, 1 << 16, "unpooled").expect("a region"));
        }
        assert_eq!(
            context.residency().allocationCount(),
            before,
            "eight allocate-and-drop cycles should leave nothing behind"
        );

        let scratch = Scratch::new();
        let mut addresses = std::collections::BTreeSet::new();
        for _ in 0..8 {
            let lease = scratch.take(&context, 1 << 16, "pooled").expect("a region");
            addresses.insert(lease.gpu_address());
        }
        assert_eq!(
            scratch.allocated(),
            1,
            "eight takes of one shape, one allocation"
        );
        assert_eq!(addresses.len(), 1, "and one address, which is the point");

        // Two live at once is two regions: a lease still held cannot be
        // handed out again, which is what `ALLOCATOR_COUNT = 2` requires.
        let a = scratch.take(&context, 1 << 16, "pooled").expect("a region");
        let b = scratch.take(&context, 1 << 16, "pooled").expect("a region");
        assert_ne!(
            a.gpu_address(),
            b.gpu_address(),
            "two live leases must not share a region -- the second fire would \
             write over the first while it executes"
        );
        assert_eq!(scratch.allocated(), 2);

        // A different shape is a different region, and asking for more bytes
        // than were leased never resizes one silently.
        let _wide = scratch.take(&context, 1 << 17, "pooled").expect("a region");
        assert_eq!(scratch.allocated(), 3);
    }
}
