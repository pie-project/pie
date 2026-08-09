//! A standalone device buffer that leaves the residency set when it dies.
//!
//! # The leak this replaces
//!
//! `ring::allocate` created a shared-storage buffer, added it to the
//! context's residency set — `addAllocation`, `commit`, `requestResidency` —
//! and returned a [`Handle`]. A `Handle` is a *view*: it is handed out for
//! heap slots and sub-ranges as well as for whole allocations, so a `Drop`
//! on it that removed the allocation would remove one the caller does not
//! own. Nothing else removed it either. Measured: fifty allocate-then-drop
//! cycles left the residency set holding **fifty allocations and 52 MB**.
//!
//! A residency set holds its own reference, so a buffer that is only dropped
//! stays alive and stays resident. That is the same defect
//! `.wiki/driver/progress-metal.md` records for the caching pool under *"a
//! count is not a release"*, in the one place where nothing counted at all.
//!
//! # Why a type rather than a `release` call
//!
//! Because a `release` call is a convention, and this crate has already paid
//! for that one twice — the pool's `State::release` that decremented counters
//! and edited nothing, and the C++ `release_standalone_buffer` whose whole
//! reason for existing is that `create_standalone_buffer` handed back
//! something with no owner. `.wiki/driver/real-metal-north-star.md` §8 states
//! the target directly: *registration and release are one type's construct
//! and drop*.
//!
//! The pattern is not new here. [`Mapped`](super::external::Mapped),
//! [`External`](super::external::External) and [`Ring`](super::ring::Ring)
//! all register on construct and remove on `Drop`. Bare `allocate` was the
//! one primitive that could not, and every caller of it leaked.
//!
//! # What ownership now means
//!
//! An `Allocation` is resident for exactly as long as it lives, so **dropping
//! one while the GPU is still reading it is now a real event** where before
//! it was free. That is not a regression introduced by this type — it is the
//! bug it makes visible. It found one: the serving seam staged a fire's
//! tables per step and dropped them at the end of the loop iteration, while
//! the fire itself was not waited for until after the loop. Under the leak
//! that was invisible and permanent; the fix is that the tables travel with
//! the fire, which is what [`InFlight`](crate::fire::run::InFlight)
//! already did for the argument table and the scalars.
//!
//! [`Deref`](std::ops::Deref) to `Handle` is deliberate and follows
//! [`Lease`](crate::fire::scratch::Lease), which is the same shape one
//! layer up: a thing that owns a region and should read like the region.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResidencySet, MTLResourceOptions};

use crate::error::{Error, Result};
use crate::device::context::Context;
use crate::device::external::{add, remove};
use crate::device::handle::Handle;
use crate::layout::region::Region;

/// One shared-storage buffer, resident for exactly as long as this value.
///
/// Construct with [`Allocation::new`]; read it through the [`Handle`] it
/// derefs to. Cloning is deliberately not offered: two owners would each
/// remove the allocation, and the second removal is against a set that no
/// longer names it. Hand out [`Handle`]s instead — they retain the buffer,
/// so a view cannot outlive the memory, only the residency.
pub struct Allocation {
    /// The whole buffer, as the view every caller wants.
    handle: Handle,
    /// Retained rather than borrowed from the context, so an `Allocation`
    /// can be stored beside things that outlive a `&Context`.
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Allocation {
    /// Allocate `len` bytes of shared storage and make them resident.
    ///
    /// `what` names the region in the error, because "the device declined
    /// 268435456 bytes" is a different bug report from "the device declined
    /// 4 bytes" and the caller is the only one who knows which region asked.
    ///
    /// # Errors
    ///
    /// A length this host's `usize` cannot hold, or the device declining it.
    pub fn new(context: &Context, len: u64, what: &'static str) -> Result<Self> {
        let options = MTLResourceOptions(
            MTLResourceOptions::StorageModeShared.0
                | MTLResourceOptions::HazardTrackingModeUntracked.0,
        );
        let length = usize::try_from(len).map_err(|_| Error::Create {
            what,
            message: format!("{len} bytes does not fit this host's usize"),
        })?;
        let buffer = context
            .device()
            .newBufferWithLength_options(length, options)
            .ok_or_else(|| Error::Create {
                what,
                message: format!("the device declined {len} bytes"),
            })?;
        add(context.residency(), &buffer);
        Ok(Self {
            handle: Handle::over(&buffer, len)?,
            residency: context.residency_handle(),
        })
    }

    /// The whole region, as a view.
    #[must_use]
    pub const fn handle(&self) -> &Handle {
        &self.handle
    }
}

impl std::ops::Deref for Allocation {
    type Target = Handle;

    fn deref(&self) -> &Handle {
        &self.handle
    }
}

impl Drop for Allocation {
    /// Out of the residency set. The buffer itself releases with the last
    /// `Handle` over it, which may be later — a view keeps the memory, this
    /// keeps it resident, and they are not the same claim.
    fn drop(&mut self) {
        remove(&self.residency, self.handle.buffer());
    }
}

impl std::fmt::Debug for Allocation {
    /// The span, not the residency set it is in — every `Allocation` is in
    /// that one, so printing it says nothing.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocation")
            .field("gpu_address", &self.handle.gpu_address())
            .field("len", &Region::len(&self.handle))
            .finish()
    }
}

#[cfg(test)]
// A device test that finds no device SAYS so by passing. Silence would make
// "this machine has no Metal 4" and "the residency set is balanced" the same
// observation.
mod tests {
    use super::Allocation;
    use crate::device::Context;
    use objc2_metal::MTLResidencySet;

    /// The leak, closed, measured against the device rather than a counter.
    ///
    /// `.wiki/driver/progress-metal.md` records the shape this belongs to:
    /// *"a subsystem's own bookkeeping cannot falsify a claim about something
    /// outside it."* The pool's eight tests all passed while `release` freed
    /// nothing, because they asked the pool. This asks
    /// `MTLResidencySet::allocationCount`, which is the device's answer and
    /// the only one that could have been wrong.
    ///
    /// Falsified by removing the `Drop` impl: the second assertion reports
    /// `left: 8, right: 0`.
    #[test]
    fn an_allocation_leaves_the_residency_set_when_it_drops() {
        let Ok(context) = Context::new() else {
            return;
        };
        let before = context.residency().allocationCount();

        // Held: every one of them is in the set, so the test can tell "the
        // drop removed it" from "it was never added".
        let live: Vec<_> = (0..8)
            .map(|_| Allocation::new(&context, 1 << 16, "residency probe").expect("a region"))
            .collect();
        assert_eq!(
            context.residency().allocationCount() - before,
            8,
            "eight live allocations should be eight entries -- if this fails, \
             construction stopped registering and the drop half proves nothing"
        );

        drop(live);
        assert_eq!(
            context.residency().allocationCount(),
            before,
            "the set should be back where it started. Fifty allocate-and-drop \
             cycles used to leave fifty allocations and 52 MB resident, and a \
             serving driver does three per fire"
        );
    }

    /// A view outliving its owner keeps the MEMORY, not the residency.
    ///
    /// Both halves matter. `Handle` retains the buffer, so a slice cannot
    /// dangle -- that is why `Handle` is safe to hand out. What it does not
    /// carry is residency, which is exactly why the owner has to outlive the
    /// GPU work rather than merely outliving the last read.
    #[test]
    fn a_handle_outlives_its_allocation_without_keeping_it_resident() {
        let Ok(context) = Context::new() else {
            return;
        };
        let before = context.residency().allocationCount();
        let allocation = Allocation::new(&context, 4096, "outlived").expect("a region");
        let view = allocation.handle().clone();
        let address = view.gpu_address();
        drop(allocation);

        assert_eq!(
            context.residency().allocationCount(),
            before,
            "the owner is gone, so the entry should be"
        );
        assert_eq!(
            view.gpu_address(),
            address,
            "the view still names the same buffer -- it retains it, so this is \
             not a dangling read, only a non-resident one"
        );
    }
}
