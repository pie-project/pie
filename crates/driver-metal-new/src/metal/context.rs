//! The device objects a step is encoded against.
//!
//! One queue, two command allocators, one residency set. That is the whole of
//! the Metal 4 context; everything else the C++ shell keeps in the same struct
//! (the heap, the compiler, the argument tables, the timing) is a separate
//! concern that happens to have been declared next to this one.
//!
//! # Why the allocators come in pairs
//!
//! An `MTL4CommandAllocator` owns the memory a command buffer was built into,
//! so resetting one while the GPU is still reading it is a use-after-free that
//! Metal will not diagnose. Two of them, used alternately, is what lets step
//! `n+1` be encoded while step `n` is still running: the allocator being reset
//! is the one whose commands finished two steps ago. The C++ shell calls this
//! `alloc[2]` and indexes it with an `ab` flag; the pair is modelled here as a
//! pair so that "which one may I reset" is a question about an index rather
//! than about a comment.
//!
//! This type does NOT reset them. It cannot: knowing that a step completed is
//! knowing about the event timeline, which belongs to the encoder that waits
//! on it. What lives here is ownership.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSError;
use objc2_metal::{
    MTL4CommandAllocator, MTL4CommandQueue, MTLDevice, MTLGPUFamily, MTLResidencySet,
    MTLResidencySetDescriptor,
};

use super::device::default_device;
use crate::error::{Error, Result};

/// How many command allocators the context keeps.
///
/// Two: see the module docs. Named rather than spelled `2` at each use so the
/// arithmetic that alternates between them cannot disagree with the array.
pub const ALLOCATOR_COUNT: usize = 2;

/// The Metal 4 device objects, and the device they came from.
pub struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocators: [Retained<ProtocolObject<dyn MTL4CommandAllocator>>; ALLOCATOR_COUNT],
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Context {
    /// Build the context on the system default device.
    ///
    /// Fails rather than degrades when the device is not a Metal 4 device.
    /// The driver is written against MTL4 types throughout -- a fallback path
    /// would be a second driver, and pretending one exists would turn a clear
    /// refusal here into an obscure nil somewhere later.
    pub fn new() -> Result<Self> {
        let device = default_device().ok_or(Error::NoDevice)?;

        // The C++ shell gates on exactly this and prints the device name. The
        // check is not decoration: `newMTL4CommandQueue` on a pre-Metal-4
        // device returns nil, and nil here surfaces three calls later as a
        // missing encoder rather than as an unsupported device.
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(Error::Create {
                what: "MTL4 context",
                message: format!(
                    "device '{}' does not support MTLGPUFamilyMetal4; this driver is Metal 4 only",
                    device.name()
                ),
            });
        }

        let queue = device.newMTL4CommandQueue().ok_or(Error::Create {
            what: "MTL4CommandQueue",
            message: String::new(),
        })?;

        // `try_map` would be tidier but is not stable for arrays, and the
        // count is 2. Collecting into a Vec and unwrapping the length back
        // into an array would be the same code with a panic in it.
        let allocators = [new_allocator(&device)?, new_allocator(&device)?];

        let residency_descriptor = MTLResidencySetDescriptor::new();
        let residency = device
            .newResidencySetWithDescriptor_error(&residency_descriptor)
            .map_err(|e| Error::Create {
                what: "MTLResidencySet",
                message: describe(&e),
            })?;

        Ok(Self {
            device,
            queue,
            allocators,
            residency,
        })
    }

    /// The device everything here was created from.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The queue steps are committed to.
    #[must_use]
    pub fn queue(&self) -> &ProtocolObject<dyn MTL4CommandQueue> {
        &self.queue
    }

    /// The allocator for parity `index`, which is taken modulo the pair.
    ///
    /// Taken modulo rather than asserted because the caller's index is a step
    /// counter, and a step counter that has to be reduced by its caller is a
    /// step counter that will eventually be reduced twice.
    #[must_use]
    pub fn allocator(&self, index: usize) -> &ProtocolObject<dyn MTL4CommandAllocator> {
        &self.allocators[index % ALLOCATOR_COUNT]
    }

    /// The residency set every allocation this context hands out belongs to.
    #[must_use]
    pub fn residency(&self) -> &ProtocolObject<dyn MTLResidencySet> {
        &self.residency
    }

    /// What this device will hold resident, in bytes.
    #[must_use]
    pub fn working_set_bytes(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// Refuse `requested` bytes up front when the device will not hold them.
    ///
    /// Metal does not report this. Every buffer is created, every bind
    /// succeeds, and the failure arrives much later as a command buffer
    /// returning `kIOGPUCommandBufferCallbackErrorOutOfMemory` from three
    /// levels down -- by which point the numbers that would explain it are
    /// gone. This is the only place they are still in hand.
    pub fn check_working_set(&self, requested: u64) -> Result<()> {
        let working_set = self.working_set_bytes();
        // A device that reports 0 is a device that declines to answer, not one
        // that will hold nothing. Refusing every allocation on a machine whose
        // driver does not publish the number would be a worse failure than the
        // one this guards against.
        if working_set != 0 && requested > working_set {
            return Err(Error::WorkingSetExceeded {
                requested,
                working_set,
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("device", &self.device.name().to_string())
            .field("working_set_bytes", &self.working_set_bytes())
            .finish_non_exhaustive()
    }
}

fn new_allocator(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTL4CommandAllocator>>> {
    device.newCommandAllocator().ok_or(Error::Create {
        what: "MTL4CommandAllocator",
        message: String::new(),
    })
}

/// What an `NSError` said, for an [`Error`] message.
pub(crate) fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything below needs a GPU. A context that cannot be built because
    /// the process cannot reach a device is not a failure of the code under
    /// test, so these skip rather than fail -- but ONLY on `NoDevice`. Every
    /// other error is a real failure and is propagated.
    fn context() -> Option<Context> {
        match Context::new() {
            Ok(ctx) => Some(ctx),
            Err(Error::NoDevice) => None,
            Err(e) => panic!("context creation failed: {e}"),
        }
    }

    #[test]
    fn a_context_builds_and_drops() {
        let Some(ctx) = context() else { return };
        // Dropping is the half worth stating: every field is a `Retained`, so
        // this is where an over-release would land, and it lands as a crash
        // rather than as a failed assertion.
        drop(ctx);
    }

    #[test]
    fn the_allocator_pair_alternates() {
        let Some(ctx) = context() else { return };
        let a = std::ptr::from_ref(ctx.allocator(0));
        let b = std::ptr::from_ref(ctx.allocator(1));
        assert!(!std::ptr::eq(a, b), "the pair must be two distinct objects");
        assert!(
            std::ptr::eq(std::ptr::from_ref(ctx.allocator(2)), a),
            "index wraps at the pair"
        );
        assert!(
            std::ptr::eq(std::ptr::from_ref(ctx.allocator(3)), b),
            "index wraps at the pair"
        );
    }

    #[test]
    fn the_working_set_is_a_real_number() {
        let Some(ctx) = context() else { return };
        let ws = ctx.working_set_bytes();
        assert!(ws > 0, "a Metal 4 device reports a working set");
        ctx.check_working_set(ws).expect("its own limit fits");
        let err = ctx
            .check_working_set(ws + 1)
            .expect_err("one byte past the limit does not");
        assert!(matches!(err, Error::WorkingSetExceeded { .. }), "{err}");
    }
}
