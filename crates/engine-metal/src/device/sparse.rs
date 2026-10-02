//! The Metal 4 half of the elastic pool: what maps a tile and what proves
//! the mapping landed.
//!
//! The shell encodes its frames on a legacy `MTLCommandQueue`, and that
//! queue cannot change a sparse buffer's mappings — only an
//! `MTL4CommandQueue` can. So the two coexist: frames go on the legacy
//! queue as before, and remaps go on a second, Metal 4 queue that shares
//! the device. Three things tie them together:
//!
//! * a [`MTLResidencySet`] attached to the legacy queue, so every heap a
//!   sparse buffer maps into is resident while a frame runs;
//! * an [`MTLSharedEvent`] the mapping queue signals after each batch of
//!   remaps, which the host waits on before it lets a frame that needs the
//!   new tiles be encoded — a growth is therefore synchronous to the host,
//!   which costs one scheduling round-trip per batch and buys a frame that
//!   never has to encode a wait;
//! * a rule, not a mechanism, for the unmap direction: tiles are taken
//!   away only when the caller has drained every frame that could read
//!   them. `Pools::release` states that precondition; nothing here can
//!   check it, because the legacy queue signals nothing.
//!
//! Absent on a device without `MTLGPUFamilyMetal4` — [`Sparse::open`]
//! answers `None` and the pool stays a fixed allocation.

use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation, MTLBuffer, MTLCommandQueue,
    MTLDevice, MTLGPUFamily, MTLHeap, MTLResidencySet, MTLResidencySetDescriptor, MTLSharedEvent,
    MTLSparseTextureMappingMode,
};

use crate::error::{Fault, Result};

/// How long one probe of a mapping wait lasts, and how many probes before
/// the wait is declared lost. Probed rather than waited for in one call so a
/// mapping queue that stopped moving is a refusal with a message, not a hang.
const WAIT_PROBE_MS: u64 = 5_000;
const WAIT_PROBES: u32 = 12;

struct Ticks {
    /// The last timeline value the mapping queue was told to signal.
    issued: u64,
    /// The last value the host has observed signalled.
    landed: u64,
    /// Whether a remap has been issued since the last signal.
    dirty: bool,
}

pub struct Sparse {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    ticks: Mutex<Ticks>,
}

// SAFETY: every Metal object here is documented thread-safe for the calls
// made on it (queue operations, event reads and waits, residency edits), and
// the only mutable state is behind a `Mutex`.
unsafe impl Send for Sparse {}
unsafe impl Sync for Sparse {}

impl std::fmt::Debug for Sparse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ticks = self.ticks();
        f.debug_struct("Sparse")
            .field("issued", &ticks.issued)
            .field("landed", &ticks.landed)
            .finish()
    }
}

impl Sparse {
    /// Whether `device` can map tiles at all.
    #[must_use]
    pub fn supported(device: &ProtocolObject<dyn MTLDevice>) -> bool {
        device.supportsFamily(MTLGPUFamily::Metal4)
    }

    /// The mapping side for `device`, or `None` when it is not a Metal 4
    /// device. The residency set is attached to `frames`, the queue the
    /// shell's frames run on, so heaps mapped here are resident there.
    pub fn open(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        frames: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Option<Arc<Sparse>>> {
        if !Sparse::supported(device) {
            return Ok(None);
        }
        let queue = device.newMTL4CommandQueue().ok_or(Fault::Device {
            call: "newMTL4CommandQueue",
            why: "the device would not open a Metal 4 command queue".to_string(),
        })?;
        let event = device.newSharedEvent().ok_or(Fault::Device {
            call: "newSharedEvent",
            why: "the device would not open a shared event".to_string(),
        })?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|error| Fault::Device {
                call: "newResidencySetWithDescriptor:error:",
                why: error.localizedDescription().to_string(),
            })?;
        frames.addResidencySet(&residency);
        Ok(Some(Arc::new(Sparse {
            device: device.clone(),
            queue,
            event,
            residency,
            ticks: Mutex::new(Ticks {
                issued: 0,
                landed: 0,
                dirty: false,
            }),
        })))
    }

    pub(crate) fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub(crate) fn residency(&self) -> &ProtocolObject<dyn MTLResidencySet> {
        &self.residency
    }

    pub(crate) fn event(&self) -> &Retained<ProtocolObject<dyn MTLSharedEvent>> {
        &self.event
    }

    /// What the timeline has actually reached.
    #[must_use]
    pub fn signalled(&self) -> u64 {
        self.event.signaledValue()
    }

    fn ticks(&self) -> std::sync::MutexGuard<'_, Ticks> {
        self.ticks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Queue one remap of `tiles` tiles of `buffer` from `first_tile`,
    /// taking them from `heap` at `heap_tile` (a map) or giving them back
    /// (an unmap, which ignores the heap). Nothing has happened until
    /// [`flush`](Self::flush) returns.
    pub(crate) fn issue(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        heap: Option<&ProtocolObject<dyn MTLHeap>>,
        mode: MTLSparseTextureMappingMode,
        first_tile: u64,
        tiles: u64,
        heap_tile: u64,
    ) {
        let operation = MTL4UpdateSparseBufferMappingOperation {
            mode,
            bufferRange: NSRange {
                location: usize::try_from(first_tile).unwrap_or(usize::MAX),
                length: usize::try_from(tiles).unwrap_or(0),
            },
            heapOffset: usize::try_from(heap_tile).unwrap_or(0),
        };
        let mut ticks = self.ticks();
        // SAFETY: the operation is a live `repr(C)` value for the call, the
        // count is one and matches, and the buffer and heap outlive it —
        // the heap is borrowed from a `Chunk` the caller still owns.
        unsafe {
            self.queue.updateBufferMappings_heap_operations_count(
                buffer,
                heap,
                std::ptr::NonNull::from(&operation),
                1,
            );
        }
        ticks.dirty = true;
    }

    /// Signal the timeline past every remap issued so far and wait for it.
    ///
    /// Returns the value the timeline reached, which is what a buffer
    /// records as its fence. Idempotent when nothing was issued.
    pub(crate) fn flush(&self) -> Result<u64> {
        let value = {
            let mut ticks = self.ticks();
            if !ticks.dirty {
                return Ok(ticks.landed);
            }
            ticks.issued += 1;
            ticks.dirty = false;
            ticks.issued
        };
        self.queue
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), value);
        if !self.await_value(value) {
            return Err(Fault::Device {
                call: "updateBufferMappings",
                why: format!(
                    "the mapping queue did not reach {value} within {} s; the kv pool's \
                     memory cannot be trusted",
                    u64::from(WAIT_PROBES) * WAIT_PROBE_MS / 1000
                ),
            });
        }
        self.ticks().landed = value;
        Ok(value)
    }

    /// Block until the timeline reaches `value`; `false` if it never did.
    pub(crate) fn await_value(&self, value: u64) -> bool {
        if self.event.signaledValue() >= value {
            return true;
        }
        (0..WAIT_PROBES).any(|_| {
            self.event
                .waitUntilSignaledValue_timeoutMS(value, WAIT_PROBE_MS)
        })
    }
}
