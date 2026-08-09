//! Buffers this context did not allocate, and the residency they still need.
//!
//! Two things live here and they answer the same question from opposite
//! sides. [`Mapped`] builds a Metal buffer over host pages -- an `mmap` of a
//! checkpoint, in practice -- so a 19 GB file is never copied into a Metal
//! allocation. [`Externals`] takes a buffer someone else allocated, an
//! authoritative channel ring being the case it exists for, and adds it to
//! this context's residency set without claiming to own it.
//!
//! # Why a no-copy buffer is not just an optimisation
//!
//! `newBufferWithLength:` over a checkpoint means the bytes exist twice: once
//! in the page cache the loader read them through and once in a Metal
//! allocation, and the second copy is dirty anonymous memory the kernel
//! cannot evict. A file-backed mapping wrapped with
//! `newBufferWithBytesNoCopy:` is demand-faulted under GPU access and its
//! pages stay CLEAN, so the kernel reclaims them under pressure and faults
//! them back when the GPU next reads. That is what weight streaming is on
//! Apple silicon, and it is why nothing here checks the working set: a
//! mapping larger than the working set is the intended case, not the error
//! case, and refusing it would defeat the entire point.
//!
//! # What the caller still owns
//!
//! Metal is handed a `deallocator:` of `nil`, so it will not unmap anything.
//! The mapping must outlive the [`Mapped`], and the buffer must be gone
//! before the pages are. Rust cannot check that -- the pages are not a Rust
//! allocation -- so [`Mapped::new`] is `unsafe` and the obligation is stated
//! there rather than in a comment beside the call.
//!
//! # Residency, and the leak it used to be
//!
//! Both of these have to be added to the residency set individually: they are
//! not inside the placement heap, so the one `addAllocation` that covers
//! every heap slot does not cover them. Both then have to be REMOVED, and
//! that is the part the C++ shell notes was leaking -- a pool that grows and
//! shrinks left every old buffer retained and resident forever. Here removal
//! is `Drop`, so forgetting it is not something the API can express.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLAllocation, MTLBuffer, MTLDevice, MTLResidencySet, MTLResourceOptions};

use crate::gpu::device::context::Context;
use crate::error::{Error, Result};
use crate::layout::region::Region;

/// This host's page size.
///
/// `newBufferWithBytesNoCopy:` requires a page-aligned address, and asked for
/// rather than assumed: it is 16 KiB on Apple silicon and 4 KiB on Intel, and
/// a constant would be wrong on one of them.
#[must_use]
pub fn page_size() -> u64 {
    // SAFETY: `sysconf` with a valid name has no preconditions.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(raw).unwrap_or(4096)
}

/// Add `buffer` to the residency set and ask for it to be made resident.
///
/// `pub(super)` so the ring, which allocates its own buffers rather than
/// adopting someone else's, keeps them resident through the same three calls
/// instead of a fourth copy of them.
pub(super) fn add(
    residency: &ProtocolObject<dyn MTLResidencySet>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
) {
    let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(buffer);
    residency.addAllocation(allocation);
    residency.commit();
    residency.requestResidency();
}

/// Drop `buffer` from the residency set.
pub(super) fn remove(
    residency: &ProtocolObject<dyn MTLResidencySet>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
) {
    let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(buffer);
    residency.removeAllocation(allocation);
    residency.commit();
    residency.requestResidency();
}

/// A Metal buffer over memory this context does not own.
///
/// See the module docs. The pages stay the caller's, and Metal is told not to
/// free them.
pub struct Mapped {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    contents: NonNull<c_void>,
    gpu_address: u64,
    len: u64,
}

impl Mapped {
    /// Wrap `len` bytes at `ptr` in a shared-storage Metal buffer.
    ///
    /// # Errors
    ///
    /// [`Error::Create`] if `ptr` is not page-aligned, if `len` is zero, or
    /// if Metal declines the mapping. The alignment is checked here rather
    /// than left to Metal because Metal answers a misaligned pointer with
    /// `nil` and no reason, and "no-copy buffer failed" over a 19 GB
    /// checkpoint gives nothing to act on.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `len` readable, writable, mapped bytes that stay
    /// mapped for at least as long as the returned value. Nothing here can
    /// check that: the pages are not a Rust allocation and Metal is given a
    /// null `deallocator:`, so unmapping them while the buffer lives leaves
    /// the GPU reading addresses that are no longer backed.
    pub unsafe fn new(context: &Context, ptr: NonNull<c_void>, len: u64) -> Result<Self> {
        if len == 0 {
            return Err(Error::Create {
                what: "no-copy buffer",
                message: "a mapping of zero bytes has no address to give the GPU".to_string(),
            });
        }
        let page = page_size();
        let address = ptr.as_ptr() as u64;
        if !address.is_multiple_of(page) {
            return Err(Error::Create {
                what: "no-copy buffer",
                message: format!(
                    "host address {address:#x} is not aligned to this host's {page}-byte page; \
                     Metal answers that with nil and no reason"
                ),
            });
        }
        let length = usize::try_from(len).map_err(|_| Error::Create {
            what: "no-copy buffer",
            message: format!("{len} bytes does not fit this host's usize"),
        })?;

        // SAFETY: the caller's obligation is exactly this call's precondition
        // -- `ptr` is `length` mapped bytes that outlive the buffer. A null
        // deallocator is what says Metal must not unmap them.
        let buffer = unsafe {
            context
                .device()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .ok_or_else(|| Error::Create {
            what: "no-copy buffer",
            message: format!(
                "the device declined a mapping of {len} bytes at {address:#x}; a length that is \
                 not a whole number of pages is the usual reason"
            ),
        })?;

        let contents = buffer.contents();
        let gpu_address = buffer.gpuAddress();
        add(context.residency(), &buffer);

        Ok(Self {
            buffer,
            residency: context.residency().retain(),
            contents,
            gpu_address,
            len,
        })
    }

    /// The Metal buffer, for binding.
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The GPU virtual address, for an argument table entry.
    #[must_use]
    pub const fn gpu_address(&self) -> u64 {
        self.gpu_address
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        remove(&self.residency, &self.buffer);
    }
}

impl std::fmt::Debug for Mapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapped")
            .field("gpu_address", &format_args!("{:#x}", self.gpu_address))
            .field("len", &self.len)
            .finish()
    }
}

// SAFETY: `contents` is the buffer's shared-storage pointer over the caller's
// pages, and the caller has promised `len` of them stay mapped for at least
// this value's lifetime.
unsafe impl Region for Mapped {
    fn contents(&self) -> NonNull<c_void> {
        self.contents
    }

    fn len(&self) -> u64 {
        self.len
    }
}

/// Buffers allocated elsewhere that this context has been asked to keep
/// resident.
///
/// Reference counted by buffer identity, because the same ring can be handed
/// in by several callers and the first one to finish must not drop it out of
/// the set from under the others. Cloning shares the registry.
#[derive(Clone, Default)]
pub struct Externals {
    state: Arc<Mutex<HashMap<usize, usize>>>,
}

impl Externals {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Make `buffer` resident for as long as the returned guard lives.
    ///
    /// Registering the same buffer twice adds it once and takes two drops to
    /// undo, which is what the count is for.
    #[must_use]
    pub fn insert(&self, context: &Context, buffer: &ProtocolObject<dyn MTLBuffer>) -> External {
        let key = std::ptr::from_ref(buffer).cast::<()>() as usize;
        let first = {
            let mut state = self.locked();
            let count = state.entry(key).or_insert(0);
            *count += 1;
            *count == 1
        };
        if first {
            add(context.residency(), buffer);
        }
        External {
            registry: self.clone(),
            residency: context.residency().retain(),
            buffer: buffer.retain(),
            key,
        }
    }

    /// How many distinct buffers are registered.
    ///
    /// Distinct, not registrations: this is the number of allocations the
    /// residency set is carrying on their behalf, which is the figure that
    /// matters against the working set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locked().len()
    }

    /// Whether no external buffer is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locked().is_empty()
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, HashMap<usize, usize>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for Externals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Externals")
            .field("buffers", &self.len())
            .finish()
    }
}

/// One registration of an external buffer. Drops out of residency when the
/// last one for that buffer goes.
pub struct External {
    registry: Externals,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Retained so the buffer cannot be freed while the residency set still
    /// names it. The C++ keeps only a raw pointer and relies on the caller.
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    key: usize,
}

impl External {
    /// The buffer this registration keeps resident.
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

impl Drop for External {
    fn drop(&mut self) {
        let last = {
            let mut state = self.registry.locked();
            match state.get_mut(&self.key) {
                Some(count) if *count > 1 => {
                    *count -= 1;
                    false
                }
                Some(_) => {
                    state.remove(&self.key);
                    true
                }
                None => false,
            }
        };
        if last {
            remove(&self.residency, &self.buffer);
        }
    }
}

impl std::fmt::Debug for External {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("External")
            .field("key", &format_args!("{:#x}", self.key))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_size_is_a_power_of_two_and_at_least_four_kib() {
        let page = page_size();
        assert!(page >= 4096, "{page}");
        assert!(page.is_power_of_two(), "{page}");
    }
}
