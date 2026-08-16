//! The launch path's view of device memory: one buffer, at an offset, with a
//! length both sides are good for — the span a fire binds when it stages a
//! stage's descriptors, an op's parameters, or a channel ring's words.
//!
//! A [`Handle`] can only be a view of a plain buffer. An
//! [`Elastic`](super::Elastic) (placement-sparse) buffer's sub-range must
//! never be represented as one: the ordinary capacity test (`bytes <= size`)
//! would pass against pages with no backing store behind them.
//!
//! [`Handle::over`] refuses a buffer whose storage mode gives the host no
//! address for its bytes. [`Handle::slice`] refuses a span that leaves its
//! parent view, using the same wrap-safe bound every [`Region`] uses.
//!
//! # Ownership
//!
//! A `Handle` retains its buffer, so the allocation cannot be freed out from
//! under a view of it. Retaining does not give exclusivity: a pooled buffer
//! whose [`Transient`](super::Transient) is dropped returns to the free list
//! even while a `Handle` still addresses its bytes, so a handle must not
//! outlive the owner it was derived from.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};

use crate::error::{Error, Result};
use crate::layout::region::Region;

/// A span of one shared-storage buffer: an address for the GPU, a pointer for
/// the host, and the length both are valid for.
///
/// Cloning is a retain, not a copy of bytes; a clone is the same span.
#[derive(Clone)]
pub struct Handle {
    /// Retained so the span cannot outlive the allocation it names.
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    contents: NonNull<c_void>,
    gpu_address: u64,
    len: u64,
}

impl Handle {
    /// A view of the first `len` bytes of `buffer`.
    ///
    /// `len` is a claim, not a measurement. Pass the length the allocation
    /// was *asked for*, not `buffer.length()`: the heap, the pool and the
    /// device all round a request up, the padding is mapped and writable, and
    /// a view that includes it is how one slot quietly reaches its
    /// neighbour's rounding.
    ///
    /// # Errors
    ///
    /// [`Error::Create`] if the buffer's storage mode gives the host no
    /// address for its bytes — a view is a host-visible span, and a private
    /// buffer has no `contents` to point at. [`Error::OutOfRange`] if `len`
    /// claims more than the buffer holds.
    pub fn over(buffer: &ProtocolObject<dyn MTLBuffer>, len: u64) -> Result<Self> {
        let mode = buffer.storageMode();
        if mode != MTLStorageMode::Shared {
            return Err(Error::Create {
                what: "buffer view",
                message: format!(
                    "storage mode {mode:?} gives the host no address for the bytes; \
                     a view is a host-visible span of a Shared buffer"
                ),
            });
        }
        let length = buffer.length() as u64;
        if len > length {
            return Err(Error::OutOfRange {
                what: "buffer view",
                offset: 0,
                bytes: len,
                len: length,
            });
        }
        Ok(Self {
            buffer: buffer.retain(),
            contents: buffer.contents(),
            gpu_address: buffer.gpuAddress(),
            len,
        })
    }

    /// The sub-span of `len` bytes starting `offset` bytes into this one.
    ///
    /// The bound is [`Region::check`]'s — written as two comparisons so an
    /// `offset + len` that wraps is refused rather than passed — and both
    /// the host pointer and the GPU address move by exactly the offset that
    /// survived it.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] if the span leaves this view. The bound is this
    /// view's own length, not the buffer's: a slice of a slice cannot reach
    /// back out into what its parent was already narrowed away from.
    pub fn slice(&self, offset: u64, len: u64) -> Result<Self> {
        self.check("handle", offset, len)?;
        // SAFETY: `check` kept `offset` within this view's `len`, and the
        // view's bytes are one mapped allocation, so the moved pointer stays
        // in bounds and non-null.
        let contents = unsafe { self.contents.cast::<u8>().add(usize_of(offset)).cast() };
        Ok(Self {
            buffer: self.buffer.clone(),
            contents,
            // In bounds of the same mapped allocation, so it cannot wrap.
            gpu_address: self.gpu_address + offset,
            len,
        })
    }

    /// The buffer the span lives in, for an API that wants the object rather
    /// than its address.
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The GPU virtual address of the first byte, for an argument table.
    #[must_use]
    pub const fn gpu_address(&self) -> u64 {
        self.gpu_address
    }
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("gpu_address", &format_args!("{:#x}", self.gpu_address))
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

// SAFETY: `contents` is the shared-storage pointer of a buffer this value
// retains, valid for `len` bytes from construction on: `over` measured the
// claim against the buffer's own length and `slice` against its parent's.
// Views of one buffer overlap by construction — a slice IS an alias of its
// parent — which is the explicit-alias case the trait's contract names; what
// keeps two of them from racing is the step boundary, exactly as for `Slot`
// and `Transient`.
unsafe impl Region for Handle {
    fn contents(&self) -> NonNull<c_void> {
        self.contents
    }

    fn len(&self) -> u64 {
        self.len
    }
}

/// Narrows a checked offset to a host index; callers have already bounded
/// it against a `usize`-sized `len`.
#[allow(clippy::cast_possible_truncation)]
const fn usize_of(v: u64) -> usize {
    v as usize
}
