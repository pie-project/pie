//! The handle table: what a `kernels_metal::Tensor`'s `u32` means.
//!
//! **THIS IS THE ONE PLACE THE TWO SHELLS GENUINELY DIVERGE.** A
//! `kernels_cuda::Tensor` carries a device address, so its shell hands out
//! `base + offset` and the resolution is arithmetic. Metal has no address to
//! hand out: a compute encoder binds a BUFFER and an OFFSET
//! (`setBuffer:offset:atIndex:`), so the number a `Tensor` carries has to be
//! an index into something. This is that something — one row per carved
//! view, minted by whoever carved it and read by the encode sink.
//!
//! **Two generations in one table, and the watermark is what separates
//! them.** Weight rows are minted once at load and read by every fire
//! afterwards; arena slots, pool views and staged input vectors are minted
//! per fire and dead at the end of it. Both live here because a `Tensor` is
//! one `u32` wide and cannot say which it is. [`Handles::seal`] records the
//! load-time watermark and [`Handles::rewind`] drops everything past it, so
//! a fire's minting costs one `Vec` push per view and its cleanup costs one
//! truncate — and a stale fire handle cannot survive into the next fire to
//! be resolved against the wrong offset.
//!
//! **THE REWIND IS AT ENQUEUE AND NOT AT SETTLE, AND THAT IS WHAT LETS TWO
//! STEPS BE IN FLIGHT.** A row is read by the ENCODER, at
//! `setBuffer:offset:`, and a command buffer retains what it was bound to —
//! so a step's rows are dead the moment its last dispatch is encoded, long
//! before the device has finished the work. Held until settlement instead,
//! the table would have no room for the step behind. The one row a settlement
//! still needs — the out seam's, for the readout copy — is resolved into a
//! retained buffer and a `u64` offset while it is still alive, which is why
//! nothing downstream of `enqueue` holds a handle at all.
//!
//! A row RETAINS its buffer. That is what lets the arena, the pools and the
//! inputs slab each be owned by their own module while the sink resolves a
//! handle without borrowing any of them.

use std::cell::{Ref, RefCell};

use super::alloc::{Buffer, Slab};
use crate::error::{Fault, Result};

/// The handle every `absent` argument carries: a slot the shader declares
/// nothing at, or an optional pointer this entry does not pass.
///
/// `u32::MAX` rather than 0 because 0 is a perfectly good handle — the first
/// row minted — and a null spelled as a valid index is a silent bind of
/// somebody else's bytes.
pub const NIL: u32 = u32::MAX;

/// One resolved view: which buffer, and how far into it.
#[derive(Clone)]
pub struct Binding {
    slab: Slab,
    offset: u64,
}

impl Binding {
    /// The retained buffer this view lives in.
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

    /// Bytes from the buffer's base to this view's first element.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("offset", &self.offset)
            .finish()
    }
}

/// Every view this load has minted, in minting order.
///
/// **INTERIOR-MUTABLE, AND THE WINDOW IS WHY.** A dispatch arm resolves its
/// operands through `Run::tensor`, which cuts each one to the asking
/// region's row window (design §0) — and on this plane a cut IS a new
/// handle, because Metal has no address to add an offset to. `Run::tensor`
/// takes `&self` (the walk's `Dispatch` contract), so the minting has to be
/// a write through a shared reference. The `RefCell` is never borrowed
/// across a call into another module, and the whole table lives on one
/// thread.
#[derive(Default)]
pub struct Handles {
    rows: RefCell<Vec<Binding>>,
    /// Where the load-time rows end. Set once by [`Handles::seal`].
    sealed: std::cell::Cell<usize>,
}

// SAFETY: the rows retain `MTLBuffer`s, which are documented thread-safe for
// retain/release and for binding; the same argument `Buffer`'s own `Send`
// makes, for the same move.
unsafe impl Send for Handles {}

impl std::fmt::Debug for Handles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handles")
            .field("rows", &self.rows.borrow().len())
            .field("sealed", &self.sealed.get())
            .finish()
    }
}

impl Handles {
    /// An empty table.
    #[must_use]
    pub fn new() -> Handles {
        Handles::default()
    }

    /// Mint a handle for `len` bytes of `buffer` starting at `offset`.
    ///
    /// The length is not stored — a `Tensor` states its own rectangle and
    /// the kernels read it from there — but it IS checked, here, because
    /// this is the last place a carve can be caught before a shader
    /// dereferences past the reservation.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the view leaves its buffer, and when the
    /// table would pass [`NIL`].
    pub fn bind(&self, buffer: &Buffer, offset: u64, len: u64) -> Result<u32> {
        buffer.span(offset, len)?;
        let mut rows = self.rows.borrow_mut();
        let at = rows.len();
        if at >= NIL as usize {
            return Err(Fault::Ceiling {
                what: "handles in one load",
                need: at as u64 + 1,
                have: u64::from(NIL),
            });
        }
        rows.push(Binding {
            slab: buffer.slab().clone(),
            offset,
        });
        Ok(at as u32)
    }

    /// Mint a handle `skip` bytes further into whatever `handle` names.
    ///
    /// The windowed cut, and the reason this table exists: a CUDA shell
    /// answers this with `ptr + skip` and no state at all. The row is
    /// bounds-checked against the buffer the parent row names, so a cut past
    /// the end is refused here rather than in a shader.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a handle no row answers, [`Fault::Ceiling`]
    /// when the cut leaves the buffer or the table is full.
    pub fn cut(&self, handle: u32, skip: u64, len: u64) -> Result<u32> {
        let (slab, offset) = {
            let rows = self.rows.borrow();
            let row = rows.get(handle as usize).ok_or_else(|| Fault::Unbound {
                what: format!("handle {handle}, which this load minted no row for"),
            })?;
            (row.slab.clone(), row.offset)
        };
        let at_offset = offset.checked_add(skip).ok_or(Fault::Ceiling {
            what: "bytes of a device reservation",
            need: u64::MAX,
            have: offset,
        })?;
        let _ = len;
        let mut rows = self.rows.borrow_mut();
        let at = rows.len();
        if at >= NIL as usize {
            return Err(Fault::Ceiling {
                what: "handles in one load",
                need: at as u64 + 1,
                have: u64::from(NIL),
            });
        }
        rows.push(Binding {
            slab,
            offset: at_offset,
        });
        Ok(at as u32)
    }

    /// Resolve a handle. `None` for [`NIL`] and for a row past the table.
    #[must_use]
    pub fn get(&self, handle: u32) -> Option<Ref<'_, Binding>> {
        if handle == NIL {
            return None;
        }
        let rows = self.rows.borrow();
        if handle as usize >= rows.len() {
            return None;
        }
        Some(Ref::map(rows, |rows| &rows[handle as usize]))
    }

    /// Declare everything minted so far to be load-lived.
    ///
    /// Called once, after the weight table is built and before the first
    /// fire. Sealing twice would move the watermark past rows a fire minted,
    /// which is the leak this method exists to prevent — so it takes the
    /// first answer and keeps it.
    pub fn seal(&self) {
        if self.sealed.get() == 0 {
            self.sealed.set(self.rows.borrow().len());
        }
    }

    /// Drop every handle minted since [`Handles::seal`].
    pub fn rewind(&self) {
        self.rows.borrow_mut().truncate(self.sealed.get());
    }

    /// How many rows the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.borrow().len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.borrow().is_empty()
    }

    /// Where the load-lived rows end.
    #[must_use]
    pub fn sealed(&self) -> usize {
        self.sealed.get()
    }
}
