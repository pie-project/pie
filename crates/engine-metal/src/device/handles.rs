//! The handle table: a `kernels_metal::Tensor`'s `u32` is an index into this
//! table, one row per carved view. Weight rows are minted once at load;
//! arena/pool/input rows are minted per fire and dropped at [`Handles::seal`]'s
//! watermark by [`Handles::rewind`], at enqueue rather than settle.

use std::cell::{Ref, RefCell};

use super::alloc::{Buffer, Slab};
use crate::error::{Fault, Result};

/// The handle for an absent argument. Not 0, since 0 is a valid handle (the
/// first row minted) and a null spelled as a valid index silently binds
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

/// Every view this load has minted, in minting order. Interior-mutable
/// because `Run::tensor` mints while taking `&self`.
#[derive(Default)]
pub struct Handles {
    rows: RefCell<Vec<Binding>>,
    /// Where the load-time rows end. Set once by [`Handles::seal`].
    sealed: std::cell::Cell<usize>,
}

// SAFETY: rows retain `MTLBuffer`s, documented thread-safe for retain/release
// and binding.
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

    /// Mint a handle for `len` bytes of `buffer` starting at `offset`. The
    /// length is not stored (a `Tensor` states its own rectangle) but is
    /// checked here, the last place a carve can be caught before a shader
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

    /// Declare everything minted so far to be load-lived. Called once, before
    /// the first fire; a second call is a no-op.
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
