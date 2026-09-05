use std::cell::{Ref, RefCell};

use super::alloc::{Buffer, Slab};
use crate::error::{Fault, Result};

pub const NIL: u32 = u32::MAX;

#[derive(Clone)]
pub struct Binding {
    slab: Slab,
    offset: u64,

    len: u64,
}

impl Binding {
    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.len.min(self.slab.size.saturating_sub(self.offset))
    }

    #[must_use]
    pub fn slab_id(&self) -> u64 {
        self.slab.id
    }
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("offset", &self.offset)
            .finish()
    }
}

#[derive(Default)]
pub struct Handles {
    rows: RefCell<Vec<Binding>>,
    sealed: std::cell::Cell<usize>,
}

impl std::fmt::Debug for Handles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handles")
            .field("rows", &self.rows.borrow().len())
            .field("sealed", &self.sealed.get())
            .finish()
    }
}

impl Handles {
    #[must_use]
    pub fn new() -> Handles {
        Handles::default()
    }

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
            len,
        });
        Ok(at as u32)
    }

    pub fn read(&self, handle: u32, len: u64) -> Result<Vec<u8>> {
        let binding = self.get(handle).ok_or_else(|| Fault::Unbound {
            what: format!("handle {handle}, which no row answers"),
        })?;
        let mut out = vec![0u8; usize::try_from(len).unwrap_or(usize::MAX)];
        binding.slab().read(binding.offset(), &mut out)?;
        Ok(out)
    }

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
        slab.span(at_offset, len)?;
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
            len,
        });
        Ok(at as u32)
    }

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

    pub fn seal(&self) {
        if self.sealed.get() == 0 {
            self.sealed.set(self.rows.borrow().len());
        }
    }

    pub fn rewind(&self) {
        self.rows.borrow_mut().truncate(self.sealed.get());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.borrow().is_empty()
    }

    #[must_use]
    pub fn sealed(&self) -> usize {
        self.sealed.get()
    }
}
