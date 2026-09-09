use std::cell::{Ref, RefCell};

use super::alloc::{Buffer, Slab};
use crate::error::{Fault, Result};

pub const NIL: u32 = u32::MAX;

#[derive(Clone)]
pub struct Binding {
    slab: Slab,
    offset: u64,
}

impl Binding {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

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

#[derive(Default)]
pub struct Handles {
    rows: RefCell<Vec<Binding>>,
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
        });
        Ok(at as u32)
    }

    pub fn read(&self, handle: u32, len: u64) -> Result<Vec<u8>> {
        let binding = self.get(handle).ok_or_else(|| Fault::Unbound {
            what: format!("handle {handle}, which no row answers"),
        })?;
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (binding, len);
            Err(Fault::Unbound {
                what: "a buffer read on a platform with no Metal buffers".to_string(),
            })
        }
        #[cfg(target_vendor = "apple")]
        {
            let mut out = vec![0u8; usize::try_from(len).unwrap_or(usize::MAX)];
            use objc2_metal::MTLBuffer as _;
            let have = binding.slab().length() as u64;
            if binding.offset().saturating_add(len) > have {
                return Err(Fault::Ceiling {
                    what: "bytes read off one handle",
                    need: binding.offset().saturating_add(len),
                    have,
                });
            }
            // SAFETY: a shared-storage buffer's contents are host-addressable
            // for its whole length, and the span was checked just above.
            unsafe {
                let base = binding.slab().contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(
                    base.add(
                        usize::try_from(binding.offset()).expect("an offset inside a live mapping"),
                    ),
                    out.as_mut_ptr(),
                    out.len(),
                );
            }
            Ok(out)
        }
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
