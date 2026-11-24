use std::cell::{Ref, RefCell};

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};

pub const NIL: u32 = u32::MAX;

pub struct Binding<B: Backend> {
    buffer: B::Buffer,
    offset: u64,

    extent: u64,

    /// The first byte of the value this binding is a window of
    /// ([`Handles::window`]), its own offset where it is a value whole.
    origin: u64,
}

impl<B: Backend> Clone for Binding<B> {
    fn clone(&self) -> Self {
        Binding {
            buffer: self.buffer.clone(),
            offset: self.offset,
            extent: self.extent,
            origin: self.origin,
        }
    }
}

impl<B: Backend> Binding<B> {
    /// The reservation this binding lies in.
    #[must_use]
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    /// The first `len` bytes this binding names, read back.
    pub fn read(&self, len: u64) -> Result<Vec<u8>> {
        let mut out = vec![0u8; usize::try_from(len).unwrap_or(usize::MAX)];
        self.buffer.read(self.offset, &mut out)?;
        Ok(out)
    }

    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The first byte of the value this binding windows: a fire's class
    /// region binds its rows of a value from the region's row offset, and a
    /// backend whose views start only at aligned bytes binds the value from
    /// here, whatever row the region starts at, and steps the kernel over
    /// the rows before it.
    #[must_use]
    pub fn origin(&self) -> u64 {
        self.origin
    }

    /// The bytes of the reservation from this binding's offset on.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.buffer.slab_size().saturating_sub(self.offset)
    }

    #[must_use]
    pub fn extent(&self) -> u64 {
        self.extent
    }

    #[must_use]
    pub fn is_host(&self) -> bool {
        self.buffer.is_host()
    }

    #[must_use]
    pub fn slab_id(&self) -> u64 {
        self.buffer.slab_id()
    }
}

impl<B: Backend> std::fmt::Debug for Binding<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Binding")
            .field("offset", &self.offset)
            .finish()
    }
}

pub struct Handles<B: Backend> {
    rows: RefCell<Vec<Binding<B>>>,
    sealed: std::cell::Cell<usize>,
}

impl<B: Backend> Default for Handles<B> {
    fn default() -> Self {
        Handles {
            rows: RefCell::new(Vec::new()),
            sealed: std::cell::Cell::new(0),
        }
    }
}

impl<B: Backend> std::fmt::Debug for Handles<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handles")
            .field("rows", &self.rows.borrow().len())
            .field("sealed", &self.sealed.get())
            .finish()
    }
}

impl<B: Backend> Handles<B> {
    #[must_use]
    pub fn new() -> Handles<B> {
        Handles::default()
    }

    pub fn bind(&self, buffer: &B::Buffer, offset: u64, len: u64) -> Result<u32> {
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
            buffer: buffer.clone(),
            offset,
            extent: len,
            origin: offset,
        });
        Ok(at as u32)
    }

    pub fn read(&self, handle: u32, len: u64) -> Result<Vec<u8>> {
        let binding = self.get(handle).ok_or_else(|| Fault::Unbound {
            what: format!("handle {handle}, which no row answers"),
        })?;
        binding.read(len)
    }

    /// `len` bytes of `handle` from byte `skip`: a value of its own.
    pub fn cut(&self, handle: u32, skip: u64, len: u64) -> Result<u32> {
        self.cut_of(handle, skip, len, false)
    }

    /// `len` bytes of `handle` from byte `skip` as a window of it (a class
    /// region's rows of a fire's value): the binding keeps where the value
    /// starts ([`Binding::origin`]).
    pub fn window(&self, handle: u32, skip: u64, len: u64) -> Result<u32> {
        self.cut_of(handle, skip, len, true)
    }

    fn cut_of(&self, handle: u32, skip: u64, len: u64, window: bool) -> Result<u32> {
        let (buffer, offset, origin) = {
            let rows = self.rows.borrow();
            let row = rows.get(handle as usize).ok_or_else(|| Fault::Unbound {
                what: format!("handle {handle}, which this load minted no row for"),
            })?;
            (row.buffer.clone(), row.offset, row.origin)
        };
        let at_offset = offset.checked_add(skip).ok_or(Fault::Ceiling {
            what: "bytes of a device reservation",
            need: u64::MAX,
            have: offset,
        })?;
        let size = buffer.slab_size();
        match at_offset.checked_add(len) {
            Some(end) if end <= size => {}
            _ => {
                return Err(Fault::Ceiling {
                    what: "bytes of a device reservation",
                    need: at_offset.saturating_add(len),
                    have: size,
                });
            }
        }
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
            buffer,
            offset: at_offset,
            extent: len,
            origin: if window { origin } else { at_offset },
        });
        Ok(at as u32)
    }

    #[must_use]
    pub fn get(&self, handle: u32) -> Option<Ref<'_, Binding<B>>> {
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

    /// Drops the rows past `len` (a mark of [`Handles::len`]), none sealed.
    pub fn rewind_to(&self, len: usize) {
        self.rows.borrow_mut().truncate(len.max(self.sealed.get()));
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
