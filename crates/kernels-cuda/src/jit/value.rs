use std::ffi::c_void;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArgValue {

    Ptr(*mut c_void),
    I32(i32),
    U32(u32),
    F32(f32),
    Usize(usize),
    I64(i64),
    Bool(bool),
    U8(u8),
    Region {

        ptr: *mut c_void,
        rows: i32,
        width: i32,
    },
    Bytes {

        ptr: *const u8,
        len: usize,
    },
}

impl ArgValue {

    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            ArgValue::Ptr(_) => "a pointer",
            ArgValue::I32(_) => "an i32",
            ArgValue::U32(_) => "a u32",
            ArgValue::F32(_) => "an f32",
            ArgValue::Usize(_) => "a usize",
            ArgValue::I64(_) => "an i64",
            ArgValue::Bool(_) => "a bool",
            ArgValue::U8(_) => "a u8 enumerator",
            ArgValue::Region { .. } => "a region",
            ArgValue::Bytes { .. } => "a by-value aggregate",
        }
    }

    #[must_use]
    pub fn cell(self) -> u64 {
        match self {
            ArgValue::Ptr(p) => p as u64,
            #[allow(clippy::cast_sign_loss)]
            ArgValue::I32(v) => u64::from(v as u32),
            ArgValue::U32(v) => u64::from(v),
            ArgValue::F32(v) => u64::from(v.to_bits()),
            ArgValue::Usize(v) => v as u64,
            #[allow(clippy::cast_sign_loss)]
            ArgValue::I64(v) => v as u64,
            ArgValue::Bool(v) => u64::from(v),
            ArgValue::U8(v) => u64::from(v),
            ArgValue::Region { .. } => {
                panic!("a region has no cell; a body pushes its parts instead")
            }
            ArgValue::Bytes { .. } => {
                panic!("an aggregate has no cell; the launch copies it instead")
            }
        }
    }
}

#[cfg(feature = "_cuda")]
pub struct Bound {

    #[allow(clippy::vec_box)]
    cells: Vec<Box<u64>>,
    blobs: Vec<Box<[u8]>>,
    slots: Vec<*mut c_void>,
}

#[cfg(feature = "_cuda")]
impl Bound {

    pub unsafe fn new(values: &[ArgValue]) -> Self {
        let mut out = Self {
            cells: Vec::with_capacity(values.len()),
            blobs: Vec::new(),
            slots: Vec::with_capacity(values.len()),
        };
        for value in values {
            match *value {
                ArgValue::Bytes { ptr, len } => {

                    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
                    let mut boxed: Box<[u8]> = bytes.to_vec().into_boxed_slice();
                    let at: *mut u8 = boxed.as_mut_ptr();
                    out.blobs.push(boxed);
                    out.slots.push(at.cast());
                }
                other => {
                    let mut boxed = Box::new(other.cell());
                    let at: *mut u64 = &raw mut *boxed;
                    out.cells.push(boxed);
                    out.slots.push(at.cast());
                }
            }
        }
        out
    }

    pub fn slots_mut(&mut self) -> &mut [*mut c_void] {
        &mut self.slots
    }
}

impl kernels::routine::Absent for ArgValue {
    fn is_absent(&self) -> bool {
        matches!(self, Self::Ptr(p) if p.is_null())
    }

    fn absent() -> Option<Self> {
        Some(Self::Ptr(core::ptr::null_mut()))
    }
}
