//! An IOSurface the Neural Engine reads or writes. The GPU reaches the same
//! memory through a no-copy Metal buffer over [`Surface::base`].

use std::ffi::{c_int, c_void};

use super::sys;

/// The element type a surface holds: the Neural Engine program reads int8
/// activations and weights and writes fp16 partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Int8,
    Fp16,
}

impl Element {
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Int8 => 1,
            Self::Fp16 => 2,
        }
    }

    /// The type name the MIL program declares the surface's tensor with.
    #[must_use]
    pub const fn mil_type(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
        }
    }
}

pub struct Surface {
    raw: sys::RawSurface,
    element: Element,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    /// `rows` rows of `width` elements. Rows are stride-aligned to 64 bytes
    /// and the allocation to 16 KB, so [`Surface::stride`] is what the GPU
    /// kernels index by, not `width`.
    pub fn new(rows: u32, width: u32, element: Element) -> Result<Surface, String> {
        let mut raw = sys::RawSurface {
            surface: std::ptr::null_mut(),
            base: std::ptr::null_mut(),
            bytes: 0,
            stride: 0,
        };
        let mut err = sys::message();
        let int8 = c_int::from(element == Element::Int8);
        let made = unsafe {
            sys::pie_ane_surface(
                rows,
                width,
                int8,
                &raw mut raw,
                err.as_mut_ptr(),
                sys::CAP as c_int,
            )
        };
        if made != 1 {
            return Err(sys::failure(&err));
        }
        Ok(Surface { raw, element })
    }

    #[must_use]
    pub fn element(&self) -> Element {
        self.element
    }

    /// The first byte of the surface's memory, shared with the GPU.
    #[must_use]
    pub fn base(&self) -> *mut u8 {
        self.raw.base.cast()
    }

    /// The allocation's length in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.raw.bytes
    }

    /// Bytes from one row to the next.
    #[must_use]
    pub fn stride(&self) -> u32 {
        self.raw.stride
    }

    pub(super) fn handle(&self) -> *mut c_void {
        self.raw.surface
    }

    #[must_use]
    pub fn mil_type(&self) -> &'static str {
        self.element.mil_type()
    }

    /// The MIL stride list of a `[1, 1, rows, width]` view of the surface.
    #[must_use]
    pub fn strides(&self, rows: u32) -> String {
        let stride = u64::from(self.raw.stride / self.element.bytes());
        let plane = u64::from(rows) * stride;
        format!("[{plane}, {plane}, {stride}, 1]")
    }

    /// The `tensor_buffer` type a MIL function declares this surface as.
    #[must_use]
    pub fn buffer_type(&self, rows: u32, width: u32) -> String {
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
            self.mil_type(),
            self.strides(rows)
        )
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { sys::pie_ane_surface_free(self.raw.surface) };
    }
}
