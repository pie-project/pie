//! A typed, shaped region of device memory, and the shape arithmetic that
//! decides how large it is.
//!
//! Port of `DeviceTensor`, split into [`TensorSpec`] (pure shape arithmetic,
//! where the interesting mistakes live) and [`DeviceTensor`] (the `cudaMalloc`)
//! so the layout logic tests on a machine with no CUDA device.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// The dtype and extents of a tensor, with no memory behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    dtype: DType,
    shape: Vec<i64>,
    numel: u64,
    nbytes: u64,
}

impl TensorSpec {
    /// Build a spec. A negative extent is rejected, not clamped: it would wrap
    /// near `2^64` and fail `cudaMalloc` with an unrelated message.
    pub fn new(dtype: DType, shape: impl Into<Vec<i64>>) -> Result<Self> {
        let shape = shape.into();
        let mut numel: u64 = 1;
        for &d in &shape {
            if d < 0 {
                return Err(Error::invalid("DeviceTensor", "negative shape".to_owned()));
            }
            numel = numel.saturating_mul(d.unsigned_abs());
        }
        let nbytes = numel.saturating_mul(dtype.size_bytes() as u64);
        Ok(Self {
            dtype,
            shape,
            numel,
            nbytes,
        })
    }

    /// The element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// The extents.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Total element count, the product of the extents.
    #[must_use]
    pub const fn numel(&self) -> u64 {
        self.numel
    }

    /// Total size in bytes.
    #[must_use]
    pub const fn nbytes(&self) -> u64 {
        self.nbytes
    }

    /// Whether this describes no memory at all — the caches use a zero-byte
    /// spec as a placeholder so a slot index stays equal to a layer index.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nbytes == 0
    }
}

/// Device memory with a dtype and a shape.
///
/// Ownership is the type, not a runtime flag: this struct owns, a borrow is a
/// `&DeviceTensor` or a raw pointer from one. Gated on `_cuda`.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct DeviceTensor {
    spec: TensorSpec,
    buffer: Option<crate::device::DeviceBuffer>,
}

#[cfg(feature = "_cuda")]
impl DeviceTensor {
    /// Allocate uninitialised device memory for `spec`. A zero-byte spec yields
    /// a null pointer, which the caches use to tell a real layer from an alias.
    pub fn allocate(alloc: &crate::device::Allocator, spec: TensorSpec) -> Result<Self> {
        let buffer = if spec.nbytes() == 0 {
            None
        } else {
            Some(alloc.alloc(usize::try_from(spec.nbytes()).map_err(|_| {
                Error::invalid("DeviceTensor::allocate", "size exceeds usize".to_owned())
            })?)?)
        };
        Ok(Self { spec, buffer })
    }

    /// The specification this was allocated from.
    #[must_use]
    pub const fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    /// The device address, or null for a zero-byte tensor.
    #[must_use]
    pub fn as_ptr(&self) -> *mut core::ffi::c_void {
        self.buffer
            .as_ref()
            .map_or(core::ptr::null_mut(), crate::device::DeviceBuffer::as_ptr)
    }

    /// Whether this tensor has no memory behind it.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.buffer.is_none()
    }

    /// The element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.spec.dtype()
    }

    /// The extents.
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        self.spec.shape()
    }

    /// Total size in bytes.
    #[must_use]
    pub const fn nbytes(&self) -> u64 {
        self.spec.nbytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_is_the_product_and_bytes_scale_with_dtype() {
        let s = TensorSpec::new(DType::Bf16, vec![4, 16, 8, 128]).unwrap();
        assert_eq!(s.numel(), 4 * 16 * 8 * 128);
        assert_eq!(s.nbytes(), s.numel() * 2);
        let f = TensorSpec::new(DType::Fp32, vec![4, 16, 8]).unwrap();
        assert_eq!(f.nbytes(), 4 * 16 * 8 * 4);
    }

    #[test]
    fn a_negative_extent_is_refused_rather_than_wrapped() {
        assert!(TensorSpec::new(DType::Bf16, vec![4, -1, 8]).is_err());
    }

    #[test]
    fn an_empty_shape_is_one_element_but_a_zero_extent_is_none() {
        assert_eq!(TensorSpec::new(DType::Fp32, vec![]).unwrap().numel(), 1);
        let z = TensorSpec::new(DType::Fp32, vec![0, 16]).unwrap();
        assert_eq!(z.numel(), 0);
        assert!(z.is_empty());
    }
}
