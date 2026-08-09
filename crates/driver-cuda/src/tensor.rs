//! A typed, shaped region of device memory, and the shape arithmetic that
//! decides how large it is.
//!
//! Port of `kernels-cuda/csrc/src/tensor.hpp`'s `DeviceTensor`. The C++ class
//! fuses two things that fail in different ways and are testable by different
//! means:
//!
//! * the **shape arithmetic** -- what `dtype` and what extents, hence how many
//!   bytes -- which is pure host code, is where the interesting mistakes live,
//!   and today can only be exercised by allocating on a GPU;
//! * the **allocation** itself, which is a `cudaMalloc` and either works or
//!   returns an error.
//!
//! They are split here into [`TensorSpec`] and [`DeviceTensor`]. The pools in
//! [`crate::store`] build a `Vec<TensorSpec>` first and allocate from it
//! second, so the layout of a KV cache -- which layers are real, which are
//! aliases, which tiers a quantised format adds -- can be compared against the
//! C++ exactly, on a machine with no CUDA device at all.
//!
//! # What the split buys
//!
//! The C++ `KvCache::allocate_per_layer` interleaves the decision and the
//! `cudaMalloc`: the shape is an argument to a call whose result is a pointer,
//! so nothing observable survives to be checked. The only way to ask "did this
//! allocate the right thing" is to allocate it. Producing the specification as
//! a value first makes that question answerable by comparing two vectors.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// The dtype and extents of a tensor, with no memory behind it.
///
/// This is the half of `DeviceTensor` that can be wrong in an interesting way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    dtype: DType,
    shape: Vec<i64>,
    numel: u64,
    nbytes: u64,
}

impl TensorSpec {
    /// Build a spec, rejecting a negative extent as `DeviceTensor::allocate`
    /// does.
    ///
    /// The C++ throws here rather than clamping, and it matters: a negative
    /// extent reaching `numel_ *= static_cast<std::size_t>(d)` would wrap to a
    /// number near `2^64` and the subsequent `cudaMalloc` would fail with an
    /// unrelated message about an impossible size.
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

    /// Whether this describes no memory at all.
    ///
    /// A zero-byte spec is the port of the C++'s default-constructed
    /// `DeviceTensor`, which the cache classes push into their layer vectors
    /// as a placeholder so that a slot index stays equal to a layer index.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nbytes == 0
    }
}

/// Device memory with a dtype and a shape.
///
/// The C++ carries an `owns_memory_` flag and a hand-written move constructor
/// that nulls the source, because a `DeviceTensor` may be either an owner or a
/// borrowed view into someone else's allocation and the destructor has to tell
/// them apart at runtime. Here ownership is the type: this struct owns, and a
/// borrowed view is a `&DeviceTensor` or a raw pointer obtained from one, so
/// the flag and the null-out have nothing to do.
/// GATED — it OWNS device memory, which is the one thing in this file
/// that needs a card. `TensorSpec` above it is arithmetic and stays
/// portable, which is what the cache geometry actually asks for.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct DeviceTensor {
    spec: TensorSpec,
    buffer: Option<crate::gpu::device::DeviceBuffer>,
}

#[cfg(feature = "_cuda")]
impl DeviceTensor {
    /// Allocate uninitialised device memory for `spec`.
    ///
    /// A zero-byte spec allocates nothing and yields a tensor whose pointer is
    /// null, matching the C++'s `if (t.nbytes_ > 0)` guard. The cache classes
    /// rely on that: they distinguish a real layer from an aliased one by
    /// asking whether its tensor is empty.
    pub fn allocate(alloc: &crate::gpu::device::Allocator, spec: TensorSpec) -> Result<Self> {
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
            .map_or(core::ptr::null_mut(), crate::gpu::device::DeviceBuffer::as_ptr)
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
