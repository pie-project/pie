//! The engine's handles, and the kernels' composites over them. A kernel
//! checks a handle's stored shape and dtype ([`Stored`]) and builds its
//! calls from handles; only the engine resolves one to a value.

use dtype::Dtype;
use kernels_xla::attn::{mla, paged, pool, ragged, rel};
use kernels_xla::tensor::{Landing, Stored};

/// A handle the engine resolves: `buf` names a binding (a root plus a row
/// window), `rows × width` is its shape in elements of `dtype`. The same
/// four fields as kernels-slang's `Tensor`, so an engine dispatch reads the
/// same on both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub buf: u32,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Tensor {
    #[must_use]
    pub const fn new(buf: u32, rows: u32, width: u32, dtype: Dtype) -> Self {
        Self {
            buf,
            rows,
            width,
            dtype,
        }
    }

    #[must_use]
    pub const fn elements(self) -> u64 {
        self.rows as u64 * self.width as u64
    }

    /// Where a result lands in this handle.
    #[must_use]
    pub const fn landing(self) -> Landing {
        Landing {
            rows: self.rows,
            width: self.width,
            dtype: self.dtype,
        }
    }
}

impl Stored for Tensor {
    fn with_dtype(self, dtype: Dtype) -> Self {
        Self { dtype, ..self }
    }
    fn with_width(self, width: u32) -> Self {
        Self { width, ..self }
    }
    fn rows(self) -> u32 {
        self.rows
    }
    fn width(self) -> u32 {
        self.width
    }
    fn dtype(self) -> Dtype {
        self.dtype
    }
}

pub type RaggedTensor = kernels_xla::tensor::RaggedTensor<Tensor>;
pub type KvPool = kernels_xla::tensor::KvPool<Tensor>;
pub type RecurrentPool = kernels_xla::tensor::RecurrentPool<Tensor>;
pub type Bank = kernels_xla::tensor::Bank<Tensor>;
pub type Committed = kernels_xla::lanes::Committed<Tensor>;
pub type DecodePlan = kernels_xla::attn::DecodePlan<Tensor>;
pub type PrefillPlan = kernels_xla::attn::PrefillPlan<Tensor>;
pub type Paged = paged::Paged<Tensor>;
pub type MlaCall = mla::MlaCall<Tensor>;
pub type PoolLse = pool::PoolLse<Tensor>;
pub type RaggedMask = ragged::RaggedMask<Tensor>;
pub type RelBias = rel::RelBias<Tensor>;
