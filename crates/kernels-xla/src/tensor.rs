use dtype::Dtype;

use crate::hlo::{Elem, Func, Val};

/// An operand as a kernel reads it: the value the engine read (`[rows,
/// width]` of the dtype's storage element) and the shape and dtype it is
/// stored in. A packed bank's codes keep their declared width and dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub val: Val,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Operand {
    #[must_use]
    pub const fn elements(self) -> u64 {
        self.rows as u64 * self.width as u64
    }

    /// The value in f32, the precision every reduction runs in.
    pub fn f32(self, f: &mut Func) -> Val {
        f.convert(self.val, Elem::F32)
    }

    /// The value converted to `elem`.
    pub fn as_elem(self, f: &mut Func, elem: Elem) -> Val {
        f.convert(self.val, elem)
    }

    /// Where a result that replaces this operand lands.
    #[must_use]
    pub const fn landing(self) -> Landing {
        Landing {
            rows: self.rows,
            width: self.width,
            dtype: self.dtype,
        }
    }
}

/// Where a kernel's result lands: its stored shape and dtype. The kernel
/// returns the value; the engine converts it to `dtype` and reshapes it to
/// `[rows, width]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Landing {
    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Landing {
    #[must_use]
    pub const fn elements(self) -> u64 {
        self.rows as u64 * self.width as u64
    }
}

/// The stored shape and dtype a kernel checks, whichever side of it a
/// tensor is on.
pub trait Stored: Copy {
    fn rows(self) -> u32;
    fn width(self) -> u32;
    fn dtype(self) -> Dtype;

    /// The same storage read as another dtype of the same width.
    #[must_use]
    fn with_dtype(self, dtype: Dtype) -> Self;

    /// The same rows read `width` wide.
    #[must_use]
    fn with_width(self, width: u32) -> Self;

    fn elements(self) -> u64 {
        u64::from(self.rows()) * u64::from(self.width())
    }

    /// The shape and dtype alone.
    fn shape(self) -> Landing {
        Landing {
            rows: self.rows(),
            width: self.width(),
            dtype: self.dtype(),
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

impl Stored for Operand {
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

impl Stored for Landing {
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

/// A handle the engine resolves: `buf` names a binding (a root plus a row
/// window), `rows × width` is its shape in elements of `dtype`. The same
/// four fields as kernels-wgpu's `Tensor`, so an engine dispatch reads the
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor<T = Tensor> {
    pub data: T,

    pub indptr: T,
}

impl<T> RaggedTensor<T> {
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<RaggedTensor<U>, E> {
        Ok(RaggedTensor {
            data: f(self.data)?,
            indptr: f(self.indptr)?,
        })
    }
}

/// A paged kv space: `keys`/`values` are `[pages * page_size, kv_heads *
/// head_dim]` planes (values may be narrower), paged by the CSR
/// `page_indptr`/`page_indices` of the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool<T = Tensor> {
    pub keys: T,

    pub values: T,

    pub page_indices: T,

    pub page_indptr: T,

    pub page_size: i32,

    /// The most pages any lane of this fire holds, rounded up to the fire's
    /// page bucket: the static bound a gathered page table is laid out to.
    /// Part of the executable's key, like the row bucket.
    pub max_pages: u32,

    pub seq_stride: u64,

    pub head_stride: u64,
}

/// A recurrent state space: `state` holds one row per slot, `slots` the slot
/// of each lane of the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool<T = Tensor> {
    pub state: T,

    pub slots: T,

    pub conv_state: T,

    pub new_conv_state: T,
}

/// A quantized weight: codes with per-group scales (and zero points when
/// affine).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank<T = Tensor> {
    pub codes: T,

    pub scales: T,

    pub biases: Option<T>,

    pub group: u32,

    pub bits: u32,
}

impl<T> Bank<T> {
    #[must_use]
    pub const fn affine(&self) -> bool {
        self.biases.is_some()
    }

    /// The bank with its codes, scales and biases read (in that order).
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<Bank<U>, E> {
        Ok(Bank {
            codes: f(self.codes)?,
            scales: f(self.scales)?,
            biases: self.biases.map(&mut f).transpose()?,
            group: self.group,
            bits: self.bits,
        })
    }
}
