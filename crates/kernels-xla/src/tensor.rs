use dtype::Dtype;

use crate::hlo::{Built, Cmp, Elem, Func, Val};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor<T> {
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

/// Where a plane's pages (or a state plane's slot blocks) lie in the plane
/// it is read from: page `p` of a fire's tables is the plane's own page
/// `p * stride + offset`. In the shared chunk pool every plane is a view of
/// the whole pool, and a plane's run of a slot starts at a multiple of its
/// own run, so a slot ref becomes the plane's page by this affine map
/// (`addr = base + slot * stride + offset`); the degenerate pools are the
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageMap {
    pub stride: i64,

    pub offset: i64,
}

impl Default for PageMap {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl PageMap {
    pub const IDENTITY: PageMap = PageMap {
        stride: 1,
        offset: 0,
    };

    #[must_use]
    pub const fn is_identity(self) -> bool {
        self.stride == 1 && self.offset == 0
    }

    /// The plane's page of each id (i32, any shape): `x * stride + offset`
    /// for `x >= 0`; a negative id (`-(s) - 1`, a slot that reads but does
    /// not write, or `-1`, a dropped row) keeps its encoding over the mapped
    /// page: `-((-x - 1) * stride + offset) - 1`. An id too large to map
    /// without overflowing i32 (a padded row's `i32::MAX`) stays as it is,
    /// out of every plane's range. The identity emits nothing.
    pub fn apply(self, f: &mut Func, ids: Val) -> Built<Val> {
        if self.is_identity() {
            return Ok(ids);
        }
        let most = (i64::from(i32::MAX) - self.offset.abs()) / self.stride.max(1);
        let zero = f.like_i(ids, 0);
        let real = f.compare(Cmp::Ge, ids, zero)?;
        let neg_one = f.like_i(ids, -1);
        let flip = f.sub(neg_one, ids)?;
        let base = f.select(real, ids, flip)?;
        let top = f.like_i(base, most);
        let fits = f.compare(Cmp::Le, base, top)?;
        let stride = f.like_i(base, self.stride);
        let mapped = f.mul(base, stride)?;
        let offset = f.like_i(mapped, self.offset);
        let mapped = f.add(mapped, offset)?;
        let neg_one = f.like_i(mapped, -1);
        let back = f.sub(neg_one, mapped)?;
        let mapped = f.select(real, mapped, back)?;
        f.select(fits, mapped, ids)
    }

    /// The id a plane page `x` maps back from, on the host, for tests and
    /// engines laying a pool: `x * stride + offset`.
    #[must_use]
    pub const fn page(self, x: i64) -> i64 {
        x * self.stride + self.offset
    }
}

/// A paged kv space: `keys`/`values` are `[pages * page_size, kv_heads *
/// head_dim]` planes (values may be narrower), paged by the CSR
/// `page_indptr`/`page_indices` of the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool<T> {
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

    /// Where the keys plane's pages lie ([`PageMap`]).
    pub keys_map: PageMap,

    /// Where the values plane's pages lie.
    pub values_map: PageMap,
}

/// A recurrent state space: `state` holds one row per slot, `slots` the slot
/// of each lane of the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool<T> {
    pub state: T,

    pub slots: T,

    pub conv_state: T,

    pub new_conv_state: T,

    /// Where each slot's block of rows lies in `state` (and the conv
    /// planes, the same plane): slot `s` is block `map(s)` ([`PageMap`]).
    pub map: PageMap,
}

/// A quantized weight: codes with per-group scales (and zero points when
/// affine).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank<T> {
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
