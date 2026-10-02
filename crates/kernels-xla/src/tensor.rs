use dtype::Dtype;

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

/// A paged kv space: `keys`/`values` are `[pages * page_size, kv_heads *
/// head_dim]` planes (values may be narrower), paged by the CSR
/// `page_indptr`/`page_indices` of the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

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
pub struct RecurrentPool {
    pub state: Tensor,

    pub slots: Tensor,

    pub conv_state: Tensor,

    pub new_conv_state: Tensor,
}

/// A quantized weight: codes with per-group scales (and zero points when
/// affine).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    pub codes: Tensor,

    pub scales: Tensor,

    pub biases: Option<Tensor>,

    pub group: u32,

    pub bits: u32,
}

impl Bank {
    #[must_use]
    pub const fn affine(&self) -> bool {
        self.biases.is_some()
    }
}
