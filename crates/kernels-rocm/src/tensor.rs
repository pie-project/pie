use dtype::Dtype;

use crate::jit::ArgValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub ptr: u64,

    pub rows: u32,

    pub width: u32,

    pub dtype: Dtype,
}

impl Tensor {
    #[must_use]
    pub const fn new(ptr: u64, rows: u32, width: u32, dtype: Dtype) -> Self {
        Self {
            ptr,
            rows,
            width,
            dtype,
        }
    }

    #[must_use]
    pub const fn arg(&self) -> ArgValue<'static> {
        ArgValue::Ptr(self.ptr)
    }

    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.rows as u64 * self.width as u64
    }

    pub const ABSENT: Tensor = Tensor::new(0, 0, 0, Dtype::U8);

    #[must_use]
    pub const fn is_absent(&self) -> bool {
        self.ptr == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaggedTensor {
    pub data: Tensor,

    pub indptr: Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPool {
    pub keys: Tensor,

    pub values: Tensor,

    pub bf16_keys: Tensor,

    pub bf16_values: Tensor,

    pub key_scales: Tensor,

    pub value_scales: Tensor,

    pub page_indices: Tensor,

    pub page_indptr: Tensor,

    pub last_page_lens: Tensor,

    pub row_valid: Tensor,

    pub env_min: Tensor,

    pub env_max: Tensor,

    pub has_envelopes: bool,

    pub page_size: i32,

    pub seq_stride: i64,

    pub head_stride: i64,

    pub layout: i32,

    pub scheme_byte: i32,

    pub block_size: i32,

    pub max_pages_per_request: i32,

    pub pages_in_batch: i32,

    /// Where heads are zero-padded to `head_stride`: the widths a key head
    /// and a value head carry (`[0, 0]`: the whole stride).
    pub live: [u32; 2],

    /// Bytes from a page's rows to the next page's in the key plane and in
    /// the value plane. The HIP kernels do not read it yet: they step pages
    /// packed ([`KvPool::packed_stride`]), and an engine refuses a pool laid
    /// any other way ([`KvPool::packed`]) before it fires one.
    pub page_stride: [u64; 2],
}

impl KvPool {
    /// The page strides of key and value planes laid page after page, each
    /// page `page_size` rows of the plane's width.
    #[must_use]
    pub const fn packed_stride(page_size: i32, keys: &Tensor, values: &Tensor) -> [u64; 2] {
        const fn page_bytes(page: u64, t: &Tensor) -> u64 {
            page * (t.width as u64 * t.dtype.bits()).div_ceil(8)
        }
        let page = if page_size > 0 { page_size as u64 } else { 0 };
        [page_bytes(page, keys), page_bytes(page, values)]
    }

    /// This pool with its planes stated page after page.
    #[must_use]
    pub const fn with_packed_stride(self) -> Self {
        Self {
            page_stride: Self::packed_stride(self.page_size, &self.keys, &self.values),
            ..self
        }
    }

    /// Whether both planes lie page after page: what the HIP kernels read.
    #[must_use]
    pub const fn packed(&self) -> bool {
        let packed = Self::packed_stride(self.page_size, &self.keys, &self.values);
        self.page_stride[0] == packed[0] && self.page_stride[1] == packed[1]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecurrentPool {
    pub slab: Tensor,

    pub slot_ids: Tensor,

    pub slot_stride_elems: i64,

    pub conv_slab: Tensor,

    pub conv_stride: i64,

    pub write_state: bool,

    pub write_state_mask: Tensor,

    pub commit_len: Tensor,

    pub begin_at: Tensor,

    pub fused_decay: bool,

    pub snapshots: Snapshots,

    /// Where the rows `slot_ids` names live. The HIP kernels do not read it
    /// yet: they find a row whole in its slot ([`Segments::ONE`]), and an
    /// engine refuses a record laid any other way before it fires one.
    pub segments: Segments,
}

/// How a recurrent pool's fixed rows sit in its chunks: a row in `pieces`
/// pieces, each in a chunk of its own `piece_off` elements into it, chunk
/// `c` in segment `c / seg_chunks` of `table`. The CUDA kernels' layout,
/// stated here so one engine states both; the HIP kernels read only
/// [`Segments::ONE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segments {
    pub table: Tensor,

    pub seg_chunks: u32,

    pub pieces: u32,

    pub piece_off: i64,

    pub piece_atoms: u32,
}

impl Segments {
    /// The slot layout: one segment, one piece, the slot as its chunk.
    pub const ONE: Segments = Segments {
        table: Tensor::ABSENT,
        seg_chunks: 0,
        pieces: 1,
        piece_off: 0,
        piece_atoms: 0,
    };

    /// Whether a row lies whole in its slot of the one slab: what the HIP
    /// kernels read.
    #[must_use]
    pub const fn whole(&self) -> bool {
        self.table.is_absent() && self.pieces <= 1 && self.piece_off == 0
    }
}

/// The rows inside each lane whose state lands in a slot of its own: a
/// lane's entries are `indptr[lane]..indptr[lane + 1]` of `rows` (inside the
/// lane, ascending) and `slots`, all i32; `count` entries in the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshots {
    pub indptr: Tensor,

    pub rows: Tensor,

    pub slots: Tensor,

    pub count: u32,
}

impl Snapshots {
    pub const NONE: Snapshots = Snapshots {
        indptr: Tensor::ABSENT,
        rows: Tensor::ABSENT,
        slots: Tensor::ABSENT,
        count: 0,
    };

    /// The kernels' `snap, snap_rows, snap_slots`.
    #[must_use]
    pub const fn args(&self) -> [ArgValue<'static>; 3] {
        [self.indptr.arg(), self.rows.arg(), self.slots.arg()]
    }
}
