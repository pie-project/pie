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

    /// Where the rows `slot_ids` names live: `slot_ids` holds `pieces`
    /// chunk ids a lane (and `snapshots.slots` as many an entry), and
    /// `slot_stride_elems` (`conv_stride` for a conv row) is a chunk's span.
    pub segments: Segments,
}

/// How a recurrent pool's fixed rows sit in its chunks
/// (`prim/state_rows.hip`): a row splits into `pieces` equal pieces along the
/// reading op's atom axis, each in a chunk of its own, `piece_off` elements
/// into it; chunk `c` is chunk `c % seg_chunks` of segment `c / seg_chunks`,
/// whose base address is entry `c / seg_chunks` of `table` (u64). An absent
/// table keeps every chunk in the one segment at the slab's pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segments {
    pub table: Tensor,

    pub seg_chunks: u32,

    pub pieces: u32,

    pub piece_off: i64,

    /// The atoms a piece holds, where the pieces are not an equal split of
    /// the row's atoms (the last one padded past them); 0: `atoms / pieces`.
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

    /// Where `op`'s rows of `slab` live, named by the ids `ids`, a row
    /// `atoms` atoms of the op's cut axis and chunks `chunk` elements apart.
    pub fn rows(
        &self,
        op: &'static str,
        slab: Tensor,
        ids: Tensor,
        chunk: i64,
        atoms: u32,
    ) -> Result<StateRows, crate::error::Error> {
        let refuse = |detail: String| crate::error::Error::Backend { op, detail };
        let pieces = self.pieces.max(1);
        let per_piece = if self.piece_atoms != 0 {
            self.piece_atoms
        } else {
            atoms / pieces
        };
        if atoms == 0
            || (self.piece_atoms == 0 && !atoms.is_multiple_of(pieces))
            || u64::from(per_piece) * u64::from(pieces) < u64::from(atoms)
            || u64::from(per_piece) * u64::from(pieces - 1) >= u64::from(atoms)
        {
            return Err(refuse(format!(
                "a state row of {atoms} atoms does not lie in {pieces} pieces of {per_piece}"
            )));
        }
        if !self.table.is_absent() && self.seg_chunks == 0 {
            return Err(refuse(
                "the state's segment table holds no chunk a segment".to_string(),
            ));
        }
        let narrow = |v: u32, what: &str| {
            i32::try_from(v).map_err(|_| refuse(format!("{v} {what} overflow an i32")))
        };
        Ok(StateRows {
            base: slab.ptr,
            segs: self.table.ptr,
            ids: ids.ptr,
            chunk,
            off: self.piece_off,
            seg_chunks: narrow(self.seg_chunks, "chunks a segment")?,
            pieces: narrow(pieces, "pieces a row")?,
            atoms: narrow(per_piece, "atoms a piece")?,
            pad: 0,
        })
    }
}

/// The kernels' `StateRows` (`prim/state_rows.hip`), passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateRows {
    base: u64,
    segs: u64,
    ids: u64,
    chunk: i64,
    off: i64,
    seg_chunks: i32,
    pieces: i32,
    atoms: i32,
    pad: i32,
}

impl StateRows {
    /// The kernels' argument.
    #[must_use]
    pub fn arg(&self) -> ArgValue<'_> {
        ArgValue::of(self)
    }

    /// The pieces a row splits into.
    #[must_use]
    pub const fn pieces(&self) -> u32 {
        self.pieces as u32
    }

    /// The atoms a piece holds.
    #[must_use]
    pub const fn atoms(&self) -> u32 {
        self.atoms as u32
    }

    /// Whether every piece starts a whole `align` elements from its
    /// segment's base (the chunk span and the piece offset are multiples; a
    /// segment's base is page-aligned).
    #[must_use]
    pub const fn keeps(&self, align: i64) -> bool {
        self.chunk % align == 0 && self.off % align == 0
    }
}

impl RecurrentPool {
    /// Where op `op`'s rows of `slab` live, a row `atoms` atoms of the op's
    /// cut axis and chunks `chunk` elements apart.
    pub fn rows(
        &self,
        op: &'static str,
        slab: Tensor,
        chunk: i64,
        atoms: u32,
    ) -> Result<StateRows, crate::error::Error> {
        self.segments.rows(op, slab, self.slot_ids, chunk, atoms)
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
