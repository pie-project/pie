//! Where a routed bank's experts, and a gathered table's row blocks, lie:
//! the weight cache's location tables (`linear/moe/table.cuh`).
//!
//! A routed bank is one or more planes, each `[experts, ...]` when
//! contiguous. An [`ExpertTable`] names, for every expert, the device address
//! of each of its planes, `table[e * planes + p]` in the plane order of the
//! bank's [`Planes`]. A null table is the contiguous bank (expert `e`'s plane
//! at `base + e * stride`): the degenerate table, which a kernel computes
//! rather than reads. A table equal to the contiguous bank's addresses lands
//! the same bits as no table.

/// The planes of a routed bank's kind, in the order an expert's table
/// entries list them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planes {
    /// One plane: bf16 / f16 / fp8 weights, `[experts, n, k]`.
    Dense,
    /// `[codes, scales, bias]`: MXFP4 (bias 0 where the bank has none).
    Mxfp4,
    /// `[codes, scales, tensor_scales]`: NVFP4 against bf16 activations.
    Nvfp4,
    /// `[codes, scales, zeros]`: affine u4 (mlx: `[codes, scales, biases]`).
    Affine,
    /// `[codes, scales]`: a block-scaled bank whose scales are their own
    /// plane (W8A16 / W8A8 block FP8, MXFP4 against e4m3 activations, either
    /// laid in atoms).
    Scaled,
    /// `[codes, atoms, tensor_scales, input_scales]`: NVFP4 against NVFP4
    /// activations.
    Nvfp4A4,
}

impl Planes {
    /// Entries an expert.
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            Planes::Dense => 1,
            Planes::Scaled => 2,
            Planes::Mxfp4 | Planes::Nvfp4 | Planes::Affine => 3,
            Planes::Nvfp4A4 => 4,
        }
    }
}

/// Where a routed bank's experts lie.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTable {
    /// A device array of [`Planes::count`] u64 addresses an expert; 0 for
    /// the contiguous bank.
    pub table: u64,
    /// The expert tier's u32 hit counter an expert, or 0.
    pub hits: u64,
    /// Whether every entry names device memory (the weight cache's chunks or
    /// staging slots): any select takes such a table. Otherwise an entry may
    /// name mapped host memory (the expert tier's), and only the selects that
    /// read through plain loads take it.
    pub located: bool,
    /// `(base, bytes)`: the span every entry falls in, which a TMA-addressed
    /// kernel maps (an entry's offset past `base` a whole number of the
    /// plane's rows); `(0, 0)` maps the bank's own extent.
    pub window: (u64, u64),
}

impl ExpertTable {
    pub const RESIDENT: ExpertTable = ExpertTable {
        table: 0,
        hits: 0,
        located: false,
        window: (0, 0),
    };

    /// A table of device addresses within `window` (`(0, 0)`: no TMA
    /// reader maps past the bank).
    #[must_use]
    pub const fn located(table: u64, window: (u64, u64)) -> ExpertTable {
        ExpertTable {
            table,
            hits: 0,
            located: true,
            window,
        }
    }

    /// Whether the table may name mapped host memory: the fast selects
    /// (tensor-core, grouped, TMA) leave such a bank to the per-route GEMV.
    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0 && !self.located
    }

    /// Whether there is a table at all: a select whose kernels take none
    /// leaves any tabled bank to one whose do.
    #[must_use]
    pub const fn tabled(&self) -> bool {
        self.table != 0
    }

    /// The table pointer a kernel takes, 0 for the contiguous bank.
    #[must_use]
    pub const fn at(&self) -> u64 {
        self.table
    }
}

/// Where a token-id-gathered table's (PLE's) row blocks lie: row block `b`'s
/// plane `p` at `table[b * planes + p]`, row `r` within it `(r % block_rows)
/// * row_stride` past that. A null table is the contiguous table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowTable {
    pub table: u64,
    /// Rows a block; any nonzero value under a null table.
    pub block_rows: u32,
}

impl RowTable {
    pub const RESIDENT: RowTable = RowTable {
        table: 0,
        block_rows: 1,
    };

    #[must_use]
    pub const fn located(table: u64, block_rows: u32) -> RowTable {
        RowTable { table, block_rows }
    }
}

/// The contiguous bank's table: `planes` given as `(base, stride)` in bytes,
/// expert `e`'s entry `p` at `base_p + e * stride_p` (a plane the bank lacks
/// as `(0, 0)`). Uploaded, it lands what no table does.
#[must_use]
pub fn contiguous_entries(experts: u32, planes: &[(u64, u64)]) -> Vec<u64> {
    (0..u64::from(experts))
        .flat_map(|e| planes.iter().map(move |&(base, stride)| base + e * stride))
        .collect()
}
