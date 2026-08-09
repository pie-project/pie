/// One launch's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
    /// This launch needs more than the 48 KB default cap, and the fire path
    pub smem_opt_in: bool,
}

/// The default dynamic shared-memory cap, in bytes.
pub const OPT_IN_ABOVE: u32 = 48 * 1024;

impl Launch {
    /// ONE THREAD PER ELEMENT, in blocks of `block`.
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self { grid: [grid, 1, 1], block: [block, 1, 1], smem: 0, smem_opt_in: false }
    }

    /// ONE BLOCK PER ROW, `block` threads wide.
    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self { grid: [rows, 1, 1], block: [block, 1, 1], smem: 0, smem_opt_in: false }
    }

    /// The same launch with `smem` bytes of dynamic shared memory.
    #[must_use]
    pub const fn smem(mut self, bytes: u32) -> Self {
        self.smem = bytes;
        self.smem_opt_in = bytes > OPT_IN_ABOVE;
        self
    }

    /// This launch's `smem` needs the cap raised.
    #[must_use]
    pub const fn opt_in_needed(&self) -> bool {
        self.smem > OPT_IN_ABOVE
    }

    /// The author's [`Launch::smem_opt_in`] and the driver's threshold
    #[must_use]
    pub const fn disagrees(&self) -> bool {
        self.smem_opt_in != self.opt_in_needed()
    }

    /// Nothing to launch — an axis is zero.
    #[must_use]
    pub const fn empty(&self) -> bool {
        self.grid[0] == 0
            || self.grid[1] == 0
            || self.grid[2] == 0
            || self.block[0] == 0
            || self.block[1] == 0
            || self.block[2] == 0
    }
}

#[cfg(feature = "_cuda")]
impl From<Launch> for crate::runtime::Launch {
    /// Drop [`Launch::smem_opt_in`] at the fire boundary.
    fn from(l: Launch) -> Self {
        Self { grid: l.grid, block: l.block, smem: l.smem }
    }
}
