//! §3.1 — [`Launch`], and the two conveniences that are not vocabulary.
//!
//! # What died here
//!
//! `LaunchRule` was a forty-variant enum: one variant per launcher shape,
//! each a small arithmetic program in data, each needing a `LaunchRule::eval`
//! arm, a Rust port, a C++ port and a row column. A launcher whose geometry
//! fitted none of the forty got `LaunchRule::Unstated` and stayed hand-
//! written, which is to say the enum's real job was to be *incomplete in a
//! way a table could record*.
//!
//! A `fn` needs none of that. The grid is an expression, and an expression
//! that fits no pattern is still an expression.
//!
//! **[`Launch::flat`] and [`Launch::per_row`] are conveniences, not
//! vocabulary.** They exist because two shapes recur often enough that
//! writing the ceiling division out every time invites a typo, not because a
//! launch must be one of them. A kernel that fits neither writes the
//! literal:
//!
//! ```ignore
//! Launch { grid: [tokens, heads / per_block, 1], block: [128, 1, 1], smem, smem_opt_in: false }
//! ```
//!
//! There is no 41st variant to add. That is the whole point.
//!
//! # The war story these two carry
//!
//! `LaunchRule::Rope` — the rule that served this family before the port —
//! sized `smem` as `cache_pairs * 2 * sizeof(float)` while the launcher it
//! described passed `0`. `rotate_yarn` declares no `extern __shared__`, so
//! the allocation was unread: it cost occupancy and could not change a byte
//! of output. The bug was invisible because it was *generous*. A rule that
//! had allocated too LITTLE would have been a silent out-of-bounds read on
//! the first fire, and nothing in the row could have caught either.
//!
//! `smem` here is written by the same function that writes the `extern
//! __shared__` read — they are two lines of one program — which is the
//! structural reason the asymmetry cannot recur.

/// One launch's geometry.
///
/// The value a host program hands the fire path. Its fields are exactly what
/// `cuLaunchKernel` takes and nothing else: no rule, no context, no
/// arithmetic — those happened in the `fn` that built it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
    /// This launch needs more than the 48 KB default cap, and the fire path
    /// must raise it with `cuFuncSetAttribute` before launching.
    ///
    /// **Derived, and checked rather than trusted.** `runtime::module`'s
    /// `fire` already raises the cap whenever `launch.smem > 48 KB` — the
    /// driver-level fact is a threshold, not a flag, and a flag that
    /// disagreed with the threshold would be a launch that fails with
    /// `CUDA_ERROR_INVALID_VALUE` for a reason no reader could see. So this
    /// field is a STATEMENT BY THE AUTHOR that the large allocation is
    /// intended, [`Launch::opt_in_needed`] is the driver fact, and
    /// [`Launch::disagrees`] is where the two are compared.
    pub smem_opt_in: bool,
}

/// The default dynamic shared-memory cap, in bytes.
///
/// `runtime::module`'s `DEFAULT_DYNAMIC_SMEM`, restated here because this is
/// the layer that decides whether to set `smem_opt_in` and it must not have
/// to guess. Above this, `cuFuncSetAttribute` with
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` is required or the
/// launch fails outright.
pub const OPT_IN_ABOVE: u32 = 48 * 1024;

impl Launch {
    /// ONE THREAD PER ELEMENT, in blocks of `block`.
    ///
    /// `grid.x = ceil(n / block)` and nothing on the other two axes. The
    /// shape of every elementwise launcher in the archive.
    ///
    /// **No `rope` kernel uses it**, and that is worth writing down here
    /// rather than discovering per family: `rope`'s ten `__global__`s are
    /// all one block per token, so they take [`Self::per_row`]. The
    /// convenience is kept because the shape is the archive's commonest and
    /// because a `Launch` vocabulary with only the shapes the pilot needed
    /// would be a vocabulary the second family has to extend — which is the
    /// `LaunchRule` mistake in miniature.
    ///
    /// `n <= 0` is the caller's business, not this function's: a `fn` that
    /// can be handed an empty extent refuses in its own words before it
    /// builds a geometry. This one saturates so that a slipped check is a
    /// no-op launch rather than a `grid.x` of four billion.
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self { grid: [grid, 1, 1], block: [block, 1, 1], smem: 0, smem_opt_in: false }
    }

    /// ONE BLOCK PER ROW, `block` threads wide.
    ///
    /// The shape of every row-parallel launcher: a norm, a softmax, a
    /// per-token rotation. `rope`'s fused QK-norm rotations are this with
    /// `rows = num_tokens` and `block = 128` (`rope.cu:45`).
    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self { grid: [rows, 1, 1], block: [block, 1, 1], smem: 0, smem_opt_in: false }
    }

    /// The same launch with `smem` bytes of dynamic shared memory.
    ///
    /// Sets [`Launch::smem_opt_in`] from the size, because the author who
    /// writes the byte count is the author who reads it in the kernel and
    /// there is no third party to disagree with.
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
    /// disagree.
    ///
    /// Only reachable through a struct literal that sets both by hand. The
    /// fire path asserts on it, which is the check `LaunchRule::Rope`'s
    /// generous-allocation bug never had.
    #[must_use]
    pub const fn disagrees(&self) -> bool {
        self.smem_opt_in != self.opt_in_needed()
    }

    /// Nothing to launch — an axis is zero.
    ///
    /// A `grid` or `block` with a zero axis is `CUDA_ERROR_INVALID_VALUE`,
    /// not a no-op, so a caller that can produce one tests here.
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
    ///
    /// It is not lost: `runtime::module`'s fire path derives the same fact
    /// from `smem` and raises the cap itself, keyed per `(device, function)`
    /// so the `cuFuncSetAttribute` happens once. The flag's job was to be
    /// READ — by a human, and by the assertion in `x::fire` — and it has
    /// been read by the time this conversion runs.
    fn from(l: Launch) -> Self {
        Self { grid: l.grid, block: l.block, smem: l.smem }
    }
}
