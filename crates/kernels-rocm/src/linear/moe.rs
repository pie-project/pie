#![allow(unsafe_code)]
use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Kernel, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

const FILE: &str = "linear/moe.hip";

const BLOCK: u32 = 256;

const MAX_EXPERTS: u32 = 512;

mod grouped;
#[cfg(feature = "rocm")]
mod lt;
mod mlp;
mod mx;
mod route;
mod select;

pub use grouped::*;
#[cfg(all(feature = "rocm", feature = "test-support"))]
pub use lt::lt_grouped_fires;
#[cfg(feature = "rocm")]
use lt::*;
pub use mlp::*;
pub use mx::*;
pub use route::*;
pub use select::*;

struct Selected {
    route_count: u32,

    top_k: i32,

    by_token: bool,
}

fn selected(op: &'static str, x: Tensor, routes: Tensor, y: &Tensor) -> Result<Selected, Error> {
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{op}` walks i32 routes");
    let top_k = nonzero(op, "the routed fan-out", routes.width)?;
    let route_count = routes.rows.checked_mul(top_k).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the route run will not launch: {} tokens x {top_k} fan-out",
                routes.rows
            ),
        )
    })?;
    debug_assert_eq!(y.rows, route_count, "the result lands one row per route");
    let by_token = if x.rows == route_count {
        false
    } else if x.rows == routes.rows {
        true
    } else {
        return Err(refuse(
            op,
            format!(
                "the activation's {} rows are neither the fire's tokens nor its routes",
                x.rows
            ),
        ));
    };
    Ok(Selected {
        route_count,
        top_k: stated(op, top_k)?,
        by_token,
    })
}

/// The planes of a routed bank's kind, in the order an expert's table
/// entries list them (`linear/table.hip`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planes {
    /// One plane: bf16 / f16 weights, `[experts, n, k]`.
    Dense,
    /// `[codes, scales, bias]`: MXFP4 (bias 0 where the bank has none).
    Mxfp4,
    /// `[codes, scales, tensor_scales]`: NVFP4 (no ROCm select reads one).
    Nvfp4,
    /// `[codes, scales, zeros]`: affine (mlx: `[codes, scales, biases]`).
    Affine,
    /// `[codes, scales]`: a block-scaled bank whose scales are their own
    /// plane (`e4m3tile128`'s f32 factors, an MXFP4 bank in lane order).
    Scaled,
    /// `[codes, atoms, tensor_scales, input_scales]`: NVFP4 against NVFP4
    /// activations (no ROCm select reads one).
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

/// Where a routed bank's experts lie: the weight cache's expert table.
///
/// A routed bank is one or more planes, each `[experts, ...]` when
/// contiguous. The table names, for every expert, the address of each of its
/// planes, `table[e * planes + p]` in the plane order of the bank's
/// [`Planes`]. A null table is the contiguous bank (expert `e`'s plane at
/// `base + e * stride`): the degenerate table, which a kernel computes
/// rather than reads. A table equal to the contiguous bank's addresses lands
/// the same bits as no table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertTable {
    /// A device array of [`Planes::count`] u64 addresses an expert; 0 for
    /// the contiguous bank.
    pub table: u64,
    /// A u32 counter an expert of the routes that read it, or 0.
    pub hits: u64,
    /// Whether every entry names device memory (the weight cache's chunks or
    /// staging slots): any select takes such a table. Otherwise an entry may
    /// name mapped host memory (the expert tier's), and only the per-route
    /// stream selects (one plain read of an expert's rows a route) take it.
    pub located: bool,
    /// `(base, bytes)`: the span every entry falls in, for a reader that
    /// maps it. No HIP kernel maps a bank (there is no TMA), so this rides
    /// along for the CUDA twin's sake.
    pub window: (u64, u64),
}

impl ExpertTable {
    pub const RESIDENT: ExpertTable = ExpertTable {
        table: 0,
        hits: 0,
        located: false,
        window: (0, 0),
    };

    /// A table of device addresses within `window`.
    #[must_use]
    pub const fn located(table: u64, window: (u64, u64)) -> ExpertTable {
        ExpertTable {
            table,
            hits: 0,
            located: true,
            window,
        }
    }

    /// The table counting each expert's routes in `hits`.
    #[must_use]
    pub const fn counting(mut self, hits: u64) -> ExpertTable {
        self.hits = hits;
        self
    }

    /// Whether the table may name mapped host memory: the batch, grouped
    /// and matrix-core selects leave such a bank to the per-route stream.
    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0 && !self.located
    }

    /// Whether there is a table at all.
    #[must_use]
    pub const fn tabled(&self) -> bool {
        self.table != 0
    }

    /// The table pointer a kernel takes, 0 for the contiguous bank.
    #[must_use]
    pub const fn at(&self) -> u64 {
        self.table
    }

    /// The table and the counter, as a kernel's last two arguments take them.
    fn args(&self) -> [ArgValue<'static>; 2] {
        [ArgValue::Ptr(self.table), ArgValue::Ptr(self.hits)]
    }
}

/// Where a token-id-gathered table's row blocks lie, as the CUDA kernels
/// take it: a u64 address a block of `block_rows` rows (null: the
/// contiguous table), and a u32 read counter a block (`hits`, or 0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowTable {
    pub table: u64,
    pub block_rows: u32,
    pub hits: u64,
}

impl RowTable {
    pub const RESIDENT: RowTable = RowTable {
        table: 0,
        block_rows: 1,
        hits: 0,
    };

    #[must_use]
    pub const fn located(table: u64, block_rows: u32) -> RowTable {
        RowTable {
            table,
            block_rows,
            hits: 0,
        }
    }

    #[must_use]
    pub const fn counting(mut self, hits: u64) -> RowTable {
        self.hits = hits;
        self
    }

    #[must_use]
    pub const fn streams(&self) -> bool {
        self.table != 0
    }
}
