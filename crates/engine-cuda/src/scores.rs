//! A capturing lane's attention score rows (`attn.scores`): state of the
//! instance reading them, not the fire's. The rows lie in the reader's
//! `Instance` slots, after its own item (`program::Plane::held`), so the
//! pool holds them where the runtime holds the instance: a bind, a park
//! and an evacuation move them with it, and nothing is backed for them at
//! load or at bind. Each fire stages a table of one row address a fire
//! lane (0 for a lane with no reader in the fire), and the score kernels
//! write through it; the reader reads its own rows.

use kernels_cuda::attn::score::ScoreRows as KernelRows;
use poem_ir::{Dim, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::exports::Exports;

/// The name a trace's captured scores are read out by.
pub const SCORES: &str = "attn.scores";

/// The load's score rows: each layer's plane, in the order a reader reads
/// them, and the fire's row table.
#[derive(Debug)]
pub struct ScoreRows {
    /// Each layer's `attn.scores` value, one plane after another in a
    /// lane's rows.
    planes: Vec<ValueId>,
    /// The floats one plane holds a lane: its heads' runs of keys.
    width: u32,
    /// One row address a fire lane, a run of the activation's span laid
    /// with the staged inputs (`Shell::staged_runs`).
    table: Buffer,
    lanes: u32,
    /// Whether the table names a row: a fire with no reader stages it
    /// only to clear what an earlier one named.
    named: bool,
}

/// What a fire's dispatch reads of the score rows.
#[derive(Debug, Clone)]
pub struct ScoreSeat {
    table: u64,
    lanes: u32,
    width: u32,
    planes: std::sync::Arc<[ValueId]>,
}

impl ScoreSeat {
    /// Where the capture writing `value` (a layer's plane) writes.
    #[must_use]
    pub fn rows(&self, value: ValueId) -> Option<KernelRows> {
        let plane = self.planes.iter().position(|&at| at == value)?;
        Some(KernelRows {
            table: self.table,
            lanes: self.lanes,
            plane: plane as u64 * u64::from(self.width),
        })
    }
}

impl ScoreRows {
    /// The score rows of a load whose trace reads scores out, a table of
    /// `lanes` rows; `None` where it reads none.
    pub(crate) fn of(trace: &Trace, exports: &Exports, lanes: u32) -> Result<Option<ScoreRows>> {
        let Some(out) = exports.outputs.iter().find(|out| out.decl.name == SCORES) else {
            return Ok(None);
        };
        let mut width = None;
        for &value in &out.values {
            let own = match trace.values.get(value.0 as usize).map(|decl| &decl.ty) {
                Some(Ty::Tensor { shape, .. }) => match shape.as_slice() {
                    [Dim::Lanes, Dim::Const(width)] => u32::try_from(*width).ok(),
                    _ => None,
                },
                _ => None,
            };
            let Some(own) = own.filter(|own| width.is_none_or(|width| width == *own)) else {
                return Err(Fault::program(
                    "scores",
                    format!(
                        "`{SCORES}`'s value {} is not a row a lane as wide as the other \
                         layers'",
                        value.0
                    ),
                ));
            };
            width = Some(own);
        }
        Ok(width.map(|width| ScoreRows {
            planes: out.values.clone(),
            width,
            table: Buffer::lent(lanes as usize * size_of::<u64>()),
            lanes,
            named: false,
        }))
    }

    /// A lane's rows: every layer's plane, one after another.
    #[must_use]
    pub fn lane_bytes(&self) -> u64 {
        self.planes.len() as u64 * u64::from(self.width) * 4
    }

    /// The floats one plane holds a lane: a reader reads layer `j` that
    /// many floats past layer `j - 1`.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The table's bytes, a run of the activation's span.
    #[must_use]
    pub fn table_bytes(&self) -> usize {
        self.table.bytes()
    }

    /// The table lies at `ptr` from here on.
    pub fn lay(&mut self, ptr: u64) {
        self.table.lay(ptr);
    }

    #[must_use]
    pub fn seat(&self) -> ScoreSeat {
        ScoreSeat {
            table: self.table.ptr(),
            lanes: self.lanes,
            width: self.width,
            planes: self.planes.clone().into(),
        }
    }

    /// Stages the fire's table on `stream`: `rows[lane]`, the row address
    /// of the fire's lane `lane` (in the fire's order), 0 where none.
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, rows: &[u64]) -> Result<()> {
        if rows.len() > self.lanes as usize {
            return Err(Fault::Ceiling {
                what: "lanes of the score row table",
                need: rows.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let named = rows.iter().any(|&row| row != 0);
        if !named && !self.named {
            return Ok(());
        }
        self.named = named;
        let mut bytes = vec![0u8; self.table.bytes()];
        for (at, row) in rows.iter().enumerate() {
            bytes[at * 8..at * 8 + 8].copy_from_slice(&row.to_le_bytes());
        }
        self.table.stage(stream, 0, &bytes)
    }
}
