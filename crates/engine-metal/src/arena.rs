//! The activation arena: one allocation for the model's whole load, and the
//! per-fire table that resolves the compiler's static offsets to handles
//! (a Metal view is a `u32` row into [`Handles`], not an address).

use kernels_metal::Tensor;
use model_compiler::{ArenaMap, FireRows};
use model_exec::store::arena::rect;
use model_ir::ValueId;

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::run::SlotTable;

/// The device bytes every arena rectangle is an offset into.
#[derive(Debug)]
pub struct Arena {
    store: Buffer,
}

impl Arena {
    /// Reserve what the carve says the largest admissible fire needs. Errs [`Fault::Deviceless`] for a non-Apple build, [`Fault::Ceiling`] for an arena past `maxBufferLength`, [`Fault::Device`] when the device declined the length.
    pub fn reserve(device: &Context, map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(device, map.bytes)?,
        })
    }

    /// The reservation itself, for whoever needs to mint a view into it that
    /// is not one of the carve's. No `base()`: use [`Handles::bind`] to name
    /// a view.
    #[must_use]
    pub fn store(&self) -> &Buffer {
        &self.store
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// The table a fire of `tokens` rows over `lanes` requests resolves through. `handles` is the fire's minting table, dropped wholesale by [`Handles::rewind`] when the fire ends. Errs [`Fault::Ceiling`] for a rectangle that leaves the reservation, or a full handle table.
    pub fn slots(
        &self,
        handles: &Handles,
        map: &ArenaMap,
        rows: FireRows,
    ) -> Result<SlotTable> {
        carve(handles, &self.store, map, rows)
    }

    /// Copy `into.len()` bytes back from `offset` bytes into this arena. Takes an offset, not an address — a caller holding the seam's handle wants [`Arena::read_view`] instead. Correct only once the fire's command buffer has been committed and waited on.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.store.read(offset, into)
    }

    /// Copy `into.len()` bytes back from `skip` bytes into whatever `handle` names. `handle` must name a view of this arena; the handle table spans every reservation the load made.
    pub fn read_view(
        &self,
        handles: &Handles,
        handle: u32,
        skip: u64,
        into: &mut [u8],
    ) -> Result<()> {
        let at = {
            let row = handles.get(handle).ok_or_else(|| Fault::Unbound {
                what: format!("handle {handle}, which this load minted no row for"),
            })?;
            row.offset()
        };
        let at = at.checked_add(skip).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: u64::MAX,
            have: self.store.bytes(),
        })?;
        self.store.read(at, into)
    }
}

/// One fire's slot table: the carve's arithmetic, and one minted handle per rectangle it names. Separated from [`Arena`] so it can be driven over a buffer the caller chose. Errs [`Fault::Ceiling`] for a rectangle that leaves `store`, or a full handle table.
pub fn carve(
    handles: &Handles,
    store: &Buffer,
    map: &ArenaMap,
    fire: FireRows,
) -> Result<SlotTable> {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        // FireRows::text_only sizes Dim::Patches/Dim::Images at zero rather than faulting.
        rows.push(match rect(map, value, fire) {
            Some(rect) => Some(Tensor::new(
                handles.bind(store, rect.offset, rect.bytes)?,
                rect.rows,
                rect.width,
                rect.dtype,
            )),
            None => None,
        });
    }
    Ok(SlotTable(rows))
}

/// How many rows each value's slot can hold, `ValueId`-indexed — the slot's reservation, not the fire's extent. `0` for a value with no rectangle or no element byte size.
#[must_use]
pub fn capacities(map: &ArenaMap) -> Vec<u32> {
    (0..map.placements.len())
        .map(|at| {
            let root = map.root(ValueId(at as u32));
            let Some(model_compiler::Placement::Arena {
                bytes,
                width,
                dtype,
                ..
            }) = map.placements.get(root.0 as usize)
            else {
                return 0;
            };
            let row = width.saturating_mul(model_compiler::arena::elem_bytes(*dtype).unwrap_or(0));
            if row == 0 {
                return 0;
            }
            u32::try_from(bytes / row).unwrap_or(u32::MAX)
        })
        .collect()
}
