use kernels_vulkan::Tensor;
use model_compiler::{ArenaMap, FireRows};
use model_exec::store::arena::rect;
use model_ir::ValueId;

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::run::SlotTable;

#[derive(Debug)]
pub struct Arena {
    store: Buffer,
}

impl Arena {
    pub fn reserve(device: &Context, map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(device, map.bytes)?,
        })
    }

    #[must_use]
    pub fn store(&self) -> &Buffer {
        &self.store
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    pub fn slots(&self, handles: &Handles, map: &ArenaMap, rows: FireRows) -> Result<SlotTable> {
        carve(handles, &self.store, map, rows)
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.store.read(offset, into)
    }

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

pub fn carve(
    handles: &Handles,
    store: &Buffer,
    map: &ArenaMap,
    fire: FireRows,
) -> Result<SlotTable> {
    let mut rows: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);

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
