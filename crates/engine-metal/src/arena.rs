#![allow(unsafe_code)]
use kernels_metal::Tensor;
use poem_compiler::{ArenaMap, FireRows};
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::device::{Buffer, Context, Handles};
use crate::error::Result;
use crate::run::SlotTable;

/// Scratch is sized for the widest fire and most fires are narrow. Held in
/// sparse heaps, it costs the widest carve a fire has used: a smaller carve
/// is a prefix of the store, and a heap costs nothing until a frame uses it,
/// then all of itself. A device with no sparse buffers holds it whole.
#[derive(Debug)]
pub struct Arena {
    store: Buffer,
    #[cfg(target_vendor = "apple")]
    sparse: Option<(
        crate::device::elastic::Elastic,
        crate::device::elastic::Arena,
    )>,
}

impl Arena {
    #[cfg(target_vendor = "apple")]
    pub fn reserve(device: &Context, map: &ArenaMap) -> Result<Arena> {
        use crate::device::elastic;
        let sparse = device.sparse()?.filter(|_| map.bytes > 0);
        let Some(sparse) = sparse else {
            return Ok(Arena {
                store: Buffer::zeroed(device, map.bytes)?,
                sparse: None,
            });
        };
        let budget = elastic::Arena::new(elastic::pages_up(map.bytes));
        let store = elastic::create(&sparse, &budget, map.bytes)?;
        Ok(Arena {
            store: store.view(),
            sparse: Some((store, budget)),
        })
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn reserve(device: &Context, map: &ArenaMap) -> Result<Arena> {
        Ok(Arena {
            store: Buffer::zeroed(device, map.bytes)?,
        })
    }

    /// Maps the store up to `bytes`, the carve the next fire uses. Nothing
    /// reads scratch before a kernel writes it, so nothing is zeroed.
    #[cfg(target_vendor = "apple")]
    pub fn cover(&mut self, bytes: u64) -> Result<()> {
        if let Some((store, _)) = self.sparse.as_mut() {
            crate::device::elastic::map_all(&mut [crate::device::elastic::Target {
                bytes,
                buffer: store,
            }])?;
        }
        Ok(())
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn cover(&mut self, _bytes: u64) -> Result<()> {
        Ok(())
    }

    /// Lets go of the store past `bytes`; the next wider fire maps it again.
    ///
    /// # Safety
    ///
    /// No frame may be in flight.
    #[cfg(target_vendor = "apple")]
    pub unsafe fn forget(&mut self, bytes: u64) -> Result<()> {
        let Some((store, _)) = self.sparse.as_mut() else {
            return Ok(());
        };
        // SAFETY: the caller's contract.
        unsafe {
            crate::device::elastic::shrink_all(&mut [crate::device::elastic::Target {
                bytes,
                buffer: store,
            }])
        }
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
        #[cfg(target_vendor = "apple")]
        if let Some((sparse, _)) = &self.sparse {
            return sparse.read_into(offset, into);
        }
        self.store.read(offset, into)
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
            let Some(poem_compiler::Placement::Arena {
                bytes,
                width,
                dtype,
                ..
            }) = map.placements.get(root.0 as usize)
            else {
                return 0;
            };
            let row = width.saturating_mul(poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0));
            if row == 0 {
                return 0;
            }
            u32::try_from(bytes / row).unwrap_or(u32::MAX)
        })
        .collect()
}
