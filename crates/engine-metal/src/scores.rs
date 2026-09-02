//! The observability slab: device bytes where the attention capture arm
//! writes its per-key mass, sized `lanes x planes x ATTN_SCORE_KV_MAX x 4B`
//! and read back host-side only between fires via [`Scores::read_lane`].

use kernels_metal::Tensor;
use model_ir::{Dtype, ValueId};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

/// Width of every score row; shared with the guest DSL constant.
pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

/// One F32 slot, in bytes.
const SLOT: u64 = 4;

/// Observation window: how many query rows at the end of each request the capture folds into the published row. Matches SnapKV and the CUDA shell's own constant.
pub(crate) const OBSERVE: u32 = 32;

/// This load's score slab, plus the numbers a launch needs to address it.
#[derive(Debug)]
pub(crate) struct Scores {
    store: Buffer,
    /// Slab planes per lane block: exported attention layers × query heads.
    planes: u32,
    /// How many query heads each exported layer contributes.
    heads: u32,
    /// How many fire lanes the slab seats — the budget's `max_lanes`.
    lanes: u32,
    /// `(value the capture arm writes, its plane base)`, one entry per
    /// exported column. Keyed by export position, not transformer layer,
    /// since a hybrid text's recurrent layers export nothing.
    planes_of: Vec<(ValueId, u32)>,
}

impl Scores {
    /// Reserve the slab for a load whose plan declares `exports` score columns, or `None` for one that declares none. Errs [`Fault::Device`] for the allocation, [`Fault::Ceiling`] when demand exceeds the device.
    pub(crate) fn reserve(
        device: &Context,
        exports: &[ValueId],
        heads: u32,
        lanes: u32,
    ) -> Result<Option<Scores>> {
        if exports.is_empty() || heads == 0 || lanes == 0 {
            return Ok(None);
        }
        let planes = u32::try_from(exports.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(heads);
        let bytes = u64::from(lanes)
            .saturating_mul(u64::from(planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT);
        let planes_of = exports
            .iter()
            .enumerate()
            .map(|(at, value)| (*value, u32::try_from(at).unwrap_or(0).saturating_mul(heads)))
            .collect();
        Ok(Some(Scores {
            store: Buffer::zeroed(device, bytes)?,
            planes,
            heads,
            lanes,
            planes_of,
        }))
    }

    /// The allocation itself.
    pub(crate) fn store(&self) -> &Buffer {
        &self.store
    }

    /// How many planes one lane's block holds — the row pitch between lanes.
    #[must_use]
    pub(crate) fn planes(&self) -> u32 {
        self.planes
    }

    /// How many query heads each exported layer contributes.
    #[must_use]
    pub(crate) fn heads(&self) -> u32 {
        self.heads
    }

    /// How many fire lanes the slab seats.
    #[must_use]
    pub(crate) fn lanes(&self) -> u32 {
        self.lanes
    }

    /// Device bytes held, for the footprint line.
    #[must_use]
    pub(crate) fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    /// Where lane `lane`'s block of planes begins, in bytes into the store.
    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        u64::from(lane)
            .saturating_mul(u64::from(self.planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT)
    }

    /// Copy one lane's whole block of planes back to the host. Called between fires only, never on the fire path. Errs [`Fault::Ceiling`] for a lane past the slab.
    pub(crate) fn read_lane(&self, lane: u32) -> Result<Vec<f32>> {
        if lane >= self.lanes {
            return Err(Fault::Ceiling {
                what: "fire lanes the score slab seats",
                need: u64::from(lane) + 1,
                have: u64::from(self.lanes),
            });
        }
        let floats = self.planes as usize * KV_MAX as usize;
        let mut raw = vec![0u8; floats * SLOT as usize];
        self.store.read(self.lane_base(lane), &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    /// The per-fire seat: the rectangle a capture launch writes, and the `(value, plane base)` pairs the dispatch arm asks for by node. Minted per fire since [`Handles::rewind`] drops everything a fire minted.
    pub(crate) fn seat(&self, handles: &Handles) -> Result<ScoreSeat> {
        let rows = self.lanes.saturating_mul(self.planes);
        let bytes = u64::from(rows)
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT);
        Ok(ScoreSeat {
            slab: Tensor::new(handles.bind(&self.store, 0, bytes)?, rows, KV_MAX, Dtype::F32),
            plane_stride: self.planes,
            observe: OBSERVE,
            planes_of: self.planes_of.clone(),
        })
    }
}

/// The seat the dispatch arm reads — everything a capture launch needs.
/// Rides [`FireBindings`](crate::run::FireBindings) since it carries a list,
/// unlike the `Copy` [`FireTables`](crate::run::FireTables).
#[derive(Clone, Debug)]
pub struct ScoreSeat {
    /// The whole slab, handed over as one rectangle. Excluded from the window
    /// re-cut: its rows are (lane, plane) pairs, not fire rows.
    pub slab: Tensor,
    /// Slab planes per lane block.
    pub plane_stride: u32,
    /// The observation window this load captures over.
    pub observe: u32,
    /// `(the value a capture arm writes, its plane base)`, one per exported
    /// column.
    pub planes_of: Vec<(ValueId, u32)>,
}

impl ScoreSeat {
    /// The plane base of the capture arm that writes `value`, or `None` when
    /// this node is not a score export.
    #[must_use]
    pub fn plane_of(&self, value: ValueId) -> Option<u32> {
        self.planes_of
            .iter()
            .find_map(|(exported, plane)| (*exported == value).then_some(*plane))
    }
}
