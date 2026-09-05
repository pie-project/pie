use kernels_wgpu::Tensor;
use model_ir::{Dtype, ValueId};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

const SLOT: u64 = 4;

pub(crate) const OBSERVE: u32 = 32;

#[derive(Debug)]
pub(crate) struct Scores {
    store: Buffer,

    planes: u32,

    heads: u32,

    lanes: u32,

    planes_of: Vec<(ValueId, u32)>,
}

impl Scores {
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

    #[must_use]
    pub(crate) fn planes(&self) -> u32 {
        self.planes
    }

    #[must_use]
    pub(crate) fn heads(&self) -> u32 {
        self.heads
    }

    #[must_use]
    pub(crate) fn bytes(&self) -> u64 {
        self.store.bytes()
    }

    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        u64::from(lane)
            .saturating_mul(u64::from(self.planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT)
    }

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

    pub(crate) fn seat(&self, handles: &Handles) -> Result<ScoreSeat> {
        let rows = self.lanes.saturating_mul(self.planes);
        let bytes = u64::from(rows)
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT);
        Ok(ScoreSeat {
            slab: Tensor::new(
                handles.bind(&self.store, 0, bytes)?,
                rows,
                KV_MAX,
                Dtype::F32,
            ),
            plane_stride: self.planes,
            observe: OBSERVE,
            planes_of: self.planes_of.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ScoreSeat {
    pub slab: Tensor,

    pub plane_stride: u32,

    pub observe: u32,

    pub planes_of: Vec<(ValueId, u32)>,
}

impl ScoreSeat {
    #[must_use]
    pub fn plane_of(&self, value: ValueId) -> Option<u32> {
        self.planes_of
            .iter()
            .find_map(|(exported, plane)| (*exported == value).then_some(*plane))
    }
}
