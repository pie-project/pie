use kernels_cuda::Tensor;
use model_ir::ValueId;

use crate::device::Buffer;
use crate::error::Result;

pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

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
    pub(crate) fn reserve(exports: &[ValueId], heads: u32, lanes: u32) -> Result<Option<Scores>> {
        if exports.is_empty() || heads == 0 || lanes == 0 {
            return Ok(None);
        }
        let planes = u32::try_from(exports.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(heads);
        let bytes = u64::from(lanes)
            .saturating_mul(u64::from(planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(4);
        let planes_of = exports
            .iter()
            .enumerate()
            .map(|(at, value)| (*value, u32::try_from(at).unwrap_or(0).saturating_mul(heads)))
            .collect();
        Ok(Some(Scores {
            store: Buffer::zeroed(usize::try_from(bytes).unwrap_or(usize::MAX))?,
            planes,
            heads,
            lanes,
            planes_of,
        }))
    }

    #[must_use]
    pub(crate) fn slab(&self) -> Tensor {
        Tensor::new(
            self.store.ptr(),
            self.lanes.saturating_mul(self.planes),
            KV_MAX,
            model_ir::Dtype::F32,
        )
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
    pub(crate) fn lanes(&self) -> u32 {
        self.lanes
    }

    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        self.store.ptr()
            + u64::from(lane)
                .saturating_mul(u64::from(self.planes))
                .saturating_mul(u64::from(KV_MAX))
                .saturating_mul(4)
    }

    pub(crate) fn read_lane(&self, lane: u32) -> crate::error::Result<Vec<f32>> {
        if lane >= self.lanes {
            return Err(crate::error::Fault::Ceiling {
                what: "fire lanes the score slab seats",
                need: u64::from(lane) + 1,
                have: u64::from(self.lanes),
            });
        }
        let floats = self.planes as usize * KV_MAX as usize;
        let mut raw = vec![0u8; floats * 4];
        let at = u64::from(lane)
            .saturating_mul(u64::from(self.planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(4);
        self.store.read(at, &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }

    #[must_use]
    pub(crate) fn seat(&self) -> ScoreSeat {
        ScoreSeat {
            slab: self.slab(),
            plane_stride: self.planes,
            observe: OBSERVE,
            planes_of: self.planes_of.clone(),
        }
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
