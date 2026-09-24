use kernels_cuda::Tensor;
use model_ir::ValueId;

use crate::api::World;
use crate::device::Buffer;
use crate::error::Result;

pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

pub(crate) const OBSERVE: u32 = 32;

/// Where a load's score planes sit inside a lane's block. A program declares
/// `layers * heads` planes, layer-major and head-minor, over the WHOLE model's
/// query heads: the number it reads off the SKU, not off a rank. A
/// tensor-parallel rank's trace exports only its own band of heads per layer,
/// so the block is laid out at the model's width and each rank writes its band
/// at `layer * heads + rank * band` — the same plane it would have at tp 1 —
/// and the bands are gathered before an epilogue reads the block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Planes per lane: exported layers x the model's query heads.
    pub planes: u32,
    /// The model's query heads per exported layer, every rank's band together.
    pub heads: u32,
    /// This rank's query heads per exported layer.
    pub band: u32,
    /// This rank's first plane of each exported layer.
    pub planes_of: Vec<(ValueId, u32)>,
}

impl Layout {
    #[must_use]
    pub(crate) fn of(exports: &[ValueId], band: u32, world: World) -> Layout {
        let size = world.size.max(1);
        let heads = band.saturating_mul(size);
        let planes = u32::try_from(exports.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(heads);
        let planes_of = exports
            .iter()
            .enumerate()
            .map(|(at, value)| {
                let layer = u32::try_from(at).unwrap_or(0).saturating_mul(heads);
                (
                    *value,
                    layer.saturating_add(world.rank.saturating_mul(band)),
                )
            })
            .collect();
        Layout {
            planes,
            heads,
            band,
            planes_of,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Scores {
    store: Buffer,
    layout: Layout,
    lanes: u32,
    world: World,
}

impl Scores {
    pub(crate) fn reserve(
        exports: &[ValueId],
        heads: u32,
        lanes: u32,
        world: World,
    ) -> Result<Option<Scores>> {
        if exports.is_empty() || heads == 0 || lanes == 0 {
            return Ok(None);
        }
        let layout = Layout::of(exports, heads, world);
        let bytes = u64::from(lanes)
            .saturating_mul(u64::from(layout.planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(4);
        Ok(Some(Scores {
            store: Buffer::zeroed(usize::try_from(bytes).unwrap_or(usize::MAX))?,
            layout,
            lanes,
            world,
        }))
    }

    #[must_use]
    pub(crate) fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    #[must_use]
    pub(crate) fn slab(&self) -> Tensor {
        Tensor::new(
            self.store.ptr(),
            self.lanes.saturating_mul(self.layout.planes),
            KV_MAX,
            model_ir::Dtype::F32,
        )
    }

    #[must_use]
    pub(crate) fn planes(&self) -> u32 {
        self.layout.planes
    }

    #[must_use]
    pub(crate) fn heads(&self) -> u32 {
        self.layout.heads
    }

    #[must_use]
    pub(crate) fn lanes(&self) -> u32 {
        self.lanes
    }

    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        self.store.ptr()
            + u64::from(lane)
                .saturating_mul(u64::from(self.layout.planes))
                .saturating_mul(u64::from(KV_MAX))
                .saturating_mul(4)
    }

    /// Fills a lane's block with every rank's band. Each exported layer's
    /// block is one in-place all-gather: this rank's band already sits at
    /// `rank * band` inside it, which is exactly where NCCL's in-place gather
    /// expects the send buffer. Every rank of the group issues the same
    /// gathers for the same lanes, since every rank fires the same frame.
    pub(crate) fn gather(&self, ctx: &kernels_cuda::jit::Ctx, lane: u32) -> Result<()> {
        if self.world.size <= 1 {
            return Ok(());
        }
        if lane >= self.lanes {
            return Err(crate::error::Fault::Ceiling {
                what: "fire lanes the score slab seats",
                need: u64::from(lane) + 1,
                have: u64::from(self.lanes),
            });
        }
        let row = u64::from(KV_MAX) * 4;
        let base = self.lane_base(lane);
        let band = self.layout.band;
        kernels_cuda::collective::grouped("attention.score_gather", || {
            for (_, mine) in &self.layout.planes_of {
                let block = mine - self.world.rank.saturating_mul(band);
                let send = Tensor::new(
                    base + u64::from(*mine) * row,
                    1,
                    band.saturating_mul(KV_MAX),
                    model_ir::Dtype::F32,
                );
                let mut whole = Tensor::new(
                    base + u64::from(block) * row,
                    1,
                    self.layout.heads.saturating_mul(KV_MAX),
                    model_ir::Dtype::F32,
                );
                kernels_cuda::collective::all_gather(ctx, send, &mut whole)?;
            }
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) fn read_lane(&self, lane: u32) -> crate::error::Result<Vec<f32>> {
        if lane >= self.lanes {
            return Err(crate::error::Fault::Ceiling {
                what: "fire lanes the score slab seats",
                need: u64::from(lane) + 1,
                have: u64::from(self.lanes),
            });
        }
        let floats = self.layout.planes as usize * KV_MAX as usize;
        let mut raw = vec![0u8; floats * 4];
        let at = u64::from(lane)
            .saturating_mul(u64::from(self.layout.planes))
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
            plane_stride: self.layout.planes,
            observe: OBSERVE,
            planes_of: self.layout.planes_of.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    const SCORES_SEAM: &str = model_compiler::EXPORT_SEAMS[2];

    /// What `serve::load` hands `Scores::reserve` for a SKU: the `scores`
    /// exports in seam order and the query heads of the first one's rectangle.
    fn exported(sku: &str) -> (Vec<ValueId>, u32) {
        let row = models::sku(sku).unwrap_or_else(|| panic!("{sku} is in the catalog"));
        let trace = (row.trace)(model_ir::Platform::Cuda);
        let exports: Vec<ValueId> = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == SCORES_SEAM)
            .flat_map(|seam| seam.values.iter().copied())
            .collect();
        let heads = exports
            .first()
            .and_then(|value| match &trace.values[value.0 as usize].ty {
                model_ir::Ty::Tensor { shape, .. } => shape.get(1).and_then(|dim| match dim {
                    model_ir::Dim::Const(heads) => u32::try_from(*heads).ok(),
                    _ => None,
                }),
                model_ir::Ty::Struct(_) => None,
            })
            .unwrap_or(0);
        (exports, heads)
    }

    #[test]
    fn a_tp2_ranks_block_is_the_tp1_block_and_its_band_sits_inside_it() {
        // gemma-4-E4B exports a plane per global layer (7 of 42) over 8 query
        // heads; a tp2 rank's trace has 4. tova's default claim is 6 * 8 = 48
        // planes, which the tp1 load's 56 seats and a per-rank count of 28
        // refused (#653).
        let (one, heads_one) = exported("gemma4-e4b-bf16-kv-bf16");
        let (two, heads_two) = exported("gemma4-e4b-bf16-kv-bf16-tp2");
        assert_eq!((one.len(), heads_one), (7, 8));
        assert_eq!((two.len(), heads_two), (7, 4));

        let whole = Layout::of(&one, heads_one, World::default());
        assert_eq!(whole.planes, 56);
        assert!(whole.planes >= 48);
        for rank in 0..2 {
            let mine = Layout::of(&two, heads_two, World { rank, size: 2 });
            assert_eq!(mine.planes, whole.planes);
            assert_eq!(mine.heads, whole.heads);
            assert_eq!(mine.band, 4);
            for (at, (_, plane)) in mine.planes_of.iter().enumerate() {
                assert_eq!(*plane, at as u32 * 8 + rank * 4);
            }
        }
    }

    #[test]
    fn every_tp2_row_that_exports_scores_seats_its_tp1_planes() {
        for row in models::skus().filter(|sku| sku.recipe.tp == 2) {
            let Some(single) = row.name.strip_suffix("-tp2") else {
                continue;
            };
            if models::sku(single).is_none() {
                continue;
            }
            let (one, heads_one) = exported(single);
            let (two, heads_two) = exported(&row.name);
            let whole = Layout::of(&one, heads_one, World::default());
            for rank in 0..2 {
                let mine = Layout::of(&two, heads_two, World { rank, size: 2 });
                assert_eq!(mine.planes, whole.planes, "{}: rank {rank}", row.name);
                assert_eq!(mine.heads, whole.heads, "{}: rank {rank}", row.name);
            }
        }
    }
}
