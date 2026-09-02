//! The observability slab where the attention capture arm writes its
//! per-key mass: `lanes x planes x ATTN_SCORE_KV_MAX x 4B`, sized to
//! `max_lanes` so the SKU's artifact is byte-identical whether or not
//! anyone captures. Every fire rewrites its whole row (zeroing past
//! `kv_len`) since a stale tail would look like live attention.

use kernels_cuda::Tensor;
use model_ir::ValueId;

use crate::device::Buffer;
use crate::error::Result;

/// The published width of every score row; the guest DSL reads the same
/// constant.
pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

/// Observation window: how many trailing query rows the capture folds into
/// the row it publishes. A decode fire (`qo_len == 1`) publishes just the
/// current token's distribution; a prefill folds its last 32 rows.
pub(crate) const OBSERVE: u32 = 32;

/// This load's score slab, plus the two numbers a launch needs to address it.
#[derive(Debug)]
pub(crate) struct Scores {
    store: Buffer,
    /// Slab planes per lane block: exported attention layers × query heads.
    planes: u32,
    /// How many query heads each exported layer contributes.
    heads: u32,
    /// How many fire lanes the slab seats — the budget's `max_lanes`.
    lanes: u32,
    /// `(value the capture arm writes, its plane base)`, one per exported
    /// column. Keyed by export position, not transformer layer, since a
    /// hybrid text's recurrent layers export nothing.
    planes_of: Vec<(ValueId, u32)>,
}

impl Scores {
    /// Reserve the slab for `exports` score columns, or `None` if empty.
    /// `heads` is the query-head count each column carries.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::error::Fault::Device) for the allocation,
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) when the deployment's
    /// lane budget and this text's plane count ask for more than the device
    /// will give.
    pub(crate) fn reserve(
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

    /// The slab as one rectangle: `[lanes * planes, KV_MAX]` F32.
    #[must_use]
    pub(crate) fn slab(&self) -> Tensor {
        Tensor::new(
            self.store.ptr(),
            self.lanes.saturating_mul(self.planes),
            KV_MAX,
            model_ir::Dtype::F32,
        )
    }

    /// Planes per lane block; the ceiling a plane count is refused against.
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


    /// Where lane `lane`'s block of planes begins, as an address.
    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        self.store.ptr()
            + u64::from(lane)
                .saturating_mul(u64::from(self.planes))
                .saturating_mul(u64::from(KV_MAX))
                .saturating_mul(4)
    }

    /// Copy one lane's `planes x KV_MAX` F32 block back to the host. Not on
    /// the fire path; only for host-side contract assertions.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::error::Fault::Ceiling) for a lane past the
    /// slab, and the device's own for the copy.
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

    /// The per-fire seat: addresses a capture launch reads, plus the
    /// `(value, plane base)` pairs the dispatch arm asks for by node.
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

/// The seat the dispatch arm reads: everything a capture launch needs.
#[derive(Clone, Debug)]
pub struct ScoreSeat {
    /// The whole slab as one rectangle; rows are (lane, plane) pairs, not
    /// fire rows, so it is not re-cut per window.
    pub slab: Tensor,
    /// Slab planes per lane block.
    pub plane_stride: u32,
    /// The observation window this load captures over.
    pub observe: u32,
    /// `(value a capture arm writes, its plane base)`, one per exported
    /// column.
    pub planes_of: Vec<(ValueId, u32)>,
}

impl ScoreSeat {
    /// Plane base of the capture arm that writes `value`, or `None` if this
    /// node is not a score export.
    #[must_use]
    pub fn plane_of(&self, value: ValueId) -> Option<u32> {
        self.planes_of
            .iter()
            .find_map(|(exported, plane)| (*exported == value).then_some(*plane))
    }
}
