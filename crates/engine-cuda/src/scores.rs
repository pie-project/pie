//! **THE OBSERVABILITY SLAB** — where the attention capture arm writes its
//! per-key mass, and the only new device bytes this axis costs
//! (`.wiki/alto/attn-score.md` §4, §6.1).
//!
//! # Why this is a slab and not an arena rectangle
//!
//! §6.1 audited the carve and found the per-key extent has no vocabulary
//! there: `RowExpr` is closed over token- and patch-shaped rows, `Budget`
//! holds no KV ceiling, and no page number reaches `model-compiler` at all.
//! It named two ways out — teach the axis vocabulary a kv term, or **stage
//! like the mask** — and this is the second, for a reason the first cannot
//! answer: the carve is a function of the MODEL TEXT, so a rectangle in it
//! would grow every artifact of every SKU whose text already declares a
//! capture window, and G4 asks that a pre-campaign SKU bake to the artifact
//! it always baked. A slab the shell owns is invisible to the compiler. The
//! artifact is byte-identical whether or not anyone ever observes anything,
//! which is the strongest form of S-3 available.
//!
//! The mask slab is the precedent in every particular
//! ([`crate::inputs`]'s header): reserved at a ceiling rather than measured,
//! addressed by a stride the fire does not move, handed to the launch whole,
//! and excluded from the window re-cut because its entries are not fire rows.
//!
//! # The rectangle, and what it costs
//!
//! ```text
//! lanes × planes × ATTN_SCORE_KV_MAX × 4 B
//!   lanes   = the budget's max_lanes: a slab row is indexed by FIRE LANE, so
//!             the capture arm needs no lane map and no second numbering
//!   planes  = exported attention layers × query heads — one distribution per
//!             (layer, head), which is what "per-head" means (§4's table)
//! ```
//!
//! For `qwen35-d0.8b` at 256 lanes — 6 attention layers of 16 heads, 96
//! planes — that is 201 MiB, and §6.1 predicted exactly this number ("an
//! arena carve reserves at the ceiling unconditionally — `max_lanes × kv_max
//! × heads × 4 B` per exported layer whether or not anyone captures"). The
//! one thing that changes is WHO pays: a load whose plan declares no
//! `attn.scores` export reserves nothing, and that is decided here rather
//! than in the bake.
//!
//! **The honest remainder, stated rather than hidden**: the lane dimension is
//! the whole budget because a slab row is a fire lane, and a fire lane is the
//! one index the capture arm already has. Cutting it to the handful of lanes
//! that actually observe wants a fire-lane → slot map staged beside the mask
//! indptr — cheap, mechanical, and not this wave's; the arithmetic above is
//! what it would remove.
//!
//! # Every fire writes every slot
//!
//! The kernel rewrites the whole `ATTN_SCORE_KV_MAX`-wide row each time,
//! zeroing `[kv_len, kv_max)` rather than leaving it. A slab outlives a fire,
//! so a tail carried over from a longer request is not absence — it is live
//! garbage from another sequence, and an eviction policy would keep a
//! position that does not exist while dropping one that does. `tail_nonzero
//! == 0` is asserted on the real model for exactly this reason.

use kernels_cuda::Tensor;
use model_ir::ValueId;

use crate::device::Buffer;
use crate::error::Result;

/// The published width of every score row — the guest DSL reads the same
/// constant, which is the whole reason it is a constant.
pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

/// The observation window: how many query rows at the END of each request the
/// capture folds into the row it publishes.
///
/// **32, AND IT IS SNAPKV'S OWN NUMBER** (Li et al., arXiv:2404.14469; the
/// C++ lineage spelled it `PIE_ATTN_SCORE_WINDOW` and defaulted to the same).
/// A decode fire carries one query row, so `min(32, qo_len)` is 1 there and
/// the published row is exactly the current token's distribution — TOVA's
/// quantity and H2O's per-step increment, unmodified. A prefill fire carries
/// the prompt, and the last 32 rows of it are the observation window every
/// SnapKV descendant defines. One statute serves all three because all three
/// asked for the same thing at different query lengths.
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
    /// `(the value the capture arm writes, its plane base)`, one entry per
    /// exported column — six on the workhorse SKU.
    ///
    /// Keyed by the EXPORT's position and not by the transformer layer,
    /// because a hybrid text's recurrent layers export nothing and a slab
    /// with holes in it would make a program's declared plane count a lie
    /// about which layers it read.
    planes_of: Vec<(ValueId, u32)>,
}

impl Scores {
    /// Reserve the slab for a load whose plan declares `exports` score
    /// columns, or `None` for one that declares none.
    ///
    /// `heads` is the query-head count each column carries — read off the
    /// carve's rectangle for the column, which is `[fire rows, heads]`.
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

    /// How many planes one lane's block holds — the row pitch between lanes,
    /// and the ceiling a guest program's declared plane count is refused
    /// against.
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
        self.store.bytes() as u64
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

    /// Copy one lane's whole block of planes back to the host — the
    /// `planes × KV_MAX` F32 rectangle a guest epilogue reads in place.
    ///
    /// **A GATE'S DOOR, AND IT IS THE ONLY ONE.** Nothing on the fire path
    /// calls this: the serving read of these numbers is the epilogue's, which
    /// happens on the device where the slab already is (§4's "only decisions
    /// cross to the host"). What a Rust gate cannot do is attach a guest
    /// program, so the contract assertions — a live row sums to one, the tail
    /// is exactly zero — come through here instead, reading the same bytes at
    /// the same addresses the intrinsic is bound at.
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

    /// The per-fire seat: the addresses a capture launch reads, and the six
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

/// The seat the dispatch arm reads — everything a capture launch needs, and
/// nothing a fire has to look up twice.
///
/// It rides [`FireBindings`](crate::run::FireBindings) rather than
/// [`FireTables`](crate::run::FireTables) for one mechanical reason: the
/// plane lookup is a list, `FireTables` is `Copy`, and six pairs cloned once
/// per fire is cheaper than either an index into the whole value space or a
/// second numbering somebody has to keep.
#[derive(Clone, Debug)]
pub struct ScoreSeat {
    /// The whole slab, handed over as one rectangle. Excluded from the window
    /// re-cut for the mask's reason: its rows are (lane, plane) pairs, not
    /// fire rows, so no seriation describes them.
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
    /// this node is not a score export — which is how a `prefill_lse` node
    /// that some other axis put in the plan runs unobserved rather than
    /// writing into a plane it does not own.
    #[must_use]
    pub fn plane_of(&self, value: ValueId) -> Option<u32> {
        self.planes_of
            .iter()
            .find_map(|(exported, plane)| (*exported == value).then_some(*plane))
    }
}
