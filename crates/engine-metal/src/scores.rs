//! **THE OBSERVABILITY SLAB** — where the attention capture arm writes its
//! per-key mass, and the only new device bytes this axis costs
//! (`.wiki/alto/attn-score.md` §4, §6.1). The mirror of
//! `engine_cuda::scores`, with the two divergences this plane always has.
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
//! capture window, and a pre-campaign SKU must bake the artifact it always
//! baked. A slab the shell owns is invisible to the compiler. The artifact is
//! byte-identical whether or not anyone ever observes anything, which is the
//! strongest form of zero-cost-when-off available.
//!
//! [`crate::inputs`] is the precedent in every particular: reserved at a
//! ceiling rather than measured, addressed by a stride the fire does not move,
//! handed to the launch whole, and excluded from the window re-cut because its
//! entries are not fire rows.
//!
//! # ONE COPY, AND NOT ONE PER ARM
//!
//! This is where the mask's precedent stops and [`crate::scratch`]'s begins,
//! and the line between them is who writes. `inputs` and `readout` are
//! duplicated per in-flight arm because the HOST writes them: the store is
//! `StorageModeShared`, so a `memcpy` into one lands in the very bytes a
//! committed command buffer is reading. **Nothing here is ever written by the
//! host.** Every byte is written by a shader — the capture arm — and read by a
//! shader later in the same command buffer: the epilogue's `attn_score`
//! intrinsic, bound at the lane's block of this rectangle. That puts the slab
//! in the arena's and the pools' class, resting on the property `serve`'s
//! header states and gates by measurement: command buffers committed to one
//! `MTLCommandQueue` execute in commit order and do not overlap. Two arms
//! cannot be inside this plane at once for the same reason two arms cannot be
//! inside the arena at once, and if that property were false the arena would
//! be wrong first.
//!
//! The one host reader is [`Scores::read_lane`], and it is a GATE's door
//! rather than a fire's: it runs between fires, after the flight that wrote
//! the bytes has been harvested.
//!
//! # A view here is a HANDLE, not an address
//!
//! The CUDA twin's `Scores::slab` is `Tensor::new(self.store.ptr(), ...)` —
//! pure arithmetic on a device address. Metal has no address to hand out: an
//! encoder binds a BUFFER and an OFFSET, so a [`kernels_metal::Tensor`]
//! carries a `u32` row of [`Handles`] and the resolution is a table lookup at
//! encode time. So [`Scores::seat`] takes the handle table and MINTS where its
//! twin only added — and it is fallible for the reason every mint here is,
//! because minting is where the rectangle is bounds-checked against the
//! reservation it claims to live in.
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
//! For `qwen35-d0.8b` at 256 lanes — 6 attention layers of 16 heads, 96 planes
//! — that is 201 MiB, and §6.1 predicted exactly this number. The one thing
//! that changes is WHO pays: a load whose plan declares no `attn.scores`
//! export reserves nothing, and that is decided here rather than in the bake.
//!
//! **The honest remainder, stated rather than hidden**: the lane dimension is
//! the whole budget because a slab row is a fire lane, and a fire lane is the
//! one index the capture arm already has. Cutting it to the handful of lanes
//! that actually observe wants a fire-lane → slot map staged beside the mask —
//! cheap, mechanical, and not this wave's; the arithmetic above is what it
//! would remove.
//!
//! # Every fire writes every slot
//!
//! The kernel rewrites the whole `ATTN_SCORE_KV_MAX`-wide row each time,
//! zeroing the tail rather than leaving it. A slab outlives a fire, so a tail
//! carried over from a longer request is not absence — it is live garbage from
//! another sequence, and an eviction policy would keep a position that does
//! not exist while dropping one that does.

use kernels_metal::Tensor;
use model_ir::{Dtype, ValueId};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

/// The published width of every score row — the guest DSL reads the same
/// constant, which is the whole reason it is one.
pub(crate) const KV_MAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

/// One F32 slot, in bytes. Spelled once because it multiplies into every
/// address in this file.
const SLOT: u64 = 4;

/// The observation window: how many query rows at the END of each request the
/// capture folds into the row it publishes.
///
/// **32, AND IT IS SNAPKV'S OWN NUMBER** (Li et al., arXiv:2404.14469; the C++
/// lineage spelled it `PIE_ATTN_SCORE_WINDOW` and defaulted to the same). A
/// decode fire carries one query row, so `min(32, qo_len)` is 1 there and the
/// published row is exactly the current token's distribution — TOVA's quantity
/// and H2O's per-step increment, unmodified. A prefill fire carries the
/// prompt, and the last 32 rows of it are the observation window every SnapKV
/// descendant defines. One statute serves all three because all three asked
/// for the same thing at different query lengths.
///
/// The same number the CUDA shell states, and deliberately: a guest that folds
/// a metal capture and a cuda one is folding the same quantity or it is
/// folding two.
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
    /// `(the value the capture arm writes, its plane base)`, one entry per
    /// exported column — six on the workhorse SKU.
    ///
    /// Keyed by the EXPORT's position and not by the transformer layer,
    /// because a hybrid text's recurrent layers export nothing and a slab with
    /// holes in it would make a program's declared plane count a lie about
    /// which layers it read.
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
    /// [`Fault::Device`] for the allocation, and [`Fault::Ceiling`] when the
    /// deployment's lane budget and this text's plane count ask for more than
    /// the device will give.
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

    /// The allocation itself — what [`crate::program::launch`] binds an
    /// epilogue's `attn_score` rectangle against, at a lane's own offset.
    pub(crate) fn store(&self) -> &Buffer {
        &self.store
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
        self.store.bytes()
    }

    /// Where lane `lane`'s block of planes begins, in BYTES into the store.
    ///
    /// The CUDA twin answers the same question with an address; here it is an
    /// offset, because that is the half of `setBuffer:offset:atIndex:` a
    /// caller supplies.
    #[must_use]
    pub(crate) fn lane_base(&self, lane: u32) -> u64 {
        u64::from(lane)
            .saturating_mul(u64::from(self.planes))
            .saturating_mul(u64::from(KV_MAX))
            .saturating_mul(SLOT)
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
    /// the same offset the intrinsic is bound at.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a lane past the slab, and the device's own for
    /// the copy.
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

    /// The per-fire seat: the rectangle a capture launch writes, and the
    /// `(value, plane base)` pairs the dispatch arm asks for by node.
    ///
    /// The mint is here rather than at [`Scores::reserve`] for the reason
    /// every mint on this plane is per fire: [`Handles::rewind`] drops
    /// everything a fire minted, so a row taken at load and rewound at the
    /// first fire's end would resolve to whatever the table held next.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the rectangle leaves its reservation or the
    /// handle table is full — the bounds check the shader cannot make.
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

/// The seat the dispatch arm reads — everything a capture launch needs, and
/// nothing a fire has to look up twice.
///
/// It rides [`FireBindings`](crate::run::FireBindings) rather than
/// [`FireTables`](crate::run::FireTables) for the reason `adapter_routes`
/// does: `FireTables` is the seats no op names AND is `Copy`, and this one
/// carries a list.
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
