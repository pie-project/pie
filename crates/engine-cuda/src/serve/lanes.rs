//! What one fire submits: lanes, their seats, guest attachments and images.

use engine::fire::{Boundary, Masking, RsReset, RsVerb, SelfCondInput};

/// One request inside a fire.
#[derive(Debug, Clone, Copy)]
pub struct Lane<'a> {
    /// Which pool slot this request's sequence lives in.
    pub slot: u32,
    /// Its fact bits, as the model's own `Classify::of` computed them.
    pub word: u64,
    /// The token ids this fire feeds it.
    pub tokens: &'a [u32],
}

/// One request inside a fire, with the page table its caller owns.
#[derive(Debug, Clone)]
pub struct Seated<'a> {
    /// The request.
    pub lane: Lane<'a>,
    /// This lane's kv pages, in sequence order. Empty means the shell's.
    pub pages: &'a [u32],
    /// How many kv tokens the slot already holds. `None` asks the shell.
    pub held: Option<u32>,
    /// The working set's flat table for a device-geometry lane; empty otherwise.
    pub translation: &'a [u32],
    /// An explicit attention mask over the lane's readable extent.
    pub mask: Option<&'a Masking>,
    /// Which adapter bank this lane's rows route to, or `None` for the base model.
    pub adapter: Option<u32>,
    /// Run the model's draft head over this lane's rows.
    pub drafts: bool,
    /// Keep this lane's per-query attention mass.
    pub captures_scores: bool,
    /// Lift the causal bound from this lane's mask bits: every row attends
    /// every key of the extent.
    pub bidirectional: bool,
    /// This lane's self-conditioning taps, or `None` for zero weights.
    pub self_cond: Option<&'a SelfCondInput>,
    /// What this lane's pass does to its recurrent state.
    pub rs: RsVerb,
    /// Whether this lane's recurrent slot arrives fresh.
    pub rs_reset: RsReset,
    /// Which of this lane's rows the device readout is pointed at, by index
    /// within the lane; `None` for the last row.
    pub readout: Option<&'a [u32]>,
}

impl<'a> Seated<'a> {
    /// A lane whose page table, token count and masking are the shell's.
    #[must_use]
    pub fn of(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            lane,
            pages: &[],
            held: None,
            translation: &[],
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            bidirectional: false,
            self_cond: None,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            readout: None,
        }
    }

    /// The same lane, reading only `mask`'s positions of its slot.
    #[must_use]
    pub fn masked(lane: Lane<'a>, mask: &'a Masking) -> Seated<'a> {
        Seated {
            mask: Some(mask),
            ..Seated::of(lane)
        }
    }

    /// The same lane, corrected by adapter `id`.
    #[must_use]
    pub fn adapted(lane: Lane<'a>, id: u32) -> Seated<'a> {
        Seated {
            adapter: Some(id),
            ..Seated::of(lane)
        }
    }

    /// The same lane, with the model's draft head run over its rows.
    #[must_use]
    pub fn drafting(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            drafts: true,
            ..Seated::of(lane)
        }
    }

    /// The same lane, with its attention's per-query mass kept.
    #[must_use]
    pub fn capturing(lane: Lane<'a>) -> Seated<'a> {
        Seated {
            captures_scores: true,
            ..Seated::of(lane)
        }
    }
}

/// One guest program attached to a fire's boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attached {
    /// Which lane of the submission this instance runs for.
    pub lane: u32,
    /// Which bound instance, as `Shell::bind_program` minted it.
    pub instance: u64,
    /// Which side of the graph.
    pub at: Boundary,
}

/// How many numbers one m-rope position is: the triple `(t, h, w)`.
pub(crate) const MROPE_COORDS: usize = 3;

/// "This tower row has no destination" — a compacting fold's tail route.
pub(crate) const PATCH_ROUTE_DROP: i32 = -1;

/// One lane's images, as the submission carries them (pre-unfolded patch rows).
#[derive(Debug, Clone, Copy)]
pub struct Media<'a> {
    /// Which lane of the submission these images belong to.
    pub lane: u32,
    /// How many patch rows each image contributes, in submission order.
    pub rows: &'a [u32],
    /// The patch rows, concatenated over this lane's images, in the plan's element.
    pub patches: &'a [u8],
    /// Where each fold-output row lands in this lane's token rows; `-1` tail past the live prefix.
    pub routes: &'a [i32],
    /// Three `i32` per patch row — each patch's `(t, h, w)` in its image's grid.
    pub positions: &'a [i32],
    /// Which rows of the learned position table each patch gathers, `taps` per row.
    pub embed_rows: &'a [i32],
    /// How much of each tap, `taps` `f32` per row; empty on the native grid.
    pub embed_weights: &'a [f32],
    /// Three `i32` per token row of the lane; empty means `(p, p, p)`.
    pub token_positions: &'a [i32],
}
