use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::channel::{ChannelId, Ticket};
use crate::error::{Error, Result};
use crate::program::InstanceId;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FireId(pub u64);

impl std::fmt::Display for FireId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FrameId(pub u64);

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mask {
    pub runs: Vec<u32>,
    pub total: u64,
}

impl Mask {
    #[must_use]
    pub fn new(runs: Vec<u32>, total: u64) -> Mask {
        Mask { runs, total }
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.total
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    #[must_use]
    pub fn words(&self) -> usize {
        usize::try_from(self.total.div_ceil(32)).unwrap_or(usize::MAX)
    }

    pub fn admitted(&self, range: Range<u64>) -> impl Iterator<Item = Range<u64>> + '_ {
        let mut at = 0u64;
        self.runs
            .iter()
            .enumerate()
            .filter_map(move |(index, &run)| {
                let span = at..at.saturating_add(u64::from(run)).min(self.total);
                at = span.end;
                (index % 2 == 1).then_some(span)
            })
            .map(move |span| span.start.max(range.start)..span.end.min(range.end))
            .filter(|span| !span.is_empty())
    }

    pub fn expand_into(&self, dst: &mut [u32]) {
        let total = usize::try_from(self.total).unwrap_or(usize::MAX);
        let mut at = 0usize;
        for (index, &run) in self.runs.iter().enumerate() {
            let end = at.saturating_add(run as usize).min(total);
            if index % 2 == 1 {
                for bit in at..end {
                    if let Some(word) = dst.get_mut(bit / 32) {
                        *word |= 1 << (bit % 32);
                    }
                }
            }
            if end == total {
                break;
            }
            at = end;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Masking {
    Extent(Mask),
    Rows(Vec<Mask>),
}

impl Masking {
    #[must_use]
    pub fn masks(&self) -> &[Mask] {
        match self {
            Masking::Extent(mask) => std::slice::from_ref(mask),
            Masking::Rows(rows) => rows,
        }
    }

    #[must_use]
    pub fn stated_rows(&self) -> Option<usize> {
        match self {
            Masking::Extent(_) => None,
            Masking::Rows(rows) => Some(rows.len()),
        }
    }
}

/// The recurrent layers replay one run for all of a lane's rows, so every row
/// must admit the same buffered slots.
pub fn buffered_reads(
    masking: Option<&Masking>,
    folded: u32,
    first: u32,
    rows: u32,
) -> Result<Vec<Range<u32>>> {
    let whole = || {
        (folded < first)
            .then_some(folded..first)
            .into_iter()
            .collect()
    };
    let read = |mask: &Mask| -> Vec<Range<u32>> {
        mask.admitted(u64::from(folded)..u64::from(first))
            .map(|span| span.start as u32..span.end as u32)
            .collect()
    };
    let Some(masking) = masking else {
        return Ok(whole());
    };
    let masks = masking.masks();
    let Some(reads) = masks.first().map(read) else {
        return Ok(whole());
    };
    for (row, mask) in masks.iter().enumerate() {
        if read(mask) != reads {
            return Err(Error::Invalid(format!(
                "row {row} admits other buffered slots than the lane's first row; its \
                 recurrent layers replay one run of them for all its rows"
            )));
        }
        let before = match masking {
            Masking::Rows(_) => row as u64,
            Masking::Extent(_) => u64::from(rows.saturating_sub(1)),
        };
        let own = u64::from(first)..u64::from(first) + before;
        let kept: u64 = mask
            .admitted(own.clone())
            .map(|span| span.end - span.start)
            .sum();
        if kept != own.end - own.start {
            return Err(Error::Invalid(format!(
                "row {row} hides a row of its own lane before it; its recurrent layers \
                 run the lane's rows in order"
            )));
        }
    }
    Ok(reads)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDelta {
    pub held: u32,
    pub pages: Vec<u32>,
    #[serde(default)]
    pub translation: Vec<u32>,
    /// The ids of the lane's pages in the windowed kv spaces, aligned with
    /// `translation` when it is stated and with `pages` otherwise; 0 is the
    /// null page a page wholly behind the window reads.
    #[serde(default)]
    pub window: Vec<u32>,
    /// Windowed pages copied `(src, dst)` before the fire runs, beside the
    /// full pages the fire's pre-launch copy moves.
    #[serde(default)]
    pub window_copies: Vec<(u32, u32)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Readout {
    #[default]
    Last,
    Rows(Vec<u32>),
    None,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lane {
    pub slot: u32,
    pub word: u64,
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub kv: KvDelta,
    pub mask: Option<Masking>,
    pub adapter: Option<u32>,
    #[serde(default)]
    pub rs: RsVerb,
    #[serde(default)]
    pub rs_reset: RsReset,
    /// The recurrent-state row the runtime's rs store assigned this lane, when
    /// it binds one; otherwise the state rides the seat.
    #[serde(default)]
    pub rs_slot: Option<u32>,
    /// Rows, ascending inside the lane, whose recurrent state the fire lands
    /// in slots of their own as it runs.
    #[serde(default)]
    pub rs_snapshots: Vec<RsSnapshot>,
    #[serde(default)]
    pub channels: Vec<Ticket>,
    pub readout: Readout,
    /// Unique to its pass in a fire, so two passes' lanes never attend each other.
    #[serde(default)]
    pub group: Option<u32>,
    /// How many more fires the frame this lane's fire belongs to sends
    /// after it (`None`: not part of a frame). An engine that runs a device
    /// token loop past its rows stops where the frame does.
    #[serde(default)]
    pub fires_after: Option<u32>,
    #[serde(default)]
    pub inputs: Vec<InputFeed>,
    /// The lane takes no seat; the state it carries, if any, is the row `rs_slot` names.
    #[serde(default)]
    pub kv_less: bool,
    /// An engine that cannot report says nothing, and the step is plainly
    /// committed.
    #[serde(default)]
    pub reports: u64,
    /// Pages of this lane's to read out as the fire computes them; only to
    /// an engine that taps ([`crate::Engine::tap_groups`]).
    #[serde(default)]
    pub tap: Option<Tap>,
}

/// Pool pages `pages[i]` read into host kv slots `slots[i]` as the fire
/// writes them, a group of their planes at a time (a layer's), each group
/// told to the engine's tap sink under `tag` as it lands.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tap {
    pub tag: u64,
    pub pages: Vec<u32>,
    pub slots: Vec<u32>,
}

/// The channel a lane feeds guest input `id` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputFeed {
    pub id: u32,
    pub channel: ChannelId,
}

impl Lane {
    #[must_use]
    pub fn decode(slot: u32, word: u64, token: u32, held: u32) -> Lane {
        Lane {
            slot,
            word,
            tokens: vec![token],
            kv: KvDelta {
                held,
                ..KvDelta::default()
            },
            ..Lane::default()
        }
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.tokens.len()).unwrap_or(u32::MAX)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if !self.positions.is_empty() && self.positions.len() != self.tokens.len() {
            return Err(Error::Invalid(format!(
                "lane in slot {} has {} positions for {} tokens",
                self.slot,
                self.positions.len(),
                self.tokens.len()
            )));
        }
        if let Readout::Rows(rows) = &self.readout
            && let Some(&row) = rows.iter().find(|&&r| r >= self.rows())
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} reads row {row} of the {} it has",
                self.slot,
                self.rows()
            )));
        }
        // A kv-less lane's mask is over its attention group's rows.
        if let Some(masking) = &self.mask
            && !self.kv_less
            && let Some(stated) = masking.stated_rows()
            && stated != self.tokens.len()
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} states {stated} per-row masks for the {} rows it carries",
                self.slot,
                self.rows()
            )));
        }
        if !self.channels.is_empty() && !serves.device_channel_commit {
            return Err(Error::unsupported("engine", F3_CHANNEL_TICKETS));
        }
        if !matches!(self.rs, RsVerb::Fold) && !serves.rs_verbs {
            return Err(Error::unsupported("engine", RS_VERBS_WITHOUT_DEVICE_HALF));
        }
        if !self.rs_snapshots.is_empty() {
            if self.rs_snapshots.len() > serves.rs_snapshots as usize {
                return Err(Error::Invalid(format!(
                    "lane in slot {} snapshots its recurrent state at {} rows, past the {} \
                     this engine lands in one fire",
                    self.slot,
                    self.rs_snapshots.len(),
                    serves.rs_snapshots
                )));
            }
            let ascending = self
                .rs_snapshots
                .iter()
                .try_fold(0, |at, snap| (at < snap.row).then_some(snap.row))
                .is_some_and(|last| last < self.rows());
            if !ascending || !matches!(self.rs, RsVerb::Fold) || self.rs_slot.is_none() {
                return Err(Error::Invalid(format!(
                    "lane in slot {} snapshots its recurrent state at rows {:?} of its {}: a \
                     snapshot is an interior row, ascending, of a lane that folds every row \
                     into a state slot of its own",
                    self.slot,
                    self.rs_snapshots
                        .iter()
                        .map(|snap| snap.row)
                        .collect::<Vec<_>>(),
                    self.rows()
                )));
            }
        }
        Ok(())
    }
}

const F3_CHANNEL_TICKETS: &str =
    "Lane::channels against an engine without the pull-validate and commit-bump kernels";

const RS_VERBS_WITHOUT_DEVICE_HALF: &str =
    "Lane::rs beyond RsVerb::Fold: this engine has no device half for it";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serves {
    pub device_channel_commit: bool,
    pub rs_verbs: bool,
    pub adapter_attachments: bool,
    pub rs_snapshots: u32,
}

impl Serves {
    pub const NONE: Serves = Serves {
        device_channel_commit: false,
        rs_verbs: false,
        adapter_attachments: false,
        rs_snapshots: 0,
    };

    #[must_use]
    pub fn of(caps: &crate::Capabilities) -> Serves {
        Serves {
            device_channel_commit: caps.device_channel_commit,
            rs_verbs: caps.rs_verbs,
            adapter_attachments: caps.adapter_attachments,
            rs_snapshots: caps.rs_snapshots,
        }
    }
}

/// The state after the lane's first `row` rows, landed in state slot `slot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsSnapshot {
    pub row: u32,
    pub slot: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsReset {
    #[default]
    Inferred,
    Fresh,
    Held,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RsVerb {
    #[default]
    Fold,
    /// Replays the buffered slots `folded..first` its mask admits, then freezes
    /// `fold` slots past `folded` into the state.
    Buffer {
        /// The buffer's pages: token `t` of page `p` holds slot
        /// `base + p * page + t`.
        pages: Vec<u32>,
        base: u32,
        folded: u32,
        fold: FoldLen,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldLen {
    Host(u32),
    Device,
}

/// A pass's program on a fire: the prologue before input staging, the epilogue after the forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    /// The first of the pass's lanes, `lanes` of them in a row.
    pub lane: u32,
    pub lanes: u32,
    pub instance: InstanceId,
    /// Every fire but the last of a cut pass: it carries the adapter and runs no stage.
    pub adapter_only: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepMedia {
    pub lane: u32,
    pub rows: Vec<u32>,
    /// Each span's patches, `3 · patch²` RGB bytes a row.
    pub patches: Vec<u8>,
    pub routes: Vec<i32>,
    pub positions: Vec<i32>,
    /// Each span's `(t, rows, cols)` patch grid.
    pub grids: Vec<i32>,
    pub token_positions: Vec<i32>,
}

impl StepMedia {
    #[must_use]
    pub fn payload_rows(&self) -> u32 {
        self.rows.iter().copied().fold(0u32, u32::saturating_add)
    }

    pub fn validate(&self, lane_rows: u32) -> Result<()> {
        let rows = self.payload_rows();
        if self.rows.is_empty() {
            return Err(Error::Invalid(format!(
                "lane {} carries a media row naming no spans; a lane with no span \
                 constructs no media row at all",
                self.lane
            )));
        }
        if rows == 0 {
            return Err(Error::Invalid(format!(
                "lane {}'s spans occupy no payload rows, and a span the tower reads \
                 nothing of has nothing to scatter",
                self.lane
            )));
        }
        for (what, have, owed) in [
            ("routes", self.routes.len(), rows as usize),
            ("grid positions", self.positions.len(), 3 * rows as usize),
        ] {
            if have != owed {
                return Err(Error::Invalid(format!(
                    "lane {}'s media carries {have} {what} for {rows} payload rows, \
                     and {owed} are owed",
                    self.lane
                )));
            }
        }
        if self.grids.len() != 3 * self.rows.len() {
            return Err(Error::Invalid(format!(
                "lane {}'s media carries {} grid entries for {} spans; each span states \
                 its `(t, rows, cols)` patch grid",
                self.lane,
                self.grids.len(),
                self.rows.len()
            )));
        }
        if !self.token_positions.is_empty() && self.token_positions.len() != 3 * lane_rows as usize
        {
            return Err(Error::Invalid(format!(
                "lane {}'s trunk rotation stream carries {} entries for {lane_rows} \
                 token rows; it is empty (scalar `(p, p, p)`) or three per row",
                self.lane,
                self.token_positions.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepVoxels {
    pub lane: u32,
    pub clips: Vec<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub payload: Vec<f32>,
}

impl StepVoxels {
    #[must_use]
    pub fn voxels(&self) -> u64 {
        self.clips
            .iter()
            .map(|[t, h, w]| u64::from(*t) * u64::from(*h) * u64::from(*w))
            .sum()
    }

    pub fn validate(&self) -> Result<()> {
        if self.clips.is_empty() {
            return Err(Error::Invalid(format!(
                "lane {} carries a voxel row naming no clips; a lane with no clip \
                 constructs no voxel row at all",
                self.lane
            )));
        }
        if self.clips.iter().any(|b| b.contains(&0)) {
            return Err(Error::Invalid(format!(
                "lane {} submits a clip with a zero side",
                self.lane
            )));
        }
        if !self.payload.is_empty() && !(self.payload.len() as u64).is_multiple_of(self.voxels()) {
            return Err(Error::Invalid(format!(
                "lane {} submits {} values for {} voxels, which is not a whole row per voxel",
                self.lane,
                self.payload.len(),
                self.voxels()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub lanes: Vec<Lane>,
    pub attachments: Vec<Attachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<StepMedia>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub voxels: Vec<StepVoxels>,
}

impl Step {
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.lanes.iter().map(Lane::rows).sum()
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.lanes.is_empty() {
            return Err(Error::Invalid("a fire carries no lanes".into()));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            lane.validate_for(serves)?;
            // A kv-less lane takes no seat: its slot names nothing.
            if !lane.kv_less
                && self.lanes[..index]
                    .iter()
                    .any(|l| !l.kv_less && l.slot == lane.slot)
            {
                return Err(Error::Invalid(format!(
                    "slot {} appears twice in one fire, at lane {index}",
                    lane.slot
                )));
            }
        }
        let lanes = u32::try_from(self.lanes.len()).unwrap_or(u32::MAX);
        for (index, attachment) in self.attachments.iter().enumerate() {
            if attachment.adapter_only && !serves.adapter_attachments {
                return Err(Error::Invalid(format!(
                    "attachment {index} is for an adapter alone, which this engine does not serve"
                )));
            }
            if attachment.lane >= lanes {
                return Err(Error::Invalid(format!(
                    "attachment names lane {} of the {lanes} this fire has",
                    attachment.lane
                )));
            }
            if self.attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attachment.instance)
            {
                return Err(Error::Invalid(format!(
                    "instance {} is attached twice to one fire, at attachment {index}; a \
                     program runs once a fire",
                    attachment.instance
                )));
            }
        }
        for (index, media) in self.media.iter().enumerate() {
            let Some(lane) = self.lanes.get(media.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "media row {index} names lane {} of the {lanes} this fire has",
                    media.lane
                )));
            };
            if self.media[..index]
                .iter()
                .any(|earlier| earlier.lane == media.lane)
            {
                return Err(Error::Invalid(format!(
                    "lane {} carries two media rows, at media row {index}; a lane's \
                     spans are one concatenation with one payload order",
                    media.lane
                )));
            }
            media.validate(lane.rows())?;
        }
        for (index, voxels) in self.voxels.iter().enumerate() {
            if self.lanes.get(voxels.lane as usize).is_none() {
                return Err(Error::Invalid(format!(
                    "voxel row {index} names lane {} of the {lanes} this fire has",
                    voxels.lane
                )));
            }
            if self.voxels[..index]
                .iter()
                .any(|earlier| earlier.lane == voxels.lane)
            {
                return Err(Error::Invalid(format!(
                    "lane {} carries two voxel rows, at voxel row {index}; a lane's \
                     clips are one concatenation with one payload order",
                    voxels.lane
                )));
            }
            voxels.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameSubmission {
    pub steps: Vec<Step>,
}

impl FrameSubmission {
    #[must_use]
    pub fn of(step: Step) -> FrameSubmission {
        FrameSubmission { steps: vec![step] }
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.steps.iter().map(Step::rows).sum()
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_for(Serves::NONE)
    }

    pub fn validate_for(&self, serves: Serves) -> Result<()> {
        if self.steps.is_empty() {
            return Err(Error::Invalid("a frame carries no steps".into()));
        }
        for step in &self.steps {
            step.validate_for(serves)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameTicket {
    pub id: FrameId,
    pub steps: Vec<FireTicket>,
}

/// The rows a lane read back of `output`, the last row or voxel output its
/// class writes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaneReadout {
    #[serde(default)]
    pub output: String,
    pub rows: u32,
    pub width: u32,
    pub values: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clips: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FireTicket {
    pub id: FireId,
    pub readouts: Vec<LaneReadout>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaEncode {
    pub image_grids: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub output_rows: Vec<u8>,
    pub output_row_indptr: Vec<u32>,
}

impl MediaEncode {
    pub fn validate(&self) -> Result<()> {
        const F32: usize = size_of::<f32>();
        const U16: usize = size_of::<u16>();
        let bad = |why: String| Err(Error::Invalid(why));

        let images = self.image_anchor_rows.len();
        let clips = self.audio_anchor_rows.len();
        if images + clips == 0 {
            return bad("an encode carries no image and no audio anchor".into());
        }
        if self.output_row_indptr.len() != images + clips + 1 {
            return bad(format!(
                "output_row_indptr has {} entries for {images} images and {clips} clips",
                self.output_row_indptr.len()
            ));
        }
        if self.output_rows.is_empty() || !self.output_rows.len().is_multiple_of(U16) {
            return bad(format!(
                "output_rows is {} bytes, which is empty or not a whole number of u16",
                self.output_rows.len()
            ));
        }

        if images == 0 {
            if !self.image_grids.is_empty()
                || !self.image_pixels.is_empty()
                || !self.image_pixel_indptr.is_empty()
                || !self.image_patch_positions.is_empty()
            {
                return bad("an image payload arrived with no image anchor to attach it to".into());
            }
        } else {
            if self.image_grids.len() != images.saturating_mul(3) {
                return bad(format!(
                    "image_grids has {} entries for {images} images",
                    self.image_grids.len()
                ));
            }
            if self.image_pixel_indptr.len() != images + 1 {
                return bad(format!(
                    "image_pixel_indptr has {} entries for {images} images",
                    self.image_pixel_indptr.len()
                ));
            }
            if self.image_pixels.is_empty() || !self.image_pixels.len().is_multiple_of(F32) {
                return bad(format!(
                    "image_pixels is {} bytes, which is empty or not a whole number of f32",
                    self.image_pixels.len()
                ));
            }
            if self.image_patch_positions.is_empty()
                || !self.image_patch_positions.len().is_multiple_of(2)
            {
                return bad(format!(
                    "image_patch_positions has {} entries, which is empty or not a whole number \
                     of pairs",
                    self.image_patch_positions.len()
                ));
            }
            partition(
                &self.image_pixel_indptr,
                "image_pixel_indptr",
                self.image_pixels.len(),
                F32,
                false,
            )?;
        }

        if clips == 0 {
            if !self.audio_features.is_empty() || !self.audio_feature_indptr.is_empty() {
                return bad("an audio payload arrived with no audio anchor to attach it to".into());
            }
        } else {
            if self.audio_feature_indptr.len() != clips + 1 {
                return bad(format!(
                    "audio_feature_indptr has {} entries for {clips} clips",
                    self.audio_feature_indptr.len()
                ));
            }
            if self.audio_features.is_empty() || !self.audio_features.len().is_multiple_of(F32) {
                return bad(format!(
                    "audio_features is {} bytes, which is empty or not a whole number of f32",
                    self.audio_features.len()
                ));
            }
            partition(
                &self.audio_feature_indptr,
                "audio_feature_indptr",
                self.audio_features.len(),
                F32,
                true,
            )?;
        }
        Ok(())
    }
}

fn partition(indptr: &[u32], name: &str, bytes: usize, align: usize, strict: bool) -> Result<()> {
    if indptr.first().copied() != Some(0) {
        return Err(Error::Invalid(format!(
            "{name} starts at {:?}, not 0",
            indptr.first()
        )));
    }
    if indptr.last().copied() != u32::try_from(bytes).ok() {
        return Err(Error::Invalid(format!(
            "{name} ends at {:?}, not the {bytes} bytes it partitions",
            indptr.last()
        )));
    }
    for w in indptr.windows(2) {
        let ordered = if strict { w[0] < w[1] } else { w[0] <= w[1] };
        if !ordered {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is empty or inverted",
                w[0], w[1]
            )));
        }
        if !(w[0] as usize).is_multiple_of(align) || !(w[1] as usize).is_multiple_of(align) {
            return Err(Error::Invalid(format!(
                "{name} segment {}..{} is not {align}-byte aligned",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(kept: &[bool]) -> Mask {
        let mut runs = vec![0u32];
        let mut keeping = false;
        for &bit in kept {
            if bit != keeping {
                runs.push(0);
                keeping = bit;
            }
            *runs.last_mut().unwrap() += 1;
        }
        Mask::new(runs, kept.len() as u64)
    }

    #[test]
    fn a_lane_replays_the_buffered_slots_its_mask_admits() {
        // Slots 0..2 are folded; the lane's one row is at slot 6.
        let ancestry = mask(&[true, true, false, true, false, true, true]);
        assert_eq!(
            ancestry.admitted(2..6).collect::<Vec<_>>(),
            vec![3..4, 5..6],
            "the kept spans in the range, in order"
        );
        let reads = buffered_reads(Some(&Masking::Rows(vec![ancestry.clone()])), 2, 6, 1);
        assert_eq!(reads.unwrap(), vec![3..4, 5..6]);
        assert_eq!(buffered_reads(None, 2, 6, 1).unwrap(), vec![2..6]);

        // Two rows from slot 5 that admit different buffered slots, and a
        // second row that hides the first: a recurrence runs neither.
        let other = mask(&[true, true, true, false, false, true, true]);
        let rows = Masking::Rows(vec![ancestry.clone(), other]);
        assert!(buffered_reads(Some(&rows), 2, 5, 2).is_err());
        let blind = mask(&[true, true, false, true, false, false, true]);
        let rows = Masking::Rows(vec![ancestry, blind]);
        assert!(buffered_reads(Some(&rows), 2, 5, 2).is_err());
    }

    #[test]
    fn kv_less_lanes_take_no_seat() {
        // A CFG step: four unseated float lanes beside a decode in seat 0.
        let float = |rs_slot| Lane {
            tokens: vec![0; 4],
            kv_less: true,
            rs_slot,
            ..Lane::default()
        };
        let mut step = Step {
            lanes: vec![float(None), float(None), float(None), float(Some(2))],
            ..Step::default()
        };
        step.lanes.push(Lane::decode(0, 0, 7, 3));
        assert!(step.validate().is_ok(), "{:?}", step.validate());
        step.lanes.push(Lane::decode(0, 0, 8, 4));
        assert!(step.validate().is_err(), "two seated lanes share seat 0");
    }
}
