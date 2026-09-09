use serde::{Deserialize, Serialize};

use crate::channel::Ticket;
use crate::error::{Error, Result};
use crate::program::InstanceId;

pub type FireId = u64;

pub type FrameId = u64;

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
    pub fn of_row(&self, row: usize) -> Option<&Mask> {
        match self {
            Masking::Extent(mask) => Some(mask),
            Masking::Rows(rows) => rows.get(row),
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDelta {
    pub held: u32,
    pub pages: Vec<u32>,
    #[serde(default)]
    pub translation: Vec<u32>,
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
    pub drafts: bool,
    pub captures_scores: bool,
    #[serde(default)]
    pub block_draft: bool,
    #[serde(default)]
    pub bidirectional: bool,
    #[serde(default)]
    pub self_cond: Option<SelfCondInput>,
    #[serde(default)]
    pub rs: RsVerb,
    #[serde(default)]
    pub rs_reset: RsReset,
    #[serde(default)]
    pub channels: Vec<Ticket>,
    pub readout: Readout,
    #[serde(default)]
    pub stream: LaneStream,
    #[serde(default)]
    pub group: Option<u32>,
    #[serde(default)]
    pub peer: Option<u32>,
    #[serde(default)]
    pub reading: u8,
    #[serde(default)]
    pub ports: Vec<PortFeed>,
    #[serde(default)]
    pub kv_less: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LaneStream {
    #[default]
    Text = 0,
    Image = 1,
    Video = 2,
    Audio = 3,
    Context = 4,
    Reference = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortKind {
    Latents,
    LaneVector,
    Context,
    AxisPositions,
    Voxels,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortFeed {
    pub kind: PortKind,
    pub port: u8,
    pub channel: u64,
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
        if let Some(masking) = &self.mask
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
        if self.bidirectional && self.mask.is_none() {
            return Err(Error::Invalid(format!(
                "lane in slot {} attends bidirectionally and states no mask; the custom-mask \
                 arm is where the causal bound is lifted, so the lane must carry one",
                self.slot
            )));
        }
        if self.bidirectional && !serves.bidirectional {
            return Err(Error::unsupported("engine", BIDIRECTIONAL_WITHOUT_ARM));
        }
        if let Some(sc) = &self.self_cond {
            let cells = self.rows() as usize * sc.taps as usize;
            let fed = sc.channels.is_some() && sc.rows.is_empty() && sc.weight_bits.is_empty();
            if sc.taps == 0 || (!fed && (sc.rows.len() != cells || sc.weight_bits.len() != cells)) {
                return Err(Error::Invalid(format!(
                    "lane in slot {} states a self-conditioning input of {} taps as {} ids and \
                     {} weights for its {} rows",
                    self.slot,
                    sc.taps,
                    sc.rows.len(),
                    sc.weight_bits.len(),
                    self.rows()
                )));
            }
        }
        if !matches!(self.rs, RsVerb::Fold) && !serves.rs_verbs {
            return Err(Error::unsupported("engine", RS_VERBS_WITHOUT_DEVICE_HALF));
        }
        if let RsVerb::Buffer {
            fold: FoldLen::Host(fold),
            replay,
            ..
        } = &self.rs
            && *fold > replay.saturating_add(self.rows())
        {
            return Err(Error::Invalid(if *replay == 0 {
                format!(
                    "lane in slot {} folds {fold} of the {} rows it carries",
                    self.slot,
                    self.rows()
                )
            } else {
                format!(
                    "lane in slot {} folds {fold} of the {} rows it carries plus the {replay} \
                     buffered token(s) it replays ahead of them",
                    self.slot,
                    self.rows()
                )
            }));
        }
        Ok(())
    }
}

const F3_CHANNEL_TICKETS: &str =
    "Lane::channels against an engine without the pull-validate and commit-bump kernels";

const RS_VERBS_WITHOUT_DEVICE_HALF: &str =
    "Lane::rs beyond RsVerb::Fold: this engine has no device half for it";
const BIDIRECTIONAL_WITHOUT_ARM: &str =
    "Lane::bidirectional: this engine's attention applies its own causal bound and cannot lift it";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serves {
    pub device_channel_commit: bool,
    pub rs_verbs: bool,
    pub bidirectional: bool,
}

impl Serves {
    pub const NONE: Serves = Serves {
        device_channel_commit: false,
        rs_verbs: false,
        bidirectional: false,
    };
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfCondInput {
    pub taps: u32,
    pub rows: Vec<u32>,
    pub weight_bits: Vec<u32>,
    #[serde(default)]
    pub channels: Option<(u64, u64)>,
}

impl SelfCondInput {
    #[must_use]
    pub fn from_channels(taps: u32, rows: u64, weights: u64) -> Self {
        Self {
            taps,
            rows: Vec::new(),
            weight_bits: Vec::new(),
            channels: Some((rows, weights)),
        }
    }

    #[must_use]
    pub fn new(taps: u32, rows: Vec<u32>, weights: &[f32]) -> SelfCondInput {
        SelfCondInput {
            taps,
            rows,
            weight_bits: weights.iter().map(|w| w.to_bits()).collect(),
            channels: None,
        }
    }

    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.weight_bits.iter().map(|&bits| f32::from_bits(bits))
    }
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
    Buffer {
        pages: Vec<u32>,
        at: u32,
        fold: FoldLen,
        #[serde(default)]
        replay: u32,
    },
    Window {
        read: Vec<u32>,
        write: Vec<u32>,
        fold: FoldLen,
    },
    FoldBuffered {
        pages: Vec<u32>,
        at: u32,
        bound: u32,
        len: FoldLen,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldLen {
    Host(u32),
    Device(eta_ir::registry::Port),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Boundary {
    Prologue,
    Epilogue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    pub lane: u32,
    pub instance: InstanceId,
    pub at: Boundary,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepMedia {
    pub lane: u32,
    pub rows: Vec<u32>,
    pub patches: Vec<f32>,
    pub routes: Vec<i32>,
    pub positions: Vec<i32>,
    pub embed_rows: Vec<i32>,
    pub embed_weights: Vec<f32>,
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
        if self.embed_rows.len() != self.embed_weights.len() {
            return Err(Error::Invalid(format!(
                "lane {}'s media carries {} position-table rows and {} weights; the \
                 two streams are read together and are the same length or both empty",
                self.lane,
                self.embed_rows.len(),
                self.embed_weights.len()
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
            if self.lanes[..index].iter().any(|l| l.slot == lane.slot) {
                return Err(Error::Invalid(format!(
                    "slot {} appears twice in one fire, at lane {index}",
                    lane.slot
                )));
            }
        }
        let lanes = u32::try_from(self.lanes.len()).unwrap_or(u32::MAX);
        for (index, attachment) in self.attachments.iter().enumerate() {
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
                     program's stages are one pass with one commit",
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerScores {
    pub layer: u32,
    pub rows: u32,
    pub heads: u32,
    pub lse: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaneReadout {
    pub rows: u32,
    pub width: u32,
    #[serde(default)]
    pub seam: ReadoutSeam,
    pub values: Vec<f32>,
    #[serde(default)]
    pub scores: Vec<LayerScores>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clips: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReadoutSeam {
    #[default]
    Logits,
    Velocity,
    Hidden,
    Pixels,
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
