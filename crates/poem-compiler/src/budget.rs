#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub buckets: Vec<u32>,
    pub max_adapters: u32,
    /// The vocab-wide logit rows a lane may read out in one fire: the
    /// readout planes are carved for `max_lanes` of them, and a fire that
    /// reads more is refused. [`crate::arena::readouts_per_lane`] sizes it
    /// to what a trace's drafting asks.
    pub readouts_per_lane: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budgets {
    pub tokens: Budget,
    pub patches: Option<PatchLadder>,
    pub voxels: Option<VoxelLadder>,
}

impl Budgets {
    #[must_use]
    pub fn of(tokens: Budget) -> Budgets {
        Budgets {
            tokens,
            patches: None,
            voxels: None,
        }
    }

    #[must_use]
    pub fn with_voxels(mut self, voxels: VoxelLadder) -> Budgets {
        self.voxels = Some(voxels);
        self
    }

    #[must_use]
    pub fn with_patches(mut self, patches: PatchLadder) -> Budgets {
        self.patches = Some(patches);
        self
    }

    #[must_use]
    pub fn ladder(&self, axis: crate::RowAxis) -> Option<Ladder<'_>> {
        match axis {
            crate::RowAxis::Tokens => Some(self.tokens.ladder()),
            crate::RowAxis::Patches => self.patches.as_ref().map(PatchLadder::ladder),
            crate::RowAxis::Voxels => self.voxels.as_ref().map(VoxelLadder::ladder),
        }
    }

    #[must_use]
    pub fn max_voxels(&self) -> u32 {
        self.voxels.as_ref().map_or(0, |ladder| ladder.max_voxels)
    }

    #[must_use]
    pub fn max_clips(&self) -> u32 {
        self.voxels.as_ref().map_or(0, |ladder| ladder.max_clips)
    }

    #[must_use]
    pub fn max_patches(&self) -> u32 {
        self.patches.as_ref().map_or(0, |ladder| ladder.max_patches)
    }

    #[must_use]
    pub fn max_images(&self) -> u32 {
        self.patches.as_ref().map_or(0, |ladder| ladder.max_images)
    }
}

impl From<Budget> for Budgets {
    fn from(tokens: Budget) -> Budgets {
        Budgets::of(tokens)
    }
}

impl From<&Budget> for Budgets {
    fn from(tokens: &Budget) -> Budgets {
        Budgets::of(tokens.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchLadder {
    pub max_patches: u32,
    pub buckets: Vec<u32>,
    pub max_images: u32,
}

impl PatchLadder {
    #[must_use]
    pub fn new(max_patches: u32, max_images: u32) -> PatchLadder {
        PatchLadder {
            max_patches,
            buckets: Vec::new(),
            max_images,
        }
    }

    #[must_use]
    pub fn ladder(&self) -> Ladder<'_> {
        Ladder {
            max_rows: self.max_patches,
            max_lanes: self.max_images,
            buckets: &self.buckets,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelLadder {
    pub max_voxels: u32,
    pub buckets: Vec<u32>,
    pub max_clips: u32,
}

impl VoxelLadder {
    #[must_use]
    pub fn new(max_voxels: u32, max_clips: u32) -> VoxelLadder {
        VoxelLadder {
            max_voxels,
            buckets: Vec::new(),
            max_clips,
        }
    }

    #[must_use]
    pub fn ladder(&self) -> Ladder<'_> {
        Ladder {
            max_rows: self.max_voxels,
            max_lanes: self.max_clips,
            buckets: &self.buckets,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ladder<'a> {
    pub max_rows: u32,
    pub max_lanes: u32,
    pub buckets: &'a [u32],
}

pub const PATCH_LATTICE_FLOOR: u32 = 64;

impl Budget {
    #[must_use]
    pub fn ladder(&self) -> Ladder<'_> {
        Ladder {
            max_rows: self.max_tokens,
            max_lanes: self.max_lanes,
            buckets: &self.buckets,
        }
    }

    #[must_use]
    pub fn new(max_lanes: u32, max_tokens: u32) -> Budget {
        Budget {
            max_lanes,
            max_tokens,
            buckets: Vec::new(),
            max_adapters: 0,
            readouts_per_lane: crate::arena::READOUTS_PER_LANE,
        }
    }
}

impl Default for Budget {
    fn default() -> Budget {
        Budget::new(256, 8192)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceProfile {
    pub sms: u32,
    pub empty_launch_us: f32,
    pub cond_fixed_us: f32,
    pub cond_per_arm_us: f32,
    pub fat_region_us: f32,

    pub side_streams: u32,

    pub fork_floor_us: f32,

    /// Whether a join's independent branches are laid side by side
    /// (`branch::split`) for the dependency schedule.
    pub branches: bool,

    /// Whether every launchable region is scheduled onto the streams by its
    /// dependencies (`stream::fork_dag`), a region an op.
    pub dag_streams: bool,

    /// The widest fire, in rows, that runs the dependency schedule on the
    /// side streams (0: every fire).
    pub dag_streams_until: u32,

    pub family_us: FamilyCosts,

    pub exclusive: Vec<String>,

    pub grouped: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilyCosts {
    pub attn: f32,
    pub linear: f32,
    pub elemwise: f32,
    pub gather: f32,
    pub collective: f32,
    pub sampling: f32,
    pub ssm: f32,
    pub custom: f32,
    pub conv: f32,
}

impl FamilyCosts {
    #[must_use]
    pub fn of(&self, op: &poem_ir::Operation) -> f32 {
        match op {
            poem_ir::Operation::Attn(_) => self.attn,
            poem_ir::Operation::Linear(_) => self.linear,
            poem_ir::Operation::Elemwise(_) => self.elemwise,
            poem_ir::Operation::Gather(_) => self.gather,
            poem_ir::Operation::Collective(_) => self.collective,
            poem_ir::Operation::Sampling(_) => self.sampling,
            poem_ir::Operation::Ssm(_) => self.ssm,
            poem_ir::Operation::Fused(op) => match op {
                poem_ir::Fused::LseSink { .. }
                | poem_ir::Fused::PoolAttention { .. }
                | poem_ir::Fused::PoolCompress { .. }
                | poem_ir::Fused::MlaPrefillExpanded { .. }
                | poem_ir::Fused::IndexTopkRope { .. } => self.attn,
                poem_ir::Fused::MatmulGlu { .. }
                | poem_ir::Fused::MatmulSwigluMatmul { .. }
                | poem_ir::Fused::MatmulSoftcap { .. }
                | poem_ir::Fused::MatmulBias { .. }
                | poem_ir::Fused::MatmulSplitRows { .. }
                | poem_ir::Fused::MoeSelectGlu { .. }
                | poem_ir::Fused::MoeExperts { .. } => self.linear,
                poem_ir::Fused::QkvFusedQknormRopeVnormWrite { .. }
                | poem_ir::Fused::SplitQkvFused { .. } => self.custom,
                poem_ir::Fused::GdnStep { .. } => self.attn,
                poem_ir::Fused::QGateNormRopeAppend { .. } => self.attn,
                poem_ir::Fused::ResidualAddRmsnorm { .. }
                | poem_ir::Fused::RmsnormResidualAdd { .. }
                | poem_ir::Fused::EmbedScaleAdd { .. }
                | poem_ir::Fused::EmbedScaleAddSelect { .. }
                | poem_ir::Fused::RmsnormRopePartialQ { .. }
                | poem_ir::Fused::HcNormProject { .. }
                | poem_ir::Fused::NormRopeLast { .. }
                | poem_ir::Fused::NormRopeLastAppend { .. }
                | poem_ir::Fused::NormRopePair { .. }
                | poem_ir::Fused::RopeYarnAppend { .. }
                | poem_ir::Fused::SplitRopeYarnAppend { .. }
                | poem_ir::Fused::QgQknormRopeAppend { .. }
                | poem_ir::Fused::NormModulate { .. }
                | poem_ir::Fused::HcPrenormProject { .. }
                | poem_ir::Fused::HcGatesNorm { .. }
                | poem_ir::Fused::ResBlendRmsnorm { .. }
                | poem_ir::Fused::HcPre { .. }
                | poem_ir::Fused::GatedResidualNormModulate { .. } => self.elemwise,
            },
            poem_ir::Operation::Conv(_) => self.conv,
        }
    }
}

impl Default for FamilyCosts {
    fn default() -> FamilyCosts {
        FamilyCosts {
            attn: 60.0,
            linear: 40.0,
            elemwise: 4.0,
            gather: 4.0,
            collective: 50.0,
            sampling: 4.0,
            ssm: 60.0,
            custom: 20.0,
            conv: 40.0,
        }
    }
}

impl Default for DeviceProfile {
    fn default() -> DeviceProfile {
        DeviceProfile {
            sms: 132,
            empty_launch_us: 1.0,
            cond_fixed_us: 5.0,
            cond_per_arm_us: 0.6,
            fat_region_us: 250.0,
            side_streams: 2,
            fork_floor_us: 20.0,
            branches: false,
            dag_streams: false,
            dag_streams_until: 0,
            family_us: FamilyCosts::default(),
            exclusive: Vec::new(),
            grouped: Vec::new(),
        }
    }
}
