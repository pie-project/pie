use std::fmt;

#[cfg(test)]
pub(crate) mod fixture;

pub mod compose;
pub mod descriptor;
pub mod fallback;
pub mod packing;
pub mod sink;
pub mod walk;

pub use compose::{
    AxisComposition, ClassWindow, Composition, Lane, LaneRow, MaskSpan, WindowTable, chunk_spans,
    compose, compose_axes, pass_spans, rung_of,
};
pub use descriptor::{
    ABI_VERSION, CLASS_BYTES, FireDescriptor, HEADER_BYTES, LANE_BYTES, MAGIC, PATCH_LANE_BYTES,
    VOXEL_LANE_BYTES,
};
pub use fallback::{Serve, answers as fallback_answers, fragmentable, max_runs};
pub use packing::{LaneFacts, Packed, group_of_lane, groups_of, pack, pack_with_classes};
pub use sink::{EagerSink, EventId, Sink};
pub use walk::{Filter, Phases, Regions, Units, walk, walk_phases, walk_regions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    UnknownWord {
        lane: u32,
        word: u64,
    },
    EmptyLane {
        lane: u32,
    },
    TooManyLanes {
        lanes: usize,
        max: u32,
    },
    TooManyRows {
        rows: u64,
        max: u32,
    },
    NoBucket {
        rows: u32,
        top: u32,
    },
    ClassTable {
        descriptor: usize,
        compiled: usize,
    },
    PrepareAfterCapture {
        region: u32,
    },
    NoSuchNode {
        node: u32,
        nodes: usize,
    },
    DescriptorShort {
        bytes: usize,
    },
    DescriptorMagic {
        saw: u32,
    },
    DescriptorLength {
        bytes: usize,
        want: u64,
    },
    DescriptorRows {
        counted: u64,
        header: u32,
    },
    DescriptorPatchRows {
        counted: u64,
        header: u32,
    },
    DescriptorAbi {
        saw: u32,
        speaks: u32,
    },
    TooManyPatches {
        patches: u64,
        max: u32,
    },
    TooManyImages {
        images: u64,
        max: u32,
    },
    NoPatchBucket {
        patches: u32,
        top: u32,
    },
    Towerless {
        lane: u32,
    },
    NoPatchLadder {
        lane: u32,
    },
    DescriptorVoxelRows {
        counted: u64,
        header: u32,
    },
    TooManyVoxels {
        voxels: u64,
        max: u32,
    },
    TooManyClips {
        clips: u64,
        max: u32,
    },
    NoVoxelBucket {
        voxels: u32,
        top: u32,
    },
    Vaeless {
        lane: u32,
    },
    NoVoxelLadder {
        lane: u32,
    },
    ClipGeometry {
        lane: u32,
        clips: u32,
        voxels: u32,
    },
    Fragmented {
        region: u32,
        runs: u32,
        bound: u32,
        promised: bool,
    },
    PatchGeometry {
        lane: u32,
        images: u32,
        patches: u32,
    },
    PatchRoute {
        at: u32,
        route: i32,
        rows: u32,
    },
    ScatteredSelection {
        mask: u32,
        value: u32,
        at: u32,
        expected: u32,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownWord { lane, word } => write!(
                f,
                "lane {lane} carries fact word {word:#b}, which is no class of \
                 this model — the runtime and the artifact disagree about what \
                 is loaded"
            ),
            Self::EmptyLane { lane } => write!(
                f,
                "lane {lane} carries no token rows, and a lane is its rows"
            ),
            Self::TooManyLanes { lanes, max } => write!(
                f,
                "this fire assembles {lanes} lanes and the artifact was baked \
                 for {max}"
            ),
            Self::TooManyRows { rows, max } => write!(
                f,
                "this fire carries {rows} token rows and every column was cut \
                 at {max}"
            ),
            Self::NoBucket { rows, top } => write!(
                f,
                "this fire carries {rows} token rows and the largest bucket is \
                 {top} — there is no graph to launch it in"
            ),
            Self::ClassTable {
                descriptor,
                compiled,
            } => write!(
                f,
                "the descriptor carries {descriptor} classes and the artifact \
                 has {compiled} — a region's mask would index the wrong window"
            ),
            Self::PrepareAfterCapture { region } => write!(
                f,
                "region {region} is host prepare work standing after the graph \
                 body, so it would write a descriptor slot the launch already \
                 read"
            ),
            Self::NoSuchNode { node, nodes } => write!(
                f,
                "the template runs node {node} of a plan that has {nodes} — \
                 this artifact was compiled from another plan"
            ),
            Self::DescriptorShort { bytes } => write!(
                f,
                "these bytes are not a fire descriptor: {bytes} of them, shorter \
                 than a header"
            ),
            Self::DescriptorMagic { saw } => write!(
                f,
                "these bytes are not a fire descriptor: {saw:#010x} where the FIRE \
                 magic should stand"
            ),
            Self::DescriptorLength { bytes, want } => write!(
                f,
                "this fire descriptor carries {bytes} bytes and its own header calls \
                 for {want}, so a record would be read half out of the next one"
            ),
            Self::DescriptorRows { counted, header } => write!(
                f,
                "this fire descriptor's class windows add up to {counted} rows and \
                 its header claims {header}"
            ),
            Self::DescriptorPatchRows { counted, header } => write!(
                f,
                "this fire descriptor's patch windows add up to {counted} patch rows \
                 and its header claims {header}"
            ),
            Self::DescriptorAbi { saw, speaks } => write!(
                f,
                "these bytes carry descriptor ABI {saw} and this build speaks {speaks} — \
                 the layout is checked and never negotiated, so the two halves are from \
                 two builds"
            ),
            Self::TooManyPatches { patches, max } => write!(
                f,
                "this fire carries {patches} patch rows and every tower column was cut \
                 at {max}"
            ),
            Self::TooManyImages { images, max } => write!(
                f,
                "this fire assembles {images} images and the artifact was baked for {max}"
            ),
            Self::ScatteredSelection {
                mask,
                value,
                at,
                expected,
            } => write!(
                f,
                "the lanes selected by fact mask {mask:#x} = {value:#x} do not stand in one \
                 run of the fire: a selected lane begins at row {at} where the previous one \
                 ended at {expected}, so their packed rectangle would overlap rows another \
                 class owns"
            ),
            Self::NoPatchBucket { patches, top } => write!(
                f,
                "this fire carries {patches} patch rows and the largest patch rung is \
                 {top} — there is no tower exec to launch it in"
            ),
            Self::Towerless { lane } => write!(
                f,
                "lane {lane} submitted images and this artifact declares no patch axis — \
                 there is no vision tower in it for them to go through"
            ),
            Self::NoPatchLadder { lane } => write!(
                f,
                "lane {lane} submitted images and this deployment stated no patch ladder"
            ),
            Self::DescriptorVoxelRows { counted, header } => write!(
                f,
                "this fire descriptor's voxel windows add up to {counted} voxel rows \
                 and its header claims {header}"
            ),
            Self::TooManyVoxels { voxels, max } => write!(
                f,
                "this fire carries {voxels} voxel rows and every VAE column was cut at {max}"
            ),
            Self::TooManyClips { clips, max } => write!(
                f,
                "this fire assembles {clips} clips and the artifact was baked for {max}"
            ),
            Self::NoVoxelBucket { voxels, top } => write!(
                f,
                "this fire carries {voxels} voxel rows and the largest voxel rung is \
                 {top} — there is no VAE exec to launch it in"
            ),
            Self::Vaeless { lane } => write!(
                f,
                "lane {lane} submitted clips and this artifact declares no voxel axis — \
                 there is no VAE in it for them to go through"
            ),
            Self::NoVoxelLadder { lane } => write!(
                f,
                "lane {lane} submitted clips and this deployment stated no voxel ladder"
            ),
            Self::ClipGeometry {
                lane,
                clips,
                voxels,
            } => write!(
                f,
                "lane {lane} declares {clips} clips and {voxels} voxel rows, and a clip \
                 is at least one voxel — its geometry and its payload disagree"
            ),
            Self::Fragmented {
                region,
                runs,
                bound,
                promised,
            } => write!(
                f,
                "region {region} runs as {runs} launches; the bake {} — a mismatched \
                 artifact and class table",
                if *promised {
                    "promised it one interval".to_string()
                } else {
                    format!("bounds it at {bound}")
                }
            ),
            Self::PatchGeometry {
                lane,
                images,
                patches,
            } => write!(
                f,
                "lane {lane} declares {images} images and {patches} patch rows, and an \
                 image is at least one patch row — its geometry and its payload disagree"
            ),
            Self::PatchRoute { at, route, rows } => write!(
                f,
                "patch route {at} lands tower output at token row {route} of a fire with \
                 {rows} rows, and the scatter would write outside the rectangle"
            ),
        }
    }
}

impl std::error::Error for Fault {}
