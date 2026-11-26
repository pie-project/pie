use eta_ir::registry::{GeometryClass, PortMask};
use serde::{Deserialize, Serialize};

use crate::chunk::ChunkFacts;
use crate::transfer::{KvHandle, MemoryDomain};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFacts {
    pub backend: String,
    pub domain: MemoryDomain,
    pub sms: u32,
    pub unified_memory: bool,
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
    pub storage_alignment: u32,
    pub storage_max_tile_bytes: u64,
    pub codegen_backend: Option<String>,
    /// The library forms this engine's guest emitters serve
    /// ([`LibraryForms`]): the planner forms a library region only of a form
    /// every engine of the process states, and plans any other as generated
    /// regions. Required: an engine states what it serves.
    pub library_forms: LibraryForms,
}

pub use eta_compiler::plan::LibraryForms;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireLimits {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub max_page_refs: u32,
    pub max_context: u32,
    /// The rows one prefill fire takes best, when the engine prefers fewer
    /// than `max_tokens` (0: no preference). A guest chunks a long prompt to
    /// it: on Metal, the Neural Engine takes its share of a fire only up to
    /// this many rows.
    #[serde(default)]
    pub prefill_rows: u32,
    /// The most lanes a fire of prompts only (no lane decoding) carries,
    /// past which the engine holds no body for it (0: as many as
    /// `max_lanes`).
    #[serde(default)]
    pub prefill_lanes: u32,
    /// 0: as many as `max_tokens`.
    #[serde(default)]
    pub max_readouts: u32,
    /// The most buffered recurrent rows one fire's lanes replay, summed over
    /// every lane, a decode lane's (`lane_carried`'s none) included: what
    /// the engine's staged replay table holds a fire (0: no bound past each
    /// lane's rung, [`crate::fire::rung_holds`]). The runtime steps no more
    /// replayed rows than this together, a fire that would carry more
    /// going in the steps after.
    #[serde(default)]
    pub replayed_rows: u32,
    /// The frames the engine would have the runtime keep posted where
    /// `[runtime] frame_dispatch_depth` is not stated (0: the runtime's own).
    #[serde(default)]
    pub dispatch_depth: u32,
}

/// What a load states beside its chunk pool ([`ChunkFacts`], which holds
/// every count, size and tier of its state).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolFacts {
    pub adapter_banks: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub device: DeviceFacts,
    pub pools: PoolFacts,
    /// The pools as chunks.
    #[serde(default)]
    pub chunks: ChunkFacts,
    pub limits: FireLimits,
    pub ports: PortMask,
    pub geometry: GeometryClass,
    pub kv_handle: Option<KvHandle>,
    pub media_encode: bool,

    #[serde(default)]
    pub rs_verbs: bool,

    /// The engine serves windowed (sliding-window attention) layers as
    /// state + buffer, as a recurrent layer is (CONTRACT-BC 19, 21): each
    /// windowed layer's ring of its last W folded rows in the lane's record
    /// ([`ChunkFacts::windowed`]), rows from the lane's folded index on in
    /// the window buffer ([`crate::SlotRole::WindowBuffer`]), and
    /// [`crate::Lane::window`] read on every lane. False: no lane states a
    /// window fold to it.
    #[serde(default)]
    pub windowed_state: bool,

    /// The rows inside one lane whose recurrent state a fire lands in slots
    /// of their own as it runs (`Lane::rs_snapshots`); 0 where it lands only
    /// the state after the lane's last row, so a pass is cut at each.
    #[serde(default)]
    pub rs_snapshots: u32,

    #[serde(default)]
    pub bidirectional_attention: bool,

    #[serde(default)]
    pub adapter_attachments: bool,

    #[serde(default)]
    pub has_lora: bool,

    #[serde(default)]
    pub has_attn_page_mask: bool,

    #[serde(default)]
    pub draft_block: u32,
    #[serde(default)]
    pub draft_mask_token: u32,
    #[serde(default)]
    pub draft_proposals_from: u32,

    /// The facts the loaded trace's rows branch on, which a lane's word is
    /// classified by.
    #[serde(default)]
    pub facts: poem_ir::Facts,

    /// The guest's inputs by id, with the lane words that read each.
    #[serde(default)]
    pub inputs: Vec<poem_ir::InputDecl>,

    #[serde(default)]
    pub classes: poem_ir::LaneClasses,

    /// What a guest stage's cost is held to on this device
    /// ([`GuestCost`]): none stated, the default, holds it to nothing.
    #[serde(default)]
    pub guest_cost: GuestCost,
    /// The frames this engine keeps in flight, its run-ahead depth (at
    /// least 1): the one number the runtime dispatches frames by and sizes
    /// a beam group's pool with (`BEAM-PAGE-LISTS.md` §3's `F`). The load
    /// request's depth is what the runtime asks for; this is what the
    /// engine runs.
    pub frames_in_flight: u32,

    /// The mixed fires (prompt rows beside one-row decode lanes) the load
    /// serves without building a program: the runtime's cutter composes no
    /// other ([`MixedLanes::fits`]).
    pub mixed_lanes: MixedLanes,

    /// The host ops this load serves ([`HostOps`]): a program that needs
    /// one it does not is refused at its bind, before any step. Required:
    /// an engine states what it serves.
    pub host_ops: HostOps,

    /// The step compositions the load's step builder forms, as the engine
    /// computed them before its arm walk ([`crate::StepShapes::of`] of the
    /// request's [`crate::LoadRequest::steps`] over its own
    /// [`crate::StepFacts`]), and those it runs eager. The runtime computes
    /// the set again from the stated facts and refuses a load stating
    /// another, by name (CONTRACT-BC 24).
    pub steps: crate::StatedSteps,
}

/// An op the runtime asks an engine for on the host, outside a fire, that
/// a program's finish relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostOp {
    /// A beam group's beam's page list (`crate::Engine::beam_list`), read
    /// at a beam search's finish.
    BeamList,
}

impl HostOp {
    /// The op's name, as the engine's verb.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            HostOp::BeamList => "beam_list",
        }
    }
}

/// The host ops an engine serves ([`Capabilities::host_ops`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostOps {
    /// [`HostOp::BeamList`].
    pub beam_list: bool,
}

impl HostOps {
    /// None served.
    pub const NONE: HostOps = HostOps { beam_list: false };

    #[must_use]
    pub fn serves(self, op: HostOp) -> bool {
        match op {
            HostOp::BeamList => self.beam_list,
        }
    }

    /// The host ops a program of `package` needs: a beam group (a
    /// `Port::Parents` read from a channel) reads its chosen beam's page
    /// list at its finish.
    #[must_use]
    pub fn needed_by(package: &eta_compiler::codegen::launch::LaunchPackage) -> Vec<HostOp> {
        let beam = package
            .ports
            .iter()
            .any(|binding| binding.port == eta_ir::registry::Port::Parents && !binding.is_const);
        if beam {
            vec![HostOp::BeamList]
        } else {
            Vec::new()
        }
    }

    /// The first op of `needed` this does not serve.
    #[must_use]
    pub fn unserved(self, needed: &[HostOp]) -> Option<HostOp> {
        needed.iter().copied().find(|&op| !self.serves(op))
    }
}

/// Which mixed fires a load lays: every one within its fire limits
/// ([`MixedLanes::Any`]), or the prompt-lane rungs it lays, each with the
/// decode lanes beside it ([`MixedLanes::Laid`], ascending by
/// `prompt_lanes`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MixedLanes {
    #[default]
    Any,
    Laid(Vec<MixedRung>),
}

/// One laid mixed shape: at most `prompt_lanes - 1` prompts (the engine
/// adds one lane of its own, the rest of the rows) beside at most
/// `decode_lanes` decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MixedRung {
    pub prompt_lanes: u32,
    pub decode_lanes: u32,
}

impl MixedLanes {
    /// Whether a fire of `prompts` prompt lanes holding `prompt_rows` rows
    /// beside `decodes` one-row decode lanes is one the load lays, within
    /// `limits`: the first rung holding the prompts and its rest lane takes
    /// the decodes, and its padded rows (the prompts' pad lanes, the
    /// decodes' pad lanes, the rest lane) fit `max_tokens`. A fire of only
    /// prompts or only decodes is not a mixed one and fits.
    #[must_use]
    pub fn fits(&self, limits: &FireLimits, prompts: u32, prompt_rows: u32, decodes: u32) -> bool {
        let within =
            prompts + decodes <= limits.max_lanes && prompt_rows + decodes <= limits.max_tokens;
        match self {
            _ if prompts == 0 || decodes == 0 => within,
            Self::Any => within,
            Self::Laid(rungs) => {
                within
                    && rungs
                        .iter()
                        .find(|r| r.prompt_lanes > prompts)
                        .is_some_and(|r| {
                            decodes <= r.decode_lanes
                                && u64::from(prompt_rows)
                                    + u64::from(r.prompt_lanes - prompts)
                                    + u64::from(r.decode_lanes)
                                    <= u64::from(limits.max_tokens)
                        })
            }
        }
    }
}

/// What bounds a guest stage's time on a device: the bound it states on one
/// submission, past which the driver faults the submission (and with it
/// every tenant's fire in it), and the rates its load measured that cost a
/// stage. The runtime refuses a registration whose stage the rates cost past
/// the bound (`eta_ir::validate::SubmitBound`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GuestCost {
    /// The device's stated bound on one submission, ns; `None` where it
    /// states none.
    #[serde(default)]
    pub submit_bound_ns: Option<u64>,
    /// What the load measured one launch takes, ns (`[rates] launch_ns`).
    #[serde(default)]
    pub launch_ns: Option<u64>,
    /// What the load measured an elementwise pass moves, bytes a ns
    /// (`[rates] elementwise_gbps`).
    #[serde(default)]
    pub elementwise_gbps: Option<f64>,
}

impl Capabilities {
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}

/// The frames in flight an engine states ([`Capabilities::frames_in_flight`])
/// from the depth it runs: refused by name at load where none or more than
/// a `u32` holds, never clamped.
pub fn stated_frames(depth: usize) -> Result<u32, crate::error::Error> {
    match u32::try_from(depth) {
        Ok(0) => Err(crate::error::Error::Program(
            "this load keeps no frame in flight; a fire needs one".to_string(),
        )),
        Ok(frames) => Ok(frames),
        Err(_) => Err(crate::error::Error::Program(format!(
            "this load keeps {depth} frames in flight, past what a u32 states"
        ))),
    }
}

#[cfg(test)]
mod mixed_lanes {
    use super::*;

    /// qwen3-0.6b's census load on v6e: 64 lanes and 2608 rows a fire;
    /// prompts at 16, 32 and 64 lanes beside 64, 32 and 16 decodes (the
    /// engine's lanes, past the fire's own).
    fn census() -> (MixedLanes, FireLimits) {
        let rung = |prompt_lanes, decode_lanes| MixedRung {
            prompt_lanes,
            decode_lanes,
        };
        (
            MixedLanes::Laid(vec![rung(16, 64), rung(32, 32), rung(64, 16)]),
            FireLimits {
                max_lanes: 64,
                max_tokens: 2608,
                max_page_refs: 0,
                max_context: 0,
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: 0,
                replayed_rows: 0,
                dispatch_depth: 0,
            },
        )
    }

    #[test]
    fn a_mixed_fire_fits_the_first_rung_holding_its_prompts() {
        let (laid, limits) = census();
        // 35 three-row prompts beside 17 decodes: the 64-lane rung holds
        // 16 decodes, so no laid shape; 31 prompts take the 32-lane rung.
        assert!(!laid.fits(&limits, 35, 105, 17));
        assert!(laid.fits(&limits, 31, 93, 17));
        assert!(
            !laid.fits(&limits, 32, 96, 17),
            "32 prompts and the rest lane pass 32 lanes"
        );
        assert!(laid.fits(&limits, 15, 45, 49));
        assert!(
            !laid.fits(&limits, 15, 45, 50),
            "the fire's own lanes past 64"
        );
        assert!(laid.fits(&limits, 48, 144, 16));
        assert!(
            !laid.fits(&limits, 64, 192, 1),
            "no rung holds 64 prompts and a rest lane"
        );
        // the padded rows: prompts, their pad lanes, the decode lanes, the rest lane
        assert!(laid.fits(&limits, 1, 2608 - 15 - 64, 1));
        assert!(!laid.fits(&limits, 1, 2608 - 15 - 64 + 1, 1));
    }
}
