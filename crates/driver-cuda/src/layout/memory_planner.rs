//! The candidate lattice: choose the forward shape and KV page size that fit
//! the device and score highest.
//!
//! Ported from `plan_cuda_memory` in `store/memory_planner.cpp` (~935 lines,
//! the largest single function in the C++ shell). The policy helpers it is
//! built from -- the targets, the ladders, the saturation score -- were ported
//! ahead of it and live in [`super::planner_policy`].
//!
//! # What this module is, and is not
//!
//! The C++ function opens by calling `cudaGetDevice`, `cudaGetDeviceProperties`
//! and `cudaMemGetInfo`, then spends the remaining nine hundred lines doing
//! arithmetic. Only the arithmetic is here. The three queries produce
//! [`DeviceMemory`] and [`DeviceProps`], and the model layer's workspace
//! formulas arrive through [`ModelCosts`].
//!
//! That boundary is not a convenience: it is what makes the function testable.
//! The C++ cannot be exercised without a GPU **and** a loaded checkpoint, so
//! the lattice, the feasibility filters and the score -- the parts that decide
//! what every deployment runs -- have no tests at all. Here they are a pure
//! function of plain data.
//!
//! # `forced_prefill` is dead in the C++
//!
//! `plan_cuda_memory` declares `constexpr int forced_prefill = 0;` and then
//! branches on `forced_prefill > 0` in five places: an extra `Ns` candidate, a
//! raised `prefill_cap`, two guards on the profile cache and the MoE
//! adjustment, and a ±1000 score override. Every one of those branches is
//! unreachable, and a comment still describes it as "an explicit
//! `PIE_CUDA_PREFILL_TOKENS` is an operator instruction" -- so the environment
//! variable that fed it was removed and the branches were left behind.
//!
//! [`FORCED_PREFILL`] preserves this exactly, dead branches included, because
//! the point of this port is to reproduce what ships. See its docs for the
//! table of what would come back to life.

use super::budget::{CudaMemoryPlan, PlannedForwardLimits};
use super::planner_policy as policy;
use super::profile_cache::Lookup;
use super::profile_key::{ProfileKey, ProfileShape};

/// Every candidate layout must hold at least this much KV, independent of the
/// request cap.
///
/// Below it a boot would admit so few sequences that admission and eviction
/// cannot recover. Named because the "no viable layout" diagnostic reports it.
pub const MIN_KV_TOKENS_FLOOR: u64 = 32768;

/// Relative budget change past which a measured profile stops describing this
/// machine.
pub const BUDGET_TOLERANCE: f64 = 0.05;

/// The operator's prefill pin -- hard-wired to zero, exactly as the C++ is.
///
/// This is `constexpr int forced_prefill = 0` in `plan_cuda_memory`, so every
/// branch guarded by `FORCED_PREFILL > 0` is unreachable. They are kept
/// because they are kept there, and because deleting them would quietly change
/// what happens if the control is ever restored. What it would do:
///
/// | site | effect when non-zero |
/// |---|---|
/// | `Ns` ladder | adds the pinned token count as a candidate |
/// | `prefill_cap` | raises the cap to at least the pinned value |
/// | profile cache | disabled -- an explicit instruction outranks a measurement |
/// | MoE TP2 adjustment | disabled, same reason |
/// | score | ±1000, which makes the pinned candidate win outright |
pub const FORCED_PREFILL: i32 = 0;

/// What `cudaGetDeviceProperties` contributes to the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProps {
    /// `cudaDeviceProp::name`, verbatim -- part of the profile-cache key.
    pub name: String,
    /// Compute capability major.
    pub major: i32,
    /// Compute capability minor.
    pub minor: i32,
    /// Streaming multiprocessors.
    pub sm_count: i32,
}

/// What `cudaMemGetInfo` contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    /// Free device bytes.
    pub free_bytes: u64,
    /// Total device bytes.
    pub total_bytes: u64,
}

/// The `Config` fields the planner reads, lifted out of the engine's config.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerConfig {
    /// `[driver] gpu_mem_utilization`.
    pub gpu_mem_utilization: f64,
    /// `[batching] memory_profile`; `"auto"` means search.
    pub memory_profile: String,
    /// Pinned token count, or 0 to let the lattice choose.
    pub max_forward_tokens: u32,
    /// Pinned request count, or 0 to let the lattice choose.
    pub max_forward_requests: u32,
    /// Pinned KV page size, or 0 to sweep.
    pub kv_page_size: u32,
    /// The RESOLVED KV format name, not the config alias -- key material.
    pub kv_cache_dtype: String,
    /// Tensor-parallel width.
    pub tp_size: i32,
    /// Speculative drafts per program, before clamping.
    pub mtp_num_drafts: i32,
    /// Whether this boot is measuring rather than serving.
    pub calibrating: bool,
    /// Recurrent-state slot multiplier (`PIE_RS_SLOT_MULT`), 1..=8.
    pub rs_slot_mult: i32,
    /// The NCCL id that identifies this tensor-parallel group.
    ///
    /// TP ranks are threads in one process, so the reduction that makes every
    /// rank agree on a plan is an in-process rendezvous keyed by this string.
    pub nccl_unique_id_hex: String,
}

/// The model-shape fields the planner reads.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelShape {
    /// Hidden width.
    pub hidden_size: i32,
    /// Layer count.
    pub num_hidden_layers: i32,
    /// Query heads.
    pub num_attention_heads: i32,
    /// KV heads before tensor-parallel division.
    pub num_key_value_heads: i32,
    /// Head dimension the kernels use for envelopes.
    pub head_dim_kernel: i32,
    /// The catalog id of the loaded row, carried OPAQUELY.
    ///
    /// The planner never compares it, branches on it, or parses it — it is
    /// copied into [`ProfileKey::model_type`](super::profile_key::ProfileKey)
    /// and nowhere else, because a measured profile belongs to the checkpoint
    /// that was measured and the cache key has to say which one. Identity as a
    /// cache key is not identity as a decision.
    ///
    /// It used to be read a second way, as `hf.model_id == "qwen3-8b"`, to
    /// pick a shape knee. That question moved out to [`ShapeKnees`], which the
    /// caller answers.
    pub model_id: String,
}

/// The model layer's cost formulas, which the planner treats as inputs.
///
/// These live in `model/` and `batch/` in the C++ and are called from inside
/// the candidate loop. They are a trait here for the same reason `cache_dir()`
/// was stubbed for the profile cache: they are the planner's **inputs**, not
/// its logic, and modelling them as inputs is what lets the lattice be
/// verified against the C++ without a checkpoint on disk.
pub trait ModelCosts {
    /// Device bytes one KV token costs on this rank, envelopes excluded.
    ///
    /// Non-zero is a precondition the planner checks and reports on.
    fn per_kv_token_bytes(&self) -> u64;

    /// Bytes of KV envelope charged per **page** rather than per token.
    ///
    /// Zero unless Quest key envelopes are switched on. Charged here because
    /// the envelopes share the arena with the pages, so a page count chosen
    /// without them leaves nothing to allocate them from.
    fn envelope_bytes_per_page(&self) -> u64;

    /// Bytes one recurrent-state slot costs, or zero for a model with none.
    fn state_slot_bytes(&self) -> u64;

    /// Every arena term that scales with the forward shape, summed.
    ///
    /// The C++ accumulates ten or so model-specific workspaces here; which of
    /// them apply is a property of the loaded checkpoint, not of the planner.
    fn arena_bytes(&self, n: i32, output_rows: i32, mtp_rows: i32) -> u64;

    /// The float section of the attention workspace.
    fn attn_float_workspace_bytes(&self, n: i32, r: i32) -> u64;

    /// The persistent per-fire input buffers.
    fn persistent_input_bytes(
        &self,
        n: i32,
        r: i32,
        max_page_refs: i32,
        max_custom_mask_bytes: i32,
    ) -> u64;

    /// Scratch the runtime-quantised GEMM path needs for `n` tokens.
    fn runtime_quant_scratch_bytes(&self, n: i32) -> u64;

    /// Whether the model keeps per-request linear-attention state.
    fn has_linear_state(&self) -> bool;
}

/// One read of the profile cache.
///
/// Both halves can be populated at once, and that is not an accident of the
/// port: the C++ signature is
/// `optional<Shape> planner_profile_cache_lookup(key, string* error)`, and it
/// writes the error **and** returns a shape when the file parsed but one entry
/// was malformed. A caller that treats a complaint as "no shape" silently
/// discards a usable measurement, so the two are kept independent here rather
/// than collapsed into [`Lookup`].
#[derive(Debug, Clone, Default)]
pub struct ProfileRead {
    /// The measured shape, if the key matched an entry.
    pub shape: Option<ProfileShape>,
    /// What was wrong with the file, if anything.
    pub complaint: Option<String>,
}

impl From<Lookup> for ProfileRead {
    fn from(l: Lookup) -> Self {
        match l {
            Lookup::Hit(shape) => Self {
                shape: Some(shape),
                complaint: None,
            },
            Lookup::Miss => Self::default(),
            Lookup::Unusable(why) => Self {
                shape: None,
                complaint: Some(why),
            },
        }
    }
}

/// The planner's read side of the profile cache, injected.
///
/// The key construction stays in the planner -- that is the part worth
/// verifying -- while the file I/O is an input, for the same reason the model
/// cost formulas are.
pub trait ProfileSource {
    /// What the cache says about this key.
    fn lookup(&self, key: &ProfileKey) -> ProfileRead;

    /// Where the cache lives, for the diagnostics that tell an operator which
    /// file to delete. A message that says "delete the cache" without naming
    /// it is not actionable.
    fn path(&self) -> String;
}

/// A [`ProfileSource`] that always misses, for boots with no cache and for
/// tests that are not about the cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProfiles;

impl ProfileSource for NoProfiles {
    fn lookup(&self, _key: &ProfileKey) -> ProfileRead {
        ProfileRead::default()
    }

    fn path(&self) -> String {
        String::new()
    }
}

/// Why the planner could not produce a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The weights left nothing to plan with.
    NoBudget {
        /// Bytes `gpu_mem_utilization` allows.
        usable: u64,
        /// Bytes already resident.
        used: u64,
        /// Bytes held back for the graph runtime.
        safety: u64,
    },
    /// The KV geometry came out to zero bytes per token.
    ZeroKvBytes,
    /// The lattice was empty. Carries the operator-facing diagnosis.
    NoViableLayout(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MIB: u64 = 1024 * 1024;
        match self {
            Self::NoBudget {
                usable,
                used,
                safety,
            } => write!(
                f,
                "cuda memory planner: no budget left after weights. usable={} MiB, \
                 used={} MiB, safety={} MiB",
                usable / MIB,
                used / MIB,
                safety / MIB
            ),
            Self::ZeroKvBytes => f.write_str("cuda memory planner: computed zero KV page bytes"),
            Self::NoViableLayout(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for PlanError {}

impl Selector {
    /// The token the verbose summary prints.
    ///
    /// Only two values, because the C++ prints `selected_from_profile ?
    /// "profiled" : "rule"` -- calibration and the preferred-shape override
    /// both report as `rule`, which is worth knowing when reading a log: the
    /// summary cannot distinguish "the score chose this" from "an override
    /// moved it".
    #[must_use]
    pub const fn as_verbose_str(self) -> &'static str {
        match self {
            Self::Profiled => "profiled",
            Self::Rule | Self::CalibrationCeiling | Self::PreferredShape => "rule",
        }
    }
}

/// How the winning candidate was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selector {
    /// A measurement from the profile cache pinned it.
    Profiled,
    /// The analytic score picked it.
    Rule,
    /// A calibration boot took the largest feasible rectangle.
    CalibrationCeiling,
    /// The Qwen3-8B prefill-shape preference moved it.
    PreferredShape,
}

/// One feasible layout and everything the selection step needs about it.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The plan this candidate would build.
    pub plan: CudaMemoryPlan,
    /// Which policy family proposed it.
    pub policy_profile: String,
    /// The decode target it was scored against.
    pub decode_target: i32,
    /// The prefill target it was scored against.
    pub prefill_target: i32,
    /// Its unified score.
    pub score: f64,
    /// Arena bytes it needs.
    pub arena_bytes: u64,
    /// KV tokens the budget affords at its page size.
    pub kv_tokens: u64,
}

/// The planner's answer, with enough context to explain itself.
#[derive(Debug, Clone)]
pub struct Planned {
    /// The chosen plan, before the tensor-parallel reduction.
    pub plan: CudaMemoryPlan,
    /// The profile family that won.
    pub policy_profile: String,
    /// How it was chosen.
    pub selector: Selector,
    /// The budget the search ran inside.
    pub budget: u64,
    /// Decode target the winner was scored against.
    pub decode_target: i32,
    /// Prefill target the winner was scored against.
    pub prefill_target: i32,
    /// How many layouts were feasible.
    pub candidate_count: usize,
    /// Bytes one recurrent-state slot cost, or zero for a model with none.
    ///
    /// Reported because the verbose summary's `logical_state_slots` is
    /// `state_slot_bytes == 0 ? 0 : max_requests`, which cannot be recovered
    /// from the plan alone.
    pub state_slot_bytes: u64,
    /// Bytes already resident when the plan was made.
    pub used_after_weights: u64,
    /// Bytes held back for the graph runtime.
    pub safety: u64,
    /// The score-ranked view of the lattice the C++'s introspection block
    /// prints.
    ///
    /// Kept because the winning SHAPE does not say why it won: two candidates
    /// can agree on every field of the plan and differ in provenance, and a
    /// changed score weight that does not flip an argmax is otherwise
    /// invisible. That invisibility is what makes the C++'s scoring untestable
    /// in place.
    pub introspection: Introspection,
    /// The profile-cache key this plan was looked up under.
    ///
    /// Handed back rather than rebuilt, because the cache's own module doc
    /// says why: reader and writer share `ProfileKey` deliberately, and "if
    /// the two sides built it independently a single disagreement would make
    /// every lookup miss silently -- the cache would look empty rather than
    /// broken". A calibration boot writes what this read.
    pub key: ProfileKey,
    /// Diagnostics the C++ writes to `std::cerr`, in order.
    ///
    /// Returned rather than printed: this crate denies `print_stderr`, and a
    /// caller that wants them on the terminal can log them, while a test can
    /// assert on them. The C++ has no such option, which is why its profile
    /// cache warnings are invisible to its own test suite.
    pub notes: Vec<String>,
}

/// Bytes held back so the graph runtime has somewhere to allocate from.
fn reserves(total_bytes: u64) -> u64 {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "mirrors the C++'s double round-trip through cudaMemGetInfo's size_t"
    )]
    let one_percent = (total_bytes as f64 * 0.01) as u64;
    let graph_runtime_reserve = (512 * 1024 * 1024).max(one_percent);
    (1024 * 1024 * 1024).min(graph_runtime_reserve)
}

/// The budget the lattice searches inside.
///
/// # Errors
///
/// [`PlanError::NoBudget`] when the weights already occupy what
/// `gpu_mem_utilization` allows.
pub fn budget_for(cfg: &PlannerConfig, mem: DeviceMemory) -> Result<u64, PlanError> {
    let current_used = mem.total_bytes - mem.free_bytes;
    let safety = reserves(mem.total_bytes);
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "mirrors the C++'s double round-trip"
    )]
    let usable = (mem.total_bytes as f64 * cfg.gpu_mem_utilization) as u64;
    if usable <= current_used + safety {
        return Err(PlanError::NoBudget {
            usable,
            used: current_used,
            safety,
        });
    }
    Ok(usable - current_used - safety)
}

/// The `Ns` (prefill token) ladder for one policy family.
fn token_ladder(
    profile: &str,
    prefill_target: i32,
    score_as_auto: bool,
    prefer_page16_prefill_knee: bool,
    calibrating: bool,
    prefill_cap: i32,
) -> Vec<i32> {
    let mut ns = vec![
        2 * prefill_target,
        prefill_target,
        1.max(prefill_target / 2),
        1024,
        512,
    ];
    if profile == "throughput" {
        ns.push(4 * prefill_target);
    }
    if profile == "capacity" {
        ns.push(1.max(prefill_target / 4));
    }
    if score_as_auto {
        ns.push(4 * prefill_target);
        ns.push(1.max(prefill_target / 4));
        if prefer_page16_prefill_knee {
            ns.push(5632);
        }
    }
    if FORCED_PREFILL > 0 {
        ns.push(FORCED_PREFILL);
    }
    // A calibration boot searches the space the DEVICE allows, not the space
    // the score prefers. The ladders above are generated from this profile's
    // targets and then clipped by `prefill_cap` -- the analytic model both
    // proposing the candidates and bounding them, so a measurement taken
    // inside it can trim the model's answer but never overrule it. Widening to
    // a geometric sweep lets the only real boundary do the cutting:
    // `arena + persistent >= budget`, which is memory, not preference.
    if calibrating {
        let mut n = 256;
        while n <= 131_072 {
            ns.push(n);
            n *= 2;
        }
    }
    policy::uniq_clip_desc(&ns, if calibrating { 131_072 } else { prefill_cap })
}

/// The `Rs` (decode request) ladder for one policy family.
fn request_ladder(
    profile: &str,
    decode_target: i32,
    score_as_auto: bool,
    calibrating: bool,
) -> Vec<i32> {
    let mut rs = vec![
        2 * decode_target,
        decode_target,
        1.max(decode_target / 2),
        256,
        128,
        64,
        32,
    ];
    if profile == "throughput" || score_as_auto {
        rs.push(4 * decode_target);
    }
    if profile == "latency" {
        rs.push(1.max(decode_target / 4));
    }
    if calibrating {
        let mut r = 16;
        while r <= 4096 {
            rs.push(r);
            r *= 2;
        }
    }
    policy::uniq_clip_desc(&rs, 4096)
}

/// The page-size term of the score.
#[expect(
    clippy::too_many_arguments,
    reason = "each is a distinct measured special case; bundling them would hide which"
)]
fn page_score(
    kv_page_size: i32,
    tp_size: i32,
    profile: &str,
    score_as_auto: bool,
    prefer_page16_prefill_knee: bool,
    r: i32,
    n: i32,
    max_page_refs: i32,
) -> f64 {
    if score_as_auto {
        if prefer_page16_prefill_knee {
            return if kv_page_size == 16 { 0.35 } else { -0.10 };
        }
        if tp_size == 1 {
            return if kv_page_size == 16 { 0.20 } else { -0.05 };
        }
        let latency_shaped = profile == "latency" && r <= 256;
        let metadata_heavy = r >= 512 || n >= 4096 || max_page_refs >= 262_144;
        if latency_shaped && !metadata_heavy {
            return if kv_page_size == 16 { 0.20 } else { -0.05 };
        }
        return if kv_page_size == 32 { 0.20 } else { 0.0 };
    }
    match profile {
        "latency" => {
            if kv_page_size == 16 {
                0.20
            } else {
                -0.20
            }
        }
        "throughput" => {
            if tp_size == 1 {
                if kv_page_size == 16 { 0.25 } else { -0.10 }
            } else if kv_page_size == 32 {
                0.25
            } else {
                0.0
            }
        }
        _ => {
            if tp_size == 1 {
                if kv_page_size == 16 { 0.15 } else { -0.05 }
            } else if kv_page_size == 32 {
                0.15
            } else {
                0.0
            }
        }
    }
}

/// Every term of the unified objective, gathered so the five formulas below
/// read as the weightings they are rather than as arithmetic.
struct ScoreTerms {
    prefill_score: f64,
    decode_score: f64,
    decode_shape_penalty: f64,
    prefill_shape_penalty: f64,
    prefill_overshoot_penalty: f64,
    kv_score: f64,
    kv_headroom: f64,
    kv_headroom_score: f64,
    kv_headroom_penalty: f64,
    min_headroom: f64,
    pressure: f64,
    page_score: f64,
    kv_tokens: u64,
    arena: u64,
    n: i32,
    r: i32,
    score_decode_target: i32,
    score_prefill_target: i32,
}

impl ScoreTerms {
    /// The `auto` objective: a different shape of formula, not a reweighting.
    fn auto(&self, tp_size: i32, prefer_raised_prefill_cap: bool) -> f64 {
        let cohort_score = policy::target_saturation_score(self.r, self.score_decode_target);
        #[expect(
            clippy::cast_precision_loss,
            reason = "token counts stay far below 2^53"
        )]
        let kv_residency_score =
            self.kv_headroom.ln_1p() + (self.kv_tokens as f64 / 131_072.0).ln_1p();
        #[expect(
            clippy::cast_precision_loss,
            reason = "byte counts stay far below 2^53"
        )]
        let arena_mib = self.arena as f64 / (1024.0 * 1024.0);
        let enough = self.kv_headroom >= self.min_headroom;
        let arena_penalty = if enough {
            self.pressure * 0.25
        } else {
            arena_mib / 1024.0 + self.pressure * 0.75
        };
        let prefill_weight = if enough {
            if tp_size > 1 { 4.0 } else { 3.0 }
        } else {
            2.0
        };
        let kv_weight = if enough { 2.0 } else { 4.0 };
        let prefill_underfill_penalty = if prefer_raised_prefill_cap {
            (-policy::log2_ratio(self.n, self.score_prefill_target)).max(0.0)
        } else {
            0.0
        };
        let prefill_target_bonus = if enough
            && self.n >= self.score_prefill_target
            && self.r >= self.score_decode_target
        {
            1.25
        } else {
            0.0
        };
        cohort_score * 6.0
            + self.decode_score * 4.0
            + self.prefill_score * prefill_weight
            + kv_residency_score * kv_weight
            + prefill_target_bonus
            + self.page_score
            - self.decode_shape_penalty * 6.0
            - prefill_underfill_penalty * if enough { 2.0 } else { 0.5 }
            - self.prefill_overshoot_penalty * 0.75
            - self.prefill_shape_penalty * 0.5
            - self.kv_headroom_penalty * 4.0
            - arena_penalty
    }

    /// The four named-profile objectives.
    fn named(&self, profile: &str) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "byte counts stay far below 2^53"
        )]
        let arena = self.arena as f64;
        match profile {
            "capacity" => {
                self.kv_score * 9.0
                    + self.kv_headroom_score * 4.0
                    + self.decode_score * 2.5
                    + self.page_score
                    - self.decode_shape_penalty * 8.0
                    - self.prefill_shape_penalty * 2.0
                    - self.kv_headroom_penalty * 4.0
                    - arena / (512.0 * 1024.0 * 1024.0)
            }
            "throughput" => {
                self.prefill_score * 3.0
                    + self.decode_score * 5.0
                    + self.kv_score * 1.25
                    + self.kv_headroom_score * 2.0
                    + self.page_score
                    - self.decode_shape_penalty * 4.0
                    - self.prefill_shape_penalty * 0.75
                    - self.kv_headroom_penalty * 3.0
                    - self.pressure
            }
            "latency" => {
                self.prefill_score
                    + self.decode_score * 1.5
                    + self.kv_score * 1.25
                    + self.kv_headroom_score
                    + self.page_score
                    - self.decode_shape_penalty * 2.0
                    - f64::from(self.r) / f64::from(1.max(self.n))
                    - self.pressure * 2.0
            }
            _ => {
                self.prefill_score * 1.5
                    + self.decode_score * 3.0
                    + self.kv_score * 3.0
                    + self.kv_headroom_score * 2.0
                    + self.page_score
                    - self.decode_shape_penalty * 4.0
                    - self.prefill_shape_penalty
                    - self.kv_headroom_penalty * 3.0
                    - self.pressure * 2.0
            }
        }
    }
}

/// Which measured shape knees this checkpoint is eligible for, stated by the
/// caller.
///
/// **The planner does not know what model it is planning for, and this is how
/// it stays that way.** Each knee below is a measurement taken on one
/// checkpoint and one card, so something has to connect "this checkpoint" to
/// "this knee" — but that something is not the driver. What stood here was a
/// `Family` enum with `Qwen35` / `Qwen35Moe` / `NemotronH` variants and a
/// `hf.model_id == "qwen3-8b"` string compare, which put four checkpoint
/// identities inside a crate that is not allowed to hold one.
///
/// The split is: the caller states WHICH knees a checkpoint qualifies for,
/// and the planner decides whether the DEVICE and the config also qualify
/// (see [`ShapePreferences::detect`]). A card being Ada, a rank count being
/// two, a profile being `auto` — those are the driver's own facts and stay
/// here. Which checkpoint earned the measurement is not.
///
/// [`Default`] is "no knee applies", which is what an ordinary load passes
/// and what every checkpoint without a measurement of its own gets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShapeKnees {
    /// This checkpoint's prefill measurement wants the cap raised to 12288 on
    /// a wide Ampere-or-later part at TP1.
    pub raised_prefill_cap: bool,
    /// This checkpoint measured a 5632-token, page-16 knee on Ada at TP2.
    pub page16_prefill_knee: bool,
    /// This checkpoint measured an 8k-workspace knee on Ada at TP2 — prompt
    /// bursts that need 128 short prompts in one prefill batch.
    pub wide_workspace_knee: bool,
    /// This checkpoint measured an N=2048 prefill knee at TP>1: decode-heavy,
    /// but still hurt when the prompt wave is split into 1k chunks.
    pub tp_prefill_knee_2048: bool,
    /// This checkpoint drafts MTP rows, which the arena has to be sized for.
    ///
    /// Separate from the knees above because it is not a tuning preference:
    /// omitting it silently UNDER-SIZES the arena by the draft rows rather
    /// than costing a few percent.
    pub mtp_draft_rows: bool,
}

/// The per-(model, GPU) shape preferences the C++ hard-codes, resolved
/// against this device.
///
/// Each is a measured knee, not a heuristic. The checkpoint half of the
/// question arrives as [`ShapeKnees`]; what is decided here is the device and
/// config half.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ShapePreferences {
    /// Raise the prefill cap to 12288.
    raised_prefill_cap: bool,
    /// 5632 tokens, page 16.
    page16_prefill_knee: bool,
    /// The 8k workspace.
    wide_workspace_knee: bool,
    /// The N=2048 knee.
    tp_prefill_knee_2048: bool,
}

impl ShapePreferences {
    fn detect(
        cfg: &PlannerConfig,
        prop: &DeviceProps,
        auto_profile: bool,
        knees: ShapeKnees,
    ) -> Self {
        let base = auto_profile && FORCED_PREFILL == 0;
        let wide = prop.sm_count >= 100;
        let ada = prop.major == 8 && prop.minor == 9;
        Self {
            raised_prefill_cap: base
                && cfg.tp_size == 1
                && prop.major >= 8
                && prop.major < 12
                && wide
                && knees.raised_prefill_cap,
            page16_prefill_knee: base
                && cfg.tp_size == 2
                && ada
                && wide
                && knees.page16_prefill_knee,
            wide_workspace_knee: base
                && cfg.tp_size == 2
                && ada
                && wide
                && knees.wide_workspace_knee,
            tp_prefill_knee_2048: knees.tp_prefill_knee_2048 && cfg.tp_size > 1,
        }
    }
}

/// Build the candidate lattice, score it, and pick a winner.
///
/// The port of `plan_cuda_memory`'s body. The device queries and the model
/// cost formulas are parameters rather than calls, which is what lets this be
/// exercised without a GPU or a checkpoint.
///
/// # Errors
///
/// [`PlanError`] when the budget, the KV geometry, or the lattice is empty.
#[expect(
    clippy::too_many_lines,
    reason = "one function in the C++, and splitting the candidate loop would \
              scatter the twenty-odd locals that make the score readable"
)]
pub fn plan(
    cfg: &PlannerConfig,
    hf: &ModelShape,
    prop: &DeviceProps,
    mem: DeviceMemory,
    knees: ShapeKnees,
    costs: &dyn ModelCosts,
    profiles: &dyn ProfileSource,
) -> Result<Planned, PlanError> {
    let mut notes = Vec::new();
    let budget = budget_for(cfg, mem)?;
    super::profile_cache::set_planner_budget_bytes(budget);

    let per_kv_token_bytes = costs.per_kv_token_bytes();
    if per_kv_token_bytes == 0 {
        return Err(PlanError::ZeroKvBytes);
    }
    let global_per_kv_token_bytes =
        per_kv_token_bytes * u64::try_from(cfg.tp_size.max(0)).unwrap_or(0);

    let auto_profile = policy::is_auto_profile(&cfg.memory_profile);
    // A small, narrow device has no throughput regime to trade into: with
    // fewer than 100 SMs and a <=2048-wide model the other three families all
    // resolve to shapes this card cannot fill, so `auto` collapses to the one
    // that is actually reachable. `score_as_auto` drops with it, because the
    // auto objective is written for a four-family search and scoring a
    // single-family lattice with it would apply the auto page and prefill
    // preferences to candidates that never competed.
    let narrow_latency_auto = auto_profile && prop.sm_count < 100 && hf.hidden_size <= 2048;
    let score_as_auto = auto_profile && !narrow_latency_auto;
    let throughput_decode_target = policy::decode_target("throughput", prop.sm_count);
    let kv_heavy_auto_model = global_per_kv_token_bytes >= 192 * 1024;
    let auto_decode_target = i32::min(
        if kv_heavy_auto_model { 256 } else { 512 },
        throughput_decode_target,
    );

    let prefs = ShapePreferences::detect(cfg, prop, auto_profile, knees);
    let base_prefill_cap = policy::prefill_candidate_cap(prop.major);
    let prefill_cap = if FORCED_PREFILL > 0 {
        FORCED_PREFILL.max(base_prefill_cap)
    } else if prefs.raised_prefill_cap {
        base_prefill_cap.max(12288)
    } else {
        base_prefill_cap
    };
    let auto_prefill_target = if prefs.page16_prefill_knee {
        5632
    } else if prefs.raised_prefill_cap {
        prefill_cap
    } else {
        prefill_cap
            .min(2 * policy::prefill_target("throughput", prop.sm_count, prop.major, cfg.tp_size))
    };

    let state_slot_bytes = costs.state_slot_bytes();
    let envelope_bytes_per_page = costs.envelope_bytes_per_page();
    let slot_mult = cfg.rs_slot_mult.clamp(1, 8);

    // Derived from the CONFIGURED profile, not the narrowed one, and hoisted
    // out of the loop below: the C++ computes it once from
    // `cfg.batching.memory_profile`, so a `narrow_latency_auto` boot still
    // sweeps both page sizes even though only `latency` proposes candidates.
    let kv_page_sizes =
        policy::kv_page_size_candidates(cfg.kv_page_size, &cfg.memory_profile, cfg.tp_size);

    let profiles_to_search: Vec<&str> = if narrow_latency_auto {
        vec!["latency"]
    } else {
        policy::policy_profiles(&cfg.memory_profile)
    };

    let mut candidates: Vec<Candidate> = Vec::new();
    for policy_profile in profiles_to_search {
        let decode_target = policy::decode_target(policy_profile, prop.sm_count);
        let prefill_target =
            policy::prefill_target(policy_profile, prop.sm_count, prop.major, cfg.tp_size);

        let mut ns = token_ladder(
            policy_profile,
            prefill_target,
            score_as_auto,
            prefs.page16_prefill_knee,
            cfg.calibrating,
            prefill_cap,
        );
        let mut rs = request_ladder(
            policy_profile,
            decode_target,
            score_as_auto,
            cfg.calibrating,
        );

        // A pinned axis is a single-candidate lattice. This is the point of
        // `pie config tune` writing what it measured: with the shape pinned,
        // the analytic score and every per-(model, GPU) special case stop
        // running at all, because there is nothing left to choose between.
        //
        // Not honoured during a calibration boot -- that boot exists to
        // search, and seeding a search from its own previous answer makes the
        // value a one-way ratchet.
        if !cfg.calibrating && cfg.max_forward_tokens > 0 {
            ns = vec![i32::try_from(cfg.max_forward_tokens).unwrap_or(i32::MAX)];
        }
        if !cfg.calibrating && cfg.max_forward_requests > 0 {
            rs = vec![i32::try_from(cfg.max_forward_requests).unwrap_or(i32::MAX)];
        }

        for &kv_page_size in &kv_page_sizes {
            let per_page_bytes = per_kv_token_bytes * u64::try_from(kv_page_size).unwrap_or(0)
                + envelope_bytes_per_page;
            if per_page_bytes == 0 {
                continue;
            }
            for &n in &ns {
                for &r0 in &rs {
                    if r0 > n {
                        continue;
                    }
                    let max_page_refs = 262_144.max(r0.wrapping_mul(512));
                    let max_custom_mask_bytes = (8 * 1024 * 1024).max(
                        (128 * 1024 * 1024).min(
                            i32::try_from(
                                (i64::from(n) * i64::from(1024.max(r0.wrapping_mul(64))) + 7) / 8,
                            )
                            .unwrap_or(i32::MAX),
                        ),
                    );
                    let output_rows = r0;
                    let mtp_rows = if knees.mtp_draft_rows {
                        r0.wrapping_mul(cfg.mtp_num_drafts.clamp(0, 32))
                    } else {
                        0
                    };

                    let attn_float_bytes = costs.attn_float_workspace_bytes(n, r0);
                    let mut arena = costs.arena_bytes(n, output_rows, mtp_rows);
                    arena += attn_float_bytes;
                    arena += 8 * 1024 * 1024;
                    let persistent_bytes =
                        costs.persistent_input_bytes(n, r0, max_page_refs, max_custom_mask_bytes);
                    let runtime_quant_scratch_bytes = costs.runtime_quant_scratch_bytes(n);
                    arena += runtime_quant_scratch_bytes;
                    arena = policy::align_up(arena, 2 * 1024 * 1024);
                    if arena + persistent_bytes >= budget {
                        continue;
                    }

                    let r = r0;
                    let state_slots = if state_slot_bytes > 0 {
                        r.wrapping_mul(slot_mult)
                    } else {
                        0
                    };
                    let state_bytes = u64::try_from(state_slots).unwrap_or(0) * state_slot_bytes;
                    let minimum_wave_kv_bytes = u64::try_from(r).unwrap_or(0) * per_page_bytes;
                    let remaining = budget - arena - persistent_bytes;
                    if state_bytes > remaining || minimum_wave_kv_bytes > remaining - state_bytes {
                        continue;
                    }
                    // Deliberately the FULL budget, not what is left after the
                    // arena: the arena is a transient graph workspace that is
                    // freed between fires, while the KV pool is the resident
                    // one. Sizing pages against the remainder would leave the
                    // pool permanently smaller than the device can hold.
                    let kv_pages = i32::try_from(
                        (budget / per_page_bytes).min(u64::try_from(i32::MAX).unwrap_or(u64::MAX)),
                    )
                    .unwrap_or(i32::MAX);
                    if kv_pages <= 0 {
                        continue;
                    }
                    let kv_tokens = u64::try_from(kv_pages).unwrap_or(0)
                        * u64::try_from(kv_page_size).unwrap_or(0);

                    // A candidate only needs enough KV to be viable for early
                    // decode; admission and eviction handle longer tails.
                    // Scoring, however, should value layouts that keep a
                    // realistic long-output cohort resident -- using the same
                    // small horizon for both made `auto` prefer very large
                    // request caps that fragmented 512-token generations.
                    let kv_heavy_model = global_per_kv_token_bytes >= 192 * 1024;
                    let low_horizon_kv_heavy = kv_heavy_model
                        && (prop.major >= 12 || mem.total_bytes >= 120 * 1024 * 1024 * 1024);
                    let min_kv_horizon = if score_as_auto {
                        if low_horizon_kv_heavy { 128.0 } else { 256.0 }
                    } else {
                        match policy_profile {
                            "latency" => 256.0,
                            "throughput" => 512.0,
                            _ => 608.0,
                        }
                    };
                    let score_kv_horizon = if score_as_auto {
                        if low_horizon_kv_heavy { 384.0 } else { 544.0 }
                    } else {
                        608.0
                    };
                    // The absolute floor, clamped to what the budget can
                    // actually supply. `kv_tokens` is a function of the budget
                    // and the page size only -- no term of the (N, R) shape
                    // enters it -- so this floor cannot choose BETWEEN
                    // candidates. It either admits every shape or refuses the
                    // model outright, and refusing is what an unclamped floor
                    // does to a KV-heavy architecture: gemma-4-31B spends
                    // 1120 KiB per context token, so 32768 tokens is 35 GiB of
                    // KV, unreachable on an 80 GiB card once its 58 GiB of
                    // weights are resident. The `R * min_kv_horizon` term --
                    // the shape-aware half -- still rejects a decode width
                    // this pool would starve.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "mirrors the C++'s ceil() through double"
                    )]
                    let horizon_floor = (f64::from(r) * min_kv_horizon).ceil() as u64;
                    let min_kv_tokens = MIN_KV_TOKENS_FLOOR.min(kv_tokens).max(horizon_floor);
                    if kv_tokens < min_kv_tokens {
                        continue;
                    }

                    let plan = CudaMemoryPlan {
                        kv_page_size,
                        max_workspace_tokens: n,
                        max_requests: r,
                        max_page_refs,
                        kv_page_bytes: per_page_bytes,
                        attn_float_workspace_bytes: attn_float_bytes,
                        runtime_quant_scratch_bytes,
                        persistent_input_bytes: persistent_bytes,
                        capacity: PlannedForwardLimits {
                            max_forward_tokens: n,
                            max_forward_requests: r,
                            max_page_refs,
                            max_logit_rows: output_rows,
                            max_prob_rows: output_rows,
                            max_custom_mask_bytes,
                            max_sampler_rows: output_rows,
                            max_logprob_labels: output_rows,
                        },
                    };

                    let score_decode_target = if score_as_auto {
                        auto_decode_target
                    } else {
                        decode_target
                    };
                    let score_prefill_target = if score_as_auto {
                        auto_prefill_target
                    } else {
                        prefill_target
                    };
                    #[expect(clippy::cast_precision_loss, reason = "counts stay below 2^53")]
                    let kv_tokens_f = kv_tokens as f64;
                    let kv_headroom = kv_tokens_f / f64::max(1.0, f64::from(r) * score_kv_horizon);
                    let min_headroom = if score_as_auto {
                        1.0
                    } else {
                        match policy_profile {
                            "capacity" | "throughput" => 1.0,
                            _ => 1.25,
                        }
                    };
                    #[expect(clippy::cast_precision_loss, reason = "byte counts stay below 2^53")]
                    let pressure = (arena + persistent_bytes + state_bytes + minimum_wave_kv_bytes)
                        as f64
                        / budget as f64;
                    let terms = ScoreTerms {
                        prefill_score: policy::target_saturation_score(n, score_prefill_target),
                        decode_score: policy::target_saturation_score(r, score_decode_target),
                        decode_shape_penalty: policy::log2_ratio(r, score_decode_target).abs(),
                        prefill_shape_penalty: policy::log2_ratio(n, score_prefill_target).abs(),
                        prefill_overshoot_penalty: policy::log2_ratio(n, score_prefill_target)
                            .max(0.0),
                        kv_score: (kv_tokens_f / 65536.0).ln_1p(),
                        kv_headroom,
                        kv_headroom_score: kv_headroom.ln_1p(),
                        kv_headroom_penalty: (min_headroom - kv_headroom).max(0.0),
                        min_headroom,
                        pressure,
                        page_score: page_score(
                            kv_page_size,
                            cfg.tp_size,
                            policy_profile,
                            score_as_auto,
                            prefs.page16_prefill_knee,
                            r,
                            n,
                            max_page_refs,
                        ),
                        kv_tokens,
                        arena,
                        n,
                        r,
                        score_decode_target,
                        score_prefill_target,
                    };
                    let mut score = if score_as_auto {
                        terms.auto(cfg.tp_size, prefs.raised_prefill_cap)
                    } else {
                        terms.named(policy_profile)
                    };
                    // A decode-heavy TP>1 shape that still suffers when the
                    // prompt wave is split into 1k-token chunks. The measured
                    // knee on L40 is N=2048.
                    if prefs.tp_prefill_knee_2048
                        && (auto_profile || policy_profile == "latency")
                        && FORCED_PREFILL == 0
                    {
                        score += if n >= 2048 { 1.5 } else { -1.5 };
                        score -= policy::log2_ratio(n, 2048).abs() * 4.0;
                    }
                    // TP2 prompt bursts on L40 that need the 8k workspace to
                    // keep 128 short prompts in one prefill batch; the 8k plan
                    // still leaves 256 recurrent slots.
                    if prefs.wide_workspace_knee {
                        score += if n >= 8192 { 1.5 } else { -1.5 };
                        score -= policy::log2_ratio(n, 8192).abs() * 4.0;
                    }
                    if FORCED_PREFILL > 0 {
                        score += if n == FORCED_PREFILL { 1000.0 } else { -1000.0 };
                    }

                    candidates.push(Candidate {
                        plan,
                        policy_profile: policy_profile.to_owned(),
                        decode_target: score_decode_target,
                        prefill_target: score_prefill_target,
                        score,
                        arena_bytes: arena,
                        kv_tokens,
                    });
                }
            }
        }
    }

    select(
        cfg,
        hf,
        prop,
        mem,
        budget,
        candidates,
        auto_profile,
        &prefs,
        costs,
        profiles,
        &mut notes,
    )
}

/// The operator-facing diagnosis for an empty lattice.
///
/// Which of the lattice's filters emptied it is not recoverable from a bare
/// "no layout fits", and the KV side is the one an operator can act on.
fn no_viable_layout(
    cfg: &PlannerConfig,
    mem: DeviceMemory,
    budget: u64,
    per_kv_token_bytes: u64,
) -> PlanError {
    const MIB: u64 = 1024 * 1024;
    let mut why = format!(
        "cuda memory planner: no viable forward/KV layout fits budget {} MiB",
        budget / MIB
    );
    let kv_tokens_at_budget = budget.checked_div(per_kv_token_bytes).unwrap_or(0);
    why += &format!(
        " (per-token KV {} KiB \u{2014} at most {kv_tokens_at_budget} KV tokens in this budget)",
        per_kv_token_bytes / 1024
    );
    if cfg.max_forward_tokens > 0 || cfg.max_forward_requests > 0 {
        // A pin is the likeliest reason a lattice that normally has hundreds
        // of candidates has none -- and the operator can act on that, where
        // they cannot act on "no layout fits".
        why += ". [driver] max_forward_tokens/max_forward_requests pin the shape to a \
                single candidate; unset them to let the planner choose, or re-run \
                `pie config tune` on this machine";
    } else if let Some(have_tokens) = budget.checked_div(per_kv_token_bytes) {
        // Unpinned, the usual reason is that the weights left too little
        // behind to clear the KV floor every candidate must meet. "No layout
        // fits" is not actionable; the shortfall and the utilization that
        // would cover it are.
        let need_bytes = MIN_KV_TOKENS_FLOOR * per_kv_token_bytes;
        let current_used = mem.total_bytes - mem.free_bytes;
        let safety = reserves(mem.total_bytes);
        // Round the advice UP to the next hundredth: truncating hands back a
        // utilization that still lands under the floor, which is worse than
        // no advice at all.
        #[expect(clippy::cast_precision_loss, reason = "byte counts stay below 2^53")]
        let need_util = if mem.total_bytes > 0 {
            (((need_bytes + current_used + safety) as f64 / mem.total_bytes as f64) * 100.0).ceil()
                / 100.0
        } else {
            0.0
        };
        why += &format!(
            ". KV needs {} KiB/token, so this budget holds ~{have_tokens} tokens, short of \
             the {MIN_KV_TOKENS_FLOOR} a layout wants before its decode width is the binding \
             term. Raise [driver] gpu_mem_utilization (>= {need_util:.2} here), shrink the \
             weights (`kv_cache_dtype`/quantization), or add a GPU",
            per_kv_token_bytes / 1024
        );
    }
    PlanError::NoViableLayout(why)
}

/// The score-ranked view of the lattice, as the C++'s introspection block
/// reports it.
#[derive(Debug, Clone)]
pub struct Introspection {
    /// Score of the highest-scoring candidate.
    pub top_score: f64,
    /// Its policy family.
    pub top_profile: String,
    /// Its page size.
    pub top_page: i32,
    /// Its token capacity.
    pub top_tokens: i32,
    /// Its request capacity.
    pub top_requests: i32,
    /// Its arena bytes.
    pub top_arena: u64,
    /// Its persistent-input bytes.
    pub top_persistent: u64,
    /// KV tokens the budget affords it.
    pub top_kv_tokens: u64,
    /// Whether the SELECTED candidate is the top-scoring one.
    ///
    /// False means an override moved the choice, which is the single most
    /// useful thing this block reports: it says whether a per-(model, GPU)
    /// special case is load-bearing or dead weight.
    pub selected_is_top: bool,
    /// The selected candidate's score, token and request capacity, for the
    /// line the C++ prints when an override moved it.
    pub selected: (f64, i32, i32),
}

/// Rank the lattice and apply the four selectors in the C++'s order.
#[expect(
    clippy::too_many_arguments,
    reason = "the split from plan() is for readability; these are its locals"
)]
fn select(
    cfg: &PlannerConfig,
    hf: &ModelShape,
    prop: &DeviceProps,
    mem: DeviceMemory,
    budget: u64,
    candidates: Vec<Candidate>,
    auto_profile: bool,
    prefs: &ShapePreferences,
    costs: &dyn ModelCosts,
    profiles: &dyn ProfileSource,
    notes: &mut Vec<String>,
) -> Result<Planned, PlanError> {
    const MIB: u64 = 1024 * 1024;
    if candidates.is_empty() {
        return Err(no_viable_layout(
            cfg,
            mem,
            budget,
            costs.per_kv_token_bytes(),
        ));
    }

    // A measured shape beats a scored one. `calibrate_memory_planner` times
    // the real forward step across the ladder and records the winner; when it
    // has run for this exact (device, model, tp, kv format) we take its answer
    // and skip the analytic score entirely. Fields the calibrator did not
    // measure stay zero and match anything.
    //
    // Two guards beyond `auto_profile`:
    //   * `FORCED_PREFILL == 0` -- an explicit prefill pin is an operator
    //     instruction; every other override defers to it, so the cache must
    //     too, or a stale file silently discards what was asked for.
    //   * not calibrating -- the sweep can only explore up to the arena this
    //     plan builds. Seeding a calibration run from its OWN previous answer
    //     makes the budget a one-way ratchet: once it lowers, no later sweep
    //     can look above the lowered value.
    let mut best: Option<usize> = None;
    let mut selector = Selector::Rule;
    let use_profile_cache = auto_profile && FORCED_PREFILL == 0 && !cfg.calibrating;
    // BUILT UNCONDITIONALLY, though only read when the cache is consulted: a
    // calibration boot does not read the cache and does WRITE it, and the two
    // sides have to agree about the key or the file the sweep fills is a file
    // no later boot can find. So there is one construction and `Planned`
    // hands it back.
    let key = ProfileKey {
        gpu_name: prop.name.clone(),
        compute_major: prop.major,
        compute_minor: prop.minor,
        sm_count: prop.sm_count,
        kv_cache_dtype: cfg.kv_cache_dtype.clone(),
        tp_size: cfg.tp_size,
        model_type: hf.model_id.clone(),
        hidden_size: hf.hidden_size,
        num_hidden_layers: hf.num_hidden_layers,
        num_attention_heads: hf.num_attention_heads,
        num_key_value_heads: hf.num_key_value_heads,
        head_dim: hf.head_dim_kernel,
    };
    if use_profile_cache {
        let read = profiles.lookup(&key);
        if let Some(why) = read.complaint {
            notes.push(format!(
                "memory planner: ignored profile cache {}: {why}",
                profiles.path()
            ));
        }
        let mut measured = read.shape;
        // A shape is only an answer to the budget it was measured under. The
        // key pins the device and the model, and neither notices that this
        // boot has materially more or less memory to give than the sweep did
        // -- another process holding VRAM, or a checkpoint requantized offline
        // by `pie model build`. Left unchecked this fails in the QUIET
        // direction: with a larger budget the measured shape is still
        // feasible, so it is selected, and the extra memory is never used.
        if let Some(m) = &measured
            && m.budget_bytes > 0
        {
            #[expect(clippy::cast_precision_loss, reason = "byte counts stay below 2^53")]
            let drift = (budget as f64 - m.budget_bytes as f64).abs() / m.budget_bytes as f64;
            if drift > BUDGET_TOLERANCE {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "mirrors the C++'s static_cast<int> of a bounded percentage"
                )]
                let pct = (drift * 100.0) as i32;
                notes.push(format!(
                    "memory planner: profile cache was measured against a {} MiB budget and \
                     this boot has {} MiB ({pct}% apart); the measurement does not describe \
                     this machine, so the scored rule decides. Re-run `pie config tune` if \
                     the change is permanent, or free the device if it is not.",
                    m.budget_bytes / MIB,
                    budget / MIB
                ));
                measured = None;
            }
        }
        if let Some(m) = measured {
            for (i, c) in candidates.iter().enumerate() {
                if !m.policy_profile.is_empty() && c.policy_profile != m.policy_profile {
                    continue;
                }
                if m.kv_page_size > 0 && c.plan.kv_page_size != m.kv_page_size {
                    continue;
                }
                if m.max_forward_tokens > 0 && c.plan.max_workspace_tokens != m.max_forward_tokens {
                    continue;
                }
                if m.max_forward_requests > 0 && c.plan.max_requests != m.max_forward_requests {
                    continue;
                }
                // The cache pins only what was measured; among the candidates
                // that satisfy it the score still decides, so a sweep over one
                // axis does not silently freeze the others.
                if best.is_none_or(|b| candidates[b].score < c.score) {
                    best = Some(i);
                    selector = Selector::Profiled;
                }
            }
            if best.is_none() {
                // A measured entry that lands on no feasible candidate is the
                // worst failure mode this cache has: it looks exactly like no
                // cache at all. Say so rather than falling through silently.
                notes.push(format!(
                    "memory planner: profile cache pins max_forward_tokens={} but no candidate \
                     layout matches it; falling back to the scored rule. Delete {} to \
                     re-calibrate.",
                    m.max_forward_tokens,
                    profiles.path()
                ));
            }
        }
    }

    // A calibration boot builds the CEILING of the feasible region rather than
    // the score's pick, because a bigger arena can run a smaller shape and not
    // the other way round. With the ceiling built, the sweep's downward-only
    // ladder stops being a restriction and becomes the correct direction.
    //
    // The ceiling is the largest explorable rectangle: the sweep runs shapes
    // with `requests <= max_requests` and `tokens <= max_workspace_tokens`, so
    // the box it can cover is the product. This is one point on a frontier,
    // not the frontier -- a taller box is a narrower one.
    if cfg.calibrating {
        let area =
            |c: &Candidate| i64::from(c.plan.max_workspace_tokens) * i64::from(c.plan.max_requests);
        let pick = max_by_key_first(&candidates, area);
        let scored = max_by_score(&candidates);
        notes.push(format!(
            "memory planner: calibration boot -- {} candidates fit the {} MiB budget; \
             building the largest at N={} R={} page_size={} (score would have picked N={})",
            candidates.len(),
            budget / MIB,
            candidates[pick].plan.max_workspace_tokens,
            candidates[pick].plan.max_requests,
            candidates[pick].plan.kv_page_size,
            candidates[scored].plan.max_workspace_tokens
        ));
        // The paragraph above justifies the starved KV pool with "a
        // calibration boot serves nothing" -- and nothing in this process
        // enforces that. Calibration does not exit; when it is done this arena
        // is what serves. Say so, because the symptom on the far side (a small
        // page pool, and the sweep's seconds on every start) reads as a
        // hardware limit rather than as a flag left on.
        notes.push(
            "memory planner: this arena is built to be MEASURED, not to serve -- its KV pool \
             is the smallest the largest forward shape leaves. Unset [driver] \
             calibrate_planner before serving from this config."
                .to_owned(),
        );
        best = Some(pick);
        selector = Selector::CalibrationCeiling;
    }

    if best.is_none() && prefs.raised_prefill_cap {
        let preferred_tokens = policy::prefill_candidate_cap(prop.major).max(12288);
        let auto_decode_target = candidates.first().map_or(0, |c| c.decode_target);
        let mut preferred: Option<usize> = None;
        for (i, c) in candidates.iter().enumerate() {
            if c.plan.max_workspace_tokens != preferred_tokens
                || c.plan.max_requests < auto_decode_target
                || c.plan.kv_page_size != 16
            {
                continue;
            }
            if preferred.is_none_or(|p| candidates[p].score < c.score) {
                preferred = Some(i);
            }
        }
        if let Some(p) = preferred {
            best = Some(p);
            selector = Selector::PreferredShape;
        }
    }

    let idx = best.unwrap_or_else(|| max_by_score(&candidates));
    // `stable_sort` by descending score keeps the first of any tie, and
    // `max_element` keeps the first maximum, so the two agree -- which is why
    // `selected_is_top` is a pointer comparison in the C++ and an index
    // comparison here.
    let top = max_by_score(&candidates);
    let winner = &candidates[idx];
    let introspection = Introspection {
        top_score: candidates[top].score,
        top_profile: candidates[top].policy_profile.clone(),
        top_page: candidates[top].plan.kv_page_size,
        top_tokens: candidates[top].plan.max_workspace_tokens,
        top_requests: candidates[top].plan.max_requests,
        top_arena: candidates[top].arena_bytes,
        top_persistent: candidates[top].plan.persistent_input_bytes,
        top_kv_tokens: candidates[top].kv_tokens,
        selected_is_top: idx == top,
        selected: (
            winner.score,
            winner.plan.max_workspace_tokens,
            winner.plan.max_requests,
        ),
    };
    Ok(Planned {
        introspection,
        plan: super::rendezvous::tp_min_plan(cfg.tp_size, &cfg.nccl_unique_id_hex, &winner.plan),
        state_slot_bytes: costs.state_slot_bytes(),
        used_after_weights: mem.total_bytes - mem.free_bytes,
        safety: reserves(mem.total_bytes),
        policy_profile: winner.policy_profile.clone(),
        selector,
        budget,
        decode_target: winner.decode_target,
        prefill_target: winner.prefill_target,
        candidate_count: candidates.len(),
        key,
        notes: std::mem::take(notes),
    })
}

/// `std::max_element` with a key: strictly-greater wins, so the FIRST maximum
/// is kept. C++'s `max_element` has the same tie rule, and the lattice is
/// generated in a fixed order, so this is what decides between equal scores.
fn max_by_key_first<K: Ord>(candidates: &[Candidate], key: impl Fn(&Candidate) -> K) -> usize {
    let mut best = 0;
    for i in 1..candidates.len() {
        if key(&candidates[best]) < key(&candidates[i]) {
            best = i;
        }
    }
    best
}

/// Same tie rule, over the score. Kept separate because `f64` is not `Ord` and
/// the comparison must stay `<` rather than becoming a total order.
fn max_by_score(candidates: &[Candidate]) -> usize {
    let mut best = 0;
    for i in 1..candidates.len() {
        if candidates[best].score < candidates[i].score {
            best = i;
        }
    }
    best
}

/// C's `%g` at the default precision of 6, which is what `std::ostream <<`
/// applies to a `double`.
///
/// Needed because the verbose summary prints `gpu_mem_utilization` and Rust's
/// `Display` for `f64` is shortest-round-trip: it renders `0.6000000000000001`
/// where C++ renders `0.6`. This is not a cosmetic difference -- the summary is
/// the operator's record of what the planner was asked for.
fn format_g6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_owned();
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".to_owned()
        } else if x > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    const P: i32 = 6;
    // The decimal exponent as %e would report it, taken from Rust's own
    // rounding at the same precision so the two agree at a rounding boundary
    // (9.9999995 is 1e+01 to six digits, not 9.99999e+00).
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let exp: i32 = sci
        .split('e')
        .nth(1)
        .and_then(|e| e.parse().ok())
        .unwrap_or(0);

    let strip = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        }
    };

    if !(-4..P).contains(&exp) {
        let (mantissa, _) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa.to_owned()), exp.abs())
    } else {
        let decimals = usize::try_from((P - 1 - exp).max(0)).unwrap_or(0);
        strip(format!("{x:.decimals$}"))
    }
}

impl Planned {
    /// The one-line summary the C++ prints when `verbose` is set.
    ///
    /// Reproduced verbatim -- field order, units and the literal `(auto)` --
    /// because it is the only account an operator has of why a boot chose the
    /// shape it did, and a log line that differs between the two
    /// implementations is a log line that cannot be compared across a
    /// migration.
    ///
    /// Note `selector` here is two-valued: an override that moved the choice
    /// still reports `rule`. That is the C++'s behaviour, not an omission.
    #[must_use]
    pub fn verbose_summary(
        &self,
        cfg: &PlannerConfig,
        prop: &DeviceProps,
        mem: DeviceMemory,
    ) -> String {
        const MIB: u64 = 1024 * 1024;
        let p = &self.plan;
        let pages = self.budget.checked_div(p.kv_page_bytes).unwrap_or(0);
        let slots = if self.state_slot_bytes == 0 {
            0
        } else {
            p.max_requests
        };
        format!(
            "memory planner: profile={} resolved_profile={} selector={} util={} \
             total={} MiB sm={} tp={} decode_target={} prefill_target={} page_size={} \
             (auto) used_after_weights={} MiB safety={} MiB budget={} MiB N={} R={} \
             page_refs={} persistent_inputs={} MiB rq_scratch={} MiB logical_kv_pages={} \
             kv_tokens={} logical_state_slots={}",
            cfg.memory_profile,
            self.policy_profile,
            self.selector.as_verbose_str(),
            format_g6(cfg.gpu_mem_utilization),
            mem.total_bytes / MIB,
            prop.sm_count,
            cfg.tp_size.max(1),
            self.decode_target,
            self.prefill_target,
            p.kv_page_size,
            self.used_after_weights / MIB,
            self.safety / MIB,
            self.budget / MIB,
            p.max_workspace_tokens,
            p.max_requests,
            p.max_page_refs,
            p.persistent_input_bytes / MIB,
            p.runtime_quant_scratch_bytes / MIB,
            pages,
            pages * u64::try_from(p.kv_page_size.max(0)).unwrap_or(0),
            slots,
        )
    }
}

impl Planned {
    /// The score-ranked report the C++'s introspection block prints.
    ///
    /// One line per entry, in the C++'s order and wording. `want` is 1 there,
    /// so exactly one candidate is listed.
    #[must_use]
    pub fn introspection_report(&self) -> Vec<String> {
        const MIB: u64 = 1024 * 1024;
        let i = &self.introspection;
        let mut out = vec![format!(
            "planner candidates: {} feasible, top {} by score (selector={})",
            self.candidate_count,
            self.candidate_count.min(1),
            self.selector.as_verbose_str()
        )];
        if self.candidate_count > 0 {
            out.push(format!(
                "  #1 {} score={} profile={} page={} N={} R={} arena={} MiB persist={} MiB kv_tok={}",
                if i.selected_is_top { "*" } else { " " },
                format_g6(i.top_score),
                i.top_profile,
                i.top_page,
                i.top_tokens,
                i.top_requests,
                i.top_arena / MIB,
                i.top_persistent / MIB,
                i.top_kv_tokens
            ));
        }
        if !i.selected_is_top {
            out.push(format!(
                "  selected is NOT the top-scoring candidate: an override moved it (N={} R={} score={})",
                i.selected.1,
                i.selected.2,
                format_g6(i.selected.0)
            ));
        }
        out
    }
}
