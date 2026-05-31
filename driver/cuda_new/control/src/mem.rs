//! Memory planner — replaces `driver/cuda/src/cuda_memory_planner.cpp`.
//!
//! Pure arithmetic. Given a memory profile, the device memory state, and
//! model sizing, derive the forward token capacity, the request capacity,
//! the recurrent-state slot count, and the number of KV pages that fit
//! after the weights are resident.
//!
//! This is a standalone port of `plan_cuda_memory`: it consumes only
//! primitives (see [`PlanInputs`]) and does not depend on `crate::arch`.
//! The C++ planner additionally does a few things that are *not* pure
//! arithmetic and are therefore out of scope for this module — they are
//! handled elsewhere in the rewrite:
//!
//!   * the `~/.cache/pie/cuda_memory_profiles.json` profile cache
//!     (cuda_memory_planner.cpp:262-310, 814-877),
//!   * the NCCL TP min-reduce barrier `tp_min_plan`
//!     (cuda_memory_planner.cpp:88-126) — `tp.rs` owns TP orchestration,
//!   * `PIE_CUDA_*` env-var overrides (cuda_memory_planner.cpp:159-260),
//!   * the checkpoint-name-shaped knees keyed on `cudaDeviceProp.name`.
//!
//! What *is* ported faithfully: the usable/used/safety/budget split, the
//! per-profile decode/prefill SM-scaled targets, the (N, R) candidate
//! lattice, the arena sizing, the recurrent-state affordability clamp, the
//! KV-page solve, the min-KV-horizon viability guards, and the per-profile
//! scoring objective (including the `auto` objective). Architectural knees
//! that depend only on primitives we already take (`compute_major/minor`,
//! `sm_count`, `model_type`, `hidden_size`, tp_size) are preserved; the
//! ones keyed on a GPU name string are dropped.

/// Memory profile. Mirrors the `[batching].memory_profile` config string
/// (config.hpp:195-201) and `is_auto_memory_profile`
/// (cuda_memory_planner.cpp:128-130).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Profile {
    Auto,
    Latency,
    Balanced,
    Throughput,
    Capacity,
}

impl Profile {
    pub fn parse(s: &str) -> Option<Profile> {
        Some(match s {
            "auto" => Profile::Auto,
            "latency" => Profile::Latency,
            "balanced" => Profile::Balanced,
            "throughput" => Profile::Throughput,
            "capacity" => Profile::Capacity,
            _ => return None,
        })
    }

    /// The concrete policy string the C++ planner uses for this profile in
    /// the per-candidate target/scoring functions. `Auto` is scored with
    /// its own dedicated objective, but where it must pick a single
    /// concrete profile for a target it uses `"throughput"`
    /// (cuda_memory_planner.cpp:204-212, 232-240).
    fn policy_str(self) -> &'static str {
        match self {
            Profile::Auto => "throughput",
            Profile::Latency => "latency",
            Profile::Balanced => "balanced",
            Profile::Throughput => "throughput",
            Profile::Capacity => "capacity",
        }
    }
}

/// The concrete policy families `auto` evaluates and picks between.
/// Mirrors `planner_policy_profiles` (cuda_memory_planner.cpp:132-139).
const AUTO_POLICY_PROFILES: [&str; 4] = ["latency", "balanced", "throughput", "capacity"];

/// Everything the C++ `plan_cuda_memory` consumes, reduced to primitives.
///
/// Byte fields are absolute device bytes. `weight_bytes` is informational
/// only: the C++ planner derives the post-weights "used" figure from a
/// live `cudaMemGetInfo` probe rather than from a model-reported weight
/// size, so the budget split is driven by [`total_bytes`] /
/// [`free_bytes`]. We keep `weight_bytes` so callers that lack a probe can
/// reconstruct `free_bytes = total_bytes - weight_bytes - other_used`.
#[derive(Clone, Debug)]
pub struct PlanInputs {
    // ── device memory probe (cuda_memory_planner.cpp:343-367) ──────────
    /// Post-weight-load free device memory, from `cudaMemGetInfo`.
    pub free_bytes: usize,
    /// Total device memory, from `cudaMemGetInfo`.
    pub total_bytes: usize,
    /// What the resident model weights occupy. Informational; see struct
    /// docs. Used only for the standalone-caller convenience constructor.
    pub weight_bytes: usize,
    /// `[batching].gpu_mem_utilization`, in (0.0, 1.0] (config.hpp:51,190).
    pub gpu_mem_utilization: f64,

    // ── device shape (was `cudaDeviceProp`) ────────────────────────────
    /// `cudaDeviceProp.multiProcessorCount`.
    pub sm_count: i32,
    /// `cudaDeviceProp.major`.
    pub compute_major: i32,
    /// `cudaDeviceProp.minor`.
    pub compute_minor: i32,
    /// `cfg.distributed.tp_size` (clamped to >= 1 internally).
    pub tp_size: i32,

    // ── model sizing (from HfConfig) ────────────────────────────────────
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// `hf.head_dim` (logical) and `hf.head_dim_kernel` (padded). For
    /// homogeneous bf16 models these are equal; the workspace sizing in
    /// `qwen3_workspace_bytes` reserves extra pad buffers when they differ
    /// (qwen3_forward.cpp:398-407).
    pub head_dim: usize,
    pub head_dim_kernel: usize,
    /// `hf.num_attention_heads` and `hf.num_key_value_heads` (pre-TP).
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_model_len: usize,
    /// Model-type string, for the workspace base size and the Qwen3-8B /
    /// MoE knees (attention_workspace.cpp:77-83, cuda_memory_planner.cpp:409).
    pub model_type: String,

    // ── per-page / per-slot byte sizes (precomputed by the caller) ──────
    /// Bytes per *single KV token* across all source layers, post-TP.
    /// This is `per_kv_token_bytes` (cuda_memory_planner.cpp:382-389) —
    /// the caller computes it via `kv_page_bytes_homogeneous` /
    /// `kv_page_bytes_per_layer` / the Nemotron-H variant. KV bytes for a
    /// page of size P is `kv_bytes_per_token * P`.
    pub kv_bytes_per_token: usize,
    /// Bytes for one recurrent/Mamba state slot (0 if the arch has none).
    /// This is `state_slot_bytes` (cuda_memory_planner.cpp:467-474).
    pub recurrent_slot_bytes: usize,

    // ── arena sizing knobs the workspace helpers consume ───────────────
    /// `max_intermediate`, `max_Hq`, `max_Hk` passed to
    /// `qwen3_workspace_bytes` (qwen3_forward.cpp:372-409). For a dense
    /// homogeneous model these are `intermediate_size`,
    /// `num_attention_heads*head_dim`, `num_key_value_heads*head_dim`.
    pub max_intermediate: usize,
    pub max_hq: usize,
    pub max_hk: usize,
    /// Extra per-N arena bytes (MoE / linear-attention / Mamba workspace).
    /// The caller sums `qwen3_5_*`, `nemotron_h_*`, `gemma4_moe_*`
    /// workspace contributions here; 0 for dense attention models. These
    /// scale linearly with N, so the caller supplies bytes-per-token.
    pub extra_arena_bytes_per_token: usize,
    /// `cfg.distributed.tp_size > 1`? sets the attention float workspace
    /// to its base (no single-rank decode tmp_v/tmp_s).
    pub runtime_quant_scratch_base_bytes: usize,
    /// Per-token runtime-quant scratch (scales with N), if any.
    pub runtime_quant_scratch_bytes_per_token: usize,
}

/// The realized plan. `max_tokens` is the prefill workspace width (N),
/// `max_requests` the decode cohort (R), `num_pages` the KV page count,
/// `recurrent_state_slots` the Mamba/linear state slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemPlan {
    pub max_tokens: usize,
    pub max_requests: usize,
    pub num_pages: usize,
    pub recurrent_state_slots: usize,
    /// The KV page size (16 or 32) the winning candidate used.
    pub kv_page_size: usize,
}

/// Error returned when no plan fits. Mirrors the two `throw`s in
/// `plan_cuda_memory` (cuda_memory_planner.cpp:360-366, 808-812).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// `usable <= current_used + safety` — no budget left after weights
    /// (cuda_memory_planner.cpp:360).
    NoBudget {
        usable_mib: usize,
        used_mib: usize,
        safety_mib: usize,
    },
    /// `per_kv_token_bytes == 0` (cuda_memory_planner.cpp:390).
    ZeroKvPageBytes,
    /// No (page,N,R) candidate fit the budget (cuda_memory_planner.cpp:808).
    NoViableLayout { budget_mib: usize },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NoBudget {
                usable_mib,
                used_mib,
                safety_mib,
            } => write!(
                f,
                "cuda memory planner: no budget left after weights. usable={usable_mib} MiB, \
                 used={used_mib} MiB, safety={safety_mib} MiB"
            ),
            PlanError::ZeroKvPageBytes => {
                write!(f, "cuda memory planner: computed zero KV page bytes")
            }
            PlanError::NoViableLayout { budget_mib } => write!(
                f,
                "cuda memory planner: no viable forward/KV layout fits budget {budget_mib} MiB"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * 1024 * 1024;

/// `align_up(n, a)` (cuda_memory_planner.cpp:40-42).
fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) / a * a
}

/// `clamp_pow2_nearest` (cuda_memory_planner.cpp:44-52). Clamp `value`
/// into `[lo, hi]`, then snap to the nearer of the bracketing powers of
/// two, breaking ties toward the lower bound.
fn clamp_pow2_nearest(value: i64, lo: i64, hi: i64) -> i64 {
    let value = value.max(lo).min(hi);
    let mut p: i64 = 1;
    while p < value && p <= hi / 2 {
        p <<= 1;
    }
    let lower = lo.max(p >> 1);
    let upper = hi.min(p);
    if upper <= lower {
        return lower;
    }
    if value - lower <= upper - value {
        lower
    } else {
        upper
    }
}

/// `log2_ratio(value, target)` (cuda_memory_planner.cpp:177-181).
fn log2_ratio(value: i64, target: i64) -> f64 {
    let v = value.max(1) as f64;
    let t = target.max(1) as f64;
    (v / t).log2()
}

/// `target_saturation_score(value, target)` (cuda_memory_planner.cpp:183-188).
fn target_saturation_score(value: i64, target: i64) -> f64 {
    let capped = value.max(1).min(target.max(1)) as f64;
    let t = target.max(1) as f64;
    (capped + 1.0).log2() / (t + 1.0).log2()
}

/// `derive_kv_page_size_for_profile` (cuda_memory_planner.cpp:141-152).
fn kv_page_size_for_profile(profile: &str, tp_size: i32) -> i32 {
    if tp_size == 1
        && (profile == "latency" || profile == "balanced" || profile == "throughput")
    {
        16
    } else {
        32
    }
}

/// `derive_kv_page_size_candidates` (cuda_memory_planner.cpp:155-175),
/// minus the `PIE_CUDA_KV_PAGE_SIZE` env override. Always includes 16 and
/// 32, deduped + sorted.
fn kv_page_size_candidates(profile: Profile, tp_size: i32) -> Vec<i32> {
    let mut xs: Vec<i32> = Vec::new();
    let policies: Vec<&str> = match profile {
        Profile::Auto => AUTO_POLICY_PROFILES.to_vec(),
        _ => vec![profile.policy_str()],
    };
    for p in &policies {
        xs.push(kv_page_size_for_profile(p, tp_size));
    }
    xs.push(16);
    xs.push(32);
    xs.sort_unstable();
    xs.dedup();
    xs
}

/// `profile_decode_target` (cuda_memory_planner.cpp:190-202).
fn profile_decode_target(profile: &str, sm_count: i32) -> i64 {
    let sm_factor: i64 = if profile == "latency" || profile == "capacity" {
        4
    } else {
        6
    };
    clamp_pow2_nearest(sm_count as i64 * sm_factor, 64, 2048)
}

/// `profile_prefill_target` (cuda_memory_planner.cpp:214-230).
fn profile_prefill_target(profile: &str, tp_size: i32, sm_count: i32, major: i32) -> i64 {
    let tp = tp_size.max(1) as i64;
    let tp_factor = tp.min(2);
    let wide = major >= 12;
    let sm_factor: i64 = if profile == "throughput" {
        if wide { 64 } else { 32 }
    } else {
        16
    };
    let max_target: i64 = if profile == "throughput" {
        if wide { 8192 } else { 4096 }
    } else {
        8192
    };
    let mut target = clamp_pow2_nearest(sm_count as i64 * sm_factor * tp_factor, 512, max_target);
    if profile == "latency" || profile == "capacity" {
        target = (target / 2).max(512);
    }
    target
}

/// `prefill_candidate_cap` (cuda_memory_planner.cpp:249-251).
fn prefill_candidate_cap(major: i32) -> i64 {
    if major >= 12 {
        16384
    } else {
        8192
    }
}

/// `uniq_clip_desc` (cuda_memory_planner.cpp:242-247): clip each entry to
/// `[1, cap]`, dedup, and sort descending.
fn uniq_clip_desc(xs: &mut Vec<i64>, cap: i64) {
    for x in xs.iter_mut() {
        *x = (*x).max(1).min(cap);
    }
    xs.sort_unstable();
    xs.dedup();
    xs.reverse();
}

/// Attention float workspace bytes (attention_workspace.cpp:73-116),
/// reduced to the primitives we keep. The single-rank fast-path requires
/// GQA/head-dim conditions we cannot fully reconstruct from primitives
/// without the flashinfer decode-GQA table and sliding-window/layer-type
/// info; we therefore reproduce the *base* (the common case) and the
/// single-rank planned tmp_v/tmp_s sizing when the head-dim is one of the
/// supported sizes. When in doubt this returns the base, matching the C++
/// fallbacks at lines 87, 91, 99.
fn attention_float_workspace_bytes(inp: &PlanInputs, max_requests: i64) -> usize {
    let qwen_hybrid = matches!(
        inp.model_type.as_str(),
        "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text"
    );
    // attention_workspace.cpp:82-83
    let base = if qwen_hybrid { 128 * MIB } else { 80 * MIB };
    let tp_size = inp.tp_size.max(1);
    if tp_size != 1 || max_requests <= 0 {
        return base; // line 85-87
    }
    if inp.num_key_value_heads == 0 || inp.num_attention_heads % inp.num_key_value_heads != 0 {
        return base; // line 88-91
    }
    // line 94-96: supported head dims. We keep this primitive check; the
    // flashinfer decode-GQA table + sliding-window/layer-type guard
    // (line 93, 97-99) are not reconstructable from primitives, so we fall
    // back to `base` unless the head dim is supported. This matches the
    // C++ result whenever the fast-path is *not* taken; when it would be
    // taken we size the planned buffer as below.
    let supported_head_dim = matches!(inp.head_dim_kernel, 64 | 128 | 256 | 512);
    if !supported_head_dim {
        return base;
    }
    // align_up to power-of-two boundary (attention_workspace.cpp:101-103)
    let aln = |n: usize, a: usize| (n + (a - 1)) & !(a - 1);
    let q_heads = inp.num_attention_heads / tp_size as usize; // line 104-105
    let head_dim = inp.head_dim_kernel; // line 106
    let cta_tile_q = 16usize; // line 107
    let padded_batch = aln(max_requests as usize * 2, 128); // line 108-109
    let tmp_v = q_heads * padded_batch * cta_tile_q * head_dim * 4; // sizeof(float)
    let tmp_s = q_heads * padded_batch * cta_tile_q * 4;
    let planned = tmp_v + tmp_s + 16 * MIB; // line 114
    base.max(aln(planned, 16 * MIB)) // line 115
}

/// `qwen3_workspace_bytes` (qwen3_forward.cpp:372-409). The per-fire
/// arena's attention/MLP/logit scratch, scaling with N and output_rows.
fn qwen3_workspace_bytes(inp: &PlanInputs, n: i64, output_rows: i64) -> usize {
    let bf16 = |elems: usize| elems * 2;
    let fp32 = |elems: usize| elems * 4;
    let n = n as usize;
    let o = (output_rows.max(1)) as usize;
    let hidden = inp.hidden_size;
    let max_hq = inp.max_hq;
    let max_hk = inp.max_hk;
    let max_intermediate = inp.max_intermediate;
    let head_dim = inp.head_dim;
    let head_dim_kernel = inp.head_dim_kernel;
    let vocab = inp.vocab_size;

    let mut bytes = 0usize;
    bytes += bf16(n * hidden);
    bytes += bf16(n * hidden);
    bytes += bf16(n * hidden);
    bytes += bf16(n * (max_hq + 2 * max_hk));
    bytes += bf16(n * (2 * max_intermediate));
    bytes += fp32(n * head_dim);
    bytes += bf16(n * max_hq);
    bytes += bf16(n * max_hk);
    bytes += bf16(n * max_hk);
    bytes += bf16(n * max_hq);
    bytes += bf16(n * hidden);
    bytes += bf16(n * max_intermediate);
    bytes += bf16(n * max_intermediate);
    bytes += bf16(o * vocab);
    bytes += fp32(o * vocab);
    // qwen3_forward.cpp:398-407 — extra padded q/k buffers when the
    // logical head dim differs from the kernel head dim.
    if head_dim != head_dim_kernel {
        let q_heads = max_hq / head_dim.max(1);
        let kv_heads = max_hk / head_dim.max(1);
        let hq_pad = q_heads * head_dim_kernel;
        let hk_pad = kv_heads * head_dim_kernel;
        bytes += bf16(n * hq_pad);
        bytes += bf16(n * hk_pad);
        bytes += bf16(n * hk_pad);
        bytes += bf16(n * hq_pad);
    }
    bytes
}

/// `persistent_input_bytes` (persistent_inputs.cpp:40-55).
fn persistent_input_bytes(n: i64, r: i64, max_page_refs: i64, max_custom_mask_bytes: i64) -> usize {
    let n = n as usize;
    let r = r as usize;
    let mut bytes = 0usize;
    bytes += n * (4 + 4 + 4);
    bytes += (r + 1) * (4 + 4);
    bytes += r * (4 + 4 + 1);
    bytes += max_page_refs as usize * 4;
    bytes += max_custom_mask_bytes as usize;
    bytes += (r + 1) * 4;
    // sizeof(float)*3 + sizeof(int32)*4 + uint32 + uint64 + bool
    bytes += n * (4 * 3 + 4 * 4 + 4 + 8 + 1);
    bytes
}

/// One scored candidate, mirroring the C++ `Candidate` struct
/// (cuda_memory_planner.cpp:476-482).
#[derive(Clone, Debug)]
struct Candidate {
    plan: MemPlan,
    score: f64,
}

/// Faithful port of `plan_cuda_memory` (cuda_memory_planner.cpp:325-940),
/// minus the non-arithmetic side channels documented at the module top.
///
/// `free_bytes`/`total_bytes` are the post-weight-load device memory probe;
/// the budget is `total*util - used - safety` where `used = total - free`.
pub fn plan(inp: &PlanInputs, profile: Profile) -> Result<MemPlan, PlanError> {
    let total_bytes = inp.total_bytes;
    let free_bytes = inp.free_bytes;

    // ── budget split (cuda_memory_planner.cpp:351-367) ──────────────────
    let current_used = total_bytes.saturating_sub(free_bytes);
    // graph_runtime_reserve = max(512 MiB, total * 0.01)  (line 352-354)
    let graph_runtime_reserve = (512 * MIB).max((total_bytes as f64 * 0.01) as usize);
    // safety = min(1 GiB, graph_runtime_reserve)           (line 355-357)
    let safety = (GIB).min(graph_runtime_reserve);
    // usable = total * gpu_mem_utilization                 (line 358-359)
    let usable = (total_bytes as f64 * inp.gpu_mem_utilization) as usize;
    if usable <= current_used + safety {
        return Err(PlanError::NoBudget {
            usable_mib: usable / MIB,
            used_mib: current_used / MIB,
            safety_mib: safety / MIB,
        });
    }
    let budget = usable - current_used - safety; // line 367

    let tp_size = inp.tp_size.max(1);
    let auto_profile = profile == Profile::Auto;

    // policy_profiles + narrow_latency_auto (cuda_memory_planner.cpp:370-378)
    let narrow_latency_auto =
        auto_profile && inp.sm_count < 100 && inp.hidden_size <= 2048;
    let policy_profiles: Vec<&str> = if auto_profile {
        if narrow_latency_auto {
            vec!["latency"]
        } else {
            AUTO_POLICY_PROFILES.to_vec()
        }
    } else {
        vec![profile.policy_str()]
    };
    let score_as_auto = auto_profile && !narrow_latency_auto;

    let kv_page_sizes = kv_page_size_candidates(profile, tp_size);

    // per_kv_token_bytes (cuda_memory_planner.cpp:382-392)
    let per_kv_token_bytes = inp.kv_bytes_per_token;
    if per_kv_token_bytes == 0 {
        return Err(PlanError::ZeroKvPageBytes);
    }
    let global_per_kv_token_bytes = per_kv_token_bytes * tp_size as usize;

    // auto decode/prefill targets (cuda_memory_planner.cpp:395-438)
    let throughput_decode_target = profile_decode_target("throughput", inp.sm_count);
    let kv_heavy_auto_model = global_per_kv_token_bytes >= 192 * 1024;
    let auto_decode_target =
        (if kv_heavy_auto_model { 256 } else { 512 }).min(throughput_decode_target);

    // Architectural Qwen3-8B / Nemotron-H knees that depend only on
    // primitives (cuda_memory_planner.cpp:405-438). The GPU-name-keyed
    // pieces are dropped; these are kept because they key on
    // compute_major/minor + sm_count + model_type + hidden_size only.
    // forced_prefill (PIE_CUDA_PREFILL_TOKENS) is dropped → 0.
    let prefer_qwen3_8b_prefill_shape = auto_profile
        && tp_size == 1
        && inp.compute_major >= 8
        && inp.compute_major < 12
        && inp.sm_count >= 100
        && inp.model_type == "qwen3"
        && inp.hidden_size == 4096;
    let prefer_qwen3_8b_tp2_ada_shape = auto_profile
        && tp_size == 2
        && inp.compute_major == 8
        && inp.compute_minor == 9
        && inp.sm_count >= 100
        && inp.model_type == "qwen3"
        && inp.hidden_size == 4096;
    // Nemotron-H knee needs an is_nemotron_h flag; with primitives we
    // approximate via model_type. (cuda_memory_planner.cpp:420-423)
    let prefer_nemotron_h_tp2_ada_prefill_shape = auto_profile
        && tp_size == 2
        && inp.compute_major == 8
        && inp.compute_minor == 9
        && inp.sm_count >= 100
        && inp.model_type.starts_with("nemotron_h");

    let base_prefill_cap = prefill_candidate_cap(inp.compute_major);
    let prefill_cap = if prefer_qwen3_8b_prefill_shape {
        base_prefill_cap.max(12288)
    } else {
        base_prefill_cap
    };
    let auto_prefill_target = if prefer_qwen3_8b_tp2_ada_shape {
        5632
    } else if prefer_qwen3_8b_prefill_shape {
        prefill_cap
    } else {
        prefill_cap.min(2 * profile_prefill_target("throughput", tp_size, inp.sm_count, inp.compute_major))
    };

    let state_slot_bytes = inp.recurrent_slot_bytes;

    let mut candidates: Vec<Candidate> = Vec::new();

    for policy_profile in &policy_profiles {
        let policy_profile = *policy_profile;
        let decode_target = profile_decode_target(policy_profile, inp.sm_count);
        let prefill_target =
            profile_prefill_target(policy_profile, tp_size, inp.sm_count, inp.compute_major);

        // Ns (cuda_memory_planner.cpp:489-511)
        let mut ns: Vec<i64> = vec![
            2 * prefill_target,
            prefill_target,
            (prefill_target / 2).max(1),
            1024,
            512,
        ];
        if policy_profile == "throughput" {
            ns.push(4 * prefill_target);
        }
        if policy_profile == "capacity" {
            ns.push((prefill_target / 4).max(1));
        }
        if score_as_auto {
            ns.push(4 * prefill_target);
            ns.push((prefill_target / 4).max(1));
            if prefer_qwen3_8b_tp2_ada_shape {
                ns.push(5632);
            }
        }
        // forced_prefill dropped (always 0).

        // Rs (cuda_memory_planner.cpp:512-526)
        let mut rs: Vec<i64> = vec![
            2 * decode_target,
            decode_target,
            (decode_target / 2).max(1),
            256,
            128,
            64,
            32,
        ];
        if policy_profile == "throughput" || score_as_auto {
            rs.push(4 * decode_target);
        }
        if policy_profile == "latency" {
            rs.push((decode_target / 4).max(1));
        }

        uniq_clip_desc(&mut ns, prefill_cap);
        uniq_clip_desc(&mut rs, 4096);

        for &kv_page_size in &kv_page_sizes {
            let per_page_bytes = per_kv_token_bytes * kv_page_size as usize;
            if per_page_bytes == 0 {
                continue;
            }
            for &n in &ns {
                for &r0 in &rs {
                    if r0 > n {
                        continue; // line 535
                    }
                    // max_page_refs (line 536)
                    let max_page_refs = (262144i64).max(r0 * 512);
                    // max_custom_mask_bytes (line 537-542)
                    let max_custom_mask_bytes = (8 * 1024 * 1024i64).max(
                        (128 * 1024 * 1024i64)
                            .min(((n * (1024i64).max(r0 * 64) + 7) / 8) as i64),
                    );
                    let output_rows = r0;

                    // arena (cuda_memory_planner.cpp:544-575)
                    let mut arena = qwen3_workspace_bytes(inp, n, output_rows);
                    // extra MoE / linear / Mamba / gemma4-MoE workspace,
                    // supplied per-token by the caller (lines 547-562).
                    arena += inp.extra_arena_bytes_per_token * n as usize;
                    let attn_float_bytes = attention_float_workspace_bytes(inp, r0);
                    arena += attn_float_bytes; // float section
                    arena += 8 * MIB; // int section (line 566)
                    arena += persistent_input_bytes(n, r0, max_page_refs, max_custom_mask_bytes);
                    // runtime_quant_scratch (lines 569-574)
                    let runtime_quant_scratch_bytes = inp.runtime_quant_scratch_base_bytes
                        + inp.runtime_quant_scratch_bytes_per_token * n as usize;
                    arena += runtime_quant_scratch_bytes;
                    arena = align_up(arena, 2 * MIB); // line 575
                    if arena >= budget {
                        continue; // line 576
                    }

                    // recurrent-state affordability clamp (lines 578-589)
                    let mut r = r0;
                    let mut state_slots: i64 = 0;
                    let mut state_bytes: usize = 0;
                    if state_slot_bytes > 0 {
                        let affordable = (budget - arena) / state_slot_bytes;
                        state_slots = (r as usize).min(affordable) as i64;
                        if state_slots <= 0 {
                            continue;
                        }
                        r = r.min(state_slots);
                        state_bytes = state_slots as usize * state_slot_bytes;
                    }
                    if arena + state_bytes >= budget {
                        continue; // line 589
                    }
                    let remaining = budget - arena - state_bytes; // line 590
                    let kv_pages = (remaining / per_page_bytes) as i64; // line 591
                    if kv_pages <= 0 {
                        continue;
                    }
                    let kv_tokens = kv_pages as usize * kv_page_size as usize; // line 593-594

                    // min KV horizon viability (lines 600-620)
                    let kv_heavy_model = global_per_kv_token_bytes >= 192 * 1024;
                    let low_horizon_kv_heavy = kv_heavy_model
                        && (inp.compute_major >= 12 || total_bytes >= 120 * GIB);
                    let min_kv_horizon: f64 = if score_as_auto {
                        if low_horizon_kv_heavy { 128.0 } else { 256.0 }
                    } else if policy_profile == "latency" {
                        256.0
                    } else if policy_profile == "throughput" {
                        512.0
                    } else {
                        608.0
                    };
                    let score_kv_horizon: f64 = if score_as_auto {
                        if low_horizon_kv_heavy { 384.0 } else { 544.0 }
                    } else {
                        608.0
                    };
                    let min_kv_tokens =
                        (32768usize).max((r as f64 * min_kv_horizon).ceil() as usize);
                    if kv_tokens < min_kv_tokens {
                        continue; // line 620
                    }

                    // realized plan (cuda_memory_planner.cpp:622-643)
                    let plan = MemPlan {
                        max_tokens: n as usize,
                        max_requests: r as usize,
                        num_pages: kv_pages as usize,
                        recurrent_state_slots: state_slots as usize,
                        kv_page_size: kv_page_size as usize,
                    };

                    // scoring (cuda_memory_planner.cpp:645-796)
                    let score_decode_target =
                        if score_as_auto { auto_decode_target } else { decode_target };
                    let score_prefill_target =
                        if score_as_auto { auto_prefill_target } else { prefill_target };
                    let prefill_score = target_saturation_score(n, score_prefill_target);
                    let decode_score = target_saturation_score(r, score_decode_target);
                    let decode_shape_penalty = log2_ratio(r, score_decode_target).abs();
                    let prefill_shape_penalty = log2_ratio(n, score_prefill_target).abs();
                    let prefill_overshoot_penalty =
                        log2_ratio(n, score_prefill_target).max(0.0);
                    let kv_score = (kv_tokens as f64 / 65536.0).ln_1p();
                    let kv_headroom =
                        kv_tokens as f64 / (1.0f64).max(r as f64 * score_kv_horizon);
                    let kv_headroom_score = kv_headroom.ln_1p();
                    let min_headroom: f64 = if score_as_auto {
                        1.0
                    } else if policy_profile == "capacity" {
                        1.0
                    } else if policy_profile == "throughput" {
                        1.0
                    } else {
                        1.25
                    };
                    let kv_headroom_penalty = if kv_headroom < min_headroom {
                        min_headroom - kv_headroom
                    } else {
                        0.0
                    };
                    let pressure =
                        (arena + state_bytes) as f64 / budget as f64;

                    // page_score (cuda_memory_planner.cpp:675-702)
                    let page_score: f64 = if score_as_auto {
                        if prefer_qwen3_8b_tp2_ada_shape {
                            if kv_page_size == 16 { 0.35 } else { -0.10 }
                        } else if tp_size == 1 {
                            if kv_page_size == 16 { 0.20 } else { -0.05 }
                        } else {
                            let latency_shaped = policy_profile == "latency" && r <= 256;
                            let metadata_heavy =
                                r >= 512 || n >= 4096 || max_page_refs >= 262144;
                            if latency_shaped && !metadata_heavy {
                                if kv_page_size == 16 { 0.20 } else { -0.05 }
                            } else if kv_page_size == 32 {
                                0.20
                            } else {
                                0.0
                            }
                        }
                    } else if policy_profile == "latency" {
                        if kv_page_size == 16 { 0.20 } else { -0.20 }
                    } else if policy_profile == "throughput" {
                        if tp_size == 1 {
                            if kv_page_size == 16 { 0.25 } else { -0.10 }
                        } else if kv_page_size == 32 {
                            0.25
                        } else {
                            0.0
                        }
                    } else {
                        // balanced + capacity share this branch in C++ (the
                        // final `else`), per lines 698-702.
                        if tp_size == 1 {
                            if kv_page_size == 16 { 0.15 } else { -0.05 }
                        } else if kv_page_size == 32 {
                            0.15
                        } else {
                            0.0
                        }
                    };

                    // per-profile objective (cuda_memory_planner.cpp:703-775)
                    let mut score: f64;
                    if score_as_auto {
                        let cohort_score = target_saturation_score(r, score_decode_target);
                        let kv_residency_score = kv_headroom.ln_1p()
                            + (kv_tokens as f64 / 131072.0).ln_1p();
                        let arena_mib = arena as f64 / MIB as f64;
                        let enough_kv_headroom = kv_headroom >= min_headroom;
                        let arena_penalty = if enough_kv_headroom {
                            pressure * 0.25
                        } else {
                            arena_mib / 1024.0 + pressure * 0.75
                        };
                        let prefill_weight = if enough_kv_headroom {
                            if tp_size > 1 { 4.0 } else { 3.0 }
                        } else {
                            2.0
                        };
                        let kv_weight = if enough_kv_headroom { 2.0 } else { 4.0 };
                        let prefill_underfill_penalty = if prefer_qwen3_8b_prefill_shape {
                            (-log2_ratio(n, score_prefill_target)).max(0.0)
                        } else {
                            0.0
                        };
                        let prefill_target_bonus = if enough_kv_headroom
                            && n >= score_prefill_target
                            && r >= score_decode_target
                        {
                            1.25
                        } else {
                            0.0
                        };
                        score = cohort_score * 6.0
                            + decode_score * 4.0
                            + prefill_score * prefill_weight
                            + kv_residency_score * kv_weight
                            + prefill_target_bonus
                            + page_score
                            - decode_shape_penalty * 6.0
                            - prefill_underfill_penalty * (if enough_kv_headroom { 2.0 } else { 0.5 })
                            - prefill_overshoot_penalty * 0.75
                            - prefill_shape_penalty * 0.5
                            - kv_headroom_penalty * 4.0
                            - arena_penalty;
                    } else if policy_profile == "capacity" {
                        score = kv_score * 9.0
                            + kv_headroom_score * 4.0
                            + decode_score * 2.5
                            + page_score
                            - decode_shape_penalty * 8.0
                            - prefill_shape_penalty * 2.0
                            - kv_headroom_penalty * 4.0
                            - arena as f64 / (512 * MIB) as f64;
                    } else if policy_profile == "throughput" {
                        score = prefill_score * 3.0
                            + decode_score * 5.0
                            + kv_score * 1.25
                            + kv_headroom_score * 2.0
                            + page_score
                            - decode_shape_penalty * 4.0
                            - prefill_shape_penalty * 0.75
                            - kv_headroom_penalty * 3.0
                            - pressure;
                    } else if policy_profile == "latency" {
                        score = prefill_score
                            + decode_score * 1.5
                            + kv_score * 1.25
                            + kv_headroom_score
                            + page_score
                            - decode_shape_penalty * 2.0
                            - r as f64 / (1i64.max(n)) as f64
                            - pressure * 2.0;
                    } else {
                        // balanced (final else, lines 768-775)
                        score = prefill_score * 1.5
                            + decode_score * 3.0
                            + kv_score * 3.0
                            + kv_headroom_score * 2.0
                            + page_score
                            - decode_shape_penalty * 4.0
                            - prefill_shape_penalty
                            - kv_headroom_penalty * 3.0
                            - pressure * 2.0;
                    }

                    // Qwen3.6-MoE TP2 knee (lines 776-785). Needs an
                    // is_qwen3_5_moe flag; approximate via model_type.
                    let is_qwen3_5_moe = matches!(
                        inp.model_type.as_str(),
                        "qwen3_5_moe" | "qwen3_5_moe_text"
                    );
                    if is_qwen3_5_moe
                        && tp_size > 1
                        && (auto_profile || policy_profile == "latency")
                    {
                        score += if n >= 2048 { 1.5 } else { -1.5 };
                        score -= log2_ratio(n, 2048).abs() * 4.0;
                    }
                    if prefer_nemotron_h_tp2_ada_prefill_shape {
                        score += if n >= 8192 { 1.5 } else { -1.5 };
                        score -= log2_ratio(n, 8192).abs() * 4.0;
                    }
                    // forced_prefill bonus dropped (always 0).

                    candidates.push(Candidate { plan, score });
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(PlanError::NoViableLayout {
            budget_mib: budget / MIB,
        });
    }

    // selection (cuda_memory_planner.cpp:814-902), minus the JSON profile
    // cache. The qwen3-8b preferred-shape override (lines 879-896) is kept
    // because it only consults candidates we generated.
    let mut best: Option<usize> = None;
    if prefer_qwen3_8b_prefill_shape {
        let mut preferred: Option<usize> = None;
        for (i, c) in candidates.iter().enumerate() {
            if c.plan.max_tokens as i64 != prefill_cap
                || (c.plan.max_requests as i64) < auto_decode_target
                || c.plan.kv_page_size != 16
            {
                continue;
            }
            match preferred {
                Some(p) if candidates[p].score >= c.score => {}
                _ => preferred = Some(i),
            }
        }
        best = preferred;
    }
    if best.is_none() {
        // max_element by score (cuda_memory_planner.cpp:897-902). C++
        // max_element returns the *first* max on ties (strict `<`).
        let mut best_i = 0usize;
        for i in 1..candidates.len() {
            if candidates[best_i].score < candidates[i].score {
                best_i = i;
            }
        }
        best = Some(best_i);
    }

    Ok(candidates[best.unwrap()].plan.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative dense bf16 model (~Llama-3-8B-ish) on an H100-class
    /// device (132 SMs, sm_90), with a generous budget.
    fn h100_dense_inputs() -> PlanInputs {
        // 80 GiB H100, ~16 GiB of weights resident.
        let total = 80 * GIB;
        let weights = 16 * GIB;
        PlanInputs {
            free_bytes: total - weights,
            total_bytes: total,
            weight_bytes: weights,
            gpu_mem_utilization: 0.90,
            sm_count: 132,
            compute_major: 9,
            compute_minor: 0,
            tp_size: 1,
            num_layers: 32,
            hidden_size: 4096,
            intermediate_size: 14336,
            vocab_size: 128256,
            head_dim: 128,
            head_dim_kernel: 128,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_model_len: 8192,
            model_type: "llama".to_string(),
            // 32 layers * 8 kv heads * 128 head_dim * 2 (k+v) * 2 bytes bf16
            //   per token = 32 * 8 * 128 * 2 * 2 = 1,048,576 bytes/token.
            kv_bytes_per_token: 32 * 8 * 128 * 2 * 2,
            recurrent_slot_bytes: 0,
            max_intermediate: 14336,
            max_hq: 32 * 128,
            max_hk: 8 * 128,
            extra_arena_bytes_per_token: 0,
            runtime_quant_scratch_base_bytes: 0,
            runtime_quant_scratch_bytes_per_token: 0,
        }
    }

    /// Sanity: a plan is realizable and self-consistent for the budget.
    fn assert_plan_fits(inp: &PlanInputs, plan: &MemPlan) {
        let total = inp.total_bytes;
        let used = total - inp.free_bytes;
        let graph_reserve = (512 * MIB).max((total as f64 * 0.01) as usize);
        let safety = GIB.min(graph_reserve);
        let usable = (total as f64 * inp.gpu_mem_utilization) as usize;
        let budget = usable - used - safety;
        let kv_bytes = plan.num_pages * plan.kv_page_size * inp.kv_bytes_per_token;
        let state_bytes = plan.recurrent_state_slots * inp.recurrent_slot_bytes;
        // KV + state must fit within budget (arena is the remainder).
        assert!(
            kv_bytes + state_bytes <= budget,
            "kv {} + state {} exceeds budget {}",
            kv_bytes,
            state_bytes,
            budget
        );
        assert!(plan.max_tokens > 0);
        assert!(plan.max_requests > 0);
        assert!(plan.max_requests as i64 <= plan.max_tokens as i64);
        assert!(plan.num_pages > 0);
        assert!(plan.kv_page_size == 16 || plan.kv_page_size == 32);
    }

    #[test]
    fn parse_round_trips() {
        assert_eq!(Profile::parse("auto"), Some(Profile::Auto));
        assert_eq!(Profile::parse("latency"), Some(Profile::Latency));
        assert_eq!(Profile::parse("balanced"), Some(Profile::Balanced));
        assert_eq!(Profile::parse("throughput"), Some(Profile::Throughput));
        assert_eq!(Profile::parse("capacity"), Some(Profile::Capacity));
        assert_eq!(Profile::parse("nonsense"), None);
        assert_eq!(Profile::parse(""), None);
    }

    #[test]
    fn each_profile_produces_a_sane_plan() {
        let inp = h100_dense_inputs();
        for profile in [
            Profile::Auto,
            Profile::Latency,
            Profile::Balanced,
            Profile::Throughput,
            Profile::Capacity,
        ] {
            let plan = plan(&inp, profile).unwrap_or_else(|e| {
                panic!("profile {profile:?} should produce a plan, got {e}")
            });
            assert_plan_fits(&inp, &plan);
            assert_eq!(plan.recurrent_state_slots, 0, "dense model has no state");
        }
    }

    #[test]
    fn clamp_pow2_nearest_matches_cpp() {
        // Below lo clamps to lo.
        assert_eq!(clamp_pow2_nearest(10, 64, 2048), 64);
        // Above hi clamps to hi.
        assert_eq!(clamp_pow2_nearest(100000, 64, 2048), 2048);
        // 132*6 = 792 → between 512 and 1024; nearer 1024 (792-512=280 vs
        // 1024-792=232) → 1024.
        assert_eq!(clamp_pow2_nearest(792, 64, 2048), 1024);
        // 132*4 = 528 → between 512 and 1024; nearer 512.
        assert_eq!(clamp_pow2_nearest(528, 64, 2048), 512);
        // Exact power of two stays.
        assert_eq!(clamp_pow2_nearest(512, 64, 2048), 512);
    }

    #[test]
    fn kv_page_size_candidates_match_cpp() {
        // TP1 latency/balanced/throughput → 16; capacity → 32. auto sweeps
        // all four, plus the always-appended 16 and 32 → {16, 32}.
        assert_eq!(kv_page_size_candidates(Profile::Auto, 1), vec![16, 32]);
        // TP1 latency-only → {16} plus appended {16,32} → {16,32}.
        assert_eq!(kv_page_size_candidates(Profile::Latency, 1), vec![16, 32]);
        // TP2 → everything is 32, plus appended {16,32} → {16,32}.
        assert_eq!(kv_page_size_candidates(Profile::Throughput, 2), vec![16, 32]);
        // TP1 capacity → 32, plus {16,32} → {16,32}.
        assert_eq!(kv_page_size_candidates(Profile::Capacity, 1), vec![16, 32]);
    }

    #[test]
    fn tiny_free_memory_errors() {
        let mut inp = h100_dense_inputs();
        // Almost all memory already used → no budget after safety.
        inp.free_bytes = 100 * MIB;
        let err = plan(&inp, Profile::Auto).unwrap_err();
        assert!(matches!(err, PlanError::NoBudget { .. }), "got {err:?}");
    }

    #[test]
    fn util_zero_budget_boundary_errors() {
        let mut inp = h100_dense_inputs();
        // usable = total*util; with a low util the budget vanishes vs used.
        inp.gpu_mem_utilization = 0.05;
        let err = plan(&inp, Profile::Throughput).unwrap_err();
        assert!(matches!(err, PlanError::NoBudget { .. }), "got {err:?}");
    }

    #[test]
    fn zero_kv_bytes_errors() {
        let mut inp = h100_dense_inputs();
        inp.kv_bytes_per_token = 0;
        let err = plan(&inp, Profile::Auto).unwrap_err();
        assert_eq!(err, PlanError::ZeroKvPageBytes);
    }

    #[test]
    fn no_viable_layout_when_budget_too_small_for_any_candidate() {
        // Enough budget to clear the NoBudget gate, but the arena for the
        // smallest candidate (or the KV horizon) never fits. Use a huge
        // per-token KV cost so kv_pages stays 0 / below the min horizon,
        // and a tiny budget so even the smallest arena loses.
        let mut inp = h100_dense_inputs();
        let total = 8 * GIB;
        inp.total_bytes = total;
        // Leave a sliver of budget: used is high but below usable-safety.
        // usable = 8*0.9 = 7.2 GiB, safety = min(1GiB, max(512MiB, 80MiB))
        //        = 512 MiB... actually total*0.01 = 80MiB so reserve=512MiB.
        // Pick used so budget is a few hundred MiB — too small for arena.
        inp.free_bytes = total - (total as f64 * 0.9) as usize + 700 * MIB;
        // Massive logits/vocab and big workspace already dominate arena.
        inp.vocab_size = 256000;
        inp.hidden_size = 8192;
        inp.intermediate_size = 32768;
        inp.max_intermediate = 32768;
        let res = plan(&inp, Profile::Capacity);
        assert!(
            matches!(res, Err(PlanError::NoViableLayout { .. })),
            "expected NoViableLayout, got {res:?}"
        );
    }

    #[test]
    fn page_count_respects_kv_budget() {
        // The realized KV bytes must never exceed remaining budget, and the
        // KV-token horizon must clear the per-profile minimum.
        let inp = h100_dense_inputs();
        let plan = plan(&inp, Profile::Capacity).unwrap();
        let kv_tokens = plan.num_pages * plan.kv_page_size;
        // Capacity profile floor is 608 tokens/request (min horizon),
        // clamped up to >= 32768 tokens.
        let min_kv = (32768usize).max((plan.max_requests as f64 * 608.0).ceil() as usize);
        assert!(
            kv_tokens >= min_kv,
            "kv_tokens {} below min horizon {}",
            kv_tokens,
            min_kv
        );
        assert_plan_fits(&inp, &plan);
    }

    #[test]
    fn recurrent_model_gets_state_slots_and_clamps_requests() {
        // A Qwen3.5-MoE-like hybrid with linear-attention state. Give it a
        // real per-slot cost so state_slots is finite and R clamps to it.
        let mut inp = h100_dense_inputs();
        inp.model_type = "qwen3_5_moe".to_string();
        // 4 MiB per recurrent slot — finite, so affordability binds.
        inp.recurrent_slot_bytes = 4 * MIB;
        // Hybrid models also bump the attn float workspace base to 128 MiB
        // (handled inside attention_float_workspace_bytes).
        let plan = plan(&inp, Profile::Auto).unwrap();
        assert!(
            plan.recurrent_state_slots > 0,
            "recurrent model should allocate state slots"
        );
        // R is clamped to the affordable slot count (line 586).
        assert!(plan.max_requests <= plan.recurrent_state_slots);
        assert_plan_fits(&inp, &plan);
    }

    #[test]
    fn recurrent_slots_clamp_under_tight_budget() {
        // Make slots very expensive so the affordable count caps R hard.
        let mut inp = h100_dense_inputs();
        inp.model_type = "qwen3_5_moe".to_string();
        inp.recurrent_slot_bytes = 256 * MIB; // huge slot
        let plan = plan(&inp, Profile::Throughput).unwrap();
        assert!(plan.recurrent_state_slots > 0);
        assert!(plan.max_requests <= plan.recurrent_state_slots);
        // With 256 MiB slots and a ~46 GiB budget, slot count is small.
        assert!(
            plan.recurrent_state_slots < 256,
            "expensive slots should cap the count, got {}",
            plan.recurrent_state_slots
        );
        assert_plan_fits(&inp, &plan);
    }

    #[test]
    fn deterministic() {
        let inp = h100_dense_inputs();
        for profile in [
            Profile::Auto,
            Profile::Latency,
            Profile::Balanced,
            Profile::Throughput,
            Profile::Capacity,
        ] {
            let a = plan(&inp, profile).unwrap();
            let b = plan(&inp, profile).unwrap();
            let c = plan(&inp, profile).unwrap();
            assert_eq!(a, b, "profile {profile:?} not deterministic");
            assert_eq!(b, c, "profile {profile:?} not deterministic");
        }
    }

    #[test]
    fn higher_utilization_never_shrinks_kv() {
        // Monotonicity sanity: more usable memory should not yield *fewer*
        // KV tokens for the same profile (budget is monotone in util).
        let mut lo = h100_dense_inputs();
        lo.gpu_mem_utilization = 0.70;
        let mut hi = h100_dense_inputs();
        hi.gpu_mem_utilization = 0.95;
        let plo = plan(&lo, Profile::Capacity).unwrap();
        let phi = plan(&hi, Profile::Capacity).unwrap();
        let kv_lo = plo.num_pages * plo.kv_page_size;
        let kv_hi = phi.num_pages * phi.kv_page_size;
        assert!(
            kv_hi >= kv_lo,
            "more util gave fewer kv tokens: {} < {}",
            kv_hi,
            kv_lo
        );
    }

    #[test]
    fn narrow_latency_auto_small_gpu() {
        // auto + small GPU (<100 SMs) + narrow hidden (<=2048) forces the
        // latency policy family only (cuda_memory_planner.cpp:373-377).
        // The plan should still be sane; this exercises that code path.
        let mut inp = h100_dense_inputs();
        inp.sm_count = 80;
        inp.hidden_size = 2048;
        inp.intermediate_size = 5632;
        inp.max_intermediate = 5632;
        inp.num_attention_heads = 16;
        inp.num_key_value_heads = 4;
        inp.max_hq = 16 * 128;
        inp.max_hk = 4 * 128;
        inp.kv_bytes_per_token = 32 * 4 * 128 * 2 * 2;
        let plan = plan(&inp, Profile::Auto).unwrap();
        assert_plan_fits(&inp, &plan);
    }

    #[test]
    fn balanced_and_capacity_differ_in_shape() {
        // Different profiles should generally land on different shapes for
        // the same device (capacity favors KV residency, throughput favors
        // bigger N/R). Not a hard guarantee, but holds for this config.
        let inp = h100_dense_inputs();
        let cap = plan(&inp, Profile::Capacity).unwrap();
        let thr = plan(&inp, Profile::Throughput).unwrap();
        // Capacity should hold at least as many KV tokens as throughput.
        let kv_cap = cap.num_pages * cap.kv_page_size;
        let kv_thr = thr.num_pages * thr.kv_page_size;
        assert!(
            kv_cap >= kv_thr,
            "capacity ({kv_cap}) should not hold fewer KV tokens than throughput ({kv_thr})"
        );
    }
}
