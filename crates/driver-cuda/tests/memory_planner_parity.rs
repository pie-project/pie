//! Differential parity for [`driver_cuda::layout::memory_planner`] against
//! the real `store/memory_planner.cpp`.
//!
//! The C++ oracle in `tests/oracle/memory_planner/` compiles the shipping
//! 1,221-line source and sweeps `plan_cuda_memory` over a grid of device
//! shapes, model shapes and configurations. This file sweeps the identical
//! grid through the Rust port and requires the transcripts to be
//! byte-identical, then pins the **C++'s** FNV-1a 64 hash of it.
//!
//! Run `tests/oracle/memory_planner/run.sh` to regenerate the golden.
//!
//! # Why this test exists in this shape
//!
//! `plan_cuda_memory` decides what every deployment runs -- the forward
//! buffer, the request cap, the KV page size -- and has no tests at all,
//! because exercising it needs a GPU **and** a loaded checkpoint. Both of
//! those are inputs, not logic, so the port takes them as parameters and this
//! sweep can cover eight device shapes on one machine. The C++ has only ever
//! run on whatever card was in front of it.

#![expect(
    clippy::unreadable_literal,
    reason = "the golden hash is one opaque token; grouping it invites a typo"
)]

use driver_cuda::layout::memory_planner as mp;
use driver_cuda::layout::budget::CudaMemoryPlan;
use driver_cuda::layout::profile_key::{ProfileKey, ProfileShape};
use std::fmt::Write as _;

/// FNV-1a 64 of the **C++** transcript.
///
/// Hand-written rather than `DefaultHasher`, whose output is explicitly not
/// stable across Rust releases.
const GOLDEN_FNV1A64: u64 = 0x6c6a8167324e6af1;

/// Rows the sweep produces, pinned separately: a hash mismatch says only
/// "something moved", while a row count says whether the sweep itself shrank.
const GOLDEN_ROWS: usize = 963;

const GIB: u64 = 1024 * 1024 * 1024;

// ---------------------------------------------------------------------------
// The stubbed model costs, mirroring tests/oracle/memory_planner/oracle.cpp.
// ---------------------------------------------------------------------------

/// Which architecture the case models. Mirrors the oracle's `Fam`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fam {
    Dense,
    Qwen35,
    Qwen35Moe,
    NemotronH,
    Gemma4,
    Dsv4,
    Kimi,
    Glm5,
    KimiK3,
}

struct Case {
    hf: mp::ModelShape,
    fam: Fam,
    tp: i32,
    envelopes: bool,
    rs_bf16: bool,
    head_dim: i32,
    max_intermediate: i32,
    prefill_graph_capable: bool,
}

impl Case {
    fn hidden(&self) -> u64 {
        u64::try_from(self.hf.hidden_size.max(0)).unwrap_or(0)
    }
    fn tpd(&self) -> u64 {
        u64::try_from(self.tp.max(1)).unwrap_or(1)
    }
    fn layers(&self) -> u64 {
        u64::try_from(self.hf.num_hidden_layers.max(0)).unwrap_or(0)
    }
    fn linear_layers(&self) -> i32 {
        if matches!(self.fam, Fam::Qwen35 | Fam::Qwen35Moe) {
            self.hf.num_hidden_layers * 3 / 4
        } else {
            0
        }
    }
}

/// `n * hidden + r * 8192`, divided by tp -- the shared shape of the five
/// MLA-family workspace stubs, which differ only by their prime multiplier.
fn mla_shape(c: &Case, n: i32, r: i32) -> u64 {
    (u64::try_from(n.max(0)).unwrap_or(0) * c.hidden()
        + u64::try_from(r.max(0)).unwrap_or(0) * 8192)
        / c.tpd()
}

impl mp::ModelCosts for Case {
    fn per_kv_token_bytes(&self) -> u64 {
        let layers = self.layers();
        let heads =
            u64::try_from((self.hf.num_key_value_heads / self.tp.max(1)).max(1)).unwrap_or(1);
        let head_dim = u64::try_from(self.hf.head_dim_kernel.max(0)).unwrap_or(0);
        match self.fam {
            Fam::Dsv4 => {
                layers * (u64::try_from(self.head_dim.max(0)).unwrap_or(0) * 2)
                    + layers * 512
            }
            Fam::Kimi | Fam::Glm5 | Fam::KimiK3 => {
                layers
                    * (u64::from(Self::KV_LORA_RANK) + u64::from(Self::QK_ROPE_HEAD_DIM))
                    * 2
                    + layers * 2
            }
            // The per-layer path sums identical layers here, so it lands on
            // the homogeneous figure -- but by a different route, which is the
            // point of keeping it separate in the oracle.
            Fam::Gemma4 => layers * heads * head_dim * 4,
            Fam::NemotronH => layers * heads * head_dim * 4 / 2,
            _ => layers * heads * head_dim * 4,
        }
    }

    fn envelope_bytes_per_page(&self) -> u64 {
        if self.envelopes {
            2 * 2
                * self.layers()
                * u64::try_from((self.hf.num_key_value_heads / self.tp.max(1)).max(1))
                    .unwrap_or(1)
                * u64::try_from(self.hf.head_dim_kernel.max(0)).unwrap_or(0)
        } else {
            0
        }
    }

    fn state_slot_bytes(&self) -> u64 {
        if self.fam == Fam::NemotronH {
            // The C++ OVERWRITES the qwen path's value for nemotron rather
            // than adding to it, so the linear-state terms never apply here.
            return 28 * self.hidden() * 4 / self.tpd();
        }
        let has_linear = matches!(self.fam, Fam::Qwen35 | Fam::Qwen35Moe | Fam::KimiK3);
        if !has_linear {
            return 0;
        }
        let tp = self.tp.max(1);
        let k_dim = u64::try_from((self.hf_linear_key_heads() / tp).max(0)).unwrap_or(0) * 128;
        let v_dim = u64::try_from((self.hf_linear_value_heads() / tp).max(0)).unwrap_or(0) * 128;
        let conv_dim = 2 * k_dim + v_dim;
        let elem = if self.rs_bf16 { 2 } else { 4 };
        let per_slot_recurrent =
            u64::try_from((self.hf_linear_value_heads() / tp).max(0)).unwrap_or(0) * 128 * 128 * elem;
        let per_slot_conv = 4 * conv_dim * 2;
        u64::try_from(self.linear_layers().max(0)).unwrap_or(0)
            * (per_slot_recurrent + per_slot_conv)
    }

    fn arena_bytes(&self, n: i32, output_rows: i32, mtp_rows: i32) -> u64 {
        let mut a = u64::try_from(n.max(0)).unwrap_or(0) * self.hidden() * 2
            + u64::try_from(output_rows.max(0)).unwrap_or(0) * 4096
            + u64::try_from(self.max_intermediate.max(0)).unwrap_or(0) * 128
            + u64::try_from(self.hf.num_attention_heads.max(0)).unwrap_or(0) * 64
            + u64::try_from(self.hf.num_key_value_heads.max(0)).unwrap_or(0) * 32
            + u64::try_from(mtp_rows.max(0)).unwrap_or(0) * 2048;
        let n_hidden = u64::try_from(n.max(0)).unwrap_or(0) * self.hidden() / self.tpd();
        match self.fam {
            Fam::Qwen35 => a += n_hidden * 3,
            Fam::Qwen35Moe => a += n_hidden * 3 + n_hidden * 5,
            Fam::NemotronH => a += n_hidden * 7,
            Fam::Gemma4 => a += u64::try_from(n.max(0)).unwrap_or(0) * self.hidden() * 11,
            Fam::Dsv4 => a += mla_shape(self, n, output_rows) * 13,
            Fam::Kimi => a += mla_shape(self, n, output_rows) * 17,
            Fam::Glm5 => {
                a += (u64::try_from(n.max(0)).unwrap_or(0) * self.hidden()
                    + u64::try_from(output_rows.max(0)).unwrap_or(0) * 8192
                    + u64::try_from(self.hf_max_position().max(0)).unwrap_or(0) * 4)
                    / self.tpd()
                    * 19;
            }
            Fam::KimiK3 => a += mla_shape(self, n, output_rows) * 23,
            Fam::Dense => {}
        }
        a
    }

    fn attn_float_workspace_bytes(&self, n: i32, r: i32) -> u64 {
        let base = u64::try_from(n.max(0)).unwrap_or(0) * 512
            + u64::try_from(r.max(0)).unwrap_or(0) * 1024;
        if self.prefill_graph_capable { base * 2 } else { base }
    }

    fn persistent_input_bytes(
        &self,
        n: i32,
        r: i32,
        max_page_refs: i32,
        max_custom_mask_bytes: i32,
    ) -> u64 {
        u64::try_from(n.max(0)).unwrap_or(0) * 64
            + u64::try_from(r.max(0)).unwrap_or(0) * 256
            + u64::try_from(max_page_refs.max(0)).unwrap_or(0) * 4
            + u64::try_from(max_custom_mask_bytes.max(0)).unwrap_or(0)
    }

    fn runtime_quant_scratch_bytes(&self, n: i32) -> u64 {
        u64::try_from(n.max(0)).unwrap_or(0) * self.hidden() * 2 + 128 * 1024
    }

    fn has_linear_state(&self) -> bool {
        matches!(self.fam, Fam::Qwen35 | Fam::Qwen35Moe | Fam::KimiK3)
    }
}

impl Case {
    /// MLA compression widths, fixed by the oracle's `make_hf`.
    const KV_LORA_RANK: u32 = 512;
    /// Rotary half of the MLA head, likewise fixed.
    const QK_ROPE_HEAD_DIM: u32 = 64;

    const fn hf_linear_key_heads(&self) -> i32 {
        16
    }
    const fn hf_linear_value_heads(&self) -> i32 {
        32
    }
    const fn hf_max_position(&self) -> i32 {
        131_072
    }
}

/// The profile-cache stub, driven by the sweep exactly as the oracle's global
/// is.
struct StubProfiles {
    shape: Option<ProfileShape>,
    complaint: String,
}

impl mp::ProfileSource for StubProfiles {
    fn lookup(&self, _key: &ProfileKey) -> mp::ProfileRead {
        // Both halves independently: the C++ writes its error out-parameter
        // and still returns the shape, and one of the sweep's cases exercises
        // exactly that.
        mp::ProfileRead {
            shape: self.shape.clone(),
            complaint: if self.complaint.is_empty() {
                None
            } else {
                Some(self.complaint.clone())
            },
        }
    }

    fn path(&self) -> String {
        "/stub/planner_profiles.json".to_owned()
    }
}

// ---------------------------------------------------------------------------
// The sweep.
// ---------------------------------------------------------------------------

struct DeviceCase {
    label: &'static str,
    name: &'static str,
    major: i32,
    minor: i32,
    sms: i32,
    total_gib: u64,
}

fn devices() -> Vec<DeviceCase> {
    vec![
        DeviceCase { label: "l40s", name: "NVIDIA L40S", major: 8, minor: 9, sms: 142, total_gib: 45 },
        DeviceCase { label: "a100", name: "NVIDIA A100-SXM4-80GB", major: 8, minor: 0, sms: 108, total_gib: 80 },
        DeviceCase { label: "h100", name: "NVIDIA H100 80GB HBM3", major: 9, minor: 0, sms: 132, total_gib: 80 },
        DeviceCase { label: "b200", name: "NVIDIA B200", major: 10, minor: 0, sms: 148, total_gib: 180 },
        DeviceCase { label: "rtx5090", name: "NVIDIA GeForce RTX 5090", major: 12, minor: 0, sms: 170, total_gib: 32 },
        DeviceCase { label: "l4", name: "NVIDIA L4", major: 8, minor: 9, sms: 58, total_gib: 24 },
        DeviceCase { label: "t4", name: "Tesla T4", major: 7, minor: 5, sms: 40, total_gib: 16 },
        DeviceCase { label: "ada6000", name: "NVIDIA RTX 6000 Ada", major: 8, minor: 9, sms: 142, total_gib: 48 },
    ]
}

struct ModelCase {
    label: &'static str,
    fam: Fam,
    model_type: &'static str,
    hidden: i32,
    layers: i32,
    kv_heads: i32,
    head_dim: i32,
}

fn models() -> Vec<ModelCase> {
    vec![
        ModelCase { label: "qwen3-8b", fam: Fam::Dense, model_type: "qwen3", hidden: 4096, layers: 36, kv_heads: 8, head_dim: 128 },
        ModelCase { label: "qwen3-0.6b", fam: Fam::Dense, model_type: "qwen3", hidden: 1024, layers: 28, kv_heads: 8, head_dim: 128 },
        ModelCase { label: "llama3-70b", fam: Fam::Dense, model_type: "llama", hidden: 8192, layers: 80, kv_heads: 8, head_dim: 128 },
        ModelCase { label: "narrow", fam: Fam::Dense, model_type: "llama", hidden: 2048, layers: 24, kv_heads: 4, head_dim: 64 },
        ModelCase { label: "qwen35", fam: Fam::Qwen35, model_type: "qwen3_next", hidden: 4096, layers: 48, kv_heads: 4, head_dim: 128 },
        ModelCase { label: "qwen35moe", fam: Fam::Qwen35Moe, model_type: "qwen3_next_moe", hidden: 2048, layers: 48, kv_heads: 2, head_dim: 128 },
        ModelCase { label: "nemotron", fam: Fam::NemotronH, model_type: "nemotron_h", hidden: 4480, layers: 62, kv_heads: 8, head_dim: 128 },
        ModelCase { label: "gemma4", fam: Fam::Gemma4, model_type: "gemma4", hidden: 3840, layers: 60, kv_heads: 4, head_dim: 256 },
        ModelCase { label: "dsv4", fam: Fam::Dsv4, model_type: "deepseek_v4", hidden: 7168, layers: 61, kv_heads: 1, head_dim: 576 },
        ModelCase { label: "kimi", fam: Fam::Kimi, model_type: "kimi", hidden: 7168, layers: 61, kv_heads: 1, head_dim: 576 },
        ModelCase { label: "glm5", fam: Fam::Glm5, model_type: "glm5", hidden: 5120, layers: 92, kv_heads: 1, head_dim: 576 },
        ModelCase { label: "kimik3", fam: Fam::KimiK3, model_type: "kimi_k3", hidden: 7168, layers: 61, kv_heads: 1, head_dim: 576 },
        ModelCase { label: "kvheavy", fam: Fam::Dense, model_type: "llama", hidden: 8192, layers: 96, kv_heads: 64, head_dim: 128 },
    ]
}

fn make_case(m: &ModelCase, cfg: &mp::PlannerConfig) -> Case {
    Case {
        hf: mp::ModelShape {
            hidden_size: m.hidden,
            num_hidden_layers: m.layers,
            num_attention_heads: (m.hidden / 128).max(1),
            num_key_value_heads: m.kv_heads,
            head_dim_kernel: m.head_dim,
            model_type: m.model_type.to_owned(),
        },
        fam: m.fam,
        tp: cfg.tp_size,
        envelopes: false,
        rs_bf16: true,
        head_dim: m.head_dim,
        max_intermediate: m.hidden * 4,
        prefill_graph_capable: matches!(m.fam, Fam::Qwen35 | Fam::NemotronH),
    }
}

fn describe(p: &CudaMemoryPlan) -> String {
    let c = &p.capacity;
    format!(
        "page={} N={} R={} refs={} pgB={} attnB={} rqB={} persB={} cap=[{},{},{},{},{},{},{},{}]",
        p.kv_page_size,
        p.max_workspace_tokens,
        p.max_requests,
        p.max_page_refs,
        p.kv_page_bytes,
        p.attn_float_workspace_bytes,
        p.runtime_quant_scratch_bytes,
        p.persistent_input_bytes,
        c.max_forward_tokens,
        c.max_forward_requests,
        c.max_page_refs,
        c.max_logit_rows,
        c.max_prob_rows,
        c.max_custom_mask_bytes,
        c.max_sampler_rows,
        c.max_logprob_labels,
    )
}

struct Sweep {
    out: String,
    used_frac: f64,
    used_bytes: Option<u64>,
    calibrating: bool,
    envelopes: bool,
    rs_bf16: bool,
    profile: Option<ProfileShape>,
    complaint: String,
    last_budget: u64,
}

impl Sweep {
    fn new() -> Self {
        Self {
            out: String::from("# memory_planner oracle v1\n"),
            used_frac: 0.5,
            used_bytes: None,
            calibrating: false,
            envelopes: false,
            rs_bf16: true,
            profile: None,
            complaint: String::new(),
            last_budget: 0,
        }
    }

    /// Append a pre-formatted row, for searches whose ANSWER is the output.
    fn row(&mut self, line: &str) {
        self.out.push_str(line);
        self.out.push('\n');
    }

    /// Plan without emitting a row and report only the budget.
    ///
    /// Routed through `run` for the same reason the oracle's `probe_budget`
    /// is: a second copy of the setup could drift from the one under test.
    fn probe_budget(&mut self, d: &DeviceCase, m: &ModelCase, cfg: &mp::PlannerConfig) -> u64 {
        let saved = std::mem::take(&mut self.out);
        self.run("probe", d, m, cfg);
        self.out = saved;
        self.last_budget
    }

    fn run(&mut self, id: &str, d: &DeviceCase, m: &ModelCase, cfg: &mp::PlannerConfig) {
        let total = d.total_gib * GIB;
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "mirrors the oracle's double round-trip"
        )]
        let used = match self.used_bytes {
            Some(u) => u,
            None => (total as f64 * self.used_frac) as u64,
        };
        let mem = mp::DeviceMemory {
            free_bytes: total.saturating_sub(used),
            total_bytes: total,
        };
        let prop = mp::DeviceProps {
            name: d.name.to_owned(),
            major: d.major,
            minor: d.minor,
            sm_count: d.sms,
        };
        let mut cfg = cfg.clone();
        cfg.calibrating = self.calibrating;
        let mut case = make_case(m, &cfg);
        case.envelopes = self.envelopes;
        case.rs_bf16 = self.rs_bf16;

        let family = match m.fam {
            Fam::Qwen35 => mp::Family::Qwen35,
            Fam::Qwen35Moe => mp::Family::Qwen35Moe,
            Fam::NemotronH => mp::Family::NemotronH,
            _ => mp::Family::Generic,
        };
        let profiles = StubProfiles {
            shape: self.profile.clone(),
            complaint: self.complaint.clone(),
        };

        let ranks = i32::from(cfg.tp_size > 1 && !cfg.nccl_unique_id_hex.is_empty())
            * cfg.tp_size;
        let multirank = ranks > 1;

        let (envelopes, rs_bf16) = (self.envelopes, self.rs_bf16);
        let (shape, complaint) = (self.profile.clone(), self.complaint.clone());
        let result = if multirank {
            // The rendezvous is a real barrier; a single caller would block on
            // it forever, so the ranks are spawned exactly as the oracle does.
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..ranks)
                    .map(|_| {
                        let cfg = cfg.clone();
                        let prop = prop.clone();
                        let shape = shape.clone();
                        let complaint = complaint.clone();
                        s.spawn(move || {
                            let mut c = make_case(m, &cfg);
                            c.envelopes = envelopes;
                            c.rs_bf16 = rs_bf16;
                            let hf = c.hf.clone();
                            let p = StubProfiles { shape, complaint };
                            mp::plan(&cfg, &hf, &prop, mem, family, &c, &p)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
            })
        } else {
            let hf = case.hf.clone();
            vec![mp::plan(&cfg, &hf, &prop, mem, family, &case, &profiles)]
        };

        let render = |r: &Result<mp::Planned, mp::PlanError>| match r {
            Ok(p) => describe(&p.plan),
            Err(e) => format!("FAILED {e}"),
        };
        let mut out = render(&result[0]);
        if multirank && result.iter().any(|r| render(r) != out) {
            out = "RANK-DIVERGENCE".to_owned();
        }

        let notes = if multirank {
            "\u{1}MULTIRANK".to_owned()
        } else {
            result[0]
                .as_ref()
                .map(|p| {
                    // The C++'s emission order: selection notes first (they
                    // are written as the decision is made), then the
                    // introspection block, then the verbose summary.
                    let mut acc = p.notes.iter().fold(String::new(), |mut a, n| {
                        let _ = write!(a, "[pie-driver-cuda] {n}\u{1f}");
                        a
                    });
                    for line in p.introspection_report() {
                        let _ = write!(acc, "[pie-driver-cuda] {line}\u{1f}");
                    }
                    // The oracle passes `verbose=true`: the summary is a
                    // shipping code path and the only place the selector and
                    // the resolved profile are observable at all.
                    let _ = write!(
                        acc,
                        "[pie-driver-cuda] {}\u{1f}",
                        p.verbose_summary(&cfg, &prop, mem)
                    );
                    acc
                })
                .unwrap_or_default()
        };
        // Read the process GLOBAL, not `Planned::budget`, because that is
        // what the C++ prints -- and `set_planner_budget_bytes` is sticky: a
        // plan that throws after publishing leaves the previous boot's figure
        // visible to `planner_budget_bytes()`. Harmless in the driver (a
        // throwing planner means no boot at all, so nothing reads it), but it
        // IS the observable behaviour, so the transcript compares it.
        let budget = driver_cuda::layout::profile_cache::planner_budget_bytes();
        self.last_budget = budget;
        let _ = writeln!(self.out, "{id}|{budget}|{out}|{notes}");
    }
}

fn base_config() -> mp::PlannerConfig {
    mp::PlannerConfig {
        gpu_mem_utilization: 0.90,
        memory_profile: "auto".to_owned(),
        max_forward_tokens: 0,
        max_forward_requests: 0,
        kv_page_size: 0,
        kv_cache_dtype: "auto".to_owned(),
        tp_size: 1,
        mtp_num_drafts: 0,
        calibrating: false,
        rs_slot_mult: 1,
        nccl_unique_id_hex: String::new(),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_rust_planner_reproduces_the_cpp_transcript_byte_for_byte() {
    let devs = devices();
    let mods = models();
    let mut sw = Sweep::new();

    for d in &devs {
        for m in &mods {
            let id = format!("grid/{}/{}", d.label, m.label);
            sw.run(&id, d, m, &base_config());
        }
    }

    for profile in ["auto", "latency", "balanced", "throughput", "capacity", "bogus"] {
        for d in &devs {
            for m in [&mods[0], &mods[3], &mods[5], &mods[8], &mods[12]] {
                let mut c = base_config();
                c.memory_profile = profile.to_owned();
                let id = format!("profile/{profile}/{}/{}", d.label, m.label);
                sw.run(&id, d, m, &c);
            }
        }
    }

    for tp in [1, 2, 4, 8] {
        for d in [&devs[0], &devs[2]] {
            for m in [&mods[0], &mods[5], &mods[6], &mods[11]] {
                let mut c = base_config();
                c.tp_size = tp;
                c.nccl_unique_id_hex = format!("tp{tp}{}{}", d.label, m.label);
                let id = format!("tp/{tp}/{}/{}", d.label, m.label);
                sw.run(&id, d, m, &c);
            }
        }
    }

    for step in 0..=40 {
        let util = 0.0025_f64.mul_add(f64::from(step), 0.50);
        for m in [&mods[2], &mods[8]] {
            let mut c = base_config();
            c.gpu_mem_utilization = util;
            let id = format!("cliff/{util:.4}/{}", m.label);
            sw.run(&id, &devs[0], m, &c);
        }
    }
    for util in [0.05, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.99, 1.0] {
        for m in [&mods[0], &mods[12]] {
            let mut c = base_config();
            c.gpu_mem_utilization = util;
            let id = format!("util/{util:.2}/{}", m.label);
            sw.run(&id, &devs[0], m, &c);
        }
    }

    for used_gib in [0_u64, 8, 20, 30, 38, 40, 41, 44, 45] {
        sw.used_bytes = Some(used_gib * GIB);
        let id = format!("used/{used_gib}gib");
        sw.run(&id, &devs[0], &mods[0], &base_config());
    }
    sw.used_bytes = None;

    for n in [0_u32, 512, 2048, 8192, 12288, 65536] {
        for r in [0_u32, 32, 256, 1024] {
            let mut c = base_config();
            c.max_forward_tokens = n;
            c.max_forward_requests = r;
            let id = format!("pin/{n}/{r}");
            sw.run(&id, &devs[0], &mods[0], &c);
        }
    }
    for page in [0_u32, 16, 32, 64] {
        let mut c = base_config();
        c.kv_page_size = page;
        let id = format!("pinpage/{page}");
        sw.run(&id, &devs[0], &mods[0], &c);
    }

    for d in [&devs[0], &devs[4]] {
        for m in [&mods[0], &mods[5]] {
            sw.calibrating = true;
            let id = format!("calib/{}/{}", d.label, m.label);
            sw.run(&id, d, m, &base_config());
            sw.calibrating = false;
        }
    }
    {
        let mut c = base_config();
        c.max_forward_tokens = 1024;
        c.max_forward_requests = 32;
        sw.calibrating = true;
        sw.run("calib/pinned-ignored", &devs[0], &mods[0], &c);
        sw.calibrating = false;
    }

    {
        let cases: Vec<(&str, &str, i32, i32, i32, f64)> = vec![
            ("exact", "throughput", 16, 2048, 256, 1.0),
            ("page-only", "", 32, 0, 0, 1.0),
            ("tokens-only", "", 0, 4096, 0, 1.0),
            ("requests-only", "", 0, 0, 512, 1.0),
            ("profile-only", "latency", 0, 0, 0, 1.0),
            ("no-budget", "balanced", 16, 1024, 128, 0.0),
            ("drift-small", "balanced", 16, 1024, 128, 1.03),
            ("drift-edge", "balanced", 16, 1024, 128, 1.05),
            ("drift-over", "balanced", 16, 1024, 128, 1.20),
            ("drift-under", "balanced", 16, 1024, 128, 0.50),
            ("nomatch", "capacity", 16, 999, 0, 1.0),
            ("nomatch-profile", "nonesuch", 0, 0, 0, 1.0),
        ];
        let mut c = base_config();
        sw.run("prof/warmup", &devs[0], &mods[0], &c);
        let real_budget = sw.last_budget;
        for (label, profile, page, n, r, scale) in cases {
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "mirrors the oracle's double round-trip"
            )]
            let budget_bytes =
                if scale == 0.0 { 0 } else { (real_budget as f64 / scale) as u64 };
            sw.profile = Some(ProfileShape {
                policy_profile: profile.to_owned(),
                kv_page_size: page,
                max_forward_tokens: n,
                max_forward_requests: r,
                budget_bytes,
            });
            let id = format!("prof/{label}");
            sw.run(&id, &devs[0], &mods[0], &c);
            sw.profile = None;
        }
        sw.complaint = "schema version 9 is newer than this build".to_owned();
        sw.run("prof/complaint-only", &devs[0], &mods[0], &c);
        sw.profile = Some(ProfileShape {
            policy_profile: "throughput".to_owned(),
            kv_page_size: 16,
            max_forward_tokens: 0,
            max_forward_requests: 0,
            budget_bytes: real_budget,
        });
        sw.run("prof/complaint-and-shape", &devs[0], &mods[0], &c);
        sw.profile = None;
        sw.complaint.clear();

        sw.profile = Some(ProfileShape {
            policy_profile: "capacity".to_owned(),
            kv_page_size: 32,
            max_forward_tokens: 512,
            max_forward_requests: 0,
            budget_bytes: real_budget,
        });
        let mut named = base_config();
        named.memory_profile = "latency".to_owned();
        sw.run("prof/named-profile-ignores-cache", &devs[0], &mods[0], &named);
        sw.calibrating = true;
        sw.run("prof/calibrating-ignores-cache", &devs[0], &mods[0], &c);
        sw.calibrating = false;
        sw.profile = None;

        sw.complaint = "entry 3: max_forward_tokens is not a number".to_owned();
        sw.calibrating = true;
        sw.run("prof/calibrating-hides-complaint", &devs[0], &mods[0], &c);
        sw.calibrating = false;
        sw.complaint.clear();

        sw.profile = Some(ProfileShape {
            policy_profile: "capacity".to_owned(),
            kv_page_size: 16,
            max_forward_tokens: 999,
            max_forward_requests: 0,
            budget_bytes: real_budget,
        });
        sw.calibrating = true;
        sw.run("prof/calibrating-hides-nomatch", &devs[0], &mods[0], &c);
        sw.calibrating = false;
        sw.profile = None;

        let mut drift_util = 0.0_f64;
        let mut drift_budget = 0_u64;
        for step in 0..400 {
            let mut probe = base_config();
            probe.gpu_mem_utilization = 1e-6_f64.mul_add(-f64::from(step), 0.90);
            let b = sw.probe_budget(&devs[0], &mods[0], &probe);
            if b != 0 && b.is_multiple_of(21) {
                drift_util = probe.gpu_mem_utilization;
                drift_budget = b;
                break;
            }
        }
        let exact = drift_budget / 21 * 20;
        sw.row(&format!(
            "prof/drift-exact-scan|{drift_budget}|util={drift_util:.6} cached={exact}|-"
        ));
        if exact != 0 {
            c.gpu_mem_utilization = drift_util;
            sw.profile = Some(ProfileShape {
                policy_profile: "balanced".to_owned(),
                kv_page_size: 16,
                max_forward_tokens: 1024,
                max_forward_requests: 128,
                budget_bytes: exact,
            });
            sw.run("prof/drift-exact", &devs[0], &mods[0], &c);
            sw.profile = None;
        }
    }

    for tp in [2, 4, 8] {
        for d in [&devs[0], &devs[2], &devs[7]] {
            for m in &mods {
                let mut c = base_config();
                c.tp_size = tp;
                c.nccl_unique_id_hex.clear();
                sw.run(&format!("tpsolo/{tp}/{}/{}", d.label, m.label), d, m, &c);
                for prof in ["latency", "throughput"] {
                    let mut c2 = base_config();
                    c2.tp_size = tp;
                    c2.nccl_unique_id_hex.clear();
                    c2.memory_profile = (*prof).to_owned();
                    sw.run(
                        &format!("tpsolo/{tp}/{}/{}/{prof}", d.label, m.label),
                        d,
                        m,
                        &c2,
                    );
                }
            }
        }
    }

    for env in [false, true] {
        for bf16 in [true, false] {
            sw.envelopes = env;
            sw.rs_bf16 = bf16;
            for m in [&mods[0], &mods[4], &mods[6]] {
                let id = format!(
                    "switch/env{}/bf16{}/{}",
                    u8::from(env),
                    u8::from(bf16),
                    m.label
                );
                sw.run(&id, &devs[0], m, &base_config());
            }
        }
    }
    sw.envelopes = false;
    sw.rs_bf16 = true;

    for drafts in [0, 1, 4, 32, 64, -1] {
        for m in [&mods[0], &mods[4], &mods[5]] {
            let mut c = base_config();
            c.mtp_num_drafts = drafts;
            let id = format!("mtp/{drafts}/{}", m.label);
            sw.run(&id, &devs[0], m, &c);
        }
    }

    for mult in ["", "1", "2", "4", "8", "9", "0", "-3", "abc"] {
        // The C++ reads PIE_RS_SLOT_MULT with atoi and clamps to 1..=8; the
        // Rust takes the resolved value, so the parse is applied here.
        let parsed: i32 = mult.parse().unwrap_or(0);
        let resolved = if (1..=8).contains(&parsed) { parsed } else { 1 };
        let mut c = base_config();
        c.rs_slot_mult = resolved;
        for m in [&mods[4], &mods[6]] {
            let label = if mult.is_empty() { "unset" } else { mult };
            let id = format!("slotmult/{label}/{}", m.label);
            sw.run(&id, &devs[0], m, &c);
        }
    }

    for sms in [40, 64, 99, 100, 128] {
        for hidden in [1024, 2048, 2049, 4096] {
            let d = DeviceCase {
                label: devs[0].label,
                name: devs[0].name,
                major: devs[0].major,
                minor: devs[0].minor,
                sms,
                total_gib: devs[0].total_gib,
            };
            let m = ModelCase {
                label: mods[0].label,
                fam: mods[0].fam,
                model_type: mods[0].model_type,
                hidden,
                layers: mods[0].layers,
                kv_heads: mods[0].kv_heads,
                head_dim: mods[0].head_dim,
            };
            let id = format!("narrow/{sms}/{hidden}");
            sw.run(&id, &d, &m, &base_config());
        }
    }

    for tp in [1, 2] {
        for key in ["", "deadbeef"] {
            let mut c = base_config();
            c.tp_size = tp;
            c.nccl_unique_id_hex = key.to_owned();
            let id = format!(
                "rendezvous/tp{tp}/{}",
                if key.is_empty() { "unkeyed" } else { "keyed" }
            );
            sw.run(&id, &devs[0], &mods[0], &c);
        }
    }

    let rows = sw.out.matches('\n').count();
    if let Ok(path) = std::env::var("MP_RUST_OUT") {
        std::fs::write(path, &sw.out).expect("write transcript");
    }
    assert_eq!(rows, GOLDEN_ROWS, "row count moved: the sweep itself changed");
    assert_eq!(
        fnv1a64(sw.out.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript diverged from the C++. Set MP_RUST_OUT=/tmp/rust.txt and \
         MP_ORACLE_OUT=/tmp/cpp.txt, run tests/oracle/memory_planner/run.sh, and diff."
    );
}
