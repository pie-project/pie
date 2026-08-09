//! Behavioural parity with the C++ llama_like config/plan-state surface —
//! slice A of gate-plan-state.
//!
//! The oracle in `tests/oracle/llama_like_cfg/` compiles the REAL
//! `model/llama_like/llama_like.cpp` (the whole TU; `--gc-sections`
//! discards the forward body after type-checking) and drives its host-pure
//! surface: struct defaults, the rope mapping, the fused-post env gate, and
//! the three graph-layout functions over a 576-point plan-state grid. This
//! test replays the same sweep against the port and requires the
//! transcripts to be byte-identical.
//!
//! Run `tests/oracle/llama_like_cfg/run.sh` to regenerate
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash, never this
//! file's.
//!
//! Floats are recorded as IEEE bit patterns on both sides — a golden that
//! compared `%g` to `{}` would be a claim about formatting, not values.
//!
//! The plan caches are opaque on both sides; each side's recorder returns
//! the layout value the driver stored, with the SAME stored values on the
//! same grid walk, so the golden is about the driver's branch structure and
//! mixing and cannot be satisfied by an accidentally-agreeing flashinfer.

use std::fmt::Write as _;

use driver_cuda_new::model::config::{HfConfig, RopeScaling};
use driver_cuda_new::model::llama_like::{
    LlamaLikeForwardCfg, LlamaLikePlanState, PlanLayouts, apply_rope_config,
    decode_fused_post_enabled_from, llama_like_decode_graph_layout,
    llama_like_prefill_graph_capturable, llama_like_supergraph_graph_layout,
    rope_kind_from_hf_config,
};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x71ab439f9f674881;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 648;

const SEP: char = '\u{1f}';

/// The oracle's plan-cache recorder, reproduced.
struct FakePlans;

/// A decode plan that answers with what the grid stored into it.
struct FakeDecodePlan {
    layout: u32,
}

/// A prefill plan likewise, plus the capturable bit.
struct FakePrefillPlan {
    layout: u32,
    capturable: bool,
}

impl PlanLayouts for FakePlans {
    type DecodePlan = FakeDecodePlan;
    type PrefillPlan = FakePrefillPlan;

    fn decode_plan_graph_layout(&self, plan: &FakeDecodePlan) -> u32 {
        plan.layout
    }
    fn prefill_plan_graph_layout(&self, plan: &FakePrefillPlan) -> u32 {
        plan.layout
    }
    fn prefill_plan_graph_capturable(&self, plan: &FakePrefillPlan) -> bool {
        plan.capturable
    }
    fn xqa_decode_graph_layout(&self, max_pages_per_seq: i32) -> u8 {
        (0xE0u32 ^ (max_pages_per_seq as u32).wrapping_mul(7)) as u8
    }
}

type State = LlamaLikePlanState<FakeDecodePlan, FakePrefillPlan>;

fn kv(out: &mut String, section: &str, field: &str, value: impl std::fmt::Display) {
    writeln!(out, "{section}{SEP}{field}{SEP}{value}").unwrap();
}

fn sweep_cfg_defaults(out: &mut String) {
    let c = LlamaLikeForwardCfg::default();
    let s = "cfg-default";
    kv(out, s, "use_qk_norm", u8::from(c.use_qk_norm));
    kv(out, s, "use_qkv_bias", u8::from(c.use_qkv_bias));
    kv(out, s, "norm_placement", c.norm_placement as i32);
    kv(out, s, "rope_kind", c.rope_kind as i32);
    kv(out, s, "yarn_factor", c.yarn_factor.to_bits());
    kv(out, s, "yarn_low_freq_factor", c.yarn_low_freq_factor.to_bits());
    kv(out, s, "yarn_high_freq_factor", c.yarn_high_freq_factor.to_bits());
    kv(out, s, "yarn_original_max_position", c.yarn_original_max_position);
    kv(out, s, "yarn_beta_fast", c.yarn_beta_fast.to_bits());
    kv(out, s, "yarn_beta_slow", c.yarn_beta_slow.to_bits());
    kv(out, s, "yarn_attention_factor", c.yarn_attention_factor.to_bits());
    kv(out, s, "sliding_window", c.sliding_window);
    kv(out, s, "per_layer_window_left.len", c.per_layer_window_left.len());
    kv(out, s, "force_prefill_path", u8::from(c.force_prefill_path));
    kv(out, s, "use_xqa_decode", u8::from(c.use_xqa_decode));
    kv(out, s, "decode_plan_cuda_graph", u8::from(c.decode_plan_cuda_graph));
    kv(out, s, "use_prefill_decode_plan", u8::from(c.use_prefill_decode_plan));
    kv(
        out,
        s,
        "prefill_decode_full_attention_min_requests",
        c.prefill_decode_full_attention_min_requests,
    );
    kv(
        out,
        s,
        "prefill_decode_full_attention_min_kv_pages",
        c.prefill_decode_full_attention_min_kv_pages,
    );
    kv(out, s, "prefill_decode_min_kv_pages", c.prefill_decode_min_kv_pages);
    kv(out, s, "tp_size", c.tp_size);
    kv(out, s, "tp_comm_null", u8::from(c.tp_comm.is_null()));
    kv(out, s, "emit_logits", u8::from(c.emit_logits));
    kv(out, s, "logits_argmax_chunk_tokens", c.logits_argmax_chunk_tokens);
    kv(out, s, "mrope_section_t", c.mrope_section_t);
    kv(out, s, "mrope_section_h", c.mrope_section_h);
    kv(out, s, "mrope_section_w", c.mrope_section_w);
}

fn sweep_state_defaults(out: &mut String) {
    let st = State::default();
    let s = "state-default";
    kv(out, s, "decode_plan_null", u8::from(st.decode_plan.is_none()));
    kv(out, s, "prefill_plan_null", u8::from(st.prefill_plan.is_none()));
    kv(out, s, "prefill_decode_plan_null", u8::from(st.prefill_decode_plan.is_none()));
    kv(out, s, "mask_decode_plan_null", u8::from(st.mask_decode_plan.is_none()));
    kv(
        out,
        s,
        "depth_prefix_decode_plan_null",
        u8::from(st.depth_prefix_decode_plan.is_none()),
    );
    kv(out, s, "depth_band_plans.len", st.depth_band_plans.len());
    for (i, p) in st.depth_band_plans.iter().enumerate() {
        kv(out, s, &format!("depth_band_plans_null.{i}"), u8::from(p.is_none()));
    }
    kv(out, s, "depth_band_prefill_plans.len", st.depth_band_prefill_plans.len());
    for (i, p) in st.depth_band_prefill_plans.iter().enumerate() {
        kv(
            out,
            s,
            &format!("depth_band_prefill_plans_null.{i}"),
            u8::from(p.is_none()),
        );
    }
    for (i, v) in st.depth_band_k.iter().enumerate() {
        kv(out, s, &format!("depth_band_k.{i}"), v);
    }
    for (i, v) in st.depth_band_rows.iter().enumerate() {
        kv(out, s, &format!("depth_band_rows.{i}"), v);
    }
    kv(out, s, "depth_band_count", st.depth_band_count);
    kv(out, s, "mixed_mid_decode_plan_null", u8::from(st.mixed_mid_decode_plan.is_none()));
    kv(out, s, "mixed_mid_start", st.mixed_mid_start);
    kv(out, s, "spatial_mask_split", st.spatial_mask_split);
    kv(out, s, "spatial_mask_row_split", st.spatial_mask_row_split);
    kv(out, s, "use_prefill_plan", u8::from(st.use_prefill_plan));
    kv(out, s, "use_prefill_decode_plan", u8::from(st.use_prefill_decode_plan));
    kv(out, s, "use_mask_decode_plan", u8::from(st.use_mask_decode_plan));
    kv(out, s, "prefill_score_window", st.prefill_score_window);
    kv(out, s, "lora_staged_null", u8::from(st.lora_staged.is_none()));
    kv(out, s, "lora_staged_table_null", u8::from(st.lora_staged_table.is_null()));
    kv(out, s, "use_xqa_decode", u8::from(st.use_xqa_decode));
    kv(out, s, "xqa_max_pages_per_seq", st.xqa_max_pages_per_seq);
    kv(out, s, "prefill_decode_qo_indptr_h.len", st.prefill_decode_qo_indptr_h.len());
}

fn sweep_rope(out: &mut String) {
    for kind in [RopeScaling::None, RopeScaling::Llama3, RopeScaling::OriginalYarn] {
        let hf = HfConfig { rope_scaling_kind: kind, ..HfConfig::default() };
        writeln!(
            out,
            "rope-kind{SEP}{}{SEP}{}",
            kind as i32,
            rope_kind_from_hf_config(&hf) as i32
        )
        .unwrap();
    }

    // The oracle's grid, value for value.
    struct Block {
        kind: RopeScaling,
        factor: f32,
        low: f32,
        high: f32,
        omp: i32,
        bfast: f32,
        bslow: f32,
        afactor: f32,
    }
    let grid = [
        Block {
            kind: RopeScaling::None,
            factor: 1.0,
            low: 1.0,
            high: 4.0,
            omp: 8192,
            bfast: 32.0,
            bslow: 1.0,
            afactor: 1.0,
        },
        Block {
            kind: RopeScaling::Llama3,
            factor: 32.0,
            low: 0.001953125,
            high: 0.0078125,
            omp: 16,
            bfast: 16.0,
            bslow: 2.0,
            afactor: 0.75,
        },
        Block {
            kind: RopeScaling::OriginalYarn,
            factor: 2.5,
            low: -1.5,
            high: 3.25,
            omp: 4096,
            bfast: 48.0,
            bslow: 0.5,
            afactor: 1.25,
        },
        Block {
            kind: RopeScaling::Llama3,
            factor: 8.0,
            low: 1.0,
            high: 4.0,
            omp: 32768,
            bfast: 32.0,
            bslow: 1.0,
            afactor: -2.0,
        },
    ];
    for (i, b) in grid.iter().enumerate() {
        let hf = HfConfig {
            rope_scaling_kind: b.kind,
            rope_factor: b.factor,
            rope_low_freq_factor: b.low,
            rope_high_freq_factor: b.high,
            rope_original_max_position: b.omp,
            rope_beta_fast: b.bfast,
            rope_beta_slow: b.bslow,
            rope_attention_factor: b.afactor,
            ..HfConfig::default()
        };
        let mut cfg = LlamaLikeForwardCfg::default();
        apply_rope_config(&mut cfg, &hf);
        writeln!(
            out,
            "apply-rope{SEP}{i}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
            cfg.rope_kind as i32,
            cfg.yarn_factor.to_bits(),
            cfg.yarn_low_freq_factor.to_bits(),
            cfg.yarn_high_freq_factor.to_bits(),
            cfg.yarn_original_max_position,
            cfg.yarn_beta_fast.to_bits(),
            cfg.yarn_beta_slow.to_bits(),
            cfg.yarn_attention_factor.to_bits(),
        )
        .unwrap();
    }
}

fn sweep_layouts(out: &mut String) {
    let ops = FakePlans;
    let mut i: i32 = 0;
    for sm in [-1, 0, 2] {
        for use_mask in 0..2 {
            for mask_present in 0..2 {
                for decode_present in 0..2 {
                    for xqa in 0..2 {
                        for pd in 0..3 {
                            for pf in 0..4 {
                                let st = State {
                                    spatial_mask_split: sm,
                                    use_mask_decode_plan: use_mask != 0,
                                    mask_decode_plan: (mask_present != 0).then(|| {
                                        FakePrefillPlan {
                                            layout: 0x70 + (i as u32 % 7),
                                            capturable: false,
                                        }
                                    }),
                                    decode_plan: (decode_present != 0).then(|| {
                                        FakeDecodePlan { layout: 0x10 + (i as u32 % 5) }
                                    }),
                                    use_xqa_decode: xqa != 0,
                                    xqa_max_pages_per_seq: if xqa != 0 {
                                        3 + (i % 4)
                                    } else {
                                        0
                                    },
                                    use_prefill_decode_plan: pd != 0,
                                    prefill_decode_plan: (pd == 2).then(|| {
                                        FakePrefillPlan {
                                            layout: 0x50 + (i as u32 % 3),
                                            capturable: false,
                                        }
                                    }),
                                    use_prefill_plan: pf != 0,
                                    prefill_plan: (pf >= 2).then(|| FakePrefillPlan {
                                        layout: 0x30 + (i as u32 % 9),
                                        capturable: pf == 3,
                                    }),
                                    ..State::default()
                                };

                                writeln!(
                                    out,
                                    "layout{SEP}{i}{SEP}{sm}{SEP}{use_mask}{SEP}\
                                     {mask_present}{SEP}{decode_present}{SEP}{xqa}{SEP}\
                                     {}{SEP}{pd}{SEP}{pf}{SEP}{}{SEP}{}{SEP}{}",
                                    st.xqa_max_pages_per_seq,
                                    llama_like_decode_graph_layout(&ops, &st),
                                    llama_like_supergraph_graph_layout(&ops, &st),
                                    i32::from(llama_like_prefill_graph_capturable(&ops, &st)),
                                )
                                .unwrap();
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn sweep_fused_post(out: &mut String) {
    use std::ffi::OsStr;
    let axis: [(&str, Option<&OsStr>); 5] = [
        ("unset", None),
        ("empty", Some(OsStr::new(""))),
        ("zero", Some(OsStr::new("0"))),
        ("one", Some(OsStr::new("1"))),
        ("other", Some(OsStr::new("x"))),
    ];
    for (label, value) in axis {
        writeln!(
            out,
            "fused_post{SEP}{label}{SEP}{}",
            i32::from(decode_fused_post_enabled_from(value))
        )
        .unwrap();
    }
}

fn transcript() -> String {
    let mut out = String::new();
    sweep_cfg_defaults(&mut out);
    sweep_state_defaults(&mut out);
    sweep_rope(&mut out);
    sweep_layouts(&mut out);
    sweep_fused_post(&mut out);
    out
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_port_reproduces_the_cpp_transcript() {
    let text = transcript();
    let rows = text.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — sweep shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        // The transcript is large; on mismatch, dump it for diffing against
        // `LLC_ORACLE_OUT=... tests/oracle/llama_like_cfg/run.sh`.
        let path = std::env::temp_dir().join("llama_like_cfg_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}

/// The cached env-reading form agrees with the pure form for THIS process's
/// environment — the one observation a single process can make.
#[test]
fn the_cached_gate_agrees_with_the_pure_form() {
    let expected = decode_fused_post_enabled_from(
        std::env::var_os("PIE_CUDA_DECODE_FUSED_POST").as_deref(),
    );
    assert_eq!(
        driver_cuda_new::model::llama_like::decode_fused_post_enabled(),
        expected
    );
}
