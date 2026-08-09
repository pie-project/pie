//! Behavioural parity with the C++ llama-like weight binders.
//!
//! The oracle in `tests/oracle/weight_bind/` compiles the real
//! `driver-cuda/csrc/src/model/llama_like/qwen3.cpp` — unmodified — against a
//! stub `LoadedModel`, and prints every slot of every layer plus the order in
//! which names were probed. This reproduces both.
//!
//! Run `tests/oracle/weight_bind/run.sh` to regenerate [`GOLDEN_FNV1A64`].
//!
//! # Why the transcript records nulls, and probe order
//!
//! A null slot here is not an absence, it is a decision: the forward path
//! reads a missing `q_norm` as "this architecture has no per-head norm" and
//! skips an RMSNorm, reads a missing `qkv_proj_fused` as "the loader declined
//! to fuse" and takes three narrow GEMMs instead of one wide one. Nothing
//! downstream checks. A binder that filled a slot it should have left empty
//! produces a model that loads, runs, and is quietly wrong — so every slot of
//! every layer is in the transcript, filled or not.
//!
//! Probe order is in there for a narrower reason. When a name is present under
//! two spellings, the order of the lookups is what decides which one binds.
//! OLMo-3 is exactly that case: `post_attention_layernorm` exists in both
//! schemas and means different things, so the oracle populates *both*
//! spellings and the transcript shows which one won.

use std::fmt::Write as _;

use driver_cuda_new::model::weight_bind::{
    BindConfig, BindError, Qwen3Weights, WeightSource, bind_llama_like, bind_olmo3, bind_phi3,
};
use driver_cuda_new::model::weight_view::{QuantKind, QuantMeta};

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0x123b_43e1_51d9_afbf;

/// Rows the transcript must contain.
const GOLDEN_ROWS: usize = 891;

const SEP: char = '\u{1f}';

/// The stub `LoadedModel`: a name set, a probe log, and a quant side-map.
#[derive(Default)]
struct Engine {
    cfg: BindConfig,
    names: Vec<String>,
    quant: Vec<(String, QuantMeta)>,
    probes: std::cell::RefCell<Vec<String>>,
}

impl Engine {
    fn add(&mut self, name: impl Into<String>) {
        self.names.push(name.into());
    }

    fn add_quant(&mut self, name: impl Into<String>, m: QuantMeta) {
        self.quant.push((name.into(), m));
    }

    /// Resolve a bound handle back to the name it came from. The C++ side does
    /// the same with a pointer→name map, so the transcript is about the
    /// binding rather than about an allocator.
    fn name_of(&self, h: Option<usize>) -> &str {
        match h {
            None => "null",
            Some(i) => self.names.get(i).map_or("unknown", String::as_str),
        }
    }
}

impl WeightSource for Engine {
    type Handle = usize;

    fn config(&self) -> BindConfig {
        self.cfg
    }

    fn get(&self, name: &str) -> Option<usize> {
        self.probes.borrow_mut().push(name.to_owned());
        self.names.iter().position(|n| n == name)
    }

    fn quant_meta(&self, name: &str) -> Option<QuantMeta> {
        self.quant.iter().find(|(n, _)| n == name).map(|(_, m)| *m)
    }
}

fn layer_prefix(i: i32) -> String {
    format!("model.layers.{i}.")
}

fn populate_llama_like(
    e: &mut Engine,
    layers: i32,
    lm_head: bool,
    bias: bool,
    qk_norm: bool,
    qkv_fused: bool,
    gate_up_fused: bool,
) {
    e.cfg.num_hidden_layers = layers;
    e.add("model.embed_tokens.weight");
    e.add("model.norm.weight");
    if lm_head {
        e.add("lm_head.weight");
    }
    for i in 0..layers {
        let p = layer_prefix(i);
        for s in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            e.add(format!("{p}{s}"));
        }
        if bias {
            for w in ["q", "k", "v"] {
                e.add(format!("{p}self_attn.{w}_proj.bias"));
            }
        }
        if qk_norm {
            for w in ["q", "k"] {
                e.add(format!("{p}self_attn.{w}_norm.weight"));
            }
        }
        if qkv_fused {
            e.add(format!("{p}self_attn.qkv_proj.fused.weight"));
        }
        if gate_up_fused {
            e.add(format!("{p}mlp.gate_up_proj.fused.weight"));
        }
    }
}

fn populate_olmo3(e: &mut Engine, layers: i32, lm_head: bool, bias: bool) {
    e.cfg.num_hidden_layers = layers;
    e.add("model.embed_tokens.weight");
    e.add("model.norm.weight");
    if lm_head {
        e.add("lm_head.weight");
    }
    for i in 0..layers {
        let p = layer_prefix(i);
        for s in [
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            e.add(format!("{p}{s}"));
        }
        if bias {
            for w in ["q", "k", "v"] {
                e.add(format!("{p}self_attn.{w}_proj.bias"));
            }
        }
    }
}

/// The C++ dumps the seven quant slots in declaration order; a `BTreeMap`
/// iterates alphabetically, so the order is stated here instead.
const QUANT_SLOTS: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn dump_weights(out: &mut String, label: &str, e: &Engine, w: &Qwen3Weights<usize>) {
    let top = |out: &mut String, k: &str, v: &str| {
        writeln!(out, "top{SEP}{label}{SEP}{k}{SEP}{v}").unwrap();
    };
    top(out, "embed", e.name_of(Some(w.embed)));
    top(out, "final_norm", e.name_of(Some(w.final_norm)));
    top(out, "lm_head", e.name_of(Some(w.lm_head)));
    top(
        out,
        "lm_head_aliases_embed",
        if w.lm_head_is_tied() { "1" } else { "0" },
    );
    top(out, "num_layers", &w.layers.len().to_string());

    for (i, l) in w.layers.iter().enumerate() {
        let pre = format!("layer{SEP}{label}{SEP}{i}{SEP}");
        for (name, h) in [
            ("attn_norm", Some(l.attn_norm)),
            ("mlp_norm", Some(l.mlp_norm)),
            ("q_proj", Some(l.q_proj)),
            ("k_proj", Some(l.k_proj)),
            ("v_proj", Some(l.v_proj)),
            ("o_proj", Some(l.o_proj)),
            ("q_bias", l.q_bias),
            ("k_bias", l.k_bias),
            ("v_bias", l.v_bias),
            ("q_norm", l.q_norm),
            ("k_norm", l.k_norm),
            ("gate_proj", Some(l.gate_proj)),
            ("up_proj", Some(l.up_proj)),
            ("down_proj", Some(l.down_proj)),
            ("qkv_proj_fused", l.qkv_proj_fused),
            ("gate_up_proj_fused", l.gate_up_proj_fused),
        ] {
            writeln!(out, "{pre}{name}{SEP}{}", e.name_of(h)).unwrap();
        }
        for slot in QUANT_SLOTS {
            let v = l.quant_for(slot).map_or_else(
                || "none".to_owned(),
                |q| {
                    format!(
                        "kind={},gs={},axis={}",
                        q.kind as i32, q.group_size, q.channel_axis
                    )
                },
            );
            writeln!(out, "{pre}{slot}_quant{SEP}{v}").unwrap();
        }
    }
}

fn dump_probes(out: &mut String, label: &str, e: &Engine) {
    let probes = e.probes.borrow();
    writeln!(out, "probe_count{SEP}{label}{SEP}{}", probes.len()).unwrap();
    for p in probes.iter() {
        if p.starts_with("model.layers.") && !p.starts_with("model.layers.0.") {
            continue;
        }
        writeln!(out, "probe{SEP}{label}{SEP}{p}").unwrap();
    }
}

type Binder = fn(&Engine) -> Result<Qwen3Weights<usize>, BindError>;

fn run(out: &mut String, label: &str, e: &Engine, bind: Binder) {
    match bind(e) {
        Ok(w) => {
            dump_weights(out, label, e, &w);
            dump_probes(out, label, e);
        }
        Err(err) => {
            writeln!(out, "throw{SEP}{label}{SEP}{}", err.cpp_message()).unwrap();
            dump_probes(out, label, e);
        }
    }
}

/// Script 1 — the config grid, each flag varied independently.
fn script_config_grid(out: &mut String) {
    struct Case {
        label: &'static str,
        have_lm_head: bool,
        tie: bool,
        bias: bool,
        qk_norm: bool,
    }
    let cases = [
        Case { label: "base", have_lm_head: true, tie: false, bias: false, qk_norm: false },
        Case { label: "tied_no_head", have_lm_head: false, tie: true, bias: false, qk_norm: false },
        Case { label: "untied_no_head", have_lm_head: false, tie: false, bias: false, qk_norm: false },
        Case { label: "head_and_tie", have_lm_head: true, tie: true, bias: false, qk_norm: false },
        Case { label: "bias", have_lm_head: true, tie: false, bias: true, qk_norm: false },
        Case { label: "qk_norm", have_lm_head: true, tie: false, bias: false, qk_norm: true },
        Case { label: "bias_and_qk_norm", have_lm_head: true, tie: false, bias: true, qk_norm: true },
        Case {
            label: "qk_norm_flag_without_tensors",
            have_lm_head: true,
            tie: false,
            bias: false,
            qk_norm: false,
        },
    ];
    for c in cases {
        let mut e = Engine::default();
        e.cfg.tie_word_embeddings = c.tie;
        e.cfg.attention_bias = c.bias;
        e.cfg.use_qk_norm = c.qk_norm;
        let last = c.label == "qk_norm_flag_without_tensors";
        populate_llama_like(&mut e, 2, c.have_lm_head, c.bias, c.qk_norm, false, false);
        if last {
            e.cfg.use_qk_norm = true;
        }
        run(out, c.label, &e, bind_llama_like);
    }
}

/// Script 2 — the fused-projection slots.
fn script_fusion(out: &mut String) {
    for (label, qkv, gate_up) in [
        ("fused_neither", false, false),
        ("fused_qkv_only", true, false),
        ("fused_gate_up_only", false, true),
        ("fused_both", true, true),
    ] {
        let mut e = Engine::default();
        populate_llama_like(&mut e, 1, true, false, false, qkv, gate_up);
        run(out, label, &e, bind_llama_like);
    }
}

/// Script 3 — the quant side-map, keyed per projection.
fn script_quant_sidemap(out: &mut String) {
    let mut e = Engine::default();
    populate_llama_like(&mut e, 1, true, false, false, false, false);
    for (suffix, tag) in [
        ("self_attn.q_proj.weight", 11),
        ("self_attn.k_proj.weight", 12),
        ("self_attn.v_proj.weight", 13),
        ("self_attn.o_proj.weight", 14),
        ("mlp.gate_proj.weight", 15),
        ("mlp.up_proj.weight", 16),
        ("mlp.down_proj.weight", 17),
    ] {
        e.add_quant(
            format!("{}{suffix}", layer_prefix(0)),
            QuantMeta {
                kind: QuantKind::PerGroup,
                group_size: tag,
                channel_axis: tag % 2,
                ..QuantMeta::default()
            },
        );
    }
    run(out, "quant_all", &e, bind_llama_like);

    let mut e2 = Engine::default();
    populate_llama_like(&mut e2, 1, true, false, false, false, false);
    let m = QuantMeta {
        kind: QuantKind::PerChannel,
        group_size: 0,
        channel_axis: 1,
        ..QuantMeta::default()
    };
    e2.add_quant(format!("{}self_attn.q_proj.weight", layer_prefix(0)), m);
    e2.add_quant(format!("{}mlp.down_proj.weight", layer_prefix(0)), m);
    run(out, "quant_partial", &e2, bind_llama_like);
}

/// Script 4 — the architecture variants.
fn script_variants(out: &mut String) {
    {
        let mut e = Engine::default();
        e.cfg.use_qk_norm = true;
        populate_olmo3(&mut e, 2, true, false);
        // Both norm spellings present, so the transcript records a choice.
        e.add(format!("{}input_layernorm.weight", layer_prefix(0)));
        e.add(format!("{}input_layernorm.weight", layer_prefix(1)));
        run(out, "olmo3", &e, bind_olmo3);
    }
    {
        let mut e = Engine::default();
        e.cfg.attention_bias = true;
        populate_olmo3(&mut e, 1, false, true);
        e.cfg.tie_word_embeddings = true;
        run(out, "olmo3_tied_bias", &e, bind_olmo3);
    }
    {
        let mut e = Engine::default();
        populate_llama_like(&mut e, 2, true, false, false, false, false);
        run(out, "phi3", &e, bind_phi3);
    }
    {
        let mut e = Engine::default();
        e.cfg.num_hidden_layers = 1;
        for n in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            e.add(n);
        }
        e.add(format!("{}mlp.gate_proj.weight", layer_prefix(0)));
        e.add(format!("{}mlp.up_proj.weight", layer_prefix(0)));
        run(out, "phi3_missing_qkv", &e, bind_phi3);
    }
    {
        let mut e = Engine::default();
        e.cfg.num_hidden_layers = 1;
        for n in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            e.add(n);
        }
        for w in ["q", "k", "v"] {
            e.add(format!("{}self_attn.{w}_proj.weight", layer_prefix(0)));
        }
        run(out, "phi3_missing_gate_up", &e, bind_phi3);
    }
}

/// Script 5 — the degenerate shapes.
fn script_degenerate(out: &mut String) {
    let mut e = Engine::default();
    populate_llama_like(&mut e, 0, true, false, false, false, false);
    run(out, "zero_layers", &e, bind_llama_like);

    let mut e2 = Engine::default();
    e2.cfg.num_hidden_layers = 1;
    e2.add("model.norm.weight");
    e2.add("lm_head.weight");
    run(out, "missing_embed", &e2, bind_llama_like);

    let mut e3 = Engine::default();
    e3.cfg.num_hidden_layers = 1;
    e3.add("model.embed_tokens.weight");
    e3.add("lm_head.weight");
    run(out, "missing_final_norm", &e3, bind_llama_like);
}

fn transcript() -> String {
    let mut out = String::new();
    script_config_grid(&mut out);
    script_fusion(&mut out);
    script_quant_sidemap(&mut out);
    script_variants(&mut out);
    script_degenerate(&mut out);
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Diagnostic: write the Rust transcript for a line-by-line diff.
#[test]
#[ignore = "diagnostic"]
fn dump_transcript() {
    let path = std::env::var("WB_RUST_OUT").unwrap_or_else(|_| "/tmp/wb_rust.txt".into());
    std::fs::write(&path, transcript()).unwrap();
    eprintln!("wrote {path}");
}

#[test]
fn the_rust_binders_reproduce_the_cpp_transcript() {
    let t = transcript();
    assert_eq!(
        t.lines().count(),
        GOLDEN_ROWS,
        "transcript row count drifted from the C++ oracle"
    );
    assert_eq!(
        fnv1a64(t.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript differs from the C++ oracle; \
         run tests/oracle/weight_bind/run.sh with WB_ORACLE_OUT set to diff them"
    );
}

/// Every optional slot, stated as a table the hash cannot spell out.
///
/// The hash says "a slot changed"; this says which flag was supposed to
/// control it. The three flags are set one at a time so that a port which
/// wired two together — reading `use_qk_norm` where it meant `attention_bias`,
/// say — fails here with the pair named.
#[test]
fn each_flag_controls_exactly_its_own_slots() {
    let build = |bias: bool, qk: bool| {
        let mut e = Engine::default();
        e.cfg.attention_bias = bias;
        e.cfg.use_qk_norm = qk;
        populate_llama_like(&mut e, 1, true, bias, qk, false, false);
        let mut w = bind_llama_like(&e).unwrap();
        w.layers.remove(0)
    };

    for (bias, qk) in [(false, false), (true, false), (false, true), (true, true)] {
        let l = build(bias, qk);
        assert_eq!(l.q_bias.is_some(), bias, "q_bias with bias={bias} qk={qk}");
        assert_eq!(l.k_bias.is_some(), bias, "k_bias with bias={bias} qk={qk}");
        assert_eq!(l.v_bias.is_some(), bias, "v_bias with bias={bias} qk={qk}");
        assert_eq!(l.q_norm.is_some(), qk, "q_norm with bias={bias} qk={qk}");
        assert_eq!(l.k_norm.is_some(), qk, "k_norm with bias={bias} qk={qk}");
    }
}

/// The fused slots follow the tensors, not a config flag.
///
/// Separate from the flag test above because the mechanism is different: these
/// are probed, and their absence is the loader's contract declining to fuse a
/// group rather than the architecture lacking the weights.
#[test]
fn the_fused_slots_are_decided_by_presence_not_by_config() {
    for (qkv, gate_up) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut e = Engine::default();
        // Every flag off: nothing in the config mentions fusion.
        populate_llama_like(&mut e, 1, true, false, false, qkv, gate_up);
        let f = bind_llama_like(&e).unwrap().layers[0].fusion();
        assert_eq!(f.qkv, qkv, "qkv fusion for ({qkv}, {gate_up})");
        assert_eq!(f.gate_up, gate_up, "gate_up fusion for ({qkv}, {gate_up})");
    }
}
