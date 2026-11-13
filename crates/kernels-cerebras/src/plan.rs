//! The logical plan: what the model is ([`Shape`]), how it is cut ([`Partition`]) and what the
//! build promises at run time ([`Contract`]). No colours or kernel names: the spatial pass reads
//! the [`Shape`] off the model's trace ([`super::target`]), and the partition of the layout its
//! layers take turns it and a [`Preset`] into the plan ([`super::partition::plan`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The model, as the wafer program needs it: the model's own facts, read off its trace, the same
/// for every partition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shape {
    /// Hidden size of the checkpoint (norms divide by this).
    pub hidden: u32,
    /// Vocabulary of the checkpoint (the head pads past it with copies of its last real row).
    pub vocab: u32,
    pub layers: Vec<LayerKind>,
    pub attn: AttnSpec,
    pub mlp: MlpSpec,
    pub moe: Option<MoeSpec>,
    /// Qwen3.5's gated delta net (`None`: no GDN layer).
    pub gdn: Option<GdnSpec>,
    /// Gemma's per-layer inputs.
    pub ple: Option<PleSpec>,
    pub head: HeadSpec,
    pub rope: RopeSpec,
    pub norm: NormSpec,
}

impl Shape {
    /// What tells one model's partition from another's within a layout: the hidden size, the
    /// layer count and the MLP's width (a pinned plan's made-for shape).
    #[must_use]
    pub fn signature(&self) -> [u32; 3] {
        [self.hidden, self.layers.len() as u32, self.mlp.intermediate]
    }
}

/// One decoder layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// Qwen3.5's gated delta net.
    Gdn,
    /// Full causal attention.
    Global,
    /// Sliding-window attention over the last `window` positions.
    Sliding { window: u32 },
    /// Attention reading another layer's KV (Gemma's shared KV): that layer's index.
    SharedKv { from: u32, sliding: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct AttnSpec {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// The global (full-attention) layers' head dim where it differs from the sliding layers'
    /// `head_dim` (Gemma 4: twice it).
    #[serde(default)]
    pub global_head_dim: Option<u32>,
    /// The global layers' KV heads where they differ from the sliding layers' `kv_heads`
    /// (Gemma 4 12B: one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_kv_heads: Option<u32>,
    /// Qwen3.5's sigmoid output gate.
    pub output_gate: bool,
    /// RMSNorm over each q / k head before RoPE.
    pub qk_norm: bool,
    /// gpt-oss's attention sinks.
    pub sinks: bool,
    /// Biases on the q / k / v projections.
    #[serde(default, skip_serializing_if = "is_false")]
    pub bias: bool,
    /// A bias on the output projection.
    #[serde(default, skip_serializing_if = "is_false")]
    pub o_bias: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Act {
    Silu,
    GeluTanh,
    /// gpt-oss's clamped swiglu.
    SwigluOai,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlpSpec {
    pub intermediate: u32,
    pub act: Act,
    /// Each layer's MLP width where they differ (Gemma 4's KV-sharing layers are wider); empty:
    /// every layer's is `intermediate`.
    #[serde(default)]
    pub layer_widths: Vec<u32>,
}

impl MlpSpec {
    /// Layer `l`'s MLP width.
    pub fn width(&self, l: u32) -> u32 {
        self.layer_widths
            .get(l as usize)
            .copied()
            .unwrap_or(self.intermediate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeSpec {
    pub experts: u32,
    pub top_k: u32,
    pub expert_intermediate: u32,
    /// A bias on the router's logits.
    #[serde(default, skip_serializing_if = "is_false")]
    pub router_bias: bool,
    /// Biases on the experts' gate / up and down legs.
    #[serde(default, skip_serializing_if = "is_false")]
    pub expert_bias: bool,
    /// The experts' banks quantized (gpt-oss's MXFP4); else 16-bit.
    #[serde(default, skip_serializing_if = "is_false")]
    pub quantized: bool,
}

/// Qwen3.5's gated delta net: its key / value heads and their dims (`linear_*` of the config).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GdnSpec {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PleSpec {
    /// Per-layer input width.
    pub width: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct HeadSpec {
    /// The head reads the embedding table.
    pub tied: bool,
    pub final_softcap: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f32,
    /// Rotated dims of a head (partial RoPE < head_dim).
    pub rotary_dims: u32,
    /// Each rotated pair's inverse frequency where it is not `theta`'s own (a scaled rotation:
    /// Llama 3's long waves stretched); `None`: `theta ^ (-i / span)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freqs: Option<Vec<f32>>,
    /// Pair i turns at `theta ^ (-i / span)`; `None`: `rotary_dims / 2` (a proportional
    /// rotation's is `head_dim / 2`: Gemma 4's global layers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<u32>,
    /// YaRN's stretch of the frequencies, its attention factor on cos and sin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yarn: Option<YarnSpec>,
    /// The global layers' rotation where it is not this one (Gemma 4's sliding / global).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global: Option<Box<RopeSpec>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct YarnSpec {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original: u32,
}

impl RopeSpec {
    /// Pair `i`'s inverse frequency as `theta`, `span` and YaRN state it (the trace's own
    /// frequencies aside). YaRN's ramp runs between its correction dims unrounded (HF's
    /// `truncate: false`, what gpt-oss's checkpoint states).
    pub fn inv_freq(&self, head_dim: u32) -> Vec<f64> {
        let theta = f64::from(self.theta);
        let pairs = (self.rotary_dims / 2) as usize;
        let span = f64::from(self.span.unwrap_or(self.rotary_dims / 2));
        match &self.yarn {
            None => (0..pairs)
                .map(|i| 1.0 / theta.powf(i as f64 / span))
                .collect(),
            Some(y) => {
                let d = f64::from(head_dim);
                let factor = f64::from(y.factor);
                let pos: Vec<f64> = (0..pairs).map(|i| theta.powf((2 * i) as f64 / d)).collect();
                let corr = |rot: f64| {
                    (d * (f64::from(y.original) / (rot * 2.0 * std::f64::consts::PI)).ln())
                        / (2.0 * theta.ln())
                };
                let (lo, mut hi) = (
                    corr(f64::from(y.beta_fast)).max(0.0),
                    corr(f64::from(y.beta_slow)).min(d - 1.0),
                );
                if lo == hi {
                    hi += 0.001;
                }
                pos.iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let ramp = ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
                        let ext = 1.0 - ramp;
                        (1.0 / (factor * p)) * (1.0 - ext) + (1.0 / p) * ext
                    })
                    .collect()
            }
        }
    }

    /// YaRN's attention factor on cos and sin (1: none).
    pub fn scale(&self) -> f64 {
        self.yarn
            .map_or(1.0, |y| 0.1 * f64::from(y.factor).ln() + 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct NormSpec {
    pub eps: f32,
    /// Gemma's (1 + w) gammas.
    pub unit_offset: bool,
    /// Gemma's sandwich norms.
    pub sandwich: bool,
}

/// How the model is cut over the wafer's units and PEs, in logical terms: counts and split
/// factors, never positions. Every field is a choice the spatial compiler will search over.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    /// Units (`UNX * UNY` minus gap slots today) and the layers each runs (`K`).
    pub units: u32,
    pub layers_per_unit: u32,
    /// The unit slots the units are visited over, (columns, rows) (`UNX`, `UNY`); a count of
    /// units short of the slots leaves gap slots (the snake's rule places them). `None`: the
    /// layout places its units otherwise (explicit origins).
    pub unit_grid: Option<(u32, u32)>,
    /// Weight PEs a unit row / column: the hidden dim's split (`PY`, rows) and the input dims'
    /// split (`PX`, columns; gpt-oss `AX + MX`).
    pub rows: u32,
    pub cols: u32,
    /// A row's dims (`Hc`); per row run where rows differ (Gemma 12B attention / plain rows).
    pub row_dims: Vec<RowRun>,
    /// Rows a unit keeps KV on (`PYW`; `rows` when every row does).
    pub kv_rows: u32,
    /// KV entries a KV row holds (`NE`), and Gemma's global / sliding entry counts (`NEG`, `NES`).
    pub kv_entries: KvEntries,
    /// Two-level reduction group sizes (`G2X`, `G2Y`, `G2H`, `G2W`; 0: single level).
    pub reduce: ReduceGroups,
    /// Segment widths on the weight rows / CY row (`SWA`, `SWG`, `SWV`; gpt-oss `SWR`).
    pub segments: Segments,
    /// gpt-oss: attention columns of a unit row (`AX`); the rest are MoE columns.
    pub attn_cols: Option<u32>,
    pub head: HeadPartition,
    /// Stacked unit columns' placement (a side input's storage above them), where the layers
    /// take that layout ([`super::compose::Layout::Stacked`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrangement: Option<Arrangement>,
}

/// Stacked unit columns, as a plan places them: unit column A at x 0 holds `a_units` top to
/// bottom, stacked from row `a_top`; unit column B, east of it, holds each of `b_units` level with
/// a column-A unit; above each column's top unit its side input's storage block. Unit index
/// order is the handoff order (unit 0 takes the embedding); the head sits east of column B. An
/// arrangement is made for one unit shape (`unit`: weight columns, rows, units) and refused for
/// any other ([`super::compose::stacked`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arrangement {
    pub unit: (u32, u32, u32),
    pub a_top: u32,
    pub a_units: Vec<u32>,
    /// (unit, the column-A unit it sits level with), top to bottom.
    pub b_units: Vec<(u32, u32)>,
    /// Column A's storage, column B's.
    pub stores: [StoreSpec; 2],
}

/// A side input's storage block: rows above its column's top unit, tokens a storage PE, the
/// bus's colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreSpec {
    pub rows: u32,
    pub tokens: u32,
    pub bus: u8,
}

/// A run of rows with the same dims and role (Gemma 12B: attention rows `r < RA`, plain rows).
/// `rows` is half-open (`[rows.0, rows.1)`); the runs of a partition tile `0 .. rows` in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowRun {
    pub rows: (u32, u32),
    pub dims: u32,
    pub attention: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvEntries {
    pub per_row: u32,
    pub global: Option<u32>,
    pub sliding: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceGroups {
    pub row: u32,
    pub col: u32,
    pub head: u32,
    pub token_loop: u32,
    /// The token loop's max-reduce in three levels (`G3W`): level-2 groups of this many
    /// `token_loop` groups, the level-2 sums chained by a third level (0: two levels, the
    /// chain). Same winner either way: the merge is a total order (ties to the lower id).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub token_loop3: u32,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Segments {
    pub scores: u32,
    pub gdn: u32,
    pub onorm: u32,
    pub router: Option<u32>,
}

/// The head's cut: weight columns (`HC`), rows (`HR`), vocabulary rows a head PE (`VR`), and
/// Gemma 12B's head groups (CX rows feeding one head column).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct HeadPartition {
    pub cols: u32,
    pub rows: u32,
    pub vocab_rows: u32,
    pub groups: Option<Vec<Vec<u32>>>,
}

/// What a build promises the host at run time.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contract {
    /// Positions the program serves: min(KV entries x KV rows, the RoPE tables' reach).
    pub positions: u32,
    /// RoPE table levels and digit bits (`RLV`, `RBITS`).
    pub rope_levels: u32,
    pub rope_bits: u32,
    /// Passes a launch can run (the token ring, `CAP`).
    pub cap: u32,
    /// Default posted loop length (`LOOP`).
    pub loop_len: u32,
    /// The on-wafer token loop (`WL`).
    pub token_loop: bool,
    /// The logits' prescale the head applies (gpt-oss's band head: 2^15), 1 without one.
    pub logit_scale: f32,
}

/// The logical plan: the input of the physical build ([`super::program::Program::build`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub shape: Shape,
    pub partition: Partition,
    pub contract: Contract,
    /// The kernel levers the build applies (the preset's, `K:V,...`).
    #[serde(default)]
    pub levers: String,
}

impl Plan {
    /// Checks the plan is self-consistent before any physical work: divisibility (`rows` divides
    /// the placed hidden, segments divide the half-rows they live in, head groups cover every row),
    /// `layers_per_unit * units == layers.len()`, `kv_rows <= rows`, positions within the RoPE
    /// reach.
    ///
    /// Only facts every layout states the same way are checked; a layout's own rules stay in its
    /// partition ([`super::partition`]) and its blocks.
    pub fn validate(&self) -> Result<(), String> {
        let (sh, pa, co) = (&self.shape, &self.partition, &self.contract);
        let bad = |m: String| Err(format!("plan: {m}"));
        if sh.hidden == 0 || sh.vocab == 0 || sh.layers.is_empty() {
            return bad("an empty model".into());
        }
        if pa.hidden_placed() < sh.hidden {
            return bad(format!(
                "placed hidden {} < hidden {}",
                pa.hidden_placed(),
                sh.hidden
            ));
        }
        if pa.units == 0 || pa.layers_per_unit == 0 || pa.rows == 0 || pa.cols == 0 {
            return bad("no units, layers, rows or columns".into());
        }
        let slots = pa.units * pa.layers_per_unit;
        if (sh.layers.len() as u32) > slots {
            return bad(format!(
                "{} layers on {} units of {}",
                sh.layers.len(),
                pa.units,
                pa.layers_per_unit
            ));
        }
        if pa.kv_rows == 0 || pa.kv_rows > pa.rows {
            return bad(format!("KV rows {} of {} rows", pa.kv_rows, pa.rows));
        }
        let mut next = 0;
        for run in &pa.row_dims {
            if run.rows.0 != next || run.rows.1 <= run.rows.0 {
                return bad(format!("row runs not contiguous at {:?}", run.rows));
            }
            next = run.rows.1;
        }
        if next != pa.rows {
            return bad(format!("row runs end at {next}, not {}", pa.rows));
        }
        // A segment lives in one half of a unit row and must not span its control column.
        let halves: Vec<(&str, u32, u32)> = match pa.attn_cols {
            Some(ax) => {
                if ax >= pa.cols {
                    return bad(format!("{ax} attention columns of {}", pa.cols));
                }
                let mut v = vec![("scores", pa.segments.scores, ax)];
                if let Some(r) = pa.segments.router {
                    v.push(("router", r, pa.cols - ax));
                }
                v
            }
            None => vec![
                ("scores", pa.segments.scores, pa.cols / 2),
                ("gdn", pa.segments.gdn, pa.cols / 2),
                ("onorm", pa.segments.onorm, pa.cols / 2),
            ],
        };
        for (n, w, half) in halves {
            if w != 0 && !half.is_multiple_of(w) {
                return bad(format!(
                    "{n} segments of {w} do not divide the {half} columns they live in"
                ));
            }
        }
        if let Some(groups) = &pa.head.groups {
            let mut seen = vec![false; pa.rows as usize];
            for g in groups {
                if g.is_empty() {
                    return bad("an empty head group".into());
                }
                for &r in g {
                    match seen.get_mut(r as usize) {
                        Some(s) if !*s => *s = true,
                        _ => return bad(format!("head groups: row {r} twice or past the rows")),
                    }
                }
            }
            if seen.iter().any(|s| !s) {
                return bad("head groups leave a row out".into());
            }
        }
        if co.rope_levels > 0 {
            let bits = co.rope_bits * co.rope_levels;
            let reach = if bits >= 32 { u64::MAX } else { 1u64 << bits };
            if u64::from(co.positions) > reach {
                return bad(format!(
                    "{} positions past the RoPE tables' {reach}",
                    co.positions
                ));
            }
        }
        Ok(())
    }
}

impl Partition {
    /// The hidden size as placed: every row's dims (zero-padded past the model's hidden where the
    /// rows do not divide it; Qwen3-8B 4,096 -> 4,104).
    pub fn hidden_placed(&self) -> u32 {
        self.row_dims
            .iter()
            .map(|r| (r.rows.1 - r.rows.0) * r.dims)
            .sum()
    }
}

/// A partition's choices by name (a pinned plan's, [`super::pins`], or the search's), with any
/// overrides set over them: the counts and split factors a layout's partition turns into a
/// [`Partition`] and a [`Contract`] for a [`Shape`]. Keys are the layout's (`PX`, `PY`, `K`,
/// `G2`, ...); values are numbers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub keys: BTreeMap<String, f64>,
    /// The kernel levers the preset's program ships with (`K:V,...`, see
    /// [`super::emit::params::Levers`]): the plan carries them to the build.
    pub levers: String,
    /// The shape signature of the model this preset's program was measured on (a shipping
    /// preset's; [`Shape::signature`]): a plan with no preset named prefers it for that shape.
    #[serde(default)]
    pub made_for: Option<[u32; 3]>,
    /// The stacked unit columns' placement the preset's program was measured with (its unit
    /// shape's; [`Partition::arrangement`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrangement: Option<Arrangement>,
}

impl Preset {
    pub fn new(name: &str, keys: &[(&str, f64)]) -> Preset {
        Preset {
            name: name.to_string(),
            keys: keys.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            levers: String::new(),
            made_for: None,
            arrangement: None,
        }
    }

    /// `k=v,...` over the preset's keys.
    pub fn set(mut self, kvs: &str) -> Result<Preset, String> {
        for kv in kvs.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("--set {kv}: k=v"))?;
            let v: f64 = v.parse().map_err(|e| format!("--set {kv}: {e}"))?;
            self.keys.insert(k.to_string(), v);
        }
        Ok(self)
    }

    pub fn float(&self, k: &str) -> Option<f64> {
        self.keys.get(k).copied()
    }

    pub fn int(&self, k: &str) -> Option<u32> {
        self.float(k).map(|v| v as u32)
    }

    pub fn need(&self, k: &str) -> Result<u32, String> {
        self.int(k)
            .ok_or_else(|| format!("preset {}: no {k}", self.name))
    }
}
