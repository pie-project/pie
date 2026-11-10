//! The logical plan: what the model is ([`Shape`]), how it is cut ([`Partition`]) and what the
//! build promises at run time ([`Contract`]), with the family that builds it. No coordinates,
//! colours or kernel names: the spatial pass reads the [`Shape`] off the model's trace
//! ([`super::target`]) and a family turns it and a [`Preset`] into the plan
//! ([`super::family::Family::plan`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Which program family a plan builds (each is one [`super::family::Family`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FamilyId {
    /// Qwen3.5 (GDN + gated attention), Qwen3 and Llama 3 (dense attention, no output gate):
    /// `family/qwen.rs`.
    Qwen,
    /// gpt-oss (attention / MoE column split, banded head): `family/gptoss.rs`.
    GptOss,
    /// Gemma 4 E2B / E4B (PLE storage, shared KV, sliding / global layers): `family/gemma.rs`.
    Gemma,
    /// Gemma 4 12B (attention / plain row roles, head groups): `family/gemma12.rs`.
    Gemma12,
}

impl FamilyId {
    pub const ALL: [FamilyId; 4] = [
        FamilyId::Qwen,
        FamilyId::GptOss,
        FamilyId::Gemma,
        FamilyId::Gemma12,
    ];

    /// The family's name (the plan's record, error messages).
    pub fn name(self) -> &'static str {
        match self {
            FamilyId::Qwen => "qwen",
            FamilyId::GptOss => "gptoss",
            FamilyId::Gemma => "gemma",
            FamilyId::Gemma12 => "gemma12",
        }
    }

    pub fn of_name(name: &str) -> Option<FamilyId> {
        FamilyId::ALL.into_iter().find(|f| f.name() == name)
    }
}

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
    /// Qwen3.5's sigmoid output gate.
    pub output_gate: bool,
    /// RMSNorm over each q / k head before RoPE.
    pub qk_norm: bool,
    /// gpt-oss's attention sinks.
    pub sinks: bool,
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
    /// Llama 3's long waves stretched); `None`: `theta ^ (-2 i / rotary_dims)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freqs: Option<Vec<f32>>,
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
    /// units short of the slots leaves gap slots (the family's rule places them). `None`: the
    /// family places its units otherwise (explicit origins).
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
    pub family: FamilyId,
    pub shape: Shape,
    pub partition: Partition,
    pub contract: Contract,
}

impl Plan {
    /// Checks the plan is self-consistent before any physical work: divisibility (`rows` divides
    /// the placed hidden, segments divide the half-rows they live in, head groups cover every row),
    /// `layers_per_unit * units == layers.len()`, `kv_rows <= rows`, positions within the RoPE
    /// reach.
    ///
    /// Only facts every family states the same way are checked; a family's own rules stay in its
    /// `plan` / `floor`.
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

/// A family's partition choices by name (its preset table), with any overrides set over them:
/// the counts and split factors a family turns into a [`Partition`] and a [`Contract`] for a
/// [`Shape`]. Keys are the family's (`PX`, `PY`, `K`, `G2`, ...); values are numbers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub keys: BTreeMap<String, f64>,
}

impl Preset {
    pub fn new(name: &str, keys: &[(&str, f64)]) -> Preset {
        Preset {
            name: name.to_string(),
            keys: keys.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
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
