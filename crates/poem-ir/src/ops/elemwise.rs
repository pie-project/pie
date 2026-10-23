use serde::{Deserialize, Serialize};

use crate::Op;
use crate::operands::{Part, Slot};
use crate::value::ValueId;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

/// A rotation's frequencies stated outright (a `rope_scaling` the model
/// declares: linear, dynamic NTK, YaRN, llama3, longrope): one inverse
/// frequency a turned pair, `inv_freq[i]` turning pair `i` where the plain
/// rotation would turn it at `theta^(-i/span)`, and the scale both its cos
/// and sin carry. Absent, a rotation turns at `theta` as it states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeFreqs {
    pub inv_freq: Vec<f32>,
    pub mscale: f32,
}

/// The norm a [`Elemwise::NormGate`] also lands over its output: `y =
/// rmsnorm(out) * weight` (`1 + weight` where `plus_one`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormNext {
    pub weight: ValueId,
    pub eps: f32,
    pub plus_one: bool,
    pub y: ValueId,
}

/// What an [`Elemwise::Rmsnorm`] scales its normed rows by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormWeight {
    /// Unscaled.
    None,
    /// `* weight`.
    Scale(ValueId),
    /// `* (1 + weight)`.
    PlusOne(ValueId),
}

impl Slot for NormWeight {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        match self {
            NormWeight::None => {}
            NormWeight::Scale(w) | NormWeight::PlusOne(w) => f(*w),
        }
    }
}

/// The runs of a row an [`Elemwise::Rmsnorm`] norms on their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormAxis {
    /// The whole row.
    Row,
    /// Each `head_dim`-wide head, one weight of that width shared by all.
    Head(u32),
    /// Each `group`-wide run, the weight as wide as the row.
    Group(u32),
}

impl Part for NormNext {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.weight);
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.y);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "elemwise")]
pub enum Elemwise {
    /// `x / rms(x)` over each `axis` run, times `weight`.
    Rmsnorm {
        #[input]
        x: ValueId,
        #[input]
        weight: NormWeight,
        axis: NormAxis,
        eps: f32,
        #[output]
        y: ValueId,
    },
    /// Muse Spark's post-norm and residual gate: `u = x * scale` rounded
    /// (`x` where no scale), `n = rmsnorm(u)` unscaled and rounded, and with
    /// a residual `y = alpha * res + beta * n`, `beta = sigmoid(w)` and
    /// `alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))` of the gate's
    /// `w = gate / temperature` a column (f32); else `y = n`.
    NormGate {
        #[input]
        x: ValueId,
        #[input]
        scale: Option<ValueId>,
        #[input]
        res: Option<ValueId>,
        #[input]
        gate: Option<ValueId>,
        eps: f32,
        temperature: f32,
        #[output]
        y: ValueId,
        #[part]
        next: Option<NormNext>,
    },
    /// `(x - mean) / std` over the row, times `weight` plus `bias` where
    /// given.
    Layernorm {
        #[input]
        x: ValueId,
        #[input]
        weight: Option<ValueId>,
        #[input]
        bias: Option<ValueId>,
        eps: f32,
        #[output]
        y: ValueId,
    },
    /// `x` clamped into `[lo, hi]`, in place: bounds stated outright, or
    /// learned planes read as `lo` and `hi`.
    Clamp {
        #[input]
        x: ValueId,
        #[input]
        lo: Option<ValueId>,
        #[input]
        hi: Option<ValueId>,
        bounds: Bounds,
        #[output(alias = x)]
        x_out: ValueId,
    },
    RmsnormGated {
        #[input]
        x: ValueId,
        #[input]
        gate: ValueId,
        #[input]
        weight: Option<ValueId>,
        head_dim: u32,
        eps: f32,
        act: GateActivation,
        #[output]
        y: ValueId,
    },
    RmsnormGatedBy {
        #[input]
        x: ValueId,
        #[input]
        gate: ValueId,
        #[input]
        weight: ValueId,
        heads: u32,
        eps: f32,
        #[output]
        y: ValueId,
    },
    ResidualAdd {
        #[input]
        x: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
    AddBias {
        #[input]
        bias: ValueId,
        #[input]
        out: ValueId,
        #[output(alias = out)]
        out_out: ValueId,
    },
    Standardize {
        #[input]
        x: ValueId,
        #[input]
        bias: ValueId,
        #[input]
        scale: ValueId,
        #[output(alias = x)]
        x_out: ValueId,
    },
    /// `act(x · k)`, in place: `k` the f32 `scale`, the first element of the
    /// device scalar `s`, or 1.
    Act {
        #[input]
        s: Option<ValueId>,
        #[input]
        x: ValueId,
        act: ActFn,
        scale: Option<f32>,
        #[output(alias = x)]
        x_out: ValueId,
    },
    ResBlend {
        #[input]
        prefix: ValueId,
        #[input]
        blocks: Vec<ValueId>,
        #[input]
        weight: ValueId,
        #[input]
        proj: ValueId,
        eps: f32,
        #[output]
        y: ValueId,
    },
    /// The rotary turn of each head's dims at its row's position, q and
    /// (where given) k in place. `turn` says which dims turn and how their
    /// frequencies run; `freqs` states the frequencies outright where the
    /// model's `rope_scaling` does.
    Rope {
        #[input]
        q: ValueId,
        #[input]
        k: Option<ValueId>,
        #[input(I32)]
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        turn: Turn,
        #[serde(default)]
        freqs: Option<RopeFreqs>,
        #[output(alias = q)]
        q_out: ValueId,
        #[output(alias = k)]
        k_out: Option<ValueId>,
    },
    GateSigmoidMul {
        #[input]
        x: ValueId,
        #[input]
        gate: ValueId,
        #[output(alias = x)]
        x_out: ValueId,
    },
    GateSigmoidMulHeads {
        #[input]
        x: ValueId,
        #[input]
        gate: ValueId,
        head_dim: u32,
        scale: f32,
        #[output(alias = x)]
        x_out: ValueId,
    },
    HcExpand {
        #[input]
        x: ValueId,
        streams: u32,
        #[output]
        y: ValueId,
    },
    HcRmsnormF32 {
        #[input]
        streams: ValueId,
        eps: f32,
        #[output(F32)]
        y: ValueId,
    },
    HcProject {
        #[input(F32)]
        normed: ValueId,
        #[input(F32)]
        weight: ValueId,
        stream_count: u32,
        #[output(F32)]
        mixes: ValueId,
    },
    HcGates {
        #[input(F32)]
        normed: ValueId,
        #[input]
        streams: ValueId,
        #[input]
        scale: ValueId,
        #[input]
        base: ValueId,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        comb: bool,
        #[output]
        x: ValueId,
        #[output(F32)]
        post_mix: ValueId,
        #[output(F32)]
        comb_mix: ValueId,
    },
    HcFold {
        #[input]
        x: ValueId,
        #[input]
        streams: ValueId,
        #[input]
        post_mix: ValueId,
        #[input]
        comb_mix: ValueId,
        #[output]
        y: ValueId,
    },
    HcCollapse {
        #[input(F32)]
        mixes: ValueId,
        #[input]
        streams: ValueId,
        #[input(F32)]
        scale: ValueId,
        #[input(F32)]
        base: ValueId,
        stream_count: u32,
        hc_eps: f32,
        #[output]
        y: ValueId,
    },
    HcMix {
        #[input]
        gates: ValueId,
        #[input]
        normed: ValueId,
        streams: u32,
        #[output]
        y: ValueId,
    },
    HcInject {
        #[input]
        o: ValueId,
        #[input]
        gates: ValueId,
        #[input]
        hyper: ValueId,
        streams: u32,
        #[output(alias = hyper)]
        hyper_out: ValueId,
    },
    PleGate {
        #[input]
        key: ValueId,
        #[input]
        query: ValueId,
        #[input]
        value: ValueId,
        streams: u32,
        #[output]
        y: ValueId,
    },
    Modulate {
        #[input]
        x: ValueId,
        #[input]
        m: ValueId,
        #[input(I32)]
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        #[output]
        y: ValueId,
    },
    GatedResidualAdd {
        #[input]
        r: ValueId,
        #[input]
        g: ValueId,
        #[input]
        y: ValueId,
        #[input(I32)]
        lane_of_row: Option<ValueId>,
        #[output(alias = r)]
        r_out: ValueId,
    },
    Sinusoid {
        #[input(F32)]
        t: ValueId,
        dim: u32,
        max_period: f32,
        flip_sin_cos: bool,
        scale: f32,
        #[output(F32)]
        y: ValueId,
    },
    RelativeBucketBias {
        #[input]
        embedding: ValueId,
        max_len: u32,
        num_buckets: u32,
        max_distance: f32,
        bidirectional: bool,
        #[output(F32)]
        y: ValueId,
    },
    Hadamard {
        #[input]
        x: ValueId,
        #[input]
        signs: Option<ValueId>,
        block: u32,
        #[output(alias = x)]
        x_out: ValueId,
    },
    Mul {
        #[input]
        x: ValueId,
        #[input]
        y: ValueId,
        #[output]
        z: ValueId,
    },
    Add {
        #[input]
        x: ValueId,
        #[input]
        y: ValueId,
        #[output]
        z: ValueId,
    },
    RopeAxes {
        #[input]
        x: ValueId,
        #[input(F32)]
        positions: ValueId,
        dims: [u32; 4],
        thetas: [f32; 4],
        form: RopeForm,
        rotary_dim: u32,
        head_dim: u32,
        #[output(alias = x)]
        x_out: ValueId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulateForm {
    ScaleShift,
    Scale,
    TanhGate,
}

impl ModulateForm {
    #[must_use]
    pub fn slices(self) -> u64 {
        match self {
            ModulateForm::ScaleShift => 2,
            ModulateForm::Scale | ModulateForm::TanhGate => 1,
        }
    }
}

/// The function an [`Elemwise::Act`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActFn {
    Identity,
    Silu,
    Gelu { tanh: bool },
    Tanh,
}

/// Where a [`Elemwise::Clamp`]'s bounds come from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Bounds {
    Fixed {
        lo: f32,
        hi: f32,
    },
    /// The op's `lo` and `hi` planes.
    Learned,
}

/// Which of a head's dims a [`Elemwise::Rope`] turns, and how.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Turn {
    /// Every pair: `(i, i + head_dim/2)`, or `(2i, 2i + 1)` interleaved.
    Full { interleaved: bool },
    /// The first `rotary_dim/2` pairs, pair `i` being `(i, i + span)` at
    /// `theta^(-i/span)`: `span` is `rotary_dim/2` for a plain partial
    /// rotary, `head_dim/2` for a proportional one (Gemma 4's global layers).
    Partial { rotary_dim: u32, span: u32 },
    /// The last `rotary_dim` dims (`inverse` turns them back), at frequencies
    /// YaRN ramps where it is given.
    Last {
        rotary_dim: u32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
    },
    /// Every pair at YaRN's stretched frequencies, cos and sin scaled by
    /// `attention_factor`.
    Yarn {
        yarn: Yarn,
        attention_factor: f32,
        interleaved: bool,
    },
    /// Multimodal: a `(t, h, w)` position a row, `sections` naming which
    /// pairs of the first `rotary_dim` turn by which axis.
    Mrope {
        sections: [u32; 3],
        form: MropeForm,
        rotary_dim: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateActivation {
    Silu,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MropeForm {
    Interleaved,
    Blocked,
    Split,
}
