use serde::{Deserialize, Serialize};

use crate::Op;
use crate::value::ValueId;

/// The precision a product quantizes its bf16 activations to before reading
/// a weight's codes: a property of the weight's deployment (`nvfp4a4`,
/// `mxfp4a8`); activations stay bf16 between products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActQuant {
    /// NVFP4: e2m1 under an e4m3 scale a 16, the scales under the static
    /// global `1 / input_scale` the weight's `.input_scales` plane holds (one
    /// an expert, or one for a projection).
    Nvfp4,
    /// e4m3 under a scale a `gran` (1x128 blocks, or 32). The engine picks
    /// the scale's form: ue8m0 (`2^ceil(log2(amax / 448))`, DeepSeek's) or
    /// f32 (`amax / 448`, transformers' and vLLM's against f32 weight
    /// factors), whichever its matrix cores take.
    E4m3 { gran: u32 },
}

impl ActQuant {
    /// The K one scale covers.
    #[must_use]
    pub const fn gran(self) -> u32 {
        match self {
            Self::Nvfp4 => 16,
            Self::E4m3 { gran } => gran,
        }
    }

    /// Whether the scales sit under a static global the weight states.
    #[must_use]
    pub const fn global(self) -> bool {
        matches!(self, Self::Nvfp4)
    }
}

/// How a [`Linear::MoeRoute`] scores and weighs its experts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RouteScore {
    /// Softmax over the experts (times `scale`, a per-expert gain, where given).
    Softmax,
    /// Sigmoid scores, the top-k's weights renormalized where stated, then
    /// times `scaling`.
    Sigmoid { renormalize: bool, scaling: f32 },
    /// Sigmoid scores, with `sink` slots past the `top_k` routed ones (its
    /// routes and weights are `top_k + sink` wide).
    SigmoidSink { sink: u32, scaling: f32 },
    /// `sqrt(softplus(x))` scores (DeepSeek-V4).
    SqrtSoftplus { renormalize: bool, scaling: f32 },
    /// The routes a biased pick predicts; its weights are not read.
    Predict,
    /// Experts looked up by token id in `table` (`vocab` rows), weighted by
    /// their scores of `logits`.
    Hash {
        vocab: u32,
        renormalize: bool,
        scaling: f32,
    },
}

/// The gate's activation a [`Linear::Glu`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GluAct {
    /// `silu(gate) · up`.
    Swiglu,
    /// GPT-OSS's: `gate · sigmoid(alpha · gate) · (up + 1)`.
    SwigluAlpha { alpha: f32 },
    /// `gelu_tanh(gate) · up`.
    GegluTanh,
    /// `situ_beta(gate) · up`, `up` capped at `up_cap` where stated.
    Situ { beta: f32, up_cap: Option<f32> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "linear")]
pub enum Linear {
    /// `act · wᵀ`. `quant` states the precision `act` is quantized to
    /// before it meets `w`'s codes (under `input_scales`, the weight's static
    /// global, when the format has one; `scales` the weight's block scales
    /// when they are a plane of their own beside raw codes, `fp8b128`), else
    /// the product runs at the activations' own precision. `logits` marks the
    /// vocabulary's head, which backends tile and gather as its own product.
    Matmul {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        #[input(F32)]
        input_scales: Option<ValueId>,
        #[input]
        scales: Option<ValueId>,
        quant: Option<ActQuant>,
        logits: bool,
        #[output]
        y: ValueId,
    },
    /// A gated linear unit: `act(gate) · up` over `[gate | up]` rows
    /// (`packed`, `intermediate` columns of each) or two planes (`gate`,
    /// `up`, `intermediate` wide), clamped at `limit` where stated, each row
    /// times its route's f32 weight before it rounds where `weights` is given
    /// (an expert scaling its activations before its down leg quantizes).
    Glu {
        #[input]
        packed: Option<ValueId>,
        #[input]
        gate: Option<ValueId>,
        #[input]
        up: Option<ValueId>,
        #[input(F32)]
        weights: Option<ValueId>,
        intermediate: u32,
        act: GluAct,
        limit: Option<f32>,
        #[output]
        y: ValueId,
    },
    GeluTanh {
        #[input]
        x: ValueId,
        #[output]
        y: ValueId,
    },
    /// `relu(x)^2` (Nemotron-H's ungated experts).
    Relu2 {
        #[input]
        x: ValueId,
        #[output]
        y: ValueId,
    },
    RelBias {
        #[input]
        x: ValueId,
        #[input]
        w: ValueId,
        heads: u32,
        d_rel: u32,
        extent: u32,
        #[output(F32)]
        y: ValueId,
    },
    /// A token's `top_k` of `experts`: each expert's score of `logits`
    /// (`bias` added before the pick where given, `scale` the per-expert gain
    /// or post-scale, `hint` an earlier pick to keep), the picked experts in
    /// `routes` and their weights; a hashed route reads its experts from
    /// `table` at the token's id instead.
    MoeRoute {
        #[input(I32)]
        ids: Option<ValueId>,
        #[input]
        table: Option<ValueId>,
        #[input]
        logits: ValueId,
        #[input]
        bias: Option<ValueId>,
        #[input]
        scale: Option<ValueId>,
        #[input]
        hint: Option<ValueId>,
        experts: u32,
        top_k: u32,
        score: RouteScore,
        #[output(I32)]
        routes: ValueId,
        #[output(F32)]
        weights: ValueId,
    },
    GroupRoutes {
        groups: u32,
        #[output(I32)]
        routes: ValueId,
    },
    MatmulGrouped {
        #[input]
        x: ValueId,
        #[input]
        w: ValueId,
        #[input(I32)]
        routes: ValueId,
        groups: u32,
        #[output]
        y: ValueId,
    },
    MoeMatmulSelect {
        #[input]
        x: ValueId,
        #[input]
        bank: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[output]
        y: ValueId,
    },
    MoeMatmulSelectBias {
        #[input]
        x: ValueId,
        #[input]
        bank: ValueId,
        #[input]
        bias: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[output]
        y: ValueId,
    },
    MoeMatmulSelectQuant {
        #[input]
        x: ValueId,
        #[input]
        bank: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[output]
        y: ValueId,
    },
    /// [`Linear::MoeMatmulSelectQuant`] whose activations are quantized as
    /// `quant` states first, each route's under its expert's `input_scales`
    /// when the format has a static global.
    MoeMatmulSelectActQuant {
        #[input]
        x: ValueId,
        #[input]
        bank: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input(F32)]
        input_scales: Option<ValueId>,
        quant: ActQuant,
        #[output]
        y: ValueId,
    },
    MoeWeightedSum {
        #[input]
        routed: ValueId,
        #[input(F32)]
        weights: ValueId,
        #[output]
        y: ValueId,
    },
    /// Each token's routed rows (already weighted) summed in f32, plus its
    /// `shared` row, rounded once.
    MoeSumAdd {
        #[input]
        routed: ValueId,
        #[input]
        shared: ValueId,
        #[output]
        y: ValueId,
    },
    MoeBiasSum {
        #[input]
        x: ValueId,
        #[input]
        bias: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input(F32)]
        weights: ValueId,
        #[output]
        y: ValueId,
    },
    MoeSigmoidGateAdd {
        #[input]
        routed: ValueId,
        #[input]
        shared: ValueId,
        #[input]
        gate: ValueId,
        #[output]
        y: ValueId,
    },
    LoraCorrect {
        #[input]
        x: ValueId,
        #[input]
        bank_a: ValueId,
        #[input]
        bank_b: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
}
