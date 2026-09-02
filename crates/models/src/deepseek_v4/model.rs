use model_dsl::ops::elemwise::Yarn;
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub act: Dtype,

    /// The one attention reading this text is carved for; every layer
    /// shares this schedule (query heads, kv heads, head width, window).
    /// The latent plane is shared across heads, so kv heads are these
    /// query `heads`.
    pub heads: u32,
    pub head_dim: u32,
    pub window: u32,

    /// The adapter banks this family seats. Per layer, and the same two
    /// numbers at every one: the correction is a per-lane axis, not a
    /// per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub hyper: Hyper,

    pub embed: Weight,
    /// The output projection. `None` on the toy `base` rows, whose lm_head ties
    /// the embedding; `Some` on the flash rows, which ship a distinct
    /// `lm_head.weight` (config `tie_word_embeddings = false`).
    pub head: Option<Weight>,
    /// The trunk-level hyper-connection head (`model.hc_head.*`), stated once
    /// for the whole tower. `None` on the toy, `Some` on flash.
    pub hc_head: Option<HcHead>,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
    /// The draft head, or `None` for a row without one.
    pub mtp: Option<Mtp>,
}

/// **THE DRAFT HEAD** — DeepSeek-V4-Flash's one `nextn` layer (the official
/// `MTPBlock`): the next token's embedding and the trunk's residual STREAMS
/// fused, one flash block over the fused streams, the block's own hyper head
/// and norm, and the base `lm_head`.
///
/// ```text
/// x[s] = e_proj(enorm(embed(tok))) + h_proj(hnorm(streams[s]))   per stream s
/// x    = block(x)                                                 (attn + MoE, hc gated)
/// out  = lm_head(norm(hc_head(x)))
/// ```
pub struct Mtp {
    pub enorm: Weight,
    pub hnorm: Weight,
    pub e_proj: Weight,
    /// `h_proj`, `streams` times over as one block-diagonal bank.
    pub h_proj: Weight,
    pub block: Layer,
    pub hc_head: HcHead,
    pub norm: Weight,
    pub norm_eps: f32,
}

/// Where one block is stated: its name prefix, cadence, gate kind, dtypes
/// and cache rows — the trunk's per-layer facts, and the draft head's.
struct Site {
    prefix: String,
    ratio: Option<u32>,
    hash: bool,
    /// The routed bank's expert count: the trunk's row states it, the draft
    /// head is always the checkpoint's full 256.
    experts: u32,
    split: bool,
    gate: Dtype,
    up: Dtype,
    down: Dtype,
    weights: Dtype,
    dense: Dtype,
    kv: String,
    pool: String,
    index: String,
}

pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
}

/// Trunk hyper-connection head: `base [streams]`, `fn [streams,
/// streams*hidden]` (per-token mixing function, stated but not yet fused
/// into the gate op — see `forward::flash`), `scale [1]` (config `hc_head`).
pub struct HcHead {
    pub base: Weight,
    pub dynamic: Weight,
    pub scale: Weight,
}

pub struct Mix {
    pub scale: Weight,
    pub base: Weight,
    /// The dynamic hyper-connection plane (`{attn,ffn}_hc.fn`, `[2*streams +
    /// streams², streams*hidden]`). `None` on the toy, `Some` on flash.
    pub dynamic: Option<Weight>,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn_mix: Mix,
    /// The attention sublayer's own pre-norm (`attn_norm.weight`), applied to
    /// the gated stream. `None` on the toy, which norms inside the gate;
    /// `Some` on flash, which ships the plane.
    pub attn_norm: Option<Weight>,
    pub attn: Attn,
    pub mlp_mix: Mix,
    /// The feed-forward sublayer's own pre-norm (`ffn_norm.weight`). `None` on
    /// the toy, `Some` on flash.
    pub mlp_norm: Option<Weight>,
    pub mlp: Mlp,
    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction site.
    ///
    /// The site is the attention sublayer because both ends are replicated
    /// values: the input is the gated stream `hc_gates` hands the mixer, and
    /// the output is `o_up`'s, already past `all_reduce` (`o_down` is
    /// rows-cut, `o_up` is replicated). A correction stated before the
    /// reduce would read a rows-cut partial product and be summed `tp` times.
    ///
    /// Applied before `hc_fold`, not after, so a corrected site stays one
    /// site instead of becoming a second contribution with its own mixing
    /// weights.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub struct Attn {
    pub rope_dim: u32,
    /// **THE LAYER'S OWN THETA.** Flash ropes its compressor layers at
    /// `compress_rope_theta` with the YaRN ramp beside it and its pure
    /// sliding-window layers at `rope_theta` with none — the official
    /// `Attention.__init__`'s `if self.compress_ratio` — and the pooled
    /// entries and the attention output's un-rotation ride the same pair.
    pub theta: f32,
    pub yarn: Option<Yarn>,
    pub sm_scale: f32,
    pub q_down: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub q_up: Weight,
    pub kv_down: Weight,
    pub kv_norm: Weight,
    pub kv_norm_eps: f32,
    pub o_down: Weight,
    pub o_up: Weight,
    /// The o-projection groups (`o_groups`): the attention output is reduced
    /// over this many blocks before `o_down`. One on the toy; eight on flash
    /// (`wo_a` out is `o_groups * o_lora`).
    pub o_groups: u32,
    pub sink: Weight,
    pub kv: String,
    pub pool: Option<Pool>,
    /// The NSA sparse-selection indexer, present on the ratio-4 layers only.
    /// Narrows this layer's own compressed branch: it ranks its own
    /// compressor's pooled entries (1:1 with the attention compressor's).
    /// Ratio-128 layers carry none, since `S / 128` is a key set no budget
    /// needs to cap.
    pub indexer: Option<Indexer>,
}

/// The NSA compressor pool. On the toy it is a parameter-free mean pool (ratio
/// only). On flash it carries the learned compression planes
/// (`compressor.{wkv,wgate,ape,norm}`).
pub struct Pool {
    pub ratio: u32,
    pub entries: String,
    pub compressor: Option<Compressor>,
}

/// One learned NSA compressor: a gated low-rank projection of the residual
/// into per-block compressed entries, with an intra-block absolute-position
/// plane (`ape`) and a norm.
pub struct Compressor {
    pub wkv: Weight,
    pub wgate: Weight,
    pub ape: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub yarn: Option<Yarn>,
    pub window: u32,
    pub wq_b: Weight,
    pub weights_proj: Weight,
    pub compressor: Compressor,
    pub keys: String,
}

/// The routed gate. Layers `< num_hash_layers` route by a per-token hash table
/// (`ffn.gate.tid2eid [vocab, top_k]`, an I64 lookup, not a matmul); later
/// layers carry the `noaux_tc` correction bias (`ffn.gate.e_score_correction_bias
/// [experts]`).
pub enum Gate {
    Hash { tid2eid: Weight },
    Bias { bias: Weight },
}

/// The routed experts' two projections, fused or not.
/// [`Fused`](GateUp::Fused) is one bank, one routed matmul, one packed
/// swiglu-clamp. [`Split`](GateUp::Split) is forced when `gate_proj`/
/// `up_proj` are quantized at different MLX group sizes so their scales
/// can't join one bank — each half gets its own bank and dtype, the
/// matmul fires twice, and `linear.mlp_swiglu_clamp_split` combines them.
pub enum GateUp {
    Fused(Weight),
    Split { gate: Weight, up: Weight },
}

pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
        limit: f32,
    },
    Routed {
        router: Weight,
        bias: Weight,
        gate_up: Weight,
        down: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
    /// The flash MoE: a `sqrtsoftplus`/`noaux_tc` router over the stacked
    /// `switch_mlp` experts plus one always-on shared expert, with the two
    /// gate kinds (`hash` for the first `num_hash_layers`, correction `bias`
    /// after).
    MoeFlash {
        router: Weight,
        gate: Gate,
        gate_up: GateUp,
        down: Weight,
        shared_gate_up: Weight,
        shared_down: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    pool: &'static [Option<u32>],
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    window: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    dense_inter: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    norm_eps: f32,
}

/// The flash geometry's extra facts a toy [`Dims`] has no field for — a
/// superset kept beside the toy so the toy's own `new` stays byte-identical.
struct FlashDims {
    hidden: u32,
    layers: u32,
    /// Per-layer compressor schedule (`compress_ratios`): `None` for a ratio-0
    /// layer (no compressor), `Some(4)` for a compressor+indexer layer,
    /// `Some(128)` for a compressor-only layer.
    pool: &'static [Option<u32>],
    num_hash_layers: u32,
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    kv_latent: u32,
    o_groups: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    compress_theta: f32,
    /// The YaRN ramp the compressor layers rope under (`rope_scaling`).
    yarn: Yarn,
    /// Whether this row carries the draft head ([`Mtp`]).
    draft: bool,
    window: u32,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    index_window: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    shared_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    norm_eps: f32,
}

/// The full DeepSeek-V4-Flash `compress_ratios`, one per layer. Layers 0-1
/// carry no compressor; from layer 2 up, even layers carry a compressor and
/// indexer (ratio 4), odd layers a compressor only (ratio 128):
///
/// ```text
/// [0, 0, 4, 128, 4, 128, …, 4, 128, 4, 0]
/// ```
///
/// The trailing 0 is the multi-token-prediction layer (`num_nextn_predict_layers`
/// = 1), not one of the 43 this text builds.
const FLASH_RATIOS: [Option<u32>; 43] = flash_ratios();

/// [`FLASH_RATIOS`], stated as the rule rather than forty-three literals.
const fn flash_ratios() -> [Option<u32>; 43] {
    let mut out = [None; 43];
    let mut layer = 2;
    while layer < 43 {
        out[layer] = if layer % 2 == 0 { Some(4) } else { Some(128) };
        layer += 1;
    }
    out
}

/// The mini `mlx-community/DeepSeek-V4-Flash-2bit-DQ` snapshot's five-layer
/// schedule (original layers 0, 1, 2, 3, 42, renumbered), which the name
/// bijection test holds this text against.
const FLASH_MICRO_RATIOS: [Option<u32>; 5] =
    [None, None, Some(4), Some(128), Some(4)];

/// What the routed expert block is stored as, when one `weights` dtype
/// cannot say it: one representation per projection plus per-layer
/// exceptions, mirroring the DQ conversion's per-tensor quantization.
/// Everything else in the text still reads the trunk's `weights`.
#[derive(Clone, Copy, Debug)]
pub struct Routed {
    /// The routed gate projection's representation.
    pub gate: Dtype,
    /// The layers whose gate is NOT [`gate`](Routed::gate), read from the
    /// conversion's per-tensor overrides. On the 2-bit DQ artifact the
    /// routed `gate_proj` groups by 32 on layers 0-3 and by 64 on the last
    /// (original layer 42, renumbered to 4); `up_proj`/`down_proj` group by
    /// 64 throughout.
    pub gate_at: &'static [(u32, Dtype)],
    /// The routed up projection's.
    pub up: Dtype,
    /// The routed down projection's.
    pub down: Dtype,
    /// Whether the gate and up halves are declared as two banks or one — see
    /// [`GateUp`]. A mix whose halves disagree has no choice; a uniform one
    /// stays fused and keeps its trace.
    pub split: bool,
}

impl Routed {
    /// The whole block in the trunk's own representation, fused — the
    /// reading under which this type is invisible.
    #[must_use]
    pub const fn uniform(w: Dtype) -> Routed {
        Routed {
            gate: w,
            gate_at: &[],
            up: w,
            down: w,
            split: false,
        }
    }

    /// The `mlx-community/DeepSeek-V4-Flash-2bit-DQ` conversion's own routed
    /// mix, verified against the snapshot's `config.json` and its stored
    /// `.scales` rectangles by
    /// `model/tests/the_flash_text_reads_the_mini_snapshot.rs`.
    pub const DQ_2BIT: Routed = Routed {
        gate: Dtype::U2g32,
        gate_at: &[(4, Dtype::U2g64)],
        up: Dtype::U2g64,
        down: Dtype::U2g64,
        split: true,
    };

    /// The same mix over the full forty-three layers, where the exception
    /// sits at its own layer number rather than the mini snapshot's
    /// renumbered one (mini layer 4 == full layer 42).
    ///
    /// Taken from the full artifact's `config.json`: routed `gate_proj` is
    /// `(bits 2, group 32)` on layers 0–41 and `(bits 2, group 64)` on
    /// layer 42; `up_proj`/`down_proj` are `(bits 2, group 64)` throughout.
    /// Everything outside the routed block uses the file's
    /// `(bits 4, group 64)` default (`U4g64`).
    pub const DQ_2BIT_FULL: Routed = Routed {
        gate: Dtype::U2g32,
        gate_at: &[(42, Dtype::U2g64)],
        up: Dtype::U2g64,
        down: Dtype::U2g64,
        split: true,
    };

    /// This layer's gate representation: the exception if the file states one,
    /// and the block's default otherwise.
    #[must_use]
    pub fn gate_of(&self, layer: u32) -> Dtype {
        self.gate_at
            .iter()
            .find_map(|(at, dtype)| (*at == layer).then_some(*dtype))
            .unwrap_or(self.gate)
    }
}

impl Model {
    pub fn base(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            act,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 6,
                dense_layers: 1,
                pool: &[Some(1), Some(2), Some(4), None, None, None],
                heads: 16,
                head_dim: 128,
                q_lora: 768,
                o_lora: 512,
                rope_dim: 64,
                theta: 10_000.0,
                window: 2048,
                streams: 4,
                gate_eps: 1e-6,
                alpha: 2.0,
                sinkhorn: 20,
                dense_inter: 5632,
                experts: 64,
                top_k: 6,
                moe_inter: 1024,
                renorm: false,
                scaling: 2.5,
                swiglu_limit: 7.0,
                vocab: 129_280,
                norm_eps: 1e-5,
            },
        )
    }

    /// The real DeepSeek-V4-Flash geometry, bf16 structure: 43 layers, hidden
    /// 4096, 64 MLA heads of width 512, the NSA compressor/indexer cadence, the
    /// 256-expert top-6 MoE with one shared expert, and the hyper-connection
    /// tower (`hc_mult 4`).
    pub fn flash(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::flash_mixed(w, Routed::uniform(w), act, kv, tp)
    }

    /// The real flash geometry with a routed block that is not the trunk's
    /// dtype: same forty-three layers, cadence and 256-expert top-6 MoE,
    /// but the routed experts are stated per projection instead of taken
    /// from `w`. `flash` is now this with [`Routed::uniform`].
    ///
    /// What the full 2-bit artifact is read through — a 4-bit trunk over a
    /// 2-bit routed block.
    pub fn flash_mixed(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new_flash(
            w,
            routed,
            act,
            kv,
            tp,
            Model::flash_dims(43, &FLASH_RATIOS, 3),
        )
    }

    /// [`flash_mixed`](Model::flash_mixed) with the draft head ([`Mtp`]).
    pub fn flash_mixed_mtp(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(43, &FLASH_RATIOS, 3);
        d.draft = true;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    /// The mini DQ snapshot's own geometry: the real DeepSeek-V4-Flash
    /// dimensions (hidden 4096, 64 MLA heads of width 512, moe_inter 2048,
    /// 129,280-token vocabulary) over the five renumbered layers and
    /// sixteen routed experts `mlx-community/DeepSeek-V4-Flash-2bit-DQ`
    /// publishes.
    ///
    /// Not [`flash_micro`](Model::flash_micro): that shrinks every
    /// dimension to hold this family's names against the snapshot's
    /// census, with shapes nobody's. This one is a text a checkpoint can
    /// actually be read through.
    /// [`flash_mini`](Model::flash_mini) with the draft head — the real
    /// head over the five-layer miniature, which is where the mechanism is
    /// gated before the full artifact carries it.
    pub fn flash_mini_mtp(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.experts = 16;
        d.draft = true;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    pub fn flash_mini(w: Dtype, routed: Routed, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.experts = 16;
        Model::new_flash(w, routed, act, kv, tp, d)
    }

    /// A DeepSeek-V4-Flash small enough to hold against the mini snapshot's
    /// tensor census — the five renumbered layers, sixteen experts, every
    /// organ present (both gate kinds, a compressor-only layer, two
    /// compressor+indexer layers). Its dims are shrunk; its NAMES and cadence
    /// are the mini's, which is what the bijection gate reads.
    pub fn flash_micro(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims(5, &FLASH_MICRO_RATIOS, 3);
        d.hidden = 256;
        d.heads = 8;
        d.head_dim = 64;
        d.rope_dim = 16;
        d.q_lora = 128;
        d.kv_latent = 64;
        // heads * head_dim == o_groups * hidden (8*64 == 2*256), the same
        // o-group invariant the real geometry meets at 64*512 == 8*4096.
        d.o_groups = 2;
        d.o_lora = 128;
        d.index_heads = 8;
        d.index_head_dim = 32;
        d.index_top_k = 16;
        d.moe_inter = 64;
        d.shared_inter = 64;
        d.experts = 16;
        d.vocab = 512;
        Model::new_flash(w, Routed::uniform(w), act, kv, tp, d)
    }

    fn flash_dims(layers: u32, pool: &'static [Option<u32>], hash: u32) -> FlashDims {
        FlashDims {
            hidden: 4096,
            layers,
            pool,
            num_hash_layers: hash,
            heads: 64,
            head_dim: 512,
            q_lora: 1024,
            kv_latent: 512,
            o_groups: 8,
            o_lora: 1024,
            rope_dim: 64,
            theta: 10_000.0,
            compress_theta: 160_000.0,
            yarn: Yarn {
                factor: 16.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max_position: 65_536,
            },
            draft: false,
            window: 128,
            index_heads: 64,
            index_head_dim: 128,
            index_top_k: 512,
            index_window: 128,
            streams: 4,
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
            experts: 256,
            top_k: 6,
            moe_inter: 2048,
            shared_inter: 2048,
            renorm: true,
            scaling: 1.5,
            swiglu_limit: 10.0,
            vocab: 129_280,
            norm_eps: 1e-6,
        }
    }

    fn new(weights: Dtype, act: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );

        let heads = d.heads / tp;
        let dense_inter = d.dense_inter / tp;
        let moe_inter = d.moe_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let q_w = heads as u64 * d.head_dim as u64;
        let q_lora = d.q_lora as u64;
        let o_lora = d.o_lora as u64;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], weights);
                let mix = |s: &str| Mix {
                    scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                    base: Weight::sym(
                        n(&format!("{s}_base")),
                        [2 * streams + streams * streams],
                        Dtype::F32,
                    ),
                    dynamic: None,
                };
                let (lora_a, lora_b) = crate::adapter::banks(
                    &format!("layer.{l}"),
                    ADAPTERS,
                    hidden,
                    crate::dense(weights),
                );
                Layer {
                    attn_mix: mix("attn_mix"),
                    attn_norm: None,
                    mlp_norm: None,
                    attn: Attn {
                        rope_dim: d.rope_dim,
                        theta: d.theta,
                        yarn: None,
                        sm_scale: (d.head_dim as f32).sqrt().recip(),
                        q_down: Weight::sym(n("q_down"), [q_lora, hidden], weights),
                        q_norm: norm("q_norm", q_lora),
                        q_norm_eps: d.norm_eps,
                        q_up: Weight::sym(n("q_up"), [q_w, q_lora], weights).columns(),
                        kv_down: Weight::sym(n("kv_down"), [q_w, hidden], weights).columns(),
                        kv_norm: Weight::sym(n("kv_norm"), [q_w], weights).columns(),
                        kv_norm_eps: d.norm_eps,
                        o_down: Weight::sym(n("o_down"), [o_lora, q_w], weights).rows(),
                        o_up: Weight::sym(n("o_up"), [hidden, o_lora], weights),
                        o_groups: 1,
                        sink: Weight::sym(n("attn_sink"), [heads as u64], weights).columns(),
                        kv: format!("kv.{l}"),
                        pool: d.pool[l as usize].map(|ratio| Pool {
                            ratio,
                            entries: format!("pool.{l}"),
                            compressor: None,
                        }),
                        indexer: None,
                    },
                    mlp_mix: mix("mlp_mix"),
                    mlp: if l < d.dense_layers {
                        Mlp::Dense {
                            gate_up: Weight::sym(
                                n("gate_up"),
                                [2 * dense_inter as u64, hidden],
                                weights,
                            )
                            .packed([dense_inter as u64, dense_inter as u64]),
                            down: Weight::sym(n("down"), [hidden, dense_inter as u64], weights)
                                .rows(),
                            inter: dense_inter,
                            limit: d.swiglu_limit,
                        }
                    } else {
                        Mlp::Routed {
                            router: Weight::sym(n("router"), [d.experts as u64, hidden], weights),
                            bias: Weight::sym(n("router_bias"), [d.experts as u64], weights),
                            gate_up: Weight::sym(
                                n("experts_gate_up"),
                                [d.experts as u64, 2 * moe_inter as u64, hidden],
                                weights,
                            )
                            .bank([moe_inter as u64, moe_inter as u64]),
                            down: Weight::sym(
                                n("experts_down"),
                                [d.experts as u64, hidden, moe_inter as u64],
                                weights,
                            )
                            .rows(),
                            experts: d.experts,
                            top_k: d.top_k,
                            inter: moe_inter,
                            limit: d.swiglu_limit,
                            renorm: d.renorm,
                            scaling: d.scaling,
                        }
                    },
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act,
            heads,
            head_dim: d.head_dim,
            window: d.window,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: None,
            hc_head: None,
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
            mtp: None,
        }
    }

    fn new_flash(
        weights: Dtype,
        routed: Routed,
        act: Dtype,
        kv: Dtype,
        tp: u32,
        d: FlashDims,
    ) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );

        // norms, the compressor's absolute-position embedding, and the
        // routed router ship unquantized by every MLX conversion of this
        // family, so they're stated in the compute dtype
        // (model_dsl::compute_dtype) rather than `weights`.
        let dense = crate::dense(weights);

        let heads = d.heads / tp;
        let moe_inter = d.moe_inter / tp;
        let shared_inter = d.shared_inter / tp;

        let hidden = d.hidden as u64;
        let streams = d.streams as u64;
        let hc_base = 2 * streams + streams * streams;
        let hc_fan = streams * hidden;
        let q_w = heads as u64 * d.head_dim as u64;
        let q_lora = d.q_lora as u64;
        let kv_latent = d.kv_latent as u64;
        let o_lora = d.o_lora as u64;
        let o_out = d.o_groups as u64 * o_lora;
        let idx_w = d.index_heads as u64 * d.index_head_dim as u64;
        let idx_norm_eps = d.norm_eps;

        // One learned compressor at `prefix`, its output `entries` wide over a
        // window of `ratio` positions, its norm `norm_w` wide.
        let compressor = |prefix: String, ratio: u32, entries: u64, norm_w: u64| Compressor {
            wkv: Weight::sym(format!("{prefix}.wkv"), [entries, hidden], weights),
            wgate: Weight::sym(format!("{prefix}.wgate"), [entries, hidden], weights),
            // position plane rides the gather's dtype, f32: added to gate
            // logits pre-softmax on both shaders (pool_gather_paged / the
            // CUDA twin), so f32 is what they agree on regardless of the
            // artifact's own element width.
            ape: Weight::sym(format!("{prefix}.ape"), [ratio as u64, entries], Dtype::F32),
            norm: Weight::sym(format!("{prefix}.norm"), [norm_w], dense),
            norm_eps: d.norm_eps,
        };

        // **ONE LAYER, STATED FOR A SITE.** The trunk's forty-three and the
        // draft head's one are the same block; what differs per site is its
        // name prefix, its pool cadence, its gate kind, its routed dtypes, its
        // dense dtype and its cache rows — so the block is a closure over a
        // `Site`, and the trunk and the head each state theirs.
        let layer_at = |site: Site| -> Layer {
            let prefix = site.prefix;
            let n = |s: &str| format!("{prefix}.{s}");
            let weights = site.weights;
            let dense = site.dense;
            let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], dense);
            let mix = |s: &str| Mix {
                scale: Weight::sym(n(&format!("{s}_scale")), [3], Dtype::F32),
                base: Weight::sym(n(&format!("{s}_base")), [hc_base], Dtype::F32),
                dynamic: Some(Weight::sym(
                    n(&format!("{s}_fn")),
                    [hc_base, hc_fan],
                    Dtype::F32,
                )),
            };
            let (lora_a, lora_b) = crate::adapter::banks(&prefix, ADAPTERS, hidden, dense);

            let ratio = site.ratio;
            let has_indexer = ratio == Some(4);
            let pool = ratio.map(|ratio| {
                // the 2x is the overlap, not a k/v pair: overlap_transform
                // reshapes a [ratio, 2d] block into a [2*ratio, d] window
                // — half the columns serve this block's own positions,
                // half serve the previous block's — so
                // pool_gather_paged's fanout is 2 at ratio 4 and 1 at
                // ratio 128 (which pools its own block alone).
                let entries = if has_indexer { 2 * kv_latent } else { kv_latent };
                Pool {
                    ratio,
                    entries: site.pool.clone(),
                    compressor: Some(compressor(
                        n("compressor"),
                        ratio,
                        entries,
                        kv_latent,
                    )),
                }
            });
            let indexer = has_indexer.then(|| Indexer {
                heads: d.index_heads,
                head_dim: d.index_head_dim,
                top_k: d.index_top_k,
                rope_dim: d.rope_dim,
                theta: d.compress_theta,
                yarn: Some(d.yarn),
                window: d.index_window,
                wq_b: Weight::sym(n("indexer.wq_b"), [idx_w, q_lora], weights),
                weights_proj: Weight::sym(
                    n("indexer.weights_proj"),
                    [d.index_heads as u64, hidden],
                    weights,
                ),
                compressor: compressor(
                    n("indexer.compressor"),
                    ratio.unwrap_or(4),
                    2 * d.index_head_dim as u64,
                    d.index_head_dim as u64,
                ),
                keys: site.index.clone(),
            });

            let gate = if site.hash {
                Gate::Hash {
                    tid2eid: Weight::sym(
                        n("gate.tid2eid"),
                        [d.vocab as u64, d.top_k as u64],
                        Dtype::I64,
                    ),
                }
            } else {
                Gate::Bias {
                    bias: Weight::sym(n("gate.bias"), [site.experts as u64], Dtype::F32),
                }
            };

            Layer {
                attn_mix: mix("attn_mix"),
                attn_norm: Some(norm("attn_norm", hidden)),
                mlp_norm: Some(norm("ffn_norm", hidden)),
                attn: Attn {
                    rope_dim: d.rope_dim,
                    // A compressor layer ropes at the compress theta
                    // under the YaRN ramp; a pure window layer at the
                    // base theta with none (official `Attention.__init__`).
                    theta: if ratio.is_some() { d.compress_theta } else { d.theta },
                    yarn: ratio.map(|_| d.yarn),
                    sm_scale: (d.head_dim as f32).sqrt().recip(),
                    q_down: Weight::sym(n("q_down"), [q_lora, hidden], weights),
                    q_norm: norm("q_norm", q_lora),
                    q_norm_eps: d.norm_eps,
                    q_up: Weight::sym(n("q_up"), [q_w, q_lora], weights).columns(),
                    kv_down: Weight::sym(n("kv_down"), [kv_latent, hidden], weights),
                    kv_norm: norm("kv_norm", kv_latent),
                    kv_norm_eps: d.norm_eps,
                    o_down: Weight::sym(n("o_down"), [o_out, hidden], weights),
                    o_up: Weight::sym(n("o_up"), [hidden, o_out], weights).rows(),
                    o_groups: d.o_groups,
                    // sink rides the activation's dtype, not the
                    // checkpoint's: attention.sink templates the sink
                    // plane on the activation dtype (kernels_metal::
                    // attn::sink dispatches on o.dtype, CUDA the same),
                    // so an f32 plane at a bf16 seat would return NaN.
                    // The checkpoint's own width is the import's
                    // business (matches gpt_oss's convention).
                    sink: Weight::sym(n("attn_sink"), [heads as u64], dense).columns(),
                    kv: site.kv.clone(),
                    pool,
                    indexer,
                },
                mlp_mix: mix("mlp_mix"),
                mlp: Mlp::MoeFlash {
                    router: Weight::sym(n("gate"), [site.experts as u64, hidden], dense),
                    gate,
                    gate_up: if site.split {
                        // two banks, each with its own dtype — both cut
                        // on the same intermediate axis the fused bank
                        // was, so a rank still holds a whole gate row
                        // beside the up row it multiplies.
                        let half = |what: &str, dtype: Dtype| {
                            Weight::sym(
                                n(what),
                                [site.experts as u64, moe_inter as u64, hidden],
                                dtype,
                            )
                            .bank([moe_inter as u64])
                        };
                        GateUp::Split {
                            gate: half("experts_gate", site.gate),
                            up: half("experts_up", site.up),
                        }
                    } else {
                        GateUp::Fused(
                            Weight::sym(
                                n("experts_gate_up"),
                                [site.experts as u64, 2 * moe_inter as u64, hidden],
                                site.gate,
                            )
                            .bank([moe_inter as u64, moe_inter as u64]),
                        )
                    },
                    down: Weight::sym(
                        n("experts_down"),
                        [site.experts as u64, hidden, moe_inter as u64],
                        site.down,
                    )
                    .rows(),
                    shared_gate_up: Weight::sym(
                        n("shared_gate_up"),
                        [2 * shared_inter as u64, hidden],
                        weights,
                    )
                    .packed([shared_inter as u64, shared_inter as u64]),
                    shared_down: Weight::sym(
                        n("shared_down"),
                        [hidden, shared_inter as u64],
                        weights,
                    )
                    .rows(),
                    experts: site.experts,
                    top_k: d.top_k,
                    inter: moe_inter,
                    shared_inter,
                    limit: d.swiglu_limit,
                    renorm: d.renorm,
                    scaling: d.scaling,
                },
                lora_a,
                lora_b,
            }
        };

        let layers = (0..d.layers)
            .map(|l| {
                layer_at(Site {
                    prefix: format!("layer.{l}"),
                    ratio: d.pool[l as usize],
                    hash: l < d.num_hash_layers,
                    experts: d.experts,
                    split: routed.split,
                    gate: routed.gate_of(l),
                    up: routed.up,
                    down: routed.down,
                    weights,
                    dense,
                    kv: format!("kv.{l}"),
                    pool: format!("pool.{l}"),
                    index: format!("index.{l}"),
                })
            })
            .collect();

        // **THE DRAFT HEAD** (`DeepSeek-V4-Flash`'s one `nextn` layer,
        // `mlx-community/DeepSeek-V4-Flash-MTP-bf16` restated by
        // `scripts/dsv4_mtp_companion.py`): the same block at ratio zero with
        // a bias gate, its experts in the companion's own mxfp4 and every
        // dense plane in bf16, behind the two fusion projections and its own
        // hyper head and norm. Its planes come in under `mtp.`, which the
        // `--aux` overlay lands as `aux.`.
        let mtp = d.draft.then(|| {
            let streams_n = d.streams as u64;
            Mtp {
                enorm: Weight::sym("mtp.enorm", [hidden], Dtype::Bf16),
                hnorm: Weight::sym("mtp.hnorm", [hidden], Dtype::Bf16),
                e_proj: Weight::sym("mtp.e_proj", [hidden, hidden], Dtype::Bf16),
                // `h_proj` applies per STREAM: one `[hidden, hidden]` plane
                // read `streams` times over, as the `[streams·hidden,
                // hidden]` block-diagonal bank `linear.matmul_grouped` walks.
                h_proj: Weight::sym("mtp.h_proj", [streams_n * hidden, hidden], Dtype::Bf16),
                block: layer_at(Site {
                    prefix: "mtp.decoder".to_string(),
                    ratio: None,
                    hash: false,
                    experts: DRAFT_EXPERTS,
                    split: true,
                    gate: Dtype::Mxfp4,
                    up: Dtype::Mxfp4,
                    down: Dtype::Mxfp4,
                    weights: Dtype::Bf16,
                    dense: Dtype::Bf16,
                    kv: "kv.mtp".to_string(),
                    pool: "pool.mtp".to_string(),
                    index: "index.mtp".to_string(),
                }),
                hc_head: HcHead {
                    base: Weight::sym("mtp.hc_head.base", [streams], Dtype::F32),
                    dynamic: Weight::sym("mtp.hc_head.fn", [streams, hc_fan], Dtype::F32),
                    scale: Weight::sym("mtp.hc_head.scale", [1], Dtype::F32),
                },
                norm: Weight::sym("mtp.norm", [hidden], Dtype::Bf16),
                norm_eps: d.norm_eps,
            }
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            act,
            heads,
            head_dim: d.head_dim,
            window: d.window,
            adapters: ADAPTERS,
            kv,
            hyper: Hyper {
                streams: d.streams,
                norm_eps: d.norm_eps,
                gate_eps: d.gate_eps,
                alpha: d.alpha,
                sinkhorn: d.sinkhorn,
            },
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: Some(Weight::sym("lm_head", [d.vocab as u64, hidden], weights)),
            hc_head: Some(HcHead {
                base: Weight::sym("hc_head.base", [streams], Dtype::F32),
                dynamic: Weight::sym("hc_head.fn", [streams, hc_fan], Dtype::F32),
                scale: Weight::sym("hc_head.scale", [1], Dtype::F32),
            }),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: idx_norm_eps,
            mtp,
        }
    }
}

/// What every SKU of this family seats. Not a `Dims` field: no pretrained
/// artifact states it; a deployment that wants a different one changes this
/// line and re-traces (load-time recompile, never a runtime extension).
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

/// The draft head's routed bank is always the checkpoint's own 256 experts,
/// whatever miniature the trunk was cut to.
const DRAFT_EXPERTS: u32 = 256;

impl Model {
 }
