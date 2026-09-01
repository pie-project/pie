use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub act: Dtype,

    /// The one attention reading this text is carved for. Every layer reads
    /// the same schedule, so the numbers that schedule states — query heads,
    /// kv heads, head width, window — are facts about the model, not about a
    /// layer. The latent plane is shared across heads, so the kv heads a
    /// reader restates are these query `heads`.
    pub heads: u32,
    pub head_dim: u32,
    pub window: u32,

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
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
}

pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
}

/// The trunk hyper-connection head: `base [streams]`, `fn [streams,
/// streams*hidden]`, `scale [1]` (config `hc_head`). The dynamic `fn` plane is
/// the per-token mixing function; today it is stated and read but its fusion
/// into the gate op is deferred (see `forward::flash`).
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
    /// `[slots, hidden, rank]` — the down and up planes of one correction site
    /// (palo design §8, campaign A-6).
    ///
    /// **THE SITE IS THE ATTENTION SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is the gated
    /// stream `hc_gates` hands the mixer, and the output is what `o_up`
    /// answers — which is already past `all_reduce`, because this text reduces
    /// between its two output projections (`o_down` is rows-cut, `o_up` is
    /// replicated). A correction stated before the reduce would read a rows-cut
    /// partial product, and every rank would contribute the whole `ΔW·x` for
    /// the sum to carry `tp` times.
    ///
    /// AND BEFORE `hc_fold`, not after: the fold writes the correction into
    /// the hyper-connection streams the way it writes the mixer's own output,
    /// so a corrected site stays one site instead of becoming a second
    /// contribution with its own mixing weights.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub struct Attn {
    pub rope_dim: u32,
    pub theta: f32,
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
    /// The NSA sparse-selection indexer, present on the ratio-4 layers only
    /// (`self_attn.indexer.*`). Its top-k narrows THIS LAYER'S COMPRESSED
    /// BRANCH: the keys it ranks are its own compressor's pooled entries, one
    /// per ratio-4 block and in 1:1 correspondence with the attention
    /// compressor's, and `attention.pool_lse_selected` reads the rows it
    /// chose. The ratio-128 layers carry none because `S / 128` is a key set
    /// no budget needs to cap.
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

/// **THE ROUTED EXPERTS' TWO PROJECTIONS, FUSED OR NOT.**
///
/// [`Fused`](GateUp::Fused) is the cheaper form and the one every SKU of this
/// family said until the DQ artifact arrived: ONE `[experts, 2·inter, hidden]`
/// bank read as the concat of the stored `gate_proj` and `up_proj`, one routed
/// matmul, one packed swiglu-clamp.
///
/// [`Split`](GateUp::Split) is the form an artifact can force. A [`Weight`]
/// carries ONE `Dtype`, and `Dtype` is where an MLX affine GROUP is written
/// down; `mlx-community/DeepSeek-V4-Flash-2bit-DQ` stores its routed
/// `gate_proj` at group 32 and its `up_proj` at group 64 on four of its five
/// layers, so the two halves' `.scales` planes are 2048 rows of 128 beside
/// 2048 rows of 64 and join into NO rectangle at any axis. The fused
/// declaration is not awkward there; it is unstateable. Split, each half is
/// its own bank with its own dtype, the routed matmul fires twice, and
/// `linear.mlp_swiglu_clamp_split` combines the pair.
///
/// **AND ONLY THE ROWS THAT NEED IT SAY IT.** Splitting costs a second routed
/// launch and a second bank's residency for nothing when both halves share a
/// point, so every bf16 and every uniform-4-bit row keeps `Fused` and keeps
/// its trace byte for byte.
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

/// The flash geometry, whose extra facts the toy `Dims` has no field for: the
/// MLA latent width and o-projection groups, the two hyper planes' `fn`, the
/// indexer, and the shared expert. It is a superset built beside the toy so
/// the toy `new` stays byte-identical.
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

/// The full DeepSeek-V4-Flash `compress_ratios`, one per layer. Layers 0, 1
/// and the last carry no compressor (ratio 0); twenty-one layers carry a
/// compressor and an indexer (ratio 4); the remaining nineteen carry a
/// compressor only (ratio 128). The COUNTS are the artifact's; the block
/// ORDER here is a placeholder cadence — the mini snapshot fixes only the
/// first four layers and the last, and the full artifact's exact interleave of
/// the ratio-4/ratio-128 middle is not what a bf16 structure text is held to.
#[rustfmt::skip]
const FLASH_RATIOS: [Option<u32>; 43] = [
    // layers 0,1: ratio 0 (no compressor)
    None, None,
    // 21 layers: ratio 4 (compressor + indexer)
    Some(4), Some(4), Some(4), Some(4), Some(4), Some(4), Some(4),
    Some(4), Some(4), Some(4), Some(4), Some(4), Some(4), Some(4),
    Some(4), Some(4), Some(4), Some(4), Some(4), Some(4), Some(4),
    // 19 layers: ratio 128 (compressor only)
    Some(128), Some(128), Some(128), Some(128), Some(128), Some(128), Some(128),
    Some(128), Some(128), Some(128), Some(128), Some(128), Some(128), Some(128),
    Some(128), Some(128), Some(128), Some(128), Some(128),
    // layer 42: ratio 0 (no compressor)
    None,
];

/// The mini `mlx-community/DeepSeek-V4-Flash-2bit-DQ` snapshot's five-layer
/// schedule (original layers 0, 1, 2, 3, 42, renumbered), which the name
/// bijection test holds this text against.
const FLASH_MICRO_RATIOS: [Option<u32>; 5] =
    [None, None, Some(4), Some(128), Some(4)];

/// **WHAT THE ROUTED EXPERT BLOCK IS STORED AS**, which one `weights` dtype
/// cannot always say.
///
/// A DQ conversion states its quantization PER TENSOR: the artifact's
/// `config.json` carries one default `(bits, group)` and then a list of
/// overrides, and the routed experts are where the overrides land because the
/// routed experts are where the bytes are. So this mirrors that shape — one
/// representation per projection plus the per-layer exceptions the file states
/// — rather than pretending a single dtype covers the block. Everything else
/// in the text still reads the trunk's `weights`, which is what the overrides
/// are exceptions TO.
#[derive(Clone, Copy, Debug)]
pub struct Routed {
    /// The routed gate projection's representation.
    pub gate: Dtype,
    /// The layers whose gate is NOT [`gate`](Routed::gate), read exactly as
    /// the conversion's per-tensor overrides read. **THIS IS THE LANDMINE**:
    /// on the 2-bit DQ artifact the routed `gate_proj` groups by 32 on layers
    /// 0-3 and by 64 on the LAST — original layer 42, renumbered to 4 — while
    /// `up_proj` and `down_proj` group by 64 throughout. One layer is the
    /// difference between a text that reads this artifact and one that reads
    /// four fifths of it.
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
    /// The whole block in the trunk's own representation, fused — what every
    /// row that is not a per-tensor mix says, and the reading under which this
    /// type is invisible.
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
        Model::new_flash(
            w,
            Routed::uniform(w),
            act,
            kv,
            tp,
            Model::flash_dims(43, &FLASH_RATIOS, 3),
        )
    }

    /// **THE MINI DQ SNAPSHOT'S OWN GEOMETRY** — the real DeepSeek-V4-Flash
    /// everything (hidden 4096, 64 MLA heads of width 512, moe_inter 2048, the
    /// 129 280-token vocabulary) over the FIVE renumbered layers and SIXTEEN
    /// routed experts `mlx-community/DeepSeek-V4-Flash-2bit-DQ` publishes.
    ///
    /// **NOT [`flash_micro`](Model::flash_micro), and the difference is what
    /// each is for.** `flash_micro` shrinks every dimension so the trace is
    /// cheap; it exists to hold this family's NAMES against the snapshot's
    /// census, and its shapes are nobody's. This one is a text a checkpoint can
    /// actually be read through: the names are the same and the rectangles are
    /// the file's.
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
        // `heads * head_dim == o_groups * hidden` (8·64 == 2·256), the o-group
        // invariant the real geometry meets at 64·512 == 8·4096.
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

        // **WHAT A QUANTIZED SKU'S NEIGHBOURS ARE.** Norms, the compressor's
        // absolute position embedding and the routed ROUTER are shipped
        // unquantized by every MLX conversion of this family — a 16-row gate
        // and a 4096-wide norm are not where a checkpoint's bytes are — so
        // they are stated in what the banks beside them COMPUTE in, which is
        // exactly what `model_dsl::compute_dtype` answers. At `weights =
        // bf16` this is `weights`, so every row that shipped before this line
        // existed declares what it declared.
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
            // **THE POSITION PLANE RIDES WHAT THE GATHER READS, WHICH IS
            // f32.** It is added to the gate LOGITS before their softmax —
            // `pool_gather_paged` takes it as `const device float*` and the
            // CUDA twin as `const float*` — so f32 is the width both shaders
            // agree on, and the artifact's own element is the import's
            // business, exactly as `attn_sink`'s note one screen down argues
            // in the other direction.
            ape: Weight::sym(format!("{prefix}.ape"), [ratio as u64, entries], Dtype::F32),
            norm: Weight::sym(format!("{prefix}.norm"), [norm_w], dense),
            norm_eps: d.norm_eps,
        };

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
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
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);

                let ratio = d.pool[l as usize];
                let has_indexer = ratio == Some(4);
                let pool = ratio.map(|ratio| {
                    // **THE `2 x` IS THE OVERLAP AND NOT A k/v PAIR.** The
                    // reference's `overlap_transform` reshapes a
                    // `[ratio, 2d]` block into a `[2 * ratio, d]` WINDOW —
                    // the second half of a row's columns serves the block's
                    // own positions and the first half serves the previous
                    // block's, so a pooled entry straddles two blocks
                    // (`v4mlx/compressor.py`). `pool_gather_paged`'s
                    // `col = (i >= ratio ? head_dim : 0) + d` is that
                    // reshape, and `coff` is the fanout: 2 at ratio 4, and 1
                    // at ratio 128, which pools its own block alone.
                    let entries = if has_indexer { 2 * kv_latent } else { kv_latent };
                    Pool {
                        ratio,
                        entries: format!("pool.{l}"),
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
                    keys: format!("index.{l}"),
                });

                let gate = if l < d.num_hash_layers {
                    Gate::Hash {
                        tid2eid: Weight::sym(
                            n("gate.tid2eid"),
                            [d.vocab as u64, d.top_k as u64],
                            Dtype::I64,
                        ),
                    }
                } else {
                    Gate::Bias {
                        bias: Weight::sym(n("gate.bias"), [d.experts as u64], Dtype::F32),
                    }
                };

                Layer {
                    attn_mix: mix("attn_mix"),
                    attn_norm: Some(norm("attn_norm", hidden)),
                    mlp_norm: Some(norm("ffn_norm", hidden)),
                    attn: Attn {
                        rope_dim: d.rope_dim,
                        theta: d.theta,
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
                        // **THE SINK RIDES THE ACTIVATION'S DTYPE, NOT THE
                        // CHECKPOINT'S.** The artifact ships `attn.attn_sink`
                        // as f32 and this text declared f32 to match it — but
                        // `attention.sink` templates its sink plane on the
                        // ACTIVATION (`kernels_metal::attn::sink` dispatches
                        // on `o.dtype`, and the CUDA twin on the same), so an
                        // f32 plane arrived at a `bfloat*` seat and every
                        // logit came back NaN. The checkpoint's own width is
                        // the import's business — a bf16 declaration reads the
                        // f32 tensor through an honest cast — and gpt_oss has
                        // always declared its `attn_sinks` this way.
                        sink: Weight::sym(n("attn_sink"), [heads as u64], dense).columns(),
                        kv: format!("kv.{l}"),
                        pool,
                        indexer,
                    },
                    mlp_mix: mix("mlp_mix"),
                    mlp: Mlp::MoeFlash {
                        router: Weight::sym(n("gate"), [d.experts as u64, hidden], dense),
                        gate,
                        gate_up: if routed.split {
                            // Two banks, EACH WITH ITS OWN DTYPE — the whole
                            // point of the form. Both are cut on the same
                            // intermediate axis the fused bank was cut on, at
                            // its one remaining seam, so a rank still holds a
                            // whole gate row beside the up row it multiplies.
                            let half = |what: &str, dtype: Dtype| {
                                Weight::sym(
                                    n(what),
                                    [d.experts as u64, moe_inter as u64, hidden],
                                    dtype,
                                )
                                .bank([moe_inter as u64])
                            };
                            GateUp::Split {
                                gate: half("experts_gate", routed.gate_of(l)),
                                up: half("experts_up", routed.up),
                            }
                        } else {
                            GateUp::Fused(
                                Weight::sym(
                                    n("experts_gate_up"),
                                    [d.experts as u64, 2 * moe_inter as u64, hidden],
                                    routed.gate,
                                )
                                .bank([moe_inter as u64, moe_inter as u64]),
                            )
                        },
                        down: Weight::sym(
                            n("experts_down"),
                            [d.experts as u64, hidden, moe_inter as u64],
                            routed.down,
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
                        experts: d.experts,
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
            head: Some(Weight::sym("lm_head", [d.vocab as u64, hidden], weights)),
            hc_head: Some(HcHead {
                base: Weight::sym("hc_head.base", [streams], Dtype::F32),
                dynamic: Weight::sym("hc_head.fn", [streams, hc_fan], Dtype::F32),
                scale: Weight::sym("hc_head.scale", [1], Dtype::F32),
            }),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: idx_norm_eps,
        }
    }
}

/// What every SKU of this family seats.
///
/// Not a `Dims` field, because it is not a fact about the checkpoint the way
/// `hidden` and `layers` are — no pretrained artifact states it. It is the
/// DEPLOYMENT's ceiling written where a shape has to be written, and a
/// deployment that wants a different one changes this line and re-traces,
/// which is exactly the "load-time recompile, never a runtime extension"
/// design §9 asks for.
///
/// Eight slots of rank sixteen costs the `base` row 1 MiB a layer — two planes
/// of `8 x 16 x 2048` in the compute element — and 6 MiB over its six.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        let mut b = checkpoint_dsl::Builder::new(src, self.tp);
        let mut stated = |w: &Weight| b.read_own(w);
        stated(&self.embed)?;
        if let Some(head) = &self.head {
            stated(head)?;
        }
        stated(&self.final_norm)?;
        if let Some(hc) = &self.hc_head {
            stated(&hc.base)?;
            stated(&hc.dynamic)?;
            stated(&hc.scale)?;
        }

        for layer in &self.layers {
            for mix in [&layer.attn_mix, &layer.mlp_mix] {
                stated(&mix.scale)?;
                stated(&mix.base)?;
                if let Some(dynamic) = &mix.dynamic {
                    stated(dynamic)?;
                }
            }
            if let Some(n) = &layer.attn_norm {
                stated(n)?;
            }
            if let Some(n) = &layer.mlp_norm {
                stated(n)?;
            }

            let at = &layer.attn;
            stated(&at.q_down)?;
            stated(&at.q_norm)?;
            stated(&at.q_up)?;
            stated(&at.kv_down)?;
            stated(&at.kv_norm)?;
            stated(&at.o_down)?;
            stated(&at.o_up)?;
            stated(&at.sink)?;
            if let Some(pool) = &at.pool {
                if let Some(c) = &pool.compressor {
                    stated(&c.wkv)?;
                    stated(&c.wgate)?;
                    stated(&c.ape)?;
                    stated(&c.norm)?;
                }
            }
            if let Some(ix) = &at.indexer {
                stated(&ix.wq_b)?;
                stated(&ix.weights_proj)?;
                stated(&ix.compressor.wkv)?;
                stated(&ix.compressor.wgate)?;
                stated(&ix.compressor.ape)?;
                stated(&ix.compressor.norm)?;
            }

            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    stated(gate_up)?;
                    stated(down)?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    ..
                } => {
                    stated(router)?;
                    stated(bias)?;
                    stated(gate_up)?;
                    stated(down)?;
                }
                Mlp::MoeFlash {
                    router,
                    gate,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    ..
                } => {
                    // **THE HASH LAYERS STATE THE TABLE AND NOT THE ROUTER**,
                    // which is the exact inverse of what this arm said while
                    // `linear.moe_hash_route` was missing. A lookup gate
                    // computes no logits, so `ffn.gate.weight` is a plane the
                    // forward never reads on those layers and a load that
                    // published it would hand `Shell::load` a plane no plan
                    // names; `tid2eid` is what the route now comes out of, so
                    // it is what the contract demands.
                    match gate {
                        Gate::Hash { tid2eid } => stated(tid2eid)?,
                        Gate::Bias { bias } => {
                            stated(router)?;
                            stated(bias)?;
                        }
                    }
                    match gate_up {
                        GateUp::Fused(bank) => stated(bank)?,
                        GateUp::Split { gate, up } => {
                            stated(gate)?;
                            stated(up)?;
                        }
                    }
                    stated(down)?;
                    stated(shared_gate_up)?;
                    stated(shared_down)?;
                }
            }
        }

        Ok(b.build())
    }
}
