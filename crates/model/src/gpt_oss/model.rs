use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor, Yarn};

pub struct Model<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub vocab: u32,
    pub embed: Tensor<W1>,
    pub head: Tensor<W1>,
    pub layers: Vec<Layer<W1, W2>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub struct Layer<W1: Dtype, W2: Dtype> {
    pub attn: Attn<W1>,
    pub attn_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Moe<W1, W2>,
}

pub struct Attn<W1: Dtype> {
    /// The sliding window this layer attends through, or `None` for the
    /// global layers.
    ///
    /// AN `Option<u32>` AND NOT AN `AttnKind`, unlike gemma-4's, and the
    /// difference is what the two enums carry. Gemma's arms hold two
    /// GEOMETRIES — different head widths, different kv head counts,
    /// different rope — and five accessors read them apart. gpt-oss's held
    /// nothing but this number: every other field of the layer is the same
    /// on both kinds, the sole reader called `window()`, and the enum was
    /// `Option` spelled long.
    pub window: Option<u32>,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub sm_scale: f32,
    pub rope: Yarn,
    /// `[q_width + 2 * kv_width, hidden]`, the three projections packed.
    ///
    /// ONE BANK AND ONE MATMUL, where this stated three of each. The
    /// checkpoint ships `q_proj`, `k_proj` and `v_proj` separately and
    /// `Source::Pack` concatenates them at import, which is what `gemma_4`
    /// has always done with the same three names.
    ///
    /// **WHY, MEASURED.** A decode matvec's cost on `driver-metal` is not its
    /// bytes: `k_proj` and `v_proj` are 2.95 MB each and ran at **24% of an
    /// M1 Max's streaming roof**, where the same kernel over the 1.16 GB
    /// lm_head reaches 90%. They are too small to fill the GPU and the barrier
    /// in front of the next dispatch stops them borrowing the room. Packed,
    /// one 29.5 MB matvec does the work of three.
    pub qkv_proj: Tensor<W1>,
    /// `[q_width + 2 * kv_width]` — the three biases packed the same way, so
    /// one `norm.add_bias` follows one matmul.
    pub qkv_bias: Tensor<W1>,
    pub o_proj: Tensor<W1>,
    pub o_bias: Tensor<W1>,
    pub sinks: Tensor<W1>,
    pub kv: CacheRef,
}

pub struct Moe<W1: Dtype, W2: Dtype> {
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    pub router: Tensor<W1>,
    pub router_bias: Tensor<W1>,
    pub gate_up: Tensor<W2>,
    pub gate_up_bias: Tensor<W1>,
    pub down: Tensor<W2>,
    pub down_bias: Tensor<W1>,
}

struct Dims {
    hidden: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_attention_factor: f32,
    yarn_original_max_position: u32,
    window: u32,
    experts: u32,
    top_k: u32,
    inter: u32,
    swiglu_limit: f32,
    swiglu_alpha: f32,
    vocab: u32,
    norm_eps: f32,
}

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// The head counts and the expert intermediate. `head_dim` is one head's own
/// width and does not divide; `experts` is the fan the router scores across
/// and every rank scores the whole of it, because the routed leg is cut
/// INSIDE each expert (`expert_gate_up_bank` packed on its `2 * inter` axis,
/// `expert_down_bank` on its `inter` one) and not across the bank.
///
/// A SEAM THIS CUT DOES NOT CLOSE, and it is a numeric one rather than a
/// width: `o_bias` and `expert_down_bias` are added by the row-parallel
/// statement that precedes `dist.all_reduce`, so a `world`-way deployment
/// sums them `world` times. Every other family in this catalog lands its
/// attention and its feed-forward without a bias and is unaffected. Fixing
/// it is a statement change (a bias that lands after the reduce, or a rank
/// that owns it), which is the driver-side rank story and not this column.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
    Dims {
        q_heads: cut("q_heads", d.q_heads),
        kv_heads: cut("kv_heads", d.kv_heads),
        inter: cut("inter", d.inter),
        ..d
    }
}

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Model<W1, W2, K, TP> {
    /// `openai/gpt-oss-20b`, verified from both sides: `baker_load` joins
    /// 459 of 459 params with every repr truthful against the cached
    /// checkpoint, and the model fires to ARGMAX 11 at 14.4375. The mxfp4
    /// expert stacks go in as the checkpoint's own `_blocks` and `_scales`
    /// — no repack, which is what `axes::Mxfp4` documents.
    pub fn b20() -> Self {
        assemble(Dims {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            theta: 150_000.0,
            yarn_factor: 32.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_attention_factor: 1.346_573_6,
            yarn_original_max_position: 4096,
            window: 128,
            experts: 32,
            top_k: 4,
            inter: 2880,
            swiglu_limit: 7.0,
            swiglu_alpha: 1.702,
            vocab: 201_088,
            norm_eps: 1e-5,
        })
    }

    /// UNVERIFIED against a checkpoint — no gpt-oss-120b is cached. The
    /// dims are the published 120B config and every one of them except
    /// `layers` and `experts` is shared with [`Self::b20`], which HAS been
    /// read; those two are what a file would settle. Q6 measured the
    /// related fact that 20b's tensor names are a SUBSET of 120b's, so a
    /// 120b checkpoint identifies as both rows (membership, not equality).
    pub fn b120() -> Self {
        assemble(Dims {
            hidden: 2880,
            layers: 36,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            theta: 150_000.0,
            yarn_factor: 32.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_attention_factor: 1.346_573_6,
            yarn_original_max_position: 4096,
            window: 128,
            experts: 128,
            top_k: 4,
            inter: 2880,
            swiglu_limit: 7.0,
            swiglu_alpha: 1.702,
            vocab: 201_088,
            norm_eps: 1e-5,
        })
    }
}

fn assemble<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, W2, K, TP> {
    let d = per_rank::<TP>(d);
    let hidden = d.hidden as u64;
    let hd = d.head_dim as u64;
    let q_w = d.q_heads as u64 * hd;
    let kv_w = d.kv_heads as u64 * hd;
    let experts = d.experts as u64;
    let inter = d.inter as u64;
    let sliding_at = |l: u32| l.is_multiple_of(2);

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, w: u64| Norm {
                weight: Tensor::sym(n(s), [w]),
                eps: d.norm_eps,
            };
            Layer {
                attn: Attn {
                    window: sliding_at(l).then_some(d.window),
                    q_heads: d.q_heads,
                    kv_heads: d.kv_heads,
                    head_dim: d.head_dim,
                    sm_scale: (d.head_dim as f32).sqrt().recip(),
                    rope: Yarn {
                        theta: d.theta,
                        factor: d.yarn_factor,
                        beta_fast: d.yarn_beta_fast,
                        beta_slow: d.yarn_beta_slow,
                        attention_factor: d.yarn_attention_factor,
                        original_max_position: d.yarn_original_max_position,
                    },
                    qkv_proj: Tensor::sym(n("qkv_proj"), [q_w + 2 * kv_w, hidden]).columns(),
                    qkv_bias: Tensor::sym(n("qkv_bias"), [q_w + 2 * kv_w]).columns(),
                    o_proj: Tensor::sym(n("o_proj"), [hidden, q_w]).rows(),
                    o_bias: Tensor::sym(n("o_bias"), [hidden]),
                    sinks: Tensor::sym(n("attn_sinks"), [d.q_heads as u64]).columns(),
                    kv: CacheRef::to(format!("kv.{l}")),
                },
                attn_norm: norm("attn_norm", hidden),
                mlp_norm: norm("mlp_norm", hidden),
                mlp: Moe {
                    experts: d.experts,
                    top_k: d.top_k,
                    inter: d.inter,
                    swiglu_limit: d.swiglu_limit,
                    swiglu_alpha: d.swiglu_alpha,
                    router: Tensor::sym(n("router"), [experts, hidden]),
                    router_bias: Tensor::sym(n("router_bias"), [experts]),
                    gate_up: Tensor::sym(n("expert_gate_up_bank"), [experts, 2 * inter, hidden])
                        .bank([inter, inter]),
                    gate_up_bias: Tensor::sym(n("expert_gate_up_bias"), [experts, 2 * inter])
                        .bank([inter, inter]),
                    down: Tensor::sym(n("expert_down_bank"), [experts, hidden, inter]).rows(),
                    down_bias: Tensor::sym(n("expert_down_bias"), [experts, hidden]),
                },
            }
        })
        .collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: Tensor::sym("lm_head", [d.vocab as u64, hidden]),
        layers,
        final_norm: Norm {
            weight: Tensor::sym("final_norm", [hidden]),
            eps: d.norm_eps,
        },
        _kv: PhantomData,
    }
}
