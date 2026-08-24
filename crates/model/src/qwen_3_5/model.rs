use std::marker::PhantomData;

use model_dsl::axes::{Dtype, F32, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor};

pub struct Model<W1: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub vocab: u32,
    pub embed: Tensor<W1>,
    pub head: Head<W1>,
    pub layers: Vec<Layer<W1>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub enum Head<W1: Dtype> {
    Tied,
    Bank(Tensor<W1>),
}

pub struct Layer<W1: Dtype> {
    pub mixer: Mixer<W1>,
    pub mixer_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Mlp<W1>,
}

pub enum Mixer<W1: Dtype> {
    Attn(Attn<W1>),
    Gdn(Gdn<W1>),
}

pub struct Attn<W1: Dtype> {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub qg_proj: Tensor<W1>,
    pub k_proj: Tensor<W1>,
    pub v_proj: Tensor<W1>,
    pub o_proj: Tensor<W1>,
    pub q_norm: Norm<W1>,
    pub k_norm: Norm<W1>,
    pub kv: CacheRef,
}

/// TWO OF THIS MIXER'S BANKS ARE f32 AND THE REST ARE `W1`, and that is a
/// statement about the kernels, not a quirk of one file.
///
/// * `a_log` — `ssm.gdn_prep` declares the slot `Const<Self::Tensor<f32>>`
///   and `ssm/gated_delta_net_prep.cuh` reads it through `const float*
///   __restrict__ A_log`, in the same argument list where `dt_bias` is
///   `const T*`.
/// * `norm` — `norm.rmsnorm_gated` declares `weight:
///   Const<Self::Tensor<f32>>`, and `kernels-cuda`'s claiming routine
///   `rmsnorm_gated_fp32_in` takes `Const<Tensor<f32>>`.
///
/// The shipped 35B-A3B checkpoint agrees from the other side: in an
/// otherwise BF16 file, `linear_attn.A_log` is F32 and `linear_attn.norm`
/// is F32 while `linear_attn.dt_bias` is BF16. Declaring these `W1` made
/// the plan's repr column say bf16 for sixty rows the checkpoint stores as
/// f32 — the join reported it, and no cast anyone could write would have
/// been right, because the kernel wants the f32.
pub struct Gdn<W1: Dtype> {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv_kernel: u32,
    pub in_qkvz: Tensor<W1>,
    pub in_ba: Tensor<W1>,
    pub conv: Tensor<W1>,
    pub dt_bias: Tensor<W1>,
    pub a_log: Tensor<F32>,
    pub norm: Norm<F32>,
    pub out_proj: Tensor<W1>,
    pub conv_state: CacheRef,
    pub delta_state: CacheRef,
}

pub enum Mlp<W1: Dtype> {
    Dense {
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        inter: u32,
    },
    Routed {
        router: Tensor<W1>,
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        shared_gate_up: Tensor<W1>,
        shared_down: Tensor<W1>,
        shared_gate: Tensor<W1>,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
    },
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
}

/// An SKU's feed-forward is EITHER dense with an intermediate width OR
/// routed with an expert bank, and it is spelled as a choice because the
/// two numbers do not coexist in any config this family ships. `a3b`'s
/// `text_config` states `moe_intermediate_size` and
/// `shared_expert_intermediate_size` and has NO `intermediate_size` at all
/// (`mlp_only_layers: []`, so not one layer is dense); the dense SKUs state
/// `intermediate_size` and no expert count. A struct with both fields made
/// the routed rows carry a dense width that nothing read and no file
/// stated. `model-legacy`'s `Qwen35MlpKind` is the same enum.
enum MlpDims {
    Dense { inter: u32 },
    Routed(MoeDims),
}

struct Dims {
    hidden: u32,
    layers: u32,
    attn_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_kernel: u32,
    mlp: MlpDims,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
}

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// Heads and intermediates, and nothing else. A head's own width
/// (`head_dim`, `k_dim`, `v_dim`) is the same on every rank — it is the
/// extent a norm and a rope walk, not a count of them — and `hidden`,
/// `vocab` and `experts` name the widths the reduce closes over, the table
/// every rank holds whole, and the fan the router scores across.
///
/// This is one half of a fact whose other half is the shard mark on each
/// tensor: `qg_proj`/`k_proj`/`v_proj` cut their out axis (`q_heads`,
/// `kv_heads`), `o_proj` and `out_proj` cut their in axis, the GDN
/// projections cut every packed segment (`k_heads`, `v_heads`), and both
/// feed-forwards cut their intermediate. `model/tests/
/// a_rank_cut_is_the_shard_column.rs` holds the two halves equal, which is
/// what keeps them one fact.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
    Dims {
        q_heads: cut("q_heads", d.q_heads),
        kv_heads: cut("kv_heads", d.kv_heads),
        k_heads: cut("k_heads", d.k_heads),
        v_heads: cut("v_heads", d.v_heads),
        mlp: match d.mlp {
            MlpDims::Dense { inter } => MlpDims::Dense {
                inter: cut("inter", inter),
            },
            MlpDims::Routed(m) => MlpDims::Routed(MoeDims {
                inter: cut("moe inter", m.inter),
                shared_inter: cut("shared inter", m.shared_inter),
                ..m
            }),
        },
        ..d
    }
}

impl<W1: Dtype, K: KvDtype, const TP: usize> Model<W1, K, TP> {
    /// EVERY NUMBER HERE IS A `config.json` KEY of the shipped
    /// `Qwen/Qwen3.5-35B-A3B`, read out of the cached snapshot's
    /// `text_config` and cross-checked against the safetensors headers:
    ///
    /// | field          | config key                          | value |
    /// |----------------|-------------------------------------|-------|
    /// | `hidden`       | `hidden_size`                       | 2048 |
    /// | `layers`       | `num_hidden_layers`                 | 40 |
    /// | `attn_every`   | `full_attention_interval`           | 4 |
    /// | `q_heads`      | `num_attention_heads`               | 16 |
    /// | `kv_heads`     | `num_key_value_heads`               | 2 |
    /// | `head_dim`     | `head_dim`                          | 256 |
    /// | `rotary_dim`   | `rope_parameters.partial_rotary_factor` × `head_dim` | 0.25 × 256 = 64 |
    /// | `theta`        | `rope_parameters.rope_theta`        | 10 000 000 |
    /// | `k_heads`      | `linear_num_key_heads`              | 16 |
    /// | `v_heads`      | `linear_num_value_heads`            | 32 |
    /// | `k_dim`        | `linear_key_head_dim`               | 128 |
    /// | `v_dim`        | `linear_value_head_dim`             | 128 |
    /// | `conv_kernel`  | `linear_conv_kernel_dim`            | 4 |
    /// | `experts`      | `num_experts`                       | 256 |
    /// | `top_k`        | `num_experts_per_tok`               | 8 |
    /// | `inter`        | `moe_intermediate_size`             | 512 |
    /// | `shared_inter` | `shared_expert_intermediate_size`   | 512 |
    /// | `vocab`        | `vocab_size`                        | 248 320 |
    /// | `tied`         | `tie_word_embeddings` (top level)   | false |
    /// | `norm_eps`     | `rms_norm_eps`                      | 1e-6 |
    ///
    /// FOUR OF THEM WERE WRONG, each in the direction of an older Qwen:
    /// `layers` said 48, `vocab` said 151 936 (Qwen2/3's tokenizer, not the
    /// 248 320 this file's `embed_tokens` and `lm_head` both are),
    /// `experts` said 512 and `top_k` said 10. A fifth number, a dead
    /// `mlp_inter: 5120`, stated a dense width for a model with no dense
    /// layer, and is gone with the field.
    /// `model-legacy/src/qwen_3_5/spec.rs::qwen3_5_35b_a3b` — the
    /// deployment that served this checkpoint — already said 256 experts,
    /// top_k 8, 512/512 intermediates.
    ///
    /// The header arithmetic confirms every derived width: `q_proj`
    /// [8192, 2048] is `2 × q_heads × head_dim` because `attn_output_gate`
    /// is true and the gate rides the same bank; `k_proj`/`v_proj`
    /// [512, 2048] are `kv_heads × head_dim`; `conv1d` [8192, 1, 4] is
    /// `2·k_heads·k_dim + v_heads·v_dim`; `in_proj_z` [4096, 2048] is
    /// `v_heads × v_dim`; `in_proj_a`/`in_proj_b` [32, 2048] are
    /// `v_heads`. `layer_types` puts `full_attention` at 3, 7, … 39,
    /// which is `l % 4 == 3` — what `attn_at` computes.
    ///
    /// NOT MODELLED, and named so the gap is a fact rather than a silence:
    /// `rope_parameters.mrope_interleaved` / `mrope_section [11, 11, 10]`
    /// (the vision tower's rope; the text lane uses plain partial rope),
    /// the whole `vision_config` tower, and `mtp_num_hidden_layers: 1`
    /// (the checkpoint's `mtp.*` rows). All three are checkpoint tensors no
    /// import row reads, which `baker_load` counts out loud.
    pub fn a3b() -> Self {
        assemble(Dims {
            hidden: 2048, layers: 40, attn_every: 4,
            q_heads: 16, kv_heads: 2, head_dim: 256, rotary_dim: 64, theta: 10_000_000.0,
            k_heads: 16, v_heads: 32, k_dim: 128, v_dim: 128, conv_kernel: 4,
            mlp: MlpDims::Routed(MoeDims { experts: 256, top_k: 8, inter: 512, shared_inter: 512 }),
            vocab: 248_320, tied: false, norm_eps: 1e-6,
        })
    }

    pub fn d0_8b() -> Self {
        assemble(Dims {
            hidden: 1024, layers: 24, attn_every: 4,
            q_heads: 8, kv_heads: 2, head_dim: 256, rotary_dim: 64, theta: 10_000_000.0,
            k_heads: 16, v_heads: 16, k_dim: 128, v_dim: 128, conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 3584 },
            vocab: 248_320, tied: true, norm_eps: 1e-6,
        })
    }

    /// UNVERIFIED against a checkpoint — no Qwen3.5-3B is cached, so these
    /// numbers have had no file held against them the way `a3b` and
    /// `d0_8b` now have. `vocab: 151_936` is the specific row to distrust:
    /// it is the number `a3b` was wrong by, and both SKUs that HAVE been
    /// read ship 248 320.
    pub fn d3b() -> Self {
        assemble(Dims {
            hidden: 2048, layers: 24, attn_every: 4,
            q_heads: 16, kv_heads: 2, head_dim: 256, rotary_dim: 64, theta: 10_000_000.0,
            k_heads: 16, v_heads: 32, k_dim: 128, v_dim: 128, conv_kernel: 4,
            mlp: MlpDims::Dense { inter: 8192 },
            vocab: 151_936, tied: true, norm_eps: 1e-6,
        })
    }
}

fn assemble<W1: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, K, TP> {
    let d = per_rank::<TP>(d);
    let hidden = d.hidden as u64;
    let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

    let layers = (0..d.layers).map(|l| {
        let n = |s: &str| format!("layer.{l}.{s}");
        let norm = |s: &str, w: u64| Norm { weight: Tensor::sym(n(s), [w]), eps: d.norm_eps };
        let mixer = if attn_at(l) {
            let hd = d.head_dim as u64;
            Mixer::Attn(Attn {
                q_heads: d.q_heads,
                kv_heads: d.kv_heads,
                head_dim: d.head_dim,
                rotary_dim: d.rotary_dim,
                theta: d.theta,
                sm_scale: (d.head_dim as f32).sqrt().recip(),
                qg_proj: Tensor::sym(n("qg_proj"), [2 * d.q_heads as u64 * hd, hidden]).columns(),
                k_proj: Tensor::sym(n("k_proj"), [d.kv_heads as u64 * hd, hidden]).columns(),
                v_proj: Tensor::sym(n("v_proj"), [d.kv_heads as u64 * hd, hidden]).columns(),
                o_proj: Tensor::sym(n("o_proj"), [hidden, d.q_heads as u64 * hd]).rows(),
                q_norm: norm("q_norm", hd),
                k_norm: norm("k_norm", hd),
                kv: CacheRef::to(format!("kv.{l}")),
            })
        } else {
            let k_w = d.k_heads as u64 * d.k_dim as u64;
            let v_w = d.v_heads as u64 * d.v_dim as u64;
            let qkv = 2 * k_w + v_w;
            let qkvz = qkv + v_w;
            Mixer::Gdn(Gdn {
                k_heads: d.k_heads,
                v_heads: d.v_heads,
                k_dim: d.k_dim,
                v_dim: d.v_dim,
                conv_kernel: d.conv_kernel,
                in_qkvz: Tensor::sym(n("in_qkvz"), [qkvz, hidden]).packed([k_w, k_w, v_w, v_w]),
                in_ba: Tensor::sym(n("in_ba"), [2 * d.v_heads as u64, hidden]).packed([d.v_heads as u64, d.v_heads as u64]),
                conv: Tensor::sym(n("conv"), [qkv, d.conv_kernel as u64]).packed([k_w, k_w, v_w]),
                dt_bias: Tensor::sym(n("dt_bias"), [d.v_heads as u64]).columns(),
                a_log: Tensor::<F32>::sym(n("a_log"), [d.v_heads as u64]).columns(),
                norm: Norm {
                    weight: Tensor::<F32>::sym(n("gdn_norm"), [d.v_dim as u64]),
                    eps: d.norm_eps,
                },
                out_proj: Tensor::sym(n("out_proj"), [hidden, d.v_heads as u64 * d.v_dim as u64]).rows(),
                conv_state: CacheRef::to(format!("conv.{l}")),
                delta_state: CacheRef::to(format!("delta.{l}")),
            })
        };
        let mlp = match &d.mlp {
            MlpDims::Dense { inter } => Mlp::Dense {
                gate_up: Tensor::sym(n("gate_up"), [2 * *inter as u64, hidden]).packed([*inter as u64, *inter as u64]),
                down: Tensor::sym(n("down"), [hidden, *inter as u64]).rows(),
                inter: *inter,
            },
            MlpDims::Routed(m) => Mlp::Routed {
                router: Tensor::sym(n("router"), [m.experts as u64, hidden]),
                // `[E, out, in]`, which is what the KERNEL reads and not a
                // convention picked here. `moe.matmul_select` declares
                // "`bank` is the `[E, N, K]` stack", and
                // `moe/moe_grouped_gemm.cuh` indexes it
                // `weight_base + e*N*K + n*K` with the b-fragment loaded
                // `col_major` at `ld = K` — its own words: *"W is [N, K]
                // row-major, and W^T is [K, N]; a [K, N] column-major view
                // of W^T is exactly W's own memory with leading dimension
                // K"*. So N is the output width and K the input width.
                gate_up: Tensor::sym(n("experts_gate_up"), [m.experts as u64, 2 * m.inter as u64, hidden]).bank([m.inter as u64, m.inter as u64]),
                down: Tensor::sym(n("experts_down"), [m.experts as u64, hidden, m.inter as u64]).rows(),
                shared_gate_up: Tensor::sym(n("shared_gate_up"), [2 * m.shared_inter as u64, hidden]).packed([m.shared_inter as u64, m.shared_inter as u64]),
                shared_down: Tensor::sym(n("shared_down"), [hidden, m.shared_inter as u64]).rows(),
                shared_gate: Tensor::sym(n("shared_gate"), [1, hidden]),
                experts: m.experts,
                top_k: m.top_k,
                inter: m.inter,
                shared_inter: m.shared_inter,
            },
        };
        Layer {
            mixer,
            mixer_norm: norm("mixer_norm", hidden),
            mlp_norm: norm("mlp_norm", hidden),
            mlp,
        }
    }).collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: if d.tied {
            Head::Tied
        } else {
            Head::Bank(Tensor::sym("lm_head", [d.vocab as u64, hidden]))
        },
        layers,
        final_norm: Norm { weight: Tensor::sym("final_norm", [hidden]), eps: d.norm_eps },
        _kv: PhantomData,
    }
}
