//! The qwen4 model: qwen_3's hybrid mix rebuilt around a non-summing gated residual, plus a hashed n-gram PLE.

use model_dsl::{Dtype, Weight};


pub use crate::qwen_3::model::{Attn, Gdn, Mlp};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// Per rank.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    /// `hc_count` and `hc_lowrank` in the checkpoint config.
    pub streams: u32,
    pub lowrank: u32,

    pub kv: Dtype,
    pub embed: Weight,
    /// Untied from `embed` (`tie_word_embeddings: false`).
    pub head: Weight,
    pub layers: Vec<Layer>,

    /// The final mixer's own norm is the last normalization before the head; there is no separate `model.norm`.
    pub mixer: Residual,

    /// The layer it applies to is recorded inside [`Ple`], not by this field's position.
    pub ple: Option<Ple>,
}

/// One gated-residual site.
///
/// ```text
/// normed = rmsnorm_grouped_plus_one(hyper)                  [S·H]
/// x      = meanₛ(σ(up(silu(down(normed)/S))) ⊙ normed)      [H]
/// …sublayer runs on x…
/// hyper += 2·σ(inject(normed)/S) ⊗ o                        [S·H]
/// ```
pub struct Residual {
    /// `[streams · hidden]`.
    pub norm: Weight,
    /// `[lowrank, streams · hidden]`.
    pub down: Weight,
    /// `[streams · hidden, lowrank]`.
    pub up: Weight,
    /// `[streams, streams · hidden]`; `None` on the final mixer, which never injects.
    pub inject: Option<Weight>,
    pub eps: f32,
}

pub struct Layer {
    pub mixer: Mixer,
    /// No separate input layernorm; this residual site's norm is the only one the sublayer input gets.
    pub attn_res: Residual,
    pub mlp_res: Residual,
    pub mlp: Mlp,
}

pub enum Mixer {
    /// The checkpoint's fused `q_proj` (query + output gate) reads as
    /// `qg_proj` here. Does not run the sparse-selection indexer.
    Attn(Attn),
    Gdn(Gdn),
}

/// The PLE: a hashed n-gram embedding gathered per token, gated per stream,
/// locally mixed by a dilated depthwise convolution, and added into the wide row at one layer.
///
/// ```text
/// e      = embed_concat(ngram_ids(tokens), table)            [E]
/// key    = norm_key(key_proj(e))     value = value_proj(e)   [S·H], [H]
/// gate_s = σ(signed_sqrt(keyₛ · norm_query(hyper)ₛ / √H))
/// g      = gateₛ ⊗ value                                     [S·H]
/// hyper += g + silu(conv₄,dil₃(norm_conv(g)))                [S·H]
/// ```
///
/// Hash constants (`mults`/`primes`/`offsets`) are derived from config, not read from the checkpoint.
pub struct Ple {
    /// Zero-indexed; the config's `ple_layer_ids` is one-indexed.
    pub layer: u32,
    /// `eos_token_id`; the hasher's padding id at sequence starts and eos boundaries.
    pub eos: u32,
    pub heads_per_ngram: u32,
    /// One multiplier per n-gram position; `mults.len()` is `ngram_size`.
    pub mults: Vec<u64>,
    /// Per-head prime vocab size; `offsets` holds each head's row offset in the table.
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// Primes' sum, rounded up to `make_ngram_vocab_size_divisible_by`.
    pub padded_vocab: u64,
    /// `[padded_vocab, embed_dim / heads]`.
    pub table: Weight,
    pub key_proj: Weight,
    pub value_proj: Weight,
    pub norm_key: Weight,
    pub norm_query: Weight,
    pub norm_conv: Weight,
    /// `[streams · hidden, kernel]`, depthwise; dilated by `ngram_size` (tap `j` reads `3·j` positions back).
    pub conv: Weight,
    pub conv_kernel: u32,
    pub dilation: u32,
    pub eps: f32,
    /// Trailing token-id window (`[ngram − 1]`, i32).
    pub ids_state: String,
    /// Convolution history (`[(kernel−1)·dilation + 1, streams·hidden]`).
    pub conv_state: String,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
}

struct PleDims {
    layer: u32,
    heads_per_ngram: u32,
    ngram: u32,
    base_vocab: u64,
    /// `make_ngram_vocab_size_divisible_by`.
    divisible_by: u64,
    /// `split_ngram_parts`; how many shards the table is stored as.
    split_parts: u64,
    seed: u64,
    conv_kernel: u32,
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
    streams: u32,
    lowrank: u32,
    moe: MoeDims,
    ple: Option<PleDims>,
    vocab: u32,
    eos: u32,
    norm_eps: f32,
}

/// Per-role weight dtypes; a single dtype can't express this family's checkpoints
/// (their per-tensor overrides disagree in different directions).
#[derive(Clone, Copy, Debug)]
pub struct Mix {
    /// The token embedding and the (untied) output head.
    pub embed: Dtype,
    /// Dense projections: attention banks, GDN qkv|z/output, shared expert, residual GEMMs, PLE key/value.
    pub proj: Dtype,
    /// Residual injection gates (`block_inject_weight`, `[4, 10240]`); a separate role since conversions disagree on its width.
    pub inject: Dtype,
    /// GDN's `in_proj_b | in_proj_a` pair (`[2·v_heads, hidden]`); also disagreed on by the two conversions.
    pub gdn_ba: Dtype,
    /// Routed expert banks (fused `gate|up` and `down`); most of a layer's bytes live here.
    pub experts: Dtype,
    /// Hashed n-gram table; 160-wide rows can't group by 64, so quantized tables use G32 regardless of the trunk width.
    pub table: Dtype,
}

impl Mix {
    /// Dtype mix for the mixed-4/8 conversion: dense projections raised to 8 bits; experts and table stay at the trunk width.
    #[must_use]
    pub fn of(w: Dtype) -> Mix {
        let proj = match w {
            Dtype::U4g64 => Dtype::U8g64,
            other => crate::dense(other),
        };
        let table = match w {
            Dtype::U4g64 => Dtype::U4g32,
            other => other,
        };
        Mix {
            embed: proj,
            proj,
            // Too narrow for the 8-bit predicate; stay dense while the banks beside them go to 8 bits.
            inject: crate::dense(w),
            gdn_ba: crate::dense(w),
            experts: w,
            table,
        }
    }

    /// The `Mixed-2bit` conversion's own mix, read off its `config.json`.
    pub const MIXED_2BIT: Mix = Mix {
        // Plain BF16, no `.scales` companion.
        embed: Dtype::Bf16,
        proj: Dtype::U4g64,
        inject: Dtype::U4g64,
        gdn_ba: Dtype::U4g64,
        experts: Dtype::U2g128,
        table: Dtype::U4g32,
    };

    /// Dtype the always-unquantized banks (norms, router, shared gate, convolutions) compute in.
    #[must_use]
    pub fn dense(&self) -> Dtype {
        crate::dense(self.proj)
    }
}

impl Model {
    /// The one shipped SKU (`Qwen/Qwen3.8-Flash-Next`). The vision tower and
    /// MTP arm the artifact also publishes are not declared here.
    pub fn flash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::flash_mix(Mix::of(w), kv, tp)
    }

    /// The shipped geometry with its mix stated rather than derived — same
    /// dims [`flash`](Model::flash) builds, over roles a conversion names one
    /// at a time. Exists for the arm [`Mix::of`] cannot reach: the
    /// `Mixed-2bit` conversion, whose exceptions point down from a four-bit
    /// default and whose embedding is not quantized at all.
    pub fn flash_mix(mix: Mix, kv: Dtype, tp: u32) -> Model {
        Model::new(mix, kv, tp, Model::flash_dims())
    }

    /// Geometry for `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit` (published as `mini-l4-e16-p8`).
    pub fn flash_mini(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.layers = 4;
        d.moe.experts = 16;
        let ple = d.ple.as_mut().expect("the flash dims carry a PLE");
        ple.base_vocab = 1_250_000;
        ple.split_parts = 8;
        Model::new(mix, kv, tp, d)
    }

    /// `Qwen/Qwen3.8-Flash-Next`'s `text_config`; both the shipped and mini arms start from this.
    fn flash_dims() -> Dims {
        Dims {
            hidden: 2560,
            layers: 48,
            attn_every: 4,
            q_heads: 24,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            theta: 10_000_000.0,
            k_heads: 16,
            v_heads: 48,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            streams: 4,
            lowrank: 320,
            moe: MoeDims {
                experts: 512,
                top_k: 10,
                inter: 640,
                shared_inter: 640,
            },
            ple: Some(PleDims {
                layer: 1,
                heads_per_ngram: 8,
                ngram: 3,
                base_vocab: 20_000_000,
                divisible_by: 128,
                split_parts: 128,
                seed: 1234,
                conv_kernel: 4,
            }),
            vocab: 248_320,
            eos: 248_044,
            norm_eps: 1e-6,
        }
    }

    /// A qwen4 small enough to run against the reference implementation for parity testing.
    /// No checkpoint ships this configuration.
    pub fn flash_micro(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            Mix::of(w),
            kv,
            tp,
            Dims {
                hidden: 64,
                layers: 4,
                attn_every: 2,
                q_heads: 4,
                kv_heads: 2,
                // Minimum head_dim the shipped attention kernels support.
                head_dim: 64,
                rotary_dim: 16,
                theta: 10_000_000.0,
                k_heads: 2,
                v_heads: 4,
                k_dim: 16,
                v_dim: 16,
                conv_kernel: 4,
                streams: 4,
                lowrank: 16,
                moe: MoeDims {
                    experts: 8,
                    top_k: 2,
                    inter: 32,
                    shared_inter: 32,
                },
                ple: Some(PleDims {
                    // Must be a GDN layer; PLE only rides linear-attention layers.
                    layer: 2,
                    heads_per_ngram: 2,
                    ngram: 3,
                    base_vocab: 1000,
                    divisible_by: 128,
                    split_parts: 128,
                    seed: 1234,
                    conv_kernel: 4,
                }),
                vocab: 256,
                eos: 3,
                norm_eps: 1e-6,
            },
        )
    }

    fn new(mix: Mix, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(tp == 1, "the first qwen4 texts are whole-checkpoint texts");
        let dense = mix.dense();
        // Each role's width is declared by `mix`; nothing is dequantized at load.
        let Mix {
            embed: embed_w,
            proj,
            inject: inject_w,
            gdn_ba,
            experts: experts_w,
            table: narrow_group,
        } = mix;
        let hidden = u64::from(d.hidden);
        let sh = u64::from(d.streams) * hidden;
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

        let residual = |prefix: &str, inject: bool| Residual {
            norm: Weight::sym(format!("{prefix}.norm"), [sh], dense),
            down: Weight::sym(format!("{prefix}.down"), [u64::from(d.lowrank), sh], proj),
            up: Weight::sym(format!("{prefix}.up"), [sh, u64::from(d.lowrank)], proj),
            inject: inject.then(|| {
                Weight::sym(
                    format!("{prefix}.inject"),
                    [u64::from(d.streams), sh],
                    inject_w,
                )
            }),
            eps: d.norm_eps,
        };

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let mixer = if attn_at(l) {
                    let hd = u64::from(d.head_dim);
                    Mixer::Attn(Attn {
                        rotary_dim: d.rotary_dim,
                        theta: d.theta,
                        sm_scale: (d.head_dim as f32).sqrt().recip(),
                        qg_proj: Weight::sym(
                            n("qg_proj"),
                            [2 * u64::from(q_heads) * hd, hidden],
                            proj,
                        )
                        .columns(),
                        k_proj: Weight::sym(n("k_proj"), [u64::from(kv_heads) * hd, hidden], proj)
                            .columns(),
                        v_proj: Weight::sym(n("v_proj"), [u64::from(kv_heads) * hd, hidden], proj)
                            .columns(),
                        o_proj: Weight::sym(n("o_proj"), [hidden, u64::from(q_heads) * hd], proj)
                            .rows(),
                        q_norm: Weight::sym(n("q_norm"), [hd], dense),
                        q_norm_eps: d.norm_eps,
                        k_norm: Weight::sym(n("k_norm"), [hd], dense),
                        k_norm_eps: d.norm_eps,
                        kv: format!("kv.{l}"),
                    })
                } else {
                    let k_heads = d.k_heads / tp;
                    let v_heads = d.v_heads / tp;
                    let k_w = u64::from(k_heads) * u64::from(d.k_dim);
                    let v_w = u64::from(v_heads) * u64::from(d.v_dim);
                    let qkv = u64::from(Gdn::qkv_width(k_heads, v_heads, d.k_dim, d.v_dim));
                    Mixer::Gdn(Gdn {
                        k_heads,
                        v_heads,
                        k_dim: d.k_dim,
                        v_dim: d.v_dim,
                        conv_kernel: d.conv_kernel,
                        in_qkvz: Weight::sym(n("in_qkvz"), [qkv + v_w, hidden], proj)
                            .packed([k_w, k_w, v_w, v_w]),
                        in_ba: Weight::sym(n("in_ba"), [2 * u64::from(v_heads), hidden], gdn_ba)
                            .packed([u64::from(v_heads), u64::from(v_heads)]),
                        conv: Weight::sym(n("conv"), [qkv, u64::from(d.conv_kernel)], dense)
                            .packed([k_w, k_w, v_w]),
                        dt_bias: Weight::sym(n("dt_bias"), [u64::from(v_heads)], dense).columns(),
                        a_log: Weight::sym(n("a_log"), [u64::from(v_heads)], Dtype::F32).columns(),
                        norm: Weight::sym(n("gdn_norm"), [u64::from(d.v_dim)], Dtype::F32),
                        norm_eps: d.norm_eps,
                        out_proj: Weight::sym(n("out_proj"), [hidden, v_w], proj).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let inter = d.moe.inter / tp;
                let shared_inter = d.moe.shared_inter / tp;
                Layer {
                    mixer,
                    attn_res: residual(&n("attn_res"), true),
                    mlp_res: residual(&n("mlp_res"), true),
                    mlp: Mlp::Routed {
                        // Dense: the 8-bit predicate doesn't reach the router or shared gate.
                        router: Weight::sym(n("router"), [u64::from(d.moe.experts), hidden], dense),
                        // gate and up share one dtype/group in both conversions, so this stays a single fused bank.
                        gate_up: Weight::sym(
                            n("experts_gate_up"),
                            [u64::from(d.moe.experts), 2 * u64::from(inter), hidden],
                            experts_w,
                        )
                        .bank([u64::from(inter), u64::from(inter)]),
                        down: Weight::sym(
                            n("experts_down"),
                            [u64::from(d.moe.experts), hidden, u64::from(inter)],
                            experts_w,
                        )
                        .rows(),
                        shared_gate_up: Weight::sym(
                            n("shared_gate_up"),
                            [2 * u64::from(shared_inter), hidden],
                            proj,
                        )
                        .packed([u64::from(shared_inter), u64::from(shared_inter)]),
                        shared_down: Weight::sym(
                            n("shared_down"),
                            [hidden, u64::from(shared_inter)],
                            proj,
                        )
                        .rows(),
                        shared_gate: Weight::sym(n("shared_gate"), [1, hidden], dense),
                        experts: d.moe.experts,
                        top_k: d.moe.top_k,
                        inter,
                        shared_inter,
                    },
                }
            })
            .collect();

        let ple = d.ple.as_ref().map(|p| {
            let (mults, primes, offsets) = hash_constants(p, u64::from(d.vocab));
            let total: u64 = primes.iter().sum();
            let padded_vocab = total.div_ceil(p.divisible_by) * p.divisible_by;
            let heads = u64::from(p.ngram - 1) * u64::from(p.heads_per_ngram);
            let head_width = hidden / heads;
            let shards = p.split_parts;
            Ple {
                layer: p.layer,
                eos: d.eos,
                heads_per_ngram: p.heads_per_ngram,
                mults,
                primes,
                offsets,
                padded_vocab,
                // Sharded into `split_ngram_parts` equal slices; read_concat rejoins them.
                table: Weight::sym("ple.table", [padded_vocab, head_width], narrow_group)
                    .packed(vec![padded_vocab / shards; shards as usize]),
                key_proj: Weight::sym("ple.key_proj", [sh, hidden], proj),
                value_proj: Weight::sym("ple.value_proj", [hidden, hidden], proj),
                norm_key: Weight::sym("ple.norm_key", [sh], dense),
                norm_query: Weight::sym("ple.norm_query", [sh], dense),
                norm_conv: Weight::sym("ple.norm_conv", [sh], dense),
                conv: Weight::sym("ple.conv", [sh, u64::from(p.conv_kernel)], dense),
                conv_kernel: p.conv_kernel,
                dilation: p.ngram,
                eps: d.norm_eps,
                ids_state: "ple.ids".to_string(),
                conv_state: "ple.conv".to_string(),
            }
        });

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            kv_heads,
            head_dim: d.head_dim,
            streams: d.streams,
            lowrank: d.lowrank,
            kv,
            embed: Weight::sym("embed", [u64::from(d.vocab), hidden], embed_w),
            head: Weight::sym("lm_head", [u64::from(d.vocab), hidden], embed_w),
            layers,
            mixer: residual("mixer", false),
            ple,
        }
    }

}

/// Fixed by the checkpoint's own arithmetic, not chosen here.
fn hash_constants(p: &PleDims, vocab: u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
    const PRIME_1: u64 = 10_007;
    fn splitmix64(mut v: u64) -> u64 {
        v = v.wrapping_add(GAMMA);
        v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^ (v >> 31)
    }
    fn is_prime(v: u64) -> bool {
        if v < 2 {
            return false;
        }
        if v.is_multiple_of(2) {
            return v == 2;
        }
        let mut d = 3;
        while d * d <= v {
            if v.is_multiple_of(d) {
                return false;
            }
            d += 2;
        }
        true
    }

    // Single PLE layer, so the layer index is always zero; a second PLE layer would thread it here.
    let _ = PRIME_1;
    let multiplier_max = (i64::MAX as u64) / vocab.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base = p.seed;

    let mults: Vec<u64> = (0..u64::from(p.ngram))
        .map(|i| 2 * (splitmix64(base.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half_bound) + 1)
        .collect();

    let heads = u64::from(p.ngram - 1) * u64::from(p.heads_per_ngram);
    let mut primes = Vec::new();
    let mut offsets = Vec::new();
    let mut total = 0;
    let mut prime = p.base_vocab - 1;
    for _ in 0..heads {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        primes.push(prime);
        offsets.push(total);
        total += prime;
    }
    (mults, primes, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks the derived hash constants against the checkpoint's published buffers.
    #[test]
    fn the_hash_constants_are_the_checkpoints_own() {
        let m = Model::flash(Dtype::Bf16, Dtype::Bf16, 1);
        let p = m.ple.expect("flash carries the PLE");
        assert_eq!(p.mults, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]);
        assert_eq!(
            p.primes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
                20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
                20_000_153, 20_000_159, 20_000_161, 20_000_171,
            ]
        );
        assert_eq!(p.offsets[0], 0);
        assert_eq!(p.offsets[15], 300_001_275);
        assert_eq!(p.padded_vocab, 320_001_536);
        assert_eq!(p.padded_vocab % 128, 0);
    }
}
