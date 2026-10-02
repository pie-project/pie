use model_dsl::{Dtype, Weight};

pub use crate::qwen_3::model::{Attn, Gdn, Merger, Mlp, Tower, TowerBlock};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    pub streams: u32,
    pub lowrank: u32,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,

    pub mixer: Residual,

    pub ple: Option<Ple>,

    pub mtp: Option<Mtp>,

    pub tower: Option<Tower>,
}

#[derive(Clone, Copy)]
struct TowerDims {
    depth: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    patch_width: u32,
    merge: u32,
    positions: u32,
    out_hidden: u32,
    theta: f32,
    norm_eps: f32,
    taps: u32,
}

impl TowerDims {
    const fn flash_next() -> TowerDims {
        TowerDims {
            depth: 27,
            hidden: 1152,
            heads: 16,
            inter: 4304,
            patch_width: 1536,
            merge: 2,
            positions: 2304,
            out_hidden: 2560,
            theta: 10_000.0,
            norm_eps: 1e-6,
            taps: 4,
        }
    }
}

pub struct Mtp {
    pub norm_embed: Weight,
    pub norm_hidden: Weight,
    pub fc_embed: Weight,
    pub fc_hidden: Weight,
    pub block: Layer,
    pub mixer: Residual,
    pub eps: f32,
    pub depth: u32,
}

pub const DRAFT_DEPTH: u32 = 2;

pub struct Residual {
    pub norm: Weight,
    pub down: Weight,
    pub up: Weight,
    pub inject: Option<Weight>,
    pub eps: f32,
}

pub struct Layer {
    pub mixer: Mixer,
    pub attn_res: Residual,
    pub mlp_res: Residual,
    pub mlp: Mlp,
}

#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Attn {
        attn: Attn,
        indexer: Option<Indexer>,
    },
    Gdn(Gdn),
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub ratio: u32,
    pub top_k: u32,
    pub norm_eps: f32,
    pub qk_proj: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub keys: String,
}

pub struct Ple {
    pub layer: u32,
    pub eos: u32,
    pub heads_per_ngram: u32,
    pub mults: Vec<u64>,
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub padded_vocab: u64,
    pub table: Weight,
    pub key_proj: Weight,
    pub value_proj: Weight,
    pub norm_key: Weight,
    pub norm_query: Weight,
    pub norm_conv: Weight,
    pub conv: Weight,
    pub conv_kernel: u32,
    pub dilation: u32,
    pub eps: f32,
    pub ids_state: String,
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
    divisible_by: u64,
    split_parts: u64,
    seed: u64,
    conv_kernel: u32,
}

#[derive(Clone, Copy)]
struct IndexerDims {
    heads: u32,
    head_dim: u32,
    budget: u32,
    ratio: u32,
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
    indexer: Option<IndexerDims>,
    vocab: u32,
    eos: u32,
    norm_eps: f32,
    draft: bool,
    tower: Option<TowerDims>,
}

#[derive(Clone, Copy, Debug)]
pub struct Mix {
    pub embed: Dtype,
    pub proj: Dtype,
    pub inject: Dtype,
    pub gdn_ba: Dtype,
    pub experts: Dtype,
    pub head: Dtype,
    pub table: Dtype,
}

impl Mix {
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
            head: proj,
            proj,
            inject: crate::dense(w),
            gdn_ba: crate::dense(w),
            experts: w,
            table,
        }
    }

    pub const MIXED_2BIT: Mix = Mix {
        embed: Dtype::Bf16,
        head: Dtype::U4g64,
        proj: Dtype::U4g64,
        inject: Dtype::U4g64,
        gdn_ba: Dtype::U4g64,
        experts: Dtype::U2g128,
        table: Dtype::U4g32,
    };

    #[must_use]
    pub fn dense(&self) -> Dtype {
        crate::dense(self.proj)
    }
}

impl Model {
    pub fn flash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::flash_mix(Mix::of(w), kv, tp)
    }

    pub fn flash_mix(mix: Mix, kv: Dtype, tp: u32) -> Model {
        Model::new(mix, kv, tp, Model::flash_dims())
    }

    pub fn flash_mini(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.layers = 4;
        d.moe.experts = 16;
        let ple = d.ple.as_mut().expect("the flash dims carry a PLE");
        ple.base_vocab = 1_250_000;
        ple.split_parts = 8;
        Model::new(mix, kv, tp, d)
    }

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
            indexer: Some(IndexerDims {
                heads: 4,
                head_dim: 128,
                budget: 2048,
                ratio: 4,
            }),
            vocab: 248_320,
            eos: 248_044,
            norm_eps: 1e-6,
            draft: false,
            tower: None,
        }
    }

    pub fn flash_mix_mtp(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.draft = true;
        Model::new(mix, kv, tp, d)
    }

    pub fn flash_mix_vision(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.tower = Some(TowerDims::flash_next());
        Model::new(mix, kv, tp, d)
    }

    pub fn flash_mix_mtp_vision(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.draft = true;
        d.tower = Some(TowerDims::flash_next());
        Model::new(mix, kv, tp, d)
    }

    pub fn flash_micro(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(Mix::of(w), kv, tp, Model::flash_micro_dims())
    }

    fn flash_micro_dims() -> Dims {
        Dims {
            hidden: 64,
            layers: 4,
            attn_every: 2,
            q_heads: 4,
            kv_heads: 2,
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
                layer: 2,
                heads_per_ngram: 2,
                ngram: 3,
                base_vocab: 1000,
                divisible_by: 128,
                split_parts: 128,
                seed: 1234,
                conv_kernel: 4,
            }),
            indexer: Some(IndexerDims {
                heads: 2,
                head_dim: 16,
                budget: 32,
                ratio: 4,
            }),
            vocab: 256,
            eos: 3,
            norm_eps: 1e-6,
            draft: false,
            tower: None,
        }
    }

    fn new(mix: Mix, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(tp == 1, "the first qwen4 texts are whole-checkpoint texts");
        let dense = mix.dense();
        let Mix {
            embed: embed_w,
            head: head_w,
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

        let block = |n: &dyn Fn(&str) -> String,
                     attn: bool,
                     kv_name: String,
                     index_name: Option<String>,
                     conv_name: String,
                     delta_name: String,
                     experts_w: Dtype|
         -> Layer {
            {
                let mixer = if attn {
                    let hd = u64::from(d.head_dim);
                    let indexer = index_name.zip(d.indexer).map(|(keys, ix)| {
                        assert!(
                            ix.budget.is_multiple_of(ix.ratio),
                            "a token budget of {} is not a whole number of {}-token blocks",
                            ix.budget,
                            ix.ratio,
                        );
                        let ihd = u64::from(ix.head_dim);
                        let q_w = u64::from(ix.heads) * ihd;
                        Indexer {
                            heads: ix.heads,
                            head_dim: ix.head_dim,
                            ratio: ix.ratio,
                            top_k: ix.budget / ix.ratio,
                            norm_eps: d.norm_eps,
                            qk_proj: Weight::sym(n("index_qk_proj"), [q_w + ihd, hidden], proj)
                                .packed([q_w, ihd]),
                            q_norm: Weight::sym(n("index_q_norm"), [ihd], dense),
                            k_norm: Weight::sym(n("index_k_norm"), [ihd], dense),
                            keys,
                        }
                    });
                    let attn = Attn {
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
                        kv: kv_name,
                    };
                    Mixer::Attn { attn, indexer }
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
                        conv_state: conv_name,
                        delta_state: delta_name,
                    })
                };
                let inter = d.moe.inter / tp;
                let shared_inter = d.moe.shared_inter / tp;
                Layer {
                    mixer,
                    attn_res: residual(&n("attn_res"), true),
                    mlp_res: residual(&n("mlp_res"), true),
                    mlp: Mlp::Routed {
                        router: Weight::sym(n("router"), [u64::from(d.moe.experts), hidden], dense),
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
            }
        };
        let layers = (0..d.layers)
            .map(|l| {
                block(
                    &|s: &str| format!("layer.{l}.{s}"),
                    attn_at(l),
                    format!("kv.{l}"),
                    attn_at(l).then(|| format!("index.{l}")),
                    format!("conv.{l}"),
                    format!("delta.{l}"),
                    experts_w,
                )
            })
            .collect();
        let tower = d.tower.map(|t| {
            assert_eq!(
                t.out_hidden, d.hidden,
                "a tower's `out_hidden_size` is the TRUNK's width — the merger's \
                 answer is a token row, and a mismatch would scatter a rectangle \
                 of the wrong width into the embedding"
            );
            assert_eq!(
                t.hidden % t.heads,
                0,
                "a {}-wide tower does not divide into {} heads",
                t.hidden,
                t.heads
            );
            let th = u64::from(t.hidden);
            let ti = u64::from(t.inter);
            let merged = u64::from(t.merge) * u64::from(t.merge) * th;
            let head_dim = t.hidden / t.heads;
            let n = |s: String| format!("visual.{s}");
            let plane = |s: String, dims: [u64; 2]| Weight::sym(n(s), dims, dense);
            let vec1 = |s: String, len: u64| Weight::sym(n(s), [len], dense);
            Tower {
                hidden: t.hidden,
                heads: t.heads,
                head_dim,
                merge: t.merge,
                patch_width: t.patch_width,
                taps: t.taps,
                positions: t.positions,
                theta: t.theta,
                norm_eps: t.norm_eps,
                sm_scale: (head_dim as f32).sqrt().recip(),
                patch_embed: plane("patch_embed".into(), [th, u64::from(t.patch_width)]),
                patch_embed_bias: vec1("patch_embed_bias".into(), th),
                pos_embed: plane("pos_embed".into(), [u64::from(t.positions), th]),
                blocks: (0..t.depth)
                    .map(|l| {
                        let b = |s: &str| format!("block.{l}.{s}");
                        TowerBlock {
                            norm1: vec1(b("norm1"), th),
                            norm1_bias: vec1(b("norm1_bias"), th),
                            qkv: plane(b("qkv"), [3 * th, th]),
                            qkv_bias: vec1(b("qkv_bias"), 3 * th),
                            proj: plane(b("proj"), [th, th]),
                            proj_bias: vec1(b("proj_bias"), th),
                            norm2: vec1(b("norm2"), th),
                            norm2_bias: vec1(b("norm2_bias"), th),
                            fc1: plane(b("fc1"), [ti, th]),
                            fc1_bias: vec1(b("fc1_bias"), ti),
                            fc2: plane(b("fc2"), [th, ti]),
                            fc2_bias: vec1(b("fc2_bias"), th),
                        }
                    })
                    .collect(),
                merger: Merger {
                    norm: vec1("merger_norm".into(), th),
                    norm_bias: vec1("merger_norm_bias".into(), th),
                    fc1: plane("merger_fc1".into(), [merged, merged]),
                    fc1_bias: vec1("merger_fc1_bias".into(), merged),
                    fc2: plane("merger_fc2".into(), [hidden, merged]),
                    fc2_bias: vec1("merger_fc2_bias".into(), hidden),
                },
            }
        });

        let mtp = d.draft.then(|| Mtp {
            norm_embed: Weight::sym("mtp.norm_embed", [hidden], dense),
            norm_hidden: Weight::sym("mtp.norm_hidden", [sh], dense),
            fc_embed: Weight::sym("mtp.fc_embed", [hidden, hidden], dense),
            fc_hidden: Weight::sym("mtp.fc_hidden", [sh, hidden], dense),
            block: block(
                &|s: &str| format!("mtp.layer.{s}"),
                true,
                "kv.mtp".to_string(),
                None,
                "conv.mtp".to_string(),
                "delta.mtp".to_string(),
                proj,
            ),
            mixer: residual("mtp.mixer", false),
            eps: d.norm_eps,
            depth: DRAFT_DEPTH,
        });

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
            head: Weight::sym("lm_head", [u64::from(d.vocab), hidden], head_w),
            layers,
            mixer: residual("mixer", false),
            ple,
            mtp,
            tower,
        }
    }
}

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

    #[test]
    fn the_hash_constants_are_the_checkpoints_own() {
        let m = Model::flash(Dtype::Bf16, Dtype::Bf16, 1);
        let p = m.ple.expect("flash carries the PLE");
        assert_eq!(
            p.mults,
            [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
        assert_eq!(
            p.primes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ]
        );
        assert_eq!(p.offsets[0], 0);
        assert_eq!(p.offsets[15], 300_001_275);
        assert_eq!(p.padded_vocab, 320_001_536);
        assert_eq!(p.padded_vocab % 128, 0);
    }
}
