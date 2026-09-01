//! **WHAT A QWEN4 IS**: the qwen_3 hybrid — three GatedDeltaNet mixers to one
//! gated attention, a routed MoE at every layer — rebuilt around a residual
//! that is not a sum.
//!
//! Four residual STREAMS ride every layer (`hc_count: 4`), and each sublayer
//! reads a learned sigmoid mix of them and writes back through per-stream
//! gates ([`Residual`]). The trunk's other new organ is the PLE ([`Ple`]): a
//! hashed n-gram memory — fifty-one billion parameters of it — gathered per
//! token and folded into every stream at one early layer. Both are read off
//! `Qwen/Qwen3.8-Flash-Next`'s `config.json` and `transformers`'
//! `modular_qwen4_exp.py`, and the mixer/MoE organs this family shares with
//! qwen_3 are qwen_3's own structs, reused rather than restated.

use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

use checkpoint_dsl::{Builder, Error};

pub use crate::qwen_3::model::{Attn, Gdn, Mlp};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The attention reading, stated once for every full-attention site —
    /// [`crate::qwen_3::model::Model`]'s rule, unchanged. Per rank.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,

    /// The residual stream fan (`hc_count`) and the mixer's bottleneck
    /// (`hc_lowrank`). Facts about EVERY residual site, so stated once —
    /// the same argument that hoists the attention reading.
    pub streams: u32,
    pub lowrank: u32,

    pub kv: Dtype,
    pub embed: Weight,
    /// Always a bank: `tie_word_embeddings: false` in the one shipped SKU,
    /// and a tied qwen4 would be a new fact, not a new field.
    pub head: Weight,
    pub layers: Vec<Layer>,

    /// **THE FINAL MIXER IS THE FINAL NORM.** The reference's `TextModel`
    /// ends with a combine-less [`Residual`] — `hyper_connection_mixer`,
    /// `use_combine: false` — and declares NO `model.norm`: the mixer's own
    /// grouped norm is the last normalization the logits see, and the
    /// checkpoint publishes no plane a final rmsnorm could read.
    pub mixer: Residual,

    /// The hashed n-gram memory, on the layers `ple_layer_ids` names —
    /// one, in the shipped SKU. `Option` per layer would misstate the axis:
    /// the config names layers, so the layer index lives in [`Ple`] and the
    /// trunk walk asks.
    pub ple: Option<Ple>,
}

/// One gated-residual site: the grouped norm over the wide stream row, the
/// low-rank sigmoid mixer that folds it to a layer input, and — everywhere
/// but the trunk's final mixer — the per-stream injection gates the sublayer
/// output returns through.
///
/// ```text
/// normed = rmsnorm_grouped_plus_one(hyper)                  [S·H]
/// x      = meanₛ(σ(up(silu(down(normed)/S))) ⊙ normed)      [H]
/// …sublayer runs on x…
/// hyper += 2·σ(inject(normed)/S) ⊗ o                        [S·H]
/// ```
///
/// The two GEMMs are ordinary [`Weight`] banks (quantized in the 4-bit
/// artifact), so the ops between them are the only new arithmetic — see
/// `elemwise.hc_mix` / `elemwise.hc_inject`.
pub struct Residual {
    /// `[streams · hidden]` — full width, one scale per stream element,
    /// which is what parts it from a per-head norm's shared plane.
    pub norm: Weight,
    /// `[lowrank, streams · hidden]`.
    pub down: Weight,
    /// `[streams · hidden, lowrank]`.
    pub up: Weight,
    /// `[streams, streams · hidden]`, or `None` on the trunk's final mixer,
    /// which folds and never injects.
    pub inject: Option<Weight>,
    pub eps: f32,
}

pub struct Layer {
    pub mixer: Mixer,
    /// The mixer sublayer's residual site. There is NO separate input
    /// layernorm in this family — the site's grouped norm is the only
    /// normalization the sublayer input gets, which is why [`Layer`] has no
    /// `mixer_norm` field for a checkpoint plane that does not exist.
    pub attn_res: Residual,
    pub mlp_res: Residual,
    pub mlp: Mlp,
}

pub enum Mixer {
    /// The gated attention, qwen_3's own struct: the checkpoint's
    /// `q_proj` is `[2 · q_heads · head_dim, hidden]` — query and output
    /// gate fused, `Qwen3_5Attention`'s `attn_output_gate` — so the site
    /// reads as `qg_proj` here exactly as it does there.
    ///
    /// **AND IT IS THE WHOLE OF WHAT THIS TEXT SAYS ABOUT ATTENTION** (the
    /// QSA cut, stated plainly): the artifact's full-attention layers carry a
    /// sparse-selection indexer (`self_attn.indexer.*`, budget 2048 over
    /// blocks of 4) that this declaration does not read and this forward does
    /// not run. Selection keeps every complete block until the visible
    /// context outgrows the budget, so a fire whose kv never exceeds ~2048
    /// tokens computes the reference's own logits exactly; past that, this
    /// text reads MORE context than the reference selects. The indexer is an
    /// IR campaign of its own (a pooled key cache and a selected GQA reader)
    /// and a declaration that named its planes while ignoring them would
    /// dress the cut up as coverage.
    Attn(Attn),
    Gdn(Gdn),
}

/// The PLE: a hashed n-gram embedding gathered per token, gated per stream
/// against the residual, locally mixed by a dilated depthwise convolution,
/// and added into the wide stream row at one early layer.
///
/// ```text
/// e      = embed_concat(ngram_ids(tokens), table)            [E]
/// key    = norm_key(key_proj(e))     value = value_proj(e)   [S·H], [H]
/// gate_s = σ(signed_sqrt(keyₛ · norm_query(hyper)ₛ / √H))
/// g      = gateₛ ⊗ value                                     [S·H]
/// hyper += g + silu(conv₄,dil₃(norm_conv(g)))                [S·H]
/// ```
///
/// **THE HASH CONSTANTS ARE DERIVED, NOT READ.** The checkpoint publishes
/// `layer_multipliers`, `ngram_heads_vocab_sizes` and `ngram_heads_offsets`
/// as buffers, but every one of them is a pure function of the config —
/// splitmix64 over `seed: 1234`, the first sixteen primes past twenty
/// million — so [`Model::flash`] computes them and the census test holds the
/// computation against the published buffers rather than trusting either
/// alone.
pub struct Ple {
    /// Which trunk layer folds the enrichment in (zero-indexed; the config's
    /// `ple_layer_ids` is one-indexed and names layer 2).
    pub layer: u32,
    /// `eos_token_id` — the hasher's padding id at every sequence start and
    /// across every eos boundary.
    pub eos: u32,
    pub heads_per_ngram: u32,
    /// One multiplier per n-gram position; `mults.len()` is `ngram_size`.
    pub mults: Vec<u64>,
    /// One prime vocabulary per hashed head, and its row offset in the
    /// concatenated table.
    pub primes: Vec<u64>,
    pub offsets: Vec<u64>,
    /// The padded row count of [`table`](Ple::table) — the primes' sum,
    /// rounded up to `make_ngram_vocab_size_divisible_by`.
    pub padded_vocab: u64,
    /// `[padded_vocab, embed_dim / heads]` — the concatenated shards, banded
    /// at the shard seams the checkpoint splits them at.
    pub table: Weight,
    pub key_proj: Weight,
    pub value_proj: Weight,
    pub norm_key: Weight,
    pub norm_query: Weight,
    pub norm_conv: Weight,
    /// `[streams · hidden, kernel]` — depthwise, silu-activated, and DILATED
    /// by `ngram_size`: tap `j` reads `3·j` positions back.
    pub conv: Weight,
    pub conv_kernel: u32,
    pub dilation: u32,
    pub eps: f32,
    /// The state rows: the trailing token-id window the hasher keeps
    /// (`[ngram − 1]`, i32) and the convolution's own history
    /// (`[(kernel−1)·dilation + 1, streams·hidden]`).
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
    /// `make_ngram_vocab_size_divisible_by` — the table's row padding.
    divisible_by: u64,
    /// `split_ngram_parts` — how many shards the checkpoint stores the
    /// table as. Equal to `divisible_by` in the shipped config, and a
    /// separate fact anyway.
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

/// **WHAT A QWEN4 CONVERSION STORES EACH ROLE AT**, which one `weights` dtype
/// cannot say.
///
/// This family's conversions are per-tensor mixes and they do not agree with
/// each other. `config.json` carries ONE default `(bits, group)` and then a
/// list of overrides, and which planes the overrides name is a fact about the
/// converter, not about the architecture: the mixed-4/8 stack raises the dense
/// projections ABOVE its four-bit default and leaves the expert banks there,
/// while `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit` drops the expert banks
/// BELOW its own and leaves the projections at the default. One `w` threaded
/// through a `match` can express a family whose exceptions all point the same
/// way; it cannot express both of these at once, and the derivation that tried
/// (`proj = U8g64`, `narrow_group = U4g32`, both off a single `w`) read the
/// 2-bit file's `q_proj` as eight bits and its embedding as quantized at all.
///
/// So this states the roles instead — one representation per role the file
/// actually distinguishes — and [`Mix::of`] is the mixed-4/8 derivation
/// unchanged, so every row that shipped before this type existed declares what
/// it declared.
#[derive(Clone, Copy, Debug)]
pub struct Mix {
    /// The token embedding and the (untied) output head.
    pub embed: Dtype,
    /// Every dense projection — attention's four banks, the GDN's `qkv|z` and
    /// its output, the shared expert's pair and down, the residual sites' two
    /// GEMMs, and the PLE's key/value projections.
    pub proj: Dtype,
    /// The residual sites' per-stream INJECTION gates. Its own role because
    /// the two conversions disagree about it: `block_inject_weight` is a
    /// `[4, 10240]` sliver the mixed-4/8 predicate leaves dense and the
    /// mixed-2bit converter quantizes at the default like everything else.
    pub inject: Dtype,
    /// The GDN's `in_proj_b | in_proj_a` pair — `[2·v_heads, hidden]`, the
    /// other sliver the two conversions disagree about, for the same reason.
    pub gdn_ba: Dtype,
    /// The routed expert banks, fused `gate|up` and `down` alike. **THIS IS
    /// WHERE A MIXED CONVERSION SPENDS ITS EXCEPTIONS**, because this is where
    /// the bytes are: 16 experts x 640 x 2560 x 3 is the whole of a qwen4
    /// layer next to which every projection beside it rounds to nothing.
    pub experts: Dtype,
    /// The hashed n-gram table. Its 160-wide rows cannot group by sixty-four,
    /// so a quantized one is at G32 whatever the trunk is
    /// (`dtype::Dtype::U4g32`).
    pub table: Dtype,
}

impl Mix {
    /// **THE MIXED-4/8 DERIVATION, UNCHANGED** — what `Model::new` computed
    /// off one `w` before this type existed, and what
    /// `qwen38-flash-{mlxu4,bf16}-kv-bf16` still say.
    ///
    /// `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` raises the embedding,
    /// the head, every attention and GDN in/out projection, the shared expert,
    /// the residual mixers' two GEMMs and the PLE projections to `bits: 8` and
    /// leaves the expert banks and the n-gram table at four — `qwen_3::model`'s
    /// router predicate grown to the whole dense set, which is what this
    /// family's conversion does. A bf16 stack's every role stays bf16.
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
            // The two slivers the 4/8 predicate does not reach — too narrow
            // for its own rule — where the banks beside them ride eight bits.
            inject: crate::dense(w),
            gdn_ba: crate::dense(w),
            experts: w,
            table,
        }
    }

    /// **THE `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit` CONVERSION'S OWN
    /// MIX**, read off its `config.json` and verified against its stored
    /// rectangles by `model/tests/the_flash_text_reads_the_mini_snapshot.rs`.
    ///
    /// Its `MERERUN_CONVERSION.json` states the whole of it in three counts —
    /// `q2: 144`, `q4_g32: 128`, `q4_g64: 682` — and the file agrees plane for
    /// plane:
    ///
    /// * a default of `(4, 64)`, which is where every projection sits, the
    ///   two slivers the 4/8 conversion left dense INCLUDED;
    /// * `(2, 128)` on `mlp.switch_mlp.{gate,up,down}_proj`, **UNIFORM across
    ///   the three and across every layer** — so unlike the DeepSeek-V4 DQ
    ///   artifact next door, the fused `experts_gate_up` bank is stateable
    ///   here and `deepseek_v4::model::GateUp::Split` is not needed;
    /// * `(4, 32)` on the eight `ple_embedding.ngram_embedding.shard_*`;
    /// * and **NO 8-BIT ENTRY ANYWHERE** — the embedding and the head are
    ///   plain `BF16` planes, not the `U8g64` the 4/8 derivation would have
    ///   claimed.
    pub const MIXED_2BIT: Mix = Mix {
        // Not quantized at all: `embed_tokens.weight` and `lm_head.weight`
        // ship as `BF16 [248320, 2560]` with no `.scales` companion beside
        // them, which is the one thing the single-`w` derivation could not
        // have been talked into saying.
        embed: Dtype::Bf16,
        proj: Dtype::U4g64,
        inject: Dtype::U4g64,
        gdn_ba: Dtype::U4g64,
        experts: Dtype::U2g128,
        table: Dtype::U4g32,
    };

    /// What the banks of this mix COMPUTE in — the norms, the router, the
    /// shared gate and the depthwise convolutions, which no conversion of this
    /// family quantizes. See `crate::dense`.
    #[must_use]
    pub fn dense(&self) -> Dtype {
        crate::dense(self.proj)
    }
}

impl Model {
    /// **THE ONE SHIPPED SKU** — `Qwen/Qwen3.8-Flash-Next`, every number its
    /// `text_config`'s: 48 layers at `full_attention_interval: 4`, heads
    /// 24/2 at 256 wide rotating the first quarter, GatedDeltaNet at 16/48
    /// heads of 128, 512 experts at 640 routing 10, four residual streams
    /// over a 320-wide mixer, and the tri-gram PLE on layer 2. The vision
    /// tower and the MTP arm the artifact also publishes are not declared
    /// here — the first is a second SKU when a deployment wants it, the
    /// second is a draft head no 4-bit conversion carries.
    pub fn flash(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(Mix::of(w), kv, tp, Model::flash_dims())
    }

    /// **THE MINI 2-BIT SNAPSHOT'S OWN GEOMETRY** — the real
    /// `Qwen3.8-Flash-Next` everything (hidden 2560, heads 24/2 at 256, the
    /// GatedDeltaNet at 16/48 of 128, `moe_intermediate_size` 640, the
    /// 248 320-token vocabulary) over the FOUR layers, SIXTEEN routed experts
    /// and EIGHT n-gram shards `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit`
    /// publishes as `mini-l4-e16-p8`.
    ///
    /// Four numbers move off [`flash`](Model::flash) and no others, each read
    /// off the snapshot's own `text_config`:
    ///
    /// * `num_hidden_layers: 4` — and `layer_types` is
    ///   `[linear, linear, linear, full]`, which is `attn_every: 4` unchanged;
    /// * `num_experts: 16` (`num_experts_per_tok` stays 10, so the router
    ///   selects ten of sixteen);
    /// * `ngram_vocab_size_base: 1250000`, a sixteenth of the shipped
    ///   20 000 000 — the hashed heads are the same sixteen and each gets a
    ///   prime a sixteenth the size;
    /// * `split_ngram_parts: 8`, so the table is stored as eight shards and
    ///   not the shipped one hundred and twenty-eight.
    ///
    /// **AND THE LAST TWO ARE WHERE THIS ARTIFACT AND ITS OWN CONFIG USED TO
    /// PART COMPANY.** The snapshot's PLE table shipped as eight shards of
    /// 2 500 012 rows — `320 001 536 / 16`, the SHIPPED table's padded row
    /// space cut into its 128 stored shards with eight kept — and it published
    /// the shipped model's sixteen primes past 20 000 000 beside it. Its
    /// `ngram_vocab_size_base: 1250000` reproduced neither: this derivation
    /// gives 20 001 536 rows, and NO base gives 20 000 096, which is not even a
    /// multiple of the config's own `make_ngram_vocab_size_divisible_by`. The
    /// metadata had not been carved with the table, fifteen of the sixteen
    /// head offsets pointed past the end of it, and the row imported, baked and
    /// REFUSED AT LOAD at that one rectangle.
    ///
    /// **THE ARTIFACT MOVED TO MEET THIS TEXT, WHICH IS THE RIGHT DIRECTION.**
    /// A text declares what a config SAYS; it does not invent a hashing to fit
    /// bytes a slicer produced, and declaring the stored 20 000 096 would have
    /// fitted the load while leaving the gather off the end of the plane.
    /// `benches/shrink_checkpoint.py` re-cuts the table by HEAD instead of by
    /// stored shard — head `h` of the miniature takes its own prime's worth of
    /// rows out of head `h` of the original — and rewrites the two published
    /// head buffers to the miniature's own primes and offsets. The snapshot is
    /// eight shards of 2 500 192 now, 20 001 536 rows, which is exactly what
    /// the four numbers above derive.
    ///
    /// `model/tests/the_qwen4_text_reads_the_two_bit_miniature.rs` holds that
    /// agreement against the bytes, and
    /// `engine-metal/tests/qwen4_two_bit_first_light.rs` is the first light it
    /// unblocked.
    ///
    /// **NOT [`flash_micro`](Model::flash_micro), and the difference is what
    /// each is for.** `flash_micro` shrinks every dimension so the reference
    /// can be run beside it; its shapes are nobody's file's. This one is a
    /// text a checkpoint can actually be read through: the names are the same
    /// and the rectangles are this snapshot's.
    pub fn flash_mini(mix: Mix, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::flash_dims();
        d.layers = 4;
        d.moe.experts = 16;
        let ple = d.ple.as_mut().expect("the flash dims carry a PLE");
        ple.base_vocab = 1_250_000;
        ple.split_parts = 8;
        Model::new(mix, kv, tp, d)
    }

    /// `Qwen/Qwen3.8-Flash-Next`'s own `text_config`, as one value both the
    /// shipped arm and the mini arm start from.
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

    /// **A QWEN4 SMALL ENOUGH TO HOLD AGAINST THE REFERENCE** — the text the
    /// parity gate loads, `a3b_micro`'s reason one family over: no checkpoint
    /// ships it and no deployment selects it, and what it is for is the one
    /// claim `flash` cannot be used to make — that this forward computes
    /// `modular_qwen4_exp.py`'s own logits — because that claim needs both
    /// implementations over one set of weights, and `flash` is a hundred
    /// gigabytes against a reference that runs on a CPU.
    ///
    /// Every organ is exercised: the hybrid mix (`attn_every: 2`), the
    /// routed MoE, both residual sites per layer, the final mixer, and the
    /// PLE on layer 1 with a hash base small enough that collisions actually
    /// occur.
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
                // 64 and not smaller: the fa2 lattice stamps no narrower
                // unit, and the gate wants the shipped kernels, not new ones.
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
                    // On a GDN layer, as the reference's validator requires
                    // (PLE rides linear-attention layers only); layer 2 is
                    // one under `attn_every: 2`.
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
        // **EVERYTHING LANDS AS STORED.** A quantized artifact of this family
        // quantizes almost everything, and this text declares each rectangle
        // at the width the file holds it, because every width has its own
        // reader: the routed expert banks take the grouped select, the n-gram
        // table the concatenating gather, and the projections the affine gemm
        // point a dense `linear.matmul` resolves to when its weight seats as
        // planes. Nothing is dequantized at load; the one weight transform
        // the import states is the norms' `+1` fold, taken back out
        // (`import.rs`).
        //
        // WHICH width each role is is [`Mix`]'s business, because the two
        // conversions this family has shipped disagree about it in both
        // directions — the 4/8 stack raises its projections above its default
        // and the 2-bit stack drops its experts below its own. See its doc.
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
                        // `in_proj_b`/`in_proj_a` are [48, 2560] slivers the
                        // mixed-4/8 conversion leaves DENSE — too narrow for
                        // its own predicate — where the qkv|z pair beside them
                        // rides eight bits; the mixed-2bit conversion has no
                        // such predicate and quantizes them at its default.
                        // `Mix::gdn_ba` is that disagreement.
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
                        // The router and the shared gate are DENSE in the
                        // shipped conversion — qwen_3's eight-bit predicate
                        // does not reach this family's file.
                        router: Weight::sym(n("router"), [u64::from(d.moe.experts), hidden], dense),
                        // **THE FUSED BANK IS STATEABLE ON BOTH ARTIFACTS.**
                        // A `Weight` carries ONE dtype and a `Dtype` is where
                        // an MLX affine group is written down, so a fused
                        // `[gate | up]` bank needs the two stored halves to
                        // agree on their point — `deepseek_v4::model::GateUp`
                        // is the form for an artifact where they do not. Both
                        // of this family's conversions write `gate_proj` and
                        // `up_proj` at the SAME `(bits, group)` on every
                        // layer, so this stays one bank and one routed matmul.
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
                // Banded at the checkpoint's own shard seams: `split_ngram_
                // parts` equal slices of the padded row count, which is what
                // an import's `read_concat` rejoins.
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

    /// The native-artifact read: every declared plane by its own name, whole.
    pub fn load(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
        b.read_own(&self.embed)?;
        b.read_own(&self.head)?;
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn(a) => {
                    b.read_own(&a.qg_proj)?;
                    b.read_own(&a.k_proj)?;
                    b.read_own(&a.v_proj)?;
                    b.read_own(&a.o_proj)?;
                    b.read_own(&a.q_norm)?;
                    b.read_own(&a.k_norm)?;
                }
                Mixer::Gdn(g) => {
                    b.read_own(&g.in_qkvz)?;
                    b.read_own(&g.in_ba)?;
                    b.read_own(&g.conv)?;
                    b.read_own(&g.dt_bias)?;
                    b.read_own(&g.a_log)?;
                    b.read_own(&g.norm)?;
                    b.read_own(&g.out_proj)?;
                }
            }
            for res in [&w.attn_res, &w.mlp_res] {
                b.read_own(&res.norm)?;
                b.read_own(&res.down)?;
                b.read_own(&res.up)?;
                if let Some(inject) = &res.inject {
                    b.read_own(inject)?;
                }
            }
            match &w.mlp {
                Mlp::Dense { .. } => unreachable!("every qwen4 layer routes"),
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    b.read_own(router)?;
                    b.read_own(gate_up)?;
                    b.read_own(down)?;
                    b.read_own(shared_gate_up)?;
                    b.read_own(shared_down)?;
                    b.read_own(shared_gate)?;
                }
            }
        }
        if let Some(p) = &self.ple {
            b.read_own(&p.table)?;
            b.read_own(&p.key_proj)?;
            b.read_own(&p.value_proj)?;
            b.read_own(&p.norm_key)?;
            b.read_own(&p.norm_query)?;
            b.read_own(&p.norm_conv)?;
            b.read_own(&p.conv)?;
        }
        b.read_own(&self.mixer.norm)?;
        b.read_own(&self.mixer.down)?;
        b.read_own(&self.mixer.up)?;
        Ok(b.build())
    }
}

/// The reference's own hash derivation, constant for constant:
/// `_build_layer_multipliers` and `_find_nth_prime_after` from
/// `modular_qwen4_exp.py`, held against the checkpoint's published buffers
/// by the census test. `splitmix64` is the standard finalizer; the
/// multipliers are odd and bounded so an `i64` product cannot overflow, which
/// is what lets the device hash in `u64` and agree with torch's `long`.
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

    // This family has one PLE layer, so `ple_layer_index` is zero in both
    // derivations below (the reference offsets `seed` by `PRIME_1` times the
    // index, and multiplies by nothing at index zero); a second PLE layer
    // would thread its index here.
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

    /// The derivation, held against the buffers the shipped artifact
    /// publishes (`ple_embedding.layer_multipliers` /
    /// `ngram_heads_vocab_sizes` / `ngram_heads_offsets`, read out of
    /// `Qwen3.8-Flash-Next-MLX-mixed-4_8bit`) — the census this module's doc
    /// promises. Sixteen primes, three multipliers, and the padded row count
    /// the 128 shards sum to.
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
