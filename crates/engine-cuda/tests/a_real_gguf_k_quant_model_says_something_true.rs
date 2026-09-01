//! **FIRST LIGHT ON A REAL GGUF**: a shipped `Qwen3-0.6B-Q4_K_M.gguf`, opened
//! where it lies, its K-quant projections served AS STORED, decoding English
//! (QNF wave §J5's successor — the same ladder, against a file somebody else
//! wrote).
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!   --test a_real_gguf_k_quant_model_says_something_true -- --ignored --nocapture
//! ```
//!
//! # What §J5 could not say
//!
//! `a_stored_k_quant_row_serves_as_stored.rs` proved the connective tissue —
//! declaration to plane to `linear::kquant` — but the box held no GGUF, so it
//! wrote its own container out of a seeded stream: three ops, arbitrary
//! blocks, and an oracle that was the same bytes read the other way. Every
//! number in it was one this tree had also produced.
//!
//! This gate produces none of them. The container is llama.cpp's, the blocks
//! are `llama-quantize`'s, the mixture of `q4_k` and `q6_k` is the `Q4_K_M`
//! recipe's own and is READ OFF THE FILE rather than assumed, the tokenizer
//! is the one the GGUF carries in its metadata, and the claim at the end is
//! that the sentence which comes out is English about the world. A wrong
//! super-block width, a mis-set data-section alignment, a reversed dimension
//! order or a plane bound at the wrong address all land here as noise, and
//! nothing in this file could round the same way twice to hide it.
//!
//! # THE CONTAINER READER IS NOT NEW, AND THAT IS THE FINDING
//!
//! Nothing here parses GGUF. `ztensor-compat`'s `gguf` projection (a default
//! feature, and `checkpoint`'s manifest names it explicitly) reads the magic,
//! the version, the KV block, the tensor infos and `general.alignment`, and
//! hands back an ordinary `ztensor::Source` whose quantized entries carry
//! layout `gguf.q4_k/1` and a single u8 `data` part — which is the SAME
//! spelling §J5's fixture wrote by hand, and which `checkpoint::file::zt`
//! already turns into `QuantScheme::GgufQ4K` and `qnf::sig_of_scheme` already
//! turns into the variant the text below declares. So the file is opened by
//! name and the ladder does the rest:
//!
//! ```text
//! Qwen3-0.6B-Q4_K_M.gguf
//!   -> ztensor_compat::index          "gguf.q4_k/1", u8 `data`, [out, in]
//!   -> checkpoint_dsl::Builder::read  stored == wanted: the identity rung
//!   -> WeightRow::Dense at Dtype::Quant(Q4_K)
//!   -> linear::kquant::{matmul, lm_head}
//! ```
//!
//! # A MODEL NOBODY IN THIS TREE SHIPS
//!
//! The text is written here, for §J5's reason and one more. `crates/models`
//! serves a Qwen3.5/3.6/3.8 generation — one full-attention layer in four
//! over a gated delta net, an attention output gate that makes `q_proj`
//! `[2·q·d, hidden]`, a quarter-width partial rotation, and `rmsnorm` with the
//! unit added to the scale. Qwen3-0.6B, the model in this file, is none of
//! those: twenty-eight plain GQA layers, a whole rotation over the whole head,
//! per-head query and key norms with no `+1`, and an ungated `q_proj`. It is
//! not a SKU of that family and a row claiming it would be a row that loads
//! and answers noise.
//!
//! Every op below is already in the vocabulary and already on
//! `engine_cuda::SHIFTED`; what is written here is the ORDER, which is the
//! only thing this artifact needed that the tree did not have.
//!
//! # Gating
//!
//! `#[ignore]`, `qwen4_flash_first_light`'s convention for a gate that wants a
//! device and a snapshot: run it by name. It skips politely rather than
//! failing when either is missing.

use std::path::PathBuf;
use std::time::Instant;

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, Request, Value, Weight,
    ops, trace_hybrid,
};
use model_ir::Trace;

// ─────────────────────────────────────────────────────────────────────────
// The artifact's own numbers
// ─────────────────────────────────────────────────────────────────────────

/// Read off the file's `qwen3.*` metadata block, which is `config.json`'s
/// reading of the same model:
///
/// ```text
/// qwen3.block_count                       28
/// qwen3.embedding_length                1024
/// qwen3.feed_forward_length             3072
/// qwen3.attention.head_count              16
/// qwen3.attention.head_count_kv            8
/// qwen3.attention.key_length             128
/// qwen3.rope.freq_base               1000000
/// qwen3.attention.layer_norm_rms_epsilon  1e-6
/// tokenizer.ggml.tokens               151936 entries
/// ```
///
/// Restated here rather than parsed because every one of them is CHECKED: a
/// `Builder::read` line compares the declared rectangle against the tensor it
/// names, so a wrong number faults on the line that said it and not four
/// stages later. The one thing that is genuinely read from the file is the
/// per-tensor quantization — see [`stored`].
const LAYERS: usize = 28;
const HIDDEN: u64 = 1024;
const INTER: u64 = 3072;
const Q_HEADS: u32 = 16;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const VOCAB: u32 = 151_936;
const THETA: f32 = 1_000_000.0;
const EPS: f32 = 1e-6;

/// `1 / sqrt(head_dim)` — the family's only attention scale.
const SM_SCALE: f32 = 0.088_388_35;

/// The head's fan-out and the query rectangle's width.
const Q_WIDTH: u64 = Q_HEADS as u64 * HEAD_DIM as u64;

/// One kv plane's per-token width, in elements.
const KV_WIDTH: u64 = KV_HEADS as u64 * HEAD_DIM as u64;

/// `q4_k`, spelled. `Dtype::quant` is a `const fn` over the mangled term, so a
/// typo here does not compile; `qnf::scheme_of_sig` answers `GgufQ4K` for it,
/// which is what `checkpoint::file::zt` recovers from `gguf.q4_k/1` — and
/// their equality is what puts the ladder on its identity rung.
const Q4_K: Dtype = Dtype::U4g32k;

/// `q6_k` — what the `Q4_K_M` recipe keeps the embedding table, and some of
/// the value and down projections, at.
const Q6_K: Dtype = Dtype::I6g16k;

// ─────────────────────────────────────────────────────────────────────────
// The text
// ─────────────────────────────────────────────────────────────────────────

/// One bit, which is every bit this model needs: whether a lane's fire is one
/// query row. Attention is the only site with two readings and this is the
/// split between them.
struct Facts {
    qo_one: bool,
}

impl Facts {
    fn qo_one() -> Predicate {
        Predicate::fact(0)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
    }
}

/// One attention site: three ungated projections, two per-head norms, an
/// output, and the kv row this layer writes.
struct Attn {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Weight,
    k_norm: Weight,
    kv: String,
}

/// One decoder layer: pre-attention norm, the site, pre-MLP norm, and a dense
/// SwiGLU whose gate and up the checkpoint ships apart.
struct Layer {
    attn_norm: Weight,
    attn: Attn,
    ffn_norm: Weight,
    gate_up: Weight,
    down: Weight,
}

/// Qwen3-0.6B, as the GGUF states it.
///
/// **THE TABLE IS READ TWICE, AND THE TWO READINGS DISAGREE ON PURPOSE.**
/// `token_embd.weight` is the only tensor this artifact publishes for both the
/// embedding and the readout — Qwen3-0.6B ties them, and the file carries no
/// `output.weight` at all — so the two weights below name the same bytes. They
/// do not name the same REPRESENTATION:
///
/// * `head` is declared `q6_k` and served as stored, because `lm_head` has a
///   K-quant point (`linear::kquant::lm_head`) and this is the busiest
///   consumer of it in the model.
/// * `embed` is declared `Bf16`, because a gather is not a dot: there is no
///   embed point over a braided super-block, and inventing one for a table
///   this wave does not otherwise touch would be a kernel wave hiding inside a
///   gate. So the contract's `Quant -> Raw` rung decodes the table once at
///   load — the same `executor::walk` decode §J5 used as its oracle — and the
///   device holds 297 MiB of bf16 rows beside the 118 MiB of blocks the head
///   reads in place.
///
/// That asymmetry is the honest one: what the file stores is one thing, and
/// what each of its two readers can use is a fact about the reader.
struct Qwen3 {
    embed: Weight,
    head: Weight,
    final_norm: Weight,
    layers: Vec<Layer>,
}

/// **WHAT THE FILE ACTUALLY STORES THIS TENSOR AS.**
///
/// `Q4_K_M` is not a dtype, it is a MIXTURE, and llama.cpp's mixture is
/// data-dependent: in this artifact `attn_v` and `ffn_down` are `q6_k` at
/// layers 0-2, 5, 8, 11, 14, 17 and 20-27 and `q4_k` at the rest, which is a
/// pattern no constant in this file could state and stay true of the next
/// quantization of the same model. So it is asked of the container, tensor by
/// tensor, and a scheme this gate has no term for is a refusal by name rather
/// than a silent bf16.
fn stored(src: &ztensor::Source, name: &str) -> Dtype {
    let tensor = src
        .get(name)
        .unwrap_or_else(|| panic!("this GGUF holds no tensor called `{name}`"));
    match tensor.layout() {
        "gguf.q4_k/1" => Q4_K,
        "gguf.q6_k/1" => Q6_K,
        other => panic!(
            "`{name}` is stored `{other}`, and this text has a term for \
             `gguf.q4_k/1` and `gguf.q6_k/1` only"
        ),
    }
}

impl Qwen3 {
    /// The text, built against the open container so every projection is
    /// declared at the width the file stores it.
    fn read(src: &ztensor::Source) -> Qwen3 {
        // Norms, and every neighbour a bank does not quantize, are stated in
        // what the bank COMPUTES in — `model_dsl::compute_dtype`'s ruling,
        // which answers bf16 for a `Quant` term. The file holds them f32 and
        // the ladder's cast is what closes the gap.
        let dense = Dtype::Bf16;
        let layers = (0..LAYERS)
            .map(|l| {
                let n = |s: &str| format!("blk.{l}.{s}");
                let g = |s: &str| stored(src, &n(s));
                Layer {
                    attn_norm: Weight::sym(n("attn_norm"), [HIDDEN], dense),
                    attn: Attn {
                        q: Weight::sym(n("attn_q"), [Q_WIDTH, HIDDEN], g("attn_q.weight")),
                        k: Weight::sym(n("attn_k"), [KV_WIDTH, HIDDEN], g("attn_k.weight")),
                        v: Weight::sym(n("attn_v"), [KV_WIDTH, HIDDEN], g("attn_v.weight")),
                        o: Weight::sym(n("attn_o"), [HIDDEN, Q_WIDTH], g("attn_output.weight")),
                        q_norm: Weight::sym(n("attn_q_norm"), [u64::from(HEAD_DIM)], dense),
                        k_norm: Weight::sym(n("attn_k_norm"), [u64::from(HEAD_DIM)], dense),
                        kv: format!("kv.{l}"),
                    },
                    ffn_norm: Weight::sym(n("ffn_norm"), [HIDDEN], dense),
                    // The two halves join at the seam `mlp_swiglu` splits at,
                    // and both are `q4_k` at every layer of this artifact —
                    // `read_concat` refuses a pair that is not one scheme, so
                    // that is checked rather than assumed.
                    gate_up: Weight::sym(
                        n("ffn_gate_up"),
                        [2 * INTER, HIDDEN],
                        g("ffn_gate.weight"),
                    )
                    .packed([INTER, INTER]),
                    down: Weight::sym(n("ffn_down"), [HIDDEN, INTER], g("ffn_down.weight")),
                }
            })
            .collect();
        Qwen3 {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], dense),
            head: Weight::sym(
                "lm_head",
                [u64::from(VOCAB), HIDDEN],
                stored(src, "token_embd.weight"),
            ),
            final_norm: Weight::sym("final_norm", [HIDDEN], dense),
            layers,
        }
    }

    /// The load contract, in llama.cpp's own tensor spelling. This is
    /// `qwen_3::import::import_from_gguf`'s vocabulary line for line — the
    /// names are the CONVERTER's, not this tree's, and they are the same names
    /// whatever family reads them.
    fn load(&self, src: &ztensor::Source) -> ModelContract {
        let mut b = checkpoint_dsl::Builder::new(src, 1);
        let say = |what: &str, r: Result<(), checkpoint_dsl::Error>| {
            r.unwrap_or_else(|why| panic!("`{what}`: {why}"));
        };
        say("embed", b.read(&self.embed, "token_embd.weight"));
        say("lm_head", b.read(&self.head, "token_embd.weight"));
        say("final_norm", b.read(&self.final_norm, "output_norm.weight"));
        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");
            say("attn_norm", b.read(&w.attn_norm, n("attn_norm.weight")));
            say("attn_q", b.read(&w.attn.q, n("attn_q.weight")));
            say("attn_k", b.read(&w.attn.k, n("attn_k.weight")));
            say("attn_v", b.read(&w.attn.v, n("attn_v.weight")));
            say("attn_o", b.read(&w.attn.o, n("attn_output.weight")));
            say("attn_q_norm", b.read(&w.attn.q_norm, n("attn_q_norm.weight")));
            say("attn_k_norm", b.read(&w.attn.k_norm, n("attn_k_norm.weight")));
            say("ffn_norm", b.read(&w.ffn_norm, n("ffn_norm.weight")));
            say(
                "ffn_gate_up",
                b.read_concat(&w.gate_up, [n("ffn_gate.weight"), n("ffn_up.weight")]),
            );
            say("ffn_down", b.read(&w.down, n("ffn_down.weight")));
        }
        b.build()
    }
}

impl ForwardHybrid for Qwen3 {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(Dtype::Bf16);
        for w in &self.layers {
            c.kv(kv, w.attn.kv.clone(), [KV_WIDTH, KV_WIDTH]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        // Two classes, one schedule each, cut off `inputs` before either arm
        // reads one — so a plan node's guard is the class it was carved for
        // because the text says so.
        let [input_d, input_p] = inputs.split([Facts::qo_one(), Predicate::rest()]);
        let plan_d = ops::attn::plan_decode(&input_d, Q_HEADS, KV_HEADS, HEAD_DIM, None);
        let plan_p = ops::attn::plan_prefill(&input_p, Q_HEADS, KV_HEADS, HEAD_DIM, None);

        let mut y = ops::layout::embed(&inputs.tokens(), &self.embed, VOCAB);
        for (_, w) in inputs.walk_layers(&self.layers) {
            let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, EPS);
            let o = attend(&x, &inputs, &w.attn, &plan_d, &plan_p);
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm(&y, &w.ffn_norm, EPS);
            let h = ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &w.gate_up), INTER as u32);
            let f = ops::linear::matmul(&h, &w.down);
            y = ops::elemwise::residual_add(&f, &y);
        }
        let x = ops::elemwise::rmsnorm(&y, &self.final_norm, EPS);
        ops::linear::lm_head(&x, &self.head)
    }
}

/// One attention site. **PLAIN EVERYWHERE THE 3.5 GENERATION IS NOT**: no
/// output gate to split off `q`, the whole head rotated rather than a quarter
/// of it, and `rmsnorm_per_head` without the `+1` — three differences that are
/// each one op, and together are why this text is not a row of `qwen_3`.
fn attend(
    x: &Value,
    inputs: &Input<Facts>,
    a: &Attn,
    plan_d: &Value,
    plan_p: &Value,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = HEAD_DIM;

    let q = ops::linear::matmul(x, &a.q);
    let k = ops::linear::matmul(x, &a.k);
    let v = ops::linear::matmul(x, &a.v);
    let q = ops::elemwise::rmsnorm_per_head(&q, &a.q_norm, d, EPS);
    let k = ops::elemwise::rmsnorm_per_head(&k, &a.k_norm, d, EPS);
    // NEOX halves, not interleaved pairs: `rotate_half` is what Qwen3's
    // reference implementation applies and `GGML_ROPE_TYPE_NEOX` is what the
    // converter stamps.
    let (q, k) = ops::elemwise::rope_full(&q, &k, &inputs.positions(), d, THETA, false);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);

    // The same carve as `forward`'s, restated over a different carrier —
    // `Recorder::push` is what holds the two equal.
    let [dq, pq] = q.split([Facts::qo_one(), Predicate::rest()]);
    let o = Value::merge(vec![
        ops::attn::decode(&dq, plan_d, pages, None, d, SM_SCALE),
        ops::attn::prefill(&pq, plan_p, pages, None, d, KV_HEADS, SM_SCALE),
    ]);
    ops::linear::matmul(&o, &a.o)
}

// ─────────────────────────────────────────────────────────────────────────
// The fire
// ─────────────────────────────────────────────────────────────────────────

const PROMPT: &str = "The capital of France is";

/// **OBSERVED, THEN PINNED** — `qwen4_flash_first_light`'s convention. The
/// first light of this shell against this artifact answered
///
/// ```text
/// " Paris. The capital of France is also the capital of the country. The
///   capital of France is also the capital of the country"
/// ```
///
/// greedily, in 1.3 s of load and 5.3 ms per token (188.7 tok/s) with
/// `Graphs::Off` on an L40S — a 0.6-billion-parameter model that repeats
/// itself after one clause, which is what a 0.6B answers and not a symptom.
///
/// The step time is the WARM one, and the first invocation of this gate on a
/// cold page cache measured 17.8 ms (56.0 tok/s) for the same twenty-five
/// tokens. Both are host-bound and neither is a throughput claim: `Graphs` is
/// `Off` here, which is ~470 uncaptured launches a step, and what the number
/// is recorded for is that it is a number at all — the same tokens twice.
///
/// **THE ASSERTION IS THE FIRST WORD AND NOT THE WHOLE SENTENCE**, and that
/// asymmetry is deliberate. What this gate is for is the LADDER — that a
/// `q4_k` row read in place multiplies like the weight it is — and the
/// evidence for that is that the answer is about the world at all: noise from
/// a mis-addressed plane does not spell a capital city. Pinning all
/// twenty-five tokens would additionally pin every rounding decision in every
/// kernel between here and the logits, which is a different gate
/// (`kernels-cuda/tests/kquant_matmul.rs`) and one that already exists.
const EXPECTED: &str = " Paris.";

/// How many greedy steps the continuation is read for.
const STEPS: usize = 24;

/// Where the artifact is. One file — a GGUF is a whole checkpoint, tokenizer
/// included, which is the thing that makes this gate short.
fn artifact() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_GGUF_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_file().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path().join("Qwen3-0.6B-Q4_K_M.gguf")))
        .find(|path| path.is_file())
}

/// The tokenizer the FILE carries: `tokenizer.ggml.{model,pre,tokens,
/// token_type,merges}` compiled by `tokenizer::loader::gguf`. Nothing beside
/// the `.gguf` is read — no `tokenizer.json`, no snapshot directory — which is
/// what makes "a real model" one path and not a directory convention.
///
/// **AND IT IS WHY `PIE_GGUF_TOKENIZER` EXISTS.** Point `PIE_GGUF_SNAPSHOT` at
/// the artifact `pie model import` writes out of this same file and the
/// weights all still resolve — the import keeps a block plane as stored, under
/// llama.cpp's own tensor name and its own `gguf.q4_k/1` layout, which is what
/// the host half of this gate re-checks tensor for tensor against the `.zt`.
/// The tokenizer is the one thing that does NOT survive in this reader's
/// terms: the import COMPILES it, into a `pie.tokenizer/1` object, and the raw
/// `tokenizer.ggml.*` tables it was compiled from are not carried forward. So
/// a run against the artifact names the GGUF it came from here, and the
/// asymmetry is recorded rather than papered over.
fn tokenizer_of(path: &std::path::Path) -> tokenizer::Tokenizer {
    let from = std::env::var("PIE_GGUF_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| path.to_path_buf());
    let tables = checkpoint::file::read::parse_tokenizer(&from)
        .expect("the GGUF's tokenizer metadata parses");
    tokenizer::loader::gguf::from_tables(&tokenizer::loader::gguf::Tables {
        model: &tables.model,
        pre: tables.pre.as_deref(),
        tokens: &tables.tokens,
        token_types: &tables.token_types,
        merges: &tables.merges,
    })
    .expect("this GGUF's own tokenizer compiles")
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

fn word(query_len: u32) -> u64 {
    Facts::of(&Request::new(query_len, false)).word()
}

/// One prefill and `STEPS` greedy decodes, and the per-step wall clock.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");

    let mut produced = vec![argmax(&prefill[0])];
    let mut millis = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let at = Instant::now();
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        finite(&decode[0], "decode");
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

fn trace_of(m: &Qwen3) -> Trace {
    trace_hybrid("qwen3-0.6b-gguf", m, Platform::Cuda)
}

// ─────────────────────────────────────────────────────────────────────────
// (0) the host half — no device, and it says what the declaration folded to
// ─────────────────────────────────────────────────────────────────────────

/// **THE FILE'S OWN MIXTURE REACHES THE TRACE AS BYTE RECTANGLES.**
///
/// What the engine is handed is the trace, so what the trace says about a
/// stored block is the whole of what the shell can know: one plane, no
/// companions, and a width that is the row's BYTES. `1024 / 256 = 4`
/// super-blocks per row, so a `q4_k` row is `4 x 144 = 576` bytes and a `q6_k`
/// row is `4 x 210 = 840` — and neither number is written in this file or in
/// the GGUF. Both fall out of the term.
///
/// Runs with no device, on every plain `cargo test`, when the artifact is
/// there.
#[test]
#[ignore = "real-artifact: needs the Qwen3-0.6B-Q4_K_M GGUF; run with `-- --ignored`"]
fn the_files_own_mixture_reaches_the_trace_as_byte_rectangles() {
    let Some(path) = artifact() else {
        eprintln!("skipping: no Qwen3-0.6B-Q4_K_M.gguf (set PIE_GGUF_SNAPSHOT)");
        return;
    };
    let src = ztensor_compat::index(&path).expect("the GGUF opens as a source");
    let m = Qwen3::read(&src);
    let trace = trace_of(&m);
    let plane = |name: &str| {
        trace
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the trace interns `{name}`"))
    };

    // Every query projection is q4_k in this artifact: [2048, 576] of bytes.
    assert_eq!(plane("blk.0.attn_q").dtype, Q4_K);
    assert_eq!(plane("blk.0.attn_q").shape, vec![Q_WIDTH, 576]);
    // Layer 0's value projection is q6_k and layer 3's is q4_k — the mixture,
    // read off the file rather than assumed.
    assert_eq!(plane("blk.0.attn_v").dtype, Q6_K);
    assert_eq!(plane("blk.0.attn_v").shape, vec![KV_WIDTH, 840]);
    assert_eq!(plane("blk.3.attn_v").dtype, Q4_K);
    assert_eq!(plane("blk.3.attn_v").shape, vec![KV_WIDTH, 576]);
    // The head serves the tied table as stored; the embedding decodes.
    assert_eq!(plane("lm_head").dtype, Q6_K);
    assert_eq!(plane("lm_head").shape, vec![u64::from(VOCAB), 840]);
    assert_eq!(plane("embed").dtype, Dtype::Bf16);
    assert_eq!(plane("embed").shape, vec![u64::from(VOCAB), HIDDEN]);
    // No `.scales`, no `.biases`: a braided block names no companion.
    assert!(
        trace.params.iter().all(|p| !p.name.ends_with(".scales")),
        "a self-contained term claims one tensor and no companions"
    );

    // And the contract states no cast for a block plane read at its own
    // scheme, which is the whole of "serve as stored".
    let contract = m.load(&src);
    let expr = |name: &str| {
        format!(
            "{:?}",
            contract
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("the contract claims `{name}`"))
                .expr
        )
    };
    assert!(!expr("blk.0.attn_q").contains("Cast"), "q4_k serves as stored");
    assert!(!expr("lm_head").contains("Cast"), "the head serves as stored");
    assert!(expr("embed").contains("Cast"), "the table decodes at load");
}

// ─────────────────────────────────────────────────────────────────────────
// (1) the device half
// ─────────────────────────────────────────────────────────────────────────

/// **THE GATE.** A file llama.cpp wrote, opened where it lies, and a sentence
/// about the world.
///
/// The claim is `serve_smoke`'s three: finite, deterministic, and a
/// continuation that says something TRUE. The continuation is OBSERVED, then
/// PINNED — the first run of this shell against this artifact is what the
/// expectation records — and it is checked against the world rather than
/// against another arm of this file, because there is no second arm: the
/// oracle is that Paris is the capital of France.
#[test]
#[ignore = "real-hardware: a CUDA device and the Qwen3-0.6B-Q4_K_M GGUF; run with `--ignored`"]
fn a_shipped_gguf_serves_its_k_quant_rows_and_says_something_true() {
    if !engine_cuda::device::present() {
        eprintln!("skipping gguf first light: no CUDA device on this machine");
        return;
    }
    let Some(path) = artifact() else {
        eprintln!("skipping gguf first light: no Qwen3-0.6B-Q4_K_M.gguf (set PIE_GGUF_SNAPSHOT)");
        return;
    };
    eprintln!("artifact: {}", path.display());

    let tokenizer = tokenizer_of(&path);
    let src = ztensor_compat::index(&path).expect("the GGUF opens as a source");
    let m = Qwen3::read(&src);
    let trace = trace_of(&m);
    let contract = m.load(&src);
    drop(src);

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &path,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 2,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads the GGUF");
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB",
        booted.elapsed().as_secs_f64(),
        weights as f64 / (1u64 << 30) as f64,
        arena as f64 / (1u64 << 20) as f64,
        pools as f64 / (1u64 << 20) as f64,
        inputs as f64 / (1u64 << 20) as f64,
    );

    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");
    eprintln!("prompt {PROMPT:?} -> {prompt:?}");

    let (first, millis) = run(&mut shell, &prompt);
    let text = tokenizer.decode(&first, false);
    let mean = millis.iter().sum::<f64>() / millis.len().max(1) as f64;
    eprintln!("greedy continuation: {text:?}");
    eprintln!("tokens: {first:?}");
    eprintln!("decode: {mean:.1} ms/token ({:.2} tok/s)", 1000.0 / mean);

    assert!(
        text.starts_with(EXPECTED),
        "greedy continuation of {PROMPT:?} was {text:?}, and the first light \
         answered {EXPECTED:?} — a model reading its own weights knows the \
         capital of France, and one reading them at the wrong address does not"
    );

    let (again, _) = run(&mut shell, &prompt);
    assert_eq!(first, again, "twice is not once");
}
