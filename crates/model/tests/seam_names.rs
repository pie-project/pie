//! The SEAM: what a forward pass names, against what `wire()` can answer.
//!
//! Two vocabularies invent names for the same tensors and nothing compares
//! them. The DSL invents TRACE names (`layer.3.qkv`) as it records a
//! forward pass; a load contract invents PUBLISHED names
//! (`model.layers.3.self_attn.qkv_proj.fused.weight`) as it authors the
//! staging; and [`model::shared::weight_names::wire`] is the one bridge between
//! them. A trace name `wire()` cannot answer reaches the driver's
//! resolver, which returns `None` — and `Resolver::weight`'s own doc says
//! what that is:
//!
//! > `None` — which is DRIFT, not absence: a trace that names a weight the
//! > store lacks was traced against a different binding.
//!
//! **The seam fails silently by construction.** `Wiring::alias` records a
//! row only `if self.has(&published)`, so a name the contract never
//! published is dropped with no diagnostic, and the refusal surfaces one
//! name at a time at FIRE time as `BindRefusal::UnknownWeight`. The design
//! knows it is drift and detects it one request too late. This moves the
//! whole class to CI.
//!
//! ## What this asks, precisely
//!
//! Not "does this checkpoint wire" — that needs a checkpoint. It asks the
//! stronger and cheaper question: **can `wire()` EVER emit this name, for
//! any checkpoint at all?** A name outside its reachable set is
//! unanswerable by construction, and no fixture can rescue it.
//!
//! So the published side is a `|_| true` predicate under each of the three
//! naming schemes `wire()` recognises, which yields the maximal set. That
//! makes a listed stem a genuine hole rather than a fixture artefact, and
//! it makes this test's failure mode the safe one: it can miss a
//! deployment-specific gap, and it cannot invent one.
//!
//! ## What it does NOT catch, and one instance found while writing it
//!
//! **A name that resolves to the WRONG tensor.** This compares spellings,
//! so a trace name `wire()` answers is a name it considers answered —
//! whether or not the tensor behind it is the right one. gemma-2 was the
//! instance: it is a SANDWICH-norm family with four norms per layer, and
//! `llama_like`'s pre-norm branch mapped its `mlp_norm` to
//! `post_attention_layernorm` where the forward means
//! `pre_feedforward_layernorm`. The name resolves; the tensor is wrong.
//!
//! That one is repaired — `weight_names::wire` branches on the SANDWICH
//! PAIR rather than on `input_layernorm`, whose presence never told the
//! two apart, and the repair has its own assertion beside it. The class
//! is what stays uncaught here, so it is written down: the shape of what
//! a test misses belongs beside the test.
//!
//! Layer indices are normalised to `layer.*`, because a trace at four
//! layers and a `wire()` at eight would otherwise disagree about
//! everything for no reason. The question is about SPELLINGS.

#![cfg(feature = "contract")]

use std::collections::{BTreeMap, BTreeSet};

use model::catalog::LoadShape;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_ir::trace::{FireClass, ForwardPlan};

/// Which naming scheme a family's checkpoint follows.
///
/// `wire()` picks its builders by SIGNATURE — one tensor only that family
/// ships — so which builders run is a property of the checkpoint, and the
/// reachable set has to be computed the same way or the test answers a
/// different question than the driver does.
///
/// An earlier draft used one all-true predicate and took the union over
/// schemes. That opened every gate at once, so `gpt_oss`'s builder ran for
/// every family and its `layer.*.router` made nemotron-h's router look
/// answerable — which it is not, because a nemotron checkpoint ships no
/// attention sinks and that builder never runs for it. The union answers
/// "can `wire()` emit this SPELLING for anyone", and the question worth
/// asking is "can it emit this name for THIS family's checkpoint".
#[derive(Clone, Copy)]
enum Scheme {
    /// Plain HF: `model.embed_tokens.weight`, `model.layers.N.…`.
    LlamaLike,
    /// Plain HF plus the per-head attention sink only gpt-oss ships.
    GptOss,
    /// The VL prefix WITH the per-layer embedding table.
    Gemma4,
    /// The VL prefix WITHOUT it.
    Qwen35,
}

impl Scheme {
    /// A checkpoint that publishes everything its scheme allows, and
    /// nothing another scheme's gate would recognise.
    fn published(self) -> fn(&str) -> bool {
        match self {
            Self::LlamaLike => {
                |n: &str| !n.starts_with("model.language_model.") && !n.ends_with("self_attn.sinks")
            }
            Self::GptOss => |n: &str| !n.starts_with("model.language_model."),
            Self::Gemma4 => |n: &str| n.starts_with("model.language_model."),
            Self::Qwen35 => |n: &str| {
                if !n.starts_with("model.language_model.")
                    || n == "model.language_model.embed_tokens_per_layer.weight"
                {
                    return false;
                }
                // A HYBRID stack, because that is what qwen3.5 is: three
                // gated-delta-net layers then one full-attention layer,
                // measured on `Qwen3.6-35B-A3B-4bit`, and each layer
                // ships one kind's tensors and not the other's.
                //
                // This predicate used to answer yes to every name, which
                // was harmless while `wire()` read a `layer_types` list
                // off the config. It stopped being harmless when that
                // branch started asking the CHECKPOINT — `w.has(q_proj)`
                // is then true at every layer, the linear-attention arm
                // never runs, and `conv`, `a_log`, `dt_bias`,
                // `gate_norm` and the four `in_proj`s all became
                // unreachable at once. The test reported the first of
                // them as a seam gap; there is no gap, there was no
                // hybrid checkpoint to ask.
                let Some(rest) = n.strip_prefix("model.language_model.layers.") else {
                    return true;
                };
                let Some((index, _)) = rest.split_once('.') else {
                    return true;
                };
                let Ok(layer) = index.parse::<u32>() else {
                    return true;
                };
                let full = layer % 4 == 3;
                if n.contains(".self_attn.") {
                    full
                } else if n.contains(".linear_attn.") {
                    !full
                } else {
                    true
                }
            },
        }
    }
}

/// Every trace name `wire()` can emit for a checkpoint of this scheme,
/// layer-normalised.
fn answerable(scheme: Scheme) -> BTreeSet<String> {
    // Four layers and a 128-wide head: enough for the per-layer loops to
    // run and for a KV-shared tail to be distinguishable from the layers
    // that project their own. `wire()` reads only the layer count and the
    // shared-tail length, which is why a `LoadShape` replaced the
    // 136-field config this used to build.
    let shape = LoadShape::dense(4, 128, false);
    let published = scheme.published();
    let w = model::shared::weight_names::wire(shape, &published);
    w.aliases
        .iter()
        .map(|(t, _)| normalise(t))
        .chain(w.joins.iter().map(|(t, _)| normalise(t)))
        .collect()
}

/// `layer.3.qkv` -> `layer.*.qkv`.
fn normalise(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for part in name.split('.') {
        if !out.is_empty() {
            out.push('.');
        }
        if part.parse::<u64>().is_ok() {
            out.push('*');
        } else {
            out.push_str(part);
        }
    }
    out
}

/// The `Arg::Weight` stems a family's decode plan names, layer-normalised
/// and narrowed to the ones that are actually TENSORS.
///
/// Two kinds of name come out of `Arg::Weight` and are not weights, and
/// counting either would report a hole where there is none.
///
/// **`scale.…` is a HOST SCALAR.** `dsl::cuda::scalar_mul` given no value
/// names a `scale.*` that the driver looks up in `ctx.scales`, a table it
/// built from a config — the arm strips the prefix and never touches the
/// weight store. `wire()` has a third channel for these
/// (`Wiring::scalars`, which the driver reads into
/// `gemma_layer_scalars`), and it maps PUBLISHED names rather than trace
/// names, so it cannot be checked the same way. That channel is
/// unchecked; this test says so rather than pretending otherwise.
///
/// **The empty stem is a WEIGHTLESS statement.**
/// `norm::per_head_rmsnorm_bf16` is the V-norm without a gamma, and the
/// trace records an `Arg::Weight("")` for the slot it does not fill.
/// Nothing resolves it because nothing needs to.
fn stems(plan: &ForwardPlan) -> BTreeSet<String> {
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        4
    ];
    let l = lower(
        plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the corpus lowers");
    l.args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(name) if !name.is_empty() && !name.starts_with("scale.") => {
                Some(normalise(name))
            }
            Arg::Weight(_) | Arg::Named { .. } | Arg::Arena { .. } | Arg::Raised { .. } => None,
        })
        .collect()
}

/// Every declared forward family, at its own facts fixture, decode class.
///
/// The same eleven `golden_plans.rs` holds — deliberately, because a
/// family that has a golden and no row here is a family whose seam nobody
/// is checking, and the two lists diverging is itself the bug.
fn corpus() -> Vec<(&'static str, Scheme, ForwardPlan)> {
    use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    use model::shared::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};

    vec![
        (
            "llama_like",
            Scheme::LlamaLike,
            model::shared::llama_like::forward::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            ),
        ),
        (
            "qwen3_5",
            Scheme::Qwen35,
            model::qwen_3_5::forward::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_5_0_8b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma_4",
            Scheme::Gemma4,
            model::gemma_4::forward::gemma4_cuda(
                &Gemma4Facts::gemma_4_e4b(),
                &Gemma4CudaFacts::gemma_4_e4b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gpt_oss",
            Scheme::GptOss,
            model::gpt_oss::forward::gpt_oss_cuda(
                &GptOssFacts::gpt_oss_20b(),
                &GptOssCudaFacts::gpt_oss_20b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma_2",
            Scheme::LlamaLike,
            model::gemma_2::forward::gemma2_cuda(
                &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma3n",
            Scheme::LlamaLike,
            model::gemma_3n::forward::gemma3n_cuda(
                &model::gemma_3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "deepseek_v4",
            Scheme::LlamaLike,
            model::deepseek_v4::forward::dsv4_cuda(
                &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "glm5",
            Scheme::LlamaLike,
            model::glm_5::forward::glm5_cuda(
                &model::glm_5::forward::facts::Glm5Facts::glm5_106b_a12b(),
                FireClass::Decode,
            ),
        ),
        (
            "kimi_k2",
            Scheme::LlamaLike,
            model::kimi_k2::forward::kimi_cuda(
                &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
                &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "kimi_k3",
            Scheme::LlamaLike,
            model::kimi_k3::forward::kimi_k3_cuda(
                &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "nemotron_h",
            Scheme::LlamaLike,
            model::nemotron_h::forward::nemotron_h_cuda(
                &model::nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
                FireClass::Decode,
            ),
        ),
    ]
}

/// The stems `wire()` cannot answer today, per family — the seam's debt.
///
/// A CLOSED list, and that is the whole point: it shrinks as builders land
/// and a stem that JOINS it is a family that has started naming something
/// nothing can resolve, which is a fire away from `UnknownWeight`. Sorted
/// within a family so the diff when one leaves is one line.
///
/// `gpt_oss` WAS the row that bit, and is the reason this list is a test
/// rather than a note: it was the only family with both a `FACTS_ROWS`
/// entry in the CUDA shell and a Prefill arm, so a gpt-oss checkpoint
/// LOADED, reported itself healthy, and died at its first fire. It is
/// wired now, and its line is kept empty rather than deleted because an
/// empty line is the record that the debt was paid.
///
/// FIVE families are wired end to end and SIX still owe. The six are not
/// yet reachable for other reasons, so their debt is owed and not due --
/// which is a different thing from safe, and the difference is what the
/// gpt-oss line is here to remember. `the_header_counts_what_the_list_
/// holds` keeps those two numbers honest, because a count in prose is
/// the first thing to rot when a builder lands.
#[rustfmt::skip]
const NOT_YET_WIRED: &[(&str, &[&str])] = &[
    // The three families `wire()` has builders for, and the only three
    // that can serve a checkpoint end to end today.
    ("llama_like", &[]),
    ("qwen3_5", &[]),
    ("gemma_4", &[]),
    // WIRED, and the two missing names were the smaller half of it. This
    // read "both halves of the seam chose different words" -- true, and
    // it hid the real defect, which was that the word `mlp_norm` DID
    // resolve, to `post_attention_layernorm`, where gemma-2's forward
    // means `pre_feedforward_layernorm`. A test that asks what a family
    // can NAME cannot see a name that resolves to the wrong tensor; the
    // sandwich placement's own test in `weight_names` reads the target.
    ("gemma_2", &[]),
    // WAS the row that bit: the only family with both a `FACTS_ROWS`
    // entry in the CUDA shell and a Prefill arm, so a gpt-oss checkpoint
    // loaded, reported itself healthy, and died at its first fire on
    // `UnknownWeight("layer.0.router")`. Wired now. The families below
    // owe the same debt and are not yet reachable, so theirs is not due.
    ("gpt_oss", &[]),
    // Its two sandwich norms went with gemma-2's -- same placement, same
    // branch. What is left is the AltUp and Laurel machinery, which is
    // gemma3n's alone.
    ("gemma3n", &[
        "layer.*.altup_correct_norm",
        "layer.*.altup_norm",
        "layer.*.laurel_post_norm",
    ]),
    // MLA and the latent cache: three families, one shape. `kv_b_proj`
    // and `q_a_norm` are the latent projection's two halves and all
    // three name them.
    // AND SIX HYPER-CONNECTION NAMES, which are this family's alone. Every
    // `hc_pre` and the one `hc_head` reads an affine pair -- a `scale` and
    // a `base` the kernel dereferences per token -- and a layer states two
    // pre-mixes, one before its attention and one before its MLP, so the
    // pairs do not share. The head's is not layer-scoped because the
    // collapse runs once for the whole tower.
    //
    // Trace names on the same footing as `attn_sink` and `router_bias`
    // above: no witnessed checkpoint spells them, so `project.rs` claims
    // none of them in the manifest and they are owed here instead.
    ("deepseek_v4", &[
        "hc_head_base",
        "hc_head_scale",
        "layer.*.attn_sink",
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.hc_attn_base",
        "layer.*.hc_attn_scale",
        "layer.*.hc_mlp_base",
        "layer.*.hc_mlp_scale",
        "layer.*.kv_norm",
        "layer.*.router_bias",
    ]),
    // The DSA indexer's LayerNorm joins its three projections, and for the
    // reason `glm_5/project.rs` gives about all five at once: the
    // checkpoint's spelling for anything under the indexer is not written
    // down in this tree. The norm is a PAIR because the kernel subtracts
    // the row mean and adds a bias -- `w[d]` and `b[d]`, both per element.
    ("glm5", &[
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.idx_k_norm",
        "layer.*.idx_k_norm_bias",
        "layer.*.idx_weights_proj",
        "layer.*.idx_wk",
        "layer.*.idx_wq_b",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("kimi_k2", &[
        "layer.*.experts.down_packed",
        "layer.*.experts.down_scale",
        "layer.*.experts.gate_packed",
        "layer.*.experts.gate_scale",
        "layer.*.experts.up_packed",
        "layer.*.experts.up_scale",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("kimi_k3", &[
        "layer.*.attn_res_norm",
        "layer.*.attn_res_proj",
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.kda_a_log",
        "layer.*.kda_dt_bias",
        "layer.*.kda_k_conv",
        "layer.*.kda_o_norm",
        "layer.*.kda_q_conv",
        "layer.*.kda_v_conv",
        "layer.*.kv_a_norm",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("nemotron_h", &[
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.up",
        "layer.*.mamba_a_log",
        "layer.*.mamba_conv",
        "layer.*.mamba_d",
        "layer.*.mamba_dt_bias",
        "layer.*.mamba_norm",
        "layer.*.norm",
        "layer.*.router",
        "layer.*.router_bias",
    ]),
];

/// The header's two numbers are the list's two numbers.
///
/// The sentence above used to read "`gpt_oss` is the row that bites
/// TODAY ... the other eight are not yet reachable", with the gpt-oss
/// entry ten lines below it saying "Wired now". Both halves stale, in
/// opposite directions, in one paragraph: the example had been fixed and
/// the count had never been recounted. A number in prose beside a list
/// is a second reading of that list, so it gets held to it.
#[test]
fn the_header_counts_what_the_list_holds() {
    let wired = NOT_YET_WIRED.iter().filter(|(_, w)| w.is_empty()).count();
    let owing = NOT_YET_WIRED.len() - wired;
    assert_eq!(
        (wired, owing),
        (5, 6),
        "the list now holds {wired} wired and {owing} owing; the doc on \
         `NOT_YET_WIRED` says FIVE and SIX. A family whose list emptied \
         is a builder landing — say so in both places, or the next \
         reader believes the older one."
    );
}

/// Every weight a family's decode plan names is one `wire()` can emit, or
/// is written down.
///
/// The failure message is the whole value: a name that JOINED means a
/// forward pass started asking for something no checkpoint can answer, and
/// a name that LEFT means a builder landed and the line should go.
#[test]
fn every_traced_weight_is_a_name_wire_can_emit() {
    let anchors = answerable(Scheme::LlamaLike);
    assert!(
        anchors.contains("layer.*.qkv") && anchors.contains("embed"),
        "the answerable set lost its anchors, so `wire()`'s shape changed \
         rather than a family's: {anchors:?}"
    );

    let expected: BTreeMap<&str, BTreeSet<&str>> = NOT_YET_WIRED
        .iter()
        .map(|(f, names)| (*f, names.iter().copied().collect()))
        .collect();

    let mut actual: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for (family, scheme, plan) in corpus() {
        let can = answerable(scheme);
        let missing: BTreeSet<String> = stems(&plan)
            .into_iter()
            .filter(|s| !can.contains(s))
            .collect();
        actual.insert(family, missing);
    }

    assert_eq!(
        actual.keys().copied().collect::<BTreeSet<_>>(),
        expected.keys().copied().collect::<BTreeSet<_>>(),
        "the family list moved: NOT_YET_WIRED and the corpus must name the \
         same families, or a family's seam is unchecked"
    );

    let mut report = String::new();
    for (family, missing) in &actual {
        let want: BTreeSet<String> = expected[family].iter().map(|s| (*s).to_string()).collect();
        for joined in missing.difference(&want) {
            report.push_str(&format!(
                "  {family}: `{joined}` is named by the forward pass and \
                 `wire()` can never emit it — the first fire that reaches \
                 this weight fails with UnknownWeight.\n"
            ));
        }
        for left in want.difference(missing) {
            report.push_str(&format!(
                "  {family}: `{left}` is wired now — delete its line from \
                 NOT_YET_WIRED.\n"
            ));
        }
    }
    assert!(report.is_empty(), "the seam moved:\n{report}");
}

/// The fact that could only ever be false.
///
/// `serve.rs` derives kimi's fused latent projection as
/// `aliases.contains_key("layer.0.q_kv_a_fused")`. The contract publishes
/// that join and the forward consumes it — but `wire()` has no kimi
/// builder, so the alias is never created, the fact is permanently
/// `false`, and the fusion is paid for at load and never read. Were it
/// ever true, the launch would fail with `UnknownWeight`.
///
/// Its own test because it is the SHARPEST case: not a name a fire has not
/// reached yet, but a name whose absence is silently load-bearing
/// somewhere else. When a kimi builder lands this flips, and the driver's
/// derivation starts telling the truth for the first time.
#[test]
fn kimis_fused_latent_projection_is_still_unreachable() {
    let can = answerable(Scheme::LlamaLike);
    assert!(
        !can.contains("layer.*.q_kv_a_fused"),
        "a kimi builder landed: `wire()` can now emit `q_kv_a_fused`, so \
         `serve`'s `aliases.contains_key(\"layer.0.q_kv_a_fused\")` is \
         no longer permanently false. Check that the forward, the contract \
         and the driver agree before deleting this test."
    );
}
