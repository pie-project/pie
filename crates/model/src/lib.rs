//! What a model IS: the one registry that turns a model's name into an
//! implementation, and the implementations it dispatches to.
//!
//! ## Three aspects, one crate
//!
//! * [`contract`] — a row to the author that writes its load contract.
//! * [`instruct`] — a row to the chat template that formats for it.
//! * `<generation>::forward` — the forward pass, written in
//!   `model-compiler`'s tracing eDSL. UNGATED, like the rest of the
//!   crate's shape vocabulary: it used to sit behind a non-default
//!   `forward` feature on the argument that `pie model convert` wants
//!   the first two and not a tracer, and every consumer in the tree
//!   turned it on anyway while `model-compiler` stayed non-optional.
//!
//! The first two are ASPECTS OF A ROW and the third is not: a forward pass is
//! reached by a driver naming a family's text directly, not by a row. That
//! asymmetry is real and is why `forward` sits on the generation module
//! rather than beside the other two on [`catalog::Variant`].
//!
//! Neither of the first two is a registry any more, and saying so twice is
//! how this file used to contradict itself. [`contract::author`] takes a
//! `&dyn Variant` and calls `row.author`; [`instruct::create`] takes an id,
//! asks [`catalog::find`] for the row, and calls `row.chat`. The lookup they
//! used to each own — one keyed on `config.json`'s `model_type`, one on
//! `architectures[0]` — is [`catalog`], once, for both. See "The registries"
//! below for what those separate keys cost.
//!
//! There was a fourth thing here — `ffi`, a `#[repr(C)]` door and a committed
//! `include/pie_forward.h`, 5,382 lines of it. It existed so the C++ drivers
//! could trace a declaration across the ABI. Both drivers are Rust now and
//! call this crate through its own types, so the door had no building behind
//! it; `worker/src/embedded_driver.rs` had already recorded that when it
//! dropped the link anchors that kept the entry points alive. It is deleted
//! rather than kept warm: an ABI with no caller does not stay correct, it
//! just stays compiling.
//!
//! They used to partition the model space differently — templates split by
//! release, contracts by storage schema — and the whole point of one row is
//! that they no longer do. A generation is reached through a *row*, never as
//! a module path, and every N:1 reuse is written out on that one partition: a
//! dozen `model_type`s share Llama 3's dense author, and ChatML serves four
//! vendors. Two rows may still answer the same author and different
//! templates; what they can no longer do is disagree about which model they
//! are talking about.
//!
//! A generation with no implementation of its own gets no module: `mixtral`,
//! `deepseek_v2`, `deepseek_v3`, `ministral3`, `gemma3n`, `kimi_k25` and
//! `qwen3_6` are rows naming the generation whose implementation they share.
//!
//! ## Why one crate, when a generation used to be one
//!
//! Twenty generation crates, a `common`, and this registry were twenty-two
//! packages. The crate boundary was there to stop one generation reaching into
//! another — and it had already stopped doing that: `qwen_2` depended on
//! `qwen_3` for ChatML, because the thing they share is real. What the boundary
//! still cost was `common`, a crate that existed only because siblings needed
//! somewhere to share from, and a feature matrix that named every generation
//! twice.
//!
//! So the isolation moves from the crate system to a `#[test]`
//! (`tests/sibling_isolation.rs`), stated as one rule:
//!
//! > **A generation module may name [`families`] and the shared root. It may
//! > not name a sibling.**
//!
//! And the sharing that broke the old rule gets the home the rule implies:
//! ChatML — which `qwen3`, `qwen3_5`, `nemotron_h`, `glm_moe_dsa` and `qwen2`
//! all bind — is [`shared::chatml`], not a file inside whichever generation
//! happened to write it down first.
//!
//! The module layout mirrors the old crate layout on purpose. A generation
//! that grows big enough to want its own crate again can be lifted back out
//! mechanically.
//!
//! ## The shared root
//!
//! Everything a generation may reach: the authoring DSL ([`builder`]), the
//! vocabulary it is parameterized by ([`facts`], [`policy`]), the shared shapes
//! ([`probe`], [`moe`], [`mlx`]), and the chat trait and [`decoders`] behind
//! it. The test for whether something belongs here is unchanged and still
//! enforced (`tests/common_is_thin.rs`): it must be *about models in general*,
//! never a fact about one generation's checkpoints or templates.
//!
//! **Thin means no knowledge, not few lines.** [`builder`] is the largest file
//! here and that is correct — it is the authoring DSL, and every generation
//! calls its passes directly. Bulk is only a smell when it is knowledge.
//!
//! [`multimodal`] is the exception: host-side image and audio decode is a
//! third aspect and it is family-aware (it dispatches on a `VisionArch`), so
//! it is neither shared vocabulary nor one generation's fact. It sits at the
//! root until it is split per generation the way chat and contract are.
//!
//! The model *service* — the global model/tokenizer cache — is not here. It
//! was runtime machinery rather than family knowledge and lives in
//! `engine/src/model.rs`; only [`ModelMetadata`] stayed, because the worker
//! reads it without linking the runtime.

// ── The descriptor aspect is GONE ────────────────────────────────────
//
// `config` (1,563 lines), `descriptor` (443) and `facts` (153) are
// deleted, and so is `deployment_cuda` (1,436): a `pie.model/1`
// descriptor normalized from a 136-field schema, a reader that parsed it
// back, a `ModelFacts` projection of the result, and eleven per-family
// derivations over all of it. That is 3,595 lines whose entire job was
// to turn strings a checkpoint says about itself into numbers, ONCE PER
// REPRESENTATION, with nothing holding the representations to each
// other.
//
// The numbers were always the same numbers. They are stated once now, in
// `catalog`, and the only thing left that a file has to answer is the
// declared encoding — see `encoding`.

/// What a driver needs to serve a checkpoint, with no family name in it.
///
/// The answer to "what is this model made of", shaped so that a driver
/// CANNOT ask the question again. See the module doc for why this is a
/// value rather than the `Box<dyn PlannedFamily>` the drivers built per
/// fire.
pub mod deployment;

// The `Instruct` trait and its events, AND the `create` registry that picks an
// implementation for an `arch_name`. Those were two files in two crates: a
// generation could not depend on the registry that dispatches to it, so the
// vocabulary had to sit below both. One crate, one module.
#[cfg(feature = "chat")]
pub mod instruct;

// ── Cross-generation sharing ─────────────────────────────────────────
// The ONLY thing a generation module may name. What lives there is what more
// than one generation binds -- not what one generation wrote first -- plus
// the general vocabulary (`builder`, `policy`, `probe`, `moe`, `mlx`,
// `weight_names`, `decoders`) that used to be scattered across this root.
//
// Ungated as a whole because its contents are not: each module inside carries
// its own aspect gate, and the two kinds of shared thing do not share one.
// The root that is left is the catalog and its answers.
pub mod shared;

// ── The registries ───────────────────────────────────────────────────
//
// `catalog` is THE registry. There used to be three -- `contract::HF_ROWS`
// keyed on a `config.json` `model_type`, `deployment_cuda::FACTS_ROWS` keyed
// on the same string, and `instruct::create` keyed on `architectures[0]` --
// and nothing held them to each other. `qwen3_moe` authored as a GDN mixture
// and deployed as a dense llama in the same tree; the chat table's `_ =>`
// arm answered ChatML for a generation that had never heard of `<|im_end|>`.
//
// One row now answers all three, so the answers cannot disagree.

/// The tensor manifest: what a checkpoint of a row MUST contain.
///
/// Identity and validation are the same operation here, which is the
/// whole reason a manifest is first-class. Ungated: every aspect that
/// can name a row can ask what it is made of.
pub mod manifest;

/// The catalog: one row per model, one row per answer.
pub mod catalog;

/// What a checkpoint's FILES say about how its numbers are stored.
///
/// Beside the catalog rather than in it, because an encoding is a
/// property of a file and a row is a property of a model. Qwen3-8B is
/// one row and four downloads.
pub mod encoding;

/// The load path itself: a row in, a plan out, stated once for every
/// driver. Sits with `contract` because it is that registry's caller.
#[cfg(feature = "contract")]
pub mod boot;
#[cfg(feature = "contract")]
pub mod contract;
/// The ingest aspect: a foreign checkpoint vocabulary in, this crate's out.
/// Sits with `contract` because it is the layer below it -- the same
/// question, one step earlier.
#[cfg(feature = "contract")]
pub mod ingest;
#[cfg(feature = "chat")]
pub mod multimodal;

// ── The generations ──────────────────────────────────────────────────
//
// Named `<vendor>_<generation>`, a version's dots as underscores. Each gates
// its own aspects, so a generation that implements only one is an empty module
// under the other.
//
// `qwen_3` used to be absent by the families rule -- everything it held was
// ChatML, which four other generations bind. It is back, because a
// generation now holds something no family can: its ROWS. `shared::chatml`
// still speaks for it.
pub mod csm;
pub mod deepseek_r1;
pub mod deepseek_v4;
pub mod gemma_2;
pub mod gemma_3;
pub mod gemma_3n;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k2;
pub mod kimi_k3;
pub mod llama_2;
pub mod llama_3;
pub mod mistral_3;
pub mod nemotron_h;
pub mod olmo_2;
pub mod olmo_3;
pub mod phi_3;
pub mod qwen_2;
pub mod qwen_3;
pub mod qwen_3_5;

/// One row that describes no real checkpoint, so that a test can afford to
/// write one. Absent unless asked for; see the module for why a closed set
/// needs a door and why this is not one.
#[cfg(feature = "test-rows")]
pub mod test_rows;

// ── Neither aspect ───────────────────────────────────────────────────
//
// What a served model's compiled metadata IS. Outside both aspects because the
// worker reads it without linking either.
mod metadata;
pub use metadata::ModelMetadata;
