//! What a model IS: the two registries that turn a model's name into an
//! implementation, and the implementations they dispatch to.
//!
//! ## Three aspects, one crate
//!
//! * [`contract`] — `model_type` to the author that writes its load contract.
//! * [`instruct`] — `arch_name` to the chat template that formats for it.
//! * `<generation>::forward` — the forward pass, written in
//!   `model-compiler`'s tracing eDSL, and [`ffi`] the door a driver reaches it
//!   through. Behind the non-default `forward` feature, because `pie model
//!   convert` wants the first two and not a tracer.
//!
//! The first two are REGISTRIES and the third is not: a forward pass is
//! reached by the driver naming a family over the C ABI, not by a row. That
//! asymmetry is real and is why `forward` sits on the generation module
//! rather than in a table beside the other two.
//!
//! They deliberately partition the model space differently: templates split by
//! release, contracts by storage schema. So a generation is reached through a
//! *row*, never as a module path, and every N:1 reuse is written out — a dozen
//! `model_type`s share Llama 3's dense author, and ChatML serves four vendors.
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
//! all bind — is [`families::chatml`], not a file inside whichever generation
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

// ── The shared root: the contract aspect ─────────────────────────────
#[cfg(feature = "contract")]
pub mod builder;
#[cfg(feature = "contract")]
pub mod facts;
#[cfg(feature = "contract")]
pub mod mlx;
#[cfg(feature = "contract")]
pub mod moe;
#[cfg(feature = "contract")]
pub mod policy;
#[cfg(feature = "contract")]
pub mod probe;

// ── The shared root: the chat aspect ─────────────────────────────────
#[cfg(feature = "chat")]
pub mod decoders;
// The `Instruct` trait and its events, AND the `create` registry that picks an
// implementation for an `arch_name`. Those were two files in two crates: a
// generation could not depend on the registry that dispatches to it, so the
// vocabulary had to sit below both. One crate, one module.
#[cfg(feature = "chat")]
pub mod instruct;

// ── Cross-generation sharing ─────────────────────────────────────────
// The only thing a generation module may name besides the root. What lives
// here is what more than one generation binds -- not what one generation
// wrote first.
// Gated on either aspect that has a family module in it: `chatml` is chat's,
// `llama_like`'s forward pass is the forward aspect's. The directory is the
// home for what more than one generation binds, whichever aspect binds it.
#[cfg(any(feature = "chat", feature = "forward"))]
pub mod families;

/// The `#[repr(C)]` boundary a driver traces a declaration across.
///
/// `types` is the published vocabulary, `arena` turns a traced
/// `ForwardPlan` into it, and `entry` holds the `extern "C"` functions.
/// The committed `include/pie_forward.h` is the C view of exactly those.
///
/// It is here, and not in `model-compiler`, because it is how a DRIVER
/// reaches a declaration — and a declaration is a model's. The toolchain
/// that traces one has no business owning the door.
#[cfg(feature = "forward")]
pub mod ffi;

// ── The registries ───────────────────────────────────────────────────
#[cfg(feature = "contract")]
pub mod contract;
#[cfg(feature = "chat")]
pub mod multimodal;

// ── The generations ──────────────────────────────────────────────────
//
// Named `<vendor>_<generation>`, a version's dots as underscores. Each gates
// its own aspects, so a generation that implements only one is an empty module
// under the other.
//
// `qwen_3` is absent by the families rule: everything it held was ChatML,
// which four other generations bind, so it is `families::chatml`.
pub mod csm;
pub mod deepseek_r1;
pub mod deepseek_v4;
pub mod gemma_2;
pub mod gemma_3;
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
pub mod qwen_3_5;

// ── Neither aspect ───────────────────────────────────────────────────
//
// What a served model's compiled metadata IS. Outside both aspects because the
// worker reads it without linking either.
mod metadata;
pub use metadata::ModelMetadata;
