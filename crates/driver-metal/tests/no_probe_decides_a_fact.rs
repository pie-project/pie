//! No tensor probe decides a model fact, and the refusal happens first.
//!
//! # Why this exists
//!
//! `model/text.rs` held `facts_from_with`, which rebuilt the model's own
//! twenty-nine `LlamaLikeFacts` out of the projected geometry and NINE
//! `has_tensor` probes:
//!
//! | probe | fact it decided |
//! |---|---|
//! | `layers.0.self_attn.q_norm.weight` | per-head q/k norms |
//! | `layers.0.self_attn.qkv_proj.weight` | a fused QKV projection |
//! | `layers.0.self_attn.q_proj.bias` | the Qwen-2 attention biases |
//! | `layers.0.mlp.gate.weight` | a routed FFN |
//! | `layers.0.mlp.shared_expert.gate_proj.weight` | a shared expert |
//! | `layers.0.pre_feedforward_layernorm.weight` | the sandwich norm AND the norm variant |
//! | `layers.0.self_attn.sinks` | attention sinks |
//! | `per_layer_model_projection.weight` | gemma-4's per-layer embeddings |
//! | `layers.0.mlp.experts.gate_proj.weight` | the expert bank's encoding |
//!
//! Eight of those nine are MODEL facts, and every one of them was already
//! stated by the `model::catalog` row the checkpoint had been matched to.
//! Deriving them a second time is not redundancy, it is a second answer: the
//! norm-variant probe read `(1 + w)` for gemma-4 — whose gains are a plain
//! multiplier — because gemma-4 ships the norm the probe asked about. The
//! result agreed with MLX to three digits on its largest value and was off by
//! a third on its ordinary ones, which is a model that runs, never faults and
//! is quietly wrong.
//!
//! The ninth is an ENCODING, and it stays: `mlx-community/gpt-oss-20b-MXFP4-Q4`
//! names 98 tensors as affine/64/4 and leaves the expert banks at the
//! top-level mxfp4/32 default, and no row may state which format a publisher
//! packed a bank in.
//!
//! # What it checks
//!
//! Two properties of `serve/load.rs`, both by reading it:
//!
//! 1. The load path asks the tensors exactly one question, and the answer
//!    feeds a `MetalBinding` rather than a fact.
//! 2. The "no Metal text for this row" refusal is REACHED BEFORE STAGING. The
//!    placement is the whole value of that refusal — on the 31B gemma the
//!    alternative is 17 GB of weights read to reach an answer identification
//!    had already settled — and nothing in the type system holds a statement
//!    in place. Moving it below `weights::load::load` is a two-line diff that
//!    compiles, passes every other test, and costs a minute per refused load.
//!
//! # What it does not check
//!
//! That the facts are RIGHT. `crates/model`'s own tests own that, and this
//! could not add to them: the point of the refactor is that this crate has no
//! opinion left to check.
//!
//! # Why it is not gated
//!
//! It reads files. `serve/load.rs` is on disk whether or not `metal-4`
//! compiles it, so this runs in the portable half and the no-GPU job catches
//! a re-introduced probe without a Mac — the same argument `layering.rs`
//! makes for itself, and the reason neither is in `Cargo.toml`'s
//! `required-features` list.

use std::path::{Path, PathBuf};

/// `src/serve/load.rs`, which is the whole subject.
///
/// One file rather than a walk, because the claim is about one path. A probe
/// somewhere else in the crate is a different question — `lowering/` reads
/// tensor NAMES constantly, and it must: resolving a text's stated weight to
/// a staged buffer is exactly its job. What must not happen is a name being
/// asked about in order to decide what the model IS, and `serve/load.rs` is
/// where that decision would have to live, because it is the only place with
/// both the staged tensors and the row in scope.
fn load_path() -> (PathBuf, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/serve/load.rs");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} could not be read ({e}). The load path moved and this guard is \
             now looking at nothing, which is the failure mode \
             `no_family_names.rs` records `one_normalizer.rs` having",
            path.display()
        )
    });
    (path, text)
}

/// Lines that are not comments, with their 1-based numbers.
///
/// Comments are excluded for the reason `no_family_names.rs` excludes them:
/// the paragraphs above the refusals in `serve/load.rs` QUOTE the probes they
/// deleted, name the tensors they used to ask about, and would otherwise be
/// counted as the thing they are recording the removal of. A doc that cannot
/// mention what it replaced is a doc that gets written without the argument.
fn code(text: &str) -> Vec<(usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.starts_with("//") && !l.is_empty())
        .collect()
}

/// The one tensor name the load path may spell, and it decides an encoding.
const ENCODING_PROBE: &str = "loaded.mxfp4.contains";

/// The load path asks the tensors nothing that decides a model fact.
///
/// Measured as: no line of code in `serve/load.rs` contains a per-layer
/// tensor name. That is a blunt instrument on purpose. A tensor name in this
/// file is either a probe or a comment, comments are already excluded, and
/// the ONE legitimate probe left names its tensor in
/// `model/binding.rs::EXPERT_BANK` — a constant, in the module that owns the
/// binding, precisely so that "which tensors does the load ask about" is a
/// question answerable by reading one file.
#[test]
fn no_tensor_name_is_spelled_in_the_load_path() {
    let (path, text) = load_path();
    // The shapes a per-layer weight name takes in this checkpoint vocabulary.
    // `layers.0.` is mlx_lm's flattened spelling and the one every deleted
    // probe used; the other two are the prologue and epilogue names.
    const NAMES: &[&str] = &["layers.0.", "model.layers.", "embed_tokens", "lm_head."];
    let found: Vec<String> = code(&text)
        .into_iter()
        .filter(|(_, l)| NAMES.iter().any(|n| l.contains(n)))
        .map(|(i, l)| format!("{i}: {l}"))
        .collect();
    assert!(
        found.is_empty(),
        "{} spells a tensor name in code. Every model fact this driver needs \
         is stated by the `model::catalog` row the checkpoint was matched to, \
         and a probe here is a second answer to a question that already has \
         one — which is how the norm variant came to be decided by whether \
         gemma-4 shipped a `pre_feedforward_layernorm`:\n  {}",
        path.display(),
        found.join("\n  ")
    );
}

/// The load path's only question of the tensors is about an ENCODING.
///
/// The companion to the test above, and the half that would notice a probe
/// spelled indirectly — through a helper, a constant or a name built at run
/// time. There is exactly one place the staged tensors are consulted, it is
/// the MXFP4 set rather than the tensor index, and its answer goes into a
/// `MetalBinding`.
///
/// `loaded.tensors.contains_key` is named in the failure message because it
/// is what the deleted code called, twice, and what a re-introduced probe
/// would most naturally reach for.
#[test]
fn the_load_paths_one_tensor_question_feeds_a_binding() {
    let (path, text) = load_path();
    let probes: Vec<(usize, &str)> = code(&text)
        .into_iter()
        .filter(|(_, l)| l.contains(".contains_key(") || l.contains(".contains("))
        .collect();
    assert_eq!(
        probes.len(),
        1,
        "{} asks the staged tensors {} questions; it may ask one, and that \
         one must decide an ENCODING. `loaded.tensors.contains_key(..)` is \
         what `facts_from_with` called to decide a model fact:\n  {}",
        path.display(),
        probes.len(),
        probes
            .iter()
            .map(|(i, l)| format!("{i}: {l}"))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    let (line, src) = probes[0];
    assert!(
        src.contains(ENCODING_PROBE),
        "{}:{line} asks the tensors `{src}`, which is not the expert bank's \
         format. The one probe this load may make is `{ENCODING_PROBE}`, and \
         it answers what a publisher packed a bank in — not what the model is",
        path.display()
    );
    assert!(
        src.contains("binding::observed"),
        "{}:{line} probes the tensors outside `binding::observed`. That \
         function takes an affine format and one question deliberately: it \
         cannot see the geometry, so it cannot smuggle a model fact into a \
         binding, and a probe that reaches the load path by another route \
         loses that guarantee",
        path.display()
    );
}

/// The "no Metal text" refusal is reached before a byte is staged.
///
/// # The placement is the whole value of the refusal
///
/// A row this build has no Metal text for is refused by `serve/load.rs`
/// BEFORE `weights::load::load` reads the checkpoint. The old code's own
/// message admitted what the other order costs — *"the checkpoint loaded, but
/// nothing states its forward pass"* — and on `gemma-4-31b-it-4bit` that
/// sentence is 17 GB of staging spent to reach an answer `catalog::identify`
/// had settled before any of it started.
///
/// It is the same rule `weights::stage::fits_on_this_gpu` states for itself:
/// asked before a byte is read is the only moment it can be asked usefully.
/// Nothing in the type system holds either in place, which is what this test
/// is for — the refusal and the staging call are two statements in one
/// function, and swapping them compiles.
///
/// The refusal is answered with `binding::ANY_ENCODING` rather than the real
/// binding, and that is what MAKES the placement possible: the real binding
/// needs `moe_mxfp4`, which needs the tensors, which needs the staging this
/// gate exists to precede. It is sound because a row that refuses Metal
/// refuses it for every encoding — which is not an argument left in prose,
/// it is `binding::a_row_is_served_the_same_way_at_every_encoding`.
#[test]
fn the_no_text_refusal_precedes_the_staging_call() {
    let (path, text) = load_path();
    let lines = code(&text);
    let at = |needle: &str| -> Vec<usize> {
        lines
            .iter()
            .filter(|(_, l)| l.contains(needle))
            .map(|(i, _)| *i)
            .collect()
    };

    let refusals = at("binding::serves(");
    assert_eq!(
        refusals.len(),
        1,
        "{} asks the row whether this build has a Metal text {} times; one \
         gate is the point. Two gates for one question is what `serves(arch)` \
         and the gelu-gate refusal were, and they disagreed about gemma-4",
        path.display(),
        refusals.len()
    );
    let staging = at("weights::load::load(");
    assert_eq!(
        staging.len(),
        1,
        "{} stages weights from {} places; this guard orders the refusal \
         against THE staging call and cannot do that if there are several",
        path.display(),
        staging.len()
    );

    assert!(
        refusals[0] < staging[0],
        "{}: the Metal-text refusal is at line {} and the weights are staged \
         at line {}. A refusal after staging is 17 GB of reading spent to \
         reach an answer the row gave for free — the exact cost the old \
         `serves(arch)`/gelu-gate pairing paid on every gemma-4, and the \
         reason this gate sits where it does.",
        path.display(),
        refusals[0],
        staging[0]
    );
}

/// The gelu-gate contradiction has no refusal left, and no probe either.
///
/// What stood at `serve/load.rs:217` rejected a checkpoint shipping
/// `pre_feedforward_layernorm` while the projected geometry could only state
/// a SiLU gate. Its message named its own repair — *"either an activation on
/// `Deployment` or a `Variant::trace` that can be asked for a Metal text"* —
/// and the second one exists: the row's text states the activation, this
/// driver states none, and two statements that cannot disagree cannot
/// contradict each other.
///
/// Pinned because the deletion is the load-bearing half of this change and it
/// leaves no symbol behind to reference. A reader who does not know why it
/// went will re-add it the first time a gemma checkpoint is refused somewhere
/// less obvious — and re-adding it means re-adding a tensor probe, which is
/// what the two tests above are for.
#[test]
fn the_gelu_contradiction_is_not_representable() {
    let (path, text) = load_path();
    let found: Vec<String> = code(&text)
        .into_iter()
        .filter(|(_, l)| l.contains("gelu_gate") || l.contains("pre_feedforward"))
        .map(|(i, l)| format!("{i}: {l}"))
        .collect();
    assert!(
        found.is_empty(),
        "{} is deciding something about an ACTIVATION again. A `Deployment` \
         states none and the row's text states it — the driver has no third \
         answer to reconcile, which is why the refusal that used to compare \
         them is gone:\n  {}",
        path.display(),
        found.join("\n  ")
    );
}
