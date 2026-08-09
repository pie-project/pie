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
//! That same checkpoint later added a TENTH, for the same reason. Its
//! `quantization` block carries 122 overrides and not 98: the other 24 are
//! the `mlp.router` gates at affine/64/**8**, a second affine point in one
//! file. `layers.0.mlp.router.weight` is asked its point, and the alternative
//! — a rule here about what a router's name looks like — would have been a
//! model fact decided by a probe, which is the thing this file exists to
//! prevent. Two probes, both encodings, is the invariant holding rather than
//! bending.
//!
//! # What it checks
//!
//! Two properties of `serve/load.rs`, both by reading it:
//!
//! 1. Every question the load path puts to the tensors decides an ENCODING,
//!    and each answer feeds a `MetalBinding` rather than a fact.
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

/// Every way `serve/load.rs` may mention the staged tensors, and what each
/// one decides.
///
/// `loaded` is the only handle that function has on the checkpoint's bytes —
/// one `let` at line 192, one function in the file — so a list of the ways it
/// is touched is a COMPLETE list of the questions this load asks. That is
/// what makes the test below an instrument rather than a restatement: a new
/// probe cannot reach the tensors without adding a line this list does not
/// have.
///
/// Three of the six entries are not questions: the `let` that produces the
/// handle, the move that gives it away, and the region walk. They are here
/// because a list that only held questions would need a rule for what a
/// question looks like, and the rule is what a new probe would be written to
/// slip past. Every MENTION, adjudicated, has no such gap.
const TENSOR_QUESTIONS: &[(&str, &str)] = &[
    (
        "loaded.affine_point(",
        "which affine point the checkpoint's dense projections were written \
         at, and the refusal when there is more than one the driver can hold",
    ),
    (
        "loaded.affine_point_of(",
        "which point ONE named tensor was written at — the router gate, whose \
         width gpt-oss states separately from its stack's",
    ),
    (
        ENCODING_PROBE,
        "whether the expert banks reached the device still in the \
         checkpoint's own MXFP4 rather than an affine repack",
    ),
    (
        "&loaded.regions",
        "not a question of the tensors at all: the staged regions, walked to \
         record which buffer each weight address belongs to",
    ),
    (
        "let loaded = crate::weights::load::load(",
        "not a question either: this is where the bytes arrive. It is the \
         only `let` that produces the handle, which is what makes this list \
         complete",
    ),
    (
        "self.model = Some(loaded)",
        "and this is where they leave, moved to the field that owns them. \
         Neither end asks anything",
    ),
];

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

/// Every question the load path puts to the tensors is about an ENCODING.
///
/// The companion to the test above, and the half that would notice a probe
/// spelled indirectly — through a helper, a constant or a name built at run
/// time. The other test forbids a tensor NAME here; this one forbids a
/// tensor QUESTION here, whatever it is spelled with.
///
/// # Why "one question" became two, and why that is not a weakening
///
/// This test used to assert the count was exactly one. It was, until
/// gpt-oss-20b-MXFP4-Q4 turned out to publish TWO affine points in one file:
/// 98 dense tensors at 64/4 and 24 `mlp.router` gates at 64/**8**. A second
/// question was the only honest answer, because the alternative is a rule
/// about what a router's name looks like — which is a MODEL fact, decided
/// here, which is the exact thing this file exists to prevent.
///
/// So the invariant was never the number. It is that each question's answer
/// is an ENCODING: what format a publisher packed some bytes in, which no
/// `model::catalog` row may state and none does. [`TENSOR_QUESTIONS`] carries
/// that adjudication per question, in prose, next to the expression that asks
/// it — so adding a third means writing down what it decides, and a probe
/// that decides a model fact has nowhere to write itself down.
///
/// # The narrow signature is still the guarantee
///
/// Both encoding questions are passed to `binding::observed` as CLOSURES
/// rather than answered here, and the assertion below pins them to the lines
/// directly beneath that call. `observed` cannot see the geometry, so it
/// cannot smuggle a model fact into a binding; a probe answered on its own
/// line, before the call, would have the whole of `load_model` in scope and
/// would lose that.
#[test]
fn every_tensor_question_the_load_asks_decides_an_encoding() {
    let (path, text) = load_path();
    let lines = code(&text);
    let unaccounted: Vec<String> = lines
        .iter()
        .filter(|(_, l)| l.contains("loaded"))
        .filter(|(_, l)| !TENSOR_QUESTIONS.iter().any(|(q, _)| l.contains(q)))
        .map(|(i, l)| format!("{i}: {l}"))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "{} consults the staged tensors in a way `TENSOR_QUESTIONS` does not \
         account for. `loaded` is this function's only handle on the \
         checkpoint's bytes, so every such line is a question this load asks \
         — and the price of asking one is writing down what it decides, \
         beside the others, where it can be read as an encoding or caught as \
         a fact. `loaded.tensors.contains_key(..)` is what `facts_from_with` \
         called to decide a model fact:\n  {}",
        path.display(),
        unaccounted.join("\n  ")
    );
    // The two ENCODING probes sit in the argument list of `binding::observed`
    // and nowhere else. Pinned positionally rather than by presence, because
    // a probe hoisted to its own `let` above the call still "appears in the
    // file" while having lost the one property that makes it safe.
    let at = lines
        .iter()
        .position(|(_, l)| l.contains("binding::observed("))
        .unwrap_or_else(|| panic!("{} no longer calls `binding::observed`", path.display()));
    let arms: Vec<&str> = lines[at + 1..].iter().take(3).map(|(_, l)| *l).collect();
    for probe in ["loaded.affine_point_of(", ENCODING_PROBE] {
        assert!(
            arms.iter().any(|l| l.contains(probe)),
            "{}:{} does not hand `{probe}` to `binding::observed` as an \
             argument. That function takes its questions as closures \
             deliberately: it cannot see the geometry, so it cannot smuggle a \
             model fact into a binding, and a probe answered before the call \
             loses that guarantee. What follows the call is:\n  {}",
            path.display(),
            lines[at].0,
            arms.join("\n  ")
        );
    }
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
