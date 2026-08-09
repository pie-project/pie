//! Do the names a plan states exist in a checkpoint, and does `src/names.rs`
//! close the gap?
//!
//! `src/names.rs` is 360 lines that were copied into this crate from
//! `driver-vulkan` and never checked against a real artifact. It is a
//! translation table: a plan binds `layer.0.down.zeros` and a loader publishes
//! `layers.0.mlp.down_proj.biases`. The claim it makes is precise and large --
//! *every* name a text binds resolves to a tensor the loader states -- and
//! until this file ran, nothing in this crate had measured any of it. Its own
//! unit tests hold the table against strings the same commit wrote down.
//!
//! `driver-vulkan/tests/checkpoint.rs` is the template and its numbers are
//! what this re-measures rather than assumes. This is not a loader and does
//! not pretend to be one.
//!
//! # What it measured, on real checkpoints, on this machine
//!
//! **Zero of 704**, on a stock `Qwen/Qwen3-0.6B` HuggingFace snapshot. Not one
//! weight name a qwen3 plan states is a tensor name that checkpoint holds. The
//! plan says `layer.0.down`; the checkpoint says
//! `model.layers.0.mlp.down_proj.weight`. And the plan wants `embed.scales`
//! and `embed.zeros`, which a bfloat16 checkpoint does not hold under ANY
//! spelling -- they are outputs of quantizing rather than tensors anyone
//! published. So a weight loader for this crate is a CONVERSION, not a lookup.
//!
//! **704 of 704 still disagree after the conversion**, because the two sides
//! use different conventions rather than different files: the loader publishes
//! `layers.0.mlp.down_proj.biases` where the text binds `layer.0.down.zeros`.
//!
//! **Through `Naming::mlx()`, nothing is left over** -- for qwen3. Every one of
//! the 704 resolves to a tensor the compiled plan publishes, and the sizes it
//! states are the model's own arithmetic: `embed` is
//! 151936 x 1024 / 2 = 77_791_232 bytes, and the whole bound model is
//! 335_372_288.
//!
//! Those four numbers are the Vulkan file's, to the byte, measured from a
//! DIFFERENT artifact -- that file's fixture is `mlx-community/Qwen3-0.6B-4bit`
//! and this one is the unquantised release with the encode run at load. Two
//! publishers, one answer. The table is not specialised to a repo.
//!
//! # And a finding: this crate's table is NOT `driver-vulkan`'s any more
//!
//! The brief for this file said `src/names.rs` was byte-identical to
//! `driver-vulkan/src/names.rs`. It is not, and the difference is load-bearing.
//! `driver-vulkan`'s table carries 22 roles and a `bias_suffix`; this one
//! carries 19 and folds `.bias` into `zero_point_suffix`. What that costs is
//! measured by [`every_name_the_six_texts_bind_is_one_this_table_can_spell`]
//! and again on a real checkpoint by [`the_loader_states_the_names_this_driver_binds`]:
//!
//! - **228 of the 4664 names** the six texts in `tests/arena.rs` bind are
//!   outside this table's shape entirely -- `spellings` answers with an empty
//!   vector, which the table's own doc calls "drift rather than a spelling
//!   this table has not learned".
//! - They are six spellings over two texts. qwen2.5-1.5B contributes 84:
//!   `{q,k,v}_bias`, three a layer over 28. gpt-oss-20B contributes 144:
//!   the same three over its 24 layers, plus `expert_{gate,up,down}.bias`,
//!   the additive term a routed expert bank carries beside its codec's planes.
//! - Every one of them is a tensor the loader DOES publish. Measured against
//!   the `openai/gpt-oss-20b` plan: `layers.0.self_attn.q_proj.bias` and
//!   `layers.0.mlp.experts.down_proj.bias` are both there, and
//!   `driver-vulkan`'s three extra roles resolve all 144 -- 727 of 727 rather
//!   than 583 of 727, and 11_177_235_072 bytes rather than 11_163_718_272.
//!
//! So this is not a checkpoint that lacks something. It is this crate's copy of
//! the table having been left behind when the texts gained attention biases
//! (`driver-vulkan/tests/checkpoint.rs` records that change: 648 weights became
//! 732) and when `.bias` was measured to be the additive term rather than the
//! codec's zero point. Two checks here fail because of it. **They are not
//! adjusted to pass**; the failure is the report, and the repair is three rows
//! and one field in `src/names.rs`, which this file may not write.
//!
//! # The `.bias` this table still offers as a zero point is not one
//!
//! `zero_point_suffix` here is `[".biases", ".bias"]`, and the comment beside
//! it says they are "the same role one character apart". Measured on gpt-oss,
//! they are not: `layers.0.self_attn.q_proj.bias` is `[4096]`, one value per
//! output row, and `layers.0.self_attn.q_proj.biases` is `[4096, 45]`, one per
//! group of 64. **A factor of 45.** The checkpoint publishes both, and 120 of
//! that model's 122 resolved zero points have both spellings available:
//! `.biases` is tried first and wins, and that ordering is the only thing
//! standing between this table and binding an 8 KiB additive bias where a
//! 368 KiB zero-point plane belongs -- on a backend where an out-of-bounds
//! storage read returns zero and says nothing.
//! [`the_zero_point_this_table_finds_is_a_zero_point`] pins it.
//!
//! # Why it must not skip, and what it does instead
//!
//! `PIE_CHECKPOINT` names snapshot directories, colon-separated, the way a
//! `PATH` does -- the convention the Vulkan file established, kept because a
//! build box has no checkpoint and a test that failed there is a test that gets
//! deleted.
//!
//! But a skip that could have been a measurement is the failure mode this file
//! was written against. Pointed at this machine's cache, the Vulkan file
//! reports `embeds [151936, 1024], which is no fixture this file states` and
//! passes green three times over, because both its fixtures are `*-4bit`
//! conversions that are not downloaded here. Three green tests, nothing
//! measured, and the crate's largest untested claim still untested.
//!
//! So with `PIE_CHECKPOINT` unset this walks `~/.cache/huggingface/hub` and
//! measures whatever it finds. A checkpoint is IDENTIFIED by its
//! `model.embed_tokens.weight` shape and its layer count, not by the directory
//! it came from -- a name a variable claims is not evidence, and the failure
//! mode of guessing wrong is not an error, it is this file reporting the
//! FIXTURE's names as ones the checkpoint is missing, which reads exactly like
//! a loader defect. The layer count is this file's addition to the Vulkan
//! version's rule and it is there because auto-discovery walks thirty
//! directories rather than one: two models can share an embedding table's
//! shape, and the second one to match would be measured against the first's
//! text.
//!
//! # What a checkpoint costs to reach, twice over
//!
//! **A stock `Qwen/Qwen3-0.6B` cannot be identified by this build.**
//! `catalog::identify` answers `qwen3-0.6b: unexpected lm_head`. That
//! snapshot's `config.json` says `tie_word_embeddings: true` and its
//! `model.safetensors` publishes `lm_head.weight` *and*
//! `model.embed_tokens.weight` -- tied and exported anyway, which is ordinary
//! for an HF export and which the manifest treats as a contradiction. Left
//! where it was found, the way `driver-vulkan` left it: loosening another
//! crate's manifest from a driver's test is how a refusal that meant something
//! becomes one nobody remembers. The row is taken by id and the shape is
//! checked here instead. `openai/gpt-oss-20b` identifies cleanly, so the
//! `identify` path is exercised rather than described -- see
//! [`the_catalog_places_the_checkpoints_it_can`].
//!
//! **An unquantised checkpoint refuses the default binding.**
//! `boot::compile_load_plan_for` with `Binding::MLX_IN_PLACE` answers `Metal
//! llama needs quantized weights: this checkpoint carries no `.scales`
//! tensors`, and that message names the other way in: encode the floats at
//! load. So [`compiled_plan_for`] takes the documented path first and falls
//! back to the `RuntimeQuant::Int4` policy for a bf16 release, printing which
//! it used and holding it against what the fixture says to expect. That is not
//! a workaround -- `mlx.rs` states the two as equally valid ways to serve an
//! unquantized release -- and it is what turns the one checkpoint this machine
//! has for the model in question into a measurement instead of a skip.
//!
//! # A second finding, about the artifacts on this machine
//!
//! `~/.cache/pie/models/{qwen-3-0.6b,llama-3.2-1b-instruct}` look like the
//! obvious inputs and are not readable here: both begin `ZTEN0001`, while the
//! `ztensor` 2.1.1 this workspace resolves opens on `89 5a 54 32 0d 0a 1a 0a`
//! and answers `cannot detect the format`. They are v1 artifacts under a v2
//! reader -- the same thing `driver-vulkan` recorded, still true. The HF
//! snapshot cache is what the numbers above were measured against.
//!
//! `Qwen/Qwen3-1.7B` is cached and is deliberately NOT a fixture. Its role set
//! is qwen3-0.6B's exactly -- 28 layers, per-head qk-norm, tied -- so the 704
//! names would match and the file would look like it had measured a second
//! model while measuring one text twice. Its widths are worth knowing anyway
//! and are recorded here rather than asserted: 968_001_536 bytes bound, and
//! **87** weights over [`GUESS`], against qwen3-0.6B's 3 -- which is, to the
//! number, the finding the Vulkan file needed a whole second fixture to make.
//!
//! # Why every count here is pinned
//!
//! `tests/arena.rs` states the argument in full and this file follows it: a
//! sweep that iterated nothing passes exactly as loudly as one that iterated
//! everything and agreed, so each check asserts the SIZE of what it walked.
//! The cost is that these numbers move whenever `crates/model` changes a text
//! or `model-loader` changes a contract, and that is the right trade only
//! while updating them stays MECHANICAL -- every assertion below prints both
//! numbers, so the new one is the answer. If it stops being mechanical, or if
//! these start being updated without anybody reading the direction they moved,
//! the honest fix is to assert the coverage rather than the volume.

use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_compiler::trace::FireClass;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The block a sibling shell holds every non-`embed` weight under.
///
/// `driver-vulkan/tests/device.rs` stages real bytes for a whole plan and has
/// to pick a size, because `Arg::Weight` states none: four mebibytes for
/// everything, and a per-model number for `embed` alone -- 96 MiB for qwen3-
/// 0.6B, 192 MiB for qwen3-30B-A3B -- because the tied table is nineteen times
/// the guess and the head read 74 MiB past the end of it, silently, until a
/// numeric run found every logit at `-0`.
///
/// THIS crate stages nothing. `tests/arena.rs` answers all 14948 arena
/// operands with one `Placeholder(1 << 30)`, which is a size and not an
/// allocation, and `tests/device.rs` builds its buffers by hand for single
/// kernels. So no test here has ever been wrong about a weight's size, because
/// none of them has ever stated one -- and the widths below are the first
/// statement in this crate of how large a weight actually is.
///
/// Four mebibytes is the yardstick anyway, and deliberately not this crate's
/// 1 GiB placeholder: the interesting number is how many weights a fixed block
/// is too small for, and a stand-in chosen to be larger than anything answers
/// that question with a zero that means nothing. See [`Fixture::over`].
const GUESS: u64 = 4 << 20;

/// The stand-in `tests/arena.rs` answers every weight with.
///
/// Not a guess -- it is generous on purpose, and the only thing worth asking
/// of it is whether "generous" is still true of a real model. It is: gpt-oss's
/// `embed` is the largest weight either fixture binds and it is under a third
/// of this.
const PLACEHOLDER: u64 = 1 << 30;

// ---------------------------------------------------------------------------
// The texts, and the names they bind
// ---------------------------------------------------------------------------

/// `LlamaLikeMetalFacts::synthetic()` with the one line this backend answers
/// differently, copied from `tests/arena.rs` so both files ask the same
/// question of the same plans.
///
/// `synthetic()` is `driver-metal`'s answer sheet and its `add_bias: false` is
/// read off the ABSENCE of a `Source::OutWidth` arm in that driver's binder;
/// this backend's binder has one. What it changes is which LAUNCHES a text
/// carries, not which WEIGHTS it binds -- measured both ways on qwen2.5-1.5B,
/// the one text here whose checkpoint ships attention biases: 732 names either
/// way, and no name in one set that is not in the other. Stated anyway,
/// because a file that measured this crate's names under another crate's
/// binder would be measuring the wrong plan the moment that stops being true.
fn wgpu_facts() -> LlamaLikeMetalFacts {
    LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    }
}

/// The six texts `tests/arena.rs` lowers, and the backend facts each is
/// lowered under.
///
/// The same list, in the same order, for the same reason that file gives: a
/// table that had quietly specialised to one architecture leaves names over
/// exactly here. `phi3_mini` is absent from both, and that is a `model-compiler`
/// signature check that PANICS rather than returns -- see `tests/arena.rs`.
fn texts() -> Vec<(&'static str, LlamaLikeFacts, LlamaLikeMetalFacts)> {
    vec![
        ("qwen3_0_6b", LlamaLikeFacts::qwen3_0_6b(), wgpu_facts()),
        (
            "gpt_oss_20b",
            LlamaLikeFacts::gpt_oss_20b(),
            LlamaLikeMetalFacts::gpt_oss_20b(),
        ),
        (
            "qwen3_30b_a3b",
            LlamaLikeFacts::qwen3_30b_a3b(),
            wgpu_facts(),
        ),
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b(), wgpu_facts()),
        (
            "mistral_7b_v03",
            LlamaLikeFacts::mistral_7b_v03(),
            wgpu_facts(),
        ),
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), wgpu_facts()),
    ]
}

/// Every weight name one text's lowering states, sidecars included.
///
/// A `scale.` marker is left out: it is a constant riding the weight slot
/// rather than a tensor, so no loader publishes one and no binder looks one
/// up. None of these six states any -- measured, zero of 4664 -- and the
/// filter stays because the marker is `model-compiler`'s to reintroduce and a
/// driver that bound one would be asking a checkpoint for a scalar.
fn names_a_text_binds(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
    class: FireClass,
) -> BTreeSet<String> {
    let rows = match class {
        FireClass::Decode => 1,
        _ => 64,
    };
    let text = llama_like_metal(facts, metal, class);
    let low = lower(
        &text,
        &vec![Row::default(); rows],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    low.args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) => Some(n.clone()),
            _ => None,
        })
        .filter(|n| !n.starts_with("scale."))
        .collect()
}

/// One fixture's decode names, which is what a checkpoint is held against.
///
/// Decode and not prefill because they bind the same set -- asserted, over all
/// six texts, in [`every_name_the_six_texts_bind_is_one_this_table_can_spell`]
/// -- and decode is the plan a driver runs 99 times out of 100.
fn names_the_text_binds(fixture: &Fixture) -> BTreeSet<String> {
    let wanted = names_a_text_binds(&(fixture.facts)(), &(fixture.metal)(), FireClass::Decode);
    assert_eq!(
        wanted.len(),
        fixture.bound,
        "the {} text binds {} weights, not {}",
        fixture.id,
        wanted.len(),
        fixture.bound
    );
    wanted
}

// ---------------------------------------------------------------------------
// The checkpoints this file knows how to recognise
// ---------------------------------------------------------------------------

/// A checkpoint this file is written against, and the text it belongs to.
///
/// The snapshot names itself. `model.embed_tokens.weight`'s shape plus the
/// layer count is enough to tell these apart, and both are facts the ARTIFACT
/// states rather than ones a path claims. That matters because the failure
/// mode of guessing wrong is not an error -- it is this file reporting the
/// fixture's names as ones the checkpoint is missing, which reads exactly like
/// a loader defect.
struct Fixture {
    /// The catalog row, taken by id. See [`compiled_plan_for`] for why not
    /// always `catalog::identify`.
    id: &'static str,
    /// The repository the numbers below were measured against.
    repo: &'static str,
    /// The forward-facts fixture whose text states the names.
    facts: fn() -> LlamaLikeFacts,
    /// The backend facts that text is lowered under.
    metal: fn() -> LlamaLikeMetalFacts,
    /// `model.embed_tokens.weight`'s shape as PUBLISHED -- the release's own
    /// storage, so `[vocab, hidden]` for a bf16 export and the packed width
    /// for a `*-4bit` one.
    embed: &'static [i64],
    /// How many `model.layers.N.` indices the checkpoint carries. Half of the
    /// identification, and the half the Vulkan file does not have; see the
    /// module doc on why auto-discovery needs it.
    layers: usize,
    /// Whether the release is quantised, i.e. whether
    /// `Binding::MLX_IN_PLACE` compiles a plan without a runtime encode.
    /// Stated so that a checkpoint which changes side is a visible failure
    /// rather than a silent change of path.
    quantised: bool,
    /// How many weights the decode text binds, sidecars included.
    bound: usize,
    /// How many of those are ALREADY names the compiled plan publishes,
    /// before any translation.
    ///
    /// The Vulkan file asserts this is zero for every fixture, as the negative
    /// control for the table: without it, a table that had quietly stopped
    /// translating would still read as "still fine". It is zero for qwen3 and
    /// it is ONE for gpt-oss -- `lm_head.scales`, and the coincidence is
    /// legible rather than lucky. gpt-oss is untied, so the loader keeps the
    /// head under its own name `lm_head`, and `.scales` is the one sidecar
    /// suffix both sides spell the same way. Pinned per fixture rather than
    /// asserted zero, because "zero" was one model's answer.
    raw_agree: usize,
    /// Their total size in bytes, as the load plan states them.
    ///
    /// Pinned rather than printed, because the point of these numbers is that
    /// they can CHANGE: a contract that started fusing projections, or a
    /// target that changed a block layout, moves them, and a test that only
    /// printed would let that happen quietly.
    total: u64,
    /// `embed`'s own size, which is `vocab * hidden / 2` for every affine-U4
    /// plan here and is asserted as that arithmetic as well as as a constant.
    embed_bytes: u64,
    /// `layer.0.q_proj`'s size, likewise `q_heads * head_dim * hidden / 2`.
    /// Two derivations rather than one, because a single one that happened to
    /// hold would not tell a stated width from a coincidence.
    q_proj_bytes: u64,
    /// How many weights are larger than [`GUESS`], the block
    /// `driver-vulkan/tests/device.rs` holds every non-`embed` name under.
    ///
    /// Three for qwen3-0.6B -- `embed` and its two sidecars -- and ONE HUNDRED
    /// AND NINETY-EIGHT for gpt-oss-20B. That gap is the finding: "4 MiB
    /// covers everything except the tied table" was never a rule, it was one
    /// model's arithmetic, and the Vulkan file made the same discovery going
    /// from qwen3 to qwen2.5 (3 to 87). Kept here because this backend has the
    /// same silence: an out-of-bounds storage read returns zero.
    over: usize,
}

/// Every checkpoint shape this file is written against.
///
/// Two, and the second is not a duplicate of the first: gpt-oss is a different
/// generation with a different role set -- routed experts, attention sinks, an
/// untied head, additive biases everywhere -- a different quantisation
/// (MXFP4 as published, affine-U4 as bound) and 33x the weight. A table that
/// had quietly specialised to qwen3 leaves names over here, and it does.
///
/// Both are on this machine. Neither is a `*-4bit` conversion, which is what
/// the Vulkan file's two fixtures are, so nothing below shares an artifact
/// with the numbers it re-measures.
const FIXTURES: &[Fixture] = &[
    Fixture {
        id: "qwen3-0.6b",
        repo: "Qwen/Qwen3-0.6B",
        facts: LlamaLikeFacts::qwen3_0_6b,
        metal: wgpu_facts,
        embed: &[151_936, 1024],
        layers: 28,
        quantised: false,
        bound: 704,
        raw_agree: 0,
        total: 335_372_288,
        embed_bytes: 77_791_232,
        q_proj_bytes: 1_048_576,
        over: 3,
    },
    Fixture {
        id: "gpt-oss-20b",
        repo: "openai/gpt-oss-20b",
        facts: LlamaLikeFacts::gpt_oss_20b,
        metal: LlamaLikeMetalFacts::gpt_oss_20b,
        embed: &[201_088, 2880],
        layers: 24,
        quantised: true,
        bound: 727,
        raw_agree: 1,
        // MEASURED WITH THE SIX ROLES THIS CRATE'S TABLE IS MISSING, spelled
        // by hand the way `driver-vulkan`'s table spells them, because the
        // shipped table cannot name 144 of these and a total over the 583 it
        // CAN name would be a number that changes when the defect is fixed.
        // What is pinned is the model's size, which is a fact about the
        // artifact: 11_177_235_072 bytes, of which the shipped table reaches
        // 11_163_718_272.
        total: 11_177_235_072,
        embed_bytes: 289_566_720,
        q_proj_bytes: 5_898_240,
        over: 198,
    },
];

/// The snapshot directories to measure, and whether a person named them.
///
/// `PIE_CHECKPOINT` first, colon-separated: one checkpoint proves the
/// conversion works, a second proves it is a CONVERSION rather than a table
/// that happens to spell one model's names, and that is a different claim.
///
/// With it unset this walks the local HuggingFace cache instead of returning
/// nothing. The flag comes back with the list because the two sources deserve
/// different manners: a directory a person NAMED and which turns out to be
/// unreadable, or to be a model no fixture states, is worth a printed line,
/// and thirty cache entries that are datasets, kernels and models this file
/// has never heard of are not.
fn snapshots() -> (Vec<String>, bool) {
    if let Ok(v) = std::env::var("PIE_CHECKPOINT")
        && !v.trim().is_empty()
    {
        return (
            v.split(':')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            true,
        );
    }
    (hugging_face_cache(), false)
}

/// Every `models--*/snapshots/*` directory under the local HuggingFace cache.
///
/// `HF_HOME`/`HF_HUB_CACHE` are honoured because a machine that moved its cache
/// did so to a disk with room on it, and a file that only knew `~/.cache` would
/// report "unmeasured" on exactly the machine with the most artifacts.
/// Datasets and kernel repositories are skipped by prefix: they are neither,
/// and parsing them would print reasons about things nobody asked about.
fn hugging_face_cache() -> Vec<String> {
    let hub = match (std::env::var("HF_HUB_CACHE"), std::env::var("HF_HOME")) {
        (Ok(dir), _) if !dir.is_empty() => std::path::PathBuf::from(dir),
        (_, Ok(home)) if !home.is_empty() => std::path::PathBuf::from(home).join("hub"),
        _ => match std::env::var("HOME") {
            Ok(home) => std::path::PathBuf::from(home).join(".cache/huggingface/hub"),
            Err(_) => return Vec::new(),
        },
    };
    let Ok(repos) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for repo in repos.flatten() {
        if !repo.file_name().to_string_lossy().starts_with("models--") {
            continue;
        }
        let Ok(revisions) = std::fs::read_dir(repo.path().join("snapshots")) else {
            continue;
        };
        for revision in revisions.flatten() {
            out.push(revision.path().to_string_lossy().into_owned());
        }
    }
    // Sorted, so that a machine with two revisions of one repository measures
    // the same one on every run. Which of two revisions is measured is
    // arbitrary; that it is arbitrary AND unstable would make a pinned byte
    // count flap.
    out.sort();
    out
}

/// The fixture whose embed shape and layer count this checkpoint's are, or
/// `None` with the reason printed when a person asked for it by name.
fn fixture_of(
    dir: &str,
    meta: &model_loader::checkpoint::CheckpointMetadata,
    named: bool,
) -> Option<&'static Fixture> {
    let shape = meta
        .tensors
        .iter()
        .find(|t| t.name == "model.embed_tokens.weight")
        .map(|t| t.shape.clone())
        .unwrap_or_default();
    let layers = layer_count(meta);
    let found = FIXTURES
        .iter()
        .find(|f| shape == f.embed && layers == f.layers);
    if found.is_none() && named {
        eprintln!(
            "{dir} embeds {shape:?} over {layers} layers, which is no fixture this file states"
        );
    }
    found
}

/// How many `model.layers.N.` indices a checkpoint carries.
fn layer_count(meta: &model_loader::checkpoint::CheckpointMetadata) -> usize {
    meta.tensors
        .iter()
        .filter_map(|t| t.name.strip_prefix("model.layers."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|n| n.parse::<usize>().ok())
        .collect::<BTreeSet<_>>()
        .len()
}

/// Every checkpoint that named a fixture, paired with it, at most one per
/// fixture.
///
/// The de-duplication is where the two sources part again. Two snapshots of
/// the same model would run a check twice and prove it once, so a person who
/// NAMES the same model twice is told; a cache that happens to hold two
/// revisions of one repository is not, because that is an ordinary thing for a
/// cache to hold and failing there would make this file's discovery its own
/// worst enemy.
fn measurable(named: bool, dirs: &[String]) -> Vec<(String, &'static Fixture)> {
    let mut out: Vec<(String, &'static Fixture)> = Vec::new();
    for dir in dirs {
        let meta = match model_loader::checkpoint::read::parse_checkpoint_metadata(
            std::path::Path::new(dir),
        ) {
            Ok(meta) => meta,
            Err(e) => {
                if named {
                    eprintln!("{dir} is not readable as a checkpoint ({e}), so it is unmeasured");
                }
                continue;
            }
        };
        let Some(fixture) = fixture_of(dir, &meta, named) else {
            continue;
        };
        if let Some((first, _)) = out.iter().find(|(_, f)| f.id == fixture.id) {
            assert!(
                !named,
                "`{}` and `{dir}` are both {} -- two snapshots of one model run a check \
                 twice and prove it once",
                first, fixture.id
            );
            eprintln!(
                "{dir} is a second {} snapshot; measuring {first}",
                fixture.id
            );
            continue;
        }
        out.push((dir.clone(), fixture));
    }
    out
}

/// Every fixture that could be measured at all, or a printed reason.
///
/// One entry point for all five checkpoint-driven tests, so the skip
/// conditions cannot drift apart between them, and one place where "nothing
/// was measured" is said out loud. A test that passed silently on a machine
/// with no artifact would be reporting the absence of the checkpoint as the
/// presence of agreement.
fn measured(what: &str) -> Vec<(String, &'static Fixture)> {
    let (dirs, named) = snapshots();
    let found = measurable(named, &dirs);
    if found.is_empty() {
        eprintln!(
            "no checkpoint this file states was found in {} ({} candidates), so {what} COULD \
             NOT BE MEASURED",
            if named {
                "PIE_CHECKPOINT"
            } else {
                "the HuggingFace cache"
            },
            dirs.len()
        );
    } else {
        for (dir, f) in &found {
            eprintln!("measuring {} ({}) at {dir}", f.id, f.repo);
        }
    }
    found
}

// ---------------------------------------------------------------------------
// 1. The raw disagreement
// ---------------------------------------------------------------------------

/// A plan's weight names and a checkpoint's tensor names overlap completely or
/// not at all -- never partly.
#[test]
fn the_names_a_plan_states_are_names_a_checkpoint_holds() {
    for (dir, fixture) in measured("the raw name agreement") {
        let meta =
            model_loader::checkpoint::read::parse_checkpoint_metadata(std::path::Path::new(&dir))
                .expect("it parsed once already");
        let held: BTreeSet<&str> = meta.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            held.len() > 100,
            "{} holds only {} tensors, so this is not a whole checkpoint",
            fixture.id,
            held.len()
        );

        let wanted = names_the_text_binds(fixture);

        // The interesting number is not "how many are missing" but "how many
        // overlap", because the two safe answers are ALL and NONE and the
        // dangerous answer is in between. None means a loader must convert;
        // all means it can look up. A partial overlap means a loader could
        // load the names that happen to agree, leave the rest at whatever the
        // arena held, and produce logits -- wrong ones, with nothing refused.
        let shared: Vec<&String> = wanted
            .iter()
            .filter(|n| held.contains(n.as_str()))
            .collect();
        eprintln!(
            "{}: {} of {} plan names are raw checkpoint tensors",
            fixture.id,
            shared.len(),
            wanted.len()
        );
        assert!(
            shared.is_empty(),
            "{} of {} plan names are also raw checkpoint tensors ({:?}). Neither none nor \
             all, which is the one answer a loader cannot act on: it would load the \
             agreeing names and silently leave the rest unwritten.",
            shared.len(),
            wanted.len(),
            &shared[..shared.len().min(8)]
        );

        // MEASURED: zero of 704 on `Qwen/Qwen3-0.6B` and zero of 727 on
        // `openai/gpt-oss-20b`. The plan says `layer.0.down` and both
        // checkpoints say `model.layers.0.mlp.down_proj.weight`; the plan also
        // wants `embed.scales` and `embed.zeros`, which neither release holds
        // under any spelling -- they are outputs of quantizing, not tensors
        // anyone published.
        //
        // So a weight loader for this crate is not a lookup and cannot be. It
        // is the conversion `model-loader`'s `plan::compile` already exists to
        // describe, and it belongs above a driver: a driver that knew how to
        // turn `model.layers.0.mlp.down_proj.weight` into `layer.0.down` plus
        // scales and zeros would be a driver with opinions about checkpoint
        // conventions.
        eprintln!(
            "  none of {} names are checkpoint tensors; loading is a conversion, not a lookup",
            wanted.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The table's own claim, which needs no artifact
// ---------------------------------------------------------------------------

/// `Naming` has a spelling for every name any text this crate lowers binds.
///
/// This is `src/names.rs`'s headline claim -- "the nineteen roles the six texts
/// in `tests/arena.rs` bind, and no others. Measured, not guessed" -- asked of
/// the texts rather than of the comment. It needs no checkpoint, no adapter and
/// no download, which is the point: the claim is about a table and six texts,
/// both of which are in this build, so the build that changes a text is the
/// build that finds out.
///
/// An empty answer from `spellings` is NOT treated as a match here for the same
/// reason the table's own doc gives: it means the name is outside the text's
/// shape, and a caller that read it as "no spelling to try" would turn a role
/// the table has never heard of into a silent miss at load.
///
/// # It fails, and that is the finding
///
/// 228 of 4664, over two of the six texts. qwen2.5-1.5B: `{q,k,v}_bias`, three
/// a layer over 28. gpt-oss-20B: the same three over 24 layers, plus
/// `expert_{gate,up,down}.bias`, the additive term a routed expert bank carries
/// beside its codec's planes. `driver-vulkan`'s copy of this table spells all
/// six -- three roles with the `.bias` in the path and an empty
/// `weight_suffix`, plus a `bias_suffix` and a `Sidecar::Bias` for the routed
/// banks -- and this crate's copy was left behind when the texts gained them.
/// Not adjusted to pass; see the module doc.
#[test]
fn every_name_the_six_texts_bind_is_one_this_table_can_spell() {
    let naming = driver_wgpu::names::Naming::mlx();
    let mut total = 0usize;
    let mut orphans: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (text, facts, metal) in texts() {
        let decode = names_a_text_binds(&facts, &metal, FireClass::Decode);
        let prefill = names_a_text_binds(&facts, &metal, FireClass::Prefill);
        // The two classes bind the same weights -- the row count changes every
        // rectangle and no operand's NAME -- and this is where that is checked
        // rather than assumed, so that measuring decode alone below is a
        // measured shortcut and not a hopeful one.
        assert_eq!(
            decode, prefill,
            "{text} binds different weights at decode and prefill"
        );
        total += decode.len();
        let missing: Vec<String> = decode
            .iter()
            .filter(|n| naming.spellings(n).is_empty())
            .cloned()
            .collect();
        eprintln!(
            "{text}: {} weights, {} outside the table's shape",
            decode.len(),
            missing.len()
        );
        if !missing.is_empty() {
            orphans.insert(text.to_string(), missing);
        }
    }
    // The floor `tests/arena.rs` argues for: a sweep that walked nothing
    // passes exactly as loudly as one that walked everything and agreed.
    assert_eq!(
        total, 4664,
        "the six texts bind {total} weights, not 4664 -- a text changed, so read which \
         direction this moved before repinning it"
    );

    let orphaned: usize = orphans.values().map(Vec::len).sum();
    assert!(
        orphans.is_empty(),
        "{orphaned} of {total} names these texts bind are outside `Naming`'s shape entirely, \
         so `spellings` answers with nothing and a loader would leave them at whatever the \
         arena held. By text: {:?}. `driver-vulkan/src/names.rs` spells all of them -- it \
         carries `q_bias`/`k_bias`/`v_bias` as roles whose path ends in `.bias`, and a \
         `bias_suffix` with its own `Sidecar::Bias` for the routed expert banks -- and \
         `crates/driver-wgpu/src/names.rs` still folds `.bias` into `zero_point_suffix` and \
         has no bias roles at all. This is drift between two copies of one table, not a \
         checkpoint that lacks something: the loader publishes every one of these (measured \
         on `openai/gpt-oss-20b`: `layers.0.self_attn.q_proj.bias` and \
         `layers.0.mlp.experts.down_proj.bias` are both in the compiled plan). First few \
         per text: {:?}",
        orphans
            .iter()
            .map(|(t, n)| (t.as_str(), n.len()))
            .collect::<Vec<_>>(),
        orphans
            .iter()
            .map(|(t, n)| (t.as_str(), &n[..n.len().min(3)]))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// 3. The conversion, run
// ---------------------------------------------------------------------------

/// The load plan `model-loader` would compile for one snapshot.
///
/// Shared by the three checkpoint tests, because compiling it is the fiddly
/// half and a second copy would be a second place for the conditions to drift
/// apart.
fn compiled_plan_for(dir: &str, fixture: &Fixture) -> model_loader::plan::LoadPlan {
    let path = std::path::Path::new(dir);
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(path)
        .expect("it parsed once already");

    // THE ROW BY NAME, NOT BY `catalog::identify`, and the reason is a finding
    // rather than a shortcut: `identify` refuses a stock `Qwen/Qwen3-0.6B`
    // with `unexpected lm_head`, because that export ships a tied head as a
    // real tensor. See the module doc, and
    // `the_catalog_places_the_checkpoints_it_can` for the half that IS
    // exercised.
    let row = model::catalog::find(fixture.id)
        .unwrap_or_else(|| panic!("this build has no `{}` row", fixture.id));

    let config =
        match model_loader::checkpoint::read::read_meta(&meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
            _ => std::fs::read_to_string(path.join("config.json"))
                .unwrap_or_else(|e| panic!("{dir}/config.json: {e}")),
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .expect("the config states an encoding");

    // `BackendKind::Vulkan`, and this backend is not Vulkan -- it is whichever
    // of Vulkan, Metal or DX12 the adapter answered on, which is a thing no
    // load plan can be compiled against because there is no `Wgpu` arm and a
    // plan is compiled before an adapter is asked. What a target changes is
    // alignment, tile budget and which transforms are claimed; it does not
    // change what the tensors are CALLED. The two arms this shell can actually
    // sit on are held against each other below, so "it does not matter which"
    // is measured here rather than assumed by an alias.
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );
    assert_eq!(
        target.tile_map_mask,
        model_loader::plan::StorageTarget::for_backend(
            model_loader::types::BackendKind::Metal,
            0,
            1
        )
        .tile_map_mask,
        "the Vulkan target no longer admits what Metal's does, so a wgpu shell on a Metal \
         adapter and the same shell on a Vulkan one would be handed different plans -- and \
         this crate picks its backend at runtime"
    );

    // THE DOCUMENTED PATH FIRST. `Binding::MLX_IN_PLACE` is what a driver boot
    // asks for and what `driver-vulkan`'s version of this file uses.
    let documented = model::boot::compile_load_plan_for(
        path,
        &meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    );
    let refusal = match documented {
        Ok((plan, _)) => {
            assert!(
                fixture.quantised,
                "{} compiled without a runtime encode, so this release is quantised and the \
                 fixture says it is not",
                fixture.id
            );
            eprintln!(
                "{}: plan compiled through `Binding::MLX_IN_PLACE`, {} tensors",
                fixture.id,
                plan.tensors.len()
            );
            return plan;
        }
        Err(e) => e.to_string(),
    };
    assert!(
        !fixture.quantised,
        "{} was refused ({refusal}), and the fixture says this release is quantised",
        fixture.id
    );
    // AND THE OTHER WAY THE REFUSAL ITSELF NAMES. An unquantised release has
    // no `.scales`, every projection binds through the affine-U4 path, and
    // `crates/model/src/shared/mlx.rs` states the two remedies in the same
    // sentence: pre-quantise the repo, or encode at load. `RuntimeQuant::Int4`
    // is the second, and taking it is what turns the one `Qwen/Qwen3-0.6B` on
    // this machine into a measurement instead of a skip. Everything below is
    // about NAMES and WIDTHS, and both are the same on either road -- the
    // module doc records that the byte totals came out identical to the
    // Vulkan file's, measured from the pre-quantised repo.
    assert!(
        refusal.contains("needs quantized weights"),
        "{} was refused for a reason this file does not know how to answer: {refusal}",
        fixture.id
    );
    eprintln!(
        "{}: `MLX_IN_PLACE` refused an unquantised release, so the plan is compiled with \
         `RuntimeQuant::Int4` -- the remedy the refusal names",
        fixture.id
    );
    let policy = model::shared::policy::Policy {
        projections: model::shared::policy::Projections::InPlace,
        naming: model::shared::policy::Naming::Mlx,
        runtime_quant: model::shared::policy::RuntimeQuant::Int4,
        moe_request: model::shared::policy::Mxfp4MoeRequest::Auto,
        component: model::shared::policy::Component::Full,
        stream_routed_experts: false,
        knobs: model::shared::policy::FamilyKnobs::default(),
    };
    let (contract, _) =
        model::contract::author_with_policy(row, &encoding, &meta, &target, &policy)
            .unwrap_or_else(|e| panic!("the loader would not author `{}`: {e}", fixture.id));
    let plan = model_loader::plan::compile(&meta, &contract, target.clone()).unwrap_or_else(|e| {
        panic!(
            "the loader would not compile a plan for `{}`: {e}",
            fixture.id
        )
    });
    // `compile_load_plan_for` runs this and the policy path does not, so it is
    // run here: a snapshot that moved under a plan compiled against it is a
    // refusal, and dropping it would mean the two roads through this function
    // checked different things.
    model_loader::checkpoint::read::verify_declared_files(&plan, path).unwrap_or_else(|e| {
        panic!(
            "the plan for `{}` names a file that is not there: {e}",
            fixture.id
        )
    });
    eprintln!(
        "{}: plan compiled, {} tensors",
        fixture.id,
        plan.tensors.len()
    );
    plan
}

/// The loader's own plan states the names this driver binds -- through
/// `Naming`, and not otherwise.
///
/// The first test measured the raw checkpoint and found zero of 704, which
/// settled that loading is a CONVERSION. This is the other half of that
/// sentence: that the conversion already exists and produces what this driver
/// asks for. Same snapshot, through `catalog::identify`'s row ->
/// `contract::author` -> `plan::compile`, comparing the tensor names the
/// compiled plan publishes against the weight names the lowering states.
///
/// Nothing here executes the plan. Executing it needs a `model-loader`
/// executor this crate has not written, and a target changes alignment and
/// tile budget rather than what the tensors are CALLED -- so the names are
/// measurable now and the bytes are not, and this measures the half that is.
///
/// # It passes for qwen3 and fails for gpt-oss
///
/// 704 of 704 raw disagreements, nothing left over after translation. And 726
/// of 727 for gpt-oss, of which 144 are names `Naming` cannot decompose at
/// all. See the module doc: six roles, all of them in `driver-vulkan`'s copy of
/// this table and none in this one.
#[test]
fn the_loader_states_the_names_this_driver_binds() {
    for (dir, fixture) in measured("the loader's name agreement") {
        let plan = compiled_plan_for(&dir, fixture);
        names_agree(&plan, fixture);
    }
}

/// One snapshot's half of the test above.
///
/// Split out so the loop stays a loop and the reasoning stays in one place:
/// every claim below is about a plan and a text, not about which model they
/// belong to.
fn names_agree(plan: &model_loader::plan::LoadPlan, fixture: &Fixture) {
    let published: BTreeSet<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        published.len() > 100,
        "the plan published {} tensors, so it is not a whole model",
        published.len()
    );

    let wanted = names_the_text_binds(fixture);

    // THE NEGATIVE CONTROL FOR THE TABLE ITSELF. Held directly, all 704 of
    // qwen3's disagree -- the two sides share no convention, which is the
    // measurement `src/names.rs` exists because of. Without this, a table that
    // had quietly stopped translating would still read as "still fine".
    //
    // Pinned per fixture rather than asserted zero, because zero was one
    // model's answer: gpt-oss is untied, so its head keeps the name `lm_head`
    // on both sides and `lm_head.scales` agrees by itself. One name out of
    // 727 is a coincidence with a reason, and it is stated so that a SECOND
    // one has to be explained rather than absorbed.
    let agreeing: Vec<&String> = wanted
        .iter()
        .filter(|n| published.contains(n.as_str()))
        .collect();
    eprintln!(
        "{}: {} of {} names agree with the plan before translation",
        fixture.id,
        agreeing.len(),
        wanted.len()
    );
    assert_eq!(
        agreeing.len(),
        fixture.raw_agree,
        "{} of {} names the {} text binds are already names the loader publishes, and this \
         file states {}. The names: {:?}",
        agreeing.len(),
        wanted.len(),
        fixture.id,
        fixture.raw_agree,
        &agreeing[..agreeing.len().min(8)]
    );

    // A name the table cannot decompose is reported rather than skipped, or
    // the filter below would hide it: an empty answer means the name is
    // outside the table's shape, which is drift and not a spelling it has yet
    // to learn.
    let undecomposed: Vec<&String> = wanted
        .iter()
        .filter(|n| driver_wgpu::names::Naming::mlx().spellings(n).is_empty())
        .collect();

    // THROUGH THE TRANSLATION, which is the whole point. What is under test is
    // whether the table is TOTAL: a role it does not carry, or a spelling this
    // loader does not publish, shows up as a name left over.
    let naming = driver_wgpu::names::Naming::mlx();
    let missing: Vec<&String> = wanted
        .iter()
        .filter(|n| {
            let spellings = naming.spellings(n);
            !spellings.is_empty() && !spellings.iter().any(|s| published.contains(s.as_str()))
        })
        .collect();
    eprintln!(
        "{}: after `Naming::mlx()`, {} names resolve, {} resolve to nothing published, {} \
         cannot be decomposed at all",
        fixture.id,
        wanted.len() - missing.len() - undecomposed.len(),
        missing.len(),
        undecomposed.len()
    );

    // The same all-or-nothing shape as the first test, and for the same
    // reason: a partial answer is the one a driver cannot act on, because it
    // would bind the agreeing names and leave the rest at whatever the arena
    // held.
    assert!(
        missing.is_empty(),
        "{} of {} names the {} text binds resolve to no tensor the loader publishes; the \
         first few are {:?}, which this table spells {:?}. Loading is a conversion, and \
         this is the conversion not being the one this driver needs.",
        missing.len(),
        wanted.len(),
        fixture.id,
        &missing[..missing.len().min(8)],
        missing
            .first()
            .map(|n| naming.spellings(n))
            .unwrap_or_default()
    );
    assert!(
        undecomposed.is_empty(),
        "{} of {} names the {} text binds are not in `Naming`'s shape at all: {:?}. Every \
         one of them IS a tensor this plan publishes -- {:?} -- so the loader is not \
         missing them, this crate's table cannot say their names. \
         `driver-vulkan/src/names.rs` can. See the module doc.",
        undecomposed.len(),
        wanted.len(),
        fixture.id,
        &undecomposed[..undecomposed.len().min(8)],
        published
            .iter()
            .filter(|n| n.ends_with(".bias"))
            .take(4)
            .collect::<Vec<_>>()
    );
}

/// The `.bias` this table offers as a spelling of `.zeros` is a different
/// tensor from the one it means.
///
/// `zero_point_suffix` is `[".biases", ".bias"]` here and the comment beside it
/// calls them "the same role one character apart". A checkpoint that publishes
/// both says otherwise, and gpt-oss publishes both: the zero point is one value
/// per GROUP of 64 and the additive bias is one per output ROW, so they differ
/// by a factor of `group_size` and are read at different bindings.
///
/// Nothing here is broken today -- `.biases` is tried first and wins. What this
/// pins is the margin, and the margin is thin: of gpt-oss's 122 resolved zero
/// points, **120 sit beside an additive `.bias` this table also offers as a
/// spelling of `.zeros`**, and only the order of two strings in a slice
/// decides which one a loader would bind. qwen3's 197 have no `.bias` beside
/// them at all, which is why one fixture could not have found this.
///
/// A checkpoint that shipped `.bias` and no `.biases`, or an edit that
/// reordered those two strings, binds a 45x undersized buffer as a zero-point
/// plane -- and on this backend an out-of-bounds storage read returns zero
/// rather than faulting. `driver-vulkan` removed `.bias` from that list for
/// exactly this reason and gave it a row of its own.
#[test]
fn the_zero_point_this_table_finds_is_a_zero_point() {
    let naming = driver_wgpu::names::Naming::mlx();
    let mut checked = 0usize;
    for (dir, fixture) in measured("the zero point's identity") {
        let plan = compiled_plan_for(&dir, fixture);
        let by_name: HashMap<&str, &model_loader::types::TensorDecl> =
            plan.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        let mut here = 0usize;
        let mut both = 0usize;
        for traced in names_the_text_binds(fixture) {
            let Some(base) = traced.strip_suffix(".zeros") else {
                continue;
            };
            let spellings = naming.spellings(&traced);
            let Some(resolved) = spellings.iter().find(|s| by_name.contains_key(s.as_str())) else {
                continue;
            };
            assert!(
                resolved.ends_with(".biases"),
                "`{traced}` resolved to `{resolved}`, which is the additive bias and not the \
                 codec's zero point"
            );
            checked += 1;
            here += 1;
            // And where the additive bias exists too, the two are measurably
            // different tensors rather than two names for one.
            let additive = format!("{}.bias", &resolved[..resolved.len() - ".biases".len()]);
            if let (Some(zero), Some(add)) = (
                by_name.get(resolved.as_str()),
                by_name.get(additive.as_str()),
            ) {
                assert_ne!(
                    zero.shape, add.shape,
                    "`{resolved}` and `{additive}` have the same shape, so this checkpoint \
                     no longer distinguishes the zero point from the additive bias and the \
                     hazard this test pins has changed shape -- re-measure before deleting it"
                );
                both += 1;
                if both > 1 {
                    continue;
                }
                eprintln!(
                    "  {} vs {}: {:?} against {:?}, and `{}` offers both as spellings of \
                     `.zeros`",
                    resolved,
                    additive,
                    zero.shape,
                    add.shape,
                    base.rsplit('.').next().unwrap_or(base)
                );
                // One report per fixture is the finding; 84 of them is noise.
                // The LOOP still runs, because the assertion above is the
                // check and stopping at the first pair would leave 83 of them
                // unmade.
            }
        }
        eprintln!(
            "{}: {here} zero points resolved, {both} of them beside an additive `.bias` this \
             table also offers as a spelling of `.zeros`",
            fixture.id
        );
    }
    if checked == 0 {
        eprintln!("no `.zeros` name resolved, so the zero point's identity is unmeasured");
    }
}

/// The catalog places the checkpoints it can, and says why for the ones it
/// cannot.
///
/// `compiled_plan_for` takes its row by id, because a stock `Qwen/Qwen3-0.6B`
/// is refused. That refusal is a fact about `crates/model`'s manifest and is
/// left where it was found -- but taking the row by id on every path would
/// leave `catalog::identify` untouched by this file, and then "the loader can
/// place a real snapshot" would be a claim nobody here had run. gpt-oss
/// identifies, so it is run.
#[test]
fn the_catalog_places_the_checkpoints_it_can() {
    let mut placed = 0usize;
    for (dir, fixture) in measured("the catalog's identification") {
        let meta =
            model_loader::checkpoint::read::parse_checkpoint_metadata(std::path::Path::new(&dir))
                .expect("it parsed once already");
        match model::catalog::identify(&meta, &model::catalog::Override::default()) {
            Ok(row) => {
                assert_eq!(
                    row.id(),
                    fixture.id,
                    "the catalog placed {dir} as `{}`, and this file measures it as `{}`",
                    row.id(),
                    fixture.id
                );
                placed += 1;
                eprintln!("{}: identified", fixture.id);
            }
            Err(e) => {
                // MEASURED: `qwen3-0.6b: unexpected lm_head` for the stock
                // release, whose `config.json` says `tie_word_embeddings:
                // true` and whose safetensors publishes the head anyway. The
                // reason is asserted, not just the refusal, so that this
                // silently becoming a DIFFERENT refusal is a failure.
                let reason = e.to_string();
                assert!(
                    reason.contains(fixture.id),
                    "{dir} was refused without naming `{}`: {reason}",
                    fixture.id
                );
                eprintln!(
                    "{}: not identified -- {}",
                    fixture.id,
                    reason.lines().nth(1).unwrap_or(&reason).trim()
                );
            }
        }
    }
    if placed == 0 {
        eprintln!("no checkpoint was identified, so `catalog::identify` is unmeasured here");
    }
}

// ---------------------------------------------------------------------------
// 4. The widths, which a plan does not carry and a loader does
// ---------------------------------------------------------------------------

/// How large every weight this driver binds actually is.
///
/// # The number `lib.rs` says does not exist
///
/// "A plan does not say how large a tensor is" is true of a PLAN, and this
/// crate has never had to find out: `tests/arena.rs` answers every one of its
/// 14948 arena operands with one `Placeholder(1 << 30)`, a size with no
/// allocation behind it, which works because nothing there dispatches. The
/// moment something does, the size stops being free -- and on this backend an
/// undersized weight does not fail, an out-of-bounds storage read returns
/// zero. `driver-vulkan/tests/device.rs` paid for that discovery once already:
/// `embed` at four mebibytes made every logit `-0`.
///
/// A load plan has the widths, because `TensorDecl` carries a shape and an
/// encoding and `encoding_nbytes` turns those into bytes. So this asks it, and
/// the answer is what a real load would have to allocate.
///
/// The widths are checked for being PLAUSIBLE as well as present: `embed` is
/// `vocab * hidden / 2` and `layer.0.q_proj` is `q_heads * head_dim * hidden /
/// 2`, both derived from the same `LlamaLikeFacts` the text was lowered from,
/// both at four bits. A stated width that is merely non-zero would pass a plan
/// that had declared everything as one row.
#[test]
fn the_loader_states_how_large_every_weight_this_driver_binds_is() {
    for (dir, fixture) in measured("the widths") {
        let plan = compiled_plan_for(&dir, fixture);
        widths_are_stated(&plan, fixture);
    }
}

/// One snapshot's half of the test above.
fn widths_are_stated(plan: &model_loader::plan::LoadPlan, fixture: &Fixture) {
    let naming = driver_wgpu::names::Naming::mlx();
    let by_name: HashMap<&str, &model_loader::types::TensorDecl> =
        plan.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut total: u64 = 0;
    let mut widest: (String, u64) = (String::new(), 0);
    let mut over = 0usize;
    let mut measured = 0usize;
    let mut unresolved: Vec<String> = Vec::new();
    let mut bytes_of: BTreeMap<String, u64> = BTreeMap::new();
    for traced in names_the_text_binds(fixture) {
        let spellings = naming.spellings(&traced);
        let Some(decl) = spellings.iter().find_map(|s| by_name.get(s.as_str())) else {
            unresolved.push(traced);
            continue;
        };
        let bytes = model_loader::types::encoding_nbytes(&decl.shape, &decl.encoding)
            .unwrap_or_else(|| panic!("`{}` states no width", decl.name));
        assert!(bytes > 0, "`{}` is zero bytes", decl.name);
        total += bytes;
        measured += 1;
        if bytes > widest.1 {
            widest = (traced.clone(), bytes);
        }
        if bytes > GUESS {
            over += 1;
        }
        bytes_of.insert(traced, bytes);
    }
    eprintln!(
        "{}: {measured} of {} weights sized, {total} bytes, widest {} at {}, {over} over the \
         {} MiB block",
        fixture.id,
        fixture.bound,
        widest.0,
        widest.1,
        GUESS >> 20
    );

    // THE WIDTHS ARE THE MODEL'S OWN ARITHMETIC, derived from the facts the
    // text was lowered from rather than from a constant this file could have
    // copied out of a passing run. Both are affine-U4, so a row of `n` values
    // is `n / 2` bytes.
    let facts = (fixture.facts)();
    let embed = bytes_of
        .get("embed")
        .copied()
        .expect("every fixture binds an embedding");
    assert_eq!(
        embed,
        u64::from(facts.vocab) * u64::from(facts.hidden) / 2,
        "`embed` is {embed} bytes and the text's own geometry says {} x {} at four bits",
        facts.vocab,
        facts.hidden
    );
    let q_proj = bytes_of
        .get("layer.0.q_proj")
        .copied()
        .expect("every fixture binds a query projection");
    assert_eq!(
        q_proj,
        u64::from(facts.q_heads) * u64::from(facts.head_dim) * u64::from(facts.hidden) / 2,
        "`layer.0.q_proj` is {q_proj} bytes and the geometry says {} heads x {} x {}",
        facts.q_heads,
        facts.head_dim,
        facts.hidden
    );
    assert_eq!(embed, fixture.embed_bytes, "`embed` in bytes");
    assert_eq!(q_proj, fixture.q_proj_bytes, "`layer.0.q_proj` in bytes");
    assert_eq!(widest.0, "embed", "the widest weight");
    // And the one thing this crate's own stand-in can be wrong about. It is
    // not, by a factor of three -- but "generous" is a claim about a model, and
    // the models keep getting larger.
    assert!(
        widest.1 < PLACEHOLDER,
        "`{}` is {} bytes, which is larger than the `Placeholder(1 << 30)` tests/arena.rs \
         answers every weight with -- that stand-in has stopped being generous",
        widest.0,
        widest.1
    );

    // AND THE FINDING NOTHING IN THIS CRATE COULD HAVE MADE FOR ITSELF. Three
    // of qwen3-0.6B's 704 are larger than the four-mebibyte block a sibling
    // shell holds every non-`embed` name under -- `embed` and its two sidecars
    // -- so that block is safe for the other 701, provably rather than
    // hopefully. For gpt-oss it is ONE HUNDRED AND NINETY-EIGHT.
    //
    // So "4 MiB is enough for everything except the tied table" was never a
    // rule; it is one model's arithmetic. Recorded here because the failure it
    // guards is silent on this backend too.
    assert_eq!(
        over,
        fixture.over,
        "weights larger than the {} MiB block driver-vulkan/tests/device.rs holds",
        GUESS >> 20
    );
    for sidecar in ["embed", "embed.scales", "embed.zeros"] {
        let decl = naming
            .spellings(sidecar)
            .iter()
            .find_map(|s| by_name.get(s.as_str()).copied())
            .unwrap_or_else(|| panic!("`{sidecar}` resolves to nothing this plan publishes"));
        assert!(
            model_loader::types::encoding_nbytes(&decl.shape, &decl.encoding).unwrap() > GUESS,
            "`{sidecar}` was expected to be one of the ones that overflow"
        );
    }

    // Reported before the totals are asserted, because a name with no width is
    // a name this crate could not allocate for and that is the more useful
    // failure to read first.
    assert!(
        unresolved.is_empty(),
        "{} of {} weights the {} text binds have NO stated width, because `Naming` resolves \
         them to nothing: {:?}. A driver that loaded this plan would allocate for {measured} \
         weights and leave {} bound to whatever the arena held. See the module doc -- the \
         same six roles.",
        unresolved.len(),
        fixture.bound,
        fixture.id,
        &unresolved[..unresolved.len().min(8)],
        unresolved.len()
    );
    assert_eq!(
        measured, fixture.bound,
        "the {} text binds {measured} sizeable weights, not {}",
        fixture.id, fixture.bound
    );
    // MEASURED. For qwen3-0.6B this is the Vulkan file's number to the byte,
    // from the unquantised release with the encode run at load rather than
    // from `mlx-community/Qwen3-0.6B-4bit`.
    assert_eq!(total, fixture.total, "the whole model's bound weights");
}
