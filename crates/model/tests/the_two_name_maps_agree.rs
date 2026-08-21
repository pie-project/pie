//! THE MLX NAME MAP ANSWERS EXACTLY WHAT THE METAL TEXT ASKS.
//!
//! A trace names its tensors in the text's own words -- `layer.7.qkv`,
//! `layer.7.mlp_norm` -- and a checkpoint names them in its exporter's:
//! `layers.7.self_attn.qkv_proj.fused.weight`. `Names::mlx` is the map
//! between the two, and `driver-metal`'s `Store::checkpoint_names` is its
//! only caller.
//!
//! It is DATA, and data is the thing a unit test cannot hold. Its own doc
//! records what happened the last time someone tried: an earlier draft
//! "assumed the HuggingFace spelling and was wrong on all three -- it was
//! self-consistent, and the test that held the text against it passed,
//! because both sides were this file." So nothing held it at all, and
//! until this file it was the one map in the crate that no test called.
//!
//! The question that is not circular is not what the map says, but whether
//! its VOCABULARY is the vocabulary the Metal text speaks -- and neither
//! side can answer that alone, because the text is in `crates/model` and
//! the resolution is in `driver-metal`. `Store::checkpoint_names` answers a
//! handle it does not know with an EMPTY candidate list, so a drifted name
//! is not a compile error and not a refusal: it is a load that asks for a
//! tensor and is told there is no such tensor.
//!
//! Both directions matter, and they fail differently. A handle the text
//! asks for and the map lacks is a load that cannot resolve. A key the map
//! answers that nothing asks for is what a RENAMED handle leaves behind --
//! harmless on its own, and the only visible trace of the half of a rename
//! that was forgotten.
//!
//! # This is not a restatement of the map
//!
//! Nothing here asserts what any role resolves to. The comparison is
//! between two sets of names produced by two different files, so it holds
//! even though it knows nothing about what the checkpoint calls anything.
#![cfg(feature = "contract")]

use model::catalog::{self, Deployed, MetalBinding};
use model::shared::weight_names::Names;
use model_ir::trace::FireClass;
use std::collections::BTreeSet;

/// The binding `tests/catalog_backends.rs` traces every row against.
const BINDING: MetalBinding = MetalBinding {
    qmm_partial_rows: false,
    qmm_fp16_precast: true,
    qmm_tile: None,
    quant_group: 64,
    quant_bits: 4,
    router_quant_group: 0,
    router_quant_bits: 0,
    moe_mxfp4: false,
    fuse_residual_gemv: true,
    paged_multi_batch: true,
    qmm_multi_batch: true,
    // FALSE, and unlike `add_bias` that is not a wrong answer quietly taken:
    // only `driver-vulkan` fires the fused form, and the two texts compute the
    // same thing at different dispatch counts, so no name moves either way.
    fused_qk_rope: false,
    // TRUE, and the only one of the four whose value is a copy of something
    // this crate cannot see. `driver-metal::model::binding::build_kernels` is
    // where the claim lives; the layering forbids a dependency on it, so this
    // restates it and `every_role_the_mlx_map_answers_is_one_some_trace_asks
    // _for` is what catches the restatement going stale — with `false` here,
    // the Metal text states no bias, no trace asks for `q_bias`, and the map
    // entry that answers it looks like dead weight.
    add_bias: true,
};

/// Every tensor name every row that serves Metal actually asks for.
///
/// Collected by serializing each plan and taking every `weight` field
/// rather than by matching on `OpKind`, because a match has to be extended
/// when a new op carrying a weight is added and this must not be a place
/// that silently stops looking. The plan is `Serialize` precisely so that
/// goldens can pin it, and this reads the same bytes a golden would.
fn names_metal_traces_ask_for() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut served = 0;
    for row in catalog::catalog() {
        for class in [FireClass::Prefill, FireClass::Decode] {
            let Ok(plan) = row.trace(class, Deployed::metal(&BINDING)) else {
                continue;
            };
            served += 1;
            let json = serde_json::to_value(&plan).expect("a plan serializes");
            collect_weights(&json, &mut names);
        }
    }
    println!("SERVED={served} NAMES={}", names.len());
    if let Some(n) = names.iter().next() {
        println!("SAMPLE={n}");
    }
    assert!(
        served > 0,
        "no row in the catalog traced a Metal text, so this file compares \
         the MLX map against an empty vocabulary and passes vacuously"
    );
    names
}

fn collect_weights(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                // A CUDA text names one tensor per op and a Metal text
                // names a list, because a quantised launch takes the
                // weight and its sidecars together.
                if let ("weight", Some(name)) = (k.as_str(), val.as_str()) {
                    out.insert(name.to_string());
                }
                if let ("weights", Some(items)) = (k.as_str(), val.as_array()) {
                    out.extend(items.iter().filter_map(|i| i.as_str()).map(str::to_string));
                }
                collect_weights(val, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_weights(item, out);
            }
        }
        _ => {}
    }
}

/// Sidecars a quantised tensor carries, which the map states separately in
/// `weight_suffix` and `zero_point_suffix` rather than keying per role.
const COMPANIONS: &[&str] = &[".scales", ".zeros", ".biases", ".bias", ".weight"];

/// The role a trace name asks for, with any layer index removed.
fn role(traced: &str) -> &str {
    // A companion rides its parent's name: the map keys the ROLE and
    // states the suffixes separately, so `embed.scales` is the `embed`
    // question asked about a sidecar.
    let traced = COMPANIONS
        .iter()
        .find_map(|c| traced.strip_suffix(c))
        .unwrap_or(traced);
    traced
        .strip_prefix("layer.")
        .map_or(traced, |rest| match rest.split_once('.') {
            Some((_index, role)) => role,
            None => rest,
        })
}

/// Roles the map answers that no Metal text asks for, and why.
///
/// `qkv` and `gate_up` are the map's own documented case: no Metal
/// deployment joins its projections, so the Metal text names the three and
/// the two separately and these two keys exist for a deployment that does
/// join. The rest belong to generations that reach no Metal text at all --
/// `attn_sinks` to gpt-oss, the `shared_*` bank to the shared-expert
/// mixtures -- and `tests/catalog_backends.rs` is where which generations
/// those are is stated and held.
///
/// Named rather than counted, so a generation gaining a Metal text has to
/// delete a line here.
const ANSWERED_BUT_UNASKED: &[&str] = &[
    "qkv",
    "gate_up",
    "attn_sinks",
    "shared_gate",
    "shared_up",
    "shared_down",
    "shared_gate_proj",
];

#[test]
fn every_name_a_metal_text_asks_for_is_one_the_mlx_map_answers() {
    let names = Names::mlx();
    assert!(
        !names.roles.is_empty() && !names.globals.is_empty(),
        "the MLX map is empty, so every assertion below holds vacuously"
    );

    let traced = names_metal_traces_ask_for();
    // A FLOOR ON PURPOSE, and the widest gap in this sweep: it reads 5,596
    // and asks for twenty. The subject is every traced name across every
    // model, which moves whenever a layer count or a family's graph moves,
    // so a census here would be re-measured without being read and would
    // teach the next person that the number means nothing. Twenty catches
    // the walk producing nothing, which at this scale is the only failure
    // it can have; the size is written here so nobody re-measures it to
    // discover that.
    assert!(
        traced.len() >= 20,
        "the Metal texts named only {} tensors, which is too few for this \
         comparison to mean anything",
        traced.len()
    );

    let mut unknown = Vec::new();
    for name in &traced {
        let r = role(name);
        let known = if name.starts_with("layer.") {
            names.roles.contains_key(r)
        } else {
            names.globals.contains_key(r)
        };
        if !known {
            unknown.push(r.to_string());
        }
    }
    unknown.sort();
    unknown.dedup();
    assert!(
        unknown.is_empty(),
        "the Metal texts ask for {unknown:?} but the MLX map has no entry \
         for them, so a Metal load asks for those tensors and is told there \
         is no such tensor -- add the entry to `Names::mlx`"
    );
}

/// A role the map answers that no text asks for is either dead or a
/// deployment that does not exist yet, and the difference has to be said.
///
/// This is the direction that catches half a rename. gemma-4 has two
/// texts, and they do not use the same handles: the CUDA one says
/// `pre_ffw_norm` and `ple_proj` where the Metal one says `mlp_norm` and
/// `ple_out`. Renaming a handle in one text leaves the other text's key
/// here with nothing asking for it, which is the only sign that the rename
/// stopped halfway.
#[test]
fn every_role_the_mlx_map_answers_is_one_some_trace_asks_for() {
    let names = Names::mlx();
    let traced: BTreeSet<String> = names_metal_traces_ask_for()
        .iter()
        .map(|n| role(n).to_string())
        .collect();

    let mut dead: Vec<&str> = names
        .roles
        .keys()
        .chain(names.globals.keys())
        .map(String::as_str)
        .filter(|r| !traced.contains(*r) && !ANSWERED_BUT_UNASKED.contains(r))
        .collect();
    dead.sort_unstable();
    assert!(
        dead.is_empty(),
        "the MLX map answers for {dead:?}, which no trace asks for -- \
         either the text renamed the handle and this is what the old \
         spelling left behind, or the entry was written for a text that \
         was never committed"
    );
}

/// The prefix and the two suffix lists are shape, not vocabulary.
///
/// They are the only parts of the map the two tests above cannot see: a
/// role's set of keys is unchanged by what its names are glued to. Each
/// property here is one the map's own doc states as a reason, held as a
/// rule rather than restated as a value.
///
/// The suffix ORDER is the subtle one. `checkpoint_names` builds the cross
/// product of a role's paths and these suffixes in order, and
/// `checkpoint_name` takes the first candidate the checkpoint actually
/// has -- so the bare spelling must come last. A checkpoint that ships
/// both `…q_proj` and `…q_proj.weight` is not hypothetical: the bare name
/// exists for gpt-oss's `self_attn.sinks`, a vector that hangs under no
/// module, and putting it first would make every quantised tensor's
/// module resolve ahead of its own weight.
#[test]
fn the_prefix_and_suffixes_are_shaped_the_way_resolution_needs() {
    let names = Names::mlx();

    assert!(
        names.layer_prefix.ends_with('.'),
        "the layer prefix is {:?}, and resolution glues it straight to the \
         layer index -- without the separator `layers.` and `7` become one \
         word and every layer-scoped tensor resolves to a name no \
         checkpoint has",
        names.layer_prefix
    );

    assert!(
        names.weight_suffix.contains(&String::new()),
        "no empty weight suffix, so a role whose tensor IS the value rather \
         than hanging under a module -- gpt-oss ships `self_attn.sinks`, \
         not `self_attn.sinks.weight` -- can never resolve"
    );
    assert_eq!(
        names.weight_suffix.last(),
        Some(&String::new()),
        "the empty weight suffix is not last, so the bare spelling is tried \
         before `.weight` and a checkpoint holding both resolves to the \
         module instead of to the tensor"
    );

    // This used to require `zero_point_suffix` to hold BOTH `.biases` and
    // `.bias`, on the theory that the two spellings were one role. They are
    // not: measured on `mlx-community/gpt-oss-20b-MXFP4-Q4`, an expert bank
    // publishes `bias` of `[32, 2880]` -- one value per output row -- while
    // the zero point beside `scales` would be `[32, 2880, 90]`, one per
    // group, and `qmv_routed_bias` reads them at two different buffers.
    // Conflating them read the bias off a null pointer. So the invariant is
    // not that the list is long, it is that the two lists are DISJOINT.
    assert!(
        !names.zero_point_suffix.is_empty(),
        "no zero-point spelling at all, so an affine weight can never find \
         the tensor that shifts it and every quantised value is decoded \
         around the wrong origin"
    );
    // Non-empty too, and it matters more than it used to: `bias_suffix` was
    // read only by the MXFP4 expert bank when it was added, and it is now how
    // the Qwen-2 family's `q_bias`/`k_bias`/`v_bias` roles reach their
    // tensors. Emptied, the disjointness loop below passes vacuously and
    // seven models go back to being served without their biases.
    assert!(
        !names.bias_suffix.is_empty(),
        "no additive-bias spelling at all, so a role that names one -- the \
         Qwen-2 projections, the MXFP4 expert banks -- resolves to nothing \
         and its kernel reads an unbound buffer"
    );
    for zero in &names.zero_point_suffix {
        assert!(
            !names.bias_suffix.contains(zero),
            "`{zero}` is listed as both a zero point and an additive bias, \
             and they are different tensors of different shapes -- whichever \
             list is consulted first answers for both and the other's \
             buffer is never bound"
        );
    }
    for suffix in names
        .weight_suffix
        .iter()
        .filter(|s| !s.is_empty())
        .chain(&names.zero_point_suffix)
        .chain(&names.bias_suffix)
    {
        assert!(
            suffix.starts_with('.'),
            "the suffix {suffix:?} does not begin with a separator, so it \
             runs into the name it is appended to"
        );
    }
}
