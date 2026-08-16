//! What every Metal text must satisfy, checked once and reusable.
//!
//! Four families need texts and one has one. The checks that found the
//! defects in `llama_like`'s were the same four every time, so they are
//! written here **over a `ForwardPlan`** rather than over that one text —
//! a new family gets them by adding three lines to `texts()`, and gets them
//! the moment its first statement exists rather than after it is finished.
//!
//! # What each check caught, so none is deleted for looking obvious
//!
//! | check | what it found in `llama_like` |
//! |---|---|
//! | every symbol has a row | `attn::split_qkv_bf16`, which turned out to need a scalar channel and not a shader |
//! | every row states its file | three rows pointing at files that do not define them |
//! | every symbol is an INSTANTIATED point | four symbols named as bare stems, which resolve in the table and not in any shader |
//! | every launch becomes a legal grid | the `Unstated` rows for the whole batched lane |
//! | every weight name has a spelling | the map assuming HuggingFace naming |
//!
//! Two of those are only findable by *running* — a stem resolves through
//! `sig_in` because the row carries axes, and only the shader disagrees. So
//! this file holds the ones that are answerable on the host, and
//! `tests/device_text_fire.rs` holds the rest.

use std::collections::BTreeSet;

use driver_metal::lowering::dispatch::{Geometry, Undispatchable, plan_launch};
use driver_metal::lowering::executor::{Frame, Resolver, Slice};
use driver_metal::lowering::resolve::{Names, Store};
use model_compiler::lower::{Arg, Fire, Lowered, Row, lower};
use model_ir::trace::{FireClass, ForwardPlan, ValueId};

/// A text under test: how to trace it, and the geometry its fires run at.
struct Text {
    /// What to call it when a check fails.
    name: &'static str,
    /// Traced for a class.
    plan: fn(FireClass) -> ForwardPlan,
    /// The fire geometry the rules evaluate at.
    geometry: Geometry,
}

/// Every Metal text that exists.
///
/// **Add a row here when a family gets a text.** That is the whole cost of
/// joining this harness, and the point of writing it over `ForwardPlan`.
fn texts() -> Vec<Text> {
    vec![
        Text {
            name: "llama_like",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen3_0_6b(),
                    &LlamaLikeMetalFacts::synthetic(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
        // The same text at the WIDTH the device gates actually run. Every
        // other llama_like entry here is 128 wide; Llama-3.2-1B is 64, and
        // 64 is where `dsl::metal::sdpa` prefers the matrix-unit prefill
        // kernel. Without this entry the only 64-wide text is gpt-oss's,
        // which has sinks -- so `sdpa_paged_mma_sink` was named and plain
        // `sdpa_paged_mma` was not, on a build that dispatches it against
        // MLX twelve times per device run.
        Text {
            name: "llama_like (llama-3.2-1b, d=64)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::llama_3_2_1b(),
                    &LlamaLikeMetalFacts::synthetic(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 32,
                kv_heads: 8,
                head_dim: 64,
                rotary_dims: 64,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
        // The SAME text at a different fact, which is what a second entry here is
        // for. qwen3-moe is a llama-like attention with a routed FFN, so it joins
        // by naming a fixture rather than by being a family -- and every check
        // below then applies to the mixture's six statements without knowing they
        // are a mixture.
        Text {
            name: "llama_like (qwen3-moe)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen3_30b_a3b(),
                    &LlamaLikeMetalFacts::synthetic(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 32,
                kv_heads: 4,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 128,
                experts_per_token: 8,
                ..Geometry::default()
            },
        },
        // The SAME mixture with the dense expert a mixture may also have.
        //
        // `shared_intermediate` is the only field the seven shape fixtures
        // all state identically -- 0, meaning no shared expert -- and it is
        // therefore the one excuse `every_shape_predicate_is_stated_more_
        // than_one_way_or_excused` needs. Seen from the KERNEL side it is
        // the same hole: `shared_expert_combine` and its strided twin are
        // compiled into every Metal build, `llama_like_metal` names them
        // whenever the field is non-zero, and no text had ever set it, so
        // the slot conformance below had never inspected either.
        //
        // 512 is `Qwen3.6-35B-A3B`'s measured `shared_expert_intermediate_
        // size`. That row's own family is a GDN hybrid this driver refuses,
        // but the number is a real one rather than a plausible one, and the
        // dense leg it selects is the same leg qwen2-moe publishes.
        Text {
            name: "llama_like (qwen3-moe, shared expert)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts {
                        shared_intermediate: 512,
                        ..LlamaLikeFacts::qwen3_30b_a3b()
                    },
                    &LlamaLikeMetalFacts::synthetic(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 32,
                kv_heads: 4,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 128,
                experts_per_token: 8,
                ..Geometry::default()
            },
        },
        // The BIAS seam, and the reason is NOT that the symbol was
        // uncovered. It was written here claiming to be the only entry
        // with `qkv_bias: true`; a symbol census over all five says
        // otherwise -- this text names 12 symbols and none of them is
        // unique to it, because gpt-oss has attention biases too and has
        // been carrying `add_bias_bfloat16` all along.
        //
        // What it adds, measured: `add_bias` evaluated at a SECOND
        // geometry. gpt-oss reaches it at 64 q heads of 64, so the bias
        // spans 4096 and 512; Qwen-2.5-1.5B reaches it at 12 q heads of
        // 128, so 1536 and 256. `every_launch_of_every_text_becomes_a_
        // legal_grid` is a per-text check evaluated at the text's own
        // `Geometry`, and one shape is not a rule.
        //
        // And it states an INTENT the other entry holds by accident.
        // gpt-oss is here for sinks, its own SwiGLU and MXFP4 banks; its
        // biases are incidental, so a change to that fixture could take
        // the whole bias seam with it and nothing would say so. This row
        // is named for the seam, so losing it is a deletion rather than a
        // side effect.
        //
        // `add_bias` is a CAPABILITY on the metal facts and `qkv_bias` is
        // a fact about the checkpoint; `synthetic()` states the first and
        // `qwen2_5_1_5b()` the second, and both must hold for the text to
        // state the bias at all.
        Text {
            name: "llama_like (qwen2 qkv bias)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen2_5_1_5b(),
                    &LlamaLikeMetalFacts::synthetic(),
                    class,
                )
            },
            // Qwen2.5-1.5B as measured: 12 q heads, 2 kv, hidden 1536 / 12.
            // Standard rope over the whole head, so `rotary_dims == head_dim`.
            geometry: Geometry {
                q_heads: 12,
                kv_heads: 2,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
        // gpt-oss, and it joins the same way: attention SINKS, its own SwiGLU and
        // an alternating window are three facts, not a family. What is new is one
        // weight per layer and one symbol.
        Text {
            name: "llama_like (gpt-oss)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::gpt_oss_20b(),
                    &LlamaLikeMetalFacts::gpt_oss_20b(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                rotary_dims: 64,
                n_experts: 32,
                experts_per_token: 4,
                ..Geometry::default()
            },
        },
        // The three gemma facts that ARE facts. NOT gemma4 — its per-layer
        // embeddings are a side network of nine kernels no fact makes appear —
        // but the geglu, the readout softcap and the alternating window all reach
        // the device through this executor, so a gemma4 text has only the PLE and
        // the branch structure left to state.
        Text {
            name: "llama_like (gemma facts)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen3_0_6b(),
                    &LlamaLikeMetalFacts::gemma_like(),
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
        // The SAME shape at the OTHER affine width. `mlx-community`
        // publishes 8-bit snapshots of these rows beside the 4-bit ones and
        // `model::binding::observed` reads whichever arrived, so `_b_8` is a
        // point a production load reaches. `kernels-metal` instantiates it
        // (`affine_qmv_fast_bfloat16_gs_{32,64,128}_b_8` all exist) and no
        // text had ever named one -- so the slot conformance below, which
        // only looks at kernels a text names, had never looked at them.
        Text {
            name: "llama_like (8-bit affine)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen3_0_6b(),
                    &LlamaLikeMetalFacts {
                        affine_bits: 8,
                        ..LlamaLikeMetalFacts::synthetic()
                    },
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
        // The SAME family without its side network, which is not a smaller
        // gemma but a DIFFERENT branch: `llama_like_metal` reads the PLE
        // when `per_layer_emb_dim` is set and `layer_scalar` when it is not,
        // and `gemma_4::project::metal_facts` states `per_layer_scalar:
        // f.ple_dim == 0`. So gemma-4-31b and 26b-a4b take this arm and the
        // E-series takes the one above.
        //
        // It had no text. `per_layer_scalar` was on the list of predicates
        // this crate branches on that NO fixture turns on, and it was the
        // last one on that list with a shipped Metal row behind it -- the
        // 31b serves, and only its seventeen gigabytes keep it off this
        // machine's device rig.
        Text {
            name: "llama_like (gemma facts, scalar instead of PLE)",
            plan: |class| {
                use model::shared::llama_like::forward::facts::{
                    LlamaLikeFacts, LlamaLikeMetalFacts,
                };
                model::shared::llama_like::forward::llama_like_metal(
                    &LlamaLikeFacts::qwen3_0_6b(),
                    &LlamaLikeMetalFacts {
                        per_layer_emb_dim: 0,
                        per_layer_scalar: true,
                        ..LlamaLikeMetalFacts::gemma_like()
                    },
                    class,
                )
            },
            geometry: Geometry {
                q_heads: 16,
                kv_heads: 8,
                head_dim: 128,
                rotary_dims: 128,
                n_experts: 0,
                experts_per_token: 0,
                ..Geometry::default()
            },
        },
    ]
}

/// Answers every name, so a check is about the walk and not about a store.
struct Anything;

impl Resolver for Anything {
    fn weight(&mut self, _: &str) -> Option<Slice> {
        Some(Slice {
            address: 0x1000_0000,
            bytes: 1 << 30,
        })
    }
    fn named(&mut self, _: ValueId) -> Option<Slice> {
        Some(Slice {
            address: 0x2000_0000,
            bytes: 1 << 30,
        })
    }
}

/// Both fire classes, at a row count that exercises each lane.
///
/// The prefill's count is DERIVED, not written. `gemm_at` puts the tiled
/// GEMM behind `GuardPred::TokensMultipleOf(qmm_tile)`, so a prefill whose
/// rows are not a multiple of that tile takes the `otherwise` arm and this
/// whole file checks the matvec twice instead of checking the GEMM once.
/// That is not hypothetical: the count was written `16` when the tile was
/// 16, `d7e0f0d4f` widened the tile to 32 for a measured 4.5x, and
/// `affine_qmm_t` and `affine_qmm_t_residual` went dark in a suite that
/// still passed every other assertion. Reading the number off the fixture
/// is what makes the next widening a no-op here.
fn fires(text: &Text) -> Vec<(FireClass, Lowered)> {
    let tile = model::shared::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic()
        .qmm_tile
        .0
        .max(1) as usize;
    [(FireClass::Decode, 1usize), (FireClass::Prefill, tile)]
        .into_iter()
        .map(|(class, rows)| {
            let plan = (text.plan)(class);
            let low = lower(
                &plan,
                &vec![
                    Row {
                        samples: true,
                        ..Row::default()
                    };
                    rows
                ],
                Fire {
                    captures_across_splits: false,
                },
            )
            .unwrap_or_else(|why| panic!("{}: {class:?} does not lower: {why:?}", text.name));
            (class, low)
        })
        .collect()
}

#[test]
fn every_symbol_every_text_states_is_one_something_can_dispatch() {
    // This used to ask three questions of the ROW: does the symbol have one,
    // does it state a file, does it state a rule. The third went with the
    // rule interpreter and the other two went with the rows -- every Metal
    // family has retired them.
    //
    // What replaces them is one question the answer to which cannot rot: is
    // there a routine whose stem claims this symbol. That is the same
    // resolution `plan_launch` performs, so a fault here is a launch that
    // refuses `Unclaimed` at run time, named at its source instead of at the
    // dispatch.
    //
    // The file half moved twice. A crossed family's file is stated by its
    // BODY, in the `Fire` it returns, which
    // `every_launch_of_every_text_becomes_a_legal_grid` runs; and beside each
    // instantiated name in `ENTRYPOINTS`, which
    // `every_routine_agrees_with_the_shader_its_stem_names` reads.
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                if driver_metal::lowering::routine::crossed(symbol).is_none() {
                    faults.push(format!(
                        "{}/{class:?}: `{symbol}` is claimed by no routine stem, \
                         so nothing can dispatch it",
                        text.name
                    ));
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_symbol_is_an_instantiated_point_and_not_a_bare_stem() {
    // The check that only exists because running found it. A stem RESOLVES --
    // `crossed` claims `embed_gather_4bit` with the routine of that name, and
    // the test above is satisfied — while no shader exports it, because what
    // the shader exports is the axis point `embed_gather_4bit_bfloat16`.
    //
    // The row's axis product was the set to check against. Every family has
    // retired its rows, and the same set survives as the CENSUS: the
    // `(file, entrypoint)` pairs the families state, which is what
    // `device_kernels.rs` builds a pipeline for. A symbol that is not one of
    // them is a symbol no pipeline can be built from.
    let census: BTreeSet<String> = kernels_metal::entrypoints().into_iter().collect();
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                if census.contains(symbol.as_str()) {
                    continue;
                }
                // A stem, or a point that was never instantiated: the same
                // failure at the device either way, so the report shows what
                // the census DOES carry for the routine that claims it.
                let near: Vec<&String> = census
                    .iter()
                    .filter(|e| e.starts_with(symbol.as_str()))
                    .take(4)
                    .collect();
                faults.push(format!(
                    "{}/{class:?}: `{symbol}` is not an instantiated entry point. \
                     The census carries {near:?}. A stem resolves in the registry \
                     and in no shader -- spell the point from the deployment's facts.",
                    text.name
                ));
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

/// Every dark stem still names a shipped shader.
///
/// `routine::DARK` is a list of excuses, and an excuse outliving its subject
/// is how the next one gets believed. Each entry claims a kernel exists that
/// this backend deliberately does not cross; if the shader has left the tree
/// the entry is dead weight that also silently blocks a stem lookup.
#[test]
fn every_dark_stem_names_a_kernel_that_is_still_here() {
    let shipped: BTreeSet<String> = kernels_metal::entrypoints().into_iter().collect();
    for (stem, why) in driver_metal::lowering::routine::DARK {
        assert!(
            shipped.iter().any(|e| e == stem
                || e.strip_prefix(*stem)
                    .is_some_and(|rest| rest.starts_with('_'))),
            "`{stem}` is not crossed because {why} -- but no shipped \
             entrypoint starts with it, so the argument has outlived its \
             subject."
        );
    }
}

#[test]
fn every_launch_of_every_text_becomes_a_legal_grid() {
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            let frame = Frame {
                arena: Slice {
                    address: 0x8000_0000,
                    bytes: low.arena_bytes as u64,
                },
            };
            for launch in &low.launches {
                match plan_launch(&low, launch, frame, text.geometry, &mut Anything) {
                    Ok(plan) => {
                        for d in &plan {
                            let threads: u64 = d.grid.iter().map(|&n| u64::from(n)).product();
                            let group: u64 = d.threadgroup.iter().map(|&n| u64::from(n)).product();
                            if threads == 0 || group == 0 || group > 1024 {
                                faults.push(format!(
                                    "{}/{class:?}: `{}` wants grid {:?} in groups of {:?}",
                                    text.name, d.symbol, d.grid, d.threadgroup
                                ));
                            }
                        }
                    }
                    Err(Undispatchable::Unclaimed { .. }) => {}
                    Err(other) => {
                        faults.push(format!("{}/{class:?}: {other:?}", text.name));
                    }
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_weight_name_every_text_states_has_a_checkpoint_spelling() {
    let (tensors, named) = (Default::default(), Default::default());
    let store = Store::new(Names::mlx(), &tensors, &named);
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for arg in &low.args {
                let Arg::Weight(name) = arg else { continue };
                // A `scale.` marker is a constant riding the weight slot; the
                // binder never looks it up.
                if name.starts_with("scale.") {
                    continue;
                }
                if store.checkpoint_name(name).is_none() {
                    faults.push(format!(
                        "{}/{class:?}: `{name}` has no spelling in `Names::mlx`",
                        text.name
                    ));
                }
            }
        }
    }
    faults.sort();
    faults.dedup();
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

/// The two gemma texts are one deployment either side of ONE branch.
///
/// `llama_like_metal` reads the per-layer embeddings when
/// `per_layer_emb_dim` is set and `layer_scalar` when it is not, and
/// `gemma_4::project::metal_facts` chooses between them with
/// `per_layer_scalar: f.ple_dim == 0`. So the E-series takes one arm and
/// gemma-4-31b and 26b-a4b take the other, and until the second text
/// above existed only one of the two had ever been emitted.
///
/// Asserted as an EXCHANGE rather than as a presence: a text that named
/// both would mean the branch is not a branch, and a text that named
/// neither would pass a check written as "the scalar one has a scalar".
#[test]
fn the_two_gemma_texts_trade_the_side_network_for_a_scalar() {
    let named = |name: &str| -> BTreeSet<String> {
        let text = texts()
            .into_iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("`{name}` is one of the rows in `texts()`"));
        fires(&text)
            .into_iter()
            .flat_map(|(_, low)| low.kernels)
            .collect()
    };
    let ple = named("llama_like (gemma facts)");
    let scalar = named("llama_like (gemma facts, scalar instead of PLE)");
    let has = |set: &BTreeSet<String>, stem: &str| set.iter().any(|k| k.contains(stem));

    assert!(
        has(&ple, "ple_"),
        "the side network is nine kernels and this text is the only one that emits them: {ple:?}"
    );
    assert!(
        !has(&ple, "layer_scalar"),
        "a text with a PLE takes the other arm"
    );
    assert!(
        has(&scalar, "layer_scalar"),
        "`per_layer_scalar` is the last predicate on this crate's \
         branched-on-but-never-fired list with a shipped Metal row behind \
         it, and this text is what fires it: {scalar:?}"
    );
    assert!(
        !has(&scalar, "ple_"),
        "a text without a PLE emits none of its nine"
    );
}

/// The two affine widths, asserted as an EXCHANGE rather than a presence.
///
/// `affine_bits` was the last scalar on the Metal facts' branched-on list
/// that no fixture stated twice, and the reason it survived a census is that
/// the width is exercised where it is CHOSEN -- `binding::observed` reads it
/// off the checkpoint, `AffineFormat::kernel_suffix` spells it -- and never
/// where it is EMITTED. Both halves of that sentence are needed: a suffix
/// function that agrees with itself proves nothing about whether the symbol
/// it names was ever compiled into this build, and the slot conformance
/// above only inspects kernels some text asks for.
///
/// Presence is the wrong relation for the same reason it was wrong for the
/// gemma pair. `_b_4` appearing in the 8-bit text would mean a projection
/// read its width from somewhere other than the facts, and that defect
/// leaves every symbol present.
#[test]
fn the_two_affine_widths_are_an_exchange_not_a_default() {
    let named = |name: &str| -> BTreeSet<String> {
        let text = texts()
            .into_iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("`{name}` is one of the rows in `texts()`"));
        fires(&text)
            .into_iter()
            .flat_map(|(_, low)| low.kernels)
            .collect()
    };
    let four = named("llama_like");
    let eight = named("llama_like (8-bit affine)");
    fn widths(set: &BTreeSet<String>) -> BTreeSet<String> {
        set.iter()
            .filter_map(|k| k.split("_b_").nth(1))
            .map(|tail| tail.split('_').next().unwrap_or(tail).to_string())
            .collect()
    }
    let one = |w: &str| BTreeSet::from([w.to_string()]);

    assert_eq!(
        widths(&four),
        one("4"),
        "the 4-bit text names one width: {four:?}"
    );
    assert_eq!(
        widths(&eight),
        one("8"),
        "`kernels-metal` instantiates `affine_qmv_fast_bfloat16_gs_{{32,64,128}}_b_8` \
         and until this text existed no plan in this crate had ever asked for one: {eight:?}"
    );
}

#[test]
fn the_harness_covers_every_family_that_has_a_text() {
    // The check that keeps the harness honest. A family whose text lands and
    // is not added to `texts()` gets none of the above, and the failure would
    // be silence — which is the one failure mode a conformance suite cannot
    // afford.
    //
    // NAMED rather than counted, for the reason `DELIBERATE` gives a few
    // hundred lines down: a count passes when a row is added and another
    // removed in the same run, and it says nothing about WHICH row the sixth
    // is. The names are the claim.
    //
    // NINE entries over ONE text, and the gap is the interesting part: every
    // one of them joined by naming a fixture rather than by being a family,
    // so a routed FFN, attention sinks, a geglu with a softcap, a q/k/v bias
    // and now a per-layer scalar all reach the device with no second text and
    // no per-family branch anywhere in the executor.
    //
    // The ninth joined for a different reason, and it is the exception that
    // shows what a fixture is FOR. Llama-3.2-1B states nothing the other
    // eight do not; it is 64 heads wide where they are 128, and the width is
    // a fact the DSL branches on. Every shape here was checked at one width
    // until it landed.
    let named: Vec<&str> = texts().iter().map(|t| t.name).collect();
    assert_eq!(
        named,
        [
            "llama_like",
            "llama_like (llama-3.2-1b, d=64)",
            "llama_like (qwen3-moe)",
            "llama_like (qwen3-moe, shared expert)",
            "llama_like (qwen2 qkv bias)",
            "llama_like (gpt-oss)",
            "llama_like (gemma facts)",
            "llama_like (8-bit affine)",
            "llama_like (gemma facts, scalar instead of PLE)",
        ],
        "a Metal text or fixture landed or left. Add or remove its row in \
         `texts()` — everything above is per-text and a shape not listed is a \
         shape not checked."
    );
}

/// How many buffers a shader's entry point declares.
///
/// # Why this is parsed rather than declared
///
/// `KernelSig` has an `operands` field and the CUDA table uses it. When this
/// was written **no Metal row declared one** — the C++ shell bound by hand
/// from tables that are retiring, so nothing ever needed the arity written
/// down. Forty-eight rows came to state their operands, and then every row on
/// this backend retired.
///
/// It is still parsed, and for a better reason than the original: a routine
/// and its shader are two statements of the same arity in two languages, and
/// the only way one can check the other is if this side is read rather than
/// declared. The shader is the ABI; the signature is a claim about it — and
/// unlike a row, the signature is what the driver actually binds from, so a
/// disagreement found here is a misbinding rather than a stale comment.
///
/// The parse is deliberately crude and *conservative*: find the template body
/// by its stem, take its parameter list, and count distinct `[[buffer(N)]]`
/// indices. A kernel it cannot find contributes nothing, so this never invents
/// a gap.
fn declared_buffers(root: &std::path::Path, file: &str, stem: &str) -> Option<usize> {
    let params = param_list(root, file, stem)?;
    let mut seen = BTreeSet::new();
    let mut rest = params.as_str();
    while let Some(i) = rest.find("[[buffer(") {
        rest = &rest[i + 9..];
        if let Some(j) = rest.find(')')
            && let Ok(n) = rest[..j].trim().parse::<usize>()
        {
            seen.insert(n);
        }
    }
    // The HIGHEST index plus one, not the count. A row is positional — its
    // n-th operand is buffer n — so a kernel with gaps in its indices needs a
    // row that covers them, and `kv_append_paged` has gaps: it declares
    // 0,1,2,3,5,10,12..15 and leaves the rest to a ring ABI it does not read.
    // `Source::Unbound` is what a row says in a gap, and the operands doc
    // already asks for exactly that: *"a row lists every operand the callee
    // has, defaulted or not"*.
    seen.iter().next_back().map(|&n| n + 1)
}

/// A shader entry's parameter list, by its template stem.
///
/// Delegates to [`macro_param_list`], which the slot-name check downstream
/// already used and this did not: it tries `void <stem>(` first, exactly as
/// this used to, and then follows a macro INVOCATION whose first argument is
/// the stem back to the `#define` that stamps the kernel. Two parsers in one
/// file where one strictly dominates was its own small drift.
///
/// Five more of the ninety-nine resolve that way, all five through
/// `quant/qmv.metal`'s `instantiate_gptoss_qmv(<entrypoint>, <template>, ..)`:
/// `affine_qmv_routed`, `affine_qmv_routed_bias`, `mxfp4_qmv_routed_bias`,
/// `affine_qmv_tail` and `affine_qmv_tail_bias`. The first three a text names
/// and all three agree. The two nothing names are exactly what the widening
/// added to `SHORT`, and nothing new landed in `MISBOUND`. Every widening so
/// far has had that shape -- the named kernels hold, the unnamed ones do not
/// -- which is the reason to keep widening.
fn param_list(root: &std::path::Path, file: &str, stem: &str) -> Option<String> {
    let src = std::fs::read_to_string(root.join(file)).ok()?;
    macro_param_list(&src, stem, 0)
}

/// Where a shader's first WRITABLE buffer sits, and where the trace's first
/// output sits.
///
/// A `device T*` with no `const` is an output; `const device` is an input and
/// `constant` is a scalar. So the index of the first writable buffer is the
/// index the kernel expects its first output at — and the trace states inputs,
/// then outputs, then weights, so its first output sits right after its
/// inputs.
///
/// When those two disagree, **every operand of that launch is bound at the
/// wrong slot**.
fn first_writable(root: &std::path::Path, file: &str, stem: &str) -> Option<usize> {
    let params = param_list(root, file, stem)?;
    let mut best: Option<usize> = None;
    let mut rest = params.as_str();
    let mut cursor = 0usize;
    while let Some(i) = rest.find("[[buffer(") {
        let decl = &rest[..i];
        let after = &rest[i + 9..];
        let j = after.find(')')?;
        let index: usize = after[..j].trim().parse().ok()?;
        // The declaration for THIS buffer is the text since the last comma.
        let decl = decl.rsplit(',').next().unwrap_or(decl);
        let writable = decl.contains("device") && !decl.contains("const");
        if writable && best.is_none_or(|b| index < b) {
            best = Some(index);
        }
        cursor += i + 9 + j;
        let _ = cursor;
        rest = &after[j..];
    }
    best
}

/// **The routine's signature agrees with its shader.**
///
/// `model::executor` used to bind "operands in the trace's stated order" —
/// inputs, then outputs, then weights, at buffers `0..n`. That is the
/// COMPILER's convention and it is not the kernels'. `affine_qmv_fast`
/// declares `w, scales, biases, x, y`: weights first. So the activation bound
/// where the packed weight belongs, and every operand after it was one slot
/// further wrong — on all nine of `llama_like`'s statements.
///
/// A `kernel!` row's `operands` column was the first answer, and this test
/// checked that column against the `.metal` source. The routines replaced it:
/// a body's Rust signature IS the argument table, in order, and the arm in
/// `lowering/arm.rs` is what fills it. So the comparison moves to the
/// signature, and it gets stronger by moving — the column was stated on
/// forty-eight of a hundred rows, and every routine has a signature.
///
/// The row is still what joins the two. It states `symbol` and `file`, which
/// is where the shader is; the routine states the parameters. Neither alone
/// can be checked against the MSL.
///
/// Two claims, and the second is the one that caught the original bug:
///
/// * same count. A signature with more buffers than the shader declares
///   binds past the end of the argument table.
/// * the writable buffer in the same place. `Ty::BufMut` and `Ty::F32sMut`
///   are the routine saying "this one is written", and the shader says it
///   with the absence of `const`. A routine that puts its output where the
///   shader put an input does not fault; it writes over its own input and
///   returns something plausible.
/// Stems whose signature accounts for FEWER slots than their shader declares.
///
/// Not a fault by itself. A body may pad -- `moe::qmm_t_routed` repeats one
/// argument five times to push its last operand out to slot 12 -- and padding
/// is invisible to a signature, so this is the set where the signature alone
/// cannot decide. It is written down so that it cannot grow quietly, which is
/// the only guarantee available without running the body.
///
/// Every entry is a quantized GEMM, a precast, a split-K, or the gated
/// DeltaNet prefill pair, and every one of them is a kernel whose shader
/// numbers its buffers from a SHARED argument table: `affine_qmm_t` binds
/// `w, scales, biases, x, y` at 0..4 and its variants keep those numbers
/// while adding operands at 8, 12 or 13. The C++ shell encoded them into one
/// table and ran several kernels against it. A positional list reaches those
/// indices only by padding up to them.
///
/// No text names any of them: the census of quant symbols a `model-dsl` text
/// can spell is `affine_qmm_t`, `affine_qmm_t_residual`, `affine_qmv_fast`,
/// `affine_qmv_fast_residual`, `affine_qmv_routed`, `affine_qmv_routed_bias`
/// and `mxfp4_qmv_routed_bias`, and none of these is one. They are
/// transcriptions of rows that stated no `operands` -- and a row that stated
/// none was bound positionally too, so the row and the shader were wrong in
/// the same way and agreed. This is the first statement of the pair that can
/// be checked against the other.
const SHORT: &[&str] = &[
    "affine_qmm_t_bias_fp16_precast",
    "affine_qmm_t_fp16_precast",
    "affine_qmm_t_residual_fp16_precast",
    "affine_qmm_t_routed",
    "affine_qmm_t_routed_fp16",
    "affine_qmm_t_splitk_fp16_precast",
    "affine_qmm_t_strided",
    "affine_qmm_t_strided_fp16_precast",
    "affine_qmv_tail",
    "affine_qmv_tail_bias",
    "affine_qmv_wide_strided",
    "cast_qmm_input_bfloat16_to_float16",
    "cast_qmm_input_strided_bfloat16_to_float16",
    "gdn_core_recurrent_prefill",
    "gdn_prep_prefill",
    "mxfp4_qmm_t_routed_bias",
    "qmm_splitk_reduce",
];

/// Stems whose first WRITABLE argument is not at the position their shader
/// declares its first writable buffer at.
///
/// This one is a fault, and padding cannot explain it: `qmm_t_fp16_precast`
/// names `w, scales, biases, y, half_in, k, n`, so its result is the fourth
/// bound position, and `affine_qmm_t_fp16_precast` declares
/// `device bfloat* y [[buffer(4)]]` with buffer 3 not declared at all. Bound
/// as written, the GEMM writes its result where the shader reads `x` and
/// reads its activation from buffer 12, which nothing binds.
/// `cast_qmm_input_bfloat16_to_float16` is the sharpest: three buffers, at 3,
/// 12 and 13, against a six-value list that starts at 0.
///
/// Listed rather than fixed because the fix is not local. There is no `Ty`
/// meaning "bind nothing here" -- that absence is why
/// `driver-metal`'s `DARK` exists at all, for `silu_mul_strided`, whose
/// entrypoint leaves a slot empty -- so a routine reaches slot 12 only by
/// naming twelve arguments and having its arm supply the ones the kernel
/// ignores, which is what `attn::kv_append_paged`'s `ring_*` arguments are.
/// Every entry here is unreached by any text, so the list costs nothing that
/// runs; what it must not do is stay unsaid.
const MISBOUND: &[&str] = &[
    "affine_qmm_t_bias_fp16_precast",
    "affine_qmm_t_fp16_precast",
    "affine_qmm_t_residual_fp16_precast",
    "affine_qmm_t_splitk",
    "affine_qmm_t_splitk_fp16_precast",
    "affine_qmm_t_strided_fp16_precast",
    "cast_qmm_input_bfloat16_to_float16",
    "cast_qmm_input_strided_bfloat16_to_float16",
    "gdn_core_recurrent_prefill",
    "qmm_splitk_reduce",
];

#[test]
fn every_routine_agrees_with_the_shader_its_stem_names() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels");

    let mut disagrees: Vec<String> = Vec::new();
    let mut unrouted: Vec<String> = Vec::new();
    let mut short: Vec<&str> = Vec::new();
    let mut compared = 0usize;

    // The census, keyed by STEM. This walked `kernels_metal::KERNELS`, taking
    // each row's `symbol` and `file` -- and every family has retired its
    // rows, so that walk now visits nothing and passes. The two facts it
    // needed survive elsewhere: the stem is what the routine registry states,
    // and the file is what `ENTRYPOINTS` carries beside each instantiated
    // name. A stem's file is the file of any symbol it claims; they agree,
    // because a template and its instantiations are one declaration.
    let shaders = kernels_metal::shaders();
    let file_of = |stem: &str| {
        shaders
            .iter()
            .find(|(_, entry)| {
                entry
                    .strip_prefix(stem)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .map(|(file, _)| *file)
    };

    // A DARK stem answers nothing on purpose and the registry states the
    // argument for each. Anything else with a shader and no routine is a
    // kernel nothing can reach, which is a louder fault than a signature that
    // disagrees.
    for (stem, _) in driver_metal::lowering::routine::DARK {
        if file_of(stem).is_none() {
            unrouted.push(format!("  {stem} is dark and its shader is gone"));
        }
    }

    for (symbol, routine) in driver_metal::lowering::routine::stems() {
        let Some(file) = file_of(symbol) else {
            unrouted.push(format!("  {symbol} names no shader in the census"));
            continue;
        };
        // The SLOTS the signature accounts for, which is not the number of
        // its pointers.
        //
        // This counted pointers, on the reasoning that "only the pointers are
        // buffers". MSL does not agree: `add_bias` declares
        // `const constant int& width [[buffer(2)]]`, so its third slot is a
        // scalar bound to a buffer, and a rule that skips scalars reads that
        // kernel as two-slotted against a three-slotted shader. It reads
        // `affine_qmv_fast` as five against seven and `kv_append` as five
        // against eight -- all three of which are live, load-bearing and
        // right. A slot is a POSITION in the argument list `lay_out` walks,
        // whatever rides in it.
        //
        // `Env` arguments are supplied by the environment and never bound, so
        // they take no position. A TRAILING `InPacked` is a field of the
        // struct an earlier argument binds and binds nothing itself, so the
        // last position that binds is the last one that is not one --
        // `layout::row_gather` names five arguments, the fifth is the packed
        // count, and its shader declares four buffers.
        let bound: Vec<kernels::Ty> = routine
            .args
            .iter()
            .filter(|(_, prov)| *prov != kernels::routine::Provenance::Env)
            .map(|(ty, _)| *ty)
            .collect();
        let slots = bound
            .iter()
            .rposition(|ty| *ty != kernels::Ty::InPacked)
            .map_or(0, |at| at + 1);
        if let Some(buffers) = declared_buffers(&root, file, symbol) {
            compared += 1;
            // ONE DIRECTION IS A FAULT AND THE OTHER IS NOT DECIDABLE HERE.
            //
            // A body may PAD, and `moe::qmm_t_routed` does: nine arguments,
            // and a dispatch list of thirteen because `pad` is repeated five
            // times to push `tile_expert` out to slot 12, which is where its
            // shader declares it. So the signature's count is a LOWER bound
            // on what the body binds, and a signature naming fewer slots than
            // the shader declares may be padding correctly.
            //
            // More is never padding. A value bound past the last declared
            // buffer is a value the kernel cannot read.
            if slots > buffers {
                disagrees.push(format!(
                    "  {symbol} -> `{}`: signature binds {slots} slot(s), \
                     shader declares {buffers}",
                    routine.name
                ));
            } else if slots < buffers {
                short.push(symbol);
            }
        }
        if let Some(writes) = first_writable(&root, file, symbol) {
            // Both mutable pointer kinds, because the shader says "writable"
            // with the absence of `const` and does not distinguish an opaque
            // buffer from a typed one. `gdn_prep` writes four `F32sMut` and
            // no `BufMut` at all, so asking only about `BufMut` would read it
            // as a kernel that writes nothing.
            //
            // Positions among BOUND arguments, for the reason above -- and
            // unlike the count, this one padding cannot rescue: a repeated
            // pad before the output moves the output's position too, so the
            // signature and the body agree about which position it is unless
            // the pad comes after it, which would put the output where the
            // shader does not declare one either way.
            let by_routine = bound.iter().position(|ty| {
                matches!(
                    ty,
                    kernels::Ty::BufMut
                        | kernels::Ty::F32sMut
                        | kernels::Ty::I32sMut
                        | kernels::Ty::U32sMut
                        | kernels::Ty::U8sMut
                )
            });
            if by_routine != Some(writes) && !MISBOUND.contains(&symbol) {
                disagrees.push(format!(
                    "  {symbol} -> `{}`: shader writes buffer {writes}, the \
                     signature puts its first writable at {by_routine:?}",
                    routine.name
                ));
            }
        }
    }

    disagrees.sort();
    disagrees.dedup();
    unrouted.sort();
    unrouted.dedup();
    short.sort_unstable();
    short.dedup();

    assert_eq!(
        short, SHORT,
        "the set of routines whose signature accounts for fewer slots than \
         their shader declares has moved. A line arriving is a routine that \
         cannot reach its shader's last buffer unless its body pads to it; a \
         line leaving is one that no longer needs to, and its entry has to go \
         with it."
    );

    assert!(
        unrouted.is_empty(),
        "a routine stem that reaches no shipped shader, so nothing it plans \
         can be built:\n{}",
        unrouted.join("\n")
    );
    assert!(
        disagrees.is_empty(),
        "a routine describes a kernel it does not match, and the arm fills it \
         anyway:\n{}",
        disagrees.join("\n")
    );
    // SEVENTY-THREE of ninety-nine, and the gap is the point of writing the
    // number down rather than a floor.
    //
    // `param_list` finds a declaration by `void <stem>(`, or by a macro
    // invocation whose first argument is the stem. Twenty-six stems are
    // neither, because MSL kernels here are STAMPED and a stamped kernel's
    // parameter list is written under a name that is not the entrypoint's:
    //
    // * `[[host_name("neox_decode_" #name)]] void rope_neox_decode<itype>` --
    //   the entrypoint and the declaration have different names, and only the
    //   declaration carries buffers.
    // * `instantiate_sdpa_tiled_impl("sdpa_paged_tiled_sink", ...)` -- the
    //   entrypoint arrives as a STRING argument, and the `[[host_name]]` that
    //   consumes it is built from a macro parameter.
    //
    // A parser that follows the census back to the longest `[[host_name]]`
    // literal a stem claims reaches 98 of 99; it was written and measured
    // before this sentence, against `scripts/metal-kernel-audit.py`'s expanded
    // census. Landing it widens the COMPARISON by thirty-one stems, and
    // `SHORT` and `MISBOUND` are what the sixty-eight already here had to say
    // before that is worth doing -- a widening onto an unsettled comparison
    // reads as a wall of noise and gets excused wholesale.
    assert!(
        compared >= 73,
        "only {compared} symbol(s) were compared against a shader, which is \
         fewer than this has ever reached -- the loop is no longer reading \
         the sources it claims to."
    );
}

/// **The slots nothing fills, held against the shader that declares them.**
///
/// A row's `Source::Unbound` operand was a slot nobody fills, and a slot
/// nobody fills is read anyway -- on this backend, whatever the last dispatch
/// left there. Two tests asked about it: this one counted the holes, and a
/// second counted the `Source::Param` scalars a statement had to supply.
///
/// Both read row COLUMNS, and every Metal family has retired its rows. The
/// statement half did not disappear with them -- it moved to where it is
/// enforced rather than counted. A routine's argument list is positional and
/// total: it cannot express a hole (which is why `silu_mul_strided` is
/// [`DARK`] rather than crossed), and a scalar it cannot source is a
/// `plan_launch` refusal, which
/// [`every_launch_of_every_text_becomes_a_legal_grid`] runs for every launch
/// of every text. A count that used to be thirteen and reached zero is now a
/// thing that cannot be built.
///
/// What does not survive that move is the ARGUMENT: seventeen slots with a
/// paragraph each saying why they are empty. Those are facts about SHADERS,
/// so they are held against shaders here.
///
/// Writing the list down against the real parameter lists corrected it twice,
/// which is the point:
///
/// * six of `kv_append_paged`'s seven "ring" slots are not declared at all.
///   The shader takes 0,1,2,3,5,10,12,13,14,15 and the row invented names for
///   the GAPS so that its positional operand list could reach past them.
/// * the seventh, `ring_15`, is `src_row_stride` -- a slot the shader really
///   declares, under a name the row made up. A row that names a real slot
///   after an imaginary one is the exact drift two statements of one fact
///   produce, and nothing could see it while the row was the only statement.
const DELIBERATE: &[(&str, usize, &str)] = &[
    // A shared ring ABI `kv_append_paged` does not read. The shader declares
    // NOTHING at these indices -- they are holes in its `[[buffer(n)]]`
    // numbering -- so the empty name is the claim: nothing to fill.
    ("kv_append_paged", 4, ""),
    ("kv_append_paged", 6, ""),
    ("kv_append_paged", 7, ""),
    ("kv_append_paged", 8, ""),
    ("kv_append_paged", 9, ""),
    ("kv_append_paged", 11, ""),
    // Declared, and its real name. See the correction above.
    ("kv_append_paged", 15, "src_row_stride"),
    // A slot the OTHER instantiation of the same kernel fills. `sinks` is
    // `LlamaLikeMetalFacts::attn_sinks`'s, and no text in `texts()` sets it;
    // `bias` is `affine_qmv_routed_bias`'s; `per_expert_scale` is
    // `router_topk_scaled`'s.
    //
    // These were listed per INSTANTIATED symbol -- `_d_64`, `_d_128`,
    // `_d_256` -- because a row is a point set and the hole was found at each
    // point. The slot is a property of the template, so one entry per stem
    // says the same thing once. Width mattered to the old test for a reason
    // worth keeping: gemma-4 states two attention geometries, and the shared
    // gemma fixture used to leave `global_head_dim` at 0, so every gemma text
    // emitted one width. Stating the 256 reached an instantiation nothing had
    // ever fired.
    ("sdpa_paged_decode", 16, "sinks"),
    ("sdpa_paged_tiled", 16, "sinks"),
    ("sdpa_paged_mma", 16, "sinks"),
    ("affine_qmv_routed", 7, "bias"),
    ("router_topk", 4, "per_expert_scale"),
    // The one hole that is a property of a CODEC rather than of a sibling
    // instantiation. `biases` is the affine zero-point plane, and MXFP4 has
    // none: its scales are E8M0 block exponents with nothing to subtract.
    // The kernel takes the pointer and ignores it, which is why the slot may
    // be empty -- and why it may not be SOURCED. It was, from `Weight(2)`,
    // copied index-for-index off the affine row, and that pushed the additive
    // bias this symbol does read to a `Weight(3)` the codec's weight list
    // never reaches. See `model_dispatch::the_mxfp4_expert_bank_reads_a_bias_
    // and_is_handed_one`.
    ("mxfp4_qmv_routed_bias", 2, "biases"),
];

#[test]
fn every_slot_argued_for_is_the_slot_the_shader_declares() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels");
    let shaders = kernels_metal::shaders();

    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (stem, slot, want) in DELIBERATE {
        // The stem must still be one this backend crosses, or the argument is
        // about a kernel that has left.
        let claims = |name: &str| {
            name.strip_prefix(*stem)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
        };
        let Some((file, _)) = shaders.iter().find(|(_, entry)| claims(entry)) else {
            wrong.push(format!("  {stem}: no shader in the census names it"));
            continue;
        };
        let Some(names) = buffer_names(&root, file, stem) else {
            wrong.push(format!(
                "  {stem}: {file} declares no parameter list this can read"
            ));
            continue;
        };
        checked += 1;
        let got = names.get(*slot).map_or("", String::as_str);
        if got != *want {
            wrong.push(format!(
                "  {stem} [{slot}]: argued for as `{want}`, {file} declares \
                 `{got}`"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "{} argued slot(s) are not what the shader declares. An excuse \
         outliving its subject is how the next one gets believed.\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert_eq!(
        checked,
        DELIBERATE.len(),
        "every argued slot has to reach a parameter list; a stem this could \
         not parse is an argument nothing holds"
    );
}

/// Prefill TILES the attention decode walks row by row, in every text.
///
/// This test used to assert the opposite, and it is the same measurement
/// either way: `dsl::metal::sdpa` matched on `(paged, sinks)` and nothing
/// else, so a prefill launched `sdpa_paged_decode` -- "one threadgroup per
/// (q_batch_head, query_row)", its own header -- and every one of those
/// threadgroups walked the whole page table alone. Thirty-two rows read the
/// key run thirty-two times.
///
/// The shader that fixes it was already in the build, compiled at every
/// head width this driver serves, with the cost of NOT using it written in
/// its header: fitting `a + b*n` to the 30B checkpoint puts the quadratic
/// term at 39% of prefill time at n = 2048, and the traffic that term
/// implies is about 527 GB/s -- some 5% of this machine's fp16 peak.
/// Bandwidth, not arithmetic. So the fix is not a faster inner loop, it is
/// reading the run once for a tile of 32 query rows instead of once each.
///
/// What this asserts is the EXCHANGE, not that it is right: decode still
/// names the vector kernel, because at one row a tile of 32 is 31 rows of
/// wasted grid, and prefill names the tiled one. Whether the tiled kernel
/// computes the same numbers is not this file's question --
/// `device_real_weights` asks MLX.
#[test]
fn prefill_tiles_the_attention_decode_walks_row_by_row() {
    let mut checked = 0usize;
    for text in texts() {
        let per_class: Vec<BTreeSet<String>> = fires(&text)
            .into_iter()
            .map(|(_, low)| {
                low.kernels
                    .into_iter()
                    .filter(|k| k.starts_with("sdpa_"))
                    .collect()
            })
            .collect();
        assert_eq!(per_class.len(), 2, "`fires` runs decode and prefill");
        let (decode, prefill) = (&per_class[0], &per_class[1]);
        assert!(
            !decode.is_empty() && !prefill.is_empty(),
            "{}: names no attention kernel at all",
            text.name
        );
        assert!(
            decode.iter().all(|k| k.contains("_decode")),
            "{}: decode names attention outside the vector family: {decode:?}",
            text.name
        );
        // `_tiled` stages a run of keys for 32 query rows to share; `_mma`
        // tiles the same 32 rows onto the matrix unit and is preferred at
        // `_d_64`, where it is measured 3.4x-3.9x cheaper on the quadratic
        // term. Either is the shape this test is about. What must not appear
        // is `_decode`, which re-reads the whole key run once per row.
        assert!(
            prefill
                .iter()
                .all(|k| k.contains("_tiled") || k.contains("_mma")),
            "{}: prefill names attention outside the tiled and mma families: \
             {prefill:?}. A prefill on the decode kernel is the quadratic \
             read this test exists to keep from coming back",
            text.name
        );
        // The two are the SAME kernel at a different shape, so the sink half
        // has to travel: a text whose decode has sinks must have a prefill
        // with sinks, or gpt-oss silently loses its per-head logit on the
        // one class where the whole prompt goes through.
        assert_eq!(
            decode.iter().filter(|k| k.contains("_sink")).count(),
            prefill.iter().filter(|k| k.contains("_sink")).count(),
            "{}: the sink survives one fire class and not the other. \
             decode {decode:?}, prefill {prefill:?}",
            text.name
        );
        checked += 1;
    }
    assert!(checked >= 8, "every text is asked, not a subset: {checked}");
}
// ── A test that was here, and the stronger one it duplicated ─────────────
//
// `every_source_a_metal_row_states_is_one_the_operand_walk_names` stood here.
// It source-parsed `lowering/dispatch.rs` for `Source::X` mentions and
// asserted every non-`Unbound` source a `KERNELS_METAL` row states appears
// among them, on the reading that the walk's `_ => nothing` would otherwise
// bind an empty region.
//
// `driver-metal` already had that claim, made better, and I did not look
// before writing: `lowering::dispatch`'s own
// `every_source_the_table_names_is_one_this_binder_resolves` matches on the
// real `kernels::Source` enum rather than on the file's TEXT, covers the
// whole table rather than the sources some text happens to state, and splits
// the question the way the binder is split -- an operand `reorder` misses is
// a null pointer and a scalar `param_layout` misses is an unwritten index,
// which are different defects and this one could not tell them apart. Its
// doc names `Source::OutWidth` as the hole it was written for, with the
// Qwen-2 q/k/v biases that went unserved for it.
//
// Two names for one idea, so the weaker reading goes. What was kept from
// this one is its falsifier, moved into `dispatch.rs`'s own `mod tests` as
// `a_pointer_the_walk_cannot_place_refuses_instead_of_binding_a_null`,
// beside `a_scalar_the_walk_cannot_place_is_not_a_refusal` -- because the
// static check asks a HAND-KEPT list (`by_reorder`) that is a second
// spelling of the match, and the one thing it cannot see is the two of them
// drifting apart. `_ => nothing` is now a `BindRefusal::UnboundPointer` that
// names the row, the operand and the source.

/// A Metal entrypoint's buffer parameter NAMES, indexed by `[[buffer(n)]]`.
///
/// Slots the declaration skips come back empty: `[[buffer(n)]]` is explicit
/// and may have gaps -- `kv_append_paged` declares 0,1,2,3,5,10,12,13,14,15 --
/// so the vector is sized by the HIGHEST index and a gap is a name of `""`.
///
/// # Why this does not replace [`declared_buffers`]
///
/// It resolves one level of MACRO, which that does not, so it reaches the
/// stamped kernels that are most of this tree. Widening `declared_buffers`
/// the same way would put sixty more signatures under
/// `every_routine_agrees_with_the_shader_its_stem_names` in one commit, which
/// is a good change and a separate one: that test compares COUNTS and a new
/// disagreement there is a real finding to be read, not a parser change to be
/// landed alongside it.
fn buffer_names(root: &std::path::Path, file: &str, stem: &str) -> Option<Vec<String>> {
    let src = std::fs::read_to_string(root.join(file)).ok()?;
    let list = macro_param_list(&src, stem, 0)?;
    let mut out: Vec<String> = Vec::new();
    for param in list.split(',') {
        let Some(mark) = param.find("[[buffer(") else {
            continue;
        };
        let rest = &param[mark + "[[buffer(".len()..];
        let end = rest.find(')')?;
        let Ok(slot) = rest[..end].trim().parse::<usize>() else {
            continue;
        };
        // The identifier immediately before the attribute: `const device
        // bfloat* bias [[buffer(7)]]` -> `bias`.
        let name = param[..mark]
            .split_whitespace()
            .next_back()
            .unwrap_or("")
            .trim_start_matches('*')
            .trim_start_matches('&')
            .to_owned();
        if out.len() <= slot {
            out.resize(slot + 1, String::new());
        }
        out[slot] = name;
    }
    (!out.is_empty()).then_some(out)
}

/// The parameter list of the declaration `stem` names, following macros.
///
/// Three shapes. Written out: `template <...> [[kernel]] void stem(...)`.
/// STAMPED: `gptoss_qmv_kernel(qmv_routed_bias, true, true, 1)`, where the
/// list lives once inside `#define gptoss_qmv_kernel(name, ...)`. Stamped and
/// then INSTANTIATED: `instantiate_gptoss_qmv(mxfp4_qmv_routed_bias,
/// qmv_routed_bias, ...)`, whose `#define` declares the types alone with no
/// names at all.
///
/// So a body is accepted only when its list carries `[[buffer(`, and the
/// invocation's remaining arguments are followed when it does not -- the
/// third shape hands off to the second, which is where the names are.
fn macro_param_list(src: &str, stem: &str, depth: u32) -> Option<String> {
    fn between(src: &str, at: usize) -> Option<String> {
        let open = at + src[at..].find('(')?;
        // Depth-counted: `[[buffer(0)]]` closes a parenthesis the signature
        // did not open, so stopping at the first `)` finds a list of one.
        let mut depth = 0i32;
        for (i, c) in src[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(src[open + 1..open + i].to_owned());
                    }
                }
                _ => {}
            }
        }
        None
    }
    let named = |list: Option<String>| list.filter(|l| l.contains("[[buffer("));

    if depth > 3 {
        return None;
    }
    if let Some(at) = src.find(&format!("void {stem}("))
        && let Some(list) = named(between(src, at))
    {
        return Some(list);
    }
    for line in src.lines() {
        let head = line.split("//").next().unwrap_or(line).trim();
        if head.starts_with('#') {
            continue;
        }
        let Some(open) = head.find('(') else { continue };
        let call = head[..open].trim();
        if call.is_empty() || !call.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let args: Vec<&str> = head[open + 1..]
            .split(',')
            .map(|a| a.trim().trim_end_matches(')'))
            .collect();
        if args.first() != Some(&stem) {
            continue;
        }
        if let Some(define) = src.find(&format!("#define {call}("))
            && let Some(off) = src[define..].find("void ")
            && let Some(list) = named(between(src, define + off))
        {
            return Some(list);
        }
        for arg in &args[1..] {
            if let Some(list) = macro_param_list(src, arg, depth + 1) {
                return Some(list);
            }
        }
    }
    None
}
