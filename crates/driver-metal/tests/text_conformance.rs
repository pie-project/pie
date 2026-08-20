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

use std::collections::{BTreeMap, BTreeSet};

use driver_metal::lowering::dispatch::{Geometry, Undispatchable, plan_launch};
use driver_metal::lowering::executor::{FireTable, Frame, Resolver, Slice};
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

/// The quantisation every text in this file is traced at, as the base every
/// entry spreads.
///
/// It used to spread `Geometry::default()`, which is zeros — honest for a
/// type the driver fills from a binding, and a statement that this deployment
/// quantises to groups of zero at zero bits. Every trace here names
/// `..._gs_64_b_4`, so every quant routine refused `Narrow { what: "the group
/// size", at: 0 }` before a rectangle was ever computed, and the walk that
/// checks grids reported those refusals as if they were the driver's.
///
/// The two numbers match `dispatch.rs`'s own fixtures, and `plan_routine`
/// checks the spelling a routine composes from them against the one the trace
/// states — so a text traced at some other quantisation fails here by name
/// rather than silently agreeing.
const QUANTISED: Geometry = Geometry {
    q_heads: 0,
    kv_heads: 0,
    head_dim: 0,
    rotary_dims: 0,
    n_experts: 0,
    experts_per_token: 0,
    group: 64,
    bits: 4,
    // Zero is "one attention shape", which is every text here but the two
    // gemma ones -- and those state their own beside their first.
    global_head_dim: 0,
    global_kv_heads: 0,
    full_attn_every: 0,
    v_heads: 0,
    v_dim: 0,
    // Zero is "one affine point", which is every text here but gpt-oss,
    // whose router gates arrived at their own width.
    router_group: 0,
    router_bits: 0,
};

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
                ..QUANTISED
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
                ..QUANTISED
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
                ..QUANTISED
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
                ..QUANTISED
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
                ..QUANTISED
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
                // The SECOND affine point, which is the whole reason this
                // row needed `build_kernels_at`. `gpt-oss-20b-MXFP4-Q4`
                // lists 98 tensors at group 64 / 4 bits and its 24
                // `mlp.router` gates at group 64 / EIGHT, and this text
                // names `affine_qmv_fast_bfloat16_gs_64_b_8` for the gate
                // beside `_b_4` for everything else.
                //
                // A production load reads it off the checkpoint by name --
                // `model::binding::observed`, and only when it differs.
                router_group: 64,
                router_bits: 8,
                ..QUANTISED
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
                // The SECOND attention shape, which is the whole reason this
                // fixture is named for gemma. `LlamaLikeMetalFacts::gemma_like`
                // states `global_head_dim: 256` over `global_kv_heads: 4`
                // against the paired `qwen3_0_6b`'s 128 over 8, and its
                // `window_left` is `-1` at `l % 6 == 5` -- one full layer
                // every six, spelled the way the KV pool spells it.
                //
                // A production load reads all three off the pool's `Shape`,
                // which `batch::geometry` derived from the same windows.
                global_head_dim: 256,
                global_kv_heads: 4,
                full_attn_every: 6,
                v_heads: 0,
                v_dim: 0,
                ..QUANTISED
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
                bits: 8,
                ..QUANTISED
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
                // The SECOND attention shape, which is the whole reason this
                // fixture is named for gemma. `LlamaLikeMetalFacts::gemma_like`
                // states `global_head_dim: 256` over `global_kv_heads: 4`
                // against the paired `qwen3_0_6b`'s 128 over 8, and its
                // `window_left` is `-1` at `l % 6 == 5` -- one full layer
                // every six, spelled the way the KV pool spells it.
                //
                // A production load reads all three off the pool's `Shape`,
                // which `batch::geometry` derived from the same windows.
                global_head_dim: 256,
                global_kv_heads: 4,
                full_attn_every: 6,
                v_heads: 0,
                v_dim: 0,
                ..QUANTISED
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
    // The four questions that arrived with DEFAULTS, and a default is `None`.
    // This fixture's whole promise is in its name, and it stopped keeping it
    // the moment the trait grew a method it did not implement -- so
    // `sdpa_paged_decode` refused for want of a page size and `gdn_core` for
    // want of a slab, in a test about whether a grid is legal. A refusal is
    // not a grid, so the walk below reported the fixture and called it the
    // driver.
    fn kv(&mut self, _: u16, _: bool) -> Option<Slice> {
        Some(Slice {
            address: 0x3000_0000,
            bytes: 1 << 30,
        })
    }
    fn slab(&mut self, _: u16, _: &'static str) -> Option<Slice> {
        Some(Slice {
            address: 0x4000_0000,
            bytes: 1 << 30,
        })
    }
    fn fire(&mut self, _: FireTable) -> Option<Slice> {
        Some(Slice {
            address: 0x5000_0000,
            bytes: 1 << 20,
        })
    }
    fn pool(&mut self, which: FireTable) -> Option<u32> {
        Some(match which {
            FireTable::KvPageSize => 16,
            _ => 1 << 12,
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
            // `multi_token` IS THE PREFILL, and a row count is not.
            //
            // These rows were `Row::default()` at both classes, so the
            // prefill lane was thirty-two rows each carrying a ONE-token
            // query window -- a batched decode, which is a real fire and not
            // the one this asks about. `GuardPred::WindowOne` reads exactly
            // that field (`lower::walk`), and `llama_like`'s attention arms
            // on it, so the fixture answered "every row is one token" and the
            // plan correctly named `sdpa_paged_decode`. The test that exists
            // to keep a prefill off the per-row kernel was reading the guard
            // that puts a BATCHED DECODE there on purpose.
            let low = lower(
                &plan,
                &vec![
                    Row {
                        samples: true,
                        multi_token: class != FireClass::Decode,
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
/// What a shader declares at every buffer index it uses.
///
/// # Why this is parsed rather than declared
///
/// `KernelSig` has an `operands` field and the CUDA table uses it. When this
/// was written **no Metal row declared one** -- the C++ shell bound by hand
/// from tables that are retiring, so nothing ever needed the arity written
/// down. Forty-eight rows came to state their operands, and then every row on
/// this backend retired.
///
/// It is still parsed, and for a better reason than the original: a routine
/// and its shader are two statements of the same call in two languages, and
/// the only way one can check the other is if this side is READ rather than
/// declared. The shader is the ABI; the routine is a claim about it -- and
/// unlike a row, the routine is what the driver actually binds from, so a
/// disagreement found here is a misbinding rather than a stale comment.
///
/// # The four kinds
///
/// MSL says what a buffer is for in its declaration and the distinctions are
/// not decoration:
///
/// * `device T* y` -- writable. The kernel's result.
/// * `const device T* x` -- readable. An input, a weight, or a device-side
///   scalar the host never sees.
/// * `const constant int& k` -- a scalar, bound to a buffer all the same.
///   Counting only POINTERS as buffers was tried first and read `add_bias`,
///   `affine_qmv_fast` and `kv_append` as short when all three were right.
/// * `const constant Params& p` -- a struct by reference, which a routine
///   binds as one opaque handle. It is a buffer of BYTES and asking whether
///   the routine put a pointer or a scalar there answers nothing, so it is
///   the one kind this does not compare.
///
/// The parse is deliberately crude and *conservative*: find the declaration
/// that names the buffers, take its parameter list, and read the kind out of
/// the text before each `[[buffer(N)]]`. A kernel it cannot find contributes
/// nothing, so this never invents a gap.
fn shader_slots(
    root: &std::path::Path,
    file: &str,
    stem: &str,
    entry: &str,
) -> Option<BTreeMap<usize, &'static str>> {
    let params = param_list(root, file, stem, entry)?;
    let mut slots = BTreeMap::new();
    let mut rest = params.as_str();
    while let Some(i) = rest.find("[[buffer(") {
        // The declaration for THIS buffer is the text since the last comma.
        let decl = rest[..i].rsplit(',').next().unwrap_or(&rest[..i]);
        let after = &rest[i + 9..];
        let j = after.find(')')?;
        let index: usize = after[..j].trim().parse().ok()?;
        let kind = if decl.contains('&') {
            let base = decl
                .split('&')
                .next()
                .unwrap_or(decl)
                .split_whitespace().rfind(|w| *w != "const" && *w != "constant" && *w != "device")
                .unwrap_or("");
            if PRIMITIVE.contains(&base) {
                "scalar"
            } else {
                "packed"
            }
        } else if decl.contains("device") && !decl.contains("const") {
            "out"
        } else {
            "in"
        };
        slots.insert(index, kind);
        rest = &after[j..];
    }
    (!slots.is_empty()).then_some(slots)
}

/// The MSL types a `&` parameter has to be for the slot to hold ONE number.
///
/// Anything else behind a reference is a struct, and a struct arrives as
/// bytes the routine hands over as a single handle.
const PRIMITIVE: &[&str] = &[
    "int", "uint", "float", "half", "bool", "short", "char", "long", "bfloat", "uint32_t",
    "int32_t", "uint8_t", "size_t", "ushort", "uchar",
];

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
/// and all three agreed on arrival. The two nothing names did not, and were
/// found to be short of their shader's twelve buffers by the check below --
/// which is the shape every widening so far has had, and the reason to keep
/// widening: the named kernels hold, the unnamed ones are where the faults
/// have been sitting unread.
fn param_list(root: &std::path::Path, file: &str, stem: &str, entry: &str) -> Option<String> {
    let src = std::fs::read_to_string(root.join(file)).ok()?;
    macro_param_list(&src, stem, 0)
        .or_else(|| quoted_macro(&src, stem).and_then(|f| declaration(&src, &f)))
        .or_else(|| host_name_alias(&src, entry).and_then(|f| declaration(&src, &f)))
}

/// The parameter list of `void <name>(..)`, taking the first that names its
/// buffers.
///
/// A template is written TWICE in these files: once as a definition, whose
/// parameters are named and carry `[[buffer(N)]]`, and once per instantiation
/// as a declaration, whose parameters are bare types. Only the first says
/// anything, and which comes first in the file is not fixed, so this walks
/// every occurrence rather than the first.
fn declaration(src: &str, name: &str) -> Option<String> {
    let needle = format!("void {name}(");
    let mut at = 0usize;
    while let Some(i) = src[at..].find(&needle) {
        let start = at + i;
        if let Some(list) = between_parens(src, start)
            && list.contains("[[buffer(")
        {
            return Some(list);
        }
        at = start + 1;
    }
    None
}

/// The C++ name behind a macro invocation whose first argument is the stem AS
/// A STRING.
///
/// `instantiate_sdpa_tiled_impl("sdpa_paged_tiled_sink", bfloat16, ..)` is the
/// only way this entrypoint's name appears anywhere: the `[[host_name]]` that
/// consumes it is `fn "_" #name "_d_" #d`, so the literal in the source is a
/// macro PARAMETER and no text in the file spells the entrypoint except the
/// call. [`macro_param_list`] already follows unquoted invocations; this is
/// the same step for a quoted one, and it stops at the `#define`'s `void`
/// rather than at its parameters, because a stamping macro's parameters are
/// bare types.
fn quoted_macro(src: &str, stem: &str) -> Option<String> {
    let quoted = format!("\"{stem}\"");
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
        if head[open + 1..].split(',').next().map(str::trim) != Some(quoted.as_str()) {
            continue;
        }
        if let Some(define) = src.find(&format!("#define {call}(")) {
            return void_name(&src[define..]);
        }
    }
    None
}

/// The C++ name behind the longest `[[host_name]]` literal the ENTRY starts
/// with.
///
/// A stem is not a C++ function name. `rope/neox.metal` stamps its kernels as
/// `[[host_name("neox_decode_" #name)]] void rope_neox_decode<itype>`, so the
/// entrypoint the census carries and the declaration that names the buffers
/// have different names and only the census joins them.
///
/// LONGEST, because the literals nest: `router_topk_` is a prefix of
/// `router_topk_scaled_` and both are prefixes of the scaled entry. NON-EMPTY,
/// because a literal built entirely from macro parameters is a prefix of
/// everything and would match whichever such kernel came first in the file --
/// that is how `sdpa_paged_tiled_sink` was read against a twenty-buffer
/// `sdpa_paged_tiled_strided` it has nothing to do with, and reported as short
/// by two.
fn host_name_alias(src: &str, entry: &str) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    let mut at = 0usize;
    while let Some(i) = src[at..].find("[[host_name(") {
        let start = at + i + "[[host_name(".len();
        at = start;
        let mut cursor = start;
        let mut literal = String::new();
        loop {
            let bytes = src.as_bytes();
            while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\t' | b'\\' | b'\n') {
                cursor += 1;
            }
            if src[cursor..].starts_with('"') {
                let Some(j) = src[cursor + 1..].find('"') else {
                    break;
                };
                literal.push_str(&src[cursor + 1..cursor + 1 + j]);
                cursor += j + 2;
            } else {
                break;
            }
        }
        if literal.is_empty() || !entry.starts_with(&literal) {
            continue;
        }
        if best.as_ref().is_some_and(|(len, _)| *len >= literal.len()) {
            continue;
        }
        let window = &src[start..floor_boundary(src, start + 400)];
        if let Some(name) = void_name(window) {
            best = Some((literal.len(), name));
        }
    }
    best.map(|(_, name)| name)
}

/// A byte index at or before `at` that `src` may be sliced at.
///
/// A window is arithmetic — *four hundred bytes past the macro* — and a byte
/// count is not a position. These shaders' comments are ruled with box
/// drawing, so `start + 400` lands inside a three-byte `─` and the slice
/// panics rather than answering. `str::floor_char_boundary` is the same
/// walk and is unstable, so this is it written out.
fn floor_boundary(src: &str, at: usize) -> usize {
    let mut at = at.min(src.len());
    while at > 0 && !src.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The identifier after the next `void` in `src`.
///
/// Any whitespace, not a space. These stamping lines wrap between the return
/// type and the name — `template [[host_name("qmm_splitk_reduce_f32_bfloat16")]]
/// [[kernel]] void` ends a line and `qmm_splitk_reduce<bfloat, float>(` starts
/// the next — so a needle of `"void "` finds nothing there and the stem was
/// dropped from the comparison silently. And `void` must be a word: `avoid`
/// ends in it.
fn void_name(src: &str) -> Option<String> {
    let mut at = 0usize;
    while let Some(i) = src[at..].find("void") {
        let start = at + i;
        let after = start + "void".len();
        at = after;
        let word = src[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if !word || !src[after..].starts_with(char::is_whitespace) {
            continue;
        }
        let rest = src[after..].trim_start();
        let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_')?;
        if end > 0 {
            return Some(rest[..end].to_owned());
        }
    }
    None
}

/// The text between the parentheses opened at or after `at`, depth-counted.
fn between_parens(src: &str, at: usize) -> Option<String> {
    let open = at + src[at..].find('(')?;
    let close = balanced(src.as_bytes(), open)?;
    Some(src[open + 1..close].to_owned())
}

/// The index of the bracket that closes the one at `start`.
fn balanced(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (at, byte) in bytes.iter().enumerate().skip(start) {
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(at);
                }
            }
            _ => {}
        }
    }
    None
}

/// A comma-separated list, split where the commas are not inside anything —
/// and a `//` comment is "inside something".
///
/// COMMENTS COUNTED AS ARGUMENTS. This walked the characters of a parameter
/// list or a dispatch list and split on every top-level comma, and a comma in
/// the PROSE beside an argument is a top-level comma. `moe::router_topk`
/// dispatches eight values and its list carries two explanatory comments with
/// one comma each, so this read ten -- and the shader declares eight, so the
/// conformance gate failed a routine that agrees with its shader down to the
/// slot. The house style puts the argument FOR an argument next to it, which
/// makes this the common case rather than the odd one.
fn split_top(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut comment = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if comment {
            comment = c != '\n';
            continue;
        }
        if c == '/' && chars.peek() == Some(&'/') {
            comment = true;
            continue;
        }
        match c {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if c == ',' && depth == 0 {
            out.push(cur.trim().to_owned());
            cur = String::new();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

/// The declared type of every entry in a routine's DISPATCH LIST, in order.
///
/// Not the signature. A routine's signature is a lower bound on what its body
/// binds, because a body may PAD: `moe::qmm_t_routed` names nine arguments and
/// dispatches thirteen values, repeating `pad` five times to push
/// `tile_expert` out to slot 12 where its shader declares it. Reading the
/// signature, that routine is indistinguishable from one that binds nine
/// values and never reaches slot 12 -- which is the actual defect, and which
/// several routines here had.
///
/// The list is right there in the body, so this reads it: the parameter list
/// for the name-to-type map, then the last `&[..]` in the body, which is the
/// slice handed to `ctx.dispatch`. Every routine has exactly one dispatch, so
/// "the last" is "the only".
///
/// A `Ctx` that RECORDED instead of encoding would be exact where this is a
/// parse, and it is not available: a `Fire` is built inside a dispatch, and
/// calling every routine means inventing arguments for every routine.
fn dispatch_list(name: &str) -> Option<Vec<(String, String)>> {
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/src");
    for family in FAMILIES {
        let text = std::fs::read_to_string(src.join(format!("{family}.rs")))
            .unwrap_or_else(|e| panic!("cannot read {family}.rs: {e}"));
        let Some(at) = text.find(&format!("\npub fn {name}(")) else {
            continue;
        };
        let open = at + format!("\npub fn {name}").len();
        let close = balanced(text.as_bytes(), open)?;
        let mut types: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        for part in split_top(&text[open + 1..close]) {
            if let Some((field, ty)) = part.split_once(':') {
                types.insert(field.trim().to_owned(), ty.trim().to_owned());
            }
        }
        let body = close + text[close..].find('{')?;
        let end = balanced(text.as_bytes(), body)?;
        let body = &text[body..end];
        // A DISPATCHED VALUE NEED NOT BE A PARAMETER.
        //
        // The signature was the whole table when this was written, and it is
        // not: `ctx.ask` binds what the ENVIRONMENT carries rather than what
        // the statement does, and such a value is dispatched beside the
        // parameters. Reading only the signature leaves those entries with an
        // empty type, and an empty type is not `InPacked`, so the strip below
        // stopped at the first one and `row_gather` counted the request count
        // -- a field of the struct at buffer 3 -- as a fifth buffer.
        for (at, _) in body.match_indices("ctx.ask::<") {
            let Some(name) = body[..at].rsplit_once("let ").map(|(_, n)| n) else {
                continue;
            };
            let name = name.split(['=', ':']).next().unwrap_or_default().trim();
            let Some(ty) = body[at + "ctx.ask::<".len()..].split(',').next() else {
                continue;
            };
            if !name.is_empty() {
                types.insert(name.to_owned(), ty.trim().to_owned());
            }
        }
        let list = body.rfind("&[")?;
        let list_end = balanced(body.as_bytes(), list + 1)?;
        return Some(
            split_top(&body[list + 2..list_end])
                .iter()
                .map(|item| {
                    let head = item.split('.').next().unwrap_or(item).trim();
                    (
                        head.to_owned(),
                        types.get(head).cloned().unwrap_or_default(),
                    )
                })
                .collect(),
        );
    }
    None
}

/// The families whose modules carry routine bodies.
const FAMILIES: &[&str] = &[
    "attn", "layout", "mlp", "moe", "norm", "ptir", "quant", "rope", "sample", "ssm",
];

/// What a routine's argument type is, in the vocabulary [`shader_slots`]
/// answers in.
///
/// Both mutable pointer kinds count as `out`, because the shader says
/// "writable" with the ABSENCE of `const` and does not distinguish an opaque
/// buffer from a typed one. `gdn_prep` writes four `F32sMut` and no `BufMut`
/// at all, so asking only about `BufMut` would read it as a kernel that
/// writes nothing.
///
/// `Env<T>` is a number the environment supplies rather than the trace, which
/// changes where it comes from and not what the slot holds. `InPacked` is a
/// field of a struct an earlier argument binds and returns [`None`]: it
/// occupies a position without binding one.
fn routine_kind(ty: &str) -> Option<&'static str> {
    const OUT: &[&str] = &["BufMut", "F32sMut", "I32sMut", "U32sMut", "U8sMut"];
    const IN: &[&str] = &["Buf", "F32s", "I32s", "U32s", "U8s"];
    const SCALAR: &[&str] = &["i32", "u32", "f32", "usize"];
    if OUT.contains(&ty) {
        Some("out")
    } else if IN.contains(&ty) {
        Some("in")
    } else if SCALAR.contains(&ty)
        || ty
            .strip_prefix("Env<")
            .and_then(|rest| rest.strip_suffix('>'))
            .is_some_and(|inner| SCALAR.contains(&inner))
    {
        Some("scalar")
    } else {
        None
    }
}

/// The ORPHAN this absorbed, and what it was doing loose in the file.
///
/// Everything from here to the next rule sat above `fn balanced`, a
/// bracket-matching helper it says nothing about -- left behind when the
/// test it documents moved down past the helpers. A doc comment with no
/// item under it is not a compile error and `-D warnings` only reached it
/// once this crate's clippy step was run, which no machine had done on this
/// branch. It is kept whole rather than deleted: it records the ORIGINAL
/// defect, which the ledger below is the descendant of.
///
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
///
/// ---
///
/// **Every routine's dispatch list, held against its shader's buffers.**
///
/// This check kept two ledgers and now keeps none, and the emptying is the
/// result rather than a tidy-up.
///
/// `SHORT` held stems whose list was shorter than their shader's buffer
/// count. It could not be an assertion while the check read the SIGNATURE, because a
/// signature does not state padding: `moe::qmm_t_routed` takes nine arguments
/// and dispatches thirteen, repeating one `pad` into the holes, so from the
/// signature a body that reaches slot 12 correctly and a body that stops at
/// slot 7 read alike. Seventeen stems were collected and seven of them were
/// right. Reading the dispatch list -- the list `lay_out` actually walks, and
/// which is written out in the body -- separated the two, and left ten faults.
///
/// `MISBOUND` held a sharper half, which no length check can see: a list of
/// the right LENGTH whose first writable value lands where the shader declares
/// an input. `affine_qmm_t_splitk` dispatched eleven against eleven and put
/// its output at 4 where the shader writes at 8, so bound as written it wrote
/// its partials over an activation and returned something plausible.
///
/// All seventeen were one defect. The quant shaders number their buffers from
/// a SHARED argument table -- `affine_qmm_t` binds `w, scales, biases, x, y`
/// at 0..4, and its precast, strided, split-K and tail variants keep those
/// numbers while moving the activation to 12 and adding operands at 7, 8 and
/// 13. A positional list reaches index 12 only by passing through 3..11, so
/// each of them takes a `pad` argument bound at every index its shader leaves
/// undeclared, which is the idiom `qmm_t_routed` already documents. Nothing
/// was reading garbage on purpose; the holes were simply not addressable from
/// a list that stopped early, and an unbound index on this driver is not an
/// error, it is whatever the previous step wrote at that address.
///
/// Nothing caught them because they were transcriptions of rows that stated no
/// `operands`, and a row that stated none was bound positionally too: the row
/// and the shader were wrong in the same way and agreed. No `model-dsl` text
/// names any of the seventeen, which is why they could be survived at all.
///
/// With both ledgers empty, all three axes fail. A stem arriving in the
/// failure means one of three things and none is allowed: a value bound past
/// the last declared buffer, which the kernel cannot read; a declared buffer
/// the list never reaches, which nothing binds; or a slot whose KIND is
/// wrong -- a pointer where the shader reads a number, a number where it
/// reads a pointer, or a read where it writes.
///
/// The kind axis is the one that found `affine_qmm_t_bias` and
/// `affine_qmm_t_strided_residual`, and it is the only one that could have.
/// Both dispatched exactly as many values as their shaders declare buffers,
/// and both put their result at buffer 4 where the shader writes. What they
/// got wrong was the ORDER of the three values after it: they bound the extra
/// pointer at 5 and `K, N` at 6 and 7, where the shader reads `K, N` at 5 and
/// 6 and the pointer at 7. `qmm_t_strided_residual` even carried a doc
/// asserting the wrong order as a fact about the shader.
///
/// A routine may STATE a hole, and where it does the kind axis steps back.
/// `pad` and `kv_append_paged`'s `ring_4`..`ring_15` are arguments whose whole
/// content is "the shader declares something here and this caller has nothing
/// for it"; asking whether such a slot holds the right kind asks about a value
/// nobody reads. Naming the hole is what earns the exemption, which is why the
/// idiom is a named argument rather than a repeated one.
#[test]
fn every_routine_agrees_with_the_shader_its_stem_names() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels");

    let mut disagrees: Vec<String> = Vec::new();
    let mut unrouted: Vec<String> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();
    let mut compared = 0usize;

    // The census, keyed by STEM. This walked `kernels_metal::KERNELS`, taking
    // each row's `symbol` and `file` -- and every family has retired its
    // rows, so that walk now visits nothing and passes. The two facts it
    // needed survive elsewhere: the stem is what the routine registry states,
    // and the file is what `ENTRYPOINTS` carries beside each instantiated
    // name. A stem's file is the file of any symbol it claims; they agree,
    // because a template and its instantiations are one declaration.
    //
    // The ENTRY matters too, and only for the stems whose declaration is
    // written under another name -- `host_name_alias` needs a name the census
    // carries to join on. The SHORTEST claimed entry is the stem's own: a
    // stem claims every entry it prefixes, so `affine_qmm_t` claims
    // `affine_qmm_t_fp16_precast_bfloat16_..` as well as its own
    // instantiations, and taking whichever came first read it against the
    // precast's thirteen buffers. The shortest is the one with the least name
    // that is not the stem's.
    let shaders = kernels_metal::shaders();
    let claim = |stem: &str| {
        shaders
            .iter()
            .filter(|(_, entry)| {
                entry
                    .strip_prefix(stem)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .min_by_key(|(_, entry)| entry.len())
            .copied()
    };

    // A DARK stem answers nothing on purpose and the registry states the
    // argument for each. Anything else with a shader and no routine is a
    // kernel nothing can reach, which is a louder fault than a signature that
    // disagrees.
    for (stem, _) in driver_metal::lowering::routine::DARK {
        if claim(stem).is_none() {
            unrouted.push(format!("  {stem} is dark and its shader is gone"));
        }
    }

    for (symbol, routine) in driver_metal::lowering::routine::stems() {
        let Some((file, entry)) = claim(symbol) else {
            unrouted.push(format!("  {symbol} names no shader in the census"));
            continue;
        };
        // THE SLOTS THE BODY BINDS, read from the dispatch list.
        //
        // Two readings of this were wrong before it was read from the body.
        // The first counted POINTERS, on the reasoning that only pointers are
        // buffers; MSL does not agree, since `add_bias` declares
        // `const constant int& width [[buffer(2)]]` and its third slot is a
        // scalar bound to a buffer. That rule read `affine_qmv_fast` as five
        // against seven and `kv_append` as five against eight, all of which
        // are live and right.
        //
        // The second counted SIGNATURE POSITIONS, which is the right unit and
        // the wrong list: a body may pad, padding is not in the signature, and
        // so `slots < buffers` had to be collected rather than failed. Ten of
        // the seventeen it collected were real and seven were bodies that
        // padded correctly. The dispatch list is what `lay_out` walks, so it
        // is what this walks.
        //
        // A TRAILING `InPacked` is a field of the struct an earlier value
        // binds and binds nothing itself, so the last position that binds is
        // the last one that is not one -- `layout::row_gather` dispatches five
        // values, the fifth is the packed count, and its shader declares four
        // buffers.
        let Some(dispatched) = dispatch_list(routine.name) else {
            unrouted.push(format!(
                "  {symbol} -> `{}` has no readable dispatch list",
                routine.name
            ));
            continue;
        };
        let slots = dispatched
            .iter()
            .rposition(|(_, ty)| ty != "InPacked")
            .map_or(0, |at| at + 1);
        let Some(shader) = shader_slots(&root, file, symbol, entry) else {
            unresolved.push(format!(
                "  {symbol} -> `{}`: nothing in {file} declares buffers under \
                 `{symbol}`, under the shortest entry it claims (`{entry}`), \
                 or under the name that entry's `host_name` aliases",
                routine.name
            ));
            continue;
        };
        compared += 1;
        let buffers = shader.keys().next_back().map_or(0, |&n| n + 1);
        // BOTH DIRECTIONS ARE FAULTS.
        //
        // More is a value bound past the last declared buffer, which the
        // kernel cannot read. Fewer is a declared buffer the list never
        // reaches, which nothing binds -- and an unbound index on this driver
        // is not a fault either, it is whatever the previous step left at
        // that address.
        if slots != buffers {
            disagrees.push(format!(
                "  {symbol} -> `{}`: dispatches {slots} slot(s), shader \
                 declares {buffers}",
                routine.name
            ));
        }
        for (at, (held, ty)) in dispatched.iter().enumerate() {
            // A STATED HOLE. The argument's name is the statement, and it is
            // why `kv_append_paged` spells six of them out instead of
            // repeating one handle: three other backends declare the same
            // sixteen slots, and a folded pad would make this port's call a
            // different call from theirs.
            if held == "pad" || held.starts_with("ring_") {
                continue;
            }
            let (Some(&want), Some(got)) = (shader.get(&at), routine_kind(ty)) else {
                continue;
            };
            // `packed` is a struct behind a reference. It arrives as one
            // opaque handle and answers nothing about pointers or numbers.
            if want != "packed" && want != got {
                disagrees.push(format!(
                    "  {symbol} -> `{}`: slot {at} holds `{held}: {ty}`, \
                     which is an {got}, and the shader declares an {want} \
                     there",
                    routine.name
                ));
            }
        }
    }

    disagrees.sort();
    disagrees.dedup();
    unrouted.sort();
    unrouted.dedup();

    assert!(
        unrouted.is_empty(),
        "a routine stem that reaches no shipped shader, so nothing it plans \
         can be built:\n{}",
        unrouted.join("\n")
    );
    // Named rather than counted. This used to be the arithmetic below alone,
    // and when the parser lost a stem the failure said `98 != 99` and left
    // the reader to find which -- so the diagnosis of a narrowed parser was
    // strictly worse than the diagnosis of a wrong argument.
    unresolved.sort();
    unresolved.dedup();
    assert!(
        unresolved.is_empty(),
        "a routine stem was not compared against a shader at all, which is a \
         declaration this can no longer FIND rather than a declaration that \
         agrees:\n{}",
        unresolved.join("\n")
    );
    assert!(
        disagrees.is_empty(),
        "a routine describes a kernel it does not match, and the arm fills it \
         anyway:\n{}",
        disagrees.join("\n")
    );
    // NINETY-NINE of ninety-nine. This read sixty-eight when it was first
    // written and seventy-three after one widening, and the number was kept
    // as a floor precisely because the gap was the interesting part.
    //
    // The gap closed in two steps, both of them the same realisation: a stem
    // is not a C++ function name. MSL kernels here are STAMPED, and a stamped
    // kernel's parameters are written under whatever name the template has:
    //
    // * `[[host_name("neox_decode_" #name)]] void rope_neox_decode<itype>` --
    //   the entrypoint and the declaration disagree on the name, and only the
    //   declaration carries buffers. `host_name_alias` joins them through the
    //   census, which is the only thing that knows both.
    // * `instantiate_sdpa_tiled_impl("sdpa_paged_tiled_sink", ..)` -- the
    //   entrypoint is a STRING argument and appears nowhere else in the file.
    //   `quoted_macro` follows the invocation to the `#define`.
    //
    // The widening was worth doing only once the comparison was settled, and
    // it paid immediately: the twenty-six stems it added carried five faults,
    // which is a higher rate than the seventy-three that came before.
    assert_eq!(
        compared,
        driver_metal::lowering::routine::stems().count(),
        "a routine stem was not compared against a shader at all, which \
         means a declaration this can no longer find rather than a \
         declaration that agrees"
    );
}

/// **The slots nothing fills, held against the shader that declares them.**
///
/// A row's unsourced operand was a slot nobody fills, and a slot
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
    // SLOT 3, NOT 4. `RouterParams` was buffer 3 and `per_expert_scale` was
    // 4; the struct was unwound into one `const constant uint&` per field at
    // ascending indices AFTER the operands, so the scale moved down into the
    // index the block vacated. `moe/route.metal` says so at its declaration.
    // An excuse that keeps the old index points at `n_experts`, which is a
    // slot the unscaled instantiation very much does fill.
    ("router_topk", 3, "per_expert_scale"),
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
// asserted every non-`None` source a `KERNELS_METAL` row states appears
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
/// # Why this is not [`shader_slots`]
///
/// It asks a narrower question -- what a slot is CALLED -- and answers it for
/// the stems [`macro_param_list`] alone can reach, which is no longer all of
/// them. `shader_slots` resolves two further ways and reaches every stem;
/// this does not need to, because the names it checks are the ones a `Row`
/// spells and those come from the unstamped declarations.
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
