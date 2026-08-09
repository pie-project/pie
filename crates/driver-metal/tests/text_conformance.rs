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

use driver_metal::lowering::dispatch::{Geometry, Undispatchable, plan_one};
use driver_metal::lowering::executor::{Frame, Resolver, Slice};
use driver_metal::lowering::resolve::{Names, Store};
use model_compiler::lower::{Arg, Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ForwardPlan, ValueId};

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
fn fires(text: &Text) -> Vec<(FireClass, Lowered)> {
    [(FireClass::Decode, 1usize), (FireClass::Prefill, 16)]
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
fn every_symbol_every_text_states_has_a_row_that_states_its_file_and_rule() {
    // Three questions with one answer shape, so one walk asks all three: a
    // symbol with no row has no contract, a row with no file cannot be
    // compiled at run time, and a row with no rule cannot be given a grid.
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                match kernels::sig_in(kernels_metal::KERNELS, symbol) {
                    None => faults.push(format!("{}/{class:?}: `{symbol}` has no row", text.name)),
                    Some(sig) if sig.file.is_none() => {
                        faults.push(format!(
                            "{}/{class:?}: `{symbol}` states no file",
                            text.name
                        ));
                    }
                    Some(sig) if sig.launch == kernels::LaunchRule::Unstated => {
                        faults.push(format!(
                            "{}/{class:?}: `{symbol}` states no rule",
                            text.name
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_symbol_is_an_instantiated_point_and_not_a_bare_stem() {
    // The check that only exists because running found it. A row carries
    // AXES, so `sig_in` resolves a stem — `embed_gather_4bit` matches its own
    // row — and the table is satisfied while no shader exports that name.
    //
    // The test is the row's own product: a symbol must be one of the
    // entrypoints its axes generate. A row with no axes generates exactly its
    // own symbol, so an unparameterised kernel passes trivially, which is
    // right.
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue; // the check above owns this
                };
                let points = sig.entrypoints();
                if !points.iter().any(|p| p == symbol) {
                    faults.push(format!(
                        "{}/{class:?}: `{symbol}` is a STEM, not an entry point. \
                         Its row instantiates {:?}. A stem resolves here and in no \
                         shader — spell the point from the deployment's facts.",
                        text.name,
                        points.iter().take(4).collect::<Vec<_>>()
                    ));
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
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
                match plan_one(
                    &low,
                    launch,
                    kernels_metal::KERNELS,
                    frame,
                    text.geometry,
                    &mut Anything,
                ) {
                    Ok(d) => {
                        let threads: u64 = d.grid.iter().map(|&n| u64::from(n)).product();
                        let group: u64 = d.threadgroup.iter().map(|&n| u64::from(n)).product();
                        if threads == 0 || group == 0 || group > 1024 {
                            faults.push(format!(
                                "{}/{class:?}: `{}` wants grid {:?} in groups of {:?}",
                                text.name, d.symbol, d.grid, d.threadgroup
                            ));
                        }
                    }
                    Err(Undispatchable::NoRow { .. } | Undispatchable::NoFile { .. }) => {}
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

#[test]
fn the_harness_covers_every_family_that_has_a_text() {
    // The check that keeps the harness honest. A family whose text lands and
    // is not added to `texts()` gets none of the above, and the failure would
    // be silence — which is the one failure mode a conformance suite cannot
    // afford.
    //
    // Counted rather than named: the list is short and its growth is the whole
    // remaining plan (`.wiki/new-driver/metal.md` task 5).
    //
    // FIVE entries over ONE text, and the gap is the interesting part: every
    // one of them joined by naming a fixture rather than by being a family,
    // so a routed FFN, attention sinks, a geglu with a softcap and a q/k/v
    // bias all reach the device with no second text and no per-family branch
    // anywhere in the executor.
    assert_eq!(
        texts().len(),
        5,
        "a Metal text or fixture landed or left. Add or remove its row in \
         `texts()` — everything above is per-text and a shape not listed is a \
         shape not checked."
    );
}

/// How many buffers a shader's entry point declares.
///
/// # Why this is parsed rather than declared
///
/// `KernelSig` has an `operands` field and the CUDA table uses it, but **no
/// Metal row declares one**: the C++ shell bound by hand from tables that are
/// retiring, so nothing ever needed the arity written down. Until the rows
/// carry it, the shader is the only statement of how many buffers a kernel
/// takes — so this reads the shader.
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
fn param_list(root: &std::path::Path, file: &str, stem: &str) -> Option<String> {
    let src = std::fs::read_to_string(root.join(file)).ok()?;
    let at = src.find(&format!("void {stem}("))?;
    let open = src[at..].find('(')? + at;
    // Depth-counted, because a parameter list is full of parentheses:
    // `[[buffer(0)]]` closes one the signature did not open, and stopping at
    // the first `)` finds a list of one operand for every kernel there is.
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in src[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    Some(src[open..close?].to_string())
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

/// **The operand order, and the rows that have not stated one.**
///
/// `model::executor` used to bind "operands in the trace's stated order" —
/// inputs, then outputs, then weights, at buffers `0..n`. That is the
/// COMPILER's convention and it is not the kernels'. `affine_qmv_fast`
/// declares `w, scales, biases, x, y`: weights first. So the activation bound
/// where the packed weight belongs, and every operand after it was one slot
/// further wrong — on all nine of `llama_like`'s statements, which is every
/// one whose shader could be found.
///
/// The fix is the field the CUDA table has always filled and no Metal row did:
/// [`KernelSig::operands`], each carrying a [`Source`] that says where its
/// value comes from. `dispatch::reorder` binds BY that when a row states it.
///
/// So the number to shrink is **rows the text names that state no operands**.
/// Each is a launch still bound positionally, which is right by accident or
/// not at all.
///
/// And for the rows that DO state them, this checks the statement against the
/// shader: same buffer count, and the writable buffer in the same place. A row
/// that describes a kernel it does not match is worse than one that describes
/// nothing, because the executor believes it.
#[test]
fn a_row_that_states_its_operands_agrees_with_its_shader() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels");

    let mut unstated: Vec<String> = Vec::new();
    let mut disagrees: Vec<String> = Vec::new();

    for text in texts() {
        for (_, low) in fires(&text) {
            let mut seen = BTreeSet::new();
            for symbol in &low.kernels {
                if !seen.insert(symbol.clone()) {
                    continue;
                }
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue;
                };
                let Some(file) = sig.file else { continue };
                if sig.operands.is_empty() {
                    unstated.push(format!("  {symbol}"));
                    continue;
                }
                // `Ty::InPacked` operands are FIELDS of a preceding packed
                // struct, not buffers of their own, so they are named by the
                // row and absent from the shader's parameter list. Counting
                // them here would make a row that spells a struct's contents
                // look like a row that miscounted its kernel.
                let slots = sig
                    .operands
                    .iter()
                    .filter(|o| o.ty != kernels::Ty::InPacked)
                    .count();
                if let Some(buffers) = declared_buffers(&root, file, sig.symbol)
                    && buffers != slots
                {
                    disagrees.push(format!(
                        "  {symbol}: row states {slots} buffer operands, shader \
                         declares {buffers}"
                    ));
                }
                if let Some(writes) = first_writable(&root, file, sig.symbol) {
                    // A writable buffer is one the row declares `BufMut`, and
                    // the `Ty` is the precise signal where the `Source` is
                    // not: the KV writes have NO result — their whole effect
                    // is on the cache, so `KvKeys` is writable there — while
                    // attention READS the same store, so `KvKeys` is
                    // read-only there. Two passes over this got it wrong in
                    // both directions before asking the type.
                    let row_writes = sig
                        .operands
                        .iter()
                        .position(|o| matches!(o.ty, kernels::Ty::BufMut));
                    if row_writes != Some(writes) {
                        disagrees.push(format!(
                            "  {symbol}: shader writes buffer {writes}, row puts its \
                             output at {row_writes:?}"
                        ));
                    }
                }
            }
        }
    }
    unstated.sort();
    unstated.dedup();
    disagrees.sort();
    disagrees.dedup();

    assert!(
        disagrees.is_empty(),
        "a row describes a kernel it does not match, and the executor believes \
         it:\n{}",
        disagrees.join("\n")
    );

    eprintln!(
        "{} symbol(s) still bound positionally:\n{}",
        unstated.len(),
        unstated.join("\n")
    );
    // ZERO, from fourteen. **Every symbol `llama_like` names states its
    // operands**, so no launch is bound positionally any more and the
    // trace-order/kernel-order mismatch is closed.
    //
    // It may not grow: a new symbol arrives with no row operands, and this is
    // what says so before it reaches a GPU that will not.
    //
    // What ZERO does NOT mean, and the distinction is the whole remaining
    // gap: a row states where a value goes, and the TEXT still has to state
    // the value. Three holes are visible in the rows themselves, written
    // there as `Unbound` because a positional row cannot omit a slot:
    //
    // * the gathers want the token IDS — a fire value `Source` has no name
    //   for, and `Arg::Named` is the channel the text would state it on;
    // * the paged attention wants six of the fire's TABLES — which request
    //   owns each token, the page CSR, the mask and its stride;
    // * `dsl::metal::rope` states ONE launch carrying q and k, and the kernel
    //   rotates ONE buffer in place. The statement should be two, and until it
    //   is, the second tensor is not rotated at all.
    //
    // The first two are the text stating what it carries. The third is a
    // defect the rows made visible: nothing before this could see that a
    // statement's shape disagreed with its kernel's.
    assert!(
        unstated.is_empty(),
        "{} row(s) state no operands and are bound POSITIONALLY, and the \
         trace's order is not the kernel's:\n{}",
        unstated.len(),
        unstated.join("\n")
    );
}

/// **How many slots the statements actually fill.**
///
/// The rows say where every buffer goes; that is `a_row_that_states_its_
/// operands_agrees_with_its_shader`, and it is at zero. This asks the other
/// half: does the STATEMENT supply a value for each slot the row names?
///
/// A row's `Unbound` operand is a slot nobody fills, and a slot nobody fills
/// is read anyway — on this backend, whatever the last dispatch left there. So
/// this counts them, and the count is the last measurable distance between the
/// executor and a fire worth checking against a checkpoint.
/// The slots a row leaves unbound on purpose, with the argument for each.
///
/// `(symbol, buffer, operand)`. A row is POSITIONAL, so a slot the kernel
/// declares and nothing fills has to be listed rather than dropped — dropping
/// it shifts every operand after it. What this table adds is WHY, per slot,
/// and the test holds the set exactly: a new hole fails, and so does an
/// argument whose slot has since been filled.
const DELIBERATE: &[(&str, usize, &str)] = &[
    // SEVEN buffers of a shared ring ABI `kv_append_paged` declares and does
    // not read. Nothing fills them because nothing should.
    ("kv_append_paged_bfloat16", 4, "ring_4"),
    ("kv_append_paged_bfloat16", 6, "ring_6"),
    ("kv_append_paged_bfloat16", 7, "ring_7"),
    ("kv_append_paged_bfloat16", 8, "ring_8"),
    ("kv_append_paged_bfloat16", 9, "ring_9"),
    ("kv_append_paged_bfloat16", 11, "ring_11"),
    ("kv_append_paged_bfloat16", 15, "ring_15"),
    // A slot the OTHER instantiation of the same kernel fills. `sinks` is
    // `LlamaLikeMetalFacts::attn_sinks`'s, and no text in `texts()` sets it;
    // `bias` is `affine_qmv_routed_bias`'s; `per_expert_scale` is
    // `router_topk_scaled`'s.
    ("sdpa_paged_decode_bfloat16_d_128", 16, "sinks"),
    ("affine_qmv_routed_bfloat16_gs_64_b_4", 7, "bias"),
    ("router_topk_bfloat16", 4, "per_expert_scale"),
    // The one hole that is a property of a CODEC rather than of a sibling
    // instantiation. `biases` is the affine zero-point plane, and MXFP4 has
    // none: its scales are E8M0 block exponents with nothing to subtract.
    // The kernel takes the pointer and ignores it, which is why the slot may
    // be empty — and why it may not be SOURCED. It was, from `Weight(2)`,
    // copied index-for-index off the affine row, and that pushed the additive
    // bias this symbol does read to a `Weight(3)` the codec's weight list
    // never reaches. See `model_dispatch::the_mxfp4_expert_bank_reads_a_bias_
    // and_is_handed_one`.
    ("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", 2, "biases"),
];

#[test]
fn every_slot_a_row_names_is_a_slot_a_statement_fills() {
    let mut holes: Vec<String> = Vec::new();
    for text in texts() {
        for (_, low) in fires(&text) {
            let mut seen = BTreeSet::new();
            for symbol in &low.kernels {
                if !seen.insert(symbol.clone()) {
                    continue;
                }
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue;
                };
                for (slot, o) in sig.operands.iter().enumerate() {
                    if matches!(o.source, kernels::Source::Unbound) {
                        holes.push(format!("  {symbol}: buffer {slot} (`{}`)", o.name));
                    }
                }
            }
        }
    }
    holes.sort();
    holes.dedup();

    eprintln!(
        "{} slot(s) no statement fills:\n{}",
        holes.len(),
        holes.join("\n")
    );
    // Every one is NAMED rather than counted.
    //
    // This used to assert `holes.len() <= 10` against a paragraph listing
    // which ten. A count is the weaker claim by exactly the amount that
    // matters: it passes when a deliberate hole is filled and an accidental
    // one opens in the same run, and it says nothing about WHICH slot the
    // eleventh is. The paragraph was already doing the real work; this makes
    // it the assertion.
    let deliberate: BTreeSet<String> = DELIBERATE
        .iter()
        .map(|(symbol, slot, name)| format!("  {symbol}: buffer {slot} (`{name}`)"))
        .collect();
    let found: BTreeSet<String> = holes.iter().cloned().collect();
    let opened: Vec<&String> = found.difference(&deliberate).collect();
    let closed: Vec<&String> = deliberate.difference(&found).collect();
    assert!(
        opened.is_empty(),
        "{} slot(s) no statement fills and no argument covers. A slot nobody \
         fills is read anyway.\n{}",
        opened.len(),
        opened
            .iter()
            .map(|h| h.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        closed.is_empty(),
        "{} slot(s) are argued for below and no row leaves them unbound. An \
         excuse outliving its subject is how the next one gets believed.\n{}",
        closed.len(),
        closed
            .iter()
            .map(|h| h.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// **Do the statements carry the scalars their rows name?**
///
/// A row says `in_vec_size: I32 <- Param(0)`; the statement has to state a
/// scalar for slot 0 or the kernel reads a buffer nobody wrote. This is the
/// operand question one level down, and the same answer applies: a slot nobody
/// fills is read anyway.
///
/// It is separate from the operand check because it fails differently. A
/// missing OPERAND is a wrong pointer; a missing SCALAR is a wrong extent, and
/// a kernel told its output is zero wide computes nothing and reports success.
#[test]
fn every_scalar_a_row_names_is_a_scalar_the_statement_states() {
    let mut short: Vec<String> = Vec::new();
    for text in texts() {
        for (_, low) in fires(&text) {
            let mut seen = BTreeSet::new();
            for launch in &low.launches {
                let symbol = &low.kernels[launch.kernel as usize];
                if !seen.insert(symbol.clone()) {
                    continue;
                }
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue;
                };
                let wants = sig
                    .operands
                    .iter()
                    .filter_map(|o| match o.source {
                        kernels::Source::Param(i) | kernels::Source::ParamF32(i) => {
                            Some(usize::from(i) + 1)
                        }
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                let states = (launch.params.end - launch.params.start) as usize;
                if states < wants {
                    short.push(format!(
                        "  {symbol}: row names {wants} scalar(s), statement states {states}"
                    ));
                }
            }
        }
    }
    short.sort();
    short.dedup();

    eprintln!(
        "{} statement(s) state fewer scalars than their row names:\n{}",
        short.len(),
        short.join("\n")
    );
    // ZERO, down from THIRTEEN — which was every statement but the QKV split,
    // the only one that had ever stated a scalar. Every projection was told
    // its extents were zero and computed nothing; every rope was told its head
    // width was zero; the attention was told its strides were zero and read
    // one key forever.
    //
    // The rows made it askable. Before they named their `Param` slots nothing
    // stated how many scalars a kernel wants, so nothing could notice that
    // none were supplied.
    //
    // The last four all wanted the SAME thing and it was not the text's to
    // give: the KV pool's strides and its page size — the shape the DRIVER
    // allocated (`model::kv::Shape`), which no model text can know. They are
    // `Source::KvHeadStride`/`KvSeqStride`/`KvPageSize` now, answered by the
    // resolver beside the pages themselves and APPENDED to the statement's own
    // scalars by `dispatch::param_layout`.
    //
    // Writing them down found the layout was the other way around from the
    // names: the pool is `[page, token, head, dim]`, so one head is `head_dim`
    // away and one TOKEN is a whole interleaved row away. Swapping the two is
    // a fire that reads real memory and attends to the wrong tokens.
    assert_eq!(
        short.len(),
        0,
        "the gap REOPENED at {}. A kernel told its output is zero wide computes \
         nothing and reports success.\n{}",
        short.len(),
        short.join("\n")
    );
}

/// Every text states where its answer is.
///
/// A fire that runs and computes a distribution nobody can find is a fire that
/// did nothing, and until the `out` seam was stated that was literally the
/// engine seam's behaviour: it encoded every launch, waited, and dropped the
/// arena. The two places that DID read logits guessed, in two dialects — the
/// reference gate took the widest arena region and the sampler had no path at
/// all.
///
/// So this pins the derived answer against the guess. They must agree: if the
/// widest region is not the exit seam's value, then either the text states the
/// wrong exit or the arena holds something wider than a vocabulary, and both
/// are worth failing over.
#[test]
fn every_text_says_where_its_answer_lands_and_it_is_the_widest_thing_there() {
    for text in texts() {
        for (class, low) in fires(&text) {
            let r = low.readout.unwrap_or_else(|| {
                panic!(
                    "{}: {class:?} states no exit seam, so a driver running it \
                     has no way to find the distribution it computed",
                    text.name
                )
            });
            // The guess the reference gate makes, computed the same way.
            let widest = low
                .args
                .iter()
                .filter_map(|a| match a {
                    model_compiler::lower::Arg::Arena { at, width, bytes } => {
                        Some((*at, *width as usize * *bytes as usize))
                    }
                    _ => None,
                })
                .max_by_key(|(_, span)| *span)
                .expect("a fire binds at least one arena region");
            assert_eq!(
                r.at, widest.0,
                "{}: {class:?} states its exit at {} but the widest arena \
                 region is at {} ({} bytes). A vocabulary is the widest thing \
                 a fire holds, so a disagreement means one of the two is not \
                 the read-out.",
                text.name, r.at, widest.0, widest.1
            );
            assert_eq!(
                r.rows, low.n_requests,
                "{}: {class:?} reads out {} rows over {} requests. A fire \
                 samples one distribution per REQUEST, so anything else is \
                 the row axis coming from the wrong place -- tokens, most \
                 likely, which is the bug that made a two-token prefill \
                 answer the first token's question.",
                text.name, r.rows, low.n_requests
            );
            assert_eq!(
                r.bytes, 2,
                "{}: {class:?} declares a {}-byte read-out. The metal readout \
                 is bf16 because `affine_qmv_fast` writes bf16; a text that \
                 says four gets a vocabulary read as half zeros.",
                text.name, r.bytes
            );
            assert!(
                r.vocab > 1000,
                "{}: {class:?} reads out {} elements, which is not a \
                 vocabulary",
                text.name,
                r.vocab
            );
        }
    }
}
