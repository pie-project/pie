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

use driver_metal_new::model::dispatch::{Geometry, Undispatchable, plan_one};
use driver_metal_new::model::executor::{Frame, Resolver, Slice};
use driver_metal_new::model::resolve::{Names, Store};
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
    vec![Text {
        name: "llama_like",
        plan: |class| {
            use model::families::llama_like::forward::facts::{
                LlamaLikeFacts, LlamaLikeMetalFacts,
            };
            model::families::llama_like::forward::llama_like_metal(
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
    }]
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
    [(FireClass::Decode, 1usize), (FireClass::Prefill, 8)]
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
                        faults.push(format!("{}/{class:?}: `{symbol}` states no file", text.name));
                    }
                    Some(sig) if sig.launch == kernels::LaunchRule::Unstated => {
                        faults.push(format!("{}/{class:?}: `{symbol}` states no rule", text.name));
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
    // Counted rather than named: the list of Metal texts is short and its
    // growth is the whole remaining plan (`.wiki/new-driver/metal.md` task 5).
    assert_eq!(
        texts().len(),
        1,
        "a Metal text landed or left. Add or remove its row in `texts()` — \
         everything above is per-text and a family not listed is a family not \
         checked."
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
    (!seen.is_empty()).then_some(seen.len())
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
                if let Some(buffers) = declared_buffers(&root, file, sig.symbol)
                    && buffers != sig.operands.len()
                {
                    disagrees.push(format!(
                        "  {symbol}: row states {} operands, shader declares {buffers} buffers",
                        sig.operands.len()
                    ));
                }
                if let Some(writes) = first_writable(&root, file, sig.symbol) {
                    let row_writes = sig
                        .operands
                        .iter()
                        .position(|o| matches!(o.source, kernels::Source::Out(_)));
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
    // EIGHT, measured 2026-08-10, down from fourteen. Seven rows state their
    // operands now — `rms_single_row`, `silu_mul`, `split_qkv_bf16` and the
    // four projections, which were the loud case: `affine_qmv_fast`
    // declares its weights FIRST and the trace states them last, so every
    // projection of every layer bound its activation where the packed weight
    // belongs.
    //
    // **Every remaining entry is a launch whose operands are at the wrong
    // buffers**, and this may only shrink. What each still needs, so the next
    // reader has a map rather than a count:
    //
    // | symbol | what is missing |
    // |---|---|
    // | `embed_gather_4bit`, `..._mb_4bit` | the token IDS. A fire value the text does not state; `Arg::Named` is the channel |
    // | `neox_decode`, `neox_mb` | the POSITIONS. `Source::Positions` already exists for exactly this |
    // | `kv_append`, `kv_append_paged` | the layer's K and V pages, plus `pos` and three strides. **`Source` has no variant for a state store** — the op carries `StateRef { KvCache, layer }` and the pool holds the pages, so this needs one |
    // | `sdpa_vector_decode`, `sdpa_paged_decode` | the same K/V pages, six strides, a scale and a window. Blocked on the same variant |
    //
    // So the backlog is TWO pieces of work and not eight: teach the text to
    // state its fire values (ids, positions — `Source::Positions` is already
    // there), and give `Source` a way to name the KV store. The second is the
    // only new vocabulary, and it is what the attention pair and the KV writes
    // both wait on.
    assert!(
        unstated.len() <= 8,
        "the backlog GREW to {}. A row with no operands is bound positionally, \
         and the trace's order is not the kernel's.\n{}",
        unstated.len(),
        unstated.join("\n")
    );
}
