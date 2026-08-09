//! The binder over a REAL Metal lowering.
//!
//! Not a synthetic launch list: `llama_like`'s Metal text — the only one that
//! exists today — traced for both fire classes, lowered over plain rows, and
//! every launch bound through `model::executor::bind`. GPU-free, so it runs
//! wherever the workspace does.
//!
//! What it proves, and each is a precondition for the dispatch half:
//!
//! * every arena offset the lowering assigns is inside the arena it sized, so
//!   `arena_bytes` and the offsets agree with each other;
//! * every weight and named value the trace states reaches the resolver — the
//!   map is the only per-family piece left, as designed;
//! * every kernel symbol the lowering emits has a **stated row** in
//!   `kernels_metal::KERNELS`, so an entry point exists for the dispatch half
//!   to compile by name. That is the claim the whole approach rests on: on
//!   this backend a symbol is a name, so a lowering that states one the table
//!   knows needs no arm written to receive it.
//!
//! The third check is the one that can fail as texts grow, and it is the
//! useful failure: `kernels-metal` carries 98 `kernel!` rows against
//! `kernels-cuda`'s 226, so a text naming a symbol with no row is the expected
//! way to discover the next row to write.

use std::collections::BTreeSet;

use driver_metal::lowering::executor::{BindRefusal, Frame, Resolver, Slice, bind};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

/// Answers every name with a distinct region and records what was asked.
///
/// The extents are deliberately generous: this test is about whether the names
/// resolve and the arena offsets are sound, not about the store's sizes.
#[derive(Default)]
struct Sentinels {
    weights: BTreeSet<String>,
    named: BTreeSet<ValueId>,
}

impl Resolver for Sentinels {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        self.weights.insert(name.to_string());
        Some(Slice {
            address: 0x1000_0000,
            bytes: 1 << 30,
        })
    }
    fn named(&mut self, value: ValueId) -> Option<Slice> {
        self.named.insert(value);
        Some(Slice {
            address: 0x2000_0000,
            bytes: 1 << 30,
        })
    }
}

fn lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        class,
    );
    let rows: Vec<Row> = vec![
        Row {
            samples: true,
            ..Row::default()
        };
        rows
    ];
    lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the metal text lowers")
}

/// The arena the lowering sized, at a recognisable base.
fn frame(lowered: &Lowered) -> Frame {
    Frame {
        arena: Slice {
            address: 0x8000_0000,
            bytes: lowered.arena_bytes as u64,
        },
    }
}

#[test]
fn every_launch_of_the_metal_text_binds_in_both_fire_classes() {
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        let low = lowered(class, rows);
        assert!(
            !low.launches.is_empty(),
            "{class:?} lowered to no launches at all"
        );
        let frame = frame(&low);
        let mut store = Sentinels::default();
        for launch in &low.launches {
            match bind(&low, launch, frame, &mut store) {
                Ok(bound) => assert_eq!(
                    bound.kernel, low.kernels[launch.kernel as usize],
                    "the bound symbol is the one the lowering named"
                ),
                Err(refusal) => panic!(
                    "{class:?}: launch of `{}` (op {}) refused: {refusal:?}",
                    low.kernels[launch.kernel as usize], launch.op
                ),
            }
        }
    }
}

#[test]
fn no_arena_operand_addresses_past_the_arena_the_lowering_sized() {
    // The lowering assigns offsets and reports `arena_bytes`; nothing else
    // checks that the two agree. A frame sized exactly to its own report is
    // the strictest version of that question.
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        let low = lowered(class, rows);
        let frame = frame(&low);
        let mut store = Sentinels::default();
        for launch in &low.launches {
            if let Err(BindRefusal::ArenaOutOfBounds {
                at, arena_bytes, ..
            }) = bind(&low, launch, frame, &mut store)
            {
                panic!(
                    "{class:?}: `{}` addresses arena byte {at} of {arena_bytes}",
                    low.kernels[launch.kernel as usize]
                );
            }
        }
    }
}

#[test]
fn every_symbol_the_lowering_names_has_a_row_in_the_metal_table() {
    // The claim the approach rests on. A symbol is a NAME on this backend, so
    // the dispatch half compiles the entry point the lowering states — but
    // only if the table knows the contract, because the row is where the
    // contract lives.
    // `sig_in` is the crate's own answer: an exact symbol match, or the row
    // whose axes cover an instantiated point (`silu_mul` covers
    // `silu_mul_bfloat16`). Comparing against `KernelSig::name` instead would
    // be comparing a dsl-side name to a recorded symbol.
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        for symbol in &lowered(class, rows).kernels {
            if kernels::sig_in(kernels_metal::KERNELS, symbol).is_none() {
                missing.insert(symbol.clone());
            }
        }
    }
    // The known gap CLOSED. `attn::split_qkv_bf16` was the one symbol the
    // text named that no `kernel!` row declared, and the fix was not the
    // shader it looked like: the kernel needs `q_width` as a dispatch
    // constant, and nothing could pass it one. `OpKind::Launch::params` is
    // the channel the trace already carried for exactly that, so the text now
    // states the launch and its two widths outright (`dsl::metal::split_qkv`)
    // and the driver forwards them knowing nothing about what they mean.
    //
    // CUDA still carries its split as `driver_internal` — a kernel the driver
    // launches that no text has to name. Metal has no such category and this
    // is why it needs none.
    assert!(
        missing.is_empty(),
        "the metal text names {} symbol(s) with no `kernel!` row: {missing:?}\n\
         A row is where the contract lives; add it in the module beside the .metal.",
        missing.len()
    );
}

#[test]
fn every_symbol_the_lowering_names_states_the_file_that_defines_it() {
    // Metal compiles at run time from `(path, entry name)`, so a symbol whose
    // row does not say which file defines it cannot be dispatched — and the
    // only things that knew the file were the per-family plans that retire.
    // Demand-driven: a row gets its `file` when a text names it, and this is
    // what asks.
    let mut unstated: BTreeSet<String> = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        for symbol in &lowered(class, rows).kernels {
            match kernels::sig_in(kernels_metal::KERNELS, symbol) {
                Some(sig) if sig.file.is_some() => {}
                // The known gap has no row at all; `every_symbol_..._row`
                // owns it.
                None => {}
                Some(_) => {
                    unstated.insert(symbol.clone());
                }
            }
        }
    }
    assert!(
        unstated.is_empty(),
        "{} symbol(s) the metal text names have a row that states no file: {unstated:?}\n\
         Add `file = Some(\"dir/file.metal\")` to the row.",
        unstated.len()
    );
}

#[test]
fn the_text_states_weights_by_concrete_layer_rather_than_by_template() {
    // The trace is layer-unrolled, so a weight name is `layer.3.q_proj` and
    // never a template the driver would have to expand. If that stopped
    // holding, the resolver's map would need to become a parser.
    let low = lowered(FireClass::Decode, 1);
    let frame = frame(&low);
    let mut store = Sentinels::default();
    for launch in &low.launches {
        bind(&low, launch, frame, &mut store).expect("binds");
    }
    assert!(
        !store.weights.is_empty(),
        "the text names no weights at all"
    );
    for name in &store.weights {
        assert!(
            !name.contains('{') && !name.contains('*'),
            "`{name}` looks like a template, not a concrete name"
        );
    }
}
