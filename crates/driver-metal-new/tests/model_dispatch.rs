//! The dispatch half over a REAL Metal lowering.
//!
//! `tests/model_bind.rs` proved the operands resolve. This proves the other
//! half: that every launch the lowering states becomes a **grid** — a symbol
//! whose row states its file, its rule, and a rule that evaluates at the
//! rectangle's own dims.
//!
//! What it is really measuring is the size of the executor. The walk under
//! test is `dispatch::plan`, which has no arm for any kernel and no branch on
//! any family, and it dispatches `llama_like`'s whole Metal text. If a text
//! naming a new symbol needed a line here, that would show up as this test
//! failing to compile rather than failing to run — and it does not.

use std::collections::BTreeSet;

use driver_metal_new::model::dispatch::{
    Dispatch, Geometry, Undispatchable, dims_of, pipelines_needed, plan_one,
};
use driver_metal_new::model::executor::{Frame, Resolver, Slice};
use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

/// Answers every name with a generous region: this test is about grids, and
/// `model_bind.rs` already owns whether the names resolve.
#[derive(Default)]
struct Sentinels;

impl Resolver for Sentinels {
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

/// qwen3-0.6b's geometry, which is the checkpoint the smokes use.
fn geometry() -> Geometry {
    Geometry {
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        rotary_dims: 128,
        n_experts: 0,
        experts_per_token: 0,
    }
}

fn lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        class,
    );
    lower(
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
    .expect("the metal text lowers")
}

fn frame(lowered: &Lowered) -> Frame {
    Frame {
        arena: Slice {
            address: 0x8000_0000,
            bytes: lowered.arena_bytes as u64,
        },
    }
}

/// The one symbol with no `kernel!` row. `model_bind.rs` owns the argument;
/// here it is only the launch this walk is allowed to refuse.
const KNOWN_GAP: &str = "attn::split_qkv_bf16";

/// Plan every launch, returning the dispatches and the refusals separately.
fn planned(low: &Lowered) -> (Vec<Dispatch<'_>>, Vec<Undispatchable>) {
    let frame = frame(low);
    let mut store = Sentinels;
    let mut ok = Vec::new();
    let mut refused = Vec::new();
    for launch in &low.launches {
        match plan_one(
            low,
            launch,
            kernels_metal::KERNELS,
            frame,
            geometry(),
            &mut store,
        ) {
            Ok(d) => ok.push(d),
            Err(e) => refused.push(e),
        }
    }
    (ok, refused)
}

#[test]
fn every_launch_whose_symbol_has_a_row_becomes_a_grid() {
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 8)] {
        let low = lowered(class, rows);
        let (dispatches, refused) = planned(&low);

        // Nothing may refuse for any reason other than the recorded gap.
        for why in &refused {
            match why {
                Undispatchable::NoRow { symbol, .. } if symbol == KNOWN_GAP => {}
                other => panic!("{class:?}: a launch refused for a NEW reason: {other:?}"),
            }
        }
        assert!(
            !dispatches.is_empty(),
            "{class:?}: nothing dispatched at all"
        );

        // A grid of no threads runs nothing and reports success, which is the
        // failure this crate exists to make impossible.
        for d in &dispatches {
            let threads: u64 = d.grid.iter().map(|&n| u64::from(n)).product();
            let per_group: u64 = d.threadgroup.iter().map(|&n| u64::from(n)).product();
            assert!(
                threads > 0,
                "{class:?}: `{}` dispatches a grid of no threads: {:?}",
                d.symbol,
                d.grid
            );
            assert!(
                per_group > 0 && per_group <= 1024,
                "{class:?}: `{}` wants {per_group} threads a threadgroup",
                d.symbol
            );
            assert!(
                !d.args.is_empty(),
                "{class:?}: `{}` dispatches with no operands",
                d.symbol
            );
        }
    }
}

#[test]
fn the_only_symbol_the_walk_refuses_is_the_one_with_no_row() {
    // Stated as its own test so that closing the gap fails HERE, loudly,
    // rather than quietly widening what the walk tolerates.
    let mut refusals: BTreeSet<String> = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 8)] {
        for why in planned(&lowered(class, rows)).1 {
            refusals.insert(match why {
                Undispatchable::NoRow { symbol, .. }
                | Undispatchable::NoFile { symbol, .. }
                | Undispatchable::Ungeometric { symbol, .. }
                | Undispatchable::Unbound { symbol, .. } => symbol,
            });
        }
    }
    assert_eq!(
        refusals,
        [KNOWN_GAP.to_string()].into_iter().collect::<BTreeSet<_>>(),
        "the set of symbols this backend cannot dispatch has changed"
    );
}

#[test]
fn a_fire_compiles_each_of_its_symbols_once_however_often_it_names_them() {
    // 24 layers restate the same nine kernels. The dispatch list is long and
    // the compile list is short, and that difference is what makes a cold
    // start bounded by the TEXT rather than by the fire.
    let low = lowered(FireClass::Decode, 1);
    let (dispatches, _) = planned(&low);
    let needed = pipelines_needed(&dispatches);
    assert!(
        needed.len() < dispatches.len() / 4,
        "{} pipelines for {} dispatches — the cache is not deduplicating",
        needed.len(),
        dispatches.len()
    );
    let symbols: BTreeSet<&str> = needed.iter().map(|(_, s)| *s).collect();
    assert_eq!(
        symbols.len(),
        needed.len(),
        "a symbol appears twice in the compile list"
    );
    for (file, symbol) in &needed {
        assert!(
            file.ends_with(".metal"),
            "`{symbol}` states `{file}`, which is not a shader"
        );
    }
}

#[test]
fn a_rectangles_dims_come_from_the_rectangle_and_the_fire_and_nowhere_else() {
    // The driver derives no geometry: rows are the rectangle's, width is the
    // operand's, and the head counts are handed in. A wider fire must move
    // `rows` and nothing else about how a rectangle is read.
    //
    // Note what is NOT asserted: that a symbol has one width. `rms_single_row`
    // serves the attention norm at 1024 and the qk-norm at 2048 in the same
    // fire, because a width is the OPERAND's and not the kernel's — which is
    // the property that makes one rule serve every use of a kernel.
    for (class, rows) in [(FireClass::Decode, 1u32), (FireClass::Prefill, 8)] {
        let low = lowered(class, rows as usize);
        for launch in &low.launches {
            let symbol = low.kernels[launch.kernel as usize].as_str();
            let dims = dims_of(&low, launch, geometry());
            assert_eq!(dims.rows, rows, "`{symbol}` at {rows} rows");
            assert_eq!(dims.q_heads, geometry().q_heads, "the fire states the rest");
            assert!(
                dims.width > 0,
                "`{symbol}` states no widthed operand, so no rule can size it"
            );
        }
    }
}

#[test]
fn the_batched_lane_is_the_row_count_and_not_a_second_vocabulary() {
    // The planning documents recorded "which of the two rule sets a row means"
    // as a question to answer before M>1 could be dispatched. It dissolves:
    // where the lanes differ they are DIFFERENT SYMBOLS, each stating its own
    // row, and the rest is `dims.rows`.
    let decode: BTreeSet<String> = lowered(FireClass::Decode, 1).kernels.into_iter().collect();
    let prefill: BTreeSet<String> = lowered(FireClass::Prefill, 8).kernels.into_iter().collect();
    let only_batched: Vec<&String> = prefill.difference(&decode).collect();
    assert!(
        !only_batched.is_empty(),
        "the two lanes name identical symbol sets, so this claim is untestable here"
    );
    // And every one of them dispatches, which is what says the row carries the
    // lane rather than the driver picking it.
    for (_, refused) in [planned(&lowered(FireClass::Prefill, 8))] {
        for why in refused {
            assert!(
                matches!(&why, Undispatchable::NoRow { symbol, .. } if symbol == KNOWN_GAP),
                "a batched symbol did not dispatch: {why:?}"
            );
        }
    }
}

/// The whole host path, joined: a sealed frame's step becomes rows, the rows
/// become rectangles, and the rectangles become grids.
///
/// This is `DriverBackend::launch`'s body with the device taken out. What is
/// missing after it is the buffers, not the decisions.
mod from_a_frame {
    use super::*;
    use driver_metal_new::model::frame::{Step, fire_class, lower_step, sig};

    fn plan_for(class: FireClass) -> model_compiler::trace::ForwardPlan {
        llama_like_metal(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeMetalFacts::synthetic(),
            class,
        )
    }

    #[test]
    fn a_decode_step_reaches_grids_without_the_driver_deciding_anything() {
        // One token a request: a decode, four lanes.
        let step = Step {
            token_ids: &[11, 22, 33, 44],
            qo_indptr: &[0, 1, 2, 3, 4],
            sampling_indices: &[0, 1, 2, 3],
            ..Step::default()
        };
        assert_eq!(fire_class(&step), FireClass::Decode);

        let low = lower_step(&plan_for(fire_class(&step)), &step).expect("the step lowers");
        let mut store = Sentinels;
        let mut grids = 0;
        for launch in &low.launches {
            match plan_one(
                &low,
                launch,
                kernels_metal::KERNELS,
                frame(&low),
                geometry(),
                &mut store,
            ) {
                Ok(d) => {
                    assert!(d.grid.iter().all(|&n| n > 0));
                    grids += 1;
                }
                Err(Undispatchable::NoRow { symbol, .. }) if symbol == KNOWN_GAP => {}
                Err(other) => panic!("a frame-driven launch refused: {other:?}"),
            }
        }
        assert!(grids > 300, "only {grids} grids came out of a 24-layer fire");
    }

    #[test]
    fn a_region_table_changes_the_rows_and_therefore_the_fire() {
        // The seriation's output IS the row feature points, so a step whose
        // regions differ must lower differently from one whose regions do not.
        // Today the Metal text splits on nothing, so the rectangle COUNT is
        // unchanged — and that is the monomorphism `tests/polymorphism.rs`
        // measures, showing up here from the frame end.
        let plain = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 4],
            ..Step::default()
        };
        let seriated = Step {
            region_row_indptr: &[0, 2, 4],
            region_sig: &[sig::TRUNCATED, 0],
            region_k: &[4, u32::MAX],
            ..plain.clone()
        };
        let plan = plan_for(FireClass::Prefill);
        let a = lower_step(&plan, &plain).expect("lowers");
        let b = lower_step(&plan, &seriated).expect("lowers");
        assert_eq!(
            a.launches.len(),
            b.launches.len(),
            "the text gained a guard — rewrite this and tests/polymorphism.rs \
             to assert WHICH axes split"
        );
    }
}
