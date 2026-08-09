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

use driver_metal::lowering::dispatch::{
    Dispatch, Geometry, Undispatchable, dims_of, pipelines_needed, plan_one,
};
use driver_metal::lowering::executor::{Frame, Resolver, Slice};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
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
    // SIXTEEN rows for a prefill, not eight, and it is a precondition rather than
    // a round number: `qmm_t.metal` has no `M` argument -- its header says the
    // driver only selects it when `M % BM == 0`, so the row count lives in the
    // grid -- and `QMM_BMS` starts at sixteen. `Rule::Qmm` refuses anything else
    // with `Ungeometric::PartialTile`.
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        let low = lowered(class, rows);
        let (dispatches, refused) = planned(&low);

        // NOTHING may refuse. The `split_qkv` gap closed when the text
        // started stating its two widths as launch params, so every symbol
        // this text names now reaches a grid.
        assert!(
            refused.is_empty(),
            "{class:?}: {} launch(es) refused: {refused:?}",
            refused.len()
        );
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
fn there_is_no_symbol_this_backend_cannot_dispatch() {
    // This used to record one: `attn::split_qkv_bf16`, the symbol with no row.
    // It was never a missing shader — it was a kernel needing `q_width` as a
    // dispatch constant with no channel to receive one. The text states it now
    // and the driver forwards it, so the set is empty.
    let mut refusals: BTreeSet<String> = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
        for why in planned(&lowered(class, rows)).1 {
            refusals.insert(match why {
                Undispatchable::NoRow { symbol, .. }
                | Undispatchable::NoFile { symbol, .. }
                | Undispatchable::Ungeometric { symbol, .. }
                | Undispatchable::Unbound { symbol, .. }
                | Undispatchable::Conditional { symbol, .. } => symbol,
            });
        }
    }
    assert!(
        refusals.is_empty(),
        "this backend cannot dispatch: {refusals:?}"
    );
}

#[test]
fn a_statement_that_states_scalars_carries_them_to_its_dispatch() {
    // The QKV split used to be this test's example -- three outputs, each a
    // fraction of the work, and a kernel that cannot find the boundary
    // between them from any operand shape. It is not lowered here any more:
    // `LlamaLikeMetalFacts::synthetic` states `qkv_fused: false`, because no
    // Metal deployment publishes a fused bank, so the text projects three
    // separate matvecs and there is nothing to split.
    //
    // Paged decode attention makes the point harder. Two of its six scalars
    // could not come off a shape even in principle: `params[2]` is a f32
    // BIT-CAST into a u32 slot, and `params[4]` is `u32::MAX` standing for
    // "no sliding window" -- a sentinel, not a measurement. A driver that
    // reconstructed scalars from operand extents would have to invent both.
    let low = lowered(FireClass::Decode, 1);
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let sdpa = planned(&low)
        .0
        .into_iter()
        .find(|d| d.symbol.starts_with("sdpa_paged_decode"))
        .expect("the text states a paged decode attention");

    assert_eq!(sdpa.params.len(), 6, "the statement's six scalars");
    assert_eq!(
        sdpa.params[0],
        facts.q_heads / facts.kv_heads,
        "the GQA group, 16 query heads over 8 KV heads"
    );
    assert_eq!(sdpa.params[1], facts.kv_heads, "the KV head count");

    let scale = f32::from_bits(sdpa.params[2]);
    let want = 1.0 / (facts.head_dim as f32).sqrt();
    assert!(
        (scale - want).abs() < 1e-9,
        "the softmax scale rides through a u32 slot as bits: got {scale}, want {want}"
    );

    assert_eq!(
        sdpa.params[4],
        u32::MAX,
        "no sliding window, said as a sentinel rather than as an absence"
    );
    assert!(sdpa.grid[0] > 0, "the dispatch covers something");

    // One placement per scalar, and the row states where each goes. That
    // is the difference between binding positionally -- where a shader
    // reordering its buffers is a silent miscompile -- and binding where
    // the kernel reads.
    assert_eq!(
        sdpa.param_slots.len(),
        sdpa.params.len(),
        "every stated scalar is placed"
    );
    let slots: BTreeSet<u32> = sdpa.param_slots.iter().map(|p| p.slot as u32).collect();
    assert_eq!(
        slots.len(),
        sdpa.param_slots.len(),
        "two scalars placed in one slot would overwrite: {:?}",
        sdpa.param_slots
    );
    assert!(
        sdpa.param_slots.iter().any(|p| p.slot == 4 && p.at == 0),
        "the first scalar sits at buffer 4, offset 0"
    );

    // It used to be the ONLY statement carrying scalars, and this asserted
    // that — "the channel is not a general escape hatch that grew". It grew,
    // deliberately: the rows named their `Param` slots and every statement but
    // four states them now, because a projection told its extents are zero
    // computes nothing. So the assertion inverts.
    let without: Vec<&str> = planned(&low)
        .0
        .into_iter()
        .filter(|d| d.params.is_empty())
        .map(|d| d.symbol)
        .collect();
    let unexpected: BTreeSet<&str> = without
        .into_iter()
        // `silu_mul` takes none and its row names none — a kernel with no
        // scalars is not a statement missing them.
        .filter(|s| {
            !s.starts_with("kv_append") && !s.starts_with("sdpa_") && *s != "silu_mul_bfloat16"
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "a statement other than the KV writes and the attentions carries no \
         scalars: {unexpected:?}. Those four want the POOL's strides, which \
         the text cannot state."
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
    for (class, rows) in [(FireClass::Decode, 1u32), (FireClass::Prefill, 16)] {
        let low = lowered(class, rows as usize);
        for launch in &low.launches {
            let symbol = low.kernels[launch.kernel as usize].as_str();
            let sig = kernels::sig_in(driver_metal::lowering::dispatch::table(), symbol)
                .expect("every symbol this text states has a row");
            let dims = dims_of(sig, &low, launch, geometry());
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
    let prefill: BTreeSet<String> = lowered(FireClass::Prefill, 16)
        .kernels
        .into_iter()
        .collect();
    let only_batched: Vec<&String> = prefill.difference(&decode).collect();
    assert!(
        !only_batched.is_empty(),
        "the two lanes name identical symbol sets, so this claim is untestable here"
    );
    // And every one of them dispatches, which is what says the row carries the
    // lane rather than the driver picking it.
    let refused = planned(&lowered(FireClass::Prefill, 16)).1;
    assert!(
        refused.is_empty(),
        "a batched symbol did not dispatch: {refused:?}"
    );
}

/// The whole host path, joined: a sealed frame's step becomes rows, the rows
/// become rectangles, and the rectangles become grids.
///
/// This is `DriverBackend::launch`'s body with the device taken out. What is
/// missing after it is the buffers, not the decisions.
mod from_a_frame {
    use super::*;
    use driver_metal::lowering::frame::{Step, fire_class, lower_step, sig};

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
                Err(other) => panic!("a frame-driven launch refused: {other:?}"),
            }
        }
        assert!(
            grids > 300,
            "only {grids} grids came out of a 24-layer fire"
        );
    }

    #[test]
    fn a_region_table_changes_the_rows_and_therefore_the_fire() {
        // The seriation's output IS the row feature points, so a step whose
        // regions differ lowers differently from one whose regions do not.
        // This is where the two tasks meet: the region table supplies the
        // rows, and the text's depth axis is what makes them matter.
        let plain = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 4],
            ..Step::default()
        };
        // Full-depth rows first, truncated last — the order a depth split
        // requires, and the order the scheduler's seriation produces.
        let seriated = Step {
            region_row_indptr: &[0, 2, 4],
            region_sig: &[0, sig::TRUNCATED],
            region_k: &[u32::MAX, 4],
            ..plain.clone()
        };
        let plan = plan_for(FireClass::Prefill);
        let a = lower_step(&plan, &plain).expect("lowers");
        let b = lower_step(&plan, &seriated).expect("a seriated step lowers");

        let work = |low: &model_compiler::lower::Lowered| -> u64 {
            low.launches
                .iter()
                .map(|l| u64::from(l.rows.end - l.rows.start))
                .sum()
        };
        assert!(
            work(&b) < work(&a),
            "the truncated region did no less work than the full one \
             ({} against {}), so the depth axis is not reaching the frame",
            work(&b),
            work(&a)
        );
    }

    #[test]
    fn a_region_table_the_scheduler_did_not_seriate_is_refused() {
        // The truncated region FIRST. A rectangle is a row range, so at layer
        // 4 the alive set would be a suffix and there is no honest way to
        // state that — the lowering says so rather than covering the wrong
        // rows. This is the contract the frame bridge inherits the moment the
        // text states an axis, and it is why the region table's ORDER matters.
        let unseriated = Step {
            token_ids: &[1, 2, 3, 4],
            qo_indptr: &[0, 4],
            region_row_indptr: &[0, 2, 4],
            region_sig: &[sig::TRUNCATED, 0],
            region_k: &[4, u32::MAX],
            ..Step::default()
        };
        assert!(
            lower_step(&plan_for(FireClass::Prefill), &unseriated).is_err(),
            "an unseriated region table lowered anyway"
        );
    }
}

/// The resolver's map, against the whole of what the text asks for.
///
/// `model_bind.rs` proves the names reach *a* resolver. This proves the real
/// one knows all of them — which is the difference between a map that exists
/// and a map that is complete.
mod the_map {
    use std::collections::HashMap;

    use driver_metal::lowering::resolve::{Names, Store};

    use super::*;

    #[test]
    fn every_name_the_text_states_has_a_checkpoint_spelling() {
        let names = Names::mlx();
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let store = Store::new(names, &tensors, &named);

        let mut unknown: BTreeSet<String> = BTreeSet::new();
        for (class, rows) in [(FireClass::Decode, 1), (FireClass::Prefill, 16)] {
            let low = lowered(class, rows);
            for arg in &low.args {
                if let model_compiler::lower::Arg::Weight(name) = arg {
                    // A `scale.` marker is a constant riding the weight slot,
                    // not a tensor; the binder never looks it up.
                    if name.starts_with("scale.") {
                        continue;
                    }
                    if store.checkpoint_name(name).is_none() {
                        unknown.insert(name.clone());
                    }
                }
            }
        }
        assert!(
            unknown.is_empty(),
            "the text states {} name(s) the map cannot spell: {unknown:?}\n\
             Add the role to `Names::mlx`, or the name is drift.",
            unknown.len()
        );
    }

    #[test]
    fn an_affine_projection_asks_for_all_three_of_its_tensors() {
        // The property the `proj_repr` fact buys. A text that left its
        // projections dense would name ONE tensor where `affine_qmv_fast`
        // reads three, and the driver would have had to derive the other two
        // from a naming convention nobody told it.
        let low = lowered(FireClass::Decode, 1);
        let qkv = low
            .launches
            .iter()
            // By PREFIX: the symbol is the instantiated point
            // (`affine_qmv_fast_bfloat16_gs_64_b_4`), because a bare stem does
            // not resolve to any entry point the shader exports.
            .find(|l| low.kernels[l.kernel as usize].starts_with("affine_qmv_fast_bfloat16"))
            .expect("the text states a quantized projection");
        let weights: Vec<&str> = low.args[qkv.args.start as usize..qkv.args.end as usize]
            .iter()
            .filter_map(|a| match a {
                model_compiler::lower::Arg::Weight(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(weights.len(), 3, "packed weight, scales, zero point");
        assert!(weights[1].ends_with(".scales"));
        assert!(weights[2].ends_with(".zeros"));
    }
}

/// A statement's STATE, which is not one of its operands.
///
/// The KV cache outlives the fire, so no traced value stands for it and a
/// statement names it as `StateRef { KvCache, layer }`. Every backend has
/// answered that with a hand-written arm; `Source::KvKeys`/`KvValues` let the
/// ROW ask, and `Resolver::kv` is where the asking lands.
mod state {
    use std::collections::HashMap;

    use driver_metal::lowering::executor::{Resolver, Slice};
    use driver_metal::lowering::resolve::{Names, Store};

    use super::*;

    /// Distinct addresses per (layer, side), so a wrong one names itself.
    fn pages(layer: u16, values: bool) -> Option<Slice> {
        Some(Slice {
            address: 0x7000_0000u64.wrapping_add(u64::from(layer) << 16)
                + if values { 0x8000 } else { 0 },
            bytes: 1 << 20,
        })
    }

    #[test]
    fn a_kv_write_binds_the_pages_of_its_own_layer() {
        let low = lowered(FireClass::Decode, 1);
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let kv = |l: u16, v: bool| pages(l, v);
        let mut store = Store::new(Names::mlx(), &tensors, &named).with_kv(&kv);

        // Answer the weights too, so a refusal is about state and nothing else.
        assert!(store.weight("layer.0.attn_norm").is_none());

        let mut checked = 0;
        for launch in &low.launches {
            if !low.kernels[launch.kernel as usize].starts_with("kv_append") {
                continue;
            }
            let d = plan_one(
                &low,
                launch,
                kernels_metal::KERNELS,
                frame(&low),
                geometry(),
                &mut store,
            )
            .expect("a kv write plans");
            let layer = launch.layers.start;
            // Buffers 2 and 3 are the cache, which the ROW says and this
            // reads back: keys then values, of this statement's own layer.
            assert_eq!(
                d.args[2].slice.address,
                pages(layer, false).unwrap().address,
                "layer {layer}: keys"
            );
            assert_eq!(
                d.args[3].slice.address,
                pages(layer, true).unwrap().address,
                "layer {layer}: values"
            );
            checked += 1;
        }
        assert!(
            checked >= 24,
            "only {checked} kv writes; a 24-layer text has one a layer"
        );
    }

    #[test]
    fn a_rope_binds_the_fires_positions_because_its_row_names_them() {
        // A text cannot state the positions — they are this fire's data, not
        // this model's structure — so the ROW names which table and the
        // resolver answers. A kernel wanting them and a driver that KNEW to
        // bind them is the hand-written arm this crate removes.
        use driver_metal::lowering::executor::FireTable;

        let low = lowered(FireClass::Decode, 1);
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let tables = |t: FireTable| {
            (t == FireTable::Positions).then_some(Slice {
                address: 0x5150_0000,
                bytes: 64,
            })
        };
        let mut store = Store::new(Names::mlx(), &tensors, &named).with_fire(&tables);
        let launch = low
            .launches
            .iter()
            .find(|l| low.kernels[l.kernel as usize].starts_with("neox"))
            .expect("the text rotates");
        let d = plan_one(
            &low,
            launch,
            kernels_metal::KERNELS,
            frame(&low),
            geometry(),
            &mut store,
        )
        .expect("a rotation plans");
        // Buffer 1 is `position`, which the row says and this reads back.
        assert_eq!(d.args[1].slice.address, 0x5150_0000);
        assert_eq!(d.args.len(), 5, "x, position, and three scalars");
    }

    #[test]
    fn a_resolver_with_no_pool_binds_a_region_that_addresses_nothing() {
        // The default. A binder's own tests have no pool and must not need
        // one, and a statement that asks and gets nothing binds a region
        // addressing nothing — the same honest answer a missing scale gets,
        // and not a skipped slot, which would shift every operand after it.
        let low = lowered(FireClass::Decode, 1);
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let mut store = Store::new(Names::mlx(), &tensors, &named);
        let launch = low
            .launches
            .iter()
            .find(|l| low.kernels[l.kernel as usize].starts_with("kv_append"))
            .expect("the text writes KV");
        let d = plan_one(
            &low,
            launch,
            kernels_metal::KERNELS,
            frame(&low),
            geometry(),
            &mut store,
        )
        .expect("it still plans");
        assert_eq!(d.args[2].slice.address, 0);
        assert_eq!(d.args[2].slice.bytes, 0);
        // SIXTEEN, which is `kv_append_paged`'s width. The text names the
        // paged variant for every fire now -- the POOL is paged, so a decode
        // that named the contiguous one would walk it with contiguous
        // arithmetic -- and the paged row is positional over a shared ring ABI
        // it does not read, which is where most of the sixteen go.
        assert_eq!(d.args.len(), 16, "and every other slot is still in place");
    }
}

/// A rule may take its extent from the STATEMENT, not only from the fire.
///
/// `Dims` is filled from the fire's geometry — one `rotary_dims`, one
/// `head_dim`, one of everything — which is right until a deployment states
/// the number per layer. gemma-4 does: `partial_rotary_factor: 0.25` means
/// its full-attention layers rotate 128 of their 512 channels while its
/// sliding layers rotate all 256 of theirs. One fire-wide number cannot be
/// both, and rotating the wrong count returns fluent text rather than
/// failing.
///
/// The rope rows answer it with `grid_param`, naming which of the statement's
/// own scalars carries the extent. This asserts the grid actually moves —
/// a row-level declaration that produced one grid everywhere would be
/// decorative.
#[test]
fn a_row_can_say_its_grid_extent_comes_from_the_statement() {
    let facts = LlamaLikeFacts {
        layers: 12,
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    // gemma-4's shape: one full-attention layer in six, twice as wide per
    // head, rotating a quarter of it.
    let metal = LlamaLikeMetalFacts {
        global_head_dim: 512,
        global_kv_heads: 4,
        full_partial_rotary: 0.25,
        window_left: (0..12)
            .map(|l| if (l + 1) % 6 == 0 { -1 } else { 1024 })
            .collect(),
        ..LlamaLikeMetalFacts::synthetic()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the gemma-shaped text lowers");

    let table = driver_metal::lowering::dispatch::table();
    let geometry = Geometry {
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        // The FIRE's number, which is the sliding layers' and which the full
        // layers must NOT take.
        rotary_dims: 256,
        ..Geometry::default()
    };
    let mut by_layer: BTreeSet<(u16, u32)> = BTreeSet::new();
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        if !symbol.starts_with("neox") {
            continue;
        }
        let sig = kernels::sig_in(table, symbol).expect("a rope row");
        by_layer.insert((
            launch.layers.start,
            dims_of(sig, &low, launch, geometry).rotary_dims,
        ));
    }
    assert!(!by_layer.is_empty(), "the text states rope launches");
    for (layer, rotary) in &by_layer {
        let full = (layer + 1) % 6 == 0;
        assert_eq!(
            *rotary,
            if full { 128 } else { 256 },
            "layer {layer} (full={full}) rotates {rotary} channels"
        );
    }
    // And both answers are present, or the declaration proved nothing.
    let widths: BTreeSet<u32> = by_layer.iter().map(|(_, r)| *r).collect();
    assert_eq!(
        widths,
        BTreeSet::from([128, 256]),
        "one extent everywhere means the statement's scalar was not read"
    );
}

/// The routed MXFP4 leg binds the bias its ONE symbol reads.
///
/// `dsl::metal::routed_qmv` has a single instantiation for an MXFP4 bank —
/// `mxfp4_qmv_routed_bias`, group 32, 4 bits — and it is a BIASED one: the
/// template is `qmv_routed_bias`, `BIASED` is on, and the kernel adds
/// `bias_row[out_row + row]` from `buffer(7)` to every output it writes.
///
/// That buffer had nothing in it. The row named `Weight(3)` for it, after
/// giving `Weight(2)` to the codec's zero-point plane, and MXFP4 has no
/// zero-point plane: `MatW::scale_names` yields `.scales` alone, so the
/// statement's weight list was two names long and neither index existed.
/// `reorder` answered both with an address of zero and the launch went ahead.
///
/// Nothing had ever run it. No catalog row states `moe_mxfp4`, so the only
/// checkpoint whose experts are MXFP4 — gpt-oss — refuses on Metal before it
/// reaches a text, and the one symbol that always reads a bias is the one
/// symbol nobody had fired.
///
/// # What this holds, and how each half fails
///
/// The slot the row leaves unbound must address nothing and the slot it
/// sources must address something. Re-sourcing `biases` pushes `bias` back to
/// an index the list never reaches, which is now `BindRefusal::UnstatedWeight`
/// and shows up here as a refusal; dropping `routed_qmv`'s `.bias` name does
/// the same. Sourcing the unread slot from somewhere shows up as the first
/// assertion.
#[test]
fn the_mxfp4_expert_bank_reads_a_bias_and_is_handed_one() {
    let plan = llama_like_metal(
        &LlamaLikeFacts::gpt_oss_20b(),
        &LlamaLikeMetalFacts::gpt_oss_20b(),
        FireClass::Decode,
    );
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the routed MXFP4 text lowers");

    let facts = LlamaLikeFacts::gpt_oss_20b();
    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };
    let frame = frame(&low);
    let table = driver_metal::lowering::dispatch::table();

    // The symbol carries its instantiation point, which is the OTHER half of
    // the same omission: the arm returned a bare `mxfp4_qmv_routed_bias` and
    // `quant/qmv.metal` exports only the pointed name, so the fire died at
    // pipeline construction with "exports no such entry point".
    let mut seen = 0usize;
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        if symbol != "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4" {
            continue;
        }
        seen += 1;
        let sig = kernels::sig_in(table, symbol).expect("the MXFP4 routed row");
        let mut store = Sentinels;
        let d = plan_one(&low, launch, table, frame, geometry, &mut store)
            .expect("the routed MXFP4 leg plans");
        for (slot, operand) in sig.operands.iter().enumerate() {
            let bound = d.args[slot].slice;
            // `biases` is in the ABI because a row is positional, and unread
            // because the codec has no such plane. `bias` is read per output
            // row. Naming them off the row rather than by index, so that a
            // row reordered upstream is still being asked the right question.
            match operand.name {
                "biases" => assert_eq!(
                    bound.bytes, 0,
                    "the zero-point slot MXFP4 does not have addresses {bound:?}"
                ),
                "bias" => assert!(
                    bound.bytes > 0,
                    "the additive bias the kernel reads addresses nothing"
                ),
                _ => {}
            }
        }
    }
    // Three legs per layer, twenty-four layers: a decode that dispatched none
    // of them would pass every assertion above.
    assert_eq!(seen, 3 * 24, "the routed MXFP4 legs the text states");
}

/// The bias add gets a `width` slot, and the width is the projection's.
///
/// # Why a plan-level test and not only a device one
///
/// `norm::add_bias` states its row pitch as `Source::OutWidth(0)` — the row
/// DERIVES the number from the operand it biases rather than making a text
/// repeat it. `param_layout`'s source match had no arm for that and ended in
/// `_ => continue`, so the slot was never emitted at all: the kernel's
/// `const constant int& width [[buffer(2)]]` would have read whatever the
/// encoder last left at index 2.
///
/// That is a silent failure at every level above it. The plan is well-formed,
/// the pipeline compiles, the fire runs, and the arithmetic is wrong — which
/// is why `LlamaLikeMetalFacts::add_bias` defaulted itself off and said so in
/// prose for as long as the arm was missing, and why seven Qwen-2.5 rows have
/// been served on Metal without their q/k/v projection biases.
///
/// The assertion is on the WIDTHS and not merely on the slot's existence: an
/// arm that emitted a slot holding zero would satisfy "there is a slot" and
/// still multiply every row index by nothing.
#[test]
fn the_bias_add_is_handed_the_width_it_derives() {
    let facts = LlamaLikeFacts {
        qkv_bias: true,
        o_bias: false,
        router_bias: false,
        // qwen-2 has no q/k norm. Left on, a norm would sit between the
        // projection and the bias and this would prove nothing about either.
        qk_norm: model::shared::llama_like::spec::QkNorm::Off,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    let metal = LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the metal text lowers");
    let frame = frame(&low);
    let mut store = Sentinels;

    let g = geometry();
    // What the three projections are worth per row, which is what the three
    // biases must each be handed.
    let widths = [
        g.q_heads * g.head_dim,
        g.kv_heads * g.head_dim,
        g.kv_heads * g.head_dim,
    ];
    let mut seen = Vec::new();
    for launch in &low.launches {
        let d = plan_one(&low, launch, kernels_metal::KERNELS, frame, g, &mut store)
            .expect("every launch of this text plans");
        if !d.symbol.starts_with("add_bias") {
            continue;
        }
        let slot = d
            .param_slots
            .iter()
            .find(|s| s.slot == 2)
            .unwrap_or_else(|| panic!("`{}` binds no scalar at buffer 2", d.symbol));
        assert!(!slot.packed, "the row places `width` itself");
        let value = slot
            .value
            .and_then(|i| d.params.get(usize::from(i)))
            .copied()
            .expect("the slot points at a staged scalar");
        seen.push(value);
    }
    assert_eq!(
        seen.len(),
        3 * facts.layers as usize,
        "one bias per projection per layer"
    );
    // In text order, and the text states q, k, v.
    for (i, got) in seen.iter().enumerate() {
        assert_eq!(
            *got,
            widths[i % 3],
            "the {}th bias of layer {} takes the wrong row pitch",
            i % 3,
            i / 3
        );
    }
}
