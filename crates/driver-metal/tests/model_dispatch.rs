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
    Dispatch, Geometry, Undispatchable, facts_of, named_tile, pipelines_needed, plan_launch,
};
use driver_metal::lowering::executor::{FireTable, Frame, Resolver, Slice};
use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_ir::trace::{FireClass, ValueId};

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
    /// The KV pages, and the pool's three strides.
    ///
    /// A stub that declined these read as a driver that CANNOT DISPATCH the
    /// paged statements -- `kv_append_paged` and both `sdpa_paged` legs
    /// refused as *"the KV page size: the pool has none"*, which is the
    /// driver answering correctly about a rig that had not been asked to
    /// hold a pool. The refusal is right and the rig was wrong, so the rig
    /// answers now.
    ///
    /// qwen3-0.6b's own numbers, matching [`geometry`], because a stride
    /// invented here would let a grid that reads past a page pass.
    fn kv(&mut self, _: u16, _: bool) -> Option<Slice> {
        Some(Slice {
            address: 0x3000_0000,
            bytes: 1 << 30,
        })
    }
    fn pool(&mut self, which: FireTable) -> Option<u32> {
        Some(match which {
            FireTable::KvHeadStride => 128,
            FireTable::KvSeqStride => 8 * 128,
            FireTable::KvPageSize => 256,
            _ => return None,
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
        // qwen3-0.6b's checkpoint is 4-bit over groups of 64.
        group: 64,
        bits: 4,
        // A dense row with one head width and one affine point states none
        // of the five axes a row can hold twice, and zero is what
        // `geometry_from_deployment` puts there for such a row.
        ..Geometry::default()
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
        match plan_launch(low, launch, frame, geometry(), &mut store) {
            Ok(d) => ok.extend(d),
            Err(e) => refused.push(e),
        }
    }
    (ok, refused)
}

#[test]
fn the_gemm_covers_exactly_the_output_it_writes() {
    // `affine_qmm_t` is a C++ template over `(dtype x group x bits x BM x BK x
    // BN)` -- `y_row = tid.y * BM`, `y_col = tid.x * BN` -- so the tile is
    // COMPILED into the entrypoint and the grid must be threadgroups times
    // that tile. The driver used to derive the tile from the fire's geometry
    // rather than read it off the name, and at this shape the two disagreed:
    // `(64, 64)` derived against `_bm_32_bn_32` named. A threadgroup count
    // computed for a 64-wide tile and handed to a 32-wide kernel covers a
    // QUARTER of the output, so every long prefill's projections were three
    // quarters whatever the arena held, and no layer reported anything.
    //
    // 512 rows and not the 16 the test below uses, because 16 never reaches
    // this arm: the GEMM is guarded by `TokensMultipleOf(32)`, so every
    // correctness oracle in this crate -- each of which prefills one or two
    // tokens -- resolved to the matvec and could not have seen it. The one
    // test that did reach it measured TIME.
    let low = lowered(FireClass::Prefill, 512);
    let frame = frame(&low);
    let mut store = Sentinels;
    let mut seen = 0usize;
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        // The tile in the NAME is the only tile there is; a symbol carrying
        // none is not a GEMM point.
        let Some((bm, bn)) = named_tile(symbol) else {
            continue;
        };
        let dims = facts_of(&low, launch, geometry());
        let d = plan_launch(&low, launch, frame, geometry(), &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
            .unwrap_or_else(|e| panic!("`{symbol}` dispatches: {e:?}"));

        // `dispatchThreads` takes THREADS, so the threadgroup count -- which
        // is what the kernel indexes by -- is the quotient.
        let groups = [
            d.grid[0] / d.threadgroup[0],
            d.grid[1] / d.threadgroup[1],
            d.grid[2] / d.threadgroup[2],
        ];
        assert_eq!(
            (groups[0] * bn, groups[1] * bm),
            (dims.width, dims.rows),
            "`{symbol}`: {groups:?} threadgroups at ({bm}, {bn}) do not cover \
             {} rows of {}",
            dims.rows,
            dims.width
        );
        seen += 1;
    }
    // Without this the test passes by planning no GEMM at all, which is
    // exactly how the disagreement survived for as long as it did.
    assert!(seen > 0, "no GEMM was planned, so nothing was checked");
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
                Undispatchable::Unclaimed { symbol, .. }
                | Undispatchable::Unbound { symbol, .. }
                | Undispatchable::Refused { symbol, .. }
                | Undispatchable::Misspelled { symbol, .. }
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
    // could not come off a shape even in principle: `params[3]` is a f32
    // BIT-CAST into a u32 slot, and `params[5]` is `u32::MAX` standing for
    // "no sliding window" -- a sentinel, not a measurement. A driver that
    // reconstructed scalars from operand extents would have to invent both.
    let low = lowered(FireClass::Decode, 1);
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let sdpa = planned(&low)
        .0
        .into_iter()
        .find(|d| d.symbol.starts_with("sdpa_paged_decode"))
        .expect("the text states a paged decode attention");

    // The SHADER's order, not the statement's: `sdpa_paged.metal` numbers
    // its constants gqa_factor(4), page_size(9), n_kv_heads(10), scale(11),
    // attention_mask_stride(13), window(15), and the packed array follows
    // the buffer numbers. `page_size` sits second and comes off the POOL,
    // which is the one scalar here that no statement states.
    assert_eq!(sdpa.params.len(), 6, "the statement's six scalars");
    assert_eq!(
        sdpa.params[0],
        facts.q_heads / facts.kv_heads,
        "the GQA group, 16 query heads over 8 KV heads"
    );
    assert_eq!(sdpa.params[1], 256, "the page size, off the pool");
    assert_eq!(sdpa.params[2], facts.kv_heads, "the KV head count");

    let scale = f32::from_bits(sdpa.params[3]);
    let want = 1.0 / (facts.head_dim as f32).sqrt();
    assert!(
        (scale - want).abs() < 1e-9,
        "the softmax scale rides through a u32 slot as bits: got {scale}, want {want}"
    );

    assert_eq!(
        sdpa.params[5],
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
            // A PACKED BLOCK IS NOT AN EMPTY RUN. `rms_single_row` forwards
            // `ctx.params()` whole -- the shader reads a `RmsParams` struct by
            // field -- so its scalars ride as ONE staged block and `params` is
            // empty by construction. It read as "carries no scalars" only
            // while the block went unrecognised, which is what
            // `ParamSlot::packed` says and `packed_params_cover_the_struct`
            // measures.
            !s.starts_with("kv_append")
                && !s.starts_with("sdpa_")
                && *s != "silu_mul_bfloat16"
                && *s != "rms_single_row_bfloat16"
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
            let dims = facts_of(&low, launch, geometry());
            assert_eq!(dims.rows, rows, "`{symbol}` at {rows} rows");
            assert_eq!(dims.q_heads(), geometry().q_heads, "the fire states the rest");
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
    // SIXTY-FOUR rows, and the number is load-bearing. The tiled matmul is
    // guarded by `GuardPred::TokensMultipleOf(k)`, and `k` grew from 16 to 32
    // -- so a 16-row prefill stopped being a multiple of it, fell back to the
    // vector kernel, and named exactly the symbols a decode names. The
    // assertion below then failed as "untestable here", which is true and
    // reads like the claim was wrong rather than the fixture stale.
    let decode: BTreeSet<String> = lowered(FireClass::Decode, 1).kernels.into_iter().collect();
    let prefill: BTreeSet<String> = lowered(FireClass::Prefill, 64)
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

    fn plan_for(class: FireClass) -> model_ir::trace::ForwardPlan {
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
            // Each lane reads ITS OWN only row, so each names 0. This said
            // `[0, 1, 2, 3]` -- the fire's row numbers -- which is a table the
            // engine never emits: it was written to match a driver that read
            // the field absolutely, not to state the contract. Under the
            // numbering the scheduler actually uses, lane 3 naming row 3 of a
            // one-row request is now refused by name.
            sampling_indices: &[0, 0, 0, 0],
            sampling_indptr: &[0, 1, 2, 3, 4],
            ..Step::default()
        };
        assert_eq!(fire_class(&step), FireClass::Decode);

        let low = lower_step(&plan_for(fire_class(&step)), &step).expect("the step lowers");
        let mut store = Sentinels;
        let mut grids = 0;
        for launch in &low.launches {
            match plan_launch(&low, launch, frame(&low), geometry(), &mut store) {
                Ok(d) => {
                    for d in &d {
                        assert!(d.grid.iter().all(|&n| n > 0));
                        grids += 1;
                    }
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
        // FOUR WIRE ROWS, and that is what a region table indexes.
        //
        // This was one request of four tokens with regions at `[0, 2, 4]`,
        // which reads as a table over TOKEN rows -- and the bridge translates
        // through `qo_indptr` exactly as `driver-cuda` does, so wire rows 2
        // and 4 of a one-request step resolved to `u32::MAX` and every
        // seriated case here refused before it reached the thing it asks
        // about. Two tokens a request keeps the fire a prefill and gives the
        // two regions two wire rows each.
        let plain = Step {
            token_ids: &[1, 2, 3, 4, 5, 6, 7, 8],
            qo_indptr: &[0, 2, 4, 6, 8],
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
            token_ids: &[1, 2, 3, 4, 5, 6, 7, 8],
            qo_indptr: &[0, 2, 4, 6, 8],
            region_row_indptr: &[0, 2, 4],
            region_sig: &[sig::TRUNCATED, 0],
            region_k: &[4, u32::MAX],
            ..Step::default()
        };
        // The ORDER is what must refuse it. Stated over one request, this
        // step's regions named wire rows that did not exist and the refusal
        // came from the translation instead -- a green assertion about
        // something else entirely. The seriated twin above lowers under the
        // same numbering, so the difference between them is the order alone.
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
/// answered that with a hand-written arm; `Source::Named(<keys::KvKeys as keys::Fact>::KEY)`/`KvValues` let the
/// ROW ask, and `Resolver::kv` is where the asking lands.
mod state {
    use std::collections::HashMap;

    use driver_metal::layout::kv::Shape;
    use driver_metal::lowering::executor::{Resolver, Slice};
    use driver_metal::lowering::resolve::{Names, Store};

    use super::*;

    /// qwen3-0.6b's pool, matching [`geometry`].
    ///
    /// `kv_append_paged` and both `sdpa_paged` legs read a page size, so a
    /// store without one refuses them -- correctly, because a page size of
    /// zero would send the ring's arithmetic through the wrong rows. These
    /// tests are about the PAGES, so the pool is stated and the refusal they
    /// would otherwise all take is out of the way.
    fn pool() -> Shape {
        Shape {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
            page_size: 256,
            pages: 64,
            element_bytes: 2,
            global_head_dim: 0,
            global_kv_heads: 0,
            full_attn_every: 0,
        }
    }

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
        let mut store = Store::new(Names::mlx(), &tensors, &named)
            .with_kv(&kv)
            .with_pool(pool());

        // Answer the weights too, so a refusal is about state and nothing else.
        assert!(store.weight("layer.0.attn_norm").is_none());

        let mut checked = 0;
        for launch in &low.launches {
            if !low.kernels[launch.kernel as usize].starts_with("kv_append") {
                continue;
            }
            let d = plan_launch(&low, launch, frame(&low), geometry(), &mut store)
                .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
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
        let d = plan_launch(&low, launch, frame(&low), geometry(), &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
            .expect("a rotation plans");
        // Buffer 1 is `position`, which the row says and this reads back.
        assert_eq!(d.args[1].slice.address, 0x5150_0000);
        assert_eq!(d.args.len(), 5, "x, position, and three scalars");
    }

    #[test]
    fn a_resolver_with_no_pool_binds_a_region_that_addresses_nothing() {
        // A statement that asks for pages and gets nothing binds a region
        // addressing nothing — the same honest answer a missing scale gets,
        // and not a skipped slot, which would shift every operand after it.
        //
        // The POOL is stated and the PAGES are not, which is the split this
        // test used to miss: it held neither and read the resulting refusal
        // as a plan. A page POINTER is a buffer, and a buffer nothing hands
        // out is bindable as nothing; a page SIZE is arithmetic, and there
        // is no value that stands for "unknown" in a multiply. So the pages
        // bind empty here and the size refuses in
        // `a_paged_write_with_no_page_size_refuses_rather_than_reading_row_zero`.
        let low = lowered(FireClass::Decode, 1);
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let mut store = Store::new(Names::mlx(), &tensors, &named).with_pool(pool());
        let launch = low
            .launches
            .iter()
            .find(|l| low.kernels[l.kernel as usize].starts_with("kv_append"))
            .expect("the text writes KV");
        let d = plan_launch(&low, launch, frame(&low), geometry(), &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
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

    /// The other half of the split above: the SIZE has no empty answer.
    #[test]
    fn a_paged_write_with_no_page_size_refuses_rather_than_reading_row_zero() {
        let low = lowered(FireClass::Decode, 1);
        let (tensors, named) = (HashMap::new(), HashMap::new());
        let kv = |l: u16, v: bool| pages(l, v);
        let mut store = Store::new(Names::mlx(), &tensors, &named).with_kv(&kv);
        let launch = low
            .launches
            .iter()
            .find(|l| low.kernels[l.kernel as usize].starts_with("kv_append"))
            .expect("the text writes KV");
        let why = plan_launch(&low, launch, frame(&low), geometry(), &mut store)
            .expect_err("a write with no page size cannot be planned");
        assert!(
            format!("{why:?}").contains("the KV page size"),
            "and says which number is missing: {why:?}"
        );
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

    let geometry = Geometry {
        q_heads: 32,
        kv_heads: 16,
        head_dim: 256,
        // The FIRE's number, which is the sliding layers' and which the full
        // layers must NOT take.
        rotary_dims: 256,
        group: 64,
        bits: 4,
        // The SECOND head width, and how often a layer takes it. Without
        // these the geometry knows of no full layer at all, `is_full_attention`
        // answers false everywhere, and every layer rotates the fire's 256 --
        // which is this test's own bug, passing before the axis existed
        // because there was then nothing to state.
        global_head_dim: 512,
        global_kv_heads: 4,
        full_attn_every: 6,
        v_heads: 0,
        v_dim: 0,
        ..Geometry::default()
    };
    // The PLANNED grid, not `facts_of`. `Facts::rotary_dims` is the fire's
    // number by construction -- it is the FALLBACK `stated` takes when a
    // statement carries none -- so reading it back could only ever return
    // what was put in, and this test read it and proved nothing. `rope_grid`
    // puts the resolved count in `grid[0]` as pairs, which is where a rope
    // told the wrong extent stops covering its channels.
    let frame = frame(&low);
    let mut store = Sentinels;
    let mut by_layer: BTreeSet<(u16, u32)> = BTreeSet::new();
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        if !symbol.starts_with("neox") {
            continue;
        }
        let d = plan_launch(&low, launch, frame, geometry, &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
            .expect("a rotation plans");
        by_layer.insert((launch.layers.start, d.grid[0] * 2));
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
        // gpt-oss's MXFP4: blocks of 32, four bits.
        group: 32,
        bits: 4,
        // The router gate arrives at its own point -- see the same two
        // numbers in `text_conformance`'s gpt-oss row. Not read by the
        // assertions below, which plan only the expert bank's leg, but a
        // geometry that understated it would compose the gate's symbol
        // wrong for anything that did.
        router_group: 64,
        router_bits: 8,
        ..Geometry::default()
    };
    let frame = frame(&low);
    // The kernel's own parameter names, by slot. This asked the ROW, whose
    // `operands` column listed them -- and `quant` has retired its rows, so
    // there is no column left to ask. Reading `quant/qmv.metal` is the
    // stronger question anyway: the row was a transcription of this list and
    // could be wrong about it, where the shader IS the ABI the pipeline is
    // built from.
    let slots = buffer_names(&kernels_dir(), "quant/qmv.metal", "mxfp4_qmv_routed_bias");
    assert!(
        slots.len() > 4,
        "the parameter list for `mxfp4_qmv_routed_bias` was not found in \
         quant/qmv.metal; this test reads the shader for its slot names"
    );

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
        let mut store = Sentinels;
        let d = plan_launch(&low, launch, frame, geometry, &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
            .expect("the routed MXFP4 leg plans");
        for (slot, name) in slots.iter().enumerate() {
            let Some(arg) = d.args.get(slot) else {
                continue;
            };
            let bound = arg.slice;
            // `biases` is in the ABI because the argument list is positional,
            // and unread because the codec has no such plane. `bias` is read
            // per output row. Naming them off the SHADER rather than by
            // index, so that a list reordered upstream is still being asked
            // the right question.
            match name.as_str() {
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
/// `norm::add_bias` states its row pitch as `Source::Slot(Kind::OutWidth, 0)` — the row
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
        let d = plan_launch(&low, launch, frame, g, &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
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

/// gemma's PLE tail, whose middle statement is the one strided kernel a live
/// text names.
fn gemma_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_metal(
        &LlamaLikeFacts {
            layers: 4,
            ..LlamaLikeFacts::qwen3_0_6b()
        },
        &LlamaLikeMetalFacts::gemma_like(),
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
    .expect("the gemma text lowers")
}

/// The strided activation can reach every row it is given.
///
/// `geglu_tanh_strided` is FLAT — its row states `LaunchRule::Elementwise`,
/// so the grid is `[width * rows, 1, 1]` and the body recovers the row by
/// dividing the thread id by `p.width`. Two things therefore have to hold, and
/// neither held before this test was written:
///
/// * the thread count is the whole rectangle, not one row of it, and
/// * `params[0]` is the rectangle's TRUE width, because it is the divisor.
///
/// The kernel exists for M>1 and for nothing else — its own header says
/// gemma's PLE slice "at M=1 is a byte offset and the flat kernel above
/// serves; at M>1 it is not" — so a launch that covers one row makes the
/// symbol pointless. It covered one row: a grid of [2048, 1, 1] against a
/// `uint2` body whose `gid.y` was structurally zero, on every gemma layer of
/// every prefill.
#[test]
fn the_strided_activation_reaches_every_row_it_is_given() {
    const ROWS: usize = 8;
    let low = gemma_lowered(FireClass::Prefill, ROWS);
    let frame = frame(&low);
    let mut store = Sentinels;
    let mut seen = 0usize;
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        if !symbol.starts_with("geglu_tanh_strided") {
            continue;
        }
        let dims = facts_of(&low, launch, geometry());
        let d = plan_launch(&low, launch, frame, geometry(), &mut store)
            .map(|d| d.into_iter().next().expect("one statement, one dispatch"))
            .unwrap_or_else(|e| panic!("`{symbol}` dispatches: {e:?}"));

        assert_eq!(
            (d.grid[0] as u64, d.grid[1], d.grid[2]),
            (u64::from(dims.width) * u64::from(dims.rows), 1, 1),
            "`{symbol}`: a grid of {:?} over a {} x {} rectangle",
            d.grid,
            dims.rows,
            dims.width
        );
        // `GegluStridedParams` is `{width, unused, gate_pitch, up_pitch,
        // out_pitch}` and the body divides by the first word.
        //
        // READ OFF THE STATEMENT, NOT THE DISPATCH. This body forwards
        // `ctx.params()` whole, so its words ride as ONE staged block and
        // `Dispatch::params` is empty by construction -- `ParamSlot::packed`
        // is what carries them. The statement's own run is the same words in
        // the same order, and it is what the block points into.
        let stated =
            &low.params[launch.params.start as usize..launch.params.end as usize];
        assert_eq!(
            stated[0], dims.width,
            "`{symbol}`: the body divides the thread id by params[0] to get \
             its row, so a stated {} against a real width of {} puts every \
             row but the first at the wrong offset",
            stated[0], dims.width
        );
        seen += 1;
    }
    assert!(
        seen > 0,
        "no strided activation was planned, so nothing was checked"
    );
}

/// The readout covers the rows the fire READS OUT, and not its stream.
///
/// A prefill of one request samples one row, so the gather that compacts it
/// and the head that projects it are one-row rectangles. They were the
/// fire's whole row window instead, and `Rule::Qmv` reads that window as its
/// M: on Llama-3.2-1B the head ran as a 2048-row matvec against a 128256
/// vocabulary and cost 904 ms of a 2184 ms prefill — 41% of the fire —
/// producing 2047 distributions nothing reads. The answer stayed right,
/// which is why only a profile found it.
///
/// Stated here rather than in `model-compiler` because it takes a real
/// text: the two statements are metal's own (`dsl::metal::sample_rows` and
/// `dsl::metal::lm_head`), not the shared epilogue, and it is the pairing of
/// a `[Requests, ..]` output with a launch rule that reads the rectangle
/// that does the damage.
#[test]
fn the_readout_covers_the_sampled_rows_and_not_the_stream() {
    const ROWS: usize = 2048;
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    let mut rows = vec![Row::default(); ROWS];
    // One request, sampled at its last row — what a prefill step is.
    rows[ROWS - 1].samples = true;
    let low = lower(
        &plan,
        &rows,
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the metal text lowers");

    let readout: Vec<_> = low
        .launches
        .iter()
        .filter(|l| {
            let k = low.kernels[l.kernel as usize].as_str();
            k.starts_with("row_gather") || k.starts_with("affine_qmv_fast")
        })
        .collect();
    assert_eq!(
        readout.len(),
        2,
        "a prefill states exactly the gather and the head; found {:?}",
        readout
            .iter()
            .map(|l| &low.kernels[l.kernel as usize])
            .collect::<Vec<_>>()
    );
    for l in readout {
        assert_eq!(
            l.rows,
            0..1,
            "`{}` covers {:?} of a {ROWS}-row fire that samples ONE row",
            low.kernels[l.kernel as usize],
            l.rows,
        );
    }

    // And the body is untouched: every other statement still covers the
    // stream, so the narrowing above is about the readout's row space and
    // not a rule that leaked into the layers.
    let body = low
        .launches
        .iter()
        .filter(|l| {
            let k = low.kernels[l.kernel as usize].as_str();
            !k.starts_with("row_gather") && !k.starts_with("affine_qmv_fast")
        })
        .count();
    assert!(body > 0, "a prefill lowers a body");
    assert!(
        low.launches
            .iter()
            .filter(|l| {
                let k = low.kernels[l.kernel as usize].as_str();
                !k.starts_with("row_gather") && !k.starts_with("affine_qmv_fast")
            })
            .all(|l| l.rows == (0..ROWS as u32)),
        "a body statement stopped covering the stream"
    );
}

/// The shader tree, as the driver finds it.
fn kernels_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("a crates directory")
        .join("kernels-metal/kernels")
}

/// A Metal entrypoint's buffer parameter NAMES, indexed by `[[buffer(n)]]`.
///
/// The row's `operands` column used to carry this list, and a row is a
/// transcription: it could disagree with the kernel and nothing would notice
/// until a dispatch bound the wrong plane. The `.metal` is the declaration
/// the pipeline is actually built from, so a test that wants to say "the
/// slot called `bias`" should read it here.
///
/// Slots the list skips come back empty. `[[buffer(n)]]` is explicit and may
/// have gaps -- `kv_append_paged` declares 0,1,2,3,5,10,12..15 -- so the
/// vector is sized by the HIGHEST index, not by the count.
fn buffer_names(root: &std::path::Path, file: &str, stem: &str) -> Vec<String> {
    let Ok(src) = std::fs::read_to_string(root.join(file)) else {
        return Vec::new();
    };
    let Some(list) = param_list(&src, stem) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for param in list.split(',') {
        let Some(mark) = param.find("[[buffer(") else {
            continue;
        };
        let rest = &param[mark + "[[buffer(".len()..];
        let Some(end) = rest.find(')') else { continue };
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
    out
}

/// The parameter list of the declaration `stem` names, `[[buffer(` and all.
///
/// Three shapes, and the third is most of this tree.
///
/// 1. Written out: `template <...> [[kernel]] void stem(...)`.
/// 2. STAMPED: `gptoss_qmv_kernel(qmv_routed_bias, true, true, 1)`, where the
///    list lives once inside `#define gptoss_qmv_kernel(name, ...)` with
///    `name` substituted -- so `void qmv_routed_bias(` appears nowhere.
/// 3. Stamped and then INSTANTIATED:
///    `instantiate_gptoss_qmv(mxfp4_qmv_routed_bias, qmv_routed_bias, ...)`,
///    whose own `#define` declares `void fn<itype, codec>(...)` with the
///    types alone and no names at all.
///
/// So a macro body is accepted only when its list carries `[[buffer(`, and
/// the invocation's remaining arguments are followed when it does not: (3)
/// hands off to (2), which is where the names are. Bounded at three hops,
/// and a chain that does not end returns `None` rather than a guess.
fn param_list(src: &str, stem: &str) -> Option<String> {
    fn between(src: &str, at: usize) -> Option<String> {
        let open = at + src[at..].find('(')?;
        // Depth-counted, because a parameter list is full of parentheses:
        // `[[buffer(0)]]` closes one the signature did not open, and stopping
        // at the first `)` finds a list of one operand for every kernel.
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

    fn named(list: Option<String>) -> Option<String> {
        list.filter(|l| l.contains("[[buffer("))
    }

    fn resolve(src: &str, stem: &str, depth: u32) -> Option<String> {
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
                if let Some(list) = resolve(src, arg, depth + 1) {
                    return Some(list);
                }
            }
        }
        None
    }

    resolve(src, stem, 0)
}
