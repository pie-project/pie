//! A windowed recipe (CONTRACT-BC 19, 21): an engine that serves windowed
//! layers as ring + buffer asks `chunk::plan_windowed` with its ring hook.
//! The windowed rows' unfolded rows lie in one window buffer (a list of the
//! kv page's rows of every windowed layer), the ring region in the record
//! (or at chunk starts), sized as the hook answers, and no window list is
//! left. The facts an engine states from it hold together as the engine
//! crate's `ChunkFacts::validate` checks them.

use poem_compiler::catalog;
use poem_compiler::chunk::{
    self, ChunkProfile, ChunkRecipe, RingLayout, RingPlacement, RingRegion, RowId, SlotLayout,
    WindowBufferSlot, WindowedRow,
};
use poem_compiler::{Budget, Budgets, DeviceProfile};
use poem_ir::{CacheRow, Platform, Trace};

fn on(chunk: ChunkProfile) -> DeviceProfile {
    DeviceProfile {
        chunk,
        ..DeviceProfile::default()
    }
}

fn trace(name: &str) -> Trace {
    catalog::deployment(name)
        .unwrap_or_else(|| panic!("{name} is in the catalog"))
        .trace(Platform::Cuda)
}

/// An engine indexing per-layer rings: the plain form.
struct Plain(u64);

impl RingLayout for Plain {
    fn ring_region(&self, layers: &[WindowedRow], _: &WindowBufferSlot<'_>) -> RingRegion {
        chunk::plain_rings(layers, self.0).region
    }
}

/// An engine whose ring is pages laid like window-buffer slots, on the slot
/// grid from a chunk's start (Metal's): `R / P` pages, `R = ceil(W / P) * P
/// + P` rows of the widest window, `P` the buffer's rows.
struct Paged;

impl RingLayout for Paged {
    fn ring_region(&self, layers: &[WindowedRow], buffer: &WindowBufferSlot<'_>) -> RingRegion {
        let p = u64::from(buffer.tokens);
        let w = layers
            .iter()
            .map(|l| u64::from(l.window))
            .max()
            .unwrap_or(0);
        let pages = w.div_ceil(p) + 1;
        let bytes = pages * buffer.bytes;
        let mut layer_bytes: Vec<u64> = layers
            .iter()
            .map(|layer| {
                pages
                    * buffer
                        .rows
                        .iter()
                        .filter(|run| layer.rows.contains(&run.row))
                        .map(|run| p * run.per_token)
                        .sum::<u64>()
            })
            .collect();
        let laid: u64 = layer_bytes.iter().sum();
        if let Some(last) = layer_bytes.last_mut() {
            *last += bytes - laid;
        }
        RingRegion {
            bytes,
            atom: buffer.bytes,
            placement: RingPlacement::ChunkStart,
            layer_bytes,
        }
    }
}

fn plan(trace: &Trace, hook: &dyn RingLayout) -> ChunkRecipe {
    chunk::plan_windowed(
        trace,
        &Budgets::of(Budget::default()),
        &on(ChunkProfile::SERVER_CUDA),
        hook,
    )
    .expect("a windowed recipe")
}

/// The record's bytes: the state record (the ring inside it, where it lies
/// there) and the ring shape's.
fn record_bytes(recipe: &ChunkRecipe) -> u64 {
    let of = |shape: Option<&SlotLayout>| match shape {
        Some(SlotLayout::Fixed { record, .. }) => *record,
        _ => 0,
    };
    let windowed = recipe.windowed.as_ref().expect("windowed");
    of(recipe.fixed()) + of(windowed.ring_shape(recipe))
}

/// What the engine crate's `ChunkFacts::validate` asks of the facts an
/// engine states from the recipe: layers ascending and distinct, each ring
/// of some rows and bytes, the record holding them all, one window buffer
/// (a list of rate 1 and no window) and no window list.
fn facts_hold(recipe: &ChunkRecipe) {
    let windowed = recipe.windowed.as_ref().expect("windowed");
    let layers: Vec<u32> = windowed
        .layers
        .iter()
        .map(|l| l.layer.expect("a windowed layer is a layer's"))
        .collect();
    assert!(layers.windows(2).all(|p| p[0] < p[1]), "{layers:?}");
    for (layer, bytes) in windowed.layers.iter().zip(&windowed.region.layer_bytes) {
        assert!(layer.window > 0 && *bytes > 0, "{layer:?}: {bytes}");
        assert!(*bytes >= u64::from(layer.window) * layer.per_token);
    }
    let rings: u64 = windowed.region.layer_bytes.iter().sum();
    assert_eq!(
        rings, windowed.region.bytes,
        "the facts state what the hook answered"
    );
    assert!(
        rings <= record_bytes(recipe),
        "{rings} past the record's {}",
        record_bytes(recipe)
    );
    assert!(
        matches!(
            recipe.window_buffer(),
            Some(SlotLayout::List { window: None, .. })
        ),
        "{}",
        recipe.shapes()
    );
    assert_eq!(
        recipe.windows().count(),
        0,
        "no window list: {}",
        recipe.shapes()
    );
    let run = windowed.ring_run(recipe).expect("the ring's run");
    assert_eq!(run.row, RowId::Ring);
}

fn windowed_rows(trace: &Trace) -> Vec<(u32, u32)> {
    trace
        .caches
        .iter()
        .enumerate()
        .filter_map(|(at, row)| match row {
            CacheRow::Kv {
                window: Some(w), ..
            } => Some((at as u32, *w)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_gemma_window_lies_in_the_record_beside_its_global_kv_list() {
    let trace = trace("gemma4-e4b-mini-l6-bf16-kv-bf16");
    let windowed = windowed_rows(&trace);
    assert!(!windowed.is_empty(), "gemma has windowed rows");
    let recipe = plan(&trace, &Plain(256));
    facts_hold(&recipe);
    let ring = recipe.windowed.as_ref().unwrap();
    // The global layers keep the kv list; the buffer is not it.
    let (
        Some(SlotLayout::List {
            tokens: page,
            rows: kv,
            ..
        }),
        Some(SlotLayout::List { tokens, rows, .. }),
    ) = (recipe.list(), recipe.window_buffer())
    else {
        panic!("a kv list and a window buffer: {}", recipe.shapes());
    };
    assert_eq!(tokens, page, "the buffer's slot is the kv page");
    assert!(
        kv.iter()
            .all(|run| !windowed.iter().any(|(at, _)| run.row == RowId::Cache(*at)))
    );
    assert_eq!(
        rows.len(),
        windowed.len(),
        "every windowed row in the buffer"
    );
    // The ring lies first in the record, at its slot's start, aligned.
    let run = ring.ring_run(&recipe).unwrap();
    assert_eq!((run.offset, run.slots.first().copied()), (0, Some(0)));
    assert!(ring.ring_shape.is_none());
    assert_eq!(chunk::plain_rings(&ring.layers, 256).region, ring.region);
    let offsets = chunk::plain_rings(&ring.layers, 256).offsets;
    assert!(offsets.windows(2).all(|p| p[0] < p[1]) && offsets[0] == 0);
    // Each layer's rows run in the buffer slot where its runs say.
    assert!(!ring.buffer_runs(&recipe, 0).is_empty());
    // A ring laid at chunk starts (the paged hook) is a shape of its own.
    let paged = plan(&trace, &Paged);
    facts_hold(&paged);
    assert!(paged.windowed.as_ref().unwrap().ring_shape.is_some());
}

#[test]
fn each_rank_holds_each_layer_ring_at_its_own_layers_cut() {
    let trace = trace("gptoss-20b-mini-bf16-mxfp4-kv-bf16");
    let recipe = plan(&trace, &Plain(256));
    let windowed = recipe.windowed.as_ref().unwrap();
    let whole = windowed.rank_layer_bytes(&recipe, 1);
    assert_eq!(
        whole, windowed.region.layer_bytes,
        "one rank holds the region"
    );
    let halves = windowed.rank_layer_bytes(&recipe, 2);
    for (whole, half) in whole.iter().zip(&halves) {
        assert!(
            *half < *whole && *half >= whole.div_ceil(2),
            "{half} of {whole}"
        );
    }
    // The ranks' recipe sizes the ring from the layers' shares.
    let rank = recipe.per_rank(2);
    let run = windowed.ring_run(&recipe).unwrap();
    let ring = rank
        .fixed
        .iter()
        .find(|r| r.row == RowId::Ring)
        .expect("the ring on a rank");
    let held = ring.piece_bytes * u64::from(run.pieces);
    assert!(held >= halves.iter().sum::<u64>() && held < whole.iter().sum::<u64>());

    // A layer whose rows the ranks do not cut keeps its share whole on
    // each rank; the others still halve.
    let mut kept = trace.clone();
    let first = kept
        .caches
        .iter()
        .position(|row| {
            matches!(
                row,
                CacheRow::Kv {
                    window: Some(_),
                    ..
                }
            )
        })
        .unwrap();
    if let CacheRow::Kv { shard, .. } = &mut kept.caches[first] {
        *shard = poem_ir::Shard::Replicated;
    }
    let recipe = plan(&kept, &Plain(256));
    let windowed = recipe.windowed.as_ref().unwrap();
    let halves = windowed.rank_layer_bytes(&recipe, 2);
    assert_eq!(
        halves[0], windowed.region.layer_bytes[0],
        "its own cut: none"
    );
    assert!(halves[1] < windowed.region.layer_bytes[1]);
}
