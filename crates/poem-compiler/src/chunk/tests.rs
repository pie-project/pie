use poem_ir::{
    CacheRow, Def, Dim, Dtype, Node, Operation, Platform, Shard, Ssm, Trace, Ty, ValueDecl, ValueId,
};

use std::collections::BTreeMap;

use super::*;
use crate::{Budget, Budgets, DeviceProfile};

fn trace(caches: Vec<CacheRow>) -> Trace {
    Trace {
        name: "hand-built".to_string(),
        platform: Platform::Cuda,
        params: Vec::new(),
        caches,
        values: Vec::new(),
        nodes: Vec::new(),
        outputs: Vec::new(),
        drafter: None,
        facts: Default::default(),
        inputs: Vec::new(),
    }
}

fn kv(name: &str, width: u64, window: Option<u32>) -> CacheRow {
    CacheRow::Kv {
        name: name.to_string(),
        planes: vec![width, width],
        dtype: Dtype::Bf16,
        space: u32::from(window.is_some()),
        window,
        head_dim: 128,
        live: None,
        shard: Shard::Cut {
            axis: 0,
            segments: vec![width, width],
            heads: Vec::new(),
        },
        rows: None,
    }
}

fn state(name: &str, slab: Vec<u64>, dtype: Dtype) -> CacheRow {
    CacheRow::State {
        name: name.to_string(),
        slab,
        dtype,
        shard: Shard::Cut {
            axis: 0,
            segments: Vec::new(),
            heads: Vec::new(),
        },
    }
}

/// A gated-delta step reading cache row `row` in `layer`.
fn delta_reads(trace: &mut Trace, row: u32, layer: u32) {
    let mut value = |def: Def, width: u64| {
        trace.values.push(ValueDecl {
            def,
            ty: Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(width)],
                dtype: Dtype::Bf16,
            },
        });
        ValueId(u32::try_from(trace.values.len() - 1).unwrap())
    };
    let input = |n| Def::Input(poem_ir::RuntimeInput::Mask { space: n });
    let (qkv, z, gates) = (value(input(0), 8), value(input(1), 8), value(input(2), 8));
    let cache = value(Def::Cache(row), 1);
    let y = value(Def::Op(0), 8);
    trace.nodes.push(Node {
        op: Operation::Ssm(Ssm::GatedDelta {
            qkv,
            z,
            gates,
            state: cache,
            k_heads: 1,
            v_heads: 1,
            k_dim: 1,
            v_dim: 1,
            y,
        }),
        guard: poem_ir::Guard::Always,
        layer: Some(layer),
    });
}

fn budgets() -> Budgets {
    Budgets::of(Budget::default())
}

fn on(chunk: ChunkProfile) -> DeviceProfile {
    DeviceProfile {
        chunk,
        ..DeviceProfile::default()
    }
}

#[test]
fn full_rows_of_every_space_share_one_list_and_a_window_rings() {
    let mut caches: Vec<CacheRow> = (0..4).map(|l| kv(&format!("kv.{l}"), 1024, None)).collect();
    caches.extend((0..8).map(|l| kv(&format!("swa.{l}"), 2048, Some(1024))));
    if let CacheRow::Kv { space, .. } = &mut caches[1] {
        *space = 7;
    }
    let recipe = plan(&trace(caches), &budgets(), &on(ChunkProfile::SERVER_CUDA)).unwrap();
    let lists: Vec<_> = recipe
        .slots
        .iter()
        .filter(|s| matches!(s, SlotLayout::List { .. }))
        .collect();
    assert_eq!(lists.len(), 1);
    let SlotLayout::List {
        tokens,
        bytes,
        rows,
    } = lists[0]
    else {
        unreachable!()
    };
    assert_eq!(rows.len(), 4);
    assert_eq!(tokens % TOKEN_GRAIN, 0);
    assert_eq!(*bytes, u64::from(*tokens) * 4 * 4096);
    assert!(u64::from(*tokens) * 4096 >= MIN_RUN);
    let SlotLayout::Ring {
        tokens,
        cap_slots,
        cap,
        ..
    } = recipe.rings().next().unwrap()
    else {
        unreachable!()
    };
    assert_eq!(cap.window, 1024);
    assert_eq!(*cap_slots, cap.tokens().div_ceil(*tokens) + 1);
    assert_eq!(
        recipe.chunk_bytes % bytes,
        0,
        "C is a multiple of the list slot"
    );
    assert!(recipe.fixed().is_none());
}

/// What is wrong with the fixed `rows` of a slot of `bytes`: a piece past
/// its slot or atoms, pieces of a cut row off its grain, two runs sharing
/// bytes of a slot.
fn overlaps(recipe: &ChunkRecipe, rows: &[FixedRun], bytes: u64, slots: u32) -> Vec<String> {
    let mut wrong = Vec::new();
    let mut held: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
    for run in rows {
        let atom = recipe.atoms[&run.row];
        let end = run.offset + run.piece_bytes(&atom);
        if run.slots.len() != run.pieces as usize
            || end > bytes
            || u64::from(run.pieces) * run.atoms_per_piece < atom.extent
            || u64::from(run.pieces - 1) * run.atoms_per_piece >= atom.extent
            || (run.is_cut() && run.atoms_per_piece % atom.grain != 0)
        {
            wrong.push(format!("{run:?} of {atom:?}"));
        }
        for &slot in &run.slots {
            if slot >= slots {
                wrong.push(format!("{run:?} past slot {slots}"));
            }
            held.entry(slot).or_default().push((run.offset, end));
        }
    }
    for (slot, mut spans) in held {
        spans.sort_unstable();
        for pair in spans.windows(2) {
            if pair[1].0 < pair[0].1 {
                wrong.push(format!("slot {slot}: {pair:?}"));
            }
        }
    }
    wrong
}

#[test]
fn a_row_past_the_slot_is_cut_on_heads_into_equal_pieces_at_one_offset() {
    // 12 layers of a 6 MiB f32 state [96 heads, 64, 256] and a bf16 conv
    // Metal keeps in f32 (128 KiB), on a device whose chunks hold at most
    // 4 MiB.
    let mut caches = Vec::new();
    for l in 0..12 {
        caches.push(state(&format!("ssm.{l}"), vec![96, 64, 256], Dtype::F32));
        caches.push(state(&format!("conv.{l}"), vec![4, 8192], Dtype::Bf16));
    }
    let mut t = trace(caches);
    for l in 0..12 {
        delta_reads(&mut t, 2 * l, l);
    }
    let recipe = plan(&t, &budgets(), &on(ChunkProfile::ON_DEVICE_METAL)).unwrap();
    let Some(SlotLayout::Fixed {
        bytes,
        record,
        pieces_per_ws,
        rows,
        ..
    }) = recipe.fixed()
    else {
        panic!("no fixed shape")
    };
    assert_eq!(*record, 12 * ((6 << 20) + (128 << 10)));
    assert!(*bytes <= recipe.chunk_bytes && *bytes >= MIN_PIECE);
    assert!(u64::from(*pieces_per_ws) * bytes >= *record);
    assert_eq!(rows.len(), 24);
    let wrong = overlaps(&recipe, rows, *bytes, *pieces_per_ws);
    assert!(wrong.is_empty(), "{wrong:#?}");
    for run in rows {
        let atom = recipe.atoms[&run.row];
        if atom.bytes * atom.extent == 6 << 20 {
            // Every state is cut alike: equal pieces of whole heads.
            assert!(run.is_cut(), "{run:?}");
            assert_eq!(atom.contract, Contract::Heads);
            assert_eq!(run.atoms_per_piece % atom.grain, 0);
            assert_eq!(run.slots.len(), run.pieces as usize);
        } else {
            assert_eq!((run.pieces, run.atoms_per_piece), (1, atom.extent));
        }
    }
    // Each rank of 4 holds a quarter of every cut row's pieces, in as many
    // slots at offsets common to them, and as many chunks.
    let rank = recipe.per_rank(4);
    assert_eq!(rank.slots.len(), recipe.slots.len());
    assert!(rank.chunk_bytes <= recipe.chunk_bytes / 4 + (64 << 10));
    assert_eq!(rank.fixed.len(), rows.len());
    for (run, on) in rows.iter().zip(&rank.fixed) {
        assert_eq!(run.row, on.row);
        if run.is_cut() {
            assert_eq!(on.atoms_per_piece * 4, run.atoms_per_piece);
        }
    }
}

#[test]
fn rows_under_the_slot_stay_whole_and_share_slots() {
    // 12 layers of a 4 MiB state and a 64 KiB conv past a chunk of 32 MiB:
    // no row is cut; whole rows pack into shared slots.
    let mut caches = Vec::new();
    for l in 0..12 {
        caches.push(state(&format!("ssm.{l}"), vec![128, 64, 128], Dtype::F32));
        caches.push(state(&format!("conv.{l}"), vec![4, 8192], Dtype::Bf16));
    }
    let mut t = trace(caches);
    for l in 0..12 {
        delta_reads(&mut t, 2 * l, l);
    }
    let recipe = plan(&t, &budgets(), &on(ChunkProfile::SERVER_CUDA)).unwrap();
    let Some(SlotLayout::Fixed {
        bytes,
        pieces_per_ws,
        rows,
        ..
    }) = recipe.fixed()
    else {
        panic!("no fixed shape")
    };
    assert!(*bytes >= 4 << 20);
    assert!(rows.iter().all(|r| !r.is_cut()));
    assert!(*pieces_per_ws < 24);
    let wrong = overlaps(&recipe, rows, *bytes, *pieces_per_ws);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_few_bytes_of_ids_share_a_chunk_with_a_thousand_others() {
    let caches = vec![
        kv("kv.0", 512, None),
        CacheRow::State {
            name: "replay".to_string(),
            slab: vec![1],
            dtype: Dtype::I32,
            shard: Shard::Replicated,
        },
        CacheRow::State {
            name: "engram".to_string(),
            slab: vec![3],
            dtype: Dtype::I32,
            shard: Shard::Replicated,
        },
    ];
    let recipe = plan(&trace(caches), &budgets(), &on(ChunkProfile::SERVER_CUDA)).unwrap();
    let fixed = recipe.fixed().unwrap();
    assert_eq!(fixed.bytes(), FIXED_SLOT_FLOOR);
    assert!(fixed.per_chunk(recipe.chunk_bytes) >= 1024);
    assert_eq!(recipe.fixed_sets_per_ws(), 1);
    // The copy-on-write reserve is one record a lane, once for the pool.
    assert_eq!(recipe.cow_sets(), budgets().tokens.max_lanes);
    assert_eq!(recipe.cow_slots(), u64::from(budgets().tokens.max_lanes));
}

#[test]
fn a_tiled_profile_aligns_every_row_and_slot_to_its_tile() {
    let caches = vec![
        kv("kv.0", 192, None),
        kv("kv.1", 192, None),
        state("s", vec![3, 1000], Dtype::Bf16),
    ];
    let recipe = plan(&trace(caches), &budgets(), &on(ChunkProfile::TPU)).unwrap();
    let tile = ChunkProfile::TPU.tile_bytes;
    for shape in &recipe.slots {
        assert_eq!(shape.bytes() % tile, 0);
        for run in shape.runs() {
            assert_eq!(run.offset % tile, 0);
        }
    }
    assert_eq!(recipe.chunk_bytes % tile, 0);
}

#[test]
fn a_small_window_folds_into_the_fixed_record() {
    let caches = vec![kv("kv.0", 512, None), kv("swa.0", 64, Some(64))];
    let recipe = plan(&trace(caches), &budgets(), &on(ChunkProfile::SERVER_CUDA)).unwrap();
    assert_eq!(recipe.rings().count(), 0);
    let Some(SlotLayout::Fixed { folded, .. }) = recipe.fixed() else {
        panic!("the ring folds into a fixed record")
    };
    assert_eq!(folded.len(), 1);
}

#[test]
fn an_uncuttable_row_past_every_binding_is_refused() {
    let caches = vec![CacheRow::State {
        name: "ids".to_string(),
        slab: vec![1 << 30],
        dtype: Dtype::I32,
        shard: Shard::Replicated,
    }];
    let refused = plan(
        &trace(caches),
        &budgets(),
        &on(ChunkProfile::ON_DEVICE_METAL),
    );
    assert!(matches!(refused, Err(ChunkError::Uncuttable { .. })));
}

#[test]
fn a_packed_kv_row_takes_its_stored_bytes_a_token() {
    // KvU4 packs a head of `head_dim` to `head_dim / 2 + 2` bytes; with no
    // head_dim stated, the codec's 256-wide anchor (130 bytes); a
    // block-scaled row is declared in its own bytes.
    let u4 = |head_dim| CacheRow::Kv {
        name: "kv.0".to_string(),
        planes: vec![1024, 1024],
        dtype: Dtype::KvU4,
        space: 0,
        window: None,
        head_dim,
        live: None,
        shard: Shard::Replicated,
        rows: None,
    };
    let rows = CacheRow::Kv {
        name: "kv.1".to_string(),
        planes: vec![584],
        dtype: Dtype::U8,
        space: 0,
        window: None,
        head_dim: 512,
        live: None,
        shard: Shard::Replicated,
        rows: Some(poem_ir::KvRows::V4),
    };
    for (row, want) in [(u4(128), 2 * 8 * 66), (u4(0), 2 * 4 * 130), (rows, 584)] {
        let census = census::census(&trace(vec![row]), false).unwrap();
        assert_eq!(census.tokens[0].per_token(), want);
    }
}
