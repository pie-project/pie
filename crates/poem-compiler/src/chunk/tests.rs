use poem_ir::{
    CacheRow, Def, Dim, Dtype, Node, Operation, Platform, Shard, Ssm, Trace, Ty, ValueDecl, ValueId,
};

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

#[test]
fn a_record_past_the_chunk_is_cut_on_heads_into_equal_pieces() {
    // 12 layers of a 4 MiB f32 state [128 heads, 64, 128] and a 64 KiB conv.
    let mut caches = Vec::new();
    for l in 0..12 {
        caches.push(state(&format!("ssm.{l}"), vec![128, 64, 128], Dtype::F32));
        caches.push(state(&format!("conv.{l}"), vec![4, 8192], Dtype::Bf16));
    }
    let mut t = trace(caches);
    for l in 0..12 {
        delta_reads(&mut t, 2 * l, l);
    }
    let profile = on(ChunkProfile::SERVER_CUDA);
    let recipe = plan(&t, &budgets(), &profile).unwrap();
    let Some(SlotLayout::Fixed {
        bytes,
        record,
        pieces_per_ws,
        pieces,
        ..
    }) = recipe.fixed()
    else {
        panic!("no fixed shape")
    };
    assert_eq!(*record, 12 * ((4 << 20) + (64 << 10)));
    assert!(*bytes <= recipe.chunk_bytes && *bytes >= MIN_PIECE);
    assert!(u64::from(*pieces_per_ws) * bytes >= *record);
    for piece in pieces {
        let atom = recipe.atoms[&piece.row];
        assert!(piece.span.end <= atom.extent);
        assert!(piece.offset + (piece.span.end - piece.span.start) * atom.bytes <= *bytes);
        if atom.extent > 1 && piece.span != (0..atom.extent) {
            assert_eq!(atom.contract, Contract::Heads);
            assert_eq!(piece.span.start % atom.grain, 0);
        }
    }
    // Each rank of 4 holds a quarter of every cut row and as many chunks.
    let rank = recipe.per_rank(4);
    assert_eq!(rank.slots.len(), recipe.slots.len());
    assert!(rank.chunk_bytes <= recipe.chunk_bytes / 4 + (64 << 10));
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
    assert_eq!(recipe.fixed_sets(), 2);
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
