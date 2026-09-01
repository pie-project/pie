//! **WHICH `mlxu4` PLANES A TILED READER COULD SERVE** (§J4b's flip, censused
//! before it is taken).
//!
//! ```text
//! cargo test -p models --test which_mlxu4_planes_a_tiled_reader_could_serve \
//!   -- --nocapture
//! ```
//!
//! §J4b landed the whole tiled chain and flipped no shipping SKU. The recipe
//! it recorded is per-family: a `*-mlxu4-*` row's PROJECTIONS get
//! `Dtype::U4g64tiled` and its import swaps `read` for `read_repack`. What
//! the recipe does not say, because it is a fact about each family's file
//! rather than about the layout, is WHICH families have a projection
//! declared `U4g64` at all.
//!
//! They do not all. The tiled point serves one shape — a two-dimensional
//! affine weight, four bits to a code, sixty-four codes to a factor — and
//! `model_dsl::Weight::planes` refuses anything else BY NAME:
//!
//! ```text
//! `{name}` is a tiled affine projection declared {shape}; the tiled point
//! reads a two-dimensional weight
//! ```
//!
//! So a family whose `mlxu4` row spends `U4g64` only on ROUTED EXPERT BANKS
//! — three-dimensional, read by the grouped select and not by
//! `linear::matmul` — has no target for the flip, and forcing one is that
//! panic. This file is the census that says which is which, so the next
//! author reads it instead of rediscovering it against a hundred-gigabyte
//! import.
//!
//! **THE FINDING, AND WHY IT IS THE ONE THAT MATTERS.** `qwen38-flash-mlxu4-
//! kv-bf16` is the only shipping `mlxu4` SKU with a real checkpoint on the
//! CUDA box (`pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`), and it is
//! the one family whose `mlxu4` row has NO two-dimensional `U4g64` plane:
//! its mixed-4/8 conversion raises every projection to eight bits
//! (`qwen_4::model::Model::new`'s `proj`), leaves the n-gram table at four
//! bits grouped by thirty-two (`narrow_group`), and spends the bare `w` on
//! `experts_gate_up` and `experts_down` alone.

use std::collections::BTreeMap;

use model_dsl::{Dtype, Platform};

/// Every catalog SKU whose name says `mlxu4`, with its trace.
fn mlxu4_rows() -> Vec<(&'static str, model_dsl::Trace)> {
    models::catalog()
        .into_iter()
        .filter(|row| row.0.contains("mlxu4"))
        .map(|row| (row.0, row.2(Platform::Cuda)))
        .collect()
}

/// The `U4g64` code planes a trace interns, as `(name, shape)`.
fn u4_planes(trace: &model_dsl::Trace) -> Vec<(String, Vec<u64>)> {
    trace
        .params
        .iter()
        .filter(|p| p.dtype == Dtype::U4g64)
        .map(|p| (p.name.clone(), p.shape.clone()))
        .collect()
}

/// The `U4g64tiled` code planes a trace interns — the ones §J4b's flip has
/// already moved into m16n8k16 fragment order.
fn tiled_planes(trace: &model_dsl::Trace) -> Vec<(String, Vec<u64>)> {
    trace
        .params
        .iter()
        .filter(|p| p.dtype == Dtype::U4g64tiled)
        .map(|p| (p.name.clone(), p.shape.clone()))
        .collect()
}

/// **THE CENSUS, PRINTED.** One line per `mlxu4` SKU: how many `U4g64` code
/// planes it interns, and how many of those are two-dimensional — which is
/// to say, how many the tiled point could be made to read.
#[test]
fn the_census_says_which_rows_have_a_two_dimensional_u4_plane() {
    for (sku, trace) in mlxu4_rows() {
        let planes = u4_planes(&trace);
        let flat = planes.iter().filter(|(_, s)| s.len() == 2).count();
        let banks = planes.len() - flat;

        // Which distinct weights, by the name with the layer index taken
        // out — a 48-layer text says the same three things 48 times.
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for (name, shape) in &planes {
            let kind: String = name
                .split('.')
                .map(|part| {
                    if part.parse::<u64>().is_ok() {
                        "N"
                    } else {
                        part
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
            *kinds.entry(format!("{kind} {shape:?}")).or_default() += 1;
        }

        println!(
            "{sku}: {flat} flat u4 planes, {banks} banked, {} tiled",
            tiled_planes(&trace).len(),
        );
        for (kind, count) in kinds {
            println!("    {count:4} x {kind}");
        }
    }
}

/// **THE FLIP HAS NO TARGET IN THE ONE FAMILY WITH A CHECKPOINT.**
///
/// Every `U4g64` plane `qwen38-flash-mlxu4-kv-bf16` interns is a routed
/// expert bank, and every one of them is three-dimensional. There is no
/// projection to declare `U4g64tiled`, and declaring a bank one is the panic
/// `Weight::planes` names.
#[test]
fn the_flash_row_spends_its_four_bits_on_banks_and_nothing_else() {
    let rows = mlxu4_rows();
    let (_, trace) = rows
        .iter()
        .find(|(sku, _)| *sku == "qwen38-flash-mlxu4-kv-bf16")
        .expect("the flash mlxu4 row ships");
    let planes = u4_planes(trace);

    assert!(
        !planes.is_empty(),
        "the flash row interns no four-bit plane at all, which is not the \
         checkpoint this family reads"
    );
    let flat: Vec<_> = planes.iter().filter(|(_, s)| s.len() == 2).collect();
    assert!(
        flat.is_empty(),
        "the flash row interns {} two-dimensional U4g64 planes ({:?}) — the \
         census this file records says it interns none, and if that changed \
         the tiled flip has a target here after all",
        flat.len(),
        flat.iter().map(|(n, _)| n).take(4).collect::<Vec<_>>(),
    );
    for (name, shape) in &planes {
        assert_eq!(
            shape.len(),
            3,
            "`{name}` is a four-bit plane of rank {}, which is neither the \
             projection the tiled point reads nor the bank the select does",
            shape.len(),
        );
        assert!(
            name.contains("experts_"),
            "`{name}` is a four-bit bank that is not a routed expert — the \
             census says the flash row's four bits are the expert banks alone"
        );
    }
}

/// **AND THE RECIPE WAS TAKEN, HERE** (§J4b's flip, landed).
///
/// `qwen_3::Model::new` declares a `proj` width beside its `gate` one:
/// `U4g64tiled` wherever the stack is `U4g64`, spent on the dense
/// projections the forward hands to `ops::linear::matmul` — the attention's
/// four, the GDN's three, the dense MLP's two and the routed layer's shared
/// expert. What it is NOT spent on is the three sets that have no tiled
/// reader, and this test is the statement of that boundary:
///
/// ```text
/// embed              the affine gather, and for a tied row the head too
/// experts_gate_up    three-dimensional, read by the grouped select
/// experts_down
/// mtp.*              read as slices of one stored tensor
/// ```
///
/// So the smallest row the flip reaches interns its projections TILED and
/// keeps exactly one row-major four-bit plane: the embedding.
#[test]
fn the_smallest_row_the_flip_reaches_declares_its_projections_tiled() {
    let rows = mlxu4_rows();
    let (_, trace) = rows
        .iter()
        .find(|(sku, _)| *sku == "qwen35-d0.8b-mlxu4-kv-bf16")
        .expect("the qwen3.5 0.8b mlxu4 row ships");

    let tiled = tiled_planes(trace);
    assert!(
        !tiled.is_empty(),
        "the flip is not on this row: it interns no `U4g64tiled` plane"
    );
    for (name, shape) in &tiled {
        assert_eq!(
            shape.len(),
            2,
            "`{name}` is declared tiled at rank {}, and the tiled point reads a \
             two-dimensional weight",
            shape.len(),
        );
        assert!(
            !name.starts_with("embed"),
            "`{name}` is the embedding declared tiled — the affine gather has no \
             tiled reader, and §J4b's recipe says never"
        );
    }

    // And what is LEFT four-bit and row-major is the embedding, alone. This
    // row ties its head to the table, so that one plane is both the gather's
    // and the readout's — which is exactly why it cannot move.
    let flat: Vec<_> = u4_planes(trace)
        .into_iter()
        .filter(|(_, s)| s.len() == 2)
        .collect();
    assert_eq!(
        flat.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        vec!["embed"],
        "the row-major four-bit remainder of this row is not the embedding alone"
    );
    println!(
        "qwen35-d0.8b-mlxu4-kv-bf16: {} tiled planes, row-major u4 remainder {:?}",
        tiled.len(),
        flat.iter().map(|(n, s)| (n, s)).collect::<Vec<_>>(),
    );
}
