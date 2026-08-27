use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use model::contract::ModelError;
use model_dsl::Plane;

#[test]
fn every_sku_names_a_checkpoint_it_can_be_built_from() {
    let mut faults = Vec::new();
    let catalog: BTreeSet<&str> = model::catalog().into_iter().map(|(sku, ..)| sku).collect();

    let mut rows: BTreeMap<&str, usize> = BTreeMap::new();
    for (sku, _) in model::imports() {
        if !catalog.contains(sku) {
            faults.push(format!(
                "an `IMPORTS` row states where `{sku}` comes from and the \
                 catalog ships no such SKU"
            ));
        }
        *rows.entry(sku).or_default() += 1;
    }

    for sku in &catalog {
        match rows.get(sku).copied().unwrap_or(0) {
            1 => {}
            0 => faults.push(format!(
                "`{sku}` ships and names no import contract, so it can be \
                 selected and never obtained"
            )),
            n => faults.push(format!(
                "`{sku}` names {n} import rows; a SKU reads a checkpoint one \
                 way, and `import_of` returns whichever is written first"
            )),
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn every_sku_names_exactly_one_chat_template() {
    let mut faults = Vec::new();
    let catalog: BTreeSet<&str> = model::catalog().into_iter().map(|(sku, ..)| sku).collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (sku, _) in model::template::templates() {
        if !catalog.contains(sku) {
            faults.push(format!(
                "a `TEMPLATES` row states how `{sku}` is talked to and the \
                 catalog ships no such SKU"
            ));
        }
        *counts.entry(sku).or_default() += 1;
    }

    for sku in &catalog {
        match counts.get(sku).copied().unwrap_or(0) {
            1 => {}
            0 => faults.push(format!(
                "`{sku}` ships and names no chat template, so a runtime that \
                 loaded it could not write it a turn"
            )),
            n => faults.push(format!(
                "`{sku}` names {n} chat templates; `template_of` returns the \
                 first and the rest are unreachable"
            )),
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn a_sku_name_states_the_world_its_row_ships() {
    let mut faults = Vec::new();

    for (sku, tp, ..) in model::catalog() {
        let named = match sku.rsplit_once("-tp") {
            Some((_, ranks)) => ranks.parse::<u32>().unwrap_or_else(|why| {
                panic!("`{sku}` ends in a world of `{ranks}` ranks, which is no number: {why}")
            }),
            None => 1,
        };
        if named != tp {
            faults.push(format!(
                "`{sku}` names a world of {named} rank(s) and its catalog row \
                 ships tp {tp}; the name a runtime selects by and the world it \
                 gets are the same fact"
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn planes_do_not_move_a_param() {
    let planes = [Plane::Cuda, Plane::Metal, Plane::Wgpu, Plane::Vulkan];

    for (sku, _, trace, _) in model::catalog() {
        let first = trace(planes[0]);
        for plane in &planes[1..] {
            let other = trace(*plane);
            assert_eq!(
                first.params, other.params,
                "`{sku}` declares different weights on {plane:?} than on {:?}; \
                 an artifact can no longer be one file",
                planes[0],
            );
            assert_eq!(
                first.caches, other.caches,
                "`{sku}` declares different caches on {plane:?} than on {:?}",
                planes[0],
            );
        }
    }
}

#[test]
fn every_import_row_reads_the_checkpoint_it_is_handed() {
    let dir = scratch();
    let path = dir.join("holds-nothing.zt");
    write_a_checkpoint_of_one_stranger(&path);
    let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
        panic!(
            "{}: the checkpoint just written does not open: {why}",
            path.display()
        )
    });

    let mut faults = Vec::new();
    for (sku, import) in model::imports() {
        match import(&src) {
            Ok(_) => faults.push(format!(
                "`{sku}` states a whole contract over a checkpoint holding one \
                 tensor no model reads, so its import table never asked the \
                 file what it holds"
            )),
            Err(ModelError::Missing(_) | ModelError::Illegible { .. }) => {}
            Err(why @ ModelError::Incompatible { .. }) => {
                faults.push(format!(
                    "`{sku}` refuses a checkpoint that holds nothing it reads \
                     with `{why}`, and a file that states none of its planes \
                     is missing them, not storing them in another \
                     representation"
                ));
            }
        }
    }

    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn scratch() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir().join(format!(
        "model_import_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

fn write_a_checkpoint_of_one_stranger(path: &Path) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .object("a.tensor.no.model.in.this.catalog.reads", |o| {
            o.shape(vec![1u64])
                .part("data", |p| p.dtype(ztensor::DType::U8).bytes(&[0u8]))
        })
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}
