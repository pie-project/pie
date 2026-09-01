use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint_dsl::Error;
use model_dsl::Platform;

#[test]
fn every_sku_names_a_checkpoint_it_can_be_built_from() {
    let mut faults = Vec::new();
    let catalog: BTreeSet<&str> = models::catalog().into_iter().map(|(sku, ..)| sku).collect();

    let mut rows: BTreeMap<&str, usize> = BTreeMap::new();
    for (sku, ..) in models::imports() {
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
    let catalog: BTreeSet<&str> = models::catalog().into_iter().map(|(sku, ..)| sku).collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (sku, _) in models::template::templates() {
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
fn every_sku_names_exactly_one_tokenizer_contract() {
    let mut faults = Vec::new();
    let catalog: BTreeSet<&str> = models::catalog().into_iter().map(|(sku, ..)| sku).collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (sku, _) in models::tokenizer::contracts() {
        if !catalog.contains(sku) {
            faults.push(format!(
                "a `TOKENIZERS` row states what `{sku}` reads from its \
                 vocabulary and the catalog ships no such SKU"
            ));
        }
        *counts.entry(sku).or_default() += 1;
    }

    for sku in &catalog {
        match counts.get(sku).copied().unwrap_or(0) {
            1 => {}
            0 => faults.push(format!(
                "`{sku}` ships and names no tokenizer contract, so serve boot \
                 could not check its vocabulary demands"
            )),
            n => faults.push(format!(
                "`{sku}` names {n} tokenizer contracts; `contract_of` returns \
                 the first and the rest are unreachable"
            )),
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn a_sku_name_states_the_world_its_row_ships() {
    let mut faults = Vec::new();

    for (sku, tp, ..) in models::catalog() {
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

const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// **A PLATFORM MOVES NO PARAM'S VALUE, AND MAY MOVE ITS ARRANGEMENT** (§J4c).
///
/// This asserted `first.params == other.params` outright, under the sentence
/// *"an artifact can no longer be one file"*. The first half was the invariant
/// worth keeping and the second was one step too far, and §M had already taken
/// it back before this test was written: a tier key is "a function of the
/// RECIPE — backend, tensor parallelism, precision", so a `.zt` is
/// SETUP-SPECIFIC and one text is legitimately two artifacts. `Dtype::placed`
/// is that ruling reaching a plane — `U4g64tiled`'s `repr` IS `U4g64`'s,
/// because a repack moves no value — and `Platform::placement` is what
/// resolves one for the shell that will read it.
///
/// So the equality below is by FIELD, and the dtype column is checked for what
/// actually has to hold:
///
/// - **the same param, in the same place**: name, order, count, shard and
///   source are the artifact's identity and no placement touches them;
/// - **the same algebra**: two platforms' dtypes for one param must share a
///   [`Dtype::repr`], which is exactly "the codes mean the same numbers". A
///   platform that moved a param from `U4g64` to `Mxfp4` would be two models
///   and this is what says so;
/// - **and the same rectangle unless a placement pads it**: a placed plane may
///   band its output axis (`Weight::planes`' tiled arm), so a shape that
///   differs is admitted only between two dtypes that already differ, and
///   refused between two that agree.
#[test]
fn platforms_move_only_a_params_arrangement() {
    for (sku, _, trace, _) in models::catalog() {
        let first = trace(PLATFORMS[0]);
        for platform in &PLATFORMS[1..] {
            let other = trace(*platform);
            assert_eq!(
                first.params.len(),
                other.params.len(),
                "`{sku}` declares {} params on {:?} and {} on {platform:?}; a \
                 placement is an arrangement and never a plane more or less",
                first.params.len(),
                PLATFORMS[0],
                other.params.len(),
            );
            for (a, b) in first.params.iter().zip(&other.params) {
                let at = format!("`{sku}`'s `{}` on {:?} vs {platform:?}", a.name, PLATFORMS[0]);
                assert_eq!(a.name, b.name, "{at}: two platforms name it differently");
                assert_eq!(a.shard, b.shard, "{at}: the rank cut moved");
                assert_eq!(a.source, b.source, "{at}: the provenance moved");
                assert_eq!(
                    a.dtype.repr(),
                    b.dtype.repr(),
                    "{at}: declared {:?} and {:?}, which are not one algebra — a \
                     platform may choose an ARRANGEMENT of a plane's bytes and \
                     never what they mean",
                    a.dtype,
                    b.dtype,
                );
                if a.dtype == b.dtype {
                    assert_eq!(
                        a.shape, b.shape,
                        "{at}: one dtype and two rectangles",
                    );
                }
            }
            assert_eq!(
                first.caches, other.caches,
                "`{sku}` declares different caches on {platform:?} than on {:?}",
                PLATFORMS[0],
            );
        }
    }
}

/// **EVERY PARAM A TRACE DECLARES IS IN AN ORDER ITS PLATFORM READS** (§J4c).
///
/// The general rule this wave installs, asserted over the whole catalog rather
/// than over the family that found it. A model text states which of its
/// weights an arrangement is LEGAL on — the tiled point serves `y = act x W^T`
/// over a two-dimensional weight and nothing else, so an embedding's gather
/// and a routed expert bank are left row-major by name — and
/// `model_dsl::place` states whether the setup being traced for reads it.
///
/// What it is holding against: `qwen_3::model`'s projection flip stated
/// `U4g64tiled` for every platform, and the seven `*-mlxu4-*` rows of that
/// family stopped serving on Metal — a raw snapshot refused at
/// `validate_target_support` and a converted artifact loaded in 0.1s and
/// answered nonsense, because `kernels_metal::linear::quant` indexes an affine
/// bank row-major and has no fragment-order twin. That was one text's bug and
/// this is the shape of it, so the next text that reaches for a placement is
/// caught here and not by a first light.
#[test]
fn no_trace_declares_a_plane_its_platform_cannot_read() {
    let mut faults = Vec::new();
    for (sku, _, trace, _) in models::catalog() {
        for platform in PLATFORMS {
            for param in trace(platform).params {
                if !platform.reads_placement(param.dtype) {
                    faults.push(format!(
                        "`{sku}` declares `{}` as {:?} in its {platform:?} trace, \
                         and that shell has no reader for the arrangement",
                        param.name, param.dtype,
                    ));
                }
            }
        }
    }
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
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
    let mut sharded = 0usize;
    for (sku, tp, import) in models::imports() {
        let refusal = match import(&src, tp) {
            Ok(_) => {
                faults.push(format!(
                    "`{sku}` states a whole contract over a checkpoint holding \
                     one tensor no model reads, so its import table never asked \
                     the file what it holds"
                ));
                continue;
            }
            Err(Error::Missing(why)) => why.to_string(),
            Err(Error::Illegible { detail, .. }) => detail,
            Err(why @ Error::Incompatible { .. }) => {
                faults.push(format!(
                    "`{sku}` refuses a checkpoint that holds nothing it reads \
                     with `{why}`, and a file that states none of its planes \
                     is missing them, not storing them in another \
                     representation"
                ));
                continue;
            }
        };
        // **A SHARDED ROW REFUSES FOR A DIFFERENT REASON AND IS COUNTED AS
        // ONE.** An import states the WHOLE checkpoint, so a row built for
        // more than one rank stops at the first weight it reads and never gets
        // as far as asking this file what it holds — which is a true refusal
        // and not the one this test is about. Separating them is what keeps
        // the pass honest: without it, the day every row started refusing at
        // the width, this test would still be green and would be asserting
        // nothing at all about the checkpoint.
        if tp > 1 {
            assert!(
                refusal.contains("WHOLE checkpoint"),
                "`{sku}` is built for {tp} ranks, so it should refuse at the \
                 width before it reads anything, and it refused with: {refusal}"
            );
            sharded += 1;
        } else if refusal.contains("WHOLE checkpoint") {
            faults.push(format!(
                "`{sku}` is a one-rank row and refused at the width: {refusal}"
            ));
        }
    }
    assert!(
        sharded > 0,
        "no import row is built for more than one rank, so the arm above is \
         dead — if the catalog lost its sharded rows, delete it"
    );

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
