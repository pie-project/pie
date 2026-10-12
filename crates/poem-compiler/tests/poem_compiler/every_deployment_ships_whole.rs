use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use poem::import::Error;

#[test]
fn every_deployment_ships_whole_every_case() {
    a_deployment_name_states_the_world_its_row_ships();
    every_import_row_reads_the_checkpoint_it_is_handed();
}

fn a_deployment_name_states_the_world_its_row_ships() {
    let mut faults = Vec::new();

    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
        let (deployment, tp) = (row.name.as_str(), row.deploy.tp);
        let named = match deployment.rsplit_once("-tp") {
            Some((_, ranks)) => ranks.parse::<u32>().unwrap_or_else(|why| {
                panic!(
                    "`{deployment}` ends in a world of `{ranks}` ranks, which is no number: {why}"
                )
            }),
            None => 1,
        };
        if named != tp {
            faults.push(format!(
                "`{deployment}` names a world of {named} rank(s) and its catalog row \
                 ships tp {tp}; the name a runtime selects by and the world it \
                 gets are the same fact"
            ));
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

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
    for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
        let (deployment, tp) = (row.name.as_str(), row.deploy.tp);
        let refusal = match row.contract(&src, poem::Platform::Cuda) {
            Ok(_) => {
                faults.push(format!(
                    "`{deployment}` states a whole contract over a checkpoint holding \
                     one tensor no model reads, so its import table never asked \
                     the file what it holds"
                ));
                continue;
            }
            Err(Error::Missing(why)) => why.to_string(),
            Err(Error::Illegible { detail, .. }) => detail,
            Err(why @ Error::Incompatible { .. }) => {
                faults.push(format!(
                    "`{deployment}` refuses a checkpoint that holds nothing it reads \
                     with `{why}`, and a file that states none of its planes \
                     is missing them, not storing them in another \
                     representation"
                ));
                continue;
            }
        };
        if tp > 1 {
            assert!(
                refusal.contains("WHOLE checkpoint"),
                "`{deployment}` is built for {tp} ranks, so it should refuse at the \
                 width before it reads anything, and it refused with: {refusal}"
            );
            sharded += 1;
        } else if refusal.contains("WHOLE checkpoint") {
            faults.push(format!(
                "`{deployment}` is a one-rank row and refused at the width: {refusal}"
            ));
        }
    }
    assert!(
        sharded > 0,
        "`poem_compiler::catalog::splits()` yields no row of more than one rank, so the sharded arm \
         above checked nothing"
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
        .add(
            "a.tensor.no.model.in.this.catalog.reads",
            vec![1u64],
            ztensor::Leaf::U8,
            &[0u8],
        )
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}
