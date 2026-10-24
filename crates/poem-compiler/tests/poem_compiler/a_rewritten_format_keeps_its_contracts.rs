//! A package whose formats are rewritten to say the same thing another way
//! keeps its import contracts: run once with `PIE_CONTRACT_SNAPSHOTS=<dir>`
//! before the rewrite to record, for each checkpoint under
//! `PIE_CHECKPOINTS` (one directory a checkpoint; its safetensors may be
//! sparse, their headers all a contract reads), what every one-rank
//! deployment reads it into (a contract the checkpoint compiles a load plan
//! from, as `pie model import` compiles it) or why it refuses; run again
//! after to hold each to the recorded one, byte for byte.
//!
//! `cargo test -p pie-poem-compiler --test poem_compiler -- --ignored a_rewritten_format`

use std::path::PathBuf;

use poem_compiler::catalog::deployments;
use poem_ir::Platform;

#[test]
#[ignore = "a migration's check, run by hand around a rewrite"]
fn a_rewritten_format_keeps_its_contracts() {
    let snapshots =
        PathBuf::from(std::env::var("PIE_CONTRACT_SNAPSHOTS").expect("PIE_CONTRACT_SNAPSHOTS"));
    let checkpoints = PathBuf::from(std::env::var("PIE_CHECKPOINTS").expect("PIE_CHECKPOINTS"));
    std::fs::create_dir_all(&snapshots).expect("the snapshot dir");
    let (mut recorded, mut held, mut differ) = (0, 0, Vec::new());
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&checkpoints)
        .expect("the checkpoints dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for dir in dirs {
        let src = match checkpoint::file::snapshot::open(&dir) {
            Ok(src) => src,
            Err(why) => panic!("{}: {why}", dir.display()),
        };
        let metadata = checkpoint::file::read::parse_metadata(&dir)
            .unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
        // What `pie model import` compiles a contract against: a contract the
        // checkpoint's own bytes cannot deliver is no reading of it.
        let target = checkpoint::plan::StorageTarget {
            tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
            max_tile_bytes: 64 << 20,
            ..checkpoint::plan::StorageTarget::default()
        };
        let checkpoint = dir.file_name().expect("a name").to_string_lossy().to_string();
        let mut lines = Vec::new();
        let mut renames: std::collections::BTreeMap<String, Vec<(String, String)>> =
            std::collections::BTreeMap::new();
        for deployment in deployments().filter(|d| d.deploy.tp == 1) {
            let (read, renamed) = poem::star::renamed(|| deployment.contract(&src, Platform::Cuda));
            renames.insert(deployment.name.clone(), renamed);
            let said = match read {
                Ok(contract) => match checkpoint::plan::compile(&metadata, &contract, target.clone()) {
                    Ok(_) => serde_json::to_string(&contract).expect("a contract serializes"),
                    Err(why) => format!("refused: no plan: {why}"),
                },
                Err(why) => format!("refused: {why}").replace('\n', "\\n"),
            };
            lines.push(format!("{}\t{said}", deployment.name));
        }
        let now = lines.join("\n");
        // A rewrite may word its refusals anew: `PIE_CONTRACT_REFUSALS=loose`
        // holds only that a deployment refuses, not why.
        let loose = std::env::var("PIE_CONTRACT_REFUSALS").is_ok_and(|v| v == "loose");
        let shown = |text: &str| -> String {
            if !loose {
                return text.to_string();
            }
            text.lines()
                .map(|l| match l.split_once("\trefused: ") {
                    Some((name, _)) => format!("{name}\trefused"),
                    None => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let file = snapshots.join(format!("{checkpoint}.txt"));
        match std::fs::read_to_string(&file) {
            Err(_) => {
                std::fs::write(&file, &now).expect("records");
                recorded += 1;
            }
            Ok(then) if shown(&renaming(&then, &renames)) == shown(&renaming(&now, &Default::default())) => held += 1,
            Ok(then) => {
                let (then, now) = (shown(&renaming(&then, &renames)), shown(&renaming(&now, &Default::default())));
                let (then, now): (Vec<&str>, Vec<&str>) =
                    (then.lines().collect(), now.lines().collect());
                let moved: Vec<&str> = now
                    .iter()
                    .zip(&then)
                    .filter(|(a, b)| a != b)
                    .map(|(a, _)| a.split('\t').next().unwrap_or(""))
                    .collect();
                differ.push(format!("{checkpoint}: {moved:?}"));
            }
        }
    }
    println!("recorded {recorded}, held {held}, differ {}", differ.len());
    assert!(differ.is_empty(), "contracts moved:\n{}", differ.join("\n"));
}

/// The recorded `text` with each weight stated under a name now named by its
/// path (`renames`): its tensors, the planes beside them and what reads them.
fn renaming(
    text: &str,
    renames: &std::collections::BTreeMap<String, Vec<(String, String)>>,
) -> String {
    fn walk(v: &mut serde_json::Value, key: Option<&str>, rename: &dyn Fn(&str) -> Option<String>) {
        match v {
            serde_json::Value::String(s) if matches!(key, Some("name" | "Out" | "of" | "zero_points")) => {
                if let Some(new) = rename(s) {
                    *s = new;
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, key, rename);
                }
            }
            serde_json::Value::Object(fields) => {
                for (k, item) in fields.iter_mut() {
                    walk(item, Some(k.as_str()), rename);
                }
            }
            _ => {}
        }
    }
    text.lines()
        .map(|line| match line.split_once('\t') {
            Some((deployment, json)) if json.starts_with('{') => {
                let mut ours: Vec<&(String, String)> =
                    renames.get(deployment).into_iter().flatten().collect();
                ours.sort_by_key(|(stated, _)| std::cmp::Reverse(stated.len()));
                let rename = |name: &str| -> Option<String> {
                    ours.iter().find_map(|(stated, path)| {
                        let plane = name.strip_prefix(stated.as_str())?;
                        (plane.is_empty() || plane.starts_with('.')).then(|| format!("{path}{plane}"))
                    })
                };
                let mut v: serde_json::Value = serde_json::from_str(json).expect("a recorded contract");
                walk(&mut v, None, &rename);
                format!("{deployment}\t{}", serde_json::to_string(&v).expect("serializes"))
            }
            _ => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
