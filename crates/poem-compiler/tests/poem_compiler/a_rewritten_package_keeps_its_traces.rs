//! A package rewritten to say the same thing another way keeps its traces:
//! run once with `PIE_TRACE_SNAPSHOTS=<dir>` before the rewrite to record
//! every deployment's trace, and again after it to hold each new trace to
//! the recorded one under `poem_ir::canon`. `PIE_TRACE_PACKAGES` names the
//! packages, comma-separated.
//!
//! `cargo test -p pie-poem-compiler --test poem_compiler -- --ignored a_rewritten_package`

use std::path::PathBuf;

use poem_compiler::catalog::{deployments, splits};
use poem_ir::{Platform, Trace, canon};

#[test]
#[ignore = "a migration's check, run by hand around a rewrite"]
fn a_rewritten_package_keeps_its_traces() {
    let dir = PathBuf::from(std::env::var("PIE_TRACE_SNAPSHOTS").expect("PIE_TRACE_SNAPSHOTS"));
    let packages: Vec<String> = std::env::var("PIE_TRACE_PACKAGES")
        .expect("PIE_TRACE_PACKAGES")
        .split(',')
        .map(str::to_string)
        .collect();
    std::fs::create_dir_all(&dir).expect("the snapshot dir");
    let (mut recorded, mut held, mut differ) = (0, 0, Vec::new());
    // Every listed deployment, and each at every count of ranks it shards
    // across.
    let all = deployments().chain(splits());
    for deployment in all.filter(|d| packages.iter().any(|p| p == d.package.name())) {
        for platform in [Platform::Cuda, Platform::Metal] {
            let file = dir.join(format!("{}.{platform:?}.json", deployment.name));
            let (traced, renames) = poem::star::renamed(|| deployment.try_trace(platform));
            let trace = match traced {
                Ok(trace) => trace,
                // A deployment its platform refuses has no trace to keep; one
                // that traced before and does not now has changed.
                Err(why) if file.exists() => {
                    differ.push(format!(
                        "{} on {platform:?} no longer traces: {why:?}",
                        deployment.name
                    ));
                    continue;
                }
                Err(_) => continue,
            };
            if file.exists() {
                let mut before: Trace =
                    serde_json::from_slice(&std::fs::read(&file).expect("read"))
                        .expect("a recorded trace");
                // A weight now named by its path is the one recorded under
                // its stated name, its planes beside it; the longest stated
                // name a param's begins with is its weight's.
                let mut renames = renames.clone();
                renames.sort_by_key(|(stated, _)| std::cmp::Reverse(stated.len()));
                for param in &mut before.params {
                    for (stated, path) in &renames {
                        if let Some(plane) = param.name.strip_prefix(stated.as_str())
                            && (plane.is_empty() || plane.starts_with('.'))
                        {
                            param.name = format!("{path}{plane}");
                            break;
                        }
                    }
                }
                match canon::difference(&before, &trace) {
                    None => held += 1,
                    Some(why) => differ.push(format!("{} on {platform:?}: {why}", deployment.name)),
                }
            } else {
                std::fs::write(&file, serde_json::to_vec(&trace).expect("serialize"))
                    .expect("write");
                recorded += 1;
            }
        }
    }
    eprintln!("recorded {recorded}, held {held}, differ {}", differ.len());
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}
