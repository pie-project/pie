use std::collections::BTreeSet;
use std::path::PathBuf;

use poem::Platform;
use poem_compiler::catalog::repository;

fn fixtures() -> Vec<(String, serde_json::Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/poem_compiler/adopted");
    let mut out: Vec<(String, serde_json::Value)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|why| panic!("{}: {why}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let id = p.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).unwrap();
            (id, serde_json::from_str(&text).unwrap())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn first_difference(a: &serde_json::Value, b: &serde_json::Value, at: &str) -> Option<String> {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            keys.into_iter().find_map(|k| {
                first_difference(
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    &format!("{at}.{k}"),
                )
            })
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{at}: {} entries, adopted {}", x.len(), y.len()));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .find_map(|(i, (x, y))| first_difference(x, y, &format!("{at}[{i}]")))
        }
        _ if a == b => None,
        _ => Some(format!("{at}: listed {a}, adopted {b}")),
    }
}

#[test]
fn every_adopting_family_reproduces_its_listed_models() {
    let catalog = repository();
    let fixtures = fixtures();
    let mut faults = Vec::new();
    let mut covered = BTreeSet::new();
    for (id, fixture) in &fixtures {
        let Some((package, listed)) = catalog.model(id) else {
            faults.push(format!("{id}: the catalog lists no such model"));
            continue;
        };
        covered.insert(package.name().to_string());
        if !package.adopts() {
            faults.push(format!("{id}: `{}` states no `adopt`", package.name()));
            continue;
        }
        let source = fixture["source"].as_str().unwrap_or(id);
        let adoption = match package.adopt(
            "roundtrip",
            id,
            &fixture["config"],
            &fixture["special"],
            source,
        ) {
            Ok(adoption) => adoption,
            Err(why) => {
                faults.push(format!("{id} ({source}): {why:#}"));
                continue;
            }
        };
        let manifest = adoption.package.manifest();
        let Some(adopted) = manifest.model(id) else {
            faults.push(format!("{id}: the adoption holds no model `{id}`"));
            continue;
        };
        for (what, same) in [
            ("layers", listed.layers == adopted.layers),
            ("vocab", listed.vocab == adopted.vocab),
            ("arch", listed.arch == adopted.arch),
            ("template", listed.template == adopted.template),
            ("tokenizer", listed.tokenizer == adopted.tokenizer),
            ("parts", listed.parts == adopted.parts),
        ] {
            if !same {
                faults.push(format!(
                    "{id}: the adopted model's {what} is not the listed one's"
                ));
            }
        }
        let deploys: Vec<_> = package
            .manifest()
            .deployments
            .iter()
            .filter(|(of, d)| of == id && d.drafter.is_none())
            .map(|(_, d)| d.clone())
            .collect();
        for deploy in deploys {
            let name = format!("{id}/{deploy:?}");
            let listed = package.trace(id, &deploy, &name, Platform::Cuda);
            let adopted = adoption.package.trace(id, &deploy, &name, Platform::Cuda);
            match (listed, adopted) {
                (Ok(listed), Ok(adopted)) => {
                    let listed = serde_json::to_value(&listed).unwrap();
                    let adopted = serde_json::to_value(&adopted).unwrap();
                    if let Some(at) = first_difference(&listed, &adopted, "trace") {
                        faults.push(format!("{id} at {:?}: {at}", deploy.weights));
                    }
                }
                (Ok(_), Err(why)) => faults.push(format!(
                    "{id} at {:?}: the adoption does not trace: {why:#}",
                    deploy.weights
                )),
                (Err(_), _) => {}
            }
        }
    }
    for package in catalog.packages().iter().filter(|p| p.adopts()) {
        if !covered.contains(package.name()) {
            faults.push(format!(
                "`{}` adopts, but no fixture under adopted/ checks it against a model it lists",
                package.name()
            ));
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}
