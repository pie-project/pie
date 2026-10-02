use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Shell, World};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn write_rows(path: &Path, rows: &[Vec<f32>]) {
    let mut file = std::fs::File::create(path).expect("the output file opens");
    let mut bytes = Vec::with_capacity(rows.len() * rows.first().map_or(0, Vec::len) * 4);
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    file.write_all(&bytes).expect("the logits are written");
}

#[test]
fn every_probe_is_dumped() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: this machine publishes no CUDA device");
        return;
    }
    let (Ok(probes), Ok(out)) = (
        std::env::var("PIE_PARITY_PROBES"),
        std::env::var("PIE_PARITY_OUT"),
    ) else {
        eprintln!(
            "not asked: set PIE_PARITY_PROBES and PIE_PARITY_OUT, and PIE_PARITY_ARTIFACT or PIE_PARITY_SNAPSHOT + PIE_PARITY_SKU"
        );
        return;
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("the dump directory exists");
    let steps: usize = std::env::var("PIE_PARITY_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let context: u32 = std::env::var("PIE_PARITY_CONTEXT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let (snapshot, sku, contract) = match (
        std::env::var("PIE_PARITY_ARTIFACT"),
        std::env::var("PIE_PARITY_SNAPSHOT"),
    ) {
        (Ok(artifact), _) => {
            let artifact = PathBuf::from(artifact);
            let stamp = checkpoint::file::serve::stamp_of(&artifact)
                .expect("the artifact reads")
                .expect("the artifact carries a serving stamp");
            let sku = models::sku(&stamp.sku).unwrap_or_else(|| panic!("no SKU {}", stamp.sku));
            let trace = (sku.trace)(Platform::Cuda);
            let source = ztensor_compat::index(&artifact).expect("the artifact opens");
            let contract =
                checkpoint_dsl::own_contract(&source, &trace.params, sku.recipe.tp, Platform::Cuda)
                    .unwrap_or_else(|why| {
                        panic!("the artifact holds every plane of {}: {why}", sku.name)
                    });
            (artifact, sku, contract)
        }
        (_, Ok(snapshot)) => {
            let snapshot = PathBuf::from(snapshot);
            let name = std::env::var("PIE_PARITY_SKU")
                .expect("PIE_PARITY_SKU names the row that reads the snapshot");
            let sku = models::sku(&name).unwrap_or_else(|| panic!("no SKU {name}"));
            let mut shards: Vec<PathBuf> = if snapshot.is_dir() {
                std::fs::read_dir(&snapshot)
                    .expect("the snapshot lists")
                    .filter_map(|e| {
                        let path = e.ok()?.path();
                        let name = path.file_name()?.to_str()?;
                        (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
                    })
                    .collect()
            } else {
                vec![snapshot.clone()]
            };
            shards.sort();
            let source = ztensor_compat::index_all(&shards).expect("the snapshot opens");
            let contract = sku
                .contract(&source, Platform::Cuda)
                .unwrap_or_else(|why| panic!("{name}'s import reads the snapshot: {why}"));
            (snapshot, sku, contract)
        }
        _ => {
            eprintln!("not asked: set PIE_PARITY_ARTIFACT or PIE_PARITY_SNAPSHOT + PIE_PARITY_SKU");
            return;
        }
    };
    let trace = (sku.trace)(Platform::Cuda);
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let budget = |key: &str| -> Option<u64> {
        let text = std::env::var(key).ok()?;
        let text = text.trim();
        for (suffix, unit) in [("GiB", 1u64 << 30), ("MiB", 1u64 << 20)] {
            if let Some(count) = text.strip_suffix(suffix) {
                return Some(
                    (count.trim().parse::<f64>().expect("a unit count") * unit as f64) as u64,
                );
            }
        }
        Some(text.parse().expect("a byte count"))
    };
    let budgets = Budgets {
        device: budget("PIE_PARITY_DEVICE_BUDGET"),
        host: budget("PIE_PARITY_HOST_BUDGET"),
    };
    let residency = {
        let target = checkpoint::plan::StorageTarget::for_backend(
            checkpoint::types::BackendKind::Cuda,
            0,
            1,
        );
        let prospect = engine_cuda::weights::prospect(&trace, &contract, &snapshot, target)
            .unwrap_or_else(|why| panic!("the prospect reads the artifact: {why}"));
        Plan::cut(&prospect.ranking, budgets)
            .unwrap_or_else(|why| panic!("the budgets plan: {why}"))
    };
    let tag = std::env::var("PIE_PARITY_TAG").unwrap_or_else(|_| "pie".to_string());

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        classify: sku.classify,
        trace,
        contract: &contract,
        checkpoint: &snapshot,
        budget: Budget::new(4, context),
        patches: None,
        profile: None,
        page_size: 16,
        context,
        slots: 4,
        pages: 4 * context / 16,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency,
        world: World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    eprintln!(
        "loaded {} in {:.1}s",
        sku.name,
        booted.elapsed().as_secs_f64()
    );

    let battery: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&probes).expect("the probes read")).expect("json");
    let mut slot = 0u32;
    for probe in battery["probes"].as_array().expect("`probes` is a list") {
        let name = probe["name"].as_str().expect("a probe has a name");
        let ids: Vec<u32> = probe["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect();
        let started = Instant::now();

        let tf_slot = slot % 4;
        slot += 1;
        shell.open(tf_slot).expect("the slot opens");
        let mut tf_rows: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        for id in &ids {
            let fed = [*id];
            let got = shell
                .fire(&[Lane {
                    slot: tf_slot,
                    word: word(1),
                    tokens: &fed,
                }])
                .expect("a teacher-forced fire returns");
            tf_rows.push(got.into_iter().next().expect("one row"));
        }

        let gen_slot = slot % 4;
        slot += 1;
        shell.open(gen_slot).expect("the slot opens");
        let got = shell
            .fire(&[Lane {
                slot: gen_slot,
                word: word(ids.len() as u32),
                tokens: &ids,
            }])
            .expect("the prefill fires");
        let mut gen_rows: Vec<Vec<f32>> = vec![got.into_iter().next().expect("one row")];
        let mut produced: Vec<u32> = Vec::with_capacity(steps);
        for _ in 0..steps {
            let nxt = argmax(gen_rows.last().expect("a row"));
            produced.push(nxt);
            let fed = [nxt];
            let got = shell
                .fire(&[Lane {
                    slot: gen_slot,
                    word: word(1),
                    tokens: &fed,
                }])
                .expect("a decode fires");
            gen_rows.push(got.into_iter().next().expect("one row"));
        }

        write_rows(&out.join(format!("{name}.{tag}.tf.f32")), &tf_rows);
        write_rows(&out.join(format!("{name}.{tag}.gen.f32")), &gen_rows);
        let argmaxes: Vec<u32> = tf_rows.iter().map(|r| argmax(r)).collect();
        std::fs::write(
            out.join(format!("{name}.{tag}.json")),
            serde_json::to_string(&serde_json::json!({
                "ids": ids, "argmax": argmaxes, "gen": produced, "vocab": tf_rows[0].len(),
            }))
            .expect("json"),
        )
        .expect("the summary is written");
        eprintln!(
            "  {name}: {} tokens, gen={:?}  ({:.1}s)",
            ids.len(),
            &produced[..produced.len().min(12)],
            started.elapsed().as_secs_f64()
        );
    }
}
