#![cfg(target_vendor = "apple")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "dsv4-flash-u4g64-u2g64-kv-bf16";
const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

const REFERENCE: &str = "tests/dsv4-parity/reference.json";

const NEAR_TIE: f32 = 0.5;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && container(path).is_some();
    let homes = [std::env::var("HOME").unwrap_or_default()];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(REPO)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| usable(path))
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

fn word(query_len: u32) -> u64 {
    models::deepseek_v4::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn load(checkpoint: &Path, context: u32) -> Shell {
    let trace = (models::sku(SKU)
        .expect("the catalog ships the 2-bit SKU")
        .trace)(Platform::Metal);
    let container = container(checkpoint).expect("the snapshot holds a tensor container");
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = models::sku(SKU)
        .expect("the catalog ships an import for the SKU")
        .contract(&source, Platform::Metal)
        .expect("the 2-bit SKU's import contract fits the real DQ checkpoint");
    drop(source);
    let booted = Instant::now();
    let shell = Shell::load(Boot {
        voxels: None,
        trace,
        contract: &contract,
        checkpoint,
        budget: Budget::new(4, context),
        patches: None,
        profile: None,
        page_size: 16,
        context,
        slots: 4,
        pages: 4 * context / 16,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the 2-bit shell loads");
    eprintln!(
        "loaded {SKU} on {} in {:.1}s",
        shell.device_name(),
        booted.elapsed().as_secs_f64()
    );
    shell
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

fn top(row: &serde_json::Value) -> Vec<(u32, f32)> {
    row.as_array()
        .expect("a top-k row is a list")
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("a top-k entry is [id, logit]");
            (
                pair[0].as_u64().expect("an id") as u32,
                pair[1].as_f64().expect("a logit") as f32,
            )
        })
        .collect()
}

fn read(pie: u32, reference: &[(u32, f32)]) -> Result<(), Option<f32>> {
    let (best, best_logit) = reference[0];
    if pie == best {
        return Ok(());
    }
    match reference.iter().find(|(id, _)| *id == pie) {
        Some((_, logit)) => Err(Some(best_logit - logit)),
        None => Err(None),
    }
}

#[test]
fn every_probe_answers_the_reference_to_the_bf16_floor() {
    if !engine_metal::device::present() {
        eprintln!("skipping: this machine publishes no Metal device");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!("skipping: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(REFERENCE)).expect("the fixture reads"))
            .expect("the fixture is JSON");
    let dump = std::env::var("PIE_DSV4_PARITY_OUT").ok().map(PathBuf::from);
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir).expect("the dump directory exists");
    }
    let steps = fixture["steps"]
        .as_u64()
        .expect("the fixture states its steps") as usize;
    let probes = fixture["probes"].as_array().expect("`probes` is a list");

    let mut shell = load(&checkpoint, 512);
    let mut slot = 0u32;
    let mut next_slot = || {
        let s = slot;
        slot = (slot + 1) % 4;
        s
    };

    let mut faults: Vec<String> = Vec::new();
    let mut agreed = 0usize;
    let mut total = 0usize;
    for probe in probes {
        let name = probe["name"].as_str().expect("a probe has a name");
        let ids: Vec<u32> = probe["ids"]
            .as_array()
            .expect("a probe has ids")
            .iter()
            .map(|v| v.as_u64().expect("an id is an integer") as u32)
            .collect();
        assert!(
            ids.len() + steps < 128,
            "{name}: {} tokens plus {steps} steps reaches a ratio-128 compressed row the \
             reference does not model",
            ids.len()
        );
        let tf_top = probe["tf_top"]
            .as_array()
            .expect("teacher-forced top-k rows");
        let gen_top = probe["gen_top"].as_array().expect("greedy top-k rows");
        assert_eq!(
            tf_top.len(),
            ids.len(),
            "{name}: one reference row per prompt token"
        );
        let started = Instant::now();

        let tf_slot = next_slot();
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
            assert_eq!(got.len(), 1);
            tf_rows.push(got.into_iter().next().expect("one row"));
        }
        let mut flips = Vec::new();
        for (at, row) in tf_rows.iter().enumerate() {
            total += 1;
            match read(argmax(row), &top(&tf_top[at])) {
                Ok(()) => agreed += 1,
                Err(Some(margin)) if margin < NEAR_TIE => {
                    flips.push(format!("@{at} (near-tie, reference margin {margin:.3})"));
                }
                Err(Some(margin)) => faults.push(format!(
                    "{name}: teacher-forced position {at} chose {} where the reference chose {} \
                     by {margin:.3} logits",
                    argmax(row),
                    top(&tf_top[at])[0].0
                )),
                Err(None) => faults.push(format!(
                    "{name}: teacher-forced position {at} chose {}, outside the reference's top-5",
                    argmax(row)
                )),
            }
        }

        let gen_slot = next_slot();
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
        gen_rows.truncate(steps.max(1));
        let reference_gen: Vec<u32> = probe["gen"]
            .as_array()
            .expect("the reference's continuation")
            .iter()
            .map(|v| v.as_u64().expect("an id") as u32)
            .collect();
        let mut diverged: Option<usize> = None;
        for (at, row) in gen_rows.iter().enumerate() {
            match read(argmax(row), &top(&gen_top[at])) {
                Ok(()) => {}
                Err(Some(margin)) if margin < NEAR_TIE => {
                    diverged = Some(at);
                    break;
                }
                Err(Some(margin)) => {
                    faults.push(format!(
                        "{name}: greedy step {at} chose {} where the reference chose {} by \
                         {margin:.3} logits",
                        argmax(row),
                        reference_gen[at]
                    ));
                    diverged = Some(at);
                    break;
                }
                Err(None) => {
                    faults.push(format!(
                        "{name}: greedy step {at} chose {}, outside the reference's top-5",
                        argmax(row)
                    ));
                    diverged = Some(at);
                    break;
                }
            }
        }

        eprintln!(
            "  {name}: {} tokens, teacher-forced flips {:?}; greedy {}; {:.1}s",
            ids.len(),
            flips,
            match diverged {
                Some(at) => format!("forks at step {at} of {steps} (a near-tie)"),
                None => format!("all {steps} steps agree"),
            },
            started.elapsed().as_secs_f64(),
        );

        if let Some(dir) = &dump {
            write_rows(&dir.join(format!("{name}.pie.tf.f32")), &tf_rows);
            write_rows(&dir.join(format!("{name}.pie.gen.f32")), &gen_rows);
            let argmaxes: Vec<u32> = tf_rows.iter().map(|r| argmax(r)).collect();
            let summary = serde_json::json!({
                "ids": ids,
                "argmax": argmaxes,
                "gen": produced,
                "vocab": tf_rows[0].len(),
            });
            std::fs::write(
                dir.join(format!("{name}.pie.json")),
                serde_json::to_string(&summary).expect("json"),
            )
            .expect("the summary is written");
        }
    }
    eprintln!("teacher-forced argmax agreement over the battery: {agreed}/{total}");
    assert!(
        faults.is_empty(),
        "the miniature parts from its reference by more than a near-tie:\n{}\n",
        faults.join("\n")
    );
}
