use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Seated, Shell, World};
use poem::{Platform, Request};
use poem_compiler::Budget;

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
            "not asked: set PIE_PARITY_PROBES and PIE_PARITY_OUT, and PIE_PARITY_ARTIFACT or PIE_PARITY_SNAPSHOT + PIE_PARITY_DEPLOYMENT"
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

    let (snapshot, deployment, contract) = match (
        std::env::var("PIE_PARITY_ARTIFACT"),
        std::env::var("PIE_PARITY_SNAPSHOT"),
    ) {
        (Ok(artifact), _) => {
            let artifact = PathBuf::from(artifact);
            let stamp = checkpoint::file::serve::stamp_of(&artifact)
                .expect("the artifact reads")
                .expect("the artifact carries a serving stamp");
            // A stated row serves the artifact's planes if they are its own:
            // `nvfp4a4` reads the planes an `nvfp4` import lands.
            let name = std::env::var("PIE_PARITY_DEPLOYMENT").unwrap_or(stamp.deployment);
            // A row the catalog does not list (a whole model's own export)
            // still reads as its name spells.
            let deployment = poem_compiler::catalog::deployment(&name)
                .or_else(|| {
                    poem_compiler::catalog::repository()
                        .parse(&name)
                        .map(|d| &*Box::leak(Box::new(d)))
                })
                .unwrap_or_else(|| panic!("no deployment {name}"));
            let trace = deployment.trace(Platform::Cuda);
            let source = ztensor_compat::index(&artifact).expect("the artifact opens");
            let contract = poem::import::own_contract(
                &source,
                &trace.params,
                deployment.deploy.tp,
                Platform::Cuda,
            )
            .unwrap_or_else(|why| {
                panic!(
                    "the artifact holds every plane of {}: {why}",
                    deployment.name
                )
            });
            (artifact, deployment, contract)
        }
        (_, Ok(snapshot)) => {
            let snapshot = PathBuf::from(snapshot);
            let name = std::env::var("PIE_PARITY_DEPLOYMENT")
                .expect("PIE_PARITY_DEPLOYMENT names the row that reads the snapshot");
            // A row the catalog does not list (a listed row at another kv,
            // say) still reads as its name spells.
            let deployment = poem_compiler::catalog::deployment(&name)
                .or_else(|| {
                    poem_compiler::catalog::repository()
                        .parse(&name)
                        .map(|d| &*Box::leak(Box::new(d)))
                })
                .unwrap_or_else(|| panic!("no deployment {name}"));
            // As an import opens it: its `config.json` rides along, which a
            // format reads (an int4 export's `quantization_config`).
            let source = checkpoint::file::snapshot::open(&snapshot).expect("the snapshot opens");
            let contract = deployment
                .contract(&source, Platform::Cuda)
                .unwrap_or_else(|why| panic!("{name}'s import reads the snapshot: {why}"));
            (snapshot, deployment, contract)
        }
        _ => {
            eprintln!(
                "not asked: set PIE_PARITY_ARTIFACT or PIE_PARITY_SNAPSHOT + PIE_PARITY_DEPLOYMENT"
            );
            return;
        }
    };
    let trace = deployment.trace(Platform::Cuda);
    let facts = deployment.trace(poem::Platform::Cuda).facts;
    let word = |query_len: u32| facts.word(&Request::new(query_len, false));
    // `PIE_PARITY_DSPARK`: after each greedy step, a drafted block anchored
    // at the step's pick (the deployment's block drafter, read out every
    // row), its logits beside the reference's `forward_spec`.
    let drafter = std::env::var_os("PIE_PARITY_DSPARK").map(|_| {
        trace
            .drafter
            .expect("PIE_PARITY_DSPARK reads a deployment that drafts a block")
    });
    let block_word = |rows: u32| facts.word(&Request::new(rows, false).drafting_a_block(true));
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

    // Each process times its dense GEMM tactics afresh, and a near tie can
    // land on another cuBLASLt algorithm (another rounding) run to run;
    // `PIE_PARITY_CACHE` names a directory the chosen tactics persist in,
    // so two runs over it read bit for bit alike.
    let tactics = std::env::var_os("PIE_PARITY_CACHE").map(std::path::PathBuf::from);

    // `PIE_PARITY_BATCH` (e.g. "8,64") also reads batched decode: that many
    // lanes, each its own prompt of `PIE_PARITY_BATCH_PREFIX` tokens and a
    // few more, prefilled in one fire and then teacher-forced
    // `PIE_PARITY_BATCH_STEPS` tokens a lane in fires of every lane at once,
    // and each lane again alone. The probes above fire one row at a time, so
    // only this reaches a decode's 2..64-row shapes.
    let batches: Vec<usize> = std::env::var("PIE_PARITY_BATCH")
        .ok()
        .map(|list| {
            list.split(',')
                .map(|b| b.trim().parse().expect("a lane count"))
                .collect()
        })
        .unwrap_or_default();
    let batch_prefix: usize = std::env::var("PIE_PARITY_BATCH_PREFIX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let batch_steps: usize = std::env::var("PIE_PARITY_BATCH_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let lanes_most = batches.iter().copied().max().unwrap_or(0).max(4);
    let lane_prefix = |lane: usize| batch_prefix + (lane * 5) % 61;
    let tokens_most = (0..lanes_most)
        .map(lane_prefix)
        .sum::<usize>()
        .max(context as usize) as u32;
    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        trace,
        contract: &contract,
        checkpoint: &snapshot,
        budget: Budget::new(lanes_most as u32, tokens_most),
        patches: None,
        page_size: 16,
        context,
        slots: lanes_most as u32,
        kv_budget: None,
        memory_share: std::env::var("PIE_PARITY_MEMORY_SHARE")
            .ok()
            .map_or(engine::load::DEFAULT_DEVICE_MEMORY_SHARE, |share| {
                share.parse().expect("a device memory share")
            }),
        ordinal: 0,
        knobs: Knobs {
            // Eager, at the registry defaults; `PIE_PARITY_KERNEL` adds
            // `[kernel]` decisions (a chip profile's, `;`-separated) to
            // read under.
            // `PIE_PARITY_DEVICE_PROFILE` reads under the device's own
            // resolved profile instead (graphs, bodies and side streams as a
            // server runs them), a user file in `PIE_PARITY_PROFILES`
            // overlaid.
            chip: if std::env::var_os("PIE_PARITY_DEVICE_PROFILE").is_some() {
                None
            } else {
                Some(match std::env::var("PIE_PARITY_KERNEL") {
                    Ok(kernel) => engine_cuda::chip::profile(&format!(
                        "[engine]\ngraphs = \"off\"\n[kernel]\n{}\n",
                        kernel.replace(';', "\n")
                    )),
                    Err(_) => engine_cuda::chip::pinned(Graphs::Off, None),
                })
            },
            profiles: std::env::var_os("PIE_PARITY_PROFILES").map(PathBuf::from),
            ..Knobs::default()
        },
        cache_dir: tactics.as_deref(),
        runahead: engine::runahead::Runahead::F1,
        residency,
        world: World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    eprintln!(
        "loaded {} in {:.1}s",
        deployment.name,
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
        let mut draft_rows: Vec<Vec<f32>> = Vec::new();
        let mut drafts: Vec<Vec<u32>> = Vec::new();
        for step in 0..steps {
            let nxt = argmax(gen_rows.last().expect("a row"));
            produced.push(nxt);
            let fed = [nxt];
            // The committed length: a drafted block above it never was.
            let held = (ids.len() + step) as u32;
            let got = shell
                .fire_rows(
                    &[Seated {
                        held: Some(held),
                        ..Seated::of(Lane {
                            slot: gen_slot,
                            word: word(1),
                            tokens: &fed,
                        })
                    }],
                    &[],
                )
                .expect("a decode fires");
            gen_rows.push(got.into_iter().next().expect("one row"));
            if let Some(d) = drafter {
                let mut block = vec![d.mask_token; d.rows as usize];
                block[0] = argmax(gen_rows.last().expect("a row"));
                let rows: Vec<u32> = (0..d.rows).collect();
                let got = shell
                    .fire_rows(
                        &[Seated {
                            held: Some(held + 1),
                            readout: Some(&rows),
                            ..Seated::of(Lane {
                                slot: gen_slot,
                                word: block_word(d.rows),
                                tokens: &block,
                            })
                        }],
                        &[engine::fire::Readout::Rows(rows.clone())],
                    )
                    .expect("a drafted block fires");
                let flat = got.into_iter().next().expect("one lane");
                let vocab = flat.len() / d.rows as usize;
                let picks: Vec<u32> = flat.chunks(vocab).map(argmax).collect();
                draft_rows.extend(flat.chunks(vocab).map(<[f32]>::to_vec));
                drafts.push(picks);
            }
        }

        write_rows(&out.join(format!("{name}.{tag}.tf.f32")), &tf_rows);
        write_rows(&out.join(format!("{name}.{tag}.gen.f32")), &gen_rows);
        let argmaxes: Vec<u32> = tf_rows.iter().map(|r| argmax(r)).collect();
        let mut summary = serde_json::json!({
            "ids": ids, "argmax": argmaxes, "gen": produced, "vocab": tf_rows[0].len(),
        });
        if drafter.is_some() {
            write_rows(&out.join(format!("{name}.{tag}.dspark.f32")), &draft_rows);
            summary["drafts"] = serde_json::json!(drafts);
        }
        std::fs::write(
            out.join(format!("{name}.{tag}.json")),
            serde_json::to_string(&summary).expect("json"),
        )
        .expect("the summary is written");
        eprintln!(
            "  {name}: {} tokens, gen={:?}  ({:.1}s)",
            ids.len(),
            &produced[..produced.len().min(12)],
            started.elapsed().as_secs_f64()
        );
    }

    for &lanes in &batches {
        let started = Instant::now();
        // Each lane's ids: a fixed LCG walk over the vocabulary's body.
        let prompts: Vec<Vec<u32>> = (0..lanes)
            .map(|lane| {
                let mut state = 0x9e37_79b9_u64 ^ ((lane as u64 + 1) * 0x2545_f491);
                (0..lane_prefix(lane) + batch_steps)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        200 + ((state >> 33) % 119_800) as u32
                    })
                    .collect()
            })
            .collect();

        // All lanes at once: one prefill fire, then a fire a step.
        let mut together: Vec<Vec<Vec<f32>>> = vec![Vec::new(); lanes];
        for lane in 0..lanes {
            shell.open(lane as u32).expect("the slot opens");
        }
        // Prefills go a lane a fire (a fire of many lanes' prompts can
        // outgrow the elastic arena); the decode steps after are what this
        // reads.
        for first in 0..lanes {
            let group = first..first + 1;
            let fired = shell
                .fire(
                    &group
                        .clone()
                        .map(|lane| Lane {
                            slot: lane as u32,
                            word: word(lane_prefix(lane) as u32),
                            tokens: &prompts[lane][..lane_prefix(lane)],
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("the batched prefill fires");
            for (lane, row) in group.zip(fired) {
                together[lane].push(row);
            }
        }
        for step in 0..batch_steps {
            let fired = shell
                .fire(
                    &(0..lanes)
                        .map(|lane| {
                            let at = lane_prefix(lane) + step;
                            Lane {
                                slot: lane as u32,
                                word: word(1),
                                tokens: &prompts[lane][at..at + 1],
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("a batched decode fires");
            for (lane, row) in fired.into_iter().enumerate() {
                together[lane].push(row);
            }
        }

        // Each lane alone, the same feed.
        let mut alone: Vec<Vec<Vec<f32>>> = vec![Vec::new(); lanes];
        for lane in 0..lanes {
            shell.open(0).expect("the slot opens");
            let prefix = lane_prefix(lane);
            let fired = shell
                .fire(&[Lane {
                    slot: 0,
                    word: word(prefix as u32),
                    tokens: &prompts[lane][..prefix],
                }])
                .expect("a lone prefill fires");
            alone[lane].push(fired.into_iter().next().expect("one row"));
            for step in 0..batch_steps {
                let at = prefix + step;
                let fired = shell
                    .fire(&[Lane {
                        slot: 0,
                        word: word(1),
                        tokens: &prompts[lane][at..at + 1],
                    }])
                    .expect("a lone decode fires");
                alone[lane].push(fired.into_iter().next().expect("one row"));
            }
        }

        let flat = |rows: &[Vec<Vec<f32>>]| rows.concat();
        write_rows(
            &out.join(format!("batch{lanes}.{tag}.together.f32")),
            &flat(&together),
        );
        write_rows(
            &out.join(format!("batch{lanes}.{tag}.alone.f32")),
            &flat(&alone),
        );
        std::fs::write(
            out.join(format!("batch{lanes}.{tag}.json")),
            serde_json::to_string(&serde_json::json!({
                "lanes": lanes, "rows_a_lane": batch_steps + 1, "vocab": together[0][0].len(),
            }))
            .expect("json"),
        )
        .expect("the summary is written");
        eprintln!(
            "  batch{lanes}: {lanes} lanes x {} rows  ({:.1}s)",
            batch_steps + 1,
            started.elapsed().as_secs_f64()
        );
    }
}
