//! **THE FIRST EXTERNAL READING OF THIS FAMILY.** Every fidelity claim dsv4
//! made before this file was self-consistency — streamed == resident, warm ==
//! cold, one road == another — and the full 89.9 GiB model answered
//! incoherently through all of them. This file is the other half: the
//! `mini-l5-e16` snapshot (five of the forty-three layers, sixteen of the 256
//! experts, real geometry) fired over a probe battery and read against
//! `scripts/dsv4_mini_parity_ref.py`, an MLX transcription of
//! `ssd-moe/deepseek-v4-flash-mlx`'s oracle — which was itself graded against
//! the official ds4 dumps — driven by the miniature's own config.
//!
//! # What it found, the first time it ran
//!
//! 0 of 239 teacher-forced positions agreed. Six deviations from the
//! official model were attributed by emulating each in the reference until
//! the reference met pie at the bf16 floor (`--pie ogroups,rope,comp_rope0,
//! hash,trunk,no_orope`): the o-projection summed the `o_groups` slices and
//! projected once where the official `einsum("bsgd,grd->bsgr")` is
//! block-diagonal; the compressor layers roped q/kv at `rope_theta` without
//! YaRN where the official `Attention.__init__` uses `compress_rope_theta`
//! with it; the pooled entries roped at the base theta too; the hash layers
//! weighted their experts uniformly where `Gate.forward` gathers the
//! sqrt-softplus scores on every layer; the trunk summed its streams where
//! `hc_head` folds them under learned sigmoid gates; and the attention output
//! was never un-rotated (`apply_rotary_emb(o[..., -rd:], freqs, True)` — MLA's
//! latent is both key and value). With the six fixed: 233 of 239, every
//! remaining flip a near-tie the reference itself decides by under a tenth
//! of a logit.
//!
//! # Shape
//!
//! `tests/dsv4-parity/reference.json` carries, per probe, the reference's
//! **top-5 (id, logit)** at every teacher-forced position and at every greedy
//! step (the full rows are 129 280 wide and are not committed). Two arms per
//! probe:
//!
//! * **teacher-forced**, one token per fire in a fresh slot, so every
//!   position's logits come back — the decode class over the whole prompt;
//! * **prefill + greedy**, the whole prompt in one fire and `steps` decodes —
//!   the prefill class, then the decode class over pie's OWN continuation.
//!
//! **THE BAR.** At every teacher-forced position pie's argmax is the
//! reference's, or the reference's own margin between its top choice and
//! pie's choice is under [`NEAR_TIE`] — a rounding-scale perturbation flips
//! only a near-tie, and a flip won by more than that is a fault. The greedy
//! arm is read the same way at every step up to the first divergence and
//! REPORTED after it: once pie's own token differs, the two continuations
//! condition on different prefixes and stop being comparable. No token index
//! is pinned; the split point is printed, not asserted.
//!
//! **NOT A LONG-HORIZON GATE.** Prompt plus steps stay under 128 tokens
//! because the reference models no compressed rows on a ratio-128 layer and
//! no indexer selection (`index_topk` 512 is far beyond every row here); the
//! ratio-128 compressor and the top-k branch are outside what this file can
//! say anything about.
//!
//! **DUMP MODE.** `PIE_DSV4_PARITY_OUT=DIR` also writes every full row
//! (`NAME.pie.{tf,gen}.f32`, `NAME.pie.json`) for
//! `scripts/dsv4_mini_parity_compare.py`, which is how the attribution above
//! was measured and how the next fault will be.
//!
//! ```text
//! cargo test -p engine-metal --release \
//!   --test the_two_bit_miniature_is_read_against_its_reference -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "dsv4-flash-u4g64-u2g64-kv-bf16";
const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

/// The reference fixture, relative to the workspace root.
const REFERENCE: &str = "tests/dsv4-parity/reference.json";

/// A flip is admitted only where the reference itself decided by less than
/// this many logits between its top choice and the one pie made. Measured:
/// after the six fixes every flip in the battery sits under 0.1, and the
/// rows' bf16 noise floor is ~1e-2 in KL; a fault of the kind this file was
/// written for moves logits by tens.
const NEAR_TIE: f32 = 0.5;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && container(path).is_some();
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
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
    let trace = (models::sku(SKU).expect("the catalog ships the 2-bit SKU").trace)(Platform::Metal);
    let container = container(checkpoint).expect("the snapshot holds a tensor container");
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = models::sku(SKU)
        .expect("the catalog ships an import for the SKU")
        .contract(&source, Platform::Metal)
        .expect("the 2-bit SKU's import contract fits the real DQ checkpoint");
    drop(source);
    let booted = Instant::now();
    let shell = Shell::load(Boot {
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

/// The reference's top-k at one position: `(id, logit)`, best first.
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

/// One position read against the reference's top-k: agreement, or the
/// reference's margin between its top and pie's choice when they differ.
/// `None` where pie chose something outside the reference's top-k — a gap
/// wider than the fixture can measure, and a fault by construction.
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
    let steps = fixture["steps"].as_u64().expect("the fixture states its steps") as usize;
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
        let tf_top = probe["tf_top"].as_array().expect("teacher-forced top-k rows");
        let gen_top = probe["gen_top"].as_array().expect("greedy top-k rows");
        assert_eq!(tf_top.len(), ids.len(), "{name}: one reference row per prompt token");
        let started = Instant::now();

        // Arm 1: teacher-forced, one token per fire.
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

        // Arm 2: prefill the prompt in one fire, then greedy decode.
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
        // The prefill row is read against the reference outright (same
        // prefix); each later row only while pie's continuation is still the
        // reference's, because the first divergence changes the prefix.
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
