//! **WHAT A GEMMA-4-26B-A4B LANE ACTUALLY DISAGREES WITH ITS CROWD ABOUT**,
//! separated into the four things it could have been.
//!
//! `throughput_probe`'s `gemma4_26b_a4b_decodes_on_many_lanes_at_once` says
//! that this checkpoint's lanes do not answer in a crowd what they answer
//! alone. That file's own assertion calls a parting of its size "a window
//! reading rows that are not its own", and the row was held open for two
//! campaigns on that suspicion — first as a dequantization width, then as a
//! window / page-table / kv-extent defect on gemma-4's attention.
//!
//! **IT IS NEITHER.** It is `linear::quant`'s vector arm changing its k
//! partition with the fire's row count, and this file is the separation that
//! says so. Every assertion below is BIT-FOR-BIT, because the three geometry
//! faults are the kind that replace an answer rather than perturb it, and the
//! one thing that does move is measured beside them.
//!
//! # The four questions, and which one answers yes
//!
//! | question | answer |
//! |---|---|
//! | A. is a lane's answer a function of its SLOT? | no, bit for bit |
//! | B. do lanes in one fire read each other? | no, bit for bit |
//! | C. does a lane's answer depend on its NEIGHBOURS' extents? | no, bit for bit |
//! | D. does it depend on HOW MANY rode with it? | **yes, at 2 and 4 lanes and not at 1 or 3** |
//!
//! **ONE TEST AND ONE LOAD**, not four: the snapshot is 13.3 GiB on a 32 GiB
//! box, and four cases would be four of them in one process with the
//! allocator's timing deciding whether two are ever live at once.
//!
//! The last row is `linear::quant::qmv_rows_fold`'s divisibility rule and
//! nothing else: it offers a fold rung that DIVIDES the batch, so two and four
//! rows take `affine_qmv_rows` and one and three take `affine_qmv_fast` — and
//! the two points are stamped at different pack widths, which is how k is
//! dealt out to a simdgroup's thirty-two lanes. A window read at the wrong row
//! does not switch itself off at three lanes and back on at four.
//!
//! `.wiki/macos-bench.md` §23 is the campaign write-up,
//! `affine_floor::the_folded_vector_point_lands_the_one_row_bits` is the same
//! fact at the kernel (one bfloat16 ulp in one element in eight thousand), and
//! `kernels_metal::tuning::DeviceTuning::qmv_rows_packs` carries the price of
//! taking it away.
//!
//! # Gating
//!
//! `#[ignore]`d: the snapshot is 13.3 GiB and this box runs one model at a
//! time. It SKIPS at run time when the device or the snapshot is missing.
//!
//! ```text
//! cargo test -p engine-metal --release --test what_the_a4b_crowd_disagrees_about \
//!     -- --ignored --nocapture --test-threads=1
//!
//! PIE_PROBE_TUNING=qmv_rows_packs=2 cargo test -p engine-metal --release \
//!     --test what_the_a4b_crowd_disagrees_about -- --ignored --nocapture
//! ```
//!
//! `PIE_PROBE_TUNING` is `throughput_probe`'s spelling: `key=value` pairs,
//! comma separated, into a `[metal.tuning]` boot document. Under
//! `qmv_rows_packs=2` or `qmv_rows_max=1` the fourth question's answer becomes
//! no as well, and [`the_row_count_is_what_moves_the_answer`] ASSERTS that —
//! so the fix, if one is ever made the default, is gated here the day it lands.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row, its cache directory, and the variable that overrides it —
/// `throughput_probe::GEMMA4_A4B`'s three fields, restated rather than shared
/// for that file's own reason: a fixture two test binaries reach into is a
/// file two lanes of work edit.
const SKU: &str = "gemma4-26b-a4b-mlxu4-kv-bf16";
const REPO: &str = "models--mlx-community--gemma-4-26b-a4b-it-4bit";
const ENV: &str = "PIE_GEMMA4_A4B_SNAPSHOT";

/// `<bos>`, id 2 — `session_c_first_light::Family::bos` is the measurement,
/// and a gemma sequence without it decodes `" la la la la"` in every lane,
/// which is four lanes agreeing about nothing.
const BOS: u32 = 2;

/// **FOUR, BECAUSE FOUR IS WHERE THE QUESTION IS SHARPEST.** The fold rungs
/// are `[2, 4, 8]` and `qmv_rows_max` is 2 here, so 1 and 3 rows take the
/// one-row point and 2 and 4 take the fold — the alternation this file's
/// headline case reads. It is also under `qmm_min_batch`, so no tile fires at
/// any width below and the vector arm is the ONLY thing that moves.
const LANES: u32 = 4;

/// Tokens a sequence may hold. One short prompt plus [`STEPS`] steps.
const CONTEXT: u32 = 256;

/// Decode fires after the prefill. Short on purpose: a bit-exact comparison
/// needs no room to amplify, and this row amplifies at every step anyway.
const STEPS: usize = 8;

/// Four prompts, deliberately of four different LENGTHS — which is what
/// [`a_lane_does_not_depend_on_who_rode_with_it`] varies and what a kv extent
/// taken from the widest lane would be caught by.
const PROMPTS: [&str; LANES as usize] = [
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
    "Two plus two equals",
];

/// **ONE SHELL AT A TIME, PER PROCESS.** `throughput_probe`'s reason: 13.3 GiB
/// of weights beside a kv pool, on a box with 32.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    models::gemma_4::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// `(worst |delta| over the row, where it fell)` — and a zero here is BIT
/// equality, because both rows come back as `f32` widenings of the same
/// `bfloat16` readout.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "two readings of one lane are one width");
    let mut at = 0usize;
    let mut most = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > most {
            most = d;
            at = i;
        }
    }
    (most, at)
}

/// The worst element over a whole reading, step by step.
fn worst_over(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| worst(x, y).0).collect()
}

/// Assert two readings are the same bits at every step, naming the step and
/// the element when they are not.
fn identical(what: &str, a: &[Vec<f32>], b: &[Vec<f32>]) {
    for (step, (x, y)) in a.iter().zip(b).enumerate() {
        let (delta, at) = worst(x, y);
        assert!(
            delta == 0.0,
            "{what}: step {step} parts by {delta} at logit {at} ({} against {}) — this \
             comparison moves NOTHING the arithmetic keys on, so a parting here is an \
             index or a window and not a summation order",
            x[at],
            y[at],
        );
    }
    eprintln!("  {what}: {} steps, bit for bit", a.len());
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(ENV) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over tailscale ssh, so `HOME` is not the owner's.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snaps = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(REPO)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| !containers(path).is_empty())
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

/// Every container of the snapshot, sorted — plural because this checkpoint is
/// sharded and a contract built over shard one refuses at the first tensor
/// that lives in shard two.
fn containers(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found
}

/// A contract lookup the boot door never reaches: the document below is opened
/// for its `[metal.tuning]` table alone.
fn no_door(
    _trace: &model_ir::Trace,
    _path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    Err("this door never loads".to_string())
}

/// `PIE_PROBE_TUNING`, [`throughput_probe`'s spelling][1], applied before the
/// load — because `kernels_metal::tuning` freezes at the first `current()`.
///
/// [1]: https://github.com/pie-project/pie
fn stated_tuning() {
    let Ok(stated) = std::env::var("PIE_PROBE_TUNING") else {
        return;
    };
    let body: String = stated
        .split(',')
        .map(str::trim)
        .filter(|pair| !pair.is_empty())
        .map(|pair| format!("{pair}\n"))
        .collect();
    if body.is_empty() {
        return;
    }
    let doc = format!("[metal.tuning]\n{body}");
    engine_metal::open(doc.as_bytes(), no_door).expect("the boot document opens");
}

/// A loaded shell at [`LANES`] slots and the four prompts, already encoded —
/// or `None` and a sentence naming what was missing.
fn ready(what: &str) -> Option<(Shell, Vec<Vec<u32>>)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(snapshot) = snapshot() else {
        eprintln!("skipping {what}: no {REPO} snapshot in the cache (name one in {ENV})");
        return None;
    };
    if !snapshot.join("tokenizer.json").exists() {
        eprintln!("skipping {what}: {snapshot:?} ships no tokenizer beside its tensors");
        return None;
    }
    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|text| {
            let mut tokens = vec![BOS];
            tokens.extend(tokenizer.encode(text));
            tokens
        })
        .collect();
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(1) as u32;

    let trace = models::trace_of(SKU).expect("the catalog ships this row");
    let import = models::import_of(SKU).expect("the catalog ships an import for it");
    let files = containers(&snapshot);
    let source = ztensor_compat::index_all(&files).expect("the shards open as one");
    // Read for this shell (§J4c): a family's text may state a `Dtype`
    // PLACEMENT, and a contract read under a different setup than the trace
    // describes different planes.
    let contract = models::placing_for(Platform::Metal, || import(&source))
        .expect("the import contract fits the checkpoint");
    drop(source);

    stated_tuning();
    let tuned = kernels_metal::tuning::current();
    let shell = match Shell::load(Boot {
        trace: trace(Platform::Metal),
        contract: &contract,
        checkpoint: &snapshot,
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: Budget::new(LANES, LANES.max(longest)),
        patches: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: LANES,
        runahead: engine::runahead::Runahead::default(),
        residency: engine_metal::ResidencyPlan::default(),
    }) {
        Ok(shell) => shell,
        Err(why) => {
            // A refused load is a skip: 13.3 GiB beside whatever else this box
            // is holding either lands or does not.
            eprintln!("skipping {what}: the load did not fit — {why}");
            return None;
        }
    };
    eprintln!(
        "loaded {SKU} on {} — qmv_rows_max {} (packs {}), qmm_min_batch {}",
        shell.device_name(),
        tuned.qmv_rows_max,
        tuned.qmv_rows_packs,
        tuned.qmm_min_batch,
    );
    Some((shell, prompts))
}

/// One lane's prefill row and [`STEPS`] decode rows after it, fired ALONE in
/// `slot`, each step fed its own argmax.
fn alone(shell: &mut Shell, slot: u32, prompt: &[u32]) -> Vec<Vec<f32>> {
    shell.open(slot).expect("the slot opens");
    let prefill = shell
        .fire(&[Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the solo prefill fires");
    let mut rows = vec![prefill[0].clone()];
    for _ in 0..STEPS {
        let fed = [argmax(rows.last().expect("a step feeds the last row back"))];
        let step = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .expect("the solo decode fires");
        rows.push(step[0].clone());
    }
    rows
}

/// One prompt per lane, each PREFILLED IN ITS OWN FIRE and then decoded
/// TOGETHER — so the prefill is the same rectangle it is alone and the only
/// thing this changes is who shares the decode step.
fn crowd(shell: &mut Shell, prompts: &[Vec<u32>]) -> Vec<Vec<Vec<f32>>> {
    let lanes = prompts.len();
    let mut rows: Vec<Vec<Vec<f32>>> = Vec::with_capacity(lanes);
    for (slot, prompt) in prompts.iter().enumerate() {
        shell.open(slot as u32).expect("the slot opens");
        let prefill = shell
            .fire(&[Lane {
                slot: slot as u32,
                word: word(prompt.len() as u32),
                tokens: prompt,
            }])
            .expect("a lane's prefill fires");
        rows.push(vec![prefill[0].clone()]);
    }
    for _ in 0..STEPS {
        let fed: Vec<[u32; 1]> = rows
            .iter()
            .map(|lane| [argmax(lane.last().expect("a step feeds back"))])
            .collect();
        let seated: Vec<Lane<'_>> = (0..lanes)
            .map(|slot| Lane {
                slot: slot as u32,
                word: word(1),
                tokens: &fed[slot],
            })
            .collect();
        let fired = shell
            .fire(&seated)
            .unwrap_or_else(|why| panic!("the {lanes}-lane decode fires: {why}"));
        for (lane, row) in fired.into_iter().enumerate() {
            rows[lane].push(row);
        }
    }
    rows
}

/// **THE FOUR QUESTIONS, OVER ONE LOAD.**
///
/// # A. The slot is not the seat
///
/// One prompt, fired ALONE, in each of the four slots in turn. A page table
/// row indexed by the slot rather than by the sequence, a write offset off by
/// a lane, a mask plane addressed at lane zero — every one of those lands
/// here, with no crowd anywhere near it. Measured: bit-identical in all four
/// slots at all nine steps.
///
/// # B. Lanes in one fire do not read each other
///
/// Four lanes carrying the SAME prompt, seated in four slots, decoded in one
/// fire. Every row of that fire has the same input, the same positions and the
/// same kv extent, so anything but four identical answers is a lane that read
/// a rectangle that is not its own — and unlike a crowd-against-solo
/// comparison there is no second fire here for a summation order to differ in.
/// Measured: bit-identical across all four lanes.
///
/// # C. A lane does not depend on who rode with it
///
/// Lane 0 decodes the same prompt in two four-lane fires: once beside three
/// copies of itself, once beside three prompts of three different LENGTHS. The
/// fire's shape is identical either way and everything about the neighbours
/// moves. This is the sharpest form of the question `throughput_probe`'s
/// assertion asks — a kv extent taken from the widest lane, a window whose row
/// interval is off by a lane, a page table indexed by the fire's seriated
/// order rather than the submitted one all make lane 0's answer a function of
/// its neighbours' lengths, and none of them survives it. Measured:
/// bit-identical.
///
/// # D. And the one thing that DOES move is the row count
///
/// The same prompt in every lane, at one, two, three and four lanes, each
/// against the same prompt fired alone. Nothing about a lane's geometry
/// changes across those four fires; the only thing that does is how many rows
/// `linear::quant::act_x_wt` is handed, and `qmv_rows_fold` offers a fold rung
/// that DIVIDES the batch:
///
/// ```text
///   width 1   no rung under two            affine_qmv_fast, two packs
///   width 2   rung 2                       affine_qmv_rows, qmv_rows_packs
///   width 3   no rung divides three        affine_qmv_fast, two packs
///   width 4   rung 2                       affine_qmv_rows, qmv_rows_packs
/// ```
///
/// So the answer alternates, and it alternates ONLY because the two points
/// deal k out to a simdgroup's thirty-two lanes differently. Stock M1 Max,
/// worst `|delta|` over the logit row at each decode step:
///
/// ```text
///   width 1   0.0000  0.0000  0.0000  0.0000  0.0000 ...
///   width 2   0.0000  2.4062  8.4141  2.1875  4.8750 ...
///   width 3   0.0000  0.0000  0.0000  0.0000  0.0000 ...
///   width 4   0.0000  2.4062  8.4141  2.1875  4.8750 ...
/// ```
///
/// # What is asserted, and what is only printed
///
/// The widths that take the ONE-ROW point are asserted bit-identical to the
/// solo fire at every tuning, because for those two the fire is running the
/// same kernel over the same rows and there is nothing left to differ.
///
/// The folding widths are asserted bit-identical **only when the table says
/// the fold reproduces the one-row point** — `qmv_rows_packs = 2`, its own
/// pack width, or `qmv_rows_max < 2`, no fold at all. Under the stock table
/// they are printed, because the drift is the ruling's ("fast ladders by
/// default, deterministic knob opt-in") and pinning a number here would gate
/// the default on a promise it does not make.
///
/// So this case is green today and STAYS green the day the default changes,
/// and it is the file that would catch a fold quietly re-minted at a third
/// pack width.
#[test]
#[ignore = "opt-in: holds 13.3 GiB for the length of the run"]
fn what_the_a4b_crowd_disagrees_about() {
    let _serial = serialized();
    let Some((mut shell, prompts)) = ready("the a4b crowd separation") else {
        return;
    };
    let tuned = kernels_metal::tuning::current();
    // The two tables under which the fold IS the one-row point: its own pack
    // width, or no fold offered at any width.
    let invariant = tuned.qmv_rows_packs == 2 || tuned.qmv_rows_max < 2;

    // ── A. The slot is not the seat.
    let by_slot: Vec<Vec<Vec<f32>>> = (0..LANES)
        .map(|slot| alone(&mut shell, slot, &prompts[0]))
        .collect();
    for slot in 1..LANES as usize {
        identical(
            &format!("A. slot {slot} against slot 0, one lane in the fire"),
            &by_slot[0],
            &by_slot[slot],
        );
    }
    let solo = by_slot.into_iter().next().expect("slot 0 was read");

    // ── B. Lanes in one fire do not read each other.
    let same: Vec<Vec<u32>> = (0..LANES).map(|_| prompts[0].clone()).collect();
    let beside_copies = crowd(&mut shell, &same);
    for lane in 1..LANES as usize {
        identical(
            &format!("B. lane {lane} against lane 0, same fire"),
            &beside_copies[0],
            &beside_copies[lane],
        );
    }

    // ── C. A lane does not depend on who rode with it.
    let beside_strangers = crowd(&mut shell, &prompts);
    let lengths: Vec<usize> = prompts.iter().map(Vec::len).collect();
    identical(
        &format!("C. lane 0 beside three copies of itself against beside {lengths:?}"),
        &beside_copies[0],
        &beside_strangers[0],
    );

    // ── D. The row count, width by width.
    for width in 1..=LANES {
        let same: Vec<Vec<u32>> = (0..width).map(|_| prompts[0].clone()).collect();
        let rows = crowd(&mut shell, &same);
        let deltas = worst_over(&solo, &rows[0]);
        eprintln!(
            "  D. width {width}: worst |delta| per step  {}",
            deltas
                .iter()
                .map(|d| format!("{d:.4}"))
                .collect::<Vec<_>>()
                .join("  "),
        );
        // `qmv_rows_fold` declines under two rows and at a batch no rung
        // divides, which for the rungs `[2, 4, 8]` is every odd width.
        let folds = width >= 2 && width % 2 == 0;
        if !folds || invariant {
            identical(
                &format!("D. width {width} against one lane alone"),
                &solo,
                &rows[0],
            );
        }
    }
    if !invariant {
        eprintln!(
            "  D. the folding widths are printed and not asserted: qmv_rows_packs is {} against \
             the one-row point's 2, so the fold reassociates and this row's router amplifies \
             it. `PIE_PROBE_TUNING=qmv_rows_packs=2` asserts all four.",
            tuned.qmv_rows_packs,
        );
    }
}
