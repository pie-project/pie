//! **A LANE'S TOKENS MUST NOT DEPEND ON HOW MANY LANES RODE WITH IT.**
//!
//! `test_curated.py`'s `greedy-decoding-is-the-same-alone-and-in-a-crowd`
//! states the property at the serving door: the same prompt at temperature 0
//! answers the same alone and 2, 4 and 8 ways at once. That gate is
//! autoregressive, so when it fails it says only "the text diverged" — a
//! dozen steps after whatever moved, in a sentence, through a server, a
//! scheduler and a sampler.
//!
//! This is the same claim with the feedback loop cut and the server taken
//! out. ONE shell, one prompt, the lane seated identically in every
//! submission — same slot, same tokens, same pages — and the only thing that
//! changes is HOW MANY OTHER LANES share the fire.
//!
//! # What is promised, and what is not
//!
//! **NOT BIT EQUALITY.** Both dense ladders pick their arm on the fire's row
//! count, which is the composition's number, and the arms round differently:
//! `linear::gemm`'s vector rung folds a simdgroup with `simd_sum` where its
//! tiles walk k in ascending chunks on the matrix unit, and
//! `linear::quant`'s vector point never materializes a weight at all. That is
//! the owner's ruling and it is deliberate:
//!
//! > We do NOT need bit-level identity. If a much faster path has small
//! > numerical drift from nondeterminism, that is obviously acceptable.
//!
//! **WHAT IS PROMISED IS THE TOKEN.** Every step the model actually decided
//! comes out the same alone and in a crowd. A step it had already tied may
//! round either way, and the line between the two is a MARGIN —
//! `throughput_probe::TIE`, a quarter of a logit, restated here as [`TIE`]
//! with that file's reasoning. Every parting is printed with the margin that
//! produced it, so a drift toward the line is readable long before it crosses
//! it; and the whole-row drift is bounded separately by [`MAX_DRIFT`], so a
//! failure that is a wrong INDEX rather than a rounding difference is caught
//! even at a step nobody had decided.
//!
//! # What this still catches
//!
//! The bugs this file was written for are not rounding. A window whose row
//! interval is off by a lane, a page table indexed by the fire's seriated
//! order rather than the submitted one, a mask plane addressed at lane zero
//! for every lane, a kv extent taken from the widest lane instead of each —
//! every one of those REPLACES a lane's answer rather than perturbing it, so
//! it lands far above both lines. Measured on this file's own vehicles, the
//! arm drift the ladders do produce is 0.45 of a logit at a bf16 prefill
//! readout and one or two bf16 ulp a decode step; a lane reading somebody
//! else's rows is a different sentence.
//!
//! # Gating
//!
//! Apple target, a Metal device, and the qwen35-d0.8b snapshot in the hugging
//! face cache — SKIPS at run time saying which was missing, rather than being
//! `#[ignore]`d on the one box that could run it.
//!
//! ```text
//! cargo test -p engine-metal --release --test a_lane_answers_the_same_in_a_crowd \
//!   -- --nocapture
//! ```
//!
//! `PIE_SMOKE_SNAPSHOT` overrides where it looks.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// **ONE CHECKPOINT THIS GATE IS ASKED OF**, because the two planes that
/// carry this disease are selected by the WEIGHT's form and not by the op:
/// `dispatch::linear` sends a dense row to `linear::gemm` and a banked one to
/// `linear::quant`, so a bf16 checkpoint exercises one ladder and a 4-bit
/// checkpoint the other, and neither says anything about the other's arms.
struct Vehicle {
    /// The catalog row.
    sku: &'static str,
    /// What a skip line calls it.
    what: &'static str,
    /// The `models--*` cache directory this snapshot lives under, and the
    /// variable that overrides it.
    repo: &'static str,
    env: &'static str,
    /// **THE PROMPT LENGTH IS THE WHOLE EXPERIMENT**, and it is per vehicle
    /// because the threshold it has to straddle is. See [`BF16`] and [`U4`].
    prompt_rows: usize,
}

/// The bf16 vehicle: the row the curated sweep runs on, and the one whose
/// dense ladder this file was written for.
///
/// THREE rows, because the dense ladder's lowest threshold is
/// `kernels_metal::linear::gemm::VECTOR_MAX_ROWS` and it is four — so one lane
/// alone takes the vector rung and two lanes together take the 8-row tile,
/// which is the one boundary on this ladder the two sides of which round
/// differently. The two TILE rungs are one kernel at two row blocks and land
/// the same bits, so a prompt chosen to straddle `TILE_M` would be asking
/// nothing.
const BF16: Vehicle = Vehicle {
    sku: "qwen35-d0.8b-bf16-kv-bf16",
    what: "the bf16 crowd gate",
    repo: "models--Qwen--Qwen3.5-0.8B",
    env: "PIE_SMOKE_SNAPSHOT",
    prompt_rows: 3,
};

/// The 4-bit vehicle — `four_bit_first_light`'s SKU, and the format every
/// north-star model on this plane ships in.
///
/// THREE rows, and for the same kind of reason the bf16 row is twenty: the
/// quantized ladder's crossover is `tuning::qmm_min_batch`, which is FIVE on
/// an M1 Max. Three rows alone is under it and two lanes' six rows is over
/// it, so the prefill fire straddles the crossover at every width below, and
/// the decode fires after it straddle it too — one row alone against `width`
/// rows together. A twenty-row prompt would have been over the crossover
/// alone as well, and would have gated nothing on this ladder.
const U4: Vehicle = Vehicle {
    sku: "qwen35-d0.8b-mlxu4-kv-bf16",
    what: "the 4-bit crowd gate",
    repo: "models--mlx-community--Qwen3.5-0.8B-4bit",
    env: "PIE_U4_SNAPSHOT",
    prompt_rows: 3,
};

/// What the prompt is made of. Truncated to the vehicle's `prompt_rows`; the
/// text only has to be long enough to reach it and ordinary enough to produce
/// a non-degenerate row.
const PROMPT: &str = "Explain why the sky appears blue, and be thorough about \
                      the physics of it, including what happens at sunset.";

/// The neighbours in the mixed-fire arm, deliberately RAGGED: a fire whose
/// lanes all carry the same row count is the one composition a width-keyed
/// arm is most likely to survive by accident.
const NEIGHBOURS: [&str; 7] = [
    "The capital of France is",
    "banana",
    "Name the largest planet in the solar system, and say why.",
    "One two three four five",
    "Kilimanjaro",
    "Write a haiku about a stone in a river.",
    "ok",
];

/// How many decode fires follow the prefill. Short on purpose — a bit-exact
/// comparison needs no room to amplify, and every step is a whole fire.
const STEPS: usize = 6;

/// The widths the crowd is made at.
const WIDTHS: [u32; 3] = [2, 4, 8];

/// **HOW CLOSE A STEP HAS TO BE BEFORE A DISAGREEMENT STOPS BEING ONE**, in
/// logits of top-two gap. `throughput_probe::TIE`'s number and
/// `throughput_probe::TIE`'s argument, restated because this file has its own
/// gate: `session_c_first_light` measures this family of checkpoints at a
/// two-implementation noise floor of 0.103 rms and records a real mlx-vs-mlx
/// flip at 0.0625, while the steps a model has genuinely decided in that file
/// are won by 0.94 to 5.13. A quarter of a logit sits an order of magnitude
/// below the smallest DECIDED step and a factor of two above the largest
/// measured tie.
const TIE: f32 = 0.25;

/// **AND THE ROW'S DRIFT IS BOUNDED TOO**, so that a step nobody had decided
/// is still gated on something. Half a logit is above the largest arm
/// difference ever measured on these ladders — 0.453125, the bf16 prefill
/// readout that named the dense vector rung — and orders of magnitude below
/// what a lane reading another lane's rows produces, which is a different
/// distribution rather than a perturbed one.
///
/// It is a SANITY bound and deliberately not tight: tightening it would turn
/// this into a gate on bf16 rounding, which is the thing the ruling says not
/// to spend anything on.
const MAX_DRIFT: f32 = 0.5;

/// **ONE SHELL AT A TIME, PER PROCESS** — the same reason `serve_smoke`
/// gives: each of these holds ~1.6 GiB resident.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, for whichever vehicle is asking.
fn snapshot(vehicle: &Vehicle) -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(vehicle.env) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over tailscale ssh, so `HOME` is not the
    // owner's — the cache is named explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(vehicle.repo)
            .join("snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists() && container(path).is_some())
    })
}

/// The container the contract is checked against — one file of the snapshot,
/// whichever one holds the tensors.
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

/// The lane word the model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// The winner and **how much it won by** — the top-two logit gap.
///
/// `throughput_probe::top2`'s reason: an argmax disagreement at a step decided
/// by five logits is a wrong answer, and one at a step decided by a hundredth
/// is two summation orders rounding a tie two ways.
fn top2(logits: &[f32]) -> (u32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    let mut winner = 0u32;
    for (at, &value) in logits.iter().enumerate() {
        if value > best {
            second = best;
            best = value;
            winner = at as u32;
        } else if value > second {
            second = value;
        }
    }
    (winner, best - second)
}

/// **WHETHER TWO READINGS OF ONE LANE DISAGREE ABOUT ANYTHING THAT WAS
/// DECIDED**, and the sentence a failure gets to say.
///
/// Two things are asked of the pair and they catch different faults. The
/// TOKEN has to match at every step, unless the step was a tie by [`TIE`] —
/// that is the property a serving fleet publishes. And the row's worst
/// element drift has to stay under [`MAX_DRIFT`] whatever the argmax did —
/// that is what still says something at a step the model had not decided, and
/// it is where a lane reading rows that are not its own lands.
fn parted(alone: &[f32], crowd: &[f32]) -> Option<String> {
    let worst = alone
        .iter()
        .zip(crowd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (mine, my_margin) = top2(alone);
    let (theirs, their_margin) = top2(crowd);
    let margin = my_margin.min(their_margin);
    let flipped = mine != theirs && margin > TIE;
    if !flipped && worst <= MAX_DRIFT {
        return None;
    }
    let why = if flipped {
        format!(
            "argmax {mine} alone (won by {my_margin:.4}) against {theirs} in the crowd \
             (won by {their_margin:.4}) — decided by {margin:.4}, over the {TIE:.2} line"
        )
    } else {
        format!("worst |delta| over the row is {worst}, over the {MAX_DRIFT:.2} bound")
    };
    Some(format!("{why} (argmax {mine} against {theirs}, worst |delta| {worst})"))
}

/// Everything the tests below share: a shell wide enough for `slots` lanes of
/// `rows` rows each, and the prompt, already truncated.
fn ready(
    vehicle: &Vehicle,
    what: &str,
    slots: u32,
    rows: u32,
) -> Option<(Shell, Vec<u32>, Vec<Vec<u32>>)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot(vehicle) else {
        eprintln!(
            "skipping {what}: no {} snapshot in the hugging face cache (set {})",
            vehicle.repo, vehicle.env
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let mut prompt = tokenizer.encode(PROMPT);
    assert!(
        prompt.len() >= vehicle.prompt_rows,
        "the prompt text tokenizes to {} rows, and this gate needs {}",
        prompt.len(),
        vehicle.prompt_rows
    );
    prompt.truncate(vehicle.prompt_rows);
    let neighbours = NEIGHBOURS
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();

    let trace = model::trace_of(vehicle.sku).expect("the catalog ships this gate's SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(vehicle.sku)
        .expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(slots, slots * rows),
        profile: None,
        page_size: 16,
        context: 256,
        slots,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    Some((shell, prompt, neighbours))
}

/// One lane's whole reading: the prefill row and every decode row after it,
/// fed its own argmax, in a fire of exactly `lanes` lanes.
type Reading = Vec<Vec<f32>>;

/// **THE PROPERTY, ON THE PREFILL AND EVERY STEP AFTER IT.** The same prompt
/// is prefilled in one fire by `width` lanes at once — which is the shape the
/// server makes when several requests arrive together — and then decoded
/// together. Every lane says at every step what it says alone, unless the
/// step was a tie; see [`parted`] for both halves of that.
///
/// The `_four_bit` twin below is the same claim over a banked checkpoint, so
/// that `linear::quant`'s ladder is asked and not `linear::gemm`'s.
#[test]
fn one_prompt_in_every_lane_says_what_it_says_alone() {
    let crossover =
        usize::try_from(kernels_metal::linear::gemm::VECTOR_MAX_ROWS).unwrap_or(usize::MAX);
    assert!(
        BF16.prompt_rows < crossover && BF16.prompt_rows * 2 >= crossover,
        "this gate is only asking anything if one lane's rows and two lanes' rows \
         fall on opposite sides of the dense ladder's vector crossover"
    );
    every_lane_says_what_it_says_alone(&BF16);
}

/// The 4-bit twin. See [`U4`] for why its prompt is three rows: the quantized
/// ladder's threshold is `tuning::qmm_min_batch` rather than
/// `gemm::VECTOR_MAX_ROWS`, and it is five rows on an M1 Max.
#[test]
fn one_prompt_in_every_lane_says_what_it_says_alone_four_bit() {
    every_lane_says_what_it_says_alone(&U4);
}

fn every_lane_says_what_it_says_alone(vehicle: &Vehicle) {
    let _serial = serialized();
    let widest = *WIDTHS.last().expect("the widths are not empty");
    let Some((mut shell, prompt, _)) = ready(
        vehicle,
        vehicle.what,
        widest,
        u32::try_from(vehicle.prompt_rows).expect("the prompt fits a u32"),
    ) else {
        return;
    };

    let alone = read(&mut shell, &prompt, 1).remove(0);
    eprintln!(
        "{} alone: {:?}",
        vehicle.sku,
        alone.iter().map(|row| argmax(row)).collect::<Vec<_>>()
    );

    for width in WIDTHS {
        let crowd = read(&mut shell, &prompt, width);
        for (lane, reading) in crowd.iter().enumerate() {
            for (step, row) in reading.iter().enumerate() {
                assert!(
                    row.len() == alone[step].len(),
                    "at width {width}, lane {lane} came back {} logits wide against {} alone",
                    row.len(),
                    alone[step].len()
                );
                if let Some(why) = parted(&alone[step], row) {
                    panic!(
                        "{}: at width {width}, lane {lane}'s step {step} is not what it \
                         says alone: {why}\n\
                         the arms this ladder picks between drift by one or two bf16 ulp; \
                         a decided step coming out the other way, or half a logit of row \
                         drift, is an index or a window that read the fire's shape",
                        vehicle.sku
                    );
                }
            }
        }
        eprintln!(
            "{} width {width}: {} lanes, every step the token it says alone",
            vehicle.sku,
            crowd.len()
        );
    }
}

/// **THE MIXED FIRE, WHICH IS WHAT A SERVER ACTUALLY MAKES.** A steady-state
/// batch is one lane decoding beside another lane prefilling: the fire's rows
/// are the decode lane's one plus the prefills' many, so the row count the
/// dispatch sees moves with the NEIGHBOURS' prompts and not with anything the
/// decode lane did. Lane 0 decodes its own prompt here while 1..width prefill
/// ragged ones.
///
/// This is the composition that crosses BOTH ladders' crossovers at once, and
/// it is the sharpest form of the question this file asks: lane 0 is a
/// one-row fire alone and part of a twelve-row fire beside one neighbour, so
/// it takes the vector arm in the first and the tile in the second. Nothing
/// about lane 0 moved, and the token it publishes may not either.
#[test]
fn a_decode_lane_says_the_same_thing_beside_any_number_of_prefills() {
    a_decode_lane_beside_prefills(&BF16);
}

/// The 4-bit twin, over `linear::quant`'s ladder rather than
/// `linear::gemm`'s.
#[test]
fn a_decode_lane_says_the_same_thing_beside_any_number_of_prefills_four_bit() {
    a_decode_lane_beside_prefills(&U4);
}

fn a_decode_lane_beside_prefills(vehicle: &Vehicle) {
    let _serial = serialized();
    let widest = *WIDTHS.last().expect("the widths are not empty");
    let longest = NEIGHBOURS
        .iter()
        .map(|text| text.len())
        .max()
        .unwrap_or(vehicle.prompt_rows);
    let Some((mut shell, prompt, neighbours)) = ready(
        vehicle,
        vehicle.what,
        widest,
        u32::try_from(vehicle.prompt_rows.max(longest)).expect("the prompt fits a u32"),
    ) else {
        return;
    };

    let alone = read(&mut shell, &prompt, 1).remove(0);

    for width in WIDTHS {
        // Lane 0 is seated exactly as it is alone: opened, prefilled by
        // itself, then stepped. What changes is who ELSE is in each step's
        // fire — a fresh prefill of a different length every time.
        shell.open(0).expect("slot 0 opens");
        let seed = shell
            .fire(&[Lane {
                slot: 0,
                word: word(u32::try_from(prompt.len()).expect("the prompt fits a u32")),
                tokens: &prompt,
            }])
            .expect("lane 0 prefills");
        let mut rows = vec![seed[0].clone()];
        let mut fed = argmax(&seed[0]);
        for step in 0..STEPS {
            let token = [fed];
            let mut lanes = vec![Lane {
                slot: 0,
                word: word(1),
                tokens: &token,
            }];
            for slot in 1..width {
                let neighbour = &neighbours[(step + slot as usize) % neighbours.len()];
                shell.open(slot).expect("a neighbour's slot opens");
                lanes.push(Lane {
                    slot,
                    word: word(u32::try_from(neighbour.len()).expect("a prompt fits a u32")),
                    tokens: neighbour,
                });
            }
            let fired = shell
                .fire(&lanes)
                .unwrap_or_else(|why| panic!("the width-{width} mixed fire fires: {why}"));
            fed = argmax(&fired[0]);
            rows.push(fired[0].clone());
        }
        for (step, row) in rows.iter().enumerate() {
            if let Some(why) = parted(&alone[step], row) {
                panic!(
                    "{}: beside {} prefilling lanes, lane 0's step {step} is not what it \
                     says alone: {why}\n\
                     the neighbours' rows changed the fire's shape, and the shape reached \
                     further into the answer than the arms' own rounding does",
                    vehicle.sku,
                    width - 1
                );
            }
        }
        eprintln!(
            "{} lane 0 beside {} prefills: every step the token it says alone",
            vehicle.sku,
            width - 1
        );
    }
}

/// **THE OTHER SHAPE A THRESHOLD CAN BE MOVED BY: A SHORTER NEIGHBOUR.** The
/// dense projection keys on the fire's rows, which a neighbour ADDS to; the
/// sdpa arbiter keys on rows-PER-REQUEST, which a short neighbour SUBTRACTS
/// from. A long prompt alone is over that second threshold and the same
/// prompt beside a one-token lane is under it, so this composition steps the
/// attention arm without touching the projection's, and the lane that did not
/// move must not move either.
///
/// **AND THE ATTENTION ARMS ARE HELD TO MORE THAN THE PROJECTIONS ARE.** The
/// two sdpa rungs are the same arithmetic — one walks a tile where the other
/// walks a row — so unlike the projection ladders there is no accepted drift
/// here, and a parting at this threshold is a defect rather than a rounding
/// difference. [`parted`]'s bound is what says so: the projections' own drift
/// is far under [`MAX_DRIFT`], and this composition does not move them.
#[test]
fn a_long_prefill_says_the_same_thing_beside_a_short_one() {
    let _serial = serialized();
    // The bf16 vehicle alone: this arm is about the sdpa arbiter's
    // rows-per-request threshold, which no projection ladder reads, so a
    // banked twin would ask the same question of the same kernels.
    let Some((mut shell, prompt, neighbours)) =
        ready(&BF16, "the ragged-prefill gate", 2, 128)
    else {
        return;
    };
    // Long enough that this lane alone is over the sdpa arbiter's
    // rows-per-request threshold, and short enough to page inside `context`.
    let long: Vec<u32> = prompt.iter().cycle().copied().take(62).collect();
    // The shortest neighbour there is, because what has to move is the MEAN.
    let short = neighbours
        .iter()
        .min_by_key(|tokens| tokens.len())
        .expect("the neighbours are not empty");
    assert!(
        long.len() > 32 && (long.len() + short.len()) / 2 < 32,
        "this gate is only asking anything if the pair's rows-per-request          ({}) falls under the threshold the lone lane's ({}) is over",
        (long.len() + short.len()) / 2,
        long.len()
    );

    shell.open(0).expect("slot 0 opens");
    let alone = shell
        .fire(&[Lane {
            slot: 0,
            word: word(u32::try_from(long.len()).expect("the prompt fits a u32")),
            tokens: &long,
        }])
        .expect("the long lane prefills alone")[0]
        .clone();

    shell.open(0).expect("slot 0 re-opens");
    shell.open(1).expect("slot 1 opens");
    let together = shell
        .fire(&[
            Lane {
                slot: 0,
                word: word(u32::try_from(long.len()).expect("the prompt fits a u32")),
                tokens: &long,
            },
            Lane {
                slot: 1,
                word: word(u32::try_from(short.len()).expect("the prompt fits a u32")),
                tokens: short,
            },
        ])
        .expect("the ragged pair prefills");
    if let Some(why) = parted(&alone, &together[0]) {
        panic!(
            "a {}-row prefill answered differently beside a {}-row one: {why}\n\
             the pair's rows-per-request is under a threshold the lone lane's is over, \
             and the two attention rungs are supposed to be the same arithmetic",
            long.len(),
            short.len()
        );
    }
    eprintln!(
        "a {}-row prefill beside a {}-row one: the same token it says alone",
        long.len(),
        short.len()
    );
}

/// `lanes` lanes, all seated on the same prompt, prefilled in ONE fire and
/// then stepped together — each fed its own argmax, which for identical lanes
/// is the same token and therefore keeps every lane's input identical for the
/// whole reading.
fn read(shell: &mut Shell, prompt: &[u32], lanes: u32) -> Vec<Reading> {
    for slot in 0..lanes {
        shell.open(slot).expect("the slot opens");
    }
    let seated: Vec<Lane<'_>> = (0..lanes)
        .map(|slot| Lane {
            slot,
            word: word(u32::try_from(prompt.len()).expect("the prompt fits a u32")),
            tokens: prompt,
        })
        .collect();
    let prefill = shell
        .fire(&seated)
        .unwrap_or_else(|why| panic!("the {lanes}-lane prefill fires: {why}"));
    assert_eq!(prefill.len() as u32, lanes, "one row of logits per lane");

    let mut readings: Vec<Reading> = prefill.iter().map(|row| vec![row.clone()]).collect();
    for step in 0..STEPS {
        let fed: Vec<[u32; 1]> = readings
            .iter()
            .map(|reading| {
                [argmax(
                    reading.last().expect("a step feeds the last row back"),
                )]
            })
            .collect();
        let seated: Vec<Lane<'_>> = (0..lanes as usize)
            .map(|slot| Lane {
                slot: slot as u32,
                word: word(1),
                tokens: &fed[slot],
            })
            .collect();
        let fired = shell
            .fire(&seated)
            .unwrap_or_else(|why| panic!("the {lanes}-lane decode step {step} fires: {why}"));
        for (lane, row) in fired.into_iter().enumerate() {
            readings[lane].push(row);
        }
    }
    readings
}
