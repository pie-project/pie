//! **A LANE'S LOGITS MUST NOT DEPEND ON HOW MANY LANES RODE WITH IT.**
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
//! changes is HOW MANY OTHER LANES share the fire. Every kernel this plane
//! owns is row-parallel, so a row's arithmetic reads that row and the answer
//! is a function of the lane's own inputs. BIT FOR BIT, or it is not.
//!
//! Not `argmax`, and not a tolerance: the whole point of the failure this
//! gate is named for is that two arms that agree to four decimal places still
//! part into two different sentences twelve steps later. The comparison is
//! `==` over the whole logit row.
//!
//! # What was broken
//!
//! `kernels_metal::linear::gemm::act_x_wt` chose its arm on the FIRE's row
//! count — `dense_gemv_t_bfloat16` below 32 rows, the tiled GEMM at or above
//! — and a fire's row count is however many rows the composition put in it.
//! The vector kernel splits K across a simdgroup's 32 lanes and folds the
//! pieces with `simd_sum`; the tile walks K in ascending 8-wide chunks on the
//! matrix unit. Same arithmetic, different order, different rounding.
//!
//! So the curated gate's own prompt — chat-templated, 31 tokens — was 31
//! rows alone and 248 rows eight ways at once, landed on opposite sides of
//! that threshold, and came back different. Measured at HEAD 6714f0580 with
//! that prompt, the prefill readout row of a lane fired alone against the
//! same lane in a crowd (this file uses a shorter one; any length under the
//! threshold does):
//!
//! ```text
//!   width 2   max |delta logit| 0.453125   -> a different sentence by token 9
//!   width 4   max |delta logit| 0.453125
//!   width 8   max |delta logit| 0.453125
//! ```
//!
//! And the tell that it was the THRESHOLD and not the crowd: a 41-token
//! prompt — over the threshold when it is alone, too — was bit-identical at
//! every width. Both arms are correct; neither is the other's bits. The fix
//! makes the narrow rung the same kernel at an 8-row block, so the pick is
//! invisible in the answer. `linear/gemm.rs`'s header is the argument.
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

/// The catalog row this gate serves — the one the curated sweep runs on.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// **THE PROMPT LENGTH IS THE WHOLE EXPERIMENT.** It must be UNDER the dense
/// projection's row threshold (`kernels_metal::linear::gemm::TILE_M`, 32) so
/// that one lane alone lands on one side of it and two lanes together land on
/// the other. A prompt already over the threshold is bit-identical at every
/// width even with the bug in, which is exactly how the bug was localized and
/// exactly why this number is not "whatever the tokenizer gave back".
const PROMPT_ROWS: usize = 20;

/// What the prompt is made of. Truncated to [`PROMPT_ROWS`]; the text only
/// has to be long enough to reach it and ordinary enough to produce a
/// non-degenerate row.
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

/// **ONE SHELL AT A TIME, PER PROCESS** — the same reason `serve_smoke`
/// gives: each of these holds ~1.6 GiB resident.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot directory: the checkpoint AND the tokenizer that goes with it.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
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

/// Where two rows first part, and by how much — the sentence a failure gets
/// to say instead of "not equal".
fn parted(alone: &[f32], crowd: &[f32]) -> Option<String> {
    let at = alone
        .iter()
        .zip(crowd.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits())?;
    let worst = alone
        .iter()
        .zip(crowd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Some(format!(
        "logit {at} is {} alone and {} in the crowd (worst |delta| over the row {worst}, \
         argmax {} against {})",
        alone[at],
        crowd[at],
        argmax(alone),
        argmax(crowd),
    ))
}

/// Everything the tests below share: a shell wide enough for `slots` lanes of
/// `rows` rows each, and the prompt, already truncated.
fn ready(what: &str, slots: u32, rows: u32) -> Option<(Shell, Vec<u32>, Vec<Vec<u32>>)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
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
        prompt.len() >= PROMPT_ROWS,
        "the prompt text tokenizes to {} rows, and this gate needs {PROMPT_ROWS}",
        prompt.len()
    );
    prompt.truncate(PROMPT_ROWS);
    let neighbours = NEIGHBOURS
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();

    let trace = model::trace_of(SKU).expect("the catalog ships this gate's SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
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
/// server makes when several requests arrive together, and the shape no test
/// in this directory made before this one — and then decoded together. Every
/// row of every lane is the row the lane produced alone.
#[test]
fn one_prompt_in_every_lane_lands_the_bits_it_lands_alone() {
    let _serial = serialized();
    let widest = *WIDTHS.last().expect("the widths are not empty");
    let Some((mut shell, prompt, _)) = ready(
        "the crowd gate",
        widest,
        u32::try_from(PROMPT_ROWS).expect("the prompt fits a u32"),
    ) else {
        return;
    };
    assert!(
        PROMPT_ROWS < usize::try_from(kernels_metal::linear::gemm::TILE_M).unwrap_or(usize::MAX)
            && PROMPT_ROWS * 2 >= usize::try_from(kernels_metal::linear::gemm::TILE_M).unwrap_or(0),
        "this gate is only asking anything if one lane's rows and two lanes' rows \
         fall on opposite sides of the dense projection's row threshold"
    );

    let alone = read(&mut shell, &prompt, 1).remove(0);
    eprintln!(
        "alone: {:?}",
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
                        "at width {width}, lane {lane}'s step {step} is not the row it is \
                         alone: {why}\n\
                         a lane's answer is a function of its own inputs, so this is an arm \
                         or an index that read the fire's shape"
                    );
                }
            }
        }
        eprintln!(
            "width {width}: {} lanes, every row bit-identical to alone",
            crowd.len()
        );
    }
}

/// **THE MIXED FIRE, WHICH IS WHAT A SERVER ACTUALLY MAKES.** A steady-state
/// batch is one lane decoding beside another lane prefilling, and it is the
/// composition a per-lane rule cannot save: the fire's rows are the decode
/// lane's one plus the prefills' many, so the row count the dispatch sees
/// moves with the NEIGHBOURS' prompts and not with anything the decode lane
/// did. Lane 0 decodes its own prompt here while 1..width prefill ragged ones.
#[test]
fn a_decode_lane_lands_the_same_bits_beside_any_number_of_prefills() {
    let _serial = serialized();
    let widest = *WIDTHS.last().expect("the widths are not empty");
    let longest = NEIGHBOURS
        .iter()
        .map(|text| text.len())
        .max()
        .unwrap_or(PROMPT_ROWS);
    let Some((mut shell, prompt, neighbours)) = ready(
        "the mixed-fire crowd gate",
        widest,
        u32::try_from(PROMPT_ROWS.max(longest)).expect("the prompt fits a u32"),
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
                    "beside {} prefilling lanes, lane 0's step {step} is not the row it is \
                     alone: {why}\n\
                     the neighbours' rows changed the fire's shape, and the shape reached \
                     the answer",
                    width - 1
                );
            }
        }
        eprintln!(
            "lane 0 beside {} prefills: every row bit-identical to alone",
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
#[test]
fn a_long_prefill_lands_the_same_bits_beside_a_short_one() {
    let _serial = serialized();
    let Some((mut shell, prompt, neighbours)) = ready("the ragged-prefill gate", 2, 128) else {
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
            "a {}-row prefill answered differently beside a {}-row one: {why}\n             the pair's rows-per-request is under a threshold the lone lane's is over",
            long.len(),
            short.len()
        );
    }
    eprintln!(
        "a {}-row prefill beside a {}-row one: bit-identical to alone",
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
