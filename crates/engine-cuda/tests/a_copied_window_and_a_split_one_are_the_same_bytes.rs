//! **`Fallback::Copy` ON DEVICE, DIFFED AGAINST THE ORACLE IT REPLACES.**
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test a_copied_window_and_a_split_one_are_the_same_bytes -- --nocapture
//! ```
//!
//! # What this is for
//!
//! P4 withdraws the `captures_scores` window — the axis that CROSSES `qo_one`
//! where every earlier axis nested inside it — and writes a fallback row for
//! each of qwen3.5's six `attention.prefill_lse` nodes. The menu it writes has
//! two entries because its cost model is bucket-keyed
//! (`model_compiler::layout`'s `CROSSOVER_ROWS`: at 64 rows a two-way split
//! measured 1.82x the ideal against a copy's 1.07x on a 3090, converging by
//! 2048): **`Fallback::Copy` below the crossover, `Fallback::Split { r }`
//! above it.** Until now the shell served both as splits — correct, and
//! roughly 1.7x more expensive than the table asked for on every bucket a
//! decode fire lands in.
//!
//! `export_axes.rs` gate (g) is the split's gate: a capturing prefill lane
//! beside a capturing decode lane beside a plain one, fired, with every lane
//! saying what it says alone. This is the copy's, against the same
//! composition and on the same weights — and against a stronger claim, because
//! a split is a free oracle here. A gather moves BYTES. It may not change a
//! single number.
//!
//! ```text
//! (a) the same fire, split then copied, is bit-identical in logits and in
//!     captured mass — every lane, every layer
//! (b) and the copy costs fewer launches, counted off the artifact rather
//!     than hoped for
//! (c) the composition really is one P4 could not seat, and the table really
//!     does ask for a copy at this fire's bucket — the two premises, asserted
//! (d) what the two cost in wall clock on this device, which nobody has
//!     measured since the 3090 table
//! ```
//!
//! # Why bit-identity is the right bar and not an aspiration
//!
//! Everything a copy changes is an ADDRESS. The same kernel runs, over the
//! same rows, with the same schedule — carved over the union rather than per
//! interval, which is a different carving of the same requests — reading the
//! same kv pages through a page table that names them in the same order. The
//! only arithmetic that could differ is a reduction order inside one launch
//! against two, and there is none: the two runs of a split attend disjoint
//! requests and never combine. So a difference of one ULP is a bug, not
//! numerical noise, and asserting exact equality is what makes this test able
//! to see one.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::{Boot, Graphs, LayerScores, Seated, Shell};
use model_compiler::{Budget, DeviceProfile, compile};
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The three prompts, one per lane. Lane 0 and lane 1 capture; lane 2 does
/// not — which is gate (g)'s composition, and the smallest one that puts a
/// plain lane's rows between the two capturing classes.
const CAPTURING: &str = "The capital of France is";
const LATE: &str = "The largest planet is";
const PLAIN: &str = "Water boils at";

/// The ceilings, named because this file bakes the same plan a second time to
/// check its own premise and two budgets would be two artifacts.
const BUDGETS: Budget = Budget {
    max_lanes: 4,
    max_tokens: 256,
    buckets: Vec::new(),
    max_adapters: 0,
};

/// The wider ceilings gate (e) needs: five lanes, and an adapter seat, which
/// is what makes the classes `{6,7}` reachable and therefore what makes the
/// withdrawn window break into all THREE of the runs P4 counted.
const WIDE: Budget = Budget {
    max_lanes: 8,
    max_tokens: 256,
    buckets: Vec::new(),
    max_adapters: 2,
};

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name, and this file's copy slab is one of them
/// (`serve_smoke.rs` argues the rule whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32, captures: bool) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false).capturing_scores(captures))
        .word()
}

fn seat<'a>(slot: u32, tokens: &'a [u32], captures: bool) -> Seated<'a> {
    let lane = engine_cuda::Lane {
        slot,
        word: word(tokens.len() as u32, captures),
        tokens,
    };
    if captures {
        Seated::capturing(lane)
    } else {
        Seated::of(lane)
    }
}

/// The fragmented composition, fired once from a clean set of slots.
///
/// **THE SLOTS ARE RE-OPENED EVERY TIME**, which is what makes the two halves
/// of the A/B the same fire rather than two consecutive steps of one
/// conversation: an open slot carries kv from whatever ran before it, and a
/// copy reading the same pages as a split is exactly what is being tested.
///
/// Lane 0 decodes (one token) while lanes 1 and 2 prefill, and lanes 0 and 1
/// capture — so the fire holds a capturing DECODE class and a capturing
/// PREFILL class with a plain class's rows standing between them, which is
/// the window P4 could not seat.
fn fire_it(
    shell: &mut Shell,
    fed: &[Vec<u32>],
) -> (Vec<Vec<f32>>, Vec<Vec<LayerScores>>, engine_cuda::FireCost) {
    for slot in 0..3 {
        shell.open(slot).expect("the slot opens");
    }
    let seated = [
        seat(0, &fed[0], true),
        seat(1, &fed[1], true),
        seat(2, &fed[2], false),
    ];
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    let out = shell
        .fire_captured(&seated, &[], &mut mass)
        .expect("the fragmented fire");
    (out, mass, shell.last_fire_cost())
}

/// Bit-for-bit, and it says WHICH number moved when it does not hold.
fn same_logits(left: &[Vec<f32>], right: &[Vec<f32>], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: lane counts");
    for (lane, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: lane {lane} vocabulary");
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: lane {lane} logit {at} — split {x} against copy {y}",
            );
        }
    }
}

fn same_mass(left: &[Vec<LayerScores>], right: &[Vec<LayerScores>], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: lane counts");
    for (lane, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: lane {lane} layer counts");
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.layer, y.layer, "{what}: lane {lane} layer order");
            assert_eq!(
                (x.rows, x.heads),
                (y.rows, y.heads),
                "{what}: lane {lane} shape"
            );
            assert_eq!(x.lse.len(), y.lse.len(), "{what}: lane {lane} length");
            for (at, (p, q)) in x.lse.iter().zip(&y.lse).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "{what}: lane {lane} layer {} entry {at} — split {p} against copy {q}",
                    x.layer,
                );
            }
        }
    }
}

// ── (c) the premises ─────────────────────────────────────────────────────

/// **NEITHER HALF OF THIS FILE IS ALLOWED TO BE VACUOUS**, and the way it
/// could be is quiet: a seriation change that seated the capture window, or a
/// crossover that put this bucket on the split side, would leave every
/// assertion below comparing a fire against itself and passing.
///
/// So the artifact is asked directly. CompiledModel at the same budgets from the same
/// catalog text: some region's window must come back in pieces under this
/// composition, and P4's table must answer `Copy` for it at this fire's
/// bucket. NO DEVICE — it is a statement about the bake.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_composition_fragments_a_window_and_the_table_asks_for_a_copy() {
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let compiled = compile(&trace, &BUDGETS, &DeviceProfile::default()).expect("the SKU bakes");

    // The three lanes of `fire_it`, as words: a capturing decode, a capturing
    // prefill and a plain prefill.
    let lanes = [
        model_exec::fire::Lane::new(word(1, true), 1),
        model_exec::fire::Lane::new(word(5, true), 5),
        model_exec::fire::Lane::new(word(4, false), 4),
    ];
    let fire = model_exec::fire::compose(&compiled, &BUDGETS, &lanes).expect("the fire composes");
    // An empty lattice is one implicit bucket, at index 0 — which is what the
    // shell computes too, and the number the table is read at.
    let bucket = 0u32;

    let mut fragmented = 0usize;
    let mut copied = 0usize;
    for region in compiled.template() {
        if fire.classes().spans(&region.mask).len() < 2 {
            continue;
        }
        fragmented += 1;
        if model_exec::fire::fallback::copies(&compiled, &region.mask, bucket) {
            copied += 1;
        }
    }
    assert!(
        fragmented > 0,
        "this composition leaves no window in pieces, so the A/B below is a fire \
         against itself",
    );
    assert_eq!(
        copied, fragmented,
        "{fragmented} windows come back in pieces and the table asks for a copy on \
         only {copied} of them",
    );
    eprintln!("the composition leaves {fragmented} windows in pieces, all of them copyable");
}

// ── (a) and (b) the diff, on device ──────────────────────────────────────

/// **THE GATE.** The same fire, served two ways, is the same bytes — and the
/// copy costs fewer launches.
///
/// The order is split first because the split is the oracle: it is what this
/// crate's other gates are written against, so its numbers are the ones a
/// regression would be measured from. Both halves are fired twice and the
/// SECOND is kept, for the reason every mixed-fire gate in this crate does
/// it: the dense autotuner tunes a GEMM shape on its second sighting, so a
/// cold first fire and a warm second one are two tactic ladders and the
/// identity would be between different arithmetic.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_same_fragmented_fire_split_and_copied_is_the_same_bytes_in_fewer_launches() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the copy/split diff") else {
        return;
    };
    let fed = vec![
        tok.encode(CAPTURING)[..1].to_vec(),
        tok.encode(LATE),
        tok.encode(PLAIN),
    ];

    shell.set_copies(false);
    let _ = fire_it(&mut shell, &fed);
    let (split_out, split_mass, split_cost) = fire_it(&mut shell, &fed);

    shell.set_copies(true);
    let _ = fire_it(&mut shell, &fed);
    let (copy_out, copy_mass, copy_cost) = fire_it(&mut shell, &fed);

    eprintln!(
        "split: {} launches, {} copied | copy: {} launches, {} copied",
        split_cost.launches, split_cost.copied, copy_cost.launches, copy_cost.copied,
    );

    // (b) THE LAUNCHES. The copy path was actually taken — a run that
    // silently fell back to the split would pass every byte comparison below
    // and prove nothing at all, which is the failure mode this counts
    // against.
    assert_eq!(split_cost.copied, 0, "the split half copied something");
    assert!(
        copy_cost.copied > 0,
        "copies are on and no region was gathered, so the diff below is a split \
         against a split",
    );
    assert!(
        copy_cost.launches < split_cost.launches,
        "the copy cost {} launches and the split {}",
        copy_cost.launches,
        split_cost.launches,
    );
    // AND THE SAVING IS EXACTLY THE ONE THE ARTIFACT PREDICTS: every copied
    // region falls from its run count to one, and nothing else moves. CompiledModel
    // again from the same text at the same budgets, so the number comes from
    // P4 rather than from the shell agreeing with itself.
    let (fragmented, extra) = predicted();
    assert_eq!(
        copy_cost.copied, fragmented,
        "{fragmented} windows come back in pieces and {} were gathered",
        copy_cost.copied,
    );
    assert_eq!(
        split_cost.launches - copy_cost.launches,
        extra,
        "the saving is not the copied regions' extra runs",
    );

    // (a) THE BYTES. Nothing about the numbers may have moved.
    same_logits(&split_out, &copy_out, "the fragmented fire");
    same_mass(&split_mass, &copy_mass, "the fragmented fire");

    // And the mass is really there — a capture that came back empty from both
    // halves would compare equal and say nothing.
    assert!(
        !split_mass[0].is_empty() && !split_mass[1].is_empty(),
        "the two capturing lanes read no attention mass",
    );
    assert!(
        split_mass[2].is_empty(),
        "the plain lane read attention mass it never asked for",
    );
    let entries: usize = split_mass[0].iter().map(|layer| layer.lse.len()).sum();
    assert!(entries > 0, "the capture columns are empty");
    eprintln!(
        "diffed {} logits per lane and {entries} capture entries, bit for bit",
        split_out[0].len(),
    );
}

/// `(how many windows this composition leaves in pieces, how many launches a
/// split pays for them beyond one apiece)` — read off a fresh bake of the same
/// text at the same budgets, which is where the number belongs.
fn predicted() -> (u32, u32) {
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let compiled = compile(&trace, &BUDGETS, &DeviceProfile::default()).expect("the SKU bakes");
    let lanes = [
        model_exec::fire::Lane::new(word(1, true), 1),
        model_exec::fire::Lane::new(word(5, true), 5),
        model_exec::fire::Lane::new(word(4, false), 4),
    ];
    let fire = model_exec::fire::compose(&compiled, &BUDGETS, &lanes).expect("the fire composes");
    let mut fragmented = 0u32;
    let mut extra = 0u32;
    for region in compiled.template() {
        let runs = fire.classes().spans(&region.mask).len() as u32;
        if runs > 1 {
            fragmented += 1;
            extra += runs - 1;
        }
    }
    (fragmented, extra)
}

// ── (e) three runs into one, which needs five lanes ──────────────────────

/// **THE FULL `r = 3`, WHICH THE THREE-LANE FIRE CANNOT REACH.**
///
/// P4's own count for the withdrawn window is `Fallback::Split { r: 3 }`: on
/// the shipped order `4 0 2 6 7 3 1 5` the mask `{4,5,6,7}` sits at positions
/// `0`, `3`, `4`, `7`, which is three runs — `[4] [6 7] [5]`. A fire realises
/// all three only if it carries a class from `{6,7}` as well as `4` and `5`
/// with a gap on either side, and `6` and `7` are the classes whose word sets
/// `has_adapter`. So the composition is FIVE lanes and one of them routes:
///
/// ```text
/// present, in the baked order:  [ 4 | 0 | 6 | 1 | 5 ]
/// mask {4,5,6,7}:                 ─      ─       ─      3 runs
///   4  capturing prefill, no adapter
///   0  plain prefill
///   6  capturing prefill WITH an adapter
///   1  plain decode
///   5  capturing decode
/// ```
///
/// Everything else is gate (a)'s claim at a wider budget: the same fire split
/// and copied is the same bytes, and the copied region costs one launch where
/// the split cost three.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_window_in_three_pieces_becomes_one_launch_and_still_the_same_bytes() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready_wide("the three-run copy") else {
        return;
    };
    register_loud(&mut shell, 0);

    let prompts = [
        tok.encode(CAPTURING),
        tok.encode(PLAIN),
        tok.encode(LATE),
        tok.encode("Iron melts at"),
        tok.encode("The moon orbits"),
    ];
    // (slot, captures, adapter, decode) — the five classes above, in the
    // order the classes come out rather than in fire order, which `compose`
    // decides.
    let held: [(u32, bool, Option<u32>, bool); 5] = [
        (0, true, None, false),    // class 4
        (1, false, None, false),   // class 0
        (2, true, Some(0), false), // class 6
        (3, false, None, true),    // class 1
        (4, true, None, true),     // class 5
    ];

    let go = |shell: &mut Shell| {
        for (slot, ..) in held {
            shell.open(slot).expect("the slot opens");
        }
        let one = [prompts[3][0]];
        let two = [prompts[4][0]];
        let seated: Vec<Seated<'_>> = held
            .iter()
            .enumerate()
            .map(|(at, &(slot, captures, adapter, decode))| {
                let tokens: &[u32] = match (decode, at) {
                    (false, at) => &prompts[at],
                    (true, 3) => &one,
                    (true, _) => &two,
                };
                Seated {
                    lane: engine_cuda::Lane {
                        slot,
                        word: wide_word(tokens.len() as u32, captures, adapter.is_some()),
                        tokens,
                    },
                    captures_scores: captures,
                    adapter,
                    ..Seated::of(engine_cuda::Lane {
                        slot,
                        word: 0,
                        tokens,
                    })
                }
            })
            .collect();
        let mut mass: Vec<Vec<LayerScores>> = Vec::new();
        let out = shell
            .fire_captured(&seated, &[], &mut mass)
            .expect("the five-lane fragmented fire");
        (out, mass, shell.last_fire_cost())
    };

    shell.set_copies(false);
    let _ = go(&mut shell);
    let (split_out, split_mass, split_cost) = go(&mut shell);
    shell.set_copies(true);
    let _ = go(&mut shell);
    let (copy_out, copy_mass, copy_cost) = go(&mut shell);

    eprintln!(
        "five lanes — split: {} launches | copy: {} launches, {} regions gathered",
        split_cost.launches, copy_cost.launches, copy_cost.copied,
    );

    // THE PREMISE: some region really did come back in THREE pieces, which is
    // the whole reason this gate exists beside the three-lane one.
    let (fragmented, extra) = predicted_wide();
    assert!(
        extra >= 2 * fragmented,
        "no window of this composition breaks into three; {fragmented} broke into \
         {} pieces in total",
        extra + fragmented,
    );
    assert_eq!(
        copy_cost.copied, fragmented,
        "not every fragmented window was gathered"
    );
    assert_eq!(
        split_cost.launches - copy_cost.launches,
        extra,
        "three-into-one is the claim and the launch counts do not show it",
    );

    same_logits(&split_out, &copy_out, "the five-lane fire");
    same_mass(&split_mass, &copy_mass, "the five-lane fire");
    assert!(
        !split_mass[0].is_empty() && !split_mass[2].is_empty() && !split_mass[4].is_empty(),
        "the three capturing lanes read no attention mass",
    );
}

/// The word a lane of the five-class fire carries.
fn wide_word(query_len: u32, captures: bool, adapter: bool) -> u64 {
    model::qwen_3::forward::Facts::of(
        &Request::new(query_len, false)
            .capturing_scores(captures)
            .adapted(adapter),
    )
    .word()
}

/// `(fragmented windows, launches a split pays beyond one apiece)` for the
/// five-lane composition, off a fresh bake.
fn predicted_wide() -> (u32, u32) {
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let compiled = compile(&trace, &WIDE, &DeviceProfile::default()).expect("the SKU bakes");
    let lanes = [
        model_exec::fire::Lane::new(wide_word(5, true, false), 5),
        model_exec::fire::Lane::new(wide_word(3, false, false), 3),
        model_exec::fire::Lane::new(wide_word(4, true, true), 4),
        model_exec::fire::Lane::new(wide_word(1, false, false), 1),
        model_exec::fire::Lane::new(wide_word(1, true, false), 1),
    ];
    let fire = model_exec::fire::compose(&compiled, &WIDE, &lanes).expect("the five lanes compose");
    let mut fragmented = 0u32;
    let mut extra = 0u32;
    for region in compiled.template() {
        let runs = fire.classes().spans(&region.mask).len() as u32;
        if runs > 1 {
            fragmented += 1;
            extra += runs - 1;
        }
    }
    (fragmented, extra)
}

/// A LOUD adapter — big enough that its correction is visibly not the
/// identity, so a copy that dropped the adapted lane's rows would not pass
/// the byte diff by accident.
fn register_loud(shell: &mut Shell, id: u32) {
    let built: Vec<(String, Vec<u8>)> = shell
        .banks()
        .iter()
        .map(|&(name, _, slot)| {
            let count = usize::try_from(slot).expect("a slot fits this host") / 2;
            let mut bytes = Vec::with_capacity(count * 2);
            for at in 0..count {
                let value = if name.ends_with(".lora_a") {
                    0.20 - ((at % 5) as f32) * 0.07
                } else {
                    0.15 + ((at % 3) as f32) * 0.05
                };
                bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            (name.to_string(), bytes)
        })
        .collect();
    let planes: Vec<engine_cuda::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| engine_cuda::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .register_adapter(id, &planes)
        .unwrap_or_else(|why| panic!("registering adapter {id}: {why}"));
}

/// f32 to bf16, round-to-nearest-even — the loader's conversion, restated so
/// the adapter registered is the one described.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

// ── (d) the wall clock, on this device ───────────────────────────────────

/// **WHAT THE TWO ACTUALLY COST HERE**, because the 1.82x-against-1.07x table
/// `model_compiler::layout` carries is from an RTX 3090 and nobody has
/// re-measured it.
///
/// NOT AN ASSERTION ABOUT WHICH IS FASTER, and deliberately so — the numbers
/// are the output and the test passes either way.
///
/// **AND IT IS NOT A MEASUREMENT OF THE CROSSOVER TABLE'S CLAIM.** That table
/// is a two-way split of a dense fp16 GEMM at `K=N=4096` against a
/// copy-then-dense, and the consumer P4 actually withdrew here is
/// `attention.prefill_lse` — a paged attention whose cost is dominated by kv
/// traffic, not by GEMM tile quanta. What a copy saves on THIS consumer is
/// launches (and one attention schedule build per interval saved), and what
/// it costs is a gather and a scatter of the window's rows. So this measures
/// that trade at two sizes and says so, rather than dressing itself up as a
/// re-measurement of `CROSSOVER_ROWS`.
///
/// Both runs are interleaved (split, copy, split, copy) so a device that
/// drifts under thermal load drifts through both.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn what_a_copy_and_a_split_cost_on_this_device() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the copy/split timing") else {
        return;
    };
    let short = vec![
        tok.encode(CAPTURING)[..1].to_vec(),
        tok.encode(LATE),
        tok.encode(PLAIN),
    ];
    // The same three classes at prefill scale: a capturing decode lane beside
    // a long capturing prefill and a long plain prefill, so the copied
    // region's window is ~200 rows in two pieces rather than 2 rows in two.
    let filler: Vec<u32> = tok.encode(&format!("{CAPTURING} {LATE} {PLAIN} ").repeat(24));
    let long = vec![
        short[0].clone(),
        filler[..100.min(filler.len())].to_vec(),
        filler[..100.min(filler.len())].to_vec(),
    ];
    const FIRES: usize = 40;
    const WARM: usize = 8;

    for (what, fed) in [("3 rows", &short), ("201 rows", &long)] {
        let mut said: Vec<(bool, f64, u32)> = Vec::new();
        for copies in [false, true, false, true] {
            shell.set_copies(copies);
            for _ in 0..WARM {
                let _ = fire_it(&mut shell, fed);
            }
            let at = Instant::now();
            let mut cost = engine_cuda::FireCost::default();
            for _ in 0..FIRES {
                cost = fire_it(&mut shell, fed).2;
            }
            said.push((
                copies,
                at.elapsed().as_secs_f64() * 1e6 / FIRES as f64,
                cost.launches,
            ));
        }
        for (copies, us, launches) in &said {
            eprintln!(
                "{what:>9}  {:<6} {us:9.1} us/fire over {FIRES} fires, {launches} launches",
                if *copies { "copy" } else { "split" },
            );
        }
        let split = (said[0].1 + said[2].1) / 2.0;
        let copy = (said[1].1 + said[3].1) / 2.0;
        eprintln!(
            "{what:>9}  copy is {:+.2}% against the split ({:+.1} us/fire over {} launches saved)",
            (copy - split) / split * 100.0,
            copy - split,
            said[0].2 - said[1].2,
        );
        assert!(said[1].2 < said[0].2, "the copy saved no launch at {what}");
    }
}

// ── (f) and the recorded graph carries it ────────────────────────────────

/// **A COPY UNDER `Graphs::On`**, which is the one place the scratch slab's
/// own contract could bite.
///
/// `Ctx::scratch` grows by `cudaFree` + `cudaMalloc`, which under
/// `cudaStreamBeginCapture` is a typed refusal rather than a corruption. The
/// record path's answer for every other scratch consumer is that a key's
/// first fires are EAGER and at the same shape (`record::WARM_FIRES`), so the
/// slab a capture pass reads has already been grown by the eager pass in
/// front of it — and the copy slab is sized off the window table, which is
/// what `record::Key` IS. So the argument says this works; this is the gate
/// for it, and a `Fault::Unwarmed` here would be the argument failing rather
/// than a surprise.
///
/// The composition is repeated so it reaches a capture and then a replay,
/// which is `export_axes`'s own recipe for the split.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_copied_window_replays_out_of_a_recorded_graph_identically() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the recorded copy") else {
        return;
    };
    let fed = vec![
        tok.encode(CAPTURING)[..1].to_vec(),
        tok.encode(LATE),
        tok.encode(PLAIN),
    ];

    shell.set_copies(true);
    shell.set_mode(Graphs::Off);
    for _ in 0..2 {
        let _ = fire_it(&mut shell, &fed);
    }
    let (eager_out, eager_mass, eager_cost) = fire_it(&mut shell, &fed);
    assert!(eager_cost.copied > 0, "the eager half copied nothing");

    shell.set_mode(Graphs::On);
    // Three fires: the first two are eager passes of the key, the third is
    // the capture, and everything after replays. `WARM_FIRES` is 2.
    for _ in 0..3 {
        let _ = fire_it(&mut shell, &fed);
    }
    let (replay_out, replay_mass, replay_cost) = fire_it(&mut shell, &fed);
    let stats = shell.graph_stats();
    eprintln!(
        "recorded copy: {} captures, {} replays, {} warming, {} declined; \
         {} launches, {} gathered",
        stats.captures,
        stats.replays,
        stats.warming,
        stats.declined,
        replay_cost.launches,
        replay_cost.copied,
    );
    assert!(
        stats.captures > 0,
        "nothing was captured, so nothing replayed"
    );
    assert!(stats.replays > 0, "no fire replayed out of a graph");

    same_logits(&eager_out, &replay_out, "the recorded copy");
    same_mass(&eager_mass, &replay_mass, "the recorded copy");
}

// ── the load ─────────────────────────────────────────────────────────────

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
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

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    load(what, BUDGETS, 4)
}

/// The same load at [`WIDE`] — five slots and an adapter seat.
fn ready_wide(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    load(what, WIDE, 8)
}

fn load(what: &str, budget: Budget, slots: u32) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
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
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    // **THIS FILE NEEDS A NON-GROUPABLE CONSUMER WITHDRAWN, AND THE DEFAULT
    // NO LONGER LEAVES ONE.** `crate::GROUPED` is named by default now, and a
    // groupable consumer is nearly free to withdraw — so `layout::choose`
    // takes the correction and seats the score window, and the catalog has
    // nothing left for a copy to gather. Emptying the list puts the score
    // window back on the losing side, which is the artifact this file prices.
    //
    // Stated on the `Boot` rather than set in the environment (alto wave P,
    // article 9): the word is a `Knobs` field now, so no `unsafe` block and no
    // argument about which thread is loading a shell.
    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget,
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs {
            grouped: false,
            ..engine_cuda::Knobs::default()
        },
        program_cache_dir: None,
        // F1's depth, kept: these gates fire one step at a time and
        // read its numbers, so a deeper ring would carve slots nothing
        // claims. `Runahead::of` is the door a deployment comes through.
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot weight artifact cache is off for a gate: a test
        // that shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}
