//! **W-2's third mix: `{dense spill via D2b}`** (alto streaming §2, "Dense
//! (static)"; build-order item 4).
//!
//! The other two mixes are green: `{T0 capped}` in
//! `a_capped_moe_serves_the_tokens_it_would_have` and `{T0+T1+T2}` in
//! `a_model_larger_than_both_budgets_serves`. Both spill ROUTED banks, whose
//! demand is dynamic. This one spills the other shape — the DENSE planes, the
//! embedding and the projections and the head, which every token of every fire
//! reads unconditionally — and asserts the same thing about it:
//!
//! ```text
//! (a) a dense model under a `device_weight_budget` below its table SERVES,
//!     where until this wave it was refused ("nothing in it is a routed-expert
//!     bank, so there is no tier to hold less of").
//! (b) it answers the uncapped load's greedy tokens BYTE FOR BYTE, and the
//!     logits are the same floats.
//! (c) what spilled is the SCHEDULE's tail. The plane a fire reads first is
//!     the plane the budget surrenders last, so the embedding stays and the
//!     late layers leave — `model_compiler::prefetch` is where that order
//!     comes from and it is a pure function of the plan.
//! (d) THE PUMP IS ARMED AND IT MOVED THE BYTES (D2b's second half, wave B8):
//!     the spilled planes under the slot cap are copied ahead into rotating
//!     device slots on a copy stream, forked and joined against compute at
//!     every region boundary. Structural, not timed — (b) is what says the
//!     rotation is right, and the step times below are what say what it bought.
//! ```
//!
//! # The physics, stated up front and measured, not gated
//!
//! Streaming §2: *a spilled dense byte crosses NVMe/PCIe EVERY STEP; step time
//! floors at `spill_bytes / min(NVMe, PCIe)`.* That is a property of the TIER
//! and not of the pump: whether the bytes arrive as a bulk `memcpy` issued
//! ahead of the read or as the read's own UVA traffic, they cross once per
//! step either way. This gate reports the measured step time beside the
//! computed floor and asserts NOTHING about it — the deliverable is the
//! capability and the identity; the speed is the trade, stated.
//!
//! And the SPEED is what the pump is for. Streaming §3 item 4 priced it before
//! the wave was spent: the tier is bandwidth-bound (~28 GB/s effective), so a
//! pump cannot buy bandwidth and can only buy OVERLAP — hiding the transfer
//! under the compute it was not overlapped with. Best case the spilled step
//! falls from `copy + compute` to `max(copy, compute)`, which on this rig is
//! the PCIe floor plus a ramp. Still measured, still printed, still gated on
//! nothing.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_spilled_dense_model_says_what_it_said -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";
const STEPS: usize = 12;

/// **The budget, as a fraction of the table.** Low enough that most of the
/// plan's layers leave the device and high enough that the embedding — which
/// the schedule reads first — stays, so the gate exercises the ORDER and not
/// just the tier.
const CAP: u64 = 2;
const OF: u64 = 5;

/// PCIe gen5 x16, the read direction, as a round number. Only used to print
/// the floor beside the measurement — nothing is asserted against it.
const PCIE_BYTES_PER_SEC: f64 = 25e9;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

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

struct Rig {
    trace: model_ir::Trace,
    contract: checkpoint::contract::ModelContract,
    checkpoint: PathBuf,
    tokenizer: tokenizer::Tokenizer,
}

fn rig(what: &str) -> Option<Rig> {
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
    let Some(one) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&[one]).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Some(Rig {
        trace,
        contract,
        checkpoint,
        tokenizer,
    })
}

fn load(rig: &Rig, residency: Plan) -> engine_cuda::Result<Shell> {
    Shell::load(Boot {
        trace: rig.trace.clone(),
        contract: &rig.contract,
        checkpoint: &rig.checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
        residency,
    })
}

/// A prefill and `STEPS` greedy decodes. Answers the tokens, the logit rows,
/// and the mean DECODE step time — the prefill is excluded because it reads a
/// different number of rows and is not what the physics sentence is about.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>, f64) {
    shell.open(0).expect("slot 0 opens");
    let mut chosen = Vec::with_capacity(STEPS);
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");
    let mut fed = argmax(&prefill[0]);
    chosen.push(fed);
    rows.push(prefill[0].clone());

    let started = Instant::now();
    for step in 0..STEPS {
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        finite(&decode[0], "decode");
        fed = argmax(&decode[0]);
        chosen.push(fed);
        rows.push(decode[0].clone());
    }
    let each = started.elapsed().as_secs_f64() / STEPS as f64;
    (chosen, rows, each)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B snapshot; \
            run it with `-- --ignored`"]
fn a_spilled_dense_model_says_what_it_said() {
    let _one = serialized();
    let Some(rig) = rig("W-2's dense-spill mix") else {
        return;
    };
    let prompt = rig.tokenizer.encode(PROMPT);

    let full = Plan::of(&rig.trace, &Default::default(), Budgets::uncapped())
        .expect("a dense plan plans")
        .device_demand();
    let budget = full * CAP / OF;
    eprintln!("qwen35-d0.8b: {full} bytes of table; the budget is {budget}");

    // ── THE PLAN. Dense planes, spilled by the compiler's schedule.
    let plan = Plan::of(&rig.trace, &Default::default(), Budgets::device(budget))
        .expect("a dense plan under a budget spills rather than refusing — that IS D2b");
    assert!(plan.streams(), "two fifths of the table cannot be held whole");
    assert!(plan.banks().is_empty(), "nothing here is a routed bank");
    assert!(!plan.groups().is_empty(), "and dense planes are what left");
    let spilled: u64 = plan.groups().iter().map(|group| group.bytes).sum();
    eprintln!(
        "planned: T0 {} bytes, T1 {} bytes, {} of {} planes spilled",
        plan.device_demand(),
        plan.host_demand(),
        plan.groups().len(),
        rig.trace.params.len(),
    );

    // ── (c) THE ORDER IS THE SCHEDULE'S, and the schedule is the compiler's.
    let schedule = model_compiler::prefetch::Schedule::of(&rig.trace);
    let rank: BTreeMap<usize, usize> = schedule
        .order()
        .into_iter()
        .enumerate()
        .map(|(at, param)| (param, at))
        .collect();
    let first = schedule.order()[0];
    assert!(
        !plan.pinned(first) && !plan.mapped(first),
        "the plane this fire reads FIRST (`{}`) is the one a budget gives up LAST",
        rig.trace.params[first].name
    );
    for group in plan.groups() {
        assert!(!group.routed, "`{}` is a dense plane", group.name);
        assert_eq!(group.planes.len(), 1, "and a group of one");
    }
    let earliest_spilled = plan
        .groups()
        .iter()
        .filter_map(|group| rank.get(&group.param).map(|at| (*at, group.name.clone())))
        .min()
        .expect("something spilled");
    eprintln!(
        "the schedule's tail spilled from rank {} (`{}`) onward",
        earliest_spilled.0, earliest_spilled.1
    );

    // ── THE GOLDEN, UNCAPPED.
    let mut resident = load(&rig, Plan::default()).expect("the uncapped shell loads");
    assert!(resident.weights_resident());
    let (golden, golden_rows, resident_step) = run(&mut resident, &prompt);
    let says = rig.tokenizer.decode(&golden, false);
    eprintln!("uncapped answers: {says:?} at {:.2} ms/step", resident_step * 1e3);
    drop(resident);

    // ── (a) THE CAPABILITY.
    let mut streamed = load(&rig, plan).expect("a spilled dense model serves");
    assert!(
        !streamed.weights_resident(),
        "a spilled load says so rather than claiming the table"
    );
    let (tokens, rows, spilled_step) = run(&mut streamed, &prompt);
    let also = rig.tokenizer.decode(&tokens, false);

    // ── (d) THE PUMP. Armed at load, and it moved bytes on every fire.
    let pumped = streamed.rotation();
    match pumped {
        Some((observed, slots, arena, rotating)) => eprintln!(
            "the pump: {slots} slots over {arena} bytes of arena rotate {rotating} \
             bytes a step; {observed:?}",
        ),
        None => eprintln!("the pump: nothing armed"),
    }
    let (observed, _, arena, rotating) = pumped.expect(
        "a spilled dense load arms the rotating pump — that is D2b's second half",
    );
    assert!(rotating > 0, "the rotation moves no bytes at all");
    assert!(
        arena < rotating,
        "an arena of {arena} bytes holding {rotating} rotating bytes is residency, \
         not a pump",
    );
    assert!(
        observed.copies >= (STEPS as u64),
        "the pump issued {} copies over {} fires",
        observed.copies,
        STEPS + 1,
    );

    // ── THE PHYSICS, MEASURED AND PRINTED, GATED ON NOTHING.
    let floor = spilled as f64 / PCIE_BYTES_PER_SEC;
    eprintln!(
        "spilled answers:  {also:?} at {:.2} ms/step\n\
         physics: {spilled} spilled bytes cross every step; the PCIe floor is \
         {:.2} ms/step and the measured cost of spilling is {:.2} ms/step \
         ({:.1}x the resident step)",
        spilled_step * 1e3,
        floor * 1e3,
        (spilled_step - resident_step) * 1e3,
        spilled_step / resident_step,
    );
    eprintln!(
        "the pump moved {} bytes over {} fires with {} late acquisitions",
        observed.bytes, observed.fires, observed.late,
    );

    assert!(
        tokens.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the spilled load answered {tokens:?}, which is one token repeated"
    );

    // ── (b) BIT-IDENTITY. The third mix.
    assert_eq!(
        tokens, golden,
        "the spilled load chose {tokens:?} and the uncapped one chose {golden:?}"
    );
    for (step, (a, b)) in golden_rows.iter().zip(&rows).enumerate() {
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: uncapped {x}, spilled {y} — the tier moved a number"
            );
        }
    }
}
