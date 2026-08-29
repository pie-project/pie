//! **THE GATE FOR `Fallback::Grouped`: THE SPLIT IS THE ORACLE.**
//!
//! Design §3's menu has four entries. `Split { r }` — run the kernel once per
//! maximal interval — is the one that needs nothing from a backend, and it has
//! been green on device since palo E. `Grouped` is the one tart expects to
//! dominate it: ONE launch handed the interval list, which the kernel walks
//! itself. Same rows, same arithmetic, fewer launches — which makes the split
//! a free oracle, and a grouped answer that disagrees with it by one bit is
//! wrong.
//!
//! So this file fires one fragmented composition against two artifacts that
//! differ in exactly one thing, and asserts both halves of the claim:
//!
//! ```text
//! (a) bit for bit     every lane's logits, every step, f32 bits compared
//! (b) fewer launches  the captured graph holds exactly 2 x nodes x (r - 1)
//!                     fewer — two kernels per correction site, r - 1 of the
//!                     intervals no longer launched for
//! ```
//!
//! **(b) IS THE HALF THAT CANNOT BE FAKED.** A grouped answer that was
//! secretly still splitting would pass (a) trivially — it would BE the split —
//! and the node count is what says it is not. It is read off the recorded
//! graph rather than off a counter this file keeps, because a graph node is a
//! launch that was actually enqueued.
//!
//! # The two artifacts, and why they are one row order
//!
//! `DeviceProfile::grouped` (a labelled PoC scaffold — read its doc)
//! names the op whose windowed consumer P4 should withdraw, and both arms name
//! the correction. `PIE_CUDA_GROUPED=off` empties `DeviceProfile::grouped`,
//! which is what this shell can SERVE. So the two bakes withdraw the same
//! consumer onto the same frontier and differ only in the answer written for
//! it: `Copy`/`Split { r }` in one, `Grouped` in the other. Two profile lists
//! rather than one, exactly so that this comparison is a comparison.
//!
//! Without the scaffold the correction is SEATED on this catalog — it wins the
//! C1P competition against `captures_scores` because `BTreeMap` puts
//! `{2,3,6,7}` before `{4,5,6,7}` and `4 > 2` — and nothing here would run.
//! The scaffold is empty by default, and
//! `a_grouped_window_is_one_window_and_a_segment_list.rs`'s third test is what
//! says the shipped bake is untouched by any of this.
//!
//! # The composition
//!
//! Eight lanes, one per class of the SKU: `{qo_one} x {has_adapter} x
//! {captures_scores}`, a decode and a prefill of each. Under the withdrawn
//! frontier `[0 2 4 6 5 7 1 3]` the adapter classes `{2,3,6,7}` land at
//! positions 1, 3, 5 and 7 with a non-adapted class between every pair — four
//! intervals, which is the worst this text can present and the most the answer
//! can save.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!   --test a_grouped_correction_says_what_a_split_one_says -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{AdapterPlane, Boot, Graphs, Lane, LayerScores, Seated, Shell};
use model_compiler::{Budget, DeviceProfile, Fallback, FamilyCosts};
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The op both profile lists name.
const CORRECTION: &str = "linear.lora_correct";

/// The adapter id every adapted lane routes to.
const ADAPTER: u32 = 0;

/// How many fragmented fires one pass runs.
const STEPS: usize = 5;

const CARRIED: &str = "The capital of France is";
const FRESH: &str = "Water freezes at a temperature of";

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole). This
/// file also writes an environment variable `Shell::load` reads, which is a
/// second reason the two arms may not overlap.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
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

fn budget(seats: u32) -> Budget {
    Budget {
        max_lanes: 8,
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: seats,
    }
}

/// The lane word the model's own `Classify` computes — the three facts this
/// composition varies, and no second opinion about any of them.
fn word(query_len: u32, adapted: bool, captures: bool) -> u64 {
    model::qwen_3::forward::Facts::of(
        &Request::new(query_len, false)
            .adapted(adapted)
            .capturing_scores(captures),
    )
    .word()
}

/// One lane, seated with the two asks its word claims. Both halves are
/// required: the shell refuses a capturing word with no capture ask
/// (`Fault::ScoreWord`) and an adapter against a word that does not route
/// (`Fault::AdapterWord`), which is what makes eight distinct classes eight
/// distinct submissions rather than eight spellings of one.
fn seat(slot: u32, tokens: &[u32], adapted: bool, captures: bool) -> Seated<'_> {
    let lane = Lane {
        slot,
        word: word(tokens.len() as u32, adapted, captures),
        tokens,
    };
    let mut seated = if captures {
        Seated::capturing(lane)
    } else {
        Seated::of(lane)
    };
    if adapted {
        seated.adapter = Some(ADAPTER);
    }
    seated
}

/// The eight lanes, as `(adapted, captures)` in fire order. Slots 0..4 carry a
/// sequence and decode; slots 4..8 prefill a fresh one every step.
const CLASSES: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

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

/// A LOUD adapter, built by hand: entries big enough that twenty-four stacked
/// corrections take the continuation somewhere else entirely.
///
/// **LOUD ON PURPOSE.** A zero adapter would make (a) pass for a grouped path
/// that corrected NO row — the identity is the identity either way — so the
/// adapter has to be one whose absence is visible, and the assertion that an
/// adapted lane says something different from its base neighbour is what says
/// the correction reached the readout at all.
fn loud(shell: &Shell) -> Vec<(String, Vec<u8>)> {
    shell
        .banks()
        .iter()
        .map(|&(name, _, slot)| {
            let count = usize::try_from(slot).expect("a slot fits this host") / 2;
            let mut bytes = Vec::with_capacity(count * 2);
            for at in 0..count {
                let sign = if at % 2 == 0 { 1.0 } else { -1.0 };
                let value = if name.ends_with(".lora_a") {
                    sign * 0.02 * (((at % 11) as f32) + 1.0)
                } else {
                    sign * 0.02 * (((at % 7) as f32) + 1.0)
                };
                bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            (name.to_string(), bytes)
        })
        .collect()
}

/// f32 to bf16, round-to-nearest-even — the conversion the loader does.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn register(shell: &mut Shell, built: &[(String, Vec<u8>)]) {
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .register_adapter(ADAPTER, &planes)
        .unwrap_or_else(|why| panic!("registering adapter {ADAPTER}: {why}"));
}

/// The fragmented sequence: four carried lanes decoding beside four fresh ones
/// prefilling, `STEPS` times, one lane per class.
///
/// **THE FRESH LANES ARE RE-OPENED EVERY STEP**, which is what makes the
/// composition repeat and therefore capture: slots 4..8 prefill the same
/// prompt over and over while slots 0..4 decode, so the class table — and so
/// the graph key — is the same from the first mixed fire on.
///
/// **THE CARRIED PREFILL RUNS UNDER `Graphs::Off` WHATEVER `mode` SAYS**, so
/// the mixed key is the only one this shell ever captures and
/// `Stats::nodes` — the most recently captured graph — is unambiguously the
/// fragmented one. Without that, the last capture is the four-lane prefill,
/// whose adapter window is one interval and whose node count is the same in
/// both arms: a green test of the wrong graph.
fn sequence(shell: &mut Shell, mode: Graphs, carried: &[u32], fresh: &[u32]) -> Vec<Vec<f32>> {
    shell.set_mode(Graphs::Off);
    for slot in 0..8 {
        shell.open(slot).expect("the slot opens");
    }
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    let opening: Vec<Seated<'_>> = CLASSES
        .iter()
        .enumerate()
        .map(|(at, &(adapted, captures))| seat(at as u32, carried, adapted, captures))
        .collect();
    let first = shell
        .fire_captured(&opening, &[], &mut mass)
        .expect("the carried prefill fires");
    let mut fed: Vec<[u32; 1]> = first.iter().map(|out| [argmax(out)]).collect();

    shell.set_mode(mode);
    let mut out: Vec<Vec<f32>> = Vec::new();
    for step in 0..STEPS {
        for slot in 4..8 {
            shell.open(slot).expect("the fresh slot re-opens");
        }
        let mut seated: Vec<Seated<'_>> = Vec::with_capacity(8);
        for (at, &(adapted, captures)) in CLASSES.iter().enumerate() {
            seated.push(seat(at as u32, &fed[at], adapted, captures));
        }
        for (at, &(adapted, captures)) in CLASSES.iter().enumerate() {
            seated.push(seat((at + 4) as u32, fresh, adapted, captures));
        }
        let said = shell
            .fire_captured(&seated, &[], &mut mass)
            .unwrap_or_else(|why| panic!("the fragmented fire at step {step}: {why}"));
        assert_eq!(said.len(), 8, "eight lanes, eight readouts");
        fed = said.iter().take(4).map(|out| [argmax(out)]).collect();
        out.extend(said);
    }
    out
}

/// What one arm produced.
struct Arm {
    /// Every lane's readout, every step, in fire order — the bits (a) compares.
    logits: Vec<Vec<f32>>,
    /// The greedy tokens, for the sentence a failure prints.
    said: Vec<u32>,
    /// Nodes in the graph the fragmented key captured — (b)'s number.
    nodes: usize,
    /// How many `Grouped` rows the loaded artifact owes for the correction,
    /// and how many `Split`/`Copy` ones. The premise, read off the bake.
    grouped_rows: usize,
    split_rows: usize,
    /// How many intervals the adapter window covers in this composition.
    runs: usize,
    /// How many correction nodes the plan states.
    corrections: usize,
}

/// Load a shell, register the adapter, and run the fragmented sequence three
/// times: warm, golden, captured.
///
/// `grouped` chooses the ARM by choosing the artifact. The variable is set
/// under the caller's serialization guard and read once, inside `Shell::load`.
///
/// **THE WARM PASS IS NOT CEREMONY.** The dense autotuner tunes a GEMM shape
/// on its second sighting (build log 11), so a cold pass and a warm one are
/// two tactic ladders and comparing across them compares the tuner. Every
/// number below comes from a steady state, exactly as `adapter_banks` and
/// `masked_axis` take theirs.
fn arm(what: &str, grouped: bool) -> Option<Arm> {
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
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .expect("the SKU declares adapter banks");
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    // SAFETY: the caller holds `ONE_AT_A_TIME`, no other thread of this
    // process is loading a shell, and `Shell::load` reads the variable once
    // before it returns.
    unsafe {
        if grouped {
            std::env::remove_var("PIE_CUDA_GROUPED");
        } else {
            std::env::set_var("PIE_CUDA_GROUPED", "off");
        }
        // **AND NEITHER ARM COPIES, BECAUSE THIS FILE PRICES GROUPED AGAINST
        // SPLIT.** `Fallback::Copy` is on by default now and is the menu's
        // answer below the crossover, so an arm that left it on would serve
        // its withdrawn consumer as a gather plus one launch — a third thing,
        // agreeing with neither arm, and the launch delta below would be
        // measuring Grouped against Copy under a name that says otherwise.
        // (The grouped arm never reaches it — `walk` gives `Grouped` the tie —
        // but it is set on both so that ONE word differs between them.)
        std::env::set_var("PIE_CUDA_FALLBACK_COPY", "off");
    }

    let budget = budget(u32::try_from(seats).expect("a capacity fits a u32"));
    let mut shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: budget.clone(),
        // **THE TWO ARMS MUST WITHDRAW THE SAME CONSUMER**, or they are two
        // artifacts and the comparison prices nothing. The withdrawal is
        // chosen by cost (`model_compiler::layout::choose`), and naming an op
        // groupable is itself a discount on withdrawing it — so the grouped
        // arm withdraws the correction because it is groupable, and the split
        // arm is told the correction is cheap instead. Same mask withdrawn,
        // same row order, and the only thing left differing is the ANSWER.
        profile: Some(DeviceProfile {
            family_us: if grouped {
                DeviceProfile::default().family_us
            } else {
                FamilyCosts {
                    linear: 1.0,
                    ..DeviceProfile::default().family_us
                }
            },
            ..DeviceProfile::default()
        }),
        page_size: 16,
        context: 512,
        slots: 8,
        ordinal: 0,
        graphs: Graphs::Off,
        // F1's depth, kept: these gates fire one step at a time and
        // read its numbers, so a deeper ring would carve slots nothing
        // claims. `Runahead::of` is the door a deployment comes through.
        runahead: engine::runahead::Runahead::F1,
    })
    .expect("the shell loads");
    let planes = loud(&shell);
    register(&mut shell, &planes);

    // ── the premise, read off the bake rather than assumed ───────────────
    let compiled = shell.compiled_model();
    let corrections: Vec<u32> = shell
        .trace()
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| model_ir::Operands::name(&node.op) == CORRECTION)
        .map(|(at, _)| at as u32)
        .collect();
    let mut grouped_rows = 0usize;
    let mut split_rows = 0usize;
    for row in &compiled.fallback.rows {
        if !corrections.contains(&row.node) {
            continue;
        }
        match row.fallback {
            Fallback::Grouped => grouped_rows += 1,
            Fallback::Copy | Fallback::Split { .. } => split_rows += 1,
            Fallback::View => {}
        }
    }

    let carried = tokenizer.encode(CARRIED);
    let fresh = tokenizer.encode(FRESH);

    // How many intervals the adapter window covers in the mixed composition
    // below — composed against the artifact this arm actually loaded.
    let mask = compiled
        .template()
        .iter()
        .find(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
        .map(|region| region.mask.clone())
        .expect("some region holds a correction");
    let mut mixed: Vec<engine::fire::Lane> = Vec::with_capacity(8);
    for &(adapted, captures) in &CLASSES {
        mixed.push(engine::fire::Lane::new(word(1, adapted, captures), 1));
    }
    for &(adapted, captures) in &CLASSES {
        let rows = fresh.len() as u32;
        mixed.push(engine::fire::Lane::new(word(rows, adapted, captures), rows));
    }
    let runs = engine::fire::compose(compiled, &budget, &mixed)
        .expect("the mixed fire composes")
        .classes()
        .spans(&mask)
        .len();

    // ── the fires ────────────────────────────────────────────────────────
    let _warm = sequence(&mut shell, Graphs::Off, &carried, &fresh);
    let golden = sequence(&mut shell, Graphs::Off, &carried, &fresh);
    let replayed = sequence(&mut shell, Graphs::On, &carried, &fresh);
    let stats = shell.graph_stats();
    assert!(
        stats.captures >= 1,
        "{what}: nothing captured, so the launch count below is nobody's: {stats:?}",
    );
    let said: Vec<u32> = golden.iter().map(|out| argmax(out)).collect();
    assert_eq!(
        said,
        replayed.iter().map(|out| argmax(out)).collect::<Vec<_>>(),
        "{what}: the replayed fires disagreed with the eager ones they were \
         captured from, which is a `graph_replay` failure and not this file's",
    );

    eprintln!(
        "{what}: r={runs}, {} correction nodes, {grouped_rows} grouped rows / \
         {split_rows} split rows, {} graph nodes over {} captures",
        corrections.len(),
        stats.nodes,
        stats.captures,
    );

    Some(Arm {
        logits: golden,
        said,
        nodes: stats.nodes,
        grouped_rows,
        split_rows,
        runs,
        corrections: corrections.len(),
    })
}

// ── the gate ─────────────────────────────────────────────────────────────

/// **THE A/B.** Same fires, same rows, two answers for one withdrawn consumer:
/// the logits must be identical to the bit, and the recorded graph must hold
/// exactly `2 x nodes x (r - 1)` fewer launches.
#[test]
fn a_grouped_correction_is_bit_identical_to_a_split_one_and_costs_fewer_launches() {
    let _serial = serialized();
    assert!(
        !engine_cuda::GROUPED.is_empty(),
        "this shell names no groupable op, and then there is no grouped arm",
    );

    let Some(split) = arm("the split arm", false) else {
        return;
    };
    let Some(grouped) = arm("the grouped arm", true) else {
        return;
    };

    // ── the premise, not vacuous ─────────────────────────────────────────
    assert!(split.corrections > 0, "the SKU states corrections");
    assert_eq!(split.corrections, grouped.corrections);
    assert!(
        split.runs > 1,
        "this composition leaves the adapter window whole, so both arms are one \
         launch and the comparison is empty",
    );
    assert_eq!(split.runs, grouped.runs, "one row order, two answers");
    assert!(
        split.split_rows > 0 && split.grouped_rows == 0,
        "the split arm's artifact does not answer `Split`/`Copy` for the \
         correction: {} split rows, {} grouped rows",
        split.split_rows,
        split.grouped_rows,
    );
    assert!(
        grouped.grouped_rows > 0 && grouped.split_rows == 0,
        "the grouped arm's artifact does not answer `Grouped` for the \
         correction: {} split rows, {} grouped rows",
        grouped.split_rows,
        grouped.grouped_rows,
    );

    // The correction actually did something: an adapted lane and its base
    // neighbour, on the same prompt in the same fire, say different things.
    // Without this a grouped path that corrected no row at all would pass
    // everything below.
    let adapted_moved = split.logits.chunks_exact(8).any(|fire| {
        // lanes 4 and 5 are the fresh prefills, base and adapted, same prompt.
        fire[4].iter().zip(&fire[5]).any(|(a, b)| a != b)
    });
    assert!(
        adapted_moved,
        "the adapted prefill lane says exactly what the base one says on the \
         same prompt, so no correction reached the readout and this gate is \
         testing plumbing that computes zero",
    );

    // ── (a) bit for bit ──────────────────────────────────────────────────
    assert_eq!(split.logits.len(), grouped.logits.len());
    for (at, (left, right)) in split.logits.iter().zip(&grouped.logits).enumerate() {
        assert_eq!(left.len(), right.len(), "readout {at}");
        let differs = left
            .iter()
            .zip(right)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            differs.is_none(),
            "fire {}, lane {} differs at logit {}: split {} vs grouped {}",
            at / 8,
            at % 8,
            differs.unwrap_or(0),
            left[differs.unwrap_or(0)],
            right[differs.unwrap_or(0)],
        );
    }
    assert_eq!(split.said, grouped.said, "the greedy tokens moved");

    // ── (b) fewer launches, by exactly the right number ──────────────────
    //
    // A correction site is TWO launches — the routed projection and the
    // accumulate (`kernels_cuda::linear::lora`) — so a split pays `2r` per
    // node and a grouped answer pays 2. Everything else in the graph is the
    // same artifact over the same composition, so the difference is exactly
    // the launches the grouped answer did not make.
    let saved = 2 * split.corrections * (split.runs - 1);
    assert_eq!(
        split.nodes.checked_sub(grouped.nodes),
        Some(saved),
        "the grouped graph holds {} nodes against the split graph's {}; the \
         answer is worth {saved} launches and this is not it",
        grouped.nodes,
        split.nodes,
    );

    eprintln!(
        "GROUPED vs SPLIT on {SKU}: r={}, {} correction nodes, {} graph nodes \
         against {} ({saved} launches saved), every logit bit-identical over \
         {} readouts",
        split.runs,
        split.corrections,
        grouped.nodes,
        split.nodes,
        split.logits.len(),
    );
}
