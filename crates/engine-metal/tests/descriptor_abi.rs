//! `DescriptorAbi`, derived from a real artifact on a real device
//! (`.wiki/palo/icb.md` §3, design §2's P8).
//!
//! **WHAT IS BEING CLAIMED.** Build log 10 keyed one CUDA exec per
//! `(composition, size)` because the shell could not see which argument of a
//! launch was an extent — `kernels-cuda` builds its argument bytes inside
//! `ctx.fire`. `kernels_metal::Encode::fire` hands the shell every argument
//! as a value and every grid axis as a number, so the table is derivable by
//! walking the same template against several synthetic descriptors and
//! reading the differences. This file does that, against the checkpoint
//! `serve_smoke` serves, and prints the census: how many slots, how many
//! components move, how many are constants encoded once, and — the part that
//! is a deliverable whether or not anything is built on it — which
//! components are not affine in the composition at all.
//!
//! **NOTHING IS DISPATCHED.** `Shell::record` runs the whole fire path and
//! substitutes a `Tape` for the encode sink: no command buffer is opened, no
//! kernel runs, and the sequence lengths this shell counts are left where
//! they were. So a probe is free and a probe of a composition the machine
//! could not afford to actually run is still legal.
//!
//! ```text
//! cargo test -p engine-metal --release --test descriptor_abi -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use engine_metal::abi::{self, At, Axis};
use engine_metal::{Boot, Lane, Recording, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_exec::fire::{Lane as FireLane, compose};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The lane word the model's own `Classify` computes for a query of this
/// length — runtime-side work, done here because this test is the runtime.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

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

fn ready(what: &str) -> Option<Shell> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!("skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache");
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c, as `serve_smoke` states it: an unstamped snapshot proceeds,
        // and the deployment's facts are stated honestly all the same.
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: Budget::new(16, 640),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 16,
        // F1's depth: one step in flight, one A/B seat set. Stated rather
        // than defaulted because these are goldens — the eager shell is what
        // a byte-identity arm compares against — and because a second seat
        // set is a second whole `Inputs` reservation on a machine this test
        // is already sized carefully for.
        runahead: engine::runahead::Runahead::F1,
        // Full residency: the whole weight table on the device, no
        // wired-slab tier, no segment cuts — the load every gate in
        // this directory measures.
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    Some(shell)
}

/// One synthetic lane: a slot, a word, and a run of token ids that mean
/// nothing.
///
/// **THE IDS DO NOT MATTER AND THAT IS THE POINT.** A recording is of the
/// walk's SHAPE — which shader at which grid over which bytes — and no
/// argument of any dispatch is a token's value. What decides the shape is the
/// word (which class) and the length (how many rows).
struct Synthetic {
    tokens: Vec<u32>,
    slot: u32,
    word: u64,
}

impl Synthetic {
    fn lane(&self) -> Lane<'_> {
        Lane {
            slot: self.slot,
            word: self.word,
            tokens: &self.tokens,
        }
    }
}

/// A batch of `lanes` lanes of `len` tokens each, at consecutive slots.
fn batch(spec: &[(u32, u32)]) -> Vec<Synthetic> {
    let mut out = Vec::new();
    let mut slot = 0;
    for (lanes, len) in spec {
        for _ in 0..*lanes {
            out.push(Synthetic {
                tokens: vec![1u32; *len as usize],
                slot,
                word: word(*len),
            });
            slot += 1;
        }
    }
    out
}

/// The composition a batch produces, as the class table's `(rows, lanes)`.
fn classes_of(shell: &Shell, batch: &[Synthetic]) -> Vec<(u32, u32)> {
    let submitted: Vec<FireLane> = batch
        .iter()
        .map(|s| FireLane::new(s.word, s.tokens.len() as u32))
        .collect();
    let composed =
        compose(shell.compiled_model(), shell.budget(), &submitted).expect("the batch composes");
    composed
        .classes()
        .as_slice()
        .iter()
        .map(|class| (class.rows, class.lanes))
        .collect()
}

/// The five batches the probe basis is walked with, named once because two
/// tests below walk the same ones and a copy of a batch is a copy that can
/// drift from the basis it defines.
fn base_batch() -> Vec<Synthetic> {
    batch(&[(2, 1), (2, 8)])
}

fn one_more_decode_lane() -> Vec<Synthetic> {
    batch(&[(3, 1), (2, 8)])
}

fn one_more_prefill_lane() -> Vec<Synthetic> {
    batch(&[(2, 1), (3, 8)])
}

/// **ONE TOKEN, NOT EIGHT, AND THAT IS AN INTEGRALITY CONSTRAINT AND NOT A
/// TASTE.** `fit::invert` reads a fire's class table back into these
/// coordinates by inverting the step matrix OVER THE INTEGERS. A prefill lane
/// moves that class by `(8, 1)` and this direction moves its rows alone; at
/// eight rows a step the pair generates only `rows ≡ 0 (mod 8)`, so a real
/// composition like a single 12-token lane has no integer coordinates and the
/// basis is refused by name. At one row a step the two directions saturate
/// the class's `(rows, lanes)` lattice and the inverse is exact.
fn one_more_prefill_token() -> Vec<Synthetic> {
    let mut out = batch(&[(2, 1), (1, 8)]);
    out.push(Synthetic {
        tokens: vec![1u32; 9],
        slot: 3,
        word: word(9),
    });
    out
}

/// The held-out composition: every direction stepped, and none of them by
/// one.
fn check_batch() -> Vec<Synthetic> {
    let mut out = batch(&[(5, 1), (3, 8)]);
    out.push(Synthetic {
        tokens: vec![1u32; 24],
        slot: 8,
        word: word(24),
    });
    out
}

/// Where [`check_batch`] stands: five decode lanes (+3), four prefill lanes
/// (+2), and `48 − 8·4 = 16` prefill rows past what those lanes carry at
/// eight each.
const CHECK_AT: [i128; 3] = [3, 2, 16];

/// What one rung did to the class table, componentwise.
///
/// **AN AXIS IS A DIFFERENCE OF COMPOSITIONS, SO IT IS TAKEN AS ONE.** This
/// used to be three `(rows, lanes)` pairs written out beside the batches, and
/// the transcription went stale the moment the artifact baked more than one
/// class: a step stated over one class against a twelve-class table is
/// refused by `fit::invert` before any law is fitted. The composer is the
/// only thing that knows how wide the table is, so the basis is read off it.
fn step_between(shell: &Shell, from: &[Synthetic], to: &[Synthetic]) -> Vec<(i32, i32)> {
    let from = classes_of(shell, from);
    let to = classes_of(shell, to);
    assert_eq!(
        from.len(),
        to.len(),
        "two compositions of one artifact carry one class table"
    );
    let step: Vec<(i32, i32)> = to
        .iter()
        .zip(&from)
        .map(|(&(there_rows, there_lanes), &(here_rows, here_lanes))| {
            (
                there_rows as i32 - here_rows as i32,
                there_lanes as i32 - here_lanes as i32,
            )
        })
        .collect();
    assert_eq!(
        step.iter().filter(|&&pair| pair != (0, 0)).count(),
        1,
        "a probe direction moves ONE class, or the slope it yields is two \
         derivatives wearing one name: {step:?}"
    );
    step
}

/// The three directions the composition can genuinely be stepped along, and
/// no fourth: a decode lane's word says one token, so that class's rows and
/// lanes move together and pretending they were two axes would fit a slope to
/// a direction no batch can walk.
fn basis(shell: &Shell) -> Vec<Axis> {
    let base = base_batch();
    vec![
        Axis::new(
            "a decode lane",
            step_between(shell, &base, &one_more_decode_lane()),
        ),
        Axis::new(
            "a prefill lane of 8 tokens",
            step_between(shell, &base, &one_more_prefill_lane()),
        ),
        Axis::new(
            "one more token in a prefill lane",
            step_between(shell, &base, &one_more_prefill_token()),
        ),
    ]
}

/// A slot's DISPATCH, which is the whole of what the table re-derives.
///
/// **`window_lanes` IS AN ANNOTATION AND NOT A COMPONENT.** `record::Slot`
/// says so in as many words — "nothing an ICB slot holds is this number" —
/// and the table is built to be diffed against `icb::rebind`, which carries
/// exactly ONE extra law per slot and it is `rows`: an arm is picked off the
/// window's rows (`abi::Pick::Rows`) and a tiling ceiling divides them, so
/// the lane count reaches neither `SlotAbi` nor the packed device tables.
/// `DescriptorAbi::slot_at` therefore leaves it at the skeleton's reading,
/// and asking it about a number it never fitted is a question about the
/// recorder rather than about the derivation. Everything the derivation does
/// claim — the shader point, the grid, the threadgroup, every argument, the
/// region, the run and the window's ROWS — is compared.
fn dispatch(slot: &engine_metal::Slot) -> engine_metal::Slot {
    let mut out = slot.clone();
    out.window_lanes = 0;
    out
}

/// Record one batch, at the coordinates the caller places it at.
fn record(shell: &mut Shell, batch: &[Synthetic], coords: Vec<i128>) -> Recording {
    for lane in batch {
        shell.open(lane.slot).expect("the slot opens");
    }
    let lanes: Vec<Lane<'_>> = batch.iter().map(Synthetic::lane).collect();
    shell
        .record(&lanes)
        .expect("the walk records")
        .at(coords)
}

#[test]
fn the_extent_table_is_derived_from_two_walks_and_verified_on_a_third() {
    let Some(mut shell) = ready("the descriptor abi derivation") else {
        return;
    };

    // The classes this artifact bakes, and which of them a word reaches.
    let baked_classes = shell.compiled_model().classes.classes.len();
    let decode = classes_of(&shell, &batch(&[(1, 1)]));
    let prefill = classes_of(&shell, &batch(&[(1, 8)]));
    eprintln!("the artifact bakes {baked_classes} classes");
    eprintln!("  one decode lane   lands as {decode:?}");
    eprintln!("  one prefill lane  lands as {prefill:?}");

    // THE PROBE BASIS, read off the composer rather than transcribed beside
    // it — see [`basis`] and [`step_between`] for why the transcription is
    // the thing that went stale.
    //
    //   d : one more decode lane          (decode rows +1, decode lanes +1)
    //   p : one more prefill lane of 8    (prefill rows +8, prefill lanes +1)
    //   t : one more prefill token        (prefill rows +1, prefill lanes +0)
    let axes = basis(&shell);
    eprintln!(
        "  the probe basis, as the composer states it: {}",
        axes.iter()
            .map(|axis| format!("{axis} {:?}", axis.step))
            .collect::<Vec<_>>()
            .join(" | ")
    );

    // base: 2 decode lanes, 2 prefill lanes of 8.
    let base = record(&mut shell, &base_batch(), vec![0, 0, 0]);
    let bumps = vec![
        // +1 decode lane
        record(&mut shell, &one_more_decode_lane(), vec![1, 0, 0]),
        // +1 prefill lane of 8
        record(&mut shell, &one_more_prefill_lane(), vec![0, 1, 0]),
        // one prefill lane grows by one token
        record(&mut shell, &one_more_prefill_token(), vec![0, 0, 1]),
    ];
    // check: every direction stepped, and none of them by one.
    let check = record(&mut shell, &check_batch(), CHECK_AT.to_vec());

    eprintln!(
        "probes: base {} slots at {:?}; bumps {:?}; check {} slots at {:?}",
        base.slots.len(),
        base.classes,
        bumps.iter().map(|b| b.slots.len()).collect::<Vec<_>>(),
        check.slots.len(),
        check.classes,
    );

    // ONE PROBE, AND ONE RUNG PER DIRECTION. The bumps above are exactly
    // `base + 1·e_k`, which is the pair an affine slope is read off; a longer
    // ladder is what a tiling law's divisor would be solved against, and this
    // file's claim is the affine one.
    let probes = abi::Probes {
        probes: vec![abi::Probe {
            base,
            ladders: bumps.into_iter().map(|bump| vec![bump]).collect(),
        }],
        check,
    };
    let surveyed = abi::survey(&axes, &probes).expect("the probes walk one template");
    let table = &surveyed.abi;
    let check = &probes.check;

    // ---- the census -------------------------------------------------
    eprintln!();
    eprintln!("== the derived DescriptorAbi for {SKU} ==");
    eprintln!("  slots (one Encode::fire = one dispatch = one ICB slot): {}", table.len());
    eprintln!("  components that move with the composition:              {}", table.affine());
    eprintln!("  components that are constant (encoded once):            {}", table.constants());
    eprintln!("  slots that rewrite nothing at all:                      {}", table.frozen());
    eprintln!("  components refused as Unaffine:                         {}", surveyed.unaffine.len());
    eprintln!("  slots whose SHADER POINT moves with the composition:    {}", surveyed.armed.len());
    eprintln!("  the probe basis: {}", axes.iter().map(ToString::to_string).collect::<Vec<_>>().join(" | "));

    // Which KIND of component moves, and along which direction.
    let mut by_place: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut by_axis: Vec<usize> = vec![0; axes.len()];
    // ONE LAW TABLE PER ARM, so a slot that picks its shader off the window
    // is counted once per point it is — which is what the per-point census
    // below prints too.
    for slot in &table.slots {
        for arm in &slot.arms {
            for (at, law) in &arm.laws {
                let key = match at {
                    At::Grid(_) => "grid axis",
                    At::Block(_) => "threadgroup axis",
                    At::Arg { .. } => "argument",
                    // The recorder enumerates exactly the grid axes, the
                    // threadgroup axes and the arguments, so a Metal law
                    // table holds no entry, shared-memory or shape component
                    // and there is nothing here to count.
                    At::Entry | At::Shared | At::Shape => "not on this plane",
                };
                *by_place.entry(key).or_default() += 1;
                for axis in law.reads() {
                    by_axis[axis] += 1;
                }
            }
        }
    }
    eprintln!("  what moves: {by_place:?}");
    for (k, axis) in axes.iter().enumerate() {
        eprintln!("    {} components read `{axis}`", by_axis[k]);
    }

    // Which arguments move: a buffer OFFSET is the windowed cut, a scalar is
    // an extent the shader reads. The split is the thing `.wiki/palo/icb.md`
    // §3 predicts and it is worth printing rather than asserting.
    let mut offsets = 0usize;
    let mut scalars = 0usize;
    for slot in &table.slots {
        for arm in &slot.arms {
            for (at, _) in &arm.laws {
                if let At::Arg { at: index, .. } = at {
                    match arm.skeleton.args[*index as usize] {
                        engine_metal::Arg::Buffer { .. } => offsets += 1,
                        _ => scalars += 1,
                    }
                }
            }
        }
    }
    eprintln!("  moving arguments: {offsets} buffer offsets, {scalars} scalars");

    eprintln!();
    eprintln!("  per shader point (slots, moving components):");
    for (point, slots, laws) in table.census() {
        eprintln!("    {slots:>5} slots {laws:>6} laws   {point}");
    }

    if !surveyed.armed.is_empty() {
        eprintln!();
        eprintln!("  ARM-SWITCHING SLOTS — the ICB must rebind these pipelines:");
        let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
        for armed in &surveyed.armed {
            let shape = armed
                .points
                .iter()
                .map(|(point, _)| point.to_string())
                .collect::<Vec<_>>()
                .join(" <-> ");
            *shapes.entry(shape).or_default() += 1;
        }
        for (shape, count) in &shapes {
            eprintln!("    {count:>5} slots: {shape}");
        }
        eprintln!("    first three: {:?}", surveyed.armed.iter().take(3).map(|a| a.slot).collect::<Vec<_>>());
    }

    if !surveyed.unaffine.is_empty() {
        eprintln!();
        eprintln!("  REFUSED as Unaffine, grouped by the shader point they stand in:");
        let mut by_point: BTreeMap<String, usize> = BTreeMap::new();
        for fault in &surveyed.unaffine {
            if let engine_metal::Fault::Unaffine { point, at, .. } = fault {
                *by_point.entry(format!("{point}  {at}")).or_default() += 1;
            }
        }
        for (what, count) in &by_point {
            eprintln!("    {count:>5} x  {what}");
        }
        eprintln!();
        eprintln!("  and the first three in full:");
        for fault in surveyed.unaffine.iter().take(3) {
            eprintln!("    {fault}");
        }
    }

    // ---- the gate ---------------------------------------------------
    // Every probe walked the same template (or `survey` would have refused),
    // and the table re-derives the CHECK walk exactly — slot for slot,
    // argument for argument. That is a stronger statement than the
    // per-component verification inside the fit: it says the whole recording
    // at a composition none of the probes visited is a function of the
    // composition's numbers alone.
    assert!(!table.is_empty(), "the artifact dispatches nothing");

    let mut switching: std::collections::BTreeSet<u32> =
        surveyed.armed.iter().map(|entry| entry.slot).collect();
    for fault in &surveyed.unaffine {
        if let engine_metal::Fault::Unaffine { slot, .. } = fault {
            switching.insert(*slot);
        }
    }
    for index in 0..table.len() {
        if switching.contains(&(index as u32)) {
            continue;
        }
        let rebuilt = table
            .slot_at(index, &check.coords)
            .expect("the table holds this slot");
        assert_eq!(
            dispatch(&rebuilt),
            dispatch(&check.slots[index]),
            "slot {index} ({}) re-derived from the table is not the slot the walk produced",
            table.slots[index].point()
        );
    }
    eprintln!(
        "  {} of {} slots re-derive the check walk exactly (the other {} either switch \
         shader or hold a component no affine law fits)",
        table.len() - switching.len(),
        table.len(),
        switching.len()
    );
}

#[test]
fn a_composition_the_probes_never_visited_is_re_derived_exactly() {
    let Some(mut shell) = ready("the abi's out-of-sample check") else {
        return;
    };
    let axes = basis(&shell);
    let base = record(&mut shell, &base_batch(), vec![0, 0, 0]);
    let bumps = vec![
        record(&mut shell, &one_more_decode_lane(), vec![1, 0, 0]),
        record(&mut shell, &one_more_prefill_lane(), vec![0, 1, 0]),
        record(&mut shell, &one_more_prefill_token(), vec![0, 0, 1]),
    ];
    let check = record(&mut shell, &check_batch(), CHECK_AT.to_vec());
    let probes = abi::Probes {
        probes: vec![abi::Probe {
            base,
            ladders: bumps.into_iter().map(|bump| vec![bump]).collect(),
        }],
        check,
    };
    let surveyed = abi::survey(&axes, &probes).expect("the probes walk one template");
    let table = &surveyed.abi;
    let switching: std::collections::BTreeSet<u32> =
        surveyed.armed.iter().map(|entry| entry.slot).collect();

    // A FOURTH composition, at coordinates no probe stood at: 4 decode lanes
    // and 4 prefill lanes of 8, which is (4,4) rather than the base's (2,2).
    let out_of_sample = record(&mut shell, &batch(&[(4, 1), (4, 8)]), vec![2, 2, 0]);
    assert_eq!(
        out_of_sample.slots.len(),
        table.len(),
        "a composition inside the artifact walks the artifact's slots"
    );
    let mut derived = 0usize;
    for (index, produced) in out_of_sample.slots.iter().enumerate() {
        if switching.contains(&(index as u32)) || produced.point != table.slots[index].point() {
            continue;
        }
        let rebuilt = table
            .slot_at(index, &out_of_sample.coords)
            .expect("the table holds this slot");
        assert_eq!(
            dispatch(&rebuilt),
            dispatch(produced),
            "slot {index} ({}) at an unvisited composition",
            table.slots[index].point()
        );
        derived += 1;
    }
    eprintln!(
        "the table re-derives {derived} of {} slots of a composition it never saw \
         ({} switch arms)",
        table.len(),
        table.len() - derived
    );
    assert!(derived > 0, "nothing at all re-derived");
}

#[test]
fn every_capture_phase_slot_is_a_dispatch_and_the_grid_is_a_number() {
    // KILL FACTOR 4, checked rather than assumed. An indirect command buffer
    // of compute commands holds DISPATCHES; a blit in the capture phase would
    // not be expressible. On this plane every device action a fire takes goes
    // through `Encode::fire` — there is no blit encoder in the shell and
    // `Buffer::write` is a memcpy into a `StorageModeShared` mapping, which
    // happens before the command buffer opens — so the recording IS the
    // complete list of what the ICB would have to carry, and this test says
    // so by counting.
    let Some(mut shell) = ready("the dispatch-only census") else {
        return;
    };
    let taped = record(&mut shell, &batch(&[(2, 1), (2, 8)]), vec![0, 0, 0]);
    let mut widest = 0usize;
    for slot in &taped.slots {
        assert!(
            slot.lanes.iter().all(|axis| *axis > 0),
            "slot at {} has a zero grid axis: {:?}",
            slot.point,
            slot.lanes
        );
        widest = widest.max(slot.args.len());
    }
    eprintln!(
        "{} dispatches, and nothing else; the widest argument list is {widest} \
         (maxKernelBufferBindCount must cover it)",
        taped.slots.len()
    );
    assert!(
        widest <= 31,
        "an ICB descriptor's maxKernelBufferBindCount would have to hold {widest}"
    );
}

#[test]
fn which_slots_pick_their_shader_off_the_window_and_where_the_thresholds_are() {
    // KILL FACTOR 5, in its real form. The note asks how often the affine
    // assumption breaks across the catalog and expects the answer to be about
    // ARGUMENTS. On this plane it is not: every argument fits, and what moves
    // is which SHADER a slot is. `kernels-metal`'s dense matmul picks a
    // gemv arm for a thin rectangle and a tiled gemm arm for a fat one, so a
    // slot is one ENTRY and not one PIPELINE.
    //
    // This sweep is the census: which slots change point, over what, and at
    // which row count. Recording costs no dispatch, so a sweep is free.
    let Some(mut shell) = ready("the arm-switch sweep") else {
        return;
    };

    // Decode rows and prefill rows, swept apart.
    let mut sweep: Vec<(String, Vec<Synthetic>)> = Vec::new();
    for lanes in [1u32, 2, 4, 8, 12, 16] {
        sweep.push((format!("{lanes} decode lanes"), batch(&[(lanes, 1)])));
    }
    for len in [2u32, 4, 8, 16, 32, 64, 128] {
        sweep.push((format!("one prefill lane of {len}"), batch(&[(1, len)])));
    }

    let mut points: Vec<(String, Vec<engine_metal::Point>)> = Vec::new();
    for (name, batch) in &sweep {
        let taped = record(&mut shell, batch, Vec::new());
        points.push((
            name.clone(),
            taped.slots.iter().map(|slot| slot.point).collect(),
        ));
    }

    eprintln!();
    eprintln!("== which slots pick their shader off the window ==");
    for (name, seen) in &points {
        eprintln!("  {name:<28} {} slots", seen.len());
    }

    // Group the sweep by slot COUNT: a composition with an empty class walks
    // fewer slots, and only equal-length recordings are comparable slot for
    // slot.
    let mut by_width: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (at, (_, seen)) in points.iter().enumerate() {
        by_width.entry(seen.len()).or_default().push(at);
    }
    for (width, members) in &by_width {
        if members.len() < 2 {
            continue;
        }
        let mut switches: BTreeMap<usize, Vec<(String, engine_metal::Point)>> = BTreeMap::new();
        for slot in 0..*width {
            let first = points[members[0]].1[slot];
            for member in &members[1..] {
                if points[*member].1[slot] != first {
                    let row = switches.entry(slot).or_default();
                    if row.is_empty() {
                        row.push((points[members[0]].0.clone(), first));
                    }
                    row.push((points[*member].0.clone(), points[*member].1[slot]));
                }
            }
        }
        eprintln!();
        eprintln!(
            "  among the {} compositions that walk {width} slots: {} slots switch shader",
            members.len(),
            switches.len()
        );
        for (slot, seen) in switches.iter().take(24) {
            eprintln!("    slot {slot}:");
            let mut said: Vec<String> = Vec::new();
            for (name, point) in seen {
                let line = format!("      {point} at {name}");
                if !said.contains(&line) {
                    said.push(line);
                }
            }
            for line in said {
                eprintln!("{line}");
            }
        }
        if switches.len() > 24 {
            eprintln!("    ... and {} more", switches.len() - 24);
        }
    }
}
