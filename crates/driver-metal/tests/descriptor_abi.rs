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
//! cargo test -p driver-metal --release --test descriptor_abi -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use driver::fire::{Lane as FireLane, compose};
use driver_metal::abi::{self, At, Axis};
use driver_metal::{Boot, Lane, Recording, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The lane word the model's own `Classify` computes for a query of this
/// length — engine-side work, done here because this test is the engine.
fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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
    if !driver_metal::device::present() {
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
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    let shell = Shell::load(Boot {
        plan,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(16, 640),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 16,
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

    // THE PROBE BASIS. Three directions the composition can genuinely be
    // stepped along, and no fourth: a decode lane's word says one token, so
    // that class's rows and lanes move together and pretending they were two
    // axes would fit a slope to a direction no batch can walk.
    //
    //   d : one more decode lane          (decode rows +1, decode lanes +1)
    //   p : one more prefill lane of 8    (prefill rows +8, prefill lanes +1)
    //   t : eight more prefill tokens     (prefill rows +8, prefill lanes +0)
    let axes = vec![
        Axis::new("a decode lane", vec![(1, 1)]),
        Axis::new("a prefill lane of 8 tokens", vec![(8, 1)]),
        Axis::new("8 more tokens in a prefill lane", vec![(8, 0)]),
    ];

    // base: 2 decode lanes, 2 prefill lanes of 8.
    let base = record(&mut shell, &batch(&[(2, 1), (2, 8)]), vec![0, 0, 0]);
    let bumps = vec![
        // +1 decode lane
        record(&mut shell, &batch(&[(3, 1), (2, 8)]), vec![1, 0, 0]),
        // +1 prefill lane of 8
        record(&mut shell, &batch(&[(2, 1), (3, 8)]), vec![0, 1, 0]),
        // one prefill lane grows by 8 tokens
        record(
            &mut shell,
            &{
                let mut b = batch(&[(2, 1), (1, 8)]);
                b.push(Synthetic {
                    tokens: vec![1u32; 16],
                    slot: 3,
                    word: word(16),
                });
                b
            },
            vec![0, 0, 1],
        ),
    ];
    // check: every direction stepped, and none of them by one.
    let check = record(
        &mut shell,
        &{
            let mut b = batch(&[(5, 1), (3, 8)]);
            b.push(Synthetic {
                tokens: vec![1u32; 24],
                slot: 8,
                word: word(24),
            });
            b
        },
        vec![3, 2, 2],
    );

    eprintln!(
        "probes: base {} slots at {:?}; bumps {:?}; check {} slots at {:?}",
        base.slots.len(),
        base.classes,
        bumps.iter().map(|b| b.slots.len()).collect::<Vec<_>>(),
        check.slots.len(),
        check.classes,
    );

    let surveyed = abi::survey(&axes, &base, &bumps, &check).expect("the probes walk one template");
    let table = &surveyed.abi;

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
    for slot in &table.slots {
        for (at, law) in &slot.laws {
            let key = match at {
                At::Lane(_) => "grid axis",
                At::Group(_) => "threadgroup axis",
                At::Arg(_) => "argument",
            };
            *by_place.entry(key).or_default() += 1;
            for axis in law.reads() {
                by_axis[axis] += 1;
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
        for (at, _) in &slot.laws {
            if let At::Arg(index) = at {
                match slot.skeleton.args[*index as usize] {
                    driver_metal::Arg::Buffer { .. } => offsets += 1,
                    _ => scalars += 1,
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
            if let driver_metal::Fault::Unaffine { point, at, .. } = fault {
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
        if let driver_metal::Fault::Unaffine { slot, .. } = fault {
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
            rebuilt, check.slots[index],
            "slot {index} ({}) re-derived from the table is not the slot the walk produced",
            table.slots[index].point
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
    let axes = vec![
        Axis::new("a decode lane", vec![(1, 1)]),
        Axis::new("a prefill lane of 8 tokens", vec![(8, 1)]),
        Axis::new("8 more tokens in a prefill lane", vec![(8, 0)]),
    ];
    let base = record(&mut shell, &batch(&[(2, 1), (2, 8)]), vec![0, 0, 0]);
    let bumps = vec![
        record(&mut shell, &batch(&[(3, 1), (2, 8)]), vec![1, 0, 0]),
        record(&mut shell, &batch(&[(2, 1), (3, 8)]), vec![0, 1, 0]),
        record(
            &mut shell,
            &{
                let mut b = batch(&[(2, 1), (1, 8)]);
                b.push(Synthetic {
                    tokens: vec![1u32; 16],
                    slot: 3,
                    word: word(16),
                });
                b
            },
            vec![0, 0, 1],
        ),
    ];
    let check = record(
        &mut shell,
        &{
            let mut b = batch(&[(5, 1), (3, 8)]);
            b.push(Synthetic {
                tokens: vec![1u32; 24],
                slot: 8,
                word: word(24),
            });
            b
        },
        vec![3, 2, 2],
    );
    let surveyed = abi::survey(&axes, &base, &bumps, &check).expect("the probes walk one template");
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
        if switching.contains(&(index as u32)) || produced.point != table.slots[index].point {
            continue;
        }
        let rebuilt = table
            .slot_at(index, &out_of_sample.coords)
            .expect("the table holds this slot");
        assert_eq!(
            &rebuilt, produced,
            "slot {index} ({}) at an unvisited composition",
            table.slots[index].point
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

    let mut points: Vec<(String, Vec<driver_metal::Point>)> = Vec::new();
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
        let mut switches: BTreeMap<usize, Vec<(String, driver_metal::Point)>> = BTreeMap::new();
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
