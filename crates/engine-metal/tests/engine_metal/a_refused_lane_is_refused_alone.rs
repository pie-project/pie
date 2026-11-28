#![cfg(target_vendor = "apple")]

//! A lane the engine refuses for its own sake (here one routed to an
//! adapter on a load that corrects nothing: `Fault::Adapterless`) refuses
//! that lane alone, through the `Engine` contract the runtime drives
//! (`engine::Error::Lanes`, `FrameTicket::refused`): at a frame's first step
//! the frame is refused naming it and nothing ran; at a later step the
//! frame stops there, its ticket naming it, and the steps before it land
//! whole. The plain lanes beside it fire on, and read their own rows.
//! engine-gpu's suite holds the same test (`runtime/lane_refusal.rs`); this
//! is Metal's mapping of its lane faults to that contract.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine_metal::serve::Seated;
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Trace, Weight, ops};

const NAME: &str = "lane-refusal-mini";
const WIDTH: u32 = 16;
const VOCAB: u32 = 16;
const LANES: u32 = 4;

/// The table's rows, read out: a lane's one row is its token's row.
struct Rows;

impl ForwardHybrid for Rows {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let x = ops::gather::gather_rows(&x, &inputs.readout_rows());
        poem::output("logits", &x, None);
    }
}

fn bf(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

/// The embedding table, bf16-exact.
fn table() -> Vec<f32> {
    let mut state = 0x5eed_u64;
    (0..VOCAB * WIDTH)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bf(((state >> 40) as f32 / 8_388_608.0) - 1.0)
        })
        .collect()
}

fn write(path: &Path, table: &[f32]) {
    let raw: Vec<u8> = table
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let objects = [Object::leaf(
        "embed",
        vec![u64::from(VOCAB), u64::from(WIDTH)],
        ztensor::Leaf::BF16,
        &raw,
    )];
    emit::write(
        path,
        &Stamp::of(Platform::Metal.backend(), NAME),
        &BTreeMap::new(),
        4096,
        &objects,
        |o, p, _| panic!("{o}/{p} is not streamed here"),
    )
    .expect("the artifact writes");
}

fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .map_err(|why| why.to_string())
}

/// A step of kv-less lanes, one a token, `routed` the ones naming an
/// adapter.
fn step_of(trace: &Trace, tokens: &[(u32, bool)]) -> Step {
    Step {
        lanes: tokens
            .iter()
            .enumerate()
            .map(|(slot, &(token, routed))| Lane {
                slot: slot as u32,
                word: trace.facts.word(&Request::new(1, false)),
                tokens: vec![token],
                // One token: its last row is its only row. (Metal serves a
                // row-selected readout only where a guest's attachment reads
                // it.)
                readout: Readout::Last,
                kv_less: true,
                adapter: routed.then_some(0),
                ..Lane::default()
            })
            .collect(),
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
        shape: None,
    }
}

/// The lane's one row against the table's row of `token`.
fn reads(table: &[f32], token: u32, values: &[f32]) {
    let want = &table[(token * WIDTH) as usize..][..WIDTH as usize];
    assert_eq!(values.len(), WIDTH as usize);
    for (c, (&have, &want)) in values.iter().zip(want).enumerate() {
        assert!(
            (have - want).abs() <= 1e-2 * want.abs().max(1.0),
            "token {token} column {c}: device {have} against host {want}"
        );
    }
}

#[test]
fn a_refused_lane_is_refused_alone() {
    let _device = crate::device_alone();
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("pie-metal-{NAME}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("metal-{NAME}.zt"));
    let table = table();
    write(&path, &table);

    let trace = poem::trace_hybrid(NAME, &Rows, Platform::Metal);
    let tokens = 16;
    // Every rung: the frames below are built by hand, three lanes wide,
    // past the one-token top the ruling picks for this miniature (the
    // runtime's step builder never forms a fire wider than the stated top).
    let boot = engine_metal::DeviceBoot {
        top: Some(engine_metal::Top::EveryRung),
        ..engine_metal::DeviceBoot::default()
    };
    let mut engine = engine_metal::Metal::new(boot, contract_for);
    engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: LANES,
                max_tokens: tokens,
                buckets: vec![tokens],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: LANES,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: engine::Allowance::WHOLE,
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    // The activation at its widest, as the other shell tests take it: a
    // fire of several lanes stages past the smallest rung's one row.
    engine
        .shell_mut()
        .expect("loaded")
        .raise_whole()
        .expect("the widest rung");

    // At the first step: the frame is refused, naming lane 1 alone.
    let refused = engine
        .submit(&FrameSubmission::of(step_of(
            &trace,
            &[(3, false), (7, true), (11, false)],
        )))
        .expect_err("a lane routed to an adapter this load lacks is refused");
    let engine::Error::Lanes(lanes) = &refused else {
        panic!("the refusal names its lane, not the frame: {refused}");
    };
    assert_eq!(
        lanes.iter().map(|fault| fault.lane).collect::<Vec<_>>(),
        vec![1],
        "lane 1 alone: {refused}"
    );

    // At the second step: the first lands whole, and the ticket names the
    // second's lane 1.
    let frame = FrameSubmission {
        steps: vec![
            step_of(&trace, &[(3, false), (11, false)]),
            step_of(&trace, &[(5, false), (7, true)]),
        ],
    };
    let mut ticket = engine
        .submit(&frame)
        .expect("a lane refused past the first step stops the frame there");
    assert_eq!(
        ticket.steps.len(),
        1,
        "the steps before the refusing one fired"
    );
    assert_eq!(
        ticket
            .refused
            .iter()
            .map(|fault| fault.lane)
            .collect::<Vec<_>>(),
        vec![1]
    );
    engine.settle_frame(&mut ticket).expect("the frame settles");
    assert_eq!(
        ticket.steps[0]
            .readouts
            .iter()
            .map(|r| r.rows)
            .collect::<Vec<_>>(),
        vec![1, 1],
        "the step before the refusing one read out both its lanes"
    );

    // The plain lanes fire on, the refused one gone.
    let mut ticket = engine
        .submit(&FrameSubmission::of(step_of(
            &trace,
            &[(5, false), (11, false)],
        )))
        .expect("the rest fires again");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    assert!(ticket.refused.is_empty());

    // And they read their own rows. Metal's `Engine` path reads a lane's
    // rows back to the host only for a guest's attachment (or under the
    // process-wide `host-rows` diagnostic), so the rows are read through
    // the same loaded shell, which asks for them.
    let shell = engine.shell_mut().expect("loaded");
    let tokens = [vec![5u32], vec![11u32], vec![3u32]];
    let lanes: Vec<Seated<'_>> = tokens
        .iter()
        .enumerate()
        .map(|(slot, tokens)| Seated {
            replay_stated: None,
            kv_less: true,
            // A kv-less lane names no slot; on a load placed for the
            // runtime every lane names its slots by ref.
            refs: true,
            ..Seated::of(engine_metal::Lane {
                slot: slot as u32,
                word: trace.facts.word(&Request::new(1, false)),
                tokens,
            })
        })
        .collect();
    let rows = shell.fire_seated(&lanes).expect("the plain lanes fire");
    for (row, token) in rows.iter().zip([5, 11, 3]) {
        reads(&table, token, row);
    }

    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
