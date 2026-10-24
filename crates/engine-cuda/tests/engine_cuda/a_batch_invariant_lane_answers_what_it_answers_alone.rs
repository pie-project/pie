#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Knobs, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const MODEL: &str = "qwen3-micro-text";
const PAGE: u32 = 16;
const CONTEXT: u32 = 512;
const LANES: u32 = 4;
const TOKENS: u32 = 512;
const VOCAB: u32 = 256;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut bytes = Vec::with_capacity(count * 2);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value: f32 = if norm { 1.0 + 0.05 * unit } else { 0.2 * unit };
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

fn load(trace: &Trace, contract: &ModelContract, container: &Path, invariant: bool) -> Shell {
    Shell::load(Boot {
        trace: trace.clone(),
        contract,
        checkpoint: container,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: LANES,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: Knobs {
            batch_invariant: invariant,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the micro loads")
}

fn tokens(len: u32, salt: u32) -> Vec<u32> {
    (0..len)
        .map(|t| (t * 37 + salt * 101 + 5) % VOCAB)
        .collect()
}

/// One fire: each lane's slot and the tokens it appends; the logits of
/// slot 0's lane, which every fire seats.
fn fire(shell: &mut Shell, trace: &Trace, lanes: &[(u32, &[u32])]) -> Vec<f32> {
    let seated: Vec<Lane<'_>> = lanes
        .iter()
        .map(|&(slot, tokens)| Lane {
            slot,
            word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
            tokens,
        })
        .collect();
    let mut logits = shell.fire(&seated).expect("the fire runs");
    let at = lanes
        .iter()
        .position(|&(slot, _)| slot == 0)
        .expect("slot 0 is seated");
    logits.swap_remove(at)
}

/// Slot 0's prompt, then three decode steps, alone and then in crowds:
/// its prompt beside longer and shorter ones, its steps at the front,
/// middle and back of the fire, among decodes and beside a prefill. Its
/// logits a step.
fn steps(shell: &mut Shell, trace: &Trace) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let prompt = tokens(29, 0);
    let next = [tokens(1, 1), tokens(1, 2), tokens(1, 3)];
    shell.open(0).unwrap();
    let alone = std::iter::once(fire(shell, trace, &[(0, &prompt)]))
        .chain(next.iter().map(|t| fire(shell, trace, &[(0, t)])))
        .collect();

    for slot in 0..LANES {
        shell.open(slot).unwrap();
    }
    let (a, b, c, d) = (tokens(61, 4), tokens(7, 5), tokens(200, 6), tokens(150, 7));
    let (a1, b1, d1) = (tokens(1, 8), tokens(1, 9), tokens(1, 10));
    let first = fire(shell, trace, &[(1, &a), (0, &prompt), (2, &b), (3, &c)]);
    shell.open(3).unwrap();
    let crowded = vec![
        first,
        fire(shell, trace, &[(0, &next[0]), (1, &a1), (2, &b1), (3, &d)]),
        fire(shell, trace, &[(2, &b1), (3, &d1), (0, &next[1])]),
        fire(shell, trace, &[(0, &next[2]), (1, &a1)]),
    ];
    (alone, crowded)
}

fn differing(alone: &[Vec<f32>], crowded: &[Vec<f32>]) -> Vec<(usize, usize, f32)> {
    alone
        .iter()
        .zip(crowded)
        .enumerate()
        .filter_map(|(step, (a, c))| {
            let off = a
                .iter()
                .zip(c)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let worst = a
                .iter()
                .zip(c)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            (off > 0).then_some((step, off, worst))
        })
        .collect()
}

/// Under `batch_invariant`, a lane's logits are the bits it answers alone
/// whichever lanes share its fire and wherever it sits among them, at its
/// prefill and at each decode step. Without the knob they may differ (a
/// fire's rows choose the GEMM tactic and the attention split); that is
/// printed, not held.
#[test]
fn a_batch_invariant_lane_answers_what_it_answers_alone() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let trace = poem_compiler::catalog::trace_of(MODEL, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let dir = std::env::temp_dir().join(format!("pie-invariant-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _scratch = Scratch(dir.clone());
    let container = dir.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("the fixture's planes are the plan's own");
    drop(source);

    let mut tuned = load(&trace, &contract, &container, false);
    let (alone, crowded) = steps(&mut tuned, &trace);
    drop(tuned);
    let spread = alone[0].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - alone[0].iter().copied().fold(f32::INFINITY, f32::min);
    assert!(spread > 1e-3, "the logits are a rectangle nothing wrote");
    eprintln!(
        "without the knob, (step, logits off, worst gap) where a crowd moved them: {:?}",
        differing(&alone, &crowded)
    );

    let mut invariant = load(&trace, &contract, &container, true);
    let (alone, crowded) = steps(&mut invariant, &trace);
    assert_eq!(
        differing(&alone, &crowded),
        vec![],
        "a crowd moved slot 0's logits under the knob"
    );
}
