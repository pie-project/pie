#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! An expert-parallel group, rank `r` on device `r % count`: each reads its
//! part of every MoE layer's experts from the one checkpoint, dispatches
//! its tokens to the ranks holding their routes and combines the outputs
//! back. Each rank's logits are the logits one rank holding every expert
//! answers for the same prompt, whether the other ranks fire their own
//! prompts in the step or idle through it serving the first rank's routes.

use checkpoint::contract::ModelContract;
use engine_cuda::ep::{EpRank, wire};
use engine_cuda::{Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::Trace;

use crate::micro::{KV, MICRO, SHARE, Scratch, close, ordinals, prompt, ranks, write_checkpoint};

const CEILING: u32 = 256;
const PAGE: u32 = 16;

struct Loaded {
    shell: Shell,
    ordinal: i32,
    _dir: Scratch,
}

// SAFETY: as `Cuda`'s: a shell's raw handles name driver objects any
// thread may use once it binds the device's context, which each thread
// here does before touching it, and one thread holds it at a time.
unsafe impl Send for Loaded {}

impl Loaded {
    /// One lane's logits, fired at the group's `bucket` on its own thread.
    fn fire_at(&mut self, bucket: u32, lane: Lane<'_>) -> Vec<Vec<f32>> {
        engine_cuda::device::ctx::bind_thread(self.ordinal).unwrap();
        self.shell.at_bucket(bucket);
        self.shell.fire(&[lane]).unwrap()
    }

    fn idle(&mut self, bucket: u32) {
        engine_cuda::device::ctx::bind_thread(self.ordinal).unwrap();
        self.shell.fire_idle(bucket).unwrap();
    }
}

/// Moves a value to the thread that drives it: see `Loaded`'s `Send`.
struct Carried<T>(T);

// SAFETY: as `Loaded`'s.
unsafe impl<T> Send for Carried<T> {}

/// A load of `trace` from `whole`'s checkpoint as `expert`'s rank on device
/// `ordinal`, up to its bodies, and the directory the checkpoint is in.
fn stage(
    trace: &Trace,
    whole: &Trace,
    expert: engine_cuda::World,
    ordinal: i32,
    tag: &str,
    graphs: bool,
) -> (engine_cuda::serve::Staged, Scratch) {
    let dir = std::env::temp_dir().join(format!("pie-ep-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let container = dir.join("micro.zt");
    write_checkpoint(&container, whole);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract: ModelContract =
        poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
            .expect("a container of the plan's own planes is read by its own names");
    drop(source);
    let graphs = if graphs { "on" } else { "off" };
    let chip = engine_cuda::chip::profile(&format!("[engine]\ngraphs = \"{graphs}\"\n"));
    engine_cuda::device::ctx::bind_thread(ordinal).unwrap();
    let staged = Shell::stage(engine_cuda::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(1, CEILING),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CEILING,
        slots: 2,
        kv_budget: Some(KV),
        memory_share: SHARE,
        ordinal,
        knobs: engine_cuda::Knobs {
            chip: Some(chip),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert,
        comm: core::ptr::null_mut(),
    })
    .expect("the micro loads");
    (staged, Scratch(dir))
}

fn load(trace: &Trace, ordinal: i32, tag: &str, graphs: bool) -> Loaded {
    let world = engine_cuda::World::default();
    let (staged, dir) = stage(trace, trace, world, ordinal, tag, graphs);
    Loaded {
        shell: staged.finish().expect("the micro arms"),
        ordinal,
        _dir: dir,
    }
}

/// A staged rank armed on the thread that will have driven it.
fn arm(staged: Carried<(engine_cuda::serve::Staged, Scratch)>, ordinal: i32) -> Carried<Loaded> {
    let Carried((staged, dir)) = staged;
    engine_cuda::device::ctx::bind_thread(ordinal).unwrap();
    Carried(Loaded {
        shell: staged.finish().expect("the rank arms"),
        ordinal,
        _dir: dir,
    })
}

/// The group's ranks, wired between staging and arming: arming fires the
/// collectives, so every rank arms at once.
fn group(lowered: &Trace, whole: &Trace, ordinals: &[i32], graphs: bool) -> Vec<Loaded> {
    let size = ordinals.len() as u32;
    let mut staged: Vec<_> = (0..size)
        .map(|rank| {
            let expert = engine_cuda::World { rank, size };
            let ordinal = ordinals[rank as usize];
            let tag = format!("rank{rank}-{graphs}");
            stage(lowered, whole, expert, ordinal, &tag, graphs)
        })
        .collect();
    let shape = staged[0]
        .0
        .shell
        .ep_shape()
        .expect("the lowered micro dispatches");
    let wires = wire(ordinals, shape.widest).expect("every rank reaches every other");
    for (r, (staged, _)) in staged.iter_mut().enumerate() {
        engine_cuda::device::ctx::bind_thread(ordinals[r]).unwrap();
        let ep = EpRank::new(wires.clone(), r as u32, &shape.layers).expect("the placement");
        staged.shell.join_ep(ep).expect("the seats");
    }
    std::thread::scope(|scope| {
        let armed: Vec<_> = (staged.into_iter().zip(ordinals))
            .map(|(staged, &ordinal)| {
                let staged = Carried(staged);
                scope.spawn(move || arm(staged, ordinal))
            })
            .collect();
        armed.into_iter().map(|h| h.join().unwrap().0).collect()
    })
}

/// A lane of `tokens` at `slot`, its word read off their count.
fn lane(word: impl Fn(usize) -> u64, slot: u32, tokens: &[u32]) -> Lane<'_> {
    Lane {
        slot,
        word: word(tokens.len()),
        tokens,
    }
}

#[test]
fn an_expert_parallel_pair_answers_what_one_rank_answers() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    check(false);
    check(true);
}

/// Eager walks, or captured bodies armed on every rank at once.
fn check(graphs: bool) {
    let size = ranks();
    let ordinals = ordinals(size);
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), size, 0).expect("the micro's MoE lowers");
    assert!(
        lowered.params.iter().any(|p| p.experts.is_some()),
        "the lowered micro holds part of a bank"
    );

    let prompts: Vec<Vec<u32>> = (0..size).map(|r| prompt(37 + 15 * r, r + 1)).collect();
    let word = |len: usize| whole.facts.word(&Request::new(len as u32, false));
    let lane = |slot, tokens| lane(word, slot, tokens);

    // One rank holding every expert, on each rank's device: the group's
    // kernels are loaded and tuned on every device before its ranks spin.
    let mut want = Vec::new();
    for &ordinal in ordinals.iter().take(engine_cuda::device::ctx::count()) {
        let tag = format!("one-{graphs}-{ordinal}");
        let mut one = load(&whole, ordinal, &tag, graphs);
        want = (prompts.iter())
            .map(|p| {
                one.shell.open(0).unwrap();
                one.shell.fire(&[lane(0, p)]).unwrap().remove(0)
            })
            .collect();
    }

    let mut ranks = group(&lowered, &whole, &ordinals, graphs);
    for loaded in &mut ranks {
        engine_cuda::device::ctx::bind_thread(loaded.ordinal).unwrap();
        loaded.shell.open(0).unwrap();
        loaded.shell.open(1).unwrap();
    }
    let widest = prompts.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let bucket = ranks[0]
        .shell
        .bucket_for(widest)
        .expect("a bucket holds every prompt");

    // Every rank brings a prompt to the step.
    let got: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let fired: Vec<_> = (ranks.iter_mut().zip(&prompts))
            .map(|(rank, p)| scope.spawn(move || rank.fire_at(bucket, lane(0, p))))
            .collect();
        fired.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r, (got, want)) in got.iter().zip(&want).enumerate() {
        close(
            &got[0],
            want,
            &format!("rank {r}'s prompt beside the others' (graphs {graphs})"),
        );
    }

    // The other ranks idle through rank 0's step, serving its routes.
    let (first, rest) = ranks.split_first_mut().expect("a group has ranks");
    let got = std::thread::scope(|scope| {
        let idle: Vec<_> = (rest.iter_mut())
            .map(|rank| scope.spawn(move || rank.idle(bucket)))
            .collect();
        let got = first.fire_at(bucket, lane(1, &prompts[0]));
        idle.into_iter().for_each(|h| h.join().unwrap());
        got
    });
    close(
        &got[0],
        &want[0],
        &format!("rank 0's prompt with the others idle (graphs {graphs})"),
    );
}
