#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! Two expert-parallel ranks on one device stand in for a pair of peers:
//! each reads its half of every MoE layer's experts from the one checkpoint, dispatches its tokens to
//! the rank holding their routes and combines the outputs back. Each rank's
//! logits are the logits one rank holding every expert answers for the same
//! prompt, whether the other rank fires its own prompt in the step or idles
//! through it serving the first rank's routes.

use checkpoint::contract::ModelContract;
use engine_cuda::ep::{EpRank, wire};
use engine_cuda::{Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::Trace;

use crate::micro::{MICRO, RANKS, Scratch, close, prompt, write_checkpoint};

const CEILING: u32 = 256;
const PAGE: u32 = 16;

struct Loaded {
    shell: Shell,
    _dir: Scratch,
}

// SAFETY: as `Cuda`'s: a shell's raw handles name driver objects any
// thread may use once it binds the device's context, which each thread
// here does before touching it, and one thread holds it at a time.
unsafe impl Send for Loaded {}

impl Loaded {
    /// One lane's logits, fired at the group's `bucket` on its own thread.
    fn fire_at(&mut self, bucket: u32, lane: Lane<'_>) -> Vec<Vec<f32>> {
        engine_cuda::device::ctx::bind_thread(0).unwrap();
        self.shell.at_bucket(bucket);
        self.shell.fire(&[lane]).unwrap()
    }

    fn idle(&mut self, bucket: u32) {
        engine_cuda::device::ctx::bind_thread(0).unwrap();
        self.shell.fire_idle(bucket).unwrap();
    }
}

/// Moves a value to the thread that drives it: see `Loaded`'s `Send`.
struct Carried<T>(T);

// SAFETY: as `Loaded`'s.
unsafe impl<T> Send for Carried<T> {}

/// A load of `trace` from `whole`'s checkpoint as `expert`'s rank, up to
/// its bodies, and the directory the checkpoint is in.
fn stage(
    trace: &Trace,
    whole: &Trace,
    expert: engine_cuda::World,
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
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
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

fn load(trace: &Trace, tag: &str, graphs: bool) -> Loaded {
    let (staged, dir) = stage(trace, trace, engine_cuda::World::default(), tag, graphs);
    Loaded {
        shell: staged.finish().expect("the micro arms"),
        _dir: dir,
    }
}

/// A staged rank armed on the thread that will have driven it.
fn arm(staged: Carried<(engine_cuda::serve::Staged, Scratch)>) -> Carried<Loaded> {
    let Carried((staged, dir)) = staged;
    engine_cuda::device::ctx::bind_thread(0).unwrap();
    Carried(Loaded {
        shell: staged.finish().expect("the rank arms"),
        _dir: dir,
    })
}

/// The group's ranks, wired between staging and arming: arming fires the
/// collectives, so every rank arms at once.
fn group(lowered: &Trace, whole: &Trace, graphs: bool) -> Vec<Loaded> {
    let mut staged: Vec<_> = (0..RANKS)
        .map(|rank| {
            let expert = engine_cuda::World { rank, size: RANKS };
            stage(
                lowered,
                whole,
                expert,
                &format!("rank{rank}-{graphs}"),
                graphs,
            )
        })
        .collect();
    let shape = staged[0]
        .0
        .shell
        .ep_shape()
        .expect("the lowered micro dispatches");
    let wires = wire(&[0, 0], shape.widest).expect("one device reaches itself");
    for (r, (staged, _)) in staged.iter_mut().enumerate() {
        let ep = EpRank::new(wires.clone(), r as u32, &shape.layers).expect("the placement");
        staged.shell.join_ep(ep).expect("the seats");
    }
    std::thread::scope(|scope| {
        let armed: Vec<_> = staged
            .into_iter()
            .map(|staged| {
                let staged = Carried(staged);
                scope.spawn(move || arm(staged))
            })
            .collect();
        armed.into_iter().map(|h| h.join().unwrap().0).collect()
    })
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
    let whole = poem_compiler::catalog::trace_of(MICRO, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let lowered =
        poem_compiler::expert::lower(whole.clone(), RANKS, 0).expect("the micro's MoE lowers");
    assert!(
        lowered.params.iter().any(|p| p.experts.is_some()),
        "the lowered micro holds part of a bank"
    );

    let (a, b) = (prompt(37, 1), prompt(52, 2));
    let word = |len: u32| whole.facts.word(&Request::new(len, false));

    // One rank holding every expert.
    let mut one = load(&whole, &format!("one-{graphs}"), graphs);
    one.shell.open(0).unwrap();
    one.shell.open(1).unwrap();
    let want_a = one
        .shell
        .fire(&[Lane {
            slot: 0,
            word: word(37),
            tokens: &a,
        }])
        .unwrap();
    let want_b = one
        .shell
        .fire(&[Lane {
            slot: 1,
            word: word(52),
            tokens: &b,
        }])
        .unwrap();
    drop(one);

    let mut ranks = group(&lowered, &whole, graphs);
    for loaded in &mut ranks {
        loaded.shell.open(0).unwrap();
        loaded.shell.open(1).unwrap();
    }
    let bucket = ranks[0]
        .shell
        .bucket_for(52)
        .expect("a bucket holds both prompts");

    // Both ranks bring a prompt to the step.
    let [r0, r1] = &mut ranks[..] else {
        unreachable!()
    };
    let lane = |slot, len, tokens| Lane {
        slot,
        word: word(len),
        tokens,
    };
    let (got_a, got_b) = std::thread::scope(|scope| {
        let first = scope.spawn(|| r0.fire_at(bucket, lane(0, 37, &a)));
        let second = scope.spawn(|| r1.fire_at(bucket, lane(0, 52, &b)));
        (first.join().unwrap(), second.join().unwrap())
    });
    close(
        &got_a[0],
        &want_a[0],
        &format!("rank 0's prompt beside rank 1's (graphs {graphs})"),
    );
    close(
        &got_b[0],
        &want_b[0],
        &format!("rank 1's prompt beside rank 0's (graphs {graphs})"),
    );

    // Rank 1 idles through rank 0's step, serving its routes.
    let got = std::thread::scope(|scope| {
        let first = scope.spawn(|| r0.fire_at(bucket, lane(1, 37, &a)));
        let idle = scope.spawn(|| r1.idle(bucket));
        idle.join().unwrap();
        first.join().unwrap()
    });
    close(
        &got[0],
        &want_a[0],
        &format!("rank 0's prompt with rank 1 idle (graphs {graphs})"),
    );
}
