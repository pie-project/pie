#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! An expert-parallel group, rank `r` on device `r % count`: each reads its
//! part of every MoE layer's experts from the one checkpoint, dispatches
//! its tokens to the ranks holding their routes and combines the outputs
//! back. Each rank's logits are the logits one rank holding every expert
//! answers for the same prompt at the step's bucket, whether the other
//! ranks fire their own prompts in the step or idle through it serving the
//! first rank's routes; and every route of the last MoE layer comes back
//! to its token as its expert answers the row that landed.
//!
//! The micro's logits move by a few tenths of their spread of 8 under a
//! bf16 step in one expert row (a route taken another way four layers
//! on): the routes are checked first, exactly where they are made, so a
//! wrong route is told apart from a kernel that rounds another way.

use checkpoint::contract::ModelContract;
use engine_cuda::ep::{EpRank, Wire, wire};
use engine_cuda::{Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{Collective, Def, Linear, Operation, Trace};

use crate::micro::{
    KV, MICRO, SHARE, Scratch, close, ordinals, prompt, ranks, values, write_checkpoint,
};

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

/// The group's ranks and their wire, wired between staging and arming:
/// arming fires the collectives, so every rank arms at once.
fn group(
    lowered: &Trace,
    whole: &Trace,
    ordinals: &[i32],
    graphs: bool,
) -> (Vec<Loaded>, Vec<Wire>) {
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
    let ranks = std::thread::scope(|scope| {
        let armed: Vec<_> = (staged.into_iter().zip(ordinals))
            .map(|(staged, &ordinal)| {
                let staged = Carried(staged);
                scope.spawn(move || arm(staged, ordinal))
            })
            .collect();
        armed.into_iter().map(|h| h.join().unwrap().0).collect()
    });
    (ranks, wires)
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
    // It answers each prompt at the step's bucket, the widest prompt's, as
    // the group fires it: a fire padded to another bucket rounds its
    // products another way, which no expert parallelism did.
    let widest = prompts.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let mut want = Vec::new();
    let mut bucket = 0;
    for &ordinal in ordinals.iter().take(engine_cuda::device::ctx::count()) {
        let tag = format!("one-{graphs}-{ordinal}");
        let mut one = load(&whole, ordinal, &tag, graphs);
        bucket = (one.shell)
            .bucket_for(widest)
            .expect("a bucket holds every prompt");
        want = (prompts.iter())
            .map(|p| {
                one.shell.open(0).unwrap();
                one.fire_at(bucket, lane(0, p)).remove(0)
            })
            .collect();
    }

    let (mut ranks, wires) = group(&lowered, &whole, &ordinals, graphs);
    for loaded in &mut ranks {
        engine_cuda::device::ctx::bind_thread(loaded.ordinal).unwrap();
        loaded.shell.open(0).unwrap();
        loaded.shell.open(1).unwrap();
    }
    assert_eq!(
        ranks[0].shell.bucket_for(widest),
        Some(bucket),
        "a rank of the group buckets the step as one rank does"
    );

    // Every rank brings a prompt to the step.
    let got: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let fired: Vec<_> = (ranks.iter_mut().zip(&prompts))
            .map(|(rank, p)| scope.spawn(move || rank.fire_at(bucket, lane(0, p))))
            .collect();
        fired.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let live: Vec<usize> = prompts.iter().map(Vec::len).collect();
    routes_return(&lowered, &whole, &wires, &ordinals, bucket, &live, graphs);
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

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `x` rounded to bf16, to nearest even, as a kernel lands it.
fn round(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn down<T: Copy + Default>(at: u64, count: usize) -> Vec<T> {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let mut out = vec![T::default(); count];
    // SAFETY: `out` holds `count` elements; `at` is a device range of at
    // least that many the wire carved; the copy is synchronous, and every
    // rank's fire has settled.
    let code = unsafe {
        rt::cudaMemcpy(
            out.as_mut_ptr().cast(),
            at as *const core::ffi::c_void,
            core::mem::size_of_val(out.as_slice()),
            rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
    };
    assert_eq!(code, rt::cudaError::cudaSuccess, "cudaMemcpy D2H");
    out
}

/// After a step at `bucket` that every rank brought `live[r]` tokens to:
/// each route of the last MoE layer came back to its token's row as its
/// expert answers the row that landed. The wire holds that layer's
/// landing and return until the next step; the experts' gate-up select,
/// GLU and down select are worked on the host from the checkpoint's
/// planes, each product rounded to bf16 as the kernels land it.
#[allow(clippy::too_many_arguments)]
fn routes_return(
    lowered: &Trace,
    whole: &Trace,
    wires: &[Wire],
    ordinals: &[i32],
    bucket: u32,
    live: &[usize],
    graphs: bool,
) {
    let at = (lowered.nodes.iter())
        .rposition(|n| matches!(n.op, Operation::Collective(Collective::Dispatch { .. })))
        .expect("the lowered micro dispatches");
    let Operation::Collective(Collective::Dispatch {
        x,
        local,
        experts,
        slots,
        top_k,
        ..
    }) = lowered.nodes[at].op
    else {
        unreachable!()
    };
    // The layer's two selects, in order, and the GLU between them.
    let mut banks = Vec::new();
    let mut inter = 0;
    for node in &lowered.nodes[at..] {
        match node.op {
            Operation::Linear(Linear::MoeSelect { bank, routes, .. }) if routes == local => {
                let Def::Weight(p) = lowered.values[bank.0 as usize].def else {
                    panic!("a bank is a weight");
                };
                banks.push(&lowered.params[p as usize].name);
            }
            Operation::Linear(Linear::Glu { intermediate, .. }) if banks.len() == 1 => {
                inter = intermediate as usize;
            }
            _ => {}
        }
    }
    let [gate_up, down_bank] = banks[..] else {
        panic!("the last layer's experts are two selects");
    };
    let plane = |name: &str| -> Vec<f32> {
        let param = (whole.params.iter())
            .find(|p| p.name == *name)
            .unwrap_or_else(|| panic!("`{name}` is one rank's plane"));
        let count = param.shape.iter().product::<u64>() as usize;
        values(name, count).into_iter().map(bf16).collect()
    };
    let (gate_up, down_bank) = (plane(gate_up), plane(down_bank));
    let width = match &lowered.values[x.0 as usize].ty {
        poem_ir::Ty::Tensor { shape, .. } => match shape.last() {
            Some(poem_ir::Dim::Const(w)) => *w as usize,
            _ => panic!("a token row is of a stated width"),
        },
        poem_ir::Ty::Struct(_) => panic!("a token row is a tensor"),
    };
    let expert = |e: usize, row: &[f32]| -> Vec<f32> {
        let gu = &gate_up[e * 2 * inter * width..][..2 * inter * width];
        let h: Vec<f32> = (gu.chunks(width))
            .map(|w| round(row.iter().zip(w).map(|(a, b)| a * b).sum()))
            .collect();
        let a: Vec<f32> = (0..inter)
            .map(|c| round(h[c] / (1.0 + (-h[c]).exp()) * h[inter + c]))
            .collect();
        let dn = &down_bank[e * width * inter..][..width * inter];
        (dn.chunks(inter))
            .map(|w| round(a.iter().zip(w).map(|(p, q)| p * q).sum()))
            .collect()
    };

    let world = wires.len();
    let (k, step, slots) = (top_k as usize, bucket as usize, slots as usize);
    let held = engine::placement::Placement::uniform(experts, world as u32, slots as u32)
        .expect("the uniform placement")
        .held;
    let routed: Vec<Vec<u16>> = (wires.iter().zip(ordinals))
        .map(|(w, &ordinal)| {
            engine_cuda::device::ctx::bind_thread(ordinal).unwrap();
            let r = w.routed(bucket, top_k, width as u32, Dtype::Bf16);
            down(r.ptr, step * k * width)
        })
        .collect();
    let mut returned: Vec<usize> = vec![0; world];
    for (d, (w, &ordinal)) in wires.iter().zip(ordinals).enumerate() {
        engine_cuda::device::ctx::bind_thread(ordinal).unwrap();
        let landing = w.landing(bucket, top_k, width as u32, Dtype::Bf16, false);
        let cap = world * step;
        let rows: Vec<u16> = down(landing.rows.ptr, cap * width);
        let slot: Vec<i32> = down(landing.local.ptr, cap * k);
        let origin: Vec<i32> = down(landing.origin.ptr, cap);
        for (i, &o) in origin.iter().enumerate() {
            let Ok(o) = usize::try_from(o) else {
                continue;
            };
            let (s, t) = (o / step, o % step);
            assert!(
                t < live[s],
                "rank {d}'s row {i} came from a token rank {s} lacks"
            );
            let row: Vec<f32> = rows[i * width..][..width]
                .iter()
                .map(|&b| bf16(b))
                .collect();
            for j in 0..k {
                let Ok(l) = usize::try_from(slot[i * k + j]) else {
                    continue;
                };
                let e = held[d * slots + l] as usize;
                let want = expert(e, &row);
                let got = &routed[s][(t * k + j) * width..][..width];
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let worst = (got.iter().zip(&want))
                    .map(|(&g, w)| (bf16(g) - w).abs())
                    .fold(0.0f32, f32::max);
                // A bf16 step or two on a few of the products' sums.
                assert!(
                    worst <= 0.02 * scale,
                    "rank {s} token {t} route {j} (expert {e}, slot {l} of rank {d}) came back \
                     {worst} off its expert's answer of scale {scale} (graphs {graphs})"
                );
                returned[s] += 1;
            }
        }
    }
    for (s, &n) in returned.iter().enumerate() {
        assert_eq!(
            n,
            live[s] * k,
            "rank {s}'s every route was served once (graphs {graphs})"
        );
    }
}
