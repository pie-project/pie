//! **The node map, against real captures** (`.wiki/palo/cuda-abi.md` §7,
//! step 3). Unlike `cuda_descriptor_abi_probe.rs` beside it, this one is a
//! GATE: every claim it makes is a rule step 4 is allowed to build on.
//!
//! The rules under test are the three answers `device::map::diff` can give,
//! and each is asked of a graph the driver actually built rather than of a
//! `Walked` this file made up (the invented ones are the unit tests inside
//! `device/map.rs`, and they are the ones that run without a GPU):
//!
//! ```text
//! same walk, twice          same fingerprint, and nothing to patch
//! one scalar moved          exactly one component, named
//! a node more               NotSameTopology — which is not a refusal
//! an ambiguous pair that
//!   agrees / disagrees      pass / a named refusal
//! ```
//!
//! The synthetic graphs are four launches of one nvrtc kernel, captured on
//! the shell's own `Graph::capture`, and the ambiguous pair is ENGINEERED:
//! two launches forked across two streams land at one depth with one symbol,
//! which is the shape the probe counted 78 of on the mixed composition and
//! the only shape where the canonical order is a guess. Building it by hand
//! is the point — the real capture's 78 are not reproducible on demand, and a
//! rule that only ever sees the easy case is not a rule.
//!
//! The last test is the real one: a qwen capture through the `keep_graphs`
//! seam, walked into a map, with its census PRINTED rather than pinned (the
//! probe's own lesson: pin the rules, print the catalogs).
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!   --test cuda_node_map -- --nocapture --test-threads=1
//! ```
//!
//! # Gating
//!
//! The whole file is behind `_cuda` because it launches kernels through
//! `cudarc` directly — there is no kernel entry in `kernels-cuda` whose
//! topology this file could bend into a fork — and every test skips at run
//! time when the machine has no device, as `graph_replay.rs` does.
#![cfg(feature = "_cuda")]

use core::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use engine_cuda::device::graph::Event;
use engine_cuda::device::map::{self, Component, Diff, NodeMap, Refused};
use engine_cuda::device::nodes::{self, Walked};
use engine_cuda::device::{Buffer, Graph};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_exec::law::{At, Law, Refuse};

/// One device at a time: every test here binds device 0 and captures on its
/// own stream, and the last one loads a whole shell whose scratch slabs are
/// process-global (`serve_smoke.rs` states that argument in full).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The kernel every synthetic graph launches. One symbol for every node on
/// purpose: what makes two nodes ambiguous is depth plus symbol, so a graph
/// of one kernel is the sharpest instrument for both halves — a chain of it
/// has no ambiguity at all, and a fork of it has nothing else.
const SOURCE: &str = r#"
extern "C" __global__ void bump(float* out, int add) {
    out[threadIdx.x] += (float)add;
}
"#;

/// A bound device, a kernel, two streams and somewhere for the pointer
/// argument to point.
struct Bench {
    stream: *mut c_void,
    forked: *mut c_void,
    func: dr::CUfunction,
    out: Buffer,
}

impl Bench {
    /// Bind, compile and open — or `None` and a sentence, when the machine
    /// has no device.
    fn open(what: &str) -> Option<Bench> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        // SAFETY: live out-parameters, and this thread is the one that will
        // capture. `cudaSetDevice` is what makes the primary context current
        // for the driver calls below.
        unsafe {
            assert_eq!(rt::cudaSetDevice(0), rt::cudaError::cudaSuccess);
            let mut stream: rt::cudaStream_t = core::ptr::null_mut();
            let mut forked: rt::cudaStream_t = core::ptr::null_mut();
            assert_eq!(
                rt::cudaStreamCreate(&raw mut stream),
                rt::cudaError::cudaSuccess
            );
            assert_eq!(
                rt::cudaStreamCreate(&raw mut forked),
                rt::cudaError::cudaSuccess
            );

            let ptx = cudarc::nvrtc::compile_ptx(SOURCE).expect("the probe kernel compiles");
            let image = std::ffi::CString::new(ptx.to_src()).expect("ptx holds no NUL");
            let mut module: dr::CUmodule = core::ptr::null_mut();
            assert_eq!(
                dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()),
                dr::CUresult::CUDA_SUCCESS
            );
            let name = std::ffi::CString::new("bump").expect("a literal holds no NUL");
            let mut func: dr::CUfunction = core::ptr::null_mut();
            assert_eq!(
                dr::cuModuleGetFunction(&raw mut func, module, name.as_ptr()),
                dr::CUresult::CUDA_SUCCESS
            );

            Some(Bench {
                stream: stream.cast(),
                forked: forked.cast(),
                func,
                out: Buffer::zeroed(1024).expect("32 floats of scratch"),
            })
        }
    }

    /// Enqueue one `bump(out, add)` on `stream`.
    fn launch(&self, stream: *mut c_void, add: i32) {
        let mut ptr = self.out.ptr();
        let mut add = add;
        let mut args: [*mut c_void; 2] = [
            core::ptr::from_mut(&mut ptr).cast(),
            core::ptr::from_mut(&mut add).cast(),
        ];
        // SAFETY: `args` names two live locals for the duration of the call,
        // which is all `cuLaunchKernel` needs — it copies the argument values
        // before returning, capture or no capture.
        let code = unsafe {
            dr::cuLaunchKernel(
                self.func,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                self.stream_of(stream),
                args.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "the probe launch enqueues");
    }

    fn stream_of(&self, stream: *mut c_void) -> dr::CUstream {
        stream.cast()
    }

    /// A chain: `adds.len()` launches on one stream, one node per add, each
    /// depending on the one before it.
    fn chain(&self, adds: &[i32]) -> Walked {
        let graph = Graph::capture(self.stream, || {
            for add in adds {
                self.launch(self.stream, *add);
            }
            Ok(())
        })
        .expect("the chain captures");
        nodes::walk(&graph).expect("the chain walks")
    }

    /// A fork: one launch, then `left` and `right` side by side across two
    /// streams, then a join. The two middle nodes share a depth AND a symbol,
    /// which is the whole reason this file exists.
    fn fork(&self, left: i32, right: i32) -> Walked {
        let graph = Graph::capture(self.stream, || {
            self.launch(self.stream, 1);
            let split = Event::new()?;
            split.record(self.stream)?;
            split.wait(self.forked)?;
            self.launch(self.stream, left);
            self.launch(self.forked, right);
            let join = Event::new()?;
            join.record(self.forked)?;
            join.wait(self.stream)?;
            self.launch(self.stream, 4);
            Ok(())
        })
        .expect("the fork captures");
        nodes::walk(&graph).expect("the fork walks")
    }
}

/// The patches of an alignment, or a sentence saying what came back instead.
fn aligned(diff: Diff) -> (Vec<map::Patch>, usize, usize) {
    match diff {
        Diff::Aligned {
            patches,
            unmoved,
            agreed,
        } => (patches, unmoved, agreed),
        Diff::NotSameTopology { held, brought } => {
            panic!("expected an alignment; the two captures fingerprinted {held} and {brought}")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The rules

#[test]
fn two_captures_of_one_graph_fingerprint_alike_and_ask_for_no_patch() {
    let _serial = serialized();
    let Some(bench) = Bench::open("two captures of one graph") else {
        return;
    };

    let held = NodeMap::from_walk(bench.chain(&[1, 2, 3]));
    let brought = bench.chain(&[1, 2, 3]);

    assert_eq!(held.len(), 3, "three launches, three nodes");
    assert_eq!(
        held.topology(),
        map::Topology::of(&brought),
        "the same walk captured twice is the same topology"
    );
    let (patches, unmoved, agreed) =
        aligned(map::diff(&held, &brought).expect("nothing to refuse"));
    assert!(
        patches.is_empty(),
        "nothing moved between two identical captures, and yet: {:?}",
        patches.iter().map(|p| (p.at, &p.moved)).collect::<Vec<_>>()
    );
    assert_eq!((unmoved, agreed), (3, 0));
}

#[test]
fn a_scalar_that_changed_between_captures_is_the_only_thing_the_patch_names() {
    let _serial = serialized();
    let Some(bench) = Bench::open("one moved scalar") else {
        return;
    };

    let held = NodeMap::from_walk(bench.chain(&[1, 2, 3]));
    let brought = bench.chain(&[1, 7, 3]);

    let (patches, unmoved, _) = aligned(map::diff(&held, &brought).expect("nothing to refuse"));
    assert_eq!(patches.len(), 1, "one launch's argument moved");
    assert_eq!(patches[0].at, 1, "the middle node of the chain");
    assert_eq!(
        patches[0].moved,
        vec![Component::new(
            1,
            At::Arg { at: 1, word: 0 },
            Law::Const(7)
        )],
        "the SECOND parameter — the pointer never moved — named in the shared \
         language: which node, which word, and the value that rides"
    );
    assert_eq!(
        patches[0].params[1].word(),
        Some(7),
        "the patch carries what the new capture wants, not what the old one had"
    );
    assert_eq!(unmoved, 2);
    assert!(
        !patches[0].node.is_null() && !patches[0].entry.is_null(),
        "a patch carries the driver's own handles, or it cannot be applied"
    );
}

#[test]
fn a_capture_with_a_launch_more_is_not_the_same_topology() {
    let _serial = serialized();
    let Some(bench) = Bench::open("a launch more") else {
        return;
    };

    let held = NodeMap::from_walk(bench.chain(&[1, 2, 3]));
    let brought = bench.chain(&[1, 2, 3, 4]);

    let answer = map::diff(&held, &brought).expect("a new shape is not a refusal");
    let Diff::NotSameTopology { held: a, brought: b } = answer else {
        panic!("four nodes aligned against three")
    };
    assert_ne!(a, b);
    assert_eq!((a.nodes, b.nodes), (3, 4));
}

#[test]
fn a_forked_pair_that_agrees_byte_for_byte_needs_no_patch() {
    let _serial = serialized();
    let Some(bench) = Bench::open("an ambiguous pair that agrees") else {
        return;
    };

    let held = NodeMap::from_walk(bench.fork(5, 5));
    let census = held.census();
    eprintln!("   the engineered fork: {census}");
    assert_eq!(census.nodes, 4, "one launch, two forked, one joined");
    assert_eq!(
        census.ambiguous, 2,
        "the two forked launches share a depth and a symbol"
    );
    assert_eq!(held.classes().len(), 1);
    assert_eq!(held.classes()[0].depth, 1);

    let (patches, unmoved, agreed) =
        aligned(map::diff(&held, &bench.fork(5, 5)).expect("identical bytes cannot mislead"));
    assert!(patches.is_empty());
    assert_eq!(
        (unmoved, agreed),
        (2, 2),
        "the pair passes BECAUSE the guess is unobservable, not because it was checked"
    );
}

#[test]
fn a_forked_pair_whose_arguments_disagree_refuses_the_alignment_by_name() {
    let _serial = serialized();
    let Some(bench) = Bench::open("an ambiguous pair that disagrees") else {
        return;
    };

    let held = NodeMap::from_walk(bench.fork(5, 6));
    assert_eq!(held.census().ambiguous, 2);

    let refused = map::diff(&held, &bench.fork(5, 9)).expect_err("the guess is refused");
    let Refused::Ambiguous {
        depth,
        symbol,
        count,
        differing,
        component,
        ..
    } = &refused
    else {
        panic!("the ambiguity is what refused, not: {refused}")
    };
    assert_eq!(*depth, 1, "the refusal names the depth");
    assert!(symbol.contains("bump"), "and the symbol: {symbol}");
    assert_eq!(*count, 2);
    assert!(
        (1..=2).contains(differing),
        "at least the node that moved, and both if the driver permuted them"
    );
    assert_eq!(component.at, At::Arg { at: 1, word: 0 });
    assert_eq!(refused.reason(), Refuse::Ambiguous);
    eprintln!("   the refusal reads: {refused}");
}

// ─────────────────────────────────────────────────────────────────────────
// The real capture

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";

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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let checkpoint = snapshot()?;
    let container = container(&checkpoint)?;
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);
    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 1024,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::On,
        knobs: engine_cuda::Knobs::default(),
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

/// **THE ONE THAT MATTERS.** A real capture, walked into a map: every
/// parameter block readable (the pin — 621 of 621 on the probe's L40S, and a
/// map cannot exist for a graph it cannot read), and the ambiguity census
/// printed (the catalog — the number belongs to a driver version and a
/// composition, and asserting it would be asserting somebody else's
/// enumeration).
#[test]
fn a_real_capture_becomes_a_node_map_and_reports_its_ambiguity_census() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the node map over a real capture") else {
        eprintln!("skipping the real-capture node map: no device or no checkpoint");
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let filler: Vec<u32> = core::iter::repeat_n(prompt[0], 32).collect();

    // Seat the slots eagerly, then warm at the ceiling before any capture —
    // the probe's doctrine, and for its reason: a scratch slab that grows
    // between two captures moves every pointer into it.
    shell.set_mode(Graphs::Off);
    let mut fed = Vec::new();
    for slot in 0..2u32 {
        shell.open(slot).expect("a slot opens");
        let out = shell
            .fire(&[Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: &prompt,
            }])
            .expect("a seating prefill fires");
        fed.push(argmax(&out[0]));
    }
    for _ in 0..2 {
        shell.open(1).expect("slot 1 opens");
        shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: core::slice::from_ref(&fed[0]),
                },
                Lane {
                    slot: 1,
                    word: word(32),
                    tokens: &filler[..32],
                },
            ])
            .expect("the ceiling warm fires");
    }

    shell.keep_graphs(true);
    shell.set_mode(Graphs::On);

    // Two compositions, so the census has something to compare against.
    for _ in 0..3 {
        shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: core::slice::from_ref(&fed[0]),
            }])
            .expect("the all-decode fire fires");
    }
    shell.open(1).expect("slot 1 re-opens");
    for _ in 0..3 {
        shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: core::slice::from_ref(&fed[0]),
                },
                Lane {
                    slot: 1,
                    word: word(8),
                    tokens: &filler[..8],
                },
            ])
            .expect("the mixed fire fires");
    }

    let kept = shell.kept_graphs();
    assert!(!kept.is_empty(), "the shell captured something to walk");

    let mut maps: Vec<(String, NodeMap)> = Vec::new();
    for (key, graph) in kept {
        let map = NodeMap::of(graph).expect("a kept graph walks into a map");
        let census = map.census();
        eprintln!("\n-- {key}\n   {census}\n   topology {}", map.topology());

        // THE PIN. A map is a coordinate system over parameter blocks; a node
        // whose block never read has no coordinates, and `Refused::Opaque`
        // exists for the case where one moves. On this SKU the probe measured
        // zero of them, and a regression here means the walk lost a form.
        assert_eq!(
            census.opaque, 0,
            "every kernel node's parameter block reads: {} of {} on {key}",
            census.readable, census.kernels,
        );

        // THE CATALOG. Which symbols the canonical order cannot tell apart,
        // biggest classes first — printed, because the number belongs to a
        // driver version and a composition rather than to this rule.
        let mut classes: Vec<&map::Ambiguous> = map.classes().iter().collect();
        classes.sort_by_key(|class| core::cmp::Reverse(class.at.len()));
        for class in classes.iter().take(6) {
            eprintln!(
                "   ambiguous: {:>3} nodes at depth {:<4} {}",
                class.at.len(),
                class.depth,
                class.symbol.chars().take(64).collect::<String>(),
            );
        }
        if classes.len() > 6 {
            eprintln!("   ... {} more classes", classes.len() - 6);
        }

        // A SECOND READING OF THE SAME GRAPH. Weaker than two captures — the
        // driver is enumerating one graph twice, not building two — but it is
        // the strongest form the exec cache offers today (one capture per
        // key, so a second capture of one composition does not exist to diff
        // against), and it exercises the whole alignment over 400+ real
        // nodes, ambiguous classes included.
        let again = nodes::walk(graph).expect("the kept graph walks twice");
        let (patches, unmoved, agreed) = aligned(
            map::diff(&map, &again).expect("one graph read twice cannot disagree with itself"),
        );
        assert!(
            patches.is_empty(),
            "a graph diffed against itself moved {} components",
            patches.iter().map(|patch| patch.moved.len()).sum::<usize>(),
        );
        assert_eq!(
            unmoved + agreed,
            census.kernels,
            "every kernel node is accounted for, ambiguous or not"
        );
        assert_eq!(agreed, census.ambiguous, "and every ambiguous one agreed");

        maps.push((format!("{key}"), map));
    }

    // TWO COMPOSITIONS ARE TWO TOPOLOGIES. The walk skips zero-row regions,
    // so all-decode and decode+prefill do not hold the same launches — the
    // fingerprint has to say so, and saying so is not a refusal.
    if let [(a_key, a), (b_key, b), ..] = maps.as_slice() {
        assert_ne!(
            a.topology(),
            b.topology(),
            "{a_key} and {b_key} are different compositions"
        );
        let again = nodes::walk(&shell.kept_graphs()[1].1).expect("the second kept graph walks");
        assert!(
            matches!(
                map::diff(a, &again).expect("a different composition is not a refusal"),
                Diff::NotSameTopology { .. }
            ),
            "{b_key}'s walk against {a_key}'s map is a miss, not a rebind"
        );
    }
}
