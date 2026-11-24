#![cfg(feature = "rocm")]
//! A shell dropped leaves the queues where its load found them: load after load in one
//! process leaves KFD's list of the process's queues where the first single
//! load's drop left it (at most one queue past the two-shell drop's: HIP
//! opens one lazily and pools it), HIP's rule (`kfd::save_areas`) flat and never under the
//! list, and the save areas the free figure holds back (`kfd::held`) flat
//! with them. A reloading server keeps the device memory its first load
//! found.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Graphs, Shell};
use kernels_rocm::hip::kfd;
use poem::{Dtype, ForwardHybrid, Input, Platform, StateRows, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
const LANES: u32 = 4;
const CYCLES: usize = 4;

/// One attention layer.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn load(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: 2 * LANES + 1,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::On, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads")
}

/// (KFD's listed queues, HIP's rule's queues, the bytes the free figure
/// holds back for them), now.
fn queues() -> (Option<u32>, u32, u64) {
    let ruled = kfd::save_areas(u32::MAX).expect("the rule's save areas read");
    let held = kfd::held().expect("the held save areas read");
    (kfd::listed_queues(), ruled.queues, held.bytes())
}

/// HIP's rule never counts fewer queues than KFD lists, where its list is
/// this process's: the free figure it sizes a flat range by is never more
/// than the card has.
fn at_least_listed(at: &str, (listed, ruled, _): (Option<u32>, u32, u64)) {
    if let Some(listed) = listed {
        assert!(
            ruled >= listed,
            "{at}: HIP's rule counts {ruled} queues and KFD lists {listed}"
        );
    }
}

#[test]
fn a_dropped_shell_takes_its_queues_with_it() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let trace = trace_hybrid("queues-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2207).write(dir.path());
    // Two shells at once: the most streams that copy at once.
    let (one, two) = (load(trace.clone(), &path), load(trace.clone(), &path));
    at_least_listed("two shells live", queues());
    drop((one, two));
    let first = queues();
    at_least_listed("both dropped", first);
    // One single-shell load first: HIP opens a hardware queue lazily, for a
    // stream the two shells' never took, and pools it once its stream is
    // destroyed, so the first single load may leave one queue (and its save
    // area) past the two-shell drop's. Bounded here, at one; every reload
    // after it must leave exactly what this one left.
    let warm = {
        drop(load(trace.clone(), &path));
        queues()
    };
    eprintln!("after the two-shell drop {first:?}, after one warm reload {warm:?}");
    at_least_listed("warm reload dropped", warm);
    let area = kfd::save_areas(u32::MAX).expect("the rule's save areas read").area();
    if let (Some(warm_listed), Some(first_listed)) = (warm.0, first.0) {
        assert!(
            warm_listed <= first_listed + 1,
            "the warm reload's drop left {warm_listed} queues listed, past the two-shell drop's \
             {first_listed} by more than HIP's one lazily opened queue"
        );
    }
    assert!(
        warm.2 <= first.2 + area,
        "the warm reload's drop left {} save-area bytes held, past the two-shell drop's {} by more \
         than one queue's {area}",
        warm.2,
        first.2
    );
    for cycle in 0..CYCLES {
        let shell = load(trace.clone(), &path);
        at_least_listed(&format!("reload {cycle} live"), queues());
        drop(shell);
        let now = queues();
        eprintln!("after reload {cycle}'s drop (listed, ruled, held bytes): {now:?}");
        at_least_listed(&format!("reload {cycle} dropped"), now);
        assert_eq!(
            now, warm,
            "reload {cycle}'s drop left the queues or their save areas grown past the warm reload's"
        );
    }
}
