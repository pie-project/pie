#![cfg(feature = "device")]
//! A load arms the bodies it would arm alone, whichever load of the process
//! ran before it: what a kernel's first launch in a process loads (code
//! objects, a library's kernels, the runtime's queues) is the process's, and
//! no load's measured figure counts it. Each order runs in a process of its
//! own (this test's binary again), so the first load of each is truly the
//! process's first.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine_cuda::{Boot, Graphs, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, StateRows, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
const LANES: u32 = 4;
const CHILD: &str = "PIE_CENSUS_ORDER";

/// One attention layer at `width`: two of them are two loads.
struct Micro {
    width: u64,
}

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
        let table = w("embed", u64::from(VOCAB), self.width);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, self.width));
        let k = ops::linear::matmul(&x, &w("k", plane, self.width));
        let v = ops::linear::matmul(&x, &w("v", plane, self.width));
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
        let h = ops::linear::matmul(&a, &w("o", self.width, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

/// (bodies resident, bodies armed) of the load at `width`.
fn census(width: u64) -> (usize, u64) {
    let trace = trace_hybrid("census-micro", &Micro { width }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2207).write(dir.path());
    let shell = load(trace, &path);
    let stats = shell.body_stats();
    (stats.census.bodies, stats.tally.armed_at_load)
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
        // No record: a flat range's load measures its bodies.
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
        steps: engine::StepRules::runtime(),
    })
    .expect("the shell loads")
}

/// The census of each width in `order`, run in a process of its own.
fn in_a_fresh_process(order: &str) -> Vec<(u64, (usize, u64))> {
    let out = std::process::Command::new(std::env::current_exe().expect("this test's binary"))
        .args([
            "--exact",
            "a_loads_census_is_its_own_in_either_order::a_loads_census_is_its_own_in_either_order",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, order)
        .output()
        .expect("the child runs");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the {order} child failed:\n{text}");
    text.lines()
        .filter_map(|line| line.split_once("[census] ").map(|(_, rest)| rest))
        .map(|rest| {
            let n: Vec<u64> = rest.split_whitespace().map(|x| x.parse().expect("a number")).collect();
            (n[0], (n[1] as usize, n[2]))
        })
        .collect()
}

#[test]
fn a_loads_census_is_its_own_in_either_order() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    if let Ok(order) = std::env::var(CHILD) {
        for width in order.split(',').map(|w| w.parse::<u64>().expect("a width")) {
            let (bodies, armed) = census(width);
            println!("[census] {width} {bodies} {armed}");
        }
        return;
    }
    let ab = in_a_fresh_process("128,256");
    let ba = in_a_fresh_process("256,128");
    eprintln!("first 128 then 256: {ab:?}; first 256 then 128: {ba:?}");
    for width in [128u64, 256] {
        let of = |runs: &[(u64, (usize, u64))]| {
            runs.iter().find(|(w, _)| *w == width).map(|(_, c)| *c).expect("a census")
        };
        assert!(of(&ab).0 > 0, "the load at {width} arms bodies");
        assert_eq!(
            of(&ab),
            of(&ba),
            "the load at {width} armed (resident, armed) apart by whether it ran first"
        );
    }
}
