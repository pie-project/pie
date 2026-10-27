#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::{Mask, Masking, Readout};
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Trace, Weight, fact, ops,
    trace_hybrid,
};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const ROWS: u32 = 7;

/// One attention layer: `stated` masks both ways, denoise attends every key, the rest causal.
struct Micro {
    stated: bool,
}

impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        inputs
            .recorder()
            .declare_switch("mode", &["encode", "denoise"]);
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        let pages = inputs.kv(KV_ROW);
        ops::attn::kv_append(
            &k,
            Some(&v),
            pages,
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = if self.stated {
            let masked = inputs.on(fact::has(fact::Mask));
            let plan = ops::attn::plan_prefill(&masked, HEADS, HEADS, HEAD_DIM, None);
            ops::attn::prefill(
                &q.on(fact::has(fact::Mask)),
                &plan,
                pages,
                None,
                HEAD_DIM,
                HEADS,
                SM_SCALE,
                false,
                Some(&masked.mask()),
                false,
                None,
                None,
                None,
                false,
                None,
                None,
                None,
            )
            .0
        } else {
            ops::attn::attend(
                &q,
                &inputs,
                KV_ROW,
                HEADS,
                HEADS,
                HEAD_DIM,
                SM_SCALE,
                None,
                None,
                Some(&!fact::switch_is("mode", "denoise")),
                false,
                None,
                None,
                false,
                false,
            )
        };
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: Trace, path: &Path, graphs: Graphs) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(graphs, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    shell
}

fn rows(
    trace: &Trace,
    path: &Path,
    graphs: Graphs,
    request: Request,
    mask: Option<&Masking>,
) -> Vec<f32> {
    let tokens: Vec<u32> = (0..ROWS).map(|at| (at * 13 + 5) % VOCAB).collect();
    let word = trace.facts.word(&request);
    let mut shell = shell(trace.clone(), path, graphs);
    let lane = Lane {
        slot: 0,
        word,
        tokens: &tokens,
    };
    let every: Vec<u32> = (0..ROWS).collect();
    let seated = Seated {
        readout: Some(&every),
        ..match mask {
            Some(mask) => Seated::masked(lane, mask),
            None => Seated::of(lane),
        }
    };
    let mut out = shell
        .fire_rows(&[seated], &[Readout::Rows(every.clone())])
        .expect("the fire runs");
    out.remove(0)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

// A wide class's lane stating no mask attends every key, as one stating the whole extent does.
#[test]
fn a_wide_class_attends_every_key_without_a_mask_every_case() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    for graphs in [Graphs::Off, Graphs::On] {
        a_wide_class_attends_every_key_without_a_mask(graphs);
    }
}

fn a_wide_class_attends_every_key_without_a_mask(graphs: Graphs) {
    let graph = trace_hybrid("wide-micro", &Micro { stated: false }, Platform::Cuda);
    let stated = trace_hybrid("wide-micro", &Micro { stated: true }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&graph, 1103).write(dir.path());

    let whole = Masking::Extent(Mask::new(vec![0, ROWS], u64::from(ROWS)));
    let want = rows(
        &stated,
        &path,
        graphs,
        Request::new(ROWS, true),
        Some(&whole),
    );
    let wide = rows(
        &graph,
        &path,
        graphs,
        Request::new(ROWS, false).switched("mode", "denoise"),
        None,
    );
    let causal = rows(&graph, &path, graphs, Request::new(ROWS, false), None);

    let off = worst(&wide, &want);
    assert!(
        off <= 1e-2,
        "{graphs:?}: the wide lane answers {off} away from the lane stating its whole extent"
    );
    let vocab = VOCAB as usize;
    assert!(
        worst(&causal[..vocab], &wide[..vocab]) > 1e-2,
        "the first row of a causal lane reads itself alone, and the wide one every row"
    );
    let last = (ROWS as usize - 1) * vocab;
    assert!(
        worst(&causal[last..], &wide[last..]) <= 1e-2,
        "the last row reads every row either way"
    );
}
