#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::fire::{Mask, Masking, Readout};
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops, trace_hybrid};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 20;
const TOKEN: u32 = 17;
const OTHER: u32 = 43;

/// One attention layer whose masked lanes read by their masks.
struct Micro;

impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) -> Value {
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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, true, false,
            None, None, None, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        ops::linear::lm_head(&h, &table)
    }
}

fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(2, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: 2,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
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

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// Two beams of one pool in one fire: beam 0 writes slot 20, beam 1 writes
/// slot 21 roped at depth 20 and reads up to its own slot. Its mask hides
/// its sibling's slot 20, so it answers what its token answers alone at
/// slot 20. Admitting the sibling, it reads it.
#[test]
fn a_beam_reads_up_to_its_slot_what_its_mask_admits() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = trace_hybrid("beam-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 2203).write(dir.path());
    let mut shell = shell(trace.clone(), &path);
    shell.open(1).expect("the second slot opens");
    let pages: Vec<u32> = (0..64 / PAGE).collect();
    let word = |rows: u32, masked: bool| trace.facts.word(&Request::new(rows, masked));
    let depth = [PROMPT];
    // `(seat, tokens, held, positions, mask)` a lane; every lane names the pool's pages.
    let mut fire = |lanes: &[(u32, &[u32], u32, &[u32], Option<&Masking>)]| {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, held, positions, mask)| Seated {
                pages: &pages,
                held: Some(held),
                positions,
                mask,
                ..Seated::of(Lane {
                    slot,
                    word: word(tokens.len() as u32, mask.is_some()),
                    tokens,
                })
            })
            .collect();
        let want: Vec<Readout> = lanes.iter().map(|_| Readout::Last).collect();
        shell.fire_rows(&seated, &want)
    };

    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    fire(&[(0, &prompt, 0, &[], None)]).expect("the prompt prefills");
    let own = Masking::Extent(Mask::new(vec![0, PROMPT + 1], u64::from(PROMPT + 1)));
    let alone = fire(&[(0, &[OTHER], PROMPT, &[], Some(&own))]).expect("a beam alone fires");

    // Beam 1's mask: the prompt and its own slot 21, not its sibling's 20.
    let ancestry = Masking::Extent(Mask::new(vec![0, PROMPT, 1, 1], u64::from(PROMPT + 2)));
    let beams = fire(&[
        (0, &[TOKEN], PROMPT, &[], Some(&own)),
        (1, &[OTHER], PROMPT + 1, &depth, Some(&ancestry)),
    ])
    .expect("the beams fire");
    let hidden = worst(&alone[0], &beams[1]);
    let sibling = Masking::Extent(Mask::new(vec![0, PROMPT + 2], u64::from(PROMPT + 2)));
    let seen = fire(&[
        (0, &[TOKEN], PROMPT, &[], Some(&own)),
        (1, &[OTHER], PROMPT + 1, &depth, Some(&sibling)),
    ])
    .expect("the beams fire");
    let admitted = worst(&alone[0], &seen[1]);
    eprintln!("hidden {hidden} admitted {admitted}");
    assert!(
        hidden <= 1e-5,
        "hiding its sibling, a beam answers what its token answers alone ({hidden})"
    );
    assert!(
        admitted > 1e-4,
        "admitting the sibling written in the same fire, a beam reads it ({admitted})"
    );
}
