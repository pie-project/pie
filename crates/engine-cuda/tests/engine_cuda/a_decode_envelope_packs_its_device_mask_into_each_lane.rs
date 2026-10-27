#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::fire::{Mask, Masking, Readout};
use engine::runahead::Runahead;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const PROMPT: u32 = 20;
/// The pool every beam's mask spans: the prompt, then a slot a beam.
const KEYS: u32 = 32;
const MAX_BEAMS: u32 = 5;

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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, true, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, runahead: Runahead) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(MAX_BEAMS, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: MAX_BEAMS,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..MAX_BEAMS {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

const TOKENS: u32 = 0;
const MASK: u32 = 1;
const POSITIONS: u32 = 2;

/// A beam step's descriptor program over `beams` lanes: its tokens, its
/// `[beams, KEYS]` ancestry mask and (stated, never read unless the program
/// computes it on the device) positions, each re-put by its epilogue so the
/// next fire finds the same cell.
fn beam_step(beams: u32) -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 4,
        host_role: HostRole::None,
        seeded: true,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::vector(beams), EtaDtype::I32),
            decl(Shape::matrix(beams, KEYS), EtaDtype::Bool),
            decl(Shape::vector(beams), EtaDtype::U32),
        ],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(TOKENS),
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(POSITIONS),
            },
            PortBinding {
                port: Port::AttnMask,
                source: PortSource::Channel(MASK),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(TOKENS),
                Op::ChanPut {
                    chan: TOKENS,
                    value: 0,
                },
                Op::ChanTake(MASK),
                Op::ChanPut {
                    chan: MASK,
                    value: 1,
                },
                Op::ChanTake(POSITIONS),
                Op::ChanPut {
                    chan: POSITIONS,
                    value: 2,
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// Beam `i` writes slot `PROMPT + i` at depth `PROMPT` and sees the prompt
/// and its own cell, not its siblings'.
fn ancestry(beam: u32) -> Vec<bool> {
    (0..KEYS)
        .map(|key| key < PROMPT || key == PROMPT + beam)
        .collect()
}

fn token(beam: u32) -> u32 {
    (beam * 11 + 7) % VOCAB
}

fn bind(shell: &mut Shell, beams: u32, device: PortMask) -> engine::InstanceId {
    let bound = crate::bound::logits(beam_step(beams), VOCAB).expect("the beam step binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0xbea0 + u64::from(beams),
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let tokens: Vec<u8> = (0..beams)
        .flat_map(|beam| (token(beam) as i32).to_le_bytes())
        .collect();
    let cells: Vec<bool> = (0..beams).flat_map(ancestry).collect();
    // A bool cell rides the wire a bit a cell.
    let mut mask = vec![0u8; cells.len().div_ceil(8)];
    for (at, _) in cells.iter().enumerate().filter(|(_, kept)| **kept) {
        mask[at / 8] |= 1 << (at % 8);
    }
    // Positions no lane is roped at: read, they would change every answer.
    let positions: Vec<u8> = (0..beams).flat_map(|_| 999u32.to_le_bytes()).collect();
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (1..=count as u64).map(engine::ChannelId).collect();
    shell
        .bind_program(
            program,
            &[(TOKENS, tokens), (MASK, mask), (POSITIONS, positions)],
            Extents {
                row_count: beams,
                token_count: beams,
                sampled_rows: beams,
                query_len: beams,
                ..Extents::default()
            },
            GeometryClass::DecodeEnvelope,
            device,
            &vec![None; count],
            &ids,
        )
        .expect("the instance binds")
}

struct Pool {
    pages: Vec<u32>,
    depth: [u32; 1],
    tokens: Vec<[u32; 1]>,
    masks: Vec<Masking>,
}

impl Pool {
    fn new(beams: u32) -> Pool {
        Pool {
            pages: (0..64 / PAGE).collect(),
            depth: [PROMPT],
            tokens: (0..beams).map(|beam| [token(beam)]).collect(),
            masks: (0..beams)
                .map(|beam| engine::ports::from_dense(&ancestry(beam), KEYS as usize))
                .collect(),
        }
    }

    /// The beams of one fire, each in its own seat over the pool's pages,
    /// stating its own mask on the host or carrying none for the device's.
    fn lanes(&self, trace: &poem::Trace, host_masks: bool) -> Vec<Seated<'_>> {
        let word = trace.facts.word(&Request::new(1, true));
        (0..self.tokens.len())
            .map(|beam| Seated {
                pages: &self.pages,
                held: Some(PROMPT + beam as u32),
                positions: &self.depth,
                mask: host_masks.then(|| &self.masks[beam]),
                ..Seated::of(Lane {
                    slot: beam as u32,
                    word,
                    tokens: &self.tokens[beam],
                })
            })
            .collect()
    }
}

fn prefill(shell: &mut Shell, trace: &poem::Trace, pages: &[u32]) {
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    let seated = Seated {
        pages,
        held: Some(0),
        ..Seated::of(Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(PROMPT, false)),
            tokens: &prompt,
        })
    };
    shell
        .fire_rows(&[seated], &[Readout::Last])
        .expect("the prompt prefills");
}

fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max)
}

/// Beams sharing a pool carry their ancestry masks on the device, in a
/// decode envelope over several lanes: each lane's row of the mask cell is
/// packed into the fire's staged mask, staged ahead or not, and the fire
/// answers what the same lanes answer stating those masks on the host. The
/// positions the runtime states are the lanes'; the program's stale ones
/// are never read, and a program that computes them on the device beside
/// lanes that state them is refused.
#[test]
fn a_decode_envelope_packs_its_device_mask_into_each_lane() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("envelope-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 3307).write(dir.path());
    let device = PortMask::of(&[Port::EmbedTokens, Port::AttnMask]);
    // Two beams pack lane by lane; five through one gather over a run table.
    for beams in [2, MAX_BEAMS] {
        let pool = Pool::new(beams);
        for runahead in [Runahead::F1, Runahead::of(2)] {
            let mut shell = shell(trace.clone(), &path, runahead);
            prefill(&mut shell, &trace, &pool.pages);
            let want: Vec<Readout> = (0..beams).map(|_| Readout::Last).collect();
            let host = shell
                .fire_rows(&pool.lanes(&trace, true), &want)
                .expect("the host-masked beams fire");
            let instance = bind(&mut shell, beams, device);
            let attached = [Attached {
                lane: 0,
                lanes: beams,
                instance,
                adapter_only: false,
            }];
            for fire in 0..2 {
                let got = shell
                    .fire_media(&pool.lanes(&trace, false), &attached, &[])
                    .expect("the device-masked beams fire");
                let off = worst(&got, &host);
                assert!(
                    off <= 1e-5,
                    "{beams} beams, {runahead:?}, fire {fire}: the device's masks answer {off} \
                     away from the host's"
                );
            }
            let sibling = shell
                .fire_rows(
                    &Pool {
                        masks: (0..beams)
                            .map(|_| {
                                Masking::Extent(Mask::new(vec![0, PROMPT + beams], u64::from(KEYS)))
                            })
                            .collect(),
                        ..Pool::new(beams)
                    }
                    .lanes(&trace, true),
                    &want,
                )
                .expect("the beams seeing their siblings fire");
            assert!(
                worst(&sibling, &host) > 1e-4,
                "{beams} beams: a mask admitting the siblings answers otherwise"
            );
        }
    }

    let pool = Pool::new(2);
    let mut shell = shell(trace.clone(), &path, Runahead::F1);
    prefill(&mut shell, &trace, &pool.pages);
    let instance = bind(
        &mut shell,
        2,
        PortMask::of(&[Port::EmbedTokens, Port::AttnMask, Port::Positions]),
    );
    let attached = [Attached {
        lane: 0,
        lanes: 2,
        instance,
        adapter_only: false,
    }];
    let refused = shell
        .fire_media(&pool.lanes(&trace, false), &attached, &[])
        .expect_err("two sources for one port are refused");
    assert!(format!("{refused}").contains("one side"), "{refused}");
}
