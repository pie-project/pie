#![cfg(feature = "cuda")]

mod common_dit;

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use common_dit::{HEAD_DIM, Lcg, NAME, SM_SCALE, WIDTH, Weights, assert_close, bf};
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{
    Attachment, Boundary, FrameSubmission, Lane, PortFeed, PortKind, Readout, Step,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{BindExtents, InstanceBinding, ProgramRegistration};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, RaggedMask, Trace, Value, Weight, ops, seam,
    trace_hybrid,
};

const HEADS: u32 = 4;
const RANKS: usize = 4;

struct Sharded {
    tp: u32,
}

impl ForwardHybrid for Sharded {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let mine = u64::from(HEADS * HEAD_DIM / self.tp);
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let h = ops::elemwise::layernorm_no_scale(&x, 1e-6);
        let heads = |name: &str| Weight::sym(name, [mine, u64::from(WIDTH)], Dtype::Bf16).columns();
        let q = ops::linear::matmul(&h, &heads("q"));
        let k = ops::linear::matmul(&h, &heads("k"));
        let v = ops::linear::matmul(&h, &heads("v"));
        let perm = inputs.row_permutation();
        let indptr = inputs.lane_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::None,
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let y = ops::linear::matmul(
            &o,
            &Weight::sym("o", [u64::from(WIDTH), mine], Dtype::Bf16).rows(),
        );
        let y = if self.tp > 1 {
            ops::collective::all_reduce(&y)
        } else {
            y
        };
        let out = ops::elemwise::residual_add(&x, &y);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

fn plan(tp: u32) -> Trace {
    let name = if tp > 1 {
        format!("{NAME}-tp{tp}")
    } else {
        NAME.to_string()
    };
    trace_hybrid(&name, &Sharded { tp }, Platform::Cuda)
}

fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let tp = trace
        .name
        .rsplit_once("-tp")
        .and_then(|(_, width)| width.parse::<u32>().ok())
        .unwrap_or(1);
    let source = ztensor_compat::index(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, tp, Platform::Cuda)
        .map_err(|why| why.to_string())
}

fn reference(weights: &Weights, x: &[f32], rows: usize) -> Vec<f32> {
    let w = WIDTH as usize;
    let hd = HEAD_DIM as usize;
    let full = (HEADS * HEAD_DIM) as usize;
    let mut h = vec![0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..(r + 1) * w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for c in 0..w {
            h[r * w + c] = bf((row[c] - mean) * inv);
        }
    }
    let project = |name: &str| -> Vec<f32> {
        let wt = weights.get(name);
        let mut y = vec![0f32; rows * full];
        for r in 0..rows {
            for c in 0..full {
                let mut acc = 0f32;
                for i in 0..w {
                    acc = h[r * w + i].mul_add(wt[c * w + i], acc);
                }
                y[r * full + c] = bf(acc);
            }
        }
        y
    };
    let (q, k, v) = (project("q"), project("k"), project("v"));
    let mut o = vec![0f32; rows * full];
    for head in 0..HEADS as usize {
        for i in 0..rows {
            let scores: Vec<f32> = (0..rows)
                .map(|j| {
                    (0..hd)
                        .map(|d| q[i * full + head * hd + d] * k[j * full + head * hd + d])
                        .sum::<f32>()
                        * SM_SCALE
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ws: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let mass: f32 = ws.iter().sum();
            for d in 0..hd {
                let mut acc = 0f32;
                for j in 0..rows {
                    acc += bf(ws[j] / mass) * v[j * full + head * hd + d];
                }
                o[i * full + head * hd + d] = bf(acc);
            }
        }
    }
    let wo = weights.get("o");
    let mut out = vec![0f32; rows * w];
    for r in 0..rows {
        for c in 0..w {
            let mut acc = 0f32;
            for i in 0..full {
                acc = o[r * full + i].mul_add(wo[c * full + i], acc);
            }
            out[r * w + c] = bf(bf(acc) + x[r * w + c]);
        }
    }
    out
}

fn epilogue(rows: u32) -> TraceContainer {
    let decl = |shape: Shape, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::matrix(rows, WIDTH), HostRole::Writer),
            decl(Shape::matrix(rows, WIDTH), HostRole::Reader),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::IntrinsicVal {
                    intr: IntrinsicId::Velocity,
                    shape: Shape::matrix(rows, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[test]
fn four_ranks_land_what_one_rank_lands() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let devices = engine_cuda::device::count();
    if devices < RANKS {
        eprintln!("{devices} device(s) visible; this group is {RANKS}: skipping");
        return;
    }
    let tp = RANKS as u32;
    let plan = plan(tp);
    let weights = Weights::random(&super_plan(), 53);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = weights.write(dir.path());

    let boots: Vec<engine_cuda::DeviceBoot> = (0..RANKS as i32)
        .map(|ordinal| engine_cuda::DeviceBoot {
            ordinal,
            ..engine_cuda::DeviceBoot::default()
        })
        .collect();
    let mut group = engine_cuda::open_group(boots, contract_for).expect("the group opens");
    let loaded = group
        .load(LoadRequest {
            trace: plan,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 8,
                max_tokens: 64,
                buckets: vec![16, 64],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                slots: 8,
                pages: 64,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency::default(),
            ordinal: -1,
            frames_in_flight: 1,
        })
        .expect("the sharded plan loads on four ranks");
    assert!(loaded.caps.profile.has_velocity);

    let rows = [5u32, 11];
    let mut rng = Lcg::seeded(59);
    let w = WIDTH as usize;
    let inputs: Vec<Vec<f32>> = rows
        .iter()
        .map(|&n| (0..n as usize * w).map(|_| bf(rng.unit())).collect())
        .collect();
    let want: Vec<Vec<f32>> = rows
        .iter()
        .zip(&inputs)
        .map(|(&n, x)| reference(&weights, x, n as usize))
        .collect();
    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    let mut instances = Vec::new();
    let mut next_channel = 1u64;
    for (at, &n) in rows.iter().enumerate() {
        let bound = bind(epilogue(n), loaded.caps.profile.clone()).expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend = Backend::parse("cuda").expect("the cuda backend");
        let program = group
            .register_program(&ProgramRegistration {
                program_hash: 0x4a4a ^ u64::from(n),
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(
                    &stages,
                ),
                launch,
                ..Default::default()
            })
            .expect("the program registers on every rank");
        let mut channel = |host_role: HostRole| {
            let id = next_channel;
            next_channel += 1;
            group
                .register_channel(&ChannelRegistration {
                    id,
                    shape: vec![n, WIDTH],
                    dtype: ChanDType::Concrete(EtaDtype::F32),
                    host_role,
                    seeded: false,
                    extern_dir: None,
                    capacity: 2,
                    extern_name: Vec::new(),
                })
                .expect("the channel registers");
            id
        };
        let latent = channel(HostRole::Writer);
        let out = channel(HostRole::Reader);
        let instance = group
            .bind_instance(&InstanceBinding {
                program,
                channels: vec![latent, out],
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: BindExtents {
                    row_count: n,
                    token_count: n,
                    sampled_rows: n,
                    query_len: n,
                    ..BindExtents::default()
                },
            })
            .expect("the instance binds on every rank")
            .id;
        let bytes: Vec<u8> = inputs[at].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(
            group
                .publish_channel(instance, 0, &bytes)
                .expect("the cell publishes")
        );
        lanes.push(Lane {
            slot: at as u32,
            word: 0,
            tokens: vec![0; n as usize],
            readout: Readout::Rows((0..n).collect()),
            ports: vec![PortFeed {
                kind: PortKind::Latents,
                port: 0,
                channel: latent,
            }],
            ..Lane::default()
        });
        attachments.push(Attachment {
            lane: at as u32,
            instance,
            at: Boundary::Epilogue,
        });
        instances.push(instance);
    }
    let mut ticket = group
        .submit(&FrameSubmission::of(Step {
            lanes,
            attachments,
            media: Vec::new(),
            voxels: Vec::new(),
        }))
        .expect("the frame fires on four ranks");
    group.settle_frame(&mut ticket).expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 2);
    for (at, readout) in readouts.iter().enumerate() {
        assert_eq!(readout.width, WIDTH);
        assert_close(
            &readout.values,
            &want[at],
            &format!("lane {at} velocity, rank 0's readout"),
        );
        let worst = readout
            .values
            .iter()
            .zip(&want[at])
            .map(|(got, want)| (got - want).abs() / want.abs().max(1.0))
            .fold(0f32, f32::max);
        eprintln!("lane {at}: worst relative deviation from the host reference {worst:.2e}");
        assert!(
            worst <= 8.0e-3,
            "lane {at}: the four-rank answer is {worst:.2e} off the host's bf16 walk; \
             one bf16 all-reduce over four partials does not cost that much"
        );
    }
    for (at, &instance) in instances.iter().enumerate() {
        let bytes = group
            .take_channel(instance, 1)
            .expect("the channel reads")
            .expect("the channel holds a cell");
        let got: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_close(
            &got,
            &want[at],
            &format!("lane {at} velocity, the epilogue's intrinsic"),
        );
    }
    let _ = BTreeMap::<u64, u64>::new();
}

fn super_plan() -> Trace {
    plan(1)
}
