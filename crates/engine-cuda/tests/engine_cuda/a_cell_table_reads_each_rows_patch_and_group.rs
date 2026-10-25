#![cfg(feature = "device")]

//! `conv::cells`, `gather::group_rows` and `gather::group_first` through the
//! engine against a host reference: two requests, each a text lane of plain
//! rows and an image lane of one clip patched `[1, 2, 2]`, read back off the
//! velocity seam.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step, StepVoxels};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use poem::fact;
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, ops, seam};
use poem_ir::InputAxis;

use crate::common_two_axis::{Weights, contract_for};
use engine::fire::Attachment;
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{EngineCaps, GeometryClass, Graph, ReadoutKind, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;

const C: u32 = 4;
const P: [u32; 3] = [1, 2, 2];
/// `[cells axis 0 (3 + size) | cells axis 3 (4 + size) | group rows | the
/// group's image cell (3 + size)]`.
const WIDTH: u32 = 6 + 7 + 1 + 6;

struct Cells;

impl ForwardHybrid for Cells {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        inputs
            .recorder()
            .declare_switch("stream", &["text", "image"]);
        let img = inputs.on(!fact::switch_is("stream", "text"));
        // The voxel input, patched and back out as pixels: what stages the
        // clips. Every lane's velocity is read by its epilogue.
        let grid = img.grid();
        let latents = img.input("latents", Dtype::Bf16, InputAxis::Voxel, C, None);
        let patched = ops::conv::patchify(&latents, &grid, P, &img.token_grid(P));
        let tgrid = inputs.token_grid(P);
        let positions = inputs.positions();
        let first = ops::conv::cells(&tgrid, &positions, 0);
        let fourth = ops::conv::cells(&tgrid, &positions, 3);
        let groups = ops::gather::group_rows(
            &inputs.group_indptr(),
            &inputs.group_of_lane(),
            &inputs.request_of_token(),
        );
        let own = ops::conv::cells(&img.token_grid(P), &img.positions(), 0);
        let peer = ops::gather::group_first(
            &ops::gather::pack_rows(&own, &img.row_permutation()),
            &img.group_indptr(),
            &inputs.group_of_lane(),
            &inputs.request_of_token(),
        );
        let out = ops::gather::concat_cols(&[&first, &fourth, &groups, &peer]);
        seam::at(seam::VELOCITY, &[&out]);
        let back = ops::conv::unpatchify(&patched, &img.token_grid(P), P, &grid);
        let (pixels, pixel_grid) = ops::conv::upsample_nearest(&back, &grid, [1, 1, 1], false);
        seam::at(seam::PIXELS, &[&pixels, &pixel_grid]);
        out
    }
}

/// An epilogue putting the lane's `rows` velocity rows on its one channel.
fn epilogue(rows: u32) -> TraceContainer {
    let plane = Shape::matrix(rows, WIDTH);
    TraceContainer {
        names: Vec::new(),
        channels: vec![ChannelDecl {
            shape: plane,
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Velocity,
                    lane: 0,
                    shape: plane,
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The token grid of `clip`.
fn grid_of(clip: [u32; 3]) -> [u32; 3] {
    [clip[0] / P[0], clip[1] / P[1], clip[2] / P[2]]
}

/// One row's want: its cell and clip size (a clip's row) or position, its
/// group's rows and its group's image's first cell and size.
fn want_row(
    cell: Option<([u32; 3], [u32; 3])>,
    position: u32,
    group_rows: u32,
    peer: [u32; 3],
) -> Vec<f32> {
    let mut row = vec![0f32; WIDTH as usize];
    match cell {
        Some(([t, h, w], size)) => {
            for (at, v) in [t, h, w].into_iter().enumerate() {
                row[at] = v as f32;
                row[6 + at] = v as f32;
            }
            for (at, v) in size.into_iter().enumerate() {
                row[3 + at] = v as f32;
                row[6 + 4 + at] = v as f32;
            }
        }
        None => {
            row[0] = position as f32;
            row[6 + 3] = position as f32;
        }
    }
    row[13] = group_rows as f32;
    for (at, v) in peer.into_iter().enumerate() {
        row[14 + 3 + at] = v as f32;
    }
    row
}

fn image_want(clip: [u32; 3], group_rows: u32) -> Vec<f32> {
    let size = grid_of(clip);
    let mut out = Vec::new();
    for a in 0..size[0] {
        for b in 0..size[1] {
            for c in 0..size[2] {
                out.extend(want_row(Some(([a, b, c], size)), 0, group_rows, size));
            }
        }
    }
    out
}

fn text_want(rows: u32, group_rows: u32, peer: [u32; 3]) -> Vec<f32> {
    (0..rows)
        .flat_map(|j| want_row(None, j, group_rows, peer))
        .collect()
}

#[test]
fn a_cell_table_reads_each_rows_patch_and_group() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid(crate::common_two_axis::NAME, &Cells, Platform::Cuda);
    let facts = trace.facts.clone();
    let weights = Weights::random(&trace, 1);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = weights.write(dir.path());
    let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for)
        .expect("the engine opens");
    engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 64,
                buckets: vec![32, 64],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: Some(128),
                max_clips: Some(4),
            },
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the cell plan loads");

    let clips = [[2u32, 4, 6], [1, 2, 4]];
    let text_rows = [3u32, 5];
    let tokens = |clip: [u32; 3]| clip[0] * clip[1] * clip[2] / (P[0] * P[1] * P[2]);
    let word =
        |stream: &str, rows: u32| facts.word(&Request::new(rows, false).switched("stream", stream));
    let text = |slot: u32, group: u32| Lane {
        slot,
        word: word("text", text_rows[group as usize]),
        tokens: vec![0; text_rows[group as usize] as usize],
        readout: Readout::Rows((0..text_rows[group as usize]).collect()),
        group: Some(group),
        ..Lane::default()
    };
    let image = |slot: u32, group: u32| {
        let rows = tokens(clips[group as usize]);
        Lane {
            slot,
            word: word("image", rows),
            tokens: vec![0; rows as usize],
            readout: Readout::Rows((0..rows).collect()),
            group: Some(group),
            ..Lane::default()
        }
    };
    // Request 0 text first, request 1 image first.
    let lanes = vec![text(0, 0), image(1, 0), image(2, 1), text(3, 1)];
    let voxels = vec![
        StepVoxels {
            lane: 1,
            clips: vec![clips[0]],
            payload: vec![0.5; (2 * 4 * 6 * C) as usize],
        },
        StepVoxels {
            lane: 2,
            clips: vec![clips[1]],
            payload: vec![0.25; (2 * 4 * C) as usize],
        },
    ];
    let mut next = 1u64;
    let mut readers = Vec::new();
    let mut attachments = Vec::new();
    for (at, lane) in lanes.iter().enumerate() {
        let rows = lane.tokens.len() as u32;
        let velocity = eta_ir::registry::Readout {
            kind: ReadoutKind::Velocity,
            width: WIDTH,
        };
        let bound = bind(
            epilogue(rows),
            vec![Some(velocity)],
            Graph::default(),
            EngineCaps::default(),
        )
        .expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend =
            Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
        let program = engine
            .register_program(&ProgramRegistration {
                program_hash: 0xce11 ^ u64::from(rows),
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(
                    &stages,
                ),
                launch,
                ..Default::default()
            })
            .expect("the program registers");
        let channel = engine::ChannelId(next);
        next += 1;
        engine
            .register_channel(&engine::channel::ChannelRegistration {
                id: channel,
                shape: vec![rows, WIDTH],
                dtype: ChanDType::Concrete(EtaDtype::F32),
                host_role: HostRole::Reader,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
            })
            .expect("the channel registers");
        let instance = engine
            .bind_instance(&InstanceBinding {
                program,
                channels: vec![channel],
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: engine::program::BindExtents {
                    row_count: rows,
                    token_count: rows,
                    sampled_rows: rows,
                    query_len: rows,
                    ..engine::program::BindExtents::default()
                },
            })
            .expect("the instance binds")
            .id;
        attachments.push(Attachment {
            lane: at as u32,
            lanes: 1,
            instance,
            adapter_only: false,
        });
        readers.push(instance);
    }
    let mut ticket = engine
        .submit(&FrameSubmission::of(Step {
            lanes,
            attachments,
            media: Vec::new(),
            voxels,
        }))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let got: Vec<Vec<f32>> = readers
        .iter()
        .map(|&instance| {
            engine
                .take_channel(instance, 0)
                .expect("the channel reads")
                .expect("the channel holds a cell")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect();

    let group = |g: usize| text_rows[g] + tokens(clips[g]);
    let want = [
        text_want(text_rows[0], group(0), grid_of(clips[0])),
        image_want(clips[0], group(0)),
        image_want(clips[1], group(1)),
        text_want(text_rows[1], group(1), grid_of(clips[1])),
    ];
    let mut worst = 0f32;
    for (lane, (values, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(values.len(), want.len(), "lane {lane}: rows x width");
        for (i, (g, w)) in values.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            worst = worst.max(err);
            assert!(
                err == 0.0,
                "lane {lane} row {} col {}: device {g}, host {w}",
                i / WIDTH as usize,
                i % WIDTH as usize
            );
        }
    }
    eprintln!("cells, group rows and group first: max |err| {worst}");
}
