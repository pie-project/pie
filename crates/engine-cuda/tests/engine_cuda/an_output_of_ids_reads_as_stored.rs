//! An i32 output (a draft head's ids) reads into the program as stored, from
//! the row it is bound at.

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::program::Plane;
use engine_cuda::program::compile::Disk;
use engine_cuda::program::launch::INTRINSIC_STORAGE_RAW_I32;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype, Shape};

const ROWS: u32 = 2;
const DEPTH: u32 = 3;

#[test]
fn an_output_of_ids_reads_as_stored() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let container = TraceContainer {
        names: vec!["mtp.drafts".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(ROWS, DEPTH),
            dtype: ChanDType::Concrete(Dtype::I32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(ROWS, DEPTH),
                    dtype: Dtype::I32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let decl = OutputDecl {
        name: "mtp.drafts".to_string(),
        axis: Axis::Row,
        width: DEPTH,
        dtype: Dtype::I32,
        layers: 0,
        readout: true,
    };
    let bound = eta_ir::validate::bind(container, vec![vec![decl]], EngineCaps::default())
        .expect("the epilogue binds");
    let context = Context::bind(0, None).expect("a CUDA context");
    let stages = compile_bound(&bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let registration = ProgramRegistration {
        program_hash: 0x1d5,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let mut plane = Plane::new(Disk::disabled());
    let program = plane
        .register(&context, &registration)
        .expect("the program compiles for the device");
    let extents = Extents {
        row_count: ROWS,
        sampled_rows: ROWS,
        query_len: ROWS,
        ..Extents::default()
    };
    let instance = plane
        .bind(
            program,
            &[],
            extents,
            GeometryClass::Host,
            &[None],
            &[ChannelId(1)],
            context.stream(),
        )
        .expect("the instance binds");
    // Three rows of ids; the lane's rows start at the second.
    let stored: Vec<i32> = (0..(ROWS + 1) * DEPTH)
        .map(|i| 151_000 + i as i32)
        .collect();
    let mut buffer = Buffer::zeroed(stored.len() * 4).expect("a drafts plane");
    let bytes: Vec<u8> = stored.iter().flat_map(|x| x.to_le_bytes()).collect();
    buffer.write(0, &bytes).expect("the plane uploads");
    plane
        .bind_output(
            instance,
            "mtp.drafts",
            0,
            buffer.ptr(),
            INTRINSIC_STORAGE_RAW_I32,
            DEPTH,
            DEPTH,
            1,
        )
        .expect("the drafts bind");
    let fired = plane.fire(&context, instance).expect("the fire runs");
    assert!(matches!(fired, Fired::Committed), "{fired:?}");
    let cell = plane
        .instance_mut(instance)
        .expect("the instance")
        .take(0)
        .expect("a channel reads")
        .expect("the channel holds a cell");
    let got: Vec<i32> = cell
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    assert_eq!(got, stored[DEPTH as usize..].to_vec());
}
