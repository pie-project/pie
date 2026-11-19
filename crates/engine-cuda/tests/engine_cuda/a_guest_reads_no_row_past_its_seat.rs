//! A guest that reads more rows of an output than the fire seated for it is
//! refused by the output's name, and reads nothing: not the row after its
//! lane's (another lane's, here a sentinel), nor a row-pointer table's slot
//! no seat wrote.

use engine::{ChannelId, ProgramRegistration};
use engine_cuda::device::{Buffer, Context};
use engine_cuda::ProgramPlane;
use engine_cuda::program::launch::INTRINSIC_STORAGE_F32;
use engine_cuda::program::session::Fired;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype, Shape};

const WIDTH: u32 = 8;
const SENTINEL: f32 = -12_345.0;

/// A program whose epilogue puts `rows` rows of `logits`, read at lane 0,
/// in its one channel; its instance bound on `plane`.
fn bound(context: &Context, plane: &mut ProgramPlane, rows: u32) -> engine::InstanceId {
    let shape = Shape::matrix(rows, WIDTH);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(Dtype::F32),
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
                    shape,
                    dtype: Dtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let decl = OutputDecl {
        name: "logits".to_string(),
        axis: Axis::Row,
        width: WIDTH,
        dtype: Dtype::F32,
        layers: 0,
        readout: false,
    };
    let bound = eta_ir::validate::bind(container, vec![vec![decl]], EngineCaps::default())
        .expect("the epilogue binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let registration = ProgramRegistration {
        program_hash: 0x5e47 + u64::from(rows),
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let program = plane
        .register(context, &registration)
        .expect("the program compiles for the device");
    let extents = Extents {
        row_count: rows,
        sampled_rows: rows,
        query_len: rows,
        ..Extents::default()
    };
    plane
        .bind(
            program,
            &[],
            extents,
            GeometryClass::Host,
            eta_ir::registry::PortMask::NONE,
            false,
            &[None],
            &[ChannelId(1)],
            crate::guest::alone_slots(plane, program),
            &crate::guest::alone_span,
            context.stream(),
        )
        .expect("the instance binds")
}

/// The channel's cell, if the fire put one.
fn taken(plane: &mut ProgramPlane, instance: engine::InstanceId) -> Option<Vec<f32>> {
    let cell = plane
        .instance_mut(instance)
        .expect("the instance")
        .take(0)
        .expect("a channel reads")?;
    Some(
        cell.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// Rows of `WIDTH`: row `r` holds `r * 100 + column`, and the last row is
/// the sentinel throughout.
fn plane_rows(rows: u32) -> (Buffer, Vec<f32>) {
    let mut values: Vec<f32> = (0..(rows - 1) * WIDTH)
        .map(|i| (i / WIDTH * 100 + i % WIDTH) as f32)
        .collect();
    values.extend(std::iter::repeat_n(SENTINEL, WIDTH as usize));
    let mut buffer = Buffer::zeroed(values.len() * 4).expect("a logits plane");
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    buffer.write(0, &bytes).expect("the plane uploads");
    (buffer, values)
}

#[test]
fn a_guest_reads_no_row_past_its_seat() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let context = Context::bind(0, None).expect("a CUDA context");

    // A lane's own row, and its neighbour's after it: a program that reads
    // two rows from a seat of one is refused, and puts nothing.
    let mut plane = crate::guest::alone_plane();
    let instance = bound(&context, &mut plane, 2);
    let (buffer, values) = plane_rows(2);
    let seat = |plane: &mut ProgramPlane, rows: u32| {
        plane
            .bind_output(
                instance,
                "logits",
                0,
                buffer.ptr(),
                INTRINSIC_STORAGE_F32,
                WIDTH,
                WIDTH,
                0,
                rows,
            )
            .expect("the logits bind");
    };
    seat(&mut plane, 1);
    let fired = plane.fire(&context, instance).expect("the fire runs");
    let Fired::Faulted(why) = &fired else {
        panic!("a read past the seat's rows fired: {fired:?}");
    };
    assert!(why.contains("`logits`"), "{why}");
    assert_eq!(taken(&mut plane, instance), None, "a refused fire put a cell");
    // Seated with both rows, the same program reads both: the sentinel is
    // what the refused fire would have read.
    seat(&mut plane, 2);
    let fired = plane.fire(&context, instance).expect("the fire runs");
    assert!(matches!(fired, Fired::Committed), "{fired:?}");
    assert_eq!(taken(&mut plane, instance), Some(values));

    // Read a row a lane: three rows of an output seated for two lanes leave
    // the row table's third slot unwritten, and the fire is refused before
    // any kernel reads it.
    let mut plane = crate::guest::alone_plane();
    let instance = bound(&context, &mut plane, 3);
    let (buffer, values) = plane_rows(3);
    let seat = |plane: &mut ProgramPlane, lane: u32| {
        plane
            .bind_output(
                instance,
                "logits",
                lane,
                buffer.ptr() + u64::from(lane * WIDTH) * 4,
                INTRINSIC_STORAGE_F32,
                WIDTH,
                WIDTH,
                0,
                1,
            )
            .expect("the logits bind");
    };
    seat(&mut plane, 0);
    seat(&mut plane, 1);
    let fired = plane.fire(&context, instance).expect("the fire runs");
    let Fired::Faulted(why) = &fired else {
        panic!("a read past the seated lanes fired: {fired:?}");
    };
    assert!(why.contains("`logits`"), "{why}");
    assert_eq!(taken(&mut plane, instance), None, "a refused fire put a cell");
    seat(&mut plane, 2);
    let fired = plane.fire(&context, instance).expect("the fire runs");
    assert!(matches!(fired, Fired::Committed), "{fired:?}");
    assert_eq!(taken(&mut plane, instance), Some(values));
}
