//! bf16 values on the SIMT backends (CUDA, HIP): stored two bytes an
//! element, each rounded once to nearest even as the IR rounds it, computed
//! and reduced in f32.

use crate::msl_corpus::Served;
use eta_compiler::codegen::cuda::emit_fused_region;
use eta_compiler::codegen::simt::runtime::runtime_source;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, Shape};

const ROWS: u32 = 4;
const VOCAB: u32 = 4096;

fn out(shape: Shape, dtype: Dtype) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

/// The logits cast to bf16, scaled by an f32 constant, exponentiated (both
/// bf16), the exponentials landed and summed (f32).
fn subject() -> TraceContainer {
    let row = Shape::matrix(ROWS, VOCAB);
    let ops = vec![
        Op::Output {
            index: 0,
            lane: 0,
            shape: row,
            dtype: Dtype::F32,
        }, // 0 f32
        Op::Cast {
            value: 0,
            dtype: Dtype::Bf16,
        }, // 1 bf16
        Op::Const(Literal::F32(0.5)), // 2 f32
        Op::Mul(1, 2),                // 3 bf16
        Op::Exp(3),                   // 4 bf16
        Op::ReduceSum(4),             // 5 f32
        Op::ChanPut { chan: 0, value: 4 },
        Op::ChanPut { chan: 1, value: 5 },
    ];
    TraceContainer {
        names: vec!["logits".into()],
        channels: vec![out(row, Dtype::Bf16), out(Shape::vector(ROWS), Dtype::F32)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn bf16_emits_two_byte_storage_every_case() {
    a_bf16_stream_stores_halves_and_computes_in_f32();
    the_runtime_rounds_as_the_ir_does();
    a_sampler_reads_the_output_row_in_place();
}

/// A decode lane's sampler shape: the output's one row reshaped to the
/// vocabulary, cast to bf16, exponentiated and summed. The row is read
/// where the model left it, never copied into scratch.
fn a_sampler_reads_the_output_row_in_place() {
    let ops = vec![
        Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        }, // 0
        Op::Reshape {
            value: 0,
            shape: Shape::vector(VOCAB),
        }, // 1 a view of 0
        Op::Cast {
            value: 1,
            dtype: Dtype::Bf16,
        }, // 2
        Op::Exp(2),       // 3
        Op::ReduceSum(3), // 4
        Op::Reshape {
            value: 4,
            shape: Shape::vector(1),
        }, // 5
        Op::ChanPut { chan: 0, value: 5 },
    ];
    let program = TraceContainer {
        names: vec!["logits".into()],
        channels: vec![out(Shape::vector(1), Dtype::F32)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = Served::of(VOCAB).bind(program).expect("the sampler binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        source.push_str(&emit_fused_region(&format!("k{index}"), stage, region).expect("emits"));
    }
    let kernel = source.rsplit("extern \"C\"").next().unwrap_or("");
    assert!(
        kernel.contains("m1_intrinsic_row_load"),
        "the stream reads the output row"
    );
    assert!(
        !kernel.contains("ptir_parallel_intrinsic("),
        "the output row is copied into scratch:\n{kernel}"
    );
    for value in [0u32, 1, 2, 3] {
        assert!(
            !kernel.contains(&format!("offsets[{value}]")),
            "value {value} touches scratch:\n{kernel}"
        );
    }
}

fn a_bf16_stream_stores_halves_and_computes_in_f32() {
    let bound = Served::of(VOCAB)
        .bind(subject())
        .expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let mut source = String::new();
    for (index, region) in stage.fused.regions.iter().enumerate() {
        if region.row_value.is_none() {
            continue;
        }
        let emitted = emit_fused_region(&format!("k{index}"), stage, region).expect("cuda emits");
        source.push_str(
            emitted
                .split("descriptors = ptir_rowdesc;")
                .nth(1)
                .expect("a row-parallel kernel"),
        );
    }
    assert!(source.contains("// stream of "), "the bf16 chain streams");
    // Every bf16 result is rounded once where it is computed, and held as
    // the f32 its store keeps.
    for value in [1u32, 3, 4] {
        assert!(
            source.contains(&format!("const float r{value} = m1_round_bf16(")),
            "bf16 value {value} is not computed in f32 and rounded once:\n{source}"
        );
    }
    // The landed exponentials: two bytes an element, four a vector store
    // (the two words they fill), never an f32 store.
    let exps = "scratch + offsets[4]";
    assert!(source.contains(&format!("m1_store4_h({exps}")), "{source}");
    assert!(source.contains(&format!("m1_store_h({exps}")), "{source}");
    for wide in ["m1_store4_f(", "m1_store_f("] {
        assert!(
            !source.contains(&format!("{wide}{exps}")),
            "the bf16 value is stored as f32 ({wide})"
        );
    }
    // The sum reduces in f32.
    assert!(source.contains("const float red"), "{source}");
    assert!(source.contains("ptir_canonical_reduce(48u"), "{source}");
}

fn the_runtime_rounds_as_the_ir_does() {
    let runtime = runtime_source(eta_compiler::codegen::cuda::DIALECT);
    // The one bf16 store is a 16-bit store of the element's own two bytes,
    // not a read-modify-write of the word around it.
    assert!(
        runtime.contains("reinterpret_cast<m1_u16*>(data)[index] = (m1_u16)m1_bf16_bits(value);")
    );
    assert!(runtime.contains("#define M1_DT_BF16 5u"));
    assert_eq!(eta_ir::types::WIRE_BF16, 5);
    // The rounding, transcribed: it must be eta_ir's.
    assert!(
        runtime
            .contains("if ((b & 0x7fffffffu) > 0x7f800000u) return ((b >> 16) | 0x40u) & 0xffffu;")
    );
    assert!(runtime.contains("return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;"));
    fn simt_round(x: f32) -> u32 {
        let b = x.to_bits();
        if (b & 0x7fff_ffff) > 0x7f80_0000 {
            return ((b >> 16) | 0x40) & 0xffff;
        }
        b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16
    }
    let mut x = 0x0001_2345u32;
    for _ in 0..100_000 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let f = f32::from_bits(x);
        assert_eq!(
            simt_round(f),
            u32::from(eta_ir::types::bf16_bits(f)),
            "{x:#010x}"
        );
    }
}
