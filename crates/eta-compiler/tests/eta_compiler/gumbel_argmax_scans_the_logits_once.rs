use crate::msl_corpus::Served;
use eta_compiler::codegen::cuda::emit_fused_region;
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, RngKind, Shape};

const VOCAB: u32 = 16;

fn channels() -> Vec<ChannelDecl> {
    vec![
        ChannelDecl {
            shape: Shape::new(&[]).expect("a scalar shape"),
            dtype: ChanDType::Concrete(Dtype::I32),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        },
        ChannelDecl {
            shape: Shape::vector(2),
            dtype: ChanDType::Concrete(Dtype::U32),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: false,
        },
    ]
}

fn subject(divide: bool, kind: RngKind) -> TraceContainer {
    let mut ops = vec![
        Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 0,
            shape: Shape::vector(VOCAB),
        },
        Op::ChanTake(1),
    ];
    let logits = if divide {
        ops.push(Op::Const(Literal::F32(0.7)));
        ops.push(Op::Div(1, 3));
        4
    } else {
        1
    };
    let noise = ops.len() as u32;
    ops.push(Op::RngKeyed {
        state: 2,
        shape: Shape::vector(VOCAB),
        kind,
    });
    let perturbed = ops.len() as u32;
    ops.push(Op::Add(logits, noise));
    let token = ops.len() as u32;
    ops.push(Op::ReduceArgmax(perturbed));
    ops.push(Op::ChanPut {
        chan: 0,
        value: token,
    });
    TraceContainer {
        names: vec!["logits".into()],
        channels: channels(),
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// Compat's decode epilogue at temperature > 0 with no cut, at Qwen's
/// vocabulary: `argmax(logits / T + gumbel(r))` put to the next fire's
/// input and to the host, and the random stream carried to the next fire.
fn decode_subject(divide: bool, vocab: u32) -> TraceContainer {
    let row = Shape::vector(vocab);
    let one = Shape::vector(1);
    let mut ops = vec![
        Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, vocab),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 0,
            shape: row,
        },
        Op::ChanTake(1),
    ];
    let logits = if divide {
        ops.push(Op::Const(Literal::F32(0.7)));
        ops.push(Op::Div(1, 3));
        4
    } else {
        1
    };
    let noise = ops.len() as u32;
    ops.push(Op::RngKeyed {
        state: 2,
        shape: row,
        kind: RngKind::Gumbel,
    });
    let perturbed = ops.len() as u32;
    ops.push(Op::Add(logits, noise));
    let token = ops.len() as u32;
    ops.push(Op::ReduceArgmax(perturbed));
    let shaped = ops.len() as u32;
    ops.push(Op::Reshape {
        value: token,
        shape: one,
    });
    ops.push(Op::ChanPut {
        chan: 2,
        value: shaped,
    });
    ops.push(Op::ChanPut {
        chan: 3,
        value: shaped,
    });
    // Value ids count results, and a put has none.
    let step = shaped + 1;
    ops.push(Op::Iota { len: 2 });
    let next = step + 1;
    ops.push(Op::Add(2, step));
    ops.push(Op::ChanPut {
        chan: 1,
        value: next,
    });
    let token_chan = |host_role, seeded| ChannelDecl {
        shape: one,
        dtype: ChanDType::Concrete(Dtype::I32),
        capacity: 4,
        host_role,
        seeded,
    };
    TraceContainer {
        names: vec!["logits".into()],
        channels: vec![
            channels().remove(0),
            // Seeded once by the host; the device carries it from there.
            ChannelDecl {
                shape: Shape::vector(2),
                dtype: ChanDType::Concrete(Dtype::U32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            token_chan(HostRole::None, true),
            token_chan(HostRole::Reader, false),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn emitted(container: TraceContainer) -> String {
    let served = Served::of(VOCAB);
    let bound = served.bind(container).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let region = stage.fused.regions.first().expect("one fused region");
    let source = emit_fused_region("gumbel", stage, region).expect("the CUDA emitter serves it");
    let entry = source
        .find("__global__ void gumbel")
        .expect("the entry point is in the source");
    source[entry..].to_string()
}

#[test]
fn gumbel_argmax_scans_the_logits_once_every_case() {
    a_scaled_gumbel_max_becomes_one_scan_of_the_intrinsic();
    an_unscaled_gumbel_max_folds_too_with_no_divisor();
    a_uniform_draw_is_not_a_gumbel_max_and_stays_as_traced();
    no_row_stream_takes_the_argmax_a_gumbel_scan_owns();
}

/// The region as each SIMT backend that serves it emits it, whichever kind
/// of region it compiles to.
fn emitted_by_each_backend(container: TraceContainer, vocab: u32) -> Vec<(&'static str, String)> {
    let served = Served::of(vocab);
    let bound = served.bind(container).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    let region = stage.fused.regions.first().expect("one fused region");
    vec![
        (
            "cuda",
            eta_compiler::codegen::cuda::emit_region("gumbel", stage, region)
                .expect("the CUDA emitter serves it"),
        ),
        (
            "hip",
            eta_compiler::codegen::hip::emit_region("gumbel", stage, region)
                .expect("the HIP emitter serves it"),
        ),
    ]
}

/// The draw a compat request at temperature > 0 with no top-k, top-p or
/// min-p cut makes: `argmax(logits / T + gumbel)`. The fused scan reads the
/// logits and draws the noise itself, so the ops between them are never
/// computed; a row stream that also took the argmax scanned that unwritten
/// row and answered index 0 every fire ("!!!!" on Qwen), on CUDA and HIP.
fn no_row_stream_takes_the_argmax_a_gumbel_scan_owns() {
    const QWEN: u32 = 151_936;
    let subjects = [true, false].into_iter().flat_map(|divide| {
        [
            (divide, subject(divide, RngKind::Gumbel), VOCAB),
            (divide, decode_subject(divide, QWEN), QWEN),
        ]
    });
    for (divide, container, vocab) in subjects {
        for (backend, source) in emitted_by_each_backend(container, vocab) {
            assert!(
                source.contains("ptir_fast_gumbel_argmax_intrinsic("),
                "{backend} (divide {divide}): the draw folds into the one scan"
            );
            assert!(
                !source.contains("stream_candidates"),
                "{backend} (divide {divide}): a row stream also argmaxes the draw's \
                 operand, which nothing computes"
            );
        }
    }
}

fn a_scaled_gumbel_max_becomes_one_scan_of_the_intrinsic() {
    let source = emitted(subject(true, RngKind::Gumbel));
    assert!(
        source.contains("ptir_fast_gumbel_argmax_intrinsic("),
        "the head must fold into the one-scan form"
    );
    assert!(
        !source.contains("ptir_fast_argmax("),
        "no scratch-reading argmax is left behind"
    );
    let head = source.find("ptir_fast_gumbel_argmax_intrinsic(").unwrap();
    let call = &source[head..head + 700];
    assert!(
        call.contains("descriptors[p.a0]"),
        "the operand shape sizes the scan"
    );
    assert!(!call.contains("nullptr"), "the divisor rides along");
}

fn an_unscaled_gumbel_max_folds_too_with_no_divisor() {
    let source = emitted(subject(false, RngKind::Gumbel));
    let head = source
        .find("ptir_fast_gumbel_argmax_intrinsic(")
        .expect("the head folds");
    assert!(
        source[head..head + 700].contains("nullptr"),
        "nothing to divide by"
    );
}

fn a_uniform_draw_is_not_a_gumbel_max_and_stays_as_traced() {
    let source = emitted(subject(true, RngKind::Uniform));
    assert!(!source.contains("ptir_fast_gumbel_argmax_intrinsic("));
    assert!(
        source.contains("ptir_fast_argmax("),
        "the traced argmax runs over scratch"
    );
}
