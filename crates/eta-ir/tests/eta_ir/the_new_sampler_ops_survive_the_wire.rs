use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer, decode, encode,
};
use eta_ir::op::{Op, tags};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, RngKind, Shape};

fn container(ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![ChannelDecl {
            shape: Shape::vector(4),
            dtype: ChanDType::Concrete(Dtype::U32),
            capacity: 1,
            host_role: HostRole::Writer,
            seeded: true,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

#[test]
fn the_new_sampler_ops_survive_the_wire_every_case() {
    the_appended_tags_are_the_numbers_the_wire_froze();
    a_sampler_epilogue_of_new_ops_round_trips();
    an_unknown_rng_kind_is_refused_rather_than_read_as_uniform();
}

fn the_appended_tags_are_the_numbers_the_wire_froze() {
    for (tag, name) in [
        (tags::EXP, "exp"),
        (tags::LOG, "log"),
        (tags::NEG, "neg"),
        (tags::RECIP, "recip"),
        (tags::ABS, "abs"),
        (tags::SIGN, "sign"),
        (tags::CAST, "cast"),
        (tags::SIN, "sin"),
        (tags::COS, "cos"),
        (tags::SQRT, "sqrt"),
        (tags::RSQRT, "rsqrt"),
    ] {
        assert_eq!(
            eta_ir::op::spec(tag).map(|row| row.name),
            Some(name),
            "tag {tag:#04x} is not `{name}`"
        );
    }
    assert_eq!(
        [tags::SIN, tags::COS, tags::SQRT, tags::RSQRT],
        [8, 9, 10, 11]
    );
    let tags: Vec<u8> = eta_ir::op::OP_TABLE.iter().map(|row| row.tag).collect();
    let mut sorted = tags.clone();
    sorted.sort_unstable();
    assert_eq!(tags, sorted, "OP_TABLE is no longer sorted by tag");
}

fn a_sampler_epilogue_of_new_ops_round_trips() {
    let ops = vec![
        Op::ChanRead(0),
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Normal,
        },
        Op::Sin(1),
        Op::Cos(2),
        Op::Sqrt(3),
        Op::Rsqrt(4),
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Gumbel,
        },
        Op::RngKeyed {
            state: 0,
            shape: Shape::matrix(2, 4),
            kind: RngKind::Uniform,
        },
    ];
    let source = container(ops);
    let bytes = encode(&source);
    let back = decode(&bytes).expect("the container decodes");
    assert_eq!(back, source, "the epilogue did not survive the wire");

    let kinds: Vec<RngKind> = back.stages[0]
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::RngKeyed { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec![RngKind::Normal, RngKind::Gumbel, RngKind::Uniform]
    );
}

fn an_unknown_rng_kind_is_refused_rather_than_read_as_uniform() {
    let source = container(vec![
        Op::ChanRead(0),
        Op::RngKeyed {
            state: 0,
            shape: Shape::vector(4),
            kind: RngKind::Normal,
        },
    ]);
    let bytes = encode(&source);
    let kind_byte = RngKind::Normal as u8;
    let at = bytes
        .iter()
        .rposition(|&b| b == kind_byte)
        .expect("the kind byte is on the wire");
    let mut mutant = bytes.clone();
    mutant[at] = 0x7f;
    assert!(
        decode(&mutant).is_err(),
        "a kind byte no `RngKind` claims decoded anyway"
    );
}
