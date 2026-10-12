use eta_ir::op::{OP_TABLE, tags};

const GENERATED_TAGS: &[u8] = &[
    tags::EXP,
    tags::LOG,
    tags::NEG,
    tags::RECIP,
    tags::SIN,
    tags::COS,
    tags::SQRT,
    tags::RSQRT,
    tags::ABS,
    tags::SIGN,
    tags::CAST,
    tags::ADD,
    tags::SUB,
    tags::MUL,
    tags::DIV,
    tags::MAX_ELEM,
    tags::MIN_ELEM,
    tags::GT,
    tags::GE,
    tags::EQ,
    tags::NE,
    tags::LT,
    tags::LE,
    tags::AND,
    tags::OR,
    tags::NOT,
    tags::REM,
    tags::SELECT,
    tags::REDUCE_SUM,
    tags::REDUCE_MAX,
    tags::REDUCE_MIN,
    tags::REDUCE_ARGMAX,
    tags::BROADCAST,
    tags::RESHAPE,
    tags::TRANSPOSE,
    tags::PIVOT_THRESHOLD,
    tags::GATHER,
    tags::GATHER_ROW,
    tags::SCATTER_ADD,
    tags::SCATTER_SET,
    tags::IOTA,
    tags::MASK_APPLY_PACKED,
    tags::CAUSAL_MASK,
    tags::SLIDING_WINDOW_MASK,
    tags::SINK_WINDOW_MASK,
    tags::RNG,
    tags::RNG_KEYED,
    tags::CONST,
    tags::CHAN_TAKE,
    tags::CHAN_READ,
    tags::CHAN_PUT,
    tags::INTRINSIC_VAL,
];

#[test]
fn every_op_is_classified() {
    let mut unclassified: Vec<&str> = Vec::new();
    let mut both: Vec<&str> = Vec::new();
    for row in OP_TABLE {
        let library = eta_compiler::plan::library_op_for_tag(row.tag).is_some();
        let generated = GENERATED_TAGS.contains(&row.tag);
        if library && generated {
            both.push(row.name);
        } else if !library && !generated {
            unclassified.push(row.name);
        }
    }
    assert!(
        unclassified.is_empty(),
        "these ops are in neither partition, so the planner will emit them \
         inline by default: {unclassified:?} -- add each to \
         `library_op_for_tag` or to GENERATED_TAGS above"
    );
    assert!(both.is_empty(), "ops claimed by both partitions: {both:?}");
}
