//! The Rust op table against the driver's C++ one.
//!
//! Tags are already safe: `op_table.hpp` gets them by `#include <ptir_abi.h>`,
//! and `ptir_header.rs` both regenerates that header from `OP_TABLE` and
//! forbids re-typing its constants. What no test covered is the metadata the
//! driver keeps *beside* the tags -- per-op family, launch class, arity, result
//! count -- and, more sharply, whether the driver has a row for each op at all.
//!
//! It matters because of how `op_info` fails. An `OpCode` with no `case` falls
//! through to
//!
//! ```text
//! return {c, OpFamily::Map, LaunchClass::Alias, ResultKind::Custom, 0xFF, 1, "?"};
//! ```
//!
//! so a newly added op does not make the driver complain -- it makes the driver
//! quietly treat it as a one-result aliasing map. That is the C1 failure mode
//! exactly: adding an op is a multi-repository edit, and the place that forgets
//! is silent about forgetting.
//!
//! This test reads the header as text rather than compiling it, because there
//! is no C++ toolchain in this crate's test environment. That is weaker than
//! linking against it and stronger than nothing.

use std::path::PathBuf;

use pie_ir::op::{OP_TABLE, VARIADIC, tags};

/// The driver's marker for "variadic operand count".
///
/// Deliberately not [`VARIADIC`]: the two sides picked different bytes. Rust
/// says `0xFF` means variadic; the C++ header says `0xFE` means variadic and
/// reserves `0xFF` for "unknown op", which is what `op_is_known` tests. Neither
/// side is wrong on its own and nothing crosses between them today, so this is
/// recorded rather than repaired -- but it is recorded, which is the point.
const CXX_VARIADIC: u8 = 0xFE;

struct CxxRow {
    family: String,
    launch: String,
    result: String,
    arity: u8,
    results: u8,
    name: String,
}

fn op_table_hpp() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../driver/abi/include/pie_native/launch/op_table.hpp")
}

/// Parses the `case OpCode::X: return {c, F::A, L::B, R::C, arity, results, "name"};`
/// rows out of `op_info`, keyed by the quoted name.
fn cxx_rows(source: &str) -> Vec<CxxRow> {
    let mut rows = Vec::new();
    for line in source.lines() {
        let Some(rest) = line.trim().strip_prefix("case OpCode::") else {
            continue;
        };
        let Some((_, body)) = rest.split_once("return {") else {
            continue;
        };
        let Some((fields, _)) = body.split_once('}') else {
            continue;
        };
        let parts: Vec<&str> = fields.split(',').map(str::trim).collect();
        if parts.len() != 7 {
            panic!("unparsed op_info row (expected 7 fields): {line}");
        }
        let strip = |field: &str| {
            field
                .rsplit_once("::")
                .map(|(_, tail)| tail.to_string())
                .unwrap_or_else(|| field.to_string())
        };
        rows.push(CxxRow {
            family: strip(parts[1]),
            launch: strip(parts[2]),
            result: strip(parts[3]),
            arity: parse_byte(parts[4], line),
            results: parse_byte(parts[5], line),
            name: parts[6].trim_matches('"').to_string(),
        });
    }
    assert!(!rows.is_empty(), "parsed no rows out of op_table.hpp");
    rows
}

fn parse_byte(field: &str, line: &str) -> u8 {
    let parsed = field
        .strip_prefix("0x")
        .map(|hex| u8::from_str_radix(hex, 16))
        .unwrap_or_else(|| field.parse());
    parsed.unwrap_or_else(|error| panic!("bad byte `{field}` ({error}) in: {line}"))
}

/// Rust `Family` and C++ `OpFamily` are different taxonomies, not different
/// spellings of one. Writing the correspondence down is the only way a change
/// to either can be noticed; an unlisted Rust family fails the test.
fn expected_cxx_family(rust: &str) -> &'static str {
    match rust {
        "Map" => "Map",
        "CompareLogic" => "Compare",
        "Choice" => "Choice",
        "Shape" => "Shape",
        "Index" => "Index",
        "ReduceScan" => "Reduce",
        "Order" => "Order",
        "Linear" => "Linear",
        "Sampling" => "Sampling",
        // Rust splits what the driver calls `Structural` into the ops that
        // touch channels, the ones that are pure leaves, and the intrinsics.
        "Leaf" | "Channel" | "Intrinsic" => "Structural",
        other => panic!("no C++ OpFamily declared for Rust Family::{other}"),
    }
}

#[test]
fn every_rust_op_has_a_driver_row() {
    let source = std::fs::read_to_string(op_table_hpp()).expect("read op_table.hpp");
    let rows = cxx_rows(&source);

    let missing: Vec<&str> = OP_TABLE
        .iter()
        .filter(|spec| !rows.iter().any(|row| row.name == spec.name))
        .map(|spec| spec.name)
        .collect();
    assert!(
        missing.is_empty(),
        "{} ops in OP_TABLE have no `case` in op_info, so the driver's `op_info` \
         falls through to its unknown-op default and treats them as one-result \
         aliasing maps: {missing:?}",
        missing.len()
    );
}

#[test]
fn driver_rows_agree_on_arity_and_result_count() {
    let source = std::fs::read_to_string(op_table_hpp()).expect("read op_table.hpp");
    let rows = cxx_rows(&source);

    for spec in OP_TABLE {
        let Some(row) = rows.iter().find(|row| row.name == spec.name) else {
            continue; // reported by `every_rust_op_has_a_driver_row`
        };
        let expected_arity = if spec.val_operands == VARIADIC {
            CXX_VARIADIC
        } else {
            spec.val_operands
        };
        assert_eq!(
            row.arity, expected_arity,
            "`{}` takes {} value operands in OP_TABLE but {} in op_table.hpp",
            spec.name, spec.val_operands, row.arity
        );
        assert_eq!(
            row.results, spec.results,
            "`{}` defines {} results in OP_TABLE but {} in op_table.hpp",
            spec.name, spec.results, row.results
        );
    }
}

#[test]
fn driver_rows_agree_on_family() {
    let source = std::fs::read_to_string(op_table_hpp()).expect("read op_table.hpp");
    let rows = cxx_rows(&source);

    for spec in OP_TABLE {
        let Some(row) = rows.iter().find(|row| row.name == spec.name) else {
            continue;
        };
        let rust_family = format!("{:?}", spec.family);
        assert_eq!(
            row.family,
            expected_cxx_family(&rust_family),
            "`{}` is Family::{rust_family} in OP_TABLE but OpFamily::{} in op_table.hpp",
            spec.name,
            row.family
        );
    }
}

/// The driver has rows the Rust table does not. They are not errors -- the
/// header says composite and private ops live at `0xE0+` and are deliberately
/// not container ops -- but an unexpected one means the two sets have moved
/// apart, so the exceptions are listed rather than waved through.
#[test]
fn driver_rows_beyond_the_container_op_set_are_declared() {
    let source = std::fs::read_to_string(op_table_hpp()).expect("read op_table.hpp");
    let rows = cxx_rows(&source);

    let extra: Vec<&str> = rows
        .iter()
        .filter(|row| !OP_TABLE.iter().any(|spec| spec.name == row.name))
        .map(|row| row.name.as_str())
        .collect();
    let expected: Vec<&str> = Vec::new();
    assert_eq!(
        extra, expected,
        "op_table.hpp has rows with no OP_TABLE counterpart; if these are the \
         private `0xE0+` fused forms, add them to this test's allowance with a \
         note, and if they are not, one of the two tables is stale"
    );
}

/// `launch` and `result` are driver-only decisions, so there is nothing in
/// `OP_TABLE` to compare them against. Pinning the distribution at least makes
/// a wholesale reclassification visible.
#[test]
fn driver_launch_and_result_classes_are_pinned() {
    let source = std::fs::read_to_string(op_table_hpp()).expect("read op_table.hpp");
    let rows = cxx_rows(&source);

    let mut launches: Vec<String> = rows.iter().map(|row| row.launch.clone()).collect();
    launches.sort();
    launches.dedup();
    let mut results: Vec<String> = rows.iter().map(|row| row.result.clone()).collect();
    results.sort();
    results.dedup();

    assert_eq!(
        launches,
        [
            "Alias",
            "Elementwise",
            "Gather",
            "Library",
            "Materialize",
            "RowLocal",
            "Scatter",
            "Structural"
        ],
        "the driver's LaunchClass distribution changed"
    );
    assert_eq!(
        results,
        ["Bool", "Custom", "Index", "None", "SameAsInput"],
        "the driver's ResultKind distribution changed"
    );
}

// ===========================================================================
// The planner's library/generated partition
// ===========================================================================

/// Every op the fused *generated* kernel emits inline.
///
/// This list exists to be complete, not to be read. `pie_plan` answers "is
/// this a library op?" with `library_op_for_tag`, whose `_ => None` arm means
/// "emit it inline" -- a fine default for an elementwise op and a wrong one
/// for a library op, because inline is not a kernel that exists for `matmul`.
/// A default cannot tell the two apart, so the partition is pinned instead:
/// add an op to `declare_ops!` and `every_op_is_classified` fails until it
/// appears here or in `library_op_for_tag`.
const GENERATED_TAGS: &[u8] = &[
    tags::EXP,
    tags::LOG,
    tags::NEG,
    tags::RECIP,
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
        let library = pie_plan::library_op_for_tag(row.tag).is_some();
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

#[test]
fn generated_tags_are_real_ops() {
    // A stale entry here would mask a missing classification: an op deleted
    // from `declare_ops!` but left in the list keeps the count looking right.
    for tag in GENERATED_TAGS {
        assert!(
            pie_ir::op::spec(*tag).is_some(),
            "GENERATED_TAGS lists {tag:#04x}, which is not an OP_TABLE row"
        );
    }
    let mut sorted = GENERATED_TAGS.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        GENERATED_TAGS.len(),
        "duplicate tag in GENERATED_TAGS"
    );
}
