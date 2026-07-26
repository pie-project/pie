//! Reading op-tag dispatch back out of the hand-written runtime sources.
//!
//! The `.cuh` and `.metal` files under `runtime/` are the other half of the
//! emitter: Rust decides *which* helper an op is routed to, and C++ decides
//! what that helper does with it. Nothing in either language checks that the
//! two agree, and the failure is quiet — `ptir_parallel_elementwise` writes
//! nothing at all for a tag it does not know, and a `ptir_m1_execute` missing
//! an arm faults at the first fire on a real device rather than at build time.
//!
//! So the runtime sources are parsed here and compared against
//! [`pie_ir::op::OP_TABLE`]. This is the same shape as
//! `metal::preamble::file_matches_emitted_text`: one authority, and a replica
//! that is read back and checked rather than trusted.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// The body of a device function in a runtime source, by name, without the
/// outer braces.
pub fn function_body<'a>(source: &'a str, name: &str) -> &'a str {
    let at = source
        .find(name)
        .unwrap_or_else(|| panic!("{name} is not in the runtime source"));
    let open = source[at..]
        .find(") {")
        .map(|offset| at + offset + 2)
        .unwrap_or_else(|| panic!("{name} has no body"));
    let mut depth = 0usize;
    for (offset, byte) in source[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("{name} has an unbalanced body");
}

/// Every op tag `body` compares a `tag` field against, whether by equality or
/// as an endpoint of `tag >= A && tag <= B`.
///
/// Deliberately strict: a comparison form this does not understand panics
/// rather than being skipped, because a silently ignored arm is exactly the
/// failure the callers exist to catch.
pub fn tags_compared_in(body: &str) -> BTreeSet<u8> {
    let bytes = body.as_bytes();
    let mut points: Vec<(&str, u8)> = Vec::new();
    let mut at = 0usize;
    while let Some(offset) = body[at..].find("tag") {
        let start = at + offset;
        at = start + "tag".len();
        // `tag` and `p.tag` are the dispatch variables; `subtag` or `tags`
        // would be something else.
        let joined_before =
            start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
        let joined_after = bytes
            .get(at)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_');
        if joined_before || joined_after {
            continue;
        }
        let rest = body[at..].trim_start();
        let Some(operator) = ["==", "!=", ">=", "<="]
            .into_iter()
            .find(|operator| rest.starts_with(operator))
        else {
            continue;
        };
        let literal = rest[operator.len()..].trim_start();
        let digits = literal.strip_prefix("0x").unwrap_or_else(|| {
            panic!("`tag {operator}` compared against a non-hex literal: {literal:.24}")
        });
        let value = u8::from_str_radix(&digits[..2], 16)
            .unwrap_or_else(|_| panic!("bad tag literal: 0x{digits:.8}"));
        points.push((operator, value));
    }
    assert!(!points.is_empty(), "no tag comparisons found");

    let mut handled = BTreeSet::new();
    let mut index = 0usize;
    while index < points.len() {
        match points[index..] {
            [(">=", low), ("<=", high), ..] => {
                handled.extend(low..=high);
                index += 2;
            }
            [(operator @ (">=" | "<="), value), ..] => {
                panic!("unpaired `tag {operator} {value:#04x}` — not the range form expected")
            }
            [(_, value), ..] => {
                handled.insert(value);
                index += 1;
            }
            [] => break,
        }
    }
    handled
}

/// Names, for a failure message.
pub fn names(tags: impl IntoIterator<Item = u8>) -> Vec<&'static str> {
    tags.into_iter()
        .map(|tag| pie_ir::op::spec(tag).map_or("?", |spec| spec.name))
        .collect()
}

/// The single-thread interpreter both backends fall back to must have an arm
/// for every op in the table — including the library ops, which reach it when
/// a region is not fused.
///
/// It ends in `m1_fault`, so a missing arm is at least loud; but it is loud on
/// a device, mid-fire, in whichever program first uses the op. Here it is loud
/// at build time.
pub fn assert_execute_covers_the_table(source: &str, what: &str) {
    let handled = tags_compared_in(function_body(source, "void ptir_m1_execute"));
    let declared: BTreeSet<u8> = pie_ir::op::OP_TABLE.iter().map(|spec| spec.tag).collect();
    let missing = names(declared.difference(&handled).copied());
    let extra: Vec<u8> = handled.difference(&declared).copied().collect();
    assert!(
        missing.is_empty(),
        "{what} has no arm for {} op(s): {missing:?}",
        missing.len()
    );
    assert!(
        extra.is_empty(),
        "{what} dispatches on {} tag(s) the op table does not declare: {extra:02x?}",
        extra.len()
    );
}
