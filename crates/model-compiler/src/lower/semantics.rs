//! Semantic trace ops that lower without a stated kernel.

use super::*;

/// What a statement without a stated kernel lowers to.
pub(super) enum Semantic {
    Structural,
    /// Host work the backend raises before the fire's launches run. Carries
    /// the kind rather than a symbol: a prep is not a kernel.
    Prep(model_ir::trace::PrepKind),
    Unlowered(&'static str),
}

/// What a statement without a stated kernel lowers to.
///
/// THE TABLE IS GONE. Every tier-1 statement is an [`OpKind::Launch`] whose
/// symbol the DSL stated at trace time — the no-ask contract's B6. What
/// remains is the structural vocabulary: hook sites, preps, and
/// [`OpKind::Select`] ([`OpKind::LmHead`] has the walk's own epilogue arm).
/// A retired wire position reaching this match is a trace from before the
/// retirement, refused as residue rather than guessed at.
pub(super) fn semantic(kind: &OpKind, peel_tail: bool) -> Semantic {
    let _ = peel_tail;
    use OpKind::*;
    match kind {
        HookSite { .. } => Semantic::Structural,
        Prep { prep } => Semantic::Prep(*prep),
        Select { .. } => Semantic::Structural,
        _ => Semantic::Unlowered("a retired semantic op; the DSL states launches now"),
    }
}

/// The kind's name, for a refusal a human reads.
pub(super) fn kind_name(kind: &OpKind) -> &'static str {
    use OpKind::*;
    match kind {
        Select { .. } => "Select",
        LmHead { .. } => "LmHead",
        Launch { .. } => "Launch",
        Guard { .. } => "Guard",
        Prep { .. } => "Prep",
        HookSite { .. } => "HookSite",
        Peel { .. } => "Peel",
        _ => "Retired",
    }
}

/// Rows matching a lowered axis must be contiguous by seriation.
pub(super) fn contiguous(
    rows: &[Row],
    window: &Range<u32>,
    holds: fn(&Row) -> bool,
    axis: &'static str,
    at: usize,
) -> Result<Range<u32>, Uncovered> {
    let mut start = None;
    let mut end = window.start;
    for i in window.clone() {
        if holds(&rows[i as usize]) {
            if start.is_none() {
                start = Some(i);
            } else if end != i {
                return Err(Uncovered::Discontiguous { at_op: at, axis });
            }
            end = i + 1;
        }
    }
    Ok(match start {
        Some(s) => s..end,
        None => window.start..window.start,
    })
}

/// Subtracting an arm must leave one contiguous range.
pub(super) fn subtract(
    window: &Range<u32>,
    taken: &Range<u32>,
    at: usize,
) -> Result<Range<u32>, Uncovered> {
    if taken.start == window.start {
        Ok(taken.end..window.end)
    } else if taken.end == window.end {
        Ok(window.start..taken.start)
    } else {
        Err(Uncovered::Discontiguous {
            at_op: at,
            axis: "arm",
        })
    }
}
