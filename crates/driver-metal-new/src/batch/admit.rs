//! What this driver refuses to run: the recurrent-state admission gate.
//!
//! The CUDA driver can replay a buffered recurrent prefix, resolve a fold
//! length the device wrote, address a buffer whose first live token sits
//! mid-page, cut a fold boundary inside a fire's own tokens, and fold one
//! request's state while another only buffers. This driver can do none of
//! those — and each one fails *quietly* if admitted: the recurrence runs
//! from the folded state and ignores the buffer, or folds the host's upper
//! bound instead of the device's count, or reads tokens a fold already
//! absorbed, or double-folds on the next fire. Every failure corrupts a
//! recurrent state that cannot be recovered once folded.
//!
//! So the gate refuses up front, at composition, while the launch is still
//! a value the host can fix. The C++ (`compose.cpp`'s `build_launch_view`)
//! has the same list — it is the file's real content, wrapped in slice
//! plumbing — but expresses each refusal as `throw std::runtime_error("…")`:
//! eight distinct decisions whose only identity is their prose, unmatchable
//! by a caller and conflating "your launch is malformed" with "this driver
//! lacks the capability". [`Refused`] names each one, keeps the C++'s own
//! (good) prose as [`Refused::reason`], and marks which rows are at fault
//! where a row is the fault.
//!
//! The slice plumbing itself — `build_launch_view`, `OwnedLaunchView::capture`,
//! `OwnedLaunchView::view` — is dropped, not ported: it exists because the C
//! ABI hands the C++ borrowed slices that die at return, so every launch had
//! to be re-wrapped and deep-copied. The Rust engine hands the driver an
//! owned [`LaunchPlan`] already.

use driver_abi::local::{PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD, PIE_RS_FLAG_FOLD_LEN_DEVICE};
use driver_abi::plan::LaunchPlan;

/// Why this driver will not run a launch's recurrent-state shape.
///
/// The first five are capability refusals — the launch may be perfectly
/// well-formed and CUDA may run it. The `Malformed*` pair is different in
/// kind: the arrays do not agree on the row count, so the safety checks
/// cannot even be applied, and admitting a fire *because* its arrays did not
/// line up is exactly the outcome the checks exist to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// A row asks to replay buffered tokens ahead of its own.
    ///
    /// Metal has no extended token layout: it would run the new tokens from
    /// the folded state and silently ignore what is buffered — a wrong
    /// answer that then gets folded and cannot be recovered.
    BufferReplay,
    /// A row's fold length lives on the device.
    ///
    /// The wire slot carries the host's upper bound, so this driver would
    /// read a well-formed number and fold the whole buffer instead of the
    /// prefix the device actually accepted.
    DeviceFoldLength,
    /// A row's first live buffer token is mid-page.
    ///
    /// The buffer gather/scatter treat logical token 0 as physical offset 0;
    /// a non-zero head would read the tokens a fold already absorbed and
    /// overwrite the live ones.
    MidPageHead,
    /// A fold boundary lands strictly inside a fire's own tokens.
    ///
    /// That shape needs the two-call cut CUDA makes; this driver issues one
    /// call per fire and would fold the whole row, leaving the host
    /// believing tokens are still buffered that the device already absorbed
    /// — a double fold on the next fire.
    FoldInsideFire {
        /// The request row whose boundary is inside the fire.
        row: u32,
    },
    /// A fold row carries no tokens.
    ///
    /// "Compute nothing, only move the boundary" needs the replay path this
    /// driver refuses above; here it would run an empty forward and move
    /// nothing, leaving the host believing a fold happened.
    FoldOfNothing {
        /// The empty request row.
        row: u32,
    },
    /// One request folds its state while another only buffers.
    ///
    /// The two shapes are one dispatch differing in whether the state
    /// persists; CUDA has a per-row mask, this driver only the pass-level
    /// flag, so it would fold every row or none — either way one row's state
    /// is wrong, unrecoverably.
    MixedPersistence {
        /// The first row that disagrees with row zero.
        row: u32,
    },
    /// Fold lengths, slot flags and the token CSR disagree on the row count.
    MalformedFoldRows,
    /// The buffer-slot CSR and the slot flags disagree on the row count.
    MalformedBufferRows,
}

impl Refused {
    /// The refusal, in the words the host needs.
    #[must_use]
    pub fn reason(self) -> String {
        match self {
            Refused::BufferReplay => "this driver cannot replay buffered recurrent tokens; \
                 fold the buffer before appending to it"
                .to_owned(),
            Refused::DeviceFoldLength => "this driver cannot resolve a device-resident fold \
                 length; read the length back to the host and pass it as a constant"
                .to_owned(),
            Refused::MidPageHead => "this driver cannot address a buffer whose first live \
                 token is mid-page; fold whole pages only"
                .to_owned(),
            Refused::FoldInsideFire { row } => format!(
                "row {row}: this driver cannot land a fold boundary inside a fire's own \
                 tokens; fold the whole row or none of it"
            ),
            Refused::FoldOfNothing { row } => format!(
                "row {row}: this driver cannot replay a buffered prefix, so a request row \
                 carrying no tokens would fold nothing; fold in a fire that computes"
            ),
            Refused::MixedPersistence { row } => format!(
                "row {row}: this driver cannot fold one request's recurrent state while \
                 another only buffers; split the fire"
            ),
            Refused::MalformedFoldRows => "malformed RS launch: per-row fold lengths, slot \
                 flags and the token CSR must agree on the row count"
                .to_owned(),
            Refused::MalformedBufferRows => "malformed RS launch: the buffer-slot CSR and \
                 the per-row slot flags must agree on the row count"
                .to_owned(),
        }
    }

    /// Whether the launch itself is broken, as opposed to asking for a
    /// capability this driver lacks.
    #[must_use]
    pub fn is_malformed(self) -> bool {
        matches!(
            self,
            Refused::MalformedFoldRows | Refused::MalformedBufferRows
        )
    }
}

/// Admit or refuse a launch's recurrent-state shape.
///
/// Call at composition, before anything is allocated. A launch with no
/// recurrent state at all (every RS array empty) is admitted trivially.
///
/// # Errors
///
/// The first [`Refused`] that applies. The malformed checks run before the
/// per-row checks they protect, because an unexpected shape must fail rather
/// than skip them; the per-row checks run in row order in one pass (the C++
/// made two passes, so a later row's boundary fault could be reported ahead
/// of an earlier row's empty span — both refuse, this order names the
/// earlier row).
pub fn admit_recurrent(plan: &LaunchPlan) -> Result<(), Refused> {
    if plan.rs_buffer_read_lens.iter().any(|&len| len != 0) {
        return Err(Refused::BufferReplay);
    }
    if plan
        .rs_slot_flags
        .iter()
        .any(|&flags| flags & PIE_RS_FLAG_FOLD_LEN_DEVICE != 0)
    {
        return Err(Refused::DeviceFoldLength);
    }
    if plan.rs_buffer_heads.iter().any(|&head| head != 0) {
        return Err(Refused::MidPageHead);
    }

    if !plan.rs_fold_lens.is_empty() {
        if plan.rs_slot_flags.len() != plan.rs_fold_lens.len()
            || plan.qo_indptr.len() != plan.rs_fold_lens.len() + 1
        {
            return Err(Refused::MalformedFoldRows);
        }
        for (row, &fold) in plan.rs_fold_lens.iter().enumerate() {
            // Reads were refused above, so the extended row IS the fire's own
            // tokens and the boundary is directly comparable to their count.
            // A descending CSR is a malformed launch, not a huge row — the
            // C++ wrapped here.
            let tokens = plan.qo_indptr[row + 1]
                .checked_sub(plan.qo_indptr[row])
                .ok_or(Refused::MalformedFoldRows)?;
            if tokens == 0 {
                return Err(Refused::FoldOfNothing { row: row as u32 });
            }
            if plan.rs_slot_flags[row] & PIE_RS_FLAG_BUFFER_WRITE == 0 {
                continue;
            }
            if fold != 0 && fold < tokens {
                return Err(Refused::FoldInsideFire { row: row as u32 });
            }
        }
    }

    if !plan.rs_slot_flags.is_empty() {
        if plan.rs_buffer_slot_indptr.len() != plan.rs_slot_flags.len() + 1 {
            return Err(Refused::MalformedBufferRows);
        }
        let persists = |row: usize| {
            let buffered = plan.rs_buffer_slot_indptr[row + 1] > plan.rs_buffer_slot_indptr[row];
            !buffered || plan.rs_slot_flags[row] & PIE_RS_FLAG_FOLD != 0
        };
        let first = persists(0);
        for row in 1..plan.rs_slot_flags.len() {
            if persists(row) != first {
                return Err(Refused::MixedPersistence { row: row as u32 });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rows, one token each, both buffering one slab, both folding.
    fn folding_plan() -> LaunchPlan {
        LaunchPlan {
            qo_indptr: vec![0, 1, 2],
            rs_slot_ids: vec![0, 1],
            rs_slot_flags: vec![PIE_RS_FLAG_FOLD; 2],
            rs_fold_lens: vec![1, 1],
            rs_buffer_slot_ids: vec![4, 5],
            rs_buffer_slot_indptr: vec![0, 1, 2],
            ..LaunchPlan::default()
        }
    }

    #[test]
    fn a_launch_with_no_recurrent_state_is_admitted_trivially() {
        assert_eq!(admit_recurrent(&LaunchPlan::default()), Ok(()));
    }

    #[test]
    fn a_well_formed_whole_row_fold_is_admitted() {
        assert_eq!(admit_recurrent(&folding_plan()), Ok(()));
    }

    #[test]
    fn a_buffered_replay_is_refused_before_it_runs_from_the_folded_state() {
        let mut plan = folding_plan();
        plan.rs_buffer_read_lens = vec![0, 3];
        assert_eq!(admit_recurrent(&plan), Err(Refused::BufferReplay));
    }

    #[test]
    fn a_device_resident_fold_length_is_refused_not_read_as_the_upper_bound() {
        let mut plan = folding_plan();
        plan.rs_slot_flags[1] |= PIE_RS_FLAG_FOLD_LEN_DEVICE;
        assert_eq!(admit_recurrent(&plan), Err(Refused::DeviceFoldLength));
    }

    #[test]
    fn a_mid_page_buffer_head_is_refused() {
        let mut plan = folding_plan();
        plan.rs_buffer_heads = vec![0, 7];
        assert_eq!(admit_recurrent(&plan), Err(Refused::MidPageHead));
    }

    #[test]
    fn a_fold_boundary_inside_the_fires_own_tokens_is_refused() {
        let mut plan = folding_plan();
        // Row 1 spans 3 tokens but folds only 2 of them.
        plan.qo_indptr = vec![0, 1, 4];
        plan.rs_slot_flags[1] |= PIE_RS_FLAG_BUFFER_WRITE;
        plan.rs_fold_lens = vec![1, 2];
        assert_eq!(
            admit_recurrent(&plan),
            Err(Refused::FoldInsideFire { row: 1 })
        );
        // Folding the whole row is fine, and so is not folding at all.
        plan.rs_fold_lens = vec![1, 3];
        assert_eq!(admit_recurrent(&plan), Ok(()));
        plan.rs_fold_lens = vec![1, 0];
        assert_eq!(admit_recurrent(&plan), Ok(()));
    }

    #[test]
    fn a_fold_row_carrying_no_tokens_is_refused() {
        let mut plan = folding_plan();
        plan.qo_indptr = vec![0, 1, 1];
        assert_eq!(
            admit_recurrent(&plan),
            Err(Refused::FoldOfNothing { row: 1 })
        );
    }

    #[test]
    fn one_row_folding_while_another_only_buffers_is_refused() {
        let mut plan = folding_plan();
        // Row 1 buffers a slab but does not fold: its state must persist
        // while row 0's is consumed — a per-row decision this driver cannot
        // express.
        plan.rs_slot_flags = vec![PIE_RS_FLAG_FOLD, 0];
        plan.rs_fold_lens = vec![];
        assert_eq!(
            admit_recurrent(&plan),
            Err(Refused::MixedPersistence { row: 1 })
        );
        // A row with nothing buffered persists by definition and mixes fine
        // with a folding row.
        plan.rs_buffer_slot_indptr = vec![0, 1, 1];
        plan.rs_slot_flags = vec![PIE_RS_FLAG_FOLD, 0];
        assert_eq!(admit_recurrent(&plan), Ok(()));
    }

    #[test]
    fn arrays_that_disagree_on_the_row_count_fail_rather_than_skip_the_checks() {
        let mut plan = folding_plan();
        plan.rs_fold_lens = vec![1];
        assert_eq!(admit_recurrent(&plan), Err(Refused::MalformedFoldRows));

        let mut plan = folding_plan();
        plan.rs_buffer_slot_indptr = vec![0, 1];
        assert_eq!(admit_recurrent(&plan), Err(Refused::MalformedBufferRows));
        assert!(Refused::MalformedBufferRows.is_malformed());
        assert!(!Refused::BufferReplay.is_malformed());
    }
}
