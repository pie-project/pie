//! The M=1 ring's single-resident-sequence contract.
//!
//! The sealed fast path keeps one linear KV sequence in a shared ring, and a
//! fire either starts that sequence fresh or extends the one already there.
//! Everything else — a fork, a shared prefix, two sequences interleaved, a
//! rewrite of committed pages — is refused, because the ring has one history
//! and cannot hold two.
//!
//! Every refusal here is a shape that would otherwise produce **wrong tokens
//! rather than an error**: the fire would run, read a history that is not its
//! own, and return a confident continuation of someone else's conversation.
//! That is why the C++ spends nine distinct rejections on it and why they are
//! kept one for one.
//!
//! # Resident is not the same as backing
//!
//! A slot can hold real, correctly-tracked metadata — sequence id, next
//! position, page-list prefix — without being the slot the ring is currently
//! backing. `copy_state` produces exactly that: it copies a source slot's
//! bookkeeping to a destination and clears `ring_backed`, because the bytes
//! moved but the ring did not. The distinction is load-bearing and
//! [`Backing`] keeps it.

use super::member::ForwardDesc;

/// How a resident sequence's history is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// This slot is the one the shared M=1 ring is backing right now, reached
    /// through a real fire on this slot.
    Ring,
    /// The history lives in the paged KV pool.
    Paged,
    /// Neither: the metadata is accurate but arrived by `copy_state`, so the
    /// ring is backing some other slot. Such a sequence can be continued only
    /// on the paged path.
    Copied,
}

/// What a slot knows about the sequence resident in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceState {
    /// The sequence this slot holds.
    pub sequence_id: u64,
    /// The position a continuing fire must start at.
    pub next_position: u32,
    /// The ordered page list backing it, exactly as last observed. A later
    /// fire's list must carry this as a literal prefix.
    pub pages: Vec<u32>,
    /// How the history is held.
    pub backing: Backing,
}

/// What a validated fire does to the resident sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Continuation {
    /// Starts the sequence over; nothing resident is read.
    Fresh,
    /// Extends the resident sequence.
    Extends,
}

/// Why a fire cannot run on the sealed M=1 path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceRefused {
    /// The fire carries no tokens.
    NoTokens,
    /// Token and position counts disagree.
    TokensAgainstPositions {
        /// Tokens supplied.
        tokens: usize,
        /// Positions supplied.
        positions: usize,
    },
    /// Positions within a fire must be a contiguous ascending run.
    PositionsNotInOrder {
        /// The index whose successor broke the run.
        at: usize,
        /// The position there.
        position: u32,
        /// The position after it.
        next: u32,
    },
    /// A physical page id appears twice in one sequence's own list.
    ///
    /// Page *numbering* is reused across sequences by the free list and need
    /// not be adjacent — `{5, 9}` is a valid two-page allocation. A duplicate
    /// within one list is the fork/share/aliasing signal.
    DuplicatePage {
        /// The repeated id.
        page: u32,
    },
    /// A fresh fire arrived while a different sequence still backs the ring.
    RingHeldByAnother {
        /// The sequence still resident.
        resident: u64,
    },
    /// The fire is not fresh and nothing is resident to continue.
    NothingResident,
    /// The fire belongs to a different sequence than the resident one.
    DifferentSequence {
        /// The sequence resident in the slot.
        resident: u64,
        /// The sequence this fire belongs to.
        fire: u64,
    },
    /// The resident state arrived by `copy_state`, so the ring is not backing
    /// it and the sealed path cannot continue it; the paged CSR path can.
    NotRingBacked,
    /// The fire's positions do not continue where the resident sequence ended.
    PositionsDoNotExtend {
        /// Where the fire starts.
        starts_at: u32,
        /// Where the resident sequence expects to be continued.
        expected: u32,
    },
    /// The fire's page list is shorter than the resident one — a truncation.
    PageListShorter {
        /// Pages the fire lists.
        fire: usize,
        /// Pages the resident sequence has.
        resident: usize,
    },
    /// The fire's page list does not carry the resident list as a prefix — a
    /// rewrite of already-committed pages.
    PageListRewritten {
        /// The first index that differs.
        at: usize,
        /// What the resident sequence has there.
        resident: u32,
        /// What the fire lists there.
        fire: u32,
    },
}

/// Decide whether a fire may run on the sealed M=1 path.
///
/// `state` is the slot's own resident sequence, if any. `ring_held_by_another`
/// says whether some *other* slot is currently backing the ring with a
/// different sequence — a fresh fire needs the ring, and the ring holds one.
///
/// # Errors
///
/// [`SequenceRefused`], naming the shape. Every one of them is a fire that
/// would otherwise run and answer with someone else's history.
pub fn validate_continuation(
    state: Option<&SequenceState>,
    ring_held_by_another: Option<u64>,
    desc: &ForwardDesc,
) -> Result<Continuation, SequenceRefused> {
    if desc.token_ids.is_empty() {
        return Err(SequenceRefused::NoTokens);
    }
    if desc.position_ids.len() != desc.token_ids.len() {
        return Err(SequenceRefused::TokensAgainstPositions {
            tokens: desc.token_ids.len(),
            positions: desc.position_ids.len(),
        });
    }
    for (at, pair) in desc.position_ids.windows(2).enumerate() {
        if pair[1] != pair[0] + 1 {
            return Err(SequenceRefused::PositionsNotInOrder {
                at,
                position: pair[0],
                next: pair[1],
            });
        }
    }

    // A duplicate inside one sequence's own list. Sorting a copy is what the
    // C++ does; a set would allocate the same and say less about why.
    let mut sorted = desc.kv_pages.clone();
    sorted.sort_unstable();
    if let Some(pair) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(SequenceRefused::DuplicatePage { page: pair[0] });
    }

    // "Fresh" is the reset bit when the fire names a slot, and otherwise is
    // inferred from the first position being zero. The inference is the C++'s
    // and is kept, but it is worth naming as one: a member that names no slot
    // has no statement of intent, so the only evidence available is where its
    // positions start.
    let fresh = match desc.rs_slot() {
        Some((_, reset)) => reset,
        None => desc.position_ids[0] == 0,
    };

    if fresh {
        return match ring_held_by_another {
            Some(resident) => Err(SequenceRefused::RingHeldByAnother { resident }),
            None => Ok(Continuation::Fresh),
        };
    }

    let state = state.ok_or(SequenceRefused::NothingResident)?;
    if state.sequence_id != desc.sequence_id {
        return Err(SequenceRefused::DifferentSequence {
            resident: state.sequence_id,
            fire: desc.sequence_id,
        });
    }
    if state.backing != Backing::Ring {
        return Err(SequenceRefused::NotRingBacked);
    }
    if desc.position_ids[0] != state.next_position {
        return Err(SequenceRefused::PositionsDoNotExtend {
            starts_at: desc.position_ids[0],
            expected: state.next_position,
        });
    }
    if desc.kv_pages.len() < state.pages.len() {
        return Err(SequenceRefused::PageListShorter {
            fire: desc.kv_pages.len(),
            resident: state.pages.len(),
        });
    }
    if let Some((at, (&resident, &fire))) = state
        .pages
        .iter()
        .zip(&desc.kv_pages)
        .enumerate()
        .find(|(_, (r, f))| r != f)
    {
        return Err(SequenceRefused::PageListRewritten { at, resident, fire });
    }
    Ok(Continuation::Extends)
}

/// Close a sequence: clear the slot if it holds the named sequence.
///
/// # The C++ cannot close what `copy_state` created
///
/// `close_linear_sequence` clears only when
///
/// ```cpp
/// state.has_resident && (state.ring_backed || state.paged_backed) &&
///     state.resident_sequence_id == sequence_id
/// ```
///
/// and `copy_state` produces precisely the excluded state: it copies a source
/// slot's bookkeeping and sets `ring_backed = false`, so a copy taken from a
/// **ring-backed** source has neither flag while `has_resident` stays true.
/// That slot can then never be closed. Its stale metadata outlives every
/// `close_sequence` the caller issues, and the entry is a permanent resident
/// in the table the close was written to keep clean.
///
/// The backing describes *how* a sequence is resident, not *whether* it is, so
/// it has no business in the predicate. Here the id alone decides.
pub fn close(state: &mut Option<SequenceState>, sequence_id: u64) {
    if state.as_ref().is_some_and(|s| s.sequence_id == sequence_id) {
        *state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident(pages: &[u32], next: u32, backing: Backing) -> SequenceState {
        SequenceState {
            sequence_id: 7,
            next_position: next,
            pages: pages.to_vec(),
            backing,
        }
    }

    /// A fire on sequence 7 whose positions run `[from, from + n)`.
    fn fire(from: u32, n: u32, pages: &[u32], reset: bool) -> ForwardDesc {
        ForwardDesc {
            sequence_id: 7,
            token_ids: (0..n).collect(),
            position_ids: (from..from + n).collect(),
            kv_pages: pages.to_vec(),
            request_rs_slot_ids: vec![0],
            request_rs_reset: vec![reset],
            request_rs_read: vec![!reset],
            request_rs_write: vec![true],
            ..ForwardDesc::default()
        }
    }

    #[test]
    fn a_fire_that_extends_the_resident_sequence_is_admitted() {
        let state = resident(&[5, 9], 8, Backing::Ring);
        assert_eq!(
            validate_continuation(Some(&state), None, &fire(8, 4, &[5, 9, 12], false)),
            Ok(Continuation::Extends)
        );
    }

    #[test]
    fn a_copied_state_cannot_be_continued_on_the_sealed_path() {
        // The bytes are there and the metadata is accurate, but the ring is
        // backing some other slot, so this fire would read the wrong history.
        let state = resident(&[5], 4, Backing::Copied);
        assert_eq!(
            validate_continuation(Some(&state), None, &fire(4, 1, &[5], false)),
            Err(SequenceRefused::NotRingBacked)
        );
    }

    #[test]
    fn a_sequence_copied_from_a_ring_backed_source_can_still_be_closed() {
        // The defect: the C++ requires `ring_backed || paged_backed` to close,
        // and `copy_state` from a ring-backed source leaves neither, so the
        // entry becomes a permanent resident nothing can clear.
        let mut state = Some(resident(&[5], 4, Backing::Copied));
        close(&mut state, 7);
        assert_eq!(state, None, "the id matched, so the slot is free");
    }

    #[test]
    fn closing_a_different_sequence_leaves_the_slot_alone() {
        let mut state = Some(resident(&[5], 4, Backing::Ring));
        close(&mut state, 8);
        assert!(state.is_some(), "sequence 8 is not the one resident");
    }

    #[test]
    fn a_fresh_fire_needs_the_ring_and_the_ring_holds_one_sequence() {
        assert_eq!(
            validate_continuation(None, Some(3), &fire(0, 4, &[5], true)),
            Err(SequenceRefused::RingHeldByAnother { resident: 3 })
        );
        assert_eq!(
            validate_continuation(None, None, &fire(0, 4, &[5], true)),
            Ok(Continuation::Fresh)
        );
    }

    #[test]
    fn a_duplicated_page_in_one_sequences_own_list_is_the_fork_signal() {
        // Non-adjacent ids are fine — {5, 9} is a legal allocation — so the
        // check is for repetition, not for contiguity.
        assert_eq!(
            validate_continuation(None, None, &fire(0, 4, &[5, 9], true)),
            Ok(Continuation::Fresh)
        );
        assert_eq!(
            validate_continuation(None, None, &fire(0, 4, &[5, 9, 5], true)),
            Err(SequenceRefused::DuplicatePage { page: 5 })
        );
    }

    #[test]
    fn positions_must_be_a_contiguous_ascending_run_within_one_fire() {
        let mut f = fire(4, 3, &[5], false);
        f.position_ids = vec![4, 6, 7];
        let state = resident(&[5], 4, Backing::Ring);
        assert_eq!(
            validate_continuation(Some(&state), None, &f),
            Err(SequenceRefused::PositionsNotInOrder {
                at: 0,
                position: 4,
                next: 6
            })
        );
    }

    #[test]
    fn a_fire_that_does_not_start_where_the_resident_sequence_ended_is_refused() {
        let state = resident(&[5], 8, Backing::Ring);
        assert_eq!(
            validate_continuation(Some(&state), None, &fire(6, 2, &[5], false)),
            Err(SequenceRefused::PositionsDoNotExtend {
                starts_at: 6,
                expected: 8
            })
        );
    }

    #[test]
    fn the_resident_page_list_must_survive_as_a_literal_prefix() {
        let state = resident(&[5, 9], 8, Backing::Ring);
        // A rewrite of an already-committed page.
        assert_eq!(
            validate_continuation(Some(&state), None, &fire(8, 1, &[5, 11, 12], false)),
            Err(SequenceRefused::PageListRewritten {
                at: 1,
                resident: 9,
                fire: 11
            })
        );
        // A truncation.
        assert_eq!(
            validate_continuation(Some(&state), None, &fire(8, 1, &[5], false)),
            Err(SequenceRefused::PageListShorter {
                fire: 1,
                resident: 2
            })
        );
    }

    #[test]
    fn a_continuation_of_a_sequence_nothing_holds_is_refused() {
        assert_eq!(
            validate_continuation(None, None, &fire(8, 1, &[5], false)),
            Err(SequenceRefused::NothingResident)
        );
    }

    #[test]
    fn a_fire_belonging_to_another_sequence_cannot_extend_this_one() {
        let state = resident(&[5], 8, Backing::Ring);
        let mut f = fire(8, 1, &[5], false);
        f.sequence_id = 99;
        assert_eq!(
            validate_continuation(Some(&state), None, &f),
            Err(SequenceRefused::DifferentSequence {
                resident: 7,
                fire: 99
            })
        );
    }

    #[test]
    fn a_member_naming_no_slot_is_fresh_only_by_where_its_positions_start() {
        // The inference the C++ makes, kept and named. With no slot there is
        // no statement of intent, so position zero is the only evidence.
        let mut f = fire(0, 2, &[5], false);
        f.request_rs_slot_ids.clear();
        f.request_rs_reset.clear();
        assert_eq!(
            validate_continuation(None, None, &f),
            Ok(Continuation::Fresh)
        );

        let mut f = fire(4, 2, &[5], false);
        f.request_rs_slot_ids.clear();
        f.request_rs_reset.clear();
        assert_eq!(
            validate_continuation(None, None, &f),
            Err(SequenceRefused::NothingResident),
            "not starting at zero means it claims to continue something"
        );
    }
}
