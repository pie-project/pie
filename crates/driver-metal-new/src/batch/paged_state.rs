//! The paged path's recurrent-state lifecycle: may this request continue that
//! slot, and where does the slot stand after it has.
//!
//! [`sequence`](super::sequence) is the same question for the sealed M=1 ring,
//! which holds one sequence. The paged path holds many — one per slot — so the
//! contract is per request rather than per fire, and the lineage it checks is
//! the request's own page span rather than the whole member's.
//!
//! Both halves matter and they are not symmetric. Validation refuses a fire
//! that would read the wrong history; the **commit** is what makes the next
//! fire's validation mean anything, because it is the only thing that moves the
//! slot forward. A validation that passes against a slot the last fire failed
//! to commit is a validation against the state before that fire ran.

use std::collections::BTreeMap;

use super::member::ForwardDesc;
use super::sequence::{Backing, SequenceState};

/// One request's recurrent-state binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RsBinding {
    /// The slot the request reads and writes.
    pub slot: u32,
    /// Zero the state before use.
    pub reset: bool,
    /// Read the existing state.
    pub read: bool,
    /// Write the state back.
    pub write: bool,
}

/// The binding for `request`, or `None` when the member declares none.
///
/// The C++'s `request_rs_binding` also carries a legacy arm: an empty
/// `request_rs_slot_ids` with a member-level `rs_slot_id`/`rs_reset` triple,
/// valid for request 0 only. That fork is already collapsed —
/// [`ForwardDesc::rs_slot`] derives the member-level view from the per-request
/// vectors instead of storing it beside them, so there is one representation
/// and no arm to pick.
#[must_use]
pub fn rs_binding(desc: &ForwardDesc, request: usize) -> Option<RsBinding> {
    Some(RsBinding {
        slot: *desc.request_rs_slot_ids.get(request)?,
        reset: *desc.request_rs_reset.get(request)?,
        read: *desc.request_rs_read.get(request)?,
        write: *desc.request_rs_write.get(request)?,
    })
}

/// Why a paged request may not continue its slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PagedRefused {
    /// The member declares no binding for this request.
    MissingBinding {
        /// The request index.
        request: usize,
    },
    /// `reset` and `read` are not exclusive, or the request does not write.
    ///
    /// See the module's note on `write`: the C++ requires it unconditionally,
    /// so the ABI's per-request bit has exactly one legal value here.
    Flags {
        /// The binding as given.
        binding: RsBinding,
    },
    /// The request's CSR entries do not describe a span.
    Csr {
        /// The request index.
        request: usize,
    },
    /// The slot holds nothing.
    NoState {
        /// The slot named.
        slot: u32,
    },
    /// The slot's history is in the ring, not the pool: the prior fire took
    /// the sealed path and this one is paged.
    NotPagedBacked {
        /// The slot named.
        slot: u32,
        /// How its history is actually held.
        backing: Backing,
    },
    /// The slot holds a different sequence.
    DifferentSequence {
        /// The slot named.
        slot: u32,
        /// The sequence resident there.
        resident: u64,
        /// The sequence this fire belongs to.
        fire: u64,
    },
    /// The slot is not at the position this request starts from.
    PositionMismatch {
        /// The slot named.
        slot: u32,
        /// Where the slot stands.
        resident: u32,
        /// Where the request starts.
        fire: u32,
    },
    /// The request's page span does not carry the slot's pages as a prefix.
    LineageBroken {
        /// The slot named.
        slot: u32,
        /// Pages the slot has.
        resident: Vec<u32>,
        /// Pages the request spans.
        fire: Vec<u32>,
    },
}

/// The request's `[begin, end)` spans into positions and pages, checked.
fn spans(desc: &ForwardDesc, request: usize) -> Option<(usize, usize, usize, usize)> {
    let q0 = *desc.qo_indptr.get(request)? as usize;
    let q1 = *desc.qo_indptr.get(request + 1)? as usize;
    let k0 = *desc.kv_page_indptr.get(request)? as usize;
    let k1 = *desc.kv_page_indptr.get(request + 1)? as usize;
    (q1 > q0 && q1 <= desc.position_ids.len() && k1 >= k0 && k1 <= desc.kv_pages.len())
        .then_some((q0, q1, k0, k1))
}

/// Decide whether `request` may continue the slot it names.
///
/// # Errors
///
/// [`PagedRefused`], naming the slot. The C++ splits the not-matching case
/// five ways on purpose — *"say which, or the caller cannot tell a slot that
/// was never resident from one that is a token behind"* — and so does this.
pub fn validate_paged_continuation(
    states: &BTreeMap<u32, SequenceState>,
    desc: &ForwardDesc,
    request: usize,
) -> Result<(), PagedRefused> {
    let binding = rs_binding(desc, request).ok_or(PagedRefused::MissingBinding { request })?;
    if binding.reset == binding.read || !binding.write {
        return Err(PagedRefused::Flags { binding });
    }
    let (q0, _, k0, k1) = spans(desc, request).ok_or(PagedRefused::Csr { request })?;
    if binding.reset {
        return Ok(());
    }

    let slot = binding.slot;
    let state = states.get(&slot).ok_or(PagedRefused::NoState { slot })?;
    if state.backing != Backing::Paged {
        return Err(PagedRefused::NotPagedBacked {
            slot,
            backing: state.backing,
        });
    }
    if state.sequence_id != desc.sequence_id {
        return Err(PagedRefused::DifferentSequence {
            slot,
            resident: state.sequence_id,
            fire: desc.sequence_id,
        });
    }
    let want = desc.position_ids[q0];
    if state.next_position != want {
        return Err(PagedRefused::PositionMismatch {
            slot,
            resident: state.next_position,
            fire: want,
        });
    }
    let span = &desc.kv_pages[k0..k1];
    if state.pages.len() > span.len() || span[..state.pages.len()] != state.pages[..] {
        return Err(PagedRefused::LineageBroken {
            slot,
            resident: state.pages.clone(),
            fire: span.to_vec(),
        });
    }
    Ok(())
}

/// Move the slot forward past `request`.
///
/// # Errors
///
/// The same [`PagedRefused`] the validation uses, for the same shapes.
///
/// # Why this returns a result and the C++ does not
///
/// `commit_paged_request_state` opens with
///
/// ```cpp
/// if (!request_rs_binding(...) || !write ||
///     request + 1 >= desc.qo_indptr.size() ||
///     request + 1 >= desc.kv_page_indptr.size()) {
///     return;
/// }
/// ```
///
/// — a `void` that silently declines to commit. Validation ran first, so the
/// guard should be unreachable; but if it ever fires, the fire has *already
/// run on the device* and the slot is left holding pre-fire state. The next
/// continuation then validates against a position and a page list the device
/// has moved past, passes, and reads a history one fire stale. A commit that
/// cannot happen is not a no-op, and saying so costs a `Result`.
pub fn commit_paged(
    states: &mut BTreeMap<u32, SequenceState>,
    desc: &ForwardDesc,
    request: usize,
) -> Result<(), PagedRefused> {
    let binding = rs_binding(desc, request).ok_or(PagedRefused::MissingBinding { request })?;
    if !binding.write {
        return Err(PagedRefused::Flags { binding });
    }
    let (_, q1, k0, k1) = spans(desc, request).ok_or(PagedRefused::Csr { request })?;

    states.insert(
        binding.slot,
        SequenceState {
            sequence_id: desc.sequence_id,
            // One past the last position this request wrote.
            next_position: desc.position_ids[q1 - 1] + 1,
            pages: desc.kv_pages[k0..k1].to_vec(),
            backing: Backing::Paged,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(sequence: u64, from: u32, n: u32, pages: &[u32], reset: bool) -> ForwardDesc {
        ForwardDesc {
            sequence_id: sequence,
            token_ids: (0..n).collect(),
            position_ids: (from..from + n).collect(),
            kv_pages: pages.to_vec(),
            qo_indptr: vec![0, n],
            kv_page_indptr: vec![0, pages.len() as u32],
            request_rs_slot_ids: vec![3],
            request_rs_reset: vec![reset],
            request_rs_read: vec![!reset],
            request_rs_write: vec![true],
            ..ForwardDesc::default()
        }
    }

    fn paged(sequence: u64, next: u32, pages: &[u32]) -> SequenceState {
        SequenceState {
            sequence_id: sequence,
            next_position: next,
            pages: pages.to_vec(),
            backing: Backing::Paged,
        }
    }

    #[test]
    fn a_reset_request_needs_no_resident_state_at_all() {
        let states = BTreeMap::new();
        assert_eq!(
            validate_paged_continuation(&states, &desc(7, 0, 4, &[1, 2], true), 0),
            Ok(())
        );
    }

    #[test]
    fn a_continuation_matches_sequence_position_and_lineage() {
        let mut states = BTreeMap::new();
        states.insert(3, paged(7, 4, &[1, 2]));
        assert_eq!(
            validate_paged_continuation(&states, &desc(7, 4, 2, &[1, 2, 5], false), 0),
            Ok(())
        );
    }

    #[test]
    fn the_five_ways_to_not_match_are_five_different_answers() {
        // The C++'s own reason: "say which, or the caller cannot tell a slot
        // that was never resident from one that is a token behind."
        let d = desc(7, 4, 2, &[1, 2], false);

        let empty = BTreeMap::new();
        assert_eq!(
            validate_paged_continuation(&empty, &d, 0),
            Err(PagedRefused::NoState { slot: 3 })
        );

        let mut ring = BTreeMap::new();
        ring.insert(
            3,
            SequenceState {
                backing: Backing::Ring,
                ..paged(7, 4, &[1, 2])
            },
        );
        assert_eq!(
            validate_paged_continuation(&ring, &d, 0),
            Err(PagedRefused::NotPagedBacked {
                slot: 3,
                backing: Backing::Ring
            })
        );

        let mut other = BTreeMap::new();
        other.insert(3, paged(99, 4, &[1, 2]));
        assert_eq!(
            validate_paged_continuation(&other, &d, 0),
            Err(PagedRefused::DifferentSequence {
                slot: 3,
                resident: 99,
                fire: 7
            })
        );

        let mut behind = BTreeMap::new();
        behind.insert(3, paged(7, 3, &[1, 2]));
        assert_eq!(
            validate_paged_continuation(&behind, &d, 0),
            Err(PagedRefused::PositionMismatch {
                slot: 3,
                resident: 3,
                fire: 4
            })
        );

        let mut forked = BTreeMap::new();
        forked.insert(3, paged(7, 4, &[1, 9]));
        assert_eq!(
            validate_paged_continuation(&forked, &d, 0),
            Err(PagedRefused::LineageBroken {
                slot: 3,
                resident: vec![1, 9],
                fire: vec![1, 2]
            })
        );
    }

    #[test]
    fn reset_and_read_must_be_exclusive() {
        let mut both = desc(7, 0, 2, &[1], true);
        both.request_rs_read = vec![true]; // reset AND read
        assert!(matches!(
            validate_paged_continuation(&BTreeMap::new(), &both, 0),
            Err(PagedRefused::Flags { .. })
        ));

        let mut neither = desc(7, 0, 2, &[1], false);
        neither.request_rs_read = vec![false]; // neither reset nor read
        assert!(matches!(
            validate_paged_continuation(&BTreeMap::new(), &neither, 0),
            Err(PagedRefused::Flags { .. })
        ));
    }

    #[test]
    fn a_commit_that_cannot_happen_says_so_instead_of_returning_quietly() {
        // The C++ is `void` and returns on the same condition. The fire has
        // already run by then, so a silent skip leaves the slot holding
        // pre-fire state and the NEXT continuation validates against it,
        // passes, and reads a history one fire stale.
        let mut states = BTreeMap::new();
        let mut d = desc(7, 4, 2, &[1, 2], false);
        d.request_rs_write = vec![false];
        assert!(matches!(
            commit_paged(&mut states, &d, 0),
            Err(PagedRefused::Flags { .. })
        ));
        assert!(states.is_empty(), "nothing was committed");
    }

    #[test]
    fn a_commit_moves_the_slot_past_the_requests_last_token() {
        let mut states = BTreeMap::new();
        commit_paged(&mut states, &desc(7, 4, 3, &[1, 2], false), 0).expect("well formed");
        let state = &states[&3];
        assert_eq!(state.sequence_id, 7);
        assert_eq!(state.next_position, 7, "positions 4,5,6 written; next is 7");
        assert_eq!(state.pages, [1, 2]);
        assert_eq!(state.backing, Backing::Paged);
    }

    #[test]
    fn a_committed_slot_validates_the_fire_that_follows_it() {
        // The two halves against each other: whatever `commit_paged` writes is
        // exactly what the next fire's validation expects to find.
        let mut states = BTreeMap::new();
        commit_paged(&mut states, &desc(7, 0, 4, &[1, 2], true), 0).expect("first fire");
        assert_eq!(
            validate_paged_continuation(&states, &desc(7, 4, 2, &[1, 2, 5], false), 0),
            Ok(()),
            "the commit's position and pages are what the next fire is checked against"
        );
    }

    #[test]
    fn a_member_declaring_no_binding_for_the_request_is_refused_by_index() {
        let mut d = desc(7, 0, 2, &[1], true);
        d.request_rs_slot_ids.clear();
        assert_eq!(
            validate_paged_continuation(&BTreeMap::new(), &d, 0),
            Err(PagedRefused::MissingBinding { request: 0 })
        );
    }
}
