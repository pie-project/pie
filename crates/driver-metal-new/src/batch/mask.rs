//! What a wire attention mask says, when it says nothing the kernel's own
//! predicate does not already enforce.
//!
//! The engine hands the driver one packed-u32 row per query token: bit `k`
//! of row `q` set means "row q attends key position k", rows delimited by
//! `word_indptr`. Metal's paged attention takes a *dense* mask — one byte a
//! lane — and nothing builds one from the wire form, so a launch carrying
//! wire masks used to be refused outright.
//!
//! Most of them do not need building. A prefill's mask is the causal
//! pattern: row `q` attends exactly `[0, history + q]`, each row one key
//! further than the row above. That is the bound `sdpa_paged` applies on
//! its own from `qo_indptr` and the page CSR — so a mask that says only
//! this can be dropped and the answer is bit-identical, because the
//! kernel's predicate and the mask's are then the same predicate. Anything
//! else — a window, a sink, a skipped key — is a claim the kernel would not
//! make by itself, and the answer is [`None`]: the caller refuses rather
//! than quietly attending to keys the mask excluded.
//!
//! ## The formula stated a third time, and the indexing nobody checked
//!
//! The C++ `first_kv_len_disagreement` restates the CSR KV-length formula —
//! `(pages - 1) * page_size + last_page_len` — for the third time (the
//! schedule build and the paged gate are the other two), and indexes
//! `kv_page_indptr[r + 1]` and `kv_last_page_lens[r]` with no length check
//! anywhere: a mask table describing more requests than the CSR carries is
//! an out-of-bounds read. [`kv_len_disagreement`] takes the
//! [`RequestSpan`]s the schedule already built — checked at construction,
//! one owner for the formula — and compares against `seqlen`.
//!
//! Why the comparison exists at all: a prefill's page CSR carries pages
//! *reserved* for the decode that follows, and an inferlet paging at one
//! size against a driver configured for another turns a 53-key mask into a
//! 117-key CSR claim. The reserved tail is not history, and attending to it
//! reads whatever those pages held before. When the two numbers disagree,
//! the driver cannot know which one the KV write used, so the only safe
//! answer is to say which two numbers differ and stop.

use super::schedule::RequestSpan;

/// One request whose mask and page CSR state two different KV lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Disagreement {
    /// The request at fault.
    pub request: u32,
    /// The length the mask's causal prefix declares.
    pub mask: u32,
    /// The length the page CSR claims ([`RequestSpan::seqlen`]).
    pub csr: u32,
}

/// Whether one wire row is a causal prefix, and how long.
///
/// A prefix is a run of ones from bit zero with nothing after it. A row of
/// zeroes attends nothing and is not a prefix — no causal row attends
/// nothing, its own key is always in range.
fn row_prefix(words: &[u32]) -> Option<u32> {
    let mut prefix = 0u32;
    let mut saw_tail = false;
    for &word in words {
        if saw_tail {
            // Past the run of ones every remaining bit must be clear, or the
            // row attends something its prefix does not cover.
            if word != 0 {
                return None;
            }
            continue;
        }
        let ones = word.trailing_ones();
        prefix += ones;
        if ones != 32 {
            if word >> ones != 0 {
                return None;
            }
            saw_tail = true;
        }
    }
    (prefix != 0).then_some(prefix)
}

/// The per-request KV length each request's mask declares, when every row is
/// a causal prefix one key longer than the row above it.
///
/// `None` when any row says anything else — a window, a sink, a skipped
/// key, a step of more than one — or when the tables do not line up. Either
/// way the mask cannot be dropped, and Metal has no dense-mask builder, so
/// both answers send the caller to the same refusal.
#[must_use]
pub fn causal_prefix_lengths(
    words: &[u32],
    word_indptr: &[u32],
    qo_indptr: &[u32],
) -> Option<Vec<u32>> {
    let requests = qo_indptr.len().checked_sub(1)?;
    if requests == 0 {
        return None;
    }
    let rows = *qo_indptr.last()? as usize;
    if word_indptr.len() < rows + 1 {
        return None;
    }

    let mut lengths = Vec::with_capacity(requests);
    for r in 0..requests {
        let lo = qo_indptr[r];
        let hi = qo_indptr[r + 1];
        if hi <= lo {
            return None;
        }
        let mut previous = 0u32;
        for q in lo..hi {
            let begin = word_indptr[q as usize] as usize;
            let end = word_indptr[q as usize + 1] as usize;
            if begin > end || end > words.len() {
                return None;
            }
            let prefix = row_prefix(&words[begin..end])?;
            // Each row reaches exactly one key further than the row above.
            // A step of anything else is a pattern with a shape of its own.
            if q != lo && prefix != previous + 1 {
                return None;
            }
            previous = prefix;
        }
        lengths.push(previous);
    }
    Some(lengths)
}

/// The first request whose mask length disagrees with what the page CSR says
/// its KV length is, or `None` when they all agree.
///
/// `lengths` comes from [`causal_prefix_lengths`]; `spans` from the
/// schedule, which already computed and checked every `seqlen`. A mask
/// table describing more requests than the schedule carries is itself the
/// first disagreement — the C++ read past the CSR arrays there.
#[must_use]
pub fn kv_len_disagreement(lengths: &[u32], spans: &[RequestSpan]) -> Option<Disagreement> {
    for (r, &mask) in lengths.iter().enumerate() {
        let csr = match spans.get(r) {
            Some(span) => span.seqlen,
            None => {
                return Some(Disagreement {
                    request: r as u32,
                    mask,
                    csr: 0,
                });
            }
        };
        if csr != mask {
            return Some(Disagreement {
                request: r as u32,
                mask,
                csr,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::schedule::build_schedule;
    use super::*;

    /// Rows attending `[0, len)` as packed words.
    fn prefix_row(len: u32) -> Vec<u32> {
        let mut words = Vec::new();
        let mut remaining = len;
        while remaining >= 32 {
            words.push(u32::MAX);
            remaining -= 32;
        }
        words.push((1u32 << remaining) - 1);
        words
    }

    /// A mask table from per-row prefix lengths, one request per group.
    fn table(groups: &[&[u32]]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut words = Vec::new();
        let mut word_indptr = vec![0u32];
        let mut qo_indptr = vec![0u32];
        for group in groups {
            for &len in *group {
                words.extend(prefix_row(len));
                word_indptr.push(words.len() as u32);
            }
            qo_indptr.push(word_indptr.len() as u32 - 1);
        }
        (words, word_indptr, qo_indptr)
    }

    #[test]
    fn a_causal_prefill_mask_declares_its_kv_lengths() {
        // Request 0: 3 rows over 33..=35 keys (multi-word); request 1: one
        // decode row at 5 keys.
        let (words, word_indptr, qo_indptr) = table(&[&[33, 34, 35], &[5]]);
        assert_eq!(
            causal_prefix_lengths(&words, &word_indptr, &qo_indptr),
            Some(vec![35, 5])
        );
    }

    #[test]
    fn a_gap_in_a_row_is_not_causal() {
        // Row attends keys {0, 2}: bit 1 clear inside the run.
        let words = [0b101u32];
        assert_eq!(causal_prefix_lengths(&words, &[0, 1], &[0, 1]), None);
    }

    #[test]
    fn a_bit_past_a_full_word_gap_is_not_causal() {
        // First word full, second empty, third claims a key: a sink pattern.
        let words = [u32::MAX, 0, 1];
        assert_eq!(causal_prefix_lengths(&words, &[0, 3], &[0, 1]), None);
    }

    #[test]
    fn a_row_that_attends_nothing_is_not_causal() {
        let words = [0u32];
        assert_eq!(causal_prefix_lengths(&words, &[0, 1], &[0, 1]), None);
    }

    #[test]
    fn a_step_of_two_keys_between_rows_is_a_shape_of_its_own() {
        // 33 then 35 skips a key: a windowed pattern, not the causal one.
        let (words, word_indptr, qo_indptr) = table(&[&[33, 35]]);
        assert_eq!(
            causal_prefix_lengths(&words, &word_indptr, &qo_indptr),
            None
        );
    }

    #[test]
    fn tables_that_do_not_line_up_cannot_be_dropped() {
        assert_eq!(causal_prefix_lengths(&[], &[0], &[]), None, "no qo_indptr");
        assert_eq!(causal_prefix_lengths(&[], &[0], &[0]), None, "no requests");
        assert_eq!(
            causal_prefix_lengths(&[1], &[0], &[0, 1]),
            None,
            "word_indptr shorter than the rows"
        );
        assert_eq!(
            causal_prefix_lengths(&[1], &[0, 9], &[0, 1]),
            None,
            "a row past the words array"
        );
    }

    #[test]
    fn agreement_is_checked_against_the_schedules_own_seqlen() {
        // One request, 2 pages of 32 with 3 in the last: seqlen 35.
        let schedule = build_schedule(3, &[0, 3], &[0, 2], &[3], &[], &[], 32).expect("schedule");
        assert_eq!(kv_len_disagreement(&[35], &schedule.spans), None);

        // The classic mismatch: an inferlet paging at 16 against a driver at
        // 32 — the mask says 53, the CSR says 117.
        assert_eq!(
            kv_len_disagreement(&[53], &schedule.spans),
            Some(Disagreement {
                request: 0,
                mask: 53,
                csr: 35
            })
        );
    }

    #[test]
    fn a_mask_describing_more_requests_than_the_schedule_is_the_disagreement() {
        // The C++ read past the CSR arrays here.
        let schedule = build_schedule(1, &[0, 1], &[0, 1], &[1], &[], &[], 32).expect("schedule");
        assert_eq!(
            kv_len_disagreement(&[1, 7], &schedule.spans),
            Some(Disagreement {
                request: 1,
                mask: 7,
                csr: 0
            })
        );
    }
}
