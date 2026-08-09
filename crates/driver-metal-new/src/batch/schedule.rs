//! The batch shape: what one fire's CSR view means, derived once.
//!
//! The engine hands the driver a marshaled CSR/SoA view per fire — token
//! ids, `qo_indptr`, the paged-KV index arrays, the recurrent-state slots.
//! Everything the M>1 dispatch and the paged-attention read-walk need is a
//! function of those arrays: the per-request token and page spans, whether
//! the batch is a pure decode (every request contributes exactly one token),
//! and the token→request expansion the slotted kernels index by row. This
//! module is that function, ported from `batch_schedule.hpp` — the one file
//! of `batch/` the C++ kept deliberately pure, "unit-testable standalone",
//! and then shipped without a checked build.
//!
//! ## Build first, notice later
//!
//! The C++ splits the work into `build_batch_schedule` (unchecked
//! arithmetic) and `validate_paged_batch` (the geometry gate). The split
//! itself is right — the gate needs fire-time arrays the build does not —
//! but the build trusted its inputs completely: `qo_hi - qo_lo` and
//! `seqlen - new_tokens` are `u32` subtractions, so a non-ascending
//! `qo_indptr` or a span longer than its sequence produced *wrapped* spans,
//! and whether anyone noticed depended on whether that caller also ran the
//! validator. [`build_schedule`] refuses those inputs at construction: a
//! [`BatchSchedule`] that exists has coherent spans, and [`validate_paged`]
//! checks only what genuinely needs the fire-time arrays.
//!
//! ## The other three
//!
//! * `kRsFlagReset` was a hand copy of `PIE_RS_FLAG_RESET`, "duplicated
//!   rather than included" for a test harness this crate does not have. The
//!   constant is [`driver_abi::local::PIE_RS_FLAG_RESET`] here — and the
//!   MASK discipline the C++ comment insists on (a truthiness test reads a
//!   FOLD row as a fresh sequence and zeroes a live recurrent state) is kept
//!   and tested.
//! * `find_request` answered an out-of-range token with `R - 1` — a wrong
//!   request presented like a right one. It answers [`None`] here.
//! * `page_size <= 0` silently became 32. A page size the caller did not
//!   ask for is a geometry the caller cannot predict; the default is the
//!   caller's to take ([`DEFAULT_PAGE_SIZE`]), not this module's to impose.
//!
//! The validator's answers also stop being a `bool` plus a static string:
//! [`Rejected`] carries *which* request or token failed, which is the
//! difference between a fixable report and "malformed request CSR span".

use driver_abi::local::PIE_RS_FLAG_RESET;

/// The page size the shipped pools use when the runtime does not say.
///
/// A named default the *caller* applies, replacing the C++'s silent
/// `page_size > 0 ? page_size : 32` inside the build.
pub const DEFAULT_PAGE_SIZE: u32 = 32;

/// One request's slice of the batch: its token span, its KV pages, and its
/// recurrent-state slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestSpan {
    /// First token row of this request in the batch arrays.
    pub qo_lo: u32,
    /// One past the last token row.
    pub qo_hi: u32,
    /// `qo_hi - qo_lo`: one for a decode row, more for a prefill.
    pub new_tokens: u32,
    /// Base index into `kv_page_indices` for this request.
    pub pages_first: u32,
    /// How many pages back the request's KV.
    pub num_pages: u32,
    /// Total KV length AFTER this fire's tokens are appended.
    pub seqlen: u32,
    /// KV length BEFORE this fire: `seqlen - new_tokens`.
    pub pre_kv_len: u32,
    /// The recurrent-state slot (GDN); zero when none were marshaled.
    pub rs_slot: u32,
    /// The `PIE_RS_FLAG_RESET` bit: a fresh sequence whose state must be
    /// zeroed before use. Masked, never truthiness-tested — `FOLD` is also
    /// non-zero, and reading a fold row as fresh zeroes a live state.
    pub rs_is_new: bool,
}

/// The derived shape of one fire's batch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchSchedule {
    /// Total token rows (the batch/row dimension).
    pub total_tokens: u32,
    /// Number of requests.
    pub requests: u32,
    /// Every request contributes exactly one token — the decode fast path.
    pub is_pure_decode: bool,
    /// The runtime KV page size.
    pub page_size: u32,
    /// Per-request spans, in request order.
    pub spans: Vec<RequestSpan>,
    /// Owning request per token row.
    pub req_of_token: Vec<u32>,
    /// Per-token recurrent-state slot: `spans[req_of_token[t]].rs_slot`.
    /// The slotted kernels index this by token row, not by request.
    pub slot_of_token: Vec<u32>,
}

impl BatchSchedule {
    /// The shipped single-stream fast path: one token, one request.
    #[must_use]
    pub fn single(&self) -> bool {
        self.total_tokens == 1 && self.requests == 1
    }
}

/// Why a CSR view does not describe a batch.
///
/// Every variant names the request it condemns; the C++ computed a wrapped
/// span here and left the noticing to a validator the caller might not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Malformed {
    /// `qo_indptr` is empty, so there is no request count to read.
    NoRequests,
    /// `qo_indptr[r+1] < qo_indptr[r]`: the span subtraction would wrap.
    DescendingTokens {
        /// The request whose span runs backwards.
        request: u32,
    },
    /// `kv_page_indptr[r+1] < kv_page_indptr[r]`.
    DescendingPages {
        /// The request whose page span runs backwards.
        request: u32,
    },
    /// A request appends more tokens than its sequence holds afterwards:
    /// `new_tokens > seqlen`, so `pre_kv_len` would wrap.
    LongerThanItsSequence {
        /// The request at fault.
        request: u32,
    },
    /// The spans do not cover `token_ids` exactly.
    TokensNotCovered {
        /// Where the last span ended.
        covered: u32,
        /// How many token rows the batch has.
        tokens: u32,
    },
    /// A page size of zero has no geometry at all.
    ZeroPageSize,
}

impl Malformed {
    /// One line, for the refusal.
    #[must_use]
    pub fn reason(self) -> String {
        match self {
            Malformed::NoRequests => "batch has no qo_indptr, so no requests".to_owned(),
            Malformed::DescendingTokens { request } => {
                format!("request {request}: qo_indptr runs backwards")
            }
            Malformed::DescendingPages { request } => {
                format!("request {request}: kv_page_indptr runs backwards")
            }
            Malformed::LongerThanItsSequence { request } => {
                format!("request {request}: appends more tokens than its sequence holds")
            }
            Malformed::TokensNotCovered { covered, tokens } => {
                format!("request spans cover {covered} of {tokens} token rows")
            }
            Malformed::ZeroPageSize => "a page size of zero has no geometry".to_owned(),
        }
    }
}

/// Derive the batch shape from the marshaled CSR arrays.
///
/// `qo_indptr` and `kv_page_indptr` are the `R+1` prefix arrays;
/// `kv_last_page_lens` is per request; the two `rs` slices may be empty when
/// the model has no recurrent state. `page_size` is the runtime's — pass
/// [`DEFAULT_PAGE_SIZE`] if it has nothing to say.
///
/// # Errors
///
/// [`Malformed`] naming the request whose arithmetic would have wrapped, or
/// the coverage the spans miss. The C++ built the wrapped schedule and hoped
/// the validator ran.
pub fn build_schedule(
    total_tokens: u32,
    qo_indptr: &[u32],
    kv_page_indptr: &[u32],
    kv_last_page_lens: &[u32],
    rs_slot_ids: &[u32],
    rs_slot_flags: &[u8],
    page_size: u32,
) -> Result<BatchSchedule, Malformed> {
    if page_size == 0 {
        return Err(Malformed::ZeroPageSize);
    }
    if qo_indptr.is_empty() {
        return Err(Malformed::NoRequests);
    }
    let requests = (qo_indptr.len() - 1) as u32;

    let mut spans = Vec::with_capacity(requests as usize);
    let mut pure = requests > 0;
    for r in 0..requests as usize {
        let qo_lo = qo_indptr[r];
        let qo_hi = qo_indptr[r + 1];
        let new_tokens = qo_hi
            .checked_sub(qo_lo)
            .ok_or(Malformed::DescendingTokens { request: r as u32 })?;
        let (pages_first, num_pages) = match (kv_page_indptr.get(r), kv_page_indptr.get(r + 1)) {
            (Some(&lo), Some(&hi)) => (
                lo,
                hi.checked_sub(lo)
                    .ok_or(Malformed::DescendingPages { request: r as u32 })?,
            ),
            _ => (0, 0),
        };
        let seqlen = if num_pages > 0 {
            (num_pages - 1) * page_size + kv_last_page_lens.get(r).copied().unwrap_or(0)
        } else {
            0
        };
        let pre_kv_len = seqlen
            .checked_sub(new_tokens)
            .ok_or(Malformed::LongerThanItsSequence { request: r as u32 })?;
        if new_tokens != 1 {
            pure = false;
        }
        spans.push(RequestSpan {
            qo_lo,
            qo_hi,
            new_tokens,
            pages_first,
            num_pages,
            seqlen,
            pre_kv_len,
            rs_slot: rs_slot_ids.get(r).copied().unwrap_or(0),
            rs_is_new: rs_slot_flags
                .get(r)
                .is_some_and(|&flags| flags & PIE_RS_FLAG_RESET != 0),
        });
    }
    let covered = spans.last().map_or(0, |span| span.qo_hi);
    if covered != total_tokens || spans.first().is_some_and(|span| span.qo_lo != 0) {
        return Err(Malformed::TokensNotCovered {
            covered,
            tokens: total_tokens,
        });
    }

    let mut req_of_token = vec![0u32; total_tokens as usize];
    let mut token = 0usize;
    for (r, span) in spans.iter().enumerate() {
        while token < span.qo_hi as usize && token < req_of_token.len() {
            req_of_token[token] = r as u32;
            token += 1;
        }
    }
    let slot_of_token = req_of_token
        .iter()
        .map(|&r| spans[r as usize].rs_slot)
        .collect();

    Ok(BatchSchedule {
        total_tokens,
        requests,
        is_pure_decode: pure,
        page_size,
        spans,
        req_of_token,
        slot_of_token,
    })
}

/// The request owning token `t`, or `None` when no span holds it.
///
/// The C++ answered `R - 1` for a token past every span — a wrong request
/// shaped like a right one, handed to a grid derivation that then indexed
/// someone else's pages.
#[must_use]
pub fn find_request(qo_indptr: &[u32], token: u32) -> Option<u32> {
    let requests = qo_indptr.len().checked_sub(1)?;
    (0..requests)
        .find(|&r| token >= qo_indptr[r] && token < qo_indptr[r + 1])
        .map(|r| r as u32)
}

/// Why the paged-geometry gate refused a batch.
///
/// The C++ answer was `false` and a static string; which request or token
/// was at fault was exactly the part it dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    /// A parallel array is not the length the schedule requires.
    WrongArrayLengths,
    /// A request's span disagrees with the CSR contract.
    BadSpan {
        /// The request at fault.
        request: u32,
    },
    /// A request's final page length is zero or exceeds the page size.
    BadFinalPage {
        /// The request at fault.
        request: u32,
    },
    /// A request names a physical page outside the pool.
    PageOutOfPool {
        /// The request at fault.
        request: u32,
    },
    /// A request's recurrent-state slot is outside the slot table.
    SlotOutOfRange {
        /// The request at fault.
        request: u32,
    },
    /// The spans do not cover the token or page arrays exactly.
    NotCovering,
    /// A token's expansion, position, or write descriptor is inconsistent.
    BadToken {
        /// The token row at fault.
        token: u32,
    },
    /// The batch exceeds the configured token or request capacity.
    OverCapacity,
}

impl Rejected {
    /// One line, for the refusal.
    #[must_use]
    pub fn reason(self) -> String {
        match self {
            Rejected::WrongArrayLengths => "malformed paged batch vector sizes".to_owned(),
            Rejected::BadSpan { request } => {
                format!("request {request}: malformed CSR span")
            }
            Rejected::BadFinalPage { request } => {
                format!("request {request}: invalid final page length")
            }
            Rejected::PageOutOfPool { request } => {
                format!("request {request}: physical page id exceeds pool")
            }
            Rejected::SlotOutOfRange { request } => {
                format!("request {request}: recurrent-state slot exceeds table")
            }
            Rejected::NotCovering => {
                "CSR does not cover exactly the batch token/page arrays".to_owned()
            }
            Rejected::BadToken { token } => {
                format!("token {token}: position or write descriptor is inconsistent")
            }
            Rejected::OverCapacity => {
                "paged batch exceeds configured token/request capacity".to_owned()
            }
        }
    }
}

/// The final host-side gate before a paged dispatch: validate the exact
/// address formula `kv_append_paged`/`sdpa_paged` will use, including the
/// explicit write descriptors, so invalid geometry is refused before any
/// pool cell can be touched.
///
/// # Errors
///
/// [`Rejected`], naming the request or token at fault.
pub fn validate_paged(
    schedule: &BatchSchedule,
    position_ids: &[u32],
    page_indices: &[u32],
    w_page: &[u32],
    w_off: &[u32],
    total_pages: u32,
    max_slots: u32,
) -> Result<(), Rejected> {
    let tokens = schedule.total_tokens as usize;
    if schedule.total_tokens == 0
        || schedule.requests == 0
        || position_ids.len() != tokens
        || w_page.len() != tokens
        || w_off.len() != tokens
        || schedule.spans.len() != schedule.requests as usize
        || schedule.req_of_token.len() != tokens
        || schedule.slot_of_token.len() != tokens
        || schedule.page_size == 0
    {
        return Err(Rejected::WrongArrayLengths);
    }

    let mut expected_qo = 0u32;
    let mut expected_pages = 0u32;
    for (r, span) in schedule.spans.iter().enumerate() {
        let request = r as u32;
        if span.qo_lo != expected_qo
            || span.qo_lo >= span.qo_hi
            || span.qo_hi > schedule.total_tokens
            || span.pages_first != expected_pages
            || span.num_pages == 0
            || (span.pages_first as usize) + (span.num_pages as usize) > page_indices.len()
            || span.seqlen == 0
            || span.new_tokens > span.seqlen
            || span.pre_kv_len > span.seqlen
        {
            return Err(Rejected::BadSpan { request });
        }
        if span.rs_slot >= max_slots {
            return Err(Rejected::SlotOutOfRange { request });
        }
        let last = span.seqlen - (span.num_pages - 1) * schedule.page_size;
        if last == 0 || last > schedule.page_size {
            return Err(Rejected::BadFinalPage { request });
        }
        for j in 0..span.num_pages {
            if page_indices[(span.pages_first + j) as usize] >= total_pages {
                return Err(Rejected::PageOutOfPool { request });
            }
        }
        expected_qo = span.qo_hi;
        expected_pages = span.pages_first + span.num_pages;
    }
    if expected_qo != schedule.total_tokens || expected_pages as usize != page_indices.len() {
        return Err(Rejected::NotCovering);
    }

    for t in 0..tokens {
        let token = t as u32;
        let r = schedule.req_of_token[t];
        if r >= schedule.requests || schedule.slot_of_token[t] != schedule.spans[r as usize].rs_slot
        {
            return Err(Rejected::BadToken { token });
        }
        let span = &schedule.spans[r as usize];
        let pos = position_ids[t];
        if pos >= span.seqlen || w_page[t] >= total_pages || w_off[t] >= schedule.page_size {
            return Err(Rejected::BadToken { token });
        }
        let expected = page_indices[(span.pages_first + pos / schedule.page_size) as usize];
        if w_page[t] != expected || w_off[t] != pos % schedule.page_size {
            return Err(Rejected::BadToken { token });
        }
    }
    Ok(())
}

/// The capacity gate: the batch fits the configured limits.
///
/// # Errors
///
/// [`Rejected::OverCapacity`].
pub fn validate_capacity(
    schedule: &BatchSchedule,
    max_tokens: u32,
    max_requests: u32,
) -> Result<(), Rejected> {
    if schedule.total_tokens > 0
        && schedule.total_tokens <= max_tokens
        && schedule.requests > 0
        && schedule.requests <= max_requests
    {
        Ok(())
    } else {
        Err(Rejected::OverCapacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two requests: a 3-token prefill over 2 pages, then a 1-token decode.
    fn mixed() -> BatchSchedule {
        build_schedule(
            4,
            &[0, 3, 4],
            &[0, 2, 3],
            &[3, 5],
            &[7, 9],
            &[PIE_RS_FLAG_RESET, 2],
            32,
        )
        .expect("well-formed")
    }

    #[test]
    fn spans_carry_the_geometry_the_dispatch_reads() {
        let s = mixed();
        assert_eq!((s.total_tokens, s.requests), (4, 2));
        assert!(!s.is_pure_decode, "a 3-token span is a prefill");
        assert_eq!(
            s.spans[0],
            RequestSpan {
                qo_lo: 0,
                qo_hi: 3,
                new_tokens: 3,
                pages_first: 0,
                num_pages: 2,
                seqlen: 32 + 3,
                pre_kv_len: 32,
                rs_slot: 7,
                rs_is_new: true,
            }
        );
        assert_eq!(s.spans[1].seqlen, 5);
        assert_eq!(s.spans[1].pre_kv_len, 4);
        assert_eq!(s.req_of_token, [0, 0, 0, 1]);
        assert_eq!(s.slot_of_token, [7, 7, 7, 9]);
    }

    #[test]
    fn a_batch_of_single_token_requests_is_a_pure_decode() {
        let s =
            build_schedule(2, &[0, 1, 2], &[0, 1, 2], &[1, 1], &[], &[], 32).expect("well-formed");
        assert!(s.is_pure_decode);
        assert!(!s.single(), "two lanes is not the single-stream path");
        let one = build_schedule(1, &[0, 1], &[0, 1], &[1], &[], &[], 32).expect("well-formed");
        assert!(one.single() && one.is_pure_decode);
    }

    #[test]
    fn the_reset_flag_is_masked_not_truthiness_tested() {
        // FOLD (2) alone must NOT read as a fresh sequence: that zeroes a
        // live recurrent state.
        let s = build_schedule(2, &[0, 1, 2], &[0, 1, 2], &[1, 1], &[3, 4], &[2, 3], 32)
            .expect("well-formed");
        assert!(!s.spans[0].rs_is_new, "FOLD is not RESET");
        assert!(s.spans[1].rs_is_new, "RESET|FOLD still resets");
    }

    #[test]
    fn a_descending_indptr_is_refused_not_wrapped() {
        assert_eq!(
            build_schedule(4, &[0, 3, 2], &[0, 1, 2], &[3, 1], &[], &[], 32),
            Err(Malformed::DescendingTokens { request: 1 })
        );
        assert_eq!(
            build_schedule(2, &[0, 1, 2], &[0, 2, 1], &[1, 1], &[], &[], 32),
            Err(Malformed::DescendingPages { request: 1 })
        );
    }

    #[test]
    fn a_span_longer_than_its_sequence_is_refused_not_wrapped() {
        // 5 new tokens into a sequence of 1: pre_kv_len would wrap to 2^32-4.
        assert_eq!(
            build_schedule(5, &[0, 5], &[0, 1], &[1], &[], &[], 32),
            Err(Malformed::LongerThanItsSequence { request: 0 })
        );
    }

    #[test]
    fn spans_must_cover_the_token_rows_exactly() {
        assert_eq!(
            build_schedule(5, &[0, 3, 4], &[0, 1, 2], &[3, 1], &[], &[], 32),
            Err(Malformed::TokensNotCovered {
                covered: 4,
                tokens: 5
            })
        );
    }

    #[test]
    fn a_zero_page_size_is_refused_not_defaulted() {
        assert_eq!(
            build_schedule(1, &[0, 1], &[0, 1], &[1], &[], &[], 0),
            Err(Malformed::ZeroPageSize)
        );
    }

    #[test]
    fn a_token_outside_every_span_has_no_request() {
        let qo = [0u32, 3, 4];
        assert_eq!(find_request(&qo, 0), Some(0));
        assert_eq!(find_request(&qo, 3), Some(1));
        assert_eq!(
            find_request(&qo, 4),
            None,
            "the C++ answered R-1 here, a wrong request shaped like a right one"
        );
        assert_eq!(find_request(&[], 0), None);
    }

    /// The write-descriptor formula the kernels use, held exactly.
    #[test]
    fn the_paged_gate_checks_the_kernels_address_formula() {
        let s = mixed();
        // Request 0: pages [10, 11], positions 32..35 -> page 11, offsets 0..3.
        // Request 1: page [12], position 4 -> offset 4.
        let pages = [10u32, 11, 12];
        let positions = [32u32, 33, 34, 4];
        let w_page = [11u32, 11, 11, 12];
        let w_off = [0u32, 1, 2, 4];
        assert_eq!(
            validate_paged(&s, &positions, &pages, &w_page, &w_off, 16, 16),
            Ok(())
        );

        // One wrong write offset names the token.
        let bad_off = [0u32, 1, 3, 4];
        assert_eq!(
            validate_paged(&s, &positions, &pages, &w_page, &bad_off, 16, 16),
            Err(Rejected::BadToken { token: 2 })
        );
        // A page outside the pool names the request.
        assert_eq!(
            validate_paged(&s, &positions, &[10, 11, 99], &w_page, &w_off, 16, 16),
            Err(Rejected::PageOutOfPool { request: 1 })
        );
        // A slot past the table names the request.
        assert_eq!(
            validate_paged(&s, &positions, &pages, &w_page, &w_off, 16, 8),
            Err(Rejected::SlotOutOfRange { request: 1 })
        );
    }

    #[test]
    fn the_capacity_gate_bounds_tokens_and_requests() {
        let s = mixed();
        assert_eq!(validate_capacity(&s, 4, 2), Ok(()));
        assert_eq!(validate_capacity(&s, 3, 2), Err(Rejected::OverCapacity));
        assert_eq!(validate_capacity(&s, 4, 1), Err(Rejected::OverCapacity));
    }
}
