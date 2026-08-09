//! One fire: several requests' new tokens sharing a command buffer.
//!
//! This is the shape a mixed prefill+decode batch has. A prefill
//! request contributes its whole prompt; a decode request contributes
//! one token. Nothing in the fire distinguishes them — `qo_indptr` says
//! who owns which rows, and that is the only difference.
//!
//! Validation COMPOSES the schedule machinery rather than restating it:
//! [`build_schedule`] already refuses wrapped spans and non-ascending
//! CSRs at construction, [`validate_paged`] already walks the write
//! descriptors and page lists, and a `FireCsr` that validates hands
//! back the [`BatchSchedule`] those checks built — one owner for the
//! span formula, as everywhere else in this crate.

use super::schedule::{
    BatchSchedule, Malformed, Rejected, build_schedule, validate_capacity, validate_paged,
};

/// The wire form of one fire.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FireCsr {
    /// The fired token ids, one per row.
    pub token_ids: Vec<u32>,
    /// Absolute positions, one per row.
    pub position_ids: Vec<u32>,
    /// The owning request of each row.
    pub req_of_token: Vec<u32>,
    /// The PHYSICAL page each row's KV lands in.
    pub w_page: Vec<u32>,
    /// The in-page offset of each row's KV.
    pub w_off: Vec<u32>,
    /// Per-request row spans, `requests + 1` long.
    pub qo_indptr: Vec<u32>,
    /// The flat physical page ids the attention walks.
    pub kv_page_indices: Vec<u32>,
    /// Per-request page-list spans, `requests + 1` long.
    pub kv_page_indptr: Vec<u32>,
    /// Fill count of each request's last page.
    pub kv_last_page_lens: Vec<u32>,
    /// Which rows this fire samples, in readout order. The tail runs
    /// over these and no others — the LM head is the step's most
    /// expensive dispatch by two orders of magnitude, and a prefill
    /// reads one row per request.
    pub sample_rows: Vec<u32>,
    /// Whether the device argmax runs after the tail.
    pub run_argmax: bool,
}

/// Why a fire was refused before anything was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FireRefused {
    /// A per-row array disagrees with the token count.
    RowArrays {
        /// Which array.
        what: &'static str,
        /// Its length.
        len: usize,
        /// The token count every per-row array must match.
        rows: usize,
    },
    /// The CSR itself is malformed (wrapped spans, non-ascending).
    Malformed(Malformed),
    /// The paged geometry gate refused.
    Rejected(Rejected),
    /// A sampled row is out of the fire, or there are more samples than
    /// requests — a request samples one row.
    Samples {
        /// The offending entry, or the count when it is the count.
        value: u32,
        /// The bound it broke.
        bound: u32,
    },
    /// `req_of_token` disagrees with the spans `qo_indptr` declares —
    /// two statements of the same ownership, and the kernels read BOTH.
    OwnerMismatch {
        /// The row at fault.
        row: u32,
    },
}

impl FireCsr {
    /// Rows in the fire.
    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.token_ids.len()).unwrap_or(u32::MAX)
    }

    /// Requests in the fire.
    #[must_use]
    pub fn requests(&self) -> u32 {
        u32::try_from(self.qo_indptr.len().saturating_sub(1)).unwrap_or(0)
    }

    /// Check the whole fire and hand back the schedule its checks
    /// built.
    ///
    /// # Errors
    ///
    /// [`FireRefused`] naming the first incoherence.
    pub fn validate(
        &self,
        page_size: u32,
        total_pages: u32,
        max_tokens: u32,
        max_requests: u32,
        max_slots: u32,
    ) -> Result<BatchSchedule, FireRefused> {
        let rows = self.token_ids.len();
        for (what, len) in [
            ("position_ids", self.position_ids.len()),
            ("req_of_token", self.req_of_token.len()),
            ("w_page", self.w_page.len()),
            ("w_off", self.w_off.len()),
        ] {
            if len != rows {
                return Err(FireRefused::RowArrays { what, len, rows });
            }
        }
        let schedule = build_schedule(
            self.rows(),
            &self.qo_indptr,
            &self.kv_page_indptr,
            &self.kv_last_page_lens,
            &[],
            &[],
            page_size,
        )
        .map_err(FireRefused::Malformed)?;
        validate_capacity(&schedule, max_tokens, max_requests).map_err(FireRefused::Rejected)?;
        validate_paged(
            &schedule,
            &self.position_ids,
            &self.kv_page_indices,
            &self.w_page,
            &self.w_off,
            total_pages,
            max_slots,
        )
        .map_err(FireRefused::Rejected)?;
        // The two ownership statements must agree: the append reads
        // `req_of_token`, the attention walks `qo_indptr`, and a row
        // both kernels place differently is KV written to one request's
        // pages and attended from another's.
        for (row, &req) in self.req_of_token.iter().enumerate() {
            let row32 = u32::try_from(row).expect("bounded by rows");
            let owner = super::schedule::find_request(&self.qo_indptr, row32);
            if owner != Some(req) {
                return Err(FireRefused::OwnerMismatch { row: row32 });
            }
        }
        let requests = self.requests();
        if self.sample_rows.len() > requests as usize {
            return Err(FireRefused::Samples {
                value: u32::try_from(self.sample_rows.len()).unwrap_or(u32::MAX),
                bound: requests,
            });
        }
        for &sample in &self.sample_rows {
            if sample >= self.rows() {
                return Err(FireRefused::Samples {
                    value: sample,
                    bound: self.rows(),
                });
            }
        }
        Ok(schedule)
    }

    /// One single-request fire covering `token_ids` from position 0 — a
    /// whole-prompt prefill sampling its last row, with every page in
    /// one list. The shape all three device smokes hand-rolled.
    #[must_use]
    pub fn prefill(token_ids: Vec<u32>, page_size: u32, total_pages: u32) -> FireCsr {
        let n = u32::try_from(token_ids.len()).expect("a prompt is bounded");
        let positions: Vec<u32> = (0..n).collect();
        FireCsr {
            position_ids: positions.clone(),
            req_of_token: vec![0; token_ids.len()],
            w_page: positions.iter().map(|p| p / page_size.max(1)).collect(),
            w_off: positions.iter().map(|p| p % page_size.max(1)).collect(),
            qo_indptr: vec![0, n],
            kv_page_indices: (0..total_pages).collect(),
            kv_page_indptr: vec![0, total_pages],
            kv_last_page_lens: vec![if page_size == 0 {
                1
            } else {
                ((n - 1) % page_size) + 1
            }],
            sample_rows: vec![n.saturating_sub(1)],
            run_argmax: false,
            token_ids,
        }
    }

    /// One decode row: `token` at absolute `position`, request 0, the
    /// page list covering its whole history, sampling its one row.
    #[must_use]
    pub fn decode(token: u32, position: u32, page_size: u32) -> FireCsr {
        let ps = page_size.max(1);
        let pages = (position + 1).div_ceil(ps);
        FireCsr {
            token_ids: vec![token],
            position_ids: vec![position],
            req_of_token: vec![0],
            w_page: vec![position / ps],
            w_off: vec![position % ps],
            qo_indptr: vec![0, 1],
            kv_page_indices: (0..pages).collect(),
            kv_page_indptr: vec![0, pages],
            kv_last_page_lens: vec![(position % ps) + 1],
            sample_rows: vec![0],
            run_argmax: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefill_validates_and_hands_back_its_schedule() {
        let csr = FireCsr::prefill(vec![7; 17], 32, 128);
        let schedule = csr.validate(32, 128, 64, 1, 1).expect("a coherent fire");
        assert_eq!(schedule.total_tokens, 17);
        assert_eq!(schedule.requests, 1);
        assert_eq!(csr.sample_rows, vec![16]);
    }

    #[test]
    fn the_two_ownership_statements_must_agree() {
        // The append reads req_of_token, the attention walks qo_indptr;
        // a row they place differently is KV written to one request's
        // pages and attended from another's.
        let mut csr = FireCsr::prefill(vec![7; 8], 32, 4);
        csr.req_of_token[3] = 1;
        assert_eq!(
            csr.validate(32, 4, 64, 1, 1),
            Err(FireRefused::OwnerMismatch { row: 3 })
        );
    }

    #[test]
    fn samples_and_row_arrays_are_bounded() {
        let mut csr = FireCsr::prefill(vec![7; 8], 32, 4);
        csr.sample_rows = vec![8];
        assert!(matches!(
            csr.validate(32, 4, 64, 1, 1),
            Err(FireRefused::Samples { value: 8, bound: 8 })
        ));
        let mut csr = FireCsr::prefill(vec![7; 8], 32, 4);
        csr.sample_rows = vec![0, 1];
        assert!(
            matches!(
                csr.validate(32, 4, 64, 1, 1),
                Err(FireRefused::Samples { .. })
            ),
            "one sample per request"
        );
        let mut csr = FireCsr::prefill(vec![7; 8], 32, 4);
        csr.w_page.pop();
        assert!(matches!(
            csr.validate(32, 4, 64, 1, 1),
            Err(FireRefused::RowArrays { what: "w_page", .. })
        ));
    }
}
