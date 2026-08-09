//! One member's forward, described: the per-member slice of a launch.
//!
//! A wire launch carries every member's tokens, pages and sampling rows in
//! shared CSR arrays; a device-geometry member instead arrives with its
//! geometry already resolved from channel descriptors. Either way the
//! forward executor wants one self-contained description per member — its
//! tokens, its page list, its recurrent-state slots, its readout rows, and
//! the derived extents every symbolic shape resolves against. That
//! description is [`ForwardDesc`] and [`build_member_desc`] is the slice,
//! ported from `compose.cpp`'s `build_member_forward_desc`.
//!
//! ## The recorded bug this keeps fixed
//!
//! The launch's recurrent-state arrays are indexed one of two ways, and the
//! difference only shows once a batch carries more than one member: either
//! they are scoped to the resolved member's *requests*, or they are
//! launch-wide with one entry per *member*. The shipped C++ read the
//! launch-wide form from index 0 — member 1 got member 0's slot — and
//! rejected any batch whose member count differed from one member's request
//! count, so two concurrent decodes could never share a forward. The fix is
//! in the C++ now and the two forms are matched explicitly here, in the
//! same order, with a test that member 1 gets its own slot.
//!
//! ## What is typed away
//!
//! * `kv_last_page_len = 0` meant "derive it later" — a sentinel sharing a
//!   field with real fill counts (a full page is `page_size`, never 0, so
//!   it happens to be unambiguous; it is still a convention every reader
//!   must know). It is [`Option`] here, and the derivation
//!   ([`ForwardDesc::key_len`]'s three-way fallback) is a named function
//!   instead of a nested ternary.
//! * The `bool has_rs_slot` + `rs_slot_id` + `rs_reset` triple — the
//!   member-level mirror of `request_rs_*[0]` — is the [`ForwardDesc::rs_slot`]
//!   accessor: derived, not stored, so it cannot disagree with the vectors.
//! * `StructuredMaskDescriptor` reduces to the one bit this driver reads:
//!   a structured mask with no dense fallback is refused (Metal has no
//!   structured attention), so the descriptor's kind/sink/window never
//!   matter here. The full type belongs to the descriptor-resolve port.

use crate::pipeline::Extents;
use driver_abi::local::PIE_RS_FLAG_RESET;
use driver_abi::plan::LaunchPlan;

/// Channel-resolved geometry for one device-geometry member.
///
/// The consumed subset of the C++ `FireGeometry`: what
/// [`build_member_desc`] reads when a member's geometry came from
/// descriptor resolution rather than the wire. Its producer is the
/// descriptor-resolve port; until then tests build it directly.
#[derive(Clone, Debug, Default)]
pub struct ResolvedGeometry {
    /// The member's tokens, in order.
    pub token_ids: Vec<u32>,
    /// Absolute positions, parallel to `token_ids`.
    pub position_ids: Vec<u32>,
    /// Per-request token CSR.
    pub qo_indptr: Vec<u32>,
    /// The physical page list, CSR-trimmed.
    pub kv_page_indices: Vec<u32>,
    /// Per-request page CSR.
    pub kv_page_indptr: Vec<u32>,
    /// Per-request final page fill.
    pub kv_last_page_lens: Vec<u32>,
    /// Readout rows, member-local.
    pub sampling_indices: Vec<u32>,
    /// Per-request readout CSR.
    pub sampling_indptr: Vec<u32>,
    /// Whether `w_page`/`w_off` carry an explicit write descriptor.
    pub has_write_desc: bool,
    /// Physical page id per token.
    pub w_page: Vec<u32>,
    /// In-page offset per token.
    pub w_off: Vec<u32>,
    /// Whether `mask` is a dense attention mask.
    pub has_mask: bool,
    /// Dense mask bytes, `[tokens, stride]` row-major.
    pub mask: Vec<u8>,
    /// A structured mask was declared (window/sink); dense `mask` may or may
    /// not have been materialized alongside it.
    pub structured_mask: bool,
}

/// One member's forward, self-contained.
///
/// `MemberForwardDesc`. Field meanings follow the C++ exactly; the derived
/// counts at the bottom are what
/// [`extents`](Self::extents)/[`extents_from_readout`](Self::extents_from_readout)
/// copy into the launch path's [`Extents`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForwardDesc {
    /// The engine's stable identity for this instance — distinguishes "the
    /// same conversation continuing" from "a different one", which the
    /// physical page numbering alone cannot.
    pub sequence_id: u64,
    /// This fire's new tokens, in order.
    pub token_ids: Vec<u32>,
    /// Absolute positions, parallel to `token_ids`.
    pub position_ids: Vec<u32>,
    /// The member's full historical page list.
    pub kv_pages: Vec<u32>,
    /// Final page fill after this fire, when the launch stated it; `None`
    /// derives it from the positions when the key length is computed.
    pub kv_last_page_len: Option<u32>,
    /// Per-request token CSR (member-scoped).
    pub qo_indptr: Vec<u32>,
    /// Per-request page CSR (member-scoped).
    pub kv_page_indptr: Vec<u32>,
    /// Per-request final page fills (member-scoped).
    pub kv_last_page_lens: Vec<u32>,
    /// Folded recurrent-state slot per request.
    pub request_rs_slot_ids: Vec<u32>,
    /// Whether each request starts a fresh sequence (`PIE_RS_FLAG_RESET`).
    pub request_rs_reset: Vec<bool>,
    /// Whether each request's state is read before use (`!reset`).
    pub request_rs_read: Vec<bool>,
    /// Whether each request's state is written after use (always, today).
    pub request_rs_write: Vec<bool>,
    /// Whether `w_page`/`w_off` carry an explicit write descriptor. Never
    /// silently dropped: the forward rejects it rather than running the
    /// implicit-append path it contradicts.
    pub has_write_desc: bool,
    /// Physical page id per token.
    pub w_page: Vec<u32>,
    /// In-page offset per token.
    pub w_off: Vec<u32>,
    /// The member needs the paged path (device geometry or wire pages).
    pub requires_paged: bool,
    /// Dense attention mask bytes, empty when none.
    pub attention_mask: Vec<u8>,
    /// Keys per mask row: `attention_mask.len() / token_ids.len()`.
    pub attention_mask_stride: u32,
    /// A structured mask was declared. With no dense fallback it is refused
    /// at build.
    pub structured_mask: bool,
    /// Local indices into `token_ids` whose logits must be materialized.
    pub readout_local_indices: Vec<u32>,
    /// Per-request readout CSR.
    pub sampling_indptr: Vec<u32>,
    /// Requests in this member.
    pub row_count: u32,
    /// Rows read out for sampling.
    pub sampled_rows: u32,
    /// Tokens this fire contributes.
    pub token_count: u32,
    /// Pages backing the member's KV.
    pub page_count: u32,
    /// Attention query length (== `token_count`).
    pub query_len: u32,
    /// Attention key length after this fire.
    pub key_len: u32,
    /// Highest absolute position plus one.
    pub kv_len: u32,
    /// Prefer the device's greedy token and skip full-logits staging, where
    /// the family supports it. An optimization request, not a capability.
    pub greedy_token_only: bool,
}

impl ForwardDesc {
    /// The member-level recurrent-state slot: the first request's, with its
    /// reset bit.
    ///
    /// The C++ stored `has_rs_slot`/`rs_slot_id`/`rs_reset` beside the
    /// per-request vectors they mirror; derived here, so they cannot
    /// disagree.
    #[must_use]
    pub fn rs_slot(&self) -> Option<(u32, bool)> {
        let id = self.request_rs_slot_ids.first().copied()?;
        Some((id, self.request_rs_reset.first().copied().unwrap_or(false)))
    }

    /// The launch-path extents, with the sampled-row count the caller
    /// observed. `m1_extents_from_forward_desc`.
    #[must_use]
    pub fn extents(&self, sampled_rows: u32) -> Extents {
        Extents {
            kv_len: self.kv_len,
            page_count: self.page_count,
            row_count: self.row_count,
            token_count: self.token_count,
            sampled_rows,
            query_len: self.query_len,
            key_len: self.key_len,
        }
    }

    /// [`extents`](Self::extents) with the readout count as the sampled
    /// rows. `m3_extents_from_forward_desc`.
    #[must_use]
    pub fn extents_from_readout(&self) -> Extents {
        self.extents(self.readout_local_indices.len() as u32)
    }
}

/// Why a member's description could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// A page size of zero has no geometry. The C++ silently clamped to 1.
    ZeroPageSize,
    /// The launch has no per-member token CSR.
    MissingTokenCsr,
    /// This member's token span leaves the token arrays.
    BadTokenSpan,
    /// The page CSR does not have one entry per member plus one.
    BadPageCsr,
    /// This member's page span leaves the page array.
    BadPageSpan,
    /// A hybrid-attention model with no usable recurrent-state assignment.
    MissingRsAssignment,
    /// The resolved RS arrays fit neither the per-request nor the
    /// launch-wide shape.
    AmbiguousRsAssignment,
    /// The sampling CSR does not have one entry per member plus one.
    BadSamplingCsr,
    /// This member's sampling span leaves the index array.
    BadSamplingSpan,
    /// A dense mask whose byte count is not a whole number of token rows.
    BadMaskShape,
    /// A structured mask with no dense fallback; Metal has no structured
    /// attention.
    StructuredMaskUnsupported,
}

impl BuildError {
    /// One line, for the refusal.
    #[must_use]
    pub fn reason(self) -> &'static str {
        match self {
            BuildError::ZeroPageSize => "a page size of zero has no geometry",
            BuildError::MissingTokenCsr => {
                "launch is missing qo_indptr for a forward-needing member"
            }
            BuildError::BadTokenSpan => "malformed qo_indptr/token_ids for this member",
            BuildError::BadPageCsr => "malformed kv_page_indptr for this launch",
            BuildError::BadPageSpan => "malformed kv_page_indices for this member",
            BuildError::MissingRsAssignment => {
                "missing folded recurrent-state slot assignment for a hybrid-attention model"
            }
            BuildError::AmbiguousRsAssignment => {
                "resolved hybrid geometry requires exactly one folded recurrent-state slot \
                 and flag per request"
            }
            BuildError::BadSamplingCsr => "malformed sampling_indptr for this launch",
            BuildError::BadSamplingSpan => "malformed sampling_indices for this member",
            BuildError::BadMaskShape => "resolved attention mask has an invalid dense shape",
            BuildError::StructuredMaskUnsupported => {
                "structured attention mask has no dense fallback; direct structured Metal \
                 attention is not supported"
            }
        }
    }
}

/// One member's span of a CSR, checked against the array it indexes.
fn member_span(indptr: &[u32], member: usize, bound: usize) -> Option<(usize, usize)> {
    let begin = *indptr.get(member)? as usize;
    let end = *indptr.get(member + 1)? as usize;
    (begin <= end && end <= bound).then_some((begin, end))
}

/// Build one member's forward description.
///
/// `resolved` is the channel-resolved geometry for a device-geometry
/// member; `None` slices the wire launch's shared arrays at `member`.
/// `has_linear_attn` is the family fact that makes the recurrent-state
/// assignment mandatory.
///
/// # Errors
///
/// [`BuildError`], with the C++'s own words in
/// [`reason`](BuildError::reason).
#[allow(clippy::too_many_lines)]
pub fn build_member_desc(
    plan: &LaunchPlan,
    member: usize,
    member_count: usize,
    has_linear_attn: bool,
    page_size: u32,
    resolved: Option<&ResolvedGeometry>,
) -> Result<ForwardDesc, BuildError> {
    if page_size == 0 {
        return Err(BuildError::ZeroPageSize);
    }
    let mut desc = ForwardDesc::default();

    if let Some(resolved) = resolved {
        desc.token_ids = resolved.token_ids.clone();
        desc.position_ids = resolved.position_ids.clone();
        desc.kv_pages = resolved.kv_page_indices.clone();
        desc.qo_indptr = resolved.qo_indptr.clone();
        desc.kv_page_indptr = resolved.kv_page_indptr.clone();
        desc.kv_last_page_lens = resolved.kv_last_page_lens.clone();
        desc.sampling_indptr = resolved.sampling_indptr.clone();
        desc.kv_last_page_len = match resolved.kv_last_page_lens.as_slice() {
            [only] => Some(*only),
            _ => None,
        };
        desc.readout_local_indices = resolved.sampling_indices.clone();
        desc.has_write_desc = resolved.has_write_desc;
        desc.w_page = resolved.w_page.clone();
        desc.w_off = resolved.w_off.clone();
        desc.requires_paged = true;
        desc.structured_mask = resolved.structured_mask;
        if resolved.has_mask {
            if desc.token_ids.is_empty()
                || resolved.mask.is_empty()
                || !resolved.mask.len().is_multiple_of(desc.token_ids.len())
            {
                return Err(BuildError::BadMaskShape);
            }
            desc.attention_mask = resolved.mask.clone();
            desc.attention_mask_stride = (resolved.mask.len() / desc.token_ids.len()) as u32;
        } else if resolved.structured_mask {
            return Err(BuildError::StructuredMaskUnsupported);
        }
        desc.row_count = resolved
            .qo_indptr
            .len()
            .checked_sub(1)
            .map_or(1, |rows| rows.max(1)) as u32;
        // Per-row key length, when the per-request CSR is coherent: the
        // longest row bounds the attention read.
        if resolved.kv_page_indptr.len() == desc.row_count as usize + 1
            && resolved.kv_last_page_lens.len() == desc.row_count as usize
        {
            for row in 0..desc.row_count as usize {
                let pages =
                    resolved.kv_page_indptr[row + 1].saturating_sub(resolved.kv_page_indptr[row]);
                let length = if pages == 0 {
                    0
                } else {
                    (pages - 1) * page_size + resolved.kv_last_page_lens[row]
                };
                desc.key_len = desc.key_len.max(length);
            }
        }
    } else {
        if plan.qo_indptr.len() != member_count + 1 {
            return Err(BuildError::MissingTokenCsr);
        }
        let (begin, end) = member_span(
            &plan.qo_indptr,
            member,
            plan.token_ids.len().min(plan.position_ids.len()),
        )
        .ok_or(BuildError::BadTokenSpan)?;
        desc.token_ids = plan.token_ids[begin..end].to_vec();
        desc.position_ids = plan.position_ids[begin..end].to_vec();

        if !plan.kv_page_indptr.is_empty() {
            if plan.kv_page_indptr.len() != member_count + 1 {
                return Err(BuildError::BadPageCsr);
            }
            let (page_begin, page_end) =
                member_span(&plan.kv_page_indptr, member, plan.kv_page_indices.len())
                    .ok_or(BuildError::BadPageSpan)?;
            desc.kv_pages = plan.kv_page_indices[page_begin..page_end].to_vec();
            if plan.kv_last_page_lens.len() == member_count {
                desc.kv_last_page_len = Some(plan.kv_last_page_lens[member]);
            }
            // A wire fire that names KV pages is paged, exactly as a
            // resolved one is: the sealed M=1 ring path used to claim every
            // wire fire, so a prefill posted on the wire landed in the ring
            // while the decode continuing it — device-resolved, therefore
            // paged — could not find its history.
            if !desc.kv_pages.is_empty() {
                desc.requires_paged = true;
                desc.qo_indptr = vec![0, desc.token_ids.len() as u32];
                desc.kv_page_indptr = vec![0, desc.kv_pages.len() as u32];
                desc.kv_last_page_lens = vec![desc.kv_last_page_len.unwrap_or(0)];
            }
        }
    }

    let request_count = match resolved {
        Some(resolved) if resolved.qo_indptr.len() >= 2 => resolved.qo_indptr.len() - 1,
        _ => 1,
    };
    if has_linear_attn {
        let (ids, flags): (&[u32], &[u8]) = if resolved.is_some() {
            // The two indexings of the RS arrays; see the module docs for
            // the bug their conflation shipped.
            if plan.rs_slot_ids.len() == request_count && plan.rs_slot_flags.len() == request_count
            {
                (&plan.rs_slot_ids, &plan.rs_slot_flags)
            } else if request_count == 1
                && plan.rs_slot_ids.len() == member_count
                && plan.rs_slot_flags.len() == member_count
            {
                (
                    &plan.rs_slot_ids[member..=member],
                    &plan.rs_slot_flags[member..=member],
                )
            } else {
                return Err(BuildError::AmbiguousRsAssignment);
            }
        } else if plan.rs_slot_ids.len() == member_count && plan.rs_slot_flags.len() == member_count
        {
            (
                &plan.rs_slot_ids[member..=member],
                &plan.rs_slot_flags[member..=member],
            )
        } else {
            return Err(BuildError::MissingRsAssignment);
        };
        desc.request_rs_slot_ids = ids.to_vec();
        desc.request_rs_reset = flags.iter().map(|&f| f & PIE_RS_FLAG_RESET != 0).collect();
        desc.request_rs_read = desc.request_rs_reset.iter().map(|&reset| !reset).collect();
        desc.request_rs_write = vec![true; desc.request_rs_reset.len()];
    }

    if resolved.is_none() && !plan.sampling_indptr.is_empty() {
        if plan.sampling_indptr.len() != member_count + 1 {
            return Err(BuildError::BadSamplingCsr);
        }
        let (begin, end) = member_span(&plan.sampling_indptr, member, plan.sampling_indices.len())
            .ok_or(BuildError::BadSamplingSpan)?;
        desc.readout_local_indices = plan.sampling_indices[begin..end].to_vec();
    }

    if desc.qo_indptr.is_empty() {
        desc.qo_indptr = vec![0, desc.token_ids.len() as u32];
    }
    if desc.kv_page_indptr.is_empty() {
        desc.kv_page_indptr = vec![0, desc.kv_pages.len() as u32];
    }
    if desc.kv_last_page_lens.is_empty() {
        desc.kv_last_page_lens = vec![desc.kv_last_page_len.unwrap_or(0)];
    }
    if desc.sampling_indptr.is_empty() {
        desc.sampling_indptr = vec![0, desc.readout_local_indices.len() as u32];
    }

    desc.sampled_rows = desc.readout_local_indices.len() as u32;
    desc.token_count = desc.token_ids.len() as u32;
    desc.page_count = desc.kv_pages.len() as u32;
    desc.query_len = desc.token_count;
    desc.kv_len = desc
        .position_ids
        .iter()
        .map(|&position| position + 1)
        .max()
        .unwrap_or(0);
    if desc.key_len == 0 {
        desc.key_len = derive_key_len(&desc, page_size);
    }
    Ok(desc)
}

/// The attention key length, when nothing upstream stated it.
///
/// Pageless members read exactly their positions (`kv_len`). Paged members
/// read their whole page list, whose final page holds the stated fill when
/// the launch stated one, and otherwise the fill the last position implies.
/// The C++ wrote this as a three-deep nested ternary at the bottom of a
/// 260-line function.
fn derive_key_len(desc: &ForwardDesc, page_size: u32) -> u32 {
    if desc.kv_pages.is_empty() {
        return desc.kv_len;
    }
    let last_fill = desc.kv_last_page_len.unwrap_or_else(|| {
        desc.position_ids
            .last()
            .map_or(0, |&position| position % page_size + 1)
    });
    (desc.kv_pages.len() as u32 - 1) * page_size + last_fill
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-member wire launch: a 3-token prefill and a 1-token decode.
    fn wire_plan() -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![10, 11, 12, 20],
            position_ids: vec![0, 1, 2, 4],
            qo_indptr: vec![0, 3, 4],
            kv_page_indices: vec![7, 8, 9],
            kv_page_indptr: vec![0, 2, 3],
            kv_last_page_lens: vec![3, 5],
            sampling_indices: vec![2, 0],
            sampling_indptr: vec![0, 1, 2],
            rs_slot_ids: vec![4, 6],
            rs_slot_flags: vec![PIE_RS_FLAG_RESET, 0],
            ..LaunchPlan::default()
        }
    }

    #[test]
    fn a_wire_member_is_sliced_out_of_the_shared_arrays() {
        let desc = build_member_desc(&wire_plan(), 0, 2, false, 32, None).expect("member 0");
        assert_eq!(desc.token_ids, [10, 11, 12]);
        assert_eq!(desc.position_ids, [0, 1, 2]);
        assert_eq!(desc.kv_pages, [7, 8]);
        assert!(desc.requires_paged);
        assert_eq!(desc.qo_indptr, [0, 3], "member-scoped, not launch-scoped");
        assert_eq!(desc.kv_page_indptr, [0, 2]);
        assert_eq!(desc.kv_last_page_lens, [3]);
        assert_eq!(desc.readout_local_indices, [2]);
        assert_eq!(
            (desc.token_count, desc.page_count, desc.sampled_rows),
            (3, 2, 1)
        );
        assert_eq!(desc.kv_len, 3);
        assert_eq!(desc.key_len, 32 + 3, "one full page plus the stated fill");

        let second = build_member_desc(&wire_plan(), 1, 2, false, 32, None).expect("member 1");
        assert_eq!(second.token_ids, [20]);
        assert_eq!(second.kv_pages, [9]);
        assert_eq!(second.readout_local_indices, [0]);
        assert_eq!(second.key_len, 5);
    }

    #[test]
    fn a_pageless_member_reads_exactly_its_positions() {
        let mut plan = wire_plan();
        plan.kv_page_indices.clear();
        plan.kv_page_indptr.clear();
        plan.kv_last_page_lens.clear();
        let desc = build_member_desc(&plan, 0, 2, false, 32, None).expect("member");
        assert!(!desc.requires_paged);
        assert_eq!(desc.key_len, desc.kv_len);
        assert_eq!(desc.kv_page_indptr, [0, 0], "the default CSR still exists");
    }

    #[test]
    fn an_unstated_final_fill_is_derived_from_the_last_position() {
        let mut plan = wire_plan();
        plan.kv_last_page_lens.clear();
        let desc = build_member_desc(&plan, 0, 2, false, 32, None).expect("member");
        assert_eq!(desc.kv_last_page_len, None, "the sentinel is an absence");
        // Last position 2 -> fill 3 on the final page.
        assert_eq!(desc.key_len, 32 + 3);
    }

    #[test]
    fn the_wire_rs_assignment_is_per_member_and_reset_masks_the_flag() {
        let desc = build_member_desc(&wire_plan(), 0, 2, true, 32, None).expect("member 0");
        assert_eq!(desc.rs_slot(), Some((4, true)));
        assert_eq!(
            desc.request_rs_read,
            [false],
            "a fresh sequence reads nothing"
        );
        assert_eq!(desc.request_rs_write, [true]);

        let second = build_member_desc(&wire_plan(), 1, 2, true, 32, None).expect("member 1");
        assert_eq!(second.rs_slot(), Some((6, false)));
        assert_eq!(second.request_rs_read, [true]);

        let mut missing = wire_plan();
        missing.rs_slot_ids.clear();
        assert_eq!(
            build_member_desc(&missing, 0, 2, true, 32, None),
            Err(BuildError::MissingRsAssignment)
        );
    }

    #[test]
    fn malformed_spans_are_named_not_sliced() {
        let mut plan = wire_plan();
        plan.qo_indptr = vec![0, 3];
        assert_eq!(
            build_member_desc(&plan, 0, 2, false, 32, None),
            Err(BuildError::MissingTokenCsr)
        );
        let mut plan = wire_plan();
        plan.qo_indptr = vec![0, 9, 10];
        assert_eq!(
            build_member_desc(&plan, 0, 2, false, 32, None),
            Err(BuildError::BadTokenSpan)
        );
        let mut plan = wire_plan();
        plan.sampling_indptr = vec![0, 9, 10];
        assert_eq!(
            build_member_desc(&plan, 0, 2, false, 32, None),
            Err(BuildError::BadSamplingSpan)
        );
        assert_eq!(
            build_member_desc(&wire_plan(), 0, 2, false, 0, None),
            Err(BuildError::ZeroPageSize),
            "the C++ silently clamped a zero page size to one"
        );
    }

    fn resolved_two_rows() -> ResolvedGeometry {
        ResolvedGeometry {
            token_ids: vec![1, 2],
            position_ids: vec![3, 7],
            qo_indptr: vec![0, 1, 2],
            kv_page_indices: vec![5, 6, 7],
            kv_page_indptr: vec![0, 1, 3],
            kv_last_page_lens: vec![4, 8],
            sampling_indices: vec![0, 1],
            sampling_indptr: vec![0, 1, 2],
            ..ResolvedGeometry::default()
        }
    }

    #[test]
    fn resolved_geometry_is_adopted_with_the_longest_rows_key_len() {
        let plan = LaunchPlan::default();
        let desc = build_member_desc(&plan, 0, 1, false, 32, Some(&resolved_two_rows()))
            .expect("resolved");
        assert!(desc.requires_paged);
        assert_eq!(desc.row_count, 2);
        assert_eq!(desc.kv_last_page_len, None, "two rows, no single fill");
        // Row 0: 1 page, fill 4 -> 4. Row 1: 2 pages, fill 8 -> 40.
        assert_eq!(desc.key_len, 40);
        assert_eq!(desc.kv_len, 8, "positions still bound kv_len");
    }

    #[test]
    fn the_launch_wide_rs_form_gives_each_member_its_own_slot() {
        // Two members, one request each: the launch-wide form. The shipped
        // bug read index 0 for every member.
        let plan = LaunchPlan {
            rs_slot_ids: vec![4, 6],
            rs_slot_flags: vec![0, PIE_RS_FLAG_RESET],
            ..LaunchPlan::default()
        };
        let resolved = ResolvedGeometry {
            token_ids: vec![9],
            position_ids: vec![0],
            qo_indptr: vec![0, 1],
            ..ResolvedGeometry::default()
        };
        let desc = build_member_desc(&plan, 1, 2, true, 32, Some(&resolved)).expect("member 1");
        assert_eq!(
            desc.rs_slot(),
            Some((6, true)),
            "member 1 must get member 1's slot, not member 0's"
        );

        // Per-request form: two rows in one member.
        let plan = LaunchPlan {
            rs_slot_ids: vec![4, 6],
            rs_slot_flags: vec![0, 0],
            ..LaunchPlan::default()
        };
        let desc = build_member_desc(&plan, 0, 1, true, 32, Some(&resolved_two_rows()))
            .expect("two requests");
        assert_eq!(desc.request_rs_slot_ids, [4, 6]);

        // Neither shape fits: refused, not guessed.
        let plan = LaunchPlan {
            rs_slot_ids: vec![4, 6, 9],
            rs_slot_flags: vec![0, 0, 0],
            ..LaunchPlan::default()
        };
        assert_eq!(
            build_member_desc(&plan, 0, 2, true, 32, Some(&resolved_two_rows())),
            Err(BuildError::AmbiguousRsAssignment)
        );
    }

    #[test]
    fn a_dense_mask_needs_a_whole_number_of_rows_and_structured_needs_dense() {
        let mut resolved = resolved_two_rows();
        resolved.has_mask = true;
        resolved.mask = vec![0; 7];
        assert_eq!(
            build_member_desc(&LaunchPlan::default(), 0, 1, false, 32, Some(&resolved)),
            Err(BuildError::BadMaskShape)
        );
        resolved.mask = vec![1; 10];
        let desc = build_member_desc(&LaunchPlan::default(), 0, 1, false, 32, Some(&resolved))
            .expect("dense mask");
        assert_eq!(desc.attention_mask_stride, 5);

        let mut resolved = resolved_two_rows();
        resolved.structured_mask = true;
        assert_eq!(
            build_member_desc(&LaunchPlan::default(), 0, 1, false, 32, Some(&resolved)),
            Err(BuildError::StructuredMaskUnsupported)
        );
    }

    #[test]
    fn the_extents_are_the_descs_numbers_with_the_callers_rows() {
        let desc = build_member_desc(&wire_plan(), 0, 2, false, 32, None).expect("member");
        let extents = desc.extents(9);
        assert_eq!(extents.sampled_rows, 9, "the caller's observation wins");
        assert_eq!(extents.kv_len, desc.kv_len);
        assert_eq!(extents.key_len, desc.key_len);
        assert_eq!(
            desc.extents_from_readout().sampled_rows,
            desc.readout_local_indices.len() as u32
        );
    }
}
