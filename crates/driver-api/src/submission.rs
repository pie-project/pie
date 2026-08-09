//! [`FrameSubmission`] — the sealed-frame request the scheduler hands to a
//! driver backend's `launch` verb (ABI v14). A frame carries its lane roster
//! and frame-invariant tables once, plus one [`StepSubmission`] per forward
//! step; the driver executes the steps as one closed system with a single
//! completion.
//!
//! It lives HERE and not in the engine because a driver's `launch` verb takes
//! it. A Rust-linked backend that had to name the engine's copy would have to
//! depend on the engine, and the engine depends on it — so the type that
//! states the request would be the one thing making the dependency a cycle.
//! `driver-cuda` never hit this only because its `launch` crosses a C ABI and
//! reads [`PieFrameDesc`](crate::local::PieFrameDesc) instead
//! (`.wiki/driver/real-metal-north-star.md` §9, "one door").

use crate::geometry::GeometryClass;
use crate::local::{
    PIE_RS_FLAG_BUFFER_WRITE, PIE_RS_FLAG_FOLD, PIE_RS_FLAG_FOLD_LEN_DEVICE, PIE_RS_FLAG_RESET,
    TerminalCell,
};
use crate::plan::{LaunchPlan, Malformed};

fn bad<T>(why: impl Into<String>) -> Result<T, Malformed> {
    Err(Malformed(why.into()))
}

/// One CSR: `indptr` partitions `values_len` into `outer_count` segments.
///
/// Empty is the documented "not supplied" default when `allow_empty`, which is
/// how most of these members state absence.
fn csr(
    indptr: &[u32],
    name: &str,
    values_len: usize,
    outer_count: usize,
    allow_empty: bool,
) -> Result<(), Malformed> {
    if indptr.is_empty() {
        return if allow_empty {
            Ok(())
        } else {
            bad(format!("{name} is required and empty"))
        };
    }
    if indptr.len() != outer_count + 1 {
        return bad(format!(
            "{name} has {} entries, not the {} its {outer_count} segments need",
            indptr.len(),
            outer_count + 1
        ));
    }
    if indptr[0] != 0 {
        return bad(format!("{name} starts at {}, not 0", indptr[0]));
    }
    if let Some(w) = indptr.windows(2).find(|w| w[0] > w[1]) {
        return bad(format!("{name} decreases: {} then {}", w[0], w[1]));
    }
    let last = *indptr.last().unwrap_or(&0) as usize;
    if last > values_len {
        return bad(format!(
            "{name} ends at {last}, past the {values_len} values"
        ));
    }
    Ok(())
}

/// One vector that must carry exactly `outer_count` entries, or none when
/// `allow_empty`.
fn rows(len: usize, name: &str, outer_count: usize, allow_empty: bool) -> Result<(), Malformed> {
    if len == 0 && allow_empty {
        return Ok(());
    }
    if len == outer_count {
        Ok(())
    } else {
        bad(format!("{name} has {len} entries, not {outer_count}"))
    }
}

/// One forward step: the batch geometry (wire form) plus per-step metadata.
/// Batch members reference the frame roster through `roster_rows` and are
/// partitioned into ordered geometry-homogeneous sub-batches.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepSubmission {
    pub plan: LaunchPlan,
    /// Indices into [`FrameSubmission::instance_ids`], one per batch member,
    /// in sub-batch order.
    pub roster_rows: Vec<u32>,
    /// CSR over `roster_rows`; sub-batch `b` spans members
    /// `[sub_batch_indptr[b], sub_batch_indptr[b+1])`.
    pub sub_batch_indptr: Vec<u32>,
    /// `PIE_GEOMETRY_CLASS_*` per sub-batch.
    pub sub_batch_class: Vec<u32>,
    pub terminal_cells: Vec<*mut TerminalCell>,
    /// Program → wire-request attribution CSR (`roster_rows.len() + 1`
    /// entries): member `p` owns wire request rows
    /// `[row_indptr[p], row_indptr[p+1])`. Batched fires contribute one row
    /// each (a device-geometry fire's row is an empty placeholder the driver
    /// replaces with channel-resolved geometry).
    pub program_row_indptr: Vec<u32>,
    pub logical_fire_ids: Vec<u64>,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
    pub channel_ticket_indptr: Vec<u32>,
    /// tart rung ③ (0.3 re-port): the region table — the seriation's
    /// output; the driver derives every planned split from it. Empty =
    /// no table (legacy discipline).
    pub region_row_indptr: Vec<u32>,
    /// Axis bitset per region (`PIE_REGION_SIG_*`).
    pub region_sig: Vec<u32>,
    /// Depth operand per region (`PIE_MAX_LAYERS_FULL` = full).
    pub region_k: Vec<u32>,
}

/// The sealed frame handed to `DriverBackend::launch`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameSubmission {
    /// Lane roster: every bound instance participating in any step, in
    /// scheduler order. No duplicates.
    pub instance_ids: Vec<u64>,
    /// Frame-union WorkingSet page translation (committed mapping overlaid
    /// with ALL steps' prepared write targets) + its CSR partition, one
    /// segment per roster entry.
    pub kv_translation: Vec<u32>,
    pub kv_translation_indptr: Vec<u32>,
    /// Exclusive physical KV page high-water after the LAST step — the
    /// frame-union admission demand.
    pub required_kv_pages: u32,
    /// The frame's steps in execution order. Never empty.
    pub steps: Vec<StepSubmission>,
}

impl StepSubmission {
    /// Every rule `validate_step_desc` stated about a step, minus the ones
    /// the owned shape makes unstatable.
    ///
    /// # What did not survive, and why it did not need to
    ///
    /// Roughly forty of that validator's checks were `ptr/len mismatch`: a
    /// null pointer with a nonzero length, or a length with no pointer. A
    /// `Vec` cannot be in that state, so those are not weakened rules — they
    /// are rules about a representation that is gone. Likewise
    /// `single_token_mode`/`has_user_mask` "must be 0 or 1": they are `bool`.
    ///
    /// # What was NOT loosened
    ///
    /// `terminal_cells.len == roster_rows.len` is unconditional, as it was.
    /// A stale comment in `driver-cuda`'s `pie_cuda_launch` claimed the rule
    /// was too strict to adopt; the code under it had adopted it, and
    /// `entry_validation::no_validator_is_deferred` records what it caught
    /// when it did — fixtures with no terminal cell at all, and two steps
    /// sharing one, which would have had a frame report whichever finished
    /// last as the answer for both. A frame that states no cells is a frame
    /// whose members cannot be told apart on completion.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the member and the numbers that disagree.
    pub fn validate(&self, roster_len: usize) -> Result<(), Malformed> {
        let plan = &self.plan;
        let requests = self.roster_rows.len();
        let wire_rows = plan.qo_indptr.len().saturating_sub(1);

        csr(
            &self.sub_batch_indptr,
            "sub_batch_indptr",
            requests,
            self.sub_batch_class.len(),
            true,
        )?;
        if let Some(&class) = self
            .sub_batch_class
            .iter()
            .find(|&&c| GeometryClass::try_from(c).is_err())
        {
            return bad(format!("sub_batch_class names no geometry class: {class}"));
        }

        // The roster indices, and the two distinctness rules that keep one
        // member from standing in for another.
        for (index, &row) in self.roster_rows.iter().enumerate() {
            if row as usize >= roster_len {
                return bad(format!(
                    "roster_rows[{index}] is {row}, past the {roster_len}-entry frame roster"
                ));
            }
            if self.roster_rows[..index].contains(&row) {
                return bad(format!("roster_rows repeats {row} at {index}"));
            }
        }
        rows(self.terminal_cells.len(), "terminal_cells", requests, false)?;
        for (index, cell) in self.terminal_cells.iter().enumerate() {
            if self.terminal_cells[..index].contains(cell) {
                return bad(format!("terminal_cells repeats a cell at {index}"));
            }
        }

        if plan.position_ids.len() != plan.token_ids.len() {
            return bad(format!(
                "position_ids has {} entries for {} tokens",
                plan.position_ids.len(),
                plan.token_ids.len()
            ));
        }

        csr(
            &plan.qo_indptr,
            "qo_indptr",
            plan.token_ids.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.kv_page_indptr,
            "kv_page_indptr",
            plan.kv_page_indices.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.rs_buffer_slot_indptr,
            "rs_buffer_slot_indptr",
            plan.rs_buffer_slot_ids.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.rs_translation_indptr,
            "rs_translation_indptr",
            plan.rs_translation.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.sampling_indptr,
            "sampling_indptr",
            plan.sampling_indices.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.mask_indptr,
            "mask_indptr",
            plan.masks.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.image_indptr,
            "image_indptr",
            plan.image_anchor_positions.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.audio_indptr,
            "audio_indptr",
            plan.audio_anchor_rows.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.embed_block_indptr,
            "embed_block_indptr",
            plan.embed_dtypes.len(),
            wire_rows,
            true,
        )?;
        csr(
            &plan.embed_indptr,
            "embed_indptr",
            plan.embed_rows.len(),
            plan.embed_dtypes.len(),
            true,
        )?;

        if plan.embed_shapes.len() != plan.embed_dtypes.len().saturating_mul(2)
            || plan.embed_anchor_rows.len() != plan.embed_dtypes.len()
        {
            return bad(format!(
                "embedding blocks disagree: {} dtypes, {} shape words, {} anchors",
                plan.embed_dtypes.len(),
                plan.embed_shapes.len(),
                plan.embed_anchor_rows.len()
            ));
        }
        if let Some(&dtype) = plan.embed_dtypes.iter().find(|&&d| d != 2) {
            return bad(format!(
                "precomputed embeddings carry dtype tag {dtype}; only bf16 (2) is served"
            ));
        }

        // The ticket plane: heads and tails are parallel, and the CSR covers
        // every ticket it claims.
        if self.channel_expected_head.len() != self.channel_expected_tail.len() {
            return bad(format!(
                "channel ticket head/tail are not parallel: {} and {}",
                self.channel_expected_head.len(),
                self.channel_expected_tail.len()
            ));
        }
        if !self.channel_expected_head.is_empty() && self.channel_ticket_indptr.is_empty() {
            return bad("channel_ticket_indptr is required when ticket values are present");
        }
        csr(
            &self.channel_ticket_indptr,
            "channel_ticket_indptr",
            self.channel_expected_head.len(),
            requests,
            true,
        )?;
        if let Some(&last) = self.channel_ticket_indptr.last() {
            if last as usize != self.channel_expected_head.len() {
                return bad(format!(
                    "channel_ticket_indptr ends at {last}, not the {} tickets",
                    self.channel_expected_head.len()
                ));
            }
        }

        rows(
            self.logical_fire_ids.len(),
            "logical_fire_ids",
            requests,
            true,
        )?;

        if !self.program_row_indptr.is_empty() {
            csr(
                &self.program_row_indptr,
                "program_row_indptr",
                wire_rows,
                requests,
                false,
            )?;
            if plan.kv_write_lower_bounds.len() != plan.kv_write_upper_bounds.len()
                || (!plan.kv_write_lower_bounds.is_empty()
                    && plan.kv_write_lower_bounds.len() != requests)
            {
                return bad(format!(
                    "KV write bounds must be one pair per instance: {} lower, {} upper, {requests} instances",
                    plan.kv_write_lower_bounds.len(),
                    plan.kv_write_upper_bounds.len()
                ));
            }
            if let Some((lo, hi)) = plan
                .kv_write_lower_bounds
                .iter()
                .zip(&plan.kv_write_upper_bounds)
                .find(|(lo, hi)| lo > hi)
            {
                return bad(format!("KV write bound is inverted: {lo} > {hi}"));
            }
        }

        rows(
            plan.kv_last_page_lens.len(),
            "kv_last_page_lens",
            wire_rows,
            true,
        )?;

        // The recurrent-state vectors, which index each other.
        if plan.rs_slot_ids.len() != plan.rs_slot_flags.len() {
            return bad(format!(
                "rs_slot_ids has {} entries and rs_slot_flags {}",
                plan.rs_slot_ids.len(),
                plan.rs_slot_flags.len()
            ));
        }
        if !plan.rs_fold_lens.is_empty() && plan.rs_fold_lens.len() != plan.rs_slot_ids.len() {
            return bad(format!(
                "rs_fold_lens has {} entries for {} slots",
                plan.rs_fold_lens.len(),
                plan.rs_slot_ids.len()
            ));
        }
        if !plan.qo_indptr.is_empty()
            && !plan.rs_slot_ids.is_empty()
            && plan.rs_slot_ids.len() != wire_rows
        {
            return bad(format!(
                "rs_slot_ids has {} entries for {wire_rows} resolved qo rows",
                plan.rs_slot_ids.len()
            ));
        }
        const KNOWN_RS_FLAGS: u8 = PIE_RS_FLAG_RESET
            | PIE_RS_FLAG_FOLD
            | PIE_RS_FLAG_BUFFER_WRITE
            | PIE_RS_FLAG_FOLD_LEN_DEVICE;
        if let Some(&flag) = plan
            .rs_slot_flags
            .iter()
            .find(|&&f| f & !KNOWN_RS_FLAGS != 0)
        {
            return bad(format!("rs_slot_flags carries unknown bits: {flag:#04x}"));
        }

        rows(plan.context_ids.len(), "context_ids", wire_rows, true)?;
        rows(plan.kv_len.len(), "kv_len", wire_rows, true)?;
        if plan.kv_len_device.len() > 1 {
            return bad(format!(
                "kv_len_device carries {} device pointers; zero or one",
                plan.kv_len_device.len()
            ));
        }

        // The media planes.
        if !plan.image_grids.len().is_multiple_of(3) {
            return bad(format!(
                "image_grids has {} entries, not a multiple of 3",
                plan.image_grids.len()
            ));
        }
        let images = plan.image_grids.len() / 3;
        rows(
            plan.image_anchor_positions.len(),
            "image_anchor_positions",
            images,
            false,
        )?;
        rows(
            plan.image_anchor_rows.len(),
            "image_anchor_rows",
            images,
            false,
        )?;
        csr(
            &plan.image_pixel_indptr,
            "image_pixel_indptr",
            plan.image_pixels.len(),
            images,
            true,
        )?;
        csr(
            &plan.image_mrope_indptr,
            "image_mrope_indptr",
            plan.image_mrope_positions.len(),
            images,
            true,
        )?;
        csr(
            &plan.audio_feature_indptr,
            "audio_feature_indptr",
            plan.audio_features.len(),
            plan.audio_anchor_rows.len(),
            true,
        )?;
        Ok(())
    }
}

impl FrameSubmission {
    /// Every rule `validate_frame_desc` stated: the roster, the frame-union
    /// translation table, each step against the roster, and the one invariant
    /// that spans steps — a terminal cell belongs to one step only.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the member and the numbers that disagree.
    pub fn validate(&self) -> Result<(), Malformed> {
        if self.steps.is_empty() {
            return bad("a frame must carry at least one step");
        }
        for (index, &id) in self.instance_ids.iter().enumerate() {
            if self.instance_ids[..index].contains(&id) {
                return bad(format!("instance_ids repeats {id} at {index}"));
            }
        }
        let roster_len = self.instance_ids.len();
        if !self.kv_translation.is_empty() && self.kv_translation_indptr.is_empty() {
            return bad("kv_translation_indptr is required when translation values are present");
        }
        csr(
            &self.kv_translation_indptr,
            "kv_translation_indptr",
            self.kv_translation.len(),
            roster_len,
            true,
        )?;

        let mut seen: Vec<*mut TerminalCell> = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            step.validate(roster_len)
                .map_err(|e| Malformed(format!("step {index}: {e}")))?;
            for &cell in &step.terminal_cells {
                if seen.contains(&cell) {
                    return bad(format!(
                        "step {index} reuses a terminal cell an earlier step already owns"
                    ));
                }
                seen.push(cell);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::plan::EncodedMask;

    /// A distinct, stable cell address. Leaked because a terminal cell must
    /// outlive the fire that publishes into it, and a test that took `&mut`
    /// to a local would be asserting about a pointer the frame outlives.
    fn cell() -> *mut TerminalCell {
        Box::into_raw(Box::new(TerminalCell::pending()))
    }

    /// One member, one token, one page — the shape everything else perturbs.
    fn sound() -> FrameSubmission {
        FrameSubmission {
            instance_ids: vec![9],
            required_kv_pages: 1,
            steps: vec![StepSubmission {
                plan: LaunchPlan {
                    token_ids: vec![7],
                    position_ids: vec![0],
                    kv_page_indices: vec![0],
                    kv_page_indptr: vec![0, 1],
                    kv_last_page_lens: vec![1],
                    qo_indptr: vec![0, 1],
                    ..Default::default()
                },
                roster_rows: vec![0],
                sub_batch_indptr: vec![0, 1],
                sub_batch_class: vec![0],
                terminal_cells: vec![cell()],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn why(frame: &FrameSubmission) -> String {
        frame.validate().expect_err("must be refused").to_string()
    }

    #[test]
    fn the_sound_frame_is_not_refused() {
        sound().validate().unwrap();
    }

    #[test]
    fn a_frame_with_no_step_is_refused() {
        let mut f = sound();
        f.steps.clear();
        assert!(why(&f).contains("at least one step"));
    }

    #[test]
    fn a_repeated_instance_is_refused() {
        let mut f = sound();
        f.instance_ids = vec![9, 9];
        assert!(why(&f).contains("repeats 9"));
    }

    #[test]
    fn a_roster_row_past_the_roster_is_refused() {
        let mut f = sound();
        f.steps[0].roster_rows = vec![3];
        f.steps[0].sub_batch_indptr = vec![0, 1];
        assert!(why(&f).contains("past the 1-entry frame roster"));
    }

    #[test]
    fn a_repeated_roster_row_is_refused() {
        let mut f = sound();
        f.instance_ids = vec![9, 10];
        f.steps[0].roster_rows = vec![0, 0];
        f.steps[0].sub_batch_indptr = vec![0, 2];
        f.steps[0].terminal_cells = vec![cell(), cell()];
        assert!(why(&f).contains("roster_rows repeats 0"));
    }

    /// The invariant that spans steps, and the only reason the frame walks
    /// them together rather than validating each alone.
    #[test]
    fn a_terminal_cell_shared_between_two_steps_is_refused() {
        let ptr = cell();
        let mut f = sound();
        f.steps[0].terminal_cells = vec![ptr];
        let mut second = f.steps[0].clone();
        second.terminal_cells = vec![ptr];
        f.steps.push(second);
        assert!(why(&f).contains("an earlier step already owns"));
    }

    #[test]
    fn a_cell_repeated_inside_one_step_is_refused() {
        let ptr = cell();
        let mut f = sound();
        f.instance_ids = vec![9, 10];
        f.steps[0].roster_rows = vec![0, 1];
        f.steps[0].sub_batch_indptr = vec![0, 2];
        f.steps[0].terminal_cells = vec![ptr, ptr];
        assert!(why(&f).contains("terminal_cells repeats"));
    }

    /// One cell per member, unconditionally — the rule
    /// `entry_validation::no_validator_is_deferred` records catching
    /// fixtures on. A step with no cells at all is refused too: a frame
    /// whose members cannot be told apart on completion is not servable.
    #[test]
    fn every_member_states_its_own_terminal_cell() {
        let mut f = sound();
        f.instance_ids = vec![9, 10];
        f.steps[0].roster_rows = vec![0, 1];
        f.steps[0].sub_batch_indptr = vec![0, 2];
        f.steps[0].terminal_cells = vec![cell()];
        assert!(why(&f).contains("terminal_cells has 1 entries, not 2"));

        f.steps[0].terminal_cells.clear();
        assert!(why(&f).contains("terminal_cells has 0 entries, not 2"));
    }

    #[test]
    fn positions_that_do_not_cover_the_tokens_are_refused() {
        let mut f = sound();
        f.steps[0].plan.position_ids = vec![0, 1];
        assert!(why(&f).contains("position_ids has 2 entries for 1 tokens"));
    }

    #[test]
    fn a_qo_csr_that_disagrees_with_its_tokens_is_refused() {
        let mut f = sound();
        f.steps[0].plan.qo_indptr = vec![0, 5];
        assert!(why(&f).contains("qo_indptr ends at 5"));
    }

    #[test]
    fn rs_slot_ids_and_flags_must_be_parallel() {
        let mut f = sound();
        f.steps[0].plan.rs_slot_ids = vec![0];
        f.steps[0].plan.rs_slot_flags = vec![0, 0];
        assert!(why(&f).contains("rs_slot_ids has 1 entries and rs_slot_flags 2"));
    }

    #[test]
    fn an_unknown_rs_flag_bit_is_refused() {
        let mut f = sound();
        f.steps[0].plan.rs_slot_ids = vec![0];
        f.steps[0].plan.rs_slot_flags = vec![0x80];
        assert!(why(&f).contains("unknown bits"));
    }

    #[test]
    fn fold_lengths_must_match_the_slots() {
        let mut f = sound();
        f.steps[0].plan.rs_slot_ids = vec![0];
        f.steps[0].plan.rs_slot_flags = vec![0];
        f.steps[0].plan.rs_fold_lens = vec![1, 2];
        assert!(why(&f).contains("rs_fold_lens has 2 entries for 1 slots"));
    }

    /// The acceptance half, which the refusals do not cover: a fire whose
    /// recurrent-state vectors are stated PER RESOLVED ROW is served, not
    /// turned away. Ported from `local`'s
    /// `launch_validator_accepts_resolved_request_rs_vectors`.
    #[test]
    fn resolved_row_rs_vectors_are_accepted() {
        let mut f = sound();
        f.instance_ids = vec![9, 10];
        f.steps[0].roster_rows = vec![0, 1];
        f.steps[0].sub_batch_indptr = vec![0, 2];
        f.steps[0].terminal_cells = vec![cell(), cell()];
        let plan = &mut f.steps[0].plan;
        plan.token_ids = vec![7, 11];
        plan.position_ids = vec![0, 0];
        plan.qo_indptr = vec![0, 1, 2];
        plan.kv_page_indices = vec![0, 1];
        plan.kv_page_indptr = vec![0, 1, 2];
        plan.kv_last_page_lens = vec![1, 1];
        // Two resolved rows, so two slots, two flags, two fold lengths.
        plan.rs_slot_ids = vec![0, 1];
        plan.rs_slot_flags = vec![PIE_RS_FLAG_FOLD, PIE_RS_FLAG_RESET];
        plan.rs_fold_lens = vec![1, 1];
        plan.rs_buffer_slot_ids = vec![0, 1];
        plan.rs_buffer_slot_indptr = vec![0, 1, 2];
        f.validate().expect("a resolved-row RS fire is servable");
    }

    #[test]
    fn tickets_without_their_csr_are_refused() {
        let mut f = sound();
        f.steps[0].channel_expected_head = vec![1];
        f.steps[0].channel_expected_tail = vec![2];
        assert!(why(&f).contains("channel_ticket_indptr is required"));
    }

    #[test]
    fn a_ticket_csr_that_leaves_a_ticket_uncovered_is_refused() {
        let mut f = sound();
        f.steps[0].channel_expected_head = vec![1, 2];
        f.steps[0].channel_expected_tail = vec![3, 4];
        f.steps[0].channel_ticket_indptr = vec![0, 1];
        assert!(why(&f).contains("ends at 1, not the 2 tickets"));
    }

    #[test]
    fn heads_and_tails_must_be_parallel() {
        let mut f = sound();
        f.steps[0].channel_expected_head = vec![1, 2];
        f.steps[0].channel_expected_tail = vec![3];
        f.steps[0].channel_ticket_indptr = vec![0, 2];
        assert!(why(&f).contains("not parallel"));
    }

    #[test]
    fn translation_values_without_their_csr_are_refused() {
        let mut f = sound();
        f.kv_translation = vec![4];
        assert!(why(&f).contains("kv_translation_indptr is required"));
    }

    #[test]
    fn inverted_kv_write_bounds_are_refused() {
        let mut f = sound();
        f.steps[0].program_row_indptr = vec![0, 1];
        f.steps[0].plan.kv_write_lower_bounds = vec![9];
        f.steps[0].plan.kv_write_upper_bounds = vec![2];
        assert!(why(&f).contains("inverted"));
    }

    #[test]
    fn a_non_bf16_embedding_block_is_refused() {
        let mut f = sound();
        f.steps[0].plan.embed_dtypes = vec![1];
        f.steps[0].plan.embed_shapes = vec![1, 1];
        f.steps[0].plan.embed_anchor_rows = vec![0];
        assert!(why(&f).contains("only bf16"));
    }

    #[test]
    fn image_planes_must_describe_the_same_images() {
        let mut f = sound();
        f.steps[0].plan.image_grids = vec![1, 2, 3];
        assert!(why(&f).contains("image_anchor_positions has 0 entries, not 1"));
    }

    #[test]
    fn a_mask_csr_that_overruns_its_masks_is_refused() {
        let mut f = sound();
        f.steps[0].plan.masks = vec![EncodedMask::new(vec![0, 4], 4)];
        f.steps[0].plan.mask_indptr = vec![0, 3];
        assert!(why(&f).contains("mask_indptr ends at 3"));
    }

    /// The refusal names the step, because a frame carries several and
    /// "malformed" would leave the caller to find which.
    #[test]
    fn the_refusal_names_the_step_it_came_from() {
        let mut f = sound();
        let mut second = f.steps[0].clone();
        second.plan.position_ids = vec![0, 1];
        f.steps.push(second);
        assert!(why(&f).starts_with("step 1: "));
    }
}
