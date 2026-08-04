//! Batch assembly: capacity accounting + the dense-batch accumulator.

use std::collections::HashMap;

use pie_driver_abi::PieTerminalCell;

use super::fire_plan;
use super::stats::SchedulerStats;
use super::wire;
use super::worker::PendingRequest;
use crate::driver::{FrameSubmission, LaunchPlan, SchedulerLimits, StepSubmission};

/// One step's assembled wire request: the per-batch merge of its member
/// fires, before roster resolution. Field names mirror the old per-wave
/// submission so the merge logic and its tests carry over unchanged.
pub(crate) struct StepBuild {
    pub(crate) plan: LaunchPlan,
    pub(crate) instance_ids: Vec<u64>,
    pub(crate) terminal_cells: Vec<*mut PieTerminalCell>,
    pub(crate) kv_translation: Vec<u32>,
    pub(crate) kv_translation_indptr: Vec<u32>,
    pub(crate) program_row_indptr: Vec<u32>,
    pub(crate) logical_fire_ids: Vec<u64>,
    pub(crate) channel_expected_head: Vec<u64>,
    pub(crate) channel_expected_tail: Vec<u64>,
    pub(crate) channel_ticket_indptr: Vec<u32>,
}

/// `PIE_FIRE_CENSUS=1` prints one line per sealed step group (size, solo
/// contract, join refusals by clause) — the C measurement's surface.
fn fire_census_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PIE_FIRE_CENSUS").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// The fire plan's qkv_postprocess lowering, converted from MEMBER counts
/// to WIRE request rows through the step's attribution CSR — the value
/// [`StepSubmission::planned_hook_free_prefix_rows`] carries. The
/// semantics mirror `Dispatch::launch_hook_free_prefix_rows` EXACTLY (it
/// walks compiled PTIR stage plans where this walks the admission-time
/// `hook_program` stamps over the SAME spans) so the driver's cross-check
/// can refuse on drift instead of guessing which side is right:
///   * malformed/absent attribution → UNPLANNED (driver derives alone);
///   * a hook member with an empty wire span cannot be located among the
///     rows → 0, no fast prefix;
///   * otherwise the first hook member's row start IS the prefix (spans
///     are contiguous in planned order — hooks-last is what makes it
///     maximal), and with no hook members every row is in it.
fn planned_prefix_wire_rows(
    plan: &fire_plan::FirePlan,
    ordered: &[Box<PendingRequest>],
    row_indptr: &[u32],
) -> u32 {
    // A plan is sent only when it DECIDES something: at least one hook
    // member (the site's Prefix arm). Hook stamps come from tracked
    // registration, so on every planned step the driver's compiled-plan
    // walk sees the same programs and the cross-check compares like with
    // like; a hook-free step (including prebuilt/untracked fires, whose
    // programs the driver may not know) keeps the driver's own
    // derivation, whose answer is consumed by nothing anyway.
    if !ordered.iter().any(|req| req.hook_program) {
        return pie_driver_abi::PIE_HOOK_FREE_PREFIX_UNPLANNED;
    }
    if row_indptr.len() != ordered.len() + 1 {
        return pie_driver_abi::PIE_HOOK_FREE_PREFIX_UNPLANNED;
    }
    let total = *row_indptr.last().expect("indptr has a total");
    if total == 0 {
        return 0;
    }
    let mut first_hook_row = total;
    for (member, req) in ordered.iter().enumerate() {
        if !req.hook_program {
            continue;
        }
        let (lo, hi) = (row_indptr[member], row_indptr[member + 1]);
        if hi <= lo {
            return 0;
        }
        first_hook_row = first_hook_row.min(lo);
    }
    // The site IS the source; the row scan above is its span binding. The
    // two must agree by construction (fast_rows counts the leading
    // non-hook members of the same planned order).
    if let Some(site) = plan
        .sites
        .iter()
        .find(|site| site.name == fire_plan::SITE_QKV_POSTPROCESS)
    {
        // ≥1 hook member, so the site is always the Prefix arm here.
        let fast_members = match site.lowering {
            fire_plan::Lowering::Prefix { fast_rows } => fast_rows as usize,
            _ => unreachable!("a hooked step always plans the Prefix arm"),
        };
        debug_assert_eq!(
            first_hook_row,
            row_indptr[fast_members.min(ordered.len())],
            "the plan's member prefix and the row-span scan must agree"
        );
    }
    first_hook_row
}

/// NS-2: the attention_mask site's unmasked prefix, converted to WIRE
/// rows through the attribution CSR — the value
/// [`StepSubmission::planned_unmasked_prefix_rows`] carries. Meaningful
/// only on hook-free steps with at least one masked member (the
/// seriation nests mask under hooks, so a hooked step's masked members
/// are not contiguous); everything else is UNPLANNED and the driver
/// keeps the fire-level mask arm.
/// STRUCTURAL S-2: the depth union's REQUEST split — the count of
/// leading full-depth members. Planned only for the v0 shape: at least
/// one truncated member AND at least one full member, every member a
/// plain 1-token decode lane (no hooks/lora/masks/multi-token), all
/// truncated members sharing ONE k and seriated as the contiguous tail
/// (the sort key guarantees it; the scan verifies loudly-silently by
/// declining).
fn planned_full_depth_request_split(ordered: &[Box<PendingRequest>]) -> u32 {
    if !depth_union_enabled() {
        return pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED;
    }
    let truncated = ordered
        .iter()
        .filter(|r| r.request.max_layers.is_some())
        .count();
    if truncated == 0 || truncated == ordered.len() {
        return pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED;
    }
    let mut k: Option<u32> = None;
    for req in ordered.iter() {
        // AC-3 (lora x depth): an UNTRUNCATED lora member rides the
        // full-depth prefix freely — the correction is span-grouped and
        // window-free, and the seriation keeps it out of the truncated
        // tail. A single lane carrying BOTH axes still declines (its
        // correction span would cross the depth window — the PQ-tree
        // class, refused as safe degradation for now).
        if (req.hook_program && req.request.max_layers.is_some())
            || (req.lora_program && req.request.max_layers.is_some())
            // AC-1: a lane on BOTH window axes is the PQ-tree class.
            || (req.request.has_user_mask && req.request.max_layers.is_some())
            || req.request.token_ids.len() > ordered.len()
            || req
                .request
                .qo_indptr
                .windows(2)
                .any(|w| w[1] - w[0] > 1)
        {
            return pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED;
        }
        if let Some(this_k) = req.request.max_layers {
            if *k.get_or_insert(this_k) != this_k {
                return pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED;
            }
        }
    }
    // AC-1 order [plain | truncated | masked]: the truncated block is a
    // MIDDLE window ending where the masked suffix starts; every member
    // after it must be masked (full-depth), every truncated member
    // contiguous. dsplit = the block's start; its end derives from the
    // mask word driver-side.
    // AC-4/AC-5: the full-depth suffix behind the truncated middle may
    // hold hooked lanes then masked lanes (the seriation's order) —
    // both are full-depth. The driver's stash window anchors on the
    // mask word when present, the hook word otherwise.
    let masked_tail = ordered
        .iter()
        .rev()
        .take_while(|r| r.request.has_user_mask || r.hook_program)
        .count();
    let split = ordered.len() - masked_tail - truncated;
    if ordered[split..ordered.len() - masked_tail]
        .iter()
        .any(|r| r.request.max_layers.is_none())
    {
        return pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED;
    }
    split as u32
}

/// The depth union's arm switch — DEFAULT ON (`PIE_DEPTH_UNION=0`
/// disarms and restores the S-1 solo rule). The union oracle and the
/// wide battery (R=4, mixed-k decline, all-truncated control) passed
/// on the armed boots before the flip.
pub(crate) fn depth_union_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("PIE_DEPTH_UNION").is_ok_and(|v| v == "0")
    })
}

fn planned_unmasked_prefix_wire_rows(
    plan: &fire_plan::FirePlan,
    ordered: &[Box<PendingRequest>],
    row_indptr: &[u32],
) -> u32 {
    // AC-4: hooks no longer suppress the plan either — the order is
    // [plain | truncated | hooked | masked], so the mask window is
    // still the suffix and hooked lanes sit in the unmasked prefix. A
    // lane on BOTH axes (a masked hook program) remains the refusal.
    if ordered
        .iter()
        .any(|req| req.hook_program && req.request.has_user_mask)
        || !ordered.iter().any(|req| req.request.has_user_mask)
    {
        return pie_driver_abi::PIE_UNMASKED_PREFIX_UNPLANNED;
    }
    if row_indptr.len() != ordered.len() + 1 {
        return pie_driver_abi::PIE_UNMASKED_PREFIX_UNPLANNED;
    }
    let total = *row_indptr.last().expect("indptr has a total");
    if total == 0 {
        return 0;
    }
    let mut first_masked_row = total;
    for (member, req) in ordered.iter().enumerate() {
        if !req.request.has_user_mask {
            continue;
        }
        let (lo, hi) = (row_indptr[member], row_indptr[member + 1]);
        if hi <= lo {
            return 0;
        }
        first_masked_row = first_masked_row.min(lo);
    }
    if let Some(site) = plan
        .sites
        .iter()
        .find(|site| site.name == fire_plan::SITE_ATTENTION_MASK)
    {
        let unmasked_members = match site.lowering {
            fire_plan::Lowering::Prefix { fast_rows } => fast_rows as usize,
            _ => unreachable!("a masked step always plans the Prefix arm"),
        };
        debug_assert_eq!(
            first_masked_row,
            row_indptr[unmasked_members.min(ordered.len())],
            "the plan's member prefix and the row-span scan must agree"
        );
    }
    first_masked_row
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let input_tokens = req.request.token_ids.len();
    let forward_requests = req
        .wire_row_count()
        .max(req.request.rs_slot_ids.len())
        .max(1);
    let forward_tokens = input_tokens;
    let page_refs = req.request.kv_page_indices.len();
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens,
        page_refs,
    }
}

/// Admission-time shape gate: rejects a single fire whose resolved shape
/// exceeds the driver's launch limits before it can enter the queue.
pub(crate) struct AdmissionLimits {
    page_size: u32,
    limits: SchedulerLimits,
}

impl AdmissionLimits {
    pub(crate) fn new(limits: SchedulerLimits, page_size: u32) -> Self {
        Self { page_size, limits }
    }

    pub(crate) fn single_request_limit_error(&self, req: &PendingRequest) -> Option<String> {
        let usage = request_capacity_usage(req, self.page_size);
        if usage.forward_requests > self.limits.max_forward_requests {
            return Some(format!(
                "forward request has {} resolved rows, exceeding driver limit {}",
                usage.forward_requests, self.limits.max_forward_requests
            ));
        }
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding driver limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding driver limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        // Malformed shapes reject the FIRE at admission; batch build treats
        // them as invariants (a panic there would take down the scheduler
        // thread — RV-20).
        if req.request.kv_write_lower_bounds.len() > 1
            || req.request.kv_write_upper_bounds.len() > 1
            || req.request.kv_write_lower_bounds.len() != req.request.kv_write_upper_bounds.len()
        {
            return Some("per-fire KV containment bounds must be empty or scalar".to_string());
        }
        let rows = req.request.qo_indptr.len().saturating_sub(1);
        if rows > 1 && !req.request.masks.is_empty() && req.request.mask_indptr.len() != rows + 1 {
            return Some(format!(
                "multi-row fire carries {} masks without a row CSR \
                 ({} mask boundaries for {} rows)",
                req.request.masks.len(),
                req.request.mask_indptr.len(),
                rows
            ));
        }
        None
    }
}

pub(crate) fn build_batch_request(
    requests: &[Box<PendingRequest>],
    page_size: u32,
    stats: &SchedulerStats,
) -> StepBuild {
    if requests.len() == 1 && (requests[0].prebuilt || requests[0].preserves_inner_rows()) {
        // Keep the logical-fire payload intact across attempts. RETRY builds a
        // fresh native launch from this same immutable request. Ordinary
        // multi-row programs take this path too: flattening them through
        // `wire::append_request` would collapse their inner CSR to one row
        // while incorrectly retaining B recurrent-state slots.
        let req = &requests[0];
        let mut plan = req.request.clone();
        let kv_translation = std::mem::take(&mut plan.kv_translation);
        plan.required_kv_pages = plan.required_kv_pages.max(
            kv_translation
                .iter()
                .copied()
                .max()
                .map_or(0, |page| page.saturating_add(1)),
        );
        let channel_expected_head = plan.channel_expected_head.clone();
        let channel_expected_tail = plan.channel_expected_tail.clone();
        let channel_ticket_len = channel_expected_head.len() as u32;
        let rows = plan.qo_indptr.len().saturating_sub(1) as u32;
        return StepBuild {
            kv_translation_indptr: vec![0, kv_translation.len() as u32],
            kv_translation,
            program_row_indptr: vec![0, rows],
            plan,
            instance_ids: vec![req.instance_id],
            terminal_cells: vec![req.completion.terminal_cell_ptr()],
            logical_fire_ids: vec![req.logical_fire_id],
            channel_expected_head,
            channel_expected_tail,
            channel_ticket_indptr: vec![0, channel_ticket_len],
        };
    }
    let elide_decode_masks = requests.iter().all(|req| {
        req.request.single_token_mode
            && !req.request.has_user_mask
            && req.request.token_ids.len() <= 1
    });
    let use_kv_write_bounds = requests.iter().any(|req| {
        !req.request.kv_write_lower_bounds.is_empty()
            || !req.request.kv_write_upper_bounds.is_empty()
    });
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut batch_req = wire::new_batched_forward_request_with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut kv_translation = Vec::new();
        let mut kv_translation_indptr = Vec::with_capacity(requests.len() + 1);
        kv_translation_indptr.push(0);
        let mut logical_fire_ids = Vec::with_capacity(requests.len());
        let mut channel_expected_head = Vec::new();
        let mut channel_expected_tail = Vec::new();
        let mut channel_ticket_indptr = Vec::with_capacity(requests.len() + 1);
        channel_ticket_indptr.push(0);
        let mut program_row_indptr = Vec::with_capacity(requests.len() + 1);
        program_row_indptr.push(0);
        for req in requests {
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            logical_fire_ids.push(req.logical_fire_id);
            wire::append_request_with_options(
                &mut batch_req,
                &req.request,
                req.last_page_len,
                page_size,
                elide_decode_masks,
            );
            if use_kv_write_bounds {
                match (
                    req.request.kv_write_lower_bounds.as_slice(),
                    req.request.kv_write_upper_bounds.as_slice(),
                ) {
                    ([], []) => {
                        batch_req.kv_write_lower_bounds.push(0);
                        batch_req.kv_write_upper_bounds.push(u64::MAX);
                    }
                    ([lower], [upper]) => {
                        batch_req.kv_write_lower_bounds.push(*lower);
                        batch_req.kv_write_upper_bounds.push(*upper);
                    }
                    _ => panic!("per-fire KV containment bounds must be empty or scalar"),
                }
            }
            kv_translation.extend(req.request.kv_translation.iter().copied());
            kv_translation_indptr.push(kv_translation.len() as u32);
            channel_expected_head.extend(req.request.channel_expected_head.iter().copied());
            channel_expected_tail.extend(req.request.channel_expected_tail.iter().copied());
            channel_ticket_indptr.push(channel_expected_head.len() as u32);
            program_row_indptr.push(
                program_row_indptr.last().copied().unwrap_or(0)
                    + req.wire_row_count().max(1) as u32,
            );
        }
        StepBuild {
            plan: batch_req,
            instance_ids,
            terminal_cells,
            kv_translation,
            kv_translation_indptr,
            program_row_indptr,
            logical_fire_ids,
            channel_expected_head,
            channel_expected_tail,
            channel_ticket_indptr,
        }
    })
}

/// Assemble one sealed frame's submission (ABI v14) from its waves' picked
/// requests, in slot order. Returns the submission plus the flattened
/// requests in POST order (step order, member order within a step) — the
/// retirement set.
///
/// Step formation preserves the old per-wave batch semantics exactly: within
/// a wave, requests group under [`LaunchGrouping`]'s compatibility rules
/// (instance/pipeline dedup, geometry-class homogeneity, mask/solo
/// exclusions, structural budgets); each group becomes one STEP with a
/// single geometry-homogeneous sub-batch, so every step's wire payload is
/// byte-identical to the wave batch it replaces.
///
/// Frame tables:
/// - the roster is first-appearance order across steps;
/// - each roster lane's translation segment comes from its LAST fire in the
///   frame (the latest overlay — prepared write targets accumulate);
/// - `required_kv_pages` is the frame-union high-water over every member
///   (declared high-water and page-id-derived floors).
pub(crate) fn build_frame_submission(
    waves: Vec<Vec<Box<PendingRequest>>>,
    limits: SchedulerLimits,
    page_size: u32,
    stats: &SchedulerStats,
    model_sites: &[fire_plan::Site],
) -> (FrameSubmission, Vec<Box<PendingRequest>>) {
    let mut step_groups: Vec<Vec<Box<PendingRequest>>> = Vec::new();
    for wave in waves {
        if wave.is_empty() {
            continue;
        }
        let mut deferred = wave;
        // Repeated passes: incompatible members defer to the wave's next
        // step, exactly like the old deferred-class re-dispatch.
        while !deferred.is_empty() {
            let mut grouping = super::worker::LaunchGrouping::default();
            let mut group: Vec<Box<PendingRequest>> = Vec::new();
            let mut rest: Vec<Box<PendingRequest>> = Vec::new();
            let mut closed = false;
            let mut refusals: Vec<&'static str> = Vec::new();
            for req in deferred {
                if closed {
                    refusals.push("group-closed");
                    rest.push(req);
                    continue;
                }
                if let Some(reason) = grouping.refusal(&req, limits, page_size) {
                    refusals.push(reason);
                    rest.push(req);
                    continue;
                }
                closed = grouping.push(&req, limits, page_size);
                group.push(req);
            }
            // The fire census (C): one line per sealed step group — size,
            // the head's solo contract if any, and every join refusal by
            // clause. This is the measurement surface for "what does the
            // remaining partition cost": a workload whose census shows only
            // contract-bound reasons has nothing left for the scheduler to
            // relax.
            if fire_census_enabled() {
                let solo = group
                    .first()
                    .and_then(|req| req.solo_reason())
                    .unwrap_or("-");
                // Per-member fingerprint (fire id × row count): whether two
                // runs composed the SAME logical fires is what separates a
                // composition-timing difference from a numeric one when
                // their outputs disagree.
                let members: Vec<String> = group
                    .iter()
                    .map(|req| {
                        format!(
                            "{}x{}",
                            req.logical_fire_id,
                            req.request.token_ids.len()
                        )
                    })
                    .collect();
                eprintln!(
                    "[fire-census] step members={} [{}] solo={} deferred={:?}",
                    group.len(),
                    members.join(","),
                    solo,
                    refusals,
                );
            }
            debug_assert!(!group.is_empty(), "grouping always admits the head");
            if group.is_empty() {
                // Defensive: never loop forever on a malformed head.
                group.push(rest.remove(0));
            }
            step_groups.push(group);
            deferred = rest;
        }
    }

    let mut roster: Vec<u64> = Vec::new();
    let mut roster_index: HashMap<u64, u32> = HashMap::new();
    let mut lane_translation: Vec<Vec<u32>> = Vec::new();
    let mut required_kv_pages = 0u32;
    let mut steps: Vec<StepSubmission> = Vec::new();
    let mut flattened: Vec<Box<PendingRequest>> = Vec::new();

    for group in step_groups {
        // Ordered sub-batches: wire (Host-class) members first, the
        // device-resolved envelope suffix last — the driver's offset
        // fixed-decode compose requires the envelope lanes to be a
        // contiguous program suffix. That contract stays PRIMARY. Within
        // each class, attention-hook-carrying programs sort last: the
        // driver's hook-free prefix (`StageHooks::hook_free_prefix_rows`)
        // is the fused fast path, and a leading hook-free run that spans
        // ALL hook-free lanes makes that prefix maximal instead of ending
        // at whichever hook lane happened to arrive first. Stable order
        // keeps arrival otherwise. The permutation comes from the fire
        // planner — the same key the inline sort here used to apply,
        // generalized so the next divergence axis lands as planner data
        // (`fire_plan::MemberFacts`) instead of a wider sort key; the
        // plan's per-site lowerings are not consumed yet (v0).
        let facts: Vec<fire_plan::MemberFacts> = group
            .iter()
            .enumerate()
            .map(|(arrival, req)| fire_plan::MemberFacts {
                hook_program: req.hook_program,
                lora: req.lora_program,
                custom_mask: req.request.has_user_mask,
                truncated: req.request.max_layers.is_some(),
                multi_token: req.request.qo_indptr.windows(2).any(|w| w[1] - w[0] > 1),
                device_resolved_geometry: req.request.device_resolved_geometry,
                arrival,
            })
            .collect();
        // `model_sites` is the driver's own statement, from its validated
        // declared plan through the capabilities handshake (the site_table
        // module doc's wiring; `fire_plan::site_table::summary_sites` maps
        // the reported summary into the vocabulary). Empty — every dense
        // model, every driver without a declared plan — reduces this to
        // the old `plan_fire` exactly. The qkv_postprocess site is
        // CONSUMED since B (`planned_prefix_wire_rows` below): its
        // Prefix{fast_rows} crosses the wire as
        // `planned_hook_free_prefix_rows` and the driver's Peel split
        // uses it after a cross-check. The other sites remain
        // informational.
        let plan = fire_plan::plan_fire_with_model(&facts, model_sites);
        debug_assert_eq!(
            plan.sites.len(),
            3 + model_sites.len(),
            "the merged plan carries both member-fact sites and every model site"
        );
        debug_assert_eq!(
            plan.member_order,
            {
                let mut order: Vec<usize> = (0..group.len()).collect();
                order.sort_by_key(|&i| {
                    (
                        group[i].request.device_resolved_geometry,
                        group[i].request.has_user_mask,
                        group[i].hook_program,
                        // STRUCTURAL S-2 (found by AC-0: the lora x
                        // depth pair PANICKED this parity assert — the
                        // reference comparator must carry every
                        // seriation term the plan's key carries).
                        group[i].request.max_layers.is_some(),
                        !group[i]
                            .request
                            .qo_indptr
                            .windows(2)
                            .any(|w| w[1] - w[0] > 1),
                    )
                });
                order
            },
            "fire plan order must equal the stable sort it replaced"
        );
        let mut slots: Vec<Option<Box<PendingRequest>>> = group.into_iter().map(Some).collect();
        let group: Vec<Box<PendingRequest>> = plan
            .member_order
            .iter()
            .map(|&index| slots[index].take().expect("member_order is a permutation"))
            .collect();
        let wire_count = group
            .iter()
            .take_while(|req| !req.request.device_resolved_geometry)
            .count();
        let envelope_count = group.len() - wire_count;
        let mut sub_batch_indptr: Vec<u32> = vec![0];
        let mut sub_batch_class: Vec<u32> = Vec::new();
        if wire_count > 0 {
            sub_batch_indptr.push(wire_count as u32);
            sub_batch_class.push(pie_driver_abi::PIE_GEOMETRY_CLASS_HOST);
        }
        if envelope_count > 0 {
            sub_batch_indptr.push(group.len() as u32);
            sub_batch_class.push(pie_driver_abi::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE);
        }
        let build = build_batch_request(&group, page_size, stats);
        let mut roster_rows: Vec<u32> = Vec::with_capacity(build.instance_ids.len());
        for (member_index, (member, req)) in build.instance_ids.iter().zip(&group).enumerate() {
            let row = *roster_index.entry(*member).or_insert_with(|| {
                roster.push(*member);
                lane_translation.push(Vec::new());
                (roster.len() - 1) as u32
            });
            roster_rows.push(row);
            let segment = &build.kv_translation[build.kv_translation_indptr[member_index] as usize
                ..build.kv_translation_indptr[member_index + 1] as usize];
            if !segment.is_empty() {
                lane_translation[row as usize] = segment.to_vec();
            }
            // Declared high-water + WIRE page maxima only (v13 semantics):
            // translation ids are physical placements, not demand — folding
            // them in demanded a full-arena commit and broke long-context
            // shapes that oversubscribe KV through the reclaim layer.
            required_kv_pages = required_kv_pages.max(req.request.required_kv_pages).max(
                req.request
                    .kv_page_indices
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |page| page.saturating_add(1)),
            );
        }
        required_kv_pages = required_kv_pages.max(build.plan.required_kv_pages);
        // The planner's first CONSUMED lowering (fire_plan module doc):
        // the qkv_postprocess site's Prefix{fast_rows} — member counts —
        // converted to WIRE request rows through the attribution CSR and
        // handed across; the driver cross-checks it against its own
        // compiled-plan derivation and refuses the launch on drift.
        let planned_hook_free_prefix_rows =
            planned_prefix_wire_rows(&plan, &group, &build.program_row_indptr);
        let planned_unmasked_prefix_rows =
            planned_unmasked_prefix_wire_rows(&plan, &group, &build.program_row_indptr);
        // STRUCTURAL S-2: a planned depth union stamps the SUFFIX's
        // uniform k onto the merged plan (the wire merge does not carry
        // per-member max_layers); a DECLINED composed shape leaves it
        // None — every member runs full depth, the safe degradation of
        // an advisory truncation.
        let planned_full_depth_rows = planned_full_depth_request_split(&group);
        let mut merged_plan = build.plan;
        if planned_full_depth_rows != pie_driver_abi::PIE_FULL_DEPTH_UNPLANNED {
            merged_plan.max_layers = group
                .iter()
                .find_map(|r| r.request.max_layers);
        } else if let Some(k) = group[0].request.max_layers {
            // The uniform half of the PQ-tree cell: when EVERY member
            // shares one truncation, the fire-level layer bound cuts
            // every row — mask-compatible (the attention arms operate
            // inside [0, k) unchanged) — so a declined SPLIT must not
            // silently drop the members' k (found by the arc-78 probe:
            // the wire merge discards per-member max_layers).
            if group.iter().all(|r| r.request.max_layers == Some(k)) {
                merged_plan.max_layers = Some(k);
            }
        }
        steps.push(StepSubmission {
            plan: merged_plan,
            roster_rows,
            sub_batch_indptr,
            sub_batch_class,
            terminal_cells: build.terminal_cells,
            program_row_indptr: build.program_row_indptr,
            planned_hook_free_prefix_rows,
            planned_unmasked_prefix_rows,
            planned_full_depth_rows,
            logical_fire_ids: build.logical_fire_ids,
            channel_expected_head: build.channel_expected_head,
            channel_expected_tail: build.channel_expected_tail,
            channel_ticket_indptr: build.channel_ticket_indptr,
        });
        flattened.extend(group);
    }

    let mut kv_translation: Vec<u32> = Vec::new();
    let mut kv_translation_indptr: Vec<u32> = Vec::with_capacity(roster.len() + 1);
    kv_translation_indptr.push(0);
    for segment in &lane_translation {
        kv_translation.extend(segment.iter().copied());
        kv_translation_indptr.push(kv_translation.len() as u32);
    }
    (
        FrameSubmission {
            instance_ids: roster,
            kv_translation,
            kv_translation_indptr,
            required_kv_pages,
            steps,
        },
        flattened,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{LaunchPlan, WorkItemCompletion};

    fn pending(request: LaunchPlan, instance_id: u64, prebuilt: bool) -> Box<PendingRequest> {
        Box::new(PendingRequest {
            logical_fire_id: 1,
            last_page_len: 1,
            request,
            instance_id,
            completion: WorkItemCompletion::new(instance_id, 0),
            process_id: None,
            pipeline_id: None,
            prebuilt,
            hook_program: false,
            lora_program: false,
            page_mask_program: false,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: None,
            timing: None,
        })
    }

    fn wire_decode(token: u32, page: u32) -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![token],
            position_ids: vec![0],
            kv_page_indices: vec![page],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![1],
            qo_indptr: vec![0, 1],
            sampling_indices: vec![0],
            sampling_indptr: vec![0, 1],
            mask_indptr: vec![0, 0],
            single_token_mode: true,
            ..LaunchPlan::default()
        }
    }

    /// The driver-reported model sites are INFORMATIONAL this increment
    /// (nothing consumes a fire plan's site vec downstream — v0): sealing a
    /// frame with an MoE summary's expert site merged produces a submission
    /// identical to sealing without it, while the debug assert inside
    /// `build_frame_submission` pins that the merged plan really carried
    /// the site through `plan_fire_with_model`.
    #[test]
    fn model_sites_are_informational_for_the_submission() {
        let limits = SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
        };
        let waves = || {
            vec![vec![
                pending(wire_decode(11, 3), 1, false),
                pending(wire_decode(22, 4), 2, false),
            ]]
        };
        let stats = SchedulerStats::default();
        let (without_sites, retired) = build_frame_submission(waves(), limits, 16, &stats, &[]);
        assert_eq!(retired.len(), 2);

        let model_sites = [fire_plan::expert_weights_site(256, 8)];
        let (with_sites, retired) =
            build_frame_submission(waves(), limits, 16, &stats, &model_sites);
        assert_eq!(retired.len(), 2);
        // Terminal cells are per-completion heap pointers, distinct between
        // the two constructions by nature; everything else must agree.
        let scrub = |mut submission: FrameSubmission| {
            for step in &mut submission.steps {
                step.terminal_cells.clear();
            }
            submission
        };
        assert_eq!(scrub(without_sites), scrub(with_sites));
    }

    #[test]
    fn batched_fires_attribute_one_row_each() {
        // Two wire decodes around a device-geometry placeholder: every fire
        // owns exactly one wire request row; the placeholder's row is empty
        // of tokens/sampling but still holds its boundary.
        let dg = LaunchPlan {
            kv_translation: vec![7, 8],
            ..LaunchPlan::default()
        };
        let requests = vec![
            pending(wire_decode(11, 3), 1, false),
            pending(dg, 2, true),
            pending(wire_decode(22, 4), 3, false),
        ];
        let mut requests = requests;
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 1, 2, 3]);
        assert_eq!(
            sub.plan.qo_indptr,
            vec![0, 1, 1, 2],
            "placeholder row is empty"
        );
        assert_eq!(sub.plan.sampling_indptr, vec![0, 1, 1, 2]);
        assert_eq!(sub.kv_translation, vec![7, 8]);
        assert_eq!(sub.kv_translation_indptr, vec![0, 0, 2, 2]);
    }

    #[test]
    fn batch_preserves_largest_required_kv_high_water() {
        let mut first = wire_decode(11, 3);
        first.kv_translation = vec![3, 16];
        let mut second = wire_decode(22, 4);
        second.kv_translation = vec![8, 28];
        let mut requests = [pending(first, 1, false), pending(second, 2, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());

        assert_eq!(sub.plan.required_kv_pages, 29);
    }

    #[test]
    fn prebuilt_solo_owns_all_wire_rows() {
        // A prebuilt wire plan with two lanes: its single program owns both
        // rows. A device-geometry prebuilt (empty qo) owns zero rows.
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.rs_slot_ids = vec![31, 32];
        two_lane.rs_slot_flags = vec![0, crate::driver::RS_FLAG_RESET];
        let mut solo = [pending(two_lane, 1, true)];
        let sub = build_batch_request(&mut solo, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 2]);
        assert_eq!(sub.plan.qo_indptr, vec![0, 1, 2]);
        assert_eq!(sub.plan.rs_slot_ids, vec![31, 32]);

        let dg = LaunchPlan::default();
        let mut solo_dg = [pending(dg, 2, true)];
        let sub = build_batch_request(&mut solo_dg, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 0]);
    }

    #[test]
    fn ordinary_multi_row_submission_remains_intact() {
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.mask_indptr = vec![0, 0, 0];
        two_lane.rs_slot_ids = vec![17, 23];
        two_lane.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];
        let expected = two_lane.clone();

        let mut ordinary = [pending(two_lane, 9, false)];
        let sub = build_batch_request(&mut ordinary, 16, &SchedulerStats::default());

        assert_eq!(sub.instance_ids, vec![9]);
        assert_eq!(sub.program_row_indptr, vec![0, 2]);
        assert_eq!(sub.plan.qo_indptr, expected.qo_indptr);
        assert_eq!(sub.plan.kv_page_indptr, expected.kv_page_indptr);
        assert_eq!(sub.plan.sampling_indptr, expected.sampling_indptr);
        assert_eq!(sub.plan.rs_slot_ids, vec![17, 23]);
        assert_eq!(
            sub.plan.rs_slot_flags,
            vec![crate::driver::RS_FLAG_RESET, 0]
        );
    }

    /// Multi-row masks without a row CSR reject the FIRE at admission —
    /// the batch build treats the shape as an invariant, and a panic there
    /// would take down the scheduler thread (RV-20).
    #[test]
    fn multi_row_masks_without_a_row_csr_reject_at_admission() {
        let accumulator = AdmissionLimits::new(
            SchedulerLimits {
                max_forward_requests: 64,
                max_forward_tokens: 64,
                max_page_refs: 64,
            },
            16,
        );
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.masks = vec![crate::driver::command::EncodedMask::new(vec![1], 1)];
        two_lane.mask_indptr = vec![0, 1];

        let message = accumulator
            .single_request_limit_error(&pending(two_lane.clone(), 9, false))
            .expect("multi-row masks without a row CSR must reject the fire");
        assert!(message.contains("without a row CSR"), "{message}");

        // The same fire with a row CSR is admitted.
        two_lane.mask_indptr = vec![0, 1, 1];
        assert_eq!(
            accumulator.single_request_limit_error(&pending(two_lane, 9, false)),
            None
        );
    }

    #[test]
    fn multi_row_submission_cobatches_without_collapsing_csrs() {
        let mut two_lane = wire_decode(11, 3);
        two_lane.token_ids = vec![11, 22];
        two_lane.position_ids = vec![0, 0];
        two_lane.qo_indptr = vec![0, 1, 2];
        two_lane.kv_page_indices = vec![3, 4];
        two_lane.kv_page_indptr = vec![0, 1, 2];
        two_lane.kv_last_page_lens = vec![1, 1];
        two_lane.sampling_indices = vec![0, 0];
        two_lane.sampling_indptr = vec![0, 1, 2];
        two_lane.mask_indptr = vec![0, 0, 0];
        two_lane.rs_slot_ids = vec![17, 23];
        two_lane.rs_slot_flags = vec![crate::driver::RS_FLAG_RESET, 0];

        let mut requests = [
            pending(two_lane, 9, false),
            pending(wire_decode(33, 5), 10, false),
        ];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());

        assert_eq!(sub.program_row_indptr, vec![0, 2, 3]);
        assert_eq!(sub.plan.qo_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.kv_page_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.sampling_indptr, vec![0, 1, 2, 3]);
        assert_eq!(sub.plan.sampling_indices, vec![0, 0, 0]);
        assert_eq!(sub.plan.rs_slot_ids, vec![17, 23]);
        assert_eq!(sub.plan.image_indptr, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.audio_indptr, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.embed_block_indptr, vec![0, 0, 0, 0]);
    }

    #[test]
    fn device_resolved_multitoken_geometry_skips_placeholder_mask_trim() {
        let mut request = wire_decode(11, 3);
        request.token_ids = vec![11, 0, 0, 0];
        request.position_ids = vec![0; 4];
        request.qo_indptr = vec![0, 4];
        request.kv_page_indices = vec![3, 4, 5];
        request.kv_page_indptr = vec![0, 3];
        request.kv_last_page_lens = vec![6];
        request.sampling_indices = vec![0, 1, 2, 3];
        request.sampling_indptr = vec![0, 4];
        request.single_token_mode = false;
        request.device_resolved_geometry = true;

        let mut requests = [pending(request, 12, false)];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.kv_page_indices, vec![3, 4, 5]);
        assert!(sub.plan.masks.is_empty());
    }

    #[test]
    fn host_custom_mask_cobatches_with_causal_fire() {
        let mut custom = wire_decode(11, 3);
        custom.has_user_mask = true;
        custom.single_token_mode = false;
        custom.masks = vec![crate::driver::command::EncodedMask::new(vec![1], 1)];
        custom.mask_indptr = vec![0, 1];
        let mut requests = [
            pending(custom, 20, false),
            pending(wire_decode(22, 4), 21, false),
        ];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.instance_ids, vec![20, 21]);
        assert_eq!(sub.plan.mask_indptr, vec![0, 1, 2]);
        assert_eq!(sub.plan.masks.len(), 2);
        assert_eq!(sub.plan.masks[0].runs, vec![1], "explicit custom row");
        assert_eq!(
            sub.plan.masks[1].runs,
            vec![0, 1],
            "causal peer receives the synthesized compatible row"
        );
        assert!(sub.plan.has_user_mask);
        assert!(!sub.plan.single_token_mode);
    }

    #[test]
    fn host_mask_on_device_geometry_is_not_elided_as_dense() {
        let mut request = wire_decode(11, 3);
        request.device_resolved_geometry = true;
        request.has_user_mask = true;
        request.single_token_mode = false;
        request.masks = vec![crate::driver::command::EncodedMask::new(vec![0, 1], 1)];
        request.mask_indptr = vec![0, 1];

        let mut requests = [pending(request, 12, false)];
        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.masks.len(), 1);
        assert_eq!(sub.plan.mask_indptr, vec![0, 1]);
        assert!(sub.plan.has_user_mask);
    }

    #[test]
    fn deferred_multi_row_geometry_cobatches_as_zero_kv_spans() {
        let mut plan = wire_decode(11, 3);
        plan.token_ids = vec![11, u32::MAX];
        plan.position_ids = vec![0, 0];
        plan.qo_indptr = vec![0, 1, 2];
        plan.kv_page_indices.clear();
        plan.kv_page_indptr.clear();
        plan.kv_last_page_lens.clear();
        plan.sampling_indices = vec![0, 0];
        plan.sampling_indptr = vec![0, 1, 2];
        plan.mask_indptr = vec![0, 0, 0];
        plan.device_resolved_geometry = true;
        plan.kv_write_lower_bounds = vec![7];
        plan.kv_write_upper_bounds = vec![15];
        let mut requests = [pending(plan.clone(), 20, false), pending(plan, 21, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.program_row_indptr, vec![0, 2, 4]);
        assert_eq!(sub.plan.kv_page_indices, Vec::<u32>::new());
        assert_eq!(sub.plan.kv_page_indptr, vec![0, 0, 0, 0, 0]);
        assert_eq!(sub.plan.kv_last_page_lens, vec![0, 0, 0, 0]);
        assert_eq!(sub.plan.kv_write_lower_bounds, vec![7, 7]);
        assert_eq!(sub.plan.kv_write_upper_bounds, vec![15, 15]);
    }

    #[test]
    fn mixed_batches_fill_unbounded_containment_entries() {
        let plain = wire_decode(11, 3);
        let mut bounded = wire_decode(22, 4);
        bounded.kv_write_lower_bounds = vec![7];
        bounded.kv_write_upper_bounds = vec![9];
        let mut requests = [pending(plain, 20, false), pending(bounded, 21, false)];

        let sub = build_batch_request(&mut requests, 16, &SchedulerStats::default());
        assert_eq!(sub.plan.kv_write_lower_bounds, vec![0, 7]);
        assert_eq!(sub.plan.kv_write_upper_bounds, vec![u64::MAX, 9]);
    }
}
