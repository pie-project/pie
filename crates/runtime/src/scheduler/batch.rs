//! Batch assembly: capacity accounting, and the lanes a step is made of.
//!
//! # What the merge stopped being
//!
//! `build_batch_request` used to run eleven simultaneous CSR merges through
//! `scheduler::wire` — tokens, positions, page lists, mask rows, sampler
//! indices, recurrent slots, three multimodal side channels — so that an
//! engine could walk them back into per-request form. A batch is a
//! concatenation of lanes now (`crate::engine::fire`'s header), and the
//! merge is `extend`.
//!
//! What survived, because none of it was about the wire form: the admission
//! shape gate, the per-request capacity accounting, the grouping rules that
//! decide which fires may share a step, and the fire planner's seriation.

use crate::engine::completion::TerminalCell;
use crate::engine::{FrameFire, SchedulerLimits, StepFire};

use super::fire_plan;
use super::stats::SchedulerStats;
use super::worker::PendingRequest;

/// One step's assembled lanes, before the frame is sealed.
pub(crate) struct StepBuild {
    /// The lanes, in member order.
    pub(crate) lanes: Vec<crate::engine::Lane>,
    /// Which bound instance each MEMBER belongs to.
    pub(crate) instance_ids: Vec<u64>,
    /// Whether each MEMBER's instance runs a pass at the fire's boundary —
    /// [`FireRequest::boundary_program`](crate::engine::FireRequest).
    pub(crate) boundary_programs: Vec<bool>,
    /// Which lane each member's rows start at — the attribution the region
    /// tables used to carry.
    pub(crate) member_lane_indptr: Vec<u32>,
    /// One cell per member.
    pub(crate) terminal_cells: Vec<*mut TerminalCell>,
    /// One id per member.
    pub(crate) logical_fire_ids: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestCapacityUsage {
    pub(crate) forward_requests: usize,
    pub(crate) forward_tokens: usize,
    pub(crate) page_refs: usize,
}

pub(crate) fn request_capacity_usage(req: &PendingRequest, page_size: u32) -> RequestCapacityUsage {
    let forward_requests = req
        .wire_row_count()
        .max(req.request.rs.slot_ids.len())
        .max(1);
    let _ = page_size;

    RequestCapacityUsage {
        forward_requests,
        forward_tokens: req.request.tokens(),
        page_refs: req.request.pages().count(),
    }
}

/// Admission-time shape gate: rejects a single fire whose resolved shape
/// exceeds the engine's launch limits before it can enter the queue.
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
                "forward request has {} resolved rows, exceeding engine limit {}",
                usage.forward_requests, self.limits.max_forward_requests
            ));
        }
        if usage.forward_tokens > self.limits.max_forward_tokens {
            return Some(format!(
                "forward request has {} forward tokens, exceeding engine limit {}",
                usage.forward_tokens, self.limits.max_forward_tokens
            ));
        }
        if usage.page_refs > self.limits.max_page_refs {
            return Some(format!(
                "forward request has {} page refs, exceeding engine limit {}",
                usage.page_refs, self.limits.max_page_refs
            ));
        }
        // THE TWO MALFORMED-SHAPE CHECKS THAT STOOD HERE ARE UNREPRESENTABLE
        // NOW, which is the whole reason they were checks. "per-fire KV
        // containment bounds must be empty or scalar" guarded two parallel
        // `Vec<u64>`s that had to be the same length and at most one entry —
        // it is `Option<(u64, u64)>` (`FireRequest::kv_write_bounds`). "a
        // multi-row fire carries masks without a row CSR" guarded a flat
        // `masks` vector against a `mask_indptr` that cut it per lane — a
        // mask lives on its lane (`Lane::mask`), so a mask row that belongs
        // to no lane cannot be built. Both refusals are gone with the shapes
        // that could fail them.
        None
    }
}

/// Assemble one step's lanes out of its member requests.
///
/// **THIS IS A CONCATENATION.** Every member contributes its own lanes, in
/// order, and the step's lanes are those lanes. The prebuilt/multi-row
/// special case the merge needed — flattening a multi-lane request through
/// the CSR merge would collapse its inner rows to one while incorrectly
/// retaining its recurrent-state slots — cannot arise, because nothing is
/// flattened.
pub(crate) fn build_batch_request(
    requests: &[Box<PendingRequest>],
    page_size: u32,
    stats: &SchedulerStats,
) -> StepBuild {
    let _ = page_size;
    crate::probe_fire!(stats.fire.execute.batch_build_us, {
        let mut lanes = Vec::with_capacity(requests.len());
        let mut instance_ids = Vec::with_capacity(requests.len());
        let mut boundary_programs = Vec::with_capacity(requests.len());
        let mut terminal_cells = Vec::with_capacity(requests.len());
        let mut logical_fire_ids = Vec::with_capacity(requests.len());
        let mut member_lane_indptr = Vec::with_capacity(requests.len() + 1);
        member_lane_indptr.push(0);
        for req in requests {
            boundary_programs.push(req.request.boundary_program);
            instance_ids.push(req.instance_id);
            terminal_cells.push(req.completion.terminal_cell_ptr());
            logical_fire_ids.push(req.logical_fire_id);
            lanes.extend(req.request.lanes.iter().cloned());
            member_lane_indptr.push(u32::try_from(lanes.len()).unwrap_or(u32::MAX));
        }
        StepBuild {
            lanes,
            instance_ids,
            boundary_programs,
            member_lane_indptr,
            terminal_cells,
            logical_fire_ids,
        }
    })
}

/// Assemble one sealed frame's fires from its waves' picked requests, in slot
/// order. Returns the frame plus the flattened requests in POST order (step
/// order, member order within a step) — the retirement set.
///
/// Step formation preserves the old per-wave batch semantics exactly: within
/// a wave, requests group under [`LaunchGrouping`](super::worker::LaunchGrouping)'s
/// compatibility rules (instance/pipeline dedup, geometry-class homogeneity,
/// mask/solo exclusions, structural budgets); each group becomes one step.
///
/// # What the frame stopped carrying
///
/// * **the roster** (`instance_ids` + `roster_rows`) — a per-frame table of
///   bound instances that the step tables indexed. `StepFire::instances` is
///   the same association said per lane, and the members that run a pass at
///   the fire's boundary become [`Attachment`](engine_api::fire::Attachment)s
///   of the submission itself.
/// * **`kv_translation` / `kv_translation_indptr`** — a per-roster-lane page
///   rewrite, gathered from each lane's LAST fire in the frame. It is
///   [`KvDelta::translation`](engine_api::KvDelta), on the lane it is about.
/// * **`required_kv_pages`** — the frame-union page high-water, so an engine
///   could refuse before it started. An engine makes that check for itself and
///   answers [`Exhausted`](engine_api::Error::Exhausted) with the two
///   numbers in it.
/// * **`sub_batch_indptr` / `sub_batch_class`** — the runs of equal geometry
///   class, which the fire planner's own sort key already produced and this
///   table described a second time.
/// * **the region table** (`region_row_indptr`/`region_sig`/`region_k`) —
///   tart's seriation output, stated for an engine that derived its planned
///   splits from it. The palo shell derives every window from the lanes'
///   words (`engine::fire::compose`), which is the same decision made from
///   the same facts one layer down; the six `PIE_REGION_SIG_*` bits it was
///   encoded in are gone with `engine-api::plan`.
#[allow(
    clippy::vec_box,
    reason = "measured: `PendingRequest` is 1408 bytes. This vec is not a store but a \
              conveyor — requests are moved wave -> step_groups -> deferred -> back \
              out repeatedly in this function — and the box makes each of those moves \
              8 bytes instead of 1408. Unboxing would trade one allocation per request \
              for a 1408-byte memcpy on every shuffle and every Vec regrow"
)]
pub(crate) fn build_frame_submission(
    waves: Vec<Vec<Box<PendingRequest>>>,
    limits: SchedulerLimits,
    page_size: u32,
    stats: &SchedulerStats,
) -> (FrameFire, Vec<Box<PendingRequest>>) {
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
            for req in deferred {
                if closed || !grouping.accepts(&req, limits, page_size) {
                    rest.push(req);
                    continue;
                }
                closed = grouping.push(&req, limits, page_size);
                group.push(req);
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

    let mut steps: Vec<StepFire> = Vec::new();
    let mut flattened: Vec<Box<PendingRequest>> = Vec::new();

    for group in step_groups {
        // tart (0.3 re-port step 1): the fire planner's seriation. It still
        // decides the SUBMISSION order — which is a real decision about
        // which members share a window's rows — and it no longer has to
        // produce a table describing itself, because the shell seriates the
        // lanes it is given by class (`engine::fire::compose`) and the
        // runtime's order is what it seriates within.
        let facts: Vec<fire_plan::MemberFacts> = group
            .iter()
            .enumerate()
            .map(|(arrival, req)| fire_plan::MemberFacts {
                hook_program: req.hook_program,
                lora: req.lora_program,
                custom_mask: req.request.has_user_mask,
                truncated: req.request.max_layers.is_some(),
                max_layers: req.request.max_layers,
                multi_token: req.request.lanes.iter().any(|lane| lane.tokens.len() > 1),
                geometry_class: req.request.geometry,
                arrival,
            })
            .collect();
        let plan = fire_plan::plan_fire_with_model(&facts, &[]);
        let mut slots: Vec<Option<Box<PendingRequest>>> = group.into_iter().map(Some).collect();
        let group: Vec<Box<PendingRequest>> = plan
            .member_order
            .iter()
            .map(|&index| slots[index].take().expect("member_order is a permutation"))
            .collect();

        let build = build_batch_request(&group, page_size, stats);
        if super::worker::wave_trace() {
            eprintln!(
                "[step-lanes] members={} lanes={}",
                group.len(),
                build.lanes.len()
            );
        }
        // ONE INSTANCE ID PER LANE, not per member: a member with several
        // lanes is one bound instance firing several row groups.
        let mut instances = Vec::with_capacity(build.lanes.len());
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            let span = build.member_lane_indptr[member + 1] - build.member_lane_indptr[member];
            instances.extend(std::iter::repeat_n(instance, span as usize));
        }
        // THE ATTACHMENTS (`palo B2`, design §9). One per MEMBER and not per
        // lane, because a program's stages are one pass with one readiness
        // gate and one commit: a member that fires three row groups is still
        // one bound instance running one pass, and naming it three times
        // would commit its channels three times. The lane it names is the
        // member's FIRST — the row group whose readout row an epilogue's
        // `logits` intrinsic is pointed at.
        //
        // `Boundary::Epilogue` and never `Prologue`, because a guest pass at
        // a model fire is decode logic over the fire's own logits, and before
        // the graph there is no readout to point at. A member whose request
        // did not say it runs a pass at the boundary contributes nothing, so
        // a prebuilt rider's submission is byte-identical to what it was.
        let mut attachments = Vec::new();
        let mut lanes = build.lanes;
        for (member, &instance) in build.instance_ids.iter().enumerate() {
            if !build.boundary_programs[member] {
                continue;
            }
            let lane = build.member_lane_indptr[member];
            // ── THE PREDICTIONS, ONTO THE LANE THAT CARRIES THE PASS (alto
            //    design §1 article 3). The reservation this member's submit
            //    minted — one `(head, tail)` per channel, counted, never read
            //    off a device — travels on the ATTACHED lane, because that is
            //    the lane whose instance the tickets are about.
            //
            //    Only for an engine that adopted the instance's channels:
            //    `device_channel_tickets` is the runtime's own per-channel
            //    spelling of `Capabilities::device_channel_commit`, and an
            //    engine without the two control kernels refuses a stated
            //    prediction by name rather than ignoring it.
            if let Some(target) = lanes.get_mut(lane as usize)
                && group[member].request.device_channel_tickets
            {
                target.channels = group[member]
                    .request
                    .tickets
                    .iter()
                    .map(|ticket| engine_api::Ticket {
                        channel: ticket.channel,
                        expected_head: ticket.head,
                        expected_tail: ticket.tail,
                    })
                    .collect();
            }
            attachments.push(crate::engine::Attachment {
                lane,
                instance,
                at: crate::engine::Boundary::Epilogue,
            });
        }
        steps.push(StepFire {
            submission: crate::engine::Step {
                lanes,
                attachments,
            },
            terminal_cells: build.terminal_cells,
            instances,
            logical_fire_ids: build.logical_fire_ids,
        });
        flattened.extend(group);
    }

    (FrameFire { steps }, flattened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FireRequest, WorkItemCompletion};
    use tensor_ir::registry::GeometryClass;

    fn pending(request: FireRequest, instance_id: u64, prebuilt: bool) -> Box<PendingRequest> {
        Box::new(PendingRequest {
            hook_program: false,
            lora_program: false,
            logical_fire_id: 1,
            request,
            instance_id,
            completion: WorkItemCompletion::new(instance_id, 0),
            process_id: None,
            pipeline_id: None,
            prebuilt,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: None,
        })
    }

    /// One decode lane: one token, one page, last-row readout.
    fn decode(token: u32, page: u32) -> FireRequest {
        let mut request = FireRequest::one(crate::engine::fire::lane_of(0, vec![token], 0, vec![page]));
        request.single_token_mode = true;
        request
    }

    fn limits() -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
        }
    }

    /// Every member's lanes stand where the attribution says they do.
    ///
    /// This is `batched_fires_attribute_one_row_each`, kept: what it was
    /// really about is that a step can say which member each of its rows
    /// belongs to, which is what the region tables and the retirement set
    /// both read. It used to be checked through `program_row_indptr` against
    /// a merged `qo_indptr`; it is checked through `member_lane_indptr`
    /// against the lanes now, and a member that contributes no lane still
    /// holds its boundary.
    #[test]
    fn every_member_owns_a_span_of_the_steps_lanes() {
        let placeholder = FireRequest::default();
        let requests = vec![
            pending(decode(11, 3), 1, false),
            pending(placeholder, 2, true),
            pending(decode(22, 4), 3, false),
        ];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
        assert_eq!(step.member_lane_indptr, vec![0, 1, 1, 2]);
        assert_eq!(step.instance_ids, vec![1, 2, 3]);
        assert_eq!(step.lanes.len(), 2, "the placeholder contributes no lane");
        assert_eq!(step.lanes[0].tokens, vec![11]);
        assert_eq!(step.lanes[1].tokens, vec![22]);
    }

    /// A multi-lane member keeps every one of its lanes.
    ///
    /// This is `prebuilt_solo_owns_all_wire_rows` and
    /// `ordinary_multi_row_submission_remains_intact` together — they were
    /// two readings of the same claim, that the merge must not collapse a
    /// member's inner rows to one. Nothing is merged, so the claim is that
    /// the lanes arrive verbatim.
    #[test]
    fn a_multi_lane_member_arrives_lane_for_lane() {
        let mut two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        two.rs.slot_ids = vec![31, 32];
        two.rs.slot_flags = vec![0, crate::engine::RS_FLAG_RESET];
        let expected = two.lanes.clone();

        let step = build_batch_request(&[pending(two, 1, true)], 16, &SchedulerStats::default());
        assert_eq!(step.member_lane_indptr, vec![0, 2]);
        assert_eq!(step.lanes, expected);
    }

    /// Two members co-batch by concatenation, in member order.
    #[test]
    fn members_cobatch_by_concatenation() {
        let two = FireRequest {
            lanes: vec![
                crate::engine::fire::lane_of(0, vec![11], 0, vec![3]),
                crate::engine::fire::lane_of(1, vec![22], 0, vec![4]),
            ],
            ..FireRequest::default()
        };
        let requests = [pending(two, 9, false), pending(decode(33, 5), 10, false)];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());

        assert_eq!(step.member_lane_indptr, vec![0, 2, 3]);
        assert_eq!(
            step.lanes
                .iter()
                .map(|lane| lane.tokens.clone())
                .collect::<Vec<_>>(),
            vec![vec![11], vec![22], vec![33]]
        );
    }

    /// A lane's mask is the lane's, and a peer with none gets none.
    ///
    /// This is `host_mask_on_device_geometry_is_not_elided_as_dense`, kept
    /// for the half that survives: a host-lowered mask reaches the
    /// submission. What it can no longer be is *elided* or *synthesized* —
    /// the merge used to push a causal row for every unmasked peer so that
    /// the flat `masks` vector stayed parallel to the rows, and a lane that
    /// carries no mask now simply carries none.
    #[test]
    fn a_masked_lane_carries_its_own_mask_and_a_peer_carries_none() {
        let mut masked = decode(11, 3);
        masked.has_user_mask = true;
        masked.single_token_mode = false;
        masked.device_resolved_geometry = true;
        masked.lanes[0].mask = Some(crate::engine::Mask::new(vec![0, 1], 1));

        let requests = [pending(masked, 20, false), pending(decode(22, 4), 21, false)];
        let step = build_batch_request(&requests, 16, &SchedulerStats::default());
        assert_eq!(
            step.lanes[0].mask,
            Some(crate::engine::Mask::new(vec![0, 1], 1))
        );
        assert_eq!(step.lanes[1].mask, None, "an unmasked peer stays unmasked");
    }

    /// **A BATCHED FIRE'S LANES CARRY DISTINCT SLOTS** — the property
    /// `palo` build log 28 found the tree had never checked, and build log
    /// 29 gave an owner.
    ///
    /// A step is a CONCATENATION of its members' lanes, so the seat each
    /// member states is the only thing keeping two concurrent guests out of
    /// one another's pool slot. Both runtime fire paths used to state zero —
    /// fine for a solo fire, and `Step::validate` refuses the
    /// second lane of any batch built out of them by name ("slot 0 appears
    /// twice in one fire, at lane 1"), which is what took half the lanes off
    /// an eight-guest fleet.
    ///
    /// So this drives the production stamp — `pipeline::fire::
    /// stamp_lane_slots`, against the book two real working sets own their
    /// seats in — and asserts the thing the shell asserts: the assembled
    /// submission is one the contract accepts. It is RED against the
    /// `slot: 0` this replaced, on the recorded message.
    #[test]
    fn two_seated_members_batch_into_a_fire_the_contract_accepts() {
        let model = crate::store::registry::register_model(16, &[8], &[4]);
        let stores = crate::store::registry::get(model, 0);
        let (first_ws, second_ws) =
            crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
                (kv.create_working_set(), kv.create_working_set())
            });

        let mut first = decode(11, 3);
        let mut second = decode(22, 4);
        assert_eq!(
            (first.lanes[0].slot, second.lanes[0].slot),
            (0, 0),
            "both fires arrive at the seat stamp unseated — this is the defect's shape"
        );
        crate::pipeline::fire::stamp_lane_slots(&mut first, &stores, first_ws)
            .expect("a two-slot pool seats one sequence");
        crate::pipeline::fire::stamp_lane_slots(&mut second, &stores, second_ws)
            .expect("and its peer");

        let (frame, _) = build_frame_submission(
            vec![vec![pending(first, 41, false), pending(second, 42, false)]],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let step = &frame.steps[0];
        assert_eq!(step.submission.lanes.len(), 2, "both members co-batch");
        assert_ne!(
            step.submission.lanes[0].slot, step.submission.lanes[1].slot,
            "two concurrent sequences, two pool slots"
        );
        step.submission
            .validate()
            .expect("the contract accepts a fire whose lanes are seated apart");
    }

    /// A pooled device-geometry member seriates into the step's SUFFIX.
    ///
    /// This is `a_pooled_device_geometry_member_is_stamped_its_own_class`
    /// and `a_mixed_step_names_a_sub_batch_per_class`, kept as the claim
    /// that outlived the tables they checked. `sub_batch_class` was the
    /// planner's own sort key written down a second time for the engine;
    /// what it was ever protecting is that a device-resolved member is a
    /// contiguous suffix and never interleaved with the host members, and
    /// that is a property of the ORDER, which is still the runtime's.
    #[test]
    fn a_device_geometry_member_seriates_into_the_suffix() {
        let mut pooled = decode(11, 3);
        pooled.geometry = GeometryClass::DeviceGeometry;
        // Such a fire states its pages in a channel, not in the submission.
        pooled.lanes[0].kv.pages.clear();

        let (frame, flattened) = build_frame_submission(
            vec![vec![pending(pooled, 13, false), pending(decode(22, 4), 12, false)]],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let step = &frame.steps[0];
        assert_eq!(step.submission.lanes.len(), 2);
        assert_eq!(
            flattened
                .iter()
                .map(|req| req.request.geometry)
                .collect::<Vec<_>>(),
            vec![GeometryClass::Host, GeometryClass::DeviceGeometry],
            "host first, device-resolved as the suffix run"
        );
        assert_eq!(step.instances, vec![12, 13]);
    }

    /// A frame's steps carry one terminal cell per member, and no two steps
    /// share one.
    #[test]
    fn a_frames_cells_are_one_per_member_and_never_shared() {
        let (frame, flattened) = build_frame_submission(
            vec![
                vec![pending(decode(11, 3), 1, false)],
                vec![pending(decode(22, 4), 2, false)],
            ],
            limits(),
            16,
            &SchedulerStats::default(),
        );
        let cells: Vec<_> = frame.terminal_cells().collect();
        assert_eq!(cells.len(), flattened.len());
        assert_eq!(
            cells.iter().collect::<std::collections::HashSet<_>>().len(),
            cells.len(),
            "no cell is owned by two members"
        );
    }

    // FIVE TESTS STOOD HERE AND ARE GONE WITH THE SHAPES THEY PINNED:
    //
    // * `batch_preserves_largest_required_kv_high_water` — `required_kv_pages`
    //   was a frame-union high-water the runtime computed so an engine could
    //   refuse early; an engine answers `Error::Exhausted` with the two
    //   numbers itself.
    // * `multi_row_masks_without_a_row_csr_reject_at_admission` — a flat
    //   `masks` vector out of step with the `mask_indptr` cutting it is
    //   unrepresentable: a mask lives on its lane.
    // * `device_resolved_multitoken_geometry_skips_placeholder_mask_trim` —
    //   the page trim went with `scheduler::wire` (see the note at its
    //   `mod` line).
    // * `deferred_multi_row_geometry_cobatches_as_zero_kv_spans` — every
    //   assertion in it was about `kv_page_indptr`/`kv_last_page_lens`
    //   staying parallel across a merge that no longer happens.
    // * `mixed_batches_fill_unbounded_containment_entries` — the merge had
    //   to push `(0, u64::MAX)` for every member with no bounds so the two
    //   parallel `Vec<u64>`s stayed full length; the bounds are
    //   `Option<(u64, u64)>` per request.
}
