//! A device-geometry fire and the wiring of its channels.

use super::*;

/// Whether the pass's program binds `port`.
fn binds_port<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    port: Port,
) -> Anyhow<bool> {
    Ok(ctx
        .resources()
        .get(fwd)?
        .instance
        .program
        .bound
        .container
        .ports
        .iter()
        .any(|binding| binding.port == port))
}

/// Each group's pages a device-placed fire may write, and where they end:
/// its first lane's from its folded rows to its reserve. `Err` refuses a
/// lane past a media span that states no positions, or one with nothing
/// reserved past its folded rows.
fn device_write_ranges(
    groups: &LaneGroups,
    pass_lanes: &PassLanes,
    page: u64,
    states_positions: bool,
) -> Result<(Vec<Vec<u64>>, Vec<u32>), String> {
    let mut write_indexes: Vec<Vec<u64>> = Vec::with_capacity(groups.len());
    let mut write_ends: Vec<u32> = Vec::with_capacity(groups.len());
    for (_, lanes) in groups {
        let lane = lanes[0];
        let place = &pass_lanes.places[lane];
        // Past an M-RoPE span a row rotates behind its slot, which the device picks.
        if !states_positions && !place.spans.is_empty() {
            return Err(format!(
                "pipeline: lane {lane}'s rows land where the device says, past a media span \
                 whose rotation runs behind its slots; state its positions"
            ));
        }
        let reserved = pass_lanes.cursors[lane].lock().unwrap().reserved;
        let writable = device_writable(lane, place.folded, reserved)?;
        write_indexes.push(writable_pages(&writable, page).collect());
        write_ends.push(writable.end);
    }
    Ok((write_indexes, write_ends))
}

pub(super) async fn fire_device_geometry<C: FireContext>(
    ctx: &mut C,
    this: Resource<Pipeline>,
    fwd: Resource<ForwardPass>,
    frame: Option<crate::scheduler::FrameStamp>,
) -> Anyhow<Result<(), String>> {
    if ctx
        .resources()
        .get(&fwd)?
        .bindings
        .lanes
        .iter()
        .any(|lane| !lane.media.is_empty())
    {
        return Ok(Err(
            "pipeline: MediaDeviceGeometry: this pass attached media spans \
             and resolves its token ids on the device, so the host has no \
             token list to scan for their placeholder runs — media rides a \
             host-resolved geometry, where the runs can be checked"
                .to_string(),
        ));
    }
    let (pipe_fires, pipeline_failure, pipeline_scope) =
        match open_pipeline(ctx, &this, &fwd).await? {
            Ok(opened) => opened,
            Err(error) => return Ok(Err(error)),
        };

    let (rs_reps, fold) = {
        let pass = ctx.resources().get(&fwd)?;
        (pass.rs_ws.clone(), fold_of_pass(pass))
    };
    // A recurrent state a device-counted fold left unsettled is read where
    // the engine left it: the fire goes ahead of the settle.
    let pass_lanes = pass_lanes(ctx, &fwd)?;
    // A fold list the host states leaves no lane without a count.
    if let Err(error) = fold.states(pass_lanes.kv.len()) {
        return Ok(Err(format!("pipeline: fire geometry: {error}")));
    }
    let ws = pass_lanes.kv[0].clone();
    let groups = lane_groups(&pass_lanes.kv.iter().map(|ws| ws.id).collect::<Vec<_>>());
    let group_sets: Vec<KvWorkingSet> = groups
        .iter()
        .map(|(_, lanes)| pass_lanes.kv[lanes[0]].clone())
        .collect();
    let stores = crate::store::registry::get(ws.model, ws.engine);
    let pid = ctx.process_id();
    let quorum_pipeline_id = pipeline_scope.scheduler_id();
    for set in &group_sets {
        if let Err(refusal) = set.claim_pipeline_scope(&pipeline_scope) {
            return Ok(Err(format!("pipeline: KV working set {refusal}")));
        }
    }
    {
        let p = ctx.resources().get(&fwd)?;
        if let Some(e) = &p.failed {
            return Ok(Err(format!(
                "pipeline: forward-pass failed by an earlier fire: {e}"
            )));
        }
    }

    // Every page from the frozen prefix up to what `reserve` granted is backed
    // before the fire.
    let page = u64::from(stores.kv_page_size.max(1));
    let states_positions = binds_port(ctx, &fwd, Port::Positions)?;
    // A beam group's pass (`parents`): its beams' recurrent rows are each
    // their own list's (`BEAM-PAGE-LISTS.md` §11).
    let states_parents = binds_port(ctx, &fwd, Port::Parents)?;
    let (write_indexes, write_ends) =
        match device_write_ranges(&groups, &pass_lanes, page, states_positions) {
            Ok(ranges) => ranges,
            Err(refusal) => return Ok(Err(refusal)),
        };

    let split = ctx
        .resources()
        .get(&fwd)?
        .devgeo
        .clone()
        .expect("fire_device_geometry on a non-device-geometry pass");
    let wide = split.windows(2).any(|lane| lane[1] > lane[0] + 1);
    let resolved_qo_indptr = if wide {
        split.clone()
    } else {
        vec![0; pass_lanes.places.len() + 1]
    };
    let rs_ws_ids = match bound_rs_working_sets(ctx, &ws, &rs_reps, &split, &pipeline_scope)? {
        Ok(ids) => ids,
        Err(error) => return Ok(Err(error)),
    };
    // What each working set folds, as far as the host can say before the fire.
    let stated: Vec<DeviceFold> = groups
        .iter()
        .map(|(_, lanes)| {
            let first = lanes[0];
            let place = &pass_lanes.places[first];
            let reserved = pass_lanes.cursors[first].lock().unwrap().reserved;
            device_fold(
                fold.of(first),
                split[first + 1] - split[first],
                reserved.saturating_sub(place.folded),
                appends_folded(&pass_lanes, lanes),
            )
        })
        .collect();
    // The fire stages its rows whole (the device places them, so it is never
    // cut): it asks the activation rung for them as every pass does
    // ([`RungAsk`]), and is staged once the rung takes them. It asks for
    // the recurrent rows it may carry in as well: each lane's are bounded
    // by what stands unfolded in its set ([`device_lane_rows`]).
    let frozen = match lanes_frozen(
        &stores,
        &rs_ws_ids,
        &groups,
        pass_lanes.places.len(),
        crate::model::model().rs_caps().window_state,
    ) {
        Ok(frozen) => frozen,
        Err(error) => return Ok(Err(error)),
    };
    // Each group's bound on what its lanes replay, every lane's, a decode
    // lane's included (what a step is held to by the engine's stated
    // replayed rows).
    let replayed = groups
        .iter()
        .map(|(_, lanes)| {
            let cursor = pass_lanes.cursors[lanes[0]].lock().unwrap();
            let carried = device_lane_rows(0, &cursor, frozen[lanes[0]]).carried;
            carried.saturating_mul(u32::try_from(lanes.len()).unwrap_or(u32::MAX))
        })
        .fold(0u32, u32::saturating_add);
    let counted = FireRows::of(groups.iter().flat_map(|(_, lanes)| {
        let carried = {
            let cursor = pass_lanes.cursors[lanes[0]].lock().unwrap();
            device_lane_rows(0, &cursor, frozen[lanes[0]]).carried
        };
        let split = &resolved_qo_indptr;
        lanes.iter().map(move |&lane| {
            let own = split[lane + 1] - split[lane];
            FireRows {
                own,
                carried: ::engine::fire::lane_carried(own, carried),
            }
        })
    }));
    // A device-placed fire lands no rs snapshot. A lane alone on its set
    // replays every buffered slot of it, one run ending at its first row
    // (`Lineage::Set`); a lane sharing its set reads the slots its mask
    // admits, resolved on the device: not one run as far as the host can
    // say.
    let every_lane_one_run = groups
        .iter()
        .all(|(_, lanes)| lanes.len() == 1 || frozen[lanes[0]].is_none());
    let compact = fire_compact(&stores, every_lane_one_run, false);
    let fire_lanes = u32::try_from(groups.iter().map(|(_, lanes)| lanes.len()).sum::<usize>())
        .unwrap_or(u32::MAX);
    let rows = match replay_rung(&stores, ws.engine, counted, replayed, fire_lanes, compact) {
        Ok(rows) => rows,
        Err(refusal) => return Ok(Err(refusal)),
    };
    let rung_ask = ask_rung(
        &stores,
        ws.engine,
        pid,
        quorum_pipeline_id,
        rows as usize,
        rows,
    );
    if let Some(ask) = rung_ask.as_ref()
        && let Err(refusal) = ask.wait(rows).await
    {
        return Ok(Err(refusal));
    }
    let sets: Vec<(&KvWorkingSet, Written)> = group_sets
        .iter()
        .zip(&write_indexes)
        .map(|(set, indexes)| (set, Written::Pages(indexes.clone())))
        .collect();
    let set_lanes: Vec<Vec<usize>> = if rs_ws_ids.is_empty() {
        Vec::new()
    } else {
        groups.iter().map(|(_, lanes)| lanes.clone()).collect()
    };
    let rs_plan = || {
        device_rs_plan(
            &stores,
            &rs_ws_ids,
            &groups,
            &pass_lanes,
            &split,
            &fold,
            states_parents,
        )
    };
    let reserved = reserve(
        ctx,
        (ws.model, ws.engine),
        &stores,
        quorum_pipeline_id,
        &pipeline_failure,
        &sets,
        &set_lanes,
        &rs_ws_ids,
        rs_plan,
    )
    .await;
    let Reserved {
        leases,
        copies: (copy_src, copy_dst),
        kvtxns,
        rs: mut rs_prepared,
        epoch,
        unfired,
    } = match reserved {
        Ok(reserved) => reserved,
        Err(error) => return Ok(Err(error)),
    };
    let rs_copy = rs_prepared.state_copy();
    let rstxns = RsTxnsGuard::new(ws.model, ws.engine, rs_prepared.txn.take());
    // The device picks the slots a decode writes: each working set's log is
    // cut where it may start writing, its frozen length, unless the engine
    // reports what it wrote. A fold the host can place is frozen with it, in
    // the step `reserve` ended by committing the recurrent state's fold; a
    // fire that does not follow fails the pipeline, and its sets with it,
    // however this call ends (`unfired`).
    let mut logged = Vec::with_capacity(group_sets.len());
    for (((set, indexes), (_, lanes)), stated) in group_sets
        .iter()
        .zip(&write_indexes)
        .zip(&groups)
        .zip(&stated)
    {
        let place = &pass_lanes.places[lanes[0]];
        let known = u64::from(place.folded);
        let freeze = match stated {
            DeviceFold::Count(count) if place.folded_exact => Some(known + u64::from(*count)),
            _ => None,
        };
        let entry = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
            let cut = kv.folded(set.id)?.max(known);
            let seq = begin_piece(kv, set.id, cut, freeze)?.ok_or(
                crate::store::kv::KvStoreError::BadWriteSet {
                    reason: "a fire writes inside its working set's frozen prefix",
                },
            )?;
            let pages = if kv.keeps() && rs_ws_ids.is_empty() && group_sets.len() == 1 {
                let flat = kv.flat_table(set.id)?;
                Some(
                    indexes
                        .iter()
                        .filter_map(|&index| flat.get(index as usize).map(|page| (page.0, index)))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            Ok::<_, crate::store::kv::KvStoreError>((seq, pages, cut))
        });
        match entry {
            Ok(entry) => logged.push(entry),
            Err(error) => {
                let reason = format!("pipeline: KV rows: {error}");
                record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
                return Ok(Err(reason));
            }
        }
    }

    let (completion, instance_id, scheduler, cells, fwd_rep, accesses) = {
        let p = ctx.resources().get_mut(&fwd)?;
        let completion = p.bound_instance.reserve_completion();
        let accesses = p.instance.program.channel_accesses.clone();
        (
            completion,
            p.bound_instance.instance_id,
            p.scheduler.clone(),
            p.cells.clone(),
            fwd.rep(),
            accesses,
        )
    };

    let mask_qo_indptr: Vec<u32> = (0..resolved_qo_indptr.len() as u32).collect();
    let attn_mask = {
        let p = ctx.resources().get(&fwd)?;
        let bound = &p.instance.program.bound;
        let channel_bound_mask = bound.container.ports.iter().any(|binding| {
            binding.port == eta_ir::registry::Port::AttnMask
                && matches!(binding.source, eta_ir::container::PortSource::Channel(_))
        });
        if channel_bound_mask {
            Ok(geometry::FireAttnMask::Device)
        } else {
            let (shadow, shadow_cells) = (&p.host_shadow, &p.cells);
            let mut known = |chan: u32| shadow.fire_value(bound, shadow_cells, chan);
            geometry::evaluate_attn_mask(bound, &mut known, &mask_qo_indptr)
        }
    };
    let attn_mask = match attn_mask {
        Ok(mask) => mask,
        Err(error) => {
            let reason = format!("pipeline: device-geometry attention mask: {error}");
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
    };

    let mut req = crate::engine::FireRequest {
        boundary_program: true,
        lanes: resolved_qo_indptr
            .windows(2)
            .map(|span| {
                ::engine::Lane {
                    tokens: vec![0; (span[1] - span[0]) as usize],
                    ..::engine::Lane::default()
                }
                .into()
            })
            .collect(),
        carried: counted.carried,
        replayed,
        scratch_compact: compact,
        replay_rungs: fire_rungs(&stores, !rs_ws_ids.is_empty()),
        ..crate::engine::FireRequest::default()
    };
    rs_prepared.apply_to(&mut req);
    stamp_lane_facts(ctx, &fwd, &mut req)?;
    if wide {
        let p = ctx.resources().get(&fwd)?;
        let bound = &p.instance.program.bound;
        let readout = bound.container.ports.iter().find_map(|binding| {
            match (&binding.port, &binding.source) {
                (eta_ir::registry::Port::Readout, eta_ir::container::PortSource::Channel(chan)) => {
                    p.host_shadow.fire_value(bound, &p.cells, *chan)
                }
                _ => None,
            }
        });
        if let Some(rows) = readout {
            let rows = geometry::value_as_u32(&rows);
            for (lane, span) in req.lanes.iter_mut().zip(resolved_qo_indptr.windows(2)) {
                let local: Vec<u32> = rows
                    .iter()
                    .filter(|&&row| row >= span[0] && row < span[1])
                    .map(|&row| row - span[0])
                    .collect();
                lane.readout = ::engine::Readout::Rows(local);
            }
        }
    }
    let fire_wide_mask = matches!(attn_mask, geometry::FireAttnMask::Device);
    if let Err(error) = attn_mask.apply_to(&mut req) {
        let reason = format!("pipeline: device-geometry attention mask: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    let switched = Switched::of(ctx.resources().get(&fwd)?);
    // A split narrow enough to stage no host rows still knows each lane's
    // rows; its words carry them.
    let device_rows: Option<Vec<u32>> =
        (!wide).then(|| split.windows(2).map(|lane| lane[1] - lane[0]).collect());
    let lanes_of = every_lane(&req);
    stamp_lane_words(
        &mut req,
        &switched,
        &lanes_of,
        fire_wide_mask,
        device_rows.as_deref(),
    );
    let reads_each_other = reads_each_other(&req);
    if let Err(refusal) = crate::inferlet::process::said(
        "seat the lanes",
        ctx.process_id(),
        seat_lane_slots(&mut req, &stores, &groups),
    )
    .await
    {
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    if let Err(refusal) = stamp_lane_translation(&mut req, &stores, &groups) {
        record_submit_failure(ctx, &fwd, &pipeline_failure, &refusal);
        return Ok(Err(refusal));
    }
    // The device places the rows, so their engine holds each lane to what
    // its set wrote ([`::engine::extent`]), from where the host knows it.
    for (lane, place) in req.lanes.iter_mut().zip(&pass_lanes.places) {
        lane.kv.readable = place.readable;
        lane.kv.set = place.set;
    }
    req.geometry = GeometryClass::DeviceGeometry;
    tracing::debug!(
        geometry = ?req.geometry,
        lanes = req.lanes.len(),
        qo_indptr = ?resolved_qo_indptr,
        reads_rows = switched.rows_read,
        "fire geometry resolved on device path"
    );
    let ticket_reservation = TicketReservation::new(&cells, &accesses);
    ticket_reservation.apply_to(&mut req);

    let lora_program = lora_program_of(ctx, &fwd)?;
    // A decode lane whose KV is its tokens' alone asks the engine what it
    // wrote, so its slots are logged.
    let plain_decode =
        !reads_each_other && switched.plain() && plain_decode(&req, lora_program, fire_wide_mask);
    let mut parts = Vec::with_capacity(group_sets.len());
    for ((((((set, (log_seq, log_pages, boundary)), lease), kvtxn), (_, lanes)), &stated), &end) in
        group_sets
            .iter()
            .zip(logged)
            .zip(leases)
            .zip(kvtxns)
            .zip(&groups)
            .zip(&stated)
            .zip(&write_ends)
    {
        // Each lane's mask speaks of the slots from the set's boundary on,
        // and its rows land between it and what `reserve` granted: the
        // slots of the pages the fire made the set's own.
        for &lane in lanes {
            if let Some(lane) = req.lanes.get_mut(lane) {
                lane.kv.folded = u32::try_from(boundary).unwrap_or(u32::MAX);
                lane.kv.write_end = end;
            }
        }
        let mut tags = Vec::new();
        let rows = match log_pages.filter(|_| plain_decode) {
            Some(pages) => LogRows::Reported {
                tag: report_on(&mut req, 0, &mut tags),
                pages,
                page_size: stores.kv_page_size,
            },
            None => LogRows::Unknown,
        };
        let reported = matches!(rows, LogRows::Reported { .. });
        if let (DeviceFold::PastRows(_), false) = (stated, reported) {
            let first = lanes[0];
            let reason = past_rows_refusal(first, split[first + 1] - split[first]);
            record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
            return Ok(Err(reason));
        }
        // What the host could place was frozen as the fire was prepared; the
        // rest is placed when the fire settles.
        let rs = rs_ws_ids.get(lanes[0]).copied();
        let by = device_fold_at(
            &pass_lanes,
            lanes[0],
            stated,
            reported,
            rs,
            &mut req,
            &mut tags,
        );
        // A set with a recurrent state counts the fire's rows unfolded
        // until it settles, and each of its lanes reports where its rows
        // ended.
        let staged = frozen[lanes[0]].map(|_| StagedFire {
            rows: crate::inferlet::host::working_set::StagedRows::new(
                lanes
                    .iter()
                    .map(|&lane| split[lane + 1] - split[lane])
                    .sum(),
            ),
            tags: lanes
                .iter()
                .map(|&lane| report_on(&mut req, lane, &mut tags))
                .collect(),
        });
        parts.push(FireWs {
            ws: set.id,
            kvtxn,
            log: Some(LogFire { seq: log_seq, rows }),
            lease,
            fold: by.map(|by| FoldAt {
                by,
                decoded: plain_decode,
            }),
            reports: tags,
            staged,
        });
    }
    debug_assert_eq!(
        untagged_device_fold(&req),
        None,
        "a lane whose fold the device counts asks for its report"
    );
    rs::apply_shared(&mut req, stores.shared.as_ref(), Some(&rs_prepared));
    let submit_error = crate::scheduler::submit_prebuilt_tracked_async_with_kv_and_rs_copy_on(
        &scheduler,
        req,
        instance_id,
        pid,
        quorum_pipeline_id,
        completion.clone(),
        copy_src,
        copy_dst,
        rs_copy,
        frame,
        lora_program,
    )
    .err()
    .map(|error| format!("{error:#}"));
    if let Some(error) = submit_error {
        let reason = format!("pipeline: device-geometry submit failed: {error}");
        record_submit_failure(ctx, &fwd, &pipeline_failure, &reason);
        return Ok(Err(reason));
    }
    let unqueued = unfired.submitted();
    epoch.retire_on(&completion);
    for part in &mut parts {
        part.kvtxn.retire_on(&completion);
    }
    mark_device_folds(&stores, &completion, &mut parts);
    unsettle(&stores, &completion, &mut parts);
    if let Some(ask) = rung_ask {
        ask.until_settled(&completion);
    }
    ticket_reservation.commit();
    advance_shadow(ctx, &fwd)?;
    // Rows that start at the folded length and are folded whole end there;
    // any others end where the device put them.
    for (part, (_, lanes)) in parts.iter().zip(&groups) {
        let first = lanes[0];
        let (place, cell) = (&pass_lanes.places[first], &pass_lanes.cursors[first]);
        let rows = split[first + 1] - split[first];
        let end = place.folded.saturating_add(rows);
        let folded_whole = appends_folded(&pass_lanes, lanes)
            && matches!(
                &part.fold,
                Some(FoldAt { by: FoldBy::To(to), .. }) if *to == u64::from(end)
            );
        advance_cursor(cell, folded_whole.then_some(end), None);
        if let Some(staged) = &part.staged {
            cell.lock().unwrap().stage(staged.rows.clone());
        }
    }

    unqueued.queue(
        &pipe_fires,
        PendingFire {
            completion,
            checkpoints: Vec::new(),
            kv: parts,
            device_geometry: true,
            rstxn: rstxns,
            model: ws.model,
            engine: ws.engine,
            fwd_rep,
            cells,
            failure: pipeline_failure,
            retired: None,
            lane: Some(quorum_pipeline_id),
        },
    );
    Ok(Ok(()))
}

pub(super) fn wire_channels_to_pipeline<C: FireContext>(
    ctx: &mut C,
    fwd: &Resource<ForwardPass>,
    pipe_fires: &PendingFires,
) -> Anyhow<Result<(), String>> {
    if let Some(existing) = &ctx.resources().get(fwd)?.fires
        && !Arc::ptr_eq(existing, pipe_fires)
    {
        return Ok(Err(
            "pipeline: a pass cannot submit across different pipelines".into(),
        ));
    }
    let reps = ctx.resources().get(fwd)?.channel_reps.clone();
    for rep in reps {
        let cres: Resource<Channel> = Resource::new_borrow(rep);
        if let Ok(ch) = ctx.resources().get_mut(&cres) {
            match &ch.fires {
                Some(existing) if !Arc::ptr_eq(existing, pipe_fires) => {
                    return Ok(Err("pipeline: a channel is shared across pipelines \
                         (all passes binding a channel must submit on the same \
                         pipeline)"
                        .into()));
                }
                _ => ch.fires = Some(pipe_fires.clone()),
            }
        }
    }
    ctx.resources().get_mut(fwd)?.fires = Some(pipe_fires.clone());
    Ok(Ok(()))
}
