//! Host half of one fire step (`FrameShell::prepare`, plus the `enqueue` and
//! `settle` verbs of the same impl): the admission gate, descriptor-port
//! reads, composition, page geometry, the window table and mask bits — no
//! stream is touched here.

use engine::fire::{Boundary, FoldLen, Masking, RsReset, RsVerb};
use engine::frame::{Demand, Shell as FrameShell, Supply};
use model_exec::fire::{FireDescriptor, Lane as FireLane, compose_axes};

use crate::error::{Fault, Result};
use crate::record;
use crate::run::RsMove;
use crate::store::kv::{self, Seat};
use crate::window::Windows;

use super::{
    Enqueued, FireCost, MROPE_COORDS, Media, PATCH_ROUTE_DROP, Prepared, RsFire, Settled, Shell,
    StepView,
};

/// `prepare`: host-only (gate, ports, compose, lane loop, geometry, windows,
/// mask). `enqueue`: stream-only (prologue, memsets, staging write, tables,
/// schedule, the walk). `settle`: post-sync (readback, capture, epilogue,
/// `held`).
impl FrameShell for Shell {
    type Step<'a> = StepView<'a>;
    type Prepared<'a> = Prepared<'a>;
    type Enqueued<'a> = Enqueued<'a>;
    type Settled = Settled;
    type Error = Fault;

    fn prepare<'a>(
        &mut self,
        step: StepView<'a>,
        prev: Option<&Prepared<'a>>,
    ) -> Result<Prepared<'a>>
    where
        Self: 'a,
    {
        // One step per frame, so there is never a predecessor; `prev` is kept
        // in the signature for a future caller that needs it.
        let _ = prev;
        let StepView {
            lanes,
            attachments,
            media,
        } = step;
        let arming = self.arming;
        let copies = self.copies;

        // 0. The gate: nothing has launched, so a refusal here is free. Only
        // checks the submission's own shape (lane existence, no doubly-
        // attached instance).
        for (index, attached) in attachments.iter().enumerate() {
            if attached.lane as usize >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "attachment {index} names lane {} of the {} this fire has",
                        attached.lane,
                        lanes.len()
                    ),
                ));
            }
            if attachments[..index]
                .iter()
                .any(|earlier| earlier.instance == attached.instance)
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached twice to one fire, at attachment \
                         {index}; a program's stages are one pass with one commit, so \
                         firing it twice would gate against cursors the first pass \
                         already advanced",
                        attached.instance
                    ),
                ));
            }
        }

        // 0b. Descriptor ports, read off the rings the gate just approved.
        // Read here, before the prologue, since a prologue's commit would
        // move the cursors under a later read. A `GeometryClass::Host` lane
        // resolves `None` and reads the submission unchanged.
        if self.owed.is_some()
            && attachments.iter().any(|attached| {
                self.programs
                    .geometry_of(attached.instance)
                    .is_some_and(|class| class != eta_ir::registry::GeometryClass::Host)
            })
        {
            self.reap_guests()?;
        }
        let mut resolved: Vec<crate::program::Envelope> = Vec::new();
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        for attached in attachments {
            let Some(envelope) = self.programs.envelope(attached.instance)? else {
                continue;
            };
            let first = attached.lane as usize;
            let carried = envelope.lanes();
            if first + carried > lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "instance {} is attached at lane {first} and its `embed_indptr` \
                         port describes {carried} lane(s), which runs past the {} this \
                         fire carries",
                        attached.instance,
                        lanes.len()
                    ),
                ));
            }
            let held = resolved.len();
            for lane in 0..carried {
                if envelope_of[first + lane].is_some() {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {} is claimed by two attached instances; a lane's \
                             descriptor ports have one author",
                            first + lane
                        ),
                    ));
                }
                envelope_of[first + lane] = Some((held, lane));
            }
            resolved.push(envelope);
        }

        // 0c. Two device-resolved payloads (page table, mask), built once
        // here and indexed by submission lane. The mask is run-length
        // encoded the same way a host-stated mask is, so both reach the
        // attention arm as the same slab.
        //
        // A device-geometry submission carries no row split (`Lane::tokens`
        // ships empty; the split lives on the `embed_indptr` port), so the
        // row count is read off the port here.
        //
        // A guest holds working-set-relative page indexes, never a pool page
        // id, so a device-resolved page or write reference is translated
        // through `Seated::translation` right here.
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_writes: Vec<Option<(Vec<u32>, Vec<u32>)>> = vec![None; lanes.len()];
        let mut device_masks: Vec<Option<Masking>> = vec![None; lanes.len()];
        let mut lane_rows: Vec<u32> = lanes
            .iter()
            .map(|seated| seated.lane.tokens.len() as u32)
            .collect();
        for source in 0..lanes.len() {
            let Some((held, at)) = envelope_of[source] else {
                continue;
            };
            let ports = resolved[held].lane(at, source)?;
            let table = lanes[source].translation;
            // An out-of-range relative index is refused rather than
            // translated by identity.
            let translate = |page: u32, port: &str| -> Result<u32> {
                table.get(page as usize).copied().ok_or_else(|| {
                    Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source}'s `{port}` port names working-set page {page} and the \
                             table this fire was handed maps {} page(s); a guest holds relative \
                             indexes and the pool's ids are the runtime's, so an index past the \
                             table addresses somebody else's cache",
                            table.len()
                        ),
                    )
                })
            };
            device_pages[source] = ports
                .pages()?
                .map(|relative| {
                    relative
                        .iter()
                        .map(|&page| translate(page, "pages"))
                        .collect::<Result<Vec<u32>>>()
                })
                .transpose()?;
            if ports.owns_pages() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            // `w_slot` is a page reference like `pages`; `w_off` is a plain
            // in-page offset, in no space at all.
            device_writes[source] = ports
                .writes(rows)?
                .map(|(slots, offsets)| {
                    Ok::<(Vec<u32>, Vec<u32>), Fault>((
                        slots
                            .iter()
                            .map(|&page| translate(page, "w_slot"))
                            .collect::<Result<Vec<u32>>>()?,
                        offsets.to_vec(),
                    ))
                })
                .transpose()?;
            if let Some((cells, stride)) = ports.mask(rows)? {
                // A device-resolved mask is refused for more than one query
                // row: `mask::stage` intersects each row against `k <= have +
                // q`, which is only honest in the one-row case for a
                // device-geometry lane's guest-defined write order.
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane whose \
                             write descriptor is the guest's has no such order this shell \
                             can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(crate::mask::from_dense(cells, stride));
            }
        }

        // The patch-axis submission is checked here, before anything
        // launches: past this, `layout.scatter_rows` is an unchecked
        // indexed write.
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes);
        // Position-gather width, from the load, not the submission.
        let embed_taps = self.patch_seat.map_or(0, |seat| seat.embed_taps);
        let embed_weight_taps = self
            .patch_seat
            .map_or(0, |seat| if seat.embed_weights { seat.embed_taps } else { 0 });
        let mut media_of: Vec<Option<&Media<'_>>> = vec![None; lanes.len()];
        for shot in media {
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "images were submitted for lane {at} and this fire carries {}",
                        lanes.len()
                    ),
                ));
            }
            if media_of[at].is_some() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {at} was handed two media rows, and a lane's images are one \
                         concatenation with one patch order"
                    ),
                ));
            }
            let patch_rows: u64 = shot.rows.iter().map(|&rows| u64::from(rows)).sum();
            // Payload bytes, geometry rows and plan width must agree.
            let need = patch_rows * row_bytes;
            if need != shot.patches.len() as u64 || patch_rows != shot.routes.len() as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need,
                    have: shot.patches.len() as u64,
                });
            }
            // Two rotation streams: three numbers per patch row; three per
            // token row, which may be empty and reads as scalar `(p, p, p)`.
            if shot.positions.len() as u64 != patch_rows * MROPE_COORDS as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: patch_rows * MROPE_COORDS as u64,
                    have: shot.positions.len() as u64,
                });
            }
            if !shot.token_positions.is_empty()
                && shot.token_positions.len() as u64
                    != u64::from(lane_rows[at]) * MROPE_COORDS as u64
            {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: u64::from(lane_rows[at]) * MROPE_COORDS as u64,
                    have: shot.token_positions.len() as u64,
                });
            }
            // Position-gather id/weight streams are checked exactly against
            // the plan's own tap counts (0 taps = no table).
            for (what, have, owed) in [
                (
                    "the position table's gather rows",
                    shot.embed_rows.len() as u64,
                    patch_rows * embed_taps,
                ),
                (
                    "the position table's interpolation weights",
                    shot.embed_weights.len() as u64,
                    patch_rows * embed_weight_taps,
                ),
            ] {
                let _ = what;
                if have != owed {
                    return Err(Fault::PatchPayload {
                        lane: shot.lane,
                        need: owed,
                        have,
                    });
                }
            }
            // Routes are checked against this lane's own token rows (rebased
            // later, once composed). `-1` marks "no destination" and is
            // legal only when the plan declares an op that honours it
            // (`self.drops_patch_rows`).
            let rows = lane_rows[at];
            let drop = self.drops_patch_rows;
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows)
            }) {
                return Err(Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        // 1. Lane words in; `compose_axes` seriates the patch axis beside
        // the token one.
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match media_of[at] {
                None => FireLane::new(seated.lane.word, rows),
                Some(shot) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().sum(),
                ),
            })
            .collect();
        let composition = compose_axes(&self.compiled, &self.budgets, &submitted)?;
        let descriptor = FireDescriptor::of(&composition);

        // The composition places each lane's images independently of token
        // order: `patch_offset` is where its rows begin in the fire's patch
        // rectangle, `image_offset` where its images begin in the indptr.
        // Routes are rebased here by `row.row_offset`.
        let (
            patch_payload,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
        ) = if composition.patch_rows() == 0 {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        } else {
            let stride = row_bytes as usize;
            let mut payload = vec![0u8; composition.patch_rows() as usize * stride];
            // Default route is the drop sentinel, not zero: every entry no
            // lane writes has no destination, and zero is a legal token row.
            let mut routes =
                vec![
                    if self.drops_patch_rows { PATCH_ROUTE_DROP } else { 0 };
                    composition.patch_rows() as usize
                ];
            // Rotation stream `(t, h, w)` is each patch's own grid
            // coordinate, copied verbatim (unlike routes, which are rebased).
            let mut positions = vec![0i32; composition.patch_rows() as usize * MROPE_COORDS];
            // How many patch rows this plan folds into one tower output row.
            let fold = (self.patch_fold as usize).max(1);
            let taps = embed_taps as usize;
            let weight_taps = embed_weight_taps as usize;
            let mut embed_rows = vec![0i32; composition.patch_rows() as usize * taps];
            let mut embed_weights = vec![0f32; composition.patch_rows() as usize * weight_taps];
            let mut per_image = vec![0u32; composition.images() as usize];
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                let at = row.patch_offset as usize * stride;
                payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);
                // Routes land at `patch_offset / fold` (the fold's output
                // space), not at `patch_offset` (patch-row space). A
                // negative (sentinel) route is left untouched.
                let landed = (row.patch_offset as usize) / fold;
                let live = shot
                    .rows
                    .iter()
                    .map(|rows| *rows as usize)
                    .sum::<usize>()
                    / fold;
                for (j, &route) in shot.routes.iter().take(live).enumerate() {
                    routes[landed + j] = if route < 0 {
                        route
                    } else {
                        route + row.row_offset as i32
                    };
                }
                let triples = row.patch_offset as usize * MROPE_COORDS;
                positions[triples..triples + shot.positions.len()]
                    .copy_from_slice(shot.positions);
                let at_ids = row.patch_offset as usize * taps;
                embed_rows[at_ids..at_ids + shot.embed_rows.len()]
                    .copy_from_slice(shot.embed_rows);
                let at_w = row.patch_offset as usize * weight_taps;
                embed_weights[at_w..at_w + shot.embed_weights.len()]
                    .copy_from_slice(shot.embed_weights);
                for (i, &rows) in shot.rows.iter().enumerate() {
                    per_image[row.image_offset as usize + i] = rows;
                }
            }
            // Indptr for the tower's attention: `images + 1` entries, image
            // `i` owns `[segments[i], segments[i + 1])`.
            let mut segments = Vec::with_capacity(per_image.len() + 1);
            let mut at = 0i32;
            segments.push(at);
            for rows in per_image {
                at += rows as i32;
                segments.push(at);
            }
            (
                payload,
                segments,
                routes,
                positions,
                embed_rows,
                embed_weights,
            )
        };
        let rows = composition.rows();

        // 2. Fire's own vectors, in fire (composition) order, not submission
        // order.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // One mask entry per lane, seriated with the rest.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // `Some((page, offset))` for a row with its own resolved
        // `w_slot`/`w_off`; `None` where `store::kv::geometry_with` derives
        // the landing place instead.
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // Slots that arrive fresh, decided here, zeroed in `enqueue`.
        let mut fresh: Vec<u32> = Vec::new();
        // Recurrent plan, in fire order — see `RsFire`.
        let mut rs_moves: Vec<RsMove<'a>> = Vec::with_capacity(lanes.len());
        let mut rs_lens: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut rs_order: Vec<u32> = vec![0; lanes.len()];
        // One entry per token row (not per lane); empty when no lane routes.
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            // Resolved ports for this lane, or `None` for a Host-geometry
            // lane / one with no attachment — then every read below is the
            // submission's, unchanged.
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            // `have` comes from whoever owns the page table: a shell-owned
            // slot uses `self.held`, a caller-owned one its own count. A
            // device-geometry lane states its post-append extent on
            // `kv_len`, and `have` is derived as `extent - rows`.
            let have = match ports.as_ref().filter(|ports| ports.owns_pages()) {
                Some(ports) => {
                    let after = ports.extent().ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states its own page table and binds no \
                                 `kv_len` port; the page count, the last page's fill and \
                                 the attention schedules are all carved from the extent, \
                                 and no seat in this shell knows it"
                            ),
                        )
                    })?;
                    if after < row.rows {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states a readable KV extent of {after} on \
                                 its `kv_len` port and this fire writes {} row(s) into \
                                 it; the extent is AFTER the append, so it can never be \
                                 shorter than what the append adds",
                                row.rows
                            ),
                        ));
                    }
                    after - row.rows
                }
                None => match seated.held {
                    Some(held) => held,
                    None => self
                        .held
                        .get(lane.slot as usize)
                        .copied()
                        .ok_or(Fault::Ceiling {
                            what: "slots",
                            need: u64::from(lane.slot) + 1,
                            have: self.held.len() as u64,
                        })?,
                },
            };
            debug_assert_eq!(
                row.row_offset as usize,
                tokens.len(),
                "a lane's rows stand where the composition placed them"
            );
            // A sequence with `have == 0` gets its recurrent banks zeroed.
            // The decision is here; the memset itself happens in `enqueue`,
            // so a fire that later refuses never destroys state it declined
            // to rebuild. The classification (`seated.rs_reset`) is the RS
            // store's own, not derived from `have`.
            let begins = match seated.rs_reset {
                RsReset::Inferred => have == 0,
                RsReset::Fresh => true,
                RsReset::Held => false,
            };
            if begins {
                fresh.push(lane.slot);
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            // Page table from whichever author has one: a device-geometry
            // lane's resolved cell, else the submission's own.
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            // Word and mask cross-checked once (`Fault::MaskWord`). The
            // effective mask is device-resolved OR the submission's.
            let masking = device_masks[source].as_ref().or(seated.mask);
            let runs_masked_arm = self.masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            if masking.is_some() != runs_masked_arm {
                return Err(Fault::MaskWord {
                    lane: row.source,
                    word: lane.word,
                    runs_masked_arm,
                });
            }
            masks.push(crate::mask::LaneMask {
                mask: masking,
                have,
                rows: row.rows,
                bidirectional: seated.bidirectional,
            });
            slot_ids.push(lane.slot as i32);
            // Fold length resolved here: a `FoldLen::Device` row's count
            // comes from the descriptor port read in step 0b, clamped to
            // the verb's bound.
            let fire_lane = rs_moves.len();
            rs_order[row.source as usize] = fire_lane as u32;
            let port = envelope_of[source]
                .and_then(|(held, _)| resolved[held].fold_len.as_deref());
            let (verb, folded) = match &seated.rs {
                RsVerb::Fold => (RsMove::None, row.rows),
                // `fold == 0` is a pure scatter (boundary = row count,
                // invisible to length/split); nonzero lands the durable
                // state on that row while every row is still written.
                RsVerb::Buffer {
                    pages,
                    at,
                    fold,
                    replay,
                } => {
                    // The buffer read path — buffered tokens replayed ahead
                    // of this lane's rows — has no device half on this plane
                    // yet: its recurrences initialize from the folded state
                    // alone. Refused by name rather than run from the wrong
                    // state.
                    if *replay > 0 {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} replays {replay} buffered token(s) ahead of its rows \
                                 (the buffer read path), which this plane does not serve; \
                                 fold the buffer before appending to it",
                                row.source
                            ),
                        ));
                    }
                    let fold = match fold {
                        FoldLen::Host(0) => 0,
                        stated => resolve_fold_len(*stated, row.rows, fire_lane, port)?,
                    };
                    (
                        RsMove::Scatter {
                            pages: pages.as_slice(),
                            at: *at,
                            fold,
                        },
                        if fold == 0 { row.rows } else { fold },
                    )
                }
                RsVerb::Window { .. } => {
                    return Err(Fault::program(
                        "serve::rs",
                        format!(
                            "lane {} asks the device-resident window verb, which this plane does \
                             not serve yet (it replays a buffered prefix ahead of the rows)",
                            row.source
                        ),
                    ));
                }
                RsVerb::FoldBuffered {
                    pages,
                    at,
                    bound,
                    len,
                } => {
                    let (bound, len) = (*bound, *len);
                    if bound != row.rows {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} replays a buffer bounded at {bound} tokens in a fire \
                                 that gave it {} rows — the bound IS what sizes the launch, so \
                                 the two are one number",
                                row.source, row.rows
                            ),
                        ));
                    }
                    (
                        RsMove::Gather {
                            pages: pages.as_slice(),
                            // A mid-page fold leaves survivors offset inside
                            // a shared page, so replay starts from `at`.
                            at: *at,
                        },
                        resolve_fold_len(len, bound, fire_lane, port)?,
                    )
                }
            };
            if verb != RsMove::None && self.buffers.is_none() {
                return Err(Fault::Unbound {
                    what: format!(
                        "lane {}'s recurrent verb, against a plan that declares no chunked \
                         recurrence to buffer",
                        row.source
                    ),
                });
            }
            rs_moves.push(verb);
            rs_lens.push(narrow(u64::from(folded)));
            // Adapter and word cross-checked once, as mask/word above.
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            if seated.adapter.is_some() != runs_correction {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            // Draft/capture axes cross-checked the same way; these carry no
            // payload, so the failure mode is "computed and nobody reads it".
            let runs_draft_arm = self
                .exports
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.classes.contains(row.class as usize));
            if seated.drafts && self.exports.mtp.is_none() {
                return Err(Fault::Draftless { lane: row.source });
            }
            if seated.drafts != runs_draft_arm {
                return Err(Fault::DraftWord {
                    lane: row.source,
                    word: lane.word,
                    runs_draft_arm,
                });
            }
            let runs_capture_arm = self.exports.capturing.contains(row.class as usize);
            if seated.captures_scores && self.exports.scores.is_empty() {
                return Err(Fault::Scoreless { lane: row.source });
            }
            if seated.captures_scores != runs_capture_arm {
                return Err(Fault::ScoreWord {
                    lane: row.source,
                    word: lane.word,
                    runs_capture_arm,
                });
            }
            if any_adapter {
                // `-1` = base model.
                let id = seated.adapter.map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            // Host-class lane's tokens are the submission's; a
            // device-resolved lane's come from the port cell the previous
            // fire's epilogue wrote.
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    ports.check_extent(have.saturating_add(row.rows))?;
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    // Write descriptor already translated to pool pages
                    // (step 0c); `None` means the seat's own `have + row`
                    // arithmetic stands for that row.
                    match &device_writes[source] {
                        Some((slots, offsets)) => writes.extend(
                            slots
                                .iter()
                                .zip(offsets)
                                .map(|(&page, &off)| {
                                    Some((narrow(u64::from(page)), narrow(u64::from(off))))
                                }),
                        ),
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                }
                None => {
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(narrow(u64::from(have) + at as u64));
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
        }

        // Trunk's triple-wide position stream, assembled from the scalar
        // one. Empty unless the plan declares it. Default triple is
        // `(p, p, p)`; a lane that states its own triples overwrites its
        // own interval.
        let mut mrope_positions = if !self.mrope_seat {
            Vec::new()
        } else {
            let mut triples = Vec::with_capacity(positions.len() * MROPE_COORDS);
            for &at in &positions {
                triples.extend_from_slice(&[at, at, at]);
            }
            for row in composition.lanes() {
                let Some(shot) = media_of[row.source as usize] else {
                    continue;
                };
                if shot.token_positions.is_empty() {
                    continue;
                }
                let at = row.row_offset as usize * MROPE_COORDS;
                triples[at..at + shot.token_positions.len()]
                    .copy_from_slice(shot.token_positions);
            }
            triples
        };
        // The denoiser's self-conditioning taps, `[rows, taps]` twice. Every
        // fire of a plan that declares them stages them — zeros for a lane
        // that carries none (an encode lane, an arming synthetic) — so the
        // input is bound whichever class runs.
        let taps = self.self_cond_taps as usize;
        let (mut self_cond_rows, mut self_cond_weights) = if taps == 0 {
            (Vec::new(), Vec::new())
        } else {
            let mut ids: Vec<i32> = Vec::with_capacity(positions.len() * taps);
            let mut ws: Vec<f32> = Vec::with_capacity(positions.len() * taps);
            for row in composition.lanes() {
                let seated = &lanes[row.source as usize];
                let cells = row.rows as usize * taps;
                match seated.self_cond {
                    Some(sc) => {
                        if sc.taps as usize != taps
                            || sc.rows.len() != cells
                            || sc.weight_bits.len() != cells
                        {
                            return Err(Fault::program(
                                "serve::prepare",
                                format!(
                                    "lane {} states self-conditioning taps of width {} over {} \
                                     ids, and this plan reads {taps} taps over the lane's {} rows",
                                    row.source,
                                    sc.taps,
                                    sc.rows.len(),
                                    row.rows
                                ),
                            ));
                        }
                        ids.extend(sc.rows.iter().map(|&id| i32::try_from(id).unwrap_or(0)));
                        ws.extend(sc.weights());
                    }
                    None => {
                        ids.extend(std::iter::repeat_n(0, cells));
                        ws.extend(std::iter::repeat_n(0.0, cells));
                    }
                }
            }
            (ids, ws)
        };

        // 2b. Admission: the union demand of this step, committed atomically
        // before any of it runs. A demand is a watermark (highest addressed
        // page/slot + 1), not a count, since the arenas grow at the tail.
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .map(|(seat, table)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        // Shell-owned block: one past this lane's last page id.
                        self.pools.paging().base(seat.slot).saturating_add(pages)
                    } else {
                        // Runtime-tabled ids: one past the highest addressed.
                        table
                            .iter()
                            .take(pages as usize)
                            .copied()
                            .max()
                            .map_or(0, |page| u64::from(page).saturating_add(1))
                    }
                })
                .max()
                .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX)),
            state_slots: seats
                .iter()
                .map(|seat| seat.slot.saturating_add(1))
                .max()
                .unwrap_or(0),
            workspace: 0,
        };
        Supply::commit(&mut self.pools, demand)?;

        // 3. Page arithmetic, once per kv space.
        let indptr_host = kv::indptr(&seats)?;
        let paging = self.pools.paging();
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;
        // 3b. Explicit write descriptor overrides the derived `have + r`
        // landing, since several lanes appending into one shared pool would
        // otherwise collide at `have + 0`.
        if writes.iter().any(Option::is_some) {
            for geometry in &mut geometries {
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let (Some(write_page), Some(write_offset)) = (
                        geometry.write_page.get_mut(row),
                        geometry.write_offset.get_mut(row),
                    ) else {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "row {row} states an explicit write descriptor and the \
                                 page arithmetic placed {} row(s)",
                                geometry.write_page.len()
                            ),
                        ));
                    };
                    *write_page = page;
                    *write_offset = offset;
                }
            }
        }
        // `pages` (page-id count) read here, before step 4d's lane padding,
        // since the lanes that padding adds own no page.
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        // 4. Windows: every template region resolved against this
        // composition's class table. A region that doesn't seat whole gets
        // `Fallback::Split` unless copies are enabled and the fallback table
        // asks for one at this fire's bucket, in which case it gets one
        // gathered window over the compacted rectangle instead.
        let bucket = self
            .budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        // Copy policy is stored in the segmentation memo (`Shell::segments`)
        // since it's the one input `Windows::admits` needs that
        // `record::BodyKey` doesn't carry. A masked fire always takes the
        // split, so mask/present-set alone determines admissibility from the
        // key; only `self.copies` (toggled per fire by `Shell::set_copies`)
        // sits outside it.
        let copies_here = copies && masks.iter().all(|lane| lane.mask.is_none());
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            // One table per row axis, addressed by the axis.
            model_ir::PerAxis::new([
                composition.table(model_ir::RowAxis::Tokens),
                composition.table(model_ir::RowAxis::Patches),
            ]),
            &indptr_host,
            crate::window::Copies {
                bucket,
                enabled: copies_here,
                spaces: &geometries,
            },
            // Fixed-width slots, so a recorded body's baked `indptr`
            // pointer is right for every fire of its key.
            self.inputs.window_slots(),
        )?;
        // The synthetic pass is not the last fire anybody means.
        if !arming {
            self.last = FireCost {
                launches: windows.launches(),
                copied: windows.copied(),
            };
        }
        let boundaries = windows.packed();

        // 4b. Mask bits, expanded here once, off the same `have`/`rows` the
        // page geometry used. `None` means no lane masked.
        let staged = crate::mask::stage(&masks)?;

        // Body key's class ladder, built from the key's own coordinates
        // (bucket, decode class, lane ceiling), not this fire's actual
        // rows, so two fires of one bucket that split rows differently
        // reach the same body.
        let lane_ceiling = self.lane_ceiling();
        let token_axis = composition.axis(model_ir::RowAxis::Tokens);
        let patches = composition.axis(model_ir::RowAxis::Patches);
        let key = record::BodyKey::of_axes(
            &token_axis.classes,
            token_axis.bucket,
            &self.decoding,
            lane_ceiling,
            self.towered.then_some((&patches.classes, patches.bucket)),
        );
        let ladder = key.classes.clone();
        let patch_ladder = key.patch.as_ref().map(|axis| axis.classes.clone());
        // The four load-level clauses below are hoisted in front of the
        // per-fire ones, since a load that fails any of them would
        // otherwise mint a permanent memo entry no fire will ever read.
        let records_bodies = self.records_bodies();
        let (admits, world): (std::sync::Arc<[crate::window::Admit]>, bool) = if records_bodies {
            self.segmentation(
                &key,
                &windows,
                model_ir::PerAxis::from_fn(|axis| composition.axis(axis).rows),
                copies_here,
            )
        } else {
            (Vec::new().into(), true)
        };
        let bodied = records_bodies
            && !rs_moves.iter().any(|verb| !matches!(verb, RsMove::None))
            && Self::keyable_units(&self.compiled)
            // This fire must be in the world its key was derived in: a
            // resident body holds one script, so an other-world fire walks
            // eagerly instead of replaying a script cut for somebody else.
            && world
            // Cuttability asked last (the only clause that logs to an
            // operator), through a memo since it's a function of the key.
            && !self.cache.body_refused(&key)
            && self.cuttable(&key, admits.as_ref());

        // Arming pins the key a synthetic fire landed on
        // (`Shell::arm_bodies`).
        if arming {
            self.armed_body = bodied.then(|| key.clone());
        }

        // 4c-b. A bodied fire's whole-fire regions are gridded at the
        // bucket, so token ids/positions/mrope/adapter-routes/write
        // descriptors must all reach that far too. Padding is genuinely
        // empty: id 0, position 0, write descriptors and adapter routes
        // `-1`. Nothing off the bodies path moves a byte: `carve_rows` is
        // this fire's own row count there, and every resize is a no-op.
        let carve_rows = if bodied {
            composition.bucket().max(composition.rows())
        } else {
            composition.rows()
        };
        if carve_rows > rows {
            tokens.resize(carve_rows as usize, 0);
            positions.resize(carve_rows as usize, 0);
            if !mrope_positions.is_empty() {
                mrope_positions.resize(carve_rows as usize * MROPE_COORDS, 0);
            }
            if taps > 0 {
                self_cond_rows.resize(carve_rows as usize * taps, 0);
                self_cond_weights.resize(carve_rows as usize * taps, 0.0);
            }
            if any_adapter {
                adapter_routes.resize(carve_rows as usize, -1);
            }
            for geometry in &mut geometries {
                geometry.write_page.resize(carve_rows as usize, -1);
                geometry.write_offset.resize(carve_rows as usize, -1);
            }
        }

        // 4d. Lane tables (pages, tokens, rows) padded to the bucket's lane
        // ceiling too, bodies-path only, so padded lanes read genuinely
        // empty rather than whatever the last fire left there.
        //
        // Ceiling = sum of every present class's rung, each capped at the
        // load's lane ceiling, then clamped to `max_lanes`.
        let mut qo_absolute: Vec<i32> = Vec::new();
        if bodied {
            let ceiling = ladder.lane_reach(lane_ceiling).min(self.budget.max_lanes) as usize;
            for geometry in &mut geometries {
                geometry.pad_to(ceiling);
            }
            // Fire-wide row vector gets the same padding: entries past live
            // lanes repeat the last bound. Copied, not padded in place,
            // since the table's own vector is what every window's rebased
            // slice was cut from.
            qo_absolute = windows.qo_absolute_host().to_vec();
            kv::pad_indptr(&mut qo_absolute, ceiling);
            windows.stage_qo_absolute(ceiling as u32);
        }
        let geometries = geometries;

        // 5. Staging slot, claimed last (after every possible refusal above
        // has had its chance), host only. The slot's pinned bytes back the
        // async H2D `enqueue` issues, so nothing may reuse them until the
        // device has passed that copy.
        let slot = self.inputs.claim()?;
        let staged_lens = self.inputs.write_host(
            &slot,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                // Padded to the bucket (step 4d); empty (no H2D) for an
                // unbodied fire.
                qo_absolute: &qo_absolute,
                // Staged only for a bodied fire; empty keeps the eager path
                // byte-identical to what it always was.
                live: if bodied { windows.live() } else { &[] },
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                // How far this fire's own rows go before the bucket's
                // padding starts.
                live_rows: rows,
            },
        )?;

        // Bound only when it would truncate something — see `RsFire::truncates`.
        let rs_truncates = rs_lens
            .iter()
            .zip(&seats)
            .any(|(len, seat)| *len < narrow(u64::from(seat.rows)));
        // Split only when a boundary is strictly inside a row — see
        // `RsFire::splits`. `fold == rows` or `fold == 0` are both
        // single-call; only an interior boundary costs a second launch.
        let rs_splits = rs_moves.iter().zip(&seats).any(|(verb, seat)| {
            matches!(verb, RsMove::Scatter { fold, .. } if *fold > 0 && *fold < seat.rows)
        });
        Ok(Prepared {
            slot: Some(slot),
            lengths: staged_lens,
            bodied,
            admits,
            ladder,
            lane_ceiling,
            patch_ladder,
            towered: self.towered,
            lanes,
            attachments,
            composition,
            descriptor,
            patch_payload,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_embed_rows,
            patch_embed_weights,
            mrope_positions,
            self_cond_rows,
            self_cond_weights,
            windows,
            seats,
            tables,
            geometries,
            pages,
            fresh,
            demand,
            rs: RsFire {
                // A fire whose every lane folds and carries no prologue
                // attachment keeps the empty vectors and the two false
                // questions.
                write_state: rs_moves
                    .iter()
                    .any(|verb| !matches!(verb, RsMove::Scatter { fold: 0, .. })),
                predicated: {
                    let scatters = rs_moves
                        .iter()
                        .filter(|verb| matches!(verb, RsMove::Scatter { fold: 0, .. }))
                        .count();
                    let prologue = attachments.iter().any(|attached| {
                        attached.at == Boundary::Prologue
                    });
                    (scatters != 0 && scatters != rs_moves.len()) || prologue
                },
                // Bound only where it can truncate something — tidiness, not
                // a correctness rule.
                truncates: rs_truncates,
                splits: rs_splits,
                buffered: rs_moves.iter().any(|verb| !matches!(verb, RsMove::None)),
                moves: rs_moves,
                lens: rs_lens,
                order: rs_order,
            },
        })
    }

    /// Wraps [`Shell::enqueue_on`]: on success the slot moves on to
    /// `settle`, whose callback marks its pinned bytes free again. On
    /// failure there is no callback, so this synchronizes before releasing
    /// the slot — the one sync on this path, off the fast path by
    /// construction.
    fn enqueue<'a>(&mut self, prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let mut p = prepared;
        // Weight promotion between fires: copies ride the notify stream
        // behind an event on the compute stream, so no in-flight fire reads
        // a slab being replaced. Skipped during arming.
        if !self.arming {
            let (compute, notify) = (self.device.stream(), self.device.notify_stream());
            if let Some(tier) = self.weights.experts_mut() {
                tier.promote(compute, notify)?;
            }
        }
        // Slot leaves `Prepared` for this call, so a `?` inside cannot drop
        // it early.
        let slot = p
            .slot
            .take()
            .expect("a `Prepared` holds its staging slot until `enqueue` borrows it");
        match self.enqueue_on(&mut p, &slot) {
            Ok((launches, readback)) => {
                p.slot = Some(slot);
                Ok(Enqueued {
                    prepared: p,
                    launches,
                    readback,
                })
            }
            Err(fault) => {
                // This step's copies read the slot's pinned bytes and may
                // still be in flight with no callback coming, so this is
                // the wait that bounds them.
                let _ = self.device.synchronize();
                drop(slot);
                Err(fault)
            }
        }
    }

    /// The no-completion case of [`Shell::settle_step`].
    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Settled>
    where
        Self: 'a,
    {
        self.settle_step(enqueued, None)
    }
}

/// Resolves one lane's fold length: a host-stated length is itself; a
/// device-stated one reads the descriptor port's cell for this lane. Both
/// are clamped to `bound` and refuse zero.
fn resolve_fold_len(
    len: FoldLen,
    bound: u32,
    lane: usize,
    port: Option<&[u32]>,
) -> Result<u32> {
    let stated = match len {
        FoldLen::Host(n) => n,
        FoldLen::Device(which) => {
            let cells = port.ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {}, and the \
                         program attached to it resolved no such port",
                        which.name()
                    ),
                )
            })?;
            *cells.get(lane).or_else(|| cells.first()).ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on port {} whose \
                         cell carries {} entries",
                        which.name(),
                        cells.len()
                    ),
                )
            })?
        }
    };
    let folded = stated.min(bound);
    if folded == 0 {
        return Err(Fault::program(
            "serve::rs",
            format!(
                "lane {lane}'s fold length resolved to 0 against a bound of {bound}, which is \
                 not a dispatchable commit — a speculative commit must fold at least the \
                 bonus token it is guaranteed to accept"
            ),
        ));
    }
    Ok(folded)
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::{FoldLen, resolve_fold_len};

    /// Port a device-resident fold length is read from; any consuming
    /// geometry port works, since the resolver only uses the cell.
    const PORT: eta_ir::registry::Port =
        eta_ir::registry::Port::RsFoldLen;

    /// A device-resolved count is bounded by what the host knows the buffer
    /// holds; a host-stated count is clamped by the same line.
    #[test]
    fn a_device_fold_length_is_clamped_to_the_bound_it_was_promised() {
        let cells = [3u32, 9, 5];
        let port = Some(&cells[..]);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 0, port).unwrap(), 3);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 1, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(9), 8, 0, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(4), 8, 0, None).unwrap(), 4);
    }

    /// A resolved fold of zero is not a dispatchable commit — refused by
    /// name in both spellings.
    #[test]
    fn a_fold_length_that_resolves_to_zero_is_refused_by_name() {
        let cells = [0u32];
        for len in [FoldLen::Device(PORT), FoldLen::Host(0)] {
            let error = resolve_fold_len(len, 8, 0, Some(&cells[..])).unwrap_err();
            let said = error.to_string();
            assert!(said.contains("bonus token"), "{said}");
        }
        // A bound of zero clamps to zero just as loudly.
        let error = resolve_fold_len(FoldLen::Host(4), 0, 0, None).unwrap_err();
        assert!(error.to_string().contains("bonus token"), "{error}");
    }

}
