//! **THE HOST HALF OF ONE STEP** — the gate, the descriptor ports, the
//! compose, the lane loop, the page geometry, the window table and the mask
//! bits, and not one launch among them.
//!
//! A child module of [`serve`](super), and the first one that is a PHASE
//! rather than a surface. `serve.rs` claims to be the call ORDER top to
//! bottom; a single method of eighteen hundred lines standing in the middle
//! of that order is the one thing a reader cannot follow top to bottom, and
//! it is the phase whose whole contract is that it touches no stream. So it
//! is next door, under its own header, with the three-phase seam
//! ([`FrameShell`]) implemented here and the two verbs that FOLLOW it — the
//! `enqueue` wrapper and `settle` — kept beside it, because the seam is one
//! `impl` and splitting a trait implementation across files would be a reader
//! chasing three methods of one contract through three files.
//!
//! ```text
//! prepare   the gate, the descriptor ports, compose, the lane loop, page
//!           geometry, the window table, the mask bits          <- host only
//! enqueue   a promotion, a slot, and `Shell::enqueue_on`       <- next door
//! settle    the registration, and nothing that waits
//! ```
//!
//! The two free functions below are `prepare`'s alone and moved with it:
//! [`resolve_fold_len`] is the one arithmetic the lane loop delegates, and
//! `narrow` is the cast every position and page index on this path takes.

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

/// **`fire_captured`, cut at the five obligations its own sync-guard names**
/// (alto design §4).
///
/// The cut map, seam by seam:
///
/// ```text
/// prepare   the gate, the descriptor ports, compose, the lane loop, page
///           geometry, the window table, the mask bits          ← host only
/// enqueue   the prologue, the fresh-slot memsets, the staging write, the
///           arena/pool tables, the schedules, the walk          ← stream only
/// settle    the sync, the logits readback, the capture columns, the
///           epilogue, `held`                                    ← the five
/// ```
///
/// The five are the sync's own list, in the order it wrote them: the readback,
/// error attribution, staging lifetime, eviction and teardown, and bookkeeping
/// order. Every one is below the sync and every one is now in `settle`.
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
        // F1 submits one step per frame, so there is never a predecessor. The
        // parameter is here because the wave-order effects that will read it
        // are real and named — channel sequence tickets apply in wave order —
        // and a signature that had to grow a parameter later would make every
        // shell's `prepare` a breaking change at exactly the wave that can
        // least afford one.
        let _ = prev;
        let StepView {
            lanes,
            attachments,
            media,
        } = step;
        let arming = self.arming;
        let copies = self.copies;

        // ── 0. THE GATE. Nothing has launched, so a refusal here is free. ──
        //
        // Every attachment, prologue and epilogue alike, before either runs:
        // an epilogue that discovered its rings were not ready AFTER the
        // forward would leave the lane's tokens in the cache with the guest's
        // pass unrun, which is a fire the caller cannot retry.
        //
        // **AND WHAT IT NO LONGER ASKS IS READINESS** (alto E, article 4).
        // A third clause stood below: `programs.ready` over every attached
        // instance, answering `Fault::Blocked` -> `Error::Exhausted` so the
        // runtime's lane could sleep and re-submit the identical frame. That
        // was F2a's bridge — an approximation of static admission, asked per
        // instance rather than over the frame's union — and
        // `pipeline::fire::validate_frame` is the real thing now: ring
        // occupancy in slot order, host-writer staging, reader pressure,
        // proved against declared capacities before the frame is admitted.
        // Past that door a readiness miss is a CONTRACT VIOLATION, and the
        // device is what discovers it: `channel::pull_validate` compares each
        // prediction against the live pinned words and clears the commit
        // word, and `committed_or` turns the resulting non-commit into a
        // fault naming the instance and the channel. The two clauses left
        // here are about the SUBMISSION's shape — a lane that does not exist,
        // an instance attached twice — which no amount of draining fixes.
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

        // ── 0b. THE DESCRIPTOR PORTS, read off the rings the gate just
        //    approved (`palo B3`, and [`crate::program::ports`] is the whole
        //    argument).
        //
        //    STILL NOTHING HAS LAUNCHED, and in the phase split that is no
        //    longer a promise in a comment: a port read is `read_cell(channel,
        //    head)` — the committed front, which is the cell the guest's own
        //    pass takes this fire — so it is a four-byte copy off an
        //    allocation this shell owns, on the host, with no stream in
        //    reach. It happens HERE, before the prologue, because a prologue
        //    is a pass with a commit and its cursors would move under the
        //    read.
        //
        //    A lane whose instance was bound `GeometryClass::Host` resolves
        //    `None` and the two lines below it never run: its fire reads the
        //    submission, exactly as it always did, byte for byte. That is
        //    what makes the host-carried fixture the parity leverage for the
        //    device-carried one — same program, same channels, one class
        //    apart.
        //
        //    **AND AN ATTACHMENT NAMES A MEMBER, NOT A LANE.** The runtime
        //    attaches one instance per MEMBER and points it at the member's
        //    FIRST lane, because a program's stages are one pass with one
        //    commit however many row groups it fires. A decode-envelope member
        //    is one lane and the two readings coincided; a device-geometry one
        //    need not be — a beam search binds `B` lanes through one program —
        //    and the instance's own `embed_indptr` port is what says how many
        //    and where each one's rows lie. So the map below is per lane and
        //    carries the lane's INDEX WITHIN ITS MEMBER beside the envelope.
        //    **AND A PORT READ IS THE ONE HOST READ OF A GUEST CELL LEFT ON
        //    THIS PATH**, which is why the deferred epilogue batch is
        //    collected in front of it. `read_cell` reads the committed front
        //    of a ring, and on a device-carried instance that cell is written
        //    by the previous fire's `channel::scatter_publish` — a kernel
        //    which, since the boundary stopped waiting, may still be on the
        //    stream. Paid only where a port exists: a `GeometryClass::Host`
        //    instance resolves `None` without touching a ring, and the c64
        //    decode path is entirely Host.
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

        // ── 0c. THE TWO DEVICE-RESOLVED PAYLOADS, LIFTED OUT OF THE RINGS AND
        //    OWNED HERE. A page table and a masking are the only two things a
        //    port resolves that the fire path holds by REFERENCE — `tables`
        //    borrows the submission's page list, `mask::LaneMask` borrows the
        //    submission's `Masking` — and a device-resolved one is in neither
        //    submission nor rings by the time `stage` and `geometry_with` want
        //    it. So they are built once, here, indexed by SUBMISSION lane, and
        //    the composition loop below borrows out of these vectors exactly
        //    as it borrows out of the submission.
        //
        //    **THE MASK IS RUN-LENGTH ENCODED AND NOT SEPARATELY PACKED**
        //    (`crate::mask::from_dense`): the whole claim a device mask has to
        //    answer is that it reaches the attention arm as the same slab a
        //    host-stated mask of the same bools reaches it as, and sharing the
        //    expansion is how that stops being a thing to test and starts
        //    being a thing that is true.
        //
        //    **AND THE ROW COUNT COMES WITH THEM, FOR THIS CLASS ONLY.** A
        //    decode-envelope lane's submission carries placeholder ids and
        //    therefore carries its own row count, which is why nothing about
        //    that class changes. A device-GEOMETRY submission carries no row
        //    split at all — the runtime ships `Lane::tokens` empty for every
        //    lane, because the split is the instance's own `embed_indptr`
        //    port and the runtime has no more claim on it than it has on the
        //    page table beside it. dev says the same thing by building the
        //    CSR inside the compose kernel (`compose_fixed_decode` writes
        //    `qo_indptr[i + 1] = row_base + i + 1`). So the count is read off
        //    the port HERE, before `compose`, because `compose` is what turns
        //    counts into windows and row offsets and there is no later
        //    instant at which a row can appear.
        //
        //    **AND THIS IS WHERE THE TWO PAGE SPACES MEET.** A guest holds
        //    WORKING-SET-RELATIVE indexes and never a pool page id — that is
        //    `kv-working-set`'s whole surface, and it is what makes an O(1)
        //    copy-on-write fork possible, because a relative index survives
        //    the copy that moves the physical page under it. Everything below
        //    this line is in the POOL's space: `store::kv::geometry_with`
        //    pushes a table entry straight into the page CSR and the append
        //    writes through `w_slot` with no lookup. For every host-resolved
        //    geometry the runtime crosses between them before it submits
        //    (`pipeline::fire::map_lane_pages`); for THIS class it cannot,
        //    because the values are in a cell no host read, so it ships the
        //    table (`Seated::translation`) and the crossing happens here —
        //    once, on the two ports that carry page references.
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
            // A RELATIVE INDEX THE TABLE DOES NOT COVER IS A REFUSAL, and so
            // is a lane with page references and no table at all: "translate
            // by identity" is the bug this crossing exists to end, and an
            // empty table would spell it silently.
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
            // The write descriptor crosses with them: `w_slot` is a page
            // reference like `pages` is — a beam search builds it as
            // `gather(pool_ids, wpos / page_size)` out of the same
            // `ws.reserve` grant — while `w_off` is an offset inside a page
            // and is in no space at all.
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
                // **ONE ROW A LANE, AND THE REFUSAL IS THE CAUSAL BOUND.**
                // `mask::stage` intersects every restriction with `k <= have +
                // q`, which is the order the cache is written in — and a
                // device-geometry lane's write order is the guest's
                // (`w_slot`/`w_off`), so for `q > 0` this shell has no bound
                // it can honestly derive. On a ONE-row lane the term is
                // vacuous (`have + 0` is the whole extent, because `have` is
                // `kv_len - 1`), which is exactly what dev's
                // `pack_dense_mask` does — it transcribes the guest's cells
                // and applies no causality of its own. Every device-geometry
                // shape this tree admits is one row a lane
                // (`lease::detect_pooled_device_geometry` requires a rank-1
                // `[lanes]` token channel), so the wider case is refused by
                // name rather than served with a bound nobody stated.
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

        // ── **THE SECOND ROW AXIS'S SUBMISSION, JUDGED BEFORE IT IS
        //    COUNTED** (multimodal M-1e, refusal (i)). Nothing has launched,
        //    so every disagreement between a lane's declared geometry and the
        //    payload beside it is free to refuse here — and this is the only
        //    instant at which the ROUTE vector is checkable at all, because
        //    `layout.scatter_rows` is a copy with an index and no arithmetic:
        //    an entry past the rectangle is an out-of-bounds device write the
        //    kernel cannot see and the arena does not fault on.
        //
        //    Keyed by lane like [`Attached`] is, so a text-only submission
        //    passes an empty slice, allocates this `Vec` of `None` and does
        //    nothing else at all.
        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes);
        // The position gather's width, from the LOAD and not from the
        // submission (multimodal §9.2): `0` when the text states no learned
        // position table, and `0` weights when it states a native grid.
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
            // The payload's bytes, the geometry's rows and the plan's width:
            // three numbers that have to agree.
            let need = patch_rows * row_bytes;
            if need != shot.patches.len() as u64 || patch_rows != shot.routes.len() as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need,
                    have: shot.patches.len() as u64,
                });
            }
            // **THE TWO ROTATION STREAMS, AGAINST THE TWO ROW COUNTS**
            // (multimodal §6.3). The patch one is three numbers per PATCH row
            // and is owed whole; the token one is three per TOKEN row and may
            // be empty, which reads as `(p, p, p)` — the scalar rope a lane
            // gets when it says nothing.
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
            // **AND THE POSITION GATHER'S TWO STREAMS** (multimodal §9.2),
            // against the tap count THE PLAN declares. `0` taps is a text with
            // no learned position table and owes an empty slice; a native-grid
            // text declares 1 tap of ids and no weights, so the weight stream
            // is owed empty there too. Both are exact rather than
            // empty-or-exact, because unlike `token_positions` there is no
            // value the shell could synthesize: it does not know the grid.
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
            // The routes, against THIS LANE's token rows — the bound the
            // rebase below preserves, because a lane's rows are one interval
            // of the fire's.
            //
            // **AND THE DROP SENTINEL, ADMITTED BY THE PLAN AND NOT BY THIS
            // LOOP** (multimodal §8.6). A compacting fold — `layout.pool_rows`,
            // `layout.merge_rows` — answers `rows / side²` rows and leaves the
            // rest of the patch rectangle as the arena left it, and
            // `PatchRoutes` has an entry per row of the FULL rectangle, so the
            // tail needs a value meaning "nowhere". `-1` is it, the spelling
            // `AdapterRoutes` already uses for "no bank".
            //
            // It is legal exactly when the plan declares an op that HONOURS
            // it. `layout.scatter_rows` reads a negative route as a device
            // write below the base of the token rectangle, so admitting the
            // sentinel for every plan would turn this refusal into that write;
            // `self.drops_patch_rows` is read off the trace at load, and a
            // text that folds declares `layout.scatter_live_rows` and gets the
            // leniency with it. `-1` ALONE and not every negative: a `-2` is
            // still a submission that meant something this shell does not
            // serve.
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

        // 1. Lane words in. `compose` is arithmetic over a `Vec` of them:
        //    words to classes, classes to an order, counts to prefix sums.
        //    A lane that submitted images states them here, and `compose_axes`
        //    seriates the second axis beside the first.
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

        // ── **THE SECOND SERIATION, CASHED INTO THREE VECTORS.** The
        //    composition placed every lane's images: `patch_offset` is where
        //    its rows begin in the fire's patch rectangle and `image_offset`
        //    where its images begin in the indptr — and neither is derivable
        //    from the token order, which is the whole of multimodal §5.1. So
        //    the assembly PLACES rather than appends, and the patch order may
        //    differ from the order this loop walks in without any of the
        //    three vectors noticing.
        //
        //    The routes are rebased here and nowhere else: a submission says
        //    "my seventh token row" because it was written before the fire it
        //    lands in existed, and `row.row_offset` is the fire's answer to
        //    that. The bound checked lane-relatively above survives the shift
        //    because a lane's rows are one interval of the fire's.
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
            // **THE SENTINEL IS THE DEFAULT, NOT ZERO** (multimodal §8.6, §17).
            // Every entry no lane writes is a row with no destination: the
            // fold's dead tail, and the rung padding past the last real image.
            // Zero is a legal token row, so leaving them zero scatters the
            // arena's leftovers over row 0 of the fire. A plan that declares no
            // dropping scatter has no such rows — every route it states names a
            // row — and keeps the zero it always had.
            let mut routes =
                vec![
                    if self.drops_patch_rows { PATCH_ROUTE_DROP } else { 0 };
                    composition.patch_rows() as usize
                ];
            // **THE TOWER'S ROTATION STREAM, PLACED THE WAY THE PAYLOAD IS.**
            // A patch's `(t, h, w)` is its own image's grid coordinate, so it
            // is the submission's number verbatim — no rebasing, unlike the
            // routes, which name a TOKEN row and therefore have to follow the
            // seriation.
            let mut positions = vec![0i32; composition.patch_rows() as usize * MROPE_COORDS];
            // **THE POSITION GATHER, PLACED THE WAY THE PAYLOAD IS** — the
            // taps and their weights are a property of the image's grid, so
            // they ride through verbatim like the rotation stream and unlike
            // the routes. Zero-length when the plan declares no table, and
            // then the loop below copies nothing into them.
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
                // **THE ROUTES GO IN THE FOLD'S OUTPUT SPACE, NOT IN PATCH
                // ROWS** (multimodal §17). `layout.merge_rows` and
                // `layout.pool_rows` COMPACT: `side²` patch rows become one row
                // at the FRONT of the rectangle, so a lane whose patch rows
                // start at `patch_offset` has its tower output at
                // `patch_offset / fold` — and `layout.scatter_live_rows` pairs
                // `src[j]` with `routes[j]` over THOSE rows.
                //
                // Writing them at `patch_offset` instead is right for exactly
                // one lane and wrong for every lane after it, because lane 0's
                // offset is zero and `0 / fold` is `0`. With two images the
                // second lane's routes landed at 64 where the scatter read 16,
                // so its soft tokens were dropped and its placeholder rows took
                // the garbage past the fold's live prefix instead.
                //
                // **AND A SENTINEL IS NOT AN ADDRESS, SO IT IS NOT REBASED.** A
                // route names a token row relative to its lane and `row_offset`
                // is the fire's answer to that; a NEGATIVE route names no row,
                // and adding an offset to it produces one — at `row_offset =
                // 20` every dead tail row became token row 19, which is the
                // PREVIOUS lane's last row.
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
            // The indptr the tower's attention reads: `images + 1` entries,
            // image `i` owning `[segments[i], segments[i + 1])`.
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

        // 2. The fire's own vectors, in fire order — which is the seriated
        //    order the composition chose, not the order the runtime submitted.
        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // THE MASKED AXIS, IN FIRE ORDER. One entry per lane, seriated with
        // the rest — the span table is indexed by the schedule's request
        // number, which is a position in the class order and not the order
        // the runtime submitted.
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        // THE EXPLICIT WRITE DESCRIPTOR, ONE ENTRY PER TOKEN ROW IN FIRE
        // ORDER: `Some((page, offset))` for a row whose lane resolved
        // `w_slot`/`w_off` off its rings, `None` for every row whose landing
        // place `store::kv::geometry_with` derives. All `None` is every fire
        // this shell fired before the device-geometry class.
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        // THE SLOTS THAT ARRIVE FRESH, DECIDED HERE AND ZEROED IN `enqueue`.
        let mut fresh: Vec<u32> = Vec::new();
        // THE RECURRENT PLAN, IN FIRE ORDER — see [`RsFire`]. Empty vectors
        // for a fire whose every lane folds, which is every fire this shell
        // fired before F3.
        let mut rs_moves: Vec<RsMove<'a>> = Vec::with_capacity(lanes.len());
        let mut rs_lens: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut rs_order: Vec<u32> = vec![0; lanes.len()];
        // THE ADAPTER AXIS, IN FIRE ROW ORDER. One entry per token ROW —
        // `linear.lora_correct` reads `routes[row]` beside `x[row]`, so this
        // is the shape `tokens` and `positions` have and not the shape
        // `slot_ids` has. Stays empty for a fire no lane routed, and an empty
        // vector is what makes the whole axis cost that fire nothing:
        // `Inputs::write` stages no bytes, `FireBindings` binds no seat, and
        // the correction's window has no rows for the walk to dispatch.
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            // THIS LANE'S RESOLVED PORTS, CUT TO ITS OWN ROWS — `None` for a
            // lane whose instance was bound `GeometryClass::Host` and for one
            // with no attachment at all, and then every line below reads the
            // submission exactly as it always did, byte for byte.
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            // WHO KNOWS HOW LONG THE SEQUENCE IS depends on who owns its
            // pages. A shell-owned slot is one the shell opened and has been
            // counting ever since; a caller-owned one is a page table the
            // caller forked, trimmed or restored between fires, and its own
            // count is the only one that is right.
            //
            // **AND A DEVICE-GEOMETRY LANE'S IS ITS OWN `kv_len` PORT, MINUS
            // THIS FIRE'S ROWS.** `have` is not a fact this shell can hold for
            // such a lane: `self.held` counts the slots whose page table is
            // the shell's, and the runtime's `KvDelta::held` is zero because
            // the runtime could not know it either — the extent is device
            // data, computed by the epilogue that decided where the rows land.
            // What the fire actually needs `have` for is `after = have + rows`
            // (the page count, the last page's fill, the stated kv length),
            // so the honest reading is to take the extent the guest states and
            // derive `have` back from it. That is dev's own arithmetic
            // (`compose_fixed_decode`: `last_page_len = ((kv_len - 1) %
            // page_size) + 1`) reached from the other end, and
            // `store::kv::geometry_with` then computes exactly the same three
            // numbers it computes for every other lane.
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
            // A FRESH SEQUENCE ARRIVES WITH A ZEROED RECURRENT BANK, and
            // `have == 0` is the only place the contract says a sequence
            // begins.
            //
            // [`Shell::open`] says the same thing for a caller whose page
            // table is the SHELL's: it clears the slot's recurrent banks
            // because a linear-attention scan reads its whole state on its
            // first step, so a slot still holding the last sequence's
            // history would continue it. A runtime that keeps its OWN page
            // table never calls `open` — the contract has no such verb, by
            // design — and until this line nothing else cleared the banks
            // either. The kv half was fine and stayed fine: `kv_len` says
            // nothing lives past the append, so a recycled page is
            // overwritten before it is read. The recurrent half has no
            // `kv_len`.
            //
            // The launch pattern that exposed it (`palo` build log 18, and
            // `tests/gpu/tests/cuda_launch_isolation`): THREE identical
            // greedy completions through ONE booted worker. The first was
            // right — the pools were `Buffer::zeroed` at load — and the
            // second and third answered echo-shaped garbage built out of the
            // prompt's own words, because their GDN layers were still
            // running the previous launch's sequence. Every other gate in
            // this tree launches once per boot, which is why it survived.
            //
            // Cost is one `cudaMemset` over one slot's banks on the FIRST
            // fire of a sequence and never again — a chunked prefill's
            // second chunk arrives with `have > 0` — and nothing at all for
            // a plan that declares no `CacheRow::State`. **The DECISION is
            // here and the memset is in `enqueue`**, which is the phase
            // split doing its one job: this loop refuses fires, and a fire
            // that refuses after a slot was zeroed would have destroyed
            // state it then declined to rebuild.
            //
            // **AND WHOSE FACT IT IS, SINCE F3.** `have == 0` is the KV
            // store's answer to a question the RS store owns (survey §9's gap
            // list): a runtime that forks a sequence, restores a prefix or
            // recycles a seat can hand a slot whose recurrence must be zeroed
            // while its KV count is not zero, and one whose KV was trimmed to
            // nothing while its recurrence must continue. So the LANE carries
            // the classification now, and `RsReset::Inferred` — the default,
            // and every caller that has not been taught to state it — is
            // exactly the old rule, restated where it can be seen.
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
            // THE PAGE TABLE, FROM WHICHEVER AUTHOR HAS ONE. A
            // device-geometry lane's is the cell its `pages`/`page_indptr`
            // ports resolved to and the submission's is empty; every other
            // lane's is the submission's, unchanged, and an empty table is
            // still the shell's own block-per-slot paging.
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            // THE WORD AND THE MASK, CHECKED AGAINST EACH OTHER, ONCE.
            // `compose` already refused a word this artifact has no class
            // for; what it cannot know is whether the class it resolved to
            // reads a mask. Both directions are a wrong answer that looks
            // like a right one, so both are refused (`Fault::MaskWord`
            // argues each).
            //
            // **AND THE MASK IT ASKS ABOUT IS THE EFFECTIVE ONE.** A
            // device-resolved mask reaches this shell on a channel and NOT on
            // `Seated::mask`, while the lane's word says `masked` all the same
            // — the runtime stamps it from the same lowering that decided the
            // mask was device-carried. Asking `seated.mask` alone would refuse
            // every such fire by name for the one reason that is not true of
            // it: that nobody stated a mask.
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
            });
            slot_ids.push(lane.slot as i32);
            // ── THE RECURRENT VERB, RESOLVED TO ADDRESSING (design §6).
            //
            //    The fold length is resolved HERE, in compose, and not one
            //    line later: a `FoldLen::Device` row's count comes out of the
            //    descriptor port this fire already read (step 0b), is clamped
            //    to the host's bound and refuses zero — and past this point
            //    nothing can tell the two spellings apart, which is dev
            //    clearing the flag at the same instant so that no downstream
            //    reader can branch on it.
            let fire_lane = rs_moves.len();
            rs_order[row.source as usize] = fire_lane as u32;
            let port = envelope_of[source]
                .and_then(|(held, _)| resolved[held].fold_len.as_deref());
            let (verb, folded) = match &seated.rs {
                RsVerb::Fold => (RsMove::None, row.rows),
                // **THE MIXED ROW, LOWERED** (wave F3b). A zero fold is
                // the pure scatter: it truncates nothing, so its boundary
                // entry is its own row count — "at the end", which is what
                // makes it invisible to both the length seat and the split.
                // Anything else lands the durable state on that row while
                // every row is still written into the buffer, and the
                // boundary entry IS the fold. Resolved here for the same
                // reason a replay's length is: past this point nothing may
                // tell the two spellings apart.
                RsVerb::Buffer { pages, at, fold } => {
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
                            // The buffer's head: a mid-page fold leaves the
                            // survivors offset inside the page they share
                            // with the tokens it absorbed, and a replay from
                            // buffer token zero would fold those a second
                            // time (wave F3b).
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
            // THE ADAPTER AND THE WORD, CHECKED AGAINST EACH OTHER, ONCE —
            // the mask's rule above, restated for the axis beside it, and it
            // is the same two wrong answers that look right. A lane that
            // named an adapter and landed in a class outside the correction's
            // window gets the BASE MODEL and nobody is told; a lane whose
            // word put it inside the window and named none would have its
            // rows read a routes vector nothing wrote. Both are refused
            // before anything launches.
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
            // THE TWO EXPORT AXES, CHECKED THE SAME WAY, AND THE ARGUMENT
            // CHANGES IN ONE PLACE (palo C3b/C4b). The mask and the adapter
            // are PAYLOADS, so their second wrong answer is "staged and never
            // read". These carry no payload — a draft head reads the lane's
            // own hidden, a capture arm the lane's own query — so the second
            // wrong answer is "computed and nobody told": a lane whose word
            // put it inside the export's window and that asked for nothing
            // has a column written for it that no reader collects, and a lane
            // that asked and landed outside gets no column and is handed an
            // empty readout with no way to tell that from a fire that
            // captured zeros. Both are refused before anything launches.
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
                // `-1` is the base model, and it is what an unrouted lane
                // contributes to a fire some OTHER lane routed: the projection
                // half writes its waist row zero and the combine returns before
                // it reads the bank, so those rows are bit-identical to the
                // fire they would have had alone. Reachable only when the
                // artifact's correction window covers a class that carries no
                // adapter, which the check above forbids — so today every entry
                // this branch writes is a real id, and the sentinel is the
                // kernel's own floor rather than a path.
                let id = seated.adapter.map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            // WHERE THE TOKEN COMES FROM IS THE WHOLE OF `palo B3`. A
            // host-class lane's ids are in the submission, because the runtime
            // folded them and stated them. A device-resolved lane's are the
            // cell the previous fire's epilogue wrote, which the runtime could
            // not know and did not state — its `Lane::tokens` carries the row
            // COUNT and placeholders, and `tokens_for` refuses a port that
            // disagrees with the count the composition already carved for.
            // THE ROW COUNT THE COMPOSITION PLACED, which for a
            // device-geometry lane is the port's and for every other is the
            // submission's — one number either way, decided at step 0c.
            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    // The extent is a CHECK where the seat owns it and the
                    // SOURCE `have` was derived from where the guest does; the
                    // check is therefore an identity in the second case and is
                    // made anyway, because an identity that stopped holding is
                    // the first thing anybody would want to hear about.
                    ports.check_extent(have.saturating_add(row.rows))?;
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    // THE WRITE DESCRIPTOR, KEPT IN FIRE ROW ORDER FOR THE
                    // PATCH BELOW — already translated into pool pages at step
                    // 0c, which is the one place a page reference crosses
                    // spaces. It cannot be applied here: the vectors it
                    // overwrites are `kv::geometry_with`'s, and that call
                    // wants the whole seat list. `None` for a lane that binds
                    // no `w_slot`/`w_off`, and then the seat's own
                    // `have + row` arithmetic stands for its rows.
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

        // ── **THE TRUNK'S TRIPLE-WIDE POSITION STREAM, ASSEMBLED FROM THE
        //    SCALAR ONE** (multimodal §6.3). Empty unless the plan declares
        //    it, which is what makes this cost every text served before the
        //    towers exactly nothing — not a branch inside a loop, a vector
        //    that is never built.
        //
        //    THE DEFAULT IS `(p, p, p)` AND THAT IS NOT A PLACEHOLDER: a
        //    triple whose three entries agree is scalar rope to the last bit
        //    the two expressions can share, which is why a text lane in an
        //    image-carrying fire needs no submission of its own and why a
        //    text-only fire of an mrope SKU answers what it always did. A
        //    lane that DOES state triples (`get_rope_index`'s output, where
        //    image-placeholder rows take their patch's grid coordinate)
        //    overwrites its own interval, and a lane's rows are one interval
        //    of the fire's — the same fact the route rebase leans on.
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

        // ── 2b. ADMISSION (article 4). The union demand of this step,
        //    committed atomically before any of it runs.
        //
        //    **A DEMAND IS A WATERMARK, NOT A COUNT** (wave C; dev's
        //    `required_kv_pages`/`required_state_slots`,
        //    context.cpp:2087-2127). The elastic arenas grow at the tail, so
        //    what admission has to commit is the HIGHEST addressed page and
        //    slot plus one — not how many of them this step happens to touch.
        //    The two readings agree for the shell's own block-per-slot paging
        //    and diverge the moment a lane brings the runtime's page ids,
        //    where page 900 may be the only page in the fire: a count would
        //    have committed one page and let the append write into address
        //    space with nothing behind it.
        //
        //    Both axes therefore run over EVERY lane, the runtime-tabled ones
        //    included. A page id is a page id whoever minted it (article 8 —
        //    the ids are the runtime's, the bytes under them are the
        //    engine's), and the fault a slot past the pool earns is the same
        //    `Fault::Ceiling` `kv::geometry_with` raises a dozen lines below.
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let demand = Demand {
            kv_pages: seats
                .iter()
                .zip(&tables)
                .map(|(seat, table)| {
                    let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                    let pages = after.div_ceil(page_size).max(1);
                    if table.is_empty() {
                        // The shell's own block: `base(slot) + pages` is one
                        // past this lane's last page id.
                        self.pools.paging().base(seat.slot).saturating_add(pages)
                    } else {
                        // The runtime's ids: one past the highest this lane
                        // will address. `geometry_with` reads exactly
                        // `table[..pages]` and refuses a shorter table, so
                        // the same prefix is what is scanned here.
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

        // 3. Page arithmetic, once per kv space. Every space is paged the
        //    same way in v1 — one page size, one block per slot — so the
        //    vectors coincide; the loop is per space because the geometry
        //    seat is, and a plan with two page sizes changes this call and
        //    nothing above it.
        let indptr_host = kv::indptr(&seats);
        let paging = self.pools.paging();
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.spaces)
            .map(|_| kv::geometry_with(&paging, &seats, &table_refs))
            .collect::<Result<Vec<_>>>()?;
        // ── 3b. THE EXPLICIT WRITE DESCRIPTOR, OVER THE DERIVED ONE.
        //
        //    `geometry_with` lands row `r` of a lane at flat position
        //    `have + r` of that lane's page run, which is right for every
        //    sequence that appends to its own tail and WRONG the moment
        //    several lanes append into one shared pool: a beam search's `B`
        //    lanes all state the same extent, so `have + 0` names one cell for
        //    all of them and `B - 1` beams would overwrite the first. The
        //    guest computes `w_slot`/`w_off` in its own epilogue for exactly
        //    that reason, and this is where its answer replaces the derived
        //    one — after the page CSR and the last-page fill, which are still
        //    the extent's and are still carved the same way, and before
        //    anything reads them.
        //
        //    The rows are parallel: `writes` was filled in the composition's
        //    own lane order, one entry per token row, which is the order
        //    `geometry_with` fills `write_page`/`write_offset` in.
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
        // Still `mut`, and the last write to it is step 4d's lane padding.
        // `pages` is read HERE, before it, which is the honest place: it is
        // the page-id count, and the lanes that padding adds own no page.
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);

        // 4. THE WINDOWS. Every region of the template, resolved against the
        //    class table this composition built: which rows and which lanes it
        //    runs over, deduplicated, each carrying the qo boundaries a ragged
        //    view inside it is cut by — rebased, because a sub-rectangle
        //    starts at its own zero. This is the whole of what makes a mixed
        //    fire legal, and `crate::window` is where it is argued.
        //    A region P4 could not seat gets `Fallback::Split` here — one
        //    window per interval — unless this shell serves copies and P4's
        //    table asks for one at this fire's bucket, in which case it gets
        //    ONE window over the compacted rectangle instead
        //    (`crate::window::Gathered`). The bucket is a POSITION in the
        //    lattice because `FallbackRow::buckets` is a range of positions,
        //    and `Composition::bucket` is the row count that position holds;
        //    a deployment that declared no lattice has one bucket, at 0.
        let bucket = self
            .budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        // **THE COPY POLICY, AS ONE WORD, BECAUSE TWO READERS WANT IT.** The
        // window table is built with it, and the segmentation memo STORES it
        // (`Shell::segments`): it is the one input to `Windows::admits` that
        // the `record::BodyKey` does not carry, so a memo that assumed it
        // would be a memo that can serve the wrong table. A masked fire takes
        // the split — `Copies::enabled`'s own doc says which vector a gather
        // would still have to compact and why it is the page-id list's
        // problem again.
        //
        // **AND THE MASK HALF IS A KEY FUNCTION EVEN SO** (the capacity wave):
        // `Fault::MaskWord` two hundred lines up refuses any lane whose mask
        // and whose class disagree, in both directions, so `all(is_none)` here
        // is exactly "this fire's present set misses `Shell::masked`" — and
        // the present set is the key's second coordinate. Padding lanes carry
        // no mask and no rows, and a `None` cannot flip an `all` either way.
        // What is left outside the key is `self.copies` alone, which
        // `Shell::set_copies` moves between fires; `Windows::admits` carries
        // the derivation and the gate below carries the clause.
        let copies_here = copies && masks.iter().all(|lane| lane.mask.is_none());
        let mut windows = Windows::of(
            &self.trace,
            &self.compiled,
            // **ONE TABLE PER ROW AXIS, ADDRESSED BY THE AXIS.** `Windows::of`
            // resolves each region against its own capture unit's seriation,
            // and it indexes this rather than choosing between two arguments
            // whose names said which was which.
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
            // **THE BLOB'S CARVE, HANDED TO THE TABLE THAT LAYS ITSELF OUT IN
            //  IT.** `Inputs::reserve` divided the window bytes into
            //  fixed-width slots and `Windows::packed` places each window at
            //  its slot's offset, so a recorded body's baked `indptr` pointer
            //  is right for every fire of its key (`crate::window`'s header).
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

        // 4b. THE MASK BITS. A lane states its mask as runs over its own
        //    readable extent and `attention.masked` reads one bit per
        //    (query row, key position) pair with the causal bound already
        //    folded in, so the expansion happens here, once, off the same
        //    `have` and `rows` the page geometry was carved from
        //    (`crate::mask` argues every term of it). `None` is a fire no
        //    lane masked, and then no seat is bound at all.
        let staged = crate::mask::stage(&masks)?;

        // **THE BODY KEY'S SECOND HALF, BUILT BEFORE THE GATE THAT USES IT**
        // (the ceiling design's Option B): one CEILING per present class, in
        // the order the rows stand. Three readers — the key below, step 4d's
        // padding reach, and the `Run` `enqueue_on` builds — and one
        // computation, so there is no second reading to fall out of step with
        // the one the cache is keyed on.
        //
        // **AND WHAT GOES IN IS THE KEY'S OWN COORDINATES AND TWO LOAD
        // CONSTANTS**, not this fire's rows: the bucket, which class is a
        // decode class, and the lane ceiling. The class table is asked one
        // question only — which classes have rows — so two fires of one
        // bucket that split their rows differently build the SAME ladder and
        // reach the same body. That is the key collapse, and this line is
        // where it happens on the host side.
        let lane_ceiling = self.lane_ceiling();
        let token_axis = composition.axis(model_ir::RowAxis::Tokens);
        let ladder = record::Ladder::of(
            &token_axis.classes,
            token_axis.bucket,
            &self.decoding,
            lane_ceiling,
        );
        // **AND THE SECOND UNIT'S HALF BESIDE IT** (the multi-unit bodies
        // wave). `Some` exactly when the ARTIFACT states a patch axis — a
        // load constant, not this fire's patch rows — so that a text lane and
        // an image lane of one vision SKU key into one family and a fire of a
        // text-only SKU keys into the family it always did. Its rungs are all
        // the patch bucket, because the patch axis has no decode notion to
        // split on (`record::AxisKey` carries the derivation), and an
        // axis-empty fire's bucket is zero — the rung that launches no tower
        // exec at all.
        let patches = composition.axis(model_ir::RowAxis::Patches);
        let patch_ladder = self
            .towered
            .then(|| record::Ladder::flat(&patches.classes, patches.bucket));

        // ── 4c. **IS THIS FIRE A BODY'S?** (the bodies design's chunk B) —
        //    asked HERE, at the last host instant before the slot is written,
        //    because the answer decides whether the live-rows seat's words go
        //    into it. The router in `enqueue_on` reads what this writes; it
        //    cannot ask again, because by then the staging is behind it.
        //
        //    The outer clauses are the router's own, restated: a fire that
        //    records nothing, one that moves buffered bytes and one whose
        //    weights rotate are all eager for reasons `enqueue_on` argues in
        //    full, and an eager fire has no body.
        //
        //    **AND THERE IS NO ARMING CLAUSE HERE ANY MORE, WHICH IS WHAT
        //    THE FOLD'S DELETION SIMPLIFIED.** A clause used to stand in this
        //    conjunction reading "not a synthetic, unless it is the BODIES
        //    path's synthetic" — because two kinds of arming pass arrived and
        //    the fold's template had no business seating anything. There is
        //    one kind now (`Shell::arm_bodies`), and seating a body is the
        //    entire thing it was fired to do, so it takes the gate exactly as
        //    a caller's fire does. It has to: this gate is what STAGES the
        //    live-rows seat, and a body captured without the seat staged is a
        //    body captured against a geometry no replay can move. Its numbers
        //    are still nobody's — the readback, the epilogue and the `held`
        //    advance are suppressed elsewhere on `Shell::arming`.
        //
        //    **AND THREE CLAUSES THAT ARE THIS PATH'S OWN.**
        //
        //    * **EVERY CAPTURE UNIT OF THIS ARTIFACT MUST BE NAMED BY THE
        //      KEY** ([`Shell::keyable_units`]) — which is what is LEFT of the
        //      clause that used to stand here, and the clause is retired
        //      rather than amended because its premise died.
        //
        //      It read: a multi-unit artifact is never served from a body,
        //      `CompiledModel::fold_refused`'s sentence transferred whole,
        //      because a `record::BodyKey` carries ONE bucket and a fire that
        //      launches two execs has one bucket PER UNIT. The premise was the
        //      key's shape, and the key's shape changed: it carries a
        //      `record::AxisKey` for the second unit — that unit's own lattice
        //      point and its own ladder over the patch seriation — so a
        //      two-unit composition is named exactly as a one-unit one is, per
        //      unit, which is multimodal §1's "6 + 6, not 6 x 6" rather than
        //      the product it refuses. `Cut` already carried its unit, the
        //      replay loop was already per-unit, and the key was the only
        //      thing in the way.
        //
        //      **AND `CompiledModel::fold_refused` IS UNTOUCHED AND STILL
        //      TRUE.** It is a fact about the FOLD — a plane that arms one
        //      graph per bucket per key and therefore cannot hold two buckets
        //      — and the fold is retired from this shell. What died is the
        //      bodies path INHERITING it, which was a reading of somebody
        //      else's refusal and not a statement about this one.
        //
        //      What survives is a bound the key can actually state: a
        //      `record::BodyKey` names the token unit and at most one other,
        //      so an artifact with more capture units than that has no key
        //      here. `model_ir::RowAxis` has two variants and
        //      `CompiledModel::units` holds the distinct ones, so no artifact
        //      the compiler can bake reaches it today — the clause is a belt
        //      under the day a third row space is minted, and it is written
        //      because a key silently naming two of three units is the shape
        //      of bug this campaign existed to remove.
        //    * **AND SOMETHING MUST BE LEFT FOR A GRAPH TO HOLD**
        //      (`record::cuts`, the tier-2 campaign) — which is what is LEFT
        //      of the clause that used to stand here. That clause read "every
        //      present region must be one a body can be replayed over", asked
        //      of the whole window table by `Windows::covers_fire_shifted`,
        //      and it refused a whole composition over one gathered or grouped
        //      or unshifted region. The rule itself has not moved an inch —
        //      `Windows::admits` asks the same clauses — but it is asked PER
        //      REGION now, and the refused ones become ISLANDS: the body holds
        //      every stretch around them and the fire path re-issues them
        //      eagerly between the execs. So the shape of a window no longer
        //      decides whether this key has a body; it decides how much of the
        //      composition the body holds.
        //
        //      Nor does the CUT any more. A segment boundary may not fall
        //      inside a fork group or between two arms of a conditional, and a
        //      plan builder may not land on the far side of one from the
        //      launches that read its schedule — and each of those three is a
        //      rule for GROWING the island to the nearest legal boundary
        //      (`record::widen`), not a reason to refuse. An island region is
        //      served by the eager walk, which is always correct; a refusal
        //      threw away every capturable region of a twenty-eight-layer text
        //      over one withdrawn window.
        //
        //      What can still refuse is the composition the growing consumed
        //      ENTIRELY: every region an island, no exec to capture, a body
        //      that would be a script of eager stretches. `record::cuts`
        //      answers `record::Uncut::Eager` for it, once per key, and the
        //      fire WALKS.
        //
        //      Chunk 2b-ii's flip is unchanged and is now carried per region:
        //      the same `Run` this gate builds is handed the same table with
        //      `.ceilings(..)`, so the region the host says a graph holds is the
        //      region the walk carves, seats and shifts — and the region it
        //      says is an island is the region the walk leaves exactly as the
        //      eager path leaves it (`run::Held::Eager`).
        //    * **AND THE PAD MUST BE ARMED**, which is this wave's clause and
        //      the one that is about the KEY rather than about the fire.
        //      Everything a body promises is stated at a lattice point: the
        //      grids (`Run::carve_rows`, `Run::carve_lanes`), the schedules
        //      (`Run::planning`), the arena column, the staged row vectors.
        //      With `Shell::pad` off there is no lattice point — the armed
        //      `Pad::bucket` is this fire's own row count — so every one of
        //      those ceilings would be a live span wearing a key's name, and
        //      two SPLITS of one bucket would carve differently while sharing
        //      one `record::BodyKey`. The old code met that by asking the
        //      ceilings themselves for slack (`pad.bucket > pad.rows`), which
        //      quietly disarmed them for the fire that lands EXACTLY on its
        //      bucket — a real fire, and the one `Shell::arm_bodies`
        //      synthesizes by construction. So the clause moves here, once,
        //      where it is a statement about the deployment: `[engine] pad =
        //      off` is a diagnostic arm, a shell serving it has no business
        //      recording bodies, and past this line `bodied` IMPLIES an armed
        //      pad and every ceiling below is unconditional.
        // ── 4c-a. **WHICH REGIONS A BODY OF THIS COMPOSITION WOULD HOLD**
        //    (the tier-2 campaign) — `Windows::admits`, one entry per template
        //    region, computed HERE because this is the instant that has the
        //    window table and because three later readers must all take the
        //    same answer: the gate below, the `Run` the router builds
        //    (`run::Held::Eager`, which is what stands every ceiling down inside
        //    an island), and `record::Fire::admits`, which is what the capture
        //    loop is cut with and what the ledger is kept over.
        //
        //    Computed unconditionally, before the gate, because it is what the
        //    gate ASKS: the shape of a composition's windows no longer refuses
        //    the key, it decides how much of the key's composition a graph
        //    holds. The vector is one byte per region and a fire that turns
        //    out not to be a body's simply never reads it.
        //
        //    **AND IT IS DERIVED ONCE PER KEY AND NOT ONCE PER FIRE**
        //    ([`Shell::segments`]). Both this table and `record::cuts`' verdict
        //    on it are functions of the `record::BodyKey` — that is the whole
        //    argument `Windows::admits` carries, clause by clause — so a
        //    steady decode stream was allocating two vectors per fire to
        //    re-derive a constant. The memo holds the table behind an `Arc`,
        //    which is what `Prepared` carries, and the fire path allocates
        //    neither.
        //
        // `composition.bucket()` and not the lattice POSITION named `bucket`
        // above: the key's number is the one `record::Fire` carries, which is
        // the row count the launches were recorded at. The LADDER beside it is
        // that same number asked per class (the ceiling design's Option B),
        // built once above because step 4d and the `Run` both read it — and
        // there is no third field, because the copy policy cannot separate two
        // bodies that both exist (`record::BodyKey`'s header).
        //
        // Composed unconditionally now, where the gate used to compose it
        // inside its own last conjunct: the memo is keyed on it, and the
        // arming channel below wants it too, so one clone here replaces the
        // one or two that stood below.
        let key = record::BodyKey {
            bucket: composition.bucket(),
            classes: ladder.clone(),
            // **AND THE SECOND UNIT'S PAIR, WHICH IS WHAT RETIRED THE
            // MULTI-UNIT REFUSAL** (the multi-unit bodies wave). `None` on a
            // text-only artifact makes this key byte-for-byte the key it was
            // — the same `Eq`, the same `Hash`, the same `Display` — which is
            // G4's oath and is asserted rather than assumed
            // (`record`'s key tests).
            patch: patch_ladder.clone().map(|classes| record::AxisKey {
                bucket: composition.patch_bucket(),
                classes,
            }),
        };
        // **AND THE ADMISSIBILITY TABLE IS JUDGED PER AXIS** (the multi-unit
        // bodies wave). Every region is measured against the total of ITS OWN
        // rectangle — a tower region against the fire's patch rows, a trunk
        // region against its token rows — because `Windows::admit`'s
        // whole-fire clause is a comparison and the two counts are two row
        // spaces. A text-only load hands a patch total of zero and reads
        // exactly what it read before this argument existed.
        //
        // **AND THE LOAD'S OWN CLAUSES ARE ASKED IN FRONT OF IT.** The four
        // below read nothing about this fire — a deployment either serves
        // bodies, pads, records graphs and holds its weights still, or it
        // never records anything at all — so a load that answers `false` to
        // any of them was minting a permanent memo entry per distinct
        // (rows x present set) for a table no fire would ever read. They are
        // hoisted here, and the segmentation is derived only where somebody
        // is going to spend it; the remaining clauses stay below because each
        // of them IS about this fire (its buffered verbs, its world) or about
        // this key (`body_refused`, `cuttable`), and those are exactly the
        // questions the memo exists to answer once.
        let records_bodies =
            self.bodies && self.pad && self.graphs.records() && !self.weights.rotating();
        // The empty table is the inert reading `Prepared::admits` documents:
        // a fire the gate turns away reads no admission at all, because
        // `run::Ceilings::admit` asks `bodied` first and answers `None` before it
        // indexes. `true` beside it keeps `world` meaning what it says —
        // "this fire is in its key's world" — rather than borrowing the
        // conjunction's answer.
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
            // **AND THIS FIRE MUST BE IN THE WORLD ITS KEY WAS DERIVED IN**
            // (the capacity wave, `Shell::segmentation`). The copy answer is
            // the one input to `Windows::admits` the `record::BodyKey` does
            // not carry, and once the mask half is subtracted what is left of
            // it is `self.copies` — which `Shell::set_copies` flips between
            // fires. A key armed under one policy and fired under the other
            // wants the template cut in different places; a resident body
            // holds ONE script, so the fire that wants the other one is turned
            // away here and walks eagerly
            // (`record::BodyTally::eager_copy_world`) rather than replaying a
            // script that was cut for somebody else.
            //
            // A CLAUSE AND NOT A REFUSAL: nothing is refused by name and no
            // key is closed, because the same key's other-world fires are
            // still served — this is a property of the FIRE, like the rotating
            // and buffered clauses above it, and it is counted in the same
            // family for the same reason.
            && world
            // **AND THE TEMPLATE MUST BE CUTTABLE AROUND ITS ISLANDS**
            // (`Shell::cuttable`, which takes the named decline and prints it
            // once). Asked LAST, past every outer clause, because it is the
            // only one that says anything to an operator — and asked through
            // a memo, because the answer is a function of the key.
            && !self.cache.body_refused(&key)
            && self.cuttable(&key, admits.as_ref());

        // **AND THE ARMING PASS TAKES THE KEY IT JUST COMPOSED AWAY WITH IT**
        // (`Shell::armed_body`, the tier-1 key-collapse wave). This is the one
        // instant in the engine that has both the fire's window table and the
        // key's ladder in hand, and `Shell::arm_bodies` — which knows only
        // which classes it asked for — has to be able to NAME the key its
        // synthetic landed on in order to pin it. Written only under the
        // arming word, so a real fire pays one `bool` test and no clone; and
        // written as `None` for a synthetic the gate refused, so the loop
        // cannot pin a key nothing seated.
        if arming {
            self.armed_body = bodied.then(|| key.clone());
        }

        // ── 4c-b. **AND THE ROW VECTORS, STAGED OUT TO THE BUCKET'S ROW
        //    CEILING** (the grid-at-ceiling wave) — step 4d's argument on the
        //    other axis, and it arrives for the same reason one chunk later.
        //
        //    A bodied fire's whole-fire regions are gridded at the BUCKET
        //    (`Run::carve_rows`), so a launch there runs blocks for rows this
        //    fire does not have. Those blocks are retired — every seated entry
        //    opens on `r >= win[0]` — but three of the fire's row vectors are
        //    read by entries that DECLARE their rectangle rather than only
        //    addressing it, and a declaration that stops at the live rows is a
        //    refusal rather than a stale read: `layout.embed` asserts
        //    `ids.rows == y.rows`, `elemwise.rope` asserts the same of its
        //    position stream, and `elemwise.rope_mrope` REFUSES by name on it.
        //    `y` is an arena rectangle and the arena is carved at the bucket
        //    for this fire (`Shell::enqueue_on`), so the ids and the positions
        //    have to reach as far.
        //
        //    **AND THE PADDING IS GENUINELY EMPTY, WHICH IS STEP 4d'S OWN
        //    DISCIPLINE.** Token id zero, position zero, and — for a plan that
        //    rotates by a triple — three zeros: a padded row gathers the
        //    vocabulary's first embedding and rotates it by nothing, into a
        //    plane row `row_valid` marks invalid and every guard retires. The
        //    alternative is leaving the last fire's ids there, which is the
        //    thing this shell refuses to do anywhere else on the padded axis.
        //
        //    `row_valid` is NOT padded with ones — it is the one vector whose
        //    tail has to say the opposite of its head, and `inputs::Fire::live_rows`
        //    is how the staging is told where the fire's own rows stop.
        //
        //    **AND THE WRITE DESCRIPTORS COME WITH THEM, WHICH IS A STAGING
        //    FACT BEFORE IT IS A GUARD.** `Inputs::commit` copies the
        //    per-space `write_page` and `write_offset` at the ROW count it was
        //    handed, so a padded fire whose descriptors stopped at its own
        //    rows would have the copy read pinned bytes nobody wrote this
        //    frame. `-1` is what goes in the tail: `attn/kv.cuh`'s explicit
        //    writer tests `offset_in_page < 0` before it dereferences the page
        //    id, so a padded row retires there as well as at `win[0]` and at
        //    `row_valid` — three belts, and this one is the one that makes the
        //    H2D honest.
        //
        //    **AND THE ADAPTER ROUTE VECTOR IS THE FOURTH, AND IT IS A
        //    DECLARATION AND NOT A GUARD.** `linear.lora_correct` opens on
        //    `routes.rows == x.rows` — `x` is the arena rectangle, which this
        //    fire carved at the bucket — so a routes vector that stopped at
        //    the live rows is a REFUSED launch (a `debug_assert` in the
        //    correction's own door) and not a stale read. `-1` is what goes in
        //    the tail, and it is the same sentinel the branch above writes for
        //    an unrouted lane: the projection half computes a zero waist row
        //    for it and the combine returns before it reads the bank, so a
        //    padded row is the base model's nothing whatever else retires it.
        //    Written only where the axis is on at all — an empty vector is the
        //    off switch this axis is built around, and padding it would turn
        //    that switch on for a fire no lane routed.
        //
        //    Nothing off the bodies path moves a byte: `carve_rows` is this
        //    fire's own row count there, and every resize is a no-op. And the
        //    pad is not asked for again — `bodied` implies it since the gate
        //    above took that clause.
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
            if any_adapter {
                adapter_routes.resize(carve_rows as usize, -1);
            }
            for geometry in &mut geometries {
                geometry.write_page.resize(carve_rows as usize, -1);
                geometry.write_offset.resize(carve_rows as usize, -1);
            }
        }

        // ── 4d. **THE LANE TABLES, STAGED OUT TO THE BUCKET'S LANE CEILING**
        //    (the plan-at-bucket-ceiling design, chunk 2), and only on the
        //    bodies path.
        //
        //    A body is captured at one composition and replayed at another,
        //    and the chunk after this one raises what the SCHEDULES are
        //    carved at from this fire's lanes to the ceiling the bucket
        //    spells. The moment a plan is carved at a lane count larger than
        //    the fire brought, every reader that walks a padded lane reads
        //    whatever the LAST fire left in that slot of the carve — a page
        //    run that still points at somebody's pages, a length that still
        //    says tokens. The guards would mostly hold (a decode's
        //    `block_valid_mask` retires the over-launched work item,
        //    `protective_get_kv_offset` clamps a page past the bound, the
        //    live-rows seat says how many rows are the fire's own), and
        //    "mostly" is the wrong footing for a cache. So the padded lanes
        //    are made GENUINELY EMPTY here — no pages, no tokens, no rows —
        //    and every one of those guards goes back to being belt-and-braces
        //    over a reading that is already right.
        //
        //    **THE CEILING IS THE LADDER'S LANE REACH**, which is the sum of
        //    every present class's rung with each one CAPPED AT THE LOAD'S
        //    LANE CEILING — one past the last lane any window of this key may
        //    be carved to (the ceiling design's Option B, tightened by the
        //    tier-1 key-collapse wave). A rung is a row count read as a lane
        //    count for the reason the fire's bucket was — a lane is at least
        //    one row (`fire::Fault::EmptyLane`), so a class of `rung` rows can
        //    carry no more than `rung` lanes — and a lane also needs a SEAT,
        //    which is the second bound and the tighter one wherever a prefill
        //    rung (the whole bucket) runs past the seats. `record::Ladder::lane_reach`
        //    carries the argument and the deployment inequality it buys.
        //
        //    **AND IT IS THE SUM AND NOT THE FIRE'S BUCKET, BECAUSE THE
        //    CARVES ARE LAID END TO END.** Chunk 2 padded to
        //    `Composition::bucket` because there was one carve and it began at
        //    lane zero. Option B gives every class its own carve, at an origin
        //    that is the prefix sum of the rungs in front of it, so what the
        //    staging has to define is the LAST carve's end — and the sum is
        //    that number. It dominates the fire's own lanes (each class's
        //    capped rung holds its own lanes, and the sum is taken where the
        //    lanes stand); for a single-class fire it is that class's cap.
        //
        //    Clamped to `max_lanes` because THAT is what the staging was
        //    carved at (`Inputs::reserve`: `lanes + 1` bounds, `lanes`
        //    per-lane entries) — a reach above the lane ceiling is a row count
        //    no lane count can reach, and padding past the carve would smear
        //    into the region behind it. The clamp is why `Run::planning` reads
        //    the ceiling back OFF these vectors instead of recomputing it: a
        //    carve is only honest as far as the staging defined. `pad_to`
        //    never shrinks, so the clamp is safe to spell as a `min` and a
        //    reach at or below this fire's lanes moves nothing at all.
        //
        //    **AND NOTHING OFF THE BODIES PATH MOVES A BYTE.** Everything
        //    that reads these vectors ahead of this line has already read
        //    them — `pages` is the page-id count and empty lanes own no page,
        //    `Windows::of` took the geometries for its gather above and a
        //    gathered window is not a body — so the only readers of what this
        //    grows are the staging below and the host planning twins in
        //    `enqueue_on`, whose window slices are cut at live lanes either
        //    way.
        let mut qo_absolute: Vec<i32> = Vec::new();
        if bodied {
            let ceiling = ladder.lane_reach(lane_ceiling).min(self.budget.max_lanes) as usize;
            for geometry in &mut geometries {
                geometry.pad_to(ceiling);
            }
            // The fire-wide row vector gets the same treatment for the same
            // reason (chunk 2c-a's vector, this chunk's tail): entries past
            // the live lanes repeat the last bound, so `qo_absolute[lane]` is
            // DEFINED and spells a zero-row lane at every lane a ceiling plan
            // can name. Copied rather than padded in place because the
            // table's own vector is what every window's rebased slice was cut
            // from and what `Run::qo_indptr_absolute_host` slices per window;
            // the copy is what the H2D takes.
            qo_absolute = windows.qo_absolute_host().to_vec();
            kv::pad_indptr(&mut qo_absolute, ceiling);
            // **AND THE TABLE IS TOLD HOW FAR THE COPY REACHES** (the
            // plan-at-bucket-ceiling design, chunk 3). The bytes are the
            // staging's, but the SHAPE of the device reading is the window
            // table's to state (`Windows::qo_absolute`), and a decode
            // schedule carved at this ceiling hands its launch a `q_indptr`
            // that has to say it reaches lane `ceiling` — which is also the
            // number `Run::planning` reads back to learn what the staged
            // vectors cover.
            windows.stage_qo_absolute(ceiling as u32);
        }
        let geometries = geometries;

        // ── 5. THE STAGING SLOT, CLAIMED, AND THE FIRE'S VECTORS WRITTEN INTO
        //    IT — host only, no stream in reach (alto design §4:
        //    `staging.write(slot, ..)`).
        //
        //    **THIS IS WHAT F1 REFUSED AND F2b BUILT.** `Inputs::write` used
        //    to be both halves in one call against ONE device-side buffer, so
        //    a second frame in flight would have let the host write W+1's
        //    descriptor over the bytes W's launches were still reading. The
        //    claim is the fix and it is a lifetime: this slot's PINNED host
        //    bytes are the source of the async H2D `enqueue` issues, and
        //    nothing may reuse them until the GPU has passed that copy — which
        //    is the instant the settlement callback runs and drops the guard.
        //
        //    Claimed LAST, after every refusal above has had its chance, so a
        //    step that cannot compose never holds a slot at all; and released
        //    by `Prepared`'s destructor if the frame is abandoned anyway.
        let slot = self.inputs.claim()?;
        let staged_lens = self.inputs.write_host(
            &slot,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                // **THE SAME BOUNDARIES, THE SECOND READING, AND ON THE SAME
                // SWITCH** (bodies design, chunk 2c-a). One fire-wide
                // `[lanes + 1]` vector with nothing subtracted — the one the
                // table above rebased every window's copy out of, which is why
                // it is asked for rather than rebuilt — for the consumer whose
                // pointer is the PLANE's base rather than the window's
                // (`Run::plane_base`, `Run::qo_indptr_absolute`). Only a
                // bodied fire can have such a consumer, so only a bodied fire
                // pays the H2D; empty is the off switch, exactly as below.
                //
                // Since chunk 2 it is the table's vector PADDED OUT TO THE
                // BUCKET's lane ceiling (step 4d), which is why it is a local
                // `Vec` rather than the table's own slice: the entries past
                // the live lanes are zero-row lanes, so a ceiling plan finds
                // a defined bound wherever it looks. Empty stays empty, and
                // an unbodied fire hands `&[]` exactly as before.
                qo_absolute: &qo_absolute,
                // **THE LIVE-ROWS SEAT, WRITTEN ONLY FOR A BODY** (bodies
                // design, chunks A and B). `windows.live()` holds the identity
                // words — four per launch, its own full row count and row
                // offset and its own lane count and lane offset — and staging
                // them changes no arithmetic on ANY path: a
                // guard that reads the seat admits exactly the rows its launch
                // was already going to run. What it does change is the H2D
                // this fire pays, so the words go over only when something
                // means to read them, which is the bodies path and nothing
                // else. Empty is the off switch, end to end: no host bytes, no
                // copy, no seat bound, and `Ctx::stage` stays the null pointer
                // — which is what makes the EAGER path byte for byte the path
                // it was.
                //
                // The words themselves are the identity either way. A body
                // does not need them to be anything else: it is captured at
                // one composition and replayed at another ROW COUNT — and,
                // since the chunked-arm wave, at another LANE OFFSET — of the
                // same one, and the identity written by THIS fire is exactly
                // this fire's geometry.
                live: if bodied { windows.live() } else { &[] },
                slot_ids: &slot_ids,
                spaces: &geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                // **HOW FAR THE PADDING MASK HAS TO REACH** (the
                // grid-at-ceiling wave). A bodied fire's whole-fire regions
                // are gridded at the BUCKET — `Run::carve_rows`, the same
                // number `Ctx::opaque_rows` has padded their GEMMs to since D4
                // — so the rows between this fire's own and the bucket are
                // launched and then retired, and the SEAT-LESS pool writers
                // retire them off this mask alone. Zero everywhere else, which
                // is the fire's own rows and the tail nobody launches.
                // **AND WHERE THIS FIRE'S OWN ROWS STOP.** `tokens` above
                // reaches the bucket for a bodied fire (step 4c-b) so that the
                // entries which DECLARE a rectangle can declare the one their
                // launch is gridded over; this is what keeps the padding mask
                // from claiming those rows are real. Equal to the vector's own
                // length on every other path, which writes the all-valid mask
                // this staging has always written.
                live_rows: rows,
            },
        )?;

        // Bound only when it would truncate something — see `RsFire::truncates`.
        let rs_truncates = rs_lens
            .iter()
            .zip(&seats)
            .any(|(len, seat)| *len < narrow(u64::from(seat.rows)));
        // **AND SPLIT ONLY WHEN A BOUNDARY IS STRICTLY INSIDE A ROW** — see
        // `RsFire::splits`. `fold == rows` is the fire that buffers a window
        // and folds all of it, which is the single-call folding path; `fold
        // == 0` is the pure scatter, which is the single-call buffered one.
        // Only the interior boundary costs a second launch.
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
            windows,
            seats,
            tables,
            geometries,
            pages,
            fresh,
            demand,
            rs: RsFire {
                // **NOTHING AT ALL FOR THE PLAIN PATH.** A fire whose every
                // lane folds and whose lanes carry no prologue attachment
                // keeps the empty vectors and the two false questions, and
                // `enqueue` then binds the null seats every launch here has
                // always been handed.
                // `fold: 0` and not `Scatter { .. }`, because a mixed row
                // is a scatter that FOLDS (wave F3b): it moves buffered bytes
                // like a draft and lands the boundary like a commit, so it
                // answers this question the way a fold does.
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
                // **BOUND ONLY WHEN IT WOULD TRUNCATE SOMETHING**, which
                // since F3b is tidiness and no longer a correctness rule.
                // `attn/ssm.cuh`'s fla scan used to read `commit_len !=
                // nullptr` as a second thing besides the truncation —
                // `single_round`, a different bf16 rounding of the decay — so
                // a seat bound where it could change no length still changed
                // the numbers, and a replay that accepted its whole window
                // stopped being the fold it replaced. The rounding is its own
                // argument now (`RecurrentPool::fused_decay`) and the two
                // spellings agree to the bit; what is left is the same
                // "bind nothing that can do nothing" the mask above obeys.
                truncates: rs_truncates,
                splits: rs_splits,
                buffered: rs_moves.iter().any(|verb| !matches!(verb, RsMove::None)),
                moves: rs_moves,
                lens: rs_lens,
                order: rs_order,
            },
        })
    }

    /// **The whole step onto the stream, and the slot's lifetime made safe on
    /// the way out** (alto design §4; articles 1 and 7).
    ///
    /// The body is [`Shell::enqueue_on`]; what this wrapper adds is the one
    /// thing the phase split has to get right and the type system cannot:
    /// `enqueue` issues asynchronous copies OUT OF the claimed slot's pinned
    /// bytes, so from the instant `Inputs::commit` returns those bytes belong
    /// to the device until it has passed them. On the success path the slot
    /// travels on to `settle`, whose callback is exactly that instant. On a
    /// FAILURE path there is no callback — so the slot would go straight back
    /// to the ring under `Prepared`'s destructor and the next `prepare` would
    /// overwrite bytes a copy was still reading.
    ///
    /// So a failed enqueue synchronizes before it lets go. It is the one sync
    /// left on this path, it is off the fast path by construction (a step that
    /// enqueued cleanly never reaches it), and it is what makes the abort path
    /// safe rather than merely rare.
    fn enqueue<'a>(&mut self, prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let mut p = prepared;
        // ── **THE PROMOTION INSTANT** (alto design §7, wave D2; article 3
        //    applied to weights). Between two fires, and on THIS side of the
        //    phase boundary rather than in `prepare`, because `Prepared` is
        //    the type that cannot reach a stream and a promotion is three
        //    enqueues. It stands before the first launch of this step and
        //    after every launch of the last, which is exactly the window a
        //    slab may be overwritten in.
        //
        //    Nothing here waits. The copies ride the notify stream behind an
        //    event recorded on the compute stream (so no airborne fire is
        //    still reading the slot being replaced), and the compute stream
        //    waits on their completion before the launches below (so no fire
        //    reads a table entry naming bytes in flight). A round whose
        //    predecessor has not finished simply does not happen — residency
        //    is a promotion, and a promotion that would have to wait is not
        //    one. A load that streams nothing has no tier and this is a
        //    `None` check.
        //
        //    The ARMING pass is held out: it computes nobody's numbers, and
        //    letting a synthetic fire move experts would make the working set
        //    a function of what the load armed.
        if !self.arming {
            let (compute, notify) = (self.device.stream(), self.device.notify_stream());
            if let Some(tier) = self.weights.experts_mut() {
                tier.promote(compute, notify)?;
            }
        }
        // The slot leaves the `Prepared` for the length of this call, so that
        // a `?` inside the body cannot release it behind our back.
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
                // The copies this step issued read the slot's pinned bytes and
                // may still be in flight. Nothing will call back, so this is
                // the wait that bounds them.
                let _ = self.device.synchronize();
                drop(slot);
                Err(fault)
            }
        }
    }

    /// **The registration, and nothing that waits** (alto design §4; article
    /// 2, survey §7 invariant I7).
    ///
    /// The no-completion case of [`Shell::settle_step`], which is where the
    /// five obligations the old sync guarded are enumerated and rehomed.
    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Settled>
    where
        Self: 'a,
    {
        self.settle_step(enqueued, None)
    }
}

/// **Resolve one lane's fold length** (dev `batch_compose.hpp:726-768`).
///
/// Three rules, and the third is the one that matters:
///
/// 1. a host-stated length is itself,
/// 2. a device-stated one is the descriptor port's cell for this lane,
/// 3. **both are clamped to the verb's `bound` and both refuse zero** — and
///    past this function nothing can tell which spelling arrived, which is
///    dev clearing `PIE_RS_FLAG_FOLD_LEN_DEVICE` at the same instant so that
///    the replay CSR, the classifier and the kernels' `commit_len` never see
///    a placeholder.
///
/// The clamp is what makes the scheme safe: the device may name a count the
/// host never saw, but it can never name one the buffer cannot supply.
/// Refusing zero is what makes it dispatchable: a speculative commit folds at
/// least the bonus token it is guaranteed to accept, and a zero-length fold
/// is a launch that would compute nothing while claiming to have committed.
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


    /// The port a device-resident fold length would be read from. Any
    /// consuming geometry port serves: what the resolver takes is the CELL,
    /// and the port name only ever reaches a refusal's sentence.
    const PORT: eta_ir::registry::Port =
        eta_ir::registry::Port::RsFoldLen;

    /// **THE CLAMP IS WHAT MAKES A DEVICE-RESIDENT FOLD LENGTH SAFE** (alto
    /// design §6; dev `batch_compose.hpp:726-768`).
    ///
    /// The accepted count of a speculative pass is computed by the verifier on
    /// the stream, so the host cannot know it — but the host DOES know the
    /// upper bound, because it is the host that decided how many drafts the
    /// buffer holds. Clamping the resolved value to that bound is the whole
    /// safety argument: the device may name a count the host never saw, and it
    /// can never name one the buffer cannot supply.
    ///
    /// Three readings, and the third is the one a wrong implementation would
    /// get wrong:
    ///
    /// 1. a length inside the bound is itself,
    /// 2. a length past it is the bound — not a refusal, because a verifier
    ///    that accepted everything is a legal outcome and the bound is the
    ///    whole window,
    /// 3. **a host-stated length is clamped by the same line**, so the two
    ///    spellings cannot disagree about what "past the bound" means. dev
    ///    clears `FOLD_LEN_DEVICE` at exactly this point for the same reason:
    ///    past resolution, nothing downstream may branch on which spelling
    ///    arrived.
    #[test]
    fn a_device_fold_length_is_clamped_to_the_bound_it_was_promised() {
        let cells = [3u32, 9, 5];
        let port = Some(&cells[..]);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 0, port).unwrap(), 3);
        assert_eq!(resolve_fold_len(FoldLen::Device(PORT), 8, 1, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(9), 8, 0, port).unwrap(), 8);
        assert_eq!(resolve_fold_len(FoldLen::Host(4), 8, 0, None).unwrap(), 4);
    }

    /// **A FOLD OF ZERO IS NOT A DISPATCHABLE COMMIT** (dev
    /// `batch_compose.hpp:759-763`, verbatim in intent).
    ///
    /// A speculative verify accepts at least the bonus token it is guaranteed
    /// to accept, so a resolved zero is not "nothing was accepted" — it is a
    /// port that carried a placeholder, a verifier that never ran, or a
    /// program that resolved the wrong channel. Serving it would launch a
    /// replay that folds nothing while the host advances its accepted
    /// boundary as if it had, which is the one failure the whole scheme
    /// exists to make impossible. Refused by name, in both spellings.
    #[test]
    fn a_fold_length_that_resolves_to_zero_is_refused_by_name() {
        let cells = [0u32];
        for len in [FoldLen::Device(PORT), FoldLen::Host(0)] {
            let error = resolve_fold_len(len, 8, 0, Some(&cells[..])).unwrap_err();
            let said = error.to_string();
            assert!(said.contains("bonus token"), "{said}");
        }
        // The bound clamps to zero just as loudly: a verb that promised no
        // room cannot be handed a length that fits in it.
        let error = resolve_fold_len(FoldLen::Host(4), 0, 0, None).unwrap_err();
        assert!(error.to_string().contains("bonus token"), "{error}");
    }

    /// **A DEVICE-RESIDENT LENGTH AGAINST NO RESOLVED PORT IS A REFUSAL, NOT A
    /// GUESS.** The lane said the count lives on the device; if the program
    /// attached to it bound no such port there is no count anywhere, and
    /// falling back to the bound would fold the whole speculative window
    /// including the tokens the verifier rejected.
    #[test]
    fn a_device_fold_length_with_no_resolved_port_is_refused() {
        let error = resolve_fold_len(FoldLen::Device(PORT), 8, 0, None).unwrap_err();
        assert!(error.to_string().contains("resolved no such port"), "{error}");
    }
}
