use engine::fire::{FoldLen, Masking, RsReset, RsVerb};
use engine::frame::{Demand, Shell as FrameShell};
use poem_exec::fire::{FireDescriptor, Lane as FireLane, compose_axes_at};

use crate::error::{Fault, Result};
use crate::record;
use crate::run::RsMove;
use crate::store::kv::{self, Seat};
use crate::window::Windows;

use super::{
    Enqueued, InputFeedPlan, MROPE_COORDS, Media, MergeLand, PATCH_ROUTE_DROP, Prepared, RsFire,
    Settled, Shell, StepView, VoxelFeedPlan,
};

const WINDOWS_MEMO: usize = 8;

pub(super) struct WindowsMemo {
    tables: [poem_exec::fire::WindowTable; 3],
    indptr_host: Vec<i32>,
    bucket: u32,
    copies: bool,
    windows: Windows,
    packed: Vec<i32>,
}

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
        let _ = prev;
        // A step refused here fires nothing: what its lanes' device-counted
        // folds left ahead goes back to where the step found it.
        let ahead = self.rs_ahead.clone();
        // Nor does it carry any set's written end.
        let written = self.written.clone();
        let built = kernels_cuda::tally::counts().built;
        let prepared = self.prepare_step(step);
        self.count_built_in_fire(built);
        if prepared.is_err() {
            self.rs_ahead = ahead;
            self.written = written;
        }
        prepared
    }

    fn enqueue<'a>(&mut self, prepared: Prepared<'a>) -> Result<Enqueued<'a>>
    where
        Self: 'a,
    {
        let mut p = prepared;
        let slot = p
            .slot
            .take()
            .expect("a `Prepared` holds its staging slot until `enqueue` borrows it");
        let built = kernels_cuda::tally::counts().built;
        let enqueued = self.enqueue_on(&mut p, &slot);
        self.count_built_in_fire(built);
        match enqueued {
            Ok((launches, readback)) => {
                p.slot = Some(slot);
                Ok(Enqueued {
                    prepared: p,
                    launches,
                    readback,
                })
            }
            Err(fault) => {
                let _ = self.device.synchronize();
                drop(slot);
                Err(fault)
            }
        }
    }

    fn settle<'a>(&mut self, enqueued: Enqueued<'a>) -> Result<Settled>
    where
        Self: 'a,
    {
        self.settle_step(enqueued, None)
    }
}

impl Shell {
    /// Counts what this thread built since `before` (the kernels' tally) as
    /// built on a serving fire's path: an instantiation no walk of the load
    /// resolved (`engine::live::Tally::BuiltInFire`). An arming fire, and
    /// any before the load sealed its bodies, builds as the load does.
    pub(super) fn count_built_in_fire(&mut self, before: u64) {
        if self.arming || !self.cache.bodies_sealed() {
            return;
        }
        let built = kernels_cuda::tally::counts().built - before;
        if built == 0 {
            return;
        }
        self.live.tally(engine::live::Tally::BuiltInFire, built);
        let total = self.live.tallied(engine::live::Tally::BuiltInFire);
        if total.is_power_of_two() || built >= total {
            eprintln!(
                "engine-cuda: WARNING: a serving fire built {built} kernel instantiation(s) no \
                 walk of the load resolved; {total} so far (Tally::BuiltInFire). Every one a \
                 fire can reach is built before ready"
            );
        }
    }
}

fn stated_fold(len: FoldLen, lane: usize, port: Option<&[u32]>) -> Result<u32> {
    Ok(match len {
        FoldLen::Host(n) => n,
        FoldLen::Device => {
            let cells = port.ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length, and the program \
                         attached to it resolved no `fold` port"
                    ),
                )
            })?;
            fold_cell(cells, lane).ok_or_else(|| {
                Fault::program(
                    "serve::rs",
                    format!(
                        "lane {lane} states a device-resident fold length on a `fold` port \
                         whose cell carries {} entries",
                        cells.len()
                    ),
                )
            })?
        }
    })
}

/// Lane `lane`'s fold in a `fold` port's cell: a cell of one entry states
/// every lane's fold, a longer one a fold a lane, and a lane past it none.
fn fold_cell(cells: &[u32], lane: usize) -> Option<u32> {
    match cells {
        [every] => Some(*every),
        cells => cells.get(lane).copied(),
    }
}

/// The slots the state in record `slot` holds as a lane's fire finds it.
/// A buffered lane states `folded`, which a device-counted fold not yet
/// settled has left behind what `ahead` holds; a lane that starts fresh, or
/// folds its rows itself (`None`: a working set new to the record starts
/// with such a fire), leaves nothing ahead.
fn fold_held(
    ahead: &mut std::collections::HashMap<u32, u32>,
    slot: u32,
    folded: Option<u32>,
    fresh: bool,
) -> u32 {
    match (ahead.get(&slot), folded) {
        (Some(&at), Some(folded)) if at > folded && !fresh => at,
        (_, folded) => {
            ahead.remove(&slot);
            folded.unwrap_or(0)
        }
    }
}

/// A lane's table of a window list retaining `retained` tokens, where it
/// names the list's slot refs: a page wholly behind the window of the lane's
/// first row reads the null slot, the rest the refs the lane was handed, one
/// a page of its table.
fn window_refs(
    paging: &kv::Paging,
    retained: Option<u32>,
    seat: &kv::Seat,
    refs: &[u32],
) -> Result<Vec<u32>> {
    let Some(retained) = retained.filter(|_| paging.window.is_some()) else {
        return Ok(refs.to_vec());
    };
    let page_size = u64::from(paging.page_size);
    let pages = (u64::from(seat.have) + u64::from(seat.rows))
        .div_ceil(page_size)
        .max(1);
    let first = u64::from(seat.have.saturating_sub(retained)) / page_size;
    (0..pages)
        .map(|page| {
            if page < first {
                return Ok(0);
            }
            match refs.get(page as usize) {
                Some(&slot) if slot != 0 => Ok(slot),
                Some(_) => Err(Fault::Unbound {
                    what: format!(
                        "windowed page {page} of a lane holding {} tokens, which its window \
                         still reads and the runtime has released",
                        seat.have
                    ),
                }),
                None => Err(Fault::Ceiling {
                    what: "window list slot refs this lane stated",
                    need: page + 1,
                    have: refs.len() as u64,
                }),
            }
        })
        .collect()
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// What a class of a body's ladder reads out at its rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyReads {
    /// Its lanes' last rows, through the readout plane: one a lane.
    Last,
    /// A drafter's verify (a class no plain request lands): a row it checks
    /// a lane, up to the readouts a lane takes, through the readout plane.
    Verify,
    /// Every row of its rung, off a tokens-shaped plane.
    Rows,
}

/// The readout rows a body carries, and what makes them up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BodyReadouts {
    last: u32,
    verified: u32,
    every: u32,
    ceiling: u32,
}

/// The readout rows a body of `reads` (each class's reading at its rung)
/// carries. The readout plane's rows (last rows and verified rows) are the
/// lanes', which every class of the fire shares: summed over the classes,
/// they are capped once at `plane`, the rows the arena lays the plane for,
/// so the carve never passes the slot. Rows read off a tokens-shaped plane
/// add on; no readout passes the fire's `bucket`.
fn body_readouts(
    reads: &[(BodyReads, u32)],
    lane_ceiling: u32,
    max_lanes: u32,
    per_lane: u32,
    plane: u32,
    bucket: u32,
) -> BodyReadouts {
    let (mut last, mut verified, mut every) = (0u32, 0u32, 0u32);
    for &(reads, rung) in reads {
        match reads {
            BodyReads::Last => last += rung.min(lane_ceiling),
            BodyReads::Verify => verified += rung.min(lane_ceiling.saturating_mul(per_lane)),
            BodyReads::Rows => every += rung,
        }
    }
    let readout = (last.min(max_lanes) + verified).min(plane);
    BodyReadouts {
        last,
        verified,
        every,
        ceiling: (readout + every).min(bucket),
    }
}

/// The rows a fire's kernels write through its readout plane: a readout as
/// tall as the fire is launched at the fire's bucket, so the plane must
/// hold the bucket, not only the rows read out.
/// A fire that reads nothing out (an idle expert-parallel rank's) writes
/// nothing through it.
fn readout_extent(rows: u32, bucket: u32, readouts: usize) -> u64 {
    if readouts == 0 {
        return 0;
    }
    let pad = kernels_cuda::Pad { rows, bucket };
    u64::from(pad.extent(u32::try_from(readouts).unwrap_or(u32::MAX)))
}

impl Shell {
    /// Prepares a step: its lanes' pages, records and tables, ahead of its
    /// fire.
    fn prepare_step<'a>(&mut self, step: StepView<'a>) -> Result<Prepared<'a>> {
        let StepView {
            lanes,
            attachments,
            media,
            ..
        } = step;
        let arming = self.arming;
        let copies = self.copies;
        let tapped = lanes.iter().any(|seated| seated.tap.is_some());
        if tapped && (arming || !self.taps_kv()) {
            return Err(Fault::Unsupported(
                "a lane asks for a tap, and this load reads no layer out as it goes".to_string(),
            ));
        }

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

        // The lanes an attachment's pass spans.
        let span = |attached: &super::Attached| {
            let first = (attached.lane as usize).min(lanes.len());
            let end = first
                .saturating_add(attached.lanes as usize)
                .min(lanes.len());
            &lanes[first..end]
        };
        let can_runahead = self.runahead.runs_ahead()
            && attachments.iter().all(|attached| {
                match self.programs.geometry_of(attached.instance) {
                    None | Some(eta_ir::registry::GeometryClass::Host) => true,
                    Some(eta_ir::registry::GeometryClass::DecodeEnvelope) => {
                        self.programs
                            .token_device_source(attached.instance)
                            .is_some()
                            && !self.programs.reads_mask_on_host(attached.instance)
                            && lanes.get(attached.lane as usize).is_some_and(|seated| {
                                seated.lane.tokens.len() == 1
                                    && !matches!(
                                        seated.rs,
                                        RsVerb::Buffer {
                                            fold: FoldLen::Device,
                                            ..
                                        }
                                    )
                            })
                    }
                    Some(eta_ir::registry::GeometryClass::DeviceGeometry) => false,
                }
            });

        // A device-carried descriptor is read off its ring on the host, so owed
        // epilogues writing a ring this fire resolves settle first; one feeding only
        // device copies (a draft into the next verify) leaves the fire free.
        let resolves_owed = |owed: &super::GuestBatch| {
            let puts: std::collections::HashSet<usize> = owed
                .instances()
                .flat_map(|instance| self.programs.shared_puts(instance))
                .collect();
            attachments.iter().any(|attached| {
                self.programs
                    .host_resolved_rings(attached.instance)
                    .iter()
                    .any(|ring| puts.contains(ring))
            })
        };
        if !can_runahead
            && self.owed.iter().any(resolves_owed)
            && attachments.iter().any(|attached| {
                self.programs
                    .geometry_of(attached.instance)
                    .is_some_and(|class| class != eta_ir::registry::GeometryClass::Host)
            })
        {
            self.reap_guests_at("prepare")?;
        }
        super::btrace::mark("reap");
        // A prologue that moves a channel runs before the ports resolve.
        self.prologue(attachments)?;
        super::btrace::mark("prologue");
        let mut resolved: Vec<engine::ports::Envelope> = Vec::new();
        let mut token_src_of: Vec<Option<(u64, u32)>> = vec![None; lanes.len()];
        let mut envelope_of: Vec<Option<(usize, usize)>> = vec![None; lanes.len()];
        let mut fold_at: Vec<Option<u64>> = vec![None; lanes.len()];
        // A decode envelope's mask the device carries: each lane's rows of its
        // cell, `(address, keys)` in bool bytes a row.
        let mut mask_src_of: Vec<Option<(u64, u32)>> = vec![None; lanes.len()];
        // The lanes whose positions the program computes on the device.
        let mut positions_on_device = vec![false; lanes.len()];
        for attached in attachments {
            let first = attached.lane as usize;
            let src = self.programs.token_device_source(attached.instance);
            let envelope_class = self.programs.geometry_of(attached.instance)
                == Some(eta_ir::registry::GeometryClass::DecodeEnvelope);
            let spanned = span(attached);
            if self
                .programs
                .resolves(attached.instance, eta_ir::registry::Port::Positions)
            {
                positions_on_device[first..first + spanned.len()].fill(true);
            }
            // A decode envelope's device-carried tokens span the lanes of its
            // pass, a lane's rows each; a class that places its own lanes
            // carries them whole, at its first.
            if !envelope_class {
                if first < lanes.len() {
                    token_src_of[first] = src;
                }
            } else if let Some((src, native)) = src {
                let rows: u32 = spanned
                    .iter()
                    .map(|seated| seated.lane.tokens.len() as u32)
                    .sum();
                if rows * 4 != native {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "instance {} carries {} token(s) on the device and its pass's {} \
                             lane(s) carry {rows} row(s); a decode envelope's tokens span its \
                             lanes' rows",
                            attached.instance,
                            native / 4,
                            spanned.len()
                        ),
                    ));
                }
                let mut at = 0u32;
                for (lane, seated) in spanned.iter().enumerate() {
                    let bytes = seated.lane.tokens.len() as u32 * 4;
                    token_src_of[first + lane] = Some((src + u64::from(at), bytes));
                    at += bytes;
                }
            }
            if let Some((mask_src, native)) = self.programs.mask_device_source(attached.instance) {
                let rows: u32 = spanned
                    .iter()
                    .map(|seated| seated.lane.tokens.len() as u32)
                    .sum();
                if rows == 0 || native % rows != 0 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "instance {} carries its attention mask on the device in a \
                             {native}-byte cell and its {} lane(s) carry {rows} row(s); a \
                             mask holds a row of keys a row",
                            attached.instance,
                            spanned.len()
                        ),
                    ));
                }
                let keys = native / rows;
                let mut at = 0u32;
                for (lane, seated) in spanned.iter().enumerate() {
                    mask_src_of[first + lane] = Some((mask_src + u64::from(at * keys), keys));
                    at += seated.lane.tokens.len() as u32;
                }
            }
            if can_runahead && envelope_class {
                if first < lanes.len() {
                    fold_at[first] = self
                        .programs
                        .fold_device_source(attached.instance)
                        .map(|(cell, _)| cell);
                }
                continue;
            }
            let Some(mut envelope) = self.programs.envelope(attached.instance)? else {
                continue;
            };
            envelope.host_rows = lanes
                .iter()
                .skip(first)
                .take(attached.lanes.max(1) as usize)
                .map(|seated| seated.lane.tokens.len() as u32)
                .collect();
            let carried = envelope.lanes();
            // The ports state this attachment's lanes alone: one describing
            // more would set the next lanes' positions and slots.
            let owned = attached.lanes.max(1);
            if carried > owned as usize {
                return Err(Fault::Overreach {
                    instance: attached.instance,
                    lanes: attached.lane..attached.lane.saturating_add(owned),
                    carried,
                });
            }
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

        let page_tokens = self.pools.paging().page_size.max(1);
        let mut device_pages: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
        let mut device_window: Vec<Option<Vec<u32>>> = vec![None; lanes.len()];
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
            if ports.owns_slots() {
                lane_rows[source] = ports.rows();
            }
            let rows = lane_rows[source] as usize;
            if let (Some(slots), Some(reach)) = (ports.slots(rows)?, ports.reach(rows)?) {
                let relative: Vec<u32> = (0..reach.div_ceil(page_tokens)).collect();
                device_pages[source] = Some(
                    relative
                        .iter()
                        .map(|&page| translate(page, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                );
                let window = lanes[source].window;
                if !window.is_empty() {
                    device_window[source] = Some(
                        relative
                            .iter()
                            .map(|&page| window.get(page as usize).copied().unwrap_or(0))
                            .collect(),
                    );
                }
                // The table is the working set's whole, to read through: a
                // row lands only in the slots the lane may write.
                let writable = lanes[source].folded..lanes[source].write_end;
                if let Some(why) = engine::fire::misplaced_write(writable, slots) {
                    return Err(Fault::KvWrite {
                        lane: source as u32,
                        why,
                    });
                }
                device_writes[source] = Some((
                    slots
                        .iter()
                        .map(|&slot| translate(slot / page_tokens, "slots"))
                        .collect::<Result<Vec<u32>>>()?,
                    slots.iter().map(|&slot| slot % page_tokens).collect(),
                ));
            }
            if let Some((cells, stride)) = ports.mask(rows)? {
                if rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane that \
                             states its own slots has no such order this shell can derive"
                        ),
                    ));
                }
                device_masks[source] = Some(engine::ports::from_dense(cells, stride));
            }
        }

        let mut clips_of: Vec<Option<crate::voxels::Clips<'_>>> = vec![None; lanes.len()];
        for shot in step.voxels {
            let at = shot.lane as usize;
            if at >= lanes.len() {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "the lane index is past the submission",
                });
            }
            if clips_of[at].is_some() {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "a lane's clips are one submission",
                });
            }
            if shot.clips.iter().any(|b| b.contains(&0)) {
                return Err(Fault::VoxelPayload {
                    lane: shot.lane,
                    what: "a clip's box has a zero side",
                });
            }
            clips_of[at] = Some(*shot);
        }

        let row_bytes = self.patch_seat.map_or(0, |seat| seat.row_bytes);
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
            let need = patch_rows * row_bytes;
            if need != shot.patches.len() as u64 || patch_rows != shot.routes.len() as u64 {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need,
                    have: shot.patches.len() as u64,
                });
            }
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
            let grids_owed = shot.rows.len() as u64 * MROPE_COORDS as u64;
            if shot.grids.len() as u64 != grids_owed {
                return Err(Fault::PatchPayload {
                    lane: shot.lane,
                    need: grids_owed,
                    have: shot.grids.len() as u64,
                });
            }
            let rows = lane_rows[at];
            let drop = self.drops_patch_rows;
            if let Some((j, &route)) = shot.routes.iter().enumerate().find(|&(_, &route)| {
                !(drop && route == PATCH_ROUTE_DROP) && (route < 0 || route as u32 >= rows)
            }) {
                return Err(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::PatchRoute {
                        at: j as u32,
                        route,
                        rows,
                    },
                )));
            }
            media_of[at] = Some(shot);
        }

        super::btrace::mark("ports");
        let submitted: Vec<FireLane> = lanes
            .iter()
            .zip(&lane_rows)
            .enumerate()
            .map(|(at, (seated, &rows))| match (media_of[at], clips_of[at]) {
                (None, None) => FireLane::new(seated.lane.word, rows),
                (Some(shot), _) => FireLane::with_images(
                    seated.lane.word,
                    rows,
                    shot.rows.len() as u32,
                    shot.rows.iter().sum(),
                ),
                (None, Some(clips)) => FireLane::with_clips(
                    seated.lane.word,
                    rows,
                    clips.clips.len() as u32,
                    u32::try_from(clips.voxels()).unwrap_or(u32::MAX),
                ),
            })
            .collect();
        // A buffering fire, or one of a drafter's class, pads to a bucket the
        // lattice records it at (`Shell::rs_rungs`).
        let buffering = lanes.iter().any(|seated| {
            matches!(seated.rs, RsVerb::Buffer { .. }) || self.drafts_only(seated.lane.word)
        });
        let at = self.forced_bucket.take().or_else(|| {
            let rows: u32 = lane_rows.iter().sum();
            buffering
                .then(|| self.rs_rungs().into_iter().find(|&bucket| bucket >= rows))
                .flatten()
        });
        let composition = compose_axes_at(&self.compiled, &self.budgets, &submitted, at)?;
        let mut readout_rows: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut readout_first = vec![0u32; lanes.len()];
        let mut readout_count = vec![0u32; lanes.len()];
        {
            let mut placed: Vec<(u32, usize)> = composition
                .lanes()
                .iter()
                .map(|row| (row.row_offset, row.source as usize))
                .collect();
            placed.sort_unstable();
            for (row_offset, source) in placed {
                let owned = composition
                    .lanes()
                    .iter()
                    .find(|row| row.source as usize == source)
                    .map_or(0, |row| row.rows);
                let stated = lanes.get(source).and_then(|seated| seated.readout);
                let wanted: Vec<u32> = match stated {
                    Some(rows) if !rows.is_empty() => rows.to_vec(),
                    _ => vec![owned.saturating_sub(1)],
                };
                readout_first[source] = readout_rows.len() as u32;
                readout_count[source] = wanted.len() as u32;
                for row in wanted {
                    if row >= owned {
                        return Err(Fault::Ceiling {
                            what: "rows in the lane a readout names",
                            need: u64::from(row) + 1,
                            have: u64::from(owned),
                        });
                    }
                    readout_rows.push(i32::try_from(row_offset + row).unwrap_or(0));
                }
            }
        }
        // Readout rows are gathered into the readouts-carved plane; a row or
        // voxel output is read off a tokens-shaped plane and is bounded by
        // the fire's own rows.
        let readouts_ceiling = composition
            .lanes()
            .iter()
            .filter(|row| readout_count[row.source as usize] > 0)
            .map(|row| match self.exports.readback(row.class as usize) {
                Some(out) if !out.decl.readout => u64::from(self.budget.max_tokens),
                Some(out) => self
                    .compiled
                    .arena
                    .readout_ceiling_of(out.values[0])
                    .unwrap_or_else(|| poem_compiler::arena::readouts_ceiling(&self.budget)),
                None if self.exports.reads_readout_rows() => {
                    poem_compiler::arena::readouts_ceiling(&self.budget)
                }
                None => u64::from(self.budget.max_tokens),
            })
            .min()
            .unwrap_or_else(|| poem_compiler::arena::readouts_ceiling(&self.budget));
        let received = !self.compiled.expert.is_empty();
        let launched =
            super::launched_rows(self.pad, received, composition.rows(), composition.bucket());
        let readout_extent = readout_extent(composition.rows(), launched, readout_rows.len());
        if readout_extent > readouts_ceiling {
            return Err(Fault::Ceiling {
                what: "rows one fire reads out through its readout plane",
                need: readout_extent,
                have: readouts_ceiling,
            });
        }
        let descriptor = FireDescriptor::of(&composition);

        let lane_facts: Vec<poem_exec::fire::LaneFacts> = lanes
            .iter()
            .map(|seated| poem_exec::fire::LaneFacts {
                group: seated.group,
            })
            .collect();
        let (group_of_lane, packings) = if self.feeds.selections.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let groups = poem_exec::fire::group_of_lane(composition.lanes(), &lane_facts);
            let mut packings = Vec::with_capacity(self.feeds.selections.len());
            for &select in &self.feeds.selections {
                let packed =
                    poem_exec::fire::pack(select, composition.lanes(), &groups, composition.rows())
                        .map_err(poem_exec::Error::Fire)?;
                packings.push(packed);
            }
            (groups, packings)
        };
        let group_masks =
            self.group_masks_of(lanes, &composition, &group_of_lane, &packings, &clips_of)?;

        let mut merge_lands: Vec<MergeLand> = Vec::new();
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for merged in &self.feeds.merged {
                if !merged.select.holds(row.word) {
                    continue;
                }
                let fed = seated.inputs.iter().any(|feed| feed.id == merged.seat.id);
                merge_lands.push(MergeLand {
                    merge: merged.merge,
                    seat: merged.seat,
                    first: if merged.seat.per_lane() {
                        fire_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows: if merged.seat.per_lane() { 1 } else { row.rows },
                    fed,
                });
            }
        }

        let mut input_feeds: Vec<InputFeedPlan> = Vec::new();
        let mut voxel_feeds: Vec<VoxelFeedPlan> = Vec::new();
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            let seated = &lanes[row.source as usize];
            for feed in seated.inputs {
                if self.feeds.seat(feed.id).is_none() {
                    return Err(Fault::program(
                        "serve::inputs",
                        format!(
                            "lane {} feeds guest input {} and this plan reads no such input",
                            row.source, feed.id
                        ),
                    ));
                }
            }
            for (seat, readers) in &self.feeds.inputs {
                if !readers.contains(row.class as usize) {
                    continue;
                }
                let Some(feed) = seated.inputs.iter().find(|feed| feed.id == seat.id) else {
                    if arming {
                        continue;
                    }
                    if seat.voxel()
                        && clips_of
                            .get(row.source as usize)
                            .and_then(Option::as_ref)
                            .is_some_and(|shot| !shot.payload.is_empty())
                    {
                        continue;
                    }
                    return Err(Fault::program(
                        "serve::inputs",
                        format!(
                            "lane {} runs a class that reads guest input {} and feeds it no \
                             channel; a lane feeds every input its class reads from a channel \
                             cell at every submit (`Lane::inputs`)",
                            row.source, seat.id
                        ),
                    ));
                };
                let instance = attachments
                    .iter()
                    .find(|attached| {
                        (attached.lane..attached.lane + attached.lanes).contains(&row.source)
                    })
                    .map(|attached| attached.instance)
                    .ok_or_else(|| {
                        Fault::program(
                            "serve::inputs",
                            format!(
                                "lane {} feeds guest input {} off channel {} but attaches no \
                                 instance; an input's cell is read through the instance that \
                                 carries the channel",
                                row.source, seat.id, feed.channel
                            ),
                        )
                    })?;
                let rows = if seat.voxel() {
                    row.voxels
                } else if seat.per_lane() {
                    1
                } else {
                    row.rows
                };
                let bytes = u64::from(rows) * seat.row_bytes();
                let f32_bytes = u64::from(rows) * u64::from(seat.width) * 4;
                let cell = self.programs.feed_cell_bytes(instance, feed.channel)?;
                let cast = cell != bytes && seat.dtype == poem_ir::Dtype::Bf16 && cell == f32_bytes;
                if cell != bytes && !cast {
                    return Err(Fault::program(
                        "serve::inputs",
                        format!(
                            "lane {} feeds guest input {} off channel {} whose cell is {cell} \
                             bytes, and the input wants {rows} row(s) x {} {:?} = {bytes} \
                             bytes (the lane's rows by the input's width; an f32 cell of \
                             {f32_bytes} bytes would be cast)",
                            row.source, seat.id, feed.channel, seat.width, seat.dtype
                        ),
                    ));
                }
                if seat.voxel() {
                    voxel_feeds.push(VoxelFeedPlan {
                        lane: row.source,
                        first: row.voxel_offset,
                        rows,
                        width: seat.width,
                        cast,
                        bytes,
                        channel: feed.channel,
                        instance,
                    });
                    continue;
                }
                input_feeds.push(InputFeedPlan {
                    seat: *seat,
                    first: if seat.per_lane() {
                        fire_lane as u32
                    } else {
                        row.row_offset
                    },
                    rows,
                    bytes,
                    cast,
                    channel: feed.channel,
                    instance,
                });
            }
        }
        let fed_width = voxel_feeds.first().map_or(0, |feed| feed.width);
        if voxel_feeds.iter().any(|feed| feed.width != fed_width) {
            return Err(Fault::VoxelPayload {
                lane: voxel_feeds[0].lane,
                what: "two lanes feed voxel inputs of two widths in one fire (M0: one voxel \
                       class per fire)",
            });
        }
        let voxel_tables = if composition.voxel_rows() == 0 {
            // Token rows reading their cells need a token grid even with no clips: one empty clip.
            match self.voxels.as_ref() {
                Some(store) if store.seat().token_patch.is_some() => crate::voxels::Tables {
                    grid: vec![0; 4],
                    token_grid: vec![0; 4],
                    clip_lanes: vec![0],
                    ..crate::voxels::Tables::default()
                },
                _ => crate::voxels::Tables::default(),
            }
        } else {
            let Some(store) = self.voxels.as_ref() else {
                return Err(Fault::VoxelPayload {
                    lane: clips_of.iter().position(Option::is_some).unwrap_or(0) as u32,
                    what: "this load seats no voxel axis",
                });
            };
            if composition.voxel_classes().present_in_order().count() > 1 {
                return Err(Fault::VoxelPayload {
                    lane: clips_of.iter().position(Option::is_some).unwrap_or(0) as u32,
                    what: "the clips of one fire fall in two classes, and a spatial launch \
                           runs over the whole voxel rectangle (M0: one voxel class per fire)",
                });
            }
            // A clip's conv caches are State rows, read as every recurrent
            // row is: through the fire's lane table, which names each
            // lane's record. A clip names its lane.
            let lane_of: Vec<u32> = (0..lanes.len() as u32).collect();
            crate::voxels::Tables::of(
                &store.seat(),
                composition.lanes(),
                &clips_of,
                &lane_of,
                fed_width,
            )?
        };

        // The patch rows the fire launches over, the padded ones inert.
        let patch_rows = super::launched_rows(
            self.pad,
            false,
            composition.patch_rows(),
            composition.patch_bucket(),
        ) as usize;
        let (patch_payload, patch_segments, patch_routes, patch_positions, patch_grids) =
            if composition.patch_rows() == 0 {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
            } else {
                let stride = row_bytes as usize;
                let mut payload = vec![0u8; patch_rows * stride];
                let mut routes = vec![
                    if self.drops_patch_rows {
                        PATCH_ROUTE_DROP
                    } else {
                        0
                    };
                    patch_rows
                ];
                let mut positions = vec![0i32; patch_rows * MROPE_COORDS];
                let fold = (self.patch_fold as usize).max(1);
                let mut grids = vec![0i32; composition.images() as usize * MROPE_COORDS];
                let mut per_image = vec![0u32; composition.images() as usize];
                for row in composition.lanes() {
                    let Some(shot) = media_of[row.source as usize] else {
                        continue;
                    };
                    let at = row.patch_offset as usize * stride;
                    payload[at..at + shot.patches.len()].copy_from_slice(shot.patches);
                    let landed = (row.patch_offset as usize) / fold;
                    let live = shot.rows.iter().map(|rows| *rows as usize).sum::<usize>() / fold;
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
                    let at_grids = row.image_offset as usize * MROPE_COORDS;
                    grids[at_grids..at_grids + shot.grids.len()].copy_from_slice(shot.grids);
                    for (i, &rows) in shot.rows.iter().enumerate() {
                        per_image[row.image_offset as usize + i] = rows;
                    }
                }
                let mut segments = Vec::with_capacity(per_image.len() + 1);
                let mut at = 0i32;
                segments.push(at);
                for rows in per_image {
                    at += rows as i32;
                    segments.push(at);
                }
                (payload, segments, routes, positions, grids)
            };
        let rows = composition.rows();

        let mut seats: Vec<Seat> = Vec::with_capacity(lanes.len());
        let mut tables: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        let mut window_ids: Vec<std::borrow::Cow<'_, [u32]>> = Vec::with_capacity(lanes.len());
        // Each lane's slot refs of the window lists past the first.
        let mut more_windows: Vec<&[&[u32]]> = Vec::with_capacity(lanes.len());
        let mut kv_less_seats: Vec<bool> = Vec::with_capacity(lanes.len());
        let mut masks: Vec<crate::mask::LaneMask<'_>> = Vec::with_capacity(lanes.len());
        // Whether some lane's class reads the fire's masks: then every lane
        // states its entry, a mask or none.
        let mut reads_masks = false;
        let mut tokens: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut token_injects: Vec<crate::inputs::TokenInject> = Vec::new();
        let mut positions: Vec<i32> = Vec::with_capacity(rows as usize);
        let mut writes: Vec<Option<(i32, i32)>> = Vec::with_capacity(rows as usize);
        let mut slot_ids: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut fresh: Vec<u32> = Vec::new();
        let mut rs_moves: Vec<RsMove<'a>> = Vec::with_capacity(lanes.len());
        let mut rs_pages: Vec<u32> = Vec::new();
        let mut rs_reads: Vec<Vec<std::ops::Range<u32>>> = vec![Vec::new(); lanes.len()];
        let mut rs_folded: Vec<Option<u32>> = vec![None; lanes.len()];
        // Per lane, one past the last slot its rows land at, where a port
        // says where they land: what its report tells the runtime.
        let mut rs_ends: Vec<Option<u32>> = vec![None; lanes.len()];
        let mut rs_lens: Vec<i32> = Vec::with_capacity(lanes.len());
        let mut rs_order: Vec<u32> = vec![0; lanes.len()];
        // Each seat's lane, and where its rings stand and how far its fire
        // folds them (`super::ring`), a seat each.
        let mut seat_sources: Vec<usize> = Vec::with_capacity(lanes.len());
        let mut window_folds: Vec<Option<(u32, u32)>> = Vec::with_capacity(lanes.len());
        let ringed_load = self.pools.ring().is_some();
        let mut snapshots = crate::store::rs::SnapTable {
            indptr: vec![0],
            ..Default::default()
        };
        let mut adapter_routes: Vec<i32> = Vec::new();
        let any_adapter = lanes.iter().any(|seated| seated.adapter.is_some());
        if any_adapter {
            adapter_routes.reserve(rows as usize);
        }
        // A lane whose mask the device carries stages a cleared row of keys
        // its cell's bits are packed over.
        let carried_masks: Vec<Option<Masking>> = mask_src_of
            .iter()
            .map(|carried| {
                carried.map(|(_, keys)| {
                    Masking::Extent(engine::Mask::new(vec![keys], u64::from(keys)))
                })
            })
            .collect();
        // `(the lane's index in masks, its row of the device cell, keys)`.
        let mut pending_mask_injects: Vec<(usize, u64, u32)> = Vec::new();
        // What each lane writes and reads of its set (`engine::extent`).
        let mut extents = engine::extent::FireExtents::default();
        for row in composition.lanes() {
            let source = row.source as usize;
            let seated = &lanes[source];
            let lane = &seated.lane;
            let ports = match envelope_of[source] {
                Some((held, at)) => Some(resolved[held].lane(at, source)?),
                None => None,
            };
            let have = match ports.as_ref().filter(|ports| ports.owns_slots()) {
                _ if seated.kv_less => 0,
                Some(ports) => {
                    let after = ports.reach(row.rows as usize)?.ok_or_else(|| {
                        Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} states where its rows land and reaches no \
                                 slot; the page count, the last page's fill and the \
                                 attention schedules are all carved from what it reads"
                            ),
                        )
                    })?;
                    if after < row.rows {
                        return Err(Fault::program(
                            "serve::prepare",
                            format!(
                                "lane {source} reads a KV extent of {after} and this fire \
                                 writes {} row(s) into it; the extent is AFTER the append, \
                                 so it can never be shorter than what the append adds",
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
            let begins = match seated.rs_reset {
                RsReset::Inferred => have == 0,
                RsReset::Fresh => true,
                RsReset::Held => false,
            };
            if begins {
                fresh.push(seated.rs_slot.unwrap_or(lane.slot));
            }
            seats.push(Seat {
                slot: lane.slot,
                have,
                rows: row.rows,
            });
            kv_less_seats.push(seated.kv_less);
            tables.push(match &device_pages[source] {
                Some(pages) => std::borrow::Cow::Owned(pages.clone()),
                None => std::borrow::Cow::Borrowed(seated.pages),
            });
            window_ids.push(match (&device_pages[source], &device_window[source]) {
                (_, Some(ids)) => std::borrow::Cow::Owned(ids.clone()),
                (Some(_), None) => std::borrow::Cow::Borrowed(&[][..]),
                (None, None) => std::borrow::Cow::Borrowed(seated.window),
            });
            more_windows.push(seated.more_windows);
            // A kv-less lane's mask goes to the group-packed attention, not the paged prefill.
            let masking = if seated.kv_less {
                None
            } else {
                device_masks[source]
                    .as_ref()
                    .or(seated.mask)
                    .or(carried_masks[source].as_ref())
            };
            // A port has one source: the runtime states a mask it resolved, and
            // the engine resolves only one the program computes on the device.
            if !seated.kv_less
                && seated.mask.is_some()
                && (device_masks[source].is_some() || carried_masks[source].is_some())
            {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {source} states an attention mask and its program computes one \
                         on the device; a port is resolved by one side"
                    ),
                ));
            }
            if positions_on_device[source] && !seated.positions.is_empty() {
                return Err(Fault::program(
                    "serve::prepare",
                    format!(
                        "lane {source} states positions and its program computes them on the \
                         device; a port is resolved by one side"
                    ),
                ));
            }
            if !seated.kv_less
                && let Some((src, _)) = mask_src_of[source]
            {
                if row.rows != 1 {
                    return Err(Fault::program(
                        "serve::prepare",
                        format!(
                            "lane {source} carries its attention mask on the device and {} \
                             query rows; a row's keys are packed as its cell holds them, with \
                             no causal bound, so only a lane's one row reads them",
                            row.rows
                        ),
                    ));
                }
                // The guest's cell holds the lane's buffer, then its row:
                // columns from its committed boundary on.
                let extent = have
                    .saturating_sub(seated.folded.min(have))
                    .saturating_add(row.rows);
                pending_mask_injects.push((masks.len(), src, extent));
            }
            let runs_masked_arm = self.masked.contains(row.class as usize);
            reads_masks |= self.mask_read.contains(row.class as usize);
            let reads_group_mask = self.group_masked.contains(row.class as usize);
            if masking.is_some() && self.masked.is_empty() && self.group_masked.is_empty() {
                return Err(Fault::Maskless { lane: row.source });
            }
            // The mask is per-lane data: a lane stating none reads every key
            // in a masked arm too. One stated where no arm applies it is
            // refused; a group's mask is the lane's to state on any class
            // that reads one.
            if masking.is_some() && !runs_masked_arm && !reads_group_mask {
                return Err(Fault::MaskUnread {
                    lane: row.source,
                    word: lane.word,
                });
            }
            masks.push(crate::mask::LaneMask {
                // A mask a group's rows read only is staged with the group
                // masks (`group_masks`), not over the lane's keys.
                mask: masking.filter(|_| runs_masked_arm),
                have,
                folded: seated.folded.min(have),
                rows: row.rows,
                wide: self.wide.contains(row.class as usize),
            });
            slot_ids.push(seated.rs_slot.unwrap_or(lane.slot) as i32);
            if !seated.rs_snapshots.is_empty() && row.rows as usize != lane.tokens.len() {
                return Err(Fault::program(
                    "serve::rs",
                    format!(
                        "lane {} snapshots its recurrent state at rows of its own and runs \
                         {} of its {} rows in this fire",
                        row.source,
                        row.rows,
                        lane.tokens.len()
                    ),
                ));
            }
            for snap in seated.rs_snapshots {
                snapshots.rows.push(narrow(u64::from(snap.row)));
                snapshots.slots.push(narrow(u64::from(snap.slot)));
            }
            snapshots.indptr.push(narrow(snapshots.rows.len() as u64));
            let fire_lane = rs_moves.len();
            rs_order[row.source as usize] = fire_lane as u32;
            let port = envelope_of[source].and_then(|(held, _)| resolved[held].fold.as_deref());
            // The lane's rings: folded where its runtime states, or where a
            // fold counted on the device not yet settled left them, and
            // advanced by the fold its fire states (CONTRACT-BC 21).
            seat_sources.push(source);
            window_folds.push(match &seated.window_fold {
                Some(stated) if ringed_load => {
                    let slot = seated.rs_slot.unwrap_or(lane.slot);
                    let held = fold_held(
                        &mut self.rs_ahead,
                        slot,
                        Some(seated.folded),
                        seated.rs_reset == RsReset::Fresh,
                    );
                    let cell = envelope_of[source].map_or(rs_moves.len(), |(_, at)| at);
                    let to = held
                        .saturating_add(stated_fold(stated.fold, cell, port)?)
                        .min(have + row.rows);
                    if stated.fold == FoldLen::Device && matches!(seated.rs, RsVerb::Fold) {
                        self.rs_ahead.insert(slot, to);
                    }
                    Some((held, to - held))
                }
                _ => None,
            });
            let (verb, folded) = match &seated.rs {
                RsVerb::Fold => {
                    // A lane that folds its rows itself leaves nothing ahead:
                    // a working set new to this record starts so.
                    let slot = seated.rs_slot.unwrap_or(lane.slot);
                    fold_held(&mut self.rs_ahead, slot, None, false);
                    (RsMove::None, row.rows)
                }
                RsVerb::Buffer {
                    pages,
                    base,
                    folded,
                    fold,
                    lineage,
                } => {
                    rs_pages.extend_from_slice(pages);
                    // A device-counted fold not yet settled left the state
                    // past the runtime's length.
                    let slot = seated.rs_slot.unwrap_or(lane.slot);
                    let folded = &fold_held(
                        &mut self.rs_ahead,
                        slot,
                        Some(*folded),
                        seated.rs_reset == RsReset::Fresh,
                    );
                    let stated = match &ports {
                        Some(ports) => ports.slots(row.rows as usize)?,
                        None => None,
                    };
                    let first = stated
                        .and_then(|slots| slots.first().copied())
                        .unwrap_or(have);
                    if first < *folded || *folded < *base {
                        return Err(Fault::program(
                            "serve::rs",
                            format!(
                                "lane {} lands its rows at slot {first}, and its state holds \
                                 slots below {folded} with its buffer from slot {base}; a lane \
                                 writes past what its state holds, into its buffer",
                                row.source
                            ),
                        ));
                    }
                    let reads = engine::fire::buffered_reads(masking, *folded, first, *lineage)
                        .map_err(|error| {
                            Fault::program("serve::rs", format!("lane {}: {error}", row.source))
                        })?;
                    let replay = crate::run::replayed(&reads);
                    let cell = envelope_of[source].map_or(fire_lane, |(_, at)| at);
                    let to = folded
                        .saturating_add(stated_fold(*fold, cell, port)?)
                        .min(first + row.rows);
                    rs_folded[source] = Some(to - folded);
                    rs_ends[source] = Some(first + row.rows);
                    if *fold == FoldLen::Device {
                        self.rs_ahead.insert(slot, to);
                    }
                    let below: u32 = reads
                        .iter()
                        .map(|span| span.end.min(to).saturating_sub(span.start))
                        .sum();
                    let fold = below + to.saturating_sub(first).min(row.rows);
                    rs_reads[fire_lane] = reads
                        .iter()
                        .map(|span| span.start - base..span.end - base)
                        .collect();
                    (
                        RsMove::Scatter {
                            pages: pages.as_slice(),
                            at: first - base,
                            fold,
                        },
                        if fold == 0 { replay + row.rows } else { fold },
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
            // A lane that folds nothing reads the state as it stood before
            // the fire, while the lane beside it folds into the record the
            // fire writes: the two never meet in one row.
            if matches!(verb, RsMove::Scatter { fold: 0, .. })
                && let Some(read) = seated.rs_read
                && let Some(named) = slot_ids.last_mut()
            {
                *named = read as i32;
            }
            rs_moves.push(verb);
            rs_lens.push(narrow(u64::from(folded)));
            let runs_correction = self.corrected.contains(row.class as usize);
            if seated.adapter.is_some() && self.corrected.is_empty() {
                return Err(Fault::Adapterless { lane: row.source });
            }
            let unreachable = seated.adapter.is_some()
                && !runs_correction
                && !self
                    .compiled
                    .classes
                    .correction_reaches(&self.corrected, lane.word);
            if seated.adapter.is_some() != runs_correction && !unreachable {
                return Err(Fault::AdapterWord {
                    lane: row.source,
                    word: lane.word,
                    runs_correction,
                });
            }
            if any_adapter {
                let id = seated
                    .adapter
                    .map_or(-1, |id| i32::try_from(id).unwrap_or(-1));
                adapter_routes.extend(std::iter::repeat_n(id, row.rows as usize));
            }

            let rows_here = row.rows as usize;
            match ports.as_ref() {
                Some(ports) => {
                    ports.check_extent(have.saturating_add(row.rows))?;
                    let dst_off = tokens.len() as u64 * 4;
                    if let Some((src, native)) = token_src_of[source]
                        && native as usize == rows_here * 4
                    {
                        token_injects.push(crate::inputs::TokenInject {
                            dst_off,
                            src,
                            bytes: native as usize,
                        });
                    }
                    for &token in ports.tokens_for(rows_here)? {
                        tokens.push(token as i32);
                    }
                    // One source a port: positions the runtime resolved ride the
                    // lane (checked above), and the envelope carries only ones
                    // the program computes on the device.
                    match ports.positions_for(have, rows_here)? {
                        Some(stated) => positions.extend(stated.iter().map(|&p| p as i32)),
                        None if !seated.positions.is_empty() => {
                            positions.extend(seated.positions.iter().map(|&p| p as i32))
                        }
                        None => positions
                            .extend((0..rows_here).map(|at| narrow(u64::from(have) + at as u64))),
                    }
                    match &device_writes[source] {
                        Some((slots, offsets)) => {
                            writes.extend(slots.iter().zip(offsets).map(|(&page, &off)| {
                                Some((narrow(u64::from(page)), narrow(u64::from(off))))
                            }))
                        }
                        None => writes.extend(std::iter::repeat_n(None, rows_here)),
                    }
                }
                None => {
                    let dst_off = tokens.len() as u64 * 4;
                    if let Some((src, native)) = token_src_of[source]
                        && native as usize == rows_here * 4
                    {
                        token_injects.push(crate::inputs::TokenInject {
                            dst_off,
                            src,
                            bytes: native as usize,
                        });
                    }
                    for (at, token) in lane.tokens.iter().enumerate() {
                        tokens.push(*token as i32);
                        positions.push(
                            seated
                                .positions
                                .get(at)
                                .map_or_else(|| narrow(u64::from(have) + at as u64), |&p| p as i32),
                        );
                    }
                    writes.extend(std::iter::repeat_n(None, rows_here));
                }
            }
            if !seated.kv_less {
                let reach = have.saturating_add(row.rows);
                let set = (seated.set, seated.readable);
                match ports.as_ref().map(|ports| ports.slots(rows_here)).transpose()? {
                    Some(Some(slots)) => extents.lane(
                        &self.written,
                        source,
                        lane.slot,
                        set,
                        slots.iter().map(|&slot| slot..slot.saturating_add(1)),
                        reach,
                    ),
                    _ => extents.lane(
                        &self.written,
                        source,
                        lane.slot,
                        set,
                        std::iter::once(have..reach),
                        reach,
                    ),
                }
            }
        }
        // A page holds what its last owner left until its set writes it: no
        // lane reads past what its set holds written.
        extents
            .check()
            .map_err(|why| Fault::program("serve::prepare", why))?;
        self.written.commit(extents);

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
                triples[at..at + shot.token_positions.len()].copy_from_slice(shot.token_positions);
            }
            triples
        };
        super::btrace::mark("lanes");
        let page_size = u64::from(self.pools.paging().page_size).max(1);
        let kv_less = self.page_tables == 0;
        let demand = Demand {
            kv_pages: if kv_less {
                0
            } else {
                seats
                    .iter()
                    .zip(&tables)
                    .map(|(seat, table)| {
                        let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
                        let pages = after.div_ceil(page_size).max(1);
                        if table.is_empty() {
                            self.pools.paging().base(seat.slot).saturating_add(pages)
                        } else {
                            table
                                .iter()
                                .take(pages as usize)
                                .copied()
                                .max()
                                .map_or(0, |page| u64::from(page).saturating_add(1))
                        }
                    })
                    .max()
                    .map_or(0, |pages| u32::try_from(pages).unwrap_or(u32::MAX))
            },
            state_slots: slot_ids
                .iter()
                .map(|&row| (row as u32).saturating_add(1))
                .max()
                .unwrap_or(0),
            workspace: 0,
        };
        let mut kv_ranges: Vec<(u64, u64)> = Vec::new();
        for (seat, table) in seats.iter().zip(&tables).filter(|_| !kv_less) {
            let after = u64::from(seat.have).saturating_add(u64::from(seat.rows));
            let pages = after.div_ceil(page_size).max(1);
            if table.is_empty() {
                kv_ranges.push((self.pools.paging().base(seat.slot), pages));
            } else {
                for &page in table.iter().take(pages as usize) {
                    kv_ranges.push((u64::from(page), 1));
                }
            }
        }
        let paging = self.pools.paging();
        // Each window list's table a lane, narrowest window first. Where
        // lanes name slot refs a list has refs of its own; today's windowed
        // page ids name a slot of every list.
        let window_lists = if self.pools.has_windowed() {
            self.pools.geometries() - usize::from(!self.pools.windowed_geometry(0))
        } else {
            0
        };
        let mut window_tables: Vec<Vec<Vec<u32>>> = Vec::with_capacity(window_lists);
        // A load serving its windowed rows as ring + buffer composes each
        // lane's one windowed table (`super::ring`): its entries are page
        // ids already, and its window pages the buffer slots it names.
        let mut ring_work = super::ring::Work::default();
        let mut ring_buffers: Vec<u64> = Vec::new();
        let ringed = self.pools.ring_window();
        if let Some((window, page)) = ringed {
            ring_work.page_bytes = self
                .pools
                .ring()
                .map_or(0, |(_, ring)| ring.page_bytes);
            let planes = self.pools.ring_planes();
            let keys: Vec<u32> = seat_sources
                .iter()
                .map(|&source| lanes[source].rs_slot.unwrap_or(lanes[source].lane.slot))
                .collect();
            let mut null = [0i32];
            self.pools.page_ids(1, &mut null)?;
            let null = null[0];
            let mut own_next = 0u32;
            // A record a group of lanes shares folds once: the group's
            // common rows, which its lanes read from slots they share
            // (CONTRACT-BC 19). Each record's fold, as its first lane states.
            let mut record_folds: std::collections::HashMap<u32, (u32, u32)> =
                std::collections::HashMap::new();
            let stated_buffers: Vec<(u32, &[u32])> = (0..seats.len())
                .map(|lane| {
                    let stated = lanes[seat_sources[lane]].window_fold.is_some();
                    (keys[lane], if stated { window_ids[lane].as_ref() } else { &[][..] })
                })
                .collect();
            if let Some((one, other)) = super::ring::shared_pool(&stated_buffers) {
                return Err(Fault::Unbound {
                    what: format!(
                        "lanes {} and {} read one pool of buffered rows as separate hypotheses \
                         (one record, one window buffer, told apart by their masks) on a model \
                         with windowed layers: a window cuts such a pool by its slots, not by \
                         each hypothesis's positions. Windowed beams are served by a page list \
                         a lane",
                        seat_sources[one], seat_sources[other]
                    ),
                });
            }
            let mut composed_tables: Vec<Vec<u32>> = Vec::with_capacity(seats.len());
            for (lane, seat) in seats.iter().enumerate() {
                let seated = &lanes[seat_sources[lane]];
                if kv_less_seats[lane] {
                    composed_tables.push(tables[lane].to_vec());
                    continue;
                }
                let buffer_empty = window_ids[lane].is_empty();
                let (base, folded, fold) = match (&seated.window_fold, window_folds[lane]) {
                    // A lane with no buffer slot reads every row below its
                    // fire from its ring: its buffer starts at its fold's
                    // page, whatever base it states.
                    (Some(_), Some((folded, fold))) if buffer_empty => {
                        (folded / page * page, folded, fold)
                    }
                    (Some(stated), Some((folded, fold))) => (stated.base, folded, fold),
                    // A runtimeless lane folds every row it writes.
                    _ => {
                        let base = seat.have / page * page;
                        (base, seat.have, seat.rows)
                    }
                };
                let ring = self.pools.ring_pages(keys[lane])?;
                // A lane no runtime states a fold for (a runtimeless or an
                // arming lane) buffers nothing: it folds every row it
                // writes, on a record of its own.
                let stated = seated.window_fold.is_some();
                let buffer = window_ids[lane]
                    .iter()
                    .filter(|_| stated)
                    .map(|&slot| {
                        ring_buffers.push(u64::from(slot));
                        self.pools.window_slot(slot)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let alone =
                    !stated || keys.iter().filter(|&&key| key == keys[lane]).count() == 1;
                if !alone && folded + fold > seat.have {
                    return Err(Fault::program(
                        "serve::ring",
                        format!(
                            "lanes sharing a record fold rows [{folded}, {}) into it, past \
                             row {} where one of them computes its own in this fire: a group \
                             folds only the rows its lanes share",
                            folded + fold,
                            seat.have
                        ),
                    ));
                }
                let first_fold = match record_folds.get(&keys[lane]).filter(|_| stated) {
                    None => {
                        record_folds.insert(keys[lane], (folded, fold));
                        true
                    }
                    Some(&stated) if stated == (folded, fold) => false,
                    Some(&(was, by)) => {
                        return Err(Fault::program(
                            "serve::ring",
                            format!(
                                "lanes sharing a record fold it from {was} by {by} and from \
                                 {folded} by {fold}: a record folds once a fire"
                            ),
                        ));
                    }
                };
                let pools = &self.pools;
                let mut own = || {
                    let at = own_next;
                    own_next += 1;
                    Ok((pools.own_page_id(at)?, pools.own_page_at(at)?))
                };
                let composed = super::ring::compose(
                    &super::ring::RingLane {
                        ring: &ring,
                        buffer: &buffer,
                        base,
                        have: seat.have,
                        rows: seat.rows,
                        folded,
                        fold,
                        alone,
                    },
                    page,
                    window,
                    null,
                    &mut own,
                )?;
                self.ring_seen.direct += u64::from(composed.direct);
                self.ring_seen.own += u64::from(composed.own);
                self.ring_seen.read_then_folded += u64::from(composed.read_then_folded);
                // The group's fold is its first lane's; the others read.
                if first_fold {
                    ring_work.add(&composed, &planes);
                } else {
                    ring_work.fills.extend_from_slice(&composed.fills);
                }
                composed_tables.push(composed.table.iter().map(|&id| id as u32).collect());
            }
            window_tables.push(composed_tables);
        }
        for window in (0..window_lists).filter(|_| ringed.is_none()) {
            let retained = self.pools.window_tokens(window);
            window_tables.push(
                seats
                    .iter()
                    .enumerate()
                    .map(|(lane, seat)| {
                        let (table, ids) = (&tables[lane], &window_ids[lane]);
                        if kv_less_seats[lane] {
                            Ok(table.to_vec())
                        } else if self.pools.by_refs() {
                            let ids: &[u32] = match window {
                                0 => ids,
                                past => more_windows[lane].get(past - 1).copied().unwrap_or(&[]),
                            };
                            window_refs(&paging, retained, seat, ids)
                        } else {
                            kv::window_table(&paging, seat, table, ids)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        // The windowed pages this fire's tables name, a list's apart, backed
        // as they are first used.
        let window_pages: Vec<Vec<u64>> = if ringed.is_some() {
            vec![std::mem::take(&mut ring_buffers)]
        } else {
            window_tables
                .iter()
            .map(|tables| {
                tables
                    .iter()
                    .zip(&kv_less_seats)
                    .filter(|(_, kv_less)| !**kv_less)
                    .flat_map(|(table, _)| table.iter().map(|&page| u64::from(page)))
                    .collect()
            })
            .collect()
        };
        // A fire's extended run holds the widest bucket's tokens of
        // replayed rows (`store::rs::Buffers::rows_ext`), the most the
        // runtime lets one carry (`engine::fire::rung_rows`): one that
        // replays more is refused here, before a page is committed or a
        // row written.
        let replayed: u32 = rs_reads
            .iter()
            .map(|reads| crate::run::replayed(reads))
            .sum();
        if replayed > self.budget.max_tokens {
            return Err(Fault::Ceiling {
                what: "buffered recurrent rows a fire replays ahead of its own",
                need: u64::from(replayed),
                have: u64::from(self.budget.max_tokens),
            });
        }
        self.pools.commit_frame(demand, &kv_ranges, &window_pages)?;
        // The rs buffer pages the lanes name, seated before a row lands.
        self.pools.seat_rs(&rs_pages)?;
        // How the lanes treat their recurrent rows (the body's key) and the
        // tables its kernels address them through: where each lane's rows
        // land (its pages' slots) and which buffered rows it replays. An
        // arming fire states the treatment it records, its lanes landing
        // and replaying nothing.
        let rs_replays: Vec<u32> = rs_reads
            .iter()
            .map(|reads| crate::run::replayed(reads))
            .collect();
        let rs_lands = rs_moves.iter().any(|verb| !matches!(verb, RsMove::None));
        let rs_body = match self.arm_rs {
            armed if arming && armed.buffers() => armed,
            _ if rs_replays.iter().any(|replay| *replay > 0) => record::RsBody::Replays,
            _ if rs_lands => record::RsBody::Buffers,
            _ => record::RsBody::Folds,
        };
        let mut rs_lanes: Vec<kernels_cuda::runtime::channel::RsLane> = Vec::new();
        let mut rs_slots: Vec<u64> = Vec::new();
        let mut rs_tokens: Vec<u32> = Vec::new();
        if rs_body.buffers() {
            let buffers = self.buffers.as_ref().ok_or_else(|| Fault::Unbound {
                what: "the rs buffer of a fire whose lanes buffer recurrent rows, against a \
                       plan that declares no chunked recurrence to buffer"
                    .to_string(),
            })?;
            // Each lane's extended rows, its replayed then its own, lie past
            // every earlier lane's in the fire's extended run.
            let mut ext = 0u32;
            for (fire_lane, (verb, row)) in rs_moves.iter().zip(composition.lanes()).enumerate() {
                let begin = ext;
                ext += rs_replays[fire_lane] + row.rows;
                let RsMove::Scatter { pages, at, .. } = verb else {
                    rs_lanes.push(kernels_cuda::runtime::channel::RsLane {
                        ext: begin,
                        ..kernels_cuda::runtime::channel::RsLane::INERT
                    });
                    continue;
                };
                let replay_off = rs_tokens.len() as u32;
                for span in &rs_reads[fire_lane] {
                    rs_tokens.extend(span.clone());
                }
                rs_lanes.push(kernels_cuda::runtime::channel::RsLane {
                    at: *at,
                    replay: rs_replays[fire_lane],
                    replay_off,
                    page_off: rs_slots.len() as u32,
                    ext: begin,
                });
                for &page in pages.iter() {
                    rs_slots.push(buffers.slot(&self.pools, page)?);
                }
            }
        }
        // The records as the kernels read them: each row's pieces by the
        // ids of the slots they lie in, table after table.
        let device_slot_ids = self.pools.state_tables(&slot_ids)?;
        snapshots.slots = self.pools.snapshot_tables(&snapshots.slots)?;
        self.pools.fence_swaps(self.device.stream(), &kv_ranges)?;
        super::btrace::mark("commit_frame");

        let indptr_host = kv::indptr(&seats)?;
        let table_refs: Vec<&[u32]> = tables.iter().map(std::convert::AsRef::as_ref).collect();
        let mut geometries = (0..self.page_tables)
            .map(|table| match self.pools.geometry_window(table) {
                Some(window) => {
                    let refs: Vec<&[u32]> = window_tables
                        .get(window)
                        .map(|tables| tables.iter().map(Vec::as_slice).collect())
                        .unwrap_or_default();
                    kv::geometry_with(&paging, &seats, &refs)
                }
                None => kv::geometry_with(&paging, &seats, &table_refs),
            })
            .collect::<Result<Vec<_>>>()?;
        super::btrace::mark("page_arith");
        if writes.iter().any(Option::is_some) {
            let lane_of = |row: usize| {
                composition
                    .lanes()
                    .iter()
                    .position(|lane| {
                        (lane.row_offset as usize..(lane.row_offset + lane.rows) as usize)
                            .contains(&row)
                    })
                    .unwrap_or(0)
            };
            for (table, geometry) in geometries.iter_mut().enumerate() {
                let windowed = self.pools.geometry_window(table);
                for (row, stated) in writes.iter().enumerate() {
                    let Some((page, offset)) = *stated else {
                        continue;
                    };
                    let page = if let Some(window) = windowed {
                        let lane = lane_of(row);
                        let at = tables[lane]
                            .iter()
                            .position(|&held| i64::from(held) == i64::from(page));
                        match at.and_then(|at| window_tables.get(window)?.get(lane)?.get(at)) {
                            Some(&id) if id != 0 => id as i32,
                            _ => {
                                return Err(Fault::program(
                                    "serve::prepare",
                                    format!(
                                        "row {row} writes page {page}, which lane {lane}'s \
                                         windowed table holds no live page for"
                                    ),
                                ));
                            }
                        }
                    } else {
                        page
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
        let pages = geometries
            .first()
            .map_or(0, |geometry| geometry.indices.len() as u32);
        // The tables as the kernels read them: each page by the id of the
        // slot it is seated on in the state pool.
        let mut device_geometries = geometries.clone();
        for (table, geometry) in device_geometries.iter_mut().enumerate() {
            let list = self
                .pools
                .geometry_window(table)
                .map_or(0, |window| window as u8 + 1);
            // A composed windowed table names page ids already.
            if list > 0 && ringed.is_some() {
                continue;
            }
            self.pools.page_ids(list, &mut geometry.indices)?;
            self.pools.page_ids(list, &mut geometry.write_page)?;
        }

        super::btrace::mark("admit");
        let bucket = self
            .budget
            .buckets
            .iter()
            .position(|&rows| rows == composition.bucket())
            .unwrap_or(0) as u32;
        let token_axis = composition.axis(poem_ir::RowAxis::Tokens);
        // A fire of prompts alone pads to `engine.prefill_lanes`, not every lane the load seats.
        let max_lanes = match self.prefill_lanes {
            cap if cap > 0
                && token_axis
                    .classes
                    .present_in_order()
                    .all(|class| !self.decoding.contains(class as usize)) =>
            {
                self.budget.max_lanes.min(cap)
            }
            _ => self.budget.max_lanes,
        };
        let lane_ceiling = self.lane_ceiling().min(max_lanes);
        let patches = composition.axis(poem_ir::RowAxis::Patches);
        let key = record::BodyKey::of_axes(
            &token_axis.classes,
            token_axis.bucket,
            &self.decoding,
            lane_ceiling,
            &self.tiers,
            self.towered.then_some((&patches.classes, patches.bucket)),
        )
        .with_rs(rs_body);
        let copies_here = copies
            && key.classes.rungs().iter().all(|(class, _)| {
                !self.wide.contains(*class as usize)
            });
        let staged = crate::mask::stage(&masks, reads_masks)
            .map_err(|fault| fault.of_rows(composition.lanes()))?;
        let mask_injects: Vec<crate::inputs::MaskInject> = match &staged {
            Some(staged) => pending_mask_injects
                .iter()
                .map(|&(at, src, bits)| crate::inputs::MaskInject {
                    dst_off: staged.indptr[at] as u64,
                    src,
                    bits,
                })
                .collect(),
            None => Vec::new(),
        };
        super::btrace::mark("mask");

        let class_tables = [
            composition.table(poem_ir::RowAxis::Tokens),
            composition.table(poem_ir::RowAxis::Patches),
            composition.table(poem_ir::RowAxis::Voxels),
        ];
        let held = self.windows_memo.iter().position(|memo| {
            memo.bucket == bucket
                && memo.copies == copies_here
                && memo.indptr_host == indptr_host
                && memo.tables[0] == *class_tables[0]
                && memo.tables[1] == *class_tables[1]
                && memo.tables[2] == *class_tables[2]
        });
        let (mut windows, boundaries) = match held {
            Some(at) => {
                let memo = &self.windows_memo[at];
                (memo.windows.clone(), memo.packed.clone())
            }
            None => {
                let windows = Windows::of(
                    &self.trace,
                    &self.compiled,
                    poem_ir::PerAxis::new([class_tables[0], class_tables[1], class_tables[2]]),
                    &indptr_host,
                    crate::window::Copies {
                        bucket,
                        enabled: copies_here,
                        page_tables: &device_geometries,
                        mask: staged
                            .as_ref()
                            .map(|staged| (&staged.indptr[..], &staged.origins[..])),
                    },
                    self.inputs.window_slots(),
                )?;
                let packed = windows.packed();
                if windows.copied() == 0 {
                    if self.windows_memo.len() >= WINDOWS_MEMO {
                        self.windows_memo.remove(0);
                    }
                    self.windows_memo.push(WindowsMemo {
                        tables: [
                            class_tables[0].clone(),
                            class_tables[1].clone(),
                            class_tables[2].clone(),
                        ],
                        indptr_host: indptr_host.clone(),
                        bucket,
                        copies: copies_here,
                        windows: windows.clone(),
                        packed: packed.clone(),
                    });
                }
                (windows, packed)
            }
        };
        super::btrace::mark("windows_of");


        let totals = poem_ir::PerAxis::from_fn(|axis| composition.axis(axis).rows);
        // The widened table, as `segmentation` holds it: what a body was
        // recorded under is what it replays under.
        let key = key.with_islands(&record::widen(
            &self.compiled,
            &windows.admits_axes(totals, &self.shifted, &self.lane_shifted, self.staging()),
        ));
        let ladder = key.classes.clone();
        let patch_ladder = key.patch.as_ref().map(|axis| axis.classes.clone());
        let records_bodies = self.records_bodies();
        let (admits, world): (std::sync::Arc<[crate::window::Admit]>, bool) = if records_bodies {
            self.segmentation(&key, &windows, totals, copies_here)
        } else {
            (Vec::new().into(), true)
        };
        super::btrace::mark("segmentation");
        // A tapped fire is walked, so the walk can mark where each layer
        // ends on the stream.
        let snapshots_fit = snapshots.rows.len() <= crate::store::rs::SNAPSHOTS as usize;
        let bodied = records_bodies
            && !tapped
            && snapshots_fit
            && Self::keyable_units(&self.compiled)
            && world
            && !self.cache.body_refused(&key)
            && self.cuttable(&key, admits.as_ref());
        super::btrace::mark("cuttable");
        // A class reading out readout rows reads its lanes' last rows; a
        // drafter's verify (a class no plain request lands) reads one out a
        // row it checks, up to the readouts a lane takes; one reading a row
        // output reads every row of its rung, as the eager ceiling above
        // lets it. Every readout names one of the fire's rows,
        // so the rows read out never pass its bucket: the rungs summed over
        // the classes do (a mixed bucket-256 fire's two classes rung 256
        // each), and the carve a fire of the bucket takes (the forked arena,
        // an activation rung) holds no more readout rows than its tokens.
        // An unlisted ladder (a class no plain request lands, as a DiT's) is
        // the whole-bucket body: every class present at the bucket.
        let whole: Vec<(u32, u32)>;
        let rungs = if ladder.rungs().is_empty() {
            whole = token_axis
                .classes
                .present_in_order()
                .map(|class| (class, token_axis.bucket))
                .collect();
            whole.as_slice()
        } else {
            ladder.rungs()
        };
        let per_lane = self.budget.readouts_per_lane.max(1);
        let reads: Vec<(BodyReads, u32)> = rungs
            .iter()
            .map(|&(class, rung)| {
                let reads = match self.exports.readback(class as usize) {
                    Some(out) if !out.decl.readout => BodyReads::Rows,
                    None if !self.exports.reads_readout_rows() => BodyReads::Rows,
                    _ if !self.landing[class as usize]
                        .iter()
                        .any(poem_ir::Request::plain) =>
                    {
                        BodyReads::Verify
                    }
                    _ => BodyReads::Last,
                };
                (reads, rung)
            })
            .collect();
        let BodyReadouts {
            last: last_rows,
            verified,
            every: every_row,
            ceiling: readout_ceiling,
        } = body_readouts(
            &reads,
            lane_ceiling,
            max_lanes,
            per_lane,
            u32::try_from(poem_compiler::arena::readouts_ceiling(&self.budget)).unwrap_or(u32::MAX),
            composition.bucket(),
        );
        let readouts_fit = readout_rows.len() <= readout_ceiling as usize;
        let lanes_fit = composition.lane_count() <= crate::window::lane_capacity(max_lanes);
        // Why a fire of a load that records bodies walks its plan, for the
        // warning it raises (`Bodies::eager_unbodied`).
        let unbodied =
            (records_bodies && !arming && !(bodied && readouts_fit && lanes_fit)).then(|| {
                let why = if tapped {
                    "it is tapped"
                } else if !snapshots_fit {
                    "it takes more snapshots than a body holds"
                } else if !Self::keyable_units(&self.compiled) {
                    "the plan's units key no body"
                } else if !world {
                    "its expert group is not whole"
                } else if self.cache.body_refused(&key) {
                    "its body was refused when it was captured"
                } else if !bodied {
                    "its admitted set cannot be cut to a body"
                } else if !readouts_fit {
                    "it reads out more rows than its bucket's body carries"
                } else {
                    "it has more lanes than a body's window table holds"
                };
                if bodied && !readouts_fit {
                    format!(
                        "{key}: {why} ({} rows read out over {} lanes; the body carries {}: \
                         {last_rows} last rows under {max_lanes} lanes, {verified} verified \
                         rows, {every_row} every row, lane ceiling {lane_ceiling})",
                        readout_rows.len(),
                        composition.lane_count(),
                        readout_ceiling,
                    )
                } else {
                    format!("{key}: {why}")
                }
            });
        let bodied = bodied && readouts_fit && lanes_fit;
        let readout_live = readout_rows.len() as u32;
        if bodied {
            readout_rows.resize(readout_ceiling as usize, 0);
        } else {
            // An eager fire's readout rows are carved to what its readout
            // launches over (`readout_extent`), the padded ones row 0.
            readout_rows.resize(readout_rows.len().max(readout_extent as usize), 0);
        }

        if arming {
            self.armed_body = bodied.then(|| key.clone());
        }

        // Every per-row table is carved to the rows the fire launches over.
        let carve_rows = launched;
        let mut lane_of_row: Vec<i32> = Vec::with_capacity(carve_rows as usize);
        for (fire_lane, row) in composition.lanes().iter().enumerate() {
            lane_of_row.extend(std::iter::repeat_n(fire_lane as i32, row.rows as usize));
        }
        lane_of_row.resize(carve_rows as usize, 0);
        if carve_rows > rows {
            tokens.resize(carve_rows as usize, 0);
            positions.resize(carve_rows as usize, 0);
            if !mrope_positions.is_empty() {
                mrope_positions.resize(carve_rows as usize * MROPE_COORDS, 0);
            }
            if any_adapter {
                adapter_routes.resize(carve_rows as usize, -1);
            }
            for geometry in geometries.iter_mut().chain(&mut device_geometries) {
                geometry.write_page.resize(carve_rows as usize, -1);
                geometry.write_offset.resize(carve_rows as usize, -1);
            }
        }

        let mut qo_absolute: Vec<i32> = Vec::new();
        let mut lane_carve = composition.lane_count();
        if bodied {
            let capacity = crate::window::lane_capacity(max_lanes);
            lane_carve = lane_carve.max(capacity);
            for geometry in geometries.iter_mut().chain(&mut device_geometries) {
                geometry.pad_to(capacity as usize);
            }
            qo_absolute = windows.qo_absolute_host().to_vec();
            kv::pad_indptr(&mut qo_absolute, capacity as usize);
            windows.stage_qo_absolute(capacity);
        }
        let geometries = geometries;

        super::btrace::mark("windows");
        let slot = self.inputs.claim()?;
        let packing_fires: Vec<crate::inputs::PackingFire<'_>> = packings
            .iter()
            .map(|packed| crate::inputs::PackingFire {
                group_indptr: &packed.group_indptr,
                lane_indptr: &packed.lane_indptr,
                permutation: &packed.permutation,
            })
            .collect();
        let staged_lens = self.inputs.write_host(
            &slot,
            &crate::inputs::Fire {
                tokens: &tokens,
                positions: &positions,
                windows: &boundaries,
                readout_rows: &readout_rows,
                readout_live,
                qo_absolute: &qo_absolute,
                live: if bodied { windows.live() } else { &[] },
                slot_ids: &slot_ids,
                state_ids: &device_slot_ids,
                page_tables: &device_geometries,
                mask: staged.as_ref(),
                adapter_routes: any_adapter.then_some(adapter_routes.as_slice()),
                live_rows: rows,
                lane_reach: lane_carve,
                lane_of_row: &lane_of_row,
                group_of_lane: &group_of_lane,
                packings: &packing_fires,
                rs: rs_body.buffers().then_some(crate::inputs::RsFire {
                    lanes: &rs_lanes,
                    pages: &rs_slots,
                    replay_tok: &rs_tokens,
                }),
            },
        )?;

        // A buffering fire's recurrences take its commit lengths and its
        // fold predicate whatever its lanes fold, so one body serves every
        // fold; its extended run is the widest bucket's, whatever its rows
        // and whatever its lanes replay.
        let rs_rows_ext = if rs_body == record::RsBody::Replays {
            crate::store::rs::Buffers::rows_ext(self.budget.max_tokens)
        } else {
            0
        };
        let placed = match geometries.as_slice() {
            [table] if !self.pools.windowed_geometry(0) => Some(table),
            _ => None,
        };
        let kv_reports: Vec<super::KvReport> = composition
            .lanes()
            .iter()
            .filter(|row| lanes[row.source as usize].reports != 0)
            .map(|row| {
                let fold = rs_folded[row.source as usize].or_else(|| {
                    envelope_of[row.source as usize].and_then(|(held, at)| {
                        let cells = resolved[held].fold.as_ref()?;
                        fold_cell(cells, at)
                    })
                });
                let report = super::KvReport {
                    tag: lanes[row.source as usize].reports,
                    first_row: row.row_offset,
                    tokens: Some(Vec::new()),
                    positions: Vec::new(),
                    pages: Vec::new(),
                    offsets: Vec::new(),
                    fold,
                    fold_at: fold_at[row.source as usize].filter(|_| fold.is_none()),
                    end: rs_ends[row.source as usize],
                };
                let Some(table) = placed else {
                    return report;
                };
                let unsigned = |values: &[i32]| values.iter().map(|&v| v as u32).collect();
                let span = row.row_offset as usize..(row.row_offset + row.rows) as usize;
                let fed = token_injects
                    .iter()
                    .any(|inject| inject.dst_off == u64::from(row.row_offset) * 4);
                super::KvReport {
                    tokens: (!fed).then(|| unsigned(&tokens[span.clone()])),
                    positions: unsigned(&positions[span.clone()]),
                    pages: unsigned(&table.write_page[span.clone()]),
                    offsets: unsigned(&table.write_offset[span]),
                    ..report
                }
            })
            .collect();
        Ok(Prepared {
            ring_work,
            slot: Some(slot),
            lengths: staged_lens,
            token_injects,
            mask_injects,
            kv_reports,
            bodied,
            unbodied,
            admits,
            islands: key.islands.clone(),
            ladder,
            lane_ceiling,
            patch_ladder,
            towered: self.towered,
            lanes,
            attachments,
            composition,
            readout_rows,
            readout_first,
            readout_count,
            descriptor,
            patch_payload,
            voxel_tables,
            patch_segments,
            patch_routes,
            patch_positions,
            patch_grids,
            mrope_positions,
            packings,
            group_masks,
            input_feeds,
            voxel_feeds,
            merge_lands,
            lane_carve,
            windows,
            seats,
            tables,
            kv_less_seats,
            geometries,
            pages,
            fresh,
            demand,
            rs: RsFire {
                write_state: rs_body.buffers()
                    || rs_moves
                        .iter()
                        .any(|verb| !matches!(verb, RsMove::Scatter { fold: 0, .. })),
                predicated: rs_body.buffers()
                    || attachments.iter().any(|attached| {
                        !attached.adapter_only && self.programs.prologue_runs(attached.instance)
                    }),
                truncates: rs_body.buffers(),
                body: rs_body,
                moves: rs_moves,
                lens: rs_lens,
                order: rs_order,
                rows_ext: rs_rows_ext,
                snapshots,
            },
        })
    }

    /// A fire's group masks, one a packing an attention stating a mask
    /// reads, in the load's order (`Shell::stated_feeds`;
    /// `Inputs::stage_group_masks` stages the `i`th in the `i`th stretch),
    /// where some lane's class reads them (`group_masked`): the masks are
    /// data its one body reads, every ragged attention's. A group no lane of
    /// which states a mask has the offset -1, which the kernel reads as
    /// "none" (every row of the group admitted, no bit read); a lane stating
    /// none in a group beside one stating a mask admits every row.
    fn group_masks_of(
        &self,
        lanes: &[super::lanes::Seated<'_>],
        composition: &poem_exec::fire::Composition,
        group_of_lane: &[i32],
        packings: &[poem_exec::fire::Packed],
        clips_of: &[Option<crate::voxels::Clips<'_>>],
    ) -> Result<
        Vec<(
            poem_ir::Selection,
            poem_ir::Selection,
            poem_exec::fire::GroupMask,
        )>,
    > {
        let stated: Vec<Option<poem_exec::fire::LaneMask>> = lanes
            .iter()
            .map(|seated| {
                seated
                    .mask
                    .filter(|_| seated.kv_less)
                    .map(|masking| poem_exec::fire::LaneMask {
                        rows: masking
                            .masks()
                            .iter()
                            .map(|mask| mask.admitted(0..mask.total).collect())
                            .collect(),
                    })
            })
            .collect();
        let reads = composition
            .lanes()
            .iter()
            .any(|row| self.group_masked.contains(row.class as usize));
        if !reads && stated.iter().all(Option::is_none) {
            return Ok(Vec::new());
        }
        if self.stated_feeds.is_empty() {
            return Err(Fault::program(
                "serve::mask",
                "a kv-less lane states an attention mask and this model states none on any \
                 attention: only a group-packed `attention.ragged` its text spells \
                 `stated_group_mask()` reads one",
            ));
        }
        // No lane states one: every group's offset says "none".
        let unmasked = || poem_exec::fire::GroupMask {
            bits: Vec::new(),
            indptr: std::iter::repeat_n(-1, poem_exec::fire::groups_of(group_of_lane) as usize)
                .chain([0])
                .collect(),
        };
        if stated.iter().all(Option::is_none) {
            return Ok(self
                .stated_feeds
                .iter()
                .map(|&(q, kv)| (q, kv, unmasked()))
                .collect());
        }
        let patch = self
            .voxels
            .as_ref()
            .and_then(|store| store.seat().token_patch);
        let mut guests = vec![poem_exec::fire::GuestRows::default(); lanes.len()];
        for row in composition.lanes() {
            let clips = clips_of
                .get(row.source as usize)
                .copied()
                .flatten()
                .map(|clips| clips.clips);
            guests[row.source as usize] =
                poem_exec::fire::guest_rows(row, clips, patch).map_err(poem_exec::Error::Fire)?;
        }
        let stated: Vec<poem_exec::fire::LaneMask> = stated
            .into_iter()
            .zip(&guests)
            .map(|(mask, guest)| {
                mask.unwrap_or_else(|| poem_exec::fire::LaneMask {
                    rows: vec![vec![0..u64::MAX]; guest.count as usize],
                })
            })
            .collect();
        let masks: Vec<Option<&poem_exec::fire::LaneMask>> = stated.iter().map(Some).collect();
        let packed = |select: poem_ir::Selection| {
            packings
                .iter()
                .find(|packed| packed.select == select)
                .ok_or_else(|| {
                    Fault::program(
                        "serve::mask",
                        format!("no packing of selection {select:?} was staged for this fire"),
                    )
                })
        };
        let mut staged = Vec::new();
        for &(q, kv) in &self.stated_feeds {
            // An attention none of the fire's lanes reach reads no row of
            // its stretch.
            let mask = poem_exec::fire::group_mask(
                composition.lanes(),
                group_of_lane,
                packed(q)?,
                packed(kv)?,
                &masks,
                &guests,
            )
            .map_err(poem_exec::Error::Fire)?
            .unwrap_or_else(unmasked);
            staged.push((q, kv, mask));
        }
        Ok(staged)
    }
}

#[cfg(test)]
mod tests {
    use super::{BodyReads, body_readouts, fold_cell, fold_held, readout_extent};

    /// A mixed fire of two classes no plain request lands (a drafter's
    /// verifies) at the load's lanes reads out no more rows than the arena
    /// lays the readout plane for: the lanes the two classes share are
    /// counted once, not a class each (the size-only walk of bucket 256
    /// carved 64 rows into qwen35-tiny's 32-row logits slot).
    #[test]
    fn a_mixed_body_reads_out_no_more_than_the_readout_plane_holds() {
        const MAX_LANES: u32 = 2;
        const PER_LANE: u32 = 16;
        let plane = MAX_LANES * PER_LANE;
        let verifies = [(BodyReads::Verify, 256), (BodyReads::Verify, 256)];
        let body = body_readouts(&verifies, MAX_LANES, MAX_LANES, PER_LANE, plane, 256);
        assert_eq!(body.ceiling, plane, "{body:?}");
        // Last rows beside the verifies share the plane too.
        let mixed = [(BodyReads::Last, 256), (BodyReads::Verify, 256)];
        let body = body_readouts(&mixed, MAX_LANES, MAX_LANES, PER_LANE, plane, 256);
        assert!(body.ceiling <= plane, "{body:?}");
        // Rows off a tokens-shaped plane are no readout plane's: bounded by
        // the bucket alone.
        let rows = [(BodyReads::Verify, 256), (BodyReads::Rows, 64)];
        let body = body_readouts(&rows, MAX_LANES, MAX_LANES, PER_LANE, plane, 256);
        assert_eq!(body.ceiling, plane + 64, "{body:?}");
        let rows = [(BodyReads::Rows, 256), (BodyReads::Last, 2)];
        let body = body_readouts(&rows, MAX_LANES, MAX_LANES, PER_LANE, plane, 256);
        assert_eq!(body.ceiling, 256, "{body:?}");
    }

    #[test]
    fn a_fold_cell_states_every_lanes_fold_or_a_fold_a_lane() {
        assert_eq!(fold_cell(&[3], 0), Some(3));
        assert_eq!(fold_cell(&[3], 5), Some(3));
        assert_eq!(fold_cell(&[3, 4], 1), Some(4));
        assert_eq!(fold_cell(&[3, 4], 2), None);
        assert_eq!(fold_cell(&[], 0), None);
    }

    #[test]
    fn a_readout_as_tall_as_the_fire_holds_the_bucket() {
        assert_eq!(readout_extent(133, 256, 133), 256);
        assert_eq!(readout_extent(133, 256, 5), 5);
        assert_eq!(readout_extent(133, 133, 133), 133);
        assert_eq!(readout_extent(0, 64, 0), 0);
    }

    /// A device-counted fold left record 3's state at slot 92; the record
    /// then went to a new working set, whose prefill folds its own rows. Its
    /// buffered fires read the state where they state it, not at 92.
    #[test]
    fn a_folding_lane_forgets_the_fold_a_record_was_left_ahead_at() {
        let mut ahead = std::collections::HashMap::from([(3u32, 92u32), (5, 40)]);
        // The working set the fold was counted for still reads it ahead.
        assert_eq!(fold_held(&mut ahead, 3, Some(60), false), 92);
        // A new working set's prefill on the record.
        fold_held(&mut ahead, 3, None, false);
        assert_eq!(fold_held(&mut ahead, 3, Some(28), false), 28);
        // A fresh buffered lane forgets it too, and a settled fold is past it.
        assert_eq!(fold_held(&mut ahead, 5, Some(12), true), 12);
        assert!(ahead.is_empty());
        ahead.insert(7, 40);
        assert_eq!(fold_held(&mut ahead, 7, Some(48), false), 48);
        assert!(ahead.is_empty());
    }
}
