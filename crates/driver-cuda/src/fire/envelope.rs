//! Fills in the geometry a decode step's wire tables don't carry — token,
//! position and KV length, left on channels by the previous fire — and
//! translates working-set pages to physical ones. Pages derive from positions,
//! not a port, but `Pages`/`PageIndptr`/`WSlot`/`WOff` must still be drained or
//! their rings wedge the epilogue. Both branches apply `kv_translation` with the
//! same arithmetic, or two requests at logical page 0 collide.

use driver::tensor_ir::registry::Port;
use driver_api::plan::LaunchPlan;
use driver_api::submission::{FrameSubmission, StepSubmission};

use crate::program::channel::Rings;
use crate::program::session::Session;
use crate::serve::state::{ChannelState, InstanceEntry};

/// What resolving a step's geometry produced.
pub(crate) enum Composed {
    /// Nothing changed: the step's own tables are the fire.
    Wire,
    /// The step, with every member's geometry known and every page physical.
    Ready(Box<StepSubmission>),
    /// A descriptor channel holds nothing yet, so the program that fills it has
    /// not run — distinct from a refusal since the remedy differs: fire the
    /// producer first.
    Early {
        /// The instance whose channel was empty.
        instance: u64,
        /// Its dense channel index.
        channel: u32,
        /// The port that named it, so the message says which descriptor is missing.
        port: Port,
    },
}

/// Why a step could not be composed.
pub(crate) struct Refused(pub String);

/// The geometry class of one member of a step. Absent `sub_batch_indptr` is
/// `Host`; a member the table omits is `None`, not a guessed `Host` — that would
/// route a device-resolved member down the wire-plan path into empty tables.
fn class_of(step: &StepSubmission, member: usize) -> Option<u32> {
    if step.sub_batch_indptr.is_empty() {
        return Some(driver_api::local::PIE_GEOMETRY_CLASS_HOST);
    }
    for (b, window) in step.sub_batch_indptr.windows(2).enumerate() {
        if (window[0] as usize..window[1] as usize).contains(&member) {
            return step.sub_batch_class.get(b).copied();
        }
    }
    None
}

/// One roster row's slice of the frame's page translation.
fn translation(frame: &FrameSubmission, row: u32) -> &[u32] {
    let row = row as usize;
    match (
        frame.kv_translation_indptr.get(row),
        frame.kv_translation_indptr.get(row + 1),
    ) {
        (Some(&lo), Some(&hi)) if hi >= lo && hi as usize <= frame.kv_translation.len() => {
            &frame.kv_translation[lo as usize..hi as usize]
        }
        _ => &[],
    }
}

/// One working-set page as the physical page it is placed in. An empty
/// translation means the frame states none, so pages are already physical.
fn physical(segment: &[u32], logical: u32) -> Result<u32, Refused> {
    if segment.is_empty() {
        return Ok(logical);
    }
    segment
        .get(logical as usize)
        .copied()
        .filter(|&page| page != u32::MAX)
        .ok_or_else(|| {
            Refused(format!(
                "this fire names working-set page {logical}, which the frame's translation \
                 of {} page(s) does not place",
                segment.len()
            ))
        })
}

/// Valid tokens in the last KV page of a span of `len`. An exact multiple fills
/// its last page rather than starting an empty one (the `- 1` / `+ 1`). Must
/// match `driver::last_page_len` and the engine's `geometry::last_page_len`, or
/// a fire attends a page short.
fn last_page_len(len: u32, page: u32) -> u32 {
    if page == 0 || len == 0 {
        0
    } else {
        ((len - 1) % page) + 1
    }
}

/// A folded const port's cell as `u32` lanes. A bit reinterpretation for `i32`,
/// not a numeric conversion, so the in-band `-1` skip stays `0xffff_ffff` rather
/// than saturating.
fn value_lanes(value: &driver::Value) -> Vec<u32> {
    match value {
        driver::Value::U32(v) => v.clone(),
        driver::Value::I32(v) => v.iter().map(|&x| x as u32).collect(),
        driver::Value::F32(v) => v.iter().map(|&x| x.to_bits()).collect(),
        driver::Value::Bool(v) => v.iter().map(|&b| u32::from(b)).collect(),
    }
}

/// The u32 lanes of a native device cell.
fn lanes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|four| u32::from_le_bytes(four.try_into().expect("four bytes")))
        .collect()
}

/// Everything one member's compose reads out of the shell.
pub(crate) struct Sites<'a> {
    pub instances: &'a std::collections::BTreeMap<u64, InstanceEntry>,
    pub channels: &'a std::collections::BTreeMap<u64, ChannelState>,
    pub plans: &'a std::collections::BTreeMap<u64, driver::ExecPlan>,
    pub sessions: &'a mut std::collections::BTreeMap<u64, Session>,
    pub rings: &'a mut Rings,
}

/// One member's ports, as this compose needs them.
struct Ports {
    /// Dense channel per bound non-const port, by `Port` tag.
    channel: std::collections::BTreeMap<u8, u32>,
    /// The folded value of each const port.
    constant: std::collections::BTreeMap<u8, Vec<u32>>,
}

impl Ports {
    fn of(plan: &driver::ExecPlan) -> Self {
        let mut channel = std::collections::BTreeMap::new();
        let mut constant = std::collections::BTreeMap::new();
        for binding in &plan.package.ports {
            if binding.is_const {
                if let Some(folded) = plan.const_ports.iter().find(|c| c.port == binding.port) {
                    constant.insert(binding.port, value_lanes(&folded.value));
                }
            } else {
                channel.insert(binding.port, binding.channel);
            }
        }
        Self { channel, constant }
    }

    fn dense(&self, port: Port) -> Option<u32> {
        self.channel.get(&(port as u8)).copied()
    }

    fn folded(&self, port: Port) -> Option<&[u32]> {
        self.constant.get(&(port as u8)).map(Vec::as_slice)
    }
}

/// Drain every host-writer descriptor port a HOST-geometry member holds.
///
/// A host-written channel's ring has exactly one consumer on this driver:
/// [`Session::pull_channels`], which copies the engine's mirror cells into the
/// device ring and advances the mirror's `head` as it goes. That advance is the
/// ONLY thing that moves `head` -- this driver never writes the binding's head
/// word from anywhere else -- and `head` is what the engine's writer checks
/// before it stages a cell:
///
/// ```text
///     if self.writer_tail - head >= capacity { return Err(ChannelError::Full) }
/// ```
///
/// So a host writer that is never pulled gets exactly `capacity` puts and then
/// `Full` forever. The guest waits on a cell that cannot be staged, the driver
/// waits on a fire that is never submitted, and NEITHER SIDE SAYS ANYTHING. It
/// reads as a hang with an idle GPU.
///
/// [`compose`] used to pull only on the device-resolved branch, so a pass whose
/// geometry the ENGINE resolves -- the common case -- wedged the moment it fed
/// a descriptor port from the host per step. Fixtures that carry their token
/// on-device never noticed; `contrastive-decoding` and
/// `classifier-free-guidance`, which host-put the next token each step, died
/// after exactly `capacity` tokens. At the default capacity of 1 that is the
/// SECOND decode step.
///
/// Called for its effect on the cursors; the cells land in the ring the launch
/// path will not read for a host-resolved member, which costs one copy and
/// keeps one drain rather than two.
fn drain_host_writers(
    instances: &std::collections::BTreeMap<u64, InstanceEntry>,
    channels: &std::collections::BTreeMap<u64, ChannelState>,
    plans: &std::collections::BTreeMap<u64, driver::ExecPlan>,
    sessions: &mut std::collections::BTreeMap<u64, Session>,
    rings: &mut Rings,
    id: u64,
    stream: crate::device::StreamRef<'_>,
) -> Result<(), Refused> {
    let Some(instance) = instances.get(&id) else {
        return Ok(());
    };
    let Some(exec) = plans.get(&instance.program_id) else {
        return Ok(());
    };
    let Some(session) = sessions.get_mut(&id) else {
        return Ok(());
    };
    let mut host: Vec<crate::program::channel::HostChannel> =
        Vec::with_capacity(instance.channel_ids.len());
    for channel in &instance.channel_ids {
        let Some(state) = channels.get(channel) else {
            return Ok(());
        };
        host.push(state.host_plane());
    }
    // CONSUMING ports only, which is the whole of the care this needs.
    //
    // A port the fire consumes takes a fresh cell per fire, so its ring has to
    // advance or the writer wedges -- that is what this function is for. A
    // LATEST-VALUE port does not: `KvLen`, `Pages`, `PageIndptr`, `EmbedIndptr`
    // and `AttnMask` keep one committed cell that the guest REPLACES with
    // `Channel::set` rather than re-putting, precisely because nothing on the
    // device side would ever move their head.
    //
    // Draining those breaks them. `set` refuses with `Empty` unless the cell it
    // is replacing is still committed (`committed_tail <= head` in
    // `pipeline::channel`), so a drain that takes it turns the guest's next
    // `set` into an error -- and this function drained EVERY bound port when it
    // was written, which is how `tart-masked` went from a deadlock to a
    // `no cell available` the moment the seed cursors were fixed underneath it.
    let ports = Ports::of(exec);
    let bound: Vec<u32> = Port::ALL
        .iter()
        .filter(|port| port.consumes())
        .filter_map(|&port| ports.dense(port))
        .collect();
    session
        .pull_channels(rings, &mut host, &bound, stream)
        .map_err(|why| {
            Refused(format!(
                "instance {id} cannot drain its host writers: {why}"
            ))
        })?;
    Ok(())
}

/// Resolve every device-class member of `step` and translate every member's
/// pages, or say why not. `Composed::Wire` when nothing needed either.
///
/// Returns [`Refused`] naming what disagreed — the alternative being attention
/// over pages the fire did not name.
pub(crate) fn compose(
    sites: Sites<'_>,
    frame: &FrameSubmission,
    step: &StepSubmission,
    page: u32,
    stream: crate::device::StreamRef<'_>,
) -> Result<Composed, Refused> {
    let Sites {
        instances,
        channels,
        plans,
        sessions,
        rings,
    } = sites;
    let wire_rows = step.plan.qo_indptr.len().saturating_sub(1);
    let devices = (0..step.roster_rows.len())
        .map(|m| class_of(step, m))
        .collect::<Option<Vec<u32>>>()
        .ok_or_else(|| {
            Refused(format!(
                "a member of this step is in no sub-batch: {} boundaries covering members \
                 0..{}, and {} class(es)",
                step.sub_batch_indptr.len(),
                step.sub_batch_indptr.last().copied().unwrap_or(0),
                step.sub_batch_class.len()
            ))
        })?;
    let any_device = devices
        .iter()
        .any(|&c| c != driver_api::local::PIE_GEOMETRY_CLASS_HOST);
    let any_translation = !frame.kv_translation.is_empty();
    if !any_device && !any_translation {
        // Not before draining: see `drain_host_writers`. Every member here is
        // host-resolved, and returning `Wire` without touching their rings is
        // what wedged a per-step host writer after `capacity` puts.
        for &row in &step.roster_rows {
            if let Some(&id) = frame.instance_ids.get(row as usize) {
                drain_host_writers(instances, channels, plans, sessions, rings, id, stream)?;
            }
        }
        return Ok(Composed::Wire);
    }

    crate::fire::launch::sg_trace(|| {
        format!(
            "  compose: roster={:?} classes={:?} xlat={:?} indptr={:?}",
            step.roster_rows.as_slice(),
            devices,
            frame.kv_translation.as_slice(),
            frame.kv_translation_indptr.as_slice(),
        )
    });
    let mut plan = step.plan.clone();
    let (mut tokens, mut positions, mut qo) = (Vec::new(), Vec::new(), vec![0u32]);
    let (mut pages, mut page_indptr, mut lens) = (Vec::new(), vec![0u32], Vec::new());
    let (mut samples, mut sample_indptr) = (Vec::new(), vec![0u32]);

    for (member, &row) in step.roster_rows.iter().enumerate() {
        let segment = translation(frame, row);
        if devices[member] == driver_api::local::PIE_GEOMETRY_CLASS_HOST {
            let (first, last) = member_requests(&step.program_row_indptr, member, wire_rows)
                .ok_or_else(|| {
                    Refused(format!(
                        "member {member}'s rows are not placed by program_row_indptr {:?} \
                         among {wire_rows} request(s)",
                        step.program_row_indptr.as_slice()
                    ))
                })?;
            for r in first..last {
                let (lo, hi) = (
                    step.plan.qo_indptr[r] as usize,
                    step.plan.qo_indptr[r + 1] as usize,
                );
                if lo > hi || hi > step.plan.token_ids.len() || hi > step.plan.position_ids.len() {
                    return Err(Refused(format!(
                        "request {r} spans rows {lo}..{hi} of {} token(s)",
                        step.plan.token_ids.len()
                    )));
                }
                tokens.extend_from_slice(&step.plan.token_ids[lo..hi]);
                positions.extend_from_slice(&step.plan.position_ids[lo..hi]);
                qo.push(tokens.len() as u32);
                for &r0 in sampling_rows(&step.plan, r) {
                    samples.push(r0);
                }
                sample_indptr.push(samples.len() as u32);
                let (plo, phi) = match (
                    step.plan.kv_page_indptr.get(r),
                    step.plan.kv_page_indptr.get(r + 1),
                ) {
                    (Some(&plo), Some(&phi)) if phi >= plo => (plo as usize, phi as usize),
                    _ => {
                        return Err(Refused(format!(
                            "request {r} has no page span in a CSR of {} entries",
                            step.plan.kv_page_indptr.len()
                        )));
                    }
                };
                if phi > step.plan.kv_page_indices.len() {
                    return Err(Refused(format!(
                        "request {r} spans pages {plo}..{phi} of {} page indices",
                        step.plan.kv_page_indices.len()
                    )));
                }
                for &logical in &step.plan.kv_page_indices[plo..phi] {
                    pages.push(physical(segment, logical)?);
                }
                page_indptr.push(pages.len() as u32);
                lens.push(step.plan.kv_last_page_lens.get(r).copied().unwrap_or(0));
            }
            // The engine resolved this member's geometry, so nothing above read
            // its rings -- but the guest still WROTE them, and only a pull moves
            // the mirror's head. See `drain_host_writers`.
            if let Some(&id) = frame.instance_ids.get(row as usize) {
                drain_host_writers(instances, channels, plans, sessions, rings, id, stream)?;
            }
            continue;
        }

        // A device-resolved member: read the three ports off its rings.
        let id = *frame.instance_ids.get(row as usize).ok_or_else(|| {
            Refused(format!(
                "step member {member} names roster row {row} of {}",
                frame.instance_ids.len()
            ))
        })?;
        if devices[member] != driver_api::local::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE {
            return Err(Refused(format!(
                "instance {id} is geometry class {}, and this driver resolves only the \
                 decode envelope (class {}): the pool-owned device-geometry class needs \
                 the program's own pages, page CSR, write descriptor and mask cell, none \
                 of which is read here",
                devices[member],
                driver_api::local::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE
            )));
        }
        let instance = instances
            .get(&id)
            .ok_or_else(|| Refused(format!("instance {id} is not bound in this driver")))?;
        let exec = plans.get(&instance.program_id).ok_or_else(|| {
            Refused(format!(
                "instance {id} names program {} and no plan was adopted for it",
                instance.program_id
            ))
        })?;
        let ports = Ports::of(exec);
        let session = sessions
            .get_mut(&id)
            .ok_or_else(|| Refused(format!("instance {id} has no ringed channels")))?;

        // Seeds first: a decode's `Positions`/`KvLen` are host seeds on the
        // first fire (device-carried after) and must reach the ring before the
        // forward reads it, or the first decode reads a zeroed cell.
        let mut host: Vec<crate::program::channel::HostChannel> =
            Vec::with_capacity(instance.channel_ids.len());
        for channel in &instance.channel_ids {
            let state = channels.get(channel).ok_or_else(|| {
                Refused(format!(
                    "instance {id} names channel {channel}, which this driver does not hold"
                ))
            })?;
            host.push(state.host_plane());
        }
        let bound: Vec<u32> = ports.channel.values().copied().collect();
        session
            .pull_channels(rings, &mut host, &bound, stream)
            .map_err(|why| Refused(format!("instance {id} cannot be seeded: {why}")))?;

        let cursors = rings
            .cursors(stream)
            .map_err(|why| Refused(format!("instance {id} cursors: {why}")))?;
        // Peek at the committed front; the take happens once below after every
        // port is read, or a second port on the same channel reads the next cell.
        let peek = |port: Port| -> Result<Option<Vec<u32>>, Composed> {
            if let Some(folded) = ports.folded(port) {
                return Ok(Some(folded.to_vec()));
            }
            let Some(dense) = ports.dense(port) else {
                return Ok(None);
            };
            let early = Composed::Early {
                instance: id,
                channel: dense,
                port,
            };
            let Some(slot) = session.slot(dense as usize) else {
                return Err(early);
            };
            let Some(cursor) = cursors.get(slot as usize) else {
                return Err(early);
            };
            if !cursor.is_readable() {
                return Err(early);
            }
            match rings.read_cell(slot as usize, cursor.head, stream) {
                Ok(bytes) => Ok(Some(lanes(&bytes))),
                Err(_) => Err(early),
            }
        };
        let token_ids = match peek(Port::EmbedTokens) {
            Ok(Some(v)) => v,
            Ok(None) => {
                return Err(Refused(format!(
                    "instance {id} is a decode envelope and binds no EmbedTokens port"
                )));
            }
            Err(early) => return Ok(early),
        };
        let count = token_ids.len() as u32;
        let qo_indptr = match peek(Port::EmbedIndptr) {
            Ok(Some(v)) => v,
            Ok(None) => vec![0, count],
            Err(early) => return Ok(early),
        };
        let position_ids = match peek(Port::Positions) {
            Ok(Some(v)) => v,
            Ok(None) => (0..count).collect(),
            Err(early) => return Ok(early),
        };
        let kv_len = match peek(Port::KvLen) {
            Ok(Some(v)) => v,
            Ok(None) => Vec::new(),
            Err(early) => return Ok(early),
        };
        let readout = match peek(Port::Readout) {
            Ok(v) => v,
            Err(early) => return Ok(early),
        };

        if qo_indptr.len() < 2
            || qo_indptr[0] != 0
            || qo_indptr.last().copied() != Some(count)
            || qo_indptr.windows(2).any(|w| w[1] < w[0])
        {
            return Err(Refused(format!(
                "instance {id} resolves a token CSR {qo_indptr:?} over {count} token(s)"
            )));
        }
        if position_ids.len() != token_ids.len() {
            return Err(Refused(format!(
                "instance {id} resolves {} position(s) for {} token(s)",
                position_ids.len(),
                token_ids.len()
            )));
        }

        for (r, window) in qo_indptr.windows(2).enumerate() {
            let (lo, hi) = (window[0] as usize, window[1] as usize);
            tokens.extend_from_slice(&token_ids[lo..hi]);
            positions.extend_from_slice(&position_ids[lo..hi]);
            qo.push(tokens.len() as u32);
            match readout.as_deref() {
                // The rows the program named, in its own per-lane numbering.
                Some(rows) => {
                    for &r0 in rows {
                        if (r0 as usize) < hi - lo {
                            samples.push(r0);
                        }
                    }
                }
                // No readout port means each lane reads its own last row — the decode case.
                None => samples.push((hi - lo).saturating_sub(1) as u32),
            }
            sample_indptr.push(samples.len() as u32);

            // The span comes from positions, not `kv_len`, which is only
            // checked: trusting `kv_len` could serve a page this fire just
            // filled, and a mismatch means the epilogue lost a step.
            let last = position_ids[lo..hi].iter().copied().max();
            let len = last.map_or(0, |p| p.saturating_add(1));
            if let Some(&stated) = kv_len.get(r)
                && stated != len
            {
                return Err(Refused(format!(
                    "instance {id} lane {r} states a KV length of {stated} and its last \
                     position is {}, which is a span of {len}",
                    last.unwrap_or(0)
                )));
            }
            let live = if page == 0 { 0 } else { len.div_ceil(page) };
            for logical in 0..live {
                pages.push(physical(segment, logical)?);
            }
            page_indptr.push(pages.len() as u32);
            lens.push(last_page_len(len, page));
        }

        // Advance every consuming port the program bound, not just those read
        // above: `WSlot`/`WOff` are unread here but must drain, or their rings
        // wedge the epilogue.
        for &port in Port::ALL {
            if !port.consumes() {
                continue;
            }
            let Some(dense) = ports.dense(port) else {
                continue;
            };
            let Some(slot) = session.slot(dense as usize) else {
                continue;
            };
            rings.consume_front(slot as usize, stream).map_err(|why| {
                Refused(format!(
                    "instance {id} cannot consume channel {dense}: {why}"
                ))
            })?;
        }
    }

    plan.token_ids = tokens;
    plan.position_ids = positions;
    plan.qo_indptr = qo;
    plan.kv_page_indices = pages;
    plan.kv_page_indptr = page_indptr;
    plan.kv_last_page_lens = lens;
    plan.sampling_indices = samples;
    plan.sampling_indptr = sample_indptr;
    let mut out = step.clone();
    out.plan = plan;
    Ok(Composed::Ready(Box::new(out)))
}

/// The rows request `r` names in the wire plan's sampling table. Empty for a
/// plan with no table: every request reads out its own last row.
fn sampling_rows(plan: &LaunchPlan, r: usize) -> &[u32] {
    let (Some(&lo), Some(&hi)) = (plan.sampling_indptr.get(r), plan.sampling_indptr.get(r + 1))
    else {
        return &[];
    };
    if hi < lo {
        return &[];
    }
    plan.sampling_indices
        .get(lo as usize..hi as usize)
        .unwrap_or(&[])
}

/// Which wire requests one member of a step owns. An absent CSR gives the whole
/// step to the member (the single-member case); a CSR that exists but omits this
/// member is `None`, not that fallback — else one member gets another's rows.
pub(crate) fn member_requests(
    program_row_indptr: &[u32],
    member: usize,
    requests: usize,
) -> Option<(usize, usize)> {
    if program_row_indptr.len() < 2 {
        return Some((0, requests));
    }
    match (
        program_row_indptr.get(member),
        program_row_indptr.get(member + 1),
    ) {
        (Some(&s), Some(&e)) if e >= s && e as usize <= requests => Some((s as usize, e as usize)),
        _ => None,
    }
}
