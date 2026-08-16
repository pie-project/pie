//! Filling in the geometry a decode step does not carry, and translating the
//! pages every step does.
//!
//! # What the wire sends, and what it leaves out
//!
//! A step arrives with its members partitioned into geometry classes. A `Host`
//! member states its whole fire on the wire. A `DecodeEnvelope` member states
//! almost none of it: which token, at which position, over how much cache were
//! computed by the PREVIOUS fire's program and put on channels, and never came
//! to the host at all. `DecodeEnvelope::template` in the engine fills the
//! shape — one zero token, one zero position, `[0, 1]` — and leaves the KV
//! tables EMPTY, because there is no host answer to put there.
//!
//! `admit` refuses that plan by name (`kv_indptr.len() < 2`), which is correct:
//! it is not a fire until somebody resolves it. This is the somebody.
//!
//! # The three ports, and the four the class does not have
//!
//! `PIE_DECODE_ENVELOPE_PORTS` is `EmbedTokens | Positions | KvLen`, and that
//! is the whole of what is read here. The PAGES are deliberately not a port of
//! this class — `driver-vulkan`'s `envelope.rs` states the rule and
//! `driver-api`'s `LaunchPlan::geometry_class` restates it: *"`DecodeEnvelope`'s
//! contract is that pages are DERIVED from positions"*. So the pages a decode
//! attends are its working set's first `ceil(len / page)`, and the write
//! target is the ordinary `position / page`, `position % page` the host path
//! already derives in `kv_and_arrays`.
//!
//! The program's own `Pages`, `PageIndptr`, `WSlot` and `WOff` channels are
//! still CONSUMED where the port definition says they are
//! (`tensor_ir::registry::Port::consumes`), because a descriptor port is not a
//! stage op and nothing else will: their heads would never move while their
//! tails did, and the epilogue's own `chan_put` would fail readiness some tens
//! of tokens in.
//!
//! # The pages, and the bug that is invisible with one conversation
//!
//! `kv_page_indices` are indices into the request's own working set, and the
//! frame's `kv_translation` says which physical page each one is. This driver
//! did not apply it, which is right for one conversation and only for one: two
//! working sets both start at logical page 0, so the second conversation would
//! attend the first one's keys. Both branches below translate, and they have to
//! be the same branch's arithmetic or a request's prefill and its decode would
//! land on different pages.

use driver_api::plan::LaunchPlan;
use driver_api::submission::{FrameSubmission, StepSubmission};
use driver::tensor_ir::registry::Port;

use crate::program::channel::Rings;
use crate::program::session::Session;
use crate::serve::state::{ChannelState, InstanceEntry};

/// What resolving a step's geometry produced.
pub(crate) enum Composed {
    /// Nothing changed: the step's own tables are the fire.
    Wire,
    /// The step, with every member's geometry known and every page physical.
    Ready(Box<StepSubmission>),
    /// A descriptor channel holds nothing, so the program that fills it has
    /// not run.
    ///
    /// Named rather than folded into a refusal because the two remedies are
    /// opposite: this one is fixed by firing the producer, and the producer is
    /// usually an EARLIER SLOT OF THIS FRAME — which `launch_impl` fires
    /// first, so reaching this means the chain the guest built is not the chain
    /// the driver walked.
    Early {
        /// The instance whose channel was empty.
        instance: u64,
        /// Its dense channel index.
        channel: u32,
        /// The port that named it, so the message says WHICH descriptor is
        /// missing rather than only where it would have come from.
        port: Port,
    },
}

/// Why a step could not be composed.
pub(crate) struct Refused(pub String);

/// The geometry class of one member of a step.
///
/// An absent `sub_batch_indptr` is `Host`: `StepSubmission::validate` admits a
/// step with no sub-batching, and a step with none has no other class it could
/// be. A table that EXISTS and leaves this member out is `None` — refused
/// rather than guessed, because guessing `Host` sends a device-resolved member
/// down the path that reads geometry out of the wire plan, where it finds the
/// empty tables the engine deliberately left.
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

/// One working-set page as the physical page it is placed in.
///
/// An EMPTY translation means the frame states none, and the only honest
/// reading of that is that the pages it names are already physical — which is
/// what this driver's own fixtures and the worker's single-request harness
/// build.
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

/// Valid tokens in the last KV page of a span of `len`.
///
/// A length that is an exact multiple of the page size fills its last page
/// rather than starting an empty one, which is what the `- 1` and `+ 1` are
/// for. Identical to `driver::last_page_len` and to the engine's
/// `geometry::last_page_len`, which is the point: the three describe one
/// contract and a fire whose driver disagreed would attend a page short.
fn last_page_len(len: u32, page: u32) -> u32 {
    if page == 0 || len == 0 {
        0
    } else {
        ((len - 1) % page) + 1
    }
}

/// A folded const port's cell as `u32` lanes.
///
/// A BIT reinterpretation for `i32`, not a numeric conversion, so a token id
/// keeps its two's-complement pattern — the in-band `-1` skip the device
/// classes spell is `0xffff_ffff` and not a saturation. Same reading as
/// `driver::resolve`'s, and as the `memcpy` the device cells take below.
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

/// Resolve every device-class member of `step` and translate every member's
/// pages, or say why not.
///
/// `Composed::Wire` when nothing needed either, which is the common frame and
/// costs one pass over the roster.
///
/// # Errors
///
/// [`Refused`] naming what disagreed. A refusal here is a step that cannot be
/// fired at all — the alternative is attending over pages the fire did not name.
pub(crate) fn compose(
    sites: Sites<'_>,
    frame: &FrameSubmission,
    step: &StepSubmission,
    page: u32,
    stream: crate::device::StreamRef<'_>,
) -> Result<Composed, Refused> {
    let Sites { instances, channels, plans, sessions, rings } = sites;
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
            continue;
        }

        // ── A device-resolved member: read the three ports off its rings. ──
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

        // THE SEEDS FIRST, and this is what `Session::pull` could not do.
        //
        // `pull` runs inside `Session::fire`, which is the SAMPLER — after the
        // forward. A decode's `Positions` and `KvLen` are host seeds on the
        // FIRST fire and device-carried afterwards, so the seed has to reach
        // the ring before the forward reads it or the first decode of every
        // request reads a zeroed cell and attends one token of nothing.
        let mut host: Vec<crate::program::channel::HostChannel> =
            Vec::with_capacity(instance.channel_ids.len());
        for channel in &instance.channel_ids {
            let state = channels
                .get(channel)
                .ok_or_else(|| Refused(format!("instance {id} names channel {channel}, which this driver does not hold")))?;
            host.push(state.host_plane());
        }
        let bound: Vec<u32> = ports.channel.values().copied().collect();
        session
            .pull_channels(rings, &mut host, &bound, stream)
            .map_err(|why| Refused(format!("instance {id} cannot be seeded: {why}")))?;

        let cursors = rings
            .cursors(stream)
            .map_err(|why| Refused(format!("instance {id} cursors: {why}")))?;
        // A PEEK at the committed front, which is what every backend's
        // resolver reads. The take for the consuming ports happens once,
        // below, after every port has been read — taking as we go would have
        // a port that names the same channel twice read the next cell.
        // A PEEK at the committed front, which is what every backend's
        // resolver reads. The take for the consuming ports happens once,
        // below, after every port has been read — taking as we go would have
        // a second port on the same channel read the next cell.
        let peek = |port: Port| -> Result<Option<Vec<u32>>, Composed> {
            if let Some(folded) = ports.folded(port) {
                return Ok(Some(folded.to_vec()));
            }
            let Some(dense) = ports.dense(port) else {
                return Ok(None);
            };
            let early = Composed::Early { instance: id, channel: dense, port };
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
                // The rows the program NAMED, in its own per-lane numbering,
                // which is the same convention the wire branch carries.
                Some(rows) => {
                    for &r0 in rows {
                        if (r0 as usize) < hi - lo {
                            samples.push(r0);
                        }
                    }
                }
                // No readout port is "each lane reads its own last row",
                // which is the decode case this class exists for.
                None => samples.push((hi - lo).saturating_sub(1) as u32),
            }
            sample_indptr.push(samples.len() as u32);

            // THE SPAN, FROM THE POSITIONS. `driver-vulkan`'s `envelope.rs`
            // argues the choice and it is not a preference: a row writes where
            // its position says, so reading one page fewer than it writes is
            // an attention over a page this fire itself filled. `kv_len` is
            // read and CHECKED against it rather than used, because the two
            // disagreeing is a program whose epilogue lost a step.
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

        // THE CONSUMING PORTS, ADVANCED — see `Rings::consume_front`. Every
        // one the program bound, not only the ones read above: `WSlot` and
        // `WOff` are not read by this class and are consumed by the port
        // definition, so leaving them would fill their rings and wedge the
        // epilogue's own put.
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
                Refused(format!("instance {id} cannot consume channel {dense}: {why}"))
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

/// The rows request `r` names in the wire plan's sampling table.
///
/// Empty for a plan with no table, which means every request reads out its own
/// last row — the decode case.
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

/// Which wire requests one member of a step owns.
///
/// An ABSENT CSR gives the whole step to the member, which is the
/// single-member case. A CSR that EXISTS and does not describe this member is
/// `None` rather than that same fallback: the two are different claims, and
/// taking the fallback for the second hands one member every other member's
/// rows.
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
