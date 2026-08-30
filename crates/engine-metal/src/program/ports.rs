//! The descriptor-port plane, Metal half: what a fire reads out of a guest
//! instance's DEVICE rings instead of out of the submission.
//!
//! **THE ONE VALUE A HOST CANNOT KNOW IS THE SAMPLED TOKEN.** Every legacy
//! decode fixture in `tests/inferlets` carries its next token on the device:
//! the epilogue writes the sampled id straight into the channel the `embed`
//! port reads, and the host never sees it. Everything else those epilogues
//! carry — the position, the readable extent, the write slot and offset, the
//! page CSR — is pure arithmetic over the KV length, which the runtime's host
//! shadow (`eta_compiler::eval::pareval`) folds per fire. So the gap
//! between a servable decode loop and an unservable one is exactly one line
//! of guest source (`tok_in.put(&token)`), and exactly one port:
//! [`Port::EmbedTokens`].
//!
//! ```text
//! fire N            forward ──▶ logits ──▶ epilogue ── put ──▶ tok_in ring
//!                                                                  │
//! fire N+1   embed ◀── THIS MODULE ────────── read committed cell ──┘
//! ```
//!
//! **RESOLUTION IS A READ OF THE COMMITTED CELL, NOT A SECOND RING.** The
//! shell already owns the instance's rings and its cursors
//! ([`Rings`](super::launch::Rings), [`Cursor`](super::launch::Cursor)), and
//! `head` is by definition the cell the guest's own pass will take this fire.
//! So resolving a port is `read_cell(channel, head)` — the same address the
//! lane table hands the emitted kernel as `committed_cell`, read by the shell
//! a moment earlier. Nothing is consumed here: the attached instance's own
//! pass-atomic commit advances `head` for every consuming port
//! ([`Port::consumes`]), and draining a second time would spend two cells per
//! fire and desynchronize the parity diff on the first loop-carried channel.
//!
//! **WHY A HOST-SIDE READ IS THE HONEST MECHANISM FOR THIS SHELL, AND WHY IT
//! IS CHEAPER HERE THAN ANYWHERE IT HAS EVER BEEN.** The lineage had both
//! shapes. The C++ generation composed the whole geometry in a device kernel
//! taking seven cell addresses per lane, with the cursor positions PREDICTED
//! from tickets so a chained fire never waited; the Rust rewrite withdrew that
//! to `fire::envelope::compose`, a host function that `read_cell`s the
//! committed front of each bound port. Prediction bought a wait that this
//! plane pays in one place and one place only: a ring lives in a
//! `StorageModeShared` buffer on unified memory, so the committed cell is at a
//! host address the whole time and no transfer is involved at any point.
//! Reading a port is a `memcpy` out of mapped memory — not a `cudaMemcpy`, not
//! a staging hop, and measurable against nothing.
//!
//! **WHAT IT DOES COST IS A FENCE, AND THE FENCE IS THE CALLER'S.** The
//! sentence that stood here said [`Prepared::launch_region`] had already
//! waited on the command buffer that wrote the cell, which was true of a fire
//! path that waited and is not true of this one: an epilogue is encoded into
//! the MODEL fire's buffer ([`Plane::stage_into`](super::Plane::stage_into))
//! and its cursors advance one frame later, at
//! [`Plane::settle_launched`](super::Plane::settle_launched), out of the
//! harvest. So a `head` read taken while a previous epilogue is still airborne
//! is the cell of the fire before last — which on a loop-carried decode
//! channel is the token of two steps ago, a wrong answer that looks like a
//! right one. `Shell::fence_instances` is what makes the read honest, and
//! `serve::stage` takes it over exactly the attached instances before it
//! resolves anything. What it costs is what the dependency already implied: a
//! decode loop is serial by construction, and two unrelated inferlets in one
//! frame fence nothing of each other's.
//!
//! What the read BUYS is the whole point: the token stops travelling through
//! the runtime's asynchronous host plane (`take_channel` out, a guest await,
//! `publish_channel` back in), and a second decode step can be submitted
//! behind the first inside one frame.
//!
//! **WHAT IS SERVED, AND WHAT IS DERIVED.** [`PortMask::DECODE_ENVELOPE`] is
//! three ports and this module answers each in its own way:
//!
//! ```text
//! EmbedTokens   read from the ring          the device DECIDED it
//! Positions     read from the ring          the guest may renumber
//! KvLen         read from the ring, CHECKED against the shell's own count
//! ```
//!
//! The remaining four of [`PortMask::DEVICE_GEOMETRY`] — `Pages`,
//! `PageIndptr`, `WSlot`, `WOff` — are NOT read, and this shell does not claim
//! them. A decode-envelope lane's page table is the SHELL's (`KvDelta::pages`
//! empty means exactly that), so `store::kv::geometry_with` derives all four
//! from the seat, and reading the guest's copy would be reading a second
//! opinion about a table the guest does not own. A load that wanted to claim
//! them would have to let the guest's page ids reach the pool, which is the
//! pooled device-geometry class and a different piece of work.
//!
//! **A METAL CELL IS A WIRE CELL, SO THERE IS NO CONVERSION IN THIS FILE.**
//! The CUDA twin reads a NATIVE cell here — one byte per bool lane — and the
//! ports it serves happen to be integer ports, so the difference never showed.
//! On this plane it cannot show at all: `ptir_m1_runtime.metal` packs and
//! unpacks bools on the device (the `0x90`/`0x91`/`0x92` tags), so a channel
//! cell is bit-packed wire bytes for every dtype including `Bool`. That is a
//! real ABI difference between the two shells, not a simplification of one.
//! It is invisible below, because a geometry index is `I32` or `U32` and those
//! are four bytes on both.
//!
//! [`Port::consumes`]: eta_ir::registry::Port::consumes
//! [`PortMask::DECODE_ENVELOPE`]: eta_ir::registry::PortMask::DECODE_ENVELOPE
//! [`PortMask::DEVICE_GEOMETRY`]: eta_ir::registry::PortMask::DEVICE_GEOMETRY
//! [`Prepared::launch_region`]: super::launch::Prepared::launch_region

use std::sync::atomic::{AtomicU64, Ordering};

use eta_exec::{ExecPlan, Value};
use eta_ir::Dtype;
use eta_ir::registry::Port;
use eta_ir::types::name_or_unknown;

use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor, Rings};

/// How many envelopes this PROCESS has resolved.
///
/// **AN ABSENCE HAS NO OUTPUT, SO IT IS COUNTED — AND IT IS COUNTED WHERE AN
/// OBSERVER CAN REACH IT.** The claim run-ahead rests on is a negative: a
/// chained decode's token never travelled to the host. Nothing happens when a
/// round trip does not happen, so the only way a gate can tell "the second
/// fire chained" from "the second fire waited" is to count the thing that
/// DOES happen — one envelope resolved per attached device-carried lane per
/// fire. A serving test reaches a websocket and a JSON result, never a
/// [`Shell`](crate::Shell); process-global is what makes the number legible
/// from there, and this shell's own header already says a process serves one
/// fire at a time.
static RESOLVED: AtomicU64 = AtomicU64::new(0);

/// How many descriptor-port envelopes this process has resolved off guest
/// device rings. See [`RESOLVED`].
#[must_use]
pub fn resolved() -> u64 {
    RESOLVED.load(Ordering::Relaxed)
}

/// What one lane's descriptor ports resolved to, this fire.
///
/// `None` on a field is "the program binds no such port", which is a legal
/// program: [`Port::EmbedIndptr`] defaults to one run over every token and
/// [`Port::Positions`] to the seat's own count. Only [`Envelope::tokens`] has
/// no default — a fire with no token ids is a fire with no rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    /// [`Port::EmbedTokens`]: the ids this lane embeds.
    pub tokens: Option<Vec<u32>>,
    /// [`Port::Positions`]: each id's position in its sequence.
    pub positions: Option<Vec<u32>>,
    /// [`Port::KvLen`]: the readable extent AFTER this fire's writes land.
    pub kv_len: Option<Vec<u32>>,
}

impl Envelope {
    /// True when nothing was bound — the shape an attached program with no
    /// descriptor port at all resolves to.
    ///
    /// The test that authored this shape on purpose for the other shell
    /// (runtime/tests/cuda_program_epilogue.rs) is gone — deleted as
    /// misplaced, not superseded — so nothing exercises it today.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_none() && self.positions.is_none() && self.kv_len.is_none()
    }

    /// This lane's token ids, checked against the row count the composition
    /// already placed.
    ///
    /// **THE SUBMISSION STATES THE SHAPE AND THE PORT STATES THE VALUES.** A
    /// decode-envelope lane arrives carrying placeholder ids — the runtime
    /// could not know them, which is the whole reason the class exists — but
    /// it does know how many, because the token CSR is host-derivable in
    /// every trace this class admits. So the length is already load-bearing:
    /// `compose` placed this lane's rows at `row_offset`, the arena carved
    /// rectangles for them and the page CSR counted them. A port that hands
    /// back a different count is a fire whose rows have already been
    /// allocated for somebody else.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the port carries a different number of ids
    /// than the lane has rows, or when the program binds no token port at all
    /// — which is a program the caller should not have bound in this class.
    pub fn tokens_for(&self, lane: usize, rows: usize) -> Result<&[u32]> {
        let Some(tokens) = &self.tokens else {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {lane} is bound in a device-resolved geometry class and \
                     its program binds no `embed_tokens` port, so there is nothing on \
                     the device for this fire to embed"
                ),
            ));
        };
        if tokens.len() != rows {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {lane}'s `embed_tokens` port carries {} id(s) and the fire \
                     placed {rows} row(s) for it; the composition has already carved \
                     the arena rectangles and the page CSR at {rows}",
                    tokens.len()
                ),
            ));
        }
        Ok(tokens)
    }

    /// This lane's positions, checked against the run the seat is about to
    /// write.
    ///
    /// **READ AND USED, AND CHECKED BECAUSE THE PAGES ARE NOT.** The value
    /// the port carries is what reaches RoPE — that is what serving
    /// [`Port::Positions`] means. But the page CSR, the write descriptor and
    /// the attention schedules are all carved from the seat's `have .. have +
    /// rows`, and this shell owns a decode lane's page table
    /// (`KvDelta::pages` empty), so a position list that is not that run
    /// describes a fire attending pages nobody staged. The check is what
    /// makes the two one fact instead of two.
    ///
    /// A program that binds no `positions` port is not refused: the run IS
    /// the default, and `Some` versus `None` is the only difference between
    /// stating it and meaning it.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the port's run is not `have .. have + rows`.
    pub fn positions_for(&self, lane: usize, have: u32, rows: usize) -> Result<Option<&[u32]>> {
        let Some(positions) = &self.positions else {
            return Ok(None);
        };
        let natural = positions.len() == rows
            && positions
                .iter()
                .enumerate()
                .all(|(at, &position)| u64::from(position) == u64::from(have) + at as u64);
        if !natural {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {lane}'s `positions` port carries {positions:?} and this fire \
                     writes rows {have}..{}; the page CSR, the write descriptor and the \
                     attention schedules are carved from the second, so a fire that ran \
                     would rope one extent and attend another",
                    u64::from(have) + rows as u64
                ),
            ));
        }
        Ok(Some(positions))
    }

    /// This lane's readable extent, checked against what the seat says it
    /// will be.
    ///
    /// **A CHECK, NOT A SOURCE.** The extent the shell fires with is
    /// `have + rows` — the seat's own arithmetic, which is also what the
    /// attention schedules, the page CSR and the write descriptor are carved
    /// from. Taking the guest's number instead would let one port silently
    /// disagree with the four the shell derives, and the failure is a fire
    /// that attends the wrong pages. Taking it as a CHECK is what makes
    /// [`Port::KvLen`] served rather than ignored: a guest whose count has
    /// drifted from the shell's is a named refusal on the fire that drifts,
    /// not a wrong answer forever.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the port states an extent this fire is not
    /// about to reach.
    pub fn check_extent(&self, lane: usize, after: u32) -> Result<()> {
        let Some(kv_len) = &self.kv_len else {
            return Ok(());
        };
        // A `[lanes]` cell over a one-lane instance is one entry; a program
        // that binds a wider one is stating extents for lanes this
        // attachment does not carry, and the first is this lane's.
        let Some(&stated) = kv_len.first() else {
            return Ok(());
        };
        if stated != after {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {lane} states a readable KV extent of {stated} on its `kv_len` \
                     port and this fire reaches {after}; the shell carves the page CSR, \
                     the write descriptor and the attention schedules from {after}, so a \
                     fire that ran would attend a different extent than the guest thinks \
                     it wrote"
                ),
            ));
        }
        Ok(())
    }
}

/// Resolve one instance's descriptor ports out of its rings.
///
/// Constant ports come from the plan's folded values and channel-bound ones
/// from the committed cell — the same cell the lane table will hand the
/// emitted kernel, so the shell and the guest read one value and not two.
///
/// # Errors
///
/// [`Fault::Program`] for a port naming a channel the instance does not
/// carry, a cell whose element type is not an integer (a geometry index is
/// not an activation), or a const port with no folded value; and whatever the
/// read said.
pub fn resolve(
    plan: &ExecPlan,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Envelope> {
    let mut out = Envelope::default();
    RESOLVED.fetch_add(1, Ordering::Relaxed);
    for binding in &plan.package.ports {
        let slot = match binding.port {
            Port::EmbedTokens => &mut out.tokens,
            Port::Positions => &mut out.positions,
            Port::KvLen => &mut out.kv_len,
            // Every other port is either derived by the shell (the page
            // family, the write descriptor) or has no reader on this path
            // (`Readout` is `Lane::readout`, `AttnMask` is a class this shell
            // declines). Skipped rather than refused: a program that binds
            // them is legal, and its cells are drained by its own commit.
            _ => continue,
        };
        let value = if binding.is_const {
            let folded = plan
                .const_ports
                .iter()
                .find(|folded| folded.port == binding.port)
                .ok_or_else(|| {
                    Fault::program(
                        "program::ports",
                        format!(
                            "port {} is declared const and no folded value was kept for it",
                            binding.port.name()
                        ),
                    )
                })?;
            as_u32(binding.port, &folded.value)?
        } else {
            read_cell(binding.port, binding.channel, rings, cursors, shapes)?
        };
        *slot = Some(value);
    }
    Ok(out)
}

/// One port's committed cell, as geometry indices.
fn read_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<u32>> {
    let index = channel as usize;
    let (Some(cursor), Some(shape)) = (cursors.get(index).copied(), shapes.get(index).copied())
    else {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {} names channel {channel}, which this instance does not carry",
                port.name()
            ),
        ));
    };
    if !matches!(shape.dtype, Dtype::I32 | Dtype::U32) {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} holds {}, and a geometry index is not an \
                 activation",
                port.name(),
                name_or_unknown(shape.dtype)
            ),
        ));
    }
    // **AN EMPTY RING IS A REFUSAL AND NOT A ZERO.** `head == tail` is a
    // channel nothing has committed into — a fresh instance nobody seeded and
    // whose epilogue has not run — and the cell at `head` then holds whatever
    // the allocation came with. A shell that read it would embed token zero,
    // or garbage, on the first fire of every decode loop and never say so.
    // The readiness gate (`Session::blocked_channel`) catches this for a
    // channel whose program DECLARES `NeedsFull`, which is every decode
    // fixture; this is the same fact asked of the port itself, so a program
    // that declares no readiness on the channel it binds a port to is refused
    // here rather than answered from an unwritten cell.
    if cursor.tail == cursor.head {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {} names channel {channel}, whose ring holds nothing committed \
                 (head and tail are both {}); the port's value for this fire is the \
                 cell the guest's own pass takes, and no pass has published one",
                port.name(),
                cursor.head
            ),
        ));
    }
    // `head`, not `tail`: the committed cell is the one the guest's own pass
    // takes this fire, and it is the address the lane table publishes as
    // `committed_cell`. Reading `tail` would read the cell this fire's
    // epilogue is about to WRITE, which on a loop-carried channel is the
    // token of the fire after this one.
    //
    // The bytes need no unpacking: an `I32`/`U32` cell is four little-endian
    // bytes a lane on the wire and on this device both, and the guard above
    // has already refused every dtype where the two could differ.
    let cell = rings.read_cell(index, cursor.head)?;
    Ok(cell
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

/// A folded constant, as geometry indices.
fn as_u32(port: Port, value: &Value) -> Result<Vec<u32>> {
    match value {
        Value::U32(lanes) => Ok(lanes.clone()),
        // Reinterpreted, not converted: an in-band `-1` is a skip sentinel and
        // saturating it to zero would embed row zero instead of skipping.
        Value::I32(lanes) => Ok(lanes.iter().map(|&lane| lane as u32).collect()),
        other => Err(Fault::program(
            "program::ports",
            format!(
                "the folded constant for port {} is {}, and a geometry index is not an \
                 activation",
                port.name(),
                name_or_unknown(other.dtype())
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::Envelope;

    /// The check is an EQUALITY against the seat, and the sentence names both
    /// numbers — a refusal that named only one would leave the reader
    /// guessing which side drifted.
    #[test]
    fn an_extent_that_disagrees_with_the_seat_is_refused_by_both_numbers() {
        let envelope = Envelope {
            kv_len: Some(vec![9]),
            ..Envelope::default()
        };
        let refusal = envelope
            .check_extent(0, 6)
            .expect_err("the port says nine and the fire reaches six");
        let text = format!("{refusal}");
        assert!(text.contains('9'), "names what the port stated: {text}");
        assert!(text.contains('6'), "and what the fire reaches: {text}");
    }

    /// The agreeing case is silent, and so is a program that binds no
    /// `kv_len` port at all — an attached epilogue with no descriptor port is
    /// the served shape, not an omission.
    #[test]
    fn an_agreeing_extent_and_an_unbound_one_both_pass() {
        Envelope {
            kv_len: Some(vec![6]),
            ..Envelope::default()
        }
        .check_extent(0, 6)
        .expect("the port and the seat agree");
        Envelope::default()
            .check_extent(0, 6)
            .expect("nothing to check");
        assert!(Envelope::default().is_empty());
    }
}
