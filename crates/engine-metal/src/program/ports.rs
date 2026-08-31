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
//! **WHAT IS SERVED, AND HOW EACH PORT ANSWERS.** [`PortMask::DEVICE_GEOMETRY`]
//! is seven ports and this module answers each in its own way — plus
//! [`Port::EmbedIndptr`], which is not a value the fire needs but the LANE
//! SPLIT the other seven are cut by, and [`Port::AttnMask`], which is a
//! rectangle of bools rather than an index vector:
//!
//! ```text
//! EmbedIndptr   read from the ring   the member's own lane CSR: how many
//!                                    lanes this instance carries and where
//!                                    each one's rows are in the flat vectors
//! EmbedTokens   read from the ring   the device DECIDED it
//! Positions     read from the ring   the guest renumbers (a beam's logical
//!                                    position is not its pool offset), and
//!                                    the natural-run check therefore applies
//!                                    only to a lane whose pages are OURS
//! KvLen         read from the ring   a SOURCE when the guest states its own
//!                                    pages, a CHECK against the seat when it
//!                                    does not
//! Pages         read from the ring   one flat run, cut per lane by
//! PageIndptr    read from the ring   `page_indptr` — the wire CSR's own shape
//! WSlot         read from the ring   the explicit write descriptor: the page
//! WOff          read from the ring   and the offset THIS row lands at, which
//!                                    a `have + row` derivation cannot spell
//!                                    when B lanes append into one flat pool
//! AttnMask      read from the ring   a dense `[rows, pool]` bool rectangle,
//!                                    run-length encoded here and expanded by
//!                                    `crate::mask::stage` into the very slab
//!                                    a host-stated mask expands into
//! ```
//!
//! **THE FOUR THE DECODE ENVELOPE DID NOT NEED ARE THE FOUR THAT MOVE THE
//! WRITE.** A decode-envelope lane's page table is the SHELL's
//! (`KvDelta::pages` empty), so `store::kv::geometry_with` derives the page
//! run, the write page and the write offset from the seat, and reading the
//! guest's copy would have been reading a second opinion about a table the
//! guest does not own. A device-GEOMETRY lane's table is the GUEST's: a beam
//! search keeps one shared flat pool, every beam appends at `fill + lane`, and
//! there is no `have` per lane from which `have + row` names those cells —
//! two beams would derive the same offset. So the four ports are read and
//! USED, and the seat's own derivation is overwritten with them.
//!
//! **A MEMBER IS NOT A LANE.** One attached instance can carry several lanes
//! (a beam search binds `B` of them through one program), and the runtime's
//! attachment names only the member's FIRST lane. `EmbedIndptr` is what closes
//! that gap: its `[lanes + 1]` CSR says how many lanes the member carries and
//! cuts every flat vector — tokens, positions, write descriptors, mask rows —
//! into per-lane runs. It is also the only place a pooled device-geometry fire
//! learns its ROW COUNTS at all: such a submission ships `Lane::tokens` empty
//! on every lane, because the split is the instance's and the runtime has no
//! more claim on it than it has on the page table beside it. A member that
//! binds no such port is one lane, which is every decode-envelope guest and is
//! why nothing about them changes.
//!
//! **A METAL CELL IS A WIRE CELL, AND THAT IS WHERE THE TWO SHELLS PART.**
//! The CUDA twin reads a NATIVE cell here — one byte per bool lane — and for
//! the eight integer ports the difference never shows. On this plane it
//! cannot: `ptir_m1_runtime.metal` packs and unpacks bools on the device (the
//! `0x90`/`0x91`/`0x92` tags), so a `Bool` channel's cell is
//! `numel.div_ceil(8)` BIT-PACKED bytes and a byte-per-lane reading of it
//! would take eight mask cells for one and call the other seven zero. That is
//! a real ABI difference between the two shells, not a simplification of one,
//! and [`read_bool_cell`] is where it is paid: `eta_exec`'s own
//! `encode_wire`/`decode_wire` pair owns the bit order (LSB-first inside each
//! byte) and is private to that crate, so the shift is spelled once here and
//! held against the encoder by a test rather than shared with it. It is
//! invisible for every other port, because a geometry index is `I32` or `U32`
//! and those are four bytes on both.
//!
//! [`Port::consumes`]: eta_ir::registry::Port::consumes
//! [`PortMask::DECODE_ENVELOPE`]: eta_ir::registry::PortMask::DECODE_ENVELOPE
//! [`PortMask::DEVICE_GEOMETRY`]: eta_ir::registry::PortMask::DEVICE_GEOMETRY
//! [`Prepared::launch_region`]: super::launch::Prepared::launch_region

use std::sync::atomic::{AtomicU64, Ordering};

use eta_exec::{ExecPlan, Value};
use eta_ir::Dtype;
use eta_ir::registry::{GeometryClass, Port};
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

/// What one INSTANCE's descriptor ports resolved to, this fire.
///
/// `None` on a field is "the program binds no such port", which is a legal
/// program: [`Port::EmbedIndptr`] defaults to one run over every token and
/// [`Port::Positions`] to the seat's own count. Only [`Envelope::tokens`] has
/// no default — a fire with no token ids is a fire with no rows.
///
/// **THE VECTORS ARE THE MEMBER'S, NOT THE LANE'S.** One instance may carry
/// several lanes; [`Envelope::qo_indptr`] is the CSR that cuts these flat
/// vectors into them, and [`Envelope::lane`] is the only thing that does the
/// cutting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    /// [`Port::EmbedIndptr`]: `[lanes + 1]` row bounds — how many lanes this
    /// instance carries and where each one's rows lie in the flat vectors.
    pub qo_indptr: Option<Vec<u32>>,
    /// [`Port::EmbedTokens`]: the ids this instance embeds, all lanes end to
    /// end.
    pub tokens: Option<Vec<u32>>,
    /// [`Port::Positions`]: each id's position in its sequence.
    pub positions: Option<Vec<u32>>,
    /// [`Port::KvLen`]: each lane's readable extent AFTER this fire's writes
    /// land.
    pub kv_len: Option<Vec<u32>>,
    /// [`Port::Pages`]: the page ids every lane may address, one flat run cut
    /// by [`Envelope::page_indptr`]. In the GUEST's space — working-set
    /// relative — until `serve::prepare` translates it.
    pub pages: Option<Vec<u32>>,
    /// [`Port::PageIndptr`]: `[lanes + 1]` bounds cutting [`Envelope::pages`].
    pub page_indptr: Option<Vec<u32>>,
    /// [`Port::WSlot`]: the page each token ROW is appended into, in the same
    /// guest space [`Envelope::pages`] is in.
    pub w_slot: Option<Vec<u32>>,
    /// [`Port::WOff`]: that row's offset inside that page, which is in no
    /// space at all.
    pub w_off: Option<Vec<u32>>,
    /// [`Port::AttnMask`]: a dense `[rows, keys]` bool rectangle, row-major,
    /// at whatever key width the guest built it — which is the POOL's width
    /// and not the extent's (see [`crate::mask`]'s "a mask may be LONGER").
    pub mask: Option<Vec<bool>>,
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
        self.qo_indptr.is_none()
            && self.tokens.is_none()
            && self.positions.is_none()
            && self.kv_len.is_none()
            && self.pages.is_none()
            && self.page_indptr.is_none()
            && self.w_slot.is_none()
            && self.w_off.is_none()
            && self.mask.is_none()
    }

    /// **THIS INSTANCE STATES ITS OWN PAGE TABLE**, which is what separates a
    /// device-GEOMETRY lane from a decode-ENVELOPE one: the former's pages,
    /// write descriptor and readable extent are all the guest's, the latter's
    /// are all this shell's and only the ids it embeds are not.
    #[must_use]
    pub fn owns_pages(&self) -> bool {
        self.pages.is_some() && self.page_indptr.is_some()
    }

    /// How many LANES this instance carries.
    ///
    /// **ONE, UNLESS THE INSTANCE STATES ITS OWN PAGE TABLE.** The multi-lane
    /// reading is the device-GEOMETRY class's and only that class's, because
    /// it is the only class whose submission carries no row split for the CSR
    /// to be checked against: a decode-envelope member arrives with its rows
    /// already placed and its extent already carved from a seat, and reading
    /// its `embed_indptr` as a lane count would let a traced-for-B program
    /// silently serve one lane of a submission that placed one. So the class
    /// that needs the CSR reads it and the class that does not keeps the
    /// contract it had, byte for byte ([`Envelope::owns_pages`] is the same
    /// split [`LanePorts::positions_for`] turns on).
    #[must_use]
    pub fn lanes(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_pages() && csr.len() >= 2 => csr.len() - 1,
            _ => 1,
        }
    }

    /// Lane `at` of this instance's ports, for a submission lane `source` the
    /// refusals name.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a lane past the CSR, and for a CSR that is not
    /// monotone — a span that runs backwards would slice the flat vectors at
    /// a wrapped length and read another lane's rows.
    pub fn lane(&self, at: usize, source: usize) -> Result<LanePorts<'_>> {
        let rows = match &self.qo_indptr {
            // See [`Envelope::lanes`]: the CSR cuts the flat vectors only for
            // the class that states no row split anywhere else.
            Some(csr) if self.owns_pages() && csr.len() >= 2 => {
                let (Some(&start), Some(&end)) = (csr.get(at), csr.get(at + 1)) else {
                    return Err(Fault::program(
                        "program::ports",
                        format!(
                            "lane {source} is lane {at} of its instance, whose \
                             `embed_indptr` port carries {} bound(s) and therefore \
                             describes {} lane(s)",
                            csr.len(),
                            self.lanes()
                        ),
                    ));
                };
                if end < start {
                    return Err(Fault::program(
                        "program::ports",
                        format!(
                            "lane {source}'s `embed_indptr` port states rows \
                             {start}..{end}, which runs backwards"
                        ),
                    ));
                }
                start as usize..end as usize
            }
            // No CSR: the instance is one lane and every flat vector is its.
            _ => 0..self.tokens.as_ref().map_or(0, Vec::len),
        };
        Ok(LanePorts {
            envelope: self,
            at,
            rows,
            source,
        })
    }

    /// How many token ROWS the whole instance's flat vectors span.
    fn spanned(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_pages() && csr.len() >= 2 => {
                csr.last().copied().unwrap_or(0) as usize
            }
            _ => self.tokens.as_ref().map_or(0, Vec::len),
        }
    }
}

/// One LANE's share of an instance's resolved descriptor ports.
///
/// Every accessor cuts the member-wide vector by this lane's CSR span, and
/// every one of them refuses rather than clamps: a port whose length
/// disagrees with the composition is a fire whose rows have already been
/// allocated for somebody else.
#[derive(Clone, Debug)]
pub struct LanePorts<'a> {
    envelope: &'a Envelope,
    /// Which lane of the INSTANCE this is.
    at: usize,
    /// Its rows, as the instance's own token CSR cuts them.
    rows: std::ops::Range<usize>,
    /// Which lane of the SUBMISSION it is, for the refusals to name.
    source: usize,
}

impl LanePorts<'_> {
    /// See [`Envelope::owns_pages`].
    #[must_use]
    pub fn owns_pages(&self) -> bool {
        self.envelope.owns_pages()
    }

    /// How many token ROWS this lane carries, as its instance's own token CSR
    /// cuts them.
    ///
    /// **THE ROW SPLIT IS A DESCRIPTOR PORT LIKE THE REST OF THE GEOMETRY.** A
    /// device-geometry submission states no row counts — `Lane::tokens` is
    /// empty on every lane, because the runtime owns predictions and not the
    /// split — so this is where a fire of that class learns how many rows it
    /// places, and everything downstream (the windows, the row offsets, the
    /// arena rectangles, the page CSR) is carved from it.
    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.rows.end.saturating_sub(self.rows.start)).unwrap_or(u32::MAX)
    }

    /// This lane's token ids, checked against the row count the composition
    /// already placed.
    ///
    /// **THE SUBMISSION STATES THE SHAPE AND THE PORT STATES THE VALUES.** A
    /// decode-envelope lane arrives carrying placeholder ids — the runtime
    /// could not know them, which is the whole reason the class exists — but
    /// it does know how many, because the token CSR is host-derivable in
    /// every trace that class admits. So the length is already load-bearing:
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
    pub fn tokens_for(&self, rows: usize) -> Result<&[u32]> {
        let source = self.source;
        let Some(tokens) = &self.envelope.tokens else {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source} is bound in a device-resolved geometry class and \
                     its program binds no `embed_tokens` port, so there is nothing on \
                     the device for this fire to embed"
                ),
            ));
        };
        let ids = self.slice(tokens, "embed_tokens")?;
        if ids.len() != rows {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source}'s `embed_tokens` port carries {} id(s) and the fire \
                     placed {rows} row(s) for it; the composition has already carved \
                     the arena rectangles and the page CSR at {rows}",
                    ids.len()
                ),
            ));
        }
        Ok(ids)
    }

    /// This lane's positions.
    ///
    /// **READ AND USED, AND CHECKED ONLY WHERE THE PAGES ARE NOT THE
    /// GUEST'S.** For a decode-ENVELOPE lane the page CSR, the write
    /// descriptor and the attention schedules are all carved from the seat's
    /// `have .. have + rows`, and this shell owns that lane's page table
    /// (`KvDelta::pages` empty), so a position list that is not that run
    /// describes a fire attending pages nobody staged. The check is what
    /// makes the two one fact instead of two.
    ///
    /// For a device-GEOMETRY lane the run is not the seat's and the check
    /// would be false: a beam's logical position advances by one per step
    /// while the cell it writes is `fill + lane` of a shared flat pool, and
    /// the two numbers are unrelated by construction. What ties the fire
    /// together there is the write descriptor (`w_slot`/`w_off`) and the
    /// ancestry mask, both of which this same instance states — so the
    /// positions are taken as what reaches RoPE and nothing else, which is
    /// what serving [`Port::Positions`] means.
    ///
    /// A program that binds no `positions` port is not refused: the run IS
    /// the default, and `Some` versus `None` is the only difference between
    /// stating it and meaning it.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the port's run is not `have .. have + rows` on
    /// a lane whose pages are this shell's, and for a span the flat vector
    /// does not cover.
    pub fn positions_for(&self, have: u32, rows: usize) -> Result<Option<&[u32]>> {
        let Some(positions) = &self.envelope.positions else {
            return Ok(None);
        };
        let stated = self.slice(positions, "positions")?;
        if self.owns_pages() {
            return Ok(Some(stated));
        }
        let natural = stated.len() == rows
            && stated
                .iter()
                .enumerate()
                .all(|(at, &position)| u64::from(position) == u64::from(have) + at as u64);
        if !natural {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {}'s `positions` port carries {stated:?} and this fire \
                     writes rows {have}..{}; the page CSR, the write descriptor and the \
                     attention schedules are carved from the second, so a fire that ran \
                     would rope one extent and attend another",
                    self.source,
                    u64::from(have) + rows as u64
                ),
            ));
        }
        Ok(Some(stated))
    }

    /// This lane's stated readable extent, or `None` for a program that binds
    /// no `kv_len` port.
    ///
    /// **A SOURCE ON A DEVICE-GEOMETRY LANE AND A CHECK ON EVERY OTHER.** The
    /// extent a decode-envelope fire reaches is `have + rows` — the seat's own
    /// arithmetic, which is also what its page CSR and write descriptor are
    /// carved from — so taking the guest's number instead would let one port
    /// silently disagree with four the shell derives
    /// ([`LanePorts::check_extent`] is that reading). A device-geometry lane
    /// has no such seat: its pages, its write descriptor and its extent are
    /// one statement by one author, and `have` is derived BACK from it as
    /// `kv_len - rows`.
    #[must_use]
    pub fn extent(&self) -> Option<u32> {
        // A `[lanes]` cell states one entry per lane of the instance; a
        // program that binds a wider one is stating extents for lanes this
        // attachment does not carry, and the first is this lane's.
        self.envelope
            .kv_len
            .as_ref()
            .and_then(|lens| lens.get(self.at).or_else(|| lens.first()))
            .copied()
    }

    /// This lane's readable extent, checked against what the seat says it
    /// will be.
    ///
    /// **A CHECK, NOT A SOURCE**, and on a device-geometry lane an IDENTITY —
    /// see [`LanePorts::extent`], where `have` was derived back from this very
    /// number. Taking it as a CHECK is what makes [`Port::KvLen`] served
    /// rather than ignored: a guest whose count has drifted from the shell's
    /// is a named refusal on the fire that drifts, not a wrong answer forever.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the port states an extent this fire is not
    /// about to reach.
    pub fn check_extent(&self, after: u32) -> Result<()> {
        let Some(stated) = self.extent() else {
            return Ok(());
        };
        if stated != after {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {} states a readable KV extent of {stated} on its `kv_len` \
                     port and this fire reaches {after}; the shell carves the page CSR, \
                     the write descriptor and the attention schedules from {after}, so a \
                     fire that ran would attend a different extent than the guest thinks \
                     it wrote",
                    self.source
                ),
            ));
        }
        Ok(())
    }

    /// This lane's page table, in sequence order and in the GUEST's space, or
    /// `None` when the program binds no page family and the table is
    /// therefore this shell's.
    ///
    /// **ONE FLAT RUN, CUT BY `page_indptr`** — the wire CSR's own shape, and
    /// the shape `pipeline::fire::geometry`'s `compact_page_envelope` hands
    /// every other engine. A guest that keeps a `[lanes, P]` envelope tiles it
    /// so that lane `l`'s live prefix stands at `page_indptr[l]`, because that
    /// is what the CSR MEANS; the alternative reading — a fixed stride the
    /// CSR only counts — is not derivable from a cell whose declared rank this
    /// plane does not carry, so it is refused rather than guessed at, by the
    /// bounds check below.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a `pages` port with no `page_indptr` beside it
    /// (or the reverse), for a CSR shorter than the instance's lanes, and for
    /// a span the flat page run does not cover.
    pub fn pages(&self) -> Result<Option<&[u32]>> {
        let source = self.source;
        match (&self.envelope.pages, &self.envelope.page_indptr) {
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source}'s program binds one of `pages`/`page_indptr` and not \
                     the other; a flat page run with no CSR to cut it names no lane's \
                     table, and a CSR with nothing to cut names no pages"
                ),
            )),
            (Some(pages), Some(csr)) => {
                let (Some(&start), Some(&end)) = (csr.get(self.at), csr.get(self.at + 1)) else {
                    return Err(Fault::program(
                        "program::ports",
                        format!(
                            "lane {source} is lane {} of its instance and its \
                             `page_indptr` port carries {} bound(s)",
                            self.at,
                            csr.len()
                        ),
                    ));
                };
                let (start, end) = (start as usize, end as usize);
                if end < start || end > pages.len() {
                    return Err(Fault::program(
                        "program::ports",
                        format!(
                            "lane {source}'s `page_indptr` port cuts pages {start}..{end} \
                             out of the {} its `pages` port carries",
                            pages.len()
                        ),
                    ));
                }
                Ok(Some(&pages[start..end]))
            }
        }
    }

    /// This lane's explicit write descriptor — `(page, offset)` per token row
    /// — or `None` when the program binds neither half and the seat's own
    /// `have + row` arithmetic stands.
    ///
    /// **THE ONE DERIVATION A SEAT CANNOT SPELL.** `store::kv::geometry_with`
    /// lands row `r` at flat position `have + r` of the lane's page run, which
    /// is right for every sequence that appends to its own tail. A beam search
    /// keeps ONE flat pool behind `B` lanes and appends beam `b` at
    /// `fill + b`; every lane's `have` is the same number, so `have + r` names
    /// one cell for all of them and `B - 1` beams would overwrite the first.
    /// The guest computes the descriptor in its own epilogue and states it
    /// here, and this is where the derived answer is replaced by it.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a program binding one half and not the other,
    /// and for a descriptor whose length is not this lane's row count.
    pub fn writes(&self, rows: usize) -> Result<Option<(&[u32], &[u32])>> {
        let source = self.source;
        match (&self.envelope.w_slot, &self.envelope.w_off) {
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source}'s program binds one of `w_slot`/`w_off` and not the \
                     other; a page with no offset in it addresses no cell"
                ),
            )),
            (Some(slots), Some(offsets)) => {
                let slots = self.slice(slots, "w_slot")?;
                let offsets = self.slice(offsets, "w_off")?;
                if slots.len() != rows || offsets.len() != rows {
                    return Err(Fault::program(
                        "program::ports",
                        format!(
                            "lane {source}'s write descriptor carries {} page(s) and {} \
                             offset(s) for the {rows} row(s) this fire placed",
                            slots.len(),
                            offsets.len()
                        ),
                    ));
                }
                Ok(Some((slots, offsets)))
            }
        }
    }

    /// This lane's dense attention mask — `rows` rectangles of `stride` bools,
    /// row-major — or `None` for a program that binds no `attn_mask` port.
    ///
    /// The stride is the rectangle's own key width, which is the width the
    /// GUEST built it at: a beam search states `[B, POOL]` over the pool it
    /// reserved, and the pool does not shrink as the extent grows. Clipping
    /// that surplus is [`crate::mask`]'s rule and it is the same rule for a
    /// host-stated mask.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] when the rectangle does not divide into the
    /// instance's rows, and for a span it does not cover.
    pub fn mask(&self, rows: usize) -> Result<Option<(&[bool], usize)>> {
        let source = self.source;
        let Some(dense) = &self.envelope.mask else {
            return Ok(None);
        };
        let spanned = self.envelope.spanned().max(1);
        if dense.len() % spanned != 0 {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source}'s `attn_mask` port carries {} cell(s) over the \
                     {spanned} row(s) its instance spans, which is not a rectangle",
                    dense.len()
                ),
            ));
        }
        let stride = dense.len() / spanned;
        if stride == 0 {
            return Err(Fault::program(
                "program::ports",
                format!("lane {source}'s `attn_mask` port states a zero-wide key axis"),
            ));
        }
        let (start, end) = (self.rows.start * stride, self.rows.end * stride);
        if end > dense.len() || end - start != rows * stride {
            return Err(Fault::program(
                "program::ports",
                format!(
                    "lane {source}'s `attn_mask` port holds rows {}..{} of a {}-cell \
                     rectangle {stride} wide, and this fire placed {rows} row(s) for it",
                    self.rows.start,
                    self.rows.end,
                    dense.len()
                ),
            ));
        }
        Ok(Some((&dense[start..end], stride)))
    }

    /// This lane's share of a member-wide per-ROW vector.
    fn slice<'v, T>(&self, flat: &'v [T], port: &str) -> Result<&'v [T]> {
        flat.get(self.rows.clone()).ok_or_else(|| {
            Fault::program(
                "program::ports",
                format!(
                    "lane {}'s `{port}` port carries {} entrie(s) and its instance's \
                     token CSR cuts rows {}..{} out of it",
                    self.source,
                    flat.len(),
                    self.rows.start,
                    self.rows.end
                ),
            )
        })
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
/// not an activation) or — for [`Port::AttnMask`] — not a bool, a ring with
/// nothing committed in it, or a const port with no folded value; and
/// whatever the read said.
pub fn resolve(
    plan: &ExecPlan,
    class: GeometryClass,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Envelope> {
    let mut out = Envelope::default();
    RESOLVED.fetch_add(1, Ordering::Relaxed);
    for binding in &plan.package.ports {
        if !resolves(class, binding.port) {
            continue;
        }
        // **THE MASK IS THE ONE PORT THAT IS NOT AN INDEX VECTOR**, so it is
        // read on its own arm: a `[rows, keys]` rectangle of bools, bit-packed
        // on this plane, and `as_u32` has nothing true to say about it.
        if binding.port == Port::AttnMask {
            if binding.is_const {
                // A const mask is a mask the HOST folded, and the runtime
                // lowers those to run-length `Masking` on the submission
                // (`fire::geometry::lower_attn_mask_evaluated`). Reading it
                // here as well would stage the same restriction twice.
                continue;
            }
            out.mask = Some(read_bool_cell(
                binding.port,
                binding.channel,
                rings,
                cursors,
                shapes,
            )?);
            continue;
        }
        let slot = match binding.port {
            Port::EmbedIndptr => &mut out.qo_indptr,
            Port::EmbedTokens => &mut out.tokens,
            Port::Positions => &mut out.positions,
            Port::KvLen => &mut out.kv_len,
            Port::Pages => &mut out.pages,
            Port::PageIndptr => &mut out.page_indptr,
            Port::WSlot => &mut out.w_slot,
            Port::WOff => &mut out.w_off,
            // `resolves` already let only the ports above through; what is
            // left has no reader on this path (`Readout` is `Lane::readout`)
            // or is the recurrent buffered family, which is RESERVED — and on
            // this plane also unimplemented, so a fire that read one would
            // have nowhere to put it.
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

/// **DOES AN INSTANCE BOUND IN `class` RESOLVE `port` OFF ITS OWN RINGS?**
///
/// The rule the whole plane turns on, written once. Three groups:
///
/// ```text
/// the class's own ports    GeometryClass::ports() — the contract's set
/// EmbedIndptr              the ROW SPLIT the wide class's flat vectors are
///                          cut by, and which no submission of that class
///                          states; nothing for the narrow class, whose
///                          submission places its own rows
/// AttnMask                 the wide class only; a host-known mask is lowered
///                          to run lengths on the submission instead
/// ```
///
/// **AND THE FIRST GROUP IS WHY THIS IS A FUNCTION AND NOT AN `if`.** Almost
/// every attention guest binds `pages` and `page_indptr` — the SDK's
/// `KvGeometry` sugar makes them part of an ordinary bind — and states them as
/// `0 .. reserved`, because a guest holds WORKING-SET-RELATIVE indexes and
/// never pool page ids. For a decode-envelope pass the RUNTIME is what crosses
/// that gap: it folds the port host-side and translates through the working
/// set's flat table (`pipeline::fire::map_lane_pages`). An engine that read
/// the guest's copy as well would take the relative index for a pool id, and
/// then every lane in the process addresses pages `0, 1, ...` and reads back
/// somebody else's cache — invisible alone, invisible under a homogeneous
/// load, and a wrong answer the moment two different guests share a device.
///
/// So the set is the CLASS's, not "whatever the program happened to bind".
#[must_use]
pub fn resolves(class: GeometryClass, port: Port) -> bool {
    if class.ports().contains(port) {
        return true;
    }
    matches!(port, Port::EmbedIndptr | Port::AttnMask) && class == GeometryClass::DeviceGeometry
}

/// One port's committed cell, as geometry indices.
fn read_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<u32>> {
    let (cursor, shape) = cell_of(port, channel, cursors, shapes)?;
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
    // `head`, not `tail`: the committed cell is the one the guest's own pass
    // takes this fire, and it is the address the lane table publishes as
    // `committed_cell`. Reading `tail` would read the cell this fire's
    // epilogue is about to WRITE, which on a loop-carried channel is the
    // token of the fire after this one.
    //
    // The bytes need no unpacking: an `I32`/`U32` cell is four little-endian
    // bytes a lane on the wire and on this device both, and the guard above
    // has already refused every dtype where the two could differ.
    let cell = rings.read_cell(channel as usize, cursor.head)?;
    Ok(cell
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

/// One port's committed cell, as a dense rectangle of bools.
///
/// **THE MASK IS THE ONE PORT WHOSE CELL IS NOT WORDS, AND ON THIS PLANE IT IS
/// NOT BYTES EITHER.** The CUDA twin's `Rings::read_cell` hands back one byte
/// a bool; this one hands back the WIRE cell, which for `Bool` is
/// `numel.div_ceil(8)` bit-packed bytes — the encoding
/// `ptir_m1_runtime.metal`'s `0x90`/`0x91`/`0x92` handlers write. So the
/// unpacking is real work rather than a reading, and getting the bit order
/// backwards would mask the wrong keys and answer fluently rather than fault.
///
/// **THE ORDER IS `eta_exec`'s, QUOTED RATHER THAN INVENTED.** Its
/// `encode_wire` writes lane `j` as `dst[j / 8] |= 1 << (j % 8)` — LSB-first
/// inside each byte — and its `decode_wire` reads it back the same way. That
/// pair is private to its crate (alto E withdrew the export, and this shell may
/// not put it back), so the shift is spelled here; it is spelled ONCE, and this
/// module's `a_packed_cell_unpacks_lsb_first` is the gate that holds it against
/// the encoder's own reading.
fn read_bool_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<bool>> {
    let (cursor, shape) = cell_of(port, channel, cursors, shapes)?;
    if shape.dtype != Dtype::Bool {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} holds {}, and an attention mask is a \
                 rectangle of bools",
                port.name(),
                name_or_unknown(shape.dtype)
            ),
        ));
    }
    let cell = rings.read_cell(channel as usize, cursor.head)?;
    if cell.len() != shape.cell_bytes() {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} declares {} bool lane(s) — {} packed byte(s) \
                 — and its cell is {} byte(s)",
                port.name(),
                shape.numel,
                shape.cell_bytes(),
                cell.len()
            ),
        ));
    }
    Ok(unpack_bits(&cell, shape.numel))
}

/// `numel` bool lanes out of the wire's packed bytes, LSB-first inside each
/// byte. See [`read_bool_cell`] for whose order this is.
fn unpack_bits(packed: &[u8], numel: usize) -> Vec<bool> {
    (0..numel)
        .map(|lane| packed[lane / 8] >> (lane % 8) & 1 != 0)
        .collect()
}

/// The cursor and the shape one port's channel stands at, with the two facts
/// every read of a committed cell has to establish first.
///
/// **AN EMPTY RING IS A REFUSAL AND NOT A ZERO.** `head == tail` is a channel
/// nothing has committed into — a fresh instance nobody seeded and whose
/// epilogue has not run — and the cell at `head` then holds whatever the
/// allocation came with. A shell that read it would embed token zero, or
/// garbage, on the first fire of every decode loop and never say so. The
/// readiness gate (`Session::blocked_channel`) catches this for a channel
/// whose program DECLARES `NeedsFull`, which is every decode fixture; this is
/// the same fact asked of the port itself, so a program that declares no
/// readiness on the channel it binds a port to is refused here rather than
/// answered from an unwritten cell.
fn cell_of(
    port: Port,
    channel: u32,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<(Cursor, ChannelShape)> {
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
    Ok((cursor, shape))
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
    use super::{Envelope, Port};

    use eta_ir::registry::GeometryClass;

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
            .lane(0, 0)
            .expect("one lane")
            .check_extent(6)
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
        .lane(0, 0)
        .expect("one lane")
        .check_extent(6)
        .expect("the port and the seat agree");
        Envelope::default()
            .lane(0, 0)
            .expect("one lane")
            .check_extent(6)
            .expect("nothing to check");
        assert!(Envelope::default().is_empty());
    }

    /// **A MEMBER IS NOT A LANE, AND THE TOKEN CSR IS WHAT SAYS SO.** Three
    /// beams through one instance: the flat vectors are the member's and each
    /// lane takes its own row, its own extent, its own page run and its own
    /// mask rectangle.
    #[test]
    fn one_instances_flat_vectors_cut_into_its_lanes_by_the_token_csr() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 1, 2, 3]),
            tokens: Some(vec![11, 22, 33]),
            positions: Some(vec![7, 7, 7]),
            kv_len: Some(vec![9, 9, 9]),
            // Two pages a lane, six in the flat run, cut 0..2 / 2..4 / 4..6.
            pages: Some(vec![40, 41, 40, 41, 40, 41]),
            page_indptr: Some(vec![0, 2, 4, 6]),
            w_slot: Some(vec![41, 41, 41]),
            w_off: Some(vec![6, 7, 8]),
            // [3 rows, 4 keys], row-major.
            mask: Some(vec![
                true, false, false, false, //
                true, true, false, false, //
                true, true, true, false,
            ]),
        };
        assert_eq!(envelope.lanes(), 3);
        assert!(envelope.owns_pages());

        for (at, (token, off)) in [(11u32, 6u32), (22, 7), (33, 8)].into_iter().enumerate() {
            let lane = envelope.lane(at, at).expect("the CSR describes it");
            assert_eq!(lane.rows(), 1);
            assert_eq!(lane.tokens_for(1).expect("one row"), &[token]);
            assert_eq!(lane.extent(), Some(9));
            assert_eq!(lane.pages().expect("a page run"), Some(&[40u32, 41][..]));
            let (slots, offs) = lane.writes(1).expect("a descriptor").expect("bound");
            assert_eq!((slots, offs), (&[41u32][..], &[off][..]));
            let (cells, stride) = lane.mask(1).expect("a rectangle").expect("bound");
            assert_eq!(stride, 4);
            assert_eq!(cells.len(), 4);
            assert_eq!(
                cells.iter().filter(|kept| **kept).count(),
                at + 1,
                "lane {at} keeps its own prefix"
            );
        }
    }

    /// **A LANE WHOSE PAGES ARE THE GUEST'S RENUMBERS FREELY.** The
    /// natural-run check is what ties a decode-envelope lane's positions to
    /// the seat its pages are carved from; a beam's logical position and the
    /// flat pool cell it writes are unrelated by construction, so the check
    /// would be false there and is not made.
    #[test]
    fn positions_are_checked_against_the_seat_only_where_the_seat_owns_the_pages() {
        let carried = |pages: bool| Envelope {
            tokens: Some(vec![5]),
            positions: Some(vec![3]),
            pages: pages.then(|| vec![40]),
            page_indptr: pages.then(|| vec![0, 1]),
            ..Envelope::default()
        };
        // The shell's pages: this fire writes row 20, the port says 3.
        let refusal = carried(false)
            .lane(0, 0)
            .expect("one lane")
            .positions_for(20, 1)
            .expect_err("a decode envelope may not renumber");
        assert!(format!("{refusal}").contains("positions"));
        // The guest's pages: the same 3 is the position that reaches RoPE.
        assert_eq!(
            carried(true)
                .lane(0, 0)
                .expect("one lane")
                .positions_for(20, 1)
                .expect("a device-geometry lane states its own"),
            Some(&[3u32][..])
        );
    }

    /// **THE CLASS DECIDES, AND THE PAGE FAMILY IS WHY IT MATTERS.**
    ///
    /// This is a regression gate with a measured failure behind it on the
    /// sibling plane. Widening the resolver to the whole port table — rather
    /// than to the bound class's set — makes a DECODE-ENVELOPE instance
    /// resolve its own `pages` port, and every guest states that port as
    /// `0 .. reserved` because a guest holds working-set-relative indexes. The
    /// runtime had already translated those for that class; reading them again
    /// took the relative index for a pool id.
    #[test]
    fn a_decode_envelope_resolves_its_three_ports_and_not_the_page_family() {
        for port in [Port::EmbedTokens, Port::Positions, Port::KvLen] {
            assert!(
                super::resolves(GeometryClass::DecodeEnvelope, port),
                "{} is the decode envelope's own",
                port.name()
            );
            assert!(super::resolves(GeometryClass::DeviceGeometry, port));
        }
        for port in [
            Port::Pages,
            Port::PageIndptr,
            Port::WSlot,
            Port::WOff,
            Port::AttnMask,
            Port::EmbedIndptr,
        ] {
            assert!(
                !super::resolves(GeometryClass::DecodeEnvelope, port),
                "{}'s value is the RUNTIME's to resolve for a decode envelope, and \
                 the guest's copy of it is in a different page space",
                port.name()
            );
            assert!(
                super::resolves(GeometryClass::DeviceGeometry, port),
                "{} is the wide class's",
                port.name()
            );
        }
        // And a port nothing on this path reads, in either class.
        for class in [
            GeometryClass::Host,
            GeometryClass::DecodeEnvelope,
            GeometryClass::DeviceGeometry,
        ] {
            assert!(!super::resolves(class, Port::Readout));
        }
    }

    /// Half a page family names no lane's table, and half a write descriptor
    /// addresses no cell. Both are refused by name rather than half-served.
    #[test]
    fn half_a_page_family_and_half_a_write_descriptor_are_refused() {
        let pages_only = Envelope {
            pages: Some(vec![40]),
            ..Envelope::default()
        };
        assert!(pages_only.lane(0, 0).expect("one lane").pages().is_err());
        let slot_only = Envelope {
            tokens: Some(vec![5]),
            w_slot: Some(vec![40]),
            ..Envelope::default()
        };
        assert!(slot_only.lane(0, 0).expect("one lane").writes(1).is_err());
    }

    /// **A PAGE ID OUTSIDE THE LANE'S OWN CSR SPAN IS REFUSED, NOT CLAMPED.**
    /// The CSR is the only thing that says which of a flat run is whose, so a
    /// span running off the end would hand one lane another's pages — which is
    /// the two-spaces failure one level up, reached from inside.
    #[test]
    fn a_page_csr_that_runs_past_its_flat_run_is_refused_by_both_numbers() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 1]),
            tokens: Some(vec![5]),
            pages: Some(vec![40, 41]),
            page_indptr: Some(vec![0, 5]),
            ..Envelope::default()
        };
        let refusal = envelope
            .lane(0, 0)
            .expect("one lane")
            .pages()
            .expect_err("a CSR cutting five pages out of two");
        let text = format!("{refusal}");
        assert!(text.contains("0..5") && text.contains('2'), "{text}");
    }

    /// **THE PACKED CELL, AGAINST THE ENCODER THAT WROTE IT.** A `Bool`
    /// channel's cell is `numel.div_ceil(8)` bit-packed bytes on this plane,
    /// and `eta_exec`'s `encode_wire` is what packs one — LSB-first inside
    /// each byte. That function is private to its crate, so this shell spells
    /// the shift itself, and this is the gate that holds the spelling: the
    /// same bytes, packed here the way the encoder packs them, must come back
    /// as the bools that went in.
    #[test]
    fn a_packed_cell_unpacks_lsb_first() {
        // Eleven lanes: two bytes, the second one three bits wide.
        let lanes: Vec<bool> = (0..11u32).map(|at| at % 3 == 0).collect();
        let mut packed = vec![0u8; lanes.len().div_ceil(8)];
        for (at, &kept) in lanes.iter().enumerate() {
            if kept {
                packed[at / 8] |= 1u8 << (at % 8);
            }
        }
        assert_eq!(packed, vec![0b0100_1001u8, 0b0000_0010]);
        assert_eq!(super::unpack_bits(&packed, lanes.len()), lanes);
        // And the surplus bits of the last byte are not lanes: a reader that
        // took the byte whole would invent five keys per rectangle row.
        assert_eq!(super::unpack_bits(&packed, lanes.len()).len(), 11);
    }

    /// A rectangle that does not divide into the instance's rows has no
    /// reading, and the number it would be tempting to invent — row zero's
    /// mask on every row — is the silent substitution the check exists to end.
    #[test]
    fn a_mask_that_is_not_a_rectangle_over_the_instances_rows_is_refused() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 1, 2]),
            tokens: Some(vec![5, 6]),
            pages: Some(vec![40, 41]),
            page_indptr: Some(vec![0, 1, 2]),
            mask: Some(vec![true, false, true]),
            ..Envelope::default()
        };
        let refusal = envelope
            .lane(0, 0)
            .expect("one lane")
            .mask(1)
            .expect_err("three cells over two rows");
        assert!(format!("{refusal}").contains("not a rectangle"));
    }
}
