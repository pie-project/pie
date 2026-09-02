//! Descriptor-port resolution, Metal half: reads ports from a guest
//! instance's committed device-ring cells instead of the submission.

use std::sync::atomic::{AtomicU64, Ordering};

use eta_exec::{ExecPlan, Value};
use eta_ir::Dtype;
use eta_ir::registry::{GeometryClass, Port};
use eta_ir::types::name_or_unknown;

use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor, Rings};

/// Envelopes resolved, one per attached device-carried lane per fire.
/// Process-global so a test can observe it.
static RESOLVED: AtomicU64 = AtomicU64::new(0);

/// How many descriptor-port envelopes this process has resolved. See [`RESOLVED`].
#[must_use]
pub fn resolved() -> u64 {
    RESOLVED.load(Ordering::Relaxed)
}

/// What one instance's descriptor ports resolved to, this fire. `None`
/// means the program binds no such port. Vectors are per-instance, not
/// per-lane; [`Envelope::qo_indptr`] is the CSR [`Envelope::lane`] cuts them by.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    /// [`Port::EmbedIndptr`]: `[lanes + 1]` row bounds cutting the flat vectors into lanes.
    pub qo_indptr: Option<Vec<u32>>,
    /// [`Port::EmbedTokens`]: ids this instance embeds, all lanes end to end.
    pub tokens: Option<Vec<u32>>,
    /// [`Port::Positions`]: each id's position in its sequence.
    pub positions: Option<Vec<u32>>,
    /// [`Port::KvLen`]: each lane's readable extent after this fire's writes land.
    pub kv_len: Option<Vec<u32>>,
    /// [`Port::Pages`]: page ids every lane may address, cut by
    /// [`Envelope::page_indptr`]. In guest (working-set-relative) space
    /// until `serve::prepare` translates it.
    pub pages: Option<Vec<u32>>,
    /// [`Port::PageIndptr`]: `[lanes + 1]` bounds cutting [`Envelope::pages`].
    pub page_indptr: Option<Vec<u32>>,
    /// [`Port::WSlot`]: the page each token row is appended into, same guest
    /// space as [`Envelope::pages`].
    pub w_slot: Option<Vec<u32>>,
    /// [`Port::WOff`]: that row's offset inside that page.
    pub w_off: Option<Vec<u32>>,
    /// [`Port::AttnMask`]: dense `[rows, keys]` bool rectangle, row-major,
    /// at the pool's key width (may be wider than the extent).
    pub mask: Option<Vec<bool>>,
    /// [`Port::RsFoldLen`]: how many buffered recurrent tokens each lane
    /// folds this fire (`crate::rs`); `[lanes]`, or one entry for every lane.
    pub fold_len: Option<Vec<u32>>,
}

impl Envelope {
    /// True when nothing was bound.
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

    /// Whether this instance states its own page table (device-geometry
    /// lanes do; decode-envelope lanes don't).
    #[must_use]
    pub fn owns_pages(&self) -> bool {
        self.pages.is_some() && self.page_indptr.is_some()
    }

    /// How many lanes this instance carries: 1 unless it states its own
    /// page table (device-geometry class only).
    #[must_use]
    pub fn lanes(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_pages() && csr.len() >= 2 => csr.len() - 1,
            _ => 1,
        }
    }

    /// Lane `at` of this instance's ports, for submission lane `source`
    /// (named in refusals).
    ///
    /// # Errors
    ///
    /// A lane past the CSR, or a CSR that runs backwards.
    pub fn lane(&self, at: usize, source: usize) -> Result<LanePorts<'_>> {
        let rows = match &self.qo_indptr {
            // CSR cuts the flat vectors only for classes with their own page table.
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

/// One lane's share of an instance's resolved descriptor ports.
/// Every accessor cuts by this lane's CSR span and refuses rather than clamps.
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

    /// How many token rows this lane carries, per its instance's token CSR.
    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.rows.end.saturating_sub(self.rows.start)).unwrap_or(u32::MAX)
    }

    /// This lane's token ids, checked against its placed row count.
    ///
    /// # Errors
    ///
    /// Wrong id count, or a program that binds no token port.
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

    /// This lane's positions, checked against `have .. have + rows` unless
    /// this lane owns its own pages (device-geometry). No `positions` port
    /// defaults to the natural run.
    ///
    /// # Errors
    ///
    /// A stated run that disagrees, on a lane whose pages are this shell's.
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

    /// This lane's device-decided recurrent fold length, or `None` if the
    /// program binds no `rs_fold_len` port.
    #[must_use]
    pub fn fold_len(&self) -> Option<u32> {
        self.envelope
            .fold_len
            .as_ref()
            .and_then(|lens| lens.get(self.at).or_else(|| lens.first()))
            .copied()
    }

    /// This lane's stated readable extent, or `None` if unbound.
    #[must_use]
    pub fn extent(&self) -> Option<u32> {
        // A `[lanes]` cell states one entry per lane; falls back to the first.
        self.envelope
            .kv_len
            .as_ref()
            .and_then(|lens| lens.get(self.at).or_else(|| lens.first()))
            .copied()
    }

    /// Checks this lane's stated extent against what the seat will reach.
    ///
    /// # Errors
    ///
    /// A stated extent that disagrees.
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

    /// This lane's page table, in guest space, or `None` if unbound.
    ///
    /// # Errors
    ///
    /// One of `pages`/`page_indptr` bound without the other, a CSR shorter
    /// than the instance's lanes, or a span the flat run doesn't cover.
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

    /// This lane's explicit write descriptor `(page, offset)` per row, or
    /// `None` when unbound and the seat's `have + row` arithmetic stands.
    ///
    /// # Errors
    ///
    /// One of `w_slot`/`w_off` bound without the other, or wrong length.
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

    /// This lane's dense attention mask: `rows` rectangles of `stride`
    /// bools, row-major, or `None` if unbound. Stride is the guest's build
    /// width; clipping surplus is [`crate::mask`]'s rule.
    ///
    /// # Errors
    ///
    /// A rectangle that doesn't divide into the instance's rows, or a span
    /// it doesn't cover.
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

    /// This lane's share of a member-wide per-row vector.
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
/// # Errors
///
/// A port names a channel the instance lacks, holds a non-integer (or, for
/// [`Port::AttnMask`], non-bool) cell, an empty ring, or a const port with
/// no folded value.
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
        // The mask is not an index vector, so it is read on its own arm.
        if binding.port == Port::AttnMask {
            if binding.is_const {
                // Already lowered to run-length `Masking` on the submission.
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
            Port::RsFoldLen => &mut out.fold_len,
            // Anything else has no reader on this path.
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

/// Whether an instance bound in `class` resolves `port` off its own rings.
/// Adds `EmbedIndptr`/`AttnMask` for the wide class only; a guest's own
/// `pages`/`page_indptr` is working-set-relative and isn't read for a
/// decode-envelope pass.
#[must_use]
pub fn resolves(class: GeometryClass, port: Port) -> bool {
    if class.ports().contains(port) {
        return true;
    }
    (matches!(port, Port::EmbedIndptr | Port::AttnMask) && class == GeometryClass::DeviceGeometry)
        || (port == Port::RsFoldLen
            && matches!(class, GeometryClass::DeviceGeometry | GeometryClass::DecodeEnvelope))
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
    // `head`: the committed cell is the one the guest's pass takes this fire.
    let cell = rings.read_cell(channel as usize, cursor.head)?;
    Ok(cell
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

/// One port's committed cell, as a dense rectangle of bools. Bit-packed
/// LSB-first (`numel.div_ceil(8)` bytes), matching `eta_exec::encode_wire`.
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

/// `numel` bool lanes out of packed bytes, LSB-first within each byte.
fn unpack_bits(packed: &[u8], numel: usize) -> Vec<bool> {
    (0..numel)
        .map(|lane| packed[lane / 8] >> (lane % 8) & 1 != 0)
        .collect()
}

/// The cursor and shape one port's channel stands at. An empty ring
/// (`head == tail`) is a refusal, not a zero — avoids embedding garbage.
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
        // Reinterpreted, not converted: -1 is a skip sentinel, not row zero.
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

    

    /// Equality check against the seat; the message names both numbers.
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

    /// Agreement is silent; so is an unbound `kv_len` port (not an omission).
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

    /// Positions are checked against the seat only when the seat owns pages.
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

    /// Half a page family or half a write descriptor is refused, not half-served.
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

    /// A page id outside the lane's CSR span is refused, not clamped.
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

}
