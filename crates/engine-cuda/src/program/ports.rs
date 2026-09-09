use std::sync::atomic::{AtomicU64, Ordering};

use eta_exec::{ExecPlan, Value};
use eta_ir::Dtype;
use eta_ir::registry::{GeometryClass, Port};
use eta_ir::types::name_or_unknown;

use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor, Rings};

static RESOLVED: AtomicU64 = AtomicU64::new(0);

#[must_use]
pub fn resolved() -> u64 {
    RESOLVED.load(Ordering::Relaxed)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    pub qo_indptr: Option<Vec<u32>>,
    pub tokens: Option<Vec<u32>>,
    pub positions: Option<Vec<u32>>,
    pub kv_len: Option<Vec<u32>>,
    pub pages: Option<Vec<u32>>,
    pub page_indptr: Option<Vec<u32>>,
    pub w_slot: Option<Vec<u32>>,
    pub w_off: Option<Vec<u32>>,
    pub mask: Option<Vec<bool>>,
    pub fold_len: Option<Vec<u32>>,
}

impl Envelope {
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
            && self.fold_len.is_none()
    }

    #[must_use]
    pub fn lanes(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_pages() && csr.len() >= 2 => csr.len() - 1,
            _ => 1,
        }
    }

    #[must_use]
    pub fn owns_pages(&self) -> bool {
        self.pages.is_some() && self.page_indptr.is_some()
    }

    pub fn lane(&self, at: usize, source: usize) -> Result<LanePorts<'_>> {
        let rows = match &self.qo_indptr {
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
            _ => 0..self.tokens.as_ref().map_or(0, Vec::len),
        };
        Ok(LanePorts {
            envelope: self,
            at,
            rows,
            source,
        })
    }

    fn spanned(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_pages() && csr.len() >= 2 => {
                csr.last().copied().unwrap_or(0) as usize
            }
            _ => self.tokens.as_ref().map_or(0, Vec::len),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LanePorts<'a> {
    envelope: &'a Envelope,
    at: usize,
    rows: std::ops::Range<usize>,
    source: usize,
}

impl LanePorts<'_> {
    #[must_use]
    pub fn owns_pages(&self) -> bool {
        self.envelope.owns_pages()
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.rows.end.saturating_sub(self.rows.start)).unwrap_or(u32::MAX)
    }

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

    #[must_use]
    pub fn extent(&self) -> Option<u32> {
        self.envelope
            .kv_len
            .as_ref()
            .and_then(|lens| lens.get(self.at).or_else(|| lens.first()))
            .copied()
    }

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
        if binding.port == Port::AttnMask {
            if binding.is_const {
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

#[must_use]
pub fn resolves(class: GeometryClass, port: Port) -> bool {
    if port == Port::RsFoldLen {
        return true;
    }
    if class.ports().contains(port) {
        return true;
    }
    matches!(port, Port::EmbedIndptr | Port::AttnMask)
        && class == GeometryClass::DeviceGeometry
}

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
    let native = rings.read_cell(index, cursor.head)?;
    Ok(native
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn read_bool_cell(
    port: Port,
    channel: u32,
    rings: &Rings,
    cursors: &[Cursor],
    shapes: &[ChannelShape],
) -> Result<Vec<bool>> {
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
    if shape.dtype != Dtype::Bool {
        return Err(Fault::program(
            "program::ports",
            format!(
                "port {}'s channel {channel} holds {}, and an attention mask is a \
                 rectangle of KEPT/DROPPED and not a rectangle of numbers",
                port.name(),
                name_or_unknown(shape.dtype)
            ),
        ));
    }
    let native = rings.read_cell(index, cursor.head)?;
    Ok(native.into_iter().map(|cell| cell != 0).collect())
}

fn as_u32(port: Port, value: &Value) -> Result<Vec<u32>> {
    match value {
        Value::U32(lanes) => Ok(lanes.clone()),
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
    use eta_ir::registry::Port;

    #[test]
    fn ports_every_case() {
        one_instances_flat_vectors_cut_into_its_lanes_by_the_token_csr();
        a_decode_envelope_resolves_its_three_ports_and_not_the_page_family();
    }

    fn one_instances_flat_vectors_cut_into_its_lanes_by_the_token_csr() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 1, 2, 3]),
            tokens: Some(vec![11, 22, 33]),
            positions: Some(vec![7, 7, 7]),
            kv_len: Some(vec![9, 9, 9]),
            pages: Some(vec![40, 41, 40, 41, 40, 41]),
            page_indptr: Some(vec![0, 2, 4, 6]),
            w_slot: Some(vec![41, 41, 41]),
            w_off: Some(vec![6, 7, 8]),
            mask: Some(vec![
                true, false, false, false,
                true, true, false, false,
                true, true, true, false,
            ]),
            ..Envelope::default()
        };
        assert_eq!(envelope.lanes(), 3);
        assert!(envelope.owns_pages());

        for (at, (token, off)) in [(11u32, 6u32), (22, 7), (33, 8)].into_iter().enumerate() {
            let lane = envelope.lane(at, at).expect("the CSR describes it");
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

    fn a_decode_envelope_resolves_its_three_ports_and_not_the_page_family() {
        use eta_ir::registry::GeometryClass;
        for port in [Port::EmbedTokens, Port::Positions, Port::KvLen] {
            assert!(
                super::resolves(GeometryClass::DecodeEnvelope, port),
                "{} is the decode envelope's own",
                port.name()
            );
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
        for class in [
            GeometryClass::DecodeEnvelope,
            GeometryClass::DeviceGeometry,
        ] {
            assert!(super::resolves(class, Port::RsFoldLen));
        }
        for class in [
            GeometryClass::DecodeEnvelope,
            GeometryClass::DeviceGeometry,
        ] {
            assert!(!super::resolves(class, Port::Readout));
        }
    }

}
