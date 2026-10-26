use eta_ir::registry::{GeometryClass, Port};

use crate::fire::{Mask, Masking};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct PortError(pub String);

type Result<T> = std::result::Result<T, PortError>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    pub qo_indptr: Option<Vec<u32>>,
    pub tokens: Option<Vec<u32>>,
    pub positions: Option<Vec<u32>>,
    pub slots: Option<Vec<u32>>,
    pub kv_len: Option<Vec<u32>>,
    pub mask: Option<Vec<bool>>,
    pub fold: Option<Vec<u32>>,
}

impl Envelope {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.qo_indptr.is_none()
            && self.tokens.is_none()
            && self.positions.is_none()
            && self.slots.is_none()
            && self.kv_len.is_none()
            && self.mask.is_none()
            && self.fold.is_none()
    }

    #[must_use]
    pub fn lanes(&self) -> usize {
        match &self.qo_indptr {
            Some(csr) if self.owns_slots() && csr.len() >= 2 => csr.len() - 1,
            _ => 1,
        }
    }

    #[must_use]
    pub fn owns_slots(&self) -> bool {
        self.slots.is_some()
    }

    pub fn lane(&self, at: usize, source: usize) -> Result<LanePorts<'_>> {
        let rows = match &self.qo_indptr {
            Some(csr) if self.owns_slots() && csr.len() >= 2 => {
                let (Some(&start), Some(&end)) = (csr.get(at), csr.get(at + 1)) else {
                    return Err(PortError(format!(
                        "lane {source} is lane {at} of its instance, whose \
                         `embed_indptr` port carries {} bound(s) and therefore \
                         describes {} lane(s)",
                        csr.len(),
                        self.lanes()
                    )));
                };
                if end < start {
                    return Err(PortError(format!(
                        "lane {source}'s `embed_indptr` port states rows \
                         {start}..{end}, which runs backwards"
                    )));
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
            Some(csr) if self.owns_slots() && csr.len() >= 2 => {
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
    pub fn owns_slots(&self) -> bool {
        self.envelope.owns_slots()
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.rows.end.saturating_sub(self.rows.start)).unwrap_or(u32::MAX)
    }

    pub fn tokens_for(&self, rows: usize) -> Result<&[u32]> {
        let source = self.source;
        let Some(tokens) = &self.envelope.tokens else {
            return Err(PortError(format!(
                "lane {source} is bound in a device-resolved geometry class and \
                 its program binds no `embed_tokens` port, so there is nothing on \
                 the device for this fire to embed"
            )));
        };
        let ids = self.slice(tokens, "embed_tokens")?;
        if ids.len() != rows {
            return Err(PortError(format!(
                "lane {source}'s `embed_tokens` port carries {} id(s) and the fire \
                 placed {rows} row(s) for it; the composition has already carved \
                 the arena rectangles and the page CSR at {rows}",
                ids.len()
            )));
        }
        Ok(ids)
    }

    pub fn positions_for(&self, have: u32, rows: usize) -> Result<Option<&[u32]>> {
        let Some(positions) = &self.envelope.positions else {
            // A row's RoPE position is its slot unless the program says so.
            return match &self.envelope.slots {
                Some(slots) => self.slice(slots, "slots").map(Some),
                None => Ok(None),
            };
        };
        let stated = self.slice(positions, "positions")?;
        if self.owns_slots() {
            return Ok(Some(stated));
        }
        let natural = stated.len() == rows
            && stated
                .iter()
                .enumerate()
                .all(|(at, &position)| u64::from(position) == u64::from(have) + at as u64);
        if !natural {
            return Err(PortError(format!(
                "lane {}'s `positions` port carries {stated:?} and this fire \
                 writes rows {have}..{}; the page CSR, the write descriptor and the \
                 attention schedules are carved from the second, so a fire that ran \
                 would rope one extent and attend another",
                self.source,
                u64::from(have) + rows as u64
            )));
        }
        Ok(Some(stated))
    }

    #[must_use]
    pub fn fold(&self) -> Option<u32> {
        self.envelope
            .fold
            .as_ref()
            .and_then(|lens| lens.get(self.at).or_else(|| lens.first()))
            .copied()
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
            return Err(PortError(format!(
                "lane {} states a readable KV extent of {stated} on its `kv_len` \
                 port and this fire reaches {after}; the shell carves the page CSR, \
                 the write descriptor and the attention schedules from {after}, so a \
                 fire that ran would attend a different extent than the guest thinks \
                 it wrote",
                self.source
            )));
        }
        Ok(())
    }

    pub fn slots(&self, rows: usize) -> Result<Option<&[u32]>> {
        let Some(slots) = &self.envelope.slots else {
            return Ok(None);
        };
        let slots = self.slice(slots, "slots")?;
        if slots.len() != rows {
            return Err(PortError(format!(
                "lane {}'s `slots` port carries {} slot(s) for the {rows} row(s) this \
                 fire placed",
                self.source,
                slots.len()
            )));
        }
        Ok(Some(slots))
    }

    pub fn reach(&self, rows: usize) -> Result<Option<u32>> {
        if let Some(extent) = self.extent() {
            return Ok(Some(extent));
        }
        Ok(self
            .slots(rows)?
            .and_then(|slots| slots.iter().max())
            .map(|&last| last + 1))
    }

    pub fn mask(&self, rows: usize) -> Result<Option<(&[bool], usize)>> {
        let source = self.source;
        let Some(dense) = &self.envelope.mask else {
            return Ok(None);
        };
        let spanned = self.envelope.spanned().max(1);
        if dense.len() % spanned != 0 {
            return Err(PortError(format!(
                "lane {source}'s `attn_mask` port carries {} cell(s) over the \
                 {spanned} row(s) its instance spans, which is not a rectangle",
                dense.len()
            )));
        }
        let stride = dense.len() / spanned;
        if stride == 0 {
            return Err(PortError(format!(
                "lane {source}'s `attn_mask` port states a zero-wide key axis"
            )));
        }
        let (start, end) = (self.rows.start * stride, self.rows.end * stride);
        if end > dense.len() || end - start != rows * stride {
            return Err(PortError(format!(
                "lane {source}'s `attn_mask` port holds rows {}..{} of a {}-cell \
                 rectangle {stride} wide, and this fire placed {rows} row(s) for it",
                self.rows.start,
                self.rows.end,
                dense.len()
            )));
        }
        Ok(Some((&dense[start..end], stride)))
    }

    fn slice<'v, T>(&self, flat: &'v [T], port: &str) -> Result<&'v [T]> {
        flat.get(self.rows.clone()).ok_or_else(|| {
            PortError(format!(
                "lane {}'s `{port}` port carries {} entrie(s) and its instance's \
                 token CSR cuts rows {}..{} out of it",
                self.source,
                flat.len(),
                self.rows.start,
                self.rows.end
            ))
        })
    }
}

#[must_use]
pub fn resolves(class: GeometryClass, port: Port) -> bool {
    if port == Port::Fold {
        return true;
    }
    if class.ports().contains(port) {
        return true;
    }
    match class {
        // RoPE positions may come from the device beside slots the host
        // placed, and so may the mask that says what the rows see: one the
        // device computes (beam ancestry) the runtime cannot resolve, for a
        // fire staged ahead or not.
        GeometryClass::DecodeEnvelope => matches!(port, Port::Positions | Port::AttnMask),
        GeometryClass::DeviceGeometry => matches!(port, Port::EmbedIndptr | Port::AttnMask),
        GeometryClass::Host => false,
    }
}

#[must_use]
pub fn from_dense(cells: &[bool], stride: usize) -> Masking {
    let rows = cells.len().checked_div(stride).unwrap_or(0);
    Masking::Rows(
        (0..rows)
            .map(|row| {
                let mut runs: Vec<u32> = Vec::new();
                let mut keeping = false;
                let mut run = 0u32;
                for &kept in &cells[row * stride..(row + 1) * stride] {
                    if kept == keeping {
                        run += 1;
                        continue;
                    }
                    runs.push(run);
                    keeping = kept;
                    run = 1;
                }
                if run != 0 {
                    runs.push(run);
                }
                Mask::new(runs, stride as u64)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::Envelope;
    use eta_ir::registry::Port;

    #[test]
    fn ports_every_case() {
        an_extent_that_disagrees_with_the_seat_is_refused_by_both_numbers();
        an_agreeing_extent_and_an_unbound_one_both_pass();
        positions_are_checked_against_the_seat_only_where_the_seat_owns_the_slots();
        a_slot_run_that_disagrees_with_the_rows_is_refused_by_both_numbers();
        one_instances_flat_vectors_cut_into_its_lanes_by_the_token_csr();
        a_row_is_roped_at_its_slot_unless_the_program_says_otherwise();
        a_decode_envelope_resolves_its_tokens_and_not_where_they_land();
    }

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

    fn positions_are_checked_against_the_seat_only_where_the_seat_owns_the_slots() {
        let carried = |slots: bool| Envelope {
            tokens: Some(vec![5]),
            positions: Some(vec![3]),
            slots: slots.then(|| vec![40]),
            ..Envelope::default()
        };
        let refusal = carried(false)
            .lane(0, 0)
            .expect("one lane")
            .positions_for(20, 1)
            .expect_err("a decode envelope may not renumber");
        assert!(format!("{refusal}").contains("positions"));
        assert_eq!(
            carried(true)
                .lane(0, 0)
                .expect("one lane")
                .positions_for(20, 1)
                .expect("a device-geometry lane states its own"),
            Some(&[3u32][..])
        );
    }

    fn a_slot_run_that_disagrees_with_the_rows_is_refused_by_both_numbers() {
        let envelope = Envelope {
            tokens: Some(vec![5, 6]),
            slots: Some(vec![40, 41]),
            ..Envelope::default()
        };
        let refusal = envelope
            .lane(0, 0)
            .expect("one lane")
            .slots(3)
            .expect_err("two slots for three rows");
        let text = format!("{refusal}");
        assert!(text.contains('2') && text.contains('3'), "{text}");
    }

    fn one_instances_flat_vectors_cut_into_its_lanes_by_the_token_csr() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 1, 2, 3]),
            tokens: Some(vec![11, 22, 33]),
            positions: Some(vec![7, 7, 7]),
            slots: Some(vec![6, 7, 8]),
            kv_len: Some(vec![9, 9, 9]),
            mask: Some(vec![
                true, false, false, false, true, true, false, false, true, true, true, false,
            ]),
            ..Envelope::default()
        };
        assert_eq!(envelope.lanes(), 3);
        assert!(envelope.owns_slots());

        for (at, (token, slot)) in [(11u32, 6u32), (22, 7), (33, 8)].into_iter().enumerate() {
            let lane = envelope.lane(at, at).expect("the CSR describes it");
            assert_eq!(lane.tokens_for(1).expect("one row"), &[token]);
            assert_eq!(lane.extent(), Some(9));
            assert_eq!(lane.slots(1).expect("a slot"), Some(&[slot][..]));
            assert_eq!(lane.reach(1).expect("a reach"), Some(9));
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

    fn a_row_is_roped_at_its_slot_unless_the_program_says_otherwise() {
        let envelope = Envelope {
            qo_indptr: Some(vec![0, 2]),
            tokens: Some(vec![1, 2]),
            slots: Some(vec![40, 41]),
            ..Envelope::default()
        };
        let lane = envelope.lane(0, 0).expect("one lane");
        assert_eq!(
            lane.positions_for(40, 2).expect("positions"),
            Some(&[40u32, 41][..])
        );
        assert_eq!(lane.reach(2).expect("a reach"), Some(42));
    }

    /// A decode envelope resolves what its rows are and what they see (its
    /// tokens, positions and a mask the device computes; a mask the runtime
    /// resolved rides the lane and wins) and never where they land, which
    /// stays the runtime's.
    fn a_decode_envelope_resolves_its_tokens_and_not_where_they_land() {
        use eta_ir::registry::GeometryClass;
        for port in [Port::EmbedTokens, Port::Positions, Port::AttnMask] {
            assert!(super::resolves(GeometryClass::DecodeEnvelope, port));
        }
        for port in [Port::Slots, Port::KvLen, Port::EmbedIndptr] {
            assert!(
                !super::resolves(GeometryClass::DecodeEnvelope, port),
                "{}'s value is the RUNTIME's to resolve for a decode envelope",
                port.name()
            );
            assert!(
                super::resolves(GeometryClass::DeviceGeometry, port),
                "{} is the wide class's",
                port.name()
            );
        }
        for class in [GeometryClass::DecodeEnvelope, GeometryClass::DeviceGeometry] {
            assert!(super::resolves(class, Port::Fold));
            assert!(!super::resolves(class, Port::Readout));
        }
    }
}
