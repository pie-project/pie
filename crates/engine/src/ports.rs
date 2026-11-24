pub use eta_ir::registry::GeometryClass;
use eta_ir::registry::{Port, PortMask};

use crate::fire::{Mask, Masking};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct PortError(pub String);

type Result<T> = std::result::Result<T, PortError>;

/// The id a program puts on its `embed_tokens` port for a row it skips: −1
/// as an `i32` cell, `u32::MAX` as the engine reads it. A beam that has
/// finished, or a lane the program idles this step, keeps its place in the
/// fire's geometry without embedding anything. Every engine that resolves
/// `embed_tokens` on the device carries a skip row the same way:
///
/// 1. **No embed.** No token is embedded for it: the engine reads no row of
///    the token table for it (a skip id is never clamped or wrapped into
///    the vocabulary, so it never reads row 0 or any other).
/// 2. **No kv write.** Its key/value write is suppressed: the slot it
///    names (or would append to) is left exactly as it was.
/// 3. **Attended by no row.** No row of the fire attends to it, its own
///    row included: its slot is masked out of every row's attention as if
///    the mask cell were false.
/// 4. **No readout.** It produces no readout row. A readout that names a
///    skip row is refused by name for that lane, never answered with
///    garbage logits.
/// 5. **Its position counts.** It still occupies its row in the lane's run:
///    later rows of the lane keep their positions, slots and causal bounds
///    as if it were a real row.
///
/// Any other id at or past the vocabulary is out of range, and the lane is
/// refused by name. An engine that cannot yet carry skip rows refuses the
/// lane by name, naming the skip.
pub const SKIP_TOKEN: u32 = u32::MAX;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    pub qo_indptr: Option<Vec<u32>>,
    pub tokens: Option<Vec<u32>>,
    pub positions: Option<Vec<u32>>,
    pub slots: Option<Vec<u32>>,
    pub kv_len: Option<Vec<u32>>,
    pub mask: Option<Vec<bool>>,
    pub fold: Option<Vec<u32>>,
    /// Rows the fire placed for each of the attachment's lanes; cuts the vectors of an
    /// envelope that places no slots of its own.
    pub host_rows: Vec<u32>,
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
            _ => self.host_rows.len().max(1),
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
            _ if self.host_rows.is_empty() => 0..self.tokens.as_ref().map_or(0, Vec::len),
            _ => {
                let placed = rows_sum(&self.host_rows).ok_or_else(|| {
                    PortError(format!(
                        "lane {source}'s instance was placed rows past a u32 across its {} \
                         lane(s)",
                        self.host_rows.len()
                    ))
                })?;
                if let Some(tokens) = &self.tokens
                    && tokens.len() != placed as usize
                {
                    return Err(PortError(format!(
                        "lane {source}'s `embed_tokens` port carries {} id(s) and the fire \
                         placed {placed} row(s) across the {} lane(s) of its instance",
                        tokens.len(),
                        self.host_rows.len()
                    )));
                }
                let Some(&rows) = self.host_rows.get(at) else {
                    return Err(PortError(format!(
                        "lane {source} is lane {at} of its instance, which the fire placed \
                         {} lane(s) for",
                        self.host_rows.len()
                    )));
                };
                // `placed` summed every lane's rows: no prefix of them is past it.
                let start = rows_sum(&self.host_rows[..at]).unwrap_or(placed);
                let end = start.checked_add(rows).ok_or_else(|| {
                    PortError(format!(
                        "lane {source}'s rows {start}+{rows} run past a u32"
                    ))
                })?;
                start as usize..end as usize
            }
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
            _ if !self.host_rows.is_empty() => self
                .host_rows
                .iter()
                .map(|&rows| rows as usize)
                .fold(0usize, usize::saturating_add),
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

    /// The fold length the `fold` port states for this lane: a one-entry
    /// list states it for every lane, a longer one only its own entry, or
    /// none past its end. Never another lane's entry of a per-lane list,
    /// which would be a wrong count told as this lane's own.
    #[must_use]
    pub fn fold(&self) -> Option<u32> {
        match self.envelope.fold.as_deref()? {
            [every] => Some(*every),
            lens => lens.get(self.at).copied(),
        }
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
        let Some(&last) = self.slots(rows)?.and_then(|slots| slots.iter().max()) else {
            return Ok(None);
        };
        last.checked_add(1).map(Some).ok_or_else(|| {
            PortError(format!(
                "lane {}'s `slots` port names slot {last}, past which no slot is counted \
                 in a u32",
                self.source
            ))
        })
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
        let past = || {
            PortError(format!(
                "lane {source}'s `attn_mask` port rows {}..{} at a stride of {stride} \
                 run past the address space",
                self.rows.start, self.rows.end
            ))
        };
        let start = self.rows.start.checked_mul(stride).ok_or_else(past)?;
        let end = self.rows.end.checked_mul(stride).ok_or_else(past)?;
        let want = rows.checked_mul(stride).ok_or_else(past)?;
        if end > dense.len() || end.checked_sub(start) != Some(want) {
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

/// The rows `lanes` were placed, or `None` past a u32.
fn rows_sum(lanes: &[u32]) -> Option<u32> {
    lanes.iter().try_fold(0u32, |sum, &rows| sum.checked_add(rows))
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
        // So may each beam's parent, over which the engine keeps the beams'
        // own page lists (`new-memory/designs/BEAM-PAGE-LISTS.md`).
        GeometryClass::DecodeEnvelope => matches!(
            port,
            Port::Positions | Port::AttnMask | Port::Parents
        ),
        GeometryClass::DeviceGeometry => matches!(port, Port::EmbedIndptr | Port::AttnMask),
        GeometryClass::Host => false,
    }
}

/// Whether the engine resolves `port` for an instance bound in `class`
/// whose program computes `device` on the device. A decode envelope's other
/// ports are resolved on the host and stated on the lanes, so each port has
/// one source: the engine never reads a cell the runtime already resolved.
#[must_use]
pub fn engine_resolves(class: GeometryClass, device: PortMask, port: Port) -> bool {
    resolves(class, port) && (class != GeometryClass::DecodeEnvelope || device.contains(port))
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
        an_envelope_placing_no_slots_cuts_by_the_host_rows();
        a_row_is_roped_at_its_slot_unless_the_program_says_otherwise();
        a_decode_envelope_resolves_its_tokens_and_not_where_they_land();
        an_engine_resolves_only_what_the_program_computes_on_the_device();
        a_one_entry_fold_states_every_lane_and_a_longer_one_its_own();
        sizes_past_their_arithmetic_are_refused_by_name();
    }

    /// A guest's slot `u32::MAX` and host rows that sum past a u32 are each
    /// refused by name, never wrapped into a small number.
    fn sizes_past_their_arithmetic_are_refused_by_name() {
        let last = Envelope {
            qo_indptr: Some(vec![0, 1]),
            tokens: Some(vec![7]),
            slots: Some(vec![u32::MAX]),
            ..Envelope::default()
        };
        let refused = last.lane(0, 0).unwrap().reach(1).unwrap_err().to_string();
        assert!(refused.contains("past which no slot"), "{refused}");

        let rows = Envelope {
            host_rows: vec![u32::MAX, 2],
            ..Envelope::default()
        };
        let refused = rows.lane(1, 1).unwrap_err().to_string();
        assert!(refused.contains("past a u32"), "{refused}");
    }

    /// A one-entry `fold` port states the fold for every lane; a longer one
    /// only each lane's own entry, and none past its end (never a
    /// neighbour's entry, a wrong count told as the lane's own).
    fn a_one_entry_fold_states_every_lane_and_a_longer_one_its_own() {
        let three = |fold: Vec<u32>| Envelope {
            qo_indptr: Some(vec![0, 1, 2, 3]),
            tokens: Some(vec![11, 22, 33]),
            slots: Some(vec![40, 41, 42]),
            fold: Some(fold),
            ..Envelope::default()
        };
        let folds = |envelope: &Envelope| -> Vec<Option<u32>> {
            (0..3)
                .map(|at| envelope.lane(at, at).expect("a lane").fold())
                .collect()
        };
        assert_eq!(
            folds(&three(vec![4])),
            [Some(4); 3],
            "one entry, every lane"
        );
        assert_eq!(
            folds(&three(vec![4, 5])),
            [Some(4), Some(5), None],
            "a per-lane list: its own entry, none past the end"
        );
        assert_eq!(folds(&three(vec![4, 5, 6])), [Some(4), Some(5), Some(6)]);
        assert_eq!(
            folds(&Envelope {
                fold: None,
                ..three(vec![])
            }),
            [None; 3]
        );
    }

    /// One source a port: what a decode envelope's program computes on the
    /// device the engine resolves, and what the runtime resolved it never
    /// reads; the wide class resolves every port it has.
    fn an_engine_resolves_only_what_the_program_computes_on_the_device() {
        use eta_ir::registry::{GeometryClass, PortMask};
        let device = PortMask::of(&[Port::EmbedTokens, Port::AttnMask]);
        let envelope = |port| super::engine_resolves(GeometryClass::DecodeEnvelope, device, port);
        assert!(envelope(Port::EmbedTokens) && envelope(Port::AttnMask));
        assert!(
            !envelope(Port::Positions),
            "positions the runtime resolved are the lane's"
        );
        assert!(!envelope(Port::Fold) && !envelope(Port::Slots));
        assert!(super::engine_resolves(
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            Port::Slots
        ));
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

    fn an_envelope_placing_no_slots_cuts_by_the_host_rows() {
        let envelope = Envelope {
            tokens: Some(vec![11, 22]),
            host_rows: vec![1, 1],
            ..Envelope::default()
        };
        assert_eq!(envelope.lanes(), 2);
        assert_eq!(envelope.lane(1, 1).unwrap().tokens_for(1).unwrap(), &[22]);
        let short = Envelope {
            host_rows: vec![1, 2],
            ..envelope
        };
        assert!(short.lane(0, 0).is_err());
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
