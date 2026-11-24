//! Windowed layers as ring + buffer (CONTRACT-BC 19, 21). The wafer program
//! attends a windowed layer over its own on-chip ring of the seated
//! sequence's last W rows, and rebuilds a lane's cache from its tokens; so
//! here, as a kv page holds its rows' tokens, a lane's record holds each
//! windowed layer's ring of its last W folded rows' tokens (row `r` at place
//! `r mod W`) and the window buffer its unfolded rows' tokens. A fire's
//! fold moves rows from the buffer (and the fire's own rows) into the
//! rings; the rings copy, park and restore with the record.

use std::collections::HashMap;

use engine::error::{Error, Result as EngineResult};
use engine::fire::{FoldLen, Lane, LaneRecord};

use super::pool::{BUFFER, RECORD};

#[derive(Default)]
pub(super) struct Windows {
    /// Each windowed layer's window, in the record's ring order.
    windows: Vec<u32>,
    /// Rows a window-buffer slot holds.
    page_size: u32,
    /// Record slot ref -> each windowed layer's ring (`None`: never folded).
    rings: HashMap<u32, Vec<Vec<Option<u32>>>>,
    /// Window-buffer slot ref -> the token at each row (`None`: never written).
    buffer: HashMap<u32, Vec<Option<u32>>>,
}

impl Windows {
    pub(super) fn new(windows: Vec<u32>, page_size: u32) -> Windows {
        Windows {
            windows,
            page_size: page_size.max(1),
            ..Windows::default()
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// The fire's window-buffer slot copies, every lane's, before any lane
    /// reads or writes.
    pub(super) fn copy_slots(&mut self, lanes: &[Lane]) -> EngineResult<()> {
        for (shape, pairs) in lanes.iter().flat_map(|lane| &lane.slot_copies) {
            if *shape != BUFFER {
                return Err(Error::Invalid(format!(
                    "a slot copy of shape {shape}: the spatial engine's only list a lane \
                     shares partly written is the window buffer ({BUFFER})"
                )));
            }
            for &[src, dst] in pairs {
                match self.buffer.get(&src).cloned() {
                    Some(cells) => self.buffer.insert(dst, cells),
                    None => self.buffer.remove(&dst),
                };
            }
        }
        Ok(())
    }

    /// The lane's record copy-on-write, then its fold: rows `folded..` by
    /// its window's fold move into its rings (from the buffer, or `tokens`,
    /// its own rows at `start..`), and its own rows past the fold go to the
    /// buffer. Checked whole before anything changes.
    pub(super) fn fold(&mut self, lane: &Lane, start: u32, tokens: &[u32]) -> EngineResult<()> {
        if self.is_empty() || lane.kv_less {
            return Ok(());
        }
        let Some(window) = &lane.window else {
            return Err(Error::Invalid(
                "a lane of a windowed load states no window fold".into(),
            ));
        };
        let FoldLen::Host(fold) = window.fold else {
            return Err(Error::unsupported(
                "cerebras-spatial",
                "FoldLen::Device: the spatial engine counts no fold on the device",
            ));
        };
        let record = match &lane.record {
            LaneRecord::Pieces(pieces) => pieces
                .iter()
                .find(|(shape, _)| *shape == RECORD)
                .and_then(|(_, refs)| refs.first().copied())
                .map(Some)
                .ok_or_else(|| {
                    Error::Invalid(format!("a lane's record names no slot of shape {RECORD}"))
                })?,
            LaneRecord::Null => None,
            LaneRecord::Unstated => {
                return Err(Error::Invalid(
                    "a lane of a windowed load states no record for its rings".into(),
                ));
            }
        };
        let folded = lane.kv.folded;
        let after = folded.saturating_add(fold);
        let end = start + tokens.len() as u32;
        if start < folded || after > end {
            return Err(Error::Invalid(format!(
                "a lane writing rows {start}..{end} folds rows {folded}..{after}: its rows \
                 start at or past its folded index and its fold ends within them"
            )));
        }
        let table: &[u32] = lane
            .tables
            .iter()
            .find(|(shape, _)| *shape == BUFFER)
            .map_or(&[], |(_, refs)| refs.as_slice());
        let size = self.page_size;
        let base = window.base;
        let place = |row: u32| -> EngineResult<(u32, usize)> {
            row.checked_sub(base)
                .and_then(|at| {
                    table
                        .get((at / size) as usize)
                        .map(|&slot| (slot, (at % size) as usize))
                })
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "row {row} has no window-buffer slot in a table of {} from row {base}",
                        table.len()
                    ))
                })
        };
        let folding: Vec<(u32, u32)> = (folded..after)
            .map(|row| {
                if row >= start {
                    return Ok((row, tokens[(row - start) as usize]));
                }
                let (slot, at) = place(row)?;
                self.buffer
                    .get(&slot)
                    .and_then(|cells| cells[at])
                    .map(|token| (row, token))
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "buffered row {row} (window-buffer slot {slot}) holds no token"
                        ))
                    })
            })
            .collect::<EngineResult<_>>()?;
        let buffering: Vec<(u32, usize, u32)> = (after.max(start)..end)
            .map(|row| place(row).map(|(slot, at)| (slot, at, tokens[(row - start) as usize])))
            .collect::<EngineResult<_>>()?;
        if let Some([run, cow]) = record {
            if cow != run {
                match self.rings.get(&run).cloned() {
                    Some(rings) => self.rings.insert(cow, rings),
                    None => self.rings.remove(&cow),
                };
            }
            if !folding.is_empty() {
                let rings = self.rings.entry(cow).or_insert_with(|| {
                    self.windows
                        .iter()
                        .map(|&w| vec![None; w as usize])
                        .collect()
                });
                for (ring, &w) in rings.iter_mut().zip(&self.windows) {
                    for &(row, token) in &folding {
                        ring[(row % w) as usize] = Some(token);
                    }
                }
            }
        }
        for (slot, at, token) in buffering {
            self.buffer
                .entry(slot)
                .or_insert_with(|| vec![None; size as usize])[at] = Some(token);
        }
        Ok(())
    }
}
