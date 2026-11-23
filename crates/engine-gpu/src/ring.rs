//! A fire's windowed layers as state + buffer (CONTRACT-BC 19, 21), host
//! side: where each lane's keys lie for the ring kernels
//! (`kernels_slang::attn::sdpa` with a [`kernels_slang::Ring`]), where its
//! rows of the fire land, and the jobs of the fold
//! (`kernels_slang::attn::ring::fold`).
//!
//! A lane of the fire reads keys below its folded index from the ring in
//! its record, keys from there to its first row of the fire from its
//! window-buffer slots (entry `i` rows `[base + i T, base + (i + 1) T)`),
//! and its own rows from the fire's slots: scratch laid as buffer slots,
//! the lane's first row of the fire at row 0 of its first. Every row of
//! the fire lands in its fire slot (`kv_append`), so the fire's reads of
//! its own rows never touch a buffer slot. The fold, after the layer's last
//! read, moves rows `[folded, folded')` (the last `R` of them) into the
//! ring and the fire's rows past `folded'` into the lane's buffer slots.

use crate::error::{Fault, Result};

/// One lane of a fire as the ring kernels see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane<'a> {
    /// The lane's folded index before the fire.
    pub folded: u32,
    /// Rows this fire folds.
    pub fold: u32,
    /// The lane's buffer table's first row (a multiple of the slot's rows).
    pub base: u32,
    /// The lane's first row of the fire, and its rows.
    pub first: u32,
    pub rows: u32,
    /// The lane's window-buffer slots as kernel units, from `base`.
    pub buffer: &'a [u32],
    /// The fire's slots of the lane's rows, as kernel units.
    pub fire: &'a [u32],
    /// Which lane of the fire's record table the lane's ring is, and its
    /// record's units.
    pub record: u32,
    pub units: &'a [i32],
    /// Whether the lane keeps its rows: a kv-less lane's rows land in its
    /// fire slots and fold nowhere.
    pub keeps: bool,
}

/// A lane of a fire as the shell seats it, before its fire slots are
/// known ([`Lane`] once they are).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seat {
    pub folded: u32,
    pub fold: u32,
    pub base: u32,
    pub first: u32,
    pub rows: u32,
    pub buffer: Vec<u32>,
    pub record: u32,
    pub keeps: bool,
}

impl Seat {
    /// The lane with its fire slots and its record's units.
    #[must_use]
    pub fn with<'a>(&'a self, fire: &'a [u32], units: &'a [i32]) -> Lane<'a> {
        Lane {
            folded: self.folded,
            fold: self.fold,
            base: self.base,
            first: self.first,
            rows: self.rows,
            buffer: &self.buffer,
            fire,
            record: self.record,
            units,
            keeps: self.keeps,
        }
    }
}

/// What a fire stages for its windowed layers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    /// `4 + pieces` ints a lane: folded, base, first, folded after the
    /// fire, its record's units (`Ring::lanes`).
    pub lanes: Vec<i32>,
    /// The page table: a lane's buffer slots of its rows below its first,
    /// then its fire slots.
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    /// Each row of the fire's fire slot and its row there (`kv_append`'s
    /// write descriptors).
    pub write_page: Vec<i32>,
    pub write_offset: Vec<i32>,
    /// The fold's job table: its count, then four ints a job.
    pub jobs: Vec<i32>,
}

/// The fire's slots its lanes' rows take, `tokens` rows a slot: a lane's
/// rows from row 0 of a slot of its own.
#[must_use]
pub fn fire_slots(rows: &[u32], tokens: u32) -> u32 {
    rows.iter().map(|&r| r.div_ceil(tokens.max(1))).sum()
}

/// The most fold jobs a fire of `rows` rows (summed over its lanes, its
/// carried rows among them) takes: a row folds into the ring or lands in a
/// buffer slot, never both.
#[must_use]
pub fn most_jobs(rows: u32, carried: u32) -> u32 {
    rows.saturating_add(carried)
}

/// Stages `lanes` for rings of at most `ring_rows` rows (a narrower layer's
/// fold drops the rows past its own), slots of `tokens` rows.
pub fn stage(lanes: &[Lane<'_>], tokens: u32, ring_rows: u32) -> Result<Staged> {
    let bad = |why: String| Fault::Program {
        at: "ring::stage",
        why,
    };
    if tokens == 0 || ring_rows == 0 {
        return Err(bad(format!(
            "rings of {ring_rows} rows in slots of {tokens}"
        )));
    }
    let mut out = Staged {
        indptr: vec![0],
        jobs: vec![0],
        ..Staged::default()
    };
    // Each destination ring row once: lanes of a group naming one record
    // fold it once.
    let mut landed: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for (at, lane) in lanes.iter().enumerate() {
        let to = lane.folded.saturating_add(lane.fold);
        if lane.base % tokens != 0 || lane.first < lane.folded {
            return Err(bad(format!(
                "lane {at} buffers from row {} with slots of {tokens}, its first row {} \
                 below its folded index {}",
                lane.base, lane.first, lane.folded
            )));
        }
        let end = lane.first.saturating_add(lane.rows);
        if to > end {
            return Err(bad(format!(
                "lane {at} folds to row {to}, past its last row {end}"
            )));
        }
        // Its buffered rows read, `[folded, first)`, lie in its buffer from
        // `base`; the rows of the fire past `to` land there.
        if lane.first > lane.folded && lane.base > lane.folded {
            return Err(bad(format!(
                "lane {at} carries rows from {} and its buffer starts at {}",
                lane.folded, lane.base
            )));
        }
        // Its buffer as the kernels read it: the rows `[folded, first)` it
        // carries, from `base`; none where it carries none (the kernels
        // read their base as its first row then).
        let carries = lane.first > lane.folded;
        let read_base = if carries { lane.base } else { lane.first };
        let buffered = lane.first.saturating_sub(read_base).div_ceil(tokens);
        let entry = |row: u32| -> Result<u32> {
            let i = (row - lane.base) / tokens;
            lane.buffer.get(i as usize).copied().ok_or_else(|| {
                bad(format!(
                    "lane {at}'s row {row} is in buffer entry {i}, past its {} slots",
                    lane.buffer.len()
                ))
            })
        };
        let fire = |row: u32| -> Result<(u32, u32)> {
            let i = (row - lane.first) / tokens;
            let unit = lane.fire.get(i as usize).copied().ok_or_else(|| {
                bad(format!(
                    "lane {at}'s row {row} is in fire slot {i}, past its {}",
                    lane.fire.len()
                ))
            })?;
            Ok((unit, (row - lane.first) % tokens))
        };
        out.lanes.extend([
            lane.folded as i32,
            read_base as i32,
            lane.first as i32,
            if lane.keeps { to as i32 } else { lane.folded as i32 },
        ]);
        out.lanes.extend_from_slice(lane.units);
        for i in 0..buffered {
            out.indices.push(entry(lane.base + i * tokens)? as i32);
        }
        for row in lane.first..end {
            let (unit, offset) = fire(row)?;
            if (row - lane.first) % tokens == 0 {
                out.indices.push(unit as i32);
            }
            out.write_page.push(unit as i32);
            out.write_offset.push(offset as i32);
        }
        out.indptr.push(out.indices.len() as i32);
        if !lane.keeps {
            continue;
        }
        // The fold: the last `ring_rows` of `[folded, to)` into the ring, a
        // row by its index (each layer's ring at its own place).
        let source = |row: u32| -> Result<(u32, u32)> {
            if row < lane.first {
                Ok((entry(row)?, (row - lane.base) % tokens))
            } else {
                fire(row)
            }
        };
        for row in lane.folded.max(to.saturating_sub(ring_rows))..to {
            if !landed.insert((lane.record, row)) {
                continue;
            }
            let (unit, at_row) = source(row)?;
            out.jobs.extend([
                unit as i32,
                at_row as i32,
                -1 - lane.record as i32,
                row as i32,
            ]);
        }
        // The fire's rows it leaves unfolded, into its buffer slots.
        for row in to.max(lane.first)..end {
            let (unit, at_row) = fire(row)?;
            if row < lane.base {
                return Err(bad(format!(
                    "lane {at} leaves row {row} unfolded below its buffer's first row {}",
                    lane.base
                )));
            }
            out.jobs.extend([
                unit as i32,
                at_row as i32,
                entry(row)? as i32,
                ((row - lane.base) % tokens) as i32,
            ]);
        }
    }
    out.jobs[0] = ((out.jobs.len() - 1) / 4) as i32;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode lane carrying buffered rows reads them from its buffer from
    /// `base`, its row from its fire slot; it folds its buffered rows from
    /// the buffer and its own from the fire slot, and leaves none.
    #[test]
    fn a_lane_reads_its_buffer_then_its_fire_slots_and_folds_from_both() {
        let lane = Lane {
            folded: 50,
            fold: 6,
            base: 48,
            first: 55,
            rows: 1,
            buffer: &[7, 9],
            fire: &[30],
            record: 0,
            units: &[4],
            keeps: true,
        };
        let s = stage(&[lane], 16, 40).unwrap();
        assert_eq!(s.lanes, [50, 48, 55, 56, 4]);
        assert_eq!(s.indices, [7, 30]);
        assert_eq!(s.indptr, [0, 2]);
        assert_eq!((s.write_page, s.write_offset), (vec![30], vec![0]));
        // Rows 50..55 from buffer slot 7 (rows 2..7 of it), row 55 from the fire slot.
        assert_eq!(s.jobs[0], 6);
        assert_eq!(&s.jobs[1..5], &[7, 2, -1, 50]);
        assert_eq!(&s.jobs[21..25], &[30, 0, -1, 55]);
    }

    /// A prefill folding most of its rows: its table starts past its first
    /// row; only the last R folded rows go to the ring, the rest of its
    /// rows into its buffer.
    #[test]
    fn a_prefill_rings_its_last_rows_and_buffers_its_tail() {
        let lane = Lane {
            folded: 0,
            fold: 100,
            base: 96,
            first: 0,
            rows: 110,
            buffer: &[3],
            fire: &[10, 11, 12, 13, 14, 15, 16],
            record: 2,
            units: &[4],
            keeps: true,
        };
        let s = stage(&[lane], 16, 40).unwrap();
        assert_eq!(s.indices, [10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(s.jobs[0], 40 + 10);
        // Row 60 is fire slot 3 row 12, ring place 20 of lane 2's record.
        assert_eq!(&s.jobs[1..5], &[13, 12, -3, 60]);
        // Row 100 is fire slot 6 row 4, into buffer slot 3 row 4.
        assert_eq!(&s.jobs[161..165], &[16, 4, 3, 4]);
    }

    /// Lanes of a group naming one record fold each ring row once.
    #[test]
    fn lanes_naming_one_record_fold_it_once() {
        let lane = |fire: &'static [u32]| Lane {
            folded: 16,
            fold: 1,
            base: 16,
            first: 16,
            rows: 2,
            buffer: &[5],
            fire,
            record: 0,
            units: &[4],
            keeps: true,
        };
        let s = stage(&[lane(&[1]), lane(&[2])], 16, 40).unwrap();
        let ring_jobs = s.jobs[1..].chunks(4).filter(|j| j[2] < 0).count();
        assert_eq!(ring_jobs, 1);
    }

    #[test]
    fn a_ring_of_no_rows_is_refused() {
        assert!(stage(&[], 16, 0).is_err());
    }
}
