//! Lane phases laid out against a fire's data ([`LaneLayout`]): which lanes
//! and which block of their slots' state each PE runs, the per-PE tables
//! and bank rows uploaded with it, and how its pieces come back.

use super::*;

/// A lane plan resolved against the fire's data: which lanes and which
/// block of their slots' state each PE runs, which bank rows it holds, and
/// how the pieces come back.
pub(super) struct LaneLayout {
    pes: usize,
    /// Per PE: `[l0, l1)` and the block `[a0, a1) × [b0, b1)`.
    ranges: Vec<(usize, usize, usize, usize, usize, usize)>,
    /// Per PE: the bank rows (slots) it holds, in local order.
    slots_of: Vec<Vec<usize>>,
    /// Per PE: the slot table with its lanes' slots rewritten to local rows.
    local_slots: Vec<Vec<u32>>,
    /// Per PE: the rows its lanes own.
    rows_of: Vec<Vec<usize>>,
    lanes_per_pe: usize,
    /// Bank slot rows one PE holds (its lanes' pages, or its share of a
    /// strided pool).
    bank_rows: usize,
    /// One slot's state as `(a, b, w)`, split as the plan says.
    block: (usize, usize, usize),
    /// Whether the plan splits slots into blocks at all.
    has_block: bool,
    header: String,
    slots: String,
    banks: Vec<(String, usize, BankSplit)>,
    row_outputs: Vec<(String, usize, usize, usize)>,
    /// Planes held by block: the block picks columns of every row, or
    /// whole rows.
    cols: Vec<ColPlane>,
    /// How row outputs held by several PEs merge.
    reduce: Vec<(String, Reduce)>,
    /// Per PE: the first page of each of its lanes it holds (page groups).
    page_of: Vec<usize>,
    /// Paged banks: local slot rows are pages, `max_pages` per lane (per PE).
    pages: Option<Paged>,
}

/// The paged form's per-PE tables.
struct Paged {
    max_pages: usize,
    /// Per PE: the CSR `indptr` with its lanes' entries pointing into the
    /// local indices, and the local indices (`lanes_per_pe × max_pages`).
    indptr: Option<(String, Vec<Vec<u32>>)>,
    indices: Option<(String, Vec<Vec<u32>>)>,
    /// Per-row page tables rewritten to local pages, per PE.
    rewritten: Vec<(String, Vec<Vec<u32>>)>,
}

impl LaneLayout {
    pub(super) fn plan(
        lane: &LanePlan,
        state: &HashMap<String, Vec<u32>>,
        at: usize,
    ) -> Result<LaneLayout> {
        use kernels_cerebras::phase::{LaneKind, PageSource};
        let pes = lane.pes.max(1) as usize;
        // A plan without banks names no slot table.
        let no_slots = Vec::new();
        let slots_words = if lane.slots.is_empty() {
            &no_slots
        } else {
            state.get(&lane.slots).ok_or_else(|| Fault::Unbound {
                what: format!("phase {at}: slot table {} is not in the fire", lane.slots),
            })?
        };
        // A strided (resident) plan runs every row on every PE, so its lanes
        // need no contiguous rows; the row count is the table's.
        let strided = lane.pages.as_ref().is_some_and(|p| p.strided);
        let all_rows: usize;
        // Lanes' rows: `(first row, count)` per lane.
        let (lanes, rows_of_lane): (usize, Vec<(usize, usize)>) = match &lane.kind {
            LaneKind::PerRow { lanes } => {
                all_rows = *lanes as usize;
                (
                    *lanes as usize,
                    (0..*lanes as usize).map(|l| (l, 1)).collect(),
                )
            }
            LaneKind::Ragged { indptr, lanes } => {
                let p = state.get(indptr).ok_or_else(|| Fault::Unbound {
                    what: format!("phase {at}: indptr {indptr} is not in the fire"),
                })?;
                let ip: Vec<i32> = p.iter().map(|w| *w as i32).collect();
                let rows = (0..*lanes as usize)
                    .map(|l| {
                        let a = ip.get(l).copied().unwrap_or(0).max(0) as usize;
                        let b = ip.get(l + 1).copied().unwrap_or(a as i32).max(a as i32) as usize;
                        (a, b - a)
                    })
                    .collect();
                all_rows = ip.get(*lanes as usize).copied().unwrap_or(0).max(0) as usize;
                (*lanes as usize, rows)
            }
            LaneKind::ByTable { lane_of_row, lanes } => {
                let t = state.get(lane_of_row).ok_or_else(|| Fault::Unbound {
                    what: format!("phase {at}: lane table {lane_of_row} is not in the fire"),
                })?;
                all_rows = t.len();
                let mut rows = vec![(usize::MAX, 0usize); *lanes as usize];
                for (r, w) in t.iter().enumerate() {
                    let l = *w as i32;
                    if l < 0 || l as usize >= rows.len() {
                        continue;
                    }
                    let entry = &mut rows[l as usize];
                    if entry.1 == 0 {
                        *entry = (r, 1);
                    } else if entry.0 + entry.1 == r || strided {
                        entry.1 += 1;
                    } else {
                        return Err(Fault::Unbound {
                            what: format!(
                                "phase {at}: lane {l}'s rows are not contiguous, which a lane plan needs"
                            ),
                        });
                    }
                }
                (
                    *lanes as usize,
                    rows.into_iter()
                        .map(|(a, n)| if n == 0 { (0, 0) } else { (a, n) })
                        .collect(),
                )
            }
        };
        // Each lane's pages (paged banks) or its one slot.
        let pages_of_lane: Vec<Vec<usize>> = match &lane.pages {
            None => rows_of_lane
                .iter()
                .map(|(first, n)| {
                    if *n == 0 {
                        return Vec::new();
                    }
                    let slot = slots_words.get(*first).map(|w| *w as i32).unwrap_or(-1);
                    if slot < 0 {
                        Vec::new()
                    } else {
                        vec![slot as usize]
                    }
                })
                .collect(),
            Some(pp) => match &pp.source {
                PageSource::Csr { indptr, indices } => {
                    let ip = state.get(indptr).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page indptr {indptr} is not in the fire"),
                    })?;
                    let ix = state.get(indices).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page indices {indices} is not in the fire"),
                    })?;
                    (0..lanes)
                        .map(|l| {
                            let a = ip.get(l).map(|w| *w as i32).unwrap_or(0).max(0) as usize;
                            let b = ip
                                .get(l + 1)
                                .map(|w| *w as i32)
                                .unwrap_or(a as i32)
                                .max(a as i32) as usize;
                            ix.get(a..b.min(ix.len()))
                                .unwrap_or(&[])
                                .iter()
                                .map(|w| *w as i32)
                                .filter(|p| *p >= 0)
                                .map(|p| p as usize)
                                .collect()
                        })
                        .collect()
                }
                PageSource::Rows { table } => {
                    let t = state.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page table {table} is not in the fire"),
                    })?;
                    rows_of_lane
                        .iter()
                        .map(|(first, n)| {
                            (*first..*first + *n)
                                .filter_map(|r| {
                                    t.get(r)
                                        .map(|w| *w as i32)
                                        .filter(|p| *p >= 0)
                                        .map(|p| p as usize)
                                })
                                .collect()
                        })
                        .collect()
                }
            },
        };
        // A row window: the lanes' rows clipped to it and counted from its start.
        let rows_of_lane: Vec<(usize, usize)> = match lane.window {
            Some((start, len)) => rows_of_lane
                .iter()
                .map(|(first, n)| {
                    let (start, end) = (start as usize, start as usize + len as usize);
                    let lo = (*first).max(start).min(end);
                    let hi = (first + n).max(start).min(end);
                    (lo - start, hi - lo)
                })
                .collect(),
            None => rows_of_lane,
        };
        let lanes_per_pe = lane.lanes_per_pe.max(1) as usize;
        let (a, b, w, a_groups, b_groups) = match lane.block {
            Some(blk) => (
                blk.a as usize,
                blk.b as usize,
                blk.w as usize,
                blk.a_groups.max(1) as usize,
                blk.b_groups.max(1) as usize,
            ),
            None => {
                let stride = lane.banks.first().map_or(1, |(_, s, _)| *s as usize);
                (1, 1, stride, 1, 1)
            }
        };
        let groups = a_groups * b_groups;
        let page_groups = lane
            .pages
            .as_ref()
            .map_or(1, |p| p.page_groups.max(1) as usize);
        let pages_per_pe = lane.pages.as_ref().map_or(1, |p| p.pages_per_pe() as usize);
        // The resident placement: every PE runs every lane, page `g` on PE
        // `g % pes` as local page `g / pes`.
        let strided_pages = lane.pages.as_ref().map_or(0, |p| p.strided_pages as usize);
        if strided && (groups != 1 || page_groups != 1) {
            return Err(Fault::Unbound {
                what: format!(
                    "phase {at}: a strided pool splits over neither heads, rows nor page groups"
                ),
            });
        }
        let mut page_of = Vec::with_capacity(pes);
        let mut ranges = Vec::with_capacity(pes);
        let mut slots_of = Vec::with_capacity(pes);
        let mut local_slots = Vec::with_capacity(pes);
        let mut rows_of = Vec::with_capacity(pes);
        let mut local_ips: Vec<Vec<u32>> = Vec::with_capacity(pes);
        let mut local_ixs: Vec<Vec<u32>> = Vec::with_capacity(pes);
        for p in 0..pes {
            if strided {
                // Every lane, its rows in order; the PE's fixed pages, the
                // lane lists naming them locally or -1.
                ranges.push((0, lanes, 0, a, 0, b));
                page_of.push(0);
                let held: Vec<usize> = (0..strided_pages).map(|s| s * pes + p).collect();
                let rows: Vec<usize> = (0..all_rows).collect();
                let mut local_ip: Vec<u32> = Vec::with_capacity(lanes + 1);
                let mut local_ix: Vec<u32> = Vec::new();
                for lane_pages in pages_of_lane.iter().take(lanes) {
                    local_ip.push(local_ix.len() as u32);
                    for &page in lane_pages {
                        local_ix.push(if page % pes == p {
                            (page / pes) as u32
                        } else {
                            u32::MAX
                        });
                    }
                }
                local_ip.push(local_ix.len() as u32);
                slots_of.push(held);
                local_slots.push(slots_words.iter().map(|_| u32::MAX).collect());
                rows_of.push(rows);
                local_ips.push(local_ip);
                local_ixs.push(local_ix);
                continue;
            }
            let pg0 = (p % page_groups) * pages_per_pe;
            page_of.push(pg0);
            let q = p / page_groups;
            let lane_group = q / groups;
            // Page groups vary fastest, then the b axis's groups (the a
            // axis's when the plan puts its partial axis first), then the
            // other, then the lane groups.
            let (ag, bg) = if lane.a_first {
                (q % a_groups, (q % groups) / a_groups)
            } else {
                ((q % groups) / b_groups, q % b_groups)
            };
            let l0 = (lane_group * lanes_per_pe).min(lanes);
            let l1 = ((lane_group + 1) * lanes_per_pe).min(lanes);
            let (a0, a1) = (ag * (a / a_groups), (ag + 1) * (a / a_groups));
            let (b0, b1) = (bg * (b / b_groups), (bg + 1) * (b / b_groups));
            ranges.push((l0, l1, a0, a1, b0, b1));
            let mut held: Vec<usize> = Vec::new();
            let mut local: Vec<u32> = slots_words.iter().map(|_| u32::MAX).collect();
            let mut rows = Vec::new();
            let mut local_ip: Vec<u32> = Vec::new();
            let mut local_ix: Vec<u32> = Vec::new();
            for l in l0..l1 {
                let (first_row, n_rows) = rows_of_lane[l];
                rows.extend(first_row..first_row + n_rows);
                let lane_pages: Vec<usize> = pages_of_lane[l]
                    .iter()
                    .copied()
                    .skip(pg0)
                    .take(pages_per_pe)
                    .collect();
                let lane_pages = &lane_pages;
                let mut firsts = Vec::with_capacity(lane_pages.len());
                for &page in lane_pages {
                    let at = match held.iter().position(|s| *s == page) {
                        Some(i) => i,
                        None => {
                            held.push(page);
                            held.len() - 1
                        }
                    };
                    firsts.push(at as u32);
                }
                if lane.pages.is_none() {
                    if let Some(&at) = firsts.first() {
                        for r in first_row..first_row + n_rows {
                            if let Some(entry) = local.get_mut(r) {
                                *entry = at;
                            }
                        }
                    }
                } else {
                    // Local CSR: this lane's pages at local_ix[local_ip[l]..].
                    while local_ip.len() <= l {
                        local_ip.push(local_ix.len() as u32);
                    }
                    local_ix.extend(firsts);
                    local_ip.push(local_ix.len() as u32);
                }
            }
            if lane.pages.is_some() {
                while local_ip.len() <= lanes {
                    local_ip.push(local_ix.len() as u32);
                }
            }
            if rows.windows(2).any(|w| w[1] != w[0] + 1) {
                return Err(Fault::Unbound {
                    what: format!(
                        "phase {at}: PE {p}'s rows {rows:?} are not one contiguous run, which a lane plan's row range needs"
                    ),
                });
            }
            slots_of.push(held);
            local_slots.push(local);
            rows_of.push(rows);
            local_ips.push(local_ip);
            local_ixs.push(local_ix);
        }
        let pages = match &lane.pages {
            None => None,
            Some(pp) => {
                let max_pages = pp.pages_per_pe() as usize;
                let (indptr, indices) = match &pp.source {
                    PageSource::Csr { indptr, indices } => {
                        let ixs: Vec<Vec<u32>> = local_ixs
                            .iter()
                            .map(|ix| {
                                let mut v = ix.clone();
                                v.resize(lanes_per_pe * max_pages, 0);
                                v
                            })
                            .collect();
                        (
                            Some((indptr.clone(), local_ips.clone())),
                            Some((indices.clone(), ixs)),
                        )
                    }
                    PageSource::Rows { .. } => (None, None),
                };
                let mut rewritten = Vec::new();
                for table in &pp.rewritten {
                    let global = state.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("phase {at}: page table {table} is not in the fire"),
                    })?;
                    let per_pe: Vec<Vec<u32>> = (0..pes)
                        .map(|p| {
                            let mut local: Vec<u32> = global.iter().map(|_| u32::MAX).collect();
                            for &r in &rows_of[p] {
                                let page = global.get(r).map(|w| *w as i32).unwrap_or(-1);
                                if page >= 0
                                    && let Some(at) =
                                        slots_of[p].iter().position(|s| *s == page as usize)
                                {
                                    local[r] = at as u32;
                                }
                            }
                            local
                        })
                        .collect();
                    rewritten.push((table.clone(), per_pe));
                }
                Some(Paged {
                    max_pages,
                    indptr,
                    indices,
                    rewritten,
                })
            }
        };
        Ok(LaneLayout {
            pes,
            ranges,
            slots_of,
            local_slots,
            rows_of,
            lanes_per_pe,
            bank_rows: if strided {
                strided_pages
            } else {
                lanes_per_pe * lane.pages.as_ref().map_or(1, |p| p.pages_per_pe() as usize)
            },
            block: (a, b, w),
            has_block: lane.block.is_some(),
            header: lane.header.clone(),
            slots: lane.slots.clone(),
            banks: lane
                .banks
                .iter()
                .map(|(n, s, split)| (n.clone(), *s as usize, *split))
                .collect(),
            row_outputs: lane
                .row_outputs
                .iter()
                .map(|(n, w, sa, sb)| (n.clone(), *w as usize, *sa as usize, *sb as usize))
                .collect(),
            reduce: lane.reduce.clone(),
            page_of,
            cols: lane.cols.clone(),
            pages,
        })
    }

    /// Whether `name` merges as a sum of partials (rounded here after).
    pub(super) fn sums(&self, name: &str) -> bool {
        self.reduce
            .iter()
            .any(|(n, r)| n == name && matches!(r, Reduce::Sum))
    }

    /// A bank's `(stride, split)`; without a block plan every bank is whole.
    fn bank_stride(&self, name: &str) -> Option<(usize, BankSplit)> {
        self.banks
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, s, split)| {
                (
                    *s,
                    if self.has_block {
                        *split
                    } else {
                        BankSplit::Whole
                    },
                )
            })
    }

    /// Words one PE holds of one slot of a bank of `stride` under `split`.
    fn bank_slot_words(&self, p: usize, stride: usize, split: BankSplit) -> usize {
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        match split {
            BankSplit::Whole => stride,
            BankSplit::Block => (a1 - a0) * (b1 - b0) * self.block.2,
            BankSplit::ByB => (b1 - b0) * (stride / self.block.1.max(1)),
        }
    }

    pub(super) fn owns(&self, name: &str) -> bool {
        name == self.header
            || (name == self.slots && self.pages.is_none())
            || self.bank_stride(name).is_some()
            || self.row_outputs.iter().any(|(n, ..)| n == name)
            || self.cols.iter().any(|c| c.name == name)
            || self.page_table(name).is_some()
    }

    /// The indices below `bound` (columns, or rows) a plane's segments pick
    /// for PE `p`'s block.
    fn block_picks(&self, p: usize, bound: usize, segments: &[Segment]) -> Vec<usize> {
        if self.block.0 == 1 && self.block.1 == 1 {
            return (0..bound).collect();
        }
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        picks(
            segments,
            (a0 as u32, a1 as u32),
            (b0 as u32, b1 as u32),
            bound as u32,
        )
        .into_iter()
        .map(|c| c as usize)
        .collect()
    }

    /// The `(rows, cols)` of a block plane PE `p` holds.
    fn plane_cells(&self, p: usize, plane: &ColPlane) -> (Vec<usize>, Vec<usize>) {
        let (rows, width) = (plane.rows as usize, plane.width as usize);
        if plane.row_block {
            let (_, _, _, a0, a1, _, _) = self.range(p);
            return (
                (a0.min(rows)..a1.min(rows)).collect(),
                self.block_picks(p, width, &plane.segments),
            );
        }
        if plane.by_rows {
            (
                self.block_picks(p, rows, &plane.segments),
                (0..width).collect(),
            )
        } else {
            (
                (0..rows).collect(),
                self.block_picks(p, width, &plane.segments),
            )
        }
    }

    /// A page plan table this plan lays out per PE.
    fn page_table(&self, name: &str) -> Option<&Vec<Vec<u32>>> {
        let pages = self.pages.as_ref()?;
        if let Some((n, t)) = &pages.indptr
            && n == name
        {
            return Some(t);
        }
        if let Some((n, t)) = &pages.indices
            && n == name
        {
            return Some(t);
        }
        pages
            .rewritten
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    /// Local slot rows one PE holds (pages when paged; the PE's share of
    /// the pool when strided).
    fn local_rows(&self) -> usize {
        self.bank_rows
    }

    fn range(&self, p: usize) -> (usize, usize, usize, usize, usize, usize, usize) {
        let (l0, l1, a0, a1, b0, b1) = self.ranges[p];
        (p, l0, l1, a0, a1, b0, b1)
    }

    /// The per-PE words of `name`, when the plan lays it out.
    pub(super) fn lay_out(&self, name: &str, words: &[u32]) -> Option<Vec<u32>> {
        if name == self.header {
            let mut out = Vec::with_capacity(self.pes * 16);
            for (p, (l0, l1, a0, a1, b0, b1)) in self.ranges.iter().enumerate() {
                let rows = &self.rows_of[p];
                let (r0, r1) = match (rows.first(), rows.last()) {
                    (Some(a), Some(b)) => (*a as u32, *b as u32 + 1),
                    _ => (0, 0),
                };
                let mut row = vec![
                    *l0 as u32,
                    *l1 as u32,
                    self.slots_of[p].len() as u32,
                    *a0 as u32,
                    *a1 as u32,
                    *b0 as u32,
                    *b1 as u32,
                    r0,
                    r1,
                    self.page_of.get(p).copied().unwrap_or(0) as u32,
                    self.pages.as_ref().map_or(1, |g| g.max_pages) as u32,
                ];
                row.resize(16, 0);
                out.extend(row);
            }
            return Some(out);
        }
        if name == self.slots && self.pages.is_none() {
            return Some(self.local_slots.concat());
        }
        if let Some(t) = self.page_table(name) {
            return Some(t.concat());
        }
        if let Some((stride, split)) = self.bank_stride(name) {
            let (b, w) = (self.block.1, self.block.2);
            let per_slot = self.bank_slot_words(0, stride, split);
            let mut out = Vec::with_capacity(self.pes * self.lanes_per_pe * per_slot);
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                for slot in &self.slots_of[p] {
                    match split {
                        BankSplit::Block => {
                            for i in a0..a1 {
                                for j in b0..b1 {
                                    let from = slot * stride + (i * b + j) * w;
                                    let mut piece: Vec<u32> =
                                        words.get(from..from + w).unwrap_or(&[]).to_vec();
                                    piece.resize(w, 0);
                                    out.extend(piece);
                                }
                            }
                        }
                        BankSplit::ByB => {
                            let row = stride / b.max(1);
                            let from = slot * stride + b0 * row;
                            let len = (b1 - b0) * row;
                            let mut piece: Vec<u32> =
                                words.get(from..from + len).unwrap_or(&[]).to_vec();
                            piece.resize(len, 0);
                            out.extend(piece);
                        }
                        BankSplit::Whole => {
                            let from = slot * stride;
                            let mut piece: Vec<u32> =
                                words.get(from..from + stride).unwrap_or(&[]).to_vec();
                            piece.resize(stride, 0);
                            out.extend(piece);
                        }
                    }
                }
                out.resize((p + 1) * self.local_rows() * per_slot, 0);
            }
            return Some(out);
        }
        if self.row_outputs.iter().any(|(n, ..)| n == name) {
            // Every PE holds the whole plane; only its block's cells come back.
            return Some(
                words
                    .iter()
                    .copied()
                    .cycle()
                    .take(words.len() * self.pes)
                    .collect(),
            );
        }
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let width = plane.width as usize;
            // A plane the fabric sums goes whole to each group's first PE
            // and as zeros to the rest, so the sum is the first's cells plus
            // what the others compute.
            let group = self.fabric_group(name);
            let mut out = Vec::new();
            for p in 0..self.pes {
                let (rows, cols) = self.plane_cells(p, plane);
                let zero = group.is_some_and(|g| p % g != 0);
                for r in &rows {
                    for c in &cols {
                        let w = words.get(r * width + c).copied().unwrap_or(0);
                        out.push(if zero { 0 } else { w });
                    }
                }
            }
            return Some(out);
        }
        None
    }

    /// The PE group width of a plane the fabric sums.
    fn fabric_group(&self, name: &str) -> Option<usize> {
        self.reduce.iter().find_map(|(n, r)| match r {
            Reduce::FabricSum { group } if n == name => Some((*group).max(1) as usize),
            _ => None,
        })
    }

    /// The whole of `name` from what the PEs hold and its `current` words.
    /// `sent` is what the PEs were given (`lay_out`'s words): a bank pastes
    /// back only the words a PE changed, since PEs may hold the same page
    /// (rows of two PEs landing in one page) and each writes its own rows.
    pub(super) fn gather_up(
        &self,
        name: &str,
        laid: &[u32],
        sent: &[u32],
        mut current: Vec<u32>,
        laid_of: &HashMap<String, Vec<u32>>,
    ) -> Vec<u32> {
        if name == self.header {
            return laid[..laid.len().min(self.pes * 16)].to_vec();
        }
        if name == self.slots || self.page_table(name).is_some() {
            return current;
        }
        if let Some((stride, split)) = self.bank_stride(name) {
            let (b, w) = (self.block.1, self.block.2);
            let per_slot = self.bank_slot_words(0, stride, split);
            let per_pe = self.local_rows() * per_slot;
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                let mut from = p * per_pe;
                let mut paste = |to: usize, len: usize, from: &mut usize| {
                    if let (Some(src), true) =
                        (laid.get(*from..*from + len), to + len <= current.len())
                    {
                        paste_changed(
                            &mut current[to..to + len],
                            src,
                            sent.get(*from..*from + len),
                        );
                    }
                    *from += len;
                };
                for slot in &self.slots_of[p] {
                    match split {
                        BankSplit::Block => {
                            for i in a0..a1 {
                                for j in b0..b1 {
                                    paste(slot * stride + (i * b + j) * w, w, &mut from);
                                }
                            }
                        }
                        BankSplit::ByB => {
                            let row = stride / b.max(1);
                            paste(slot * stride + b0 * row, (b1 - b0) * row, &mut from);
                        }
                        BankSplit::Whole => paste(slot * stride, stride, &mut from),
                    }
                }
            }
            return current;
        }
        if let Some((_, reduce)) = self.reduce.iter().find(|(n, _)| n == name) {
            return self.merge_rows(name, reduce, laid, sent, current, laid_of);
        }
        if let Some((_, width, sa, sb)) = self.row_outputs.iter().find(|(n, ..)| n == name) {
            let per_pe = current.len();
            for p in 0..self.pes {
                let (_, _, _, a0, a1, b0, b1) = self.range(p);
                for r in &self.rows_of[p] {
                    let from = p * per_pe + r * width;
                    let to = r * width;
                    if to + width > current.len() {
                        continue;
                    }
                    if self.block.0 == 1 && self.block.1 == 1 {
                        if let Some(src) = laid.get(from..from + width) {
                            current[to..to + width].copy_from_slice(src);
                        }
                        continue;
                    }
                    for i in a0..a1 {
                        for j in b0..b1 {
                            let col = i * sa + j * sb;
                            if col < *width
                                && let Some(v) = laid.get(from + col)
                            {
                                current[to + col] = *v;
                            }
                        }
                    }
                }
            }
            return current;
        }
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let width = plane.width as usize;
            let mut from = 0;
            for p in 0..self.pes {
                let (rows, cols) = self.plane_cells(p, plane);
                for r in &rows {
                    for c in &cols {
                        if let Some(v) = laid.get(from)
                            && sent.get(from) != Some(v)
                            && let Some(slot) = current.get_mut(r * width + c)
                        {
                            *slot = *v;
                        }
                        from += 1;
                    }
                }
            }
            return current;
        }
        laid.to_vec()
    }
}

impl LaneLayout {
    /// The width of a row output or a block plane.
    fn plane_width(&self, name: &str) -> Option<usize> {
        self.row_outputs
            .iter()
            .find(|(n, ..)| n == name)
            .map(|(_, w, ..)| *w)
            .or_else(|| {
                self.cols
                    .iter()
                    .find(|c| c.name == name)
                    .map(|c| c.width as usize)
            })
    }

    /// The columns of `name` PE `p` computes: a plane's block columns, or a
    /// row output's block cells (all when the plan has no block).
    fn cells_of(&self, p: usize, name: &str, width: usize) -> Vec<usize> {
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            return self.plane_cells(p, plane).1;
        }
        let Some((_, _, sa, sb)) = self.row_outputs.iter().find(|(n, ..)| n == name) else {
            return Vec::new();
        };
        if !self.has_block {
            return (0..width).collect();
        }
        let (_, _, _, a0, a1, b0, b1) = self.range(p);
        let mut cols = std::collections::BTreeSet::new();
        for i in a0..a1 {
            for j in b0..b1 {
                let col = i * sa + j * sb;
                if col < width {
                    cols.insert(col);
                }
            }
        }
        cols.into_iter().collect()
    }

    /// Where PE `p` keeps cell `(r, col)` of `name` in its copy: a row output
    /// is whole per PE; a block plane holds only its block's columns.
    fn local_at(&self, p: usize, name: &str, r: usize, col: usize) -> Option<usize> {
        if let Some(plane) = self.cols.iter().find(|c| c.name == name) {
            let (_, cols) = self.plane_cells(p, plane);
            let i = cols.binary_search(&col).ok()?;
            return Some(r * cols.len() + i);
        }
        Some(r * self.plane_width(name)? + col)
    }

    /// Merges an output held by several PEs (page groups): each PE's copy
    /// of a row is partial over its pages (and its heads).
    fn merge_rows(
        &self,
        name: &str,
        reduce: &Reduce,
        laid: &[u32],
        sent: &[u32],
        mut current: Vec<u32>,
        laid_of: &HashMap<String, Vec<u32>>,
    ) -> Vec<u32> {
        let Some(width) = self.plane_width(name) else {
            return current;
        };
        if let Reduce::FabricSum { group } | Reduce::FabricRoot { group } = reduce {
            // The fabric merged each group into its first PE: its cells are
            // the result.
            let group = (*group).max(1) as usize;
            let per_pe = laid.len() / self.pes.max(1);
            let rows = current.len().checked_div(width).unwrap_or(0);
            for p in (0..self.pes).step_by(group) {
                for r in &self.rows_of[p] {
                    if *r >= rows {
                        continue;
                    }
                    for col in self.cells_of(p, name, width) {
                        let Some(i) = self.local_at(p, name, *r, col) else {
                            continue;
                        };
                        if let (Some(now), Some(slot)) =
                            (laid.get(p * per_pe + i), current.get_mut(r * width + col))
                        {
                            *slot = *now;
                        }
                    }
                }
            }
            return current;
        }
        if matches!(reduce, Reduce::Sum) {
            // Each PE's copy holds its cells; what it changed adds up.
            let per_pe = laid.len() / self.pes.max(1);
            let rows = current.len().checked_div(width).unwrap_or(0);
            for p in 0..self.pes {
                for r in &self.rows_of[p] {
                    if *r >= rows {
                        continue;
                    }
                    for col in self.cells_of(p, name, width) {
                        let Some(i) = self.local_at(p, name, *r, col) else {
                            continue;
                        };
                        let (Some(now), Some(was)) =
                            (laid.get(p * per_pe + i), sent.get(p * per_pe + i))
                        else {
                            continue;
                        };
                        let delta = f32::from_bits(*now) - f32::from_bits(*was);
                        if let Some(slot) = current.get_mut(r * width + col) {
                            *slot = (f32::from_bits(*slot) + delta).to_bits();
                        }
                    }
                }
            }
            return current;
        }
        let rows = current.len().checked_div(width).unwrap_or(0);
        let per_pe = laid.len() / self.pes.max(1);
        let at = |p: usize, r: usize, col: usize| -> Option<f32> {
            let i = self.local_at(p, name, r, col)?;
            laid.get(p * per_pe + i).map(|w| f32::from_bits(*w))
        };
        for r in 0..rows {
            let holders: Vec<usize> = (0..self.pes)
                .filter(|p| self.rows_of[*p].contains(&r))
                .collect();
            if holders.is_empty() {
                continue;
            }
            match reduce {
                Reduce::Sum | Reduce::FabricSum { .. } | Reduce::FabricRoot { .. } => {}
                Reduce::LogSumExp => {
                    for h in 0..width {
                        let vals: Vec<f32> = holders.iter().filter_map(|p| at(*p, r, h)).collect();
                        if !vals.is_empty() {
                            current[r * width + h] = log_sum_exp2(&vals).to_bits();
                        }
                    }
                }
                Reduce::Weighted { lse } => {
                    let Some(heads) = self.plane_width(lse) else {
                        continue;
                    };
                    let lse_laid = laid_of.get(lse).map_or(&[][..], |v| v.as_slice());
                    let lse_per_pe = lse_laid.len() / self.pes.max(1);
                    let lse_at = |p: usize, h: usize| -> Option<f32> {
                        let i = self.local_at(p, lse, r, h)?;
                        lse_laid.get(p * lse_per_pe + i).map(|w| f32::from_bits(*w))
                    };
                    let d = width / heads.max(1);
                    for h in 0..heads {
                        let pairs: Vec<(usize, f32)> = holders
                            .iter()
                            .filter_map(|p| lse_at(*p, h).map(|l| (*p, l)))
                            .collect();
                        if pairs.is_empty() {
                            continue;
                        }
                        let ls: Vec<f32> = pairs.iter().map(|(_, l)| *l).collect();
                        let total = log_sum_exp2(&ls);
                        for i in 0..d {
                            let col = h * d + i;
                            let v = if total == f32::NEG_INFINITY {
                                0.0
                            } else {
                                pairs
                                    .iter()
                                    .map(|(p, l)| {
                                        (l - total).exp2() * at(*p, r, col).unwrap_or(0.0)
                                    })
                                    .sum()
                            };
                            if let Some(slot) = current.get_mut(r * width + col) {
                                *slot = v.to_bits();
                            }
                        }
                    }
                }
            }
        }
        current
    }
}

/// `log2 Σ 2^v`, stable; `-inf` when every `v` is.
fn log_sum_exp2(vals: &[f32]) -> f32 {
    let m = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return m;
    }
    m + vals.iter().map(|v| (v - m).exp2()).sum::<f32>().log2()
}

/// Copies into `dst` the words of `laid` that differ from `sent` (all of
/// them when nothing was sent).
fn paste_changed(dst: &mut [u32], laid: &[u32], sent: Option<&[u32]>) {
    for (i, v) in laid.iter().enumerate() {
        if sent.and_then(|s| s.get(i)).is_none_or(|s| s != v) {
            dst[i] = *v;
        }
    }
}
