use poem_ir::Selection;

use super::Fault;
use super::compose::LaneRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneFacts {
    pub group: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    pub select: Selection,
    pub origin: u32,
    pub rows: u32,
    pub lanes: u32,
    pub group_indptr: Vec<i32>,
    pub lane_indptr: Vec<i32>,
    pub permutation: Vec<i32>,
}

#[must_use]
pub fn group_of_lane(lanes: &[LaneRow], facts: &[LaneFacts]) -> Vec<i32> {
    let mut stated: Vec<(u32, i32)> = Vec::new();
    let mut next = 0i32;
    lanes
        .iter()
        .map(|row| {
            let group = facts.get(row.source as usize).and_then(|facts| facts.group);
            match group {
                Some(id) => match stated.iter().find(|(seen, _)| *seen == id) {
                    Some((_, dense)) => *dense,
                    None => {
                        let dense = next;
                        next += 1;
                        stated.push((id, dense));
                        dense
                    }
                },
                None => {
                    let dense = next;
                    next += 1;
                    dense
                }
            }
        })
        .collect()
}

#[must_use]
pub fn groups_of(group_of_lane: &[i32]) -> u32 {
    group_of_lane
        .iter()
        .map(|&group| u32::try_from(group + 1).unwrap_or(0))
        .max()
        .unwrap_or(0)
}

pub fn pack(
    select: Selection,
    lanes: &[LaneRow],
    group_of_lane: &[i32],
    fire_rows: u32,
) -> Result<Packed, Fault> {
    let groups = groups_of(group_of_lane) as usize;
    let chosen = chosen(select, lanes, group_of_lane);

    let mut by_row: Vec<(u32, u32)> = chosen
        .iter()
        .map(|&(_, at)| (lanes[at].row_offset, lanes[at].rows))
        .collect();
    by_row.sort_unstable();
    let origin = by_row.first().map_or(0, |(offset, _)| *offset);
    let mut end = origin;
    for (offset, rows) in &by_row {
        if *offset != end {
            return Err(Fault::ScatteredSelection {
                mask: select.mask,
                value: select.value,
                at: *offset,
                expected: end,
            });
        }
        end += rows;
    }
    let rows = end - origin;

    let mut group_indptr = vec![0i32; groups + 1];
    let mut lane_indptr = Vec::with_capacity(chosen.len() + 1);
    let mut permutation = vec![-1i32; fire_rows as usize];

    let mut packed = origin;
    lane_indptr.push(packed as i32);
    for &(_, at) in &chosen {
        let row = &lanes[at];
        for j in 0..row.rows {
            permutation[(packed + j) as usize] = (row.row_offset + j) as i32;
        }
        packed += row.rows;
        lane_indptr.push(packed as i32);
    }
    let mut bound = origin as i32;
    let mut run = 0usize;
    for (g, start) in group_indptr.iter_mut().take(groups).enumerate() {
        *start = bound;
        let mut count = 0u32;
        while run < chosen.len() && chosen[run].0 as usize == g {
            count += lanes[chosen[run].1].rows;
            run += 1;
        }
        bound += count as i32;
    }
    group_indptr[groups] = bound;
    debug_assert_eq!(bound, (origin + rows) as i32);

    Ok(Packed {
        select,
        origin,
        rows,
        lanes: chosen.len() as u32,
        group_indptr,
        lane_indptr,
        permutation,
    })
}

/// The lanes `select` holds, as (group, composition index), in packing order.
fn chosen(select: Selection, lanes: &[LaneRow], group_of_lane: &[i32]) -> Vec<(i32, usize)> {
    let mut chosen: Vec<(i32, usize)> = lanes
        .iter()
        .enumerate()
        .filter(|(_, row)| select.holds(row.word))
        .map(|(at, _)| (group_of_lane.get(at).copied().unwrap_or(0), at))
        .collect();
    chosen.sort_unstable();
    chosen
}

/// One lane's stated attention mask: per guest row of the lane, the columns
/// of its attention group it attends, as ascending ranges. A group's columns
/// are its lanes' guest rows, lane after lane in the pass's order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LaneMask {
    pub rows: Vec<Vec<std::ops::Range<u64>>>,
}

/// How a lane's rows stand in its mask: as many guest rows as it states,
/// and, for a voxel lane, the guest row (voxel) each token row attends as.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuestRows {
    pub count: u64,
    pub of_token: Option<Vec<u64>>,
}

impl GuestRows {
    fn of(&self, token: u32) -> u64 {
        match &self.of_token {
            Some(rows) => rows.get(token as usize).copied().unwrap_or(u64::MAX),
            None => u64::from(token),
        }
    }
}

/// A lane's guest rows. A voxel lane's token rows (after the model's
/// patchify) are its clips' patches of `patch` voxels, in (t, h, w) row-major
/// order from its first token row, clip after clip: each attends as its
/// patch's first voxel. Any other lane's are its rows.
pub fn guest_rows(
    lane: &LaneRow,
    clips: Option<&[[u32; 3]]>,
    patch: Option<[u32; 3]>,
) -> Result<GuestRows, Fault> {
    let Some(clips) = clips.filter(|clips| !clips.is_empty()) else {
        return Ok(GuestRows {
            count: u64::from(lane.rows),
            of_token: None,
        });
    };
    let refused = || Fault::MaskPatches {
        lane: lane.source,
        tokens: lane.rows,
        clips: clips.len() as u32,
    };
    let [pt, ph, pw] = patch.ok_or_else(refused)?;
    let mut of_token = Vec::with_capacity(lane.rows as usize);
    let mut base = 0u64;
    for &[t, h, w] in clips {
        if pt == 0
            || ph == 0
            || pw == 0
            || !t.is_multiple_of(pt)
            || !h.is_multiple_of(ph)
            || !w.is_multiple_of(pw)
        {
            return Err(refused());
        }
        let (h64, w64) = (u64::from(h), u64::from(w));
        for i in 0..t / pt {
            for j in 0..h / ph {
                for k in 0..w / pw {
                    of_token.push(
                        base + u64::from(i * pt) * h64 * w64
                            + u64::from(j * ph) * w64
                            + u64::from(k * pw),
                    );
                }
            }
        }
        base += u64::from(t) * h64 * w64;
    }
    if of_token.len() != lane.rows as usize {
        return Err(refused());
    }
    Ok(GuestRows {
        count: base,
        of_token: Some(of_token),
    })
}

/// The custom mask one ragged attention over packings `q` (its queries) and
/// `kv` (its keys) applies: for each attention group, a bit a (query, key)
/// pair of its rows there, query-major, group `g`'s from byte `indptr[g]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupMask {
    pub bits: Vec<u8>,
    pub indptr: Vec<i32>,
}

/// The mask the lanes' stated masks (`masks[source]`, each over its group's
/// guest rows; `guests[source]` how its rows stand there) make of a ragged
/// attention over packings `q` and `kv`: none when no lane in it states one.
/// A lane stating none attends every row of its group.
pub fn group_mask(
    lanes: &[LaneRow],
    group_of_lane: &[i32],
    q: &Packed,
    kv: &Packed,
    masks: &[Option<&LaneMask>],
    guests: &[GuestRows],
) -> Result<Option<GroupMask>, Fault> {
    let source = |at: usize| lanes[at].source as usize;
    let stated = |at: usize| masks.get(source(at)).copied().flatten();
    let q_chosen = chosen(q.select, lanes, group_of_lane);
    let kv_chosen = chosen(kv.select, lanes, group_of_lane);
    if !q_chosen.iter().any(|&(_, at)| stated(at).is_some()) {
        return Ok(None);
    }
    for &(_, at) in &q_chosen {
        if let Some(mask) = stated(at) {
            let rows = guests.get(source(at)).map_or(0, |guest| guest.count);
            if mask.rows.len() as u64 != rows {
                return Err(Fault::MaskRows {
                    lane: lanes[at].source,
                    stated: mask.rows.len(),
                    rows,
                });
            }
        }
    }
    // Where each lane's guest rows start among its group's: its group's
    // lanes, in the pass's order (their sources'), before it.
    let offset = |at: usize| -> u64 {
        let group = group_of_lane.get(at).copied().unwrap_or(0);
        (0..lanes.len())
            .filter(|&other| {
                group_of_lane.get(other).copied().unwrap_or(0) == group
                    && lanes[other].source < lanes[at].source
            })
            .map(|other| guests.get(source(other)).map_or(0, |guest| guest.count))
            .sum()
    };
    let groups = groups_of(group_of_lane) as usize;
    let mut bits: Vec<u8> = Vec::new();
    let mut indptr = Vec::with_capacity(groups + 1);
    for g in 0..groups {
        indptr.push(i32::try_from(bits.len()).unwrap_or(i32::MAX));
        // The group's key rows, as the columns of its guest rows they are.
        let keys: Vec<u64> = kv_chosen
            .iter()
            .filter(|&&(group, _)| group as usize == g)
            .flat_map(|&(_, at)| {
                let base = offset(at);
                let guest = &guests[source(at)];
                (0..lanes[at].rows).map(move |token| base + guest.of(token))
            })
            .collect();
        let queries: Vec<(usize, u32)> = q_chosen
            .iter()
            .filter(|&&(group, _)| group as usize == g)
            .flat_map(|&(_, at)| (0..lanes[at].rows).map(move |token| (at, token)))
            .collect();
        let cells = queries.len() as u64 * keys.len() as u64;
        let base = bits.len();
        bits.resize(
            base + usize::try_from(cells.div_ceil(8)).unwrap_or(usize::MAX),
            0,
        );
        for (row, &(at, token)) in queries.iter().enumerate() {
            let first = row as u64 * keys.len() as u64;
            let mut set = |key: usize| {
                let cell = first + key as u64;
                bits[base + (cell / 8) as usize] |= 1 << (cell % 8);
            };
            match stated(at) {
                None => (0..keys.len()).for_each(&mut set),
                Some(mask) => {
                    let ranges = &mask.rows[guests[source(at)].of(token) as usize];
                    for (key, &column) in keys.iter().enumerate() {
                        let next = ranges.partition_point(|span| span.end <= column);
                        if ranges.get(next).is_some_and(|span| span.start <= column) {
                            set(key);
                        }
                    }
                }
            }
        }
    }
    indptr.push(i32::try_from(bits.len()).unwrap_or(i32::MAX));
    Ok(Some(GroupMask { bits, indptr }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(source: u32, word: u64, row_offset: u32, rows: u32) -> LaneRow {
        LaneRow {
            source,
            word,
            class: 0,
            row_offset,
            rows,
            ..LaneRow::default()
        }
    }

    const TEXT: u64 = 1;
    const IMAGE: u64 = 2;

    #[test]
    fn packing_every_case() {
        a_joint_selection_packs_by_group_then_lane();
        a_class_selection_packs_at_its_own_window();
        a_scattered_selection_is_refused();
        a_group_mask_runs_over_its_group_in_lane_order();
        a_voxel_lanes_tokens_attend_as_their_patches_first_voxel();
    }

    fn keeps(mask: &GroupMask, group: usize, keys: u64, q: u64, k: u64) -> bool {
        let cell = q * keys + k;
        (mask.bits[mask.indptr[group] as usize + (cell / 8) as usize] >> (cell % 8)) & 1 == 1
    }

    fn a_group_mask_runs_over_its_group_in_lane_order() {
        // Two groups (CFG): lanes 0, 1 in group 0, lanes 2, 3 in group 1.
        let lanes = [
            row(0, IMAGE, 0, 2),
            row(1, TEXT, 2, 1),
            row(2, IMAGE, 3, 2),
            row(3, TEXT, 5, 1),
        ];
        let facts = [
            LaneFacts { group: Some(0) },
            LaneFacts { group: Some(0) },
            LaneFacts { group: Some(1) },
            LaneFacts { group: Some(1) },
        ];
        let groups = group_of_lane(&lanes, &facts);
        let packed = pack(Selection::ALL, &lanes, &groups, 6).unwrap();
        let guests: Vec<GuestRows> = lanes
            .iter()
            .map(|lane| guest_rows(lane, None, None).unwrap())
            .collect();
        // Lane 0's first row attends only the text row (column 2) of its
        // group; its second attends everything.
        let image = LaneMask {
            rows: vec![vec![2..3], vec![0..3]],
        };
        let masks = [Some(&image), None, None, None];
        let mask = group_mask(&lanes, &groups, &packed, &packed, &masks, &guests)
            .unwrap()
            .unwrap();
        assert_eq!(mask.indptr.len(), 3);
        assert!(!keeps(&mask, 0, 3, 0, 0));
        assert!(!keeps(&mask, 0, 3, 0, 1));
        assert!(keeps(&mask, 0, 3, 0, 2));
        assert!((0..3).all(|k| keeps(&mask, 0, 3, 1, k)));
        assert!((0..3).all(|k| keeps(&mask, 0, 3, 2, k)));
        assert!((0..3).all(|q| (0..3).all(|k| keeps(&mask, 1, 3, q, k))));
        let unmasked = [None, None, None, None];
        assert_eq!(
            group_mask(&lanes, &groups, &packed, &packed, &unmasked, &guests).unwrap(),
            None
        );
    }

    fn a_voxel_lanes_tokens_attend_as_their_patches_first_voxel() {
        let lane = row(0, IMAGE, 0, 4);
        let guest = guest_rows(&lane, Some(&[[1, 4, 4]]), Some([1, 2, 2])).unwrap();
        assert_eq!(guest.count, 16);
        assert_eq!(guest.of_token, Some(vec![0, 2, 8, 10]));
        assert!(guest_rows(&lane, Some(&[[1, 4, 4]]), Some([1, 4, 4])).is_err());
    }

    fn a_joint_selection_packs_by_group_then_lane() {
        let lanes = [
            row(0, TEXT, 0, 3),
            row(2, TEXT, 3, 2),
            row(1, IMAGE, 5, 4),
            row(3, IMAGE, 9, 5),
        ];
        let facts = [
            LaneFacts { group: Some(7) },
            LaneFacts { group: Some(7) },
            LaneFacts { group: Some(9) },
            LaneFacts { group: Some(9) },
        ];
        let groups = group_of_lane(&lanes, &facts);
        assert_eq!(groups, vec![0, 1, 0, 1]);
        let packed = pack(Selection::ALL, &lanes, &groups, 14).unwrap();
        assert_eq!(packed.origin, 0);
        assert_eq!(packed.rows, 14);
        assert_eq!(packed.group_indptr, vec![0, 7, 14]);
        assert_eq!(packed.lane_indptr, vec![0, 3, 7, 9, 14]);
        let expected: Vec<i32> = (0..3).chain(5..9).chain(3..5).chain(9..14).collect();
        assert_eq!(packed.permutation, expected);
    }

    fn a_class_selection_packs_at_its_own_window() {
        let lanes = [row(0, TEXT, 0, 3), row(2, TEXT, 3, 2), row(1, IMAGE, 5, 4)];
        let facts = [
            LaneFacts { group: Some(1) },
            LaneFacts { group: Some(1) },
            LaneFacts { group: None },
        ];
        let groups = group_of_lane(&lanes, &facts);
        assert_eq!(groups, vec![0, 1, 0]);
        let image = Selection { mask: 2, value: 2 };
        let packed = pack(image, &lanes, &groups, 9).unwrap();
        assert_eq!(packed.origin, 5);
        assert_eq!(packed.rows, 4);
        assert_eq!(packed.group_indptr, vec![5, 9, 9]);
        assert_eq!(packed.lane_indptr, vec![5, 9]);
        assert_eq!(packed.permutation, vec![-1, -1, -1, -1, -1, 5, 6, 7, 8]);
    }

    fn a_scattered_selection_is_refused() {
        let lanes = [row(0, TEXT, 0, 2), row(1, IMAGE, 2, 3), row(2, TEXT, 5, 2)];
        let facts = [LaneFacts::default(); 3];
        let groups = group_of_lane(&lanes, &facts);
        let text = Selection { mask: 1, value: 1 };
        let fault = pack(text, &lanes, &groups, 7).unwrap_err();
        assert!(matches!(fault, Fault::ScatteredSelection { .. }), "{fault}");
    }
}
