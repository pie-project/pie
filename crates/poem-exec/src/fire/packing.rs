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
    /// One attention class per packed row (-1 where the lane states none).
    pub attn_class: Vec<i32>,
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
    pack_with_classes(select, lanes, group_of_lane, fire_rows, &[])
}

/// `pack`, with `classes[source]` the attention classes a source lane
/// states (one per row), packed beside the permutation.
pub fn pack_with_classes(
    select: Selection,
    lanes: &[LaneRow],
    group_of_lane: &[i32],
    fire_rows: u32,
    classes: &[Option<&[i32]>],
) -> Result<Packed, Fault> {
    let groups = groups_of(group_of_lane) as usize;
    let mut chosen: Vec<(i32, usize)> = lanes
        .iter()
        .enumerate()
        .filter(|(_, row)| select.holds(row.word))
        .map(|(at, _)| (group_of_lane.get(at).copied().unwrap_or(0), at))
        .collect();
    chosen.sort_unstable();

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
    let mut attn_class = vec![-1i32; fire_rows as usize];
    let mut permutation = vec![-1i32; fire_rows as usize];

    let mut packed = origin;
    lane_indptr.push(packed as i32);
    for &(_, at) in &chosen {
        let row = &lanes[at];
        let stated = classes.get(row.source as usize).copied().flatten();
        for j in 0..row.rows {
            let fire_row = row.row_offset + j;
            let here = (packed + j) as usize;
            permutation[here] = fire_row as i32;
            attn_class[here] = stated
                .and_then(|c| c.get(j as usize))
                .copied()
                .unwrap_or(-1);
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
        attn_class,
    })
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
