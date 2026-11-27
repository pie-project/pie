//! Where a prompt is cut into pieces: host cuts, rows of lanes, checkpoint rows.

/// Each piece but the last ends on a page boundary; the last holds every row
/// the program reads out. The pages before the first media row are a piece
/// of their own, which seals.
/// `tail`: the last rows of the lane an output it reads is computed over.
pub(super) fn host_cuts(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tail: u32,
) -> Vec<(u32, u32)> {
    host_cuts_from(req, cuttable, share, page, forced, tail, 0)
}

/// [`host_cuts`] of the rows from `start` on: the rest of a pass cut again
/// once its rung was raised.
pub(super) fn host_cuts_from(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tail: u32,
    start: u32,
) -> Vec<(u32, u32)> {
    let rows =
        |lane: &crate::engine::FireLane| u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
    let [lane] = req.lanes.as_slice() else {
        return vec![(0, req.lanes.first().map_or(0, rows))];
    };
    let rows = rows(lane);
    if !cuttable || !separable_rows(req) {
        return vec![(0, rows)];
    }
    let (held, page) = (lane.kv.held, page.max(1));
    let media = first_media_row(req);
    let tail = tail_row(lane, rows, tail).min(media.unwrap_or(rows));
    let mut ends: Vec<u32> = forced
        .iter()
        .copied()
        .chain(media.map(|row| ((held + row) / page * page).saturating_sub(held)))
        .filter(|&end| start < end && end < rows)
        .collect();
    if share > 0 {
        let mut from = start;
        while rows - from > share {
            let mut to = ((held + from + share) / page * page).saturating_sub(held);
            if to <= from {
                to = from + share;
            }
            ends.push(to);
            from = to;
        }
    }
    ends.retain(|&end| end <= tail);
    ends.push(rows);
    ends.sort_unstable();
    ends.dedup();
    let mut from = start;
    ends.into_iter()
        .map(|to| {
            let cut = (from, to);
            from = to;
            cut
        })
        .collect()
}

pub(super) fn plain_rows(req: &crate::engine::FireRequest) -> bool {
    separable_rows(req)
        && req.lanes[0].mask.is_none()
        && matches!(req.lanes[0].readout, ::engine::Readout::Last)
}

/// Cut pieces end before it: the spans move the positions of the rows after
/// them (M-RoPE).
pub(super) fn first_media_row(req: &crate::engine::FireRequest) -> Option<u32> {
    req.media
        .iter()
        .filter(|media| media.lane == 0)
        .flat_map(|media| media.routes.iter())
        .filter_map(|&row| u32::try_from(row).ok())
        .min()
}

/// A mask is a row's own, so a masked lane qualifies.
pub(super) fn separable_rows(req: &crate::engine::FireRequest) -> bool {
    matches!(req.lanes.as_slice(), [lane] if plain_lane(lane)) && req.voxels.is_empty()
}

pub(super) fn plain_lane(lane: &::engine::Lane) -> bool {
    lane.adapter.is_none() && lane.inputs.is_empty() && !lane.kv_less
}

/// The first row the last piece must hold: what the readout names, and the
/// last `tail` rows an output the program reads is computed over.
pub(super) fn tail_row(lane: &::engine::Lane, rows: u32, tail: u32) -> u32 {
    let read = match &lane.readout {
        ::engine::Readout::Last => rows.saturating_sub(1),
        ::engine::Readout::Rows(read) => read.iter().copied().min().unwrap_or(rows),
        ::engine::Readout::None => rows,
    };
    if tail > 0 {
        read.min(rows.saturating_sub(tail))
    } else {
        read
    }
}

/// Every fire but the last runs the model's forward alone: no program stages,
/// channel values or readout.
pub(crate) fn rows_of_lanes(
    full: &crate::engine::FireRequest,
    piece: &[Option<(u32, u32)>],
    last: bool,
    page: u32,
) -> crate::engine::FireRequest {
    let mut req = full.clone();
    req.lanes = full
        .lanes
        .iter()
        .zip(piece)
        .filter_map(|(lane, &rows)| {
            let (from, to) = rows?;
            let mut lane = lane.clone();
            let (a, b) = (from as usize, to as usize);
            lane.tokens = lane.tokens[a..b].to_vec();
            if !lane.positions.is_empty() {
                lane.positions = lane.positions[a..b].to_vec();
            }
            // A row's mask counts its columns from the boundary its piece
            // states ([`restate_boundary`]): its rows are cut, its columns not.
            if let Some(::engine::Masking::Rows(masks)) = &mut lane.mask {
                *masks = masks[a..b].to_vec();
            }
            lane.kv.held += from;
            let len = lane.kv.held + (to - from);
            lane.pages.truncate(len.div_ceil(page.max(1)) as usize);
            if last {
                if let ::engine::Readout::Rows(read) = &mut lane.readout {
                    for row in read.iter_mut() {
                        *row -= from;
                    }
                }
            } else {
                lane.readout = ::engine::Readout::None;
            }
            Some(lane)
        })
        .collect();
    if !last {
        req.boundary_program = false;
        req.ring_moves.clear();
    }
    // A cut ends before the first media span ([`first_media_row`]): only
    // the last piece carries media, at its own rows.
    if last {
        let from = piece.first().copied().flatten().map_or(0, |(from, _)| from);
        let to = piece.first().copied().flatten().map_or(0, |(_, to)| to);
        for media in &mut req.media {
            for row in media.routes.iter_mut().filter(|row| **row >= 0) {
                *row -= from as i32;
            }
            if !media.token_positions.is_empty() {
                media.token_positions =
                    media.token_positions[3 * from as usize..3 * to as usize].to_vec();
            }
        }
    } else {
        req.media.clear();
    }
    req.single_token_mode = req.lanes.iter().all(|lane| lane.tokens.len() == 1);
    req
}

/// Several lanes are cut alike and aligned at their ends, so the last piece
/// holds every lane's read-out rows.
pub(crate) fn pass_pieces(
    req: &crate::engine::FireRequest,
    cuttable: bool,
    lanes_cuttable: bool,
    share: u32,
    page: u32,
    forced: &[u32],
    tails: &[u32],
) -> Vec<Vec<Option<(u32, u32)>>> {
    let tail = |lane: usize| tails.get(lane).copied().unwrap_or(0);
    let whole = || {
        vec![
            req.lanes
                .iter()
                .map(|lane| Some((0, u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX))))
                .collect(),
        ]
    };
    if req.lanes.len() <= 1 {
        return host_cuts(req, cuttable, share, page, forced, tail(0))
            .into_iter()
            .map(|cut| vec![Some(cut)])
            .collect();
    }
    if !cuttable || !lanes_cuttable || !separable_lanes(req) {
        return whole();
    }
    let page = page.max(1);
    let lanes = u32::try_from(req.lanes.len()).unwrap_or(u32::MAX);
    let share = (share / lanes / page * page).max(page);
    let per_lane: Vec<Vec<(u32, u32)>> = req
        .lanes
        .iter()
        .enumerate()
        .map(|(at, lane)| {
            let one = crate::engine::FireRequest::one(lane.clone());
            host_cuts(&one, true, share, page, &[], tail(at))
        })
        .collect();
    let count = per_lane.iter().map(Vec::len).max().unwrap_or(1);
    (0..count)
        .map(|at| {
            per_lane
                .iter()
                .map(|cuts| {
                    (at + cuts.len())
                        .checked_sub(count)
                        .map(|index| cuts[index])
                })
                .collect()
        })
        .collect()
}

pub(super) fn separable_lanes(req: &crate::engine::FireRequest) -> bool {
    req.media.is_empty()
        && req.voxels.is_empty()
        && req.lanes.iter().all(|lane| {
            lane.group.is_none() && separable_rows(&crate::engine::FireRequest::one(lane.clone()))
        })
}

/// The leading rows whose KV is their tokens' alone (before any media row), so their pages seal.
pub(super) fn keyed_rows(req: &crate::engine::FireRequest) -> u32 {
    let [lane] = req.lanes.as_slice() else {
        return 0;
    };
    let held = lane.kv.held;
    let keyed = plain_lane(lane)
        && lane.mask.is_none()
        && !lane.tokens.contains(&u32::MAX)
        && (lane.positions.is_empty()
            || lane
                .positions
                .iter()
                .enumerate()
                .all(|(i, &at)| at == held + i as u32));
    if !keyed {
        return 0;
    }
    let rows = u32::try_from(lane.tokens.len()).unwrap_or(u32::MAX);
    first_media_row(req).unwrap_or(rows).min(rows)
}

/// The rows a pass over `rows` keeps its state at: those the inferlet asks
/// for (`asked`), each floored to whole pages and within the `keyed` pages
/// the pass keys. The runtime keeps none unasked: where a state is kept is
/// the inferlet's (and its policy's) to say.
pub(crate) fn checkpoint_rows(
    asked: &[u32],
    rows: std::ops::Range<u64>,
    page_size: u32,
    keyed: usize,
) -> Vec<u64> {
    let page = u64::from(page_size.max(1));
    let mut at: Vec<u64> = asked
        .iter()
        .map(|&row| u64::from(row) / page * page)
        .filter(|&row| rows.start < row && row <= rows.end && row / page <= keyed as u64)
        .collect();
    at.sort_unstable();
    at.dedup();
    at
}
