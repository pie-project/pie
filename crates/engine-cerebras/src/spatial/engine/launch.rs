//! The engine's requests to the wafer program ([`wafer::Wafer`]): a fire's
//! lanes as launches, the token loop, page copies.

use std::collections::BTreeMap;

use engine::error::{Error, Result as EngineResult};
use engine::program::InstanceId;

use crate::spatial::wafer;
pub(super) use wafer::MODEL_NS;

/// A wafer model error as the engine's.
fn device(e: String) -> Error {
    Error::Device(format!("the wafer model: {e}"))
}

/// `copy_kv`'s token runs on the paged cache: whole pages only.
pub(super) fn copy_paged(
    w: &mut wafer::Wafer,
    runs: &[engine::transfer::KvRun],
) -> EngineResult<()> {
    let size = w.paged.map_or(0, |s| s.page_size) as u32;
    let mut pairs = Vec::new();
    for run in runs {
        if run.src_token != 0 || run.dst_token != 0 || run.tokens != size {
            return Err(Error::Unsupported {
                verb: "`copy_kv` of part of a page on the wafer (whole pages only)",
                engine: "cerebras-spatial",
            });
        }
        pairs.push((run.src_page, run.dst_page));
    }
    w.copy_pages(pairs).map_err(device)
}

/// The token loop: `tokens` at `start..`, then up to `extra` steps each fed
/// the argmax before. The tokens the cache then holds past `start` and the
/// argmax after each where known (the feed's last row on).
pub(super) fn greedy(
    w: &mut wafer::Wafer,
    tokens: &[u32],
    start: u32,
    extra: usize,
) -> EngineResult<(Vec<u32>, Vec<Option<u32>>)> {
    let n = tokens.len();
    let argmax = w.greedy(tokens, start as usize, extra).map_err(device)?;
    let new = argmax[n - 1..].to_vec();
    if new.is_empty() {
        return Err(Error::Device(
            "the model's token loop answered nothing".into(),
        ));
    }
    // Fed: the tokens, then every new one but the last; the argmax after
    // the feed's last row is new[0], after new[i] it is new[i + 1].
    let mut fed = tokens.to_vec();
    fed.extend_from_slice(&new[..new.len() - 1]);
    let mut argmax = vec![None; n - 1];
    argmax.extend(new.iter().map(|&t| Some(t)));
    Ok((fed, argmax))
}

/// The logits `which` asks for (one row for `Last`, a row a token for
/// `All`) of `tokens` at `start..`.
pub(super) fn forward(
    w: &mut wafer::Wafer,
    tokens: &[u32],
    start: u32,
    which: wafer::Logits,
) -> EngineResult<Vec<Vec<f32>>> {
    w.forward(tokens, start as usize, which)
        .map(|(_, logits)| logits)
        .map_err(device)
}

/// A fire's paged lanes on the batched wafer build: each lane's rows as
/// launches of consecutive slots and positions (a prefill chunk, at most a
/// launch's iterations; a lane with a mask a row at a time; every row's
/// logits a row at a time), the lanes' n-th pieces together, in launches of
/// lanes of the same length: up to a launch's lanes (B, or MMAX
/// micro-batches of B on a ring) when no logits come back, the head's few
/// logit lanes when they do.
pub(super) fn fire_paged(
    w: &mut wafer::Wafer,
    lanes: &[&PagedLane],
) -> EngineResult<Vec<PagedAnswer>> {
    if w.paged.is_none() {
        return Err(Error::Device(
            "a batched fire on a one-sequence build".into(),
        ));
    }
    let cap = w.cap.saturating_sub(1).max(1);
    let consecutive = |v: &[u32]| v.windows(2).all(|p| p[1] == p[0] + 1);
    // A verify (every row's argmax) runs as a prefill chunk: the head's
    // argmax comes back for each row.
    let row_by_row: Vec<bool> = lanes
        .iter()
        .map(|lane| {
            lane.visible.is_some()
                || (lane.which == wafer::Logits::All && !lane.argmax_rows)
                || !consecutive(&lane.slots)
                || !consecutive(&lane.positions)
        })
        .collect();
    // Prefill chunks that want no logits (argmax on the wafer, or none) and
    // run no token loop share their launches when a launch takes more lanes
    // than one: cut together (see [`staircase`]), the lanes' n-th pieces run
    // the same rows, not a launch for each prompt length.
    let shared: Vec<usize> = (0..lanes.len())
        .filter(|&i| {
            let lane = lanes[i];
            !row_by_row[i]
                && lane.ahead == 0
                && (lane.argmax_only || lane.argmax_rows || lane.which == wafer::Logits::None)
        })
        .collect();
    let mut cuts: Vec<Option<std::vec::IntoIter<usize>>> = (0..lanes.len()).map(|_| None).collect();
    if w.launch_lanes > 1 && shared.len() > 1 {
        let together = staircase(
            &shared
                .iter()
                .map(|&i| lanes[i].tokens.len())
                .collect::<Vec<_>>(),
            cap,
        );
        for (&i, mine) in shared.iter().zip(together) {
            cuts[i] = Some(mine.into_iter());
        }
    }
    // Each lane's pieces: (first row, rows, logits wanted, iterations).
    let mut pieces: Vec<Vec<(usize, usize, bool, usize)>> = Vec::with_capacity(lanes.len());
    for (li, lane) in lanes.iter().enumerate() {
        let n = lane.tokens.len();
        let row_by_row = row_by_row[li];
        let mut mine = Vec::new();
        let mut at = 0;
        while at < n {
            let rows = match cuts[li].as_mut().and_then(Iterator::next) {
                Some(rows) => rows,
                None if row_by_row => 1,
                None => (n - at).min(cap),
            };
            let last = at + rows == n;
            let want = match lane.which {
                _ if lane.argmax_only || lane.argmax_rows => false,
                wafer::Logits::None => false,
                wafer::Logits::Last => last,
                wafer::Logits::All => true,
            };
            // A token loop row runs its iterations past the one forced.
            let iters = if lane.ahead > 0 { 1 + lane.ahead } else { rows };
            mine.push((at, rows, want, iters));
            at += rows;
        }
        tracing::debug!(
            rows = n,
            pieces = mine.len(),
            argmax_only = lane.argmax_only,
            argmax_rows = lane.argmax_rows,
            which = ?lane.which,
            masked = lane.visible.is_some(),
            "spatial: a paged lane's launches"
        );
        pieces.push(mine);
    }
    let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); lanes.len()];
    let mut argmax: Vec<Vec<u32>> = vec![Vec::new(); lanes.len()];
    let waves = pieces.iter().map(Vec::len).max().unwrap_or(0);
    for wave in 0..waves {
        // The lanes with an n-th piece, by (rows, logits wanted, token
        // loop): a launch's lanes run the same iterations, so token loop
        // lanes share one launch at the fewest any of them may run.
        let mut groups: BTreeMap<(usize, bool, bool), Vec<usize>> = BTreeMap::new();
        for (li, mine) in pieces.iter().enumerate() {
            if let Some(&(_, rows, want, iters)) = mine.get(wave) {
                groups
                    .entry((rows, want, iters > rows))
                    .or_default()
                    .push(li);
            }
        }
        for ((rows, want, _), members) in groups {
            // A launch's lanes (MMAX micro-batches of B on a ring); the head
            // keeps logits rows for the first few only.
            let most = if want { w.logit_lanes } else { w.launch_lanes };
            for chunk in members.chunks(most.max(1)) {
                let iters = chunk
                    .iter()
                    .map(|&li| pieces[li][wave].3)
                    .min()
                    .unwrap_or(rows);
                let batch: Vec<crate::spatial::serve2d::BatchLane> = chunk
                    .iter()
                    .map(|&li| {
                        let lane = lanes[li];
                        let (first, _, _, _) = pieces[li][wave];
                        crate::spatial::serve2d::BatchLane {
                            pos: lane.positions[first],
                            slot: lane.slots[first],
                            pages: lane.pages.clone(),
                            visible: lane
                                .visible
                                .as_ref()
                                .map(|rows| rows[first].iter().flat_map(|&[a, b]| a..b).collect()),
                            forced: lane.tokens[first..first + rows].to_vec(),
                        }
                    })
                    .collect();
                let (argmax_out, rows_out) = w
                    .fire_batch(batch, iters, want)
                    .map_err(|e| Error::Device(format!("the wafer model: {e}")))?;
                for (k, &li) in chunk.iter().enumerate() {
                    if let Some(tokens) = argmax_out.get(k) {
                        argmax[li].extend(tokens.iter().take(iters));
                    }
                }
                if want {
                    for (k, &li) in chunk.iter().enumerate() {
                        if let Some(row) = rows_out.get(k) {
                            logits[li].push(row.clone());
                        }
                    }
                }
            }
        }
    }
    Ok(logits
        .into_iter()
        .zip(argmax)
        .zip(lanes)
        .map(|((rows, argmax), lane)| {
            if lane.argmax_only && lane.ahead > 0 {
                (Vec::new(), argmax)
            } else if lane.argmax_only {
                (Vec::new(), argmax.last().copied().into_iter().collect())
            } else if lane.argmax_rows {
                (Vec::new(), argmax)
            } else {
                (rows, Vec::new())
            }
        })
        .collect())
}

/// Lanes of `rows` rows each cut into pieces together: every lane with rows
/// left runs the fewest any of them has left (at most `cap`), so the lanes'
/// n-th pieces are the same length and share a launch (prompts of 11 to 16
/// rows: 11 rows for all, then 1, 1, 2, 1 for the longer ones, not a launch
/// of each length).
fn staircase(rows: &[usize], cap: usize) -> Vec<Vec<usize>> {
    let mut left = rows.to_vec();
    let mut cuts = vec![Vec::new(); rows.len()];
    while let Some(step) = left.iter().copied().filter(|&n| n > 0).min() {
        let step = step.min(cap.max(1));
        for (n, mine) in left.iter_mut().zip(&mut cuts) {
            if *n > 0 {
                mine.push(step);
                *n -= step;
            }
        }
    }
    cuts
}

/// A paged lane's answer: its logits rows, or (argmax only) its last row's
/// argmax, or (a verify) every row's.
pub(super) type PagedAnswer = (Vec<Vec<f32>>, Vec<u32>);

/// One lane of a fire on a paged model.
pub(super) struct PagedLane {
    pub(super) tokens: Vec<u32>,
    /// The KV slot each row appends at.
    pub(super) slots: Vec<u32>,
    /// The RoPE position each row rotates by.
    pub(super) positions: Vec<u32>,
    pub(super) pages: Vec<u32>,
    /// Per row, the slots it attends as `[start, end)` runs; `None`: every
    /// slot up to its own.
    pub(super) visible: Option<Vec<Vec<[u32; 2]>>>,
    pub(super) which: wafer::Logits,
    /// The lane's one reader takes only its last row's argmax: none of its
    /// logits come back (a launch carries more such lanes than lanes with
    /// logits).
    pub(super) argmax_only: bool,
    /// Its one reader takes every row's argmax (a speculative verify): the
    /// rows run as one prefill chunk and only their argmax come back.
    pub(super) argmax_rows: bool,
    /// An argmax-only decode row run `ahead` steps further in its launch
    /// (the head embedding its own argmax), within the lane's pages.
    pub(super) ahead: usize,
    /// The instance whose epilogue the wafer head's multi-lane VM samples
    /// for this lane (its one row, in its VM lane's bank).
    pub(super) vm: Option<InstanceId>,
}

#[cfg(test)]
mod tests {
    use super::staircase;

    #[test]
    fn prompts_of_other_lengths_share_their_launches() {
        // 11 to 16 rows: all at 11, then the longer ones on.
        let cuts = staircase(&[13, 11, 16, 12, 15], 31);
        assert_eq!(
            cuts,
            vec![
                vec![11, 1, 1],
                vec![11],
                vec![11, 1, 1, 2, 1],
                vec![11, 1],
                vec![11, 1, 1, 2]
            ]
        );
        // Every lane's rows, all of them.
        for (cut, n) in cuts.iter().zip([13, 11, 16, 12, 15]) {
            assert_eq!(cut.iter().sum::<usize>(), n);
        }
        // At most a launch's iterations a piece.
        assert_eq!(staircase(&[40, 70], 31), vec![vec![31, 9], vec![31, 9, 30]]);
        // One length: as one lane alone.
        assert_eq!(staircase(&[12, 12], 31), vec![vec![12], vec![12]]);
        assert_eq!(staircase(&[], 31), Vec::<Vec<usize>>::new());
    }
}
