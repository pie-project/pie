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
    // Each lane's pieces: (first row, rows, logits wanted, iterations).
    let mut pieces: Vec<Vec<(usize, usize, bool, usize)>> = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let n = lane.tokens.len();
        let consecutive = |v: &[u32]| v.windows(2).all(|p| p[1] == p[0] + 1);
        // A verify (every row's argmax) runs as a prefill chunk: the head's
        // argmax comes back for each row.
        let row_by_row = lane.visible.is_some()
            || (lane.which == wafer::Logits::All && !lane.argmax_rows)
            || !consecutive(&lane.slots)
            || !consecutive(&lane.positions);
        let mut mine = Vec::new();
        let mut at = 0;
        while at < n {
            let rows = if row_by_row { 1 } else { (n - at).min(cap) };
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
