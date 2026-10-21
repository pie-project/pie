//! A prefill streamed as it is computed. A peer asked to stream a run
//! registers it by its keys before its program starts ([`expect`]); the pass
//! that computes it is known by its pages' keys before its fires are built
//! ([`bind`]), and each of its fires taps the pages it fills ([`tap`]): the
//! engine reads them out a layer at a time as the fire runs, and each layer
//! goes to the asker as it lands, a [`Part`] of the stream its fetch took
//! ([`take`]). The stream ends once the program has ended and every tap has
//! told every layer.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;

use super::prefill::{Body, PageStream, Part};
use crate::inferlet::process::ProcessId;
use crate::store::kv::hash::Hash256;
use crate::store::kv::page_table::HostKvSlotId;
use crate::store::registry::{Stores, with_kv_lock};

/// A run no fetch took is let go after this.
const UNTAKEN_TTL: Duration = Duration::from_secs(60);

/// A run being computed here for an asker.
struct Run {
    /// Its pages' keys from token 0, and the first page it sends.
    keys: Vec<Hash256>,
    from: usize,
    /// How many groups a tap tells.
    groups: usize,
    /// The stream the asker's fetch takes, until it does.
    stream: Option<PageStream>,
    /// Where its parts go; `None` once it has ended.
    parts: Option<mpsc::UnboundedSender<Result<Part, String>>>,
    /// Taps told some layers but not all.
    open: usize,
    /// Its program has ended.
    ended: bool,
    since: Instant,
}

/// A fire's tap: its run, the run's pages it holds (from the run's first
/// sent page), the groups still to come, and the host slots it read into.
struct Tag {
    run: u64,
    first: u32,
    pages: u32,
    left: usize,
    slots: Vec<HostKvSlotId>,
}

#[derive(Default)]
struct Streams {
    runs: HashMap<u64, Run>,
    /// The pass computing each run, by its process.
    bound: HashMap<ProcessId, u64>,
    tags: HashMap<u64, Tag>,
    next: u64,
}

static STREAMS: LazyLock<Mutex<Streams>> = LazyLock::new(Mutex::default);

/// What the engine told of its taps, for [`route`] to take in order: the
/// receiving half until the first run starts it.
struct Told {
    tx: mpsc::UnboundedSender<engine::Tapped>,
    rx: Mutex<Option<mpsc::UnboundedReceiver<engine::Tapped>>>,
}

static TOLD: LazyLock<Told> = LazyLock::new(|| {
    let (tx, rx) = mpsc::unbounded_channel();
    Told {
        tx,
        rx: Mutex::new(Some(rx)),
    }
});

/// The engine's tap sink. It is called on the device's own thread, so it
/// only passes each told layer on.
pub(crate) fn sink() -> engine::TapSink {
    Arc::new(|tapped| {
        let _ = TOLD.tx.send(tapped);
    })
}

/// Registers a run to stream: `keys` from token 0, sent from page `from`.
/// `None` where this worker's engine reads no layer out as it computes.
pub(crate) async fn expect(keys: &[Hash256], from: usize) -> Option<u64> {
    let groups = crate::scheduler::tap_groups(0).await?;
    if let Some(told) = TOLD.rx.lock().unwrap().take() {
        tokio::spawn(route(told));
    }
    let page_bytes = groups.iter().flatten().map(|(_, len)| *len as usize).sum();
    let count = groups.len();
    let (tx, parts) = mpsc::unbounded_channel();
    // The asker lands the pages with its own pass's hashes.
    let stream = PageStream {
        count: keys.len() - from,
        page_bytes,
        layout: crate::scheduler::host_kv_layout(0).await,
        slots: Vec::new(),
        body: Body::Parts { groups, parts },
    };
    let mut streams = STREAMS.lock().unwrap();
    let now = Instant::now();
    let stale: Vec<u64> = streams
        .runs
        .iter()
        .filter(|(_, run)| run.stream.is_some() && now - run.since > UNTAKEN_TTL)
        .map(|(&id, _)| id)
        .collect();
    for id in stale {
        streams.runs.remove(&id);
    }
    streams.next += 1;
    let id = streams.next;
    streams.runs.insert(
        id,
        Run {
            keys: keys.to_vec(),
            from,
            groups: count,
            stream: Some(stream),
            parts: Some(tx),
            open: 0,
            ended: false,
            since: now,
        },
    );
    Some(id)
}

/// The stream of the run a fetch for `keys` from page `from` asks for: the
/// asker names its pass's pages from its own first page, so the run's keys
/// end with them.
pub(crate) fn take(keys: &[Hash256], from: usize) -> Option<PageStream> {
    let mut streams = STREAMS.lock().unwrap();
    let (&id, run) = streams
        .runs
        .iter_mut()
        .find(|(_, run)| run.stream.is_some() && asks_for(&run.keys, run.from, keys, from))?;
    let stream = run.stream.take();
    // A run already over has said all it will to the stream.
    if run.parts.is_none() {
        streams.runs.remove(&id);
    }
    stream
}

/// Whether a fetch for `keys` from page `from` (both counted from the
/// asker's pass's first page) asks for the run `run` sends from `run_from`
/// (counted from token 0).
fn asks_for(run: &[Hash256], run_from: usize, keys: &[Hash256], from: usize) -> bool {
    run.len()
        .checked_sub(keys.len())
        .is_some_and(|first| run[first..] == *keys && run_from == first + from)
}

/// The pass of `pid` over `slots` past `held` computes a registered run:
/// its fires tap from now on.
pub(crate) fn bind(
    pid: ProcessId,
    stores: &Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    held: u64,
    slots: &[Hash256],
) {
    let mut streams = STREAMS.lock().unwrap();
    if streams.runs.is_empty() || streams.bound.contains_key(&pid) {
        return;
    }
    let Some(keys) = with_kv_lock(&stores.kv, "offload-stream", |kv| {
        kv.page_keys(ws, held, slots, stores.kv_page_size)
            .ok()
            .flatten()
    }) else {
        return;
    };
    let bound: Vec<u64> = streams.bound.values().copied().collect();
    let run = streams
        .runs
        .iter()
        .find(|(id, run)| run.keys == keys && !run.ended && !bound.contains(id))
        .map(|(&id, _)| id);
    if let Some(run) = run {
        streams.bound.insert(pid, run);
    }
}

/// A fire of `pid`'s pass: if the pass computes a run, its lane taps the
/// run's pages the fire fills, into host slots of their own.
pub(crate) fn tap(pid: ProcessId, stores: &Stores, req: &mut crate::engine::FireRequest) {
    let mut streams = STREAMS.lock().unwrap();
    let Some(&run) = streams.bound.get(&pid) else {
        return;
    };
    let Some((from, groups)) = streams.runs.get(&run).map(|run| (run.from, run.groups)) else {
        return;
    };
    let [lane] = req.lanes.as_mut_slice() else {
        fail(
            &mut streams,
            run,
            "a streamed pass fires more than one lane".into(),
        );
        return;
    };
    let page = u64::from(stores.kv_page_size.max(1));
    let (held, rows) = (u64::from(lane.kv.held), lane.tokens.len() as u64);
    let filled = ((held / page) as usize).max(from)..((held + rows) / page) as usize;
    if filled.is_empty() {
        return;
    }
    let Some(pages) = lane.kv.pages.get(filled.clone()).map(<[u32]>::to_vec) else {
        fail(
            &mut streams,
            run,
            "a streamed fire's pages are past its lane's".into(),
        );
        return;
    };
    let Some(slots) = with_kv_lock(&stores.kv, "offload-stream", |kv| {
        kv.alloc_host_slots(pages.len())
    }) else {
        fail(
            &mut streams,
            run,
            format!(
                "no room in the host tier to read {} pages out to",
                pages.len()
            ),
        );
        return;
    };
    streams.next += 1;
    let tag = streams.next;
    lane.tap = Some(engine::fire::Tap {
        tag,
        pages,
        slots: slots.iter().map(|slot| slot.0).collect(),
    });
    streams.tags.insert(
        tag,
        Tag {
            run,
            first: (filled.start - from) as u32,
            pages: filled.len() as u32,
            left: groups,
            slots,
        },
    );
    if let Some(run) = streams.runs.get_mut(&run) {
        run.open += 1;
    }
}

/// The run's program has ended, as `ran` says.
pub(crate) fn done(run: u64, ran: Result<()>) {
    let mut streams = STREAMS.lock().unwrap();
    streams.bound.retain(|_, bound| *bound != run);
    match ran {
        Ok(()) => {
            if let Some(run) = streams.runs.get_mut(&run) {
                run.ended = true;
            }
            close_if_whole(&mut streams, run);
        }
        Err(error) => fail(&mut streams, run, format!("the prefill failed: {error:#}")),
    }
}

/// Every layer every tap of the run read has gone: the stream ends.
fn close_if_whole(streams: &mut Streams, id: u64) {
    if let Some(run) = streams.runs.get_mut(&id)
        && run.ended
        && run.open == 0
    {
        run.parts = None;
        if run.stream.is_none() {
            streams.runs.remove(&id);
        }
    }
}

/// The run stops short: its stream says why, to a fetch that took it or
/// one still to come.
fn fail(streams: &mut Streams, id: u64, why: String) {
    streams.bound.retain(|_, bound| *bound != id);
    let Some(run) = streams.runs.get_mut(&id) else {
        return;
    };
    run.ended = true;
    if let Some(parts) = run.parts.take() {
        tracing::warn!(target: "pie::offload", %why, "a streamed prefill stopped");
        let _ = parts.send(Err(why));
    }
    if run.stream.is_none() {
        streams.runs.remove(&id);
    }
}

/// Sends each told layer on to its run's asker, and gives a tap's host
/// slots back once it has told them all.
async fn route(mut told: mpsc::UnboundedReceiver<engine::Tapped>) {
    while let Some(tapped) = told.recv().await {
        let freed = {
            let mut streams = STREAMS.lock().unwrap();
            let Some(tag) = streams.tags.get_mut(&tapped.tag) else {
                continue;
            };
            let run = tag.run;
            match tapped.bytes {
                Ok(bytes) => {
                    let Some(left) = tag.left.checked_sub(1) else {
                        let tag = streams.tags.remove(&tapped.tag).expect("the tag told");
                        fail(
                            &mut streams,
                            run,
                            format!("a tap told group {} past its groups", tapped.group),
                        );
                        release(tag.slots);
                        continue;
                    };
                    tag.left = left;
                    let part = Part {
                        group: tapped.group,
                        first: tag.first,
                        pages: tag.pages,
                        bytes,
                    };
                    let whole = tag.left == 0;
                    if let Some(parts) = streams.runs.get(&run).and_then(|run| run.parts.as_ref()) {
                        let _ = parts.send(Ok(part));
                    }
                    if whole {
                        let tag = streams.tags.remove(&tapped.tag).expect("the tag told");
                        if let Some(run) = streams.runs.get_mut(&run) {
                            run.open -= 1;
                        }
                        close_if_whole(&mut streams, run);
                        Some(tag.slots)
                    } else {
                        None
                    }
                }
                Err(why) => {
                    let tag = streams.tags.remove(&tapped.tag).expect("the tag told");
                    fail(&mut streams, run, format!("reading a layer out: {why}"));
                    Some(tag.slots)
                }
            }
        };
        if let Some(slots) = freed {
            release(slots);
        }
    }
}

/// A tap's host slots, back to the host tier.
fn release(slots: Vec<HostKvSlotId>) {
    if let Some(stores) = crate::store::registry::try_get(0, 0) {
        with_kv_lock(&stores.kv, "offload-stream", |kv| kv.free_host_slots(slots));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fetch_names_a_run_by_its_pass_s_pages() {
        let run: Vec<Hash256> = (1..=5).map(|n| [n; 32]).collect();
        // A pass whose first page is the run's third: it asks for pages 2..
        // as 0.., and the run sends from its fourth.
        assert!(asks_for(&run, 3, &run[2..], 1));
        assert!(!asks_for(&run, 3, &run[2..], 0), "another first page");
        assert!(
            !asks_for(&run, 3, &run[1..4], 2),
            "not the run's last pages"
        );
        assert!(!asks_for(&run[..2], 0, &run, 0), "longer than the run");
    }
}
