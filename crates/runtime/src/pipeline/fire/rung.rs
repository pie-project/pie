//! The rung a pass asks for and the funding that raises it.

use super::*;

/// The rows past which a pass's ask is its process's between passes
/// ([`crate::store::shared::hold_for_process`]): a decode's few rows ask
/// only while they are queued.
pub(super) const RUNG_WANT_FROM: usize = 16;

/// How many of a pass's pieces run ahead of the one being cut while its
/// rung's raise is pending: the device stays fed, and the rest is cut wider
/// once the raise lands.
pub(super) const PIECES_AHEAD: usize = 2;

/// A pass's ask for the activation rung, raised from free room: where the
/// raise lacks chunks that kv holds (the span a lowered rung lent among
/// them), it asks no planner, since a rise funded as a kv demand would wait
/// among the waiters it starves, or ahead of them; it says it waits for
/// room ([`RaiseFunding::waits_for_room`]), the pass fires at the rung held,
/// and the rise is taken once the pool gives room back (its room signal,
/// [`crate::store::slots::SlotPool::room_watch`]). Runs beside the pass
/// until the raise lands or the pass is through (dropped). A piece that
/// cannot fire narrower asks for its rise itself ([`RungAsk::wait`]).
pub(super) struct RaiseFunding {
    pub(super) stop: Arc<tokio::sync::Notify>,
    pub(super) stopped: Arc<std::sync::atomic::AtomicBool>,
    /// The rise is short of free room: the pass fires at the rung held.
    pub(super) short: Arc<std::sync::atomic::AtomicBool>,
}

impl RaiseFunding {
    pub(super) fn start(shared: crate::store::shared::Shared, engine: usize, rows: u32) -> Self {
        let stop = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let short = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let funding = Self {
            stop: stop.clone(),
            stopped: stopped.clone(),
            short: short.clone(),
        };
        crate::rt::spawn(async move {
            let target = crate::store::shared::cap_for(&shared, rows);
            let says = |waits: bool| {
                if short.swap(waits, std::sync::atomic::Ordering::AcqRel) != waits {
                    // The pass reads it at its next cut.
                    crate::store::shared::rung_changed(engine);
                }
            };
            // A step with nothing to do waits on the rung's next change (a
            // raise, a copy, the cap), on room given back where the rise is
            // short of it, or on the stop; each armed before it looks.
            while !stopped.load(std::sync::atomic::Ordering::Acquire) {
                let watch = crate::store::shared::rung_watch(engine);
                tokio::pin!(watch);
                watch.as_mut().enable();
                let room = shared.lock().unwrap().pool().room_watch();
                crate::store::shared::raise_now(&shared, engine, rows);
                if crate::store::shared::fire_cap(engine) >= target {
                    says(false);
                    return;
                }
                // Quoted at the rung the raise is taken at: the one the
                // engine holds where that is wider
                // (`SharedPool::raise_at`).
                let quote = {
                    let pool = shared.lock().unwrap();
                    let goal = pool.rung_for(rows);
                    pool.raise_at(goal).map(|at| {
                        let (chunks, _) = pool.raise_short(at);
                        let spc = pool.list().map_or(1, |list| list.slots_per_chunk.max(1));
                        (chunks, pool.pool().held_in_span(at).saturating_mul(spc))
                    })
                };
                let Some((chunks, held)) = quote else {
                    // The engine holds a rung the guest terms no longer
                    // fit: the walk brings it down first, and tells the
                    // rung.
                    crate::store::shared::ask_walk(engine);
                    tokio::select! {
                        () = &mut watch => {}
                        () = stop.notified() => return,
                    }
                    continue;
                };
                if chunks == 0 {
                    // The room is there and state lies in the span: the
                    // walker moves it out, the engine copying it, and
                    // tells the rung.
                    says(false);
                    tokio::select! {
                        () = &mut watch => {}
                        () = stop.notified() => return,
                    }
                    continue;
                }
                // Short of free room: the chunks the rise lacks are kv's.
                // The pass fires at the rung held, and the rise is asked
                // again only once the pool gives room back.
                if !short.load(std::sync::atomic::Ordering::Acquire) {
                    tracing::debug!(
                        engine,
                        rows,
                        chunks,
                        held,
                        "a rise waits for room: {held} pages held by kv"
                    );
                }
                says(true);
                crate::store::shared::ask_walk(engine);
                tokio::select! {
                    () = room.given() => {}
                    () = &mut watch => {}
                    () = stop.notified() => return,
                }
            }
        });
        funding
    }

    /// Whether the rise is short of free room: the pass fires at the rung
    /// held until room is given back.
    pub(super) fn waits_for_room(&self) -> bool {
        self.short.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A pass's ask of the activation rung, the one rule for every fire the
/// host stages (a prompt's cut pieces, a decode's few rows, a
/// device-carried decode staged whole): a pass asks the rung for the rows
/// its fire takes before its first fire is staged and holds the ask until
/// its last has settled, whatever the cap is when it is cut; only a lone
/// row the cap covers asks nothing, no cap being narrower than it.
///
/// The scheduler keeps a fire past the cap queued until the rung is back,
/// and the walker, bringing the rung down, cuts fires to what is asked, the
/// live lanes' one row each otherwise: a fire wider than a row staged under
/// a cap that covered it, asking nothing, sat queued for good once the
/// walker took the cap down under it.
pub(super) struct RungAsk {
    pub(super) engine: usize,
    pub(super) pid: uuid::Uuid,
    pub(super) quorum: uuid::Uuid,
    /// The pool whose rung it asks, and the stores' kv, whose fire epoch a
    /// rise's copy is carried at.
    pub(super) shared: Option<crate::store::shared::Shared>,
    pub(super) kv: Arc<parking_lot::Mutex<crate::store::kv::KvStore>>,
    /// Held (the walker cuts nothing under it) until dropped.
    pub(super) want: crate::store::shared::RungWant,
    /// The raise the rung lacks for it, taken beside the pass from free
    /// room; `None` where fires may take the rows now.
    pub(super) funding: Option<RaiseFunding>,
}

/// Asks the rung on `engine` for the `rows` a fire of a pass of `own` rows
/// takes ([`RungAsk`]; `None`: a lone row the cap covers): the rung is
/// raised at once where nothing moves, a cap the walker lowered behind a
/// rung still up is lifted back, and what the raise lacks is waited for
/// beside the pass ([`RaiseFunding`]).
pub(super) fn ask_rung(
    stores: &crate::store::registry::Stores,
    engine: usize,
    pid: uuid::Uuid,
    quorum: uuid::Uuid,
    own: usize,
    rows: u32,
) -> Option<RungAsk> {
    if own <= 1 && rows <= crate::store::shared::fire_cap(engine) {
        return None;
    }
    // Held before the cap is read: the walker takes no cap down past an
    // ask, and `raise_now` lifts one it took before, both in the pool's
    // lock.
    let want = crate::store::shared::want_rows(engine, rows);
    let funding = stores.shared.as_ref().and_then(|shared| {
        crate::store::shared::raise_now(shared, engine, rows);
        (crate::store::shared::fire_cap(engine) < crate::store::shared::cap_for(shared, rows))
            .then(|| RaiseFunding::start(shared.clone(), engine, rows))
    });
    Some(RungAsk {
        engine,
        pid,
        quorum,
        shared: stores.shared.clone(),
        kv: stores.kv.clone(),
        want,
        funding,
    })
}

impl RungAsk {
    /// Returns once fires may take `width` rows, or with what the planner
    /// refused the rise. A piece of the held width returns at once; one
    /// wider cannot fire narrower (no cut brings it under the rung held),
    /// so its rise is asked as a keyed demand of its process, as a
    /// program's fire term is ([`crate::store::shared::programs::fund_span`]),
    /// served in key order; never on a rung that does not come.
    pub(super) async fn wait(&self, width: u32) -> Result<(), String> {
        let engine = self.engine;
        let (Some(shared), Some(_)) = (self.shared.as_ref(), self.funding.as_ref()) else {
            // Nothing funds a raise (no shared pool): the cap never moves.
            return Ok(());
        };
        if crate::store::shared::fire_cap(engine) >= width {
            return Ok(());
        }
        // Until the raise, the process submits nothing: no frame waits on
        // its lanes (the frame's firing frees what the raise is funded
        // with; the frame is not what funds it). Its owner is told blocked
        // from its first look short of the rung (`parks_here`) until the
        // rung is there, and unblocked as that park ends: no owner is left
        // blocked with no unblock to come.
        let funder = crate::store::shared::programs::Planner {
            pid: self.pid,
            quorum: self.quorum,
        };
        let kv = self.kv.clone();
        let epoch_now =
            move || crate::store::registry::with_kv_lock(&kv, "rung-rise", |kv| kv.current_epoch());
        let mover = crate::store::shared::programs::LaneMover {
            engine,
            epoch_now: &epoch_now,
        };
        crate::inferlet::process::said("enqueue: the activation rung", self.pid, async {
            // Parked once it waits, it stays parked until the rung is
            // there: between two waits (the funding's, the watch's) its
            // owner was unblocked, its lanes expected again, and a frame
            // holding three complete fronts waited on a lane whose process
            // waited on a raise that frame held up (the rung lowered under a
            // wider ask).
            let mut parked = None;
            loop {
                let watch = crate::store::shared::rung_watch(engine);
                tokio::pin!(watch);
                watch.as_mut().enable();
                if crate::store::shared::fire_cap(engine) >= width {
                    return Ok(());
                }
                if parked.is_none() {
                    parked = crate::inferlet::process::parks_here();
                }
                crate::store::shared::programs::fund_span(shared, || width, &funder, &mover)
                    .await?;
                crate::store::shared::raise_now(shared, engine, width);
                if crate::store::shared::fire_cap(engine) >= width {
                    return Ok(());
                }
                // Waiting on a raise of the rung: its owner is parked here.
                watch.await;
            }
        })
        .await
    }

    /// Whether the rise is short of free room, so the pass fires at the rung
    /// held: read before each piece is cut and submitted, never learnt by
    /// a wide fire refused.
    pub(super) fn fires_at_held_rung(&self) -> bool {
        self.funding
            .as_ref()
            .is_some_and(RaiseFunding::waits_for_room)
    }

    /// The ask stands until `last`, the pass's last fire, settles; its
    /// funding ends here, every fire staged.
    pub(super) fn until_settled(self, last: &crate::engine::completion::WorkItemCompletion) {
        let want = self.want;
        last.on_settled(move || drop(want));
    }
}

impl Drop for RaiseFunding {
    fn drop(&mut self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Release);
        self.stop.notify_one();
    }
}
