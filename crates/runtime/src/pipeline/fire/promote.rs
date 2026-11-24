//! Promoting a pass: sealing pages on landing, supplying what it reads, promoting and resuming it.

use super::*;

pub(super) static SEALED_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a keyed pass claimed to compute, and the rows it computes: passes
/// whose first pages it claimed wait for its fires to land. Landed whole,
/// the rows it froze seal there and then, and the waiters map them in;
/// otherwise (failed, refused, dropped) the claims go unsealed, and a waiter
/// computes the pages itself. Never waits on anything: it runs where the
/// pass's last fire resolves, or goes as the pass does.
pub(super) struct SealOnLanding {
    pub(super) claims: crate::pipeline::coalesce::ClaimGuard,
    pub(super) stores: crate::store::registry::Stores,
    pub(super) ws: crate::store::kv::page_table::WorkingSetId,
    pub(super) start: u64,
    pub(super) slots: Vec<crate::store::kv::hash::Hash256>,
    pub(super) page_size: u32,
    pub(super) failure: PipelineFailure,
}

impl SealOnLanding {
    /// Seals once every one of the pass's `fires` has settled, whatever
    /// order they land in, if each landed; a hook never run (its fire gone
    /// unresolved, or the runtime with it) counts as not landed, or drops
    /// the claims as the last hook goes.
    pub(super) fn on_landing(self, fires: &[crate::engine::WorkItemCompletion]) {
        struct Countdown {
            left: std::sync::atomic::AtomicUsize,
            whole: std::sync::atomic::AtomicBool,
            sealing: Mutex<Option<SealOnLanding>>,
        }
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        let countdown = Arc::new(Countdown {
            left: std::sync::atomic::AtomicUsize::new(fires.len()),
            whole: std::sync::atomic::AtomicBool::new(true),
            sealing: Mutex::new(Some(self)),
        });
        for fire in fires {
            let countdown = Arc::clone(&countdown);
            fire.on_settled_landed(move |landed| {
                if !landed {
                    countdown.whole.store(false, Release);
                }
                if countdown.left.fetch_sub(1, AcqRel) == 1
                    && let Some(sealing) = countdown.sealing.lock().unwrap().take()
                {
                    sealing.landed(countdown.whole.load(Acquire));
                }
            });
        }
    }

    pub(super) fn landed(self, landed: bool) {
        if landed && pipeline_failed(&self.failure).is_none() {
            let sealed =
                crate::store::registry::with_kv_lock(&self.stores.kv, "host-other", |kv| {
                    kv.seal_landed(self.ws, self.start, &self.slots, self.page_size)
                });
            match sealed {
                Ok(pages) => {
                    SEALED_PAGES.fetch_add(u64::from(pages), std::sync::atomic::Ordering::Relaxed);
                }
                Err(error) => tracing::debug!(%error, "kv supply: sealing what a pass landed"),
            }
        }
        drop(self.claims);
    }
}

pub(super) struct SupplyCounts {
    pub(super) passes: std::sync::atomic::AtomicU64,
    pub(super) tokens: std::sync::atomic::AtomicU64,
    pub(super) held: std::sync::atomic::AtomicU64,
    pub(super) found: std::sync::atomic::AtomicU64,
    pub(super) mapped: std::sync::atomic::AtomicU64,
}

pub(super) static SUPPLIED: SupplyCounts = SupplyCounts {
    passes: std::sync::atomic::AtomicU64::new(0),
    tokens: std::sync::atomic::AtomicU64::new(0),
    held: std::sync::atomic::AtomicU64::new(0),
    found: std::sync::atomic::AtomicU64::new(0),
    mapped: std::sync::atomic::AtomicU64::new(0),
};

#[derive(Clone, Copy)]
pub(super) enum Attach {
    No,
    Kv,
    /// KV pages on a fresh state would read as another sequence.
    Checkpoint(crate::store::rs::RsWorkingSetId),
}

#[derive(Default)]
pub(super) struct Supplied {
    pub(super) rows: u32,
    pub(super) keys: Vec<crate::store::kv::hash::Hash256>,
}

/// Maps in the sealed pages holding a keyed pass's first rows, short of its
/// last row's page, which is read out so always computed. Mapping freezes
/// nothing: what the set has folded is what its passes state. A recurrent
/// state is folded through the checkpoint it resumes from, so it resumes no
/// further than the `folds` rows the pass folds.
pub(super) fn supply(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    full: &crate::engine::FireRequest,
    slots: &[crate::store::kv::hash::Hash256],
    attach: Attach,
    folds: u64,
    page_size: u32,
) -> Supplied {
    use std::sync::atomic::Ordering::Relaxed;
    let held = u64::from(full.lanes[0].kv.held);
    let rows = slots.len() as u64;
    let page = u64::from(page_size.max(1));
    let limit = (held + rows - 1) / page;
    let tokens_of = |pages: u64| &full.lanes[0].tokens[..((pages * page) - held) as usize];
    let fresh = (held == 0).then(|| crate::store::kv::KvStore::fresh_page_keys(slots, page_size));
    let looked = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let keys = match fresh {
            Some(keys) => keys,
            None => {
                let Some(keys) = kv.page_keys(ws, held, slots, page_size)? else {
                    return Ok(None);
                };
                keys
            }
        };
        // A pass that may map nothing in looks nothing up.
        let run = match attach {
            Attach::No => Vec::new(),
            _ => kv.cached_run(&keys),
        };
        let clean = held.is_multiple_of(page) && kv.mapped_len(ws)? == held / page;
        let mapped = match attach {
            Attach::Kv if clean && kv.keeps() => {
                let mapped = kv.attach_run(ws, &run, limit)?;
                kv.log_rows(ws, held, tokens_of(held / page + mapped))?;
                mapped
            }
            Attach::Kv if !run.is_empty() => {
                tracing::debug!(
                    target: "pie::kv",
                    found = run.len(),
                    held,
                    why = if kv.keeps() {
                        "the set's rows end inside a page or past what it maps"
                    } else {
                        "the store keeps nothing"
                    },
                    "attach: found pages not mapped"
                );
                0
            }
            _ => 0,
        };
        Ok::<_, crate::store::kv::KvStoreError>(Some((run, keys, mapped, clean)))
    });
    let (run, keys, mut mapped, clean) = match looked {
        Ok(Some(found)) => found,
        Ok(None) => return Supplied::default(),
        Err(error) => {
            tracing::warn!(%error, "kv supply: the lookup failed; the pass computes all it writes");
            return Supplied::default();
        }
    };
    if let Attach::Checkpoint(rs) = attach
        && clean
        && held == 0
    {
        mapped = resume(
            stores,
            ws,
            rs,
            &keys,
            run.len().min((limit.min(folds / page)) as usize),
            page_size,
        );
        if mapped > 0 {
            let logged = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.log_rows(ws, 0, tokens_of(mapped))
            });
            if let Err(error) = logged {
                tracing::debug!(%error, "kv supply: logging what was mapped in");
            }
        }
    }
    let total = held + rows;
    // As far as it could map: whole pages short of the last row's.
    let found = (run.len() as u64).min(limit) * page;
    let mapped_rows = mapped * page;
    let passes = SUPPLIED.passes.fetch_add(1, Relaxed) + 1;
    let tokens = SUPPLIED.tokens.fetch_add(total, Relaxed) + total;
    let held = SUPPLIED.held.fetch_add(held, Relaxed) + held;
    let found = SUPPLIED.found.fetch_add(found, Relaxed) + found;
    let mapped_total = SUPPLIED.mapped.fetch_add(mapped_rows, Relaxed) + mapped_rows;
    if passes.is_multiple_of(256) {
        tracing::info!(
            passes,
            tokens,
            held,
            found,
            mapped = mapped_total,
            sealed_pages = SEALED_PAGES.load(Relaxed),
            "kv supply: over keyed passes, what sealed pages held and what was mapped in"
        );
    }
    Supplied {
        rows: u32::try_from(mapped_rows).unwrap_or(0),
        keys,
    }
}

/// Runs under a fire lease, so its own process is not suspended under it; the
/// run stays held until mapped in.
pub(super) async fn promote<C: FireContext>(
    ctx: &mut C,
    stores: &crate::store::registry::Stores,
    ws: &KvWorkingSet,
    full: &crate::engine::FireRequest,
    slots: &[crate::store::kv::hash::Hash256],
    pipeline_id: uuid::Uuid,
) -> Result<Option<Promoted>, String> {
    let (engine, page_size) = (ws.engine, stores.kv_page_size);
    let held = u64::from(full.lanes[0].kv.held);
    let page = u64::from(page_size.max(1));
    let limit = ((held + slots.len() as u64 - 1) / page) as usize;
    let fresh = (held == 0).then(|| crate::store::kv::KvStore::fresh_page_keys(slots, page_size));
    let found = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let keys = match fresh {
            Some(keys) => keys,
            None => {
                let Some(keys) = kv.page_keys(ws.id, held, slots, page_size)? else {
                    return Ok::<_, crate::store::kv::KvStoreError>(None);
                };
                keys
            }
        };
        let mut run = kv.cached_run_any_tier(&keys);
        run.truncate(limit);
        let below = kv.below_device(&run);
        Ok((below > 0).then_some((run, below)))
    })
    .map_err(|error| error.to_string())?;
    let Some((run, below)) = found else {
        return Ok(None);
    };
    let demand = crate::planner::Demand {
        kv_pages: u32::try_from(below).unwrap_or(u32::MAX),
        rs: Default::default(),
    };
    let Some(mut grant) = acquire_grant(ctx, pipeline_id, demand, false, || Ok(Vec::new())).await? else {
        return Ok(None);
    };
    let Ok(lease) = ws.fire_lease() else {
        return Ok(None);
    };
    let prepared = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        kv.prepare_promote(&run, grant.lend_kv())
    })
    .map_err(|error| error.to_string())?;
    let Some(txn) = prepared else {
        return Ok(None);
    };
    let mut guard = PromoteGuard {
        stores: stores.clone(),
        engine,
        txn: Some(txn),
        in_flight: Vec::new(),
        submitted: false,
    };
    let (kind, plans) = guard.txn.as_ref().map_or_else(
        || (crate::scheduler::TierKind::Kv, Vec::new()),
        |txn| {
            (
                crate::scheduler::TierKind::kv(txn.shape()),
                txn.copy_plans(),
            )
        },
    );
    for (tier, device, slots) in plans {
        let copy = crate::scheduler::TierMove {
            kind,
            tier,
            out: false,
            device,
            slots,
            source: None,
        };
        let completion = crate::scheduler::copy_tier_tracked(engine, &copy)
            .map_err(|error| format!("{error:#}"))?;
        guard.submitted = true;
        guard.in_flight.push(completion.clone());
        completion.wait().await.map_err(|error| error.to_string())?;
    }
    guard.in_flight.clear();
    let txn = guard.txn.take().expect("taken once");
    let pin =
        crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| kv.commit_promote(txn))
            .map_err(|error| error.to_string())?;
    Ok(Some(Promoted {
        pages: below as u64,
        stores: stores.clone(),
        pin: Some(pin),
        _lease: lease,
    }))
}

/// Dropped before it commits, it gives back what it took once any copy it
/// started has landed: written its destinations, not only settled.
pub(super) struct PromoteGuard {
    pub(super) stores: crate::store::registry::Stores,
    pub(super) engine: usize,
    pub(super) txn: Option<crate::store::kv::KvResidencyTxn>,
    pub(super) in_flight: Vec<crate::scheduler::ControlCompletion>,
    /// A copy of the promote was handed to the engine.
    pub(super) submitted: bool,
}

impl Drop for PromoteGuard {
    fn drop(&mut self) {
        let Some(txn) = self.txn.take() else {
            return;
        };
        let stores = self.stores.clone();
        let in_flight = std::mem::take(&mut self.in_flight);
        let abort = move || {
            crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
                kv.abort_restore(txn);
            });
            for planner in crate::planner::planners() {
                planner.pages_freed();
            }
        };
        if !self.submitted {
            abort();
            return;
        }
        if !crate::rt::has_runtime() {
            tracing::error!(
                "a promote dropped with a copy handed to the engine and no runtime; \
                 preserving its pages to avoid reuse under the copy"
            );
            return;
        }
        let engine = self.engine;
        crate::rt::spawn(async move {
            let settled = crate::scheduler::settle_copies(engine);
            if let Err(error) =
                crate::scheduler::abort_once_landed(in_flight, settled, abort).await
            {
                tracing::error!(
                    engine,
                    %error,
                    "a promote's copies did not settle; preserving its pages to avoid reuse \
                     under a copy"
                );
            }
        });
    }
}

pub(super) struct Promoted {
    /// Pages of the run that came up from below the device.
    pub(super) pages: u64,
    pub(super) stores: crate::store::registry::Stores,
    pub(super) pin: Option<crate::store::kv::ReaderPin>,
    pub(super) _lease: crate::store::kv::working_set::KvFireLease,
}

impl Drop for Promoted {
    fn drop(&mut self) {
        if let Some(pin) = self.pin.take() {
            crate::store::registry::with_kv_lock(&self.stores.kv, "host-other", |kv| {
                kv.release_reader(pin);
            });
        }
    }
}

/// Resumes a hybrid pass from its furthest checkpoint within `reach` pages,
/// which both halves are then folded through; none when either half is gone
/// by then, and `rs` holds no state again, the set having folded nothing.
pub(crate) fn resume(
    stores: &crate::store::registry::Stores,
    ws: crate::store::kv::page_table::WorkingSetId,
    rs: crate::store::rs::RsWorkingSetId,
    keys: &[crate::store::kv::hash::Hash256],
    reach: usize,
    page_size: u32,
) -> u64 {
    let adopted = {
        let mut store = stores.rs.lock().unwrap();
        (1..=reach).rev().find(|&pages| {
            let key = &keys[pages - 1];
            store.has_checkpoint(key) && store.adopt_checkpoint(rs, key).unwrap_or(false)
        })
    };
    let Some(pages) = adopted else {
        return 0;
    };
    let pages = pages as u64;
    let mapped = crate::store::registry::with_kv_lock(&stores.kv, "host-other", |kv| {
        let run = kv.cached_run(&keys[..pages as usize]);
        if run.len() as u64 != pages {
            return Ok(0);
        }
        let mapped = kv.attach_run(ws, &run, pages)?;
        if mapped == pages {
            kv.freeze(ws, pages * u64::from(page_size.max(1)))?;
        }
        Ok::<_, crate::store::kv::KvStoreError>(mapped)
    });
    match mapped {
        Ok(mapped) if mapped == pages => pages,
        outcome => {
            if let Err(error) = outcome {
                tracing::debug!(%error, "kv supply: mapping a checkpoint's pages in");
            }
            if let Err(error) = stores.rs.lock().unwrap().forget_state(rs) {
                tracing::debug!(%error, "kv supply: dropping a state not resumed");
            }
            0
        }
    }
}
