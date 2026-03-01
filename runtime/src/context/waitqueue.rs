//! Wait queue types and drain logic for deferred page allocation and restore.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;
use tokio::sync::oneshot;
use anyhow::Result;

use crate::process::ProcessId;
use crate::device::DeviceId;

use super::{CONTEXTS, ContextId, ContextState, ResidentResult};
use super::manager::ContextManager;
use std::collections::HashMap;

// =============================================================================
// PageWaiter — unified wait queue entry
// =============================================================================

/// A deferred allocation or restore request, held by the actor until pages
/// become available. Ordered by effective floor (priority_floor + age boost)
/// descending — highest first. Aging prevents starvation: every waiter's
/// effective priority grows over time until it is eventually served.
pub(crate) enum PageWaiter {
    /// Waiting for `reserve_pages` to succeed.
    Allocate {
        context_id: ContextId,
        device: DeviceId,
        num_pages: usize,
        requester: Option<ProcessId>,
        priority_floor: f64,
        enqueued_at: Instant,
        response: oneshot::Sender<Result<()>>,
    },
    /// Waiting for `ensure_resident` to succeed.
    Restore {
        context_id: ContextId,
        device: DeviceId,
        requester: Option<ProcessId>,
        priority_floor: f64,
        enqueued_at: Instant,
        response: oneshot::Sender<Result<ResidentResult>>,
    },
}

/// Priority boost per second of waiting. A waiter gains this much
/// effective priority for each second it sits in the queue.
pub(crate) const AGING_RATE: f64 = 1.0;

impl PageWaiter {
    fn priority_floor(&self) -> f64 {
        match self {
            PageWaiter::Allocate { priority_floor, .. } => *priority_floor,
            PageWaiter::Restore { priority_floor, .. } => *priority_floor,
        }
    }

    fn enqueued_at(&self) -> Instant {
        match self {
            PageWaiter::Allocate { enqueued_at, .. } => *enqueued_at,
            PageWaiter::Restore { enqueued_at, .. } => *enqueued_at,
        }
    }

    /// Priority floor boosted by waiting time. Guarantees starvation-freedom:
    /// any waiter eventually reaches the top of the queue.
    pub(crate) fn effective_floor(&self) -> f64 {
        let age = self.enqueued_at().elapsed().as_secs_f64();
        self.priority_floor() + age * AGING_RATE
    }
}

impl std::fmt::Debug for PageWaiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageWaiter::Allocate { context_id, device, num_pages, priority_floor, enqueued_at, .. } => {
                f.debug_struct("Allocate")
                    .field("context_id", context_id)
                    .field("device", device)
                    .field("num_pages", num_pages)
                    .field("priority_floor", priority_floor)
                    .field("age_ms", &enqueued_at.elapsed().as_millis())
                    .finish()
            }
            PageWaiter::Restore { context_id, device, priority_floor, enqueued_at, .. } => {
                f.debug_struct("Restore")
                    .field("context_id", context_id)
                    .field("device", device)
                    .field("priority_floor", priority_floor)
                    .field("age_ms", &enqueued_at.elapsed().as_millis())
                    .finish()
            }
        }
    }
}

impl PartialEq for PageWaiter {
    fn eq(&self, other: &Self) -> bool {
        self.effective_floor() == other.effective_floor()
    }
}
impl Eq for PageWaiter {}

impl PartialOrd for PageWaiter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PageWaiter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.effective_floor()
            .partial_cmp(&other.effective_floor())
            .unwrap_or(Ordering::Equal)
    }
}

// =============================================================================
// WaitNeeded sentinel
// =============================================================================

/// Returned by allocation/restore when no victims can be evicted.
pub(crate) enum WaitNeeded {
    /// No lower-priority contexts to evict; caller should enqueue.
    NeedPages,
    /// A real error (context not found, etc.) — propagate to caller.
    Fatal(anyhow::Error),
}

impl From<anyhow::Error> for WaitNeeded {
    fn from(e: anyhow::Error) -> Self { WaitNeeded::Fatal(e) }
}

// =============================================================================
// try_serve_waiters
// =============================================================================

impl ContextManager {
    /// Try to serve queued waiters on one or all devices using only
    /// already-free pages (free pool + unreferenced committed pages).
    ///
    /// `device`: `Some(d)` serves only device `d` (after a device-specific
    /// free event); `None` serves all devices (after a global weight change).
    ///
    /// **Never evicts active contexts.** This prevents thrashing where
    /// waiters evict contexts that are about to complete their forward
    /// passes and free pages naturally. Eviction only happens in the
    /// direct `ReservePages` / `EnsureResident` message handlers.
    pub(crate) async fn try_serve_waiters(&mut self, device: Option<usize>) {
        let devs: Vec<usize> = match device {
            Some(d) => vec![d],
            None => (0..self.wait_queues.len()).collect(),
        };
        for dev in devs {
            loop {
                // Quick check: any free pages at all?
                if self.devices[dev].available_gpu_pages() == 0 {
                    break;
                }

                let waiter = match self.wait_queues[dev].pop() {
                    Some(w) => w,
                    None => break,
                };
                match waiter {
                    PageWaiter::Allocate { context_id, device, num_pages, requester, enqueued_at, response, .. } => {
                        // Direct allocation: free pool + LRU eviction of
                        // unreferenced committed pages. No context eviction.
                        match self.devices[dev].allocate_working(num_pages) {
                            Ok(new_pages) => {
                                if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, context_id)) {
                                    ctx.working_pages.extend(&new_pages);
                                    ctx.device = Some(dev as DeviceId);
                                }
                                if let Some(pid) = requester {
                                    self.arbiter.add_working(pid, dev, num_pages);
                                }
                                let _ = response.send(Ok(()));
                                continue;
                            }
                            Err(_) => {
                                let floor = self.requester_floor(requester, dev, num_pages);
                                self.wait_queues[dev].push(PageWaiter::Allocate {
                                    context_id, device, num_pages, requester,
                                    priority_floor: floor, enqueued_at, response,
                                });
                                break;
                            }
                        }
                    }
                    PageWaiter::Restore { context_id, device, requester, enqueued_at, response, .. } => {
                        match self.ensure_resident(context_id).await {
                            Ok(replay_chunks) => {
                                // Fast path: atomically resolve pages + pin InFlight
                                let pages = if replay_chunks.is_none() {
                                    let pages = self.get_physical_page_ids(context_id).unwrap_or_default();
                                    if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, context_id)) {
                                        ctx.state = ContextState::InFlight;
                                    }
                                    pages
                                } else {
                                    HashMap::new()
                                };
                                let _ = response.send(Ok(ResidentResult { replay_chunks, pages }));
                                continue;
                            }
                            Err(WaitNeeded::NeedPages) => {
                                let floor = self.requester_floor(requester, dev, 1);
                                self.wait_queues[dev].push(PageWaiter::Restore {
                                    context_id, device, requester,
                                    priority_floor: floor, enqueued_at, response,
                                });
                                break;
                            }
                            Err(WaitNeeded::Fatal(e)) => {
                                let _ = response.send(Err(e));
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
}
