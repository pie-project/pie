use crate::model::request::Request;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::mem::{Discriminant, discriminant};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

type StreamId = u32;

trait BatchPolicy {
    fn next_poll_in(&self, queue: &[QueuedRequest], now: Instant) -> Option<Duration>;

    fn try_form_batch(
        &mut self,
        queue: &mut Vec<QueuedRequest>,
        now: Instant,
    ) -> Option<Vec<QueuedRequest>>;
}

#[derive(Debug)]
struct QueuedRequest {
    req: Request,
    enqueued_at: Instant,
    priority: u32,
    stream: StreamId,
}

#[derive(Debug)]
pub struct BatchPolicySelector {
    forward_pass: ForwardPassPolicy,
    eager: ThresholdPolicy,
}

impl BatchPolicySelector {
    pub fn new(forward_pass: ForwardPassPolicy) -> Self {
        let eager = ThresholdPolicy::eager();
        Self {
            forward_pass,
            eager,
        }
    }
}

impl BatchPolicy for BatchPolicySelector {
    fn next_poll_in(&self, queue: &[QueuedRequest], now: Instant) -> Option<Duration> {
        match queue.iter().next()?.req {
            Request::ForwardPass(_, _) => self.forward_pass.next_poll_in(queue, now),
            _ => self.eager.next_poll_in(queue, now),
        }
    }

    fn try_form_batch(
        &mut self,
        queue: &mut Vec<QueuedRequest>,
        now: Instant,
    ) -> Option<Vec<QueuedRequest>> {
        match queue.iter().next()?.req {
            Request::ForwardPass(_, _) => self.forward_pass.try_form_batch(queue, now),
            _ => self.eager.try_form_batch(queue, now),
        }
    }
}

#[derive(Debug)]
pub struct ThresholdPolicy {
    max_wait_time: Duration,
    min_size: usize,
    max_size: usize,
}

impl ThresholdPolicy {
    pub fn new(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        Self {
            max_wait_time,
            min_size,
            max_size: max_size.unwrap_or(min_size),
        }
    }

    /// Creates a policy where batches are formed as soon as at least one item is present.
    pub fn eager() -> Self {
        Self::new(Duration::from_secs_f32(0.0), 1, Some(usize::MAX))
    }

    /// Creates a policy that only uses the item count threshold (`min_size`).
    /// If `max_size` is not provided, it defaults to the given `min_size`.
    pub fn k_only(min_size: usize, max_size: Option<usize>) -> Self {
        Self::new(Duration::MAX, min_size, max_size)
    }

    /// Creates a policy that only uses the time threshold (`max_wait_time`).
    pub fn t_only(max_wait_time: Duration) -> Self {
        Self::new(max_wait_time, usize::MAX, Some(usize::MAX))
    }

    /// Creates a policy that forms a batch when either the item count or time threshold is met.
    pub fn k_or_t(max_wait_time: Duration, min_size: usize, max_size: Option<usize>) -> Self {
        Self::new(max_wait_time, min_size, max_size)
    }
}

impl BatchPolicy for ThresholdPolicy {
    fn next_poll_in(&self, queue: &[QueuedRequest], now: Instant) -> Option<Duration> {
        // If the size condition is already met, the batch is ready.
        if queue.len() >= self.min_size {
            return Some(Duration::ZERO);
        }

        let first_item_time = queue.iter().next()?.enqueued_at;

        let elapsed = now.duration_since(first_item_time);
        if elapsed < self.max_wait_time {
            Some(self.max_wait_time - elapsed)
        } else {
            // Time has expired, the batch is ready.
            Some(Duration::ZERO)
        }
    }

    fn try_form_batch(
        &mut self,
        queue: &mut Vec<QueuedRequest>,
        now: Instant,
    ) -> Option<Vec<QueuedRequest>> {
        let first_item_time = queue.iter().next()?.enqueued_at;

        if queue.len() < self.min_size && now.duration_since(first_item_time) < self.max_wait_time {
            return None;
        }

        let count = queue.len().min(self.max_size);
        let batch = queue.drain(..count).collect();

        Some(batch)
    }
}

#[derive(Debug)]
pub struct ForwardPassPolicy {
    trigger: Arc<AtomicBool>,
    max_batch_tokens: usize,
    min_wait_time: Duration,
}

impl ForwardPassPolicy {
    pub fn new(trigger: Arc<AtomicBool>, max_batch_tokens: usize, min_wait_time: Duration) -> Self {
        Self {
            trigger,
            min_wait_time,
            max_batch_tokens,
        }
    }
}

impl BatchPolicy for ForwardPassPolicy {
    fn next_poll_in(&self, queue: &[QueuedRequest], now: Instant) -> Option<Duration> {
        let first_item_time = queue.iter().next()?.enqueued_at;
        let elapsed = now.duration_since(first_item_time);
        if now.duration_since(first_item_time) < self.min_wait_time {
            Some(self.min_wait_time - elapsed)
        } else {
            None
        }
    }

    fn try_form_batch(
        &mut self,
        queue: &mut Vec<QueuedRequest>,
        now: Instant,
    ) -> Option<Vec<QueuedRequest>> {
        let first_item_time = queue.iter().next()?.enqueued_at;
        let waited_long_enough = now.duration_since(first_item_time) >= self.min_wait_time;
        if !waited_long_enough {
            return None;
        }

        if self.trigger.load(Ordering::SeqCst) {
            if self.trigger.swap(false, Ordering::SeqCst) {
                let mut tokens_in_batch = 0;
                let mut num_requests_to_drain = queue.len(); // Default to draining the entire queue.

                for (i, request) in queue.iter().enumerate() {
                    if let Request::ForwardPass(req, _) = &request.req {
                        tokens_in_batch += req.input_tokens.len() + req.input_embed_ptrs.len();

                        if tokens_in_batch >= self.max_batch_tokens {
                            num_requests_to_drain = i + 1;
                            break;
                        }
                    }
                }
                //
                // println!("Tokens in batch: {}", tokens_in_batch);
                // println!("Num requests to drain: {}", num_requests_to_drain);

                return Some(queue.drain(..num_requests_to_drain).collect());
            }
        }

        None
    }
}

#[derive(Debug)]
pub struct BatchScheduler {
    stream_lock: HashMap<StreamId, Discriminant<Request>>,
    queued: HashMap<StreamId, VecDeque<QueuedRequest>>,
    pending: HashMap<Discriminant<Request>, Vec<QueuedRequest>>,
    policy_selector: BatchPolicySelector,
}

impl BatchScheduler {
    pub fn new(policy: BatchPolicySelector) -> Self {
        Self {
            stream_lock: HashMap::new(),
            //handler_to_streams: HashMap::new(),
            queued: HashMap::new(),
            pending: HashMap::new(),
            policy_selector: policy,
        }
    }

    pub fn push(&mut self, stream: StreamId, priority: u32, req: Request, now: Instant) {
        self.queued
            .entry(stream)
            .or_default()
            .push_back(QueuedRequest {
                req,
                enqueued_at: now,
                priority,
                stream,
            });
    }

    /// Processes queued items and forms batches from pending queues.
    ///
    /// This method operates in two phases:
    /// 1. **Promotion**: It moves items from `queued` to the appropriate `pending`
    ///    batch queue, respecting the strict ordering constraints.
    /// 2. **Batching**: It checks each `pending` queue to see if a batch is ready
    ///    according to its scheduling policy.
    pub fn schedule(&mut self, now: Instant) -> Vec<Vec<Request>> {
        self.promote_queued_items();

        let mut ready_batches: Vec<Vec<Request>> = Vec::new();

        for (_, mut queue) in self.pending.iter_mut() {
            if let Some(batch) = self.policy_selector.try_form_batch(&mut queue, now) {
                // A batch is ready. Release the handler lock for all streams
                // that contributed an item to this batch.
                let mut ready = Vec::with_capacity(batch.len());
                for r in batch {
                    self.stream_lock.remove(&r.stream);
                    ready.push(r.req);
                }

                ready_batches.push(ready);
            }
        }

        ready_batches
    }

    /// Hints at the optimal duration to wait before the next poll across all managed batchers.
    ///
    /// Returns the smallest wait time hint among all active batchers, or `None` if no
    /// batcher provides a time-based hint.
    pub fn next_poll_in(&self, now: Instant) -> Option<Duration> {
        self.pending
            .values()
            .filter_map(|batcher| self.policy_selector.next_poll_in(batcher, now))
            .min()
    }

    /// **Phase 1: Promote items from `queued` to `pending` queues.**
    ///
    /// This function iterates through each stream's queue, promoting consecutive items
    /// that share the same handler type. This process stops for a given stream if it
    /// encounters an item with a different handler type than the one currently "locked"
    /// for that stream in `active_handlers`.
    fn promote_queued_items(&mut self) {
        let streams = self.queued.keys().copied().collect::<Vec<_>>();

        for stream in streams {
            let queue = self.queued.get_mut(&stream).unwrap();

            // Get the handler type currently being processed for this stream, if any.
            let mut active_handler = self.stream_lock.get(&stream).copied();

            while let Some(r) = queue.front() {
                // If the next item's handler is different from the active one,
                // we must wait. This enforces the strict ordering.
                let handler = discriminant(&r.req);
                if let Some(active_handler) = active_handler {
                    if handler != active_handler {
                        break;
                    }
                } else {
                    // This is the first item being processed for this stream; it sets the active handler.
                    active_handler = Some(handler);
                    self.stream_lock.insert(stream, handler);
                }

                // The item is eligible for promotion. Move it from `queued` to `pending`.
                let r = queue.pop_front().unwrap();
                let needs_sort = r.priority > 0;
                self.pending.entry(handler).or_default().push(r);

                if needs_sort {
                    self.pending
                        .get_mut(&handler)
                        .unwrap()
                        .sort_by_key(|r| std::cmp::Reverse(r.priority));
                }
            }
        }

        // Clean up streams whose queues have been fully drained.
        self.queued.retain(|_stream, queue| !queue.is_empty());
    }
}
