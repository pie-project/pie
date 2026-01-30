//! Messaging Service - PubSub and PushPull messaging patterns
//!
//! This module provides actors for inter-process messaging using the
//! modern actor model (Handle trait).

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

use crate::actor::{Actor, Actors, Handle, SendError};
use crate::utils::IdPool;

type ListenerId = usize;

// =============================================================================
// PubSub Actor
// =============================================================================

/// Global singleton PubSub actor.
static PUBSUB_ACTOR: LazyLock<Actor<PubSubMessage>> = LazyLock::new(Actor::new);

/// Spawns the PubSub actor.
pub fn spawn_pubsub() {
    PUBSUB_ACTOR.spawn::<PubSubActor>();
}

/// Sends a message to the PubSub actor.
pub fn pubsub_send(msg: PubSubMessage) -> Result<(), SendError> {
    PUBSUB_ACTOR.send(msg)
}

/// Messages for the PubSub actor.
#[derive(Debug)]
pub enum PubSubMessage {
    /// Broadcast a message to all subscribers of a topic.
    Publish { topic: String, message: String },
    /// Subscribe to a topic using a sender; returns a subscription id via the oneshot.
    Subscribe {
        topic: String,
        sender: mpsc::Sender<String>,
        sub_id: oneshot::Sender<ListenerId>,
    },
    /// Unsubscribe from a topic using the subscription id.
    Unsubscribe { topic: String, sub_id: ListenerId },
}

/// PubSub actor implementation.
struct PubSubActor {
    tx: UnboundedSender<(String, String)>,
    _event_loop_handle: tokio::task::JoinHandle<()>,
    subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    sub_id_pool: IdPool<ListenerId>,
}

impl Handle for PubSubActor {
    type Message = PubSubMessage;

    fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let subscribers_by_topic = Arc::new(DashMap::new());
        let _event_loop_handle =
            tokio::spawn(Self::event_loop(rx, Arc::clone(&subscribers_by_topic)));

        PubSubActor {
            tx,
            _event_loop_handle,
            subscribers_by_topic,
            sub_id_pool: IdPool::new(ListenerId::MAX),
        }
    }

    async fn handle(&mut self, msg: PubSubMessage) {
        match msg {
            PubSubMessage::Publish { topic, message } => {
                self.tx.send((topic, message)).unwrap();
            }
            PubSubMessage::Subscribe { topic, sender, sub_id } => {
                let id = self.sub_id_pool.acquire().unwrap();
                self.subscribers_by_topic
                    .entry(topic)
                    .or_insert_with(Vec::new)
                    .push((id, sender));
                let _ = sub_id.send(id).ok();
            }
            PubSubMessage::Unsubscribe { topic, sub_id } => {
                if let Some(mut subscribers) = self.subscribers_by_topic.get_mut(&topic) {
                    subscribers.retain(|(s, _)| *s != sub_id);
                    if subscribers.is_empty() {
                        drop(subscribers);
                        self.subscribers_by_topic.remove(&topic);
                    }
                }
                self.sub_id_pool.release(sub_id).unwrap();
            }
        }
    }
}

impl PubSubActor {
    async fn event_loop(
        mut rx: UnboundedReceiver<(String, String)>,
        subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let remove_topic = if let Some(mut subscribers) = subscribers_by_topic.get_mut(&topic) {
                subscribers.retain(|(_, sender)| {
                    match sender.try_send(message.clone()) {
                        Ok(_) => true,
                        Err(mpsc::error::TrySendError::Full(_)) => true,
                        Err(mpsc::error::TrySendError::Closed(_)) => false,
                    }
                });
                subscribers.is_empty()
            } else {
                false
            };

            if remove_topic {
                subscribers_by_topic.remove(&topic);
            }
        }
    }
}

// =============================================================================
// PushPull Actor
// =============================================================================

/// Global singleton PushPull actor.
static PUSHPULL_ACTOR: LazyLock<Actor<PushPullMessage>> = LazyLock::new(Actor::new);

/// Spawns the PushPull actor.
pub fn spawn_pushpull() {
    PUSHPULL_ACTOR.spawn::<PushPullActor>();
}

/// Sends a message to the PushPull actor.
pub fn pushpull_send(msg: PushPullMessage) -> Result<(), SendError> {
    PUSHPULL_ACTOR.send(msg)
}

/// Messages for the PushPull actor.
#[derive(Debug)]
pub enum PushPullMessage {
    Push { topic: String, message: String },
    Pull { topic: String, message: oneshot::Sender<String> },
    PushBlob { topic: String, message: Bytes },
    PullBlob { topic: String, message: oneshot::Sender<Bytes> },
}

/// A queue for a given topic, holding either waiting messages or pending pull requests.
enum PushPullStringQueue {
    Messages(VecDeque<String>),
    PendingPulls(VecDeque<oneshot::Sender<String>>),
}

/// A queue for a given topic, holding either waiting blobs or pending pull requests for blobs.
enum PushPullBlobQueue {
    Messages(VecDeque<Bytes>),
    PendingPulls(VecDeque<oneshot::Sender<Bytes>>),
}

/// PushPull actor implementation.
struct PushPullActor {
    tx_string: UnboundedSender<(String, String)>,
    _event_loop_handle_string: tokio::task::JoinHandle<()>,
    string_queue_by_topic: Arc<DashMap<String, PushPullStringQueue>>,

    tx_blob: UnboundedSender<(String, Bytes)>,
    _event_loop_handle_blob: tokio::task::JoinHandle<()>,
    blob_queue_by_topic: Arc<DashMap<String, PushPullBlobQueue>>,
}

impl Handle for PushPullActor {
    type Message = PushPullMessage;

    fn new() -> Self {
        let (tx_string, rx_string) = mpsc::unbounded_channel();
        let string_queue_by_topic = Arc::new(DashMap::new());
        let _event_loop_handle_string = tokio::spawn(Self::event_loop_string(
            rx_string,
            Arc::clone(&string_queue_by_topic),
        ));

        let (tx_blob, rx_blob) = mpsc::unbounded_channel();
        let blob_queue_by_topic = Arc::new(DashMap::new());
        let _event_loop_handle_blob = tokio::spawn(Self::event_loop_blob(
            rx_blob,
            Arc::clone(&blob_queue_by_topic),
        ));

        PushPullActor {
            tx_string,
            _event_loop_handle_string,
            string_queue_by_topic,
            tx_blob,
            _event_loop_handle_blob,
            blob_queue_by_topic,
        }
    }

    async fn handle(&mut self, msg: PushPullMessage) {
        match msg {
            PushPullMessage::Push { topic, message } => {
                self.tx_string.send((topic, message)).unwrap();
            }
            PushPullMessage::Pull { topic, message } => {
                let mut queue = self
                    .string_queue_by_topic
                    .entry(topic.clone())
                    .or_insert(PushPullStringQueue::PendingPulls(VecDeque::new()));

                let remove_queue = match queue.value_mut() {
                    PushPullStringQueue::Messages(q) => {
                        if let Some(sent_msg) = q.pop_front() {
                            let _ = message.send(sent_msg);
                        }
                        q.is_empty()
                    }
                    PushPullStringQueue::PendingPulls(q) => {
                        q.push_back(message);
                        false
                    }
                };

                drop(queue);

                if remove_queue {
                    self.string_queue_by_topic.remove(&topic);
                }
            }
            PushPullMessage::PushBlob { topic, message } => {
                self.tx_blob.send((topic, message)).unwrap();
            }
            PushPullMessage::PullBlob { topic, message } => {
                let mut queue = self
                    .blob_queue_by_topic
                    .entry(topic.clone())
                    .or_insert(PushPullBlobQueue::PendingPulls(VecDeque::new()));

                let remove_queue = match queue.value_mut() {
                    PushPullBlobQueue::Messages(q) => {
                        if let Some(sent_msg) = q.pop_front() {
                            let _ = message.send(sent_msg);
                        }
                        q.is_empty()
                    }
                    PushPullBlobQueue::PendingPulls(q) => {
                        q.push_back(message);
                        false
                    }
                };

                drop(queue);

                if remove_queue {
                    self.blob_queue_by_topic.remove(&topic);
                }
            }
        }
    }
}

impl PushPullActor {
    async fn event_loop_string(
        mut rx: UnboundedReceiver<(String, String)>,
        queue_by_topic: Arc<DashMap<String, PushPullStringQueue>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let mut queue = queue_by_topic
                .entry(topic.clone())
                .or_insert(PushPullStringQueue::Messages(VecDeque::new()));

            let remove_queue = match queue.value_mut() {
                PushPullStringQueue::Messages(q) => {
                    q.push_back(message);
                    false
                }
                PushPullStringQueue::PendingPulls(q) => {
                    if let Some(waiting_pull) = q.pop_front() {
                        let _ = waiting_pull.send(message);
                    }
                    q.is_empty()
                }
            };

            drop(queue);

            if remove_queue {
                queue_by_topic.remove(&topic);
            }
        }
    }

    async fn event_loop_blob(
        mut rx: UnboundedReceiver<(String, Bytes)>,
        queue_by_topic: Arc<DashMap<String, PushPullBlobQueue>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let mut queue = queue_by_topic
                .entry(topic.clone())
                .or_insert(PushPullBlobQueue::Messages(VecDeque::new()));

            let remove_queue = match queue.value_mut() {
                PushPullBlobQueue::Messages(q) => {
                    q.push_back(message);
                    false
                }
                PushPullBlobQueue::PendingPulls(q) => {
                    if let Some(waiting_pull) = q.pop_front() {
                        let _ = waiting_pull.send(message);
                    }
                    q.is_empty()
                }
            };

            drop(queue);

            if remove_queue {
                queue_by_topic.remove(&topic);
            }
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Spawns all messaging actors.
pub fn spawn() {
    spawn_pubsub();
    spawn_pushpull();
}
