//! Message Queue Service
//!
//! Unified pub/sub and push/pull messaging over a single actor.
//! Public functions send a message to the actor and (optionally) await replies.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

use crate::service::{Service, ServiceHandler};

type ListenerId = usize;

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the message-queue actor.
pub fn spawn() {
    SERVICE.spawn(|| MessageQueue::default()).expect("messaging already spawned");
}

// ---- Public API (message wrappers) ------------------------------------------

/// Broadcasts a message to all subscribers of a topic.
pub fn publish(topic: String, message: String) -> anyhow::Result<()> {
    SERVICE.send(Message::Publish { topic, message })
}

/// Subscribes to a topic. Returns a listener ID via the oneshot.
pub async fn subscribe(
    topic: String,
    sender: mpsc::Sender<String>,
) -> anyhow::Result<ListenerId> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Subscribe { topic, sender, sub_id: tx })?;
    Ok(rx.await?)
}

/// Unsubscribes from a topic using a previously returned listener ID.
pub fn unsubscribe(topic: String, sub_id: ListenerId) -> anyhow::Result<()> {
    SERVICE.send(Message::Unsubscribe { topic, sub_id })
}

/// Pushes a string message onto a topic queue.
pub fn push(topic: String, message: String) -> anyhow::Result<()> {
    SERVICE.send(Message::Push { topic, message })
}

/// Pulls the next string message from a topic queue.
pub async fn pull(topic: String) -> anyhow::Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Pull { topic, response: tx })?;
    Ok(rx.await?)
}

/// Pushes a binary blob onto a topic queue.
pub fn push_blob(topic: String, message: Bytes) -> anyhow::Result<()> {
    SERVICE.send(Message::PushBlob { topic, message })
}

/// Pulls the next binary blob from a topic queue.
pub async fn pull_blob(topic: String) -> anyhow::Result<Bytes> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::PullBlobMsg { topic, response: tx })?;
    Ok(rx.await?)
}

// ---- Messages ---------------------------------------------------------------

#[derive(Debug)]
enum Message {
    // -- PubSub --
    Publish { topic: String, message: String },
    Subscribe {
        topic: String,
        sender: mpsc::Sender<String>,
        sub_id: oneshot::Sender<ListenerId>,
    },
    Unsubscribe { topic: String, sub_id: ListenerId },

    // -- PushPull (string) --
    Push { topic: String, message: String },
    Pull { topic: String, response: oneshot::Sender<String> },

    // -- PushPull (blob) --
    PushBlob { topic: String, message: Bytes },
    PullBlobMsg { topic: String, response: oneshot::Sender<Bytes> },
}

// ---- State ------------------------------------------------------------------

/// Unified message-queue actor: pub/sub fanout + point-to-point push/pull.
struct MessageQueue {
    // PubSub
    pubsub_tx: UnboundedSender<(String, String)>,
    _pubsub_handle: tokio::task::JoinHandle<()>,
    subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    next_sub_id: ListenerId,

    // PushPull (string)
    string_tx: UnboundedSender<(String, String)>,
    _string_handle: tokio::task::JoinHandle<()>,
    string_queues: Arc<DashMap<String, StringQueue>>,

    // PushPull (blob)
    blob_tx: UnboundedSender<(String, Bytes)>,
    _blob_handle: tokio::task::JoinHandle<()>,
    blob_queues: Arc<DashMap<String, BlobQueue>>,
}

enum StringQueue {
    Messages(VecDeque<String>),
    PendingPulls(VecDeque<oneshot::Sender<String>>),
}

enum BlobQueue {
    Messages(VecDeque<Bytes>),
    PendingPulls(VecDeque<oneshot::Sender<Bytes>>),
}

impl Default for MessageQueue {
    fn default() -> Self {
        // PubSub event loop
        let (pubsub_tx, pubsub_rx) = mpsc::unbounded_channel();
        let subscribers_by_topic = Arc::new(DashMap::new());
        let _pubsub_handle =
            tokio::spawn(Self::pubsub_loop(pubsub_rx, Arc::clone(&subscribers_by_topic)));

        // PushPull string event loop
        let (string_tx, string_rx) = mpsc::unbounded_channel();
        let string_queues = Arc::new(DashMap::new());
        let _string_handle =
            tokio::spawn(Self::string_push_loop(string_rx, Arc::clone(&string_queues)));

        // PushPull blob event loop
        let (blob_tx, blob_rx) = mpsc::unbounded_channel();
        let blob_queues = Arc::new(DashMap::new());
        let _blob_handle =
            tokio::spawn(Self::blob_push_loop(blob_rx, Arc::clone(&blob_queues)));

        MessageQueue {
            pubsub_tx,
            _pubsub_handle,
            subscribers_by_topic,
            next_sub_id: 0,
            string_tx,
            _string_handle,
            string_queues,
            blob_tx,
            _blob_handle,
            blob_queues,
        }
    }
}

// ---- ServiceHandler ---------------------------------------------------------

impl ServiceHandler for MessageQueue {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            // -- PubSub -------------------------------------------------------
            Message::Publish { topic, message } => {
                self.pubsub_tx.send((topic, message)).unwrap();
            }
            Message::Subscribe { topic, sender, sub_id } => {
                let id = self.next_sub_id;
                self.next_sub_id += 1;
                self.subscribers_by_topic
                    .entry(topic)
                    .or_insert_with(Vec::new)
                    .push((id, sender));
                let _ = sub_id.send(id).ok();
            }
            Message::Unsubscribe { topic, sub_id } => {
                if let Some(mut subscribers) = self.subscribers_by_topic.get_mut(&topic) {
                    subscribers.retain(|(s, _)| *s != sub_id);
                    if subscribers.is_empty() {
                        drop(subscribers);
                        self.subscribers_by_topic.remove(&topic);
                    }
                }

            }

            // -- PushPull (string) --------------------------------------------
            Message::Push { topic, message } => {
                self.string_tx.send((topic, message)).unwrap();
            }
            Message::Pull { topic, response } => {
                let mut queue = self
                    .string_queues
                    .entry(topic.clone())
                    .or_insert(StringQueue::PendingPulls(VecDeque::new()));

                let remove = match queue.value_mut() {
                    StringQueue::Messages(q) => {
                        if let Some(msg) = q.pop_front() {
                            let _ = response.send(msg);
                        }
                        q.is_empty()
                    }
                    StringQueue::PendingPulls(q) => {
                        q.push_back(response);
                        false
                    }
                };

                drop(queue);
                if remove {
                    self.string_queues.remove(&topic);
                }
            }

            // -- PushPull (blob) ----------------------------------------------
            Message::PushBlob { topic, message } => {
                self.blob_tx.send((topic, message)).unwrap();
            }
            Message::PullBlobMsg { topic, response } => {
                let mut queue = self
                    .blob_queues
                    .entry(topic.clone())
                    .or_insert(BlobQueue::PendingPulls(VecDeque::new()));

                let remove = match queue.value_mut() {
                    BlobQueue::Messages(q) => {
                        if let Some(msg) = q.pop_front() {
                            let _ = response.send(msg);
                        }
                        q.is_empty()
                    }
                    BlobQueue::PendingPulls(q) => {
                        q.push_back(response);
                        false
                    }
                };

                drop(queue);
                if remove {
                    self.blob_queues.remove(&topic);
                }
            }
        }
    }
}

// ---- Event Loops ------------------------------------------------------------

impl MessageQueue {
    /// Fanout loop: delivers published messages to all subscribers of a topic.
    async fn pubsub_loop(
        mut rx: UnboundedReceiver<(String, String)>,
        subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let remove = if let Some(mut subs) = subscribers_by_topic.get_mut(&topic) {
                subs.retain(|(_, sender)| match sender.try_send(message.clone()) {
                    Ok(_) => true,
                    Err(mpsc::error::TrySendError::Full(_)) => true,
                    Err(mpsc::error::TrySendError::Closed(_)) => false,
                });
                subs.is_empty()
            } else {
                false
            };

            if remove {
                subscribers_by_topic.remove(&topic);
            }
        }
    }

    /// Push loop for string messages: delivers to a waiting pull or enqueues.
    async fn string_push_loop(
        mut rx: UnboundedReceiver<(String, String)>,
        queues: Arc<DashMap<String, StringQueue>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let mut queue = queues
                .entry(topic.clone())
                .or_insert(StringQueue::Messages(VecDeque::new()));

            let remove = match queue.value_mut() {
                StringQueue::Messages(q) => {
                    q.push_back(message);
                    false
                }
                StringQueue::PendingPulls(q) => {
                    if let Some(waiting) = q.pop_front() {
                        let _ = waiting.send(message);
                    }
                    q.is_empty()
                }
            };

            drop(queue);
            if remove {
                queues.remove(&topic);
            }
        }
    }

    /// Push loop for blob messages: delivers to a waiting pull or enqueues.
    async fn blob_push_loop(
        mut rx: UnboundedReceiver<(String, Bytes)>,
        queues: Arc<DashMap<String, BlobQueue>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            let mut queue = queues
                .entry(topic.clone())
                .or_insert(BlobQueue::Messages(VecDeque::new()));

            let remove = match queue.value_mut() {
                BlobQueue::Messages(q) => {
                    q.push_back(message);
                    false
                }
                BlobQueue::PendingPulls(q) => {
                    if let Some(waiting) = q.pop_front() {
                        let _ = waiting.send(message);
                    }
                    q.is_empty()
                }
            };

            drop(queue);
            if remove {
                queues.remove(&topic);
            }
        }
    }
}
