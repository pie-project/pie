//! Message Queue Service
//!
//! Unified pub/sub and push/pull messaging over a single actor.
//! All state is owned by the actor — no background tasks, no shared maps.

use std::collections::{HashMap, VecDeque};
use std::sync::LazyLock;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::service::{Service, ServiceHandler};

type ListenerId = usize;

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the message-queue actor.
pub fn spawn() {
    SERVICE.spawn(|| MessageQueue::default()).expect("messaging already spawned");
}

// ---- Public API -------------------------------------------------------------

/// Pushes a message onto a topic queue (point-to-point).
pub fn push(topic: String, message: String) -> anyhow::Result<()> {
    SERVICE.send(Message::Push { topic, message })
}

/// Pulls the next message from a topic queue (blocks until available).
pub async fn pull(topic: String) -> anyhow::Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Pull { topic, response: tx })?;
    Ok(rx.await?)
}

/// Broadcasts a message to all subscribers of a topic (fan-out).
pub fn publish(topic: String, message: String) -> anyhow::Result<()> {
    SERVICE.send(Message::Publish { topic, message })
}

/// Subscribes to a topic. Returns a listener ID.
pub async fn subscribe(
    topic: String,
    sender: mpsc::Sender<String>,
) -> anyhow::Result<ListenerId> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Subscribe { topic, sender, response: tx })?;
    Ok(rx.await?)
}

/// Unsubscribes from a topic using a previously returned listener ID.
pub fn unsubscribe(topic: String, sub_id: ListenerId) -> anyhow::Result<()> {
    SERVICE.send(Message::Unsubscribe { topic, sub_id })
}

// ---- Messages ---------------------------------------------------------------

#[derive(Debug)]
enum Message {
    Push { topic: String, message: String },
    Pull { topic: String, response: oneshot::Sender<String> },
    Publish { topic: String, message: String },
    Subscribe {
        topic: String,
        sender: mpsc::Sender<String>,
        response: oneshot::Sender<ListenerId>,
    },
    Unsubscribe { topic: String, sub_id: ListenerId },
}

// ---- State ------------------------------------------------------------------

/// Point-to-point queue: either buffered messages or waiting consumers.
enum Queue {
    Messages(VecDeque<String>),
    Waiters(VecDeque<oneshot::Sender<String>>),
}

struct MessageQueue {
    queues: HashMap<String, Queue>,
    subscribers: HashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>,
    next_sub_id: ListenerId,
}

impl Default for MessageQueue {
    fn default() -> Self {
        MessageQueue {
            queues: HashMap::new(),
            subscribers: HashMap::new(),
            next_sub_id: 0,
        }
    }
}

// ---- Handler ----------------------------------------------------------------

impl ServiceHandler for MessageQueue {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Push { topic, message } => {
                match self.queues.get_mut(&topic) {
                    Some(Queue::Waiters(w)) => {
                        if let Some(tx) = w.pop_front() {
                            let _ = tx.send(message);
                        }
                        if w.is_empty() {
                            self.queues.remove(&topic);
                        }
                    }
                    Some(Queue::Messages(q)) => {
                        q.push_back(message);
                    }
                    None => {
                        self.queues.insert(topic, Queue::Messages(VecDeque::from([message])));
                    }
                }
            }

            Message::Pull { topic, response } => {
                match self.queues.get_mut(&topic) {
                    Some(Queue::Messages(q)) => {
                        if let Some(msg) = q.pop_front() {
                            let _ = response.send(msg);
                        }
                        if q.is_empty() {
                            self.queues.remove(&topic);
                        }
                    }
                    Some(Queue::Waiters(w)) => {
                        w.push_back(response);
                    }
                    None => {
                        self.queues.insert(topic, Queue::Waiters(VecDeque::from([response])));
                    }
                }
            }

            Message::Publish { topic, message } => {
                if let Some(subs) = self.subscribers.get_mut(&topic) {
                    subs.retain(|(_, tx)| match tx.try_send(message.clone()) {
                        Ok(_) => true,
                        Err(mpsc::error::TrySendError::Full(_)) => true,
                        Err(mpsc::error::TrySendError::Closed(_)) => false,
                    });
                    if subs.is_empty() {
                        self.subscribers.remove(&topic);
                    }
                }
            }

            Message::Subscribe { topic, sender, response } => {
                let id = self.next_sub_id;
                self.next_sub_id += 1;
                self.subscribers.entry(topic).or_default().push((id, sender));
                let _ = response.send(id);
            }

            Message::Unsubscribe { topic, sub_id } => {
                if let Some(subs) = self.subscribers.get_mut(&topic) {
                    subs.retain(|(id, _)| *id != sub_id);
                    if subs.is_empty() {
                        self.subscribers.remove(&topic);
                    }
                }
            }
        }
    }
}
