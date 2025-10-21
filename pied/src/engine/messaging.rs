use super::service;
use super::service::Service;
use super::utils::IdPool;
use bytes::Bytes;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

static SERVICE_ID_MESSAGING_INST2INST: OnceLock<usize> = OnceLock::new();
static SERVICE_ID_MESSAGING_USER2INST: OnceLock<usize> = OnceLock::new();

pub fn dispatch_i2i(command: PubSubCommand) {
    let service_id = *SERVICE_ID_MESSAGING_INST2INST
        .get_or_init(|| service::get_service_id("messaging-inst2inst").unwrap());

    service::dispatch(service_id, command).unwrap();
}

pub fn dispatch_u2i(command: PushPullCommand) {
    let service_id = *SERVICE_ID_MESSAGING_USER2INST
        .get_or_init(|| service::get_service_id("messaging-user2inst").unwrap());

    service::dispatch(service_id, command).unwrap();
}

type ListenerId = usize;

#[derive(Debug)]
pub enum PubSubCommand {
    // Send {
    //     inst_id: InstanceId,
    //     message: String,
    // },
    //
    // Receive {
    //     inst_id: InstanceId,
    // },
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

#[derive(Debug)]
pub enum PushPullCommand {
    Push {
        topic: String,
        message: String,
    },

    Pull {
        topic: String,
        message: oneshot::Sender<String>,
    },

    PushBlob {
        topic: String,
        message: Bytes,
    },

    PullBlob {
        topic: String,
        message: oneshot::Sender<Bytes>,
    },
}

// impl PubSubCommand {
//     pub fn dispatch(self) -> Result<(), ServiceError> {
//         let service_id = *SERVICE_ID_MESSAGING
//             .get_or_init(move || service::get_service_id("messaging").unwrap());
//
//         service::dispatch(service_id, self)
//     }
// }

#[derive(Debug)]
pub struct PubSub {
    tx: UnboundedSender<(String, String)>,
    event_loop_handle: tokio::task::JoinHandle<()>,
    subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    sub_id_pool: IdPool<ListenerId>,
}

impl PubSub {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let subscribers_by_topic = Arc::new(DashMap::new());
        let event_loop_handle =
            tokio::spawn(Self::event_loop(rx, Arc::clone(&subscribers_by_topic)));

        PubSub {
            tx,
            event_loop_handle,
            subscribers_by_topic,
            sub_id_pool: IdPool::new(ListenerId::MAX),
        }
    }

    /// The event loop that listens for broadcast messages and dispatches them to subscribers.
    async fn event_loop(
        mut rx: UnboundedReceiver<(String, String)>,
        subscribers_by_topic: Arc<DashMap<String, Vec<(ListenerId, mpsc::Sender<String>)>>>,
    ) {
        while let Some((topic, message)) = rx.recv().await {
            //println!("subscriptions: {:?}", subscriptions.len());

            let remove_topic = if let Some(mut subscribers) = subscribers_by_topic.get_mut(&topic) {
                //println!("Received message: {:?}, {:?}", topic, message);

                // Retain only the subscribers that can receive the message.
                subscribers.retain(|(_, sender)| {
                    match sender.try_send(message.clone()) {
                        Ok(_) => true,
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            // The subscriber's channel is full; keep the subscription.
                            true
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            // The subscriber's channel is closed; remove the subscription.
                            false
                        }
                    }
                });

                // Remove the topic if no subscribers remain.
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
//#[async_trait]
impl Service for PubSub {
    type Command = PubSubCommand;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            // Command::Send { inst_id, message } => {}
            // Command::Receive { inst_id } => {}
            PubSubCommand::Publish { topic, message } => {
                // Broadcast the message.
                self.tx.send((topic, message)).unwrap();
            }
            PubSubCommand::Subscribe {
                topic,
                sender,
                sub_id,
            } => {
                // Acquire a new subscription id.
                let id = self.sub_id_pool.acquire().unwrap();

                // Insert the new subscriber into the map.
                self.subscribers_by_topic
                    .entry(topic)
                    .or_insert_with(Vec::new)
                    .push((id, sender));

                // Send back the subscription id.
                let _ = sub_id.send(id).ok();
            }
            PubSubCommand::Unsubscribe { topic, sub_id } => {
                if let Some(mut subscribers) = self.subscribers_by_topic.get_mut(&topic) {
                    // Remove the subscriber with the matching id.
                    subscribers.retain(|(s, _)| *s != sub_id);

                    // Remove the topic entirely if there are no more subscribers.
                    if subscribers.is_empty() {
                        self.subscribers_by_topic.remove(&topic);
                    }
                }
                // Release the subscription id back to the pool.
                self.sub_id_pool.release(sub_id).unwrap();
            }
        }
    }
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

pub struct PushPull {
    // Fields for String-based messages
    tx_string: UnboundedSender<(String, String)>,
    _event_loop_handle_string: tokio::task::JoinHandle<()>,
    string_queue_by_topic: Arc<DashMap<String, PushPullStringQueue>>,

    // Fields for Blob-based messages (Vec<u8>)
    tx_blob: UnboundedSender<(String, Bytes)>,
    _event_loop_handle_blob: tokio::task::JoinHandle<()>,
    blob_queue_by_topic: Arc<DashMap<String, PushPullBlobQueue>>,
}

impl PushPull {
    pub fn new() -> Self {
        // --- Setup for String messages ---
        let (tx_string, rx_string) = mpsc::unbounded_channel();
        let string_queue_by_topic = Arc::new(DashMap::new());
        let _event_loop_handle_string = tokio::spawn(Self::event_loop_string(
            rx_string,
            Arc::clone(&string_queue_by_topic),
        ));

        // --- Setup for Blob messages ---
        let (tx_blob, rx_blob) = mpsc::unbounded_channel();
        let blob_queue_by_topic = Arc::new(DashMap::new());
        let _event_loop_handle_blob = tokio::spawn(Self::event_loop_blob(
            rx_blob,
            Arc::clone(&blob_queue_by_topic),
        ));

        PushPull {
            tx_string,
            _event_loop_handle_string,
            string_queue_by_topic,
            tx_blob,
            _event_loop_handle_blob,
            blob_queue_by_topic,
        }
    }

    /// The event loop that listens for pushed string messages and matches them with pulls.
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

            // Drop the lock on the queue entry before potentially removing the topic.
            drop(queue);

            if remove_queue {
                queue_by_topic.remove(&topic);
            }
        }
    }

    /// The event loop that listens for pushed blob messages and matches them with pulls.
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

            // Drop the lock on the queue entry before potentially removing the topic.
            drop(queue);

            if remove_queue {
                queue_by_topic.remove(&topic);
            }
        }
    }
}

impl Service for PushPull {
    type Command = PushPullCommand;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            PushPullCommand::Push { topic, message } => {
                self.tx_string.send((topic, message)).unwrap();
            }
            PushPullCommand::Pull { topic, message } => {
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

                // To avoid DashMap deadlock.
                drop(queue);

                if remove_queue {
                    self.string_queue_by_topic.remove(&topic);
                }
            }
            PushPullCommand::PushBlob { topic, message } => {
                self.tx_blob.send((topic, message)).unwrap();
            }
            PushPullCommand::PullBlob { topic, message } => {
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

                // To avoid DashMap deadlock.
                drop(queue);

                if remove_queue {
                    self.blob_queue_by_topic.remove(&topic);
                }
            }
        }
    }
}
