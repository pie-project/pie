use std::collections::{HashMap, VecDeque};
use std::sync::LazyLock;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::service::{Service, ServiceHandler};

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

pub fn spawn() {
    if !SERVICE.is_spawned() {
        SERVICE
            .spawn(InboxRegistry::default)
            .expect("process inbox already spawned");
    }
}

pub fn send(process_id: String, message: String) -> Result<()> {
    SERVICE.send(Message::Deliver {
        process_id,
        message,
    })
}

pub async fn receive(process_id: String) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Receive {
        process_id,
        response: tx,
    })?;
    Ok(rx.await?)
}

pub fn clear(process_id: String) -> Result<()> {
    SERVICE.send(Message::Clear { process_id })
}

#[derive(Debug)]
enum Message {
    Deliver {
        process_id: String,
        message: String,
    },
    Receive {
        process_id: String,
        response: oneshot::Sender<String>,
    },
    Clear {
        process_id: String,
    },
}

enum Queue {
    Messages(VecDeque<String>),
    Waiters(VecDeque<oneshot::Sender<String>>),
}

#[derive(Default)]
struct InboxRegistry {
    queues: HashMap<String, Queue>,
}

impl InboxRegistry {
    fn deliver(&mut self, process_id: String, mut message: String) {
        match self.queues.get_mut(&process_id) {
            Some(Queue::Messages(queue)) => {
                queue.push_back(message);
                return;
            }
            Some(Queue::Waiters(waiters)) => {
                while let Some(waiter) = waiters.pop_front() {
                    match waiter.send(message) {
                        Ok(()) => {
                            if waiters.is_empty() {
                                self.queues.remove(&process_id);
                            }
                            return;
                        }
                        Err(returned) => {
                            tracing::debug!(
                                process_id,
                                "dropping canceled session.receive waiter while delivering signal"
                            );
                            message = returned;
                        }
                    }
                }
                self.queues.remove(&process_id);
            }
            None => {}
        }

        self.queues
            .insert(process_id, Queue::Messages(VecDeque::from([message])));
    }

    fn register_waiter(&mut self, process_id: String, response: oneshot::Sender<String>) {
        match self.queues.get_mut(&process_id) {
            Some(Queue::Messages(queue)) => {
                let Some(message) = queue.pop_front() else {
                    self.queues.remove(&process_id);
                    self.queues
                        .insert(process_id, Queue::Waiters(VecDeque::from([response])));
                    return;
                };
                match response.send(message) {
                    Ok(()) => {
                        if queue.is_empty() {
                            self.queues.remove(&process_id);
                        }
                    }
                    Err(message) => {
                        tracing::debug!(
                            process_id,
                            "session.receive waiter canceled before buffered signal delivery"
                        );
                        queue.push_front(message);
                    }
                }
            }
            Some(Queue::Waiters(waiters)) => {
                waiters.push_back(response);
            }
            None => {
                self.queues
                    .insert(process_id, Queue::Waiters(VecDeque::from([response])));
            }
        }
    }

    fn clear(&mut self, process_id: &str) {
        self.queues.remove(process_id);
    }
}

impl ServiceHandler for InboxRegistry {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Deliver {
                process_id,
                message,
            } => self.deliver(process_id, message),
            Message::Receive {
                process_id,
                response,
            } => self.register_waiter(process_id, response),
            Message::Clear { process_id } => self.clear(&process_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::InboxRegistry;

    #[tokio::test]
    async fn buffers_messages_fifo_before_receive() {
        let mut inbox = InboxRegistry::default();
        inbox.deliver("proc".into(), "first".into());
        inbox.deliver("proc".into(), "second".into());

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        inbox.register_waiter("proc".into(), tx1);
        assert_eq!(rx1.await.unwrap(), "first");

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        inbox.register_waiter("proc".into(), tx2);
        assert_eq!(rx2.await.unwrap(), "second");
    }

    #[tokio::test]
    async fn wakes_waiters_in_registration_order() {
        let mut inbox = InboxRegistry::default();
        let (tx1, rx1) = tokio::sync::oneshot::channel();
        let (tx2, rx2) = tokio::sync::oneshot::channel();

        inbox.register_waiter("proc".into(), tx1);
        inbox.register_waiter("proc".into(), tx2);
        inbox.deliver("proc".into(), "first".into());
        inbox.deliver("proc".into(), "second".into());

        assert_eq!(rx1.await.unwrap(), "first");
        assert_eq!(rx2.await.unwrap(), "second");
    }
}
