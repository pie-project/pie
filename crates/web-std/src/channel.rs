use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

struct Inner<T> {
    queue: VecDeque<T>,
    senders: usize,
    receivers: usize,
    wakers: Vec<Waker>,
}

struct Shared<T>(Mutex<Inner<T>>);

impl<T> Shared<T> {
    fn wake_all(inner: &mut Inner<T>) -> Vec<Waker> {
        std::mem::take(&mut inner.wakers)
    }
}

pub struct Sender<T> {
    shared: Arc<Shared<T>>,
}

pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

impl<T> std::fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("sending on a disconnected channel")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("receiving on an empty and disconnected channel")
    }
}

impl std::error::Error for RecvError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    Empty,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvTimeoutError {
    Timeout,
    Disconnected,
}

pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared(Mutex::new(Inner {
        queue: VecDeque::new(),
        senders: 1,
        receivers: 1,
        wakers: Vec::new(),
    })));
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
    )
}

pub fn bounded<T>(_capacity: usize) -> (Sender<T>, Receiver<T>) {
    unbounded()
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let wakers = {
            let mut inner = self.shared.0.lock().unwrap();
            if inner.receivers == 0 {
                return Err(SendError(value));
            }
            inner.queue.push_back(value);
            Shared::wake_all(&mut inner)
        };
        for waker in wakers {
            waker.wake();
        }
        Ok(())
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.shared.0.lock().unwrap().senders += 1;
        Sender {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let wakers = {
            let mut inner = self.shared.0.lock().unwrap();
            inner.senders -= 1;
            if inner.senders == 0 {
                Shared::wake_all(&mut inner)
            } else {
                Vec::new()
            }
        };
        for waker in wakers {
            waker.wake();
        }
    }
}

impl<T> Receiver<T> {
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut inner = self.shared.0.lock().unwrap();
        match inner.queue.pop_front() {
            Some(value) => Ok(value),
            None if inner.senders == 0 => Err(TryRecvError::Disconnected),
            None => Err(TryRecvError::Empty),
        }
    }

    pub fn len(&self) -> usize {
        self.shared.0.lock().unwrap().queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn poll_ready(&self, cx: &mut Context<'_>) -> Poll<()> {
        let mut inner = self.shared.0.lock().unwrap();
        if !inner.queue.is_empty() || inner.senders == 0 {
            return Poll::Ready(());
        }
        let waker = cx.waker();
        if !inner.wakers.iter().any(|w| w.will_wake(waker)) {
            inner.wakers.push(waker.clone());
        }
        Poll::Pending
    }

    pub async fn recv_async(&self) -> Result<T, RecvError> {
        std::future::poll_fn(|cx| match self.try_recv() {
            Ok(value) => Poll::Ready(Ok(value)),
            Err(TryRecvError::Disconnected) => Poll::Ready(Err(RecvError)),
            Err(TryRecvError::Empty) => match self.poll_ready(cx) {
                Poll::Ready(()) => match self.try_recv() {
                    Ok(value) => Poll::Ready(Ok(value)),
                    Err(TryRecvError::Disconnected) => Poll::Ready(Err(RecvError)),
                    Err(TryRecvError::Empty) => Poll::Pending,
                },
                Poll::Pending => Poll::Pending,
            },
        })
        .await
    }

    #[cfg(target_arch = "wasm32")]
    pub fn recv(&self) -> Result<T, RecvError> {
        crate::thread::block_on_poll(|cx| match self.try_recv() {
            Ok(value) => Poll::Ready(Ok(value)),
            Err(TryRecvError::Disconnected) => Poll::Ready(Err(RecvError)),
            Err(TryRecvError::Empty) => match self.poll_ready(cx) {
                Poll::Ready(()) => match self.try_recv() {
                    Ok(value) => Poll::Ready(Ok(value)),
                    Err(TryRecvError::Disconnected) => Poll::Ready(Err(RecvError)),
                    Err(TryRecvError::Empty) => Poll::Pending,
                },
                Poll::Pending => Poll::Pending,
            },
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn recv_timeout(&self, timeout: Duration) -> Result<T, RecvTimeoutError> {
        let mut sleep = crate::time::sleep(timeout);
        crate::thread::block_on_poll(|cx| {
            match self.try_recv() {
                Ok(value) => return Poll::Ready(Ok(value)),
                Err(TryRecvError::Disconnected) => {
                    return Poll::Ready(Err(RecvTimeoutError::Disconnected));
                }
                Err(TryRecvError::Empty) => {}
            }
            if self.poll_ready(cx).is_ready() {
                return match self.try_recv() {
                    Ok(value) => Poll::Ready(Ok(value)),
                    Err(TryRecvError::Disconnected) => {
                        Poll::Ready(Err(RecvTimeoutError::Disconnected))
                    }
                    Err(TryRecvError::Empty) => Poll::Pending,
                };
            }
            match std::pin::Pin::new(&mut sleep).poll(cx) {
                Poll::Ready(()) => Poll::Ready(Err(RecvTimeoutError::Timeout)),
                Poll::Pending => Poll::Pending,
            }
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn recv(&self) -> Result<T, RecvError> {
        unreachable!("web-std channels only block on wasm32; native code uses crossbeam")
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn recv_timeout(&self, _timeout: Duration) -> Result<T, RecvTimeoutError> {
        unreachable!("web-std channels only block on wasm32; native code uses crossbeam")
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.shared.0.lock().unwrap().receivers -= 1;
    }
}

type Probe<'a> = Box<dyn Fn(&mut Context<'_>) -> Poll<()> + 'a>;

pub struct Select<'a> {
    probes: Vec<Probe<'a>>,
}

impl<'a> Default for Select<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Select<'a> {
    pub fn new() -> Self {
        Select { probes: Vec::new() }
    }

    pub fn recv<T: 'a>(&mut self, rx: &'a Receiver<T>) -> usize {
        self.probes.push(Box::new(move |cx| rx.poll_ready(cx)));
        self.probes.len() - 1
    }

    #[cfg(target_arch = "wasm32")]
    pub fn ready(&mut self) -> usize {
        crate::thread::block_on_poll(|cx| {
            for (index, probe) in self.probes.iter().enumerate() {
                if probe(cx).is_ready() {
                    return Poll::Ready(index);
                }
            }
            Poll::Pending
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn ready(&mut self) -> usize {
        unreachable!("web-std channels only block on wasm32; native code uses crossbeam")
    }
}

pub struct SegQueue<T> {
    queue: Mutex<VecDeque<T>>,
}

impl<T> Default for SegQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SegQueue<T> {
    pub const fn new() -> Self {
        SegQueue {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    pub fn push(&self, value: T) {
        self.queue.lock().unwrap().push_back(value);
    }

    pub fn pop(&self) -> Option<T> {
        self.queue.lock().unwrap().pop_front()
    }

    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
