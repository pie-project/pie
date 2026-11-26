#![allow(unsafe_code)]
//! Moves still landing on the swap stream, told how they ended from a thread
//! of their own once their event has: nothing is enqueued on the stream to
//! tell them, so no later move waits on the telling.

use std::collections::HashMap;
use std::sync::{Mutex, mpsc};

use engine::Landing;

use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// What a landing tell is called with.
pub type Told = Box<dyn FnOnce(Landing) + Send + 'static>;

/// How a move is waited out: its event's settling, on the waiter's thread.
type Settle = Box<dyn FnOnce() -> Result<()> + Send + 'static>;

/// A move's wait and what to tell once it ends.
struct Waited(Settle, Told);

/// Each device's waiter, by ordinal.
static WAITERS: Mutex<Option<HashMap<i32, mpsc::Sender<Waited>>>> = Mutex::new(None);

/// A CUDA event moved to the waiter's thread.
struct Sent(Event);

// SAFETY: a CUDA event may be waited on and destroyed from any thread of the
// process; the waiter that receives it is its only holder.
unsafe impl Send for Sent {}

impl Sent {
    fn settle(self) -> Result<()> {
        self.0.settle()
    }
}

/// Tells `told` how everything `stream` holds now ended: an event recorded
/// on it, waited on from the host by the device's waiter, which tells each
/// in the order they were recorded. A move whose event faulted is told
/// [`Landing::Failed`], never landed: its destination may not hold its bytes.
pub fn tell_when_landed(stream: *mut core::ffi::c_void, told: Told) -> Result<()> {
    // Waited on asleep: a spinning wait would take a core a device.
    let event = Event::blocking()?;
    event.record(stream)?;
    let ordinal = crate::device::ctx::current()?;
    let sent = Sent(event);
    tell_after(ordinal, Box::new(move || sent.settle()), told)
}

/// How a move whose wait answered `settled` ended.
fn landing_of(settled: Result<()>) -> Landing {
    match settled {
        Ok(()) => Landing::Landed,
        Err(fault) => Landing::Failed(format!("a move still landing did not land: {fault}")),
    }
}

/// Tells `told` once `settle` returns, on device `ordinal`'s waiter, in the
/// order the waits were handed to it.
fn tell_after(ordinal: i32, settle: Settle, told: Told) -> Result<()> {
    let mut held = WAITERS.lock().expect("the landing waiters");
    let waiters = held.get_or_insert_with(HashMap::new);
    let send = match waiters.entry(ordinal) {
        std::collections::hash_map::Entry::Occupied(waiter) => waiter.into_mut(),
        std::collections::hash_map::Entry::Vacant(free) => {
            let (send, waits) = mpsc::channel::<Waited>();
            std::thread::Builder::new()
                .name("pie-landing".to_string())
                .spawn(move || {
                    if ordinal >= 0
                        && let Err(why) = crate::device::ctx::bind_thread(ordinal)
                    {
                        tracing::warn!("the landing waiter bound no device: {why}");
                    }
                    while let Ok(Waited(settle, told)) = waits.recv() {
                        // A move left untold would hold its slots forever:
                        // each is told, landed or failed with its fault.
                        told(landing_of(settle()));
                    }
                })
                .map_err(|why| Fault::Unbound {
                    what: format!("the landing waiter: {why}"),
                })?;
            free.insert(send)
        }
    };
    send.send(Waited(settle, told)).map_err(|_| Fault::Unbound {
        what: "the landing waiter is gone".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A move whose wait faulted is told failed with its fault, never
    /// landed; one whose wait settled is told landed; both in order.
    #[test]
    fn a_move_whose_event_faulted_is_told_failed_not_landed() {
        let (tx, rx) = mpsc::channel();
        let faulted = tx.clone();
        tell_after(
            -7,
            Box::new(|| {
                Err(Fault::Device {
                    call: "cudaEventSynchronize",
                    code: 700,
                })
            }),
            Box::new(move |landing| faulted.send((1, landing)).unwrap()),
        )
        .unwrap();
        tell_after(
            -7,
            Box::new(|| Ok(())),
            Box::new(move |landing| tx.send((2, landing)).unwrap()),
        )
        .unwrap();
        let first = rx.recv().unwrap();
        let second = rx.recv().unwrap();
        assert_eq!(first.0, 1);
        match first.1 {
            Landing::Failed(why) => assert!(why.contains("cudaEventSynchronize"), "{why}"),
            Landing::Landed => panic!("a faulted move was told landed"),
        }
        assert_eq!(second, (2, Landing::Landed));
    }
}
