#![allow(unsafe_code)]
//! Moves still landing on the swap stream, told landed from a thread of
//! their own once their event has: nothing is enqueued on the stream to tell
//! them, so no later move waits on the telling.

use std::collections::HashMap;
use std::sync::{Mutex, mpsc};

use crate::device::graph::Event;
use crate::error::{Fault, Result};

/// A move's event and what to call once it has landed.
struct Waited(Event, Box<dyn FnOnce() + Send + 'static>);

// SAFETY: a CUDA event may be waited on and destroyed from any thread of the
// process; the waiter that receives it is its only holder.
unsafe impl Send for Waited {}

/// Each device's waiter, by ordinal.
static WAITERS: Mutex<Option<HashMap<i32, mpsc::Sender<Waited>>>> = Mutex::new(None);

/// Calls `landed` once everything `stream` holds now has run: an event
/// recorded on it, waited on from the host by the device's waiter, which
/// tells each in the order they were recorded.
pub fn tell_when_landed(
    stream: *mut core::ffi::c_void,
    landed: Box<dyn FnOnce() + Send + 'static>,
) -> Result<()> {
    // Waited on asleep: a spinning wait would take a core a device.
    let event = Event::blocking()?;
    event.record(stream)?;
    let ordinal = crate::device::ctx::current()?;
    let mut held = WAITERS.lock().expect("the landing waiters");
    let waiters = held.get_or_insert_with(HashMap::new);
    let send = match waiters.entry(ordinal) {
        std::collections::hash_map::Entry::Occupied(waiter) => waiter.into_mut(),
        std::collections::hash_map::Entry::Vacant(free) => {
            let (send, waits) = mpsc::channel::<Waited>();
            std::thread::Builder::new()
                .name("pie-landing".to_string())
                .spawn(move || {
                    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
                        eprintln!("engine-cuda: the landing waiter bound no device: {why}");
                    }
                    while let Ok(Waited(event, landed)) = waits.recv() {
                        // A device fault surfaces on the stream's next use;
                        // a move left untold would hold its slots forever.
                        let _ = event.settle();
                        landed();
                    }
                })
                .map_err(|why| Fault::Unbound {
                    what: format!("the landing waiter: {why}"),
                })?;
            free.insert(send)
        }
    };
    send.send(Waited(event, landed)).map_err(|_| Fault::Unbound {
        what: "the landing waiter is gone".to_string(),
    })
}
