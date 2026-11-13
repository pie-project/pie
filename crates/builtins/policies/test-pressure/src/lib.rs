//! Publishes what a `tier-pressure` edge lets it read of the device tier,
//! and discards nothing: the host's default responder must stand back.

use policy::{Id, Subject, attr, worker};

#[derive(Default)]
struct Pressure;

impl worker::Policy for Pressure {
    fn on_tier_pressure(&mut self, id: &Id) {
        let of = Subject::Worker(id.clone());
        for name in ["device-band", "device-short-bytes", "device-total-bytes"] {
            worker::publish(name, attr(&of, name).value().as_ref());
        }
    }
}

policy::worker!(Pressure);
