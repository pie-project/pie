//! The channel vocabulary both sides say: the seed a bind carries
//! ([`ChannelValue`]) and what a driver answers when it registers one
//! ([`RegisteredChannel`]). The owning-side handle is NOT here — a driver has
//! no verb that takes a `ChannelEndpoint`, so it stays in `engine`.

use crate::local::ChannelBinding;

/// One channel's initial (seed) value delivered at bind time — the
/// driver-facing seed-table entry `InstanceBindingPlan::seed_values` carries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    /// Which channel the seed belongs to.
    pub channel: u64,
    /// The seed bytes, verbatim.
    pub bytes: Vec<u8>,
}

/// A channel a driver has registered, and the binding it answered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredChannel {
    /// Which driver holds the registration.
    pub driver_id: usize,
    /// Where the driver put the ring, as it reported it.
    pub binding: ChannelBinding,
    /// The wait slot a reader parks on.
    pub reader_wait_id: u64,
    /// The wait slot a writer parks on.
    pub writer_wait_id: u64,
}
