//! The channel vocabulary both sides say: the seed value a bind carries
//! ([`ChannelValue`]) and what a driver answers when it registers one
//! ([`RegisteredChannel`]).
//!
//! The owning-side handle an application holds is NOT here. `ChannelEndpoint`
//! has wait, poison and close semantics that are the engine's alone — a driver
//! never holds one and has no verb that takes one — so it stays in `engine`
//! beside the scheduler that closes it. What crosses is the registration and
//! its acknowledgement, which is what this module is.

use crate::local::ChannelBinding;

/// One channel's initial (seed) value delivered at bind time — `channel` is
/// the global channel identity, `bytes` its native-encoded wire payload. No
/// IR semantics live here; this is purely the driver-facing seed-table
/// entry [`InstanceBindingPlan::seed_values`](crate::instance::InstanceBindingPlan::seed_values)
/// carries, next to the `LaunchPlan`/`InstanceBindingPlan` it feeds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    /// The channel this value seeds.
    pub channel: u64,
    /// The value, in the channel's own encoding.
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
