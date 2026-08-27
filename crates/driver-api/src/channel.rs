//! The typed channel declaration — what `register_channel` states and answers.
//!
//! A channel is PTIR's only stateful construct: a bounded SPSC ring between
//! the pass and either the host or another instance. The engine registers one
//! before it binds the instances that share it, which is why this is its own
//! verb and not a field of [`InstanceBinding`](crate::program::InstanceBinding).
//!
//! # What died here
//!
//! `RegisteredChannel` carried a `ChannelBinding` — eleven `u64`/`u32` fields
//! naming device addresses and word indices (`mirror_base`, `word_base`,
//! `head_word_index`, `poison_word_index`, …) — plus a
//! `validate_channel_endpoint_binding` that checked the driver had filled them
//! consistently. That is a driver's private ring layout, published into the
//! contract so a C caller could poke it; the shells in this workspace are Rust
//! and drive their own rings. What the engine needs back is the id and the two
//! wait slots, and that is what comes back.
//!
//! The dtype and role bytes went the way of [`program`](crate::program)'s: a
//! declaration names [`ChanDType`], [`HostRole`] and [`ExternDir`], the three
//! types PTIR already has for them.

use serde::{Deserialize, Serialize};

use tensor_ir::container::{ChanDType, ExternDir, HostRole};

/// A registered channel's id, minted by the caller and acknowledged by the
/// driver.
pub type ChannelId = u64;

/// A value put into a channel — the wire cells, as the ring holds them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSeed {
    /// Which channel, in the package's declaration order.
    pub channel: u32,
    /// The cell bytes.
    pub bytes: Vec<u8>,
}

/// Everything a channel registration states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRegistration {
    /// The id the caller wants this channel to have.
    pub id: ChannelId,
    /// The cell's shape, as dims.
    pub shape: Vec<u32>,
    /// The cell's element type.
    pub dtype: ChanDType,
    /// Which end the host holds, if any.
    pub host_role: HostRole,
    /// Whether the ring arrives holding a value.
    pub seeded: bool,
    /// Whether this channel crosses to another instance, and which way.
    pub extern_dir: Option<ExternDir>,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// The extern binding's name, when `extern_dir` is `Some`.
    pub extern_name: Vec<u8>,
}

impl Default for ChannelRegistration {
    fn default() -> ChannelRegistration {
        ChannelRegistration {
            id: 0,
            shape: Vec::new(),
            dtype: ChanDType::Concrete(tensor_ir::types::DType::F32),
            host_role: HostRole::None,
            seeded: false,
            extern_dir: None,
            capacity: 0,
            extern_name: Vec::new(),
        }
    }
}

/// A registered channel, as the driver answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredChannel {
    /// The channel's id.
    pub id: ChannelId,
    /// The wait slot a reader parks on until the ring is non-empty.
    pub reader_wait_id: u64,
    /// The wait slot a writer parks on until the ring has room.
    pub writer_wait_id: u64,
}
