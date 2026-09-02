//! The typed channel declaration — what `register_channel` states and
//! answers: a bounded SPSC ring between the pass and either the host or
//! another instance, registered before the instances that share it bind.

use serde::{Deserialize, Serialize};

use eta_ir::container::{ChanDType, ExternDir, HostRole};

/// A registered channel's id, minted by the caller and acknowledged by the
/// engine.
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
            dtype: ChanDType::Concrete(eta_ir::types::Dtype::F32),
            host_role: HostRole::None,
            seeded: false,
            extern_dir: None,
            capacity: 0,
            extern_name: Vec::new(),
        }
    }
}

/// Host end of a channel, as the engine allocated it: mapped pinned memory, addressable from both sides. Owned by the engine until `close_channel`.
/// Layout: `capacity + 1` cells of `cell_bytes` at [`mirror`](Self::mirror), four `u64` control words `[head, tail, poison, closed]` at [`words`](Self::words).
/// Not `Deserialize`: these are host addresses, meaningless across a wire, so a serialized registration must arrive with `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMirror {
    /// Address of cell zero, host side.
    pub mirror: u64,
    /// Address of control word zero, host side.
    pub words: u64,
    /// Bytes one wire cell occupies.
    pub cell_bytes: u32,
    /// Cells the ring holds, not counting the spare.
    pub capacity: u32,
}

/// A registered channel, as the engine answers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredChannel {
    /// The channel's id.
    pub id: ChannelId,
    /// The wait slot a reader parks on until the ring is non-empty. Zero
    /// means the engine keeps no waker table; the caller mints its own slot.
    pub reader_wait_id: u64,
    /// The wait slot a writer parks on until the ring has room. As
    /// [`reader_wait_id`](Self::reader_wait_id), zero means "mint your own".
    pub writer_wait_id: u64,
    /// The pinned host half of this channel's ring, when the engine allocated one; `None` if the caller owns its ring instead.
    #[serde(skip)]
    pub mirror: Option<HostMirror>,
}

/// A prediction of where a channel's cursors will be when this lane's pass runs. The host never reads device state; it counts from monotone `u64` counters (never wrapped, so emptiness is `tail > head`).
/// Validated device-side only if [`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit); otherwise a stated ticket is refused by name rather than dropped silently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ticket {
    /// Which channel this predicts about.
    pub channel: ChannelId,
    /// Where the committed front will be — the cell a `take` reads.
    pub expected_head: u64,
    /// Where the pending back will be — the cell a `put` writes.
    pub expected_tail: u64,
}

impl Ticket {
    /// No claim about this end of the ring (same sentinel
    /// `kernels_cuda::channel::NO_TICKET` reads).
    pub const NONE: u64 = u64::MAX;
}
