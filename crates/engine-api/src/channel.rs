//! The typed channel declaration — what `register_channel` states and answers.
//!
//! A channel is PTIR's only stateful construct: a bounded SPSC ring between
//! the pass and either the host or another instance. The runtime registers one
//! before it binds the instances that share it, which is why this is its own
//! verb and not a field of [`InstanceBinding`](crate::program::InstanceBinding).
//!
//! # What died here
//!
//! `RegisteredChannel` carried a `ChannelBinding` — eleven `u64`/`u32` fields
//! naming device addresses and word indices (`mirror_base`, `word_base`,
//! `head_word_index`, `poison_word_index`, …) — plus a
//! `validate_channel_endpoint_binding` that checked the engine had filled them
//! consistently. That is an engine's private ring layout, published into the
//! contract so a C caller could poke it; the shells in this workspace are Rust
//! and drive their own rings. What the runtime needs back is the id and the two
//! wait slots, and that is what comes back.
//!
//! The dtype and role bytes went the way of [`program`](crate::program)'s: a
//! declaration names [`ChanDType`], [`HostRole`] and [`ExternDir`], the three
//! types PTIR already has for them.

use serde::{Deserialize, Serialize};

use tensor_ir::container::{ChanDType, ExternDir, HostRole};

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
            dtype: ChanDType::Concrete(tensor_ir::types::DType::F32),
            host_role: HostRole::None,
            seeded: false,
            extern_dir: None,
            capacity: 0,
            extern_name: Vec::new(),
        }
    }
}

/// **THE HOST END OF A CHANNEL, AS THE ENGINE ALLOCATED IT** (alto design §5,
/// survey §7 invariant I5).
///
/// An engine that declares
/// [`device_channel_commit`](crate::Capabilities::device_channel_commit) does
/// not want the caller's cells handed to it; it wants the caller to WRITE
/// THEM WHERE ITS KERNELS WILL READ THEM. So it allocates the ring's host
/// half itself — mapped pinned memory, addressable from both sides — and
/// publishes the two addresses here. The caller's ring becomes a view of
/// these bytes rather than a second allocation, and a guest round trip makes
/// no device call at all.
///
/// The layout is the one both sides already speak: `capacity + 1` cells of
/// `cell_bytes` at [`mirror`](Self::mirror), and four `u64` control words
/// `[head, tail, poison, closed]` at [`words`](Self::words).
///
/// **THE MEMORY IS THE ENGINE'S**, and it lives until `close_channel` — an
/// adopted view must not outlive the registration it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// The wait slot a reader parks on until the ring is non-empty.
    ///
    /// **Zero means the engine keeps no waker table and the caller mints its
    /// own slot.** An engine whose only business with this channel is the
    /// memory below has no park/wake machinery to offer, and inventing an id
    /// it would never signal is worse than saying so.
    pub reader_wait_id: u64,
    /// The wait slot a writer parks on until the ring has room. As
    /// [`reader_wait_id`](Self::reader_wait_id), zero means "mint your own".
    pub writer_wait_id: u64,
    /// The pinned host half of this channel's ring, when the engine allocated
    /// one. `None` is the pre-F2a shape: the caller owns its ring and cells
    /// cross through `publish_channel`/`take_channel`.
    #[serde(default)]
    pub mirror: Option<HostMirror>,
}

/// **A prediction about where a channel's cursors will be when this lane's
/// pass runs** (alto design §1 article 3: *programs own channels; the host
/// owns the predictions*).
///
/// The host never reads device state to learn a cursor. It COUNTS — tickets
/// are minted from monotone counters runtime-side — and states what it
/// believes the ring's head and tail will be at the instant the lane's pass
/// takes and puts. The device validates the prediction where the data is, in
/// the pull-validate kernel, and advances durable state only through the
/// predicated commit-bump kernel; a refused pass is unobservable and its
/// refusal reaches its successors as data on the stream, at stream speed.
///
/// **AN ENGINE VALIDATES ONE ONLY IF IT SAYS IT DOES.** The two control
/// kernels landed for CUDA in wave F2a, and
/// [`Capabilities::device_channel_commit`](crate::Capabilities::device_channel_commit)
/// is where an engine states that they did. An engine that does not is handed
/// [`Lane::channels`] only EMPTY and refuses a stated ticket by name
/// ([`Lane::validate_for`]) rather than dropping it silently — because a host
/// that predicted a cursor and was ignored would be told its pass ran against
/// the cell it named when it ran against whatever the ring happened to hold.
///
/// The counters are `u64` and MONOTONE — they are counted, not wrapped, which
/// is what lets emptiness be `tail > head` and fullness a subtraction.
/// [`Ticket::NONE`] is "this end of the ring is not claimed".
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
    /// The absent prediction: this fire makes no claim about that end of the
    /// ring. The same sentinel the device kernels read
    /// (`kernels_cuda::channel::NO_TICKET`).
    pub const NONE: u64 = u64::MAX;
}
