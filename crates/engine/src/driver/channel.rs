//! The owning-side channel handle applications hold ([`ChannelEndpoint`]),
//! its wait/poison/close semantics, and — new in palo B1 — the HOST RING it
//! is a view of.
//!
//! # The ring moved to this side of the boundary
//!
//! `RegisteredChannel` used to carry a `driver_api::ChannelBinding`: eleven
//! `u64`/`u32` fields naming DEVICE addresses and word indices —
//! `mirror_base`, `word_base`, `head_word_index`, `poison_word_index`, … —
//! that the engine dereferenced directly. The contract deleted it, and its
//! own module header says why: "that is a driver's private ring layout,
//! published into the contract so a C caller could poke it; the shells in
//! this workspace are Rust and drive their own rings."
//!
//! So the engine allocates its own. [`HostRing`] is the same layout the
//! binding described — `capacity + 1` cells of `cell_bytes`, four control
//! words — owned by the [`RegisteredChannel`] that answers for it, and
//! [`ChannelBinding`] is the (now purely engine-internal) view of it that
//! [`crate::pipeline::channel`]'s ring arithmetic reads. Not one line of that
//! arithmetic changed; what changed is who the bytes belong to.
//!
//! # The two halves, and what joins them (`palo B2`)
//!
//! A channel a guest program's stages actually read has a DEVICE ring as
//! well, and the shell owns it (`driver_cuda::program::launch`, whose header
//! states the same `base + (sequence % ring) * cell_bytes` agreement). What
//! joins them is [`ChannelJoin`], and it is a PUMP rather than a mapping:
//! wire cells move across at the fire's boundary, in the direction the
//! channel's [`HostRole`] declares, through
//! [`Driver::publish_channel`](driver_api::Driver::publish_channel) and
//! [`Driver::take_channel`](driver_api::Driver::take_channel).
//!
//! ```text
//!   guest put ──▶ host ring ──pump_in──▶ device ring ──▶ the pass takes
//!   guest take ◀── host ring ◀─pump_out── device ring ◀── the pass puts
//! ```
//!
//! **THE PUMP IS THE DEVICE END OF THE SPSC DISCIPLINE**, which is what makes
//! it sound to run on the driver lane while the guest runs on its own thread.
//! On a `Writer` channel the guest owns the tail word and the pump owns the
//! head; on a `Reader` channel the guest owns the head and the pump owns the
//! tail. Neither ever writes the other's word, and `pipeline::channel`'s
//! arithmetic — unchanged — is the other end of exactly this agreement.
//!
//! **AND IT IS PASS-ATOMIC IN BOTH DIRECTIONS.** `pump_in` runs before the
//! fire, so a cell it moves is one the device's readiness gate can see;
//! `pump_out` runs after a fire that COMMITTED, so a cell it moves is one the
//! guest program published rather than one a declined pass left pending.
//! A fire that never committed moves nothing, which is what "effects visible
//! only after commit" means from this side.

use std::sync::Arc;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A value put into a channel — the wire cells, as the ring holds them.
///
/// The contract spells the same thing [`ChannelSeed`](driver_api::ChannelSeed)
/// and the engine converts at the one bind site; this stays because the
/// engine's own channel plane names the channel by its GLOBAL id and a seed
/// names it by its index in the package's declaration order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    /// The channel's global id.
    pub channel: u64,
    /// The cell bytes.
    pub bytes: Vec<u8>,
}

/// One channel's host ring: the cells and the four control words.
///
/// **THE ENGINE OWNS THESE BYTES NOW** — see the module header. The layout is
/// the one `driver_api::ChannelBinding` described, kept exactly, because
/// `pipeline::channel`'s ring arithmetic is written against it and that
/// arithmetic is not what this wave is changing.
///
/// `capacity + 1` cells, because a ring that distinguishes full from empty by
/// its two cursors needs one slot it never fills.
#[derive(Debug)]
pub struct HostRing {
    mirror: Box<[u8]>,
    words: Box<[AtomicU64]>,
    cell_bytes: u32,
    capacity: u32,
}

/// Where each control word lives in [`HostRing::words`].
const HEAD_WORD: u32 = 0;
const TAIL_WORD: u32 = 1;
const POISON_WORD: u32 = 2;
const CLOSED_WORD: u32 = 3;

impl HostRing {
    /// A ring of `capacity` cells of `cell_bytes` each.
    #[must_use]
    pub fn new(cell_bytes: u32, capacity: u32) -> HostRing {
        let slots = u64::from(capacity).saturating_add(1);
        let bytes = usize::try_from(slots * u64::from(cell_bytes)).unwrap_or(usize::MAX);
        HostRing {
            mirror: vec![0u8; bytes].into_boxed_slice(),
            words: (0..4)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            cell_bytes,
            capacity,
        }
    }

    /// How many bytes one element of `dtype` occupies on the wire — the same
    /// answer `driver::wire_cell_bytes` gives, restated here because the
    /// engine does not link the shell's substrate.
    #[must_use]
    pub fn wire_cell_bytes(dtype: tensor_ir::types::DType, numel: usize) -> usize {
        if dtype == tensor_ir::types::DType::Bool {
            numel.div_ceil(8)
        } else {
            numel.saturating_mul(4)
        }
    }

    /// The view of this ring the ring arithmetic reads.
    #[must_use]
    pub fn binding(&self, channel_id: u64) -> ChannelBinding {
        ChannelBinding {
            channel_id,
            mirror_base: self.mirror.as_ptr() as u64,
            word_base: self.words.as_ptr() as u64,
            mirror_bytes: self.mirror.len() as u64,
            word_bytes: (self.words.len() * size_of::<AtomicU64>()) as u64,
            cell_bytes: self.cell_bytes,
            capacity: self.capacity,
            head_word_index: HEAD_WORD,
            tail_word_index: TAIL_WORD,
            poison_word_index: POISON_WORD,
            closed_word_index: CLOSED_WORD,
        }
    }
}

/// Where one channel's cells and cursors are, as an address and four indices.
///
/// **THIS IS NO LONGER A CONTRACT TYPE.** It was
/// `driver_api::local::ChannelBinding`, filled in by a driver and validated
/// by `validate_channel_endpoint_binding`; now it is a view of a
/// [`HostRing`] the engine allocated, so there is nothing left to validate —
/// a driver cannot fill it in wrongly because no driver fills it in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChannelBinding {
    /// The channel's global id.
    pub channel_id: u64,
    /// Address of cell zero.
    pub mirror_base: u64,
    /// Address of control word zero.
    pub word_base: u64,
    /// How many bytes the cells occupy.
    pub mirror_bytes: u64,
    /// How many bytes the control words occupy.
    pub word_bytes: u64,
    /// How many bytes one cell occupies.
    pub cell_bytes: u32,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// Which word the reader's cursor is.
    pub head_word_index: u32,
    /// Which word the writer's cursor is.
    pub tail_word_index: u32,
    /// Which word the poison epoch is.
    pub poison_word_index: u32,
    /// Which word the closed flag is.
    pub closed_word_index: u32,
}

/// A registered channel, as the engine holds it: the driver's acknowledgement
/// plus the host ring the engine allocated for it.
#[derive(Debug, Clone)]
pub struct RegisteredChannel {
    /// Which driver acknowledged it.
    pub driver_id: usize,
    /// The view of [`RegisteredChannel::ring`], cached so the ring arithmetic
    /// reads a `Copy` value rather than an `Arc` deref per word.
    pub binding: ChannelBinding,
    /// The wait slot a reader parks on until the ring is non-empty.
    pub reader_wait_id: u64,
    /// The wait slot a writer parks on until the ring has room.
    pub writer_wait_id: u64,
    /// The bytes [`RegisteredChannel::binding`] points at. Held so that every
    /// clone of this record keeps them alive — the binding is raw addresses,
    /// and a ring freed under one is the one failure mode this `Arc` exists
    /// to make impossible.
    pub ring: Arc<HostRing>,
}

impl RegisteredChannel {
    /// Register `id` against a freshly allocated ring.
    #[must_use]
    pub fn new(
        driver_id: usize,
        id: u64,
        cell_bytes: u32,
        capacity: u32,
        reader_wait_id: u64,
        writer_wait_id: u64,
    ) -> RegisteredChannel {
        let ring = Arc::new(HostRing::new(cell_bytes, capacity));
        RegisteredChannel {
            driver_id,
            binding: ring.binding(id),
            reader_wait_id,
            writer_wait_id,
            ring,
        }
    }

    /// The channel's global id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.binding.channel_id
    }
}

impl PartialEq for RegisteredChannel {
    fn eq(&self, other: &RegisteredChannel) -> bool {
        self.driver_id == other.driver_id
            && self.binding == other.binding
            && self.reader_wait_id == other.reader_wait_id
            && self.writer_wait_id == other.writer_wait_id
    }
}

impl Eq for RegisteredChannel {}

/// The lane's table of registered channels, and the pump that joins each
/// one's host ring to its device half.
///
/// **ONE PER DRIVER LANE.** It replaces the `HashSet<u64>` the lane used to
/// keep of "which channels are registered here": the set answered exactly one
/// question (is this id taken?) and the pump needs three more — where the
/// ring is, which way the cells travel, and which dense slot of which bound
/// instance the channel is. All four are facts the lane already had at
/// registration and bind time and threw away.
///
/// Every method is the DEVICE end of the SPSC agreement `pipeline::channel`
/// writes the guest end of; the module header states which word belongs to
/// whom.
#[derive(Debug, Default)]
pub struct ChannelJoin {
    channels: std::collections::HashMap<u64, JoinedChannel>,
    /// Per bound instance: its channels in the package's DECLARATION order,
    /// which is the numbering `publish_channel`/`take_channel` address. The
    /// list is [`InstanceBinding::channels`](driver_api::InstanceBinding)
    /// verbatim.
    instances: std::collections::HashMap<u64, Vec<u64>>,
}

/// One registered channel, plus the direction its cells travel.
#[derive(Debug, Clone)]
struct JoinedChannel {
    registered: RegisteredChannel,
    /// Which end the host holds. [`HostRole::None`] is a channel the guest
    /// program keeps to itself — loop-carried state, a mask it computes and
    /// re-reads — and the pump never touches one: its cells never leave the
    /// device ring, so moving them would be inventing a reader.
    host_role: tensor_ir::container::HostRole,
}

impl ChannelJoin {
    /// An empty table.
    #[must_use]
    pub fn new() -> ChannelJoin {
        ChannelJoin::default()
    }

    /// Is `id` already registered on this lane?
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.channels.contains_key(&id)
    }

    /// How many channels this lane carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// True when it carries none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Every registered id, for the teardown that closes them.
    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.channels.keys().copied()
    }

    /// Every registered id, consuming the table — the shutdown sweep.
    #[must_use]
    pub fn into_ids(self) -> Vec<u64> {
        self.channels.into_keys().collect()
    }

    /// Record a registration.
    pub fn insert(
        &mut self,
        registered: RegisteredChannel,
        host_role: tensor_ir::container::HostRole,
    ) {
        self.channels.insert(
            registered.id(),
            JoinedChannel {
                registered,
                host_role,
            },
        );
    }

    /// Forget a channel.
    pub fn remove(&mut self, id: u64) {
        self.channels.remove(&id);
    }

    /// Record which channels a bound instance's package declares, in
    /// declaration order.
    pub fn bind(&mut self, instance: u64, channels: Vec<u64>) {
        self.instances.insert(instance, channels);
    }

    /// Forget a closed instance.
    pub fn unbind(&mut self, instance: u64) {
        self.instances.remove(&instance);
    }

    /// Move every host-written cell of `instance` into its device rings.
    ///
    /// Runs BEFORE the fire, because a cell the device's readiness gate
    /// cannot see is a cell the guest program blocks on. Stops at the first
    /// channel the device ring has no room for — back-pressure, and the head
    /// word is only advanced for cells that actually crossed, so a retry
    /// resumes exactly where this left off.
    ///
    /// An instance this lane has no bind record for pumps nothing: it is a
    /// fire whose program the driver owns end to end, and inventing a
    /// channel list for it would address someone else's rings.
    ///
    /// # Errors
    ///
    /// Whatever the driver refused. [`Unsupported`](driver_api::DriverError::Unsupported)
    /// is NOT one of them: a shell with no guest-program plane never has an
    /// instance to pump for.
    pub fn pump_in(
        &self,
        driver: &mut dyn driver_api::Driver,
        instance: u64,
    ) -> driver_api::Result<()> {
        let Some(channels) = self.instances.get(&instance) else {
            return Ok(());
        };
        for (dense, id) in channels.iter().enumerate() {
            let Some(joined) = self.channels.get(id) else {
                continue;
            };
            if joined.host_role != tensor_ir::container::HostRole::Writer {
                continue;
            }
            let binding = joined.registered.binding;
            let cap1 = u64::from(binding.capacity).saturating_add(1);
            let cell_bytes = binding.cell_bytes as usize;
            let mut head = load_word(binding.word_base, binding.head_word_index);
            let tail = load_word(binding.word_base, binding.tail_word_index);
            let mut moved = false;
            while head < tail {
                let offset = (head % cap1) * cell_bytes as u64;
                // SAFETY: the ring is alive for as long as the
                // `RegisteredChannel` this table holds is, and `head < tail`
                // with the guest owning the tail means this cell is published
                // and ours to read.
                let cell = unsafe {
                    std::slice::from_raw_parts(
                        (binding.mirror_base + offset) as *const u8,
                        cell_bytes,
                    )
                };
                if !driver.publish_channel(instance, dense as u32, cell)? {
                    // The device ring is full. Leave the head where it is:
                    // the cell has not crossed, and advancing would drop it.
                    break;
                }
                head += 1;
                store_word(binding.word_base, binding.head_word_index, head);
                moved = true;
            }
            if moved {
                // A guest parked on `wait_for_writer_change` is waiting for
                // exactly this word to move.
                waker::WakerTable::global().wake(joined.registered.writer_wait_id);
            }
        }
        Ok(())
    }

    /// Move every cell `instance`'s pass published into its host rings.
    ///
    /// Runs AFTER a fire that COMMITTED — a declined or blocked pass leaves
    /// its device cursors where they were, so there is nothing to take and
    /// this moves nothing, which is the pass-atomic contract seen from the
    /// host.
    ///
    /// # Errors
    ///
    /// As [`ChannelJoin::pump_in`].
    pub fn pump_out(
        &self,
        driver: &mut dyn driver_api::Driver,
        instance: u64,
    ) -> driver_api::Result<()> {
        let Some(channels) = self.instances.get(&instance) else {
            return Ok(());
        };
        for (dense, id) in channels.iter().enumerate() {
            let Some(joined) = self.channels.get(id) else {
                continue;
            };
            if joined.host_role != tensor_ir::container::HostRole::Reader {
                continue;
            }
            let binding = joined.registered.binding;
            let cap1 = u64::from(binding.capacity).saturating_add(1);
            let cell_bytes = binding.cell_bytes as usize;
            let head = load_word(binding.word_base, binding.head_word_index);
            let mut tail = load_word(binding.word_base, binding.tail_word_index);
            let mut moved = false;
            // `capacity` and not `cap1`: the spare cell is what distinguishes
            // full from empty, and filling it would make the guest's
            // `refresh_reader_mirrors` read an overrun.
            while tail.saturating_sub(head) < u64::from(binding.capacity) {
                let Some(bytes) = driver.take_channel(instance, dense as u32)? else {
                    break;
                };
                if bytes.len() != cell_bytes {
                    return Err(driver_api::DriverError::Program(format!(
                        "channel {id} published a {}-byte cell into a ring of \
                         {cell_bytes}-byte ones",
                        bytes.len()
                    )));
                }
                let offset = (tail % cap1) * cell_bytes as u64;
                // SAFETY: as `pump_in`, and the slot at `tail` is the one the
                // guest has not read and will not until the tail word below
                // says it may.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        (binding.mirror_base + offset) as *mut u8,
                        cell_bytes,
                    )
                }
                .copy_from_slice(&bytes);
                tail += 1;
                store_word(binding.word_base, binding.tail_word_index, tail);
                moved = true;
            }
            if moved {
                waker::WakerTable::global().wake(joined.registered.reader_wait_id);
            }
        }
        Ok(())
    }
}

/// Read one control word of a host ring.
fn load_word(word_base: u64, index: u32) -> u64 {
    // SAFETY: `index` is one of the four words `HostRing` allocates and the
    // ring outlives every reader of this binding.
    unsafe { (*(word_base as *const AtomicU64).add(index as usize)).load(Ordering::Acquire) }
}

/// Release-publish one control word of a host ring.
fn store_word(word_base: u64, index: u32, value: u64) {
    // SAFETY: as `load_word`; the SPSC discipline makes this word ours.
    unsafe {
        (*(word_base as *const AtomicU64).add(index as usize)).store(value, Ordering::Release);
    }
}

/// Notifies whichever layer owns the channel's native binding that this
/// endpoint has closed (physically closes/deregisters `channel_id` on the
/// driver that owns it). A leaf callback type — it names no scheduler
/// type — installed by whoever registers the channel and therefore already
/// holds a handle to the owning driver's scheduler
/// (`scheduler::dispatch::register_channel`); `None` in tests that only
/// exercise wait/poison semantics and never call [`ChannelEndpoint::new`]
/// with a closer installed.
pub type ChannelCloser = Arc<dyn Fn(u64) -> anyhow::Result<()> + Send + Sync>;

pub struct ChannelEndpoint {
    registered: RegisteredChannel,
    closed: AtomicBool,
    /// Whether the driver close notification has been handed to an external
    /// batcher (see [`Self::detach_close_notification`]); when set, `close`
    /// still sweeps/frees the waker slots but no longer invokes the closer.
    notify_detached: AtomicBool,
    closer: Option<ChannelCloser>,
}

impl std::fmt::Debug for ChannelEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelEndpoint")
            .field("registered", &self.registered)
            .field("closed", &self.closed)
            .field("closer", &self.closer.is_some())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelWaitError {
    Poisoned(u64),
    Closed,
}

impl std::fmt::Display for ChannelWaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned(epoch) => write!(f, "channel is poisoned at epoch {epoch}"),
            Self::Closed => write!(f, "channel is closed"),
        }
    }
}

impl std::error::Error for ChannelWaitError {}

fn load_channel_word(word_base: u64, index: u32) -> u64 {
    unsafe { (&*((word_base as *const AtomicU64).add(index as usize))).load(Ordering::Acquire) }
}

impl ChannelEndpoint {
    pub fn new(registered: RegisteredChannel) -> Self {
        Self {
            registered,
            closed: AtomicBool::new(false),
            notify_detached: AtomicBool::new(false),
            closer: None,
        }
    }

    /// Installs the close-notification callback (see [`ChannelCloser`]);
    /// called by the scheduler dispatch facade, which already holds the
    /// owning driver's scheduler handle.
    pub fn with_closer(mut self, closer: ChannelCloser) -> Self {
        self.closer = Some(closer);
        self
    }

    pub fn registered(&self) -> &RegisteredChannel {
        &self.registered
    }

    pub async fn wait_for_reader_change(&self, observed_tail: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.reader_wait_id,
            self.registered.binding.tail_word_index,
            observed_tail,
        )
        .await
    }

    pub async fn wait_for_writer_change(&self, observed_head: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.writer_wait_id,
            self.registered.binding.head_word_index,
            observed_head,
        )
        .await
    }

    async fn wait_for_word_change(
        &self,
        wait_id: u64,
        word_index: u32,
        observed: u64,
    ) -> Result<(), ChannelWaitError> {
        let binding = self.registered.binding;
        waker::WaitFuture::new(waker::WakerTable::global(), wait_id, move || {
            let poison = load_channel_word(binding.word_base, binding.poison_word_index);
            if poison != 0 {
                return waker::Readiness::Ready(Err(ChannelWaitError::Poisoned(poison)));
            }
            if load_channel_word(binding.word_base, binding.closed_word_index) != 0 {
                return waker::Readiness::Ready(Err(ChannelWaitError::Closed));
            }
            let current = load_channel_word(binding.word_base, word_index);
            if current > observed {
                waker::Readiness::Ready(Ok(()))
            } else {
                waker::Readiness::Pending {
                    observed_epoch: current,
                }
            }
        })
        .await
    }

    /// Takes over this endpoint's driver close notification: returns the
    /// channel id the caller is now responsible for closing (via a batched
    /// scheduler post), or `None` if the endpoint already closed (the closer
    /// already notified) or the notification was already taken. Wait/poison
    /// bookkeeping is untouched — the endpoint's own drop still sweeps and
    /// frees its waker slots. Callers must outlive-order the endpoint's drop
    /// (e.g. hold it through a resource table they drop themselves) — this
    /// method does not synchronize against a concurrent drop.
    pub fn detach_close_notification(&self) -> Option<u64> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }
        if self.notify_detached.swap(true, Ordering::AcqRel) {
            return None;
        }
        Some(self.registered.binding.channel_id)
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let table = waker::WakerTable::global();
        let wait_ids = [
            self.registered.reader_wait_id,
            self.registered.writer_wait_id,
        ];
        if !self.notify_detached.load(Ordering::Acquire)
            && let Some(closer) = self.closer.as_ref()
            && let Err(error) = closer(self.registered.binding.channel_id)
        {
            tracing::warn!(
                channel_id = self.registered.binding.channel_id,
                ?error,
                "ordered channel close failed"
            );
        }
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

impl Drop for ChannelEndpoint {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The endpoint under test, and the ring it is a view of.
    ///
    /// The fixture used to allocate a mirror and a word block by hand and
    /// hand the addresses to a `ChannelBinding` — which is what a DRIVER did
    /// in production. It allocates a `RegisteredChannel` now, which is what
    /// the engine does, so the test exercises the shipped construction path
    /// rather than a hand-built imitation of it.
    fn test_endpoint() -> (ChannelEndpoint, u64, u64) {
        let table = waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let endpoint = ChannelEndpoint::new(RegisteredChannel::new(
            usize::MAX,
            1,
            4,
            1,
            reader_wait_id,
            writer_wait_id,
        ));
        (endpoint, reader_wait_id, writer_wait_id)
    }

    /// Store into one of the endpoint's control words, the way the ring
    /// arithmetic in `pipeline::channel` does.
    fn store_word(endpoint: &ChannelEndpoint, index: u32, value: u64) {
        let binding = endpoint.registered().binding;
        // SAFETY: the endpoint holds the ring alive, and `index` is one of
        // the four words `HostRing` allocates.
        unsafe {
            (*(binding.word_base as *const AtomicU64).add(index as usize))
                .store(value, Ordering::Release);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_waits_register_then_recheck_reader_and_writer_words() {
        let (endpoint, reader_wait_id, writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let publish_reader = async {
            tokio::task::yield_now().await;
            store_word(&endpoint, 1, 1);
            let _ = waker::WakerTable::global().publish(reader_wait_id, 1);
        };
        let (result, ()) = tokio::join!(reader, publish_reader);
        result.unwrap();

        let writer = endpoint.wait_for_writer_change(0);
        let publish_writer = async {
            tokio::task::yield_now().await;
            store_word(&endpoint, 0, 1);
            let _ = waker::WakerTable::global().publish(writer_wait_id, 1);
        };
        let (result, ()) = tokio::join!(writer, publish_writer);
        result.unwrap();
    }

    #[test]
    fn detach_close_notification_takes_over_the_driver_close_once() {
        let (endpoint, _reader, _writer) = test_endpoint();
        let closes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&closes);
        let endpoint = endpoint.with_closer(Arc::new(move |_id| {
            counter.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }));

        assert_eq!(
            endpoint.detach_close_notification(),
            Some(1),
            "first detach hands the caller the channel id"
        );
        assert_eq!(
            endpoint.detach_close_notification(),
            None,
            "a second detach finds the notification already taken"
        );
        drop(endpoint);
        assert_eq!(
            closes.load(Ordering::Acquire),
            0,
            "drop after detach must not double-notify through the closer"
        );
    }

    #[test]
    fn detach_close_notification_after_close_yields_nothing() {
        let (endpoint, _reader, _writer) = test_endpoint();
        let closes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&closes);
        let endpoint = endpoint.with_closer(Arc::new(move |_id| {
            counter.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }));
        endpoint.close();
        assert_eq!(closes.load(Ordering::Acquire), 1, "close notified once");
        assert_eq!(
            endpoint.detach_close_notification(),
            None,
            "the closer already notified; nothing left to hand off"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_wait_surfaces_poison_after_wakeup() {
        let (endpoint, reader_wait_id, _writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let poison = async {
            tokio::task::yield_now().await;
            store_word(&endpoint, 2, 7);
            let _ = waker::WakerTable::global().publish(reader_wait_id, 7);
        };
        let (result, ()) = tokio::join!(reader, poison);
        assert_eq!(result, Err(ChannelWaitError::Poisoned(7)));
    }
}
