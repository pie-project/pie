//! The owning-side channel handle ([`ChannelEndpoint`]), its wait/poison/close
//! semantics, and the host-ring/device-ring pump ([`ChannelJoin`]).

use std::sync::Arc;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A value put into a channel — the wire cells, as the ring holds them.
/// Keyed by the channel's global id, unlike [`ChannelSeed`](engine::ChannelSeed)
/// which uses its declaration-order index.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    /// The channel's global id.
    pub channel: u64,
    /// The cell bytes.
    pub bytes: Vec<u8>,
}

/// One channel's host ring: the cells and the four control words.
/// `capacity + 1` cells, because a ring that distinguishes full from empty by
/// its two cursors needs one slot it never fills.
#[derive(Debug)]
pub struct HostRing {
    /// The bytes, when the RUNTIME allocated them. `None` for a ring adopted
    /// from an engine (see [`HostRing::adopt`]): the memory is the engine's
    /// and this is a view of it.
    owned: Option<(Box<[u8]>, Box<[AtomicU64]>)>,
    mirror_base: u64,
    word_base: u64,
    mirror_bytes: u64,
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
        let mirror = vec![0u8; bytes].into_boxed_slice();
        let words: Box<[AtomicU64]> = (0..4)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        HostRing {
            mirror_base: mirror.as_ptr() as u64,
            word_base: words.as_ptr() as u64,
            mirror_bytes: mirror.len() as u64,
            owned: Some((mirror, words)),
            cell_bytes,
            capacity,
        }
    }

    /// A view of an engine's own pinned ring. An engine that declares
    /// [`device_channel_commit`](engine::Capabilities::device_channel_commit)
    /// allocates this channel's host half itself and publishes the two
    /// addresses as [`HostMirror`](engine::HostMirror); adopting them means
    /// no cell is ever copied and a guest round trip makes no device call.
    /// Layout matches [`HostRing::new`] cell for cell and word for word.
    ///
    /// # Safety
    ///
    /// `mirror` must address `(capacity + 1) * cell_bytes` bytes and `words`
    /// four `u64`s, both alive until the channel is closed on the engine that
    /// published them. The [`RegisteredChannel`] holding this ring is what
    /// keeps that ordering: the runtime closes the channel on the engine only
    /// after it has dropped its own record of it.
    #[must_use]
    pub unsafe fn adopt(mirror: u64, words: u64, cell_bytes: u32, capacity: u32) -> HostRing {
        let slots = u64::from(capacity).saturating_add(1);
        HostRing {
            owned: None,
            mirror_base: mirror,
            word_base: words,
            mirror_bytes: slots * u64::from(cell_bytes),
            cell_bytes,
            capacity,
        }
    }

    /// Whether these bytes are an engine's rather than the runtime's — which
    /// is exactly the question "does this channel still need a pump?".
    #[must_use]
    pub const fn adopted(&self) -> bool {
        self.owned.is_none()
    }

    /// How many bytes one element of `dtype` occupies on the wire — the same
    /// answer `engine::wire_cell_bytes` gives, restated here because the
    /// runtime does not link the shell's substrate.
    #[must_use]
    pub fn wire_cell_bytes(dtype: eta_ir::types::Dtype, numel: usize) -> usize {
        if dtype == eta_ir::types::Dtype::Bool {
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
            mirror_base: self.mirror_base,
            word_base: self.word_base,
            mirror_bytes: self.mirror_bytes,
            word_bytes: (4 * size_of::<AtomicU64>()) as u64,
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
/// A view of a [`HostRing`] the runtime allocated; not filled in by an
/// engine, so there is nothing to validate.
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

/// A registered channel, as the runtime holds it: the engine's acknowledgement
/// plus the host ring the runtime allocated for it.
#[derive(Debug, Clone)]
pub struct RegisteredChannel {
    /// Which engine acknowledged it.
    pub engine_id: usize,
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
        engine_id: usize,
        id: u64,
        cell_bytes: u32,
        capacity: u32,
        reader_wait_id: u64,
        writer_wait_id: u64,
    ) -> RegisteredChannel {
        RegisteredChannel::over(
            engine_id,
            id,
            Arc::new(HostRing::new(cell_bytes, capacity)),
            reader_wait_id,
            writer_wait_id,
        )
    }

    /// Register `id` against a ring that already exists — the engine's own,
    /// when it published one ([`HostRing::adopt`]).
    #[must_use]
    pub fn over(
        engine_id: usize,
        id: u64,
        ring: Arc<HostRing>,
        reader_wait_id: u64,
        writer_wait_id: u64,
    ) -> RegisteredChannel {
        RegisteredChannel {
            engine_id,
            binding: ring.binding(id),
            reader_wait_id,
            writer_wait_id,
            ring,
        }
    }

    /// Whether this channel's bytes are the ENGINE's, so that no pump is
    /// needed in either direction.
    #[must_use]
    pub fn adopted(&self) -> bool {
        self.ring.adopted()
    }

    /// The channel's global id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.binding.channel_id
    }
}

impl PartialEq for RegisteredChannel {
    fn eq(&self, other: &RegisteredChannel) -> bool {
        self.engine_id == other.engine_id
            && self.binding == other.binding
            && self.reader_wait_id == other.reader_wait_id
            && self.writer_wait_id == other.writer_wait_id
    }
}

impl Eq for RegisteredChannel {}

/// The lane's table of registered channels, and the pump that joins each
/// one's host ring to its device half. One per engine lane. Every method is
/// the device end of the SPSC agreement `pipeline::channel` writes the guest
/// end of.
#[derive(Debug, Default)]
pub struct ChannelJoin {
    channels: std::collections::HashMap<u64, JoinedChannel>,
    /// Per bound instance: its channels in the package's DECLARATION order,
    /// which is the numbering `publish_channel`/`take_channel` address. The
    /// list is [`InstanceBinding::channels`](engine::InstanceBinding)
    /// verbatim.
    instances: std::collections::HashMap<u64, Vec<u64>>,
}

/// One registered channel, plus the direction its cells travel.
#[derive(Debug, Clone)]
struct JoinedChannel {
    registered: RegisteredChannel,
    /// Which end the host holds. [`HostRole::None`] is a channel the guest
    /// keeps to itself; the pump never touches one.
    host_role: eta_ir::container::HostRole,
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
        host_role: eta_ir::container::HostRole,
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
    /// Runs before the fire. Stops at the first channel the device ring has
    /// no room for; the head word only advances for cells that crossed, so a
    /// retry resumes where this left off. An unbound instance pumps nothing.
    ///
    /// # Errors
    ///
    /// Whatever the engine refused.
    pub fn pump_in(
        &self,
        engine: &mut dyn engine::Engine,
        instance: u64,
    ) -> engine::Result<()> {
        let Some(channels) = self.instances.get(&instance) else {
            return Ok(());
        };
        for (dense, id) in channels.iter().enumerate() {
            let Some(joined) = self.channels.get(id) else {
                continue;
            };
            if joined.host_role != eta_ir::container::HostRole::Writer {
                continue;
            }
            // Adopted channel: the guest already wrote its cell where the
            // engine reads it, and the engine advances the head word at
            // settle — nothing to move here. Only the wake remains, since
            // the engine has no waker table; unconditional, so a spurious
            // wake just costs the waiter a re-read.
            if joined.registered.adopted() {
                waker::WakerTable::global().wake(joined.registered.writer_wait_id);
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
                if !engine.publish_channel(instance, dense as u32, cell)? {
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
    /// Runs after a fire that committed; a declined or blocked pass leaves
    /// nothing to take.
    ///
    /// # Errors
    ///
    /// As [`ChannelJoin::pump_in`].
    pub fn pump_out(
        &self,
        engine: &mut dyn engine::Engine,
        instance: u64,
    ) -> engine::Result<()> {
        self.pump_out_with(engine, instance, None)
    }

    /// As [`ChannelJoin::pump_out`], but an adopted channel's wake is handed
    /// back via `deferred` instead of fired in place — needed when the engine
    /// settles asynchronously, so the frame's completion callback wakes it
    /// once the device is actually done. `deferred: None` wakes in place.
    ///
    /// # Errors
    ///
    /// As [`ChannelJoin::pump_out`].
    pub fn pump_out_with(
        &self,
        engine: &mut dyn engine::Engine,
        instance: u64,
        mut deferred: Option<&mut Vec<u64>>,
    ) -> engine::Result<()> {
        let Some(channels) = self.instances.get(&instance) else {
            return Ok(());
        };
        for (dense, id) in channels.iter().enumerate() {
            let Some(joined) = self.channels.get(id) else {
                continue;
            };
            if joined.host_role != eta_ir::container::HostRole::Reader {
                continue;
            }
            // Adopted channel: as in pump_in, the engine already wrote the
            // committed cell in place; only the wake remains.
            if joined.registered.adopted() {
                match deferred.as_deref_mut() {
                    Some(wakes) => wakes.push(joined.registered.reader_wait_id),
                    None => {
                        waker::WakerTable::global().wake(joined.registered.reader_wait_id);
                    }
                }
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
                let Some(bytes) = engine.take_channel(instance, dense as u32)? else {
                    break;
                };
                if bytes.len() != cell_bytes {
                    return Err(engine::Error::Program(format!(
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
/// endpoint has closed (closes/deregisters `channel_id` on the owning
/// engine). `None` in tests that only exercise wait/poison semantics.
pub type ChannelCloser = Arc<dyn Fn(u64) -> anyhow::Result<()> + Send + Sync>;

pub struct ChannelEndpoint {
    registered: RegisteredChannel,
    closed: AtomicBool,
    /// Whether the engine close notification has been handed to an external
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

    /// Installs the close-notification callback (see [`ChannelCloser`]).
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

    /// Takes over this endpoint's engine close notification: returns the
    /// channel id the caller must now close, or `None` if already closed or
    /// already taken. Caller must outlive-order the endpoint's drop; this
    /// does not synchronize against a concurrent one.
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

