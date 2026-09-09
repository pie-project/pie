use std::sync::Arc;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

type OwnedRing = (Box<[u8]>, Box<[AtomicU64]>);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    pub channel: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct HostRing {
    owned: Option<OwnedRing>,
    mirror_base: u64,
    word_base: u64,
    mirror_bytes: u64,
    cell_bytes: u32,
    capacity: u32,
}

const HEAD_WORD: u32 = 0;
const TAIL_WORD: u32 = 1;
const POISON_WORD: u32 = 2;
const CLOSED_WORD: u32 = 3;

impl HostRing {
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

    /// # Safety
    ///
    /// `mirror` and `words` must point at a live mapping of at least
    /// `capacity + 1` cells of `cell_bytes` that outlives the returned ring.
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

    #[must_use]
    pub const fn adopted(&self) -> bool {
        self.owned.is_none()
    }

    #[must_use]
    pub fn wire_cell_bytes(dtype: eta_ir::types::Dtype, numel: usize) -> usize {
        if dtype == eta_ir::types::Dtype::Bool {
            numel.div_ceil(8)
        } else {
            numel.saturating_mul(4)
        }
    }

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChannelBinding {
    pub channel_id: u64,
    pub mirror_base: u64,
    pub word_base: u64,
    pub mirror_bytes: u64,
    pub word_bytes: u64,
    pub cell_bytes: u32,
    pub capacity: u32,
    pub head_word_index: u32,
    pub tail_word_index: u32,
    pub poison_word_index: u32,
    pub closed_word_index: u32,
}

#[derive(Debug, Clone)]
pub struct RegisteredChannel {
    pub engine_id: usize,
    pub binding: ChannelBinding,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    pub ring: Arc<HostRing>,
}

impl RegisteredChannel {
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

    #[must_use]
    pub fn adopted(&self) -> bool {
        self.ring.adopted()
    }

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

#[derive(Debug, Default)]
pub struct ChannelJoin {
    channels: std::collections::HashMap<u64, JoinedChannel>,
    instances: std::collections::HashMap<u64, Vec<u64>>,
}

#[derive(Debug, Clone)]
struct JoinedChannel {
    registered: RegisteredChannel,
    host_role: eta_ir::container::HostRole,
}

impl ChannelJoin {
    #[must_use]
    pub fn new() -> ChannelJoin {
        ChannelJoin::default()
    }

    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.channels.contains_key(&id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.channels.keys().copied()
    }

    #[must_use]
    pub fn into_ids(self) -> Vec<u64> {
        self.channels.into_keys().collect()
    }

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

    pub fn remove(&mut self, id: u64) {
        self.channels.remove(&id);
    }

    pub fn bind(&mut self, instance: u64, channels: Vec<u64>) {
        self.instances.insert(instance, channels);
    }

    pub fn unbind(&mut self, instance: u64) {
        self.instances.remove(&instance);
    }

    pub fn pump_in(&self, engine: &mut dyn engine::Engine, instance: u64) -> engine::Result<()> {
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
                    break;
                }
                head += 1;
                store_word(binding.word_base, binding.head_word_index, head);
                moved = true;
            }
            if moved {
                waker::WakerTable::global().wake(joined.registered.writer_wait_id);
            }
        }
        Ok(())
    }

    pub fn pump_out(&self, engine: &mut dyn engine::Engine, instance: u64) -> engine::Result<()> {
        self.pump_out_with(engine, instance, None)
    }

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

fn load_word(word_base: u64, index: u32) -> u64 {
    // SAFETY: `index` is one of the four words `HostRing` allocates and the
    // ring outlives every reader of this binding.
    unsafe { (*(word_base as *const AtomicU64).add(index as usize)).load(Ordering::Acquire) }
}

fn store_word(word_base: u64, index: u32, value: u64) {
    // SAFETY: as `load_word`; the SPSC discipline makes this word ours.
    unsafe {
        (*(word_base as *const AtomicU64).add(index as usize)).store(value, Ordering::Release);
    }
}

pub type ChannelCloser = Arc<dyn Fn(u64) -> anyhow::Result<()> + Send + Sync>;

pub struct ChannelEndpoint {
    registered: RegisteredChannel,
    closed: AtomicBool,
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
