use std::sync::Arc;

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use engine::{ChannelId, InstanceId};

/// A ring the runtime allocates for an engine that maps none.
#[derive(Debug)]
struct OwnedRing {
    words: Box<[AtomicU64]>,
    cells: Box<[AtomicU64]>,
}

impl engine::channel::MirrorMemory for OwnedRing {
    fn words(&self) -> &[AtomicU64] {
        &self.words
    }

    fn cells(&self) -> &[AtomicU8] {
        &[]
    }

    fn cell_words(&self) -> &[AtomicU64] {
        &self.cells
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    pub channel: ChannelId,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct HostRing {
    adopted: bool,
    memory: Arc<dyn engine::channel::MirrorMemory>,
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
        let cells = (0..bytes.div_ceil(8)).map(|_| AtomicU64::new(0)).collect();
        let words: Box<[AtomicU64]> = (0..4)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        HostRing {
            mirror_bytes: bytes as u64,
            memory: Arc::new(OwnedRing { words, cells }),
            adopted: false,
            cell_bytes,
            capacity,
        }
    }

    /// A ring over the engine's mapping, which it holds alive; `None` if the
    /// mapping is short of four words or of `capacity + 1` cells.
    #[must_use]
    pub fn adopt(mirror: engine::HostMirror) -> Option<HostRing> {
        let slots = u64::from(mirror.capacity).saturating_add(1);
        let mirror_bytes = slots * u64::from(mirror.cell_bytes);
        let (words, cells) = (mirror.memory.words(), Cells::of(&*mirror.memory));
        if words.len() < 4 || (cells.len() as u64) < mirror_bytes {
            return None;
        }
        Some(HostRing {
            adopted: true,
            mirror_bytes,
            cell_bytes: mirror.cell_bytes,
            capacity: mirror.capacity,
            memory: mirror.memory,
        })
    }

    #[must_use]
    pub const fn adopted(&self) -> bool {
        self.adopted
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
    pub fn binding(&self, channel_id: ChannelId) -> ChannelBinding {
        ChannelBinding {
            channel_id,
            memory: Arc::clone(&self.memory),
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

#[derive(Debug, Clone)]
pub struct ChannelBinding {
    pub channel_id: ChannelId,
    pub memory: Arc<dyn engine::channel::MirrorMemory>,
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
    /// The engine registered the channel itself: an instance bound to it
    /// reads the engine's ring. `false` where the engine answered
    /// `Unsupported` and the runtime keeps the channel alone; a ring of it
    /// is then the instance's own.
    pub engine_registered: bool,
}

impl RegisteredChannel {
    #[must_use]
    pub fn new(
        engine_id: usize,
        id: ChannelId,
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
        id: ChannelId,
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
            engine_registered: true,
        }
    }

    #[must_use]
    pub fn adopted(&self) -> bool {
        self.ring.adopted()
    }

    #[must_use]
    pub fn id(&self) -> ChannelId {
        self.binding.channel_id
    }
}

impl PartialEq for RegisteredChannel {
    fn eq(&self, other: &RegisteredChannel) -> bool {
        self.engine_id == other.engine_id
            && self.id() == other.id()
            && Arc::ptr_eq(&self.ring, &other.ring)
            && self.reader_wait_id == other.reader_wait_id
            && self.writer_wait_id == other.writer_wait_id
    }
}

impl Eq for RegisteredChannel {}

#[derive(Debug, Default)]
pub struct ChannelJoin {
    channels: std::collections::HashMap<ChannelId, JoinedChannel>,
    instances: std::collections::HashMap<InstanceId, Vec<ChannelId>>,
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
    pub fn contains(&self, id: ChannelId) -> bool {
        self.channels.contains_key(&id)
    }

    /// Whether the engine registered channel `id` itself, so a ring of it
    /// is the channel's and not an instance's own.
    #[must_use]
    pub fn engine_registered(&self, id: ChannelId) -> bool {
        self.channels
            .get(&id)
            .is_some_and(|joined| joined.registered.engine_registered)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = ChannelId> + '_ {
        self.channels.keys().copied()
    }

    #[must_use]
    pub fn into_ids(self) -> Vec<ChannelId> {
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

    pub fn remove(&mut self, id: ChannelId) {
        self.channels.remove(&id);
    }

    pub fn bind(&mut self, instance: InstanceId, channels: Vec<ChannelId>) {
        self.instances.insert(instance, channels);
    }

    pub fn unbind(&mut self, instance: InstanceId) {
        self.instances.remove(&instance);
    }

    pub fn pump_in(
        &self,
        engine: &mut dyn engine::Engine,
        instance: InstanceId,
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
            if joined.registered.adopted() {
                waker::WakerTable::global().wake(joined.registered.writer_wait_id);
                continue;
            }
            let binding = &joined.registered.binding;
            let (words, cells) = (binding.memory.words(), Cells::of(&*binding.memory));
            let cap1 = u64::from(binding.capacity).saturating_add(1);
            let cell_bytes = binding.cell_bytes as usize;
            let mut head = load_word(words, binding.head_word_index);
            let tail = load_word(words, binding.tail_word_index);
            let mut moved = false;
            let mut cell = Vec::new();
            while head < tail {
                let offset = ((head % cap1) * cell_bytes as u64) as usize;
                cell.resize(cell_bytes, 0);
                cells.load(offset, &mut cell);
                let kind = engine.kind();
                let Some(channels) = engine.channels() else {
                    return Err(engine::Error::unsupported(kind, "publish_channel"));
                };
                if !channels.publish(instance, dense as u32, &cell)? {
                    break;
                }
                head += 1;
                store_word(words, binding.head_word_index, head);
                moved = true;
            }
            if moved {
                waker::WakerTable::global().wake(joined.registered.writer_wait_id);
            }
        }
        Ok(())
    }

    pub fn pump_out(
        &self,
        engine: &mut dyn engine::Engine,
        instance: InstanceId,
    ) -> engine::Result<()> {
        self.pump_out_with(engine, instance, None)
    }

    pub fn pump_out_with(
        &self,
        engine: &mut dyn engine::Engine,
        instance: InstanceId,
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
            let binding = &joined.registered.binding;
            let (words, cells) = (binding.memory.words(), Cells::of(&*binding.memory));
            let cap1 = u64::from(binding.capacity).saturating_add(1);
            let cell_bytes = binding.cell_bytes as usize;
            let head = load_word(words, binding.head_word_index);
            let mut tail = load_word(words, binding.tail_word_index);
            let mut moved = false;
            while tail.saturating_sub(head) < u64::from(binding.capacity) {
                let kind = engine.kind();
                let Some(channels) = engine.channels() else {
                    return Err(engine::Error::unsupported(kind, "take_channel"));
                };
                let Some(bytes) = channels.take(instance, dense as u32)? else {
                    break;
                };
                if bytes.len() != cell_bytes {
                    return Err(engine::Error::Program(format!(
                        "channel {id} published a {}-byte cell into a ring of \
                         {cell_bytes}-byte ones",
                        bytes.len()
                    )));
                }
                let offset = ((tail % cap1) * cell_bytes as u64) as usize;
                cells.store(offset, &bytes);
                tail += 1;
                store_word(words, binding.tail_word_index, tail);
                moved = true;
            }
            if moved {
                waker::WakerTable::global().wake(joined.registered.reader_wait_id);
            }
        }
        Ok(())
    }
}

fn load_word(words: &[AtomicU64], index: u32) -> u64 {
    words[index as usize].load(Ordering::Acquire)
}

fn store_word(words: &[AtomicU64], index: u32, value: u64) {
    words[index as usize].store(value, Ordering::Release);
}

/// A ring's cells, a word at a time where its memory offers words: the word
/// that publishes a cell orders the accesses to it.
#[derive(Clone, Copy)]
pub(crate) struct Cells<'a> {
    bytes: &'a [AtomicU8],
    words: &'a [AtomicU64],
}

impl<'a> Cells<'a> {
    pub(crate) fn of(memory: &'a dyn engine::channel::MirrorMemory) -> Cells<'a> {
        Cells {
            bytes: memory.cells(),
            words: memory.cell_words(),
        }
    }

    pub(crate) fn len(self) -> usize {
        self.bytes.len().max(self.words.len() * 8)
    }

    /// The words wholly within `len` bytes at `offset`, if any.
    fn whole(self, offset: usize, len: usize) -> Option<std::ops::Range<usize>> {
        let (first, last) = (
            offset.div_ceil(8),
            ((offset + len) / 8).min(self.words.len()),
        );
        (first < last).then_some(first..last)
    }

    pub(crate) fn load(self, offset: usize, out: &mut [u8]) {
        let Some(whole) = self.whole(offset, out.len()) else {
            return self.load_bytes(offset, out);
        };
        let (head, rest) = out.split_at_mut(whole.start * 8 - offset);
        let (body, tail) = rest.split_at_mut(whole.len() * 8);
        self.load_bytes(offset, head);
        for (chunk, word) in body
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(&self.words[whole.clone()])
        {
            *chunk = word.load(Ordering::Relaxed).to_ne_bytes();
        }
        self.load_bytes(whole.end * 8, tail);
    }

    pub(crate) fn store(self, offset: usize, bytes: &[u8]) {
        let Some(whole) = self.whole(offset, bytes.len()) else {
            return self.store_bytes(offset, bytes);
        };
        let (head, rest) = bytes.split_at(whole.start * 8 - offset);
        let (body, tail) = rest.split_at(whole.len() * 8);
        self.store_bytes(offset, head);
        for (chunk, word) in body
            .as_chunks::<8>()
            .0
            .iter()
            .zip(&self.words[whole.clone()])
        {
            word.store(u64::from_ne_bytes(*chunk), Ordering::Relaxed);
        }
        self.store_bytes(whole.end * 8, tail);
    }

    fn load_bytes(self, at: usize, out: &mut [u8]) {
        for (at, byte) in (at..).zip(out) {
            *byte = match self.bytes.get(at) {
                Some(cell) => cell.load(Ordering::Relaxed),
                None => self.words[at / 8].load(Ordering::Relaxed).to_ne_bytes()[at % 8],
            };
        }
    }

    /// Bytes short of a word. Memory that offers words alone only the host
    /// touches, and a compare-and-swap keeps the word's other bytes.
    fn store_bytes(self, at: usize, bytes: &[u8]) {
        for (at, &byte) in (at..).zip(bytes) {
            match self.bytes.get(at) {
                Some(cell) => cell.store(byte, Ordering::Relaxed),
                None => {
                    self.words[at / 8].update(Ordering::Relaxed, Ordering::Relaxed, |word| {
                        let mut word = word.to_ne_bytes();
                        word[at % 8] = byte;
                        u64::from_ne_bytes(word)
                    });
                }
            }
        }
    }
}

pub type ChannelCloser = Arc<dyn Fn(ChannelId) -> anyhow::Result<()> + Send + Sync>;

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
        let binding = &self.registered.binding;
        let words = binding.memory.words();
        waker::WaitFuture::new(waker::WakerTable::global(), wait_id, move || {
            let poison = load_word(words, binding.poison_word_index);
            if poison != 0 {
                return waker::Readiness::Ready(Err(ChannelWaitError::Poisoned(poison)));
            }
            if load_word(words, binding.closed_word_index) != 0 {
                return waker::Readiness::Ready(Err(ChannelWaitError::Closed));
            }
            let current = load_word(words, word_index);
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

    pub fn detach_close_notification(&self) -> Option<ChannelId> {
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
                channel_id = self.registered.binding.channel_id.0,
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

    #[test]
    fn a_cell_reads_back_at_any_offset_and_leaves_its_neighbours() {
        let ring = HostRing::new(1, 63);
        let cells = Cells::of(&*ring.memory);
        for (offset, len) in [(0, 64), (3, 21), (8, 16), (13, 2), (60, 4)] {
            cells.store(0, &[0xff; 64]);
            let bytes: Vec<u8> = (1..=len as u8).collect();
            cells.store(offset, &bytes);
            let mut want = [0xff; 64];
            want[offset..offset + len].copy_from_slice(&bytes);
            let mut got = [0; 64];
            cells.load(0, &mut got);
            assert_eq!(got, want);
            let mut cell = vec![0; len];
            cells.load(offset, &mut cell);
            assert_eq!(cell, bytes);
        }
    }
}
