//! Host channel cells: the host endpoint of a guest-constructed channel (Writer = host-puts/pass-consumes, Reader = pass-puts/host-takes; cells are dtype-native, only the wire packs bool to bits).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::{ChannelBinding, ChannelEndpoint};
use eta_ir::container::{self, ChanDType, ChannelDecl, ExternDir, HostRole};
use eta_ir::types::Dtype;

/// Sentinel: a run-ahead ticket that neither consumes nor publishes.
pub const TICKET_NONE: u64 = u64::MAX;

/// Process-wide monotonic channel id source (0 is a null sentinel). A
/// channel keeps its id across every pass it binds into.
static NEXT_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);

/// Mint the next process-wide global channel identity.
pub fn next_channel_id() -> u64 {
    NEXT_CHANNEL_ID.fetch_add(1, Ordering::Relaxed)
}

/// The shared host state behind one guest `channel` resource.
#[derive(Clone, Debug)]
pub struct ChannelCell {
    /// Engine device channel-registry key, stable across every bound pass.
    pub global_id: u64,
    /// Declared dims; checked against the container decl at bind.
    pub shape: Vec<u32>,
    pub dtype: Dtype,
    pub capacity: u32,
    /// Stamped at bind; `None` = not yet bound to a forward pass.
    pub role: Option<HostRole>,
    pub seeded: bool,
    /// Whether this cell's seed was consumed by a first fire.
    pub seed_taken: bool,
    endpoint: Option<Arc<ChannelEndpoint>>,
    declared_dtype: Option<ChanDType>,
    extern_name: Option<String>,
    attachments: Vec<ChannelAttachment>,
    /// Host-staged cells (seeds pre-first-fire, Writer cells otherwise), FIFO.
    staged: VecDeque<Vec<u8>>,
    /// Host copies of Writer ring entries not yet claimed by a submitted
    /// fire; the runtime's only record of post-bind Writer puts.
    ring_host_copies: VecDeque<Vec<u8>>,
    writer_tail: u64,
    /// Device-produced cells awaiting host `take`/`read`, FIFO.
    produced: VecDeque<Vec<u8>>,
    /// Device-ring sequences assigned to submitted fires; immutable tickets.
    device_reserved_head: u64,
    device_reserved_tail: u64,
    /// Engine-owned mirror for every pass bound to this Reader channel.
    reader: Option<ReaderMirror>,
    /// `Some(reason)` once a fire feeding this channel failed; every later
    /// `take`/`read` errors with it.
    poisoned: Option<String>,
    /// Host replacement for the current committed front, so `set` changes
    /// the standing cell without displacing a value queued for next fire.
    front_override: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct ReaderMirror {
    mirror_base: u64,
    word_base: u64,
    cell_bytes: usize,
    cap1: u64,
    head_word_index: usize,
    tail_word_index: usize,
    poison_word_index: usize,
    closed_word_index: usize,
    /// Sequences already copied out of the mirror (reader-side cursor).
    copied_tail: u64,
}

#[derive(Clone, Debug)]
struct ChannelAttachment {
    instance_id: u64,
    extern_dir: Option<ExternDir>,
}

/// A channel host-op failure (surfaced to the guest as a WIT `result` error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelError {
    WrongRole { role: HostRole, op: &'static str },
    Empty,
    Poisoned(String),
    BadLength { expected: usize, got: usize },
    MissingSeed,
    /// A second `put` on a seeded non-Writer channel before its first fire.
    SeedAlreadyStaged,
    Full,
    /// The committed front is currently claimed by a submitted fire.
    InFlight,
    Closed,
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ChannelError::*;
        match self {
            WrongRole { role, op } => write!(f, "{op} is illegal on a {role:?} channel"),
            Empty => write!(f, "no cell available"),
            Poisoned(reason) => write!(f, "channel is poisoned: {reason}"),
            BadLength { expected, got } => {
                write!(f, "{got} bytes, expected {expected} (shape×dtype)")
            }
            MissingSeed => write!(f, "seeded but no seed was put before the first fire"),
            SeedAlreadyStaged => write!(f, "a seed is already staged (a seed is exactly one put)"),
            Full => write!(f, "channel is full"),
            InFlight => write!(f, "channel front is in use by an in-flight fire"),
            Closed => write!(f, "channel is closed"),
        }
    }
}
impl std::error::Error for ChannelError {}

impl ChannelCell {
    /// A fresh, unbound cell (the guest `channel` constructor).
    pub fn new(shape: Vec<u32>, dtype: Dtype, capacity: u32) -> Self {
        ChannelCell {
            global_id: next_channel_id(),
            shape,
            dtype,
            capacity,
            role: None,
            seeded: false,
            seed_taken: false,
            endpoint: None,
            declared_dtype: None,
            extern_name: None,
            attachments: Vec::new(),
            staged: VecDeque::new(),
            ring_host_copies: VecDeque::new(),
            writer_tail: 0,
            produced: VecDeque::new(),
            device_reserved_head: 0,
            device_reserved_tail: 0,
            reader: None,
            poisoned: None,
            front_override: None,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Native (unpacked) bytes per cell: `numel` for bool, `numel*4` otherwise.
    pub fn native_len(&self) -> usize {
        self.numel() * container::const_elem_size(self.dtype)
    }

    /// Whether the cell's constructor-declared geometry matches a container
    /// channel declaration (bind-time validation).
    #[cfg(test)]
    pub fn matches_decl(&self, decl: &ChannelDecl) -> Result<(), String> {
        self.validate_attachment(decl, None)
    }

    pub fn validate_attachment(
        &self,
        decl: &ChannelDecl,
        extern_binding: Option<(&str, ExternDir)>,
    ) -> Result<(), String> {
        let decl_dims = decl.shape.dims().to_vec();
        if self.shape != decl_dims {
            return Err(format!(
                "shape {:?} != declared {:?}",
                self.shape, decl_dims
            ));
        }
        let decl_dtype = decl.dtype.program_dtype();
        if self.dtype != decl_dtype {
            return Err(format!("dtype {:?} != declared {decl_dtype:?}", self.dtype));
        }
        if self.capacity != decl.capacity {
            return Err(format!(
                "capacity {} != declared {}",
                self.capacity, decl.capacity
            ));
        }
        if let Some(dtype) = self.declared_dtype
            && dtype != decl.dtype
        {
            return Err(format!(
                "declared dtype {:?} conflicts with prior {dtype:?}",
                decl.dtype
            ));
        }
        if !self.attachments.is_empty() {
            let Some((name, dir)) = extern_binding else {
                // Same-guest cross-pass chaining: a device-only channel (no
                // host role, never seeded) may attach to multiple passes.
                // Host-visible or seeded channels keep the one-pass rule.
                if decl.host_role != HostRole::None
                    || decl.seeded
                    || self.role != Some(HostRole::None)
                    || self.seeded
                {
                    return Err(
                        "a host-visible or seeded channel may attach to only one pass".into(),
                    );
                }
                return Ok(());
            };
            if decl.host_role != HostRole::None || decl.seeded {
                return Err("shared extern channels cannot have a host role or seed".into());
            }
            if self.extern_name.as_deref() != Some(name) {
                return Err(format!(
                    "extern binding name {name:?} conflicts with prior {:?}",
                    self.extern_name
                ));
            }
            if self
                .attachments
                .iter()
                .any(|attachment| attachment.extern_dir == Some(dir))
            {
                return Err(format!("extern {dir:?} endpoint is already claimed"));
            }
        } else if extern_binding.is_some() && (decl.host_role != HostRole::None || decl.seeded) {
            return Err("extern channels cannot have a host role or seed".into());
        }
        Ok(())
    }

    pub fn bind(&mut self, decl: &ChannelDecl) {
        if self.attachments.is_empty() {
            if let Some(endpoint) = &self.endpoint {
                let binding = endpoint.registered().binding;
                self.device_reserved_head =
                    load_word(binding.word_base, binding.head_word_index as usize);
                self.device_reserved_tail =
                    load_word(binding.word_base, binding.tail_word_index as usize);
                if decl.host_role == HostRole::Writer {
                    self.writer_tail = self.writer_tail.max(self.device_reserved_tail);
                }
            } else {
                self.device_reserved_head = 0;
                self.device_reserved_tail = u64::from(decl.seeded);
            }
            if decl.seeded && decl.host_role == HostRole::Writer {
                self.writer_tail = self.writer_tail.max(1);
            }
        }
        self.role = Some(decl.host_role);
        self.seeded = decl.seeded;
        self.declared_dtype = Some(decl.dtype);
    }

    pub fn attach(
        &mut self,
        instance_id: u64,
        decl: &ChannelDecl,
        extern_binding: Option<(&str, ExternDir)>,
    ) -> Result<(), String> {
        self.validate_attachment(decl, extern_binding)?;
        self.bind(decl);
        if let Some((name, _)) = extern_binding {
            self.extern_name = Some(name.to_string());
        }
        self.attachments.push(ChannelAttachment {
            instance_id,
            extern_dir: extern_binding.map(|(_, dir)| dir),
        });
        Ok(())
    }

    pub fn detach(&mut self, instance_id: u64) {
        self.attachments
            .retain(|attachment| attachment.instance_id != instance_id);
    }

    pub fn endpoint(&self) -> Option<Arc<ChannelEndpoint>> {
        self.endpoint.clone()
    }

    pub fn reserve_device_ticket(&mut self, consume: bool, publish: bool) -> (u64, u64) {
        let expected_head = if consume {
            let expected = self.device_reserved_head;
            self.device_reserved_head += 1;
            expected
        } else {
            TICKET_NONE
        };
        let expected_tail = if publish {
            let expected = self.device_reserved_tail;
            self.device_reserved_tail += 1;
            expected
        } else {
            TICKET_NONE
        };
        (expected_head, expected_tail)
    }

    pub fn rollback_device_ticket(&mut self, expected_head: u64, expected_tail: u64) -> bool {
        let mut complete = true;
        if expected_tail != TICKET_NONE {
            if self.device_reserved_tail == expected_tail + 1 {
                self.device_reserved_tail = expected_tail;
            } else {
                complete = false;
            }
        }
        if expected_head != TICKET_NONE {
            if self.device_reserved_head == expected_head + 1 {
                self.device_reserved_head = expected_head;
            } else {
                complete = false;
            }
        }
        complete
    }

    pub fn reader_wait_state(&self) -> Option<(Arc<ChannelEndpoint>, u64)> {
        Some((self.endpoint.clone()?, self.reader.as_ref()?.copied_tail))
    }

    pub fn writer_wait_state(&self) -> Option<(Arc<ChannelEndpoint>, u64)> {
        let endpoint = self.endpoint.clone()?;
        let binding = endpoint.registered().binding;
        Some((
            endpoint,
            load_word(binding.word_base, binding.head_word_index as usize),
        ))
    }

    pub fn attach_endpoint(&mut self, endpoint: Arc<ChannelEndpoint>) -> Result<(), String> {
        let binding = endpoint.registered().binding;
        if binding.channel_id != self.global_id {
            return Err(format!(
                "channel {} received endpoint {}",
                self.global_id, binding.channel_id
            ));
        }
        if let Some(existing) = &self.endpoint {
            if !Arc::ptr_eq(existing, &endpoint) {
                return Err(format!(
                    "channel {} endpoint already registered",
                    self.global_id
                ));
            }
            return Ok(());
        }
        if self.role == Some(HostRole::Reader) {
            self.attach_reader_mirror(
                0,
                binding.mirror_base,
                binding.word_base,
                binding.cell_bytes,
                binding.capacity,
                0,
                binding.head_word_index,
                binding.tail_word_index,
                binding.poison_word_index,
                binding.closed_word_index,
            )?;
        }
        self.endpoint = Some(endpoint);
        // Flush any pre-endpoint staged Writer cells into the shared ring
        // (a seeded Writer flushes only after its seed settles).
        if self.role == Some(HostRole::Writer)
            && let Err(error) = self.flush_writer_staging()
        {
            return Err(format!(
                "channel {}: staging flush: {error}",
                self.global_id
            ));
        }
        Ok(())
    }

    /// Host `put` a dtype-native cell. Pre-bind this stages freely; post-bind
    /// it must be a Writer stage cell or the one seed on a not-yet-fired
    /// `seeded` channel. Once the Writer endpoint exists, a put writes
    /// directly into the pinned ring cell and release-publishes the tail word.
    pub fn put(&mut self, native: Vec<u8>) -> Result<(), ChannelError> {
        self.put_ref(&native)
    }

    pub fn put_ref(&mut self, native: &[u8]) -> Result<(), ChannelError> {
        let expected = self.native_len();
        if native.len() != expected {
            return Err(ChannelError::BadLength {
                expected,
                got: native.len(),
            });
        }
        match self.role {
            None | Some(HostRole::Writer) => {}
            Some(role) => {
                if !(self.seeded && !self.seed_taken) {
                    return Err(ChannelError::WrongRole { role, op: "put" });
                }
                if !self.staged.is_empty() {
                    return Err(ChannelError::SeedAlreadyStaged);
                }
            }
        }
        if self.role == Some(HostRole::Writer)
            && !(self.seeded && !self.seed_taken)
            && let Some(endpoint) = self.endpoint.clone()
        {
            debug_assert!(
                self.staged.is_empty(),
                "writer staging must flush when the endpoint attaches"
            );
            return self.write_writer_ring(endpoint.registered().binding, native);
        }
        let consumed = self
            .endpoint
            .as_ref()
            .map(|endpoint| {
                let binding = endpoint.registered().binding;
                load_word(binding.word_base, binding.head_word_index as usize)
            })
            .unwrap_or(0);
        let in_flight = self.writer_tail.saturating_sub(consumed);
        if in_flight.saturating_add(self.staged.len() as u64) >= u64::from(self.capacity) {
            return Err(ChannelError::Full);
        }
        self.staged.push_back(native.to_vec());
        Ok(())
    }

    /// Atomically replace the committed front cell. Queue cursors and
    /// occupancy are unchanged. A front already claimed by a fire is
    /// immutable until that fire advances the device head. A front already
    /// delivered (a seed, which crossed at bind and left this ring empty) is
    /// replaced in `front_override` instead of in the ring.
    pub fn set(&mut self, native: Vec<u8>) -> Result<(), ChannelError> {
        self.set_ref(&native)
    }

    pub fn set_ref(&mut self, native: &[u8]) -> Result<(), ChannelError> {
        let expected = self.native_len();
        if native.len() != expected {
            return Err(ChannelError::BadLength {
                expected,
                got: native.len(),
            });
        }
        if let Some(reason) = &self.poisoned {
            return Err(ChannelError::Poisoned(reason.clone()));
        }

        if let Some(endpoint) = self.endpoint.clone() {
            let binding = endpoint.registered().binding;
            let poison = load_word(binding.word_base, binding.poison_word_index as usize);
            if poison != 0 {
                return Err(ChannelError::Poisoned(format!(
                    "engine published poison epoch {poison}"
                )));
            }
            if load_word(binding.word_base, binding.closed_word_index as usize) != 0 {
                return Err(ChannelError::Closed);
            }

            // Pull visible Reader cells into the host queue before replacing
            // its front copy.
            if self.role == Some(HostRole::Reader) {
                self.refresh_reader_mirrors()?;
            }
            let head = load_word(binding.word_base, binding.head_word_index as usize);
            let tail = load_word(binding.word_base, binding.tail_word_index as usize);
            let committed_tail = if self.role == Some(HostRole::Writer) {
                tail.saturating_sub(self.ring_host_copies.len() as u64)
            } else {
                tail
            };
            if committed_tail <= head {
                // A delivered seed rode `InstanceBinding::seeds` straight
                // into the shell's ring, so `committed_tail` is 0 even
                // though the seed is the front for the whole run. It can't
                // be rewritten in the ring (already full at capacity 1), so
                // the replacement is recorded in `front_override` instead.
                if self.role == Some(HostRole::Writer)
                    && self.seeded
                    && self.seed_taken
                    && head == tail
                    && self.ring_host_copies.is_empty()
                {
                    // A submitted fire that already claimed the seed owns it.
                    if self.device_reserved_head >= self.writer_tail {
                        return Err(ChannelError::InFlight);
                    }
                    self.front_override = Some(native.to_vec());
                    return Ok(());
                }
                return Err(ChannelError::Empty);
            }
            if self.device_reserved_head > head {
                return Err(ChannelError::InFlight);
            }

            self.replace_ring_cell(binding, head, native)?;
            if self.role == Some(HostRole::Reader) {
                let front = self.produced.front_mut().ok_or(ChannelError::Empty)?;
                *front = native.to_vec();
            }
            self.front_override = Some(native.to_vec());
            return Ok(());
        }

        Err(ChannelError::Empty)
    }

    fn replace_ring_cell(
        &self,
        binding: ChannelBinding,
        sequence: u64,
        native: &[u8],
    ) -> Result<(), ChannelError> {
        let cell_bytes = binding.cell_bytes as usize;
        let wire_len = if self.dtype == Dtype::Bool {
            native.len().div_ceil(8)
        } else {
            native.len()
        };
        if wire_len != cell_bytes {
            return Err(ChannelError::BadLength {
                expected: cell_bytes,
                got: wire_len,
            });
        }
        let cap1 = u64::from(binding.capacity).saturating_add(1);
        let offset = (sequence % cap1) * cell_bytes as u64;
        let cell = unsafe {
            std::slice::from_raw_parts_mut((binding.mirror_base + offset) as *mut u8, cell_bytes)
        };
        if self.dtype == Dtype::Bool {
            pack_bool_into(native, cell);
        } else {
            cell.copy_from_slice(native);
        }
        // Re-publish the unchanged tail so the replacement happens-before
        // the next consumer's acquire.
        let tail = load_word(binding.word_base, binding.tail_word_index as usize);
        store_word(binding.word_base, binding.tail_word_index as usize, tail);
        Ok(())
    }

    pub fn front_override(&self) -> Option<Vec<u8>> {
        self.front_override.clone()
    }

    pub fn consume_front_override(&mut self) {
        self.front_override = None;
    }

    /// Write one cell into the engine-shared Writer ring at `tail % cap1`,
    /// then release-publish the incremented tail word. The spare `+1` ring
    /// cell distinguishes full from empty.
    fn write_writer_ring(
        &mut self,
        binding: ChannelBinding,
        native: &[u8],
    ) -> Result<(), ChannelError> {
        let poison = load_word(binding.word_base, binding.poison_word_index as usize);
        if poison != 0 {
            return Err(ChannelError::Poisoned(format!(
                "engine published poison epoch {poison}"
            )));
        }
        if load_word(binding.word_base, binding.closed_word_index as usize) != 0 {
            return Err(ChannelError::Closed);
        }
        let head = load_word(binding.word_base, binding.head_word_index as usize);
        if self.writer_tail.saturating_sub(head) >= u64::from(self.capacity) {
            return Err(ChannelError::Full);
        }
        let cell_bytes = binding.cell_bytes as usize;
        let wire_len = if self.dtype == Dtype::Bool {
            native.len().div_ceil(8)
        } else {
            native.len()
        };
        if wire_len != cell_bytes {
            return Err(ChannelError::BadLength {
                expected: cell_bytes,
                got: wire_len,
            });
        }
        let cap1 = u64::from(binding.capacity).saturating_add(1);
        let offset = (self.writer_tail % cap1) * cell_bytes as u64;
        // SAFETY: mirror validated at registration (mirror_bytes >=
        // cell_bytes * cap1) and stays alive until channel close; SPSC
        // discipline makes this cell ours.
        let cell = unsafe {
            std::slice::from_raw_parts_mut((binding.mirror_base + offset) as *mut u8, cell_bytes)
        };
        if self.dtype == Dtype::Bool {
            pack_bool_into(native, cell);
        } else {
            cell.copy_from_slice(native);
        }
        self.writer_tail += 1;
        store_word(
            binding.word_base,
            binding.tail_word_index as usize,
            self.writer_tail,
        );
        self.ring_host_copies.push_back(native.to_vec());
        Ok(())
    }

    /// Flush pre-endpoint staged Writer cells into the shared ring, FIFO.
    pub fn flush_writer_staging(&mut self) -> Result<(), ChannelError> {
        if self.role != Some(HostRole::Writer) || (self.seeded && !self.seed_taken) {
            return Ok(());
        }
        let Some(endpoint) = self.endpoint.clone() else {
            return Ok(());
        };
        let binding = endpoint.registered().binding;
        while let Some(native) = self.staged.pop_front() {
            if let Err(error) = self.write_writer_ring(binding, &native) {
                self.staged.push_front(native);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Number of host-staged cells.
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Frame validation: host-known cells a Writer channel can still feed to
    /// future fires.
    pub fn writer_available_cells(&self) -> u64 {
        self.writer_tail
            .saturating_add(self.staged.len() as u64)
            .saturating_sub(self.device_reserved_head)
    }

    /// Frame validation: the Reader ring's (reserved publications, consumed)
    /// pressure pair.
    pub fn reader_ring_pressure(&self) -> (u64, u64) {
        let consumed = self
            .reader
            .as_ref()
            .map(|reader| load_word(reader.word_base, reader.head_word_index))
            .unwrap_or(0);
        (self.device_reserved_tail, consumed)
    }

    /// Frame validation: a device-only ring's structural backlog (reserved
    /// publish tickets minus reserved consume tickets).
    pub fn device_ring_backlog(&self) -> u64 {
        self.device_reserved_tail
            .saturating_sub(self.device_reserved_head)
    }

    /// Frame validation: whether the host side knows a committed value
    /// exists for a latest-value (read-only-bound) channel.
    pub fn has_committed_front(&self) -> bool {
        self.seeded
            || !self.staged.is_empty()
            || !self.ring_host_copies.is_empty()
            || self.front_override.is_some()
            || self.device_reserved_tail > 0
    }

    /// Host `take` a produced cell (Reader), FIFO.
    pub fn take(&mut self) -> Result<Vec<u8>, ChannelError> {
        self.refresh_reader_mirrors()?;
        if let Some(reason) = &self.poisoned {
            return Err(ChannelError::Poisoned(reason.clone()));
        }
        if let Some(role) = self.role
            && role != HostRole::Reader
        {
            return Err(ChannelError::WrongRole { role, op: "take" });
        }
        let value = self.produced.pop_front().ok_or(ChannelError::Empty)?;
        self.front_override = None;
        if let Some(reader) = &self.reader {
            let head = load_word(reader.word_base, reader.head_word_index);
            store_word(reader.word_base, reader.head_word_index, head + 1);
        }
        Ok(value)
    }

    /// Host `read` (peek, non-consuming) a produced cell (Reader).
    pub fn read(&mut self) -> Result<Vec<u8>, ChannelError> {
        self.refresh_reader_mirrors()?;
        if let Some(reason) = &self.poisoned {
            return Err(ChannelError::Poisoned(reason.clone()));
        }
        if let Some(role) = self.role
            && role != HostRole::Reader
        {
            return Err(ChannelError::WrongRole { role, op: "read" });
        }
        self.produced.front().cloned().ok_or(ChannelError::Empty)
    }

    /// Poison the cell with the failed fire's error. First poison wins.
    pub fn poison(&mut self, reason: &str) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason.to_string());
        }
    }

    /// Pop this `seeded` channel's staged seed for the first fire. Errors if
    /// nothing was staged.
    #[cfg(test)]
    pub fn take_seed(&mut self) -> Result<Vec<u8>, ChannelError> {
        let seed = self.staged.pop_front().ok_or(ChannelError::MissingSeed)?;
        self.seed_taken = true;
        Ok(seed)
    }

    pub fn peek_seed(&self) -> Result<Vec<u8>, ChannelError> {
        self.staged
            .front()
            .cloned()
            .ok_or(ChannelError::MissingSeed)
    }

    /// The seed has landed in the engine. Drop the host copy and reconcile
    /// this cell's ring: a seed rides `InstanceBinding::seeds` straight into
    /// the shell's ring (never through this host ring), so this ring's words
    /// are still zero at bind even though `bind` already charged
    /// `writer_tail`. Unreconciled, the next put would see
    /// `writer_tail - head >= capacity` and answer `Full` forever; this sets
    /// `head == tail == writer_tail`. No-op for an *adopted* ring, where the
    /// engine already wrote the seed at bind and owns the head cursor.
    pub fn commit_seed(&mut self) {
        let _ = self.staged.pop_front();
        self.seed_taken = true;
        if self.role != Some(HostRole::Writer) {
            return;
        }
        let Some(endpoint) = self.endpoint.clone() else {
            return;
        };
        if endpoint.registered().adopted() {
            return;
        }
        let binding = endpoint.registered().binding;
        // Only when the ring is untouched, so this never rewinds cursors a
        // fire has already moved.
        if load_word(binding.word_base, binding.tail_word_index as usize) == 0
            && load_word(binding.word_base, binding.head_word_index as usize) == 0
        {
            store_word(
                binding.word_base,
                binding.tail_word_index as usize,
                self.writer_tail,
            );
            store_word(
                binding.word_base,
                binding.head_word_index as usize,
                self.writer_tail,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_reader_mirror(
        &mut self,
        _instance_id: u64,
        mirror_base: u64,
        word_base: u64,
        cell_bytes: u32,
        capacity: u32,
        _mirror_offset: u64,
        head_word_index: u32,
        tail_word_index: u32,
        poison_word_index: u32,
        closed_word_index: u32,
    ) -> Result<(), String> {
        if self.role != Some(HostRole::Reader) {
            return Err(format!(
                "channel {}: engine mirror bound to {:?}, expected Reader",
                self.global_id, self.role
            ));
        }
        if mirror_base == 0 || word_base == 0 || cell_bytes == 0 {
            return Err(format!(
                "channel {}: engine returned an invalid mirror binding",
                self.global_id
            ));
        }
        if self.reader.is_some() {
            return Err(format!(
                "channel {}: endpoint mirror already attached",
                self.global_id
            ));
        }
        let native_len = self.native_len();
        let packed_bool_len = self.numel().div_ceil(8);
        let cell_bytes = cell_bytes as usize;
        if cell_bytes != native_len && !(self.dtype == Dtype::Bool && cell_bytes == packed_bool_len)
        {
            return Err(format!(
                "channel {}: mirror cell has {cell_bytes} bytes, expected {native_len}",
                self.global_id
            ));
        }
        self.reader = Some(ReaderMirror {
            mirror_base,
            word_base,
            cell_bytes,
            cap1: u64::from(capacity).saturating_add(1),
            head_word_index: head_word_index as usize,
            tail_word_index: tail_word_index as usize,
            poison_word_index: poison_word_index as usize,
            closed_word_index: closed_word_index as usize,
            copied_tail: 0,
        });
        Ok(())
    }

    /// Peek the most recently release-published mirror cell without
    /// touching the take cursor.
    pub fn latest_reader_value(
        &mut self,
        _instance_id: u64,
    ) -> Result<Option<Vec<u8>>, ChannelError> {
        let dtype = self.dtype;
        let numel = self.numel();
        let native_len = self.native_len();
        let Some(reader) = self.reader.as_ref() else {
            return Ok(None);
        };
        let tail = load_word(reader.word_base, reader.tail_word_index);
        if tail == 0 {
            return Ok(None);
        }
        let wire = read_mirror_cell(reader, tail - 1);
        decode_reader_cell(dtype, numel, native_len, wire).map(Some)
    }

    fn refresh_reader_mirrors(&mut self) -> Result<(), ChannelError> {
        let dtype = self.dtype;
        let numel = self.numel();
        let native_len = self.native_len();
        let mut poison_reason = None;
        let mut closed = false;
        let mut visible_tail = 0;
        if let Some(reader) = self.reader.as_ref() {
            let poison = load_word(reader.word_base, reader.poison_word_index);
            if poison != 0 {
                poison_reason = Some(format!("engine published poison epoch {poison}"));
            } else if load_word(reader.word_base, reader.closed_word_index) != 0 {
                closed = true;
            } else {
                let tail = load_word(reader.word_base, reader.tail_word_index);
                if tail.saturating_sub(reader.copied_tail) >= reader.cap1 {
                    poison_reason = Some(format!(
                        "channel mirror overrun (tail {tail}, copied {}, capacity {})",
                        reader.copied_tail, reader.cap1
                    ));
                } else if tail > reader.copied_tail {
                    visible_tail = tail;
                }
            }
        }
        if let Some(reason) = poison_reason {
            self.poison(&reason);
        }
        let (reader, produced) = (&mut self.reader, &mut self.produced);
        if let Some(reader) = reader {
            while reader.copied_tail < visible_tail {
                let wire = read_mirror_cell(reader, reader.copied_tail);
                produced.push_back(decode_reader_cell(dtype, numel, native_len, wire)?);
                reader.copied_tail += 1;
            }
        }
        if closed && self.produced.is_empty() && self.poisoned.is_none() {
            return Err(ChannelError::Closed);
        }
        Ok(())
    }
}

/// A forward pass's bound cells, dense declaration order (`cells[i]` backs the
/// container's channel `i`).
pub type BoundCells = Vec<Arc<Mutex<ChannelCell>>>;

/// A first-class, guest-constructed channel — the WIT
/// `pie:inferlet/forward.channel` resource. The shared [`ChannelCell`] is
/// Arc'd so a pass that bound it survives the guest dropping the handle.
pub struct Channel {
    pub cell: Arc<Mutex<ChannelCell>>,
    /// Set at submit: the feeding pipeline's in-flight fire queue.
    /// `None` until first submit.
    pub fires: Option<crate::pipeline::fire::PendingFires>,
}

/// Process-teardown close batching: walks the process's resource table,
/// takes over the engine close notification from every guest channel
/// endpoint still holding one, and returns the channel ids grouped by
/// owning engine. The caller posts one batched close per engine, preserving
/// the engine's instance-before-channel close order.
pub fn detach_channel_close_notifications(
    resources: &mut wasmtime::component::ResourceTable,
) -> Vec<(usize, Vec<u64>)> {
    let mut by_engine: std::collections::BTreeMap<usize, Vec<u64>> =
        std::collections::BTreeMap::new();
    for entry in resources.iter_mut() {
        let Some(channel) = entry.downcast_ref::<Channel>() else {
            continue;
        };
        let Some(endpoint) = channel.cell.lock().unwrap().endpoint() else {
            continue;
        };
        let Some(channel_id) = endpoint.detach_close_notification() else {
            continue;
        };
        by_engine
            .entry(endpoint.registered().engine_id)
            .or_default()
            .push(channel_id);
    }
    by_engine.into_iter().collect()
}

/// The next host-known Writer value on `cell` — the native value the engine
/// will pull for the next submitted fire (`None`: not a Writer channel, or
/// nothing pending).
pub fn staged_put_bytes(cell: &Arc<Mutex<ChannelCell>>) -> Option<Vec<u8>> {
    let c = cell.lock().unwrap();
    if c.role != Some(HostRole::Writer) {
        return None;
    }
    c.staged
        .front()
        .cloned()
        .or_else(|| c.ring_host_copies.front().cloned())
}

/// A submitted fire consumed one Writer entry: drop the ring host copy
/// backing [`staged_put_bytes`]'s front so the next fire sees the next value.
pub fn consume_writer_host_copy(cell: &Arc<Mutex<ChannelCell>>) {
    let mut c = cell.lock().unwrap();
    c.consume_front_override();
    if c.role != Some(HostRole::Writer) {
        return;
    }
    c.ring_host_copies.pop_front();
}

fn load_word(word_base: u64, index: usize) -> u64 {
    // SAFETY: aligned atomic word array alive until instance close; mirrors
    // detach before that close.
    unsafe { (&*((word_base as *const AtomicU64).add(index))).load(Ordering::Acquire) }
}

fn store_word(word_base: u64, index: usize, value: u64) {
    unsafe { (&*((word_base as *const AtomicU64).add(index))).store(value, Ordering::Release) }
}

fn read_mirror_cell(reader: &ReaderMirror, sequence: u64) -> Vec<u8> {
    let slot = (sequence % reader.cap1) * reader.cell_bytes as u64;
    let ptr = (reader.mirror_base + slot) as *const u8;
    // SAFETY: binding validates the mirror extent; engine owns it through
    // instance close.
    unsafe { std::slice::from_raw_parts(ptr, reader.cell_bytes).to_vec() }
}

fn decode_reader_cell(
    dtype: Dtype,
    numel: usize,
    native_len: usize,
    wire: Vec<u8>,
) -> Result<Vec<u8>, ChannelError> {
    let native = if dtype == Dtype::Bool {
        if wire.len() == native_len {
            wire.into_iter().map(|byte| u8::from(byte != 0)).collect()
        } else {
            unpack_bool(&wire, numel)
        }
    } else {
        wire
    };
    if native.len() != native_len {
        return Err(ChannelError::BadLength {
            expected: native_len,
            got: native.len(),
        });
    }
    Ok(native)
}

/// Pack a 1-byte-per-bool cell to the bit-packed wire (LSB-first).
#[cfg(test)]
pub fn pack_bool(native: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; native.len().div_ceil(8)];
    pack_bool_into(native, &mut out);
    out
}

/// Pack directly into `out` (e.g. the pinned ring cell), no intermediate
/// allocation. `out` must hold `native.len().div_ceil(8)` bytes.
pub fn pack_bool_into(native: &[u8], out: &mut [u8]) {
    out.fill(0);
    for (i, &b) in native.iter().enumerate() {
        if b != 0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Unpack `numel` bits (LSB-first) from the wire into a 1-byte-per-bool cell.
pub fn unpack_bool(wire: &[u8], numel: usize) -> Vec<u8> {
    (0..numel)
        .map(|i| {
            let byte = wire.get(i / 8).copied().unwrap_or(0);
            (byte >> (i % 8)) & 1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::container::{ChanDType, ChannelDecl};
    use eta_ir::types::{Dtype, Shape};

    fn decl(shape: Shape, dtype: Dtype, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    // mask (bool[8], host Writer), out (i32[1], host Reader), tok (device-private, seeded)
    fn bound() -> BoundCells {
        let mk = |shape: Vec<u32>, dtype, d: &ChannelDecl| {
            let mut c = ChannelCell::new(shape, dtype, 1);
            c.matches_decl(d).unwrap();
            c.bind(d);
            Arc::new(Mutex::new(c))
        };
        vec![
            mk(
                vec![8],
                Dtype::Bool,
                &decl(Shape::vector(8), Dtype::Bool, HostRole::Writer, false),
            ),
            mk(
                vec![1],
                Dtype::I32,
                &decl(Shape::vector(1), Dtype::I32, HostRole::Reader, false),
            ),
            mk(
                vec![1],
                Dtype::I32,
                &decl(Shape::vector(1), Dtype::I32, HostRole::None, true),
            ),
        ]
    }

    fn publish_wire(cell: &Arc<Mutex<ChannelCell>>, instance_id: u64, wire: &[u8]) {
        let capacity = cell.lock().unwrap().capacity;
        let mut mirror = vec![0u8; wire.len() * capacity.saturating_add(1) as usize];
        mirror[..wire.len()].copy_from_slice(wire);
        let mirror = Box::leak(mirror.into_boxed_slice());
        let words = Box::leak(
            vec![
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ]
            .into_boxed_slice(),
        );
        cell.lock()
            .unwrap()
            .attach_reader_mirror(
                instance_id,
                mirror.as_ptr() as u64,
                words.as_ptr() as u64,
                wire.len() as u32,
                capacity,
                0,
                0,
                1,
                2,
                3,
            )
            .unwrap();
        words[1].store(1, Ordering::Release);
        let _ = instance_id;
    }

    #[test]
    fn prebind_put_stages_and_seed_pops() {
        let mut c = ChannelCell::new(vec![1], Dtype::I32, 1);
        c.put(7i32.to_le_bytes().to_vec()).unwrap();
        c.bind(&decl(Shape::vector(1), Dtype::I32, HostRole::None, true));
        assert_eq!(c.take_seed().unwrap(), 7i32.to_le_bytes().to_vec());
        assert!(c.seed_taken);
        // A second put on the fired device-private channel is illegal.
        assert_eq!(
            c.put(9i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::WrongRole {
                role: HostRole::None,
                op: "put"
            }
        );
        // A missing seed is a first-fire error.
        let mut m = ChannelCell::new(vec![1], Dtype::I32, 1);
        m.bind(&decl(Shape::vector(1), Dtype::I32, HostRole::None, true));
        assert_eq!(m.take_seed().unwrap_err(), ChannelError::MissingSeed);
    }

    #[test]
    fn set_empty_and_errors_without_changing_staging() {
        let mut cell = ChannelCell::new(vec![1], Dtype::I32, 2);
        assert_eq!(
            cell.set(1i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::Empty
        );

        cell.put(1i32.to_le_bytes().to_vec()).unwrap();
        cell.put(2i32.to_le_bytes().to_vec()).unwrap();
        assert_eq!(
            cell.put(3i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::Full
        );
        assert_eq!(
            cell.set(7i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::Empty,
            "pre-bind puts are staged, not a committed front"
        );
        assert_eq!(cell.staged.len(), 2, "set never changes staged occupancy");
        assert_eq!(cell.staged[0], 1i32.to_le_bytes());
        assert_eq!(cell.staged[1], 2i32.to_le_bytes());
        assert_eq!(
            cell.set(vec![0]).unwrap_err(),
            ChannelError::BadLength {
                expected: 4,
                got: 1
            }
        );
        cell.poison("test failure");
        assert_eq!(
            cell.set(3i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::Poisoned("test failure".into())
        );
    }

    #[test]
    fn bind_validates_constructor_geometry() {
        let c = ChannelCell::new(vec![2, 3], Dtype::U32, 1);
        assert!(
            c.matches_decl(&decl(
                Shape::matrix(2, 3),
                Dtype::U32,
                HostRole::None,
                false
            ))
            .is_ok()
        );
        assert!(
            c.matches_decl(&decl(Shape::vector(6), Dtype::U32, HostRole::None, false))
                .is_err()
        );
        assert!(
            c.matches_decl(&decl(
                Shape::matrix(2, 3),
                Dtype::I32,
                HostRole::None,
                false
            ))
            .is_err()
        );
    }

    #[test]
    fn writer_put_reader_take_roundtrip() {
        let cells = bound();
        let out_id = cells[1].lock().unwrap().global_id;
        // out (Reader) is empty until its bound mirror publishes.
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Empty
        );
        publish_wire(&cells[1], out_id, &5i32.to_le_bytes());
        assert_eq!(
            cells[1].lock().unwrap().read().unwrap(),
            5i32.to_le_bytes().to_vec(),
            "read peeks"
        );
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap(),
            5i32.to_le_bytes().to_vec(),
            "take consumes"
        );
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Empty
        );
    }

    #[test]
    fn packed_bool_mirror_decodes_to_native_bytes() {
        let mut cell = ChannelCell::new(vec![10], Dtype::Bool, 1);
        cell.bind(&decl(
            Shape::vector(10),
            Dtype::Bool,
            HostRole::Reader,
            false,
        ));
        let cell = Arc::new(Mutex::new(cell));
        let native = vec![1, 0, 0, 1, 1, 0, 1, 0, 1, 1];
        publish_wire(&cell, 88, &pack_bool(&native));
        assert_eq!(cell.lock().unwrap().take().unwrap(), native);
    }

}
