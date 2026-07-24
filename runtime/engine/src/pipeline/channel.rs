//! Host channel cells (thrust-3 P3) — the host endpoint of a first-class
//! guest-constructed channel (overview §1, §3's `mask`/`out`).
//!
//! Under the first-class WIT surface a channel exists BEFORE any forward pass:
//! the guest constructs it (shape/dtype/capacity), stages seeds via `put`, and
//! only later binds it into a `forward-pass` (which stamps the container's
//! declared `HostRole` + `seeded` onto the cell). One [`ChannelCell`] is the
//! shared state behind one guest `channel` resource; a forward pass holds
//! `Arc` clones of its bound cells (dense declaration order), so a guest drop
//! of the handle never dangles the pass.
//!
//! SPSC discipline (echo's T2): a **Writer** channel is host-puts /
//! pass-consumes (§3's `mask`), a **Reader** channel is pass-puts / host-takes
//! (§3's `out`). Pre-bind, `put` stages freely (it may be a seed); post-bind
//! the container's role is enforced: `put` is Writer-only (plus the one seed
//! put on a `seeded` channel before its first fire), `take`/`read` are
//! Reader-only.
//!
//! Fire lifecycle (the pure host halves, unit-testable here):
//!   1. guest `channel.put`s cells: pre-endpoint they stage host-side; once
//!      the Writer endpoint exists a put writes the wire bytes straight into
//!      the driver-shared pinned ring and release-publishes the tail word
//!      (bool packed to the wire, D1) — no launch-descriptor involvement;
//!   2. first bind: [`take_seed`](ChannelCell::take_seed) pops each `seeded`
//!      channel's staged cell into the instance descriptor's seed table, then
//!      [`flush_writer_staging`](ChannelCell::flush_writer_staging) moves any
//!      remaining staged Writer cells into the ring;
//!   3. the driver pulls Writer ring entries pre-pass, and publishes Reader
//!      cells and the tail word into the bound mirror at pass completion;
//!   4. guest `channel.take`/`read` load that mirror directly. A poisoned channel turns
//!      every host `take`/`read` into an error (device-side fault surfaced to
//!      the guest).
//!
//! Cells are dtype-native (1 byte / bool); only the wire packs bool to bits
//! (`pack_bool`/`unpack_bool`), matching `PortSource::Const`'s D1 note.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::driver::ChannelEndpoint;
use pie_driver_abi::PieChannelEndpointBinding;
use pie_ptir::container::{self, ChanDType, ChannelDecl, ExternDir, HostRole};
use pie_ptir::types::DType;

/// Process-wide monotonic source of GLOBAL channel identities (0 reserved as a
/// null sentinel). Minted when the guest constructs a `channel` resource; a
/// channel keeps its id across every forward-pass it binds into, so the driver
/// resolves one device cell for a channel shared by many passes (multi-pass
/// channels — W0/W3). Inferlet-scoped in practice (one runtime per inferlet).
static NEXT_CHANNEL_ID: AtomicU64 = AtomicU64::new(1);

/// Mint the next process-wide global channel identity.
pub fn next_channel_id() -> u64 {
    NEXT_CHANNEL_ID.fetch_add(1, Ordering::Relaxed)
}

/// The shared host state behind one guest `channel` resource.
#[derive(Clone, Debug)]
pub struct ChannelCell {
    /// Global channel id (minted at construction) — the driver's device
    /// channel-registry key, stable across every pass this channel binds into.
    pub global_id: u64,
    /// Declared dims (guest constructor). Checked against the container decl
    /// at bind.
    pub shape: Vec<u32>,
    pub dtype: DType,
    pub capacity: u32,
    /// The container's role, stamped at bind (`None` = not yet bound to a
    /// forward pass).
    pub role: Option<HostRole>,
    /// The container's `seeded` flag, stamped at bind.
    pub seeded: bool,
    /// Whether this cell's seed was consumed by a first fire.
    pub seed_taken: bool,
    endpoint: Option<Arc<ChannelEndpoint>>,
    declared_dtype: Option<ChanDType>,
    extern_name: Option<String>,
    attachments: Vec<ChannelAttachment>,
    /// Host-staged cells (seeds pre-first-fire; Writer stage cells otherwise),
    /// FIFO, dtype-native.
    staged: VecDeque<Vec<u8>>,
    /// Host copies of Writer ring entries not yet claimed by a submitted
    /// fire, FIFO, dtype-native. The host shadow reads the front as the next
    /// fire's value and pops it when that fire submits — the engine's only
    /// record of post-bind Writer puts (the ring itself is driver-shared).
    ring_host_copies: VecDeque<Vec<u8>>,
    writer_tail: u64,
    /// Device-produced cells awaiting host `take`/`read`, FIFO, dtype-native.
    produced: VecDeque<Vec<u8>>,
    /// Logical device-ring sequences assigned to submitted fires. These are
    /// immutable tickets, not availability projections.
    device_reserved_head: u64,
    device_reserved_tail: u64,
    /// Direct driver-owned mirror endpoints for every pass that binds this
    /// Reader channel. Submission completion publishes each endpoint's visible tail; the
    /// guest host operation copies only then.
    reader: Option<ReaderMirror>,
    /// `Some(reason)` once a fire that feeds this channel failed: every later
    /// host `take`/`read` errors with the reason. Under run-ahead the submit
    /// returns before the fire resolves, so poison IS the error channel.
    poisoned: Option<String>,
    /// Host replacement for the current committed front. The per-pass host
    /// shadow consults this after a staged Writer put, so `set` changes the
    /// standing cell without displacing a value queued for the next fire.
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
    /// Sequences already copied out of the mirror — the reader-side cursor.
    /// Visibility is the release-published tail word itself; there is no
    /// separate runtime-side publication step.
    copied_tail: u64,
}

#[derive(Clone, Debug)]
struct ChannelAttachment {
    instance_id: u64,
    extern_dir: Option<ExternDir>,
}

/// A channel host-op failure (surfaced to the guest as a WIT `result` error).
/// Cells don't know their dense index — the host layer prefixes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelError {
    /// `put` on a bound non-Writer (past its seed), or `take`/`read` on a
    /// bound non-Reader.
    WrongRole { role: HostRole, op: &'static str },
    /// A host `take`/`read` with no produced cell available yet.
    Empty,
    /// The channel is poisoned (a device-side fault), with the fire's error.
    Poisoned(String),
    /// A put or mirror payload's length doesn't match the channel's shape×dtype.
    BadLength { expected: usize, got: usize },
    /// A first fire found no staged seed on a `seeded` channel.
    MissingSeed,
    /// A second `put` on a seeded non-Writer channel before its first fire —
    /// the seed is exactly one staged cell.
    SeedAlreadyStaged,
    /// A bounded channel has no host capacity.
    Full,
    /// The committed front is currently claimed by a submitted fire.
    InFlight,
    /// The native endpoint was closed.
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
    /// A fresh, unbound cell (the guest `channel` constructor). Mints a fresh
    /// global channel id (the driver's device-registry key).
    pub fn new(shape: Vec<u32>, dtype: DType, capacity: u32) -> Self {
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
                // Same-guest cross-pass chaining (task #32 / R4-4 single-
                // pipeline streams): a DEVICE-ONLY channel — no host role,
                // never seeded — may attach to multiple passes. The cell is
                // the ring-ticket source, so fires of both passes take
                // sequential tickets in submission order, and the pipeline
                // FIFO (same-pipeline constraint, §3.4) orders producer
                // puts before consumer reads on the one device ring — the
                // prefill→decode `tok_in` handoff. Host-visible or seeded
                // channels keep the one-pass rule: their host endpoint,
                // reader mirror, and seed staging are per-pass state.
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

    pub fn permanent_retry_cause(&self, accessed: bool) -> Option<String> {
        if !accessed {
            return None;
        }
        let Some(endpoint) = &self.endpoint else {
            return Some(format!("channel {} has no native endpoint", self.global_id));
        };
        let binding = endpoint.registered().binding;
        let poison = load_word(binding.word_base, binding.poison_word_index as usize);
        if poison != 0 {
            return Some(format!(
                "channel {} is poisoned at epoch {poison}",
                self.global_id
            ));
        }
        if load_word(binding.word_base, binding.closed_word_index as usize) != 0 {
            return Some(format!("channel {} is closed", self.global_id));
        }
        None
    }

    /// F8 deadlock decidability: a DEVICE-ONLY ring (no host role, unseeded)
    /// with fewer than two pass attachments has no consumer for its puts —
    /// after `pipeline.close` a publish blocked on it can never commit
    /// (close is what makes this decidable: before it, a consumer pass
    /// could still attach).
    pub fn is_consumerless_device_ring(&self) -> bool {
        self.role == Some(HostRole::None) && !self.seeded && self.attachments.len() < 2
    }

    pub fn reserve_device_ticket(&mut self, consume: bool, publish: bool) -> (u64, u64) {
        let expected_head = if consume {
            let expected = self.device_reserved_head;
            self.device_reserved_head += 1;
            expected
        } else {
            crate::driver::command::CHANNEL_TICKET_NONE
        };
        let expected_tail = if publish {
            let expected = self.device_reserved_tail;
            self.device_reserved_tail += 1;
            expected
        } else {
            crate::driver::command::CHANNEL_TICKET_NONE
        };
        (expected_head, expected_tail)
    }

    pub fn rollback_device_ticket(&mut self, expected_head: u64, expected_tail: u64) -> bool {
        let mut complete = true;
        if expected_tail != crate::driver::command::CHANNEL_TICKET_NONE {
            if self.device_reserved_tail == expected_tail + 1 {
                self.device_reserved_tail = expected_tail;
            } else {
                complete = false;
            }
        }
        if expected_head != crate::driver::command::CHANNEL_TICKET_NONE {
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
        // Direct puts start here: move any pre-endpoint staged Writer cells
        // into the shared ring (a seeded Writer flushes after its seed is
        // settled instead — see `flush_writer_staging`).
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

    /// Host `put` a dtype-native cell. Pre-bind this stages freely (seed or
    /// early Writer cell); post-bind it must be a Writer stage cell or the one
    /// seed on a not-yet-fired `seeded` channel. Once the Writer endpoint
    /// exists (and the seed, when declared, is settled), a put is a direct
    /// shared-memory write: wire bytes land in the pinned ring cell and the
    /// tail word is release-published (plan §4.2) — independent of pipelines
    /// and submissions.
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
    /// occupancy are unchanged, so a later queued put cannot move forward
    /// between a take and put. A front already claimed by a fire is immutable
    /// until that fire advances the device head.
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
                    "driver published poison epoch {poison}"
                )));
            }
            if load_word(binding.word_base, binding.closed_word_index as usize) != 0 {
                return Err(ChannelError::Closed);
            }

            // Pull visible Reader cells into the host queue before replacing
            // its front copy. Other roles read the same head/tail words
            // directly below.
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
        binding: PieChannelEndpointBinding,
        sequence: u64,
        native: &[u8],
    ) -> Result<(), ChannelError> {
        let cell_bytes = binding.cell_bytes as usize;
        let wire_len = if self.dtype == DType::Bool {
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
        if self.dtype == DType::Bool {
            pack_bool_into(native, cell);
        } else {
            cell.copy_from_slice(native);
        }
        // Re-publish the unchanged tail so the replacement bytes happen-before
        // the next consumer's acquire of queue state without changing occupancy.
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

    /// Write one cell into the driver-shared Writer ring (plan §4.2): check
    /// poison/closed/backpressure via the shared words, write the wire bytes
    /// at `tail % cap1`, then release-publish the incremented tail word. The
    /// spare `+1` ring cell distinguishes full from empty and holds the
    /// not-yet-consumed producer cell.
    fn write_writer_ring(
        &mut self,
        binding: PieChannelEndpointBinding,
        native: &[u8],
    ) -> Result<(), ChannelError> {
        let poison = load_word(binding.word_base, binding.poison_word_index as usize);
        if poison != 0 {
            return Err(ChannelError::Poisoned(format!(
                "driver published poison epoch {poison}"
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
        let wire_len = if self.dtype == DType::Bool {
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
        // SAFETY: the binding's pinned mirror was validated at registration
        // (`mirror_bytes >= cell_bytes * cap1`) and stays alive until the
        // channel's ordered close; the SPSC discipline makes this cell ours.
        let cell = unsafe {
            std::slice::from_raw_parts_mut((binding.mirror_base + offset) as *mut u8, cell_bytes)
        };
        if self.dtype == DType::Bool {
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
    /// Runs once the endpoint exists and the seed (when declared) has been
    /// settled into the instance descriptor; steady state has no staging.
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

    /// Number of host-staged cells (bind-time validation against the declared
    /// role: a Reader / non-seeded device-private channel must have none, a
    /// seeded non-Writer channel at most one).
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Frame validation (Vesuvius, k > 1): host-known cells a Writer channel
    /// can still feed to future fires. Counted by ring sequence — the seed
    /// plus every flushed ring write (`writer_tail`) plus pre-endpoint
    /// staging, minus the consume tickets already reserved by submitted
    /// fires — so a seeded descriptor Writer's committed initial cell counts
    /// exactly once.
    pub fn writer_available_cells(&self) -> u64 {
        self.writer_tail
            .saturating_add(self.staged.len() as u64)
            .saturating_sub(self.device_reserved_head)
    }

    /// Frame validation (Vesuvius, k > 1): the Reader ring's reservation
    /// pressure as (publications reserved by accepted unsettled fires, cells
    /// the host has already consumed). Their difference is the worst-case
    /// ring occupancy if the guest drains nothing before the frame executes —
    /// deterministic at submit time, never a function of drain timing.
    pub fn reader_ring_pressure(&self) -> (u64, u64) {
        let consumed = self
            .reader
            .as_ref()
            .map(|reader| load_word(reader.word_base, reader.head_word_index))
            .unwrap_or(0);
        (self.device_reserved_tail, consumed)
    }

    /// Frame validation (Vesuvius, k > 1): a device-only ring's structural
    /// backlog — publish tickets reserved by accepted unsettled fires minus
    /// consume tickets likewise reserved. The worst-case occupancy the ring
    /// reaches once every reserved fire settles, before anything not yet
    /// submitted consumes — deterministic at submit time, never a function
    /// of drain timing.
    pub fn device_ring_backlog(&self) -> u64 {
        self.device_reserved_tail
            .saturating_sub(self.device_reserved_head)
    }

    /// Frame validation (Vesuvius, k > 1): whether the host side knows a
    /// committed value exists for a latest-value (read-only-bound) channel.
    pub fn has_committed_front(&self) -> bool {
        self.seeded
            || !self.staged.is_empty()
            || !self.ring_host_copies.is_empty()
            || self.front_override.is_some()
            || self.device_reserved_tail > 0
    }

    /// Host `take` a produced cell (Reader), FIFO. An unbound channel is
    /// simply empty.
    pub fn take(&mut self) -> Result<Vec<u8>, ChannelError> {
        self.refresh_reader_mirrors()?;
        if let Some(reason) = &self.poisoned {
            return Err(ChannelError::Poisoned(reason.clone()));
        }
        if let Some(role) = self.role {
            if role != HostRole::Reader {
                return Err(ChannelError::WrongRole { role, op: "take" });
            }
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
        if let Some(role) = self.role {
            if role != HostRole::Reader {
                return Err(ChannelError::WrongRole { role, op: "read" });
            }
        }
        self.produced.front().cloned().ok_or(ChannelError::Empty)
    }

    /// Poison the cell with the failed fire's error: every later host
    /// `take`/`read` errors with it. First poison wins (the earliest failure
    /// is the root cause).
    pub fn poison(&mut self, reason: &str) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason.to_string());
        }
    }

    /// Pop this `seeded` channel's staged seed for the first fire (D2 —
    /// per-instance data, never identity). Errors if nothing was staged.
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

    pub fn commit_seed(&mut self) {
        let _ = self.staged.pop_front();
        self.seed_taken = true;
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
                "channel {}: driver mirror bound to {:?}, expected Reader",
                self.global_id, self.role
            ));
        }
        if mirror_base == 0 || word_base == 0 || cell_bytes == 0 {
            return Err(format!(
                "channel {}: driver returned an invalid mirror binding",
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
        if cell_bytes != native_len && !(self.dtype == DType::Bool && cell_bytes == packed_bool_len)
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

    pub fn detach_reader_mirror(&mut self, _instance_id: u64) {}

    /// Peek the most recently release-published mirror cell (device-geometry
    /// reclaim reads `w_cont` this way) without touching the take cursor.
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
        // Direct-word visibility: the driver release-publishes the tail,
        // poison, and closed words from its completion callback; the gate is
        // the word itself, not a runtime-side finalize step.
        let mut poison_reason = None;
        let mut closed = false;
        let mut visible_tail = 0;
        if let Some(reader) = self.reader.as_ref() {
            let poison = load_word(reader.word_base, reader.poison_word_index);
            if poison != 0 {
                poison_reason = Some(format!("driver published poison epoch {poison}"));
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

/// A first-class, guest-constructed channel (overview §1) — the WIT
/// `pie:inferlet/forward.channel` resource. The SAME handle is bound into a
/// forward pass (dense declaration index) and used for host
/// `put`/`take`/`read`; the shared [`ChannelCell`] is Arc'd so a pass that
/// bound it survives the guest dropping the handle. Domain state (not WIT
/// glue), so it lives here rather than in `inferlet::host::forward`, which
/// only holds the `Host`/`HostChannel` impls that push/get/delete it from the
/// WASM component resource table.
pub struct Channel {
    pub cell: Arc<Mutex<ChannelCell>>,
    /// Set at SUBMIT: the feeding PIPELINE's in-flight fire queue (W3.1). A
    /// channel may be fed by several passes, but all must submit on the SAME
    /// pipeline (§3.4) — so `take`/`read` await + finalize fires from one FIFO
    /// (submission order) until the cell fills. `None` until first submit.
    pub fires: Option<crate::pipeline::fire::PendingFires>,
}

/// Process-teardown close batching: walks the process's resource table,
/// takes over the driver close notification from every guest channel
/// endpoint still holding one, and returns the channel ids grouped by
/// owning driver. The caller posts one batched close per driver AFTER
/// dropping the table (whose pass drops post the instance closes) — the
/// scheduler mailbox's per-producer FIFO then delivers every instance
/// close before the channel batch, preserving the driver's
/// instance-before-channel close order. Endpoints this walk cannot reach
/// (a cell kept alive only by a pass after the guest dropped its channel
/// handle) keep notifying one-by-one from their own drop.
pub fn detach_channel_close_notifications(
    resources: &mut wasmtime::component::ResourceTable,
) -> Vec<(usize, Vec<u64>)> {
    let mut by_driver: std::collections::BTreeMap<usize, Vec<u64>> =
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
        by_driver
            .entry(endpoint.registered().driver_id)
            .or_default()
            .push(channel_id);
    }
    by_driver.into_iter().collect()
}

/// The next host-known Writer value on `cell` — the native value the driver
/// will pull for the next submitted fire (`None`: not a Writer channel, or
/// nothing pending). Pre-endpoint values sit in `staged`; post-bind puts go
/// straight to the driver-shared ring with a host copy retained in
/// `ring_host_copies` until a consuming fire submits. The host shadow (and
/// through it the canonical-KV fire gate) reads Writer values through this.
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
/// backing [`staged_put_bytes`]'s front so the next fire sees the next
/// value. (`staged` is never popped here — pre-flush entries are pending
/// ring writes, not yet consumable by any fire.)
pub fn consume_writer_host_copy(cell: &Arc<Mutex<ChannelCell>>) {
    let mut c = cell.lock().unwrap();
    c.consume_front_override();
    if c.role != Some(HostRole::Writer) {
        return;
    }
    c.ring_host_copies.pop_front();
}

fn load_word(word_base: u64, index: usize) -> u64 {
    // SAFETY: direct-driver bind returns an aligned atomic word array that
    // remains alive until the instance is closed. Channel mirrors detach before
    // that close.
    unsafe { (&*((word_base as *const AtomicU64).add(index))).load(Ordering::Acquire) }
}

fn store_word(word_base: u64, index: usize, value: u64) {
    unsafe { (&*((word_base as *const AtomicU64).add(index))).store(value, Ordering::Release) }
}

fn read_mirror_cell(reader: &ReaderMirror, sequence: u64) -> Vec<u8> {
    let slot = (sequence % reader.cap1) * reader.cell_bytes as u64;
    let ptr = (reader.mirror_base + slot) as *const u8;
    // SAFETY: the binding validates the mirror extent, and the driver owns it
    // through instance close.
    unsafe { std::slice::from_raw_parts(ptr, reader.cell_bytes).to_vec() }
}

fn decode_reader_cell(
    dtype: DType,
    numel: usize,
    native_len: usize,
    wire: Vec<u8>,
) -> Result<Vec<u8>, ChannelError> {
    let native = if dtype == DType::Bool {
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

/// Pack a 1-byte-per-bool cell to the bit-packed wire (LSB-first, D1).
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
    use crate::driver::{self, ChannelRegistrationPlan, DriverSpec, SchedulerLimits};
    use crate::scheduler::{self, worker::BatchScheduler};
    use pie_driver_dummy_lib::DummyDriverOptions;
    use pie_ptir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
    use pie_ptir::op::Op;
    use pie_ptir::registry::Stage;
    use pie_ptir::types::{DType, Shape};
    use tokio::time::{Duration, timeout};

    fn decl(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 2,
            host_role: role,
            seeded,
        }
    }

    fn dummy_launch() -> crate::driver::LaunchPlan {
        crate::driver::LaunchPlan {
            token_ids: vec![1],
            position_ids: vec![0],
            kv_page_indptr: vec![0, 0],
            kv_last_page_lens: vec![0],
            qo_indptr: vec![0, 1],
            sampling_indices: vec![0],
            sampling_indptr: vec![0, 1],
            mask_indptr: vec![0, 0],
            single_token_mode: true,
            ..crate::driver::LaunchPlan::default()
        }
    }

    /// Plan §14 gates 4/5 over the REAL put path: `ChannelCell::put` writes
    /// the driver-shared ring directly; puts past capacity are `Full`; a
    /// fire with no put retries without effects; a consuming fire publishes
    /// the head word and notifies the writer wait slot.
    ///
    /// Lives here (not `scheduler::tests`) because it exercises `ChannelCell`
    /// directly as the driver-shared ring's memory — the scheduler itself
    /// stays ignorant of cell semantics (the completion-sink inversion);
    /// this integration test is `pipeline`-level (downward import of
    /// `scheduler`/`driver`, never the reverse).
    /// §3 shape: mask (bool [8], host Writer) + out (i32 [1], host Reader) +
    /// a device-private seeded channel (tok).
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
                DType::Bool,
                &decl(Shape::vector(8), DType::Bool, HostRole::Writer, false),
            ),
            mk(
                vec![1],
                DType::I32,
                &decl(Shape::vector(1), DType::I32, HostRole::Reader, false),
            ),
            mk(
                vec![1],
                DType::I32,
                &decl(Shape::vector(1), DType::I32, HostRole::None, true),
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
        // The release-published tail word IS the visibility gate (§4.5).
        words[1].store(1, Ordering::Release);
        let _ = instance_id;
    }

    #[test]
    fn prebind_put_stages_and_seed_pops() {
        // A guest seeds a channel BEFORE any forward pass exists.
        let mut c = ChannelCell::new(vec![1], DType::I32, 1);
        c.put(7i32.to_le_bytes().to_vec()).unwrap();
        c.bind(&decl(Shape::vector(1), DType::I32, HostRole::None, true));
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
        let mut m = ChannelCell::new(vec![1], DType::I32, 1);
        m.bind(&decl(Shape::vector(1), DType::I32, HostRole::None, true));
        assert_eq!(m.take_seed().unwrap_err(), ChannelError::MissingSeed);
    }

    #[test]
    fn set_empty_and_errors_without_changing_staging() {
        let mut cell = ChannelCell::new(vec![1], DType::I32, 2);
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
        let c = ChannelCell::new(vec![2, 3], DType::U32, 1);
        assert!(
            c.matches_decl(&decl(
                Shape::matrix(2, 3),
                DType::U32,
                HostRole::None,
                false
            ))
            .is_ok()
        );
        assert!(
            c.matches_decl(&decl(Shape::vector(6), DType::U32, HostRole::None, false))
                .is_err()
        );
        assert!(
            c.matches_decl(&decl(
                Shape::matrix(2, 3),
                DType::I32,
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
    fn mirror_tail_word_is_the_visibility_gate() {
        // Direct-word visibility (plan §4.5): the release-published tail word
        // is the gate. A zero tail hides the cell; storing the tail makes it
        // takeable with no runtime-side finalize step in between.
        let cells = bound();
        let instance_id = 77;
        let mirror = Box::leak(9i32.to_le_bytes().to_vec().into_boxed_slice());
        let words = Box::leak(
            vec![
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ]
            .into_boxed_slice(),
        );
        cells[1]
            .lock()
            .unwrap()
            .attach_reader_mirror(
                instance_id,
                mirror.as_ptr() as u64,
                words.as_ptr() as u64,
                4,
                1,
                0,
                0,
                1,
                2,
                3,
            )
            .unwrap();
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Empty
        );
        words[1].store(1, Ordering::Release);
        assert_eq!(cells[1].lock().unwrap().take().unwrap(), 9i32.to_le_bytes());
    }

    #[test]
    fn packed_bool_mirror_decodes_to_native_bytes() {
        let mut cell = ChannelCell::new(vec![10], DType::Bool, 1);
        cell.bind(&decl(
            Shape::vector(10),
            DType::Bool,
            HostRole::Reader,
            false,
        ));
        let cell = Arc::new(Mutex::new(cell));
        let native = vec![1, 0, 0, 1, 1, 0, 1, 0, 1, 1];
        publish_wire(&cell, 88, &pack_bool(&native));
        assert_eq!(cell.lock().unwrap().take().unwrap(), native);
    }

    #[test]
    fn role_discipline_enforced() {
        let cells = bound();
        // put on the Reader `out` is illegal.
        assert_eq!(
            cells[1]
                .lock()
                .unwrap()
                .put(0i32.to_le_bytes().to_vec())
                .unwrap_err(),
            ChannelError::WrongRole {
                role: HostRole::Reader,
                op: "put"
            }
        );
        // take on the Writer `mask` is illegal.
        assert_eq!(
            cells[0].lock().unwrap().take().unwrap_err(),
            ChannelError::WrongRole {
                role: HostRole::Writer,
                op: "take"
            }
        );
        // seeded device-private `tok`: put stages until its seed is taken.
        assert!(
            cells[2]
                .lock()
                .unwrap()
                .put(1i32.to_le_bytes().to_vec())
                .is_ok()
        );
    }

    #[test]
    fn put_length_validated() {
        let cells = bound();
        // mask is bool[8] → 8 native bytes.
        assert_eq!(
            cells[0].lock().unwrap().put(vec![1, 0, 1]).unwrap_err(),
            ChannelError::BadLength {
                expected: 8,
                got: 3
            }
        );
        assert!(
            cells[0]
                .lock()
                .unwrap()
                .put(vec![1, 0, 1, 0, 1, 0, 1, 0])
                .is_ok()
        );
    }

    /// Builds a Writer ring the test owns (mirror + words), attached via the
    /// direct-write path so `put` lands wire bytes in shared memory.
    fn writer_ring(
        capacity: u32,
        cell_bytes: usize,
    ) -> (ChannelCell, &'static [u8], &'static [AtomicU64]) {
        let mut declaration = decl(Shape::vector(8), DType::Bool, HostRole::Writer, false);
        declaration.capacity = capacity;
        let mut writer = ChannelCell::new(vec![8], DType::Bool, capacity);
        writer.matches_decl(&declaration).unwrap();
        writer.bind(&declaration);
        let cap1 = capacity as usize + 1;
        let mirror = Box::leak(vec![0u8; cell_bytes * cap1].into_boxed_slice());
        let words = Box::leak(
            (0..4)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        (writer, mirror, words)
    }

    fn attach_writer(
        writer: &mut ChannelCell,
        mirror: &[u8],
        words: &[AtomicU64],
        cell_bytes: u32,
        capacity: u32,
    ) {
        let table = pie_waker::WakerTable::global();
        let endpoint = ChannelEndpoint::new(crate::driver::RegisteredChannel {
            driver_id: usize::MAX,
            binding: PieChannelEndpointBinding {
                channel_id: writer.global_id,
                mirror_base: mirror.as_ptr() as u64,
                word_base: words.as_ptr() as u64,
                mirror_bytes: mirror.len() as u64,
                word_bytes: (words.len() * std::mem::size_of::<AtomicU64>()) as u64,
                cell_bytes,
                capacity,
                head_word_index: 0,
                tail_word_index: 1,
                poison_word_index: 2,
                closed_word_index: 3,
            },
            reader_wait_id: table.alloc(),
            writer_wait_id: table.alloc(),
        });
        writer.attach_endpoint(Arc::new(endpoint)).unwrap();
    }

    #[test]
    fn writer_put_writes_the_shared_ring_directly() {
        // Plan §4.2: with an endpoint attached, `put` is a shared-memory
        // write — wire bytes at `tail % cap1`, then the release-published
        // tail word. No staging, no launch-descriptor involvement.
        let (mut writer, mirror, words) = writer_ring(2, 1);
        attach_writer(&mut writer, mirror, words, 1, 2);
        writer.put(vec![1, 0, 1, 0, 0, 0, 0, 0]).unwrap(); // bits 0,2 → 5
        assert_eq!(words[1].load(Ordering::Acquire), 1, "tail word published");
        assert_eq!(mirror[0], 5, "wire bytes written to the pinned ring");
        writer.put(vec![0, 1, 0, 0, 0, 0, 0, 1]).unwrap(); // bits 1,7 → 130
        assert_eq!(words[1].load(Ordering::Acquire), 2);
        assert_eq!(mirror[1], 130);
        // capacity 2, nothing consumed → the third put backpressures.
        assert_eq!(
            writer.put(vec![0; 8]).unwrap_err(),
            ChannelError::Full,
            "tail - head >= capacity is Full"
        );
        // The driver consuming one (head word advance) frees a cell; the
        // spare +1 ring slot means the write wraps into slot 0's successor.
        words[0].store(1, Ordering::Release);
        writer.put(vec![1, 1, 0, 0, 0, 0, 0, 0]).unwrap(); // bits 0,1 → 3
        assert_eq!(words[1].load(Ordering::Acquire), 3);
        assert_eq!(mirror[2 % 3], 3, "sequence 2 lands at slot 2 of cap1=3");
    }

    #[test]
    fn writer_set_replaces_only_committed_front_and_rejects_in_flight_use() {
        let mut declaration = decl(Shape::vector(8), DType::Bool, HostRole::Writer, true);
        declaration.capacity = 2;
        let mut writer = ChannelCell::new(vec![8], DType::Bool, 2);
        writer.bind(&declaration);
        writer.seed_taken = true;
        writer.writer_tail = 1;
        let mirror = Box::leak(vec![1u8, 0, 0].into_boxed_slice());
        let words = Box::leak(
            vec![
                AtomicU64::new(0),
                AtomicU64::new(1),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ]
            .into_boxed_slice(),
        );
        attach_writer(&mut writer, mirror, words, 1, 2);
        let second = vec![0, 1, 0, 0, 0, 0, 0, 0];
        writer.put(second).unwrap();

        writer.set(vec![0, 0, 1, 0, 0, 0, 0, 0]).unwrap();
        writer.set(vec![0, 0, 0, 1, 0, 0, 0, 0]).unwrap();
        assert_eq!(words[0].load(Ordering::Acquire), 0);
        assert_eq!(words[1].load(Ordering::Acquire), 2);
        assert_eq!(mirror[0], 8, "repeat set replaces the same front slot");
        assert_eq!(mirror[1], 2, "capacity>1 replacement leaves the next slot");
        assert_eq!(
            writer.put(vec![0; 8]).unwrap_err(),
            ChannelError::Full,
            "set does not release put backpressure"
        );

        assert_eq!(
            writer.reserve_device_ticket(true, false),
            (0, crate::driver::command::CHANNEL_TICKET_NONE)
        );
        assert_eq!(writer.set(vec![1; 8]).unwrap_err(), ChannelError::InFlight);
        assert_eq!(mirror[0], 8, "an in-flight front is never overwritten");
        assert_eq!(mirror[1], 2, "queued put remains intact");
    }

    #[test]
    fn cold_rebind_resynchronizes_device_tickets_from_the_live_ring() {
        let mut declaration = decl(Shape::vector(8), DType::Bool, HostRole::Writer, false);
        declaration.capacity = 2;
        let mut writer = ChannelCell::new(vec![8], DType::Bool, 2);
        writer.attach(11, &declaration, None).unwrap();
        let mirror = Box::leak(vec![0u8; 3].into_boxed_slice());
        let words = Box::leak(
            (0..4)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        attach_writer(&mut writer, mirror, words, 1, 2);

        words[0].store(3, Ordering::Release);
        words[1].store(5, Ordering::Release);
        writer.detach(11);
        writer.attach(12, &declaration, None).unwrap();

        assert_eq!(
            writer.reserve_device_ticket(true, true),
            (3, 5),
            "a persistent endpoint's actual words define the first rebind ticket"
        );
    }

    #[test]
    fn writer_staging_flushes_into_the_ring_at_attach() {
        // Pre-endpoint puts stage host-side; attaching the endpoint flushes
        // them FIFO into the ring (plan §4.2 pre-endpoint staging).
        let (mut writer, mirror, words) = writer_ring(2, 1);
        writer.put(vec![1, 0, 1, 0, 0, 0, 0, 0]).unwrap(); // 5
        writer.put(vec![0, 1, 0, 0, 0, 0, 0, 1]).unwrap(); // 130
        assert_eq!(writer.staged_len(), 2);
        assert_eq!(words[1].load(Ordering::Acquire), 0, "nothing published yet");
        attach_writer(&mut writer, mirror, words, 1, 2);
        assert_eq!(writer.staged_len(), 0, "staging drained");
        assert_eq!(words[1].load(Ordering::Acquire), 2, "both cells published");
        assert_eq!(&mirror[..2], &[5, 130]);
    }

    #[test]
    fn writer_put_surfaces_poison_and_close() {
        let (mut writer, mirror, words) = writer_ring(2, 1);
        attach_writer(&mut writer, mirror, words, 1, 2);
        words[2].store(9, Ordering::Release);
        assert!(matches!(
            writer.put(vec![0; 8]).unwrap_err(),
            ChannelError::Poisoned(_)
        ));
        assert!(matches!(
            writer.set(vec![0; 8]).unwrap_err(),
            ChannelError::Poisoned(_)
        ));
        words[2].store(0, Ordering::Release);
        words[3].store(1, Ordering::Release);
        assert_eq!(writer.put(vec![0; 8]).unwrap_err(), ChannelError::Closed);
        assert_eq!(writer.set(vec![0; 8]).unwrap_err(), ChannelError::Closed);
    }

    #[test]
    fn poison_faults_reader_with_reason() {
        let cells = bound();
        let out_id = cells[1].lock().unwrap().global_id;
        publish_wire(&cells[1], out_id, &7i32.to_le_bytes());
        cells[1].lock().unwrap().poison("fire 3 failed: OOM");
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Poisoned("fire 3 failed: OOM".into())
        );
        assert_eq!(
            cells[1].lock().unwrap().read().unwrap_err(),
            ChannelError::Poisoned("fire 3 failed: OOM".into())
        );
        // First poison wins — a later failure doesn't mask the root cause.
        cells[1].lock().unwrap().poison("fire 4 cascade");
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Poisoned("fire 3 failed: OOM".into())
        );
    }

    #[test]
    fn mirror_binding_rejects_non_reader() {
        let cells = bound();
        let err = cells[0]
            .lock()
            .unwrap()
            .attach_reader_mirror(1, 1, 1, 1, 1, 0, 0, 1, 2, 3)
            .unwrap_err();
        assert!(err.contains("expected Reader"));
    }

    #[test]
    fn bool_pack_unpack_roundtrip() {
        let native = vec![1u8, 0, 0, 1, 1, 0, 1, 0, 1, 1]; // 10 bools → 2 wire bytes
        let wire = pack_bool(&native);
        assert_eq!(wire.len(), 2);
        assert_eq!(unpack_bool(&wire, native.len()), native);
    }

    #[test]
    fn direct_mirror_roundtrips() {
        let cells = bound(); // `out` is the host-Reader at channel 1
        let out_id = cells[1].lock().unwrap().global_id;
        publish_wire(&cells[1], out_id, &5i32.to_le_bytes());
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap(),
            5i32.to_le_bytes().to_vec(),
            "the fired token reaches the guest"
        );
    }
}
