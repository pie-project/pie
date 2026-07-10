//! PTIR host channel cells (thrust-3 P3) — the host endpoint of a first-class
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
//!   1. guest `channel.put`s cells (seeds and/or Writer stage cells, D1);
//!   2. first fire: [`take_seed`](ChannelCell::take_seed) pops each `seeded`
//!      channel's staged cell into the submission's seed table; every fire:
//!      [`drain_host_puts`] **coalesces** the staged Writer puts into the
//!      submit's host-put table (bool packed to the wire, D1);
//!   3. the driver publishes Reader cells and epochs into the bound mirror;
//!   4. guest `channel.take`/`read` load that mirror directly. A poisoned channel turns
//!      every host `take`/`read` into an error (device-side fault surfaced to
//!      the guest).
//!
//! Cells are dtype-native (1 byte / bool); only the wire packs bool to bits
//! (`pack_bool`/`unpack_bool`), matching `PortSource::Const`'s D1 note.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::ptir::PtirChannelValue;
use pie_ptir::container::{self, ChannelDecl, HostRole};
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
    /// Host-staged cells (seeds pre-first-fire; Writer stage cells otherwise),
    /// FIFO, dtype-native.
    staged: VecDeque<Vec<u8>>,
    /// Device-produced cells awaiting host `take`/`read`, FIFO, dtype-native.
    produced: VecDeque<Vec<u8>>,
    /// Direct driver-owned mirror endpoints for every pass that binds this
    /// Reader channel. Completion publishes each endpoint's visible tail; the
    /// guest host operation copies only then.
    readers: Vec<ReaderMirror>,
    /// `Some(reason)` once a fire that feeds this channel failed: every later
    /// host `take`/`read` errors with the reason. Under run-ahead the submit
    /// returns before the fire resolves, so poison IS the error channel.
    poisoned: Option<String>,
}

#[derive(Clone, Debug)]
struct ReaderMirror {
    instance_id: u64,
    mirror_base: u64,
    word_base: u64,
    cell_bytes: usize,
    cap1: u64,
    mirror_offset: u64,
    tail_word_index: usize,
    poison_word_index: usize,
    published_tail: u64,
    copied_tail: u64,
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
            staged: VecDeque::new(),
            produced: VecDeque::new(),
            readers: Vec::new(),
            poisoned: None,
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
        Ok(())
    }

    /// Stamp the container's role/seeded onto the cell at bind.
    pub fn bind(&mut self, decl: &ChannelDecl) {
        self.role = Some(decl.host_role);
        self.seeded = decl.seeded;
    }

    /// Host `put` a dtype-native cell. Pre-bind this stages freely (seed or
    /// early Writer cell); post-bind it must be a Writer stage cell or the one
    /// seed on a not-yet-fired `seeded` channel.
    pub fn put(&mut self, native: Vec<u8>) -> Result<(), ChannelError> {
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
        self.staged.push_back(native);
        Ok(())
    }

    /// Number of host-staged cells (bind-time validation against the declared
    /// role: a Reader / non-seeded device-private channel must have none, a
    /// seeded non-Writer channel at most one).
    pub fn staged_len(&self) -> usize {
        self.staged.len()
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
        self.produced.pop_front().ok_or(ChannelError::Empty)
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

    /// Drain the staged Writer cells for one fire, packing bool to the wire.
    fn drain_wire(&mut self) -> Vec<Vec<u8>> {
        let pack = self.dtype == DType::Bool;
        self.staged
            .drain(..)
            .map(|cell| if pack { pack_bool(&cell) } else { cell })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_reader_mirror(
        &mut self,
        instance_id: u64,
        mirror_base: u64,
        word_base: u64,
        cell_bytes: u32,
        capacity: u32,
        mirror_offset: u64,
        tail_word_index: u32,
        poison_word_index: u32,
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
        if self
            .readers
            .iter()
            .any(|reader| reader.instance_id == instance_id)
        {
            return Err(format!(
                "channel {}: instance {instance_id} mirror already attached",
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
        self.readers.push(ReaderMirror {
            instance_id,
            mirror_base,
            word_base,
            cell_bytes,
            cap1: u64::from(capacity).saturating_add(1),
            mirror_offset,
            tail_word_index: tail_word_index as usize,
            poison_word_index: poison_word_index as usize,
            published_tail: 0,
            copied_tail: 0,
        });
        Ok(())
    }

    pub fn detach_reader_mirror(&mut self, instance_id: u64) {
        self.readers
            .retain(|reader| reader.instance_id != instance_id);
    }

    /// Make one completed instance's release-published mirror tail visible to
    /// subsequent `take`/`read` calls. No value is copied here.
    pub fn publish_reader_mirror(&mut self, instance_id: u64) -> Result<bool, String> {
        let Some(reader) = self
            .readers
            .iter_mut()
            .find(|reader| reader.instance_id == instance_id)
        else {
            return Err(format!(
                "channel {}: no mirror for instance {instance_id}",
                self.global_id
            ));
        };
        let poison = load_word(reader.word_base, reader.poison_word_index);
        if poison != 0 {
            let reason = format!("driver published poison epoch {poison}");
            self.poison(&reason);
            return Ok(true);
        }
        let tail = load_word(reader.word_base, reader.tail_word_index);
        if tail < reader.published_tail {
            return Err(format!(
                "channel {}: mirror tail regressed from {} to {tail}",
                self.global_id, reader.published_tail
            ));
        }
        if tail.saturating_sub(reader.copied_tail) > reader.cap1 {
            return Err(format!(
                "channel {}: mirror overrun (tail {tail}, copied {}, capacity {})",
                self.global_id, reader.copied_tail, reader.cap1
            ));
        }
        reader.published_tail = tail;
        Ok(false)
    }

    pub fn latest_reader_value(
        &mut self,
        instance_id: u64,
    ) -> Result<Option<Vec<u8>>, ChannelError> {
        let dtype = self.dtype;
        let numel = self.numel();
        let native_len = self.native_len();
        let Some(reader) = self
            .readers
            .iter()
            .find(|reader| reader.instance_id == instance_id)
        else {
            return Ok(None);
        };
        if reader.published_tail == 0 {
            return Ok(None);
        }
        let wire = read_mirror_cell(reader, reader.published_tail - 1);
        decode_reader_cell(dtype, numel, native_len, &wire).map(Some)
    }

    fn refresh_reader_mirrors(&mut self) -> Result<(), ChannelError> {
        let dtype = self.dtype;
        let numel = self.numel();
        let native_len = self.native_len();
        let (readers, produced) = (&mut self.readers, &mut self.produced);
        for reader in readers.iter_mut() {
            while reader.copied_tail < reader.published_tail {
                let wire = read_mirror_cell(reader, reader.copied_tail);
                produced.push_back(decode_reader_cell(dtype, numel, native_len, &wire)?);
                reader.copied_tail += 1;
            }
        }
        Ok(())
    }
}

/// A forward pass's bound cells, dense declaration order (`cells[i]` backs the
/// container's channel `i`).
pub type BoundCells = Vec<Arc<Mutex<ChannelCell>>>;

/// Coalesce every staged Writer put across a pass's cells into the submit's
/// host-put table (D1): one [`PtirChannelValue`] per staged cell, in
/// channel-then-FIFO order, keyed by the channel's GLOBAL id, bool packed to
/// the wire. Clears the Writer queues (shipped).
pub fn drain_host_puts(cells: &BoundCells) -> Vec<PtirChannelValue> {
    let mut out = Vec::new();
    for cell in cells.iter() {
        let mut c = cell.lock().unwrap();
        if c.role != Some(HostRole::Writer) {
            continue;
        }
        let gid = c.global_id;
        for bytes in c.drain_wire() {
            out.push(PtirChannelValue {
                channel: gid,
                bytes,
            });
        }
    }
    out
}

fn load_word(word_base: u64, index: usize) -> u64 {
    // SAFETY: direct-driver bind returns an aligned atomic word array that
    // remains alive until the instance is closed. Channel mirrors detach before
    // that close.
    unsafe { (&*((word_base as *const AtomicU64).add(index))).load(Ordering::Acquire) }
}

fn read_mirror_cell(reader: &ReaderMirror, sequence: u64) -> Vec<u8> {
    let slot = (sequence % reader.cap1) * reader.cell_bytes as u64;
    let ptr = (reader.mirror_base + reader.mirror_offset + slot) as *const u8;
    // SAFETY: the binding validates the mirror extent, and the driver owns it
    // through instance close.
    unsafe { std::slice::from_raw_parts(ptr, reader.cell_bytes).to_vec() }
}

fn decode_reader_cell(
    dtype: DType,
    numel: usize,
    native_len: usize,
    wire: &[u8],
) -> Result<Vec<u8>, ChannelError> {
    let native = if dtype == DType::Bool {
        if wire.len() == native_len {
            wire.iter().map(|byte| u8::from(*byte != 0)).collect()
        } else {
            unpack_bool(wire, numel)
        }
    } else {
        wire.to_vec()
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
    for (i, &b) in native.iter().enumerate() {
        if b != 0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
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
    use pie_ptir::container::{ChanDType, ChannelDecl};
    use pie_ptir::types::{DType, Shape};

    fn decl(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

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
        let words = Box::leak(vec![AtomicU64::new(0), AtomicU64::new(0)].into_boxed_slice());
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
            )
            .unwrap();
        words[0].store(1, Ordering::Release);
        cell.lock()
            .unwrap()
            .publish_reader_mirror(instance_id)
            .unwrap();
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
    fn mirror_tail_is_hidden_until_completion_publishes_epoch() {
        let cells = bound();
        let instance_id = 77;
        let mirror = Box::leak(9i32.to_le_bytes().to_vec().into_boxed_slice());
        let words = Box::leak(vec![AtomicU64::new(1), AtomicU64::new(0)].into_boxed_slice());
        cells[1]
            .lock()
            .unwrap()
            .attach_reader_mirror(
                instance_id,
                mirror.as_ptr() as u64,
                words.as_ptr() as u64,
                4,
                0,
                0,
                0,
                1,
            )
            .unwrap();
        assert_eq!(
            cells[1].lock().unwrap().take().unwrap_err(),
            ChannelError::Empty
        );
        cells[1]
            .lock()
            .unwrap()
            .publish_reader_mirror(instance_id)
            .unwrap();
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

    #[test]
    fn coalesce_drains_writer_puts_bool_packed() {
        let cells = bound();
        let mask_id = cells[0].lock().unwrap().global_id;
        // two staged puts on the bool[8] mask.
        cells[0]
            .lock()
            .unwrap()
            .put(vec![1, 0, 1, 0, 0, 0, 0, 0])
            .unwrap(); // bits 0,2 → 5
        cells[0]
            .lock()
            .unwrap()
            .put(vec![0, 1, 0, 0, 0, 0, 0, 1])
            .unwrap(); // bits 1,7 → 130
        // a staged seed on the device-private `tok` is NOT a Writer put.
        cells[2]
            .lock()
            .unwrap()
            .put(1i32.to_le_bytes().to_vec())
            .unwrap();
        let puts = drain_host_puts(&cells);
        assert_eq!(puts.len(), 2, "one PtirChannelValue per staged Writer cell");
        assert_eq!(
            puts[0],
            PtirChannelValue {
                channel: mask_id,
                bytes: vec![5]
            }
        );
        assert_eq!(
            puts[1],
            PtirChannelValue {
                channel: mask_id,
                bytes: vec![130]
            }
        );
        // draining clears the queue.
        assert!(drain_host_puts(&cells).is_empty());
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
            .attach_reader_mirror(1, 1, 1, 1, 1, 0, 0, 1)
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
