//! PTIR host channel store (thrust-3 P3) — the host endpoint of an instance's
//! host-facing channels (overview §1, §3's `mask`/`out`).
//!
//! SPSC discipline (echo's T2): a **Writer** channel is host-puts / pass-consumes
//! (§3's `mask`), a **Reader** channel is pass-puts / host-takes (§3's `out`). The
//! store mirrors that: `put` is Writer-only, `take`/`read` are Reader-only.
//!
//! Fire lifecycle (the pure host halves, unit-testable in isolation here; the
//! driver-submit wire between them is `TODO(P2c-fire)`, gated on charlie's
//! stage-runner):
//!   1. guest `channel.put`s stage cells on Writer channels (D1);
//!   2. [`ChannelStore::drain_host_puts`] **coalesces** every staged put into the
//!      submit's host-put table (bool packed to the wire, D1);
//!   3. the driver fires; the response carries the Reader channels' produced
//!      cells, which [`ChannelStore::marshal_response`] unpacks + enqueues;
//!   4. guest `channel.take`/`read` retrieve them. A poisoned channel turns every
//!      host `take`/`read` into an error (device-side fault surfaced to the guest).
//!
//! The store's cells are dtype-native (1 byte / bool); only the wire packs bool
//! to bits (`pack_bool`/`unpack_bool`), matching `PortSource::Const`'s D1 note.

use std::collections::{BTreeMap, VecDeque};

use pie_driver_abi::PtirChannelValue;
use pie_sampling_ir::ptir::container::{self, HostRole, TraceContainer};
use pie_sampling_ir::types::DType;

/// The host-side cell store for one instance's host-facing channels.
#[derive(Clone, Debug, Default)]
pub struct ChannelStore {
    /// Host-facing channels only, by dense channel index (ordered for a
    /// deterministic coalesced host-put table).
    channels: BTreeMap<u32, HostChannelState>,
}

#[derive(Clone, Debug)]
struct HostChannelState {
    role: HostRole,
    dtype: DType,
    /// Elements per cell (`shape.numel()`).
    numel: usize,
    /// FIFO of dtype-native cell payloads (1 byte / bool, LE otherwise).
    cells: VecDeque<Vec<u8>>,
    poisoned: bool,
}

impl HostChannelState {
    /// Native (unpacked) bytes per cell: `numel` for bool, `numel*4` otherwise.
    fn native_len(&self) -> usize {
        self.numel * container::const_elem_size(self.dtype)
    }
}

/// A channel host-op failure (surfaced to the guest as a WIT `result` error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChannelError {
    /// The index isn't a host-facing channel of this instance.
    NotHostFacing { channel: u32 },
    /// `put` on a non-Writer, or `take`/`read` on a non-Reader.
    WrongRole { channel: u32, role: HostRole, op: &'static str },
    /// A host `take`/`read` with no produced cell available yet.
    Empty { channel: u32 },
    /// The channel is poisoned (a device-side fault).
    Poisoned { channel: u32 },
    /// A put/marshal payload's length doesn't match the channel's shape×dtype.
    BadLength { channel: u32, expected: usize, got: usize },
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ChannelError::*;
        match self {
            NotHostFacing { channel } => write!(f, "channel {channel} is not host-facing"),
            WrongRole { channel, role, op } => {
                write!(f, "channel {channel}: {op} is illegal on a {role:?} channel")
            }
            Empty { channel } => write!(f, "channel {channel}: no cell available"),
            Poisoned { channel } => write!(f, "channel {channel} is poisoned"),
            BadLength { channel, expected, got } => {
                write!(f, "channel {channel}: {got} bytes, expected {expected} (shape×dtype)")
            }
        }
    }
}
impl std::error::Error for ChannelError {}

impl ChannelStore {
    /// Build a store over a container's host-facing channels (`host-role != none`).
    pub fn new(container: &TraceContainer) -> Self {
        let mut channels = BTreeMap::new();
        for (i, c) in container.channels.iter().enumerate() {
            if c.host_role != HostRole::None {
                channels.insert(
                    i as u32,
                    HostChannelState {
                        role: c.host_role,
                        dtype: c.dtype.program_dtype(),
                        numel: c.shape.numel() as usize,
                        cells: VecDeque::new(),
                        poisoned: false,
                    },
                );
            }
        }
        ChannelStore { channels }
    }

    fn get(&self, channel: u32) -> Result<&HostChannelState, ChannelError> {
        self.channels.get(&channel).ok_or(ChannelError::NotHostFacing { channel })
    }
    fn get_mut(&mut self, channel: u32) -> Result<&mut HostChannelState, ChannelError> {
        self.channels.get_mut(&channel).ok_or(ChannelError::NotHostFacing { channel })
    }

    /// Whether `channel` is a host-facing channel of this instance (the guest
    /// may only obtain a host endpoint on such a channel).
    pub fn contains(&self, channel: u32) -> bool {
        self.channels.contains_key(&channel)
    }

    /// Host `put` a cell onto a **Writer** channel (§3's `mask`). Staged (D1);
    /// shipped at the next [`drain_host_puts`](Self::drain_host_puts).
    pub fn put(&mut self, channel: u32, native: Vec<u8>) -> Result<(), ChannelError> {
        let st = self.get_mut(channel)?;
        if st.role != HostRole::Writer {
            return Err(ChannelError::WrongRole { channel, role: st.role, op: "put" });
        }
        let expected = st.native_len();
        if native.len() != expected {
            return Err(ChannelError::BadLength { channel, expected, got: native.len() });
        }
        st.cells.push_back(native);
        Ok(())
    }

    /// Host `take` a produced cell off a **Reader** channel (§3's `out`), FIFO.
    pub fn take(&mut self, channel: u32) -> Result<Vec<u8>, ChannelError> {
        let st = self.get_mut(channel)?;
        if st.poisoned {
            return Err(ChannelError::Poisoned { channel });
        }
        if st.role != HostRole::Reader {
            return Err(ChannelError::WrongRole { channel, role: st.role, op: "take" });
        }
        st.cells.pop_front().ok_or(ChannelError::Empty { channel })
    }

    /// Host `read` (peek, non-consuming) a produced cell off a **Reader** channel.
    pub fn read(&self, channel: u32) -> Result<Vec<u8>, ChannelError> {
        let st = self.get(channel)?;
        if st.poisoned {
            return Err(ChannelError::Poisoned { channel });
        }
        if st.role != HostRole::Reader {
            return Err(ChannelError::WrongRole { channel, role: st.role, op: "read" });
        }
        st.cells.front().cloned().ok_or(ChannelError::Empty { channel })
    }

    /// Poison a channel (a device-side fault): every later host `take`/`read`
    /// on it errors.
    pub fn poison(&mut self, channel: u32) -> Result<(), ChannelError> {
        self.get_mut(channel)?.poisoned = true;
        Ok(())
    }

    /// Coalesce every staged Writer put into the submit's host-put table (D1):
    /// one [`PtirChannelValue`] per staged cell, in channel-then-FIFO order,
    /// bool packed to the wire. Clears the Writer queues (shipped).
    pub fn drain_host_puts(&mut self) -> Vec<PtirChannelValue> {
        let mut out = Vec::new();
        for (&channel, st) in self.channels.iter_mut() {
            if st.role != HostRole::Writer {
                continue;
            }
            let pack = st.dtype == DType::Bool;
            for cell in st.cells.drain(..) {
                let bytes = if pack { pack_bool(&cell) } else { cell };
                out.push(PtirChannelValue { channel, bytes });
            }
        }
        out
    }

    /// Marshal a fired pass's produced Reader-channel cells back into the store
    /// (the host-side half of the response path): each `(channel, wire_bytes)`
    /// is unpacked (bool bits → 1 byte/bool) and enqueued for host `take`/`read`.
    pub fn marshal_response(
        &mut self,
        produced: &[(u32, Vec<u8>)],
    ) -> Result<(), ChannelError> {
        for (channel, wire) in produced {
            let st = self.get_mut(*channel)?;
            if st.role != HostRole::Reader {
                return Err(ChannelError::WrongRole { channel: *channel, role: st.role, op: "marshal" });
            }
            let native = if st.dtype == DType::Bool {
                unpack_bool(wire, st.numel)
            } else {
                wire.clone()
            };
            let expected = st.native_len();
            if native.len() != expected {
                return Err(ChannelError::BadLength { channel: *channel, expected, got: native.len() });
            }
            st.cells.push_back(native);
        }
        Ok(())
    }
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
    use pie_sampling_ir::ptir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
    use pie_sampling_ir::ptir::op::Op;
    use pie_sampling_ir::ptir::registry::Stage;
    use pie_sampling_ir::types::{DType, Shape};

    fn chan(shape: Shape, dtype: DType, role: HostRole) -> ChannelDecl {
        ChannelDecl { shape, dtype: ChanDType::Concrete(dtype), capacity: 2, host_role: role, seeded: false }
    }

    /// §3 shape: mask (bool [vocab], host Writer) + out (i32 [1], host Reader) +
    /// a device-private channel (no host role).
    fn store() -> ChannelStore {
        let c = TraceContainer {
            names: vec![],
            channels: vec![
                chan(Shape::vector(8), DType::Bool, HostRole::Writer), // 0 mask
                chan(Shape::vector(1), DType::I32, HostRole::Reader),  // 1 out
                chan(Shape::vector(1), DType::I32, HostRole::None),    // 2 private
            ],
            ports: vec![],
            externs: vec![],
            stages: vec![StageProgram { stage: Stage::Epilogue, ops: vec![Op::ChanTake(0)] }],
        };
        ChannelStore::new(&c)
    }

    #[test]
    fn new_registers_only_host_facing() {
        let s = store();
        assert_eq!(s.channels.len(), 2, "the device-private channel 2 is not host-facing");
        assert!(s.channels.contains_key(&0) && s.channels.contains_key(&1));
    }

    #[test]
    fn writer_put_reader_take_roundtrip() {
        let mut s = store();
        // out (Reader) is empty until a fire marshals into it.
        assert_eq!(s.take(1).unwrap_err(), ChannelError::Empty { channel: 1 });
        // a fire produced token 5 on `out`.
        s.marshal_response(&[(1, 5i32.to_le_bytes().to_vec())]).unwrap();
        assert_eq!(s.read(1).unwrap(), 5i32.to_le_bytes().to_vec(), "read peeks");
        assert_eq!(s.take(1).unwrap(), 5i32.to_le_bytes().to_vec(), "take consumes");
        assert_eq!(s.take(1).unwrap_err(), ChannelError::Empty { channel: 1 });
    }

    #[test]
    fn role_discipline_enforced() {
        let mut s = store();
        // put on the Reader `out` is illegal.
        assert_eq!(
            s.put(1, 0i32.to_le_bytes().to_vec()).unwrap_err(),
            ChannelError::WrongRole { channel: 1, role: HostRole::Reader, op: "put" }
        );
        // take on the Writer `mask` is illegal.
        assert_eq!(
            s.take(0).unwrap_err(),
            ChannelError::WrongRole { channel: 0, role: HostRole::Writer, op: "take" }
        );
        // a non-host-facing / unknown channel.
        assert_eq!(s.put(2, vec![]).unwrap_err(), ChannelError::NotHostFacing { channel: 2 });
    }

    #[test]
    fn put_length_validated() {
        let mut s = store();
        // mask is bool[8] → 8 native bytes.
        assert_eq!(
            s.put(0, vec![1, 0, 1]).unwrap_err(),
            ChannelError::BadLength { channel: 0, expected: 8, got: 3 }
        );
        assert!(s.put(0, vec![1, 0, 1, 0, 1, 0, 1, 0]).is_ok());
    }

    #[test]
    fn coalesce_drains_writer_puts_bool_packed() {
        let mut s = store();
        // two staged puts on the bool[8] mask (a capacity-2 channel).
        s.put(0, vec![1, 0, 1, 0, 0, 0, 0, 0]).unwrap(); // bits 0,2 → 0b0000_0101 = 5
        s.put(0, vec![0, 1, 0, 0, 0, 0, 0, 1]).unwrap(); // bits 1,7 → 0b1000_0010 = 130
        let puts = s.drain_host_puts();
        assert_eq!(puts.len(), 2, "one PtirChannelValue per staged cell");
        assert_eq!(puts[0], PtirChannelValue { channel: 0, bytes: vec![5] });
        assert_eq!(puts[1], PtirChannelValue { channel: 0, bytes: vec![130] });
        // draining clears the queue.
        assert!(s.drain_host_puts().is_empty());
    }

    #[test]
    fn poison_faults_reader() {
        let mut s = store();
        s.marshal_response(&[(1, 7i32.to_le_bytes().to_vec())]).unwrap();
        s.poison(1).unwrap();
        assert_eq!(s.take(1).unwrap_err(), ChannelError::Poisoned { channel: 1 });
        assert_eq!(s.read(1).unwrap_err(), ChannelError::Poisoned { channel: 1 });
    }

    #[test]
    fn marshal_response_rejects_non_reader() {
        let mut s = store();
        // producing into the Writer `mask` is a protocol error.
        assert_eq!(
            s.marshal_response(&[(0, vec![1])]).unwrap_err(),
            ChannelError::WrongRole { channel: 0, role: HostRole::Writer, op: "marshal" }
        );
    }

    #[test]
    fn bool_pack_unpack_roundtrip() {
        let native = vec![1u8, 0, 0, 1, 1, 0, 1, 0, 1, 1]; // 10 bools → 2 wire bytes
        let wire = pack_bool(&native);
        assert_eq!(wire.len(), 2);
        assert_eq!(unpack_bool(&wire, native.len()), native);
    }

    #[test]
    fn marshal_from_forward_response_roundtrips() {
        // The exact response→channel path `ptir_host::submit` runs after a fire:
        // a ForwardResponse's PTIR output table → `ptir_output_at` → the tuple
        // form `marshal_response` consumes → the guest `take` sees the token.
        use pie_driver_abi::ForwardResponse;
        let mut s = store(); // `out` is the host-Reader at channel 1
        let mut resp = ForwardResponse::default();
        resp.push_ptir_output(&[PtirChannelValue { channel: 1, bytes: 5i32.to_le_bytes().to_vec() }]);

        let produced: Vec<(u32, Vec<u8>)> = resp
            .ptir_output_at(0)
            .unwrap()
            .into_iter()
            .map(|c| (c.channel, c.bytes))
            .collect();
        s.marshal_response(&produced).unwrap();
        assert_eq!(s.take(1).unwrap(), 5i32.to_le_bytes().to_vec(), "the fired token reaches the guest");
    }
}
