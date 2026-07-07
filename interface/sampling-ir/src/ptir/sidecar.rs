//! The **bound-trace sidecar** (`PTIB` v1): the typed lowering a backend
//! consumes instead of re-implementing shape/dtype inference (option B of the
//! P0 driver-typing question — one inference impl, [`super::infer`], zero
//! drift; T8 needs bit-identical shapes on every backend).
//!
//! The runtime calls [`super::validate::bind`] once at registration and ships
//! `(container bytes, sidecar bytes)` to the driver. The sidecar carries the
//! **container hash** it was derived from — a reader MUST reject a sidecar
//! whose hash doesn't match the container it accompanies. Contents: per-op
//! result types for every stage body (SSA order), the readiness direction
//! table, and the §7.1 channel classes. Channel element types are already in
//! the container's declarations (the sidecar view has `ACT` materialized to
//! the program dtype).
//!
//! Layout (LE, tightly packed, self-delimiting) — mirrored in
//! `PTIR-CONTAINER.md` §7 and `include/ptir_abi.h`:
//!
//! ```text
//! magic "PTIB" | version:u16 = 1 | flags:u16 = 0
//! container_hash:u64
//! n_channels:u32
//!   per channel: class:u8 (0 full_ring, 1 in_place, 2 in_place_undo)
//! n_readiness:u32
//!   per entry: chan:u32 | phase:u8 (stage tag; 0xFF descriptor) | dir:u8
//!              (0 needs-full, 1 needs-empty)
//! n_stages:u32   (container order)
//!   per stage: stage:u8 | n_values:u32
//!     per value (SSA id order): dtype:u8 | shape (rank:u8 | dims:u32[rank])
//! ```

use alloc::vec::Vec;
use core::fmt;

use super::registry::{Phase, Stage};
use super::validate::{BoundTrace, ChannelClass, Direction};
use crate::types::{DType, Shape, ValueType, MAX_RANK};

/// Sidecar magic: ASCII `"PTIB"` (PTIR bound/typed).
pub const PTIB_MAGIC: [u8; 4] = *b"PTIB";
/// Sidecar format version.
pub const PTIB_VERSION: u16 = 1;

/// Serialize a [`BoundTrace`]'s typed lowering (see module docs).
pub fn encode_bound(b: &BoundTrace) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(&PTIB_MAGIC);
    w.extend_from_slice(&PTIB_VERSION.to_le_bytes());
    w.extend_from_slice(&0u16.to_le_bytes());
    w.extend_from_slice(&b.hash.to_le_bytes());
    w.extend_from_slice(&(b.classes.len() as u32).to_le_bytes());
    for c in &b.classes {
        w.push(match c {
            ChannelClass::FullRing => 0,
            ChannelClass::InPlace => 1,
            ChannelClass::InPlaceUndo => 2,
        });
    }
    w.extend_from_slice(&(b.readiness.len() as u32).to_le_bytes());
    for e in &b.readiness {
        w.extend_from_slice(&e.chan.to_le_bytes());
        w.push(e.phase.tag());
        w.push(match e.dir {
            Direction::NeedsFull => 0,
            Direction::NeedsEmpty => 1,
        });
    }
    w.extend_from_slice(&(b.container.stages.len() as u32).to_le_bytes());
    for (sp, types) in b.container.stages.iter().zip(&b.stage_types) {
        w.push(sp.stage as u8);
        w.extend_from_slice(&(types.len() as u32).to_le_bytes());
        for t in types {
            w.push(t.dtype as u8);
            w.push(t.shape.rank() as u8);
            for &d in t.shape.dims() {
                w.extend_from_slice(&d.to_le_bytes());
            }
        }
    }
    w
}

/// A decoded sidecar (reader mirror, used by tests and the mock driver; the
/// C++ driver implements an independent reader against the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct BoundSidecar {
    pub container_hash: u64,
    pub classes: Vec<ChannelClass>,
    pub readiness: Vec<(u32, u8, Direction)>,
    pub stage_types: Vec<(Stage, Vec<ValueType>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarDecodeError {
    BadMagic,
    UnsupportedVersion(u16),
    UnexpectedEof,
    UnknownTag { what: &'static str, tag: u8 },
    RankTooLarge(u8),
}

impl fmt::Display for SidecarDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SidecarDecodeError::*;
        match self {
            BadMagic => f.write_str("bad magic (expected \"PTIB\")"),
            UnsupportedVersion(v) => write!(f, "unsupported sidecar version {v}"),
            UnexpectedEof => f.write_str("unexpected end of buffer"),
            UnknownTag { what, tag } => write!(f, "unknown {what} tag 0x{tag:02x}"),
            RankTooLarge(r) => write!(f, "shape rank {r} exceeds MAX_RANK"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SidecarDecodeError {}

pub fn decode_bound(bytes: &[u8]) -> Result<BoundSidecar, SidecarDecodeError> {
    use SidecarDecodeError::*;
    struct R<'a>(&'a [u8], usize);
    impl<'a> R<'a> {
        fn take(&mut self, n: usize) -> Result<&'a [u8], SidecarDecodeError> {
            let end = self.1.checked_add(n).ok_or(UnexpectedEof)?;
            let s = self.0.get(self.1..end).ok_or(UnexpectedEof)?;
            self.1 = end;
            Ok(s)
        }
        fn u8(&mut self) -> Result<u8, SidecarDecodeError> {
            Ok(self.take(1)?[0])
        }
        fn u16(&mut self) -> Result<u16, SidecarDecodeError> {
            let s = self.take(2)?;
            Ok(u16::from_le_bytes([s[0], s[1]]))
        }
        fn u32(&mut self) -> Result<u32, SidecarDecodeError> {
            let s = self.take(4)?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        }
        fn u64(&mut self) -> Result<u64, SidecarDecodeError> {
            let s = self.take(8)?;
            Ok(u64::from_le_bytes(s.try_into().unwrap()))
        }
    }
    let mut r = R(bytes, 0);
    if r.take(4)? != PTIB_MAGIC {
        return Err(BadMagic);
    }
    let v = r.u16()?;
    if v != PTIB_VERSION {
        return Err(UnsupportedVersion(v));
    }
    r.u16()?; // flags
    let container_hash = r.u64()?;
    let n_ch = r.u32()? as usize;
    let mut classes = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        classes.push(match r.u8()? {
            0 => ChannelClass::FullRing,
            1 => ChannelClass::InPlace,
            2 => ChannelClass::InPlaceUndo,
            t => return Err(UnknownTag { what: "channel class", tag: t }),
        });
    }
    let n_rd = r.u32()? as usize;
    let mut readiness = Vec::with_capacity(n_rd);
    for _ in 0..n_rd {
        let chan = r.u32()?;
        let phase = r.u8()?;
        let dir = match r.u8()? {
            0 => Direction::NeedsFull,
            1 => Direction::NeedsEmpty,
            t => return Err(UnknownTag { what: "direction", tag: t }),
        };
        readiness.push((chan, phase, dir));
    }
    let n_st = r.u32()? as usize;
    let mut stage_types = Vec::with_capacity(n_st);
    for _ in 0..n_st {
        let st = r.u8()?;
        let stage = Stage::from_u8(st).ok_or(UnknownTag { what: "stage", tag: st })?;
        let n_vals = r.u32()? as usize;
        let mut types = Vec::with_capacity(n_vals);
        for _ in 0..n_vals {
            let dt = r.u8()?;
            let dtype = match dt {
                0 => DType::F32,
                1 => DType::I32,
                2 => DType::U32,
                3 => DType::Bool,
                t => return Err(UnknownTag { what: "dtype", tag: t }),
            };
            let rank = r.u8()?;
            if rank as usize > MAX_RANK {
                return Err(RankTooLarge(rank));
            }
            let mut dims = [0u32; MAX_RANK];
            for d in dims.iter_mut().take(rank as usize) {
                *d = r.u32()?;
            }
            types.push(ValueType::new(Shape::new(&dims[..rank as usize]).unwrap(), dtype));
        }
        stage_types.push((stage, types));
    }
    Ok(BoundSidecar { container_hash, classes, readiness, stage_types })
}

// Re-exported phase-tag helper for readers: `0xFF` is the descriptor.
pub use super::registry::PHASE_DESCRIPTOR_TAG;

#[allow(unused_imports)]
use Phase as _PhaseDocOnly;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptir::container::{
        ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer,
    };
    use crate::ptir::op::Op;
    use crate::ptir::registry::ModelProfile;
    use crate::ptir::validate::bind;
    use crate::types::Literal;
    use alloc::vec;

    #[test]
    fn sidecar_round_trips_and_pins_hash() {
        let c = TraceContainer {
            names: vec![],
            channels: vec![ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::U32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            }],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Const(Literal::U32(1)),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                ],
            }],
        externs: alloc::vec::Vec::new(),
        };
        let b = bind(c, ModelProfile::dummy()).unwrap();
        let bytes = encode_bound(&b);
        let s = decode_bound(&bytes).expect("decode");
        assert_eq!(s.container_hash, b.hash);
        assert_eq!(s.classes, b.classes);
        assert_eq!(s.stage_types.len(), 1);
        assert_eq!(s.stage_types[0].1, b.stage_types[0]);
        assert_eq!(
            s.readiness,
            b.readiness
                .iter()
                .map(|e| (e.chan, e.phase.tag(), e.dir))
                .collect::<Vec<_>>()
        );
    }
}
