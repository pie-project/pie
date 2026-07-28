//! The **bound-trace sidecar** (`PTIB` v2): typed lowering plus compiler plans.
//! consumes instead of re-implementing shape/dtype inference (option B of the
//! P0 driver-typing question — one inference impl, [`pie_ir::infer`], zero
//! drift; T8 needs bit-identical shapes on every backend).
//!
//! The runtime calls [`pie_ir::validate::bind`] once at registration and ships
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
//! magic "PTIB" | version:u16 = 2 | flags:u16 = 0
//! container_hash:u64
//! n_channels:u32
//!   per channel: class:u8 (0 full_ring, 1 in_place, 2 in_place_undo)
//! n_readiness:u32
//!   per entry: chan:u32 | phase:u8 (stage tag; 0xFF descriptor) | dir:u8
//!              (0 needs-full, 1 needs-empty)
//! n_stages:u32   (container order)
//!   per stage: stage:u8 | n_values:u32
//!     per value (SSA id order): dtype:u8 | shape (rank:u8 | dims:u32[rank])
//! n_plans:u32
//!   per stage: stage:u8 | plan_len:u32 | encoded_region_plan[plan_len]
//! ```

use alloc::vec::Vec;
use core::fmt;

use pie_ir::registry::{Phase, Stage};
use pie_ir::types::{DType, MAX_RANK, Shape, ValueType};
use pie_ir::validate::{BoundTrace, ChannelClass, Direction};

/// Sidecar magic: ASCII `"PTIB"` (PTIR bound/typed).
pub const PTIB_MAGIC: [u8; 4] = *b"PTIB";
/// Sidecar format version.
pub const PTIB_VERSION: u16 = 2;
pub const PTIB_VERSION_LEGACY: u16 = 1;

/// Serialize a [`BoundTrace`]'s typed lowering (see module docs).
pub fn encode_bound(b: &BoundTrace) -> Vec<u8> {
    let plans = crate::compile::compile_bound(b);
    encode_bound_with_plans(b, &plans)
}

/// Serialize a bound trace using compiler plans already computed by the
/// registration cache.
pub fn encode_bound_with_plans(b: &BoundTrace, plans: &[crate::compile::CompiledStage]) -> Vec<u8> {
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
    w.extend_from_slice(&(plans.len() as u32).to_le_bytes());
    for plan in plans {
        let bytes = crate::compile::encode_stage_plan(&plan);
        w.push(plan.normalized.stage as u8);
        w.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        w.extend_from_slice(&bytes);
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
    pub stage_plans: Vec<(Stage, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarDecodeError {
    BadMagic,
    UnsupportedVersion(u16),
    UnexpectedEof,
    UnknownTag { what: &'static str, tag: u8 },
    RankTooLarge(u8),
    InvalidShape,
    CountTooLarge(&'static str),
    TrailingBytes,
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
            InvalidShape => f.write_str("sidecar shape has a zero or overflowing dimension"),
            CountTooLarge(table) => {
                write!(f, "{table} count exceeds its structural or byte limit")
            }
            TrailingBytes => f.write_str("trailing bytes after bound sidecar"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SidecarDecodeError {}

#[derive(Clone, Copy)]
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SidecarDecodeError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(SidecarDecodeError::UnexpectedEof)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SidecarDecodeError::UnexpectedEof)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SidecarDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SidecarDecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, SidecarDecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, SidecarDecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn bounded_count(
        &self,
        raw_count: u32,
        minimum_record_bytes: usize,
        structural_maximum: usize,
        table: &'static str,
    ) -> Result<usize, SidecarDecodeError> {
        let count =
            usize::try_from(raw_count).map_err(|_| SidecarDecodeError::CountTooLarge(table))?;
        let minimum_bytes = count
            .checked_mul(minimum_record_bytes)
            .ok_or(SidecarDecodeError::CountTooLarge(table))?;
        if minimum_record_bytes == 0
            || count > structural_maximum
            || minimum_bytes > self.remaining()
        {
            return Err(SidecarDecodeError::CountTooLarge(table));
        }
        Ok(count)
    }

    fn length(&self, raw_length: u32, table: &'static str) -> Result<usize, SidecarDecodeError> {
        let length =
            usize::try_from(raw_length).map_err(|_| SidecarDecodeError::CountTooLarge(table))?;
        if length > self.remaining() {
            return Err(SidecarDecodeError::CountTooLarge(table));
        }
        Ok(length)
    }
}

const MAX_SIDECAR_STAGES: usize = 4;

// `preflight_bound` used to sit here: a second, hand-written parser over the
// same bytes, run to completion before `decode_bound` parsed them again.
//
// Two parsers for one format have to agree, and these had already stopped.
// Every check preflight made, decode also made -- tags, ranks, trailing bytes,
// zero and overflowing dimensions (the last two via `Shape::new`, which
// rejects both) -- with exactly one exception: the readiness *phase* tag.
// Decode read that byte and pushed it unvalidated, so the only thing standing
// between a malformed phase and a `BoundSidecar` was a separate function that
// happened to be called first.
//
// That is the whole argument against the arrangement. The asymmetry was
// invisible precisely because the two passes were far apart and looked alike,
// and it will reappear the moment someone edits one and not the other. So the
// check moved into `decode_bound` and the second parser is gone: the format is
// now read in one place, and "validated" and "decoded" cannot drift because
// they are the same walk.

pub fn decode_bound(bytes: &[u8]) -> Result<BoundSidecar, SidecarDecodeError> {
    use SidecarDecodeError::*;

    let mut r = Reader { bytes, offset: 0 };
    if r.take(4)? != PTIB_MAGIC {
        return Err(BadMagic);
    }
    let v = r.u16()?;
    if v != PTIB_VERSION && v != PTIB_VERSION_LEGACY {
        return Err(UnsupportedVersion(v));
    }
    r.u16()?; // flags
    let container_hash = r.u64()?;
    let n_ch = r.u32()?;
    let n_ch = r.bounded_count(n_ch, 1, usize::MAX, "sidecar channel table")?;
    let mut classes = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        classes.push(match r.u8()? {
            0 => ChannelClass::FullRing,
            1 => ChannelClass::InPlace,
            2 => ChannelClass::InPlaceUndo,
            t => {
                return Err(UnknownTag {
                    what: "channel class",
                    tag: t,
                });
            }
        });
    }
    let n_rd = r.u32()?;
    let n_rd = r.bounded_count(n_rd, 6, n_ch, "sidecar readiness table")?;
    let mut readiness = Vec::with_capacity(n_rd);
    for _ in 0..n_rd {
        let chan = r.u32()?;
        let phase = r.u8()?;
        if Stage::from_u8(phase).is_none() && phase != PHASE_DESCRIPTOR_TAG {
            return Err(UnknownTag {
                what: "readiness phase",
                tag: phase,
            });
        }
        let dir = match r.u8()? {
            0 => Direction::NeedsFull,
            1 => Direction::NeedsEmpty,
            t => {
                return Err(UnknownTag {
                    what: "direction",
                    tag: t,
                });
            }
        };
        readiness.push((chan, phase, dir));
    }
    let n_st = r.u32()?;
    let n_st = r.bounded_count(n_st, 5, MAX_SIDECAR_STAGES, "sidecar stage table")?;
    let mut stage_types = Vec::with_capacity(n_st);
    for _ in 0..n_st {
        let st = r.u8()?;
        let stage = Stage::from_u8(st).ok_or(UnknownTag {
            what: "stage",
            tag: st,
        })?;
        let n_vals = r.u32()?;
        let n_vals = r.bounded_count(n_vals, 2, usize::MAX, "sidecar value table")?;
        let mut types = Vec::with_capacity(n_vals);
        for _ in 0..n_vals {
            let dt = r.u8()?;
            let dtype = match dt {
                0 => DType::F32,
                1 => DType::I32,
                2 => DType::U32,
                3 => DType::Bool,
                t => {
                    return Err(UnknownTag {
                        what: "dtype",
                        tag: t,
                    });
                }
            };
            let rank = r.u8()?;
            if rank as usize > MAX_RANK {
                return Err(RankTooLarge(rank));
            }
            let rank_count =
                r.bounded_count(rank as u32, 4, MAX_RANK, "sidecar shape dimensions")?;
            let mut dims = [0u32; MAX_RANK];
            for d in dims.iter_mut().take(rank_count) {
                *d = r.u32()?;
            }
            types.push(ValueType::new(
                Shape::new(&dims[..rank_count]).ok_or(InvalidShape)?,
                dtype,
            ));
        }
        stage_types.push((stage, types));
    }
    let mut stage_plans = Vec::new();
    if v == PTIB_VERSION {
        let n_plans = r.u32()?;
        let n_plans = r.bounded_count(n_plans, 5, MAX_SIDECAR_STAGES, "sidecar plan table")?;
        stage_plans.reserve(n_plans);
        for _ in 0..n_plans {
            let stage_tag = r.u8()?;
            let stage = Stage::from_u8(stage_tag).ok_or(UnknownTag {
                what: "plan stage",
                tag: stage_tag,
            })?;
            let raw_length = r.u32()?;
            let length = r.length(raw_length, "sidecar plan payload")?;
            stage_plans.push((stage, r.take(length)?.to_vec()));
        }
    }
    if r.offset != bytes.len() {
        return Err(TrailingBytes);
    }
    Ok(BoundSidecar {
        container_hash,
        classes,
        readiness,
        stage_types,
        stage_plans,
    })
}

// Re-exported phase-tag helper for readers: `0xFF` is the descriptor.
pub use pie_ir::registry::PHASE_DESCRIPTOR_TAG;

#[allow(unused_imports)]
use Phase as _PhaseDocOnly;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use pie_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use pie_ir::op::Op;
    use pie_ir::registry::ModelProfile;
    use pie_ir::types::Literal;
    use pie_ir::validate::bind;

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
        assert_eq!(s.stage_plans.len(), 1);
        let header = crate::compile::decode_plan_header(&s.stage_plans[0].1).unwrap();
        assert_eq!(header.stage, Stage::Epilogue);
        assert_eq!(
            s.readiness,
            b.readiness
                .iter()
                .map(|e| (e.chan, e.phase.tag(), e.dir))
                .collect::<Vec<_>>()
        );
    }

    /// The one check `decode_bound` did not make for itself.
    ///
    /// Until the preflight pass was folded in, this byte was read and pushed
    /// unvalidated, and the format was only safe because a second parser had
    /// already walked it. If someone ever splits the walk in two again, this
    /// fails.
    #[test]
    fn decode_rejects_an_unknown_readiness_phase() {
        let c = TraceContainer {
            names: vec![],
            channels: vec![
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(DType::U32),
                    capacity: 1,
                    host_role: HostRole::None,
                    seeded: true,
                },
                ChannelDecl {
                    shape: Shape::vector(1),
                    dtype: ChanDType::Concrete(DType::U32),
                    capacity: 1,
                    host_role: HostRole::Reader,
                    seeded: false,
                },
            ],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanRead(0),
                    Op::ChanPut { chan: 1, value: 0 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        let b = bind(c, ModelProfile::dummy()).unwrap();
        let bytes = encode_bound(&b);
        decode_bound(&bytes).expect("the unmodified sidecar decodes");

        // magic(4) version(2) flags(2) hash(8) channel-count(4) + two class
        // bytes, then readiness-count(4), then the first entry's chan(4);
        // the phase byte follows.
        let phase_offset = 4 + 2 + 2 + 8 + 4 + 2 + 4 + 4;
        assert!(
            !b.readiness.is_empty(),
            "this fixture must produce a readiness entry"
        );
        let mut corrupted = bytes.clone();
        corrupted[phase_offset] = 0x7E;
        assert!(
            matches!(
                decode_bound(&corrupted),
                Err(SidecarDecodeError::UnknownTag {
                    what: "readiness phase",
                    tag: 0x7E,
                })
            ),
            "decode accepted an unknown readiness phase: {:?}",
            decode_bound(&corrupted)
        );
    }
}
