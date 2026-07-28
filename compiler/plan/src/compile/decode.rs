//! Reading plan bytes back.
//!
//! The mirror of [`super::encode`]. This parses input the process did not
//! necessarily write, so every count and length goes through
//! [`pie_ir::read::Reader`]'s two bounds rather than being trusted.

use pie_ir::container::{MAX_CHANNELS, MAX_OPS};
use pie_ir::op::IntrinsicId;
use pie_ir::read::{ReadError, Reader};
use pie_ir::registry::Stage;
use pie_ir::types::{DType, MAX_RANK};

use super::encode::PLAN_MAGIC;
use super::normalize::ValueDomain;
use super::region::{LibraryOp, PartitionKind, ScheduleTemplate};
use super::symbolic::SymbolicExtent;
use super::{COMPILER_VERSION, REGION_PLAN_VERSION};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedPlanHeader {
    pub stage: Stage,
    pub signature_hash: u64,
    pub singleton_regions: u32,
    pub fused_regions: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanDecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    InvalidStage,
    InvalidRecord,
    CountTooLarge(&'static str),
}

impl From<ReadError> for PlanDecodeError {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::UnexpectedEof => PlanDecodeError::Truncated,
            ReadError::CountTooLarge(table) => PlanDecodeError::CountTooLarge(table),
        }
    }
}

/// A length that must leave `required_tail` bytes behind it.
///
/// The plan format is the only one with this shape -- a variable-length op
/// payload followed by a fixed-size source count -- so it stays here rather
/// than in the shared cursor.
pub(crate) fn length_with_tail(
    reader: &Reader<'_>,
    raw_length: u32,
    required_tail: usize,
    record: &'static str,
) -> Result<usize, PlanDecodeError> {
    let length = raw_length as usize;
    let required = length
        .checked_add(required_tail)
        .ok_or(PlanDecodeError::CountTooLarge(record))?;
    if required > reader.remaining() {
        return Err(PlanDecodeError::CountTooLarge(record));
    }
    Ok(length)
}

pub(crate) fn scan_plan_shape(reader: &mut Reader<'_>) -> Result<(), PlanDecodeError> {
    let rank = reader.u8()?;
    let rank = reader.bounded_count(
        rank as u32,
        4,
        MAX_RANK,
        "planned operation shape dimensions",
    )?;
    let bytes = rank.checked_mul(4).ok_or(PlanDecodeError::CountTooLarge(
        "planned operation shape dimensions",
    ))?;
    reader.take(bytes)?;
    Ok(())
}

pub(crate) fn scan_planned_op(bytes: &[u8]) -> Result<u32, PlanDecodeError> {
    let mut reader = Reader::new(bytes);
    let tag = reader.u8()?;
    let results = match tag {
        0x01..=0x06 | 0x1E | 0x30..=0x33 | 0x3A | 0x40 | 0x41 | 0x50 | 0x64 | 0x90 | 0x91 => {
            reader.take(4)?;
            if tag == 0x50 { 2 } else { 1 }
        }
        0x07 => {
            reader.take(4)?;
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            1
        }
        0x10..=0x1D | 0x1F | 0x51 | 0x55 | 0x60 | 0x61 | 0x65 | 0x66 => {
            reader.take(8)?;
            if tag == 0x51 { 2 } else { 1 }
        }
        0x20 | 0x62 | 0x63 | 0x67 => {
            reader.take(12)?;
            1
        }
        0x68 => {
            reader.take(16)?;
            1
        }
        0x38 | 0x39 => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
            1
        }
        0x58 => {
            reader.take(4)?;
            if reader.u8()? > 2 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
            1
        }
        0x70 | 0x71 => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
            if reader.u8()? > 1 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            1
        }
        0x81 => {
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
            1
        }
        0x92 => {
            reader.take(8)?;
            0
        }
        0xA0 => {
            if reader.u16()? > IntrinsicId::AttnScore as u16 || reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            scan_plan_shape(&mut reader)?;
            1
        }
        0xA1 => {
            reader.u16()?;
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            scan_plan_shape(&mut reader)?;
            let arguments = reader.u8()? as u32;
            let arguments =
                reader.bounded_count(arguments, 4, u8::MAX as usize, "kernel argument vector")?;
            reader.take(
                arguments
                    .checked_mul(4)
                    .ok_or(PlanDecodeError::CountTooLarge("kernel argument vector"))?,
            )?;
            1
        }
        0xA2 => {
            reader.u16()?;
            let arguments = reader.u8()? as u32;
            let arguments =
                reader.bounded_count(arguments, 4, u8::MAX as usize, "sink argument vector")?;
            reader.take(
                arguments
                    .checked_mul(4)
                    .ok_or(PlanDecodeError::CountTooLarge("sink argument vector"))?,
            )?;
            0
        }
        _ => return Err(PlanDecodeError::InvalidRecord),
    };
    if reader.offset() != bytes.len() {
        return Err(PlanDecodeError::InvalidRecord);
    }
    Ok(results)
}

pub(crate) fn scan_index_vector(
    reader: &mut Reader<'_>,
    structural_maximum: usize,
    upper_bound: usize,
    ordered: bool,
    table: &'static str,
) -> Result<usize, PlanDecodeError> {
    let raw_count = reader.u32()?;
    let count = reader.bounded_count(raw_count, 4, structural_maximum, table)?;
    let byte_count = count
        .checked_mul(4)
        .ok_or(PlanDecodeError::CountTooLarge(table))?;
    let bytes = reader.take(byte_count)?;
    let mut previous = None;
    for value in bytes.chunks_exact(4) {
        let value = usize::try_from(u32::from_le_bytes(value.try_into().unwrap()))
            .map_err(|_| PlanDecodeError::CountTooLarge(table))?;
        if value >= upper_bound || (ordered && previous.is_some_and(|old| old >= value)) {
            return Err(PlanDecodeError::InvalidRecord);
        }
        previous = Some(value);
    }
    Ok(count)
}

pub(crate) fn scan_partition(
    reader: &mut Reader<'_>,
    expected_kind: PartitionKind,
    operation_count: usize,
    value_count: usize,
    channel_count: usize,
) -> Result<u32, PlanDecodeError> {
    let kind = match reader.u8()? {
        0 => PartitionKind::Singleton,
        1 => PartitionKind::Fused,
        _ => return Err(PlanDecodeError::InvalidRecord),
    };
    if kind != expected_kind || reader.u8()? > 1 {
        return Err(PlanDecodeError::InvalidRecord);
    }
    let raw_regions = reader.u32()?;
    let region_count = reader.bounded_count(raw_regions, 19, operation_count, "region table")?;
    for _ in 0..region_count {
        let region_kind = reader.u8()?;
        let library = reader.u8()?;
        let schedule = reader.u8()?;
        if region_kind > 1
            || (region_kind == 1 && library > LibraryOp::SecondParty as u8)
            || schedule > ScheduleTemplate::Library as u8
        {
            return Err(PlanDecodeError::InvalidRecord);
        }
        let nodes = scan_index_vector(
            reader,
            operation_count,
            operation_count,
            true,
            "region node vector",
        )?;
        let inputs = scan_index_vector(
            reader,
            value_count,
            value_count,
            false,
            "region input vector",
        )?;
        let outputs = scan_index_vector(
            reader,
            value_count,
            value_count,
            false,
            "region output vector",
        )?;
        let raw_sinks = reader.u32()?;
        let sinks = reader.bounded_count(raw_sinks, 8, nodes, "region sink vector")?;
        let sink_bytes = sinks
            .checked_mul(8)
            .ok_or(PlanDecodeError::CountTooLarge("region sink vector"))?;
        for sink in reader.take(sink_bytes)?.chunks_exact(8) {
            let channel = usize::try_from(u32::from_le_bytes(sink[..4].try_into().unwrap()))
                .map_err(|_| PlanDecodeError::CountTooLarge("region sink vector"))?;
            let value = usize::try_from(u32::from_le_bytes(sink[4..].try_into().unwrap()))
                .map_err(|_| PlanDecodeError::CountTooLarge("region sink vector"))?;
            if channel >= channel_count || value >= value_count {
                return Err(PlanDecodeError::InvalidRecord);
            }
        }
        // The nucleus region's shape is a wire ABI. Two forms exist: the plain
        // `(logits, top_p, rng)` recipe and the scaled one, which additionally
        // carries the temperature divisor and the pre-scale logits. Both are
        // 13 nodes; they differ only in arity. Accepting only the plain form
        // would reject plans this crate's own encoder produces and every CUDA
        // backend accepts (`grouped_nucleus_region_supported`,
        // `singleton_codegen`, `program_runtime`).
        if region_kind == 1
            && library == LibraryOp::NucleusSample as u8
            && (nodes != 13 || (inputs != 3 && inputs != 5) || outputs != 1 || sinks != 0)
        {
            return Err(PlanDecodeError::InvalidRecord);
        }
    }
    Ok(raw_regions)
}

/// Allocation-free structural decoder used by registration tests and backend
/// preflight. Backend codegen applies the same limits before materializing a
/// plan.
pub fn decode_plan_header(bytes: &[u8]) -> Result<EncodedPlanHeader, PlanDecodeError> {
    let mut reader = Reader::new(bytes);
    if reader.take(4)? != PLAN_MAGIC {
        return Err(PlanDecodeError::BadMagic);
    }
    if reader.u16()? != REGION_PLAN_VERSION || reader.u16()? != COMPILER_VERSION {
        return Err(PlanDecodeError::UnsupportedVersion);
    }
    let stage = Stage::from_u8(reader.u8()?).ok_or(PlanDecodeError::InvalidStage)?;
    let signature_hash = reader.u64()?;
    let signature_len = reader.u32()?;
    let signature_len = length_with_tail(&reader, signature_len, 0, "stage signature")?;
    let signature = reader.take(signature_len)?;
    if pie_ir::fnv1a64(signature) != signature_hash {
        return Err(PlanDecodeError::InvalidRecord);
    }

    let channels = reader.u32()?;
    let channel_count = reader.bounded_count(channels, 4, MAX_CHANNELS, "plan channel table")?;
    reader.take(
        channel_count
            .checked_mul(4)
            .ok_or(PlanDecodeError::CountTooLarge("plan channel table"))?,
    )?;

    let names = reader.u32()?;
    let name_count = reader.bounded_count(names, 2, u16::MAX as usize + 1, "plan name table")?;
    for _ in 0..name_count {
        let length = reader.u16()? as usize;
        reader.take(length)?;
    }

    let operations = reader.u32()?;
    let operation_count =
        reader.bounded_count(operations, 12, MAX_OPS, "normalized operation table")?;
    let mut result_count = 0u32;
    for _ in 0..operation_count {
        let raw_op_len = reader.u32()?;
        let op_len = length_with_tail(&reader, raw_op_len, 4, "normalized operation payload")?;
        result_count = result_count
            .checked_add(scan_planned_op(reader.take(op_len)?)?)
            .ok_or(PlanDecodeError::CountTooLarge("plan value table"))?;
        let sources = reader.u32()?;
        let source_count = reader.bounded_count(sources, 4, MAX_OPS, "operation source map")?;
        reader.take(
            source_count
                .checked_mul(4)
                .ok_or(PlanDecodeError::CountTooLarge("operation source map"))?,
        )?;
    }

    let values = reader.u32()?;
    let structural_values = usize::try_from(result_count)
        .map_err(|_| PlanDecodeError::CountTooLarge("plan value table"))?;
    let value_count = reader.bounded_count(values, 3, structural_values, "plan value table")?;
    if value_count != structural_values {
        return Err(PlanDecodeError::InvalidRecord);
    }
    for _ in 0..value_count {
        if reader.u8()? > DType::Bool as u8 {
            return Err(PlanDecodeError::InvalidRecord);
        }
        let rank = reader.u8()?;
        let rank = reader.bounded_count(rank as u32, 2, MAX_RANK, "symbolic type dimensions")?;
        for _ in 0..rank {
            match reader.u8()? {
                0 => {
                    reader.u32()?;
                }
                1 => {
                    if reader.u8()? > SymbolicExtent::KeyLen as u8 {
                        return Err(PlanDecodeError::InvalidRecord);
                    }
                }
                _ => return Err(PlanDecodeError::InvalidRecord),
            }
        }
        if reader.u8()? > ValueDomain::EffectToken as u8 {
            return Err(PlanDecodeError::InvalidRecord);
        }
    }

    let singleton_regions = scan_partition(
        &mut reader,
        PartitionKind::Singleton,
        operation_count,
        value_count,
        channel_count,
    )?;
    let fused_regions = scan_partition(
        &mut reader,
        PartitionKind::Fused,
        operation_count,
        value_count,
        channel_count,
    )?;
    if reader.offset() != bytes.len() {
        return Err(PlanDecodeError::InvalidRecord);
    }
    Ok(EncodedPlanHeader {
        stage,
        signature_hash,
        singleton_regions,
        fused_regions,
    })
}
