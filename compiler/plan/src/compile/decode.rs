//! Reading plan bytes back.
//!
//! The mirror of [`super::encode`]. This parses input the process did not
//! necessarily write, so every count and length goes through
//! [`pie_ir::read::Reader`]'s two bounds rather than being trusted.

use pie_ir::container::{MAX_CHANNELS, MAX_OPS};
use pie_ir::op::{IntrinsicId, tags};
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

/// Validates one planned-op record and reports how many results it defines.
///
/// The payload *layout* below is the plan encoding's own business (it is not
/// the trace container's), but the two facts this used to re-derive by hand —
/// which tags exist, and how many results each defines — belong to
/// [`pie_ir::op::OP_TABLE`]. Both are now read from there, so a new op cannot
/// be given a second, disagreeing result count here. What remains hand-written
/// is the payload scan, and `planned_op_scan_covers_every_op` pins that no
/// declared tag falls through to the catch-all.
pub(crate) fn scan_planned_op(bytes: &[u8]) -> Result<u32, PlanDecodeError> {
    let mut reader = Reader::new(bytes);
    let tag = reader.u8()?;
    let spec = pie_ir::op::spec(tag).ok_or(PlanDecodeError::InvalidRecord)?;
    match tag {
        tags::EXP
        | tags::LOG
        | tags::NEG
        | tags::RECIP
        | tags::ABS
        | tags::SIGN
        | tags::NOT
        | tags::REDUCE_SUM
        | tags::REDUCE_MAX
        | tags::REDUCE_MIN
        | tags::REDUCE_ARGMAX
        | tags::TRANSPOSE
        | tags::CUMSUM
        | tags::CUMPROD
        | tags::SORT_DESC
        | tags::IOTA
        | tags::CHAN_TAKE
        | tags::CHAN_READ => {
            reader.take(4)?;
        }
        tags::CAST => {
            reader.take(4)?;
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
        }
        tags::ADD
        | tags::SUB
        | tags::MUL
        | tags::DIV
        | tags::MAX_ELEM
        | tags::MIN_ELEM
        | tags::GT
        | tags::GE
        | tags::EQ
        | tags::NE
        | tags::LT
        | tags::LE
        | tags::AND
        | tags::OR
        | tags::REM
        | tags::TOP_K
        | tags::MATMUL
        | tags::GATHER
        | tags::GATHER_ROW
        | tags::MASK_APPLY_PACKED
        | tags::CAUSAL_MASK => {
            reader.take(8)?;
        }
        tags::SELECT | tags::SCATTER_ADD | tags::SCATTER_SET | tags::SLIDING_WINDOW_MASK => {
            reader.take(12)?;
        }
        tags::SINK_WINDOW_MASK => {
            reader.take(16)?;
        }
        tags::BROADCAST | tags::RESHAPE => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
        }
        tags::PIVOT_THRESHOLD => {
            reader.take(4)?;
            if reader.u8()? > 2 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
        }
        tags::RNG | tags::RNG_KEYED => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
            if reader.u8()? > 1 {
                return Err(PlanDecodeError::InvalidRecord);
            }
        }
        tags::CONST => {
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
        }
        tags::CHAN_PUT => {
            reader.take(8)?;
        }
        tags::INTRINSIC_VAL => {
            if reader.u16()? > IntrinsicId::AttnScore as u16 || reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            scan_plan_shape(&mut reader)?;
        }
        tags::KERNEL_CALL => {
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
        }
        tags::SINK_CALL => {
            reader.u16()?;
            let arguments = reader.u8()? as u32;
            let arguments =
                reader.bounded_count(arguments, 4, u8::MAX as usize, "sink argument vector")?;
            reader.take(
                arguments
                    .checked_mul(4)
                    .ok_or(PlanDecodeError::CountTooLarge("sink argument vector"))?,
            )?;
        }
        // Unreachable for undeclared tags — `spec` above already rejected
        // those. Reached only by a tag `declare_ops!` added and this scan did
        // not, which `planned_op_scan_covers_every_op` fails on.
        _ => return Err(PlanDecodeError::InvalidRecord),
    }
    if reader.offset() != bytes.len() {
        return Err(PlanDecodeError::InvalidRecord);
    }
    Ok(spec.results as u32)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `scan_planned_op` ends in a catch-all, and a catch-all cannot tell the
    /// difference between "this tag does not exist" and "nobody taught me
    /// this tag". `pie_ir::op::spec` handles the first case, so the second is
    /// the only way to reach `_`, and this is what notices.
    ///
    /// A one-byte record is the probe: every payload arm reads at least one
    /// more byte, so a tag the scan knows fails as `Truncated`, while a tag
    /// it forgot fails as `InvalidRecord`.
    #[test]
    fn planned_op_scan_covers_every_op() {
        let unhandled: Vec<&str> = pie_ir::op::OP_TABLE
            .iter()
            .filter(|spec| scan_planned_op(&[spec.tag]) == Err(PlanDecodeError::InvalidRecord))
            .map(|spec| spec.name)
            .collect();
        assert!(
            unhandled.is_empty(),
            "{} op(s) reach the catch-all in scan_planned_op: {unhandled:?}",
            unhandled.len()
        );
    }

    /// The dual: the plan format must not admit a tag `declare_ops!` never
    /// allocated. Without the `spec` lookup an undeclared tag could land in a
    /// range pattern and be scanned as its neighbour.
    #[test]
    fn planned_op_scan_rejects_undeclared_tags() {
        for tag in 0u8..=u8::MAX {
            if pie_ir::op::spec(tag).is_some() {
                continue;
            }
            assert_eq!(
                scan_planned_op(&[tag]),
                Err(PlanDecodeError::InvalidRecord),
                "tag {tag:#04x} is not in OP_TABLE but scan_planned_op accepted it"
            );
        }
    }
}
