//! `emit_scan_region_cuda`: the `cumsum`/`cumprod` library region. Float addition is not associative, so the fold stays sequential; parallelism is one row per thread. Scanned in the operand's dtype.

use crate::codegen::error::{EmitError, EmitterKind, RegionForm, ValueLayoutSite};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;

/// The kernel body, after the three `constexpr`s the emitter writes: the
/// operand value id, the result value id, and which fold this is.
///
/// No `__syncthreads()`: the row loop's bound differs per thread, so a
/// barrier inside it would be UB, and nothing here needs one.
const BODY: &str = r#"
  // Bound to the fused ABI, and this kernel reads only part of it.
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)lane_active;
  (void)status;
  (void)temporary;

  const M1ValueDesc scan_desc = descriptors[kInput];
  const m1_u8* scan_input = scratch + offsets[kInput];
  m1_u8* scan_output = scratch + offsets[kOutput];
  const m1_u32 scan_len = scan_desc.last;
  const m1_u32 scan_rows = scan_desc.rows;

  for (m1_u32 scan_row = threadIdx.x; scan_row < scan_rows;
       scan_row += blockDim.x) {
    const m1_u32 scan_base = scan_row * scan_len;
    float accumulated_f = kIsSum ? 0.0f : 1.0f;
    m1_u32 accumulated_u = kIsSum ? 0u : 1u;
    int accumulated_i = kIsSum ? 0 : 1;
    for (m1_u32 column = 0u; column < scan_len; ++column) {
      const m1_u32 index = scan_base + column;
      if (scan_desc.dtype == 1u) {
        const int value = m1_load_i(scan_input, index, scan_desc.dtype);
        accumulated_i = kIsSum
                            ? (int)((m1_u32)accumulated_i + (m1_u32)value)
                            : (int)((m1_u32)accumulated_i * (m1_u32)value);
        m1_store_i(scan_output, index, accumulated_i);
      } else if (scan_desc.dtype == 2u) {
        const m1_u32 value = m1_load_u(scan_input, index, scan_desc.dtype);
        accumulated_u = kIsSum ? accumulated_u + value
                               : accumulated_u * value;
        m1_store_u(scan_output, index, accumulated_u);
      } else {
        const float value = m1_load_f(scan_input, index, scan_desc.dtype);
        accumulated_f = kIsSum ? accumulated_f + value
                               : accumulated_f * value;
        m1_store_f(scan_output, index, accumulated_f);
      }
    }
  }
"#;

/// Whether `region` is the single-node scan library region this emitter
/// serves. The tag is re-read off the node rather than trusted from
/// `region.kind`, since a mislabelled region would otherwise be emitted
/// here over the wrong operand.
pub fn is_scan_region(stage: &CompiledStage, region: &Region) -> bool {
    if region.kind != RegionKind::Library(LibraryOp::Scan) || region.nodes.len() != 1 {
        return false;
    }
    let Some(op) = stage.normalized.ops.get(region.nodes[0].index()) else {
        return false;
    };
    let view = OpView::of(op);
    matches!(view.tag, tags::CUMSUM | tags::CUMPROD) && view.args.len() == 1 && view.results == 1
}

/// `emit_scan_region_cuda`.
///
/// # Errors
///
/// [`EmitError::EntryNameNotCIdentifier`] when the entry is not spellable in
/// C, [`EmitError::LibraryRegionAbiInvalid`] when the region is not the
/// single-node `cumsum`/`cumprod` this emitter serves, and the
/// plan-well-formedness refusals `validate_generated_region` raises.
pub fn emit_scan_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaScan));
    }
    if !is_scan_region(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::CudaScan));
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::CudaScan)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let node = region.nodes[0].index();
    let scan = &ops[node];
    if bases[node] as usize >= stage.normalized.value_types.len() {
        return Err(EmitError::RegionNodeOutOfRange(RegionForm::CudaScan));
    }

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);
    let _ = writeln!(source, "  constexpr m1_u32 kInput = {}u;", scan.args[0]);
    let _ = writeln!(source, "  constexpr m1_u32 kOutput = {}u;", bases[node]);
    let _ = writeln!(
        source,
        "  constexpr bool kIsSum = {};",
        scan.tag == tags::CUMSUM
    );
    source.push_str(BODY);
    source.push_str("}\n");
    Ok(source)
}
