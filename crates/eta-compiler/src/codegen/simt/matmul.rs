use crate::codegen::error::{EmitError, EmitterKind, RegionForm, ValueLayoutSite};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::{Dialect, runtime_source};
use super::singleton::valid_identifier;

// Each thread owns output elements and sums its contraction in the
// reference's order: from zero, ascending `k`, a zero left factor skipped.
const BODY: &str = r#"
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)lane_active;
  (void)temporary;

  const M1ValueDesc left_desc = descriptors[kLeft];
  const M1ValueDesc right_desc = descriptors[kRight];
  const m1_u8* left = scratch + offsets[kLeft];
  const m1_u8* right = scratch + offsets[kRight];
  m1_u8* product = scratch + offsets[kOutput];
  if (left_desc.rank != 2u || right_desc.rank != 2u ||
      left_desc.dims[1] != right_desc.dims[0]) {
    if (threadIdx.x == 0u) {
      m1_fault(&status, 0x55u);
      *commit = 0u;
    }
    return;
  }
  const m1_u32 rows = left_desc.dims[0];
  const m1_u32 inner = left_desc.dims[1];
  const m1_u32 columns = right_desc.dims[1];
  for (m1_u32 element = threadIdx.x; element < rows * columns;
       element += blockDim.x) {
    const m1_u32 row = element / columns;
    const m1_u32 column = element % columns;
    float sum = 0.0f;
    for (m1_u32 k = 0u; k < inner; ++k) {
      const float factor = m1_load_f(left, row * inner + k, left_desc.dtype);
      if (factor == 0.0f) continue;
      sum = sum + factor * m1_load_f(right, k * columns + column, right_desc.dtype);
    }
    m1_store_f(product, element, sum);
  }
"#;

pub fn is_matmul_region(stage: &CompiledStage, region: &Region) -> bool {
    if region.kind != RegionKind::Library(LibraryOp::MatMul) || region.nodes.len() != 1 {
        return false;
    }
    let Some(op) = stage.normalized.ops.get(region.nodes[0].index()) else {
        return false;
    };
    let view = OpView::of(op);
    view.tag == tags::MATMUL && view.args.len() == 2 && view.results == 1
}

pub fn emit_matmul_region(
    dialect: Dialect,
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaMatMul));
    }
    if !is_matmul_region(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::CudaMatMul));
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::CudaMatMul)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let node = region.nodes[0].index();
    let matmul = &ops[node];
    if bases[node] as usize >= stage.normalized.value_types.len() {
        return Err(EmitError::RegionNodeOutOfRange(RegionForm::CudaMatMul));
    }

    let mut source = runtime_source(dialect);
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);
    let _ = writeln!(source, "  constexpr m1_u32 kLeft = {}u;", matmul.args[0]);
    let _ = writeln!(source, "  constexpr m1_u32 kRight = {}u;", matmul.args[1]);
    let _ = writeln!(source, "  constexpr m1_u32 kOutput = {}u;", bases[node]);
    source.push_str(BODY);
    source.push_str("}\n");
    Ok(source)
}
