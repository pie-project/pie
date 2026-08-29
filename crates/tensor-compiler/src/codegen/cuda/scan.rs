//! `emit_scan_region_cuda` — the `cumsum` / `cumprod` library region.
//!
//! The partitioner cuts a scan into a region of its own
//! ([`RegionKind::Library(LibraryOp::Scan)`]) for the same reason it cuts
//! `top_k`: an inclusive prefix is a schedule barrier. This backend refused it,
//! and because the CUDA shell runs compiled regions and nothing else the
//! refusal was fatal at registration — `locally-typical-sampling` and
//! `tail-free-sampling` both cut their candidate set with
//! `cumsum(p) - p`, and `mtp-speculative-decoding` builds its accept prefix
//! with `cumprod`, so all three failed with `stage 0 region N was declined by
//! the emitter (generated region contains a non-generated boundary (scan))`.
//!
//! # THE SCAN IS SEQUENTIAL, AND THAT IS THE WHOLE DESIGN
//!
//! The reference — `engine::program`'s `tags::CUMSUM | tags::CUMPROD`, and the
//! `ptir_m1_execute` block this replaces — is a strict left-to-right fold per
//! row: `acc = acc ⊕ x[j]`, `acc` starting at the identity. Every parity gate
//! over this plane compares BYTES, not tolerances, and the device compiles
//! with `--fmad=false --prec-div=true --prec-sqrt=true` because bit-for-bit
//! reproducibility is the channel plane's first contract.
//!
//! Floating-point addition is not associative, so **no parallel scan can
//! reproduce that fold**. The usual chunked form — each worker folds its chunk
//! from the identity, then a carry is added on — computes
//! `(fold(c₀) ⊕ fold(c₁)) ⊕ …` where the reference computes one unbroken
//! chain, and the two round differently. Re-folding each chunk from its true
//! carry would fix the rounding, but obtaining the true carry at a chunk
//! boundary *is* the sequential chain, so nothing is saved. The only exact
//! parallel case is the integer one, where the wrapping arithmetic really is
//! associative — and specialising on dtype to buy a path the ranking guests
//! never take is complexity for nothing.
//!
//! So the fold stays sequential, and the parallelism this kernel does take is
//! the one that costs no rounding: **one row per thread**. A rank-2 scan over
//! `rows` rows runs `min(rows, blockDim.x)` folds at once; a rank-1 scan runs
//! one. That is strictly better than the single-thread runtime block it
//! replaces (which walked every row on one lane), and for the single-row case
//! it is already at the floor: the dependent chain of `len` FADDs is what a
//! left-to-right fold *is*, and no bit-exact implementation can be shorter.
//! The loads are sequential and independent of the accumulator, so they
//! pipeline; only the arithmetic serialises.
//!
//! In practice the rows are short. Both ranking guests scan their `k_max`
//! candidate row (128 by default), not the vocabulary, and the speculative
//! decoder's `cumprod` is over its draft length.
//!
//! # Dtype
//!
//! Scanned in the OPERAND's dtype, which the result type equals
//! (`infer.rs` pushes the operand type unchanged). This is transcribed from
//! the runtime block rather than simplified: a `u32` offset scan is exactly
//! what ragged row offsets are built from, and accumulating one through
//! `float` is exact only below 2^24 and silently is not above it — which is
//! [`Op::CumSum`]'s own stated reason for not being F32-only.
//!
//! **AND IT DISAGREES WITH THE HOST INTERPRETER, WHICH IS WORTH WRITING DOWN.**
//! `engine::program`'s scan is still the pre-widening form: it takes
//! `lanes_f32()` and publishes `Value::F32` whatever the operand's dtype was.
//! Both device runtimes widened and it did not, so an INTEGER scan is a
//! host/device divergence — one that could not be observed here until now,
//! because a scan region on this backend did not run at all. Following the
//! runtime block is the right side to be on (it is what `ptir_m1_execute` and
//! Metal do, and what the op's contract says), so this kernel does, and
//! `program_parity`'s `scan_prefix` subject stays F32 — the dtype every guest
//! that reaches a scan actually uses, and the only one the reference agrees
//! about. Closing the gap belongs in `engine::program`, not here.
//!
//! [`Op::CumSum`]: tensor_ir::op::Op::CumSum
//!
//! [`RegionKind::Library(LibraryOp::Scan)`]: crate::plan::LibraryOp::Scan

use crate::codegen::error::{EmitError, EmitterKind, RegionForm, ValueLayoutSite};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use tensor_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;

/// The kernel body, after the three `constexpr`s the emitter writes: the
/// operand value id, the result value id, and which fold this is.
///
/// No `__syncthreads()` anywhere, deliberately: the row loop's bound differs
/// per thread, so a barrier inside it would be reached by some threads and not
/// others, which is undefined behaviour rather than a slow path. Nothing here
/// needs one — every thread writes only the row it folded, and the region
/// boundary is a kernel boundary.
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
/// serves, with the operand and result the op table promises.
///
/// A library CLAIM is not evidence, so the tag is re-read off the node rather
/// than taken from `region.kind`: a region mislabelled `Library(Scan)` around
/// ordinary ops would otherwise be emitted here as a prefix over the wrong
/// operand, which is precisely the failure `validate_generated_region`'s
/// honesty check exists to stop.
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
