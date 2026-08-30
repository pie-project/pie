//! `emit_order_region_cuda` — the `Order` library regions, generated.
//!
//! Two ops, one kernel: [`Op::TopK`] and [`Op::SortDesc`]. The plan partitioner
//! cuts each into a region of its own ([`RegionKind::Library`]) because both
//! are schedule barriers — they read a whole row before they can write their
//! first result — and every backend then has to answer for that region. Metal
//! answers `top_k` with [`crate::codegen::metal::emit_grouped_topk`]; this
//! backend used to answer `Slot::Refused` for both, and because the CUDA shell
//! runs compiled regions and nothing else, a refusal was a region that did not
//! run at all — the chain around it read the zeros `Prepared::build` had
//! written and published a confident wrong answer, or, once `build_stage`
//! learned to say so, the whole program failed to register (`beam-search`
//! failed exactly here). So it answers with source now.
//!
//! **ONE EMITTER FOR BOTH, BECAUSE THEY ARE ONE OP.** `sort_desc` IS `top_k`
//! at `k = n`: the host interpreter computes both from the same
//! `sort_desc_order`, and both define the same two results in the same order —
//! values F32, then indices U32. The only difference is how much of the order
//! is written, and that is a number, not a code path. `kWidth` carries it, with
//! [`ORDER_FULL_ROW`] the sentinel for `sort_desc`, whose result width the plan
//! may only know symbolically (a `[vocab]` row is a `Dimension::Symbolic`) and
//! which therefore has to come off the runtime descriptor.
//!
//! # The kernel
//!
//! One block per lane, the fused ABI verbatim — this kernel lands in the same
//! `KernelKind::Fused` slot the shell reads for the region, so its signature
//! is not ours to choose. Per row it runs a **stable LSD radix sort of the
//! whole row** on [`m1_desc_key`], eight passes of four bits, and then writes
//! the first `kWidth` entries of the resulting order to the two results.
//!
//! `m1_desc_key` is the runtime's own monotone `float -> u32` map that
//! REVERSES value order, sends NaN to `0xFFFFFFFF` (which no finite float
//! reaches, so NaNs sort last as a group) and normalises `-0.0` to `+0.0`.
//! Sorting ascending on it therefore walks values descending, and because the
//! sort is stable and starts from the identity order, equal values come out in
//! ascending index order — which is exactly [`Op::TopK`]'s "ties → lower
//! index" and exactly what `metal::topk`'s nine-pass form produces. (Metal
//! spends a ninth pass separating NaNs; the `0xFFFFFFFF` sentinel does that
//! work here in the same eight.)
//!
//! # Why a full sort and not a selection
//!
//! Cost is `O(8·len)` **regardless of `k`**, which is the property that
//! matters — and it is also what makes `sort_desc` free once `top_k` exists.
//! `k` is not small in practice: `beam-search` asks for the beam width (2–3),
//! but `locally-typical-sampling` and `tail-free-sampling` pass a `k_max`
//! candidate bound that a caller may set anywhere up to the vocabulary, and
//! their own module docs record that the `O(k·vocab)` shape — rescan the row
//! once per pick — is what made an earlier ranking kernel "a performance knob
//! with teeth" at a 151,936-token vocabulary. A selection loop would
//! reintroduce exactly that, and would need a `k` ceiling refused by name to
//! stay honest. A sort needs no ceiling: there is no `k` this emitter declines.
//!
//! # What it costs in scratch
//!
//! Two `u32` order arrays over one row, taken from the fire's `temporary`
//! arena. `engine::program::scratch::layout` sizes that arena at
//! `4 · sizeof(u32)` per element of the WIDEST value in the stage, and the
//! ranked input is one of those values, so `2 · sizeof(u32) · last` is inside
//! it with a factor of two to spare — the same budget `metal::topk` spends,
//! and the reason both can afford an out-of-place counting sort.
//!
//! [`RegionKind::Library`]: crate::plan::RegionKind::Library
//! [`Op::TopK`]: eta_ir::op::Op::TopK
//! [`Op::SortDesc`]: eta_ir::op::Op::SortDesc
//! [`m1_desc_key`]: crate::codegen::cuda::runtime

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

/// `kCudaOrderWorkerCap` — how many threads take part in the counting sort.
///
/// The block width is the device's, not ours: `program::compile`'s
/// `launch_width` reads `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` and rounds
/// it down to a power of two anywhere in `[32, 1024]`. The per-worker digit
/// table is `workers × 16` `u32`s of SHARED memory, so letting every thread be
/// a worker would ask for 64 KiB at a 1024-wide block — past the 48 KiB a
/// kernel gets without an opt-in. Capping the workers instead costs occupancy
/// and nothing else: threads past the cap still run every barrier, they simply
/// own no chunk of the row.
const ORDER_WORKER_CAP: u32 = 256;

/// `kWidth`'s sentinel for "as wide as the row is at run time".
///
/// A NAMED value and not a bare `0` at the two sites that spell it, because it
/// is a claim: `top_k` refuses `k < 1` in `infer.rs`, so zero is a width no
/// legitimate `top_k` region can ask for and the sentinel cannot collide with
/// one. `sort_desc` is the only op that reaches it.
const ORDER_FULL_ROW: u32 = 0;

/// The kernel body, after the six `constexpr`s the emitter writes: the worker
/// cap, the input value id, the two result value ids, the result width, and
/// the full-row sentinel.
///
/// Every `__syncthreads()` here is reached by the whole block: the loops are
/// over uniform bounds and the guards inside them (`order_worker`,
/// `threadIdx.x < kOrderDigits`) never contain one. A barrier inside a
/// divergent branch is undefined behaviour on CUDA, not a slow path.
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

  constexpr m1_u32 kOrderDigits = 16u;
  constexpr m1_u32 kOrderPasses = 8u;
  __shared__ m1_u32 order_offsets[kOrderWorkerCap * kOrderDigits];
  __shared__ m1_u32 order_digit_total[kOrderDigits];
  __shared__ m1_u32 order_digit_base[kOrderDigits];

  const M1ValueDesc order_input_desc = descriptors[kInput];
  const m1_u8* order_input = scratch + offsets[kInput];
  m1_u8* order_values = scratch + offsets[kValues];
  m1_u8* order_indices = scratch + offsets[kIndices];
  const m1_u32 order_len = order_input_desc.last;
  const m1_u32 order_rows = order_input_desc.rows;
  // `sort_desc` writes the whole row and the plan may only know that row's
  // width symbolically, so it asks the descriptor rather than a literal; a
  // `top_k`'s `k` is a trace-known immediate and is spliced in as one.
  const m1_u32 order_width = kWidth == kOrderFullRow ? order_len : kWidth;
  // `k <= last` is `top_k`'s own contract (`infer.rs` refuses the trace
  // otherwise), so the clamp only ever matters if a launch package disagreed
  // with the plan it came from; it truncates rather than reading off the row.
  const m1_u32 order_count = order_width < order_len ? order_width : order_len;
  m1_u32* order_a = reinterpret_cast<m1_u32*>(temporary);
  m1_u32* order_b = order_a + order_len;

  // Contiguous chunks, so worker order IS position order and the counting sort
  // comes out stable. A strided assignment would sort the same multiset into a
  // different tie order.
  const m1_u32 order_workers =
      blockDim.x < kOrderWorkerCap ? blockDim.x : kOrderWorkerCap;
  const bool order_worker = threadIdx.x < order_workers;
  const m1_u32 order_begin =
      order_worker
          ? (m1_u32)(((m1_u64)order_len * threadIdx.x) / order_workers)
          : 0u;
  const m1_u32 order_end =
      order_worker
          ? (m1_u32)(((m1_u64)order_len * (threadIdx.x + 1u)) / order_workers)
          : 0u;

  if (order_len != 0u) {
    for (m1_u32 order_row = 0u; order_row < order_rows; ++order_row) {
      const m1_u32 order_base = order_row * order_len;
      for (m1_u32 i = threadIdx.x; i < order_len; i += blockDim.x)
        order_a[i] = i;
      __syncthreads();
      m1_u32* order_in = order_a;
      m1_u32* order_out = order_b;
      for (m1_u32 order_pass = 0u; order_pass < kOrderPasses; ++order_pass) {
        const m1_u32 order_shift = order_pass * 4u;
        m1_u32 order_counts[kOrderDigits];
        m1_u32 order_written[kOrderDigits];
        for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit) {
          order_counts[digit] = 0u;
          order_written[digit] = 0u;
        }
        if (order_worker) {
          for (m1_u32 at = order_begin; at < order_end; ++at) {
            const m1_u32 key = m1_desc_key(m1_load_f(
                order_input, order_base + order_in[at],
                order_input_desc.dtype));
            ++order_counts[(key >> order_shift) & 15u];
          }
          for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit)
            order_offsets[threadIdx.x * kOrderDigits + digit] =
                order_counts[digit];
        }
        __syncthreads();
        // The exclusive scan over (digit, worker) in digit-major order, split
        // sixteen ways rather than run by thread 0 alone: at the cap that is
        // 4096 shared-memory steps per pass on one lane, which dominated the
        // kernel at a real vocabulary.
        if (threadIdx.x < kOrderDigits) {
          m1_u32 total = 0u;
          for (m1_u32 worker = 0u; worker < order_workers; ++worker)
            total += order_offsets[worker * kOrderDigits + threadIdx.x];
          order_digit_total[threadIdx.x] = total;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32 running = 0u;
          for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit) {
            order_digit_base[digit] = running;
            running += order_digit_total[digit];
          }
        }
        __syncthreads();
        if (threadIdx.x < kOrderDigits) {
          m1_u32 running = order_digit_base[threadIdx.x];
          for (m1_u32 worker = 0u; worker < order_workers; ++worker) {
            const m1_u32 slot = worker * kOrderDigits + threadIdx.x;
            const m1_u32 held = order_offsets[slot];
            order_offsets[slot] = running;
            running += held;
          }
        }
        __syncthreads();
        if (order_worker) {
          for (m1_u32 at = order_begin; at < order_end; ++at) {
            const m1_u32 index = order_in[at];
            const m1_u32 key = m1_desc_key(m1_load_f(
                order_input, order_base + index, order_input_desc.dtype));
            const m1_u32 digit = (key >> order_shift) & 15u;
            order_out[order_offsets[threadIdx.x * kOrderDigits + digit] +
                      order_written[digit]] = index;
            ++order_written[digit];
          }
        }
        __syncthreads();
        m1_u32* order_swap = order_in;
        order_in = order_out;
        order_out = order_swap;
      }
      // The result rows are `order_width` wide, not `order_count` wide: the
      // width is the shape the plan laid out, and a clamped count leaves the
      // tail as the zeros the fire wrote rather than shifting the next row up
      // onto it.
      for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
        const m1_u32 index = order_in[at];
        m1_store_f(
            order_values,
            order_row * order_width + at,
            m1_load_f(order_input, order_base + index, order_input_desc.dtype));
        m1_store_u(order_indices, order_row * order_width + at, index);
      }
      __syncthreads();
    }
  }
"#;

/// Whether `region` is one of the single-node `Order` library regions this
/// emitter serves, with the results and operands the op table promises.
///
/// A library CLAIM is not evidence — the fused emitter's `validate` says the
/// same thing at more length — so the tag is re-read off the node rather than
/// taken from `region.kind`. A region mislabelled `Library(TopK)` around some
/// other op would otherwise be emitted here as a ranking of the wrong operand.
pub fn is_order_region(stage: &CompiledStage, region: &Region) -> bool {
    let expected = match region.kind {
        RegionKind::Library(LibraryOp::TopK) => tags::TOP_K,
        RegionKind::Library(LibraryOp::Sort) => tags::SORT_DESC,
        _ => return false,
    };
    if region.nodes.len() != 1 {
        return false;
    }
    let Some(op) = stage.normalized.ops.get(region.nodes[0].index()) else {
        return false;
    };
    let view = OpView::of(op);
    view.tag == expected && view.args.len() == 1 && view.results == 2
}

/// `emit_order_region_cuda`.
///
/// # Errors
///
/// [`EmitError::EntryNameNotCIdentifier`] when the entry is not spellable in
/// C, [`EmitError::LibraryRegionAbiInvalid`] when the region is not one of the
/// single-node `Order` lifts this emitter serves, and the plan-well-formedness
/// refusals `validate_generated_region` raises — an `Order` region sits in a
/// stage whose value table and op list this kernel indexes exactly as the
/// fused one does.
pub fn emit_order_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaOrder));
    }
    if !is_order_region(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::CudaOrder));
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::CudaOrder)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let node = region.nodes[0].index();
    let order = &ops[node];
    // Two results, so the indices value is the id after the values one; both
    // have to exist before either is spliced into a device pointer.
    if bases[node] as usize + 1 >= stage.normalized.value_types.len() {
        return Err(EmitError::RegionNodeOutOfRange(RegionForm::CudaOrder));
    }
    let width = if order.tag == tags::SORT_DESC {
        ORDER_FULL_ROW
    } else {
        order.imm
    };

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kOrderWorkerCap = {ORDER_WORKER_CAP}u;"
    );
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kOrderFullRow = {ORDER_FULL_ROW}u;"
    );
    let _ = writeln!(source, "  constexpr m1_u32 kInput = {}u;", order.args[0]);
    let _ = writeln!(source, "  constexpr m1_u32 kValues = {}u;", bases[node]);
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kIndices = {}u;",
        bases[node] + 1
    );
    let _ = writeln!(source, "  constexpr m1_u32 kWidth = {width}u;");
    source.push_str(BODY);
    source.push_str("}\n");
    Ok(source)
}
