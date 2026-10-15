//! A decode lane's sampler region — the `NucleusSample` (top-p) and
//! `RankSample` (top-k) library dataflows, or a lone `LaneCut` — as one kernel whose lane runs on
//! several blocks (`runtime/cuda/sample_lane.cuh`): the launcher gives such a
//! region as many blocks a lane as the device holds at once with the fire's
//! lanes (`rows_per_lane`), and the kernel's blocks split each row.

use alloc::string::String;
use core::fmt::Write as _;

use eta_ir::op::tags;
use eta_ir::wire::predicate_tags;

use crate::codegen::error::{EmitError, EmitterKind};
use crate::codegen::op_view::{OpView, result_bases};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::{Dialect, runtime_source};
use super::singleton::valid_identifier;

const SAMPLE_LANE: &str = include_str!("../../../runtime/cuda/sample_lane.cuh");

/// How a lane's blocks meet on CDNA: the exchange words at device scope
/// (each XCD's L2 written through and read past, never written back or
/// invalidated whole), a barrier that waits for the stores it publishes.
const HIP_LANE: &str = "
#define M1_LANE_PAUSE() __builtin_amdgcn_s_sleep(1)
#define M1_LANE_LOAD(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_PUT32(p, v) __hip_atomic_store((p), (m1_u32)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_PUT64(p, v) __hip_atomic_store((p), (m1_u64)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_GET32(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_GET64(p) __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#define M1_LANE_RELEASE() asm volatile(\"s_waitcnt vmcnt(0)\" ::: \"memory\")
#define M1_LANE_ACQUIRE() asm volatile(\"\" ::: \"memory\")
";

/// How a lane's blocks meet on a CUDA device: L2-coherent words behind the
/// device-scope fences.
const CUDA_LANE: &str = "
#define M1_LANE_PAUSE() __nanosleep(32)
#define M1_LANE_LOAD(p) (*reinterpret_cast<volatile m1_u32*>(p))
#define M1_LANE_PUT32(p, v) (*reinterpret_cast<volatile m1_u32*>(p) = (m1_u32)(v))
#define M1_LANE_PUT64(p, v) (*reinterpret_cast<volatile m1_u64*>(p) = (m1_u64)(v))
#define M1_LANE_GET32(p) (*reinterpret_cast<volatile m1_u32*>(p))
#define M1_LANE_GET64(p) (*reinterpret_cast<volatile m1_u64*>(p))
#define M1_LANE_RELEASE() __threadfence()
#define M1_LANE_ACQUIRE() __threadfence()
";

/// A sampler block's threads: at most 512, so its walks and plans keep their
/// state in registers (a 1024-thread block holds 128 a lane).
const LAUNCH_BOUNDS: &str = "__launch_bounds__(512) ";

/// The kernel's entry follows the fused prologue's last line.
const ENTRY: &str = "extern \"C\" __global__ void ";

/// The values a sampler region reads and lands, by their stage value ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerValues {
    pub logits: u32,
    pub threshold: u32,
    pub state: u32,
    pub probs: u32,
    pub keep: u32,
    pub token: u32,
    pub rank: bool,
    /// A lone cut (`LaneCut`): `probs` in, `keep` out, the rest unused.
    pub cut_only: bool,
}

/// Whether `region` is a sampler region this emitter takes.
#[must_use]
pub fn is_sampler_region(stage: &CompiledStage, region: &Region) -> bool {
    sampler_values(stage, region).is_some()
}

/// The sampler's values, if `region` is one: a library sampler region whose
/// nodes hold its max, cut, keyed noise and argmax, or a lone lane cut.
#[must_use]
pub fn sampler_values(stage: &CompiledStage, region: &Region) -> Option<SamplerValues> {
    let RegionKind::Library(library) = region.kind else {
        return None;
    };
    if !matches!(
        library,
        LibraryOp::NucleusSample | LibraryOp::RankSample | LibraryOp::LaneCut
    ) {
        return None;
    }
    let ops: alloc::vec::Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let (mut logits, mut pivot, mut state, mut token) = (None, None, None, None);
    for node in &region.nodes {
        let index = node.index();
        let op = ops.get(index)?;
        match op.tag {
            tags::REDUCE_MAX => logits = op.args.first().copied(),
            tags::PIVOT_THRESHOLD => {
                pivot = Some((op.args.first().copied()?, op.pred_payload, op.pred_tag, bases[index]))
            }
            tags::RNG_KEYED => state = op.args.first().copied(),
            tags::REDUCE_ARGMAX => token = Some(bases[index]),
            _ => {}
        }
    }
    let (probs, threshold, predicate, keep) = pivot?;
    let rank = match predicate {
        predicate_tags::RANK_LE => true,
        predicate_tags::CUMMASS_LE => false,
        _ => return None,
    };
    if library == LibraryOp::LaneCut {
        if region.nodes.len() != 1 {
            return None;
        }
        return Some(SamplerValues {
            logits: probs,
            threshold,
            state: probs,
            probs,
            keep,
            token: keep,
            rank,
            cut_only: true,
        });
    }
    if rank != (library == LibraryOp::RankSample) {
        return None;
    }
    Some(SamplerValues {
        logits: logits?,
        threshold,
        state: state?,
        probs,
        keep,
        token: token?,
        rank,
        cut_only: false,
    })
}

pub fn emit_sampler_region_in(
    dialect: Dialect,
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaFused));
    }
    let Some(values) = sampler_values(stage, region) else {
        return Err(EmitError::LibraryRegionAbiInvalid(
            crate::codegen::error::RegionForm::GroupedFused,
        ));
    };
    let Some(cut) = PROLOGUE.rfind(ENTRY) else {
        return Err(EmitError::LibraryRegionAbiInvalid(
            crate::codegen::error::RegionForm::GroupedFused,
        ));
    };
    let mut source = runtime_source(dialect);
    source.push_str(&PROLOGUE[..cut]);
    source.push_str(match dialect {
        Dialect::Hip => HIP_LANE,
        Dialect::Cuda => CUDA_LANE,
    });
    source.push_str(SAMPLE_LANE);
    source.push('\n');
    source.push_str(ENTRY);
    source.push_str(LAUNCH_BOUNDS);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);
    let SamplerValues {
        logits,
        threshold,
        state,
        probs,
        keep,
        token,
        rank,
        cut_only,
    } = values;
    source.push_str(
        "
  M1Lane g;
  g.slice = lane_row;
  g.slices = lane_blocks;
  g.arena = scratch + temporary_offset;
  g.stride = temporary_stride;
  g.parity = 0u;
  g.arrivals = 0u;
",
    );
    if cut_only {
        let _ = write!(
            source,
            "  ptir_lane_cut<{rank}>(
      g,
      scratch + offsets[{probs}],
      descriptors[{probs}u],
      scratch + offsets[{threshold}],
      descriptors[{threshold}u],
      scratch + offsets[{keep}]);
}}
"
        );
        return Ok(source);
    }
    let _ = write!(
        source,
        "
  const M1ValueDesc logits_desc = descriptors[{logits}u];
  ptir_lane_sample<{rank}>(
      g,
      scratch + offsets[{logits}],
      scratch + offsets[{threshold}],
      descriptors[{threshold}u],
      scratch + offsets[{state}],
      descriptors[{state}u],
      scratch + offsets[{probs}],
      scratch + offsets[{keep}],
      scratch + offsets[{token}],
      logits_desc.rows,
      logits_desc.last);
}}
"
    );
    Ok(source)
}
