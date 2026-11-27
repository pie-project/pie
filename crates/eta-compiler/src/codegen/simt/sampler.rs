//! A decode lane's sampler region — the `NucleusSample` (top-p) and
//! `RankSample` (top-k) library dataflows, or a lone `LaneCut` — as one kernel whose lane runs on
//! several blocks (`runtime/cuda/sample_lane.cuh`): the launcher gives such a
//! region as many blocks a lane as the device holds at once with the fire's
//! lanes (`rows_per_lane`), and the kernel's blocks split each row.

use alloc::format;
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
/// The lane arenas' geometry, which a sampler and a split select share.
pub(super) const LANE_ARENA: &str = include_str!("../../../runtime/cuda/lane_arena.cuh");

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
    /// The bf16 draw (a gumbel_argmax over the logits, f32 or bf16, at a constant
    /// temperature, its cut's probabilities a bf16 softmax), where the
    /// region is one.
    pub draw: Option<Draw>,
}

/// What the bf16 draw reads beside the values: its scale (the value of the
/// temperature's reciprocal, read from scratch as the stage holds it), and
/// the model output whose row the logits are a view of (its node and its
/// slot in the lane's table), which the draw reads where it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Draw {
    pub scale: u32,
    pub output: Option<(u32, u16)>,
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
    let mut divisor = None;
    for node in &region.nodes {
        let index = node.index();
        let op = ops.get(index)?;
        match op.tag {
            tags::REDUCE_MAX => logits = op.args.first().copied(),
            tags::PIVOT_THRESHOLD => {
                pivot = Some((
                    op.args.first().copied()?,
                    op.pred_payload,
                    op.pred_tag,
                    bases[index],
                ))
            }
            tags::RNG_KEYED => state = op.args.first().copied(),
            tags::REDUCE_ARGMAX => token = Some(bases[index]),
            tags::GUMBEL_ARGMAX if op.args.len() == 4 => {
                token = Some(bases[index]);
                state = Some(op.args[3]);
                divisor = Some(op.args[2]);
            }
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
            draw: None,
        });
    }
    if rank != (library == LibraryOp::RankSample) {
        return None;
    }
    let logits = logits?;
    let draw = match divisor {
        None => None,
        Some(divisor) => {
            let mut producers = alloc::vec![u32::MAX; stage.normalized.value_types.len()];
            for (node, op) in ops.iter().enumerate() {
                for result in 0..op.results {
                    if let Some(slot) = producers.get_mut((bases[node] + result) as usize) {
                        *slot = node as u32;
                    }
                }
            }
            let output = super::stream::output_source(stage, &ops, &producers, logits).map(|out| {
                let node = producers[out as usize];
                (
                    node,
                    crate::codegen::op_view::output_slot(stage, &ops[node as usize]),
                )
            });
            Some(Draw {
                scale: divisor,
                output,
            })
        }
    };
    Some(SamplerValues {
        logits,
        threshold,
        state: state?,
        probs,
        keep,
        token: token?,
        rank,
        cut_only: false,
        draw,
    })
}

/// The results of a sampler region's nodes that its kernel never touches:
/// it reads and writes its [`SamplerValues`] alone, so a value its nodes
/// compute between them that is none of those, and that nothing past the
/// region reads, is neither written nor read.
#[must_use]
pub fn untouched_values(stage: &CompiledStage, region: &Region) -> alloc::vec::Vec<u32> {
    let mut untouched = alloc::vec::Vec::new();
    let Some(values) = sampler_values(stage, region) else {
        return untouched;
    };
    let touched = [
        values.logits,
        values.threshold,
        values.state,
        values.probs,
        values.keep,
        values.token,
    ];
    let ops: alloc::vec::Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    for node in &region.nodes {
        let index = node.index();
        let (Some(op), Some(&base)) = (ops.get(index), bases.get(index)) else {
            continue;
        };
        for value in base..base.saturating_add(op.results) {
            let read_past = region.outputs.contains(&value)
                || region.sinks.iter().any(|sink| sink.value == value);
            if !touched.contains(&value) && !read_past {
                untouched.push(value);
            }
        }
    }
    untouched
}

pub fn emit_sampler_region(
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
    source.push_str(dialect.lane);
    source.push_str(LANE_ARENA);
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
        draw,
    } = values;
    source.push_str(
        "
  M1Lane g;
  g.slice = lane_row;
  g.slices = lane_blocks;
  g.arena = lane_arenas + (m1_u64)dispatch_lane * M1_LANE_SPAN(lane_blocks);
  g.stride = M1_LANE_ARENA_STRIDE;
  g.parity = 0u;
  g.arrivals = 0u;
  g.counted = false;
",
    );
    source.push_str(dialect.cut_mark);
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
    if let Some(Draw { scale, output }) = draw {
        // The logits where the model left them (its output's row), or in
        // scratch.
        let row = match output {
            Some((node, slot)) => {
                let pitch = super::fused::output_pitch(stage);
                format!(
                    "
  const m1_u32 draw_index = dispatch_lane * {pitch}u + {slot}u;
  const m1_u32 draw_width = intrinsic_widths[draw_index];
  const M1DrawRow draw_row{{
      reinterpret_cast<const m1_u8*>(intrinsic_bases[draw_index]),
      intrinsic_modes[draw_index],
      intrinsic_strides[draw_index] == 0u ? draw_width : intrinsic_strides[draw_index],
      (m1_u64)intrinsic_offsets[draw_index] + (m1_u64)params[{node}u].imm2,
      intrinsic_rows[draw_index]}};"
                )
            }
            None => format!(
                "
  const M1DrawRow draw_row{{scratch + offsets[{logits}], descriptors[{logits}u].dtype == M1_DT_BF16 ? 1u : 0u, descriptors[{logits}u].last, 0u, descriptors[{logits}u].rows}};"
            ),
        };
        source.push_str(&row);
        let _ = write!(
            source,
            "
  const M1ValueDesc logits_desc = descriptors[{logits}u];
  ptir_lane_draw<{rank}>(
      g,
      draw_row,
      m1_load_f(scratch + offsets[{scale}], 0u, descriptors[{scale}u].dtype),
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
