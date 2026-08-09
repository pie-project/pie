//! The gpt-oss M=1 step: bound once, fired per token.
//!
//! Three of the four bind passes are the SHARED ones — the Go* kinds
//! carry their weight names in `weight_binds`, the sink attention reads
//! the ring at the plain SDPA's slots, and the all-full-attention view
//! ([`gptoss_decode_geometry`]) gives every layer its KV pair — so this
//! step owns only the assembly order and the family's consts walk. What
//! it must NOT forget is the YaRN table's keepalive: the rope dispatches
//! hold its GPU address in their argument tables, and a dropped handle is
//! a use-after-free the fault reporter attributes to an innocent rope.

use std::collections::HashMap;
use std::path::Path;

use crate::batch::{
    Dispatch, GptOssGeometry, GptOssSlot, Kernel, MbSlot, ScratchSchedule, build_gptoss_dag,
    build_gptoss_dag_mb, gptoss_decode_geometry, plan_gptoss_psos, qmm_bm_slot,
};
use crate::shader::Request;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_decode_dag, bind_scratch, encode_decode_step};
use super::bind_mb::{MbBindOffsets, bind_decode_dag_mb};
use super::context::Context;
use super::encoder::Stepper;
use super::gptoss_bind::bind_gptoss_consts;
use super::handle::Handle;
use super::pipeline::Compiler;
use super::program::Pso;
use super::step_mb::MbPsos;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// One bound gpt-oss decode step.
#[derive(Debug)]
pub struct GptOssStep {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The compiled pipelines (from `gptoss_step_plan`).
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The YaRN table the rope tables point into.
    _freqs: Handle,
}

impl GptOssStep {
    /// Build the DAG, bind everything, and hold the result ready.
    ///
    /// The passes run in the shared step's order — weights/state/IO,
    /// scratch, constants — so a prepared step is never half-bound.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover this
    /// DAG.
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &GptOssGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_gptoss_dag(g, true);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "gpt-oss step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = gptoss_decode_geometry(g);
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag(context, &mut tables, storage, &dag, &shared, false)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        let freqs = bind_gptoss_consts(
            context,
            &mut tables,
            &mut consts,
            &dag,
            g,
            tuning,
            max_ctx,
            1,
            0,
        )?;
        Ok(GptOssStep {
            dag,
            tables,
            consts,
            psos,
            force_barriers: false,
            _freqs: freqs,
        })
    }

    /// Encode and run the whole DAG as one command buffer.
    ///
    /// # Errors
    ///
    /// A kind with no compiled pipeline, or a command buffer that does
    /// not retire clean.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<Timing> {
        self.fire_prefix(stepper, self.dag.len())
    }

    /// Encode and run only `[0, end)` of the DAG — the bisect's stage
    /// probe: with the ordinary recycled pool, the LAST dispatch's output
    /// slot still holds its value when the prefix retires, so a truncated
    /// fire reads any stage without the no-recycle allocation.
    ///
    /// # Errors
    ///
    /// As [`fire`](Self::fire).
    pub fn fire_prefix(&self, stepper: &mut Stepper<'_>, end: usize) -> Result<Timing> {
        stepper.run(|encoder| {
            encode_decode_step(
                encoder,
                &self.tables,
                &self.dag,
                &self.psos,
                self.force_barriers,
                0,
                end,
            )
        })
    }
}

/// The family's own slot-keyed MB pipelines: the routed MXFP4 GEMM
/// lattice. Kept by slot, not kind, because which entry a routed dispatch
/// runs is the `qmm_bn`/`qmm_bm` the DAG builder wrote on it.
#[derive(Clone, Debug, Default)]
pub struct GptOssMbPsos {
    /// Slot → pipeline.
    pub by_slot: HashMap<GptOssSlot, Pso>,
}

/// Compile the family's slot-keyed MB pipelines for `g` — empty for the
/// affine bank, whose routed projections keep the matvec at every batch.
///
/// # Errors
///
/// A named instantiation that fails to compile.
pub fn load_gptoss_mb_psos(
    compiler: &Compiler,
    context: &Context,
    kernels_dir: &Path,
    g: &GptOssGeometry,
) -> Result<GptOssMbPsos> {
    let plan: Vec<_> = plan_gptoss_psos(g)
        .into_iter()
        .filter(|r| matches!(r.slot, GptOssSlot::QmmRoutedBias { .. }))
        .collect();
    let requests: Vec<Request> = plan
        .iter()
        .map(|r| Request::new(kernels_dir.join(r.file), r.entry.clone()))
        .collect();
    let mut psos = GptOssMbPsos::default();
    if requests.is_empty() {
        return Ok(psos);
    }
    let pipelines = compiler.compile_batch(context, &requests).all()?;
    for (i, request) in plan.iter().enumerate() {
        psos.by_slot.insert(request.slot, pipelines[i].clone());
    }
    Ok(psos)
}

fn bn_slot(bn: u32) -> usize {
    match bn {
        64 => 2,
        32 => 1,
        _ => 0,
    }
}

/// The pipeline one gpt-oss MB dispatch runs: the three tables the C++
/// juggled, each asked only what it owns — the routed lattice for a tiled
/// mixture dispatch, the shared GEMM lattice for a tiled dense one, the
/// by-kind plan for everything else.
///
/// A dispatch that DECIDED a tile (`qmm_bn > 0`) whose pipeline is
/// missing returns `None` and the fire refuses. Falling back to the
/// matvec would run it under the GEMM's grid — the exact
/// grid-against-wrong-pipeline mismatch the builder exists to prevent,
/// reintroduced at the last table.
#[must_use]
pub fn gptoss_mb_pso<'a>(
    d: &Dispatch,
    base: &'a StepPsos,
    mb: &'a MbPsos,
    go: &'a GptOssMbPsos,
) -> Option<&'a Pso> {
    match d.kind {
        Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoExpertDown if d.qmm_bn > 0 => {
            go.by_slot.get(&GptOssSlot::QmmRoutedBias {
                width: qmm_bm_slot(d.qmm_bm),
                bn: bn_slot(d.qmm_bn),
            })
        }
        Kernel::GoQmvQ | Kernel::GoQmvK | Kernel::GoQmvV | Kernel::GoQmvO if d.qmm_bn > 0 => {
            mb.by_slot.get(&MbSlot::QmmTBias {
                bm: qmm_bm_slot(d.qmm_bm),
                bn: bn_slot(d.qmm_bn),
            })
        }
        // The head is the one unbiased projection.
        Kernel::LmHeadUntied if d.qmm_bn > 0 => mb.by_slot.get(&MbSlot::QmmT {
            bm: qmm_bm_slot(d.qmm_bm),
            bn: bn_slot(d.qmm_bn),
        }),
        _ => base.by_kind.get(&d.kind),
    }
}

/// One bound N-token gpt-oss fire: the paged KV, the sampled-row tail.
#[derive(Debug)]
pub struct GptOssMbStep {
    /// The MB dispatch list, tiles decided.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The by-kind pipelines (from `gptoss_mb_plan`).
    pub base: StepPsos,
    /// The shared dense GEMM lattice (bias feature on).
    pub mb: MbPsos,
    /// The family's routed GEMM lattice.
    pub go: GptOssMbPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The YaRN table the rope tables point into.
    _freqs: Handle,
}

impl GptOssMbStep {
    /// Build the N-token DAG, bind everything, hold the result ready.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover the
    /// MB DAG.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &GptOssGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        base: StepPsos,
        mb: MbPsos,
        go: GptOssMbPsos,
        n_tokens: u32,
        head_rows: u32,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_gptoss_dag_mb(g, tuning, n_tokens, head_rows, 0, false);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "gpt-oss mb step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the MB DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = gptoss_decode_geometry(g);
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag_mb(
            context,
            &mut tables,
            storage,
            &dag,
            &shared,
            false,
            MbBindOffsets::default(),
        )?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        let freqs = bind_gptoss_consts(
            context,
            &mut tables,
            &mut consts,
            &dag,
            g,
            tuning,
            max_ctx,
            n_tokens,
            head_rows,
        )?;
        Ok(GptOssMbStep {
            dag,
            tables,
            consts,
            base,
            mb,
            go,
            force_barriers: false,
            _freqs: freqs,
        })
    }

    /// Encode and run the whole MB DAG as one command buffer.
    ///
    /// # Errors
    ///
    /// A dispatch no table serves — including a decided tile whose
    /// pipeline is missing — or a command buffer that does not retire
    /// clean.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<Timing> {
        self.fire_prefix(stepper, self.dag.len())
    }

    /// [`fire`](Self::fire) over `[0, end)` — the same bisect probe the
    /// M=1 step carries.
    ///
    /// # Errors
    ///
    /// As [`fire`](Self::fire).
    pub fn fire_prefix(&self, stepper: &mut Stepper<'_>, end: usize) -> Result<Timing> {
        stepper.run(|encoder| {
            let run_ends = crate::batch::concurrent_run_ends(&self.dag);
            for (i, d) in self.dag.iter().enumerate().take(end) {
                let pso =
                    gptoss_mb_pso(d, &self.base, &self.mb, &self.go).ok_or(Error::Create {
                        what: "gpt-oss mb pso",
                        message: format!(
                            "no pipeline serves {:?} (bn {}, bm {})",
                            d.kind, d.qmm_bn, d.qmm_bm
                        ),
                    })?;
                encoder.set_pipeline(pso);
                encoder.set_argument_table_for(&self.tables, d.ordinal)?;
                encoder.dispatch(
                    d.launch.grid.map(|v| v as usize),
                    d.launch.tg.map(|v| v as usize),
                )?;
                if self.force_barriers || crate::batch::barrier_after(&self.dag, i, &run_ends) {
                    encoder.barrier(super::encoder::Visibility::Device);
                }
            }
            Ok(())
        })
    }
}
