//! The llama M=1 step: bound once, fired per token.
//!
//! Three of the four bind passes are the SHARED ones — the kinds carry
//! their weight names in `weight_binds`, the attention reads the ring at
//! the plain SDPA's slots, and [`llama_decode_geometry`] puts every
//! layer on the attention path — so this step owns only the assembly
//! order and the family's consts walk, exactly as gpt-oss's does. The
//! llama3 frequency table's keepalive is optional where YaRN's was not:
//! a geometric-series checkpoint has no table, and `None` says so
//! rather than a dummy allocation pretending to be one.

use crate::batch::{
    Dispatch, Kernel, LlamaGeometry, MbSlot, ScratchSchedule, build_llama_dag, build_llama_dag_mb,
    llama_decode_geometry, llama_is_dense_proj, qmm_bm_slot,
};
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_decode_dag, bind_scratch, encode_decode_step};
use super::bind_mb::{MbBindOffsets, bind_decode_dag_mb};
use super::context::Context;
use super::encoder::Stepper;
use super::handle::Handle;
use super::llama_bind::bind_llama_consts;
use super::program::Pso;
use super::step_mb::MbPsos;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// One bound llama decode step.
#[derive(Debug)]
pub struct LlamaStep {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The compiled pipelines (from `llama_step_plan`).
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The llama3 frequency table the rope tables point into, when the
    /// geometry carries one.
    _freqs: Option<Handle>,
}

impl LlamaStep {
    /// Build the DAG, bind everything, and hold the result ready.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover this
    /// DAG.
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &LlamaGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_llama_dag(g, tuning, true);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "llama step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = llama_decode_geometry(g);
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag(context, &mut tables, storage, &dag, &shared, false)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        let freqs = bind_llama_consts(
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
        Ok(LlamaStep {
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
    /// probe, as on the other family steps.
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

/// The pipeline one llama MB dispatch runs: the by-kind plan for
/// everything unbatched, the SHARED GEMM lattice for a decided tile —
/// this family's dense projections are unbiased and its routed matvec is
/// the shared affine one, so unlike gpt-oss it owns no slot table at
/// all. A decided tile whose pipeline is missing refuses, for the reason
/// the gpt-oss arc named: the matvec under the GEMM's grid is wrong
/// numbers, not a crash.
#[must_use]
pub fn llama_mb_pso<'a>(d: &Dispatch, base: &'a StepPsos, mb: &'a MbPsos) -> Option<&'a Pso> {
    fn bn_slot(bn: u32) -> usize {
        match bn {
            64 => 2,
            32 => 1,
            _ => 0,
        }
    }
    match d.kind {
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown if d.qmm_bn > 0 => {
            mb.by_slot.get(&MbSlot::QmmRouted {
                width: qmm_bm_slot(d.qmm_bm),
                bn: bn_slot(d.qmm_bn),
            })
        }
        _ if llama_is_dense_proj(d.kind) && d.qmm_bn > 0 => mb.by_slot.get(&MbSlot::QmmT {
            bm: qmm_bm_slot(d.qmm_bm),
            bn: bn_slot(d.qmm_bn),
        }),
        _ => base.by_kind.get(&d.kind),
    }
}

/// One bound N-token llama fire: the paged KV, the sampled-row tail.
#[derive(Debug)]
pub struct LlamaMbStep {
    /// The MB dispatch list, tiles decided.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The by-kind pipelines (from `llama_mb_plan`).
    pub base: StepPsos,
    /// The shared GEMM lattice.
    pub mb: MbPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The llama3 frequency table, when the geometry carries one.
    _freqs: Option<Handle>,
}

impl LlamaMbStep {
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
        g: &LlamaGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        base: StepPsos,
        mb: MbPsos,
        n_tokens: u32,
        head_rows: u32,
        requests: u32,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_llama_dag_mb(g, tuning, n_tokens, head_rows, requests, 0, false);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "llama mb step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the MB DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = llama_decode_geometry(g);
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
        let freqs = bind_llama_consts(
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
        Ok(LlamaMbStep {
            dag,
            tables,
            consts,
            base,
            mb,
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

    /// [`fire`](Self::fire) over `[0, end)` — the bisect probe.
    ///
    /// # Errors
    ///
    /// As [`fire`](Self::fire).
    pub fn fire_prefix(&self, stepper: &mut Stepper<'_>, end: usize) -> Result<Timing> {
        stepper.run(|encoder| {
            let run_ends = crate::batch::concurrent_run_ends(&self.dag);
            for (i, d) in self.dag.iter().enumerate().take(end) {
                let pso = llama_mb_pso(d, &self.base, &self.mb).ok_or(Error::Create {
                    what: "llama mb pso",
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
