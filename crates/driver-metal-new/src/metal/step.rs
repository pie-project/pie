//! Compiling the M=1 PSO table and driving one decode step.
//!
//! [`load_step_psos`] turns the portable plan (`plan_decode_psos`) into
//! compiled pipelines: one batch compile — shared sources front-ended once,
//! archived — then the fan-out from request to kinds that the plan already
//! recorded. [`DecodeStep`] owns everything a fired token needs and runs
//! the encode walk under the [`Stepper`].
//!
//! The scratch schedule arrives as an input: the dataflow walk that
//! produces it is the family's own and lands with the forward port. A step
//! without one binds no activation slots, which the first dispatch would
//! fault on — so [`DecodeStep::prepare`] takes it required, not optional.

use std::path::Path;

use crate::batch::{
    DagOptions, DecodeGeometry, DecodePsoPlan, Dispatch, ScratchSchedule, build_decode_dag,
};
use crate::shader::Request;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{
    ConstSlots, StepPsos, bind_decode_consts, bind_decode_dag, bind_gdn_parity, bind_scratch,
    encode_decode_step,
};
use super::context::Context;
use super::encoder::Stepper;
use super::pipeline::Compiler;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// Compile the plan's pipelines and lay them out by kind.
///
/// One batch per plan: `quant/qmv.metal`'s entrypoints come out of one
/// front-end pass, and the archive key covers the whole set.
pub fn load_step_psos(
    compiler: &Compiler,
    context: &Context,
    kernels_dir: &Path,
    plan: &DecodePsoPlan,
) -> Result<StepPsos> {
    let requests: Vec<Request> = plan
        .requests
        .iter()
        .map(|request| Request::new(kernels_dir.join(request.file), request.entry.clone()))
        .collect();
    let pipelines = compiler.compile_batch(context, &requests).all()?;
    let mut psos = StepPsos::default();
    for (i, request) in plan.requests.iter().enumerate() {
        for &kind in &request.kinds {
            psos.by_kind.insert(kind, pipelines[i].clone());
        }
        if plan.residual == Some(i) {
            psos.qmv_residual = Some(pipelines[i].clone());
        }
    }
    Ok(psos)
}

/// One bound decode step: the DAG, its argument tables, its constants and
/// its pipelines, ready to fire.
#[derive(Debug)]
pub struct DecodeStep {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache (rebound in place on width changes).
    pub consts: ConstSlots,
    /// The compiled pipelines.
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// Whether the DAG was built with the GdnPrep split (the recurrent
    /// core's slots differ).
    gdn_prep: bool,
}

impl DecodeStep {
    /// Build the DAG, bind everything, and hold the result ready.
    ///
    /// The four bind passes run in the C++'s order — weights/state/IO,
    /// scratch, constants — so a prepared step is never half-bound.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &DecodeGeometry,
        tuning: &Tuning,
        options: DagOptions,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_decode_dag(g, tuning, options);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "decode step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag(context, &mut tables, storage, &dag, g, options.gdn_prep)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        bind_decode_consts(
            context,
            &mut tables,
            &mut consts,
            &dag,
            g,
            tuning,
            max_ctx,
            options.gdn_prep,
            1,
            0,
        )?;
        Ok(DecodeStep {
            dag,
            tables,
            consts,
            psos,
            force_barriers: false,
            gdn_prep: options.gdn_prep,
        })
    }

    /// Point every GDN dispatch's conv-state binds at the half that holds
    /// the LATEST data.
    ///
    /// The conv state ping-pongs: the read and write buffers are distinct
    /// (an in-place shift races the tap reads) and step `i` reads what
    /// `i - 1` wrote, so the binds swap by the slot's own step parity —
    /// [`Parity::Even`](crate::store::Parity) is the staged orientation.
    /// Rebinding rewrites addresses the tables already hold; no encoded
    /// byte moves.
    pub fn set_gdn_parity(
        &mut self,
        context: &Context,
        storage: &DecodeStorage,
        parity: crate::store::Parity,
    ) -> Result<()> {
        bind_gdn_parity(
            context,
            &mut self.tables,
            storage,
            &self.dag,
            self.gdn_prep,
            parity,
        )
    }

    /// Encode and run the whole DAG as one command buffer, returning the
    /// step's timing.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<Timing> {
        stepper.run(|encoder| {
            encode_decode_step(
                encoder,
                &self.tables,
                &self.dag,
                &self.psos,
                self.force_barriers,
                0,
                self.dag.len(),
            )
        })
    }
}
