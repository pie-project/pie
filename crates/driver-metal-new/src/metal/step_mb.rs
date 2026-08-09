//! The multibatch step: MB pipelines, their selection, and the driver that
//! fires an N-token DAG.
//!
//! [`mb_pso`] is the C++ selection ladder, order intact: the mixture's
//! projections are asked BEFORE the shared GEMM arm and for the same
//! reason `qmv_out_size` declines to answer for them — they carry a
//! `qmm_bn` like any batched projection, so the default arm would hand
//! them the DENSE GEMM, which indexes one weight for the whole dispatch
//! and would run every expert's rows through expert 0's slice. Fluent,
//! and wrong.
//!
//! Deferred, ledgered: the FP16 staging pair (`bind_mb_fp16_qmm`) — the
//! BF16 GEMM serves every batched projection until the staging buffer and
//! its cast dispatch land, so `mb_pso` is always told FP16 is unavailable;
//! and the strided prefill kernels, which this driver does not emit yet
//! (a prefill runs as one packed N-token fire).

use std::collections::HashMap;
use std::path::Path;

use crate::batch::{
    DagOptions, DecodeGeometry, Dispatch, Kernel, MbRequest, MbSlot, ScratchSchedule,
    build_decode_dag_mb, qmm_bm_slot,
};
use crate::shader::Request;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_decode_consts, bind_scratch};
use super::bind_mb::{MbBindOffsets, bind_decode_dag_mb, bind_gdn_conv_parity};
use super::context::Context;
use super::encoder::Stepper;
use super::pipeline::Compiler;
use super::program::Pso;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// The compiled multibatch pipelines, by slot.
#[derive(Clone, Debug, Default)]
pub struct MbPsos {
    /// Slot → pipeline; array slots carry their rung indices in the key.
    pub by_slot: HashMap<MbSlot, Pso>,
}

/// Compile the multibatch plan.
pub fn load_mb_psos(
    compiler: &Compiler,
    context: &Context,
    kernels_dir: &Path,
    plan: &[MbRequest],
) -> Result<MbPsos> {
    let requests: Vec<Request> = plan
        .iter()
        .map(|request| Request::new(kernels_dir.join(request.file), request.entry.clone()))
        .collect();
    let pipelines = compiler.compile_batch(context, &requests).all()?;
    let mut psos = MbPsos::default();
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

/// The pipeline one MB dispatch runs. `base` supplies the M=1 kernels that
/// serve unbatched dispatches unchanged.
#[must_use]
pub fn mb_pso<'a>(d: &Dispatch, base: &'a StepPsos, mb: &'a MbPsos) -> Option<&'a Pso> {
    let slot = |s: MbSlot| mb.by_slot.get(&s);
    match d.kind {
        Kernel::EmbedUntied | Kernel::EmbedGather => slot(MbSlot::EmbedMb),
        Kernel::Rope | Kernel::RopeK => slot(MbSlot::RopeMb),
        Kernel::GdnPrepSlotted => slot(MbSlot::GdnPrepSlotted),
        Kernel::GdnCoreSlotted => slot(MbSlot::GdnRecurrentSlotted),
        Kernel::KvAppendPaged => slot(MbSlot::KvAppendPaged),
        Kernel::SdpaPaged => slot(MbSlot::SdpaPaged),
        Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown => {
            if d.qmm_bn > 0 {
                let routed = slot(MbSlot::QmmRouted {
                    width: qmm_bm_slot(d.qmm_bm),
                    bn: bn_slot(d.qmm_bn),
                });
                if routed.is_some() {
                    return routed;
                }
            }
            base.by_kind.get(&d.kind)
        }
        _ => {
            if d.qmm_bn > 0 {
                let wide = qmm_bm_slot(d.qmm_bm);
                let bn = bn_slot(d.qmm_bn);
                let gemm = if d.fuse_residual {
                    slot(MbSlot::QmmTResidual { bm: wide, bn })
                } else {
                    slot(MbSlot::QmmT { bm: wide, bn })
                };
                if gemm.is_some() {
                    return gemm;
                }
            }
            if d.fuse_residual {
                base.qmv_residual.as_ref()
            } else {
                base.by_kind.get(&d.kind)
            }
        }
    }
}

/// One bound multibatch step, ready to fire at its built token count.
#[derive(Debug)]
pub struct MbStep {
    /// The MB dispatch list.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The M=1 pipelines (unbatched dispatches run unchanged).
    pub base: StepPsos,
    /// The multibatch pipelines.
    pub mb: MbPsos,
    gdn_prep: bool,
}

impl MbStep {
    /// Build and bind an N-token step.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &DecodeGeometry,
        tuning: &Tuning,
        options: DagOptions,
        schedule: &ScratchSchedule,
        base: StepPsos,
        mb: MbPsos,
        n_tokens: u32,
        max_ctx: u32,
    ) -> Result<Self> {
        Self::prepare_at(
            context,
            storage,
            g,
            tuning,
            options,
            schedule,
            base,
            mb,
            n_tokens,
            max_ctx,
            MbBindOffsets::default(),
        )
    }

    /// [`prepare`](Self::prepare) with this step's IO offsets — a prefill
    /// stream prepares one step per prompt row, each bound at its own row.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_at(
        context: &Context,
        storage: &DecodeStorage,
        g: &DecodeGeometry,
        tuning: &Tuning,
        options: DagOptions,
        schedule: &ScratchSchedule,
        base: StepPsos,
        mb: MbPsos,
        n_tokens: u32,
        max_ctx: u32,
        offsets: MbBindOffsets,
    ) -> Result<Self> {
        let dag = build_decode_dag_mb(g, tuning, n_tokens, 0, options);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "mb step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the MB DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag_mb(
            context,
            &mut tables,
            storage,
            &dag,
            g,
            options.gdn_prep,
            offsets,
        )?;
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
            n_tokens,
            0,
        )?;
        Ok(MbStep {
            dag,
            tables,
            consts,
            base,
            mb,
            gdn_prep: options.gdn_prep,
        })
    }

    /// Swap the slotted GDN conv ping-pong to the half this fire reads.
    pub fn set_gdn_parity(
        &mut self,
        context: &Context,
        storage: &DecodeStorage,
        parity: crate::store::Parity,
    ) -> Result<()> {
        let _ = self.gdn_prep;
        bind_gdn_conv_parity(context, &mut self.tables, storage, &self.dag, parity)
    }

    /// Encode and run the whole MB DAG as one command buffer.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<Timing> {
        stepper.run(|encoder| {
            let run_ends = crate::batch::concurrent_run_ends(&self.dag);
            for (i, d) in self.dag.iter().enumerate() {
                let pso = mb_pso(d, &self.base, &self.mb).ok_or(Error::Create {
                    what: "mb pso",
                    message: format!("no pipeline serves {:?}", d.kind),
                })?;
                encoder.set_pipeline(pso);
                encoder.set_argument_table_for(&self.tables, d.ordinal)?;
                encoder.dispatch(
                    d.launch.grid.map(|v| v as usize),
                    d.launch.tg.map(|v| v as usize),
                )?;
                if crate::batch::barrier_after(&self.dag, i, &run_ends) {
                    encoder.barrier(super::encoder::Visibility::Device);
                }
            }
            Ok(())
        })
    }
}
