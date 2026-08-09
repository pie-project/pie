//! The llama engine: the family-shaped part of an executor, over
//! everything the arc built below it — staged storage, the MB step, the
//! fire contract.
//!
//! A fire of R rows is one wider launch on this family, so the engine
//! is a cache of PREPARED steps keyed by the fire's shape: the port
//! bakes launches and tiles into the DAG (decide-once), so a new shape
//! is a new prepared step rather than a re-encode — and a decode fleet
//! and a prefill of the same row count are DIFFERENT shapes, because
//! the 64-fleet row block differs. The KV pages live in the storage,
//! not the steps, so every prepared step reads and appends the same
//! cache — which is what makes a prefill's pages readable by the
//! decode fires that follow it.

use std::collections::HashMap;
use std::path::Path;

use model_loader::plan::LoadPlan;

use crate::batch::{
    FireCsr, FireRefused, IoSlot, LlamaGeometry, MbFeatures, build_llama_dag_mb,
    build_scratch_schedule, llama_decode_geometry, llama_mb_plan, plan_multibatch_psos,
};
use crate::region::Region as _;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::StepPsos;
use super::context::Context;
use super::encoder::Stepper;
use super::handle::Handle;
use super::llama_step::LlamaMbStep;
use super::pipeline::Compiler;
use super::step_mb::{MbPsos, load_mb_psos};
use super::storage::{DecodeStorage, stage_decode_storage, write_fire_io};

/// One fire's shape — the prepared-step cache key. Every field changes
/// the DAG the step bakes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FireShape {
    rows: u32,
    samples: u32,
    requests: u32,
}

/// A loaded llama-family model, ready to fire.
#[derive(Debug)]
pub struct LlamaEngine {
    /// The family shape this engine serves.
    pub geometry: LlamaGeometry,
    /// The tuning every prepared step reads.
    pub tuning: Tuning,
    /// The context length the KV was sized for.
    pub max_ctx: u32,
    /// The staged model: weights, KV pages, IO, scratch.
    pub storage: DecodeStorage,
    base: StepPsos,
    mb: MbPsos,
    steps: HashMap<FireShape, LlamaMbStep>,
}

impl LlamaEngine {
    /// Stage the model and compile its pipelines.
    ///
    /// # Errors
    ///
    /// A staging or compilation refusal.
    #[allow(clippy::too_many_arguments)] // one construction; a builder would hide it
    pub fn new(
        context: &Context,
        compiler: &Compiler,
        kernels_dir: &Path,
        plan: &LoadPlan,
        snapshot: &Path,
        geometry: LlamaGeometry,
        tuning: Tuning,
        max_ctx: u32,
    ) -> Result<Self> {
        let shared = llama_decode_geometry(&geometry);
        let slot_bytes =
            crate::batch::scratch_slot_elems(&shared, &tuning, geometry.max_tokens) * 4;
        let storage = stage_decode_storage(context, plan, snapshot, &shared, max_ctx, slot_bytes)?;
        let base =
            super::step::load_step_psos(compiler, context, kernels_dir, &llama_mb_plan(&geometry))?;
        let mb_plan = plan_multibatch_psos(geometry.quant, MbFeatures::default(), &tuning);
        let mb = load_mb_psos(compiler, context, kernels_dir, &mb_plan)?;
        Ok(LlamaEngine {
            geometry,
            tuning,
            max_ctx,
            storage,
            base,
            mb,
            steps: HashMap::new(),
        })
    }

    /// A fresh sequence: the KV is the only thing carried between
    /// tokens, so zeroing it is the whole reset.
    ///
    /// # Errors
    ///
    /// A write refusal.
    pub fn reset(&mut self) -> Result<()> {
        for kv in self.storage.kv.iter().flatten() {
            // SAFETY: reset happens at fire boundaries; no command
            // buffer is in flight.
            unsafe {
                kv.k_pages.zero(0, kv.k_pages.len())?;
                kv.v_pages.zero(0, kv.v_pages.len())?;
            }
        }
        Ok(())
    }

    /// Fire `csr`: validate, write the IO, run the shape's prepared
    /// step — preparing it on first use. The logits slot holds one row
    /// per SAMPLED row, in `csr.sample_rows` order.
    ///
    /// # Errors
    ///
    /// A refused fire (with the incoherence named), a bind refusal on a
    /// first-seen shape, or a command buffer that does not retire clean.
    pub fn fire(
        &mut self,
        context: &Context,
        stepper: &mut Stepper<'_>,
        csr: &FireCsr,
    ) -> Result<()> {
        csr.validate(
            self.geometry.kv_page_size,
            self.geometry.total_pages,
            self.geometry.max_tokens,
            self.geometry.max_requests,
            1,
        )
        .map_err(|refused: FireRefused| Error::Create {
            what: "llama fire",
            message: format!("{refused:?}"),
        })?;
        let shape = FireShape {
            rows: csr.rows(),
            samples: u32::try_from(csr.sample_rows.len()).unwrap_or(u32::MAX),
            requests: csr.requests(),
        };
        if !self.steps.contains_key(&shape) {
            let dag = build_llama_dag_mb(
                &self.geometry,
                &self.tuning,
                shape.rows,
                shape.samples,
                shape.requests,
                0,
                false,
            );
            let schedule = build_scratch_schedule(&dag, false).map_err(|err| Error::Create {
                what: "llama fire schedule",
                message: format!("{err:?}"),
            })?;
            let step = LlamaMbStep::prepare(
                context,
                &self.storage,
                &self.geometry,
                &self.tuning,
                &schedule,
                self.base.clone(),
                self.mb.clone(),
                shape.rows,
                shape.samples,
                shape.requests,
                self.max_ctx,
            )?;
            self.steps.insert(shape, step);
        }
        // SAFETY: the previous fire retired — `fire` blocks on the
        // stepper until the command buffer completes.
        unsafe { write_fire_io(&self.storage, csr)? };
        self.steps
            .get(&shape)
            .expect("inserted above")
            .fire(stepper)?;
        Ok(())
    }

    /// Where the sampled rows' logits land: bf16, `vocab` per row.
    ///
    /// # Errors
    ///
    /// Paging off — the slot does not exist.
    pub fn logits(&self) -> Result<&Handle> {
        self.storage.io[IoSlot::Logits as usize]
            .as_ref()
            .ok_or(Error::Create {
                what: "llama logits",
                message: "the logits IO slot was not allocated".to_string(),
            })
    }
}
