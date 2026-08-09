//! The gpt-oss engine: the llama engine's assembly over this family's
//! steps, plus the one thing only this family does at load — the
//! quantization trio is SOLVED from the staged tensors before a single
//! pipeline is planned, because the plan's entry names depend on the
//! answer and the config cannot be trusted for it.

use std::collections::HashMap;
use std::path::Path;

use model_loader::plan::LoadPlan;

use crate::batch::{
    FireCsr, FireRefused, GptOssGeometry, IoSlot, MbFeatures, build_gptoss_dag_mb,
    build_scratch_schedule, gptoss_decode_geometry, gptoss_mb_plan, plan_multibatch_psos,
    solve_quant_into,
};
use crate::region::Region as _;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::StepPsos;
use super::context::Context;
use super::encoder::Stepper;
use super::gptoss_step::{GptOssMbPsos, GptOssMbStep, load_gptoss_mb_psos};
use super::handle::Handle;
use super::pipeline::Compiler;
use super::step_mb::{MbPsos, load_mb_psos};
use super::storage::{DecodeStorage, stage_decode_storage, write_fire_io};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FireShape {
    rows: u32,
    samples: u32,
}

/// A loaded gpt-oss model, ready to fire.
#[derive(Debug)]
pub struct GptOssEngine {
    /// The family shape, quantization trio SOLVED.
    pub geometry: GptOssGeometry,
    /// The tuning every prepared step reads.
    pub tuning: Tuning,
    /// The context length the KV was sized for.
    pub max_ctx: u32,
    /// The staged model.
    pub storage: DecodeStorage,
    base: StepPsos,
    mb: MbPsos,
    go: GptOssMbPsos,
    steps: HashMap<FireShape, GptOssMbStep>,
}

impl GptOssEngine {
    /// Stage the model, SOLVE the trio off what was staged, and compile
    /// the pipelines the solution names.
    ///
    /// # Errors
    ///
    /// A staging refusal, an unsolvable trio, or a compilation refusal.
    #[allow(clippy::too_many_arguments)] // one construction
    pub fn new(
        context: &Context,
        compiler: &Compiler,
        kernels_dir: &Path,
        plan: &LoadPlan,
        snapshot: &Path,
        mut geometry: GptOssGeometry,
        tuning: Tuning,
        max_ctx: u32,
    ) -> Result<Self> {
        // Stage FIRST, then solve: the staging reads no quantization
        // field, and the staged extents are the only honest witness.
        let shared = gptoss_decode_geometry(&geometry);
        let slot_bytes =
            crate::batch::gptoss_scratch_elems_mb(&geometry, &tuning, geometry.max_tokens) * 4;
        let storage = stage_decode_storage(context, plan, snapshot, &shared, max_ctx, slot_bytes)?;
        solve_quant_into(&mut geometry, |name| {
            storage.weights.get(name).map(crate::region::Region::len)
        })
        .map_err(|refused| Error::Create {
            what: "gpt-oss quant solve",
            message: refused.0,
        })?;
        let base = super::step::load_step_psos(
            compiler,
            context,
            kernels_dir,
            &gptoss_mb_plan(&geometry),
        )?;
        let mb_plan = plan_multibatch_psos(
            crate::batch::AffineFormat {
                bits: geometry.proj_bits,
                group: 64,
            },
            MbFeatures {
                bias: true,
                ..MbFeatures::default()
            },
            &tuning,
        );
        let mb = load_mb_psos(compiler, context, kernels_dir, &mb_plan)?;
        let go = load_gptoss_mb_psos(compiler, context, kernels_dir, &geometry)?;
        Ok(GptOssEngine {
            geometry,
            tuning,
            max_ctx,
            storage,
            base,
            mb,
            go,
            steps: HashMap::new(),
        })
    }

    /// A fresh sequence: zero the pages.
    ///
    /// # Errors
    ///
    /// A write refusal.
    pub fn reset(&mut self) -> Result<()> {
        for kv in self.storage.kv.iter().flatten() {
            // SAFETY: reset happens at fire boundaries.
            unsafe {
                kv.k_pages.zero(0, kv.k_pages.len())?;
                kv.v_pages.zero(0, kv.v_pages.len())?;
            }
        }
        Ok(())
    }

    /// Fire `csr` — validate, write the IO, run the shape's prepared
    /// step.
    ///
    /// # Errors
    ///
    /// A refused fire, a bind refusal on a first-seen shape, or a
    /// command buffer that does not retire clean.
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
            what: "gpt-oss fire",
            message: format!("{refused:?}"),
        })?;
        let shape = FireShape {
            rows: csr.rows(),
            samples: u32::try_from(csr.sample_rows.len()).unwrap_or(u32::MAX),
        };
        if !self.steps.contains_key(&shape) {
            let dag = build_gptoss_dag_mb(
                &self.geometry,
                &self.tuning,
                shape.rows,
                shape.samples,
                0,
                false,
            );
            let schedule = build_scratch_schedule(&dag, false).map_err(|err| Error::Create {
                what: "gpt-oss fire schedule",
                message: format!("{err:?}"),
            })?;
            let step = GptOssMbStep::prepare(
                context,
                &self.storage,
                &self.geometry,
                &self.tuning,
                &schedule,
                self.base.clone(),
                self.mb.clone(),
                self.go.clone(),
                shape.rows,
                shape.samples,
                self.max_ctx,
            )?;
            self.steps.insert(shape, step);
        }
        // SAFETY: the previous fire retired.
        unsafe { write_fire_io(&self.storage, csr)? };
        self.steps
            .get(&shape)
            .expect("inserted above")
            .fire(stepper)?;
        Ok(())
    }

    /// Where the sampled rows' logits land.
    ///
    /// # Errors
    ///
    /// Paging off — the slot does not exist.
    pub fn logits(&self) -> Result<&Handle> {
        self.storage.io[IoSlot::Logits as usize]
            .as_ref()
            .ok_or(Error::Create {
                what: "gpt-oss logits",
                message: "the logits IO slot was not allocated".to_string(),
            })
    }
}
