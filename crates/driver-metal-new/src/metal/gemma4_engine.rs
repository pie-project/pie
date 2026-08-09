//! The gemma4 engine: the shared assembly over this family's steps,
//! with its two load-time solves — the per-owning-layer KV restage and
//! the alt-quant probe off the staged extents.

use std::collections::HashMap;
use std::path::Path;

use model_loader::plan::LoadPlan;

use crate::batch::{
    AffineFormat, FireCsr, FireRefused, Gemma4Geometry, IoSlot, MbFeatures, bits_from_extents,
    build_gemma4_dag_mb, build_scratch_schedule, gemma4_decode_geometry, gemma4_mb_plan,
    plan_multibatch_psos,
};
use crate::region::Region as _;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::StepPsos;
use super::context::Context;
use super::encoder::Stepper;
use super::gemma4_step::{Gemma4MbStep, stage_gemma4_kv};
use super::handle::Handle;
use super::pipeline::Compiler;
use super::step_mb::{MbPsos, load_mb_psos};
use super::storage::{DecodeStorage, stage_decode_storage, write_fire_io};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FireShape {
    rows: u32,
    samples: u32,
}

/// A loaded gemma4 model, ready to fire.
#[derive(Debug)]
pub struct Gemma4Engine {
    /// The family shape, alt format solved.
    pub geometry: Gemma4Geometry,
    /// The tuning every prepared step reads.
    pub tuning: Tuning,
    /// The context length the KV was sized for.
    pub max_ctx: u32,
    /// The staged model, KV restaged per owning layer.
    pub storage: DecodeStorage,
    base: StepPsos,
    mb: MbPsos,
    steps: HashMap<FireShape, Gemma4MbStep>,
}

impl Gemma4Engine {
    /// Stage, restage the KV at the family's own shapes, solve the alt
    /// format off the staged extents, and compile what the solution
    /// names.
    ///
    /// # Errors
    ///
    /// A staging or compilation refusal.
    #[allow(clippy::too_many_arguments)] // one construction
    pub fn new(
        context: &Context,
        compiler: &Compiler,
        kernels_dir: &Path,
        plan: &LoadPlan,
        snapshot: &Path,
        mut geometry: Gemma4Geometry,
        tuning: Tuning,
        max_ctx: u32,
    ) -> Result<Self> {
        let shared = gemma4_decode_geometry(&geometry);
        let slot_bytes =
            crate::batch::scratch_slot_elems(&shared, &tuning, geometry.max_tokens) * 4;
        let mut storage =
            stage_decode_storage(context, plan, snapshot, &shared, max_ctx, slot_bytes)?;
        stage_gemma4_kv(context, &mut storage, &geometry, max_ctx)?;
        // Which tensors are 8-bit is read off the checkpoint, never the
        // config — both shipped exemption sets are reachable from here.
        let bits_of = |name: &str| -> Option<u32> {
            let w = storage
                .weights
                .get(&format!("{name}.weight"))
                .map(crate::region::Region::len)?;
            let s = storage
                .weights
                .get(&format!("{name}.scales"))
                .map(crate::region::Region::len)?;
            bits_from_extents(w, s)
        };
        if let Some(bits) = bits_of("layers.0.mlp.down_proj").filter(|&b| b != geometry.quant.bits)
        {
            geometry.alt_quant_ffn = true;
            geometry.ffn_quant = AffineFormat { bits, group: 64 };
        }
        if let Some(bits) = bits_of("layers.0.router.proj").filter(|&b| b != geometry.quant.bits) {
            geometry.alt_quant_router = true;
            geometry.ffn_quant = AffineFormat { bits, group: 64 };
        }
        let base = super::step::load_step_psos(
            compiler,
            context,
            kernels_dir,
            &gemma4_mb_plan(&geometry),
        )?;
        let mb_plan = plan_multibatch_psos(
            geometry.quant,
            MbFeatures {
                sdpa_d256: geometry.head_dim == 256,
                d512: geometry.global_head_dim == 512,
                routed: geometry.is_moe(),
                ..MbFeatures::default()
            },
            &tuning,
        );
        let mb = load_mb_psos(compiler, context, kernels_dir, &mb_plan)?;
        Ok(Gemma4Engine {
            geometry,
            tuning,
            max_ctx,
            storage,
            base,
            mb,
            steps: HashMap::new(),
        })
    }

    /// A fresh sequence: zero the owning layers' pages.
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
            what: "gemma4 fire",
            message: format!("{refused:?}"),
        })?;
        let shape = FireShape {
            rows: csr.rows(),
            samples: u32::try_from(csr.sample_rows.len()).unwrap_or(u32::MAX),
        };
        if !self.steps.contains_key(&shape) {
            let dag = build_gemma4_dag_mb(
                &self.geometry,
                &self.tuning,
                shape.rows,
                shape.samples,
                0,
                false,
            );
            let schedule = build_scratch_schedule(&dag, false).map_err(|err| Error::Create {
                what: "gemma4 fire schedule",
                message: format!("{err:?}"),
            })?;
            let step = Gemma4MbStep::prepare(
                context,
                &self.storage,
                &self.geometry,
                &self.tuning,
                &schedule,
                self.base.clone(),
                self.mb.clone(),
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
                what: "gemma4 logits",
                message: "the logits IO slot was not allocated".to_string(),
            })
    }
}
