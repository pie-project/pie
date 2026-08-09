//! The gemma4 M=1 step: the family whose KV region the shared staging
//! cannot describe.
//!
//! Three axes break the uniform layout the other families share:
//! per-layer head widths (256 sliding, 512 full), per-layer KV head
//! counts (the 26B's full layers carry their own), and OWNERSHIP — the
//! KV-shared tail re-attends pages an earlier layer wrote, so twenty of
//! thirty-five layers allocate nothing and READ another's slots. So
//! [`stage_gemma4_kv`] rebuilds the KV vector at each owning layer's
//! own shape, and [`bind_gemma4_dag`] resolves every attention read
//! through [`Gemma4Geometry::kv_source`] — the geometry already refused
//! any stack whose sharers have no source.

use crate::batch::{
    Dispatch, Gemma4Geometry, IoSlot, Kernel, ScratchSchedule, WeightBind, build_gemma4_dag,
    gemma4_decode_geometry, weight_binds,
};
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_scratch, encode_decode_step};
use super::context::Context;
use super::encoder::Stepper;
use super::gemma4_bind::bind_gemma4_consts;
use super::handle::Handle;
use super::ring::allocate;
use super::storage::{DecodeStorage, KvSlots};
use super::tables::Tables;
use super::timing::Timing;

/// Replace the shared staging's uniform KV vector with this family's:
/// each OWNING layer at its own `[n_kv_heads_of, max_ctx, head_dim_of]`
/// extent, shared layers at `None` — the bind walk resolves their
/// source, and a shared layer that allocated pages of its own would be
/// memory nothing ever reads.
///
/// # Errors
///
/// An allocation refusal.
pub fn stage_gemma4_kv(
    context: &Context,
    storage: &mut DecodeStorage,
    g: &Gemma4Geometry,
    max_ctx: u32,
) -> Result<()> {
    let mut kv: Vec<Option<KvSlots>> = Vec::with_capacity(g.n_layers as usize);
    for layer in 0..g.n_layers {
        if g.is_kv_shared(layer) {
            kv.push(None);
            continue;
        }
        let bytes = u64::from(g.n_kv_heads_of(layer))
            * u64::from(max_ctx)
            * u64::from(g.head_dim_of(layer))
            * 2;
        kv.push(Some(KvSlots {
            k_pages: allocate(context, bytes, "gemma4 kv k")?,
            v_pages: allocate(context, bytes, "gemma4 kv v")?,
        }));
    }
    storage.kv = kv;
    Ok(())
}

fn bind_handle(
    context: &Context,
    tables: &mut Tables,
    ord: u32,
    index: u8,
    handle: &Handle,
) -> Result<()> {
    tables.bind_address(context, ord, index as usize, handle.gpu_address())
}

fn io(storage: &DecodeStorage, slot: IoSlot) -> Result<&Handle> {
    storage.io[slot as usize].as_ref().ok_or(Error::Create {
        what: "gemma4 io slot",
        message: "an IO slot this DAG binds was not allocated".to_string(),
    })
}

/// The family's weight/IO/KV walk — its own, not the shared one, for
/// ONE reason: the attention reads `kv[kv_source(layer)]`, and only
/// this geometry can answer whose pages those are.
///
/// # Errors
///
/// An unstaged weight, a missing IO slot, or a sharer whose source
/// resolves to a layer without pages.
pub fn bind_gemma4_dag(
    context: &Context,
    tables: &mut Tables,
    storage: &DecodeStorage,
    dag: &[Dispatch],
    g: &Gemma4Geometry,
) -> Result<()> {
    let shared = gemma4_decode_geometry(g);
    for d in dag {
        let ord = d.ordinal;
        for WeightBind { bind_index, tensor } in weight_binds(d.kind, d.layer, &shared, false) {
            let handle = storage.weights.get(&tensor).ok_or_else(|| Error::Create {
                what: "gemma4 weight bind",
                message: format!("unstaged weight {tensor}"),
            })?;
            bind_handle(context, tables, ord, bind_index, handle)?;
        }
        let kv_of = |layer: u32| -> Result<&KvSlots> {
            storage.kv[layer as usize].as_ref().ok_or(Error::Create {
                what: "gemma4 kv slots",
                message: format!("layer {layer} owns no KV pages"),
            })
        };
        match d.kind {
            Kernel::EmbedGather | Kernel::G4PleTokenGather => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::EMBED_TOKEN_ID,
                    io(storage, IoSlot::TokenId)?,
                )?;
            }
            Kernel::Rope | Kernel::RopeK => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROPE_POSITION,
                    io(storage, IoSlot::Position)?,
                )?;
            }
            Kernel::KvAppend => {
                let layer = d.layer.expect("a KV append has a layer");
                let kv = kv_of(layer)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_K_PAGES,
                    &kv.k_pages,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_V_PAGES,
                    &kv.v_pages,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_POSITION,
                    io(storage, IoSlot::Position)?,
                )?;
            }
            Kernel::KvAppendPaged => {
                let layer = d.layer.expect("a KV append has a layer");
                let kv = kv_of(layer)?;
                let row = |slot: IoSlot| io(storage, slot);
                bind_handle(context, tables, ord, 2, &kv.k_pages)?;
                bind_handle(context, tables, ord, 3, &kv.v_pages)?;
                bind_handle(context, tables, ord, 4, row(IoSlot::Position)?)?;
                bind_handle(context, tables, ord, 8, row(IoSlot::KvPageIndices)?)?;
                bind_handle(context, tables, ord, 9, row(IoSlot::KvPageIndptr)?)?;
                bind_handle(context, tables, ord, 11, row(IoSlot::ReqOfToken)?)?;
                bind_handle(context, tables, ord, 13, row(IoSlot::WPage)?)?;
                bind_handle(context, tables, ord, 14, row(IoSlot::WOff)?)?;
            }
            Kernel::SdpaPaged => {
                let layer = d.layer.expect("an attention read has a layer");
                let source = g.kv_source(layer).ok_or(Error::Create {
                    what: "gemma4 kv source",
                    message: format!("layer {layer} shares KV but has no source"),
                })?;
                let kv = kv_of(source)?;
                bind_handle(context, tables, ord, 1, &kv.k_pages)?;
                bind_handle(context, tables, ord, 2, &kv.v_pages)?;
                bind_handle(context, tables, ord, 5, io(storage, IoSlot::Position)?)?;
                bind_handle(context, tables, ord, 6, io(storage, IoSlot::ReqOfToken)?)?;
                bind_handle(context, tables, ord, 7, io(storage, IoSlot::KvPageIndices)?)?;
                bind_handle(context, tables, ord, 8, io(storage, IoSlot::KvPageIndptr)?)?;
                bind_handle(context, tables, ord, 12, io(storage, IoSlot::AttnMask)?)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    13,
                    io(storage, IoSlot::AttnMaskStride)?,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    14,
                    io(storage, IoSlot::AttnMaskEnabled)?,
                )?;
            }
            Kernel::G4RowGather => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROW_GATHER_ROWS,
                    io(storage, IoSlot::SampleRows)?,
                )?;
            }
            // The attention reads its SOURCE's pages: itself when it
            // owns them, else the most recent earlier owner of its own
            // attention type.
            Kernel::Sdpa | Kernel::G4SdpaSliding => {
                let layer = d.layer.expect("an attention read has a layer");
                let source = g.kv_source(layer).ok_or(Error::Create {
                    what: "gemma4 kv source",
                    message: format!("layer {layer} shares KV but has no source"),
                })?;
                let kv = kv_of(source)?;
                bind_handle(context, tables, ord, super::bind::slot::SDPA_K, &kv.k_pages)?;
                bind_handle(context, tables, ord, super::bind::slot::SDPA_V, &kv.v_pages)?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_N,
                    io(storage, IoSlot::SeqLen)?,
                )?;
            }
            Kernel::QmvLmHead | Kernel::LmHeadUntied => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::QMV_OUT,
                    io(storage, IoSlot::Logits)?,
                )?;
            }
            // In place over the logits: cap·tanh(logits/cap).
            Kernel::G4Softcap => {
                bind_handle(context, tables, ord, 0, io(storage, IoSlot::Logits)?)?;
                bind_handle(context, tables, ord, 1, io(storage, IoSlot::Logits)?)?;
            }
            Kernel::Argmax => {
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ARGMAX_LOGITS,
                    io(storage, IoSlot::Logits)?,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ARGMAX_NEXT_TOKEN,
                    io(storage, IoSlot::NextToken)?,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ARGMAX_PARAMS,
                    &storage.argmax_params,
                )?;
                bind_handle(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ARGMAX_EOS_FLAG,
                    &storage.eos_flag,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// One bound gemma4 decode step.
#[derive(Debug)]
pub struct Gemma4Step {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The compiled pipelines (from `gemma4_step_plan`).
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
}

impl Gemma4Step {
    /// Build the DAG, bind everything, hold the result ready.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover this
    /// DAG.
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &Gemma4Geometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_gemma4_dag(g, true);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "gemma4 step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_gemma4_dag(context, &mut tables, storage, &dag, g)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        bind_gemma4_consts(
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
        Ok(Gemma4Step {
            dag,
            tables,
            consts,
            psos,
            force_barriers: false,
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

    /// [`fire`](Self::fire) over `[0, end)` — the bisect probe.
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

/// The pipeline one gemma4 MB dispatch runs. The one selection only
/// this family makes: which PAGED attention instantiation serves a
/// layer is its head width — d256 for the sliding layers, d512 full —
/// resolved from `d.layer`, with the launch identical either way.
#[must_use]
pub fn gemma4_mb_pso<'a>(
    d: &Dispatch,
    g: &Gemma4Geometry,
    base: &'a StepPsos,
    mb: &'a super::step_mb::MbPsos,
) -> Option<&'a super::program::Pso> {
    fn bn_slot(bn: u32) -> usize {
        match bn {
            64 => 2,
            32 => 1,
            _ => 0,
        }
    }
    match d.kind {
        // A decided tile whose pipeline is missing REFUSES — the matvec
        // under the GEMM's grid is the named bug.
        Kernel::G4ExpertGate | Kernel::G4ExpertUp | Kernel::G4ExpertDown if d.qmm_bn > 0 => {
            mb.by_slot.get(&crate::batch::MbSlot::QmmRouted {
                width: crate::batch::qmm_bm_slot(d.qmm_bm),
                bn: bn_slot(d.qmm_bn),
            })
        }
        _ if d.qmm_bn > 0 => mb.by_slot.get(&crate::batch::MbSlot::QmmT {
            bm: crate::batch::qmm_bm_slot(d.qmm_bm),
            bn: bn_slot(d.qmm_bn),
        }),
        // A full layer's paged attention is the d512 instantiation —
        // same launch, different width — and missing it refuses too: the
        // d256 pipeline over 512-wide heads strides past every head.
        Kernel::SdpaPaged if d.layer.is_some_and(|l| g.is_full_attn(l)) => {
            mb.by_slot.get(&crate::batch::MbSlot::SdpaPagedD512)
        }
        _ => base.by_kind.get(&d.kind),
    }
}

/// One bound N-token gemma4 fire.
#[derive(Debug)]
pub struct Gemma4MbStep {
    /// The MB dispatch list, tiles decided.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The by-kind pipelines (from `gemma4_mb_plan`).
    pub base: StepPsos,
    /// The shared lattice (GEMMs + the d512 paged attention).
    pub mb: super::step_mb::MbPsos,
    /// The geometry the per-layer resolutions read.
    pub geometry: Gemma4Geometry,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
}

impl Gemma4MbStep {
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
        g: &Gemma4Geometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        base: StepPsos,
        mb: super::step_mb::MbPsos,
        n_tokens: u32,
        head_rows: u32,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = crate::batch::build_gemma4_dag_mb(g, tuning, n_tokens, head_rows, 0, false);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "gemma4 mb step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the MB DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_gemma4_dag(context, &mut tables, storage, &dag, g)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        super::gemma4_bind::bind_gemma4_consts(
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
        Ok(Gemma4MbStep {
            dag,
            tables,
            consts,
            base,
            mb,
            geometry: g.clone(),
            force_barriers: false,
        })
    }

    /// Encode and run the whole MB DAG as one command buffer.
    ///
    /// # Errors
    ///
    /// A dispatch no table serves, or a command buffer that does not
    /// retire clean.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<super::timing::Timing> {
        stepper.run(|encoder| {
            let run_ends = crate::batch::concurrent_run_ends(&self.dag);
            for (i, d) in self.dag.iter().enumerate() {
                let pso = gemma4_mb_pso(d, &self.geometry, &self.base, &self.mb).ok_or(
                    Error::Create {
                        what: "gemma4 mb pso",
                        message: format!(
                            "no pipeline serves {:?} (bn {}, bm {})",
                            d.kind, d.qmm_bn, d.qmm_bm
                        ),
                    },
                )?;
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
