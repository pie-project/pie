//! The decode runner: prefill streams, fleet fires, page bookkeeping and
//! the per-slot conv orientation.
//!
//! This is the first arm of the forward orchestration: what the device
//! smoke did by hand — feeding IO rows, assigning pages and slots, keeping
//! the GDN ping-pong straight — done once, in one place, so a caller holds
//! requests and tokens rather than buffer offsets.
//!
//! # The per-slot orientation, and the join copy
//!
//! The conv ping-pong's read/write binds are per FIRE, but which half holds
//! a slot's LATEST window is per SLOT — a slot that sat out a fire did not
//! flip. A fleet fire binds one orientation for everyone, so a slot whose
//! orientation disagrees is NORMALIZED on join: its latest conv window is
//! copied into the other half (both GDN buffers, every GDN layer, that
//! slot's slice only) and its orientation flipped. The copy is small — one
//! slot's `conv_dim × Kc` floats per layer — and the alternative is the
//! race the C++ documents: a bind that reads the stale half is finite,
//! quiet and wrong. Equal-length lockstep fleets never pay it.
//!
//! # Pages and slots
//!
//! Page allocation here is the simplest sound policy: the pool is split
//! into `max_requests` equal ranges, request r owning
//! `[r * per, (r + 1) * per)`. The real allocator (grow, fork, free) is
//! engine policy and arrives with the cutover wiring; nothing below bakes
//! the split into the kernels — the page table is rewritten per fire.

use crate::batch::{
    DagOptions, DecodeGeometry, EntryNames, IoSlot, MbFeatures, PsoFeatures, ScratchSchedule,
    build_decode_dag_mb, build_scratch_uses, concurrent_run_ends, plan_decode_psos,
    plan_multibatch_psos, schedule_scratch,
};
use crate::region::Region as _;
use crate::store::Parity;
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::context::Context;
use super::encoder::Stepper;
use super::pipeline::Compiler;
use super::step::load_step_psos;
use super::step_mb::{MbStep, load_mb_psos};
use super::storage::DecodeStorage;

/// One decode lane: a request continuing at `position` with `token`.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// The request (its page range and IO row follow from it).
    pub request: u32,
    /// The recurrent-state slot the request holds.
    pub slot: u32,
    /// The token this fire feeds.
    pub token: u32,
    /// Its absolute position.
    pub position: u32,
}

/// The runner. Owns the staged storage, the prepared steps per width, and
/// the bookkeeping the fires need.
pub struct Decoder<'ctx> {
    context: &'ctx Context,
    storage: DecodeStorage,
    geometry: DecodeGeometry,
    tuning: Tuning,
    options: DagOptions,
    max_ctx: u32,
    kernels_dir: std::path::PathBuf,
    compiler: Compiler,
    stepper: Stepper<'ctx>,
    steps: std::collections::HashMap<u32, MbStep>,
    /// Which half holds each slot's latest conv window (the parity the
    /// next fire that touches it must bind).
    orientation: Vec<Parity>,
    pages_per_request: u32,
}

impl<'ctx> Decoder<'ctx> {
    /// Build the runner over staged storage.
    pub fn new(
        context: &'ctx Context,
        storage: DecodeStorage,
        geometry: DecodeGeometry,
        tuning: Tuning,
        options: DagOptions,
        max_ctx: u32,
        kernels_dir: std::path::PathBuf,
    ) -> Result<Self> {
        let compiler = Compiler::new(context)?;
        let stepper = Stepper::new(context)?;
        let pages_per_request = geometry.total_pages / geometry.max_requests.max(1);
        Ok(Decoder {
            context,
            orientation: vec![Parity::Even; geometry.max_slots as usize],
            storage,
            geometry,
            tuning,
            options,
            max_ctx,
            kernels_dir,
            compiler,
            stepper,
            steps: std::collections::HashMap::new(),
            pages_per_request,
        })
    }

    /// The staged storage (the smoke reads logits through it).
    #[must_use]
    pub fn storage(&self) -> &DecodeStorage {
        &self.storage
    }

    fn schedule_for(&self, n_tokens: u32) -> Result<ScratchSchedule> {
        let dag = build_decode_dag_mb(&self.geometry, &self.tuning, n_tokens, 0, self.options);
        let (uses, values) = build_scratch_uses(&dag);
        let ends = concurrent_run_ends(&dag);
        schedule_scratch(dag.len(), &uses, &ends, values, false).map_err(|err| Error::Create {
            what: "decoder schedule",
            message: format!("{err}"),
        })
    }

    fn step_for(&mut self, n_tokens: u32) -> Result<()> {
        if self.steps.contains_key(&n_tokens) {
            return Ok(());
        }
        let features = PsoFeatures {
            gdn: true,
            gated_attention: true,
            sdpa_d256: self.geometry.head_dim == 256,
            routed: self.geometry.is_moe(),
            untied: !self.geometry.tied_embeddings,
            ..PsoFeatures::default()
        };
        let base = load_step_psos(
            &self.compiler,
            self.context,
            &self.kernels_dir,
            &plan_decode_psos(&EntryNames::bf16_g64_b4(), features),
        )?;
        let mb_features = MbFeatures {
            gdn: true,
            sdpa_d256: self.geometry.head_dim == 256,
            ..MbFeatures::default()
        };
        let mb = load_mb_psos(
            &self.compiler,
            self.context,
            &self.kernels_dir,
            &plan_multibatch_psos(self.geometry.quant, mb_features, &self.tuning),
        )?;
        let schedule = self.schedule_for(n_tokens)?;
        let step = MbStep::prepare(
            self.context,
            &self.storage,
            &self.geometry,
            &self.tuning,
            self.options,
            &schedule,
            base,
            mb,
            n_tokens,
            self.max_ctx,
        )?;
        self.steps.insert(n_tokens, step);
        Ok(())
    }

    fn io(&self, slot: IoSlot) -> Result<&super::handle::Handle> {
        self.storage.io[slot as usize]
            .as_ref()
            .ok_or(Error::Create {
                what: "decoder io",
                message: "paging is off; the decoder needs the paged IO family".to_string(),
            })
    }

    fn write_u32s(&self, slot: IoSlot, values: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: the buffers are host-owned between fires.
        unsafe { self.io(slot)?.write(0, &bytes) }
    }

    /// Rewrite the page table: request r owns its fixed range.
    fn write_page_table(&self) -> Result<()> {
        let pages: Vec<u32> = (0..self.geometry.total_pages).collect();
        self.write_u32s(IoSlot::KvPageIndices, &pages)?;
        let indptr: Vec<u32> = (0..=self.geometry.max_requests)
            .map(|r| r * self.pages_per_request)
            .collect();
        self.write_u32s(IoSlot::KvPageIndptr, &indptr)?;
        let slots: Vec<u32> = (0..self.geometry.max_requests).collect();
        self.write_u32s(IoSlot::RsSlotIds, &slots)?;
        self.write_u32s(
            IoSlot::KvLastPageLens,
            &vec![1; self.geometry.max_requests as usize],
        )
    }

    /// Copy `slot`'s latest conv window into the other half and flip its
    /// orientation, so it can join a fire binding `target`.
    fn normalize_slot(&mut self, slot: u32, target: Parity) -> Result<()> {
        if self.orientation[slot as usize] == target {
            return Ok(());
        }
        let g = &self.geometry;
        let slot_bytes = u64::from(g.gdn_conv_dim) * u64::from(g.gdn_conv_k) * 4;
        let at = u64::from(slot) * slot_bytes;
        for state in self.storage.gdn.iter().flatten() {
            // Latest is where the CURRENT orientation would read from.
            let (latest, other) = match self.orientation[slot as usize] {
                Parity::Even => (&state.conv_state, &state.conv_state_out),
                Parity::Odd => (&state.conv_state_out, &state.conv_state),
            };
            // SAFETY: fires are synchronous; the GPU is idle between them.
            unsafe { other.copy(at, latest, at, slot_bytes)? };
        }
        self.orientation[slot as usize] = target;
        Ok(())
    }

    /// Fire one width-`lanes.len()` step over the given lanes.
    ///
    /// Every touched slot is normalized to the fire's orientation first;
    /// the fire's orientation is the majority's, so lockstep fleets never
    /// pay a copy.
    pub fn fire(&mut self, lanes: &[Lane]) -> Result<()> {
        let n = u32::try_from(lanes.len()).expect("a fire is small");
        assert!(n > 0, "a fire carries at least one lane");
        self.step_for(n)?;
        let even = lanes
            .iter()
            .filter(|l| self.orientation[l.slot as usize] == Parity::Even)
            .count();
        let target = if even * 2 >= lanes.len() {
            Parity::Even
        } else {
            Parity::Odd
        };
        for lane in lanes {
            self.normalize_slot(lane.slot, target)?;
        }
        self.write_page_table()?;
        let mut indptr = vec![0u32];
        indptr.extend((1..=n).collect::<Vec<_>>());
        self.write_u32s(IoSlot::QoIndptr, &indptr)?;
        self.write_u32s(
            IoSlot::TokenId,
            &lanes.iter().map(|l| l.token).collect::<Vec<_>>(),
        )?;
        self.write_u32s(
            IoSlot::Position,
            &lanes.iter().map(|l| l.position).collect::<Vec<_>>(),
        )?;
        self.write_u32s(
            IoSlot::SeqLen,
            &lanes.iter().map(|l| l.position + 1).collect::<Vec<_>>(),
        )?;
        self.write_u32s(
            IoSlot::ReqOfToken,
            &lanes.iter().map(|l| l.request).collect::<Vec<_>>(),
        )?;
        self.write_u32s(
            IoSlot::SlotOfToken,
            &lanes.iter().map(|l| l.slot).collect::<Vec<_>>(),
        )?;
        let page =
            |l: &Lane| l.request * self.pages_per_request + l.position / self.geometry.kv_page_size;
        self.write_u32s(IoSlot::WPage, &lanes.iter().map(page).collect::<Vec<_>>())?;
        self.write_u32s(
            IoSlot::WOff,
            &lanes
                .iter()
                .map(|l| l.position % self.geometry.kv_page_size)
                .collect::<Vec<_>>(),
        )?;
        let step = self.steps.get_mut(&n).expect("prepared above");
        step.set_gdn_parity(self.context, &self.storage, target)?;
        step.fire(&mut self.stepper)?;
        for lane in lanes {
            self.orientation[lane.slot as usize] = match target {
                Parity::Even => Parity::Odd,
                Parity::Odd => Parity::Even,
            };
        }
        Ok(())
    }

    /// Run a request's whole prompt as a stream of single-token fires and
    /// return its next token.
    pub fn prefill(&mut self, request: u32, slot: u32, tokens: &[u32]) -> Result<u32> {
        for (position, &token) in tokens.iter().enumerate() {
            self.fire(&[Lane {
                request,
                slot,
                token,
                position: u32::try_from(position).expect("a prompt is small"),
            }])?;
        }
        Ok(self.argmax_row(0))
    }

    /// Greedy generation: prefill, then decode until `max_new` tokens or
    /// an id in `eos` — the serving loop's shape, one lane.
    pub fn greedy(
        &mut self,
        request: u32,
        slot: u32,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>> {
        let mut token = self.prefill(request, slot, prompt)?;
        let mut out = vec![token];
        let mut position = u32::try_from(prompt.len()).expect("a prompt is small");
        while out.len() < max_new && !eos.contains(&token) {
            self.fire(&[Lane {
                request,
                slot,
                token,
                position,
            }])?;
            token = self.argmax_row(0);
            out.push(token);
            position += 1;
        }
        Ok(out)
    }

    /// Every byte this runner holds on the device: the staged regions, the
    /// state, the pools and the const slots. The soak's instrument — a
    /// decode loop must not move this number.
    #[must_use]
    pub fn footprint_bytes(&self) -> u64 {
        use crate::region::Region as _;
        let s = &self.storage;
        let mut total = s.weights_region.len();
        for state in s.gdn.iter().flatten() {
            total += state.conv_state.len()
                + state.conv_state_out.len()
                + state.recurrent_state.len()
                + state.conv_bias_zero.len();
        }
        for kv in s.kv.iter().flatten() {
            total += kv.k_pages.len() + kv.v_pages.len();
        }
        for io in s.io.iter().flatten() {
            total += io.len();
        }
        for scratch in &s.scratch {
            total += scratch.len();
        }
        total += s.argmax_params.len() + s.eos_flag.len();
        // Const slots are 256-byte-class buffers; count them by presence so
        // an unbounded cache shows up as growth.
        for step in self.steps.values() {
            total += 256 * step.consts.len() as u64;
        }
        total
    }

    /// The device-visible argmax of logits row `row`.
    #[must_use]
    pub fn argmax_row(&self, row: usize) -> u32 {
        let vocab = self.geometry.vocab as usize;
        let logits = self.storage.io[IoSlot::Logits as usize]
            .as_ref()
            .expect("paged logits");
        // SAFETY: fires are synchronous; the last one retired.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                logits.contents().cast::<u8>().as_ptr().add(row * vocab * 2),
                vocab * 2,
            )
        };
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if v > best.1 {
                best = (u32::try_from(i).expect("vocab fits"), v);
            }
        }
        best.0
    }
}
