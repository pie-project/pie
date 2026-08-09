//! The executor's first half: binding a flat launch's operands
//! (retirement plan phase C).
//!
//! `model_compiler::lower` turns a traced fire into rectangles whose
//! operands are [`Arg`]s — an arena offset, a backend-named value, or a
//! weight name. The C++ declared executor binds those against `ws.*`
//! fields per family; THIS binder is the family-independent replacement
//! the flat list was designed for: three resolution rules, stated once.
//!
//! What binding is NOT: dispatch. A bound launch still has to reach its
//! `pie_k_*` entry with the operands in the row's own order, and that
//! per-kernel arm is the executor's other half — it grows kernel by
//! kernel beside the bridge. Splitting the two means the binder is pure
//! host logic, provable against a real lowered trace with no GPU and no
//! bridge in the build.

use std::ffi::c_void;

use model_compiler::lower::{Arg, Buffers, Launch, Lowered};
use model_compiler::trace::ValueId;

/// The frame's activation arena: one device block of
/// [`Lowered::arena_bytes`], allocated per fire (or reused across them —
/// the binder only ADDRESSES it).
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Device base of the arena.
    pub arena: *mut c_void,
    /// Its extent — [`Lowered::arena_bytes`] at allocation time. Offsets
    /// are checked against it, because an arena reused across fires can
    /// be SMALLER than the new fire needs, and a launch that addressed
    /// past it would corrupt whatever the allocator placed next.
    pub arena_bytes: usize,
}

/// Resolves the names the trace states against the driver's stores.
///
/// The one thing that stays per-family is a MAP rather than a switch —
/// `lower.rs`'s own words — and this is that map's seam. The live
/// implementation reads the loaded model's tensor store and the fire's
/// seam values; tests answer with sentinels.
pub trait Resolver {
    /// The device pointer for a weight the trace names
    /// (`layer.3.q_proj`), or `None` — which is DRIFT, not absence: a
    /// trace that names a weight the store lacks was traced against a
    /// different binding.
    fn weight(&mut self, name: &str) -> Option<*const c_void>;
    /// The device pointer for a backend-named value (the observed query,
    /// the logits — `Buffers::NAMED`).
    fn named(&mut self, value: ValueId) -> Option<*mut c_void>;
}

/// One resolved operand: where it is, and how wide one row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundArg {
    /// The device address.
    pub ptr: *mut c_void,
    /// Elements per row, for the args that carry one ([`Arg::Arena`],
    /// [`Arg::Named`]); zero for a weight, whose extent is the tensor's.
    pub width: u32,
}

/// A launch with every operand resolved — what a dispatch arm consumes.
#[derive(Debug)]
pub struct BoundLaunch<'a> {
    /// The kernel's symbol, resolved through [`Lowered::kernels`].
    pub kernel: &'a str,
    /// The rectangle, in the op's own row space.
    pub rows: std::ops::Range<u32>,
    /// The layer range.
    pub layers: std::ops::Range<u16>,
    /// Operands in the trace's stated order: inputs, outputs, weights.
    pub args: Vec<BoundArg>,
}

/// Why a launch refused to bind. Every variant is a DRIFT diagnosis, not
/// a runtime condition — the C++ executor's `throw_drift` shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindRefusal {
    /// An arena operand addresses past the frame's arena.
    ArenaOutOfBounds {
        /// The offending offset.
        at: usize,
        /// What the frame actually holds.
        arena_bytes: usize,
    },
    /// The trace names a weight the resolver does not hold.
    UnknownWeight(String),
    /// The trace names a seam value the resolver does not bind.
    UnknownNamed(ValueId),
}

/// What one launch needs beyond its bound args: the op join.
///
/// `Launch::args` carries VALUES; the op the launch lowers carries the
/// rest — the weight its statement names and the accumulate flag — which
/// is exactly the `plan.weight_name(op)` read the C++ executor does. The
/// join is computed once per lowering, so the arms read a slot instead of
/// re-matching `OpKind` per fire.
#[derive(Debug, Clone, Default)]
pub struct LaunchSpec {
    /// The weight the op names, when it names one. Concrete — the trace
    /// is layer-unrolled, so this is `layer.3.q_proj`, never a template.
    pub weight: Option<String>,
    /// `Matmul::beta_one`: the residual fold. The launch then carries the
    /// accumulate target as its LAST arg (inputs, then outputs — the
    /// output aliases the residual input's bytes).
    pub beta_one: bool,
    /// The op's OUTPUT placements, resolved through `value_offset` — the
    /// values a launch writes that its args do not carry (the fused qkv's
    /// observed-query pin, the attention output the o_proj reads).
    pub outs: Vec<Arg>,
    /// The SECOND weight an op names, when it names two (`GdnPrep`'s
    /// `dt_bias` beside its `a_log`).
    pub weight2: Option<String>,
    /// The per-request store this op addresses (`OpKind::state_ref`) —
    /// how a GDN arm learns its state layer, the C++ executor's
    /// `op.param1` read.
    pub state: Option<model_compiler::trace::StateRef>,
    /// `RmsnormPerHead`'s head width: the launch's rows are token rows,
    /// the kernel's rows are `tokens * (width / head_dim)` of `head_dim`.
    pub per_head_dim: Option<u32>,
    /// `Rope`'s partial-rotary channel count, when the op states one.
    pub rope_partial: Option<u32>,
    /// FOREIGN values a launch consumes that its statement does not
    /// carry as args — nemotron's mamba block, where the dt/dA prep and
    /// the scan read the SPLIT's raw `dt` and the PARAMS prep's fp32
    /// tables (the C++ hand pass wires them through its workspace).
    /// Order, when present: `[dt_raw, a, d, dt_bias, dt_pre, da_pre]`.
    pub aux: Vec<Arg>,
    /// How many of the launch's args are INPUTS, and how many OUTPUTS.
    ///
    /// A lowered `Launch` hands the binder one flat run in stated order
    /// (inputs, then outputs, then the weights the statement names), and
    /// the generated dispatch needs the split: a row sources its operands
    /// as `In(i)` / `Out(i)` / `Weight(i)`, which are three spans in the
    /// C++ context and three SLICES of one run here. The counts come off
    /// the op, which is the only place that knows them.
    pub n_in: usize,
    /// See [`Self::n_in`].
    pub n_out: usize,
    /// `OpKind::Launch::params` — the wire scalars a statement carries
    /// that no operand shape gives.
    ///
    /// One reader today, and it is the reason this field exists: the
    /// attention dispatches state their `window_left` here. The arms used
    /// to read it off `AttnCtx::window_left_by_layer`, which the driver
    /// built from a config — so the trace said one thing about a layer
    /// and the driver believed another, and nothing made them agree. A
    /// statement that says which window it attends over is the whole
    /// point of the field; the ctx stays as the fallback for a statement
    /// that carries none.
    ///
    /// `i32` on the wire as `u32`, so `-1` reads back as `0xFFFF_FFFF` —
    /// [`window_of`] does the cast in one place rather than at each arm.
    pub params: Vec<u32>,
}

/// A wire param read back as `f32`.
///
/// `params` is `[u32]` because that is what a wire carries; a float rides
/// it as its BITS (`f32::to_bits`), never as a rounded integer — which is
/// the whole reason this is one function rather than a cast at each arm.
fn param_f32(spec: &LaunchSpec, at: usize) -> Option<f32> {
    spec.params.get(at).copied().map(f32::from_bits)
}

/// The window a launch attends over: the STATEMENT's, or the context's
/// where a statement carries none.
///
/// The fallback is not a preference. It is what a trace written before
/// the dispatches carried a window means, and it disappears when the last
/// such trace does.
#[cfg(feature = "bridge")]
fn window_of(spec: &LaunchSpec, a: &AttnCtx, layer: u32) -> i32 {
    #[allow(clippy::cast_possible_wrap)]
    if let Some(&stated) = spec.params.first() {
        return stated as i32;
    }
    a.window_left_by_layer
        .get(layer as usize)
        .copied()
        .unwrap_or(a.window_left)
}

/// The per-launch op join over a whole lowering.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    specs: Vec<LaunchSpec>,
}

impl DispatchPlan {
    /// Join `lowered`'s launches with the ops that produced them.
    #[must_use]
    pub fn new(plan: &model_compiler::trace::ForwardPlan, lowered: &Lowered) -> Self {
        use model_compiler::trace::OpKind;
        use model_compiler::trace::Dim;
        let width_of = |v: ValueId| -> u32 {
            plan.values[v as usize]
                .shape
                .0
                .iter()
                .filter_map(|d| match d {
                    Dim::Const(w) => Some(*w),
                    _ => None,
                })
                .product::<u32>()
                .max(1)
        };
        let out_arg = |v: ValueId| -> Arg {
            match lowered.value_offset.get(v as usize) {
                Some(&at) if at != Buffers::NAMED => Arg::Arena {
                    at,
                    width: width_of(v),
                    bytes: plan
                        .values
                        .get(v as usize)
                        .map_or(2, |i| model_compiler::lower::dtype_bytes(i.dtype)),
                },
                _ => Arg::Named { value: v, width: width_of(v) },
            }
        };
        // A value-producing GUARD's outputs belong to every launch of its
        // regions (the region's launches "bind the same output buffer and
        // record no SSA outputs of their own" — the recurrence three-way).
        // Map each region op back to its owning guard, once.
        let mut guard_of: Vec<Option<usize>> = vec![None; plan.ops.len()];
        for (g, op) in plan.ops.iter().enumerate() {
            if let OpKind::Guard { arms, else_ops } = &op.kind {
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>()
                    + *else_ops as usize;
                for slot in guard_of.iter_mut().skip(g + 1).take(span) {
                    *slot = Some(g);
                }
            }
        }
        // nemotron's mamba block wires values ACROSS statements: the
        // dt/dA prep and the scan consume the SPLIT's raw `dt` and the
        // PARAMS prep's fp32 tables, none of which their own statements
        // carry (the C++ hand pass routes them through its workspace).
        // Collect them per layer so the arms read a slot.
        let mut mamba_aux: std::collections::BTreeMap<u16, [Option<Arg>; 6]> =
            std::collections::BTreeMap::new();
        for launch in &lowered.launches {
            let op = &plan.ops[launch.op as usize];
            let layer = launch.layers.start;
            match lowered.kernels[launch.kernel as usize].as_str() {
                "ssm::nemotron_mamba_split_bf16" if op.outputs.len() == 3 => {
                    mamba_aux.entry(layer).or_default()[0] = Some(out_arg(op.outputs[2]));
                }
                "ssm::nemotron_prepare_mamba_params" if op.outputs.len() == 3 => {
                    let e = mamba_aux.entry(layer).or_default();
                    for (i, &v) in op.outputs.iter().enumerate() {
                        e[1 + i] = Some(out_arg(v));
                    }
                }
                "ssm::nemotron_prepare_mamba_dt_da" if op.outputs.len() == 2 => {
                    let e = mamba_aux.entry(layer).or_default();
                    e[4] = Some(out_arg(op.outputs[0]));
                    e[5] = Some(out_arg(op.outputs[1]));
                }
                _ => {}
            }
        }
        // The LoRA correction's qkv_in: the statement carries only its
        // in-place [q, v]; the INPUT is "the buffer the projections
        // read" (the C++ arm's own words) — the same layer's qkv/q_proj
        // GEMM's activation arg, collected here.
        let mut lora_x: std::collections::BTreeMap<u16, Arg> = std::collections::BTreeMap::new();
        for launch in &lowered.launches {
            let op = &plan.ops[launch.op as usize];
            if let OpKind::Matmul { weight, .. } = &op.kind
                && (weight.ends_with(".qkv") || weight.ends_with(".q_proj"))
                && launch.args.end > launch.args.start
            {
                lora_x
                    .entry(launch.layers.start)
                    .or_insert_with(|| lowered.args[launch.args.start as usize].clone());
            }
        }
        // The PAIR-form activations (`swiglu` / `swiglu_clamp` / `situ`)
        // state ONE operand where their launcher takes gate AND up: the
        // DSL records one input ("whether the binding materialised it as
        // one buffer or two is a BUFFER question"), the C++ arm reads
        // `ws.up` from its workspace, and the declarations drop the
        // second projection outright (`let _up = matmul(...)`). So `up`
        // is invisible to the statement on both sides and the join has
        // to hand it over — the same service `spec.aux` does for the
        // mamba scan and the LoRA correction. The layer's `up`
        // projection is the one whose weight name says so.
        let mut pair_up: std::collections::BTreeMap<u16, Arg> =
            std::collections::BTreeMap::new();
        for launch in &lowered.launches {
            let op = &plan.ops[launch.op as usize];
            // The `up` projection names itself, whatever the deployment
            // spells around it (`dense_up_proj`, `up_proj`, `shared.up`).
            // The FUSED `gate_up` bank is excluded on purpose: it is the
            // packed operand the CHUNKED forms take, not an `up` half.
            let names_up = |w: &str| {
                let seg = w.rsplit('.').next().unwrap_or(w);
                seg != "gate_up"
                    && (seg == "up"
                        || seg.starts_with("up_")
                        || seg.ends_with("_up")
                        || seg.contains("_up_"))
            };
            if let OpKind::Matmul { weight, .. } = &op.kind
                && names_up(weight)
                && !op.outputs.is_empty()
            {
                pair_up.entry(launch.layers.start).or_insert_with(|| out_arg(op.outputs[0]));
            }
        }
        let mamba_aux_of = |layer: u16| -> Vec<Arg> {
            mamba_aux
                .get(&layer)
                .map(|slots| slots.iter().filter_map(Clone::clone).collect::<Vec<_>>())
                .filter(|v: &Vec<Arg>| v.len() == 6)
                .unwrap_or_default()
        };
        let specs = lowered
            .launches
            .iter()
            .map(|launch| {
                let op = &plan.ops[launch.op as usize];
                let out_values: &[ValueId] = if op.outputs.is_empty() {
                    guard_of[launch.op as usize]
                        .map_or(&[], |g| plan.ops[g].outputs.as_slice())
                } else {
                    &op.outputs
                };
                let outs: Vec<Arg> = out_values.iter().map(|&v| out_arg(v)).collect();
                let mut spec = match &op.kind {
                    OpKind::Embed { weight }
                    | OpKind::Rmsnorm { weight, .. }
                    | OpKind::RmsnormPerHead { weight, .. }
                    | OpKind::AddBias { weight }
                    | OpKind::RmsnormGated { weight }
                    | OpKind::CausalConv1d { weight, .. }
                    | OpKind::LmHead { weight } => LaunchSpec {
                        weight: Some(weight.clone()),
                        ..LaunchSpec::default()
                    },
                    OpKind::Matmul { weight, beta_one, .. } => LaunchSpec {
                        weight: Some(weight.clone()),
                        beta_one: *beta_one,
                        ..LaunchSpec::default()
                    },
                    OpKind::GdnPrep { a_log, dt_bias } => LaunchSpec {
                        weight: Some(a_log.clone()),
                        weight2: Some(dt_bias.clone()),
                        ..LaunchSpec::default()
                    },
                    // A lowered `Launch` states its weights as
                    // `Arg::Weight`s; the FIRST also rides the spec so
                    // constant-naming arms (`scale.*`) can read the name
                    // the bound pointer lost.
                    OpKind::Launch { weights, params, .. } => LaunchSpec {
                        weight: weights.first().cloned(),
                        weight2: weights.get(1).cloned(),
                        params: params.clone(),
                        ..LaunchSpec::default()
                    },
                    _ => LaunchSpec::default(),
                };
                spec.outs = outs;
                spec.n_in = op.inputs.len();
                spec.n_out = op.outputs.len();
                spec.state = op.kind.state_ref();
                if let OpKind::RmsnormPerHead { head_dim, .. } = op.kind {
                    spec.per_head_dim = Some(head_dim);
                }
                if let OpKind::Rope { partial, .. } = op.kind {
                    spec.rope_partial = partial;
                }
                if matches!(
                    lowered.kernels[launch.kernel as usize].as_str(),
                    "ssm::nemotron_prepare_mamba_dt_da" | "ssm::nemotron_mamba_ssm_batched_bf16"
                ) {
                    spec.aux = mamba_aux_of(launch.layers.start);
                }
                if matches!(
                    lowered.kernels[launch.kernel as usize].as_str(),
                    "mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16" | "mlp::situ_bf16"
                ) && let Some(up) = pair_up.get(&launch.layers.start)
                {
                    spec.aux = vec![up.clone()];
                }
                if lowered.kernels[launch.kernel as usize] == "pie_lora_qkv_correction"
                    && let Some(x) = lora_x.get(&launch.layers.start)
                {
                    spec.aux = vec![x.clone()];
                }
                spec
            })
            .collect();
        Self { specs }
    }

    /// The spec for launch `i` — index-parallel with
    /// [`Lowered::launches`].
    #[must_use]
    pub fn spec(&self, i: usize) -> &LaunchSpec {
        &self.specs[i]
    }
}

/// FlashInfer's decode plan cache, owned across the bridge.
///
/// The C++ type is INCOMPLETE on purpose (`struct DecodePlanCache;`), so
/// this is a handle, never a layout — created by the hand-written extras
/// (`pie_x_make_decode_plan`, the factory's `release()`), destroyed by the
/// factory's own deleter. Plain [`Drop`] rather than the crate's explicit
/// `release(&mut ops)` pattern, deliberately: destruction is a pure host
/// `delete` with no CUDA ordering and no recorder seam — there is no
/// oracle that needs to see it.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct DecodePlan {
    cache: *mut c_void,
}

#[cfg(feature = "bridge")]
impl DecodePlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        let cache = unsafe { crate::launch::ffi::pie_x_make_decode_plan() };
        assert!(!cache.is_null(), "make_decode_plan returned null");
        Self { cache }
    }

    /// The raw handle a dispatch arm passes as the `DecodePlanCache&`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache
    }

    /// Where the plan's int arrays sit inside the workspace's
    /// `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        unsafe { crate::launch::ffi::pie_x_set_decode_plan_int_base(self.cache, bytes) };
    }

    /// Run FlashInfer's decode planner over the fire's HOST page indptr.
    ///
    /// The caller brackets this with the workspace's
    /// `begin_plan_update`/`end_plan_update`, exactly as the C++ does —
    /// the planner stages into the view's pinned slot.
    // Safe by design like the seam methods: the view's pointers are the
    // workspace's own, and the stream is the caller's live handle.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_decode(
        &mut self,
        kv_page_indptr_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::launch::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        window_left: i32,
    ) {
        self.plan_decode_variant(
            kv_page_indptr_h,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            stream,
            enable_cuda_graph,
            false,
            window_left,
        );
    }

    /// [`Self::plan_decode`] with the `full_attention_variant` flag
    /// exposed — gemma-4 plans TWO decode caches, one per layer kind
    /// (`decode_plan_full` / `decode_plan_sliding` in the C++), because
    /// the kinds disagree on head dim and the planner bakes it in.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_decode_variant(
        &mut self,
        kv_page_indptr_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::launch::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        full_attention_variant: bool,
        window_left: i32,
    ) {
        let num_requests =
            i32::try_from(kv_page_indptr_h.len() - 1).expect("request count fits i32");
        unsafe {
            crate::launch::ffi::pie_x_plan_attention_flashinfer_decode_bf16(
                self.cache,
                kv_page_indptr_h.as_ptr(),
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                workspace,
                stream,
                enable_cuda_graph,
                full_attention_variant,
                false,
                window_left,
            );
        }
    }
}

#[cfg(feature = "bridge")]
impl Default for DecodePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bridge")]
impl Drop for DecodePlan {
    fn drop(&mut self) {
        unsafe { crate::launch::ffi::pie_x_destroy_decode_plan(self.cache) };
    }
}

/// FlashInfer's prefill plan cache — [`DecodePlan`]'s twin, owned the same
/// way for the same reasons.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct PrefillPlan {
    cache: *mut c_void,
}

#[cfg(feature = "bridge")]
impl PrefillPlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        let cache = unsafe { crate::launch::ffi::pie_x_make_prefill_plan() };
        assert!(!cache.is_null(), "make_prefill_plan returned null");
        Self { cache }
    }

    /// The raw handle a dispatch arm passes.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache
    }

    /// Run FlashInfer's prefill planner over the fire's HOST CSRs.
    ///
    /// Bracket with the workspace's plan-update fence, as with
    /// [`DecodePlan::plan_decode`].
    // Safe by design like the seam methods: the view's pointers are the
    // workspace's own, and the stream is the caller's live handle.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_prefill(
        &mut self,
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::launch::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        window_left: i32,
    ) {
        let num_requests =
            i32::try_from(qo_indptr_h.len() - 1).expect("request count fits i32");
        let total_tokens =
            i32::try_from(*qo_indptr_h.last().expect("a CSR has a last entry"))
                .expect("token count fits i32");
        unsafe {
            crate::launch::ffi::pie_x_plan_attention_flashinfer_prefill_bf16(
                self.cache,
                qo_indptr_h.as_ptr(),
                kv_page_indptr_h.as_ptr(),
                kv_last_page_lens_h.as_ptr(),
                total_tokens,
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                workspace,
                stream,
                enable_cuda_graph,
                window_left,
                false,
                false,
                true,
                false,
                false,
            );
        }
    }
}

#[cfg(feature = "bridge")]
impl Default for PrefillPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bridge")]
impl Drop for PrefillPlan {
    fn drop(&mut self) {
        unsafe { crate::launch::ffi::pie_x_destroy_prefill_plan(self.cache) };
    }
}

/// The scalar facts a dispatch arm reads beside its bound operands.
///
/// Everything else an arm needs is IN the launch: row counts from
/// `rows`, per-operand widths from the args. What remains is the
/// deployment's constants — the same values the C++ arms read off their
/// facts structs — and the per-fire handles.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct DispatchCtx {
    /// The fire's stream.
    pub stream: *mut c_void,
    /// The cuBLAS handle `gemm::act_x_w` routes through.
    pub cublas: *mut c_void,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Rope theta, for the table fill.
    pub rope_theta: f32,
    /// PER-LAYER rope theta for families whose `rope_parameters` split
    /// by layer kind (gemma-4: sliding 1e4, full 1e6 — the C++ expands
    /// the HF map into `per_layer_rope_theta[L]` at parse time). Empty
    /// means uniform [`Self::rope_theta`].
    pub rope_theta_by_layer: Vec<f32>,
    /// PER-LAYER partial-rotary width, the C++ `rotary_of(l)` table
    /// (`max(2, 2*int(0.5*factor*head_dim))` per layer). Consulted only
    /// when a `rope::rope_partial_bf16` op states no width of its own —
    /// gemma-4's Q-ONLY form on KV-shared full layers, whose dsl
    /// statement carries one operand and no param. Empty means every
    /// partial-rope op states its width.
    pub rotary_by_layer: Vec<u32>,
    /// Head width, for the table fill.
    pub head_dim: i32,
    /// The head COUNTS, which a row may name the same way it names
    /// `head_dim` — a fire-wide geometry fact, not something an arm
    /// derives from a width. Added because the rows say `num_q_heads` /
    /// `num_kv_heads` and a context that spelled them otherwise would
    /// need a translation table between the declaration and the driver,
    /// which is the thing being removed.
    pub num_q_heads: i32,
    /// See [`Self::num_q_heads`].
    pub num_kv_heads: i32,
    /// Vocabulary rows the embed weight holds.
    pub vocab: i32,
    /// The packed gate‖up order `chunked_swiglu` was bound with.
    pub gate_second: bool,
    /// GPT-J adjacent-pair rotation (`rope_interleave`), vs NeoX half/half.
    pub rope_interleaved: bool,
    /// The fire's token ids (device i32, one per row) — the embed's
    /// input, which is the backend's to provide rather than an arg.
    pub token_ids: *mut c_void,
    /// The fire's positions (device i32, one per row) — the rope table's
    /// input, provided the same way.
    pub positions: *mut c_void,
    /// gemma's FINAL logit softcap (`cap * tanh(x / cap)` over the
    /// logits) — the value behind the `attn::logit_softcap_bf16` launch,
    /// which the trace states only when the deployment configures it.
    pub final_logit_softcap: f32,
    /// gemma-4's per-layer embedding width (`ple_dim`) — what the PLE
    /// relay transpose divides its flat `[N, layers*dim]` row by. Zero
    /// on families without a PLE.
    pub ple_dim: i32,
    /// The scalar constants `norm::scalar_mul_bf16` launches name in
    /// their `scale.<name>` weight slot — `sqrt(hidden)` on the
    /// embedding, gemma's query pre-scale. Resolved here by NAME because
    /// a scale is a constant, not a tensor (the dsl's own words).
    pub scales: std::collections::BTreeMap<String, f32>,
    /// The sigmoid router's `norm_topk_prob` (`cfg` on the C++ side) —
    /// what `moe::topk_sigmoid_bias_fp32` normalizes by.
    pub moe_norm_topk: bool,
    /// The router's `routed_scaling_factor`; 1.0 when unstated.
    pub moe_routed_scaling: f32,
    /// gpt-oss's YaRN parameters, in the launcher's own order —
    /// `(factor, beta_fast, beta_slow, attention_factor)` — and the
    /// original context the scaling is measured against. Zero on the
    /// families whose rope is plain.
    pub yarn: [f32; 4],
    /// `original_max_position_embeddings` for the YaRN rope.
    pub yarn_original_max: i32,
    /// gpt-oss's clamped GLU constants: `swiglu_limit` is a config value
    /// (which is why the family states its own activation kernel rather
    /// than passing a limit through the ordinary one) and `alpha` its
    /// companion.
    pub glu_limit: f32,
    /// The clamped GLU's alpha.
    pub glu_alpha: f32,
    /// SiTU's `beta` and `linear_beta` — the tanh-gated activation's own
    /// constants, which is why it is not a swiglu variant (the tanh
    /// saturates far enough out that a bf16 intermediate loses the
    /// distinction the gate exists to make).
    pub situ_beta: f32,
    /// SiTU's linear beta.
    pub situ_linear_beta: f32,
    /// The WNA16 experts' quantisation group size.
    pub wna16_group_size: i32,
    /// gemma3n's AltUp rank-K residual: the stream count `K` and the
    /// ACTIVE stream's index (`cfg.altup_active_idx`). The launches that
    /// address `[K, tokens, hidden]` values state a zero width — a
    /// three-dimensional shape has no single row width — so the count
    /// comes from here rather than from an operand.
    pub altup_streams: i32,
    /// The active stream's index.
    pub altup_active: i32,
    /// Per-layer `gaussian_inverse_cdf(activation_sparsity)` — the
    /// `std_mult` the sparse layers' `gaussian_topk` takes. A HOST
    /// derivation from the config (the C++ computes it per layer at fire
    /// time); empty means the deployment states no sparse layer.
    pub altup_std_mult_by_layer: Vec<f32>,
    /// The fire's PEEL WINDOW, device-resident: `[start, count]`, or null
    /// when this fire has no row split. The `_devwin` statements in a
    /// peel's tail read it — their grid spans every lane and out-of-window
    /// rows early-out, which is what lets one capture replay across
    /// different splits. See [`crate::cuda::PeelWindowWord`].
    pub peel_window: *const u32,
    /// The fire's FULL row count, which a `_devwin` launch spans
    /// regardless of how many rows its own region serves. `Launch::rows`
    /// gives the region; this gives the lane space it sits in.
    pub rows_total: i32,
    /// The fire's STAGED LoRA state and its xAᵀ scratch (`ws.gate` in
    /// the C++) — what a stated `pie_lora_qkv_correction` launch
    /// applies. Null/None on adapter-free fires. A raw pointer because
    /// the state outlives the fire on the caller's side (the plan
    /// state's `lora_staged` slot) and the ctx carries no lifetime.
    pub lora: Option<(*const super::lora::LoraFireState, *mut c_void)>,
}

#[cfg(feature = "bridge")]
impl DispatchCtx {
    /// The theta a layer-tagged rope launch fires with: the per-layer
    /// entry when the family splits theta by layer kind, else the
    /// uniform value.
    fn theta_of(&self, layer: usize) -> f32 {
        self.rope_theta_by_layer
            .get(layer)
            .copied()
            .unwrap_or(self.rope_theta)
    }
}

/// The fire's attention context: what the attention arms need beyond
/// args and the op join — the planned cache, the workspace, the per-layer
/// KV views, and the fire's device-resident page CSRs and write
/// descriptors. The ENGINE'S half of a fire, assembled once.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct AttnCtx {
    /// The planned [`DecodePlan`]'s handle. Null on a pure-prefill fire.
    pub decode_plan: *mut c_void,
    /// The FULL-attention layers' decode plan, for families whose two
    /// layer kinds disagree on head dim (gemma-4: 512 vs 256 — the C++
    /// keeps `decode_plan_full` beside `decode_plan_sliding`). Null on
    /// single-kind families; the decode arm picks it when the layer's
    /// window says FULL (`window_left_by_layer[l] == -1`).
    pub decode_plan_full: *mut c_void,
    /// The planned [`PrefillPlan`]'s handle. Null on a pure-decode fire.
    pub prefill_plan: *mut c_void,
    /// The workspace, as launchers take it.
    pub workspace: crate::launch::AttentionWorkspaceView,
    /// One KV view per layer, indexed by the launch's layer.
    pub layers: Vec<crate::launch::KvCacheLayerView>,
    /// Device page-index CSR.
    pub kv_page_indices_d: *const u32,
    /// Device page indptr.
    pub kv_page_indptr_d: *const u32,
    /// Device last-page lengths.
    pub kv_last_page_lens_d: *const u32,
    /// Device query indptr — prefill's token-rows-per-request CSR.
    pub qo_indptr_d: *const u32,
    /// HOST qo indptr — the planless prefill dispatch plans internally
    /// per fire and reads the CSR from the host. Null when no planless
    /// launch is stated.
    pub qo_indptr_h: *const u32,
    /// HOST kv page indptr, the planless dispatch's other host read.
    pub kv_page_indptr_h: *const u32,
    /// Requests in the fire (`indptr.len() - 1`).
    pub num_requests: i32,
    /// Pages the fire's CSR names — what the dequant staging walks.
    pub num_pages_in_batch: i32,
    /// `write_kv_to_pages`'s first-token scalar (the fire's write origin).
    pub first_token: i32,
    /// Per-row target page for this fire's KV append.
    pub w_page_d: *const u32,
    /// Per-row offset-in-page for the append.
    pub w_off_d: *const u32,
    /// Per-row validity for the append.
    pub row_valid_d: *const u8,
    /// The observed-query pin the fused qkv writes and the dispatch
    /// reads. A GUARD-owned value (the region's launches record no SSA
    /// outputs of their own), so it is fire context until the join learns
    /// to walk back to the guard op.
    pub q_out: *mut c_void,
    /// The folded attention SCORES a `WantsAttnScore` fire captures, and
    /// the device CSR saying where each request's rows begin.
    ///
    /// Both must be ARENA-STABLE, which is not a detail: scores are a
    /// FOLDED predicate (`SLOT_WANTS_ATTN_SCORE`), so one captured exec
    /// serves a fire that wants them and a fire that does not, and an
    /// address recorded now has to still mean something when the
    /// predicate goes true. `attn_score::DecodeScoreCapturePlan` exists
    /// to answer exactly that — "arena-stable folded-row base",
    /// "arena-stable device CSR base" — and these are where its answer
    /// reaches the arm.
    pub score_out: *mut f32,
    /// See [`Self::score_out`].
    pub score_indptr_d: *const i32,
    /// The attention output slot the o_proj reads — guard-owned like
    /// `q_out`, and one arena slot reused by every layer (liveness).
    pub o_out: *mut c_void,
    /// LSE scratch the decode dispatch writes.
    pub lse_out_d: *mut f32,
    /// Sliding-window extent, `-1` for none.
    pub window_left: i32,
    /// PER-LAYER window extents for alternating-window families
    /// (gemma's global/local schedule); empty means uniform
    /// [`Self::window_left`].
    pub window_left_by_layer: Vec<i32>,
    /// Logit soft cap, `0` for none.
    pub logits_soft_cap: f32,
    /// The attention scale (`1/sqrt(head_dim)` unless overridden).
    pub sm_scale: f32,
}

/// The fire's GDN context: what the linear-attention arms need beyond
/// args and the op join — the per-layer conv/recurrent state slabs, the
/// request→slot indirection, and the deployment's head geometry. The
/// C++ executor reads these off `RecurrentStateCache` + facts per launch;
/// here they are assembled once per fire, [`AttnCtx`]-style.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct GdnCtx {
    /// Key heads (compact, pre-GQA-repeat).
    pub k_h: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key head width.
    pub k_d: i32,
    /// Value head width.
    pub v_d: i32,
    /// Conv channels (`2*K_h*K_d + V_h*V_d`).
    pub conv_dim: i32,
    /// Conv window width (`linear_conv_kernel_dim`).
    pub conv_k: i32,
    /// mamba's B/C group count (`n_groups`) — nemotron's selective scan
    /// and its grouped gated norm read it; zero on GDN families. When a
    /// fire is MAMBA, the head fields above map as: `v_h` = num_heads,
    /// `v_d` = head_dim, `k_d` = state_size — the state stride formula
    /// (`v_h·k_d·v_d`) then reads `heads·state·head_dim`, which IS
    /// mamba's slab, so the two shapes share one context.
    pub n_groups: i32,
    /// Device base of each MODEL layer's conv-state slab (slot 0); zero
    /// for layers with no linear-attention state.
    pub conv_state: Vec<u64>,
    /// Elements per conv slot (`conv_k * conv_dim`).
    pub conv_stride_elems: i64,
    /// Device base of each MODEL layer's recurrent-state slab (slot 0),
    /// in the store's own dtype (fp32 or bf16 — the deployment's
    /// `state_bf16` fact); zero for non-linear layers.
    pub recurrent_state: Vec<u64>,
    /// Elements per recurrent slot.
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state (true for Decode/Prefill; the
    /// frozen-verify service classes pass false).
    pub write_state: bool,
}

/// Why a bound launch could not be dispatched.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchRefusal {
    /// No arm exists for this kernel yet. The executor grows kernel by
    /// kernel, and an explicit refusal is what keeps a missing arm from
    /// reading as a covered launch.
    NoArm(String),
    /// The op join names a weight the resolver does not hold — the same
    /// drift [`BindRefusal::UnknownWeight`] diagnoses for stated args.
    UnknownWeight(String),
    /// The arm expected the op join to name a weight and it named none —
    /// the arm and the lowering disagree about the statement's shape.
    NoWeight(String),
    /// An attention arm ran without an [`AttnCtx`], or with one whose
    /// layer list does not cover the launch's layer.
    NoAttnCtx(String),
    /// A GDN arm ran without a [`GdnCtx`], or with one whose state
    /// vectors do not cover the launch's state layer.
    NoGdnCtx(String),
    /// An output placement failed to resolve — the join and the resolver
    /// disagree.
    Out(String),
    /// The launcher this arm calls would DECLINE this shape, and
    /// declining is spelled as a silent early return rather than as an
    /// error.
    ///
    /// `moe::moe_grouped_gemm_bf16` is the one that needs this: it opens
    /// `if (max_blocks <= 0 || !supported(M, N, K)) return;`, so an arm
    /// that simply called it would leave the destination holding whatever
    /// was there and the fire would keep going. The C++ driver reads the
    /// same predicate and takes a batched-cuBLAS fallback; until that
    /// fallback exists here, saying so is the only honest answer.
    ///
    /// Smoothly wrong is the failure mode this tree keeps naming, and a
    /// silent no-op inside a GEMM is its purest form.
    ShapeDeclined {
        /// The kernel whose launcher declines.
        kernel: String,
        /// Why, in the launcher's own terms.
        why: String,
    },
    /// The arm and the lowering disagree about the operand count — a
    /// drift between the trace's statement and this arm's reading of it.
    ArgCount {
        /// The kernel whose arm refused.
        kernel: String,
        /// Operands the arm expects.
        expected: usize,
        /// Operands the launch bound.
        got: usize,
    },
}
/// The GENERATED dispatch: one branch per row that states its sources.
///
/// Ran BEFORE the hand-written match, and `false` means "not mine" —
/// either no branch names this symbol, or the branch's guard was not
/// satisfied and the statement is the other spelling. Both fall through
/// to the arm that knows, which is the same fallthrough the C++ driver's
/// generated switch has and for the same reason: a generated branch must
/// decline rather than guess.
///
/// The included file is `emit_rust_dispatch`'s output over the same
/// tables the shim and the bindings come from — one read of one table in
/// one build script, so the three cannot disagree with each other.
#[cfg(feature = "bridge")]
fn dispatch_generated(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    rows: i32,
) -> bool {
    // What a generated branch may read about an operand: its row width,
    // its row count, and the product. Free functions rather than inline
    // expressions because every branch wants them and a generator that
    // re-derived them per branch would be duplicating in the one place
    // this whole exercise removes duplication from.
    //
    // `0` for an index the run does not hold. A branch that could index
    // past its run is refused by its own guard before it gets here, so
    // this is the belt to that suspenders — it must not fault, because a
    // read past the run does not fault either, it reads the NEXT
    // operand.
    fn width_of(b: &BoundLaunch<'_>, i: usize) -> i32 {
        b.args.get(i).map_or(0, |a| i32::try_from(a.width).unwrap_or(0))
    }
    fn rows_of(b: &BoundLaunch<'_>, i: usize, rows: i32) -> i32 {
        let _ = (b, i);
        rows
    }
    fn elems_of(b: &BoundLaunch<'_>, i: usize, rows: i32) -> usize {
        (rows.max(0) as usize) * (width_of(b, i).max(0) as usize)
    }

    /// `Source::CtxNonZero`'s test: a family zeroes a context field to
    /// say "this launch is not mine". The generator emits the call and
    /// not `!= 0` because it does not know the field's TYPE, and Rust
    /// will not compare an `f32` to an integer literal — so the one
    /// thing the generator cannot know lives on the side that knows it.
    trait IsSet {
        fn is_set(self) -> bool;
    }
    impl IsSet for f32 {
        fn is_set(self) -> bool {
            self != 0.0
        }
    }
    impl IsSet for i32 {
        fn is_set(self) -> bool {
            self != 0
        }
    }
    impl IsSet for u32 {
        fn is_set(self) -> bool {
            self != 0
        }
    }
    fn is_set<T: IsSet>(v: T) -> bool {
        v.is_set()
    }

    // A JOIN FACT NO `Source` CAN NAME declines the whole branch.
    //
    // The generator binds from the ROW, and the row describes the
    // launcher. What it cannot describe is a fact the op join carries
    // ABOUT this statement — and those change the arithmetic, not the
    // operands, so a generated branch reads right and computes wrong.
    //
    // `per_head_dim` is the live one and gemma-4 is where it bites:
    // `OpKind::RmsnormPerHead` lowers to `norm::rmsnorm_bf16`, the same
    // symbol the plain kind does, and the per-head reading is `rows *
    // (width / head_dim)` rows of `head_dim` against the flat reading's
    // `rows` of `width`. A generated branch binds `Rows` and
    // `InWidth(0)` and would norm gemma-4's q/k heads as one row each.
    //
    // `rope_partial` and `aux` are the same class: a rotary width the
    // statement carries out of band, and operands the trace does not
    // state at all. `beta_one` is not — it changes which ARG is the
    // destination, which arity guards already catch.
    //
    // This is the fallthrough working as designed, not a special case:
    // the hand arm reads the join and the generated branch says "not
    // mine" rather than guessing which reading applies.
    if spec.per_head_dim.is_some() || spec.rope_partial.is_some() || !spec.aux.is_empty() {
        return false;
    }

    let n_in = spec.n_in;
    let n_out = spec.n_out;
    include!(concat!(env!("OUT_DIR"), "/rust_dispatch.rs"))
}


/// Dispatch one bound launch through its `pie_k_*` entry.
///
/// The arms cover the anchor deployment's compute backbone — embed, the
/// rope table, rmsnorm, the quantized-dispatch GEMM, chunked swiglu.
/// Operand order inside each arm is the trace's stated order (inputs,
/// then outputs, then weights), which the numeric smoke verifies — a
/// swapped operand is wrong VALUES, not a type error, and only a check
/// against host math catches it.
///
/// # Errors
///
/// See [`DispatchRefusal`].
#[cfg(feature = "bridge")]
#[allow(clippy::too_many_lines)]
pub fn dispatch<R: Resolver>(
    bound: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    frame: Frame,
    resolver: &mut R,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
    gdn: Option<&GdnCtx>,
) -> Result<(), DispatchRefusal> {
    use crate::launch::ffi;

    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("rows fit i32");

    // GENERATED FIRST. A row that states where its arguments come from
    // needs no arm, and the branch for it is emitted from the row — so
    // the hand-written match below is what is LEFT, not what is normal.
    // It shrinks as rows state their sources, which is a row's work.
    if dispatch_generated(bound, spec, ctx, rows) {
        return Ok(());
    }

    // The GDN arms' shared reads: the ctx itself, and the launch's state
    // layer's slab out of one of its per-layer vectors.
    let gdn_ctx = || -> Result<&GdnCtx, DispatchRefusal> {
        gdn.ok_or_else(|| DispatchRefusal::NoGdnCtx(bound.kernel.to_string()))
    };
    let state_layer = || -> Result<usize, DispatchRefusal> {
        spec.state
            .map(|s| s.layer as usize)
            .ok_or_else(|| DispatchRefusal::NoGdnCtx(format!("{}: op states no layer", bound.kernel)))
    };
    let slab = |v: &[u64], layer: usize, what: &str| -> Result<*mut c_void, DispatchRefusal> {
        match v.get(layer) {
            Some(&base) if base != 0 => Ok(base as *mut c_void),
            _ => Err(DispatchRefusal::NoGdnCtx(format!(
                "{}: layer {layer} has no {what} slab",
                bound.kernel
            ))),
        }
    };

    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("row count fits i32");
    // The op join's output placements: what a guard-region launch binds
    // for the value the GUARD owns (the recurrence three-way's core out).
    // The join's placements window with the args, or a launch reads its
    // input at the window and writes its output at the base.
    let win = if bound.kernel.ends_with("_devwin") { 0 } else { bound.rows.start };
    let out_slot = |i: usize, resolver: &mut R| -> Result<BoundArg, DispatchRefusal> {
        let arg = spec
            .outs
            .get(i)
            .ok_or_else(|| DispatchRefusal::Out(format!("{}: no output {i}", bound.kernel)))?;
        resolve_arg_windowed(arg, frame, resolver, win)
            .map_err(|e| DispatchRefusal::Out(format!("{}: {e:?}", bound.kernel)))
    };
    // The spec's FOREIGN values (`LaunchSpec::aux`) — nemotron's mamba
    // wiring — resolved exactly like the outs.
    let aux_slot = |i: usize, resolver: &mut R| -> Result<BoundArg, DispatchRefusal> {
        let arg = spec
            .aux
            .get(i)
            .ok_or_else(|| DispatchRefusal::Out(format!("{}: no aux {i}", bound.kernel)))?;
        resolve_arg_windowed(arg, frame, resolver, win)
            .map_err(|e| DispatchRefusal::Out(format!("{}: {e:?}", bound.kernel)))
    };
    let need = |n: usize| -> Result<(), DispatchRefusal> {
        if bound.args.len() == n {
            Ok(())
        } else {
            Err(DispatchRefusal::ArgCount {
                kernel: bound.kernel.to_string(),
                expected: n,
                got: bound.args.len(),
            })
        }
    };
    let weight = |resolver: &mut R| -> Result<*const c_void, DispatchRefusal> {
        let name = spec
            .weight
            .as_deref()
            .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
        resolver
            .weight(name)
            .ok_or_else(|| DispatchRefusal::UnknownWeight(name.to_string()))
    };

    match bound.kernel {
        // args: [y]. The token ids are the fire's input and the weight is
        // the op's — both context, neither an arg.
        "layout::embed_bf16" => {
            need(1)?;
            let y = bound.args[0];
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_layout_embed_bf16(
                    ctx.token_ids.cast_const().cast(),
                    w,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("hidden fits i32"),
                    ctx.vocab,
                    ctx.stream,
                );
            }
        }
        // args: [table]; positions are the fire's.
        "rope::rope_standard_table" => {
            need(1)?;
            let table = bound.args[0];
            unsafe {
                ffi::pie_k_rope_rope_standard_table(
                    ctx.positions.cast_const().cast(),
                    table.ptr.cast(),
                    rows,
                    ctx.head_dim,
                    ctx.rope_theta,
                    ctx.stream,
                );
            }
        }
        // args: [x, y]; the norm weight is the op's.
        "norm::rmsnorm_bf16" => {
            // `[x, y]` when the SEMANTIC `Rmsnorm` lowered here and the
            // weight reached the arm through the resolver; `[x, y, w]`
            // now that `dsl::cuda::rmsnorm` STATES the kernel and names
            // its weight, which the binder resolves like any operand.
            // Both forms are live while both spellings are.
            let (x, y) = (bound.args[0], bound.args[1]);
            let w = match bound.args.len() {
                2 => weight(resolver)?,
                3 => bound.args[2].ptr.cast_const(),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 3,
                        got,
                    });
                }
            };
            // A PER-HEAD statement reshapes to `[rows*heads, head_dim]`
            // rows over one head-wide weight — gemma-4's `ple_model_norm`
            // ([256] over an `[N, layers*256]` row) and its full layers'
            // plain q/k norms ([512] over `[N, 4096]`). Firing the flat
            // width here read past those weights — the whole-row form is
            // the same kernel with heads = 1.
            let (num_rows, hidden) = match spec.per_head_dim {
                Some(d) => (
                    rows * (i32::try_from(x.width).expect("width") / i32::try_from(d).expect("d")),
                    i32::try_from(d).expect("head_dim fits i32"),
                ),
                None => (rows, i32::try_from(x.width).expect("hidden fits i32")),
            };
            unsafe {
                ffi::pie_k_norm_rmsnorm_bf16(
                    x.ptr,
                    w,
                    y.ptr,
                    num_rows,
                    hidden,
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [act, y] with beta 0, or [act, resid_in, y] with beta 1 —
        // the residual fold, where the output aliases the residual's
        // bytes and cuBLAS accumulates in place. M/K/N come from the
        // rectangle and the widths; the weight is the op's.
        //
        // The symbol the lowering states is the DENSE matmul — a
        // `WeightRepr::Bf16` weight, which is the one representation
        // `MatW::gemm_symbol` declines to name, because there is nothing
        // to choose. Every other representation states its own symbol
        // and lands on its own arm. So this arm binds
        // `gemm::act_x_wt_bf16`, which `gemm.hpp` defines as `act_x_w`
        // with `WeightView::raw(W, BF16)` — the one view this arm ever
        // built, now assembled inside the launcher where the routing
        // lives, rather than crossing the ABI as a descriptor.
        "gemm::act_x_w" => {
            let (act, y, beta) = if spec.beta_one {
                need(3)?;
                (bound.args[0], bound.args[2], 1.0f32)
            } else {
                need(2)?;
                (bound.args[0], bound.args[1], 0.0f32)
            };
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_gemm_act_x_wt_bf16(
                    ctx.cublas,
                    act.ptr,
                    w,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("N fits i32"),
                    i32::try_from(act.width).expect("K fits i32"),
                    beta,
                );
            }
        }
        // args: [packed, y].
        "mlp::chunked_swiglu_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_swiglu_bf16(
                    packed.ptr,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("I fits i32"),
                    ctx.stream,
                    ctx.gate_second,
                );
            }
        }
        // args: [packed, rope_table, q_norm_w, k_norm_w]; the q output is
        // the observed-query PIN (outs[0], Named); the KV pages, CSRs and
        // write descriptors are the fire's ([`AttnCtx`]).
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16" => {
            need(4)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (packed, table, qw, kw) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_qkv_decode_qk_norm_rope_write_kv_bf16(
                    packed.ptr,
                    a.q_out,
                    layer.k_pages,
                    layer.v_pages,
                    qw.ptr,
                    kw.ptr,
                    ctx.positions.cast_const().cast(),
                    table.ptr.cast_const().cast(),
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.w_page_d,
                    a.w_off_d,
                    a.row_valid_d,
                    rows,
                    (i32::try_from(packed.width).expect("packed width")
                        - 2 * layer.num_kv_heads * layer.head_dim)
                        / layer.head_dim.max(1),
                    layer.num_kv_heads,
                    layer.head_dim,
                    layer.page_size,
                    layer.hnd_layout,
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [q (the pin)]; o is the op's arena output; the plan, the
        // workspace and the layer view are the fire's.
        // args: as the plain decode dispatch, plus the SCORE outputs.
        // `_capture` is capturing scores, not capturing a graph —
        // `WantsAttnScore` is the guard that selects this spelling.
        //
        // The score buffers ride the ctx rather than the statement because
        // they must be arena-STABLE: the predicate is folded, so one exec
        // serves a fire that wants scores and one that does not, and an
        // address recorded now has to still be right when it goes true.
        "attn::dispatch_attention_flashinfer_decode_capture" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            if a.score_out.is_null() || a.score_indptr_d.is_null() {
                return Err(DispatchRefusal::NoAttnCtx(format!(
                    "{}: the fire published no score buffers",
                    bound.kernel
                )));
            }
            let (q, o, lse) = match bound.args.len() {
                1 => (bound.args[0], a.o_out, a.lse_out_d),
                2 => (bound.args[0], bound.args[1].ptr, a.lse_out_d),
                3 => (bound.args[0], bound.args[1].ptr, bound.args[2].ptr.cast()),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            let window = window_of(spec, a, u32::from(bound.layers.start));
            let plan = if window == -1 && !a.decode_plan_full.is_null() {
                a.decode_plan_full
            } else {
                a.decode_plan
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_decode_capture(
                    plan.cast_const(),
                    q.ptr,
                    *layer,
                    o,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    a.score_out,
                    a.score_indptr_d,
                    window,
                    a.logits_soft_cap,
                    a.sm_scale,
                    lse,
                );
            }
        }
        "attn::dispatch_attention_flashinfer_decode" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            // [q] with the output guard-owned (`AttnCtx::o_out`), or
            // [q, o] when the op records its output as an SSA arg.
            // [q] with the output guard-owned, [q, o] when the op
            // records its output, or [q, o, lse] when the fire CONSUMES
            // the log-sum-exp downstream — gpt-oss's sink rescale reads
            // it, so its trace states the buffer instead of leaving it to
            // the context.
            let (q, o, lse) = match bound.args.len() {
                1 => (bound.args[0], a.o_out, a.lse_out_d),
                2 => (bound.args[0], bound.args[1].ptr, a.lse_out_d),
                3 => (bound.args[0], bound.args[1].ptr, bound.args[2].ptr.cast()),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            let window = window_of(spec, a, u32::from(bound.layers.start));
            // Two-kind families keep a second plan for the FULL layers
            // (gemma-4's 512-wide kind) — the C++'s `cur_full ?
            // decode_plan_full : decode_plan_sliding` selection, keyed
            // here on the layer's window because FULL is exactly the
            // unbounded kind.
            let plan = if window == -1 && !a.decode_plan_full.is_null() {
                a.decode_plan_full
            } else {
                a.decode_plan
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_decode(
                    plan.cast_const(),
                    q.ptr,
                    *layer,
                    o,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    window,
                    a.logits_soft_cap,
                    a.sm_scale,
                    lse,
                );
            }
        }
        // args: [packed, q_raw, k_raw, v] — one input, then the op's THREE
        // outputs stated as args (SplitQkv's outputs are values).
        // args: [packed, q, k, v] — the same four the host-window form
        // takes, and the SAME base pointers. The difference is where the
        // row window comes from: a peel's tail addresses rows at absolute
        // offsets in a full-N buffer, so the split rides in device memory
        // and the grid spans every lane, out-of-window rows early-outing
        // on `win[0]`/`win[1]`. That is what makes the launch replayable
        // across splits, which is the whole reason the region asks for
        // this kernel instead of choosing it.
        //
        // Note the operands are NOT windowed by the caller here, and that
        // is the kernel's stated contract ("Buffers are BASE pointers")
        // rather than an oversight — the binder's base-resolving
        // behaviour, which §4's fourth decline-rule works around for the
        // host-window form, is exactly right for this one.
        "attn::split_qkv_bf16_devwin" => {
            need(4)?;
            let (packed, q, k, v) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            if ctx.peel_window.is_null() {
                return Err(DispatchRefusal::NoArm(
                    "attn::split_qkv_bf16_devwin: the fire published no peel window".into(),
                ));
            }
            unsafe {
                ffi::pie_k_attn_split_qkv_bf16_devwin(
                    packed.ptr,
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    ctx.peel_window,
                    ctx.rows_total,
                    i32::try_from(q.width).expect("q width"),
                    i32::try_from(k.width).expect("kv width"),
                    ctx.stream,
                );
            }
        }
        "attn::split_qkv_bf16" => {
            need(4)?;
            let (packed, q, k, v) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_split_qkv_bf16(
                    packed.ptr,
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    rows,
                    i32::try_from(q.width).expect("q width"),
                    i32::try_from(k.width).expect("kv width"),
                    ctx.stream,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out, q_norm_w, k_norm_w]. The KERNEL
        // is in-place on (q, k); the lowering assigned separate in/out
        // buffers, so the arm stages in→out with a d2d copy, then runs the
        // kernel over the outs — the only reading under which both the
        // row's signature and the launch's buffer assignment are honest.
        "rope::qk_rmsnorm_rope_bf16" => {
            need(6)?;
            let (q_in, k_in, q_out, k_out, qw, kw) = (
                bound.args[0],
                bound.args[1],
                bound.args[2],
                bound.args[3],
                bound.args[4],
                bound.args[5],
            );
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            unsafe {
                ffi::pie_k_rope_qk_rmsnorm_rope_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    qw.ptr,
                    kw.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    i32::try_from(k_out.width).expect("k width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [k_curr, v_curr]; the layer view, the CSRs and the fire
        // scalars are the fire's.
        // args: [k_curr, v_curr]. The write-descriptor spelling: the fire
        // steers a graph replay, so the destination page and offset of
        // every row are DESCRIPTORS the host published rather than
        // something the kernel derives from the CSRs. `HasWriteDesc` is
        // the guard that picks it, and `AttnCtx` already carried the three
        // descriptor arrays — the arm was simply never written, because
        // nothing in the corpus set the mark.
        "attn::write_kv_explicit_bf16" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            if a.w_page_d.is_null() || a.w_off_d.is_null() {
                return Err(DispatchRefusal::NoAttnCtx(format!(
                    "{}: the fire published no write descriptors",
                    bound.kernel
                )));
            }
            let (k_curr, v_curr) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_write_kv_explicit_bf16(
                    *layer,
                    k_curr.ptr,
                    v_curr.ptr,
                    a.w_page_d,
                    a.w_off_d,
                    rows,
                    ctx.stream,
                    a.row_valid_d,
                );
            }
        }
        "attn::write_kv_to_pages" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (k_curr, v_curr) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_write_kv_to_pages(
                    *layer,
                    k_curr.ptr,
                    v_curr.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    rows,
                    a.num_requests,
                    ctx.stream,
                    a.row_valid_d,
                    a.first_token,
                );
            }
        }
        // args: [] — everything is the fire's. A no-op on a native cache,
        // and the arm still fires it: the launch is stated, so it runs.
        "attn::dequant_kv_cache_layer_to_bf16_active" => {
            need(0)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            unsafe {
                ffi::pie_k_attn_dequant_kv_cache_layer_to_bf16_active(
                    *layer,
                    a.kv_page_indices_d,
                    a.num_pages_in_batch,
                    ctx.stream,
                );
            }
        }
        // args: [q]; o is guard-owned ([`AttnCtx::o_out`]); the pages are
        // the layer's bf16 MIRRORS — the native alias, the decode lesson.
        // The prefill sibling of the score-capturing decode dispatch, and
        // the same story: `WantsAttnScore` selects it, the score buffers
        // ride the ctx because they must be arena-stable under a folded
        // predicate, and it takes one more output than the decode form —
        // `folded_out` beside `score_out`, since a prefill's raw scores
        // and their per-request fold are different extents.
        //
        // The fold shares `score_out`'s slot here: an empty CSR makes both
        // zero-length, which is what a fire that wants no scores means,
        // and a fire that does want them needs `DecodeScoreCapturePlan`'s
        // layout for both anyway.
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            if a.score_out.is_null() || a.score_indptr_d.is_null() {
                return Err(DispatchRefusal::NoAttnCtx(format!(
                    "{}: the fire published no score buffers",
                    bound.kernel
                )));
            }
            let (q, o) = match bound.args.len() {
                1 => (bound.args[0], a.o_out),
                2 => (bound.args[0], bound.args[1].ptr),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_prefill_capture_bf16(
                    a.prefill_plan.cast_const(),
                    q.ptr,
                    layer.k_bf16_pages,
                    layer.v_bf16_pages,
                    o,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    a.score_out,
                    a.score_out,
                    a.score_indptr_d,
                    // The OBSERVATION window, not the attention one. The
                    // launcher refuses `<= 0`, and `window_left` is -1 on
                    // a family that attends the whole context — passing it
                    // here reads as "no window" to one layer and "invalid"
                    // to the other. It is a driver policy, which is what
                    // `attn_score` is for.
                    i32::try_from(crate::model::attn_score::default_attn_score_window())
                        .unwrap_or(i32::MAX),
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        "attn::dispatch_attention_flashinfer_prefill_bf16" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            // [q] with the output guard-owned, or [q, o] as SSA.
            let (q, o) = match bound.args.len() {
                1 => (bound.args[0], a.o_out),
                2 => (bound.args[0], bound.args[1].ptr),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_prefill_bf16(
                    a.prefill_plan.cast_const(),
                    q.ptr,
                    layer.k_bf16_pages,
                    layer.v_bf16_pages,
                    o,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out] — the same staged-in-place
        // shape as `qk_rmsnorm_rope`: the kernel rotates (q, k) where they
        // lie, the lowering may assign fresh out buffers, so the arm
        // stages in→out then rotates the outs.
        "rope::rope_bf16" => {
            need(4)?;
            let (q_in, k_in, q_out, k_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            unsafe {
                ffi::pie_k_rope_rope_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    i32::try_from(k_out.width).expect("k width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.stream,
                    ctx.rope_interleaved,
                );
            }
        }
        // args: [act, expert_ids, stage, out] — the grouped expert GEMM,
        // one launch for every (block, expert) route.
        //
        // `stage` is the DESTINATION the pointer build named and `out`
        // aliases it (`in_place = &[(0, 2)]`), so the arm stages only if
        // the assignment did not give them one buffer.
        //
        // THE PREDICATE IS THE POINT. The launcher opens with
        // `if (max_blocks <= 0 || !supported(M, N, K)) return;` — it
        // DECLINES by doing nothing. Qwen3.5-35B-A3B's gate_up is exactly
        // such a shape (`K = hidden = 2048`, and the kernel bounds
        // `K <= 512` because past that cuBLAS wins; the measurements are
        // in `moe_grouped_gemm.cu`'s header). So an arm that just called
        // it would write nothing for gate_up and the mixture would answer
        // fluently from an untouched buffer.
        //
        // The C++ path reads the same predicate and falls back to a
        // batched cuBLAS over the pointer arrays `build_moe_ptrs_aligned`
        // fills. That symbol has no arm yet, so this one REFUSES instead
        // of guessing — and the two are therefore coupled: the pointer
        // build is the keystone of the aligned path, not its leftover.
        "moe::moe_grouped_gemm_bf16" => {
            need(4)?;
            let (a_in, expert_ids, stage, out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let w = weight(resolver)?;
            #[allow(clippy::cast_possible_wrap)]
            let block = spec.params.first().copied().unwrap_or(0) as i32;
            #[allow(clippy::cast_possible_wrap)]
            let max_blocks = spec.params.get(1).copied().unwrap_or(0) as i32;
            let n = i32::try_from(out.width).expect("dim");
            let k = i32::try_from(a_in.width).expect("dim");
            // Mirrors `moe_grouped_gemm_bf16_supported`. Duplicated on
            // purpose and marked as such: the alternative is a launcher
            // that answers "did nothing" the same way it answers "done".
            const FRAG: i32 = 16;
            const SHORT_K: i32 = 512;
            const N_TILE: i32 = 64;
            if max_blocks <= 0
                || block != FRAG
                || k > SHORT_K
                || n % N_TILE != 0
                || k % FRAG != 0
            {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: "moe::moe_grouped_gemm_bf16".into(),
                    why: format!(
                        "the grouped kernel serves M == {FRAG}, K <= {SHORT_K}, \
                         N % {N_TILE} == 0 and K % {FRAG} == 0; this launch is \
                         M = {block}, N = {n}, K = {k} over {max_blocks} blocks. \
                         The batched-cuBLAS fallback needs \
                         moe::build_moe_ptrs_aligned_bf16, which has no arm yet"
                    ),
                });
            }
            if stage.ptr != out.ptr {
                stage_d2d(ctx, &bound.rows, out, stage);
            }
            unsafe {
                ffi::pie_k_moe_moe_grouped_gemm_bf16(
                    a_in.ptr,
                    w,
                    out.ptr,
                    expert_ids.ptr.cast::<i32>(),
                    max_blocks,
                    block,
                    n,
                    k,
                    ctx.stream,
                );
            }
        }
        // ── The MIXTURE's landing pair, both in-place ───────────────
        // Neither can generate: `emit_rust_dispatch` skips every
        // `in_place` row because a generated branch binds `Out(0)` and
        // calls, with nowhere to stage the copy the aliasing needs. The
        // rows already state their sources; staging is the whole
        // difference, and it is `stage_d2d` in both.

        // args: [src, weights, residual, out] — the routed combine that
        // ACCUMULATES. `out += sum_k(src[t, k] * w[t, k])`, so the
        // residual is staged into `out` first and the kernel adds onto
        // it. The plain `token_batched_weighted_sum_bf16` writes instead,
        // which is why only this spelling is in-place.
        "moe::token_batched_weighted_sum_add_bf16" => {
            need(4)?;
            let (src, weights, resid, out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, out, resid);
            // `src` is `[Tokens, top_k, hidden]` and `out` is
            // `[Tokens, hidden]`, so the route count is the ratio of
            // their row widths. Derived from what the two args SAY
            // rather than from a config the arm would have to be told.
            let hidden = i32::try_from(out.width).expect("dim");
            let top_k = i32::try_from(src.width / out.width.max(1)).unwrap_or(1).max(1);
            unsafe {
                ffi::pie_k_moe_token_batched_weighted_sum_add_bf16(
                    out.ptr,
                    src.ptr,
                    weights.ptr.cast::<f32>(),
                    rows,
                    top_k,
                    hidden,
                    ctx.stream,
                );
            }
        }
        // args: [x, y, out] — the SHARED expert's landing, and the
        // operand order is the trap the row's own comment names: `y` is
        // the ADDEND, not the accumulator. `out = out + sigmoid(x·gate) *
        // y`, so the routed block's output stages into `out` and the
        // shared expert's contribution is gated onto it.
        "mlp::sigmoid_dot_scalar_gate_add_bf16" => {
            need(3)?;
            let (x, y, out) = (bound.args[0], bound.args[1], bound.args[2]);
            let w = weight(resolver)?;
            stage_d2d(ctx, &bound.rows, out, y);
            unsafe {
                ffi::pie_k_mlp_sigmoid_dot_scalar_gate_add_bf16(
                    x.ptr,
                    w,
                    out.ptr,
                    y.ptr,
                    rows,
                    i32::try_from(out.width).expect("dim"),
                    ctx.stream,
                );
            }
        }
        // args: [a, b, out] — out = a + b. The kernel is the in-place
        // `y += x` over flat elements, so: stage a→out, add b.
        "norm::residual_add_bf16" => {
            need(3)?;
            let (a_in, b_in, out_arg) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, out_arg, a_in);
            let n = (bound.rows.end - bound.rows.start) as usize * out_arg.width as usize;
            unsafe {
                ffi::pie_k_norm_residual_add_bf16(out_arg.ptr, b_in.ptr, n, ctx.stream);
            }
        }
        // args: [x, out] — out = x + bias, the bias being the op's weight.
        // The kernel is in-place, so: stage x→out, add.
        "norm::add_bias_bf16" => {
            need(2)?;
            let (x_in, out_arg) = (bound.args[0], bound.args[1]);
            let w = weight(resolver)?;
            stage_d2d(ctx, &bound.rows, out_arg, x_in);
            unsafe {
                ffi::pie_k_norm_add_bias_bf16(
                    out_arg.ptr,
                    w,
                    rows,
                    i32::try_from(out_arg.width).expect("dim"),
                    ctx.stream,
                );
            }
        }
        // args: [packed, padded] / [padded, packed]. What
        // `head_dim_padded` COSTS, for a deployment whose logical head
        // width is not one this build instantiated — phi3's 96 rounding
        // up to 128.
        //
        // A hand arm rather than a stated row, and it is the third of
        // §4's classes: the head COUNT is derived arithmetic. The op sits
        // on either the q side or the kv side, so a fixed
        // `Ctx("num_q_heads")` would be right half the time; and the
        // padded width is the other operand's extent, which no `Source`
        // names. Both fall out of the two widths and the logical head dim
        // the ctx carries, which is what this computes.
        "attn::pad_head_dim_bf16" | "attn::strip_head_dim_bf16" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            let pad = bound.kernel == "attn::pad_head_dim_bf16";
            // The PACKED side is whichever end is `head_dim` wide.
            let packed = if pad { src } else { dst };
            let padded = if pad { dst } else { src };
            let hd = ctx.head_dim;
            if hd <= 0 {
                return Err(DispatchRefusal::NoArm(format!(
                    "{}: the ctx states no head_dim",
                    bound.kernel
                )));
            }
            let heads = i32::try_from(packed.width).unwrap_or(0) / hd;
            if heads <= 0 {
                return Err(DispatchRefusal::NoArm(format!(
                    "{}: a packed row of {} is not a multiple of head_dim {hd}",
                    bound.kernel, packed.width
                )));
            }
            let hd_padded = i32::try_from(padded.width).unwrap_or(0) / heads;
            unsafe {
                if pad {
                    ffi::pie_k_attn_pad_head_dim_bf16(
                        src.ptr.cast_const(),
                        dst.ptr,
                        rows,
                        heads,
                        hd,
                        hd_padded,
                        ctx.stream,
                    );
                } else {
                    ffi::pie_k_attn_strip_head_dim_bf16(
                        src.ptr.cast_const(),
                        dst.ptr,
                        rows,
                        heads,
                        hd,
                        hd_padded,
                        ctx.stream,
                    );
                }
            }
        }
        // ── The qwen3_5 hybrid's arms ────────────────────────────────
        // args: [x, y]. Whole-row for the block/final norms; the
        // per-head q/k norms are the same symbol over `tokens * heads`
        // rows of `head_dim` — the op join says which reading applies.
        "norm::rmsnorm_gemma_bf16" => {
            // `[x, y]` when the SEMANTIC `Rmsnorm` lowered here and the
            // weight reached the arm through the resolver; `[x, y, w]`
            // now that `dsl::cuda::rmsnorm` STATES the kernel and names
            // its weight, which the binder resolves like any operand.
            // Both forms are live while both spellings are.
            let (x, y) = (bound.args[0], bound.args[1]);
            let w = match bound.args.len() {
                2 => weight(resolver)?,
                3 => bound.args[2].ptr.cast_const(),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 3,
                        got,
                    });
                }
            };
            let (num_rows, hidden) = match spec.per_head_dim {
                Some(d) => (
                    rows * (i32::try_from(x.width).expect("width") / i32::try_from(d).expect("d")),
                    i32::try_from(d).expect("head_dim fits i32"),
                ),
                None => (rows, i32::try_from(x.width).expect("hidden fits i32")),
            };
            unsafe {
                ffi::pie_k_norm_rmsnorm_gemma_bf16(
                    x.ptr, w, y.ptr, num_rows, hidden, ctx.eps, ctx.stream,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out] — in-place pair, staged like
        // `rope::rope_bf16` — or [q_in, q_out], the KV-SHARED layers'
        // Q-ONLY form: gemma-4's shared full layers rotate q through the
        // same launcher with `num_kv_heads = 0` and the q buffer riding
        // the k slot (`declared_forward.cpp`'s `RopeQOnlyPartial` — NOT
        // a fallback to a generic rope). The rotary width is the op's
        // statement; the head dim is the LAYER'S (gemma-4's full layers
        // run 512 where the fire-wide `ctx.head_dim` says 256), read off
        // the kv view the layer tag names.
        "rope::rope_partial_bf16" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            // Three places a rotary width can come from, in the order
            // that prefers what a STATEMENT said:
            //
            //   `spec.params[0]`  `dsl::cuda::rope_partial` — the stated
            //                     form, which carries it as a wire param;
            //   `spec.rope_partial`  the semantic `Rope { partial }`;
            //   `ctx.rotary_by_layer`  the fire's table, for a family
            //                     whose width is per-layer and whose
            //                     statement carries none.
            //
            // The first two are the same fact under two spellings, and
            // both spellings are live — qwen3_5's prefill states the
            // launch, its decode records the semantic op.
            let rotary = spec
                .params
                .first()
                .copied()
                .filter(|r| *r > 0)
                .or(spec.rope_partial)
                .or_else(|| {
                    ctx.rotary_by_layer
                        .get(bound.layers.start as usize)
                        .copied()
                        .filter(|r| *r > 0)
                })
                .ok_or_else(|| {
                    DispatchRefusal::NoArm(format!(
                        "{}: op states no rotary width and the fire carries no per-layer table",
                        bound.kernel
                    ))
                })?;
            let (q_out, k_ptr, kv_heads) = match bound.args.len() {
                4 => {
                    let (q_in, k_in, q_out, k_out) =
                        (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
                    stage_d2d(ctx, &bound.rows, q_out, q_in);
                    stage_d2d(ctx, &bound.rows, k_out, k_in);
                    (
                        q_out,
                        k_out.ptr,
                        i32::try_from(k_out.width).expect("k width") / layer.head_dim.max(1),
                    )
                }
                2 => {
                    let (q_in, q_out) = (bound.args[0], bound.args[1]);
                    stage_d2d(ctx, &bound.rows, q_out, q_in);
                    (q_out, q_out.ptr, 0)
                }
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 4,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_rope_rope_partial_bf16(
                    q_out.ptr,
                    k_ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / layer.head_dim.max(1),
                    kv_heads,
                    layer.head_dim,
                    i32::try_from(rotary).expect("rotary fits i32"),
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.stream,
                );
            }
        }
        // args: [packed, q_out, gate_out] — the 2×-wide gated q pack's
        // per-head de-interleave.
        "layout::split_q_gate_bf16" => {
            need(3)?;
            let (packed, q_out, gate_out) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_layout_split_q_gate_bf16(
                    packed.ptr,
                    q_out.ptr,
                    gate_out.ptr,
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.stream,
                );
            }
        }
        // args: [x, gate] in place, or [x, gate, out] when the lowering
        // assigned distinct buffers — staged, the in-place contract.
        "mlp::sigmoid_gate_inplace_bf16" => {
            let (x, gate) = match bound.args.len() {
                2 => (bound.args[0], bound.args[1]),
                3 => {
                    let (x_in, gate, out) = (bound.args[0], bound.args[1], bound.args[2]);
                    stage_d2d(ctx, &bound.rows, out, x_in);
                    (out, gate)
                }
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 2,
                        got,
                    });
                }
            };
            let n = rows * i32::try_from(x.width).expect("width fits i32");
            unsafe {
                ffi::pie_k_mlp_sigmoid_gate_inplace_bf16(x.ptr, gate.ptr, n, ctx.stream);
            }
        }
        // args: [x, y, conv_weight]. The bias rides the conv binding
        // (`<name>_bias`, null when the checkpoint has none); the state
        // slab, slot indirection and window geometry are the fire's.
        "ssm::causal_conv1d_update_batched_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let layer = state_layer()?;
            let (x, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            let bias = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(&format!("{n}_bias")))
                .unwrap_or(std::ptr::null());
            let state = slab(&g.conv_state, layer, "conv")?;
            unsafe {
                ffi::pie_k_ssm_causal_conv1d_update_batched_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    bias,
                    state,
                    g.slot_ids_d,
                    g.conv_stride_elems,
                    y.ptr,
                    rows,
                    g.conv_dim,
                    g.conv_k,
                    ctx.stream,
                );
            }
        }
        // args: [x, y, conv_weight] — the prefill walk over the fire's
        // qo CSR; requests come from the attention context.
        "ssm::causal_conv1d_prefill_batched_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let (x, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            let bias = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(&format!("{n}_bias")))
                .unwrap_or(std::ptr::null());
            let state = slab(&g.conv_state, layer, "conv")?;
            unsafe {
                ffi::pie_k_ssm_causal_conv1d_prefill_batched_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    bias,
                    y.ptr,
                    state,
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.conv_stride_elems,
                    a.num_requests,
                    g.conv_dim,
                    g.conv_k,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [qkv_post, a, b, q, k, v, g, beta] — three inputs, the
        // op's five fp32 results; `a_log` (fp32-widened) and `dt_bias`
        // are the op's two named parameters.
        "ssm::qwen_gdn_post_conv_prep_bf16" => {
            need(8)?;
            let g = gdn_ctx()?;
            let a_log = weight(resolver)?;
            let dt_name = spec
                .weight2
                .as_deref()
                .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
            let dt_bias = resolver
                .weight(dt_name)
                .ok_or_else(|| DispatchRefusal::UnknownWeight(dt_name.to_string()))?;
            unsafe {
                ffi::pie_k_ssm_qwen_gdn_post_conv_prep_bf16(
                    bound.args[0].ptr.cast_const(),
                    bound.args[1].ptr.cast_const(),
                    bound.args[2].ptr.cast_const(),
                    a_log,
                    dt_bias,
                    bound.args[3].ptr.cast(),
                    bound.args[4].ptr.cast(),
                    bound.args[5].ptr.cast(),
                    bound.args[6].ptr.cast(),
                    bound.args[7].ptr.cast(),
                    rows,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    g.conv_dim,
                    ctx.stream,
                );
            }
        }
        // args: [q, k, v, g, beta, out] — the decode recurrence against
        // the layer's bf16 state slab (the deployment's `state_bf16`
        // fact picked this symbol at trace time).
        "ssm::recurrent_gated_delta_step_batched_state_bf16" => {
            need(6)?;
            let g = gdn_ctx()?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            unsafe {
                ffi::pie_k_ssm_recurrent_gated_delta_step_batched_state_bf16(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state,
                    g.slot_ids_d,
                    g.state_stride_elems,
                    bound.args[5].ptr.cast(),
                    rows,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                );
            }
        }
        // args: [q, k, v, g, beta] — the fp32-STATE FLA prefill
        // recurrence (the `state_bf16: false` deployments' text).
        "ssm::chunk_gated_delta_prefill_batched" => {
            need(5)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            let core_out = out_slot(0, resolver)?;
            unsafe {
                ffi::pie_k_ssm_chunk_gated_delta_prefill_batched(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state.cast(),
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.state_stride_elems,
                    core_out.ptr.cast(),
                    a.num_requests,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [q, k, v, g, beta, out] — the chunked FLA prefill
        // recurrence over the fire's qo CSR.
        "ssm::chunk_gated_delta_prefill_batched_state_bf16" => {
            need(5)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            // The core output is the GUARD's value, not an SSA arg of
            // this region launch — the join walked back to it.
            let core_out = out_slot(0, resolver)?;
            unsafe {
                ffi::pie_k_ssm_chunk_gated_delta_prefill_batched_state_bf16(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state,
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.state_stride_elems,
                    core_out.ptr.cast(),
                    a.num_requests,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [x, gate, y] — the GDN landing norm: per (row, value
        // head) over the trailing head width, weight fp32-widened.
        "norm::rmsnorm_gated_fp32_in_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let (x, gate, y) = (bound.args[0], bound.args[1], bound.args[2]);
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_norm_rmsnorm_gated_fp32_in_bf16(
                    x.ptr.cast_const(),
                    gate.ptr.cast_const(),
                    w,
                    y.ptr,
                    rows * g.v_h,
                    g.v_d,
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // ── gemma's arms ─────────────────────────────────────────────
        // args: [x_in, x_out, scale-name] — `x *= s`, the constant named
        // in the weight slot, resolved through `DispatchCtx::scales`.
        "norm::scalar_mul_bf16" => {
            need(3)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            // A NUMBER when the statement carries one, a NAME otherwise.
            // `dsl::cuda::scalar_mul` takes `Option<f32>`: given a value it
            // rides the params as bits, and given none the row names a
            // `scale.*` the driver looks up in a table it built from a
            // config. The first form is the one a reader can check against
            // the text; the second is what it replaces.
            let s = match param_f32(spec, 0) {
                Some(by) => by,
                None => {
                    let name = spec
                        .weight
                        .as_deref()
                        .and_then(|n| n.strip_prefix("scale."))
                        .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
                    *ctx.scales
                        .get(name)
                        .ok_or_else(|| DispatchRefusal::UnknownWeight(format!("scale.{name}")))?
                }
            };
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            let n = (bound.rows.end - bound.rows.start) as usize * x_out.width as usize;
            unsafe {
                ffi::pie_k_norm_scalar_mul_bf16(x_out.ptr, s, n, ctx.stream);
            }
        }
        // args: [gate, up, y] — gemma's GeGLU (tanh approximation); the
        // lowering lands y on the gate's bytes (the kernel's in-place
        // contract), staged when it assigned distinct buffers.
        //
        // TWO sites share this symbol and the arm no longer tells them
        // apart, which is the point. The PLE gate's second operand used
        // to be the WHOLE `[L, Tokens, ple_dim]` relay, so this arm added
        // `layer * N * ple_dim` to reach the layer's slice — and to know
        // WHEN to add it, forked on the out width against `ctx.ple_dim`.
        // The declaration states a `select` at `l` now; the layer axis
        // leads, so the slice IS a select, and `Buffers::assign` places it
        // at `offset(relay) + l · N · ple_dim` without being told. A
        // select allocates nothing.
        //
        // What went with the arithmetic is worth naming: the width fork
        // was a driver deciding which SITE it was serving from a number,
        // and the two sites now differ only in which values they name.
        "mlp::geglu_tanh_bf16" => {
            need(3)?;
            let (gate, up, y) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, y, gate);
            let n = rows * i32::try_from(y.width).expect("width fits i32");
            unsafe {
                ffi::pie_k_mlp_geglu_tanh_bf16(
                    y.ptr.cast_const(),
                    up.ptr.cast_const(),
                    y.ptr,
                    n,
                    ctx.stream,
                );
            }
        }
        // args: [x_in, x_out] — `cap * tanh(x / cap)` over the logits;
        // the cap is the deployment's final-softcap fact.
        "attn::logit_softcap_bf16" => {
            need(2)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            let n = (bound.rows.end - bound.rows.start) as usize * x_out.width as usize;
            unsafe {
                ffi::pie_k_attn_logit_softcap_bf16(
                    x_out.ptr,
                    ctx.final_logit_softcap,
                    n,
                    ctx.stream,
                );
            }
        }
        // args: [packed, q_out, q_norm, k_norm] — gemma-4's fused local
        // decode post: split the packed projection, norm q/k, rope them
        // (rounded), norm v, write k/v straight to the pages. Only the
        // query survives as a value.
        "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" => {
            need(4)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (packed, q_out, qw, kw) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    packed.ptr.cast_const(),
                    q_out.ptr,
                    layer.k_pages,
                    layer.v_pages,
                    qw.ptr.cast_const(),
                    kw.ptr.cast_const(),
                    ctx.positions.cast_const().cast(),
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.row_valid_d,
                    rows,
                    i32::try_from(q_out.width).expect("q width") / layer.head_dim.max(1),
                    layer.num_kv_heads,
                    layer.head_dim,
                    layer.page_size,
                    layer.hnd_layout,
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // The rounded fused norm+rope, BOTH shapes the driver reaches:
        // [q, k, q_norm, k_norm] — the local pair, in place — and
        // [q_in, q_out, q_norm] — a KV-SHARED layer's Q-ONLY form, which
        // the driver reaches by passing `num_kv_heads = 0`, never by a
        // generic rope.
        "rope::qk_rmsnorm_rope_bf16_rounded" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, k, qw, kw, kv_heads) = match bound.args.len() {
                4 => (
                    bound.args[0],
                    bound.args[1].ptr,
                    bound.args[2],
                    bound.args[3].ptr.cast_const(),
                    layer.num_kv_heads,
                ),
                // The PREFILL pair: the lowering states distinct in/out
                // buffers for the in-place kernel, staged like
                // `rope::rope_bf16`'s.
                6 => {
                    let (q_in, k_in, q_out, k_out) =
                        (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
                    stage_d2d(ctx, &bound.rows, q_out, q_in);
                    stage_d2d(ctx, &bound.rows, k_out, k_in);
                    (
                        q_out,
                        k_out.ptr,
                        bound.args[4],
                        bound.args[5].ptr.cast_const(),
                        layer.num_kv_heads,
                    )
                }
                3 => {
                    let (q_in, q_out, qw) = (bound.args[0], bound.args[1], bound.args[2]);
                    stage_d2d(ctx, &bound.rows, q_out, q_in);
                    (q_out, std::ptr::null_mut(), qw, std::ptr::null(), 0)
                }
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 4,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_rope_qk_rmsnorm_rope_bf16_rounded(
                    q.ptr,
                    k,
                    qw.ptr.cast_const(),
                    kw,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    kv_heads,
                    layer.head_dim,
                    ctx.theta_of(bound.layers.start as usize),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [q, o] — the PLANLESS flashinfer prefill (plans
        // internally per fire; reads the host CSR mirrors).
        "attn::attention_flashinfer_prefill" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, o) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_attention_flashinfer_prefill(
                    q.ptr.cast_const(),
                    *layer,
                    o.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.qo_indptr_h,
                    a.kv_page_indptr_h,
                    rows,
                    a.num_requests,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    a.workspace,
                    ctx.stream,
                    window_of(spec, a, u32::from(bound.layers.start)),
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        // args: [q, o] — the naive paged prefill, for the head dims
        // flashinfer's prefill template refuses (gemma-4's 512).
        "attn::attention_naive_paged" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, o) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_attention_naive_paged(
                    q.ptr.cast_const(),
                    *layer,
                    o.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    rows,
                    a.num_requests,
                    a.num_pages_in_batch,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    ctx.stream,
                    window_of(spec, a, u32::from(bound.layers.start)),
                    a.sm_scale,
                );
            }
        }
        // ── gemma-4's arms ───────────────────────────────────────────
        // args: [src, dst] — the PLE relay: `[N, layers*dim]` transposed
        // to `[layers, N, dim]` so each layer reads a contiguous slice.
        "layout::transpose_bf16_nld_to_lnd" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            if ctx.ple_dim <= 0 {
                return Err(DispatchRefusal::NoArm(format!(
                    "{}: the fire states no ple_dim",
                    bound.kernel
                )));
            }
            unsafe {
                ffi::pie_k_layout_transpose_bf16_nld_to_lnd(
                    src.ptr.cast_const().cast(),
                    dst.ptr.cast(),
                    rows,
                    i32::try_from(src.width).expect("width") / ctx.ple_dim,
                    ctx.ple_dim,
                    ctx.stream,
                );
            }
        }
        // args: [x, hidden_in, hidden_out, norm_out, w, next_w] — FOUR
        // statements in one launch: norm x, land on the stream, scale,
        // norm THAT with the next block's weight. The scale is 1 at the
        // attention landing; the PLE landing carries the layer's own
        // scalar, resolved from `DispatchCtx::scales` by the weight's
        // name (the C++ reads `layer_scalar_value` the same way).
        "norm::rmsnorm_residual_add_scale_rmsnorm_bf16" => {
            need(6)?;
            let (x, hid_in, hid_out, norm_out, w, next_w) = (
                bound.args[0],
                bound.args[1],
                bound.args[2],
                bound.args[3],
                bound.args[4],
                bound.args[5],
            );
            let scale = spec
                .weight
                .as_deref()
                .filter(|n| n.ends_with("ple_norm"))
                .map_or(1.0, |n| ctx.scales.get(n).copied().unwrap_or(1.0));
            stage_d2d(ctx, &bound.rows, hid_out, hid_in);
            unsafe {
                ffi::pie_k_norm_rmsnorm_residual_add_scale_rmsnorm_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    hid_out.ptr,
                    scale,
                    next_w.ptr.cast_const(),
                    norm_out.ptr,
                    rows,
                    i32::try_from(x.width).expect("hidden fits i32"),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [x, hidden_in, hidden_out, w] — the two-statement form:
        // norm x, land on the stream (gemma-4's post-feedforward norm).
        "norm::rmsnorm_residual_add_bf16" => {
            need(4)?;
            let (x, hid_in, hid_out, w) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, hid_out, hid_in);
            unsafe {
                ffi::pie_k_norm_rmsnorm_residual_add_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    hid_out.ptr,
                    rows,
                    i32::try_from(x.width).expect("hidden fits i32"),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [packed, y] — GeGLU over the packed gate‖up bank.
        "mlp::chunked_geglu_tanh_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_geglu_tanh_bf16(
                    packed.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("I fits i32"),
                    ctx.stream,
                    ctx.gate_second,
                );
            }
        }
        // args: [x, y] — the weightless per-head V-norm (`v / rms(v)`).
        "norm::rmsnorm_no_scale_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            let (num_rows, hidden) = match spec.per_head_dim {
                Some(d) => (
                    rows * (i32::try_from(x.width).expect("width") / i32::try_from(d).expect("d")),
                    i32::try_from(d).expect("head_dim fits i32"),
                ),
                None => (rows, i32::try_from(x.width).expect("hidden fits i32")),
            };
            unsafe {
                ffi::pie_k_norm_rmsnorm_no_scale_bf16(
                    x.ptr.cast_const(),
                    y.ptr,
                    num_rows,
                    hidden,
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // ── nemotron_h's arms ────────────────────────────────────────
        // args: [packed, gate, conv_in, dt] — the in-projection's three
        // riders, split out. Every dim is an arg's width.
        "ssm::nemotron_mamba_split_bf16" => {
            need(4)?;
            let (packed, gate, conv_in, dt) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_ssm_nemotron_mamba_split_bf16(
                    packed.ptr.cast_const(),
                    gate.ptr,
                    conv_in.ptr,
                    dt.ptr,
                    rows,
                    i32::try_from(packed.width).expect("width"),
                    i32::try_from(gate.width).expect("width"),
                    i32::try_from(conv_in.width).expect("width"),
                    i32::try_from(dt.width).expect("width"),
                    ctx.stream,
                );
            }
        }
        // args: [a_out, d_out, bias_out, W a_log, W d, W dt_bias] — the
        // load-time fp32 tables (`Lw.mamba_A/D_f32/dt_bias_f32`), stated
        // per fire because the declared trace has no load hook. Heads
        // come from the fire's mamba geometry.
        "ssm::nemotron_prepare_mamba_params" => {
            need(6)?;
            let g = gdn_ctx()?;
            let (a, d, bias) = (bound.args[0], bound.args[1], bound.args[2]);
            let (wa, wd, wb) = (bound.args[3], bound.args[4], bound.args[5]);
            unsafe {
                ffi::pie_k_ssm_nemotron_prepare_mamba_params(
                    wa.ptr.cast_const(),
                    wd.ptr.cast_const(),
                    wb.ptr.cast_const(),
                    a.ptr.cast(),
                    d.ptr.cast(),
                    bias.ptr.cast(),
                    g.v_h,
                    ctx.stream,
                );
            }
        }
        // args: [dt_raw, a, dt_out, da_out]; dt_bias rides the spec's
        // aux slots (the statement does not carry it — the C++ hand pass
        // wires it through its workspace). `time_step_min` is 0 at both
        // C++ call sites.
        "ssm::nemotron_prepare_mamba_dt_da" => {
            need(4)?;
            let (dt_raw, a, dt_out, da_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let bias = aux_slot(3, resolver)?;
            unsafe {
                ffi::pie_k_ssm_nemotron_prepare_mamba_dt_da(
                    dt_raw.ptr.cast_const(),
                    a.ptr.cast_const().cast(),
                    bias.ptr.cast_const().cast(),
                    dt_out.ptr.cast(),
                    da_out.ptr.cast(),
                    rows,
                    i32::try_from(dt_raw.width).expect("heads"),
                    0.0,
                    ctx.stream,
                );
            }
        }
        // args: [conv_out, dt_pre, y] + the aux slots [dt_raw, a, d,
        // dt_bias, dt_pre, da_pre] — the selective scan, over the
        // layer's slab and the fire's slots. On L40S (sm89) the C++
        // ALWAYS lands here: its FlashInfer SSU try refuses below sm90.
        "ssm::nemotron_mamba_ssm_batched_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let a_ctx =
                attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let (conv_out, dt_pre, y) = (bound.args[0], bound.args[1], bound.args[2]);
            let dt_raw = aux_slot(0, resolver)?;
            let a_par = aux_slot(1, resolver)?;
            let d_par = aux_slot(2, resolver)?;
            let bias = aux_slot(3, resolver)?;
            let da_pre = aux_slot(5, resolver)?;
            let state = slab(&g.recurrent_state, layer, "mamba")?;
            unsafe {
                ffi::pie_k_ssm_nemotron_mamba_ssm_batched_bf16(
                    conv_out.ptr.cast_const(),
                    dt_raw.ptr.cast_const(),
                    a_par.ptr.cast_const().cast(),
                    d_par.ptr.cast_const().cast(),
                    bias.ptr.cast_const().cast(),
                    dt_pre.ptr.cast_const().cast(),
                    da_pre.ptr.cast_const().cast(),
                    state,
                    g.slot_ids_d,
                    a_ctx.qo_indptr_d,
                    y.ptr,
                    a_ctx.num_requests,
                    g.v_h,
                    g.v_d,
                    g.k_d,
                    g.n_groups,
                    g.conv_dim,
                    g.v_h * g.v_d,
                    0.0,
                    rows != a_ctx.num_requests,
                    ctx.stream,
                );
            }
        }
        // args: [x, gate, y, W] — the grouped, gated output norm. The
        // gate is the SPLIT's contiguous copy, so its stride is its own
        // width (the C++ hand pass reads the gate in place inside the
        // packed projection, stride `projection_dim` — same values).
        "ssm::zamba_rmsnorm_gated_bf16" => {
            need(4)?;
            let g = gdn_ctx()?;
            let (x, gate, y, w) = (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let hidden = i32::try_from(x.width).expect("width");
            unsafe {
                ffi::pie_k_ssm_zamba_rmsnorm_gated_bf16(
                    x.ptr.cast_const(),
                    gate.ptr.cast_const(),
                    w.ptr.cast_const(),
                    y.ptr,
                    rows,
                    hidden,
                    i32::try_from(gate.width).expect("width"),
                    hidden / g.n_groups.max(1),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [x, y] — elementwise ReLU²; the total is rows × width
        // whichever way the shape folded its routes.
        "mlp::relu2_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_relu2_bf16(
                    x.ptr.cast_const(),
                    y.ptr,
                    rows * i32::try_from(x.width).expect("width"),
                    ctx.stream,
                );
            }
        }
        // args: [logits(f32), topk_idx, topk_w, W bias(f32)] — the
        // sigmoid router. Normalize and the scaling factor are the
        // deployment's (`cfg.norm_topk_prob`, `routed_scaling_factor`).
        "moe::topk_sigmoid_bias_fp32" => {
            need(4)?;
            let (logits, idx, wts, bias) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_moe_topk_sigmoid_bias_fp32(
                    logits.ptr.cast_const().cast(),
                    bias.ptr.cast_const().cast(),
                    idx.ptr.cast(),
                    wts.ptr.cast(),
                    rows,
                    i32::try_from(logits.width).expect("experts"),
                    i32::try_from(idx.width).expect("top_k"),
                    ctx.moe_norm_topk,
                    ctx.moe_routed_scaling,
                    ctx.stream,
                );
            }
        }
        // args: [topk_idx, norm_x, out, W stacked-expert base] — the
        // decode GEMV over the routed experts; one warp per output row.
        "moe::moe_gate_up_decode_gemv_bf16" => {
            need(4)?;
            let (idx, x, y, base) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_moe_moe_gate_up_decode_gemv_bf16(
                    idx.ptr.cast_const().cast(),
                    x.ptr.cast_const(),
                    base.ptr.cast_const(),
                    y.ptr,
                    rows,
                    top_k,
                    i32::try_from(x.width).expect("hidden"),
                    i32::try_from(y.width).expect("width") / top_k.max(1),
                    ctx.stream,
                );
            }
        }
        // args: [topk_idx, act, out, W stacked-expert base].
        "moe::moe_down_decode_gemv_bf16" => {
            need(4)?;
            let (idx, act, y, base) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_moe_moe_down_decode_gemv_bf16(
                    idx.ptr.cast_const().cast(),
                    act.ptr.cast_const(),
                    base.ptr.cast_const(),
                    y.ptr,
                    rows,
                    top_k,
                    i32::try_from(y.width).expect("width") / top_k.max(1),
                    i32::try_from(act.width).expect("i_moe"),
                    ctx.stream,
                );
            }
        }
        // args: [src, weights, out] — the K-expert combine.
        "moe::token_batched_weighted_sum_bf16" => {
            need(3)?;
            let (src, wts, out) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_moe_token_batched_weighted_sum_bf16(
                    out.ptr,
                    src.ptr.cast_const(),
                    wts.ptr.cast_const().cast(),
                    rows,
                    i32::try_from(wts.width).expect("top_k"),
                    i32::try_from(out.width).expect("hidden"),
                    ctx.stream,
                );
            }
        }
        // args: [act, y(f32), W] — the fp32-out GEMM the sigmoid router
        // reads (`act_x_wt_bf16_out_fp32`); the statement carries its
        // weight as an arg.
        "gemm::act_x_wt_bf16_out_fp32" => {
            need(3)?;
            let (act, y) = (bound.args[0], bound.args[1]);
            let w = bound.args[2].ptr.cast_const();
            unsafe {
                ffi::pie_k_gemm_act_x_wt_bf16_out_fp32(
                    ctx.cublas,
                    act.ptr.cast_const(),
                    w,
                    y.ptr.cast(),
                    rows,
                    i32::try_from(y.width).expect("n"),
                    i32::try_from(act.width).expect("k"),
                );
            }
        }
        // args: [q, v] in place; qkv_in rides the spec's aux (the same
        // layer's projection input), the staged state + scratch ride the
        // ctx. The LAYER is the op tag's — never `param1`, the bug the
        // C++'s first live A/B caught.
        "pie_lora_qkv_correction" => {
            need(2)?;
            // NO ADAPTERS STAGED IS AN ANSWER, not a refusal, and this
            // line is the one that lets a union capture record the arm at
            // all.
            //
            // Under `GuardMode::Resolve` the case is unreachable — the
            // `HasLora` guard removes the arm when no row carries an
            // adapter. Under `Union` every arm lowers and the conditional
            // decides at replay, so the arm has to be ISSUABLE with its
            // predicate false. Doing nothing is what the correction means
            // for a fire with nothing to correct.
            //
            // What makes that safe rather than merely quiet is the bucket
            // key: `BucketKey::lora_shape` is zero for a fire with no
            // adapters, so an exec recorded here serves only fires that
            // also have none. A fire that stages adapters has a different
            // shape and lands in a different bucket, where the arm records
            // its grouped launches. See `model::supergraph`.
            let Some((state, scratch)) = ctx.lora else {
                return Ok(());
            };
            let x = aux_slot(0, resolver)?;
            let (q, v) = (bound.args[0], bound.args[1]);
            unsafe {
                (*state).apply(
                    ctx.cublas,
                    i32::from(bound.layers.start),
                    x.ptr.cast_const(),
                    i32::try_from(x.width).expect("hidden"),
                    i32::try_from(q.width).expect("q width"),
                    i32::try_from(v.width).expect("v width"),
                    q.ptr,
                    v.ptr,
                    scratch,
                    ctx.stream,
                );
            }
        }
        // args: [x_in, x_out, W ""] — the WEIGHTLESS per-head norm; the
        // third arg is the statement's empty weight slot, and the row
        // takes no weight at all. In place, so staged.
        "norm::per_head_rmsnorm_bf16" => {
            need(3)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            unsafe {
                ffi::pie_k_norm_per_head_rmsnorm_bf16(
                    x_out.ptr,
                    rows,
                    i32::try_from(x_out.width).expect("width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [x, y, W] — a norm over a WINDOW of a wider row: the
        // strides are the two operands' own widths, which is the whole
        // reason this kernel is not the plain one.
        "norm::rmsnorm_strided_bf16" => {
            need(3)?;
            let (x, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_norm_rmsnorm_strided_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("hidden"),
                    i32::try_from(x.width).expect("x stride"),
                    i32::try_from(y.width).expect("y stride"),
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [attn_out, lse, out, W sink] — the sink correction, in
        // place over the dispatch's output (gpt-oss's twin under a
        // different name and a different LSE convention).
        "norm::attn_sink_correction_bf16" => {
            need(4)?;
            let (o_in, lse, o_out, sink) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, o_out, o_in);
            unsafe {
                ffi::pie_k_norm_attn_sink_correction_bf16(
                    o_out.ptr,
                    lse.ptr.cast_const().cast(),
                    sink.ptr.cast_const().cast(),
                    rows,
                    i32::try_from(o_out.width).expect("width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.stream,
                );
            }
        }
        // args: [prefix, blocks, out, W norm, W proj] — the residual
        // blend over a block-structured prefix. `B` is how many blocks
        // the prefix holds, `block_rows` how many rows each one spans.
        "attn::attn_res_blend_bf16" => {
            need(5)?;
            let (prefix, blocks, out, nw, pw) = (
                bound.args[0],
                bound.args[1],
                bound.args[2],
                bound.args[3],
                bound.args[4],
            );
            let hidden = i32::try_from(out.width).expect("hidden");
            unsafe {
                ffi::pie_k_attn_attn_res_blend_bf16(
                    prefix.ptr.cast_const(),
                    blocks.ptr.cast_const(),
                    nw.ptr.cast_const(),
                    pw.ptr.cast_const(),
                    out.ptr,
                    rows,
                    i32::try_from(blocks.width).expect("width") / hidden.max(1),
                    hidden,
                    rows,
                    ctx.eps,
                    ctx.stream,
                );
            }
        }
        // args: [act_fp16, topk_idx, gate_out, up_out] + the FOUR named
        // per-expert tables. They were unnamed until the declaration
        // learned to say what it reads — with no name in the trace an
        // executor that resolves by name cannot reach them at all, and
        // the only way in was a family's private layer struct.
        "quant::wna16_gate_up_decode_bf16" => {
            need(8)?;
            let (act, idx, gate_out, up_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_quant_wna16_gate_up_decode_bf16(
                    act.ptr.cast_const(),
                    idx.ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    bound.args[5].ptr.cast_const().cast(),
                    bound.args[6].ptr.cast_const().cast(),
                    bound.args[7].ptr.cast_const().cast(),
                    gate_out.ptr,
                    up_out.ptr,
                    rows,
                    top_k,
                    i32::try_from(act.width).expect("hidden"),
                    i32::try_from(gate_out.width).expect("intermediate"),
                    ctx.wna16_group_size,
                    ctx.stream,
                );
            }
        }
        // args: [act_fp16, topk_idx, out] + the two named down tables.
        "quant::wna16_down_decode_bf16" => {
            need(5)?;
            let (act, idx, out) = (bound.args[0], bound.args[1], bound.args[2]);
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_quant_wna16_down_decode_bf16(
                    act.ptr.cast_const(),
                    idx.ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    out.ptr,
                    rows,
                    top_k,
                    i32::try_from(out.width).expect("hidden"),
                    i32::try_from(act.width).expect("intermediate"),
                    ctx.wna16_group_size,
                    ctx.stream,
                );
            }
        }
        // ── the pair/chunked activation and router variants ─────────
        // args: [gate, y] + aux[up] — the pair form, whose `up` the
        // statement cannot name (see the join's pre-pass).
        "mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16" | "mlp::situ_bf16" => {
            need(2)?;
            let (gate, y) = (bound.args[0], bound.args[1]);
            let up = aux_slot(0, resolver).map_err(|_| {
                DispatchRefusal::NoArm(format!(
                    "{}: the pair form needs the layer's `up` projection, and the \
                     join found none",
                    bound.kernel
                ))
            })?;
            let n = rows * i32::try_from(y.width).expect("width");
            unsafe {
                match bound.kernel {
                    "mlp::swiglu_bf16" => ffi::pie_k_mlp_swiglu_bf16(
                        gate.ptr.cast_const(),
                        up.ptr.cast_const(),
                        y.ptr,
                        n,
                        ctx.stream,
                    ),
                    "mlp::swiglu_clamp_bf16" => ffi::pie_k_mlp_swiglu_clamp_bf16(
                        gate.ptr.cast_const(),
                        up.ptr.cast_const(),
                        y.ptr,
                        n,
                        ctx.glu_limit,
                        ctx.stream,
                    ),
                    _ => ffi::pie_k_mlp_situ_bf16(
                        gate.ptr.cast_const(),
                        up.ptr.cast_const(),
                        y.ptr,
                        n,
                        ctx.situ_beta,
                        ctx.situ_linear_beta,
                        ctx.stream,
                    ),
                }
            }
        }
        // args: [packed, y] — the CHUNKED forms need no aux: one packed
        // operand in, half-width out.
        "mlp::chunked_swiglu_clamp_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_swiglu_clamp_bf16(
                    packed.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("i"),
                    ctx.glu_limit,
                    ctx.stream,
                );
            }
        }
        "mlp::chunked_situ_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_situ_bf16(
                    packed.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("i"),
                    ctx.situ_beta,
                    ctx.situ_linear_beta,
                    ctx.gate_second,
                    ctx.stream,
                );
            }
        }
        // args: [logits, idx, w] or [logits, idx, w, W bias] — the two
        // bias-capable routers; the bias is null when unstated.
        "moe::topk_sigmoid_bf16" | "moe::topk_sqrtsoftplus_bf16" => {
            let (logits, idx, w) = (bound.args[0], bound.args[1], bound.args[2]);
            let bias = match bound.args.len() {
                3 => std::ptr::null(),
                4 => bound.args[3].ptr.cast_const(),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 3,
                        got,
                    });
                }
            };
            let (n, e, k) = (
                rows,
                i32::try_from(logits.width).expect("experts"),
                i32::try_from(idx.width).expect("top_k"),
            );
            unsafe {
                if bound.kernel == "moe::topk_sigmoid_bf16" {
                    ffi::pie_k_moe_topk_sigmoid_bf16(
                        logits.ptr.cast_const(),
                        idx.ptr.cast(),
                        w.ptr.cast(),
                        bias.cast(),
                        n,
                        e,
                        k,
                        ctx.moe_norm_topk,
                        ctx.moe_routed_scaling,
                        ctx.stream,
                    );
                } else {
                    ffi::pie_k_moe_topk_sqrtsoftplus_bf16(
                        logits.ptr.cast_const(),
                        idx.ptr.cast(),
                        w.ptr.cast(),
                        bias.cast(),
                        n,
                        e,
                        k,
                        ctx.moe_norm_topk,
                        ctx.moe_routed_scaling,
                        ctx.stream,
                    );
                }
            }
        }
        // args: [act, y, W] — the plain `x · Wᵀ`, weight as an arg.
        "gemm::act_x_wt_bf16" => {
            need(3)?;
            let (act, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_gemm_act_x_wt_bf16(
                    ctx.cublas,
                    act.ptr.cast_const(),
                    w.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("n"),
                    i32::try_from(act.width).expect("k"),
                    0.0,
                );
            }
        }
        // ── gpt-oss / mixtral's arms ─────────────────────────────────
        // args: [act, y, W w, W bias] — the projection with its bias
        // folded in. The bias may be null; the row says so.
        "gemm::act_x_wt_bias_bf16" => {
            need(4)?;
            let (act, y, w, bias) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_gemm_act_x_wt_bias_bf16(
                    ctx.cublas,
                    act.ptr.cast_const(),
                    w.ptr.cast_const(),
                    bias.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("n"),
                    i32::try_from(act.width).expect("k"),
                    ctx.stream,
                    0.0,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out] — YaRN over the ORIGINAL
        // context, staged like the other in-place ropes. The four scaling
        // terms and the original context are the deployment's.
        "rope::rope_yarn_original_bf16" => {
            need(4)?;
            let (q_in, k_in, q_out, k_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            let heads = |w: u32| i32::try_from(w).expect("width") / ctx.head_dim.max(1);
            unsafe {
                ffi::pie_k_rope_rope_yarn_original_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    heads(q_out.width),
                    heads(k_out.width),
                    ctx.head_dim,
                    ctx.rope_theta,
                    ctx.yarn[0],
                    ctx.yarn[1],
                    ctx.yarn[2],
                    ctx.yarn[3],
                    ctx.yarn_original_max,
                    ctx.stream,
                    ctx.rope_interleaved,
                );
            }
        }
        // args: [o_in, lse, o_out, W sinks] — the attention sink's
        // rescale, in place over the dispatch's output.
        "attn::attention_sink_rescale_bf16" => {
            need(4)?;
            let (o_in, lse, o_out, sinks) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, o_out, o_in);
            unsafe {
                ffi::pie_k_attn_attention_sink_rescale_bf16(
                    o_out.ptr,
                    lse.ptr.cast_const().cast(),
                    sinks.ptr.cast_const(),
                    rows,
                    i32::try_from(o_out.width).expect("width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.stream,
                );
            }
        }
        // args: [logits, topk_idx, topk_w] — the plain softmax router.
        "moe::topk_softmax_bf16" => {
            need(3)?;
            let (logits, idx, w) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_moe_topk_softmax_bf16(
                    logits.ptr.cast_const(),
                    idx.ptr.cast(),
                    w.ptr.cast(),
                    rows,
                    i32::try_from(logits.width).expect("experts"),
                    i32::try_from(idx.width).expect("top_k"),
                    ctx.stream,
                );
            }
        }
        // args: [in_bf16, out_fp16] — the activation the MXFP4 GEMVs read.
        "quant::bf16_to_fp16" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_quant_bf16_to_fp16(
                    src.ptr.cast_const(),
                    dst.ptr,
                    rows as usize * src.width as usize,
                    ctx.stream,
                );
            }
        }
        // args: [topk_idx, act_fp16, gate_out, up_out, W bank] — the
        // packed-expert GEMV. The C++ reaches FOUR per-expert pointer
        // arrays off its layer struct ("they are per-expert pointer
        // arrays and not tensors, which is the same reason `bind` refuses
        // to name them"); the trace names ONE bank, so the other three
        // come off it by suffix — the `_bias` convention the conv arms
        // already use. The biases are null-ok.
        "quant::mxfp4_moe_gate_up_decode_bf16" => {
            need(5)?;
            let (idx, act, gate_out, up_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let bank = spec.weight.as_deref().ok_or_else(|| {
                DispatchRefusal::NoArm(format!("{}: op names no bank", bound.kernel))
            })?;
            let packed = bound.args[4].ptr.cast_const();
            let scales = resolver
                .weight(&format!("{bank}_scales"))
                .ok_or_else(|| DispatchRefusal::UnknownWeight(format!("{bank}_scales")))?;
            let gate_bias = resolver.weight(&format!("{bank}_gate_bias"));
            let up_bias = resolver.weight(&format!("{bank}_up_bias"));
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_quant_mxfp4_moe_gate_up_decode_bf16(
                    act.ptr.cast_const(),
                    idx.ptr.cast_const().cast(),
                    packed.cast(),
                    scales.cast(),
                    gate_bias.unwrap_or(std::ptr::null()).cast(),
                    up_bias.unwrap_or(std::ptr::null()).cast(),
                    gate_out.ptr,
                    up_out.ptr,
                    rows,
                    top_k,
                    i32::try_from(act.width).expect("hidden"),
                    i32::try_from(gate_out.width).expect("width") / top_k.max(1),
                    ctx.stream,
                    std::ptr::null_mut(),
                    ctx.glu_limit,
                    ctx.glu_alpha,
                );
            }
        }
        // args: [gate_in, up, gate_out] — the CLAMPED GLU, in place on the
        // gate (the alias the declaration states).
        "mlp::gpt_oss_glu_bf16" => {
            need(3)?;
            let (gate_in, up, y) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, y, gate_in);
            unsafe {
                ffi::pie_k_mlp_gpt_oss_glu_bf16(
                    y.ptr.cast_const(),
                    up.ptr.cast_const(),
                    y.ptr,
                    rows * i32::try_from(y.width).expect("width"),
                    ctx.stream,
                    // The CLAMP is the statement's — `dsl::cuda::gpt_oss_glu`
                    // carries it as param bits, where it used to be a
                    // header default the driver re-derived from a config.
                    // `ctx.glu_limit` is the fallback for a trace that
                    // states none.
                    param_f32(spec, 0).unwrap_or(ctx.glu_limit),
                    ctx.glu_alpha,
                    std::ptr::null_mut(),
                );
            }
        }
        // args: [topk_idx, act, out, W bank] — the down GEMV. `act` is the
        // op's SECOND input, which is what the C++ arm reads
        // (`values.slot(ins[1])`), so the two drivers agree by
        // construction whatever the declaration wired there.
        "quant::mxfp4_moe_down_decode_bf16" => {
            need(4)?;
            let (idx, act, out) = (bound.args[0], bound.args[1], bound.args[2]);
            let bank = spec.weight.as_deref().ok_or_else(|| {
                DispatchRefusal::NoArm(format!("{}: op names no bank", bound.kernel))
            })?;
            let packed = bound.args[3].ptr.cast_const();
            let scales = resolver
                .weight(&format!("{bank}_scales"))
                .ok_or_else(|| DispatchRefusal::UnknownWeight(format!("{bank}_scales")))?;
            let bias = resolver.weight(&format!("{bank}_bias"));
            let top_k = i32::try_from(idx.width).expect("top_k");
            unsafe {
                ffi::pie_k_quant_mxfp4_moe_down_decode_bf16(
                    act.ptr.cast_const(),
                    idx.ptr.cast_const().cast(),
                    packed.cast(),
                    scales.cast(),
                    bias.unwrap_or(std::ptr::null()).cast(),
                    out.ptr,
                    rows,
                    top_k,
                    i32::try_from(out.width).expect("width") / top_k.max(1),
                    i32::try_from(act.width).expect("width") / top_k.max(1),
                    ctx.stream,
                );
            }
        }
        // ── gemma3n's arms (AltUp, the rank-K residual) ─────────────
        // args: [x, y] — one stream broadcast into K. `hc_mult` is the
        // ratio of the widths, so the expansion states its own K.
        "norm::hc_expand_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            let hidden = i32::try_from(x.width).expect("hidden");
            unsafe {
                ffi::pie_k_norm_hc_expand_bf16(
                    x.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("width") / hidden.max(1),
                    hidden,
                    ctx.stream,
                );
            }
        }
        // args: [x_in, x_out] — elementwise, in place; staged because the
        // lowering assigned distinct buffers.
        "norm::tanh_bf16" => {
            need(2)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            unsafe {
                ffi::pie_k_norm_tanh_bf16(
                    x_out.ptr,
                    rows * i32::try_from(x_out.width).expect("width"),
                    ctx.stream,
                );
            }
        }
        // args: [in_bf16, out_fp32] — the PREDICT coefficients are
        // `[K, K]` per token, so `k` is the square root of the width.
        "norm::altup_unpack_predict_coefs" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            let k = isqrt_exact(src.width).ok_or_else(|| {
                DispatchRefusal::NoArm(format!(
                    "{}: predict coefs width {} is not K*K",
                    bound.kernel, src.width
                ))
            })?;
            unsafe {
                ffi::pie_k_norm_altup_unpack_predict_coefs(
                    src.ptr.cast_const(),
                    dst.ptr.cast(),
                    rows,
                    k,
                    ctx.stream,
                );
            }
        }
        // args: [streams, coefs, predictions] — predict every stream from
        // the active one. The streams value is `[K, tokens, hidden]`, so
        // its stated width is K*hidden and `k` comes from the coefs.
        "norm::altup_predict_bf16" => {
            need(3)?;
            let (streams, coefs, preds) = (bound.args[0], bound.args[1], bound.args[2]);
            let k = isqrt_exact(coefs.width).ok_or_else(|| {
                DispatchRefusal::NoArm(format!(
                    "{}: predict coefs width {} is not K*K",
                    bound.kernel, coefs.width
                ))
            })?;
            unsafe {
                ffi::pie_k_norm_altup_predict_bf16(
                    streams.ptr.cast_const(),
                    coefs.ptr.cast_const().cast(),
                    preds.ptr,
                    k,
                    rows,
                    i32::try_from(streams.width).expect("width") / k.max(1),
                    ctx.stream,
                );
            }
        }
        // args: [x_in, x_out] — in place; the sparsity threshold is the
        // LAYER's `gaussian_inverse_cdf(activation_sparsity)`, a host
        // derivation the fire carries.
        "mlp::gaussian_topk_bf16" => {
            need(2)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            let std_mult = ctx
                .altup_std_mult_by_layer
                .get(bound.layers.start as usize)
                .copied()
                .ok_or_else(|| {
                    DispatchRefusal::NoArm(format!(
                        "{}: the fire carries no std_mult for layer {}",
                        bound.kernel, bound.layers.start
                    ))
                })?;
            unsafe {
                ffi::pie_k_mlp_gaussian_topk_bf16(
                    x_out.ptr,
                    rows,
                    i32::try_from(x_out.width).expect("dim"),
                    std_mult,
                    ctx.stream,
                );
            }
        }
        // args: [in_bf16, out_fp32] — the CORRECT coefficients are one
        // per stream, so the width IS K.
        "norm::altup_unpack_correct_coefs" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_norm_altup_unpack_correct_coefs(
                    src.ptr.cast_const(),
                    dst.ptr.cast(),
                    rows,
                    i32::try_from(src.width).expect("k"),
                    ctx.stream,
                );
            }
        }
        // args: [predictions, activated, coefs, corrected] — correct every
        // stream from the body's result. `h` is the ACTIVATED value's
        // width (the `[K, …]` values state none); the active index is the
        // deployment's.
        "norm::altup_correct_bf16" => {
            need(4)?;
            let (preds, activated, coefs, corrected) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_norm_altup_correct_bf16(
                    preds.ptr.cast_const(),
                    activated.ptr.cast_const(),
                    coefs.ptr.cast_const().cast(),
                    corrected.ptr,
                    i32::try_from(coefs.width).expect("k"),
                    rows,
                    i32::try_from(activated.width).expect("hidden"),
                    ctx.altup_active,
                    ctx.stream,
                );
            }
        }
        // args: [reference, target_rms_out] — the magnitude the rescale
        // below restores. `kAltupEps` is the C++'s own constexpr.
        "norm::compute_rms_bf16" => {
            need(2)?;
            let (reference, out) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_norm_compute_rms_bf16(
                    reference.ptr.cast_const(),
                    out.ptr.cast(),
                    rows,
                    i32::try_from(reference.width).expect("hidden"),
                    ALTUP_EPS,
                    ctx.stream,
                );
            }
        }
        // args: [streams, out] — the mean over K. The streams value states
        // no width, so K is the fire's.
        "norm::mean_streams_bf16" => {
            need(2)?;
            let (streams, out) = (bound.args[0], bound.args[1]);
            if ctx.altup_streams <= 0 {
                return Err(DispatchRefusal::NoArm(format!(
                    "{}: the fire states no altup stream count",
                    bound.kernel
                )));
            }
            unsafe {
                ffi::pie_k_norm_mean_streams_bf16(
                    streams.ptr.cast_const(),
                    out.ptr,
                    ctx.altup_streams,
                    rows,
                    i32::try_from(out.width).expect("hidden"),
                    ctx.stream,
                );
            }
        }
        // args: [x_in, target_rms, x_out] — in place over the projected
        // stream, restoring the magnitude `compute_rms` measured.
        "norm::magnitude_rescale_bf16" => {
            need(3)?;
            let (x_in, target, x_out) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            unsafe {
                ffi::pie_k_norm_magnitude_rescale_bf16(
                    x_out.ptr,
                    target.ptr.cast_const().cast(),
                    rows,
                    i32::try_from(x_out.width).expect("hidden"),
                    ALTUP_EPS,
                    ctx.stream,
                );
            }
        }
        other => return Err(DispatchRefusal::NoArm(other.to_string())),
    }
    Ok(())
}

/// gemma3n's AltUp epsilon — `constexpr float kAltupEps` in `gemma3n.cpp`,
/// a family constant rather than a config value.
#[cfg(feature = "bridge")]
const ALTUP_EPS: f32 = 1e-5;

/// `sqrt(w)` when `w` is a perfect square — how a `[K, K]` coefficient
/// block states its K.
#[cfg(feature = "bridge")]
fn isqrt_exact(w: u32) -> Option<i32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let r = (f64::from(w).sqrt().round()) as u32;
    (r * r == w).then(|| i32::try_from(r).unwrap_or(0))
}

/// Stage `src` into `dst` (device-to-device) when the lowering assigned an
/// in-place kernel distinct in/out buffers — the executor's half of that
/// contract, shared by the rope and elementwise-add arms.
#[cfg(feature = "bridge")]
fn stage_d2d(ctx: &DispatchCtx, rows: &std::ops::Range<u32>, dst: BoundArg, src: BoundArg) {
    if dst.ptr != src.ptr {
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let bytes = (rows.end - rows.start) as usize * src.width as usize * 2;
        let code = unsafe {
            cudaMemcpyAsync(
                dst.ptr,
                src.ptr.cast_const(),
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream.cast(),
            )
        };
        assert!(code == cudaError::cudaSuccess, "d2d stage: {code:?}");
    }
}

/// A store-backed [`Resolver`]: the per-family MAP, productized. The
/// loader (or a test) fills it; the executor asks it.
#[derive(Debug, Default)]
pub struct MapResolver {
    weights: std::collections::BTreeMap<String, *const c_void>,
    named: std::collections::BTreeMap<ValueId, *mut c_void>,
}

impl MapResolver {
    /// An empty map — every ask is a drift until something is inserted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a weight name to its device tensor.
    pub fn insert_weight(&mut self, name: impl Into<String>, ptr: *const c_void) {
        self.weights.insert(name.into(), ptr);
    }

    /// Bind a pinned seam value to its buffer.
    pub fn insert_named(&mut self, value: ValueId, ptr: *mut c_void) {
        self.named.insert(value, ptr);
    }
}

impl Resolver for MapResolver {
    fn weight(&mut self, name: &str) -> Option<*const c_void> {
        self.weights.get(name).copied()
    }
    fn named(&mut self, value: ValueId) -> Option<*mut c_void> {
        self.named.get(&value).copied()
    }
}

/// Why a fire's walk stopped.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct RunRefusal {
    /// Which launch refused.
    pub launch: usize,
    /// Its kernel.
    pub kernel: String,
    /// The refusal itself.
    pub why: RunRefusalKind,
}

/// The two ways a launch refuses.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub enum RunRefusalKind {
    /// Binding refused — see [`BindRefusal`].
    Bind(BindRefusal),
    /// Dispatch refused — see [`DispatchRefusal`].
    Dispatch(DispatchRefusal),
}

/// The prepared attention state a fire publishes, BY REGION.
///
/// A peel splits a fire's rows, and the tail region serves a different
/// row count against different requests — so it needs a different
/// prepared plan, and different KV page CSRs, and different output pins.
/// `Launch::peel`'s own doc says the first of those: a prepared plan "is
/// found by the rectangle's ROW COUNT".
///
/// The point of the type is that an arm no longer RESOLVES its state; it
/// is handed the state for the region it is executing. That is the
/// discipline `cuda.md` §3.1 asks for from the capture side — the C++'s
/// `f28ec1fed`, "the plan follows `kv_layer`; the family resolves and
/// hands the answer over" — and a peel region needs it for the same
/// reason a replayed capture does: neither may assume the fire's.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone, Copy, Default)]
pub struct AttnRegions<'a> {
    /// The fire's own state, for rectangles that span it.
    pub fire: Option<&'a AttnCtx>,
    /// A peel TAIL's state. `None` on a fire with no split, where it is
    /// never selected because no rectangle is windowed.
    pub tail: Option<&'a AttnCtx>,
}

#[cfg(feature = "bridge")]
impl<'a> AttnRegions<'a> {
    /// A fire with no split: one prepared state, every rectangle.
    #[must_use]
    pub const fn whole(fire: Option<&'a AttnCtx>) -> Self {
        Self { fire, tail: None }
    }

    /// A peeled fire: the prefix uses the fire's state, the tail its own.
    #[must_use]
    pub const fn split(fire: &'a AttnCtx, tail: &'a AttnCtx) -> Self {
        Self { fire: Some(fire), tail: Some(tail) }
    }

    /// The state a rectangle executes against.
    ///
    /// Keyed on whether the rectangle is WINDOWED, which is what makes it
    /// a tail: a peel's prefix starts at row zero and its tail does not.
    #[must_use]
    pub fn of(&self, rows: &std::ops::Range<u32>) -> Option<&'a AttnCtx> {
        if rows.start == 0 { self.fire } else { self.tail.or(self.fire) }
    }
}

/// Execute one fire: bind and dispatch every launch of the lowering, in
/// order. The walk the full-decode smoke proved, as the executor's entry.
///
/// # Errors
///
/// The first refusing launch, with its index and kernel — a drift
/// diagnosis, never a runtime condition to retry.
#[cfg(feature = "bridge")]
pub fn run<R: Resolver>(
    lowered: &Lowered,
    dplan: &DispatchPlan,
    frame: Frame,
    resolver: &mut R,
    ctx: &DispatchCtx,
    attn: AttnRegions<'_>,
    gdn: Option<&GdnCtx>,
) -> Result<usize, RunRefusal> {
    for (i, launch) in lowered.launches.iter().enumerate() {
        let kernel = || lowered.kernels[launch.kernel as usize].clone();
        let bound = bind(lowered, launch, frame, resolver).map_err(|e| RunRefusal {
            launch: i,
            kernel: kernel(),
            why: RunRefusalKind::Bind(e),
        })?;
        dispatch(&bound, dplan.spec(i), frame, resolver, ctx, attn.of(&launch.rows), gdn)
            .map_err(|e| {
            RunRefusal { launch: i, kernel: kernel(), why: RunRefusalKind::Dispatch(e) }
        })?;
    }
    Ok(lowered.launches.len())
}

/// One open conditional on the capture stack: the node, and which of its
/// two arms is currently being captured into.
#[cfg(feature = "bridge")]
struct OpenCond {
    cond: crate::cuda::Cond,
    /// The tree node ([`Lowered::conds`] index) whose body is open.
    node: u32,
}

/// The path from the root to `cond`, as tree-node indices.
#[cfg(feature = "bridge")]
fn cond_path(conds: &[model_compiler::lower::CondRegion], cond: u32) -> Vec<u32> {
    let mut path = Vec::new();
    let mut at = cond;
    while at != Launch::NO_COND {
        path.push(at);
        let Some(node) = conds.get(at as usize) else { break };
        at = node.parent;
    }
    path.reverse();
    path
}

/// Are these two tree nodes the two arms of ONE conditional?
///
/// Read off the lowering's stated pairing rather than derived from
/// `(parent, slot, param)`: a family states the same guard once per
/// layer, so those three fields identify one conditional PER LAYER, and
/// deriving would pair an arm with some other layer's else body.
#[cfg(feature = "bridge")]
fn siblings(conds: &[model_compiler::lower::CondRegion], a: u32, b: u32) -> bool {
    conds.get(a as usize).is_some_and(|x| x.sibling == b)
}

/// Execute one fire INTO A CAPTURE, rebuilding the union lowering's guard
/// tree as conditional graph nodes.
///
/// This is the supergraph's body. `lowered` must come from
/// `lower_with(.., GuardMode::Union)`: every arm is present, tagged with
/// its place in the tree, and nothing about the fire's variant bits has
/// been decided. What decides them is the device predicate word the
/// builder was constructed with — read from inside the graph, per launch,
/// with no host round-trip and no recapture.
///
/// The walk is a stack diff. Launches arrive in tree-walk order, so the
/// region path only ever extends, switches arms, or retreats; each of
/// those is one builder call.
///
/// `ctx.stream` is OVERWRITTEN per region — the arms issue onto whatever
/// stream the builder is currently capturing, which is the root at depth
/// zero and a pooled body stream inside an arm. That is the whole of what
/// "issuable into a capture" means for a dispatch arm: it is the same
/// call, onto a different stream.
///
/// # Errors
///
/// The first refusing launch, or a CUDA refusal from the builder
/// (reported against the launch that provoked it).
#[cfg(feature = "bridge")]
pub fn run_captured<R: Resolver>(
    lowered: &Lowered,
    dplan: &DispatchPlan,
    frame: Frame,
    resolver: &mut R,
    ctx: &DispatchCtx,
    attn: AttnRegions<'_>,
    gdn: Option<&GdnCtx>,
    builder: &mut crate::cuda::SupergraphBuilder<'_>,
) -> Result<usize, RunRefusal> {
    let mut ctx = ctx.clone();
    let mut stack: Vec<OpenCond> = Vec::new();

    // A CUDA refusal is reported against the launch that provoked it, so
    // that a capture failure reads like every other drift diagnosis
    // rather than like an unrelated device error.
    let cuda = |i: usize, kernel: &str, e: crate::error::Error| RunRefusal {
        launch: i,
        kernel: kernel.to_string(),
        why: RunRefusalKind::Dispatch(DispatchRefusal::NoArm(format!("capture: {e}"))),
    };

    for (i, launch) in lowered.launches.iter().enumerate() {
        let kernel = lowered.kernels[launch.kernel as usize].clone();
        let target = cond_path(&lowered.conds, launch.cond);

        // How much of the open path the target still agrees with.
        let mut keep = 0;
        while keep < stack.len() && keep < target.len() && stack[keep].node == target[keep] {
            keep += 1;
        }
        // The frame just past the agreement may be the OTHER ARM of the
        // same conditional, which is a body switch rather than a close.
        let switch_at = (keep < stack.len()
            && keep < target.len()
            && siblings(&lowered.conds, stack[keep].node, target[keep]))
        .then_some(keep);
        let close_to = switch_at.map_or(keep, |s| s + 1);

        while stack.len() > close_to {
            builder.end_body().map_err(|e| cuda(i, &kernel, e))?;
            let f = stack.pop().expect("stack is non-empty");
            builder.close_cond(&f.cond).map_err(|e| cuda(i, &kernel, e))?;
        }

        if let Some(s) = switch_at {
            builder.end_body().map_err(|e| cuda(i, &kernel, e))?;
            let want = target[s];
            let body = arm_body(&stack[s].cond, &lowered.conds, want);
            builder.begin_body(body).map_err(|e| cuda(i, &kernel, e))?;
            stack[s].node = want;
            keep = s + 1;
        }

        for &node in &target[keep..] {
            let region = lowered.conds[node as usize];
            // Always with_else: the sibling arm may arrive later, and a
            // conditional opened without an else body has nowhere to put
            // it.
            let cond = builder
                .open_cond(region.slot, true)
                .map_err(|e| cuda(i, &kernel, e))?;
            let body = arm_body(&cond, &lowered.conds, node);
            builder.begin_body(body).map_err(|e| cuda(i, &kernel, e))?;
            stack.push(OpenCond { cond, node });
        }

        ctx.stream = builder.stream().as_raw().cast::<c_void>();

        let bound = bind(lowered, launch, frame, resolver).map_err(|e| RunRefusal {
            launch: i,
            kernel: kernel.clone(),
            why: RunRefusalKind::Bind(e),
        })?;
        dispatch(&bound, dplan.spec(i), frame, resolver, &ctx, attn.of(&launch.rows), gdn)
            .map_err(|e| {
            RunRefusal { launch: i, kernel: kernel.clone(), why: RunRefusalKind::Dispatch(e) }
        })?;
    }

    // Unwind whatever the last launch left open.
    let last = lowered.launches.len().saturating_sub(1);
    while let Some(f) = stack.pop() {
        builder.end_body().map_err(|e| cuda(last, "<unwind>", e))?;
        builder.close_cond(&f.cond).map_err(|e| cuda(last, "<unwind>", e))?;
    }

    Ok(lowered.launches.len())
}

/// Which of a conditional's two bodies serves tree node `node`.
#[cfg(feature = "bridge")]
fn arm_body(
    cond: &crate::cuda::Cond,
    conds: &[model_compiler::lower::CondRegion],
    node: u32,
) -> cudarc::runtime::sys::cudaGraph_t {
    let on_true = conds.get(node as usize).is_none_or(|r| r.on_true);
    if on_true {
        cond.if_body()
    } else {
        cond.else_body().unwrap_or_else(|| cond.if_body())
    }
}

/// Resolve one [`Arg`] — the three rules, shared by [`bind`] and by the
/// arms that resolve an op's OUTPUT placements from the join.
///
/// # Errors
///
/// See [`BindRefusal`].
pub fn resolve_arg<R: Resolver>(
    arg: &Arg,
    frame: Frame,
    resolver: &mut R,
) -> Result<BoundArg, BindRefusal> {
    resolve_arg_windowed(arg, frame, resolver, 0)
}

/// [`resolve_arg`], addressing from row `row` of the operand rather than
/// from its base.
///
/// `row` is in the LAUNCH's row space, and the stride is the operand's own
/// — `width` elements of `bytes` each, which is why the lowering states
/// the element width at all.
///
/// # Errors
///
/// See [`BindRefusal`].
pub fn resolve_arg_windowed<R: Resolver>(
    arg: &Arg,
    frame: Frame,
    resolver: &mut R,
    row: u32,
) -> Result<BoundArg, BindRefusal> {
    Ok(match arg {
        Arg::Arena { at, width, bytes } => {
            let skip = row as usize * *width as usize * *bytes as usize;
            let at = *at + skip;
            if at >= frame.arena_bytes {
                return Err(BindRefusal::ArenaOutOfBounds {
                    at,
                    arena_bytes: frame.arena_bytes,
                });
            }
            BoundArg {
                ptr: unsafe { frame.arena.cast::<u8>().add(at) }.cast(),
                width: *width,
            }
        }
        Arg::Named { value, width } => BoundArg {
            ptr: resolver
                .named(*value)
                .ok_or(BindRefusal::UnknownNamed(*value))?,
            width: *width,
        },
        Arg::Weight(name) => {
            // `scale.` marks a CONSTANT riding the name slot — "a binder
            // never looks for it" (`dsl::cuda::scalar_mul`). The value
            // reaches the arm through `DispatchCtx::scales`; the operand
            // slot binds a dangling sentinel so the launch's arity holds.
            if name.starts_with("scale.") {
                BoundArg { ptr: std::ptr::NonNull::<c_void>::dangling().as_ptr(), width: 0 }
            } else {
                BoundArg {
                    ptr: resolver
                        .weight(name)
                        .ok_or_else(|| BindRefusal::UnknownWeight(name.clone()))?
                        .cast_mut(),
                    width: 0,
                }
            }
        }
    })
}

/// Bind one launch's operands against the frame and the resolver.
///
/// # Errors
///
/// See [`BindRefusal`] — each names the drift it diagnoses.
pub fn bind<'a, R: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    frame: Frame,
    resolver: &mut R,
) -> Result<BoundLaunch<'a>, BindRefusal> {
    let kernel = &lowered.kernels[launch.kernel as usize];
    // THE WINDOW, applied once and here.
    //
    // A peel's tail region serves rows `[win_start, …)` of a full-N
    // buffer, and every arm binds a pointer plus a row COUNT — so a
    // base-bound launch runs over the prefix's rows instead. That was
    // §4's fourth decline-rule (the generated branches guarded on
    // `rows.start == 0` and fell through) and it was also a live bug in
    // every hand arm, which had the same reading and no guard. Nothing
    // noticed until a fire finally peeled.
    //
    // Applied in the binder rather than in the arms because the arms are
    // not the only consumer: the op join's `outs` and `aux` resolve
    // through the same `resolve_arg`, and windowing one without the other
    // is how a launch reads its input at the window and writes its output
    // at the base.
    //
    // The `_devwin` forms are the stated exception. Their contract is
    // BASE pointers — the grid spans every lane and out-of-window rows
    // early-out on a device word — which is what makes them replayable
    // across splits, so windowing them would offset twice.
    let row = if kernel.ends_with("_devwin") { 0 } else { launch.rows.start };
    let mut args = Vec::with_capacity(launch.args.len());
    for arg in &lowered.args[launch.args.start as usize..launch.args.end as usize] {
        args.push(resolve_arg_windowed(arg, frame, resolver, row)?);
    }
    Ok(BoundLaunch { kernel, rows: launch.rows.clone(), layers: launch.layers.clone(), args })
}
