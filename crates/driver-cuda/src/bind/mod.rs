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

/// The kernel-facing records and the generated `extern "C"` bridge —
/// what a bound launch is bound TO.
pub mod abi;

/// Tier A: a device entry point loaded from a cubin, its arguments
/// marshalled from the row, and `cuLaunchKernel`. No host launcher.
pub mod device;

/// Tier A: the arithmetic a stated [`kernels::LaunchRule`] names — what
/// the C++ launchers computed inside `<<<>>>`.
pub mod launch;

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
            // WHY THIS DOES NOT SKIP OUTPUT-LESS GUARDS, which is the one
            // line that would let a DECODE be captured.
            //
            // Guards nest and this loop runs in op order, so an inner guard
            // overwrites the outer one that actually owns the value — and the
            // decode arm's innermost guard produces nothing, leaving `outs`
            // empty for its attention dispatch. `Union` reads that slot to
            // find where attention lands (`Resolve` may count launches
            // instead; `Union` may not, since every arm is present), so every
            // decode declines the graph and walks its ~400 launches by hand:
            // ~9 ms for one token on a 0.6B model.
            //
            // Adding `&& !op.outputs.is_empty()` here does make every decode
            // capture and replay, and issue drops to ~7.9 ms. It also
            // exposes two things decode capture has never been held to,
            // because no decode has ever been captured:
            //
            //   1. `pie_cuda_resize_pool` moved the KV pages without
            //      invalidating captures. FIXED — it bumps the epoch now,
            //      which is a real bug either way.
            //   2. Phi-3 FAULTS INSIDE THE CAPTURE of its first decode
            //      (`[sg] miss …launches=580`, then SIGSEGV). Not diagnosed.
            //
            // So the line stays out until (2) is understood. See
            // `.wiki/new-driver/next.md`.
            if let OpKind::Guard { arms, else_ops } = &op.kind
                && !op.outputs.is_empty()
            {
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
        let cache = unsafe { crate::bind::abi::ffi::pie_x_make_decode_plan() };
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
        unsafe { crate::bind::abi::ffi::pie_x_set_decode_plan_int_base(self.cache, bytes) };
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
        workspace: crate::bind::abi::AttentionWorkspaceView,
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
        workspace: crate::bind::abi::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        full_attention_variant: bool,
        window_left: i32,
    ) {
        let num_requests =
            i32::try_from(kv_page_indptr_h.len() - 1).expect("request count fits i32");
        unsafe {
            crate::bind::abi::ffi::pie_x_plan_attention_flashinfer_decode_bf16(
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
        unsafe { crate::bind::abi::ffi::pie_x_destroy_decode_plan(self.cache) };
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
        let cache = unsafe { crate::bind::abi::ffi::pie_x_make_prefill_plan() };
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
        workspace: crate::bind::abi::AttentionWorkspaceView,
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
            crate::bind::abi::ffi::pie_x_plan_attention_flashinfer_prefill_bf16(
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
        unsafe { crate::bind::abi::ffi::pie_x_destroy_prefill_plan(self.cache) };
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
    /// different splits. See [`crate::device::PeelWindowWord`].
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
    pub lora: Option<(*const crate::fire::lora::LoraFireState, *mut c_void)>,
    /// WHICH ROWS this fire samples, device-resident `[sampled_rows]` i32.
    ///
    /// A prefill's readout is one distribution per request and its stream
    /// is one row per token, so something has to pick — and the epilogue's
    /// gather is what picks. Null when the fire samples every row, which
    /// is the decode case and the case where the lowering states no
    /// gather at all.
    ///
    /// A device pointer on the ctx for the same reason `peel_window` is
    /// one: the launcher takes it loose, and no text can state it.
    pub sampling_indices: *const i32,
    /// How many rows that is.
    pub sampled_rows: i32,
}

#[cfg(feature = "bridge")]
impl DispatchCtx {
    /// The theta a layer-tagged rope launch fires with: the per-layer
    /// entry when the family splits theta by layer kind, else the
    /// uniform value.
    ///
    /// Named `theta` and not `theta_of` because a row says
    /// `Source::CtxByLayer("theta")` and the generated branch calls
    /// `ctx.theta(layer)`. The fallback is deliberately on this side —
    /// the table states that the value is indexed by the statement's
    /// layer, which is all it can know, and whether a family's vector is
    /// short is the driver's to answer.
    pub(crate) fn theta(&self, layer: usize) -> f32 {
        self.rope_theta_by_layer
            .get(layer)
            .copied()
            .unwrap_or(self.rope_theta)
    }

    /// gemma3n's per-layer `gaussian_inverse_cdf(activation_sparsity)` —
    /// the `std_mult` its sparse layers' `gaussian_topk` takes.
    ///
    /// Named for `Source::CtxByLayer("altup_std_mult")`, which is what
    /// the generated branch calls. Zero where the deployment states no
    /// sparse layer, which the kernel reads as "keep everything" -- the
    /// hand arm REFUSED instead, and refusing is wrong here: a layer
    /// outside the sparse set is the normal case, not a missing fact.
    pub(crate) fn altup_std_mult(&self, layer: usize) -> f32 {
        self.altup_std_mult_by_layer.get(layer).copied().unwrap_or(0.0)
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
    /// The workspace, as launchers take it — the DECODE plans'.
    pub workspace: crate::bind::abi::AttentionWorkspaceView,
    /// The PREFILL plan's own workspace.
    ///
    /// A FlashInfer plan writes its schedule into the workspace it was
    /// raised against, so a decode plan and a prefill plan sharing one is
    /// one clobbering the other. That was invisible while the shell raised
    /// only the plan this fire's text named; `.wiki/driver/graph.md` §5 ①
    /// raises every plan the geometry permits, so the two storages had to
    /// come apart. A launcher must take the workspace its own plan was
    /// raised in.
    pub prefill_workspace: crate::bind::abi::AttentionWorkspaceView,
    /// One KV view per layer, indexed by the launch's layer.
    pub layers: Vec<crate::bind::abi::KvCacheLayerView>,
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
    /// The FOLDED rows' base, distinct from [`Self::score_out`].
    ///
    /// The prefill capture writes raw scores and their per-request fold to
    /// different extents, and this used to pass `score_out` for both — safe
    /// only because the sink was always null and an empty CSR made both
    /// zero-length. A real sink has to name the two separately or the fold
    /// overwrites the rows it is folding.
    pub folded_out: *mut f32,
    /// See [`Self::score_out`].
    pub score_indptr_d: *const i32,
    /// The custom attention mask and its per-request base, one byte per
    /// `(q, kv)` pair. Published on every fire for the same reason the
    /// score sink is: `HasCustomMask` is a folded predicate, so the arm
    /// has to be RECORDABLE whether or not this fire takes it. The
    /// resident form is plain causal — the same answer the unmasked arm
    /// computes — until a program stages a real one.
    pub mask_d: *const u8,
    /// See [`Self::mask_d`].
    pub mask_indptr_d: *const i32,
    /// The attention output slot the o_proj reads — guard-owned like
    /// `q_out`, and one arena slot reused by every layer (liveness).
    pub o_out: *mut c_void,
    /// LSE scratch the decode dispatch writes.
    pub lse_out_d: *mut f32,
    /// The OBSERVATION window the score sink keeps, from `crate::boot`.
    ///
    /// Carried rather than read here, so the knob is parsed once. It was
    /// a `OnceLock` around `env::var_os` reached from two call sites —
    /// which is one parse by luck rather than by design.
    pub score_window: u32,
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
fn dispatch_generated<R: Resolver>(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    frame: Frame,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
    gdn: Option<&GdnCtx>,
    resolver: &mut R,
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
    /// `Source::InWidthOver`/`OutWidthOver`: an operand's row width
    /// divided by a context field — how many head-dims, or PLE layers,
    /// fit in a row.
    ///
    /// The `max(1)` is here rather than in the row for the reason
    /// [`IsSet`] is: what a fire that states no divisor means is the
    /// fire's business. It is belt to the guard's braces — a row stating
    /// this source is refused outright when the field is unset — and
    /// exists so that a future guard change cannot turn a refusal into a
    /// division by zero.
    /// `Source::NamedScale`: a `scale.<name>` slot resolved out of the
    /// driver's table.
    ///
    /// By NAME because a scale is a constant, not a tensor -- the table
    /// comes from the config, not the weight store, so a lookup that
    /// misses is drift rather than absence and the branch declines.
    fn named_scale(ctx: &DispatchCtx, spec: &LaunchSpec) -> Option<f32> {
        let name = spec.weight.as_deref()?.strip_prefix("scale.")?;
        ctx.scales.get(name).copied()
    }

    /// `Source::RotaryWidth`: how many channels rotate, from whichever
    /// of three places states it.
    ///
    /// The order prefers what a STATEMENT said -- the launch's own param
    /// (`dsl::cuda::rope_partial` carries it on the wire), then the
    /// semantic `Rope { partial }`, then the fire's per-layer table for
    /// a family whose width is per-layer and whose statement carries
    /// none. The first two are one fact under two spellings and both are
    /// live: qwen3_5's prefill states the launch, its decode records the
    /// semantic op.
    fn rotary_width(ctx: &DispatchCtx, spec: &LaunchSpec, layer: usize) -> Option<u32> {
        spec.params
            .first()
            .copied()
            .filter(|r| *r > 0)
            .or(spec.rope_partial)
            .or_else(|| ctx.rotary_by_layer.get(layer).copied().filter(|r| *r > 0))
    }

    /// `Source::LayerScale`: the layer's own scalar, `1.0` where it
    /// states none.
    ///
    /// One TERM of a fused norm rather than the whole launch, so a miss
    /// is one rather than a refusal -- the scale is 1 at the attention
    /// landing and the PLE landing carries the layer's own. The C++
    /// reads `layer_scalar_value` the same way.
    fn layer_scale(ctx: &DispatchCtx, spec: &LaunchSpec) -> f32 {
        spec.weight
            .as_deref()
            .filter(|n| n.ends_with("ple_norm"))
            .map_or(1.0, |n| ctx.scales.get(n).copied().unwrap_or(1.0))
    }

    /// `Source::AttnPlan`'s rule, which is the DRIVER'S and not a row's.
    ///
    /// Two-kind families keep a second decode plan for their
    /// full-attention layers, because the two kinds disagree on head dim
    /// (gemma-4: 512 vs 256). The C++ spells the choice
    /// `cur_full ? decode_plan_full : decode_plan_sliding`; the fact it
    /// turns on is `window_left_by_layer[l] == -1`, which the driver owns
    /// and a row cannot see. So a row asks for "the decode plan for my
    /// layer" and this answers.
    fn attn_plan(a: &AttnCtx, spec: &LaunchSpec, layer: u32, family: &str) -> *mut c_void {
        match family {
            "decode" => {
                if window_of(spec, a, layer) == -1 && !a.decode_plan_full.is_null() {
                    a.decode_plan_full
                } else {
                    a.decode_plan
                }
            }
            // Prefill keeps ONE plan. Spelled as a family anyway so a row
            // never names a field, which is the property that let decode
            // grow a second plan without touching a row.
            _ => a.prefill_plan,
        }
    }

    fn width_over(b: &BoundLaunch<'_>, i: usize, by: i32) -> i32 {
        width_of(b, i) / by.max(1)
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
    /// A POINTER field a fire leaves null to say "not published".
    ///
    /// `attn::split_qkv_bf16_devwin`'s peel window is the case: a fire
    /// that published none is not one that launcher can run for, and its
    /// hand arm said exactly that. Null is the pointer spelling of zero,
    /// so it is the same test and not a new one.
    impl<T> IsSet for *const T {
        fn is_set(self) -> bool {
            !self.is_null()
        }
    }
    impl<T> IsSet for *mut T {
        fn is_set(self) -> bool {
            !self.is_null()
        }
    }
    fn is_set<T: IsSet>(v: T) -> bool {
        v.is_set()
    }

    /// `Source::Gdn`'s reach into the fire's recurrent geometry.
    ///
    /// Total because of the guard, not in spite of it: every generated
    /// branch that binds a GDN field carries `&& gdn.is_some()`, so the
    /// only path here has already proved it. The panic message names the
    /// invariant rather than the symptom, because if it ever fires the
    /// bug is in the emitter's guard and nowhere near this line.
    fn g_of<'a>(g: Option<&'a GdnCtx>) -> &'a GdnCtx {
        g.expect("a generated branch binding a GDN field is guarded on gdn.is_some()")
    }

    /// `Source::ResultOrRegion`'s read: the JOIN's `i`-th output.
    ///
    /// Not `b.args[n_in + i]`, which is the launch's own SSA output, and
    /// the difference is the whole variant. A statement inside a
    /// value-producing region declares NO result of its own and binds the
    /// enclosing guard's -- the recurrence three-way and gpt-oss's
    /// attention chain are both that shape. `LaunchSpec::outs` is where
    /// the join put it.
    ///
    /// This used to be the reason such rows could not generate ("the Rust
    /// join does not carry the guard's value yet"), which stopped being
    /// true when `DispatchPlan` learned to map a region op back to its
    /// owning guard. The read was already here, spelled `out_slot`.
    fn join_out<R: Resolver>(
        spec: &LaunchSpec,
        i: usize,
        frame: Frame,
        resolver: &mut R,
    ) -> Option<BoundArg> {
        resolve_arg(spec.outs.get(i)?, frame, resolver).ok()
    }

    /// `Source::Aux`'s reach: a value the statement does NOT carry as an
    /// arg, collected onto the join by the lowering.
    ///
    /// # Why this did not exist, and what its absence looked like
    ///
    /// `Source::Aux` was added to the vocabulary and to the emitter —
    /// which emits both `join_aux(..).is_some()` in the guard and
    /// `join_aux(..)` in the bind — and this function, which both of
    /// them call, was never written. Nothing noticed because no row
    /// states `Aux`, so the emitter's arms are unreachable and the
    /// missing symbol never reaches a compiler.
    ///
    /// Sourcing one produced a branch that read correctly and "never
    /// fired". It could not have: the crate it was added to did not
    /// build, and the binary under test was the previous one. That is
    /// the second time a stale build has been mistaken for a live
    /// mystery in this file — the first is recorded in the kernel
    /// table's own comment — and both times the tell was the same, a
    /// measurement that could not be true of the code being read.
    /// `Source::InWidthIsqrt`'s arithmetic: the exact integer square
    /// root of a width, or `0` when it is not a perfect square.
    ///
    /// ZERO rather than a refusal, because a generated branch's bind
    /// expressions run after its guard and have nowhere to refuse from.
    /// The launcher rejects a zero `K`, which is the same outcome the
    /// hand arm's `NoArm` produced and one layer lower.
    fn isqrt_exact_i32(w: i32) -> i32 {
        u32::try_from(w).ok().and_then(crate::bind::isqrt_exact).unwrap_or(0)
    }

    fn join_aux<R: Resolver>(
        spec: &LaunchSpec,
        i: usize,
        frame: Frame,
        resolver: &mut R,
    ) -> Option<BoundArg> {
        resolve_arg(spec.aux.get(i)?, frame, resolver).ok()
    }

    /// `Source::GdnSlab`'s reach: the statement's own layer's entry in
    /// one of the GDN context's per-layer slab vectors.
    ///
    /// ONE function returning an `Option`, unlike the `has_kv_layer` /
    /// `kv_view` pair beside it, because all THREE ways this can be
    /// absent are the same answer: no GDN context, no layer stated by the
    /// op, or no slab at that layer. The generator emits the same call
    /// into the guard as `.is_some()` and into the argument list as
    /// `.unwrap_or(null)`, so the test and the read cannot disagree —
    /// which is what the pair achieved by construction and this achieves
    /// by being one function.
    fn gdn_slab(
        g: Option<&GdnCtx>,
        state: Option<model_compiler::trace::StateRef>,
        field: &str,
    ) -> Option<*mut c_void> {
        let g = g?;
        let layer = state?.layer as usize;
        let v: &[u64] = match field {
            "conv_state" => &g.conv_state,
            "recurrent_state" => &g.recurrent_state,
            _ => return None,
        };
        match v.get(layer) {
            Some(&base) if base != 0 => Some(base as *mut c_void),
            _ => None,
        }
    }

    /// `Source::Attn`'s reach into the fire's attention context.
    fn a_of<'a>(a: Option<&'a AttnCtx>) -> &'a AttnCtx {
        a.expect("a generated branch binding an attention field is guarded on attn.is_some()")
    }

    /// `Source::KvLayerView`'s test, and its read.
    ///
    /// Two functions rather than one returning an `Option`, because the
    /// generator emits the test into the branch GUARD and the read into
    /// the argument list, and a guard cannot bind. The pair is what makes
    /// the read total.
    fn has_kv_layer(a: Option<&AttnCtx>, layer: usize) -> bool {
        a.is_some_and(|a| a.layers.len() > layer)
    }
    fn kv_view(a: Option<&AttnCtx>, layer: usize) -> crate::bind::abi::KvCacheLayerView {
        a_of(a).layers[layer]
    }

    /// `cast_const` for a pointer that is ALREADY const.
    ///
    /// A generated bind for a `U32s`/`I32s` operand spells
    /// `(e).cast_const().cast::<u32>()`, because most sources hand back a
    /// `*mut` — an arg's `ptr` is one, and so is most of `DispatchCtx`.
    /// A context field that is already `*const` has no inherent
    /// `cast_const`, and the generator cannot know which it got without
    /// the table carrying pointer mutability, which is a fact about the
    /// DRIVER's struct and not about the launcher.
    ///
    /// Inherent methods win over trait methods, so this covers exactly
    /// the case the inherent one does not and is invisible everywhere
    /// else.
    trait AlreadyConst: Copy {
        #[allow(clippy::wrong_self_convention)]
        fn cast_const(self) -> Self {
            self
        }
    }
    impl<T> AlreadyConst for *const T {}

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
    // `rope_partial` is the same class: a rotary width the statement
    // carries out of band. `beta_one` is not — it changes which ARG is
    // the destination, which arity guards already catch.
    //
    // This is the fallthrough working as designed, not a special case:
    // the hand arm reads the join and the generated branch says "not
    // mine" rather than guessing which reading applies.
    //
    // `aux` USED TO BE ON THIS LINE and no longer is. It belonged here
    // while no row could state a foreign value; `Source::Aux` is a row
    // stating one, and a blanket refusal ahead of the match made every
    // such branch unreachable — emitted, correct, and never entered.
    // The check moved INTO the guards, where it is per row: a row that
    // states no `Aux` carries `&& spec.aux.is_empty()` and gives the
    // same answer this line gave.
    // `per_head_dim` USED TO BE ON THIS LINE, and left for the same
    // reason `aux` did: `IfPresent(&PerHeadDim, ..)` is a row stating
    // the reading, and a blanket refusal ahead of the match made every
    // such row unreachable. Rows that state neither carry
    // `&& spec.per_head_dim.is_none()` and give the same answer.
    // `rope_partial` WAS THE LAST OF THE THREE, and it went the same
    // way: `Source::RotaryWidth` is a row stating the width, and its
    // guard is the fall-through finding one. Nothing blanket is left
    // ahead of the match, which is the property that matters — every
    // refusal here is now a refusal some ROW made, and a row that
    // declines says which operand it could not source.
    let n_in = spec.n_in;
    let n_out = spec.n_out;
    // `Source::WeightNamed`'s resolve, done ONCE and before the match so
    // that a branch's guard can test it. Null when the statement names no
    // weight OR when the store lacks the name — the two are different
    // situations and the same answer here, because the branch declines in
    // both and the hand arm below reports the second as `UnknownWeight`.
    // A `None` from the resolver is DRIFT, not absence, and saying so is
    // the fallthrough's job.
    let w_named: *const c_void = spec
        .weight
        .as_deref()
        .and_then(|n| resolver.weight(n))
        .unwrap_or(core::ptr::null());
    // The SECOND named weight, resolved beside the first for the same
    // reason: a statement that names two tensors by name — the GDN
    // prep's `a_log` and `dt_bias` — needs both before the guards run.
    let w_named2: *const c_void = spec
        .weight2
        .as_deref()
        .and_then(|n| resolver.weight(n))
        .unwrap_or(core::ptr::null());
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
    use crate::bind::abi::ffi;

    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("rows fit i32");

    // GENERATED FIRST. A row that states where its arguments come from
    // needs no arm, and the branch for it is emitted from the row — so
    // the hand-written match below is what is LEFT, not what is normal.
    // It shrinks as rows state their sources, which is a row's work.
    if dispatch_generated(bound, spec, frame, ctx, attn, gdn, resolver, rows) {
        return Ok(());
    }

    // The GDN arms' shared reads: the ctx itself, and the launch's state
    // layer's slab out of one of its per-layer vectors.
    let gdn_ctx = || -> Result<&GdnCtx, DispatchRefusal> {
        gdn.ok_or_else(|| DispatchRefusal::NoGdnCtx(bound.kernel.to_string()))
    };



    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("row count fits i32");
    // The op join's output placements: what a guard-region launch binds
    // for the value the GUARD owns (the recurrence three-way's core out).
    // The join's placements window with the args, or a launch reads its
    // input at the window and writes its output at the base.
    let win = if bound.kernel.ends_with("_devwin") { 0 } else { bound.rows.start };

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


    match bound.kernel {
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
                // args: [packed, rope_table, q_norm_w, k_norm_w]; the q output is
        // the observed-query PIN (outs[0], Named); the KV pages, CSRs and
        // write descriptors are the fire's ([`AttnCtx`]).
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
                        // args: [q] with the output guard-owned, or [q, o] as SSA. The
        // custom-mask arm: `HasCustomMask` selects it, and the mask rides
        // the ctx rather than the statement for the same reason the score
        // sink does — the predicate is folded, so one exec serves the fire
        // that stages a mask and the fire that does not, and the address
        // recorded now has to still be right when it goes true.
                // ── The MIXTURE's landing pair, both in-place ───────────────
        // Neither can generate: `emit_rust_dispatch` skips every
        // `in_place` row because a generated branch binds `Out(0)` and
        // calls, with nowhere to stage the copy the aliasing needs. The
        // rows already state their sources; staging is the whole
        // difference, and it is `stage_d2d` in both.

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
                // args: [packed, q_out, q_norm, k_norm] — gemma-4's fused local
        // decode post: split the packed projection, norm q/k, rope them
        // (rounded), norm v, write k/v straight to the pages. Only the
        // query survives as a value.
                // The rounded fused norm+rope, BOTH shapes the driver reaches:
        // [q, k, q_norm, k_norm] — the local pair, in place — and
        // [q_in, q_out, q_norm] — a KV-SHARED layer's Q-ONLY form, which
        // the driver reaches by passing `num_kv_heads = 0`, never by a
        // generic rope.
                // args: [q, o] — the PLANLESS flashinfer prefill (plans
        // internally per fire; reads the host CSR mirrors).
                // args: [q, o] — the naive paged prefill, for the head dims
        // flashinfer's prefill template refuses (gemma-4's 512).
                // args: [x, hidden_in, hidden_out, norm_out, w, next_w] — FOUR
        // statements in one launch: norm x, land on the stream, scale,
        // norm THAT with the next block's weight. The scale is 1 at the
        // attention landing; the PLE landing carries the layer's own
        // scalar, resolved from `DispatchCtx::scales` by the weight's
        // name (the C++ reads `layer_scalar_value` the same way).
                // args: [dt_raw, a, dt_out, da_out]; dt_bias rides the spec's
        // aux slots (the statement does not carry it — the C++ hand pass
        // wires it through its workspace). `time_step_min` is 0 at both
        // C++ call sites.
        // args: [conv_out, dt_pre, y] + the aux slots [dt_raw, a, d,
        // dt_bias, dt_pre, da_pre] — the selective scan, over the
        // layer's slab and the fire's slots. On L40S (sm89) the C++
        // ALWAYS lands here: its FlashInfer SSU try refuses below sm90.
        // args: [x, gate, y, W] — the grouped, gated output norm. The
        // gate is the SPLIT's contiguous copy, so its stride is its own
        // width (the C++ hand pass reads the gate in place inside the
        // packed projection, stride `projection_dim` — same values).
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
        // ── the pair/chunked activation and router variants ─────────
        // args: [gate, y] + aux[up] — the pair form, whose `up` the
        // statement cannot name (see the join's pre-pass).
                // args: [logits, idx, w] or [logits, idx, w, W bias] — the two
        // bias-capable routers; the bias is null when unstated.
                // ── gemma's arms ─────────────────────────────────────────────
        // args: [x_in, x_out, scale-name] — `x *= s`, the constant named
        // in the weight slot, resolved through `DispatchCtx::scales`.
                // args: [q_in, k_in, q_out, k_out] — in-place pair, staged like
        // `rope::rope_bf16` — or [q_in, q_out], the KV-SHARED layers'
        // Q-ONLY form: gemma-4's shared full layers rotate q through the
        // same launcher with `num_kv_heads = 0` and the q buffer riding
        // the k slot (`declared_forward.cpp`'s `RopeQOnlyPartial` — NOT
        // a fallback to a generic rope). The rotary width is the op's
        // statement; the head dim is the LAYER'S (gemma-4's full layers
        // run 512 where the fire-wide `ctx.head_dim` says 256), read off
        // the kv view the layer tag names.
                other => return Err(DispatchRefusal::NoArm(other.to_string())),
    }
    Ok(())
}

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
    cond: crate::device::Cond,
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
    builder: &mut crate::device::SupergraphBuilder<'_>,
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
        // RETAIN THE NODE this launch became.
        //
        // `.wiki/driver/graph.md` §6.2: "what is missing is bookkeeping,
        // not capability". Grid and block dims ARE updatable on an
        // instantiated graph, and two fires of the same shape family
        // differing only in row count have IDENTICAL topology — so the
        // update is legal and costs tens of microseconds against a
        // recapture's milliseconds. What stopped it was that a capture
        // retained nothing, so there was no handle to update and no way to
        // say which launch a handle belonged to.
        //
        // Recorded by INDEX, so `nodes[i]` is launch `i`. A dispatch that
        // issues more than one kernel records its last, which is the one
        // whose grid the row count moves; a dispatch that issues none
        // records nothing and leaves a gap.
        builder.retain_node(i);
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
    cond: &crate::device::Cond,
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
