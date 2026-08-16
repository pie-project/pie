//! The executor's first half: binding a flat launch's operands.
//!
//! Operands are [`Arg`]s — an arena offset, a backend-named value, or a
//! weight name — resolved by three family-independent rules. Dispatch is
//! the other half, so this side stays pure host logic.

/// The kernel-facing records and the generated `extern "C"` bridge.
pub mod abi;
/// Trace symbol → what runs it.
pub mod arms;
/// The query-only fire vocabulary an arm reads facts through.
pub mod cx;
/// The driver's answer to every fact a bind arm can ask for.
pub mod facts;
/// The derived column, and the dispatch that binds a crossed symbol from it.
pub mod table;

use std::ffi::c_void;

use model_compiler::lower::{Arg, Buffers, Launch, Lowered};
use model_ir::trace::ValueId;

/// The frame's activation arena: one device block of [`Lowered::arena_bytes`].
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Device base of the arena.
    pub arena: *mut c_void,
    /// Its extent; a reused arena can be SMALLER than the new fire needs.
    pub arena_bytes: usize,
}

/// Resolves the names the trace states against the driver's stores.
pub trait Resolver {
    /// The device pointer for a weight the trace names. `None` is DRIFT: the
    /// trace was made against another binding.
    fn weight(&mut self, name: &str) -> Option<*const c_void>;
    /// The device pointer for a backend-named value (`Buffers::NAMED`).
    fn named(&mut self, value: ValueId) -> Option<*mut c_void>;
}

/// One resolved operand: where it is, and how wide one row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundArg {
    /// The device address.
    pub ptr: *mut c_void,
    /// Elements per row; zero for a weight, whose extent is the tensor's.
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

/// Why a launch refused to bind. Every variant is a DRIFT diagnosis.
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

/// What one launch needs beyond its bound args: the op join, computed once
/// per lowering so no arm re-matches `OpKind` per fire.
#[derive(Debug, Clone, Default)]
pub struct LaunchSpec {
    /// The weight the op names. Concrete (`layer.3.q_proj`), never a template.
    pub weight: Option<String>,
    /// `Matmul::beta_one`: the residual fold; its target is the LAST arg.
    pub beta_one: bool,
    /// The op's OUTPUT placements — what a launch writes but its args omit.
    pub outs: Vec<Arg>,
    /// The SECOND weight an op names, when it names two.
    pub weight2: Option<String>,
    /// The per-request store this op addresses — a GDN arm's state layer.
    pub state: Option<model_ir::trace::StateRef>,
    /// `RmsnormPerHead`'s head width; kernel rows are `tokens * width/head_dim`.
    pub per_head_dim: Option<u32>,
    /// `Rope`'s partial-rotary channel count, when the op states one.
    pub rope_partial: Option<u32>,
    /// FOREIGN values: `[dt_raw, a, d, dt_bias, dt_pre, da_pre]`, when present.
    pub aux: Vec<Arg>,
    /// How many of the launch's args are INPUTS. The args are one flat run
    /// (inputs, outputs, weights) and these counts locate the slices.
    pub n_in: usize,
    /// How many are OUTPUTS — from what the LOWERER EMITTED, which for a
    /// guard-region launch is `op.dest` and not the empty `op.outputs`.
    pub n_out: usize,
    /// The wire scalars a statement carries that no operand shape gives. `i32`
    /// sent as `u32`, so `-1` arrives as `0xFFFF_FFFF`; [`window_of`] casts it.
    pub params: Vec<u32>,
    /// What fires this symbol, resolved at load. **No string compare at fire.**
    pub route: arms::Route,
}

/// The window a launch attends over: the STATEMENT's, else the context's.
#[cfg(feature = "_cuda")]
fn window_of(spec: &LaunchSpec, a: &AttnCtx, layer: u32) -> i32 {
    #[allow(clippy::cast_possible_wrap)]
    if let Some(&stated) = spec.params.first() {
        return stated as i32;
    }
    a.window_left_by_layer.get(layer as usize).copied().unwrap_or(a.window_left)
}

/// The per-launch op join over a whole lowering.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    specs: Vec<LaunchSpec>,
    routes: Vec<arms::Route>,
    unfireable: Vec<Unfireable>,
    /// What the boot said, so a re-lowering (a warm-up) resolves the same way.
    boot: Boot,
}

/// One symbol a lowering names that nothing can fire, and why.
#[derive(Debug, Clone)]
pub struct Unfireable {
    /// The symbol the lowering states.
    pub symbol: String,
    /// Why nothing can fire it.
    pub why: kernels_cuda::Refusal,
}

impl core::fmt::Display for Unfireable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.symbol, self.why)
    }
}

/// Resolve every symbol a lowering names to the thing that will fire it: one
/// scan of the symbol table by kernel id, pure in the table and [`Boot`].
#[must_use]
pub fn resolve(lowered: &Lowered, boot: Boot) -> Vec<arms::Route> {
    lowered.kernels.iter().map(|symbol| boot.route(symbol)).collect()
}

/// What the boot decided. `None` means "the boot did not say", not "false":
/// the caller gets [`Route::Unbound`](arms::Route::Unbound), not a kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Boot {
    /// Whether the KV cache stores the model's own bf16; must match the fire's
    /// reading of the same configuration.
    pub kv_native_bf16: Option<bool>,
}

impl Boot {
    /// One symbol's route, with the boot's answer folded in.
    #[must_use]
    pub fn route(self, symbol: &str) -> arms::Route {
        if symbol != STATED_KV_WRITE {
            return arms::route(symbol);
        }
        match self.kv_native_bf16 {
            Some(native) => arms::route(kernels_cuda::attn::kv_paged::write_kv_to_pages(native)),
            // Not `Unknown`: the symbol is declared, the boot fact is not.
            None => arms::Route::Unbound("a KV cache dtype for the writer the boot chose"),
        }
    }
}

/// The name a trace states, as against the two it resolves to.
const STATED_KV_WRITE: &str = "attn::write_kv_to_pages";

impl DispatchPlan {
    /// The boot this plan was resolved against. See the field.
    #[must_use]
    pub const fn boot(&self) -> Boot {
        self.boot
    }

    /// Join `lowered`'s launches with the ops that produced them. Infallible:
    /// the refusal is [`Self::unfireable`], asked separately.
    #[must_use]
    pub fn new(plan: &model_ir::trace::ForwardPlan, lowered: &Lowered) -> Self {
        Self::with_boot(plan, lowered, Boot::default())
    }

    /// [`Self::new`] with the boot's answer — the only caller of [`resolve`].
    #[must_use]
    pub fn with_boot(
        plan: &model_ir::trace::ForwardPlan,
        lowered: &Lowered,
        boot: Boot,
    ) -> Self {
        use model_ir::trace::Dim;
        use model_ir::trace::OpKind;
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
                // NAMED is a sentinel, not an offset; the operand side must agree.
                _ => Arg::Named {
                    value: lowered.value_owner.get(v as usize).copied().unwrap_or(v),
                    width: width_of(v),
                    bytes: plan
                        .values
                        .get(v as usize)
                        .map_or(2, |i| model_compiler::lower::dtype_bytes(i.dtype)),
                },
            }
        };
        // A GUARD's outputs belong to every launch of its regions.
        let mut guard_of: Vec<Option<usize>> = vec![None; plan.ops.len()];
        for (g, op) in plan.ops.iter().enumerate() {
            // Op order and nesting: only a value-producing guard may claim its
            // span, or an inner one overwrites the outer's attention `outs`.
            if let OpKind::Guard { arms, else_ops } = &op.kind
                && !op.outputs.is_empty()
            {
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>() + *else_ops as usize;
                for slot in guard_of.iter_mut().skip(g + 1).take(span) {
                    *slot = Some(g);
                }
            }
        }
        // The LoRA correction's qkv_in: the statement carries only its
        // in-place [q, v], so the INPUT is taken from the layer's qkv/q_proj.
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
        // The PAIR-form activations state ONE operand where their launcher
        // takes gate AND up, so the join hands `up` over via `spec.aux`.
        let mut pair_up: std::collections::BTreeMap<u16, Arg> = std::collections::BTreeMap::new();
        for launch in &lowered.launches {
            let op = &plan.ops[launch.op as usize];
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
        // One scan of the symbol table, resolved here and nowhere else.
        let routes = resolve(lowered, boot);
        // The load-time refusals: what the symbol table alone already knows.
        let unfireable: Vec<Unfireable> = lowered
            .kernels
            .iter()
            .zip(&routes)
            .filter_map(|(symbol, route)| {
                route.refusal().map(|why| Unfireable { symbol: symbol.clone(), why })
            })
            .collect();
        let specs = lowered
            .launches
            .iter()
            .map(|launch| {
                let op = &plan.ops[launch.op as usize];
                let out_values: &[ValueId] = if op.outputs.is_empty() {
                    guard_of[launch.op as usize].map_or(&[], |g| plan.ops[g].outputs.as_slice())
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
                    | OpKind::LmHead { weight } => {
                        LaunchSpec { weight: Some(weight.clone()), ..LaunchSpec::default() }
                    }
                    OpKind::CausalConv1d { weight, bias, .. } => LaunchSpec {
                        weight: Some(weight.clone()),
                        weight2: bias.clone(),
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
                    // The FIRST weight rides the spec too, for `scale.*` arms.
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
                // Mirrors the lowerer's split, not the op's dataflow:
                // `op.dest` is non-empty exactly where `outputs` is empty.
                spec.n_out =
                    if op.outputs.is_empty() { op.dest.len() } else { op.outputs.len() };
                spec.state = op.kind.state_ref();
                if let OpKind::RmsnormPerHead { head_dim, .. }
                | OpKind::SplitQGate { head_dim, .. } = op.kind
                {
                    spec.per_head_dim = Some(head_dim);
                }
                if let OpKind::Rope { partial, .. } = op.kind {
                    spec.rope_partial = partial;
                }
                if matches!(
                    lowered.kernels[launch.kernel as usize].as_str(),
                    "mlp::swiglu_bf16" | "mlp::swiglu_clamp_bf16" | "mlp::situ_bf16"
                ) && let Some(up) = pair_up.get(&launch.layers.start)
                {
                    spec.aux = vec![up.clone()];
                }
                if lowered.kernels[launch.kernel as usize] == "gemm::lora_qkv_correction"
                    && let Some(x) = lora_x.get(&launch.layers.start)
                {
                    spec.aux = vec![x.clone()];
                }
                spec.route = routes[launch.kernel as usize];
                spec
            })
            .collect();
        Self { specs, routes, unfireable, boot }
    }

    /// The spec for launch `i` — index-parallel with [`Lowered::launches`].
    #[must_use]
    pub fn spec(&self, i: usize) -> &LaunchSpec {
        &self.specs[i]
    }

    /// Every symbol this lowering names that NOTHING can fire. A `Route::Rows`
    /// with no generated arm is not here; it refuses at the fire with `NoArm`.
    #[must_use]
    pub fn unfireable(&self) -> &[Unfireable] {
        &self.unfireable
    }

    /// Row-world progress: `(row-world symbols, distinct symbols)`.
    #[must_use]
    pub fn sweep_progress(&self) -> (usize, usize) {
        (self.routes.iter().filter(|r| r.is_row_world()).count(), self.routes.len())
    }
}

/// FlashInfer's decode plan cache, owned in Rust. A raw pointer, not a `Box`:
/// [`Self::as_ptr`] is `const`, and a `*mut` keeps this `!Send`.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct DecodePlan {
    cache: *mut kernels_cuda::attn::fa2::plan::DecodePlanCache,
}

#[cfg(feature = "_cuda")]
impl DecodePlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(Box::new(kernels_cuda::attn::fa2::plan::DecodePlanCache::default())),
        }
    }

    /// The raw handle a dispatch arm passes as the `DecodePlanCache&`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut kernels_cuda::attn::fa2::plan::DecodePlanCache {
        // SAFETY: `cache` came from `Box::into_raw` in `new`, is never
        // reassigned, and `&mut self` proves no other reference is live.
        unsafe { &mut *self.cache }
    }

    /// Run FlashInfer's decode planner over the fire's HOST page indptr, inside
    /// the workspace's `begin_plan_update`/`end_plan_update` fence.
    // Safe by design: the view's pointers are the workspace's own.
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

    /// [`Self::plan_decode`] with `full_attention_variant` exposed: gemma-4
    /// plans TWO decode caches, its layer kinds disagreeing on head dim.
    ///
    /// # Panics
    ///
    /// If the planner declines.
    // Safe by design: the view's pointers are the workspace's own.
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
        use kernels_cuda::attn::fa2::plan as fa2;

        let _ = stream;
        let num_requests =
            i32::try_from(kv_page_indptr_h.len() - 1).expect("request count fits i32");
        let device = fa2::plan_device();
        let max_grid_size = fa2::decode_max_grid_size(head_dim, num_q_heads, num_kv_heads);
        let planned = fa2::plan_decode(
            self.get(),
            kv_page_indptr_h,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kernels_cuda::attn::plan::Workspace {
                float_bytes: workspace.float_bytes,
                int_bytes: workspace.int_bytes,
            },
            &device,
            max_grid_size,
            enable_cuda_graph,
            full_attention_variant,
            // `hnd_layout`: `bind` has no HND deployment.
            false,
            window_left,
        );
        if let fa2::Planned::Declined(why) = planned {
            panic!("flashinfer decode plan: {why}");
        }
    }
}

#[cfg(feature = "_cuda")]
impl Default for DecodePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "_cuda")]
impl Drop for DecodePlan {
    fn drop(&mut self) {
        // SAFETY: `cache` came from `Box::into_raw` in `new` and is dropped
        // exactly once, here.
        drop(unsafe { Box::from_raw(self.cache) });
    }
}

/// FlashInfer's prefill plan cache — [`DecodePlan`]'s twin, owned the same way.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct PrefillPlan {
    cache: *mut kernels_cuda::attn::fa2::plan::PrefillPlanCache,
}

#[cfg(feature = "_cuda")]
impl PrefillPlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(
                Box::new(kernels_cuda::attn::fa2::plan::PrefillPlanCache::default()),
            ),
        }
    }

    /// The raw handle a dispatch arm passes.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut kernels_cuda::attn::fa2::plan::PrefillPlanCache {
        // SAFETY: as `DecodePlan::get`.
        unsafe { &mut *self.cache }
    }

    /// The planned cache, for a caller that fires the dispatch itself — the
    /// ViT tower, which holds no [`DispatchCtx`] for `bind::service` to serve.
    #[must_use]
    pub fn cache(&self) -> &kernels_cuda::attn::fa2::plan::PrefillPlanCache {
        // SAFETY: as `DecodePlan::get`; `&self` proves no `&mut` is live.
        unsafe { &*self.cache }
    }

    /// Run FlashInfer's prefill planner over the fire's HOST CSRs, bracketed
    /// by the workspace's plan-update fence. `kv_last_page_lens_h` is accepted
    /// and **not read**: it guards the SM90 route, which this never plans.
    ///
    /// # Panics
    ///
    /// If the planner declines.
    // Safe by design: the view's pointers are the workspace's own.
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
        self.plan_prefill_variant(
            qo_indptr_h,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            stream,
            enable_cuda_graph,
            window_left,
            PrefillPlanFlags {
                full_attention_variant: false,
                hnd_layout: false,
                causal_mask: true,
                custom_mask: false,
                wants_prefill_score: false,
            },
        );
    }

    /// [`Self::plan_prefill`] with the five variant flags exposed, for a
    /// caller that needs a non-causal plan (the ViT is bidirectional).
    ///
    /// # Panics
    ///
    /// If the planner declines.
    // Safe by design like the seam methods.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_prefill_variant(
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
        flags: PrefillPlanFlags,
    ) {
        use kernels_cuda::attn::fa2::plan as fa2;

        let _ = (stream, kv_last_page_lens_h);
        let num_requests = i32::try_from(qo_indptr_h.len() - 1).expect("request count fits i32");
        let total_tokens = i32::try_from(*qo_indptr_h.last().expect("a CSR has a last entry"))
            .expect("token count fits i32");
        let device = fa2::plan_device();
        let planned = fa2::plan_prefill(
            self.get(),
            qo_indptr_h,
            kv_page_indptr_h,
            total_tokens,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kernels_cuda::attn::plan::Workspace {
                float_bytes: workspace.float_bytes,
                int_bytes: workspace.int_bytes,
            },
            &device,
            enable_cuda_graph,
            window_left,
            flags.full_attention_variant,
            flags.hnd_layout,
            flags.causal_mask,
            flags.custom_mask,
            flags.wants_prefill_score,
        );
        if let fa2::Planned::Declined(why) = planned {
            panic!("flashinfer prefill plan: {why}");
        }
    }
}

/// The five booleans `plan_attention_flashinfer_prefill_bf16` took after its
/// numbers. Named rather than positional: `causal_mask` in `hnd_layout`'s
/// slot plans a causal ViT, and a name makes that a compile error.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct PrefillPlanFlags {
    /// `FullAttention` rather than the sliding-window variant.
    pub full_attention_variant: bool,
    /// KV pages laid out `[head, page, dim]` rather than `[page, head, dim]`.
    pub hnd_layout: bool,
    /// A causal mask; **`false` is a bidirectional layer** — a ViT, not a decoder.
    pub causal_mask: bool,
    /// A caller-supplied packed mask, supplied at the dispatch.
    pub custom_mask: bool,
    /// This plan will be dispatched through a score-capturing arm.
    pub wants_prefill_score: bool,
}

#[cfg(feature = "_cuda")]
impl Default for PrefillPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "_cuda")]
impl Drop for PrefillPlan {
    fn drop(&mut self) {
        // SAFETY: as `DecodePlan::drop`.
        drop(unsafe { Box::from_raw(self.cache) });
    }
}

/// The scalar facts a dispatch arm reads beside its bound operands; rows and
/// widths are on the launch. Integer fields are `i32` by convention.
#[cfg(feature = "_cuda")]
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
    /// PER-LAYER rope theta; empty means uniform [`Self::rope_theta`].
    pub rope_theta_by_layer: Vec<f32>,
    /// PER-LAYER partial-rotary width, read only when an op states none.
    pub rotary_by_layer: Vec<u32>,
    /// Head width, for the table fill.
    pub head_dim: i32,
    /// Query heads — a fire-wide geometry fact, not derived from a width.
    pub num_q_heads: i32,
    /// See [`Self::num_q_heads`].
    pub num_kv_heads: i32,
    /// Vocabulary rows the embed weight holds.
    pub vocab: i32,
    /// The packed gate‖up order `chunked_swiglu` was bound with.
    pub gate_second: bool,
    /// GPT-J adjacent-pair rotation (`rope_interleave`), vs NeoX half/half.
    pub rope_interleaved: bool,
    /// The fire's token ids (device i32, one per row) — the embed's input.
    pub token_ids: *mut c_void,
    /// The fire's positions (device i32, one per row) — the rope table's input.
    pub positions: *mut c_void,
    /// gemma's FINAL logit softcap, `cap * tanh(x / cap)` over the logits.
    pub final_logit_softcap: f32,
    /// gemma-4's per-layer embedding width (`ple_dim`); zero without a PLE.
    pub ple_dim: i32,
    /// The constants a `scale.<name>` slot names — a scale is not a tensor.
    pub scales: std::collections::BTreeMap<String, f32>,
    /// The sigmoid router's `norm_topk_prob`.
    pub moe_norm_topk: bool,
    /// The router's `routed_scaling_factor`; 1.0 when unstated.
    pub moe_routed_scaling: f32,
    /// YaRN `(factor, beta_fast, beta_slow, attention_factor)`; zero if plain.
    pub yarn: [f32; 4],
    /// `original_max_position_embeddings` for the YaRN rope.
    pub yarn_original_max: i32,
    /// gpt-oss's clamped GLU limit — a config value, hence its own kernel.
    pub glu_limit: f32,
    /// The clamped GLU's alpha.
    pub glu_alpha: f32,
    /// SiTU's `beta`: a tanh-gated activation, not a swiglu variant.
    pub situ_beta: f32,
    /// SiTU's linear beta.
    pub situ_linear_beta: f32,
    /// The WNA16 experts' quantisation group size.
    pub wna16_group_size: i32,
    /// Experts per token; `0` when none or the routed launches disagree.
    pub experts_per_token: i32,
    /// gemma3n's AltUp stream count `K`; a `[K, tokens, hidden]` states no width.
    pub altup_streams: i32,
    /// The active stream's index.
    pub altup_active: i32,
    /// Per-layer `std_mult` for `gaussian_topk`; empty means no sparse layer.
    pub altup_std_mult_by_layer: Vec<f32>,
    /// The fire's PEEL WINDOW, device-resident `[start, count]`, or null when
    /// unsplit. `_devwin` launches early-out against it, so a capture replays.
    pub peel_window: *const u32,
    /// The fire's FULL row count: the lane space a `_devwin` launch spans.
    pub rows_total: i32,
    /// The fire's STAGED LoRA state and xAᵀ scratch; `None` without adapters.
    /// Raw because the state outlives the fire and the ctx has no lifetime.
    pub lora: Option<(*const crate::fire::lora::LoraFireState, *mut c_void)>,
    /// The six device pointer arrays `moe::build_moe_ptrs_aligned_bf16` carves
    /// for the grouped GEMMs; `None` outside a MoE layer. A `Cell` because it
    /// is filled after construction, in issue order, on one thread.
    pub moe_ptrs: std::cell::Cell<Option<crate::fire::moe_ptrs::Arrays>>,
    /// WHICH ROWS this fire samples, device-resident `[sampled_rows]` i32, or
    /// null when it samples every row (the decode case). On the ctx because
    /// the launcher takes it loose and no text can state it.
    pub sampling_indices: *const i32,
    /// How many rows that is.
    pub sampled_rows: i32,
}

#[cfg(feature = "_cuda")]
impl DispatchCtx {
    /// The theta a layer-tagged rope launch fires with.
    pub(crate) fn theta(&self, layer: usize) -> f32 {
        self.rope_theta_by_layer.get(layer).copied().unwrap_or(self.rope_theta)
    }

    /// gemma3n's per-layer `std_mult` for `gaussian_topk`. Zero where no sparse
    /// layer is stated, which the kernel reads as "keep everything".
    #[expect(dead_code, reason = "the FLOOR half of an unbound row; see bind::arms::mlp")]
    pub(crate) fn altup_std_mult(&self, layer: usize) -> f32 {
        self.altup_std_mult_by_layer.get(layer).copied().unwrap_or(0.0)
    }
}

/// The fire's attention context: the planned caches, the workspace, the
/// per-layer KV views and the device CSRs. The ENGINE's half, assembled once.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone)]
pub struct AttnCtx {
    /// The planned [`DecodePlan`]'s handle. Null on a pure-prefill fire.
    pub decode_plan: *mut c_void,
    /// The FULL-attention layers' decode plan, for families whose layer kinds
    /// disagree on head dim. Null elsewhere; picked when the window says FULL.
    pub decode_plan_full: *mut c_void,
    /// The planned [`PrefillPlan`]'s handle. Null on a pure-decode fire.
    pub prefill_plan: *mut c_void,
    /// The workspace, as launchers take it — the DECODE plans'.
    pub workspace: crate::bind::abi::AttentionWorkspaceView,
    /// The PREFILL plan's own workspace. A plan writes its schedule into the
    /// workspace it was raised against, so sharing one is clobbering it.
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
    /// HOST qo indptr, read by the planless prefill dispatch; else null.
    pub qo_indptr_h: *const u32,
    /// HOST kv page indptr, the planless dispatch's other host read.
    pub kv_page_indptr_h: *const u32,
    /// Requests in the fire (`indptr.len() - 1`).
    pub num_requests: i32,
    /// Pages the fire's CSR names — what the dequant staging walks.
    pub num_pages_in_batch: i32,
    /// The widest single request's page count — NOT the batch total: XQA's
    /// `maxNbPagesPerSeq` is a page-table row STRIDE. Host-computed.
    pub max_pages_per_request: i32,
    /// `write_kv_to_pages`'s first-token scalar (the fire's write origin).
    pub first_token: i32,
    /// Per-row target page for this fire's KV append.
    pub w_page_d: *const u32,
    /// Per-row offset-in-page for the append.
    pub w_off_d: *const u32,
    /// Per-row validity for the append.
    pub row_valid_d: *const u8,
    /// The observed-query pin the fused qkv writes and the dispatch reads.
    pub q_out: *mut c_void,
    /// The folded attention SCORES a `WantsAttnScore` fire captures, with the
    /// CSR of each request's rows. ARENA-STABLE: one captured exec serves fires
    /// that want scores and fires that do not.
    pub score_out: *mut f32,
    /// The FOLDED rows' base: one sink would fold over the rows it folds.
    pub folded_out: *mut f32,
    /// See [`Self::score_out`].
    pub score_indptr_d: *const i32,
    /// The custom attention mask, one byte per `(q, kv)`. Published on every
    /// fire because `HasCustomMask` is folded; resident form is plain causal.
    pub mask_d: *const u8,
    /// See [`Self::mask_d`].
    pub mask_indptr_d: *const i32,
    /// The attention output slot the o_proj reads; one slot, reused per layer.
    pub o_out: *mut c_void,
    /// LSE scratch the decode dispatch writes.
    pub lse_out_d: *mut f32,
    /// The OBSERVATION window the score sink keeps, parsed once and carried.
    pub score_window: u32,
    /// Sliding-window extent, `-1` for none.
    pub window_left: i32,
    /// PER-LAYER window extents; empty means uniform [`Self::window_left`].
    pub window_left_by_layer: Vec<i32>,
    /// Logit soft cap, `0` for none.
    pub logits_soft_cap: f32,
    /// The attention scale (`1/sqrt(head_dim)` unless overridden).
    pub sm_scale: f32,
}

/// The fire's GDN context: the per-layer conv/recurrent state slabs, the
/// request→slot indirection and the head geometry, assembled once per fire.
#[cfg(feature = "_cuda")]
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
    /// mamba's B/C group count; zero on GDN. On a MAMBA fire `v_h`/`v_d`/`k_d`
    /// read as heads/head_dim/state, so `v_h·k_d·v_d` IS mamba's slab.
    pub n_groups: i32,
    /// Device base of each MODEL layer's conv-state slab (slot 0); else zero.
    pub conv_state: Vec<u64>,
    /// Elements per conv slot (`conv_k * conv_dim`).
    pub conv_stride_elems: i64,
    /// Device base of each recurrent-state slab (slot 0), in the store's dtype.
    pub recurrent_state: Vec<u64>,
    /// Elements per recurrent slot.
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state. True for every class that exists.
    pub write_state: bool,
}

/// Why a bound launch could not be dispatched.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchRefusal {
    /// No arm exists for this kernel yet.
    NoArm(String),
    /// The op join names a weight the resolver does not hold.
    UnknownWeight(String),
    /// The arm expected the op join to name a weight and it named none.
    NoWeight(String),
    /// An attention arm ran without an [`AttnCtx`] that covers its layer.
    NoAttnCtx(String),
    /// A GDN arm ran without a [`GdnCtx`] that covers its state layer.
    NoGdnCtx(String),
    /// An output placement failed to resolve.
    Out(String),
    /// The launcher this arm calls would DECLINE this shape — a silent early
    /// return, made a refusal because smoothly wrong is not allowed.
    ShapeDeclined {
        /// The kernel whose launcher declines.
        kernel: String,
        /// Why, in the launcher's own terms.
        why: String,
    },
    /// The arm and the lowering disagree about the operand count.
    ArgCount {
        /// The kernel whose arm refused.
        kernel: String,
        /// Operands the arm expects.
        expected: usize,
        /// Operands the launch bound.
        got: usize,
    },
}

// The FA2 dispatch arms are driver ops: each owns a mutable device-side plan
// cache a `Cx` must never hand over, so a failed guard here is a refusal.

/// The plan pointer a launch is handed; decode's full plan on a `-1` window.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn attn_plan(a: &AttnCtx, spec: &LaunchSpec, layer: u32, family: &str) -> *mut c_void {
    match family {
        "decode" => {
            if window_of(spec, a, layer) == -1 && !a.decode_plan_full.is_null() {
                a.decode_plan_full
            } else {
                a.decode_plan
            }
        }
        _ => a.prefill_plan,
    }
}

// MLA's absorb pair stays hand-written: both take four extents as `Param`,
// which a bind cannot carry, and both need `ctx.cublas`, which no `Cx` may
// hand over. `In(0)` is `args[0]`, `Out(0)` `args[n_in]`, `Weight(0)` next.

/// The absorb pair's shared resolution; `call` is the only difference.
#[cfg(feature = "_cuda")]
#[allow(clippy::type_complexity)]
fn mla_absorb(
    kernel: &str,
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    rows: i32,
    call: fn(
        &kernels_cuda::jit::Ctx,
        kernels::routine::In<0, c_void>,
        kernels::routine::Bank<0, c_void>,
        kernels::routine::Out<0, c_void>,
        kernels::routine::Env<kernels::keys::Rows>,
        i32,
        i32,
        i32,
        i32,
    ) -> Result<(), kernels_cuda::Refusal>,
) -> Result<(), DispatchRefusal> {
    if spec.n_in < 1 || spec.n_out < 1 || b.args.len() < spec.n_in + spec.n_out + 1 {
        return Err(DispatchRefusal::ArgCount {
            kernel: kernel.to_string(),
            expected: spec.n_in + spec.n_out + 1,
            got: b.args.len(),
        });
    }
    if spec.params.len() < 4 {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: format!(
                "the statement carries {} params and both absorbs need four -- heads, \
                 qk_nope_dim, v_head_dim, kv_lora_rank. They ride the param channel \
                 because each absorb takes the WHOLE `kv_b_proj` bank and slices it \
                 itself, so the head pitch is not any operand's extent",
                spec.params.len()
            ),
        });
    }
    if !spec.aux.is_empty() || spec.per_head_dim.is_some() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: "the op join carries an aux value or a per-head reading, and a strided \
                  GEMM over the head axis has neither -- a fact about the STATEMENT that \
                  changes the arithmetic rather than the operands"
                .to_string(),
        });
    }
    let p = |i: usize| i32::try_from(spec.params[i]).unwrap_or(i32::MAX);
    // SAFETY: `ctx.stream` is this fire's and outlives the launch, and
    // `ctx.cublas` is the engine's handle with that same stream bound.
    let cx = unsafe {
        kernels_cuda::jit::Ctx::on(ctx.stream).with_cublas(ctx.cublas)
    };
    // The arg indices must agree with `call`'s signature and nothing checks it:
    // `args[0]` `In<0>`, `args[n_in]` `Out<0>`, `args[n_in + n_out]` `Bank<0>`.
    let fired = call(
        &cx,
        kernels::routine::In { ptr: b.args[0].ptr.cast_const(), rows: 0, width: 0 },
        kernels::routine::Bank { ptr: b.args[spec.n_in + spec.n_out].ptr.cast_const() },
        kernels::routine::Out { ptr: b.args[spec.n_in].ptr, rows: 0, width: 0 },
        // The only trailing scalar with a distinct type; the four i32s permute.
        <kernels::keys::Rows as kernels::keys::Fact>::env(rows),
        p(0),
        p(1),
        p(2),
        p(3),
    );
    // Nothing to launch is an ANSWER, so `Refusal::Empty` is `Ok` here.
    match fired {
        Ok(()) | Err(kernels_cuda::Refusal::Empty { .. }) => Ok(()),
        Err(why) => Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: format!("{why:?}"),
        }),
    }
}

/// Dispatch one bound launch through its `pie_k_*` entry.
///
/// Operand order inside each arm is the trace's stated order (inputs, then
/// outputs, then weights), which only the numeric smoke verifies: a swapped
/// operand is wrong VALUES, not a type error.
///
/// # Errors
///
/// See [`DispatchRefusal`].
#[cfg(feature = "_cuda")]
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
    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("rows fit i32");

    // `Route::Rows` falls through to the `other` arm, which refuses by name.

    // The route was resolved at model load; no symbol string is compared here,
    // and a refusal in this match is the final answer, never a fallthrough.
    match spec.route {
        arms::Route::Bound(entry) => {
            // Resolved before the `Cx` exists: a `Resolver` is `&mut` and a
            // bind body must not be able to make it answer twice differently.
            let w_named: *const c_void = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(n))
                .unwrap_or(core::ptr::null());
            let w_named2: *const c_void = spec
                .weight2
                .as_deref()
                .and_then(|n| resolver.weight(n))
                .unwrap_or(core::ptr::null());
            // The suffix reach, resolved once for the reason `w_named` is. A
            // fourth suffix must be added here by hand; nothing connects it.
            let bank = spec.weight.as_deref();
            let mut suffixed = |suffix: &str| -> *const c_void {
                bank.and_then(|b| resolver.weight(&format!("{b}{suffix}")))
                    .unwrap_or(core::ptr::null())
            };
            let w_suffixed: [(&'static str, *const c_void); 4] = [
                ("_scales", suffixed("_scales")),
                ("_gate_bias", suffixed("_gate_bias")),
                ("_up_bias", suffixed("_up_bias")),
                // Without this entry gpt-oss' routed down projection fires
                // biasless over a bias its checkpoint ships.
                ("_bias", suffixed("_bias")),
            ];
            let fire = facts::Fire {
                bound,
                spec,
                ctx,
                attn,
                gdn,
                rows,
                w_named,
                w_named2,
                w_suffixed: &w_suffixed,
            };
            return entry
                .call(&cx::Cx::new(&fire), ctx.stream)
                .map_err(|r| DispatchRefusal::NoArm(format!("{}: {r}", bound.kernel)));
        }
        // Both already refused at load; reaching one is a driver bug.
        arms::Route::Unbound(why) => {
            return Err(DispatchRefusal::NoArm(format!(
                "{}: {why} (load-time refusal; the lowering was fired without \
                 DispatchPlan::unfireable being checked)",
                bound.kernel
            )));
        }
        arms::Route::Unknown => {
            return Err(DispatchRefusal::NoArm(format!(
                "{}: no contract and no row declares it (load-time refusal; \
                 the lowering was fired without DispatchPlan::unfireable \
                 being checked)",
                bound.kernel
            )));
        }
        // The driver's own ops and the row world: both fall through below.
        arms::Route::Driver | arms::Route::Rows => {}
    }

    // The GDN arms' shared read: the ctx itself.
    let _gdn_ctx = || -> Result<&GdnCtx, DispatchRefusal> {
        gdn.ok_or_else(|| DispatchRefusal::NoGdnCtx(bound.kernel.to_string()))
    };

    // The join's placements must window with the args, or a launch reads its
    // input at the window and writes its output at the base.
    let win = if bound.kernel.ends_with("_devwin") { 0 } else { bound.rows.start };

    // The spec's FOREIGN values (`LaunchSpec::aux`), resolved like the outs.
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

    // THE DRIVER-OP TABLE: arms needing what a query-only `Cx` must never
    // hand over — `ctx.cublas`, a lifetime, or a boxed mutable plan cache.
    match bound.kernel {
        // Every arm resolves its refusals before its one launch.

        // MLA's absorb pair: two crossed rows, one resolution.
        "gemm::mla_absorb_q_to_latent_bf16" => mla_absorb(
            "gemm::mla_absorb_q_to_latent_bf16",
            bound,
            spec,
            ctx,
            rows,
            kernels_cuda::gemm::mla_absorb_q_to_latent_bf16,
        )?,
        "gemm::mla_absorb_latent_to_v_bf16" => mla_absorb(
            "gemm::mla_absorb_latent_to_v_bf16",
            bound,
            spec,
            ctx,
            rows,
            kernels_cuda::gemm::mla_absorb_latent_to_v_bf16,
        )?,

        // A driver op for a LIFETIME, not a device API: the six pointer arrays
        // it fills have no stated consumer, so liveness would free them early.
        "moe::build_moe_ptrs_aligned_bf16" => {
            need(7)?;
            let (expert_ids, aligned_in) = (bound.args[0], bound.args[1]);
            let stage = crate::fire::moe_ptrs::Stage {
                gate_up: bound.args[2].ptr,
                act: bound.args[3].ptr,
                out: bound.args[4].ptr,
            };
            // The shared pair is null on this leg, not a gap: the shared
            // expert runs as its own dense `gemm::act_x_w` pair.
            let banks = crate::fire::moe_ptrs::Banks {
                gate_up: bound.args[5].ptr.cast_const(),
                down: bound.args[6].ptr.cast_const(),
                shared_gate_up: std::ptr::null(),
                shared_down: std::ptr::null(),
            };
            // `BLOCK` must match `MOE_ALIGNED_BLOCK` and the 16 the grouped GEMM
            // requires of `M`; `max_blocks` is derived, so it cannot drift.
            const BLOCK: u32 = 16;
            let aligned_rows = bound.rows.end - bound.rows.start;
            if aligned_rows == 0 || aligned_rows % BLOCK != 0 {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!(
                        "the aligned rectangle is {aligned_rows} rows, which is not a whole \
                         number of {BLOCK}-row blocks -- the padded batch and the block size \
                         disagree, and the block count is the one extent no \
                         width on an operand can carry"
                    ),
                });
            }
            let max_blocks = i32::try_from(aligned_rows / BLOCK).expect("block count fits i32");
            let bounds = crate::fire::moe_ptrs::Bounds {
                max_blocks,
                block_size: i32::try_from(BLOCK).expect("16"),
                hidden: i32::try_from(bound.args[4].width).expect("hidden fits i32"),
                moe_intermediate: i32::try_from(bound.args[3].width)
                    .expect("moe_intermediate fits i32"),
                routed_blocks: max_blocks,
            };
            // SAFETY: every pointer is a bound arg of this launch, live on
            // `ctx.stream` for the fire.
            let built = unsafe {
                crate::fire::moe_ptrs::build_for_fire(
                    expert_ids.ptr.cast_const(),
                    aligned_in.ptr.cast_const(),
                    banks,
                    stage,
                    bounds,
                    ctx.stream,
                )
            };
            match built {
                // The only write to `ctx.moe_ptrs`; the GEMM arm below reads it.
                crate::fire::moe_ptrs::Built::Ready(arrays) => ctx.moe_ptrs.set(Some(arrays)),
                crate::fire::moe_ptrs::Built::Declined(why) => {
                    // A decline here must be an error and not a quiet return:
                    // nothing after this op can run without the arrays.
                    return Err(DispatchRefusal::ShapeDeclined {
                        kernel: bound.kernel.to_string(),
                        why: format!("{why:?}"),
                    });
                }
            }
        }
        // One symbol, two implementations (grouped vs batched-cuBLAS), picked by
        // `fire::moe_grouped`. `<` not `need(n)`: the weight may not ride the run.
        "moe::moe_grouped_gemm_bf16" => {
            let stated = spec.n_in + spec.n_out;
            if bound.args.len() < stated {
                return Err(DispatchRefusal::ArgCount {
                    kernel: bound.kernel.to_string(),
                    expected: stated,
                    got: bound.args.len(),
                });
            }
            // `block_size` is param 0 and `max_blocks` param 1 — the two
            // numbers the operands carry only the PRODUCT of.
            let param = |i: usize| -> i32 {
                spec.params.get(i).and_then(|v| i32::try_from(*v).ok()).unwrap_or(0)
            };
            let bank = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(n))
                .unwrap_or(core::ptr::null());
            let out = bound.args[spec.n_in];
            // SAFETY: every pointer is a bound arg or resolved weight of this
            // launch, live on `ctx.stream`; the arrays are this fire's build.
            let fired = unsafe {
                crate::fire::moe_grouped::grouped_gemm_bf16(
                    ctx.cublas,
                    ctx.moe_ptrs.get(),
                    bound.args[0].ptr.cast_const(),
                    bank,
                    out.ptr,
                    bound.args[1].ptr.cast_const(),
                    param(1),
                    param(0),
                    i32::try_from(out.width).expect("N fits i32"),
                    i32::try_from(bound.args[0].width).expect("K fits i32"),
                    ctx.stream,
                )
            };
            if let Err(why) = fired {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!("{why:?}"),
                });
            }
        }
        // THE TENSOR-PARALLEL PAIR. Driver ops for a HANDLE that a resolved
        // plane can no longer stand in for: `plane_for` reads this launch's
        // input and the stream's capture state to pick a `RankData` slot, and
        // `note_graph_buffer` needs `&mut`. A null `car` is a routing answer —
        // the ABI form declines `NoInstance` and the launcher falls back.
        "comm::all_reduce_bf16" => {
            need(2)?;
            let count = usize::try_from(rows).unwrap_or(0)
                * usize::try_from(bound.args[1].width).unwrap_or(0);
            // SAFETY: `resident_car` answers this thread's published handle or
            // null; the two operands are this launch's, live on `ctx.stream`.
            let fired = unsafe {
                crate::fire::all_reduce::all_reduce_bf16(
                    crate::fire::all_reduce::resident_car(),
                    bound.args[0].ptr.cast_const(),
                    bound.args[1].ptr,
                    count,
                    ctx.stream,
                )
            };
            if let crate::fire::all_reduce::AllReduce::Declined(why) = fired {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!("{:?}", crate::fire::all_reduce::refusal_for(&why)),
                });
            }
        }
        // `in_place = &[(0, 1)]`, so `args[1]` is the residual read and written
        // and `args[2]` is the norm's own result; the gamma rides the weight
        // run at `args[3]`. `hidden` is the FULL width and not this rank's
        // shard: every rank holds a partial sum of the whole vector.
        "comm::all_reduce_residual_rmsnorm_bf16" => {
            need(4)?;
            let hidden = i32::try_from(bound.args[2].width).unwrap_or(0);
            // SAFETY: as the pair above; the four operands are this launch's.
            let fired = unsafe {
                crate::fire::all_reduce::all_reduce_residual_rmsnorm_bf16(
                    crate::fire::all_reduce::resident_car(),
                    bound.args[0].ptr.cast_const(),
                    bound.args[1].ptr,
                    bound.args[3].ptr.cast_const(),
                    bound.args[2].ptr,
                    rows,
                    hidden,
                    ctx.eps,
                    ctx.stream,
                )
            };
            if let crate::fire::all_reduce::AllReduce::Declined(why) = fired {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!("{:?}", crate::fire::all_reduce::refusal_for(&why)),
                });
            }
        }
        "gemm::lora_qkv_correction" => {
            need(2)?;
            // No adapters staged is an ANSWER: a union capture must record this
            // arm with its predicate false, and staged fires bucket elsewhere.
            let Some((state, scratch)) = ctx.lora else {
                return Ok(());
            };
            let x = aux_slot(0, resolver)?;
            let (q, v) = (bound.args[0], bound.args[1]);
            // SAFETY: stream and cuBLAS handle are the fire's and live across
            // the launches; `state` is the `LoraFireState` this fire staged.
            let fired = unsafe {
                let jit = kernels_cuda::jit::Ctx::on(ctx.stream).with_cublas(ctx.cublas);
                kernels_cuda::gemm::lora_qkv_correction(
                    &jit,
                    (*state).staged(),
                    i32::from(bound.layers.start),
                    x.ptr.cast_const(),
                    i32::try_from(x.width).expect("hidden"),
                    i32::try_from(q.width).expect("q width"),
                    i32::try_from(v.width).expect("v width"),
                    q.ptr,
                    v.ptr,
                    scratch,
                )
            };
            if let Err(why) = fired {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!("{why:?}"),
                });
            }
        }
        other => {
            // Names the registry asked: "NoArm" alone sent readers to `x/`.
            let registry = if spec.route.is_row_world() {
                "no generated arm and no driver-op arm"
            } else {
                "a driver op with no arm"
            };
            return Err(DispatchRefusal::NoArm(format!("{other}: {registry}")));
        }
    }
    Ok(())
}

/// A store-backed [`Resolver`]: the loader fills it, the executor asks it.
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
#[cfg(feature = "_cuda")]
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
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub enum RunRefusalKind {
    /// Binding refused — see [`BindRefusal`].
    Bind(BindRefusal),
    /// Dispatch refused — see [`DispatchRefusal`].
    Dispatch(DispatchRefusal),
}

/// The prepared attention state a fire publishes, BY REGION: a peel's tail
/// serves different requests, so an arm is handed its region's state.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct AttnRegions<'a> {
    /// The fire's own state, for rectangles that span it.
    pub fire: Option<&'a AttnCtx>,
    /// A peel TAIL's state; `None` on a fire with no split.
    pub tail: Option<&'a AttnCtx>,
}

#[cfg(feature = "_cuda")]
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

    /// The state a rectangle executes against, keyed on whether it is WINDOWED.
    #[must_use]
    pub fn of(&self, rows: &std::ops::Range<u32>) -> Option<&'a AttnCtx> {
        if rows.start == 0 { self.fire } else { self.tail.or(self.fire) }
    }
}

/// Execute one fire: bind and dispatch every launch of the lowering, in order.
///
/// # Errors
///
/// The first refusing launch, with its index and kernel.
#[cfg(feature = "_cuda")]
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
        dispatch(&bound, dplan.spec(i), frame, resolver, ctx, attn.of(&launch.rows), gdn).map_err(
            |e| RunRefusal { launch: i, kernel: kernel(), why: RunRefusalKind::Dispatch(e) },
        )?;
    }
    Ok(lowered.launches.len())
}

/// One open conditional on the capture stack, and the arm being captured.
#[cfg(feature = "_cuda")]
struct OpenCond {
    cond: crate::device::Cond,
    /// The tree node ([`Lowered::conds`] index) whose body is open.
    node: u32,
}

/// The path from the root to `cond`, as tree-node indices.
#[cfg(feature = "_cuda")]
fn cond_path(conds: &[model_compiler::lower::CondRegion], cond: u32) -> Vec<u32> {
    let mut path = Vec::new();
    let mut at = cond;
    while at != Launch::NO_COND {
        path.push(at);
        let Some(node) = conds.get(at as usize) else {
            break;
        };
        at = node.parent;
    }
    path.reverse();
    path
}

/// Are these two tree nodes the two arms of ONE conditional? Read off the
/// stated pairing, not derived from `(parent, slot, param)`: a guard repeats
/// per layer, so deriving would pair across layers.
#[cfg(feature = "_cuda")]
fn siblings(conds: &[model_compiler::lower::CondRegion], a: u32, b: u32) -> bool {
    conds.get(a as usize).is_some_and(|x| x.sibling == b)
}

/// Execute one fire INTO A CAPTURE, rebuilding the union lowering's guard
/// tree as conditional graph nodes.
///
/// `lowered` must come from `lower_with(.., GuardMode::Union)`: no variant
/// bit has been decided, and the builder's device predicate word decides them
/// inside the graph. The walk is a stack diff, and `ctx.stream` is OVERWRITTEN
/// per region — an arm issues the same call onto the capturing stream.
///
/// # Errors
///
/// The first refusing launch, or a CUDA refusal from the builder.
#[cfg(feature = "_cuda")]
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

    // A CUDA refusal is reported against the launch that provoked it.
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
        // Past the agreement may be the OTHER ARM: a body switch, not a close.
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
            // Always with_else: the sibling arm may arrive later.
            let cond = builder.open_cond(region.slot, true).map_err(|e| cuda(i, &kernel, e))?;
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
            .map_err(|e| RunRefusal {
                launch: i,
                kernel: kernel.clone(),
                why: RunRefusalKind::Dispatch(e),
            })?;
        // Retain the node by INDEX, so a same-topology fire differing only in
        // row count updates instead of recapturing; a no-kernel dispatch gaps.
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
#[cfg(feature = "_cuda")]
fn arm_body(
    cond: &crate::device::Cond,
    conds: &[model_compiler::lower::CondRegion],
    node: u32,
) -> cudarc::runtime::sys::cudaGraph_t {
    let on_true = conds.get(node as usize).is_none_or(|r| r.on_true);
    if on_true { cond.if_body() } else { cond.else_body().unwrap_or_else(|| cond.if_body()) }
}

/// Resolve one [`Arg`]: the three rules, shared by [`bind`] and by the arms
/// that resolve an op's OUTPUT placements from the join.
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

/// [`resolve_arg`], addressing from row `row` rather than the operand's base.
/// `row` is in the LAUNCH's row space; the stride is the operand's own.
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
                return Err(BindRefusal::ArenaOutOfBounds { at, arena_bytes: frame.arena_bytes });
            }
            BoundArg { ptr: unsafe { frame.arena.cast::<u8>().add(at) }.cast(), width: *width }
        }
        Arg::Named { value, width, bytes: _ } => BoundArg {
            ptr: resolver.named(*value).ok_or(BindRefusal::UnknownNamed(*value))?,
            width: *width,
        },
        Arg::Weight(name) => {
            // `scale.` marks a CONSTANT riding the name slot; the value comes
            // from `DispatchCtx::scales`, and the slot binds a sentinel.
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
    // THE WINDOW, applied here and not in the arms: a peel's tail serves rows
    // `[win_start, ..)` of a full-N buffer, and `outs`/`aux` resolve through the
    // same call. `_devwin` forms take BASE pointers, so they are not windowed.
    let row = if kernel.ends_with("_devwin") { 0 } else { launch.rows.start };
    let mut args = Vec::with_capacity(launch.args.len());
    for arg in &lowered.args[launch.args.start as usize..launch.args.end as usize] {
        args.push(resolve_arg_windowed(arg, frame, resolver, row)?);
    }
    Ok(BoundLaunch { kernel, rows: launch.rows.clone(), layers: launch.layers.clone(), args })
}
