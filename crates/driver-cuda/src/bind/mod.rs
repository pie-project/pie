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

/// Tier A: a kernel loaded from a cubin, its arguments marshalled from the
/// row, and `cuLaunchKernel`. No host launcher.
pub mod device;

/// Tier A: the `__global__` templates, compiled at run time and instantiated
/// by the names the rows state. No entry points either.
pub mod nvrtc;

/// Stage B: the device headers an `#include` resolves against, carried in the
/// binary rather than found on a path.
pub mod headers;

/// Stage C: firing a row through the unit it was compiled into at run time,
/// instead of through the generated shim.
pub mod jit;

/// Stage C, the other half: the rows the driver executes ITSELF — cuBLAS
/// through `cudarc`, and the compositions whose steps it issues in order.
/// `kernels_cuda_new::execution::Execution::Service` classified them; this
/// is the consumer that makes the classification cost the C++ its body.
///
/// # WHY THIS WAS `bridge`-GATED, AND WHY IT IS `_cuda` NOW
///
/// It was ungated, and `service.rs:44-49` argues the case at length:
/// `cudarc` `dlopen`s cuBLAS at first use, there is **no `#[link]`, no
/// `build.rs` flag and no header**, so `cargo check` with no CUDA toolkit on
/// PATH still passes — and `build.rs`'s `cargo:rustc-link-lib=cublas` is for
/// the archive's callers, not for this file.
///
/// **Every word of that is true, and it is about LINKING.** The module also
/// names [`DispatchCtx`] twenty-eight times, and `DispatchCtx` had exactly
/// one definition, `#[cfg(feature = "bridge")]`, with no `not(bridge)` arm
/// anywhere in the tree. So without the feature the module compiled and its
/// type did not exist.
///
/// **It checked the `-l` and not the `use`.** `cargo test -p driver-cuda
/// --features cuda-12` could not compile, and nothing went red for it,
/// because `[[test]] gemm_service_parity` requires `bridge` while its subject
/// module did not — so no green job ever compiled this module without the
/// type it needs.
///
/// `f38d199c2` closed that by gating the CONSUMER, and BRIDGE_RETIREMENT.md
/// §2.2 convicts the fix: **`DispatchCtx` is not the archive's.** Thirty-five
/// fields — two handle pointers, scalars, three `Vec`s, a `BTreeMap`, device
/// pointers — and not one type that `launch_bindings.rs` declares. The two
/// halves did have to agree; the gated one was the one lying. So the same
/// break closes the other way, and this module is `_cuda` with its type.
///
/// The earlier claim here — *"this module is the consumer that makes the
/// classification cost the C++ its body, which is `bridge`'s whole subject"*
/// — was a reading of the ATTRIBUTE, not of the body. The body reaches no
/// archive symbol; nothing in `driver-cuda/src` does except `abi::ffi`.
#[cfg(feature = "_cuda")]
pub mod service;

/// Stage C, the quantized half: `gemm/gemm.cpp`'s router on the weight's
/// dtype and the four cuBLASLt recipes behind it, in Rust. Separated from
/// [`service`] because it is machinery — a per-device cuBLASLt context, six
/// growable scratches and a dequantized-weight cache — where `service` is
/// argument assembly. The three entry points stay in `service` because
/// `execution::RUST_SERVED`'s spelling test reads that file.
pub mod quant_gemm;

/// Tier A: the arithmetic a stated [`kernels::LaunchRule`] names — what
/// the C++ launchers computed inside `<<<>>>`.
pub mod launch;

/// The driver's answer to `kernels_cuda_new::x::Facts` — the fire's facts,
/// as a bind body beside a `.cuh` may read them.
///
/// `.wiki/kernel-x/northstar.md` §3.3. The reads are
/// [`dispatch_generated`]'s scaffolding promoted out of a generated file
/// and given a name.
pub mod facts;

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
    /// WHAT WILL FIRE THIS LAUNCH'S SYMBOL, resolved once at model load.
    ///
    /// `.wiki/kernel-x/northstar.md` §5 step 4 — the dispatch flip. The
    /// section writes this as `lowered.kernels: Vec<&'static Entry>`; it
    /// cannot go there — `Lowered` is `model-compiler`'s, and a lowering
    /// that carried an [`Entry`](kernels_cuda_new::x::Entry) would tell a
    /// GPU-free crate exactly which symbols are JIT'd, which is the one
    /// thing §3.4 says `model-compiler` must not be able to see. So the
    /// intern lives on the op join, which is `driver-cuda`'s own
    /// per-lowering pass and is already computed once and indexed per fire.
    ///
    /// It is a [`Route`](kernels_cuda_new::x::Route) and not an
    /// `Option<&Entry>` because the question has four answers and an
    /// `Option` has two — see `Route`'s own doc, where the arity is the
    /// thing that made step 4's second half writable.
    ///
    /// [`dispatch`] reads it. **No symbol string is compared at fire time.**
    pub route: kernels_cuda_new::x::Route,
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
#[cfg(feature = "_cuda")]
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
    routes: Vec<kernels_cuda_new::x::Route>,
    unfireable: Vec<Unfireable>,
}

/// One symbol a lowering names that nothing can fire, and why.
///
/// The load-time half of [`Refusal`](kernels_cuda_new::x::Refusal), which is
/// deliberately the SAME type the fire path refuses with. §5 step 4 asks for
/// "unknown symbols refuse at load"; a second error shape for it would mean
/// the same fact printed two ways depending on when it was noticed.
#[derive(Debug, Clone)]
pub struct Unfireable {
    /// The symbol the lowering states.
    pub symbol: String,
    /// Why nothing can fire it.
    pub why: kernels_cuda_new::x::Refusal,
}

impl core::fmt::Display for Unfireable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.symbol, self.why)
    }
}

/// Resolve every symbol a lowering names to the thing that will fire it.
///
/// **§5 step 4 — the dispatch flip.** One scan of `lowered.kernels`, which is
/// the SYMBOL table (tens of entries), not the launch list (hundreds).
/// Indexed by kernel id, so every launch reads its route off its own spec and
/// no fire compares a symbol string again.
///
/// # What §5 got right and what it did not
///
/// §5 writes this as `lowered.kernels: Vec<&'static Entry>` and is right
/// about the KEY — the kernel index — and wrong about the owner and the
/// arity. Both corrections are in
/// [`Route`](kernels_cuda_new::x::Route)'s doc; the short of it is that
/// `Lowered` belongs to a GPU-free crate that must not learn which symbols
/// are JIT'd, and that `Option<&Entry>` cannot say "unknown", which is the
/// one thing step 4's second half has to be able to say.
///
/// # What `Facts` changes about this, which is nothing
///
/// §5.1 calls step 4 "the step your `Facts` change most affects", and the
/// answer turned out to be that it does not affect it at all — worth stating
/// because a reader expects otherwise. Resolution maps a symbol to static
/// data. `Facts` is constructed **per fire**, because `Fire<'a>` borrows a
/// `BoundLaunch` that does not exist until operands are resolved. So there
/// is no fire fact a load-time resolution could consult even if it wanted
/// one, and `resolve` is a pure function of the symbol table. That is why
/// this landed without touching `Cx` at all.
#[must_use]
pub fn resolve(lowered: &Lowered) -> Vec<kernels_cuda_new::x::Route> {
    lowered
        .kernels
        .iter()
        .map(|symbol| kernels_cuda_new::x::route(symbol))
        .collect()
}

impl DispatchPlan {
    /// Join `lowered`'s launches with the ops that produced them.
    ///
    /// Infallible, deliberately: this is a join, and a join over a lowering
    /// that names an unfireable symbol still joins. The refusal is
    /// [`Self::unfireable`], asked separately by the load path, so that the
    /// eighteen call sites that only want the join keep their shape.
    #[must_use]
    pub fn new(plan: &model_compiler::trace::ForwardPlan, lowered: &Lowered) -> Self {
        use model_compiler::trace::Dim;
        use model_compiler::trace::OpKind;
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
                _ => Arg::Named {
                    value: v,
                    width: width_of(v),
                },
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
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>() + *else_ops as usize;
                for slot in guard_of.iter_mut().skip(g + 1).take(span) {
                    *slot = Some(g);
                }
            }
        }
        // CROSS-STATEMENT WIRING, read off the kernel rows.
        //
        // Some blocks wire values ACROSS statements: nemotron's mamba scan
        // consumes the split's raw `dt` and the params prep's fp32 tables,
        // none of which its own statement carries (the C++ hand pass routed
        // them through its workspace). Both ends of that wiring are now
        // STATED — `KernelSig::publishes_aux` says which output fills which
        // slot, `Source::Aux(i)` says which slot an operand reads — so this
        // collects publishers into per-layer slots and knows no kernel by
        // name. What stood here matched four literal symbols, which made
        // this crate the only place that knew how one architecture's block
        // is put together.
        let sig_of = |launch: &Launch| -> Option<&'static kernels::KernelSig> {
            kernels::sig_in(
                kernels_cuda_new::table::KERNELS,
                &lowered.kernels[launch.kernel as usize],
            )
        };
        // Widest slot any row publishes or reads, so the vector is sized by
        // the table rather than by a constant this file would have to keep
        // in step with it.
        let aux_width = lowered
            .launches
            .iter()
            .filter_map(sig_of)
            .flat_map(|s| {
                s.publishes_aux
                    .iter()
                    .map(|&(slot, _)| slot)
                    .chain(s.operands.iter().filter_map(|o| match o.source {
                        kernels::Source::Aux(i) => Some(i),
                        _ => None,
                    }))
            })
            .max()
            .map_or(0, |m| usize::from(m) + 1);
        let mut aux: std::collections::BTreeMap<u16, Vec<Option<Arg>>> =
            std::collections::BTreeMap::new();
        for launch in &lowered.launches {
            let Some(sig) = sig_of(launch) else { continue };
            if sig.publishes_aux.is_empty() {
                continue;
            }
            let op = &plan.ops[launch.op as usize];
            let slots = aux
                .entry(launch.layers.start)
                .or_insert_with(|| vec![None; aux_width]);
            for &(slot, out_ix) in sig.publishes_aux {
                // Out of range is a trace that does not fit this kernel; the
                // arity guard reports it, and publishing nothing keeps the
                // `all slots filled` test below the one that decides.
                if let Some(&v) = op.outputs.get(usize::from(out_ix))
                    && let Some(cell) = slots.get_mut(usize::from(slot))
                {
                    *cell = Some(out_arg(v));
                }
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
        let mut pair_up: std::collections::BTreeMap<u16, Arg> = std::collections::BTreeMap::new();
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
                pair_up
                    .entry(launch.layers.start)
                    .or_insert_with(|| out_arg(op.outputs[0]));
            }
        }
        let aux_of = |layer: u16| -> Vec<Arg> {
            aux.get(&layer)
                .map(|slots| slots.iter().filter_map(Clone::clone).collect::<Vec<_>>())
                .filter(|v: &Vec<Arg>| v.len() == aux_width)
                .unwrap_or_default()
        };
        // THE DISPATCH FLIP — §5 step 4, resolved here and nowhere else.
        //
        // One scan of the symbol table. `route` lives in `kernels-cuda-new`
        // because all three registries it consults are that crate's; what
        // happens here is the mapping and no more.
        let routes = resolve(lowered);
        // THE LOAD-TIME REFUSALS, earned here.
        //
        // §0: *"every refusal the system can make is made at model load"*.
        // These are the ones that are knowable from the symbol table alone,
        // and they are knowable before a single operand is bound.
        let unfireable: Vec<Unfireable> = lowered
            .kernels
            .iter()
            .zip(&routes)
            .filter_map(|(symbol, route)| {
                route.refusal().map(|why| Unfireable {
                    symbol: symbol.clone(),
                    why,
                })
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
                    | OpKind::CausalConv1d { weight, .. }
                    | OpKind::LmHead { weight } => LaunchSpec {
                        weight: Some(weight.clone()),
                        ..LaunchSpec::default()
                    },
                    OpKind::Matmul {
                        weight, beta_one, ..
                    } => LaunchSpec {
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
                    OpKind::Launch {
                        weights, params, ..
                    } => LaunchSpec {
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
                // A consumer is a row that READS an aux slot, which is what
                // `Source::Aux` says. No kernel is named here.
                if sig_of(launch).is_some_and(|s| {
                    s.operands
                        .iter()
                        .any(|o| matches!(o.source, kernels::Source::Aux(_)))
                }) {
                    spec.aux = aux_of(launch.layers.start);
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
                spec.route = routes[launch.kernel as usize];
                spec
            })
            .collect();
        Self { specs, routes, unfireable }
    }

    /// The spec for launch `i` — index-parallel with
    /// [`Lowered::launches`].
    #[must_use]
    pub fn spec(&self, i: usize) -> &LaunchSpec {
        &self.specs[i]
    }

    /// Every symbol this lowering names that NOTHING can fire, with the
    /// sentence that says why.
    ///
    /// **§5 step 4's payoff**: *"unknown symbols now refuse at load"*. Two
    /// classes reach this list and they are different failures:
    ///
    /// * [`Refusal::Undeclared`] — no contract and no row declares the
    ///   symbol. `model-compiler`'s `check_plan` refuses the same thing
    ///   where the trace is BUILT; this is the driver refusing it where the
    ///   trace is LOADED, which is not the same place and, for a plan that
    ///   arrived over a wire or from an older compiler, not the same
    ///   guarantee.
    /// * [`Refusal::Unstated`] — fn-world declares it and no bind can fire
    ///   it, ever. The row world's version of this symbol had prose beside
    ///   an unsourced operand and refused, silently, at the fire. Here the
    ///   prose IS the diagnostic and it arrives before the first token.
    ///
    /// # What is NOT here, and why the list is honest about it
    ///
    /// A [`Route::Rows`](kernels_cuda_new::x::Route::Rows) symbol whose
    /// generated arm does not exist. Whether `emit_rust_dispatch` wrote an
    /// arm is decided by the row's operands all carrying a `Source`, and
    /// re-deriving that rule here would be writing the emitter's decision a
    /// second time, in a crate that cannot see the emitter's output. Those
    /// still refuse at the fire with `NoArm`, exactly as today. The gap
    /// closes when `Route::Rows` does — see its doc for the mechanical
    /// condition.
    ///
    /// [`Refusal::Undeclared`]: kernels_cuda_new::x::Refusal::Undeclared
    /// [`Refusal::Unstated`]: kernels_cuda_new::x::Refusal::Unstated
    #[must_use]
    pub fn unfireable(&self) -> &[Unfireable] {
        &self.unfireable
    }

    /// How much of this lowering still fires through the row world.
    ///
    /// `(row-world symbols, distinct symbols)`. The §5 step-5 sweep's
    /// progress, readable off any model the driver loads — which is a
    /// better progress report than a census of `.cu` files, because it
    /// counts what a real deployment actually states.
    #[must_use]
    pub fn sweep_progress(&self) -> (usize, usize) {
        (
            self.routes.iter().filter(|r| r.is_row_world()).count(),
            self.routes.len(),
        )
    }
}

/// FlashInfer's decode plan cache, owned in Rust.
///
/// # What this was, and what the handle now points at
///
/// It was a handle to an INCOMPLETE C++ type (`struct DecodePlanCache;`),
/// created by `pie_x_make_decode_plan` and destroyed by
/// `pie_x_destroy_decode_plan` — the hand-written extras, whose own header
/// said the whole reason they existed was *"a `unique_ptr` with a custom
/// deleter"*. North star §5 step 7 deleted them along with
/// `attention_flashinfer.cu`, so the pointee is now
/// [`crate::fire::flashinfer_fa2::DecodePlanCache`], a plain Rust struct, and
/// the custom deleter is [`Drop`].
///
/// # Why it is still a raw pointer and not a `Box` field
///
/// Two reasons, both about what callers already depend on:
///
/// * [`Self::as_ptr`] is `const` and is called from [`AttnCtx`], which builds
///   its plan pointers in a `const`-friendly path. `Box` does not deref in
///   `const fn`.
/// * A `*mut` field keeps `DecodePlan` `!Send` and `!Sync` exactly as it was.
///   The plan caches live behind the serve loop's locks today; making them
///   auto-`Send` here would silently widen what is allowed to hold one.
///
/// The pointer is `Box::into_raw`'d at construction and `Box::from_raw`'d in
/// `drop`, so the ownership is total and there is no path that leaks it.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct DecodePlan {
    cache: *mut crate::fire::flashinfer_fa2::DecodePlanCache,
}

#[cfg(feature = "_cuda")]
impl DecodePlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(Box::new(
                crate::fire::flashinfer_fa2::DecodePlanCache::default(),
            )),
        }
    }

    /// The raw handle a dispatch arm passes as the `DecodePlanCache&`.
    ///
    /// Still `*mut c_void` because that is what `Ty::DecodePlanCache` lowers
    /// to (`kernels/src/lib.rs:1090-1155`) and the generated dispatch has no
    /// vocabulary for a Rust type. `bind::service` casts it back.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's
    /// `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut crate::fire::flashinfer_fa2::DecodePlanCache {
        // SAFETY: `cache` came from `Box::into_raw` in `new`, is never
        // reassigned, and `&mut self` proves no other reference is live.
        unsafe { &mut *self.cache }
    }

    /// Run FlashInfer's decode planner over the fire's HOST page indptr.
    ///
    /// The caller brackets this with the workspace's
    /// `begin_plan_update`/`end_plan_update`, exactly as the C++ does. That
    /// bracket is now conservative rather than load-bearing: the descriptor
    /// stages into the cache's own `Vec<u8>` and no longer touches the view's
    /// pinned slot. It is kept because the fence also orders the workspace's
    /// int buffer against the previous step's readers, which is still true.
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
    ///
    /// # Panics
    ///
    /// If the planner declines, naming the [`Decline`] — which is what the
    /// C++ `throw` became everywhere else in this driver. The refusal stays a
    /// *type* one layer down, where it can be tested without a GPU; this is
    /// the boundary where it stops being one.
    ///
    /// [`Decline`]: crate::fire::flashinfer_fa2::Decline
    // Safe by design like the seam methods: the view's pointers are the
    // workspace's own, and the stream is the caller's live handle.
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
        use crate::fire::flashinfer_fa2 as fa2;

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
            kernels_cuda_new::plan::Workspace {
                float_bytes: workspace.float_bytes,
                int_bytes: workspace.int_bytes,
            },
            &device,
            max_grid_size,
            enable_cuda_graph,
            full_attention_variant,
            // `hnd_layout`. The C++ call site passed `false` and so does this
            // one; `bind` has no HND deployment.
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

/// FlashInfer's prefill plan cache — [`DecodePlan`]'s twin, owned the same
/// way for the same reasons.
#[cfg(feature = "_cuda")]
#[derive(Debug)]
pub struct PrefillPlan {
    cache: *mut crate::fire::flashinfer_fa2::PrefillPlanCache,
}

#[cfg(feature = "_cuda")]
impl PrefillPlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Box::into_raw(Box::new(
                crate::fire::flashinfer_fa2::PrefillPlanCache::default(),
            )),
        }
    }

    /// The raw handle a dispatch arm passes.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache.cast()
    }

    /// Where the plan's int arrays sit inside the workspace's `int_buffer`.
    ///
    /// [`DecodePlan::set_int_base`]'s twin. The C++ had no
    /// `pie_x_set_prefill_plan_int_base` — the prefill plan always sat at
    /// offset zero — so this is new surface rather than a port, and it exists
    /// because the field does: leaving it settable only by the decode side
    /// would be an asymmetry a reader would have to go and check.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.get().set_int_base(bytes);
    }

    fn get(&mut self) -> &mut crate::fire::flashinfer_fa2::PrefillPlanCache {
        // SAFETY: as `DecodePlan::get`.
        unsafe { &mut *self.cache }
    }

    /// The planned cache, for a caller that fires the dispatch itself.
    ///
    /// `tower/qwen3_vl/attn.rs` is that caller and is the reason this is not
    /// private: it holds no [`DispatchCtx`], so `bind::service` cannot serve
    /// it, and it needs the same `&PrefillPlanCache` the service reconstructs
    /// from [`Self::as_ptr`]. Handing it out as a reference rather than making
    /// the tower cast the `*mut c_void` back is the point.
    #[must_use]
    pub fn cache(&self) -> &crate::fire::flashinfer_fa2::PrefillPlanCache {
        // SAFETY: as `DecodePlan::get`; `&self` proves no `&mut` is live.
        unsafe { &*self.cache }
    }

    /// Run FlashInfer's prefill planner over the fire's HOST CSRs.
    ///
    /// Bracket with the workspace's plan-update fence, as with
    /// [`DecodePlan::plan_decode`].
    ///
    /// `kv_last_page_lens_h` is accepted and **not read**. The C++ planner
    /// read it for one purpose — `:325`'s `kv_last_page_lens_h != nullptr`
    /// guard on the SM90 route — and this lattice never plans SM90
    /// (`fire::flashinfer_fa2::plan_prefill` writes `use_sm90 = false`). The
    /// parameter is kept so that every caller still states the CSR it holds,
    /// and so that wiring an SM90 family later is a change here and not at
    /// six call sites.
    ///
    /// # Panics
    ///
    /// If the planner declines. See [`DecodePlan::plan_decode_variant`].
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
            // The five flags the C++ call site passed positionally at
            // `bind/mod.rs`'s old `pie_x_plan_attention_flashinfer_prefill_bf16`
            // call: `full_attention_variant=false`, `hnd_layout=false`,
            // `causal_mask=true`, `custom_mask=false`,
            // `wants_prefill_score=false`.
            PrefillPlanFlags {
                full_attention_variant: false,
                hnd_layout: false,
                causal_mask: true,
                custom_mask: false,
                wants_prefill_score: false,
            },
        );
    }

    /// [`Self::plan_prefill`] with the five variant flags exposed.
    ///
    /// The C++ entry point always took them; `bind`'s wrapper hard-coded a
    /// causal, uncustomised, unscored plan and every non-causal caller had to
    /// reach past it to `ffi::pie_x_plan_attention_flashinfer_prefill_bf16`
    /// directly. `tower/qwen3_vl/attn.rs` was that caller — a ViT is
    /// bidirectional — and with the `pie_x_*` extras deleted it needs a
    /// spelling that is not a back door. This is it.
    ///
    /// # Panics
    ///
    /// If the planner declines. See [`DecodePlan::plan_decode_variant`].
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
        use crate::fire::flashinfer_fa2 as fa2;

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
            kernels_cuda_new::plan::Workspace {
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
/// numbers.
///
/// A struct rather than five positional `bool`s because the C++ signature's
/// tail was `(…, window_left, full_attention_variant, hnd_layout, causal_mask,
/// custom_mask, wants_prefill_score)` and `tower/qwen3_vl/attn.rs:260` had to
/// count six `false`s to say *"bidirectional"*. Miscounting them is a silent
/// wrong answer — `causal_mask` in `hnd_layout`'s slot plans a causal ViT —
/// and named fields are the cheapest thing that makes that a compile error.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct PrefillPlanFlags {
    /// `FullAttention` rather than the sliding-window variant.
    pub full_attention_variant: bool,
    /// KV pages laid out `[head, page, dim]` rather than `[page, head, dim]`.
    pub hnd_layout: bool,
    /// A causal mask. **`false` is a bidirectional layer**, which is what a
    /// ViT wants and what no decoder layer wants.
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

/// The scalar facts a dispatch arm reads beside its bound operands.
///
/// Everything else an arm needs is IN the launch: row counts from
/// `rows`, per-operand widths from the args. What remains is the
/// deployment's constants — the same values the C++ arms read off their
/// facts structs — and the per-fire handles.
///
/// # The integer fields here are `i32` by convention, not by necessity
///
/// A table states an operand's extent as `Div(Width(i), CtxNonZero(f))` and
/// `emit_rust_dispatch` renders it by CONCATENATION — `width_of(b, i)`, which
/// is always `i32`, over `ctx.<f>` — because the emitter *"does not know the
/// field's TYPE"* (`kernels_cuda_new::abi`'s own words, and the reason
/// [`IsSet`] exists rather than a `!= 0`). There is no coercing `Source`
/// variant: the vocabulary is `Lit / Width / Mul / Div / IfPresent / Ctx /
/// CtxNonZero`.
///
/// For a while that made an UNWRITTEN RULE — every integer field a table
/// could divide by had to be `i32` here — and it broke exactly the way an
/// unwritten cross-crate rule breaks. `ple_dim` was widened to `u32` to keep
/// a field shorthand compiling against a `Deployment` that had widened too,
/// and the next table row that divided by it stopped the crate from building
/// with `cannot divide i32 by u32` — in a generated file, on a line no one
/// wrote, three hundred arms away from the change. Two readers diagnosed it
/// as a bad table row; the row was correct.
///
/// THE RULE IS GONE. `kernels_cuda_new::abi::rust_arith_of` narrows a
/// driver-declared field to the grammar's `i32` before composing it, which is
/// the same rule `cast_for` always applied to a whole operand — *"a row
/// declares `I32` and gets an i32, whatever the width of the thing it named"*
/// — now applied at every depth rather than only the top. `score_window` was
/// already `u32` and already worked; the asymmetry WAS the bug.
/// `a_declared_field_under_an_operator_is_narrowed_first` holds it over every
/// table, and widening `ple_dim` back to `u32` was measured to compile.
///
/// So `i32` here is a local choice — it matches `head_dim`, `altup_streams`
/// and the rest, and a divisor reads better signed — and a future field may
/// be whatever the driver finds honest.
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
    ///
    /// `i32` to match `head_dim` and the rest, not because it must be:
    /// two table rows divide a width by it, and the emitter narrows a
    /// declared field before composing it. See the struct's own doc.
    /// `Deployment::ple_dim` is `u32` — that is the model side's call, and
    /// the narrowing is one `try_from` at the literal that joins them.
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
    /// The fire's ROUTED-EXPERT FANOUT: how many experts each token is
    /// dispatched to, agreed across every routed launch in the fire, or `0`
    /// when the fire has none or they disagree.
    ///
    /// # Why the fire and not the statement
    ///
    /// It is a GEOMETRY axis — [`kernels_cuda_new::Dims::experts_per_token`]
    /// — and a grid is opened before an operand is read, so a rule that needs
    /// it needs it as a fact about the run. `dequant_fp4.cu:56` opened
    /// `dim3 grid(num_tokens * top_k, ..)`, `dequant_wna16.cu` the same,
    /// until §43 deleted both launchers as unreached; the kernels still take
    /// their route off `blockIdx.x` (`quant/dequant_fp4.cuh:232`,
    /// `quant/dequant_wna16.cuh:295`), so the reading is unchanged: the
    /// fanout MULTIPLIES the row axis, so getting it wrong is not a wrong
    /// scalar in a correct grid, it is the wrong number of blocks.
    ///
    /// # Why it is derived from the lowered plan and not from `Geometry`
    ///
    /// `model::deployment::Geometry` does not state it — it states hidden,
    /// heads, head width, intermediates and vocab — and this crate does not
    /// own that type. What the fire DOES hold is its own lowered launches,
    /// and the mixture statements state the fanout as a wire param at a
    /// position `dsl` fixes per statement kind. So the value is read from
    /// there, ONCE, at the single construction site, keyed on the kernel
    /// SYMBOL rather than on a bare index — see `fire::launch`'s
    /// `fire_experts_per_token`, whose doc argues why the symbol key is what
    /// makes the wrong reading unspellable.
    ///
    /// # Zero is absence
    ///
    /// A fire with no routed statement, or one whose routed statements
    /// disagree about the fanout, gets `0`, and every rule that reads it
    /// answers `Ungeometric::Empty`. A refusal is not a fallback.
    pub experts_per_token: i32,
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
    /// The six device pointer arrays `moe::build_moe_ptrs_aligned_bf16`
    /// carved for the grouped GEMMs below it.
    ///
    /// # A `Cell`, and the reason is that `lora`'s shape does not transfer
    ///
    /// `lora` is the precedent for *state one arm builds and another
    /// consumes*, and it is filled **at construction** — the plan state holds
    /// the staged adapter before the fire begins. This cannot be: the build
    /// is **step 3 of 8** of the aligned MoE lowering, so the value does not
    /// exist when the ctx is made.
    ///
    /// And every dispatch arm receives `ctx: &DispatchCtx`. **A plain field
    /// would be `None` forever** — provisioned-looking and never written,
    /// which is worse than the thread-local it replaces, because the
    /// thread-local at least worked. So: interior mutability, written through
    /// the shared reference the arms already hold.
    ///
    /// # What it buys over the thread-local
    ///
    /// `fire::moe_ptrs`' `thread_local!` was correct and said so: the build
    /// is step 3, both grouped GEMMs are below it in the same lowering, on
    /// the same thread, in issue order. **What made it a seam is that nothing
    /// stated the lifetime** — the next layer's build overwrote the cell, and
    /// it was only correct because this layer's GEMMs sat between the two.
    ///
    /// Here the lifetime is the ctx's, which is the fire's, and the struct
    /// says so. A second layer gets a second ctx.
    ///
    /// `None` on every fire outside a qwen3.5 MoE layer. Raw pointers inside,
    /// for `lora`'s reason: the arrays live in `fire::moe_ptrs`' per-thread
    /// arena, which outlives the fire on the caller's side, and this struct
    /// carries no lifetime.
    pub moe_ptrs: std::cell::Cell<Option<crate::fire::moe_ptrs::Arrays>>,
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

#[cfg(feature = "_cuda")]
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
        self.altup_std_mult_by_layer
            .get(layer)
            .copied()
            .unwrap_or(0.0)
    }
}

/// The fire's attention context: what the attention arms need beyond
/// args and the op join — the planned cache, the workspace, the per-layer
/// KV views, and the fire's device-resident page CSRs and write
/// descriptors. The ENGINE'S half of a fire, assembled once.
#[cfg(feature = "_cuda")]
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
    /// Whether this fire advances state.
    ///
    /// True for every class that exists. This used to read "the
    /// frozen-verify service classes pass false", and those classes are
    /// GONE — `FireClass` retired `FrozenVerify`, `CommitAdvance` and
    /// `StateOnly` when the driver started accepting `PIE_RS_FLAG_FOLD`,
    /// since a speculative decode writes into a buffer and folds only
    /// the accepted prefix. The field stays because the kernels take it
    /// and a class that needs `false` would arrive through here;
    /// `launch_context_is_stated` argues for the constant by name.
    pub write_state: bool,
}

/// Why a bound launch could not be dispatched.
#[cfg(feature = "_cuda")]
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
    /// same predicate and takes a batched-cuBLAS fallback. **So does this
    /// driver, since the aligned leg was finished**: the symbol is a driver
    /// op and `fire::moe_grouped` asks the predicate and picks. What reaches
    /// this variant now is a shape **neither** implementation serves — an
    /// empty padded batch, or a build that did not run — which is still a
    /// decline spelled as a silent early return one level down.
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
// ── THE ROW WORLD'S INTERPRETER STOOD HERE, AND IS DELETED ──────────────
//
// Three items, 624 lines: `dispatch_generated`, the `Arms` enum that chose
// which generated `match` it ran, and `dispatch_jit_probe`.
//
// `dispatch_generated` was the AOT half of the dispatcher. It took a
// `BoundLaunch` and ran a `match` on `bound.kernel` that `build.rs` emitted
// from the kernel tables — one arm per row that stated where each of its
// operands came from, each arm resolving that row's operands out of the
// `DispatchCtx`/`AttnCtx`/`GdnCtx` and calling the row's `pie_k_*` entry.
// The scaffolding around the `include!` (`width_of`, the operand readers,
// the nested `attn_plan`) was promoted out of the generated file so a
// generated arm read as ordinary Rust.
//
// # It is deleted because `ROW_TABLES` is empty, NOT because the archive went
//
// This is the distinction BRIDGE_RETIREMENT.md's §2 table did not have a
// column for. `emit_rust_dispatch` emits an arm per STATED row;
// `abi::stated()` drops a row with no operands; `table::ROW_TABLES` is `&[]`
// as of `5a789298b`, and every row `table::KERNELS` still holds is a
// `Contract::sig`, which states no operands deliberately — that is the third
// of the three mechanisms by which a row loses its ahead-of-time entry. So
// the generated `match` has no arms, and this function could only return
// `false`. **A dispatcher with no arms is not a smaller dispatcher.**
//
// `Arms` was a two-variant enum, `Whole` and `JitProbe`, selecting between
// `rust_dispatch.rs` and `rust_dispatch_probe.rs` at the one `include!`
// site. Both files are generated from the same empty tables.
//
// `dispatch_jit_probe` was the AOT-vs-JIT parity harness's half of the seam:
// the same rows with every unit-hosted one forced onto its JIT arm, so a row
// could be fired both ways and compared BEFORE routing deleted its shim
// entry. It had no caller — `tests/jit_parity.rs` is a two-line
// PLACEHOLDER-REFACTOR-SCAFFOLD and there is no `[[test]]` stanza for it —
// and the window it existed for is closed: routing is done, every row is
// crossed, and there is no unrouted row left to compare against. Its
// `jit-parity` feature goes with it.
//
// # The homonym test, run on all three names first
//
// §6's own rule, earned by `arm_body`: a name that reads row-world may be a
// homonym, and a classifier seeded with the claim it is checking returns the
// claim. `Arms` is a closed set of six identifier uses, all in this file,
// all at the `include!` or its two callers. `dispatch_generated` is three
// sites — the definition and the two callers deleted here.
// `dispatch_jit_probe` is three, none of them a call. None of the three has
// a second meaning anywhere in the workspace. Contrast `arm_body`,
// `OpenCond` and `cond_path`, which the ledger filed in this same class and
// which interpret `model_compiler::lower::CondRegion` and
// `device::Cond` — arms of a CUDA-GRAPH CONDITIONAL. They are live, they are
// `_cuda`, and `run_captured` is why.

// ---------------------------------------------------------------------------
// THE FA2 DISPATCH ARMS — six driver ops, resolved by hand.
//
// # Why these are hand-written and not generated
//
// Everything above this line is a SERVICE entry point: a function the
// generated shim calls with every operand already resolved. The resolution
// itself lives in `abi::emit_rust_dispatch`'s output — `rust_dispatch.rs`,
// a `match` arm per row, built from the row's `Source`s. For these six that
// arm IS the host program: it picks the plan, tests the fire's pointers,
// widens the `Or`s and calls the shim, which calls back into the function
// above.
//
// `emit_rust_dispatch` dies with `bridge` (north star §5 step 6 half B), and
// `bridge` goes when `ROW_TABLES` empties — which is what `table/attn.rs`'s
// remaining rows are. So these six resolutions have to be written by hand
// WHATEVER order the migration takes them in; writing them at the crossing
// is paying step 6's cost while the information is in front of the reader
// rather than three weeks later when it is not.
//
// # Why a driver op and not a bind
//
// `x/mod.rs`'s discriminator, answered in one line: **`DecodePlanCache` and
// `PrefillPlanCache`.** Boxed, mutable, owned by `bind::DecodePlan` /
// `bind::PrefillPlan`, raised by `Prepare::DecodePlan` / `Prepare::PrefillPlan`
// once per fire and read by every layer's dispatch. A `Cx` that could hand
// one over is §3.3's forbidden surface exactly — a cache a bind body could
// re-plan, mid-fire, from inside what is supposed to be a query.
//
// So the six take `x::SIGS`' third registration shape: a `contract!` in
// `x::attn` with **no `Entry` at all**, not even a `none:` arm — which would
// mint `Route::Unbound` and refuse the model at load — plus an entry in
// `execution::SERVICE` as `Service::DriverOp`, which is what `x::route` reads
// to answer `Route::Driver`.
//
// # What each arm reproduces
//
// Every guard the generated branch carried becomes a refusal here, and the
// order is the generated one because it is load-bearing: `attn.is_some()`
// before anything reaching `AttnCtx`, the plan's null test after it. What was
// a guard that made the branch NOT MATCH — falling through to whatever else
// serves the statement — is a refusal here, because nothing else serves these
// symbols: the row named one launcher and the statement named the row.
//
// The one guard with no counterpart is `spec.aux.is_empty() &&
// spec.per_head_dim.is_none()`. Those are join facts no `Source` can name
// (`bind/mod.rs`'s note above the generated `match`), and a statement that
// carries one is asking for a reading this launcher does not have. Refused
// with the fact named rather than ignored.
//
// **Every refusal is resolved before the first launch**, which is free here:
// each arm is one call.
// ---------------------------------------------------------------------------

/// `Source::AttnPlan`'s rule, on the side of the seam that survives `bridge`.
///
/// # Why it moved here
///
/// It was a nested `fn` inside `bind::dispatch_generated`, reachable only
/// from the generated `match` in the same body — so a hand arm could not call
/// it and would have had to copy it. **It is a DECISION and not a read**:
/// two-kind families keep a second decode plan for their full-attention
/// layers because the two kinds disagree on head dim (gemma-4: 512 vs 256),
/// and the fact it turns on — `window_left_by_layer[l] == -1` — is the
/// driver's and no row can see it. A copied decision is the shape this port
/// has already found twice in `attn` alone, so the rule moves to the surviving
/// side and `dispatch_generated`'s nested `attn_plan` forwards to it.
///
/// Prefill keeps ONE plan. Spelled as a family anyway so a caller never names
/// a field, which is the property that let decode grow a second plan without
/// touching a row.
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

/// What all six arms resolve the same way.
#[cfg(feature = "_cuda")]
struct Fa2Common<'a> {
    a: &'a AttnCtx,
    kv: crate::bind::abi::KvCacheLayerView,
    layer: u32,
    q: *const c_void,
    /// `Source::Or(&Out(0), &Attn("o_out"))` — the stated result if the
    /// statement declares one, the guard-owned arena slot if it does not.
    o: *mut c_void,
}

/// `n_in >= 1 && attn.is_some() && has_kv_layer(..)`, plus the two join
/// facts and the `o` widening, in the generated guard's order.
#[cfg(feature = "_cuda")]
fn fa2_common<'a>(
    kernel: &str,
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    attn: Option<&'a AttnCtx>,
) -> Result<Fa2Common<'a>, DispatchRefusal> {
    if spec.n_in < 1 {
        return Err(DispatchRefusal::ArgCount {
            kernel: kernel.to_string(),
            expected: 1,
            got: b.args.len(),
        });
    }
    if !spec.aux.is_empty() || spec.per_head_dim.is_some() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: "the op join carries an aux value or a per-head reading, and this launcher \
                  has neither -- a fact about the STATEMENT that changes the arithmetic \
                  rather than the operands, so binding it anyway would read right and \
                  compute wrong"
                .to_string(),
        });
    }
    let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(kernel.to_string()))?;
    let layer = usize::from(b.layers.start);
    let kv = *a
        .layers
        .get(layer)
        .ok_or_else(|| DispatchRefusal::NoAttnCtx(kernel.to_string()))?;
    let o = if spec.n_out > 0 {
        b.args[spec.n_in].ptr
    } else if a.o_out.is_null() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: "the statement declares no result and the fire published no `o_out` -- \
                  the attention output has nowhere to land"
                .to_string(),
        });
    } else {
        a.o_out
    };
    Ok(Fa2Common {
        a,
        kv,
        layer: u32::from(b.layers.start),
        q: b.args[0].ptr.cast_const(),
        o,
    })
}

/// `Source::Or(&Out(1), &Attn("lse_out_d"))` — the decode pair's LSE.
#[cfg(feature = "_cuda")]
fn fa2_lse_or(
    kernel: &str,
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    a: &AttnCtx,
) -> Result<*mut f32, DispatchRefusal> {
    if spec.n_out > 1 {
        return Ok(b.args[spec.n_in + 1].ptr.cast::<f32>());
    }
    if a.lse_out_d.is_null() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: "the statement declares no second result and the fire published no \
                  `lse_out_d` -- gpt-oss' sink rescale reads that LSE, so a null here is a \
                  launch whose second output nothing owns"
                .to_string(),
        });
    }
    Ok(a.lse_out_d)
}

/// The plan the fire raised, or the refusal that says it raised none.
#[cfg(feature = "_cuda")]
fn fa2_plan(
    kernel: &str,
    spec: &LaunchSpec,
    a: &AttnCtx,
    layer: u32,
    family: &str,
) -> Result<*const c_void, DispatchRefusal> {
    let plan = attn_plan(a, spec, layer, family);
    if plan.is_null() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: format!(
                "the fire raised no {family} plan -- a pure-prefill fire has no decode plan \
                 and a pure-decode fire has no prefill plan, and this statement asked for \
                 the one that is not there"
            ),
        });
    }
    Ok(plan.cast_const())
}

/// `Source::AttnNonZero`'s test, which was a guard and is a refusal here.
#[cfg(feature = "_cuda")]
fn fa2_set<T>(kernel: &str, p: *const T, what: &str) -> Result<(), DispatchRefusal> {
    if p.is_null() {
        return Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: format!(
                "the fire published no {what}, and this launcher's whole difference from \
                 its plain sibling is that it writes one"
            ),
        });
    }
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_decode`.
///
/// # Errors
///
/// Every guard the generated branch carried; see this section's banner.
#[cfg(feature = "_cuda")]
pub fn fa2_decode(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::dispatch_attention_flashinfer_decode";
    let c = fa2_common(K, b, spec, attn)?;
    let plan = fa2_plan(K, spec, c.a, c.layer, "decode")?;
    let lse = fa2_lse_or(K, b, spec, c.a)?;
    // SAFETY: `plan` is `bind::DecodePlan`'s own boxed cache, non-null by the
    // test above; every other pointer is a bound arg or a field of the fire's
    // `AttnCtx`, live on `ctx.stream` for the fire.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_decode(
            plan,
            c.q,
            c.kv,
            c.o,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.workspace,
            ctx.stream,
            window_of(spec, c.a, c.layer),
            c.a.logits_soft_cap,
            c.a.sm_scale,
            lse,
        );
    }
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_decode_capture`.
///
/// # Errors
///
/// [`fa2_decode`]'s, plus a fire that published no score sink -- which was
/// `Source::AttnNonZero`'s guard and is stated here one layer earlier than
/// the launcher's own throw.
#[cfg(feature = "_cuda")]
pub fn fa2_decode_capture(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::dispatch_attention_flashinfer_decode_capture";
    let c = fa2_common(K, b, spec, attn)?;
    let plan = fa2_plan(K, spec, c.a, c.layer, "decode")?;
    let lse = fa2_lse_or(K, b, spec, c.a)?;
    fa2_set(K, c.a.score_out.cast_const(), "score sink")?;
    fa2_set(K, c.a.score_indptr_d, "score index")?;
    // SAFETY: as [`fa2_decode`], plus the two score buffers, non-null by the
    // tests above. `score_out` addresses `score_indptr[batch]` floats.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_decode_capture(
            plan,
            c.q,
            c.kv,
            c.o,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.workspace,
            ctx.stream,
            c.a.score_out,
            c.a.score_indptr_d,
            window_of(spec, c.a, c.layer),
            c.a.logits_soft_cap,
            c.a.sm_scale,
            lse,
        );
    }
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_prefill_bf16`.
///
/// The PAGES loose rather than the view whole, which is what this launcher
/// takes; and `prefill_workspace`, not `workspace` -- a FlashInfer plan writes
/// its schedule into the workspace it was raised against, so a prefill reading
/// the decode plan's is one clobbering the other.
///
/// # Errors
///
/// See this section's banner.
#[cfg(feature = "_cuda")]
pub fn fa2_prefill(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::dispatch_attention_flashinfer_prefill_bf16";
    let c = fa2_common(K, b, spec, attn)?;
    let plan = fa2_plan(K, spec, c.a, c.layer, "prefill")?;
    // SAFETY: as [`fa2_decode`]; the bf16 mirrors are the layer view's own,
    // and `dequant_kv_cache_layer_to_bf16_active` inside the entry point is
    // what makes them current.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_prefill_bf16(
            plan,
            c.q,
            c.kv.k_bf16_pages,
            c.kv.v_bf16_pages,
            c.o,
            c.a.qo_indptr_d,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.prefill_workspace,
            ctx.stream,
            c.a.logits_soft_cap,
            c.a.sm_scale,
            c.a.lse_out_d,
        );
    }
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_prefill_capture_bf16`.
///
/// `window` here is the OBSERVATION window (`AttnCtx::score_window`), not the
/// attention one -- deliberately not `Source::AttnWindow`. The launcher
/// refuses `<= 0` and `window_left` is `-1` on a family that attends the whole
/// context, so the same number reads as "no window" to one and "invalid" to
/// the other.
///
/// # Errors
///
/// See this section's banner.
#[cfg(feature = "_cuda")]
pub fn fa2_prefill_capture(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::dispatch_attention_flashinfer_prefill_capture_bf16";
    let c = fa2_common(K, b, spec, attn)?;
    let plan = fa2_plan(K, spec, c.a, c.layer, "prefill")?;
    fa2_set(K, c.a.score_out.cast_const(), "score sink")?;
    fa2_set(K, c.a.score_indptr_d, "score index")?;
    // SAFETY: as [`fa2_prefill`], plus the score buffers.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_prefill_capture_bf16(
            plan,
            c.q,
            c.kv.k_bf16_pages,
            c.kv.v_bf16_pages,
            c.o,
            c.a.qo_indptr_d,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.prefill_workspace,
            ctx.stream,
            c.a.score_out,
            c.a.folded_out,
            c.a.score_indptr_d,
            i32::try_from(c.a.score_window).unwrap_or(i32::MAX),
            c.a.logits_soft_cap,
            c.a.sm_scale,
            c.a.lse_out_d,
        );
    }
    Ok(())
}

/// `attn::dispatch_attention_flashinfer_prefill_custom`.
///
/// The mask rides the CONTEXT, not the statement, for the reason the score
/// sink does: the predicate is folded, so one exec serves the fire that stages
/// a mask and the fire that does not, and the address recorded now must still
/// be right when it goes true.
///
/// # Errors
///
/// See this section's banner.
#[cfg(feature = "_cuda")]
pub fn fa2_prefill_custom(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::dispatch_attention_flashinfer_prefill_custom";
    let c = fa2_common(K, b, spec, attn)?;
    let plan = fa2_plan(K, spec, c.a, c.layer, "prefill")?;
    fa2_set(K, c.a.mask_d, "custom mask")?;
    fa2_set(K, c.a.mask_indptr_d, "custom mask index")?;
    // SAFETY: as [`fa2_prefill`], plus the mask pair, non-null by the tests
    // above. This launcher takes the layer view whole rather than the pages
    // loose.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_dispatch_attention_flashinfer_prefill_custom(
            plan,
            c.q,
            c.kv,
            c.o,
            c.a.qo_indptr_d,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.mask_d,
            c.a.mask_indptr_d,
            c.a.prefill_workspace,
            ctx.stream,
            c.a.logits_soft_cap,
            c.a.sm_scale,
            c.a.lse_out_d,
        );
    }
    Ok(())
}

/// `attn::attention_flashinfer_prefill` -- the PLANLESS prefill.
///
/// It builds an R-shaped plan on the way in, so it holds no cache and states
/// `Prepare::FireWide`. It is still a driver op, and the resource is the pair
/// `qo_indptr_h` / `kv_page_indptr_h`: **HOST mirrors of the CSR**, which the
/// planner walks on the CPU. No `Cx` query answers a host pointer, and a
/// device value read on the host is a synchronise -- which §0 forbids in a
/// fire.
///
/// `whole`, so `Source::Rows` is the fire's full lane count and `o` is the
/// stated result only -- this row carries no `Or`, because a launcher that
/// cannot be given a row window cannot be handed the guard's arena slot
/// either.
///
/// # Errors
///
/// See this section's banner; plus this one needs a stated result.
#[cfg(feature = "_cuda")]
pub fn fa2_prefill_planless(
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
    rows: i32,
) -> Result<(), DispatchRefusal> {
    const K: &str = "attn::attention_flashinfer_prefill";
    let c = fa2_common(K, b, spec, attn)?;
    if spec.n_out < 1 {
        return Err(DispatchRefusal::ArgCount {
            kernel: K.to_string(),
            expected: spec.n_in + 1,
            got: b.args.len(),
        });
    }
    // `Source::Div(&Width(&In(0)), &KvLayerField("head_dim"))`: the head COUNT,
    // which nobody carries -- the query's width over the cache's head dim.
    // `.max(1)` is the generated divisor's, and it is what keeps a layer view
    // that states no head dim from dividing by zero.
    let head_dim = i32::try_from(c.kv.head_dim).unwrap_or(i32::MAX);
    let q_width = i32::try_from(b.args[0].width).unwrap_or(0);
    let num_q_heads = q_width / head_dim.max(1);
    // SAFETY: as [`fa2_prefill`], plus the two HOST mirrors, which the entry
    // point walks on the CPU to raise its own plan.
    unsafe {
        crate::fire::flashinfer_fa2_dispatch::attn_attention_flashinfer_prefill(
            c.q,
            c.kv,
            b.args[spec.n_in].ptr,
            c.a.qo_indptr_d,
            c.a.kv_page_indices_d,
            c.a.kv_page_indptr_d,
            c.a.kv_last_page_lens_d,
            c.a.qo_indptr_h,
            c.a.kv_page_indptr_h,
            rows,
            c.a.num_requests,
            num_q_heads,
            c.a.workspace,
            ctx.stream,
            window_of(spec, c.a, c.layer),
            c.a.logits_soft_cap,
            c.a.sm_scale,
            c.a.lse_out_d,
        );
    }
    Ok(())
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
// ── MLA'S ABSORB PAIR — TWO cuBLAS DRIVER OPS ───────────────────────────
//
// `gemm::mla_absorb_q_to_latent_bf16` and `gemm::mla_absorb_latent_to_v_bf16`
// were `execution::RUST_SERVED` rows with a generated arm each. Their bodies
// are `kernels_cuda_new::x::attn::mla_absorb_*` now and their rows are gone.
//
// **The discriminator answers in one line for BOTH, and it is the same line:
// `ctx.cublas`.** That is worth recording only because the FA2 six split
// 5-1 on the same test the same evening; run as a set, this group does not
// split. Neither has a second condition to check either -- `Cx` states
// nothing here, because a `cublasHandle_t` is not something `Cx` could ever
// answer.
//
// The two arms below are ONE arm with a different callee, so the resolution
// is written once. Every number comes off `BoundLaunch::args`, `rows` and
// `LaunchSpec::params`, in the generated guard's order, exactly as the
// deleted arms read them: `In(0)` at `args[0]`, `Out(0)` at `args[n_in]`,
// `Weight(0)` at `args[n_in + n_out]` -- inputs, outputs, weights, which is
// what [`BoundLaunch::args`] documents its order to be.

/// The absorb pair's shared resolution: the guard, the three pointers and
/// the four widths.
///
/// `call` is the only difference between the two symbols, and the choice
/// belongs at the arm rather than inside a predicate here -- a boolean
/// parameter that selected the callee would be a second spelling of
/// `bound.kernel`, and two spellings of one fact is what this port keeps
/// finding.
// The gate tracks this helper's ONLY CALLER, `dispatch`, and is not a claim
// about the archive -- Sec 78's three classes, and this is the second:
// `dispatch` dies with the ROW WORLD. When that gate goes, so does this one.
#[cfg(feature = "_cuda")]
#[allow(clippy::type_complexity)]
fn mla_absorb(
    kernel: &str,
    b: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    ctx: &DispatchCtx,
    rows: i32,
    call: unsafe fn(
        *mut c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
    ) -> kernels_cuda_new::x::Fired,
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
    // SAFETY: three bound device pointers, a live `cublasHandle_t` off the
    // dispatch context with this fire's stream bound, and four widths the
    // lowering states. The extents are the statement's own, so the bank the
    // second absorb offsets into is the one the trace named.
    let fired = unsafe {
        call(
            ctx.cublas,
            b.args[0].ptr.cast_const(),
            b.args[spec.n_in + spec.n_out].ptr.cast_const(),
            b.args[spec.n_in].ptr,
            rows,
            p(0),
            p(1),
            p(2),
            p(3),
        )
    };
    // `Refusal::Empty` is the archive's `tokens <= 0 || heads <= 0`, which
    // was a bare `return` under `()`. Nothing to launch is an ANSWER, so it
    // is `Ok` here; any other refusal is the launcher declining a shape,
    // which is the variant that exists to say so. `bind::quant_gemm::empty_\
    // or_panic` applies the same rule by panicking -- one of the two should
    // eventually be the other, and neither file owns both.
    match fired {
        kernels_cuda_new::x::Fired::Launched => Ok(()),
        kernels_cuda_new::x::Fired::Declined(kernels_cuda_new::x::Refusal::Empty { .. }) => Ok(()),
        kernels_cuda_new::x::Fired::Declined(why) => Err(DispatchRefusal::ShapeDeclined {
            kernel: kernel.to_string(),
            why: format!("{why:?}"),
        }),
    }
}

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

    // THE GENERATED HALF WAS CALLED HERE, FIRST, AND IS DELETED.
    //
    // `if dispatch_generated(bound, spec, frame, ctx, attn, gdn, resolver,
    // rows, Arms::Whole) { return Ok(()); }` — the AOT arm, tried ahead of
    // the fn-world registry because a row that stated its own operand
    // sources needed no hand-written arm.
    //
    // Nothing takes its place, and nothing has to: it could only return
    // `false` once `table::ROW_TABLES` emptied, because the `match` it ran
    // is emitted per STATED row and no row states operands any more. The
    // ordering comment it carried — *"GENERATED FIRST ... the hand-written
    // match below is what is LEFT, not what is normal"* — has inverted. The
    // hand-written driver-op arms and the fn-world registry are what there
    // is, and `Route::Bound` below is the normal path.
    //
    // `Route::Rows` is still REACHABLE and still falls through to the
    // hand-written match's `other` arm, which refuses by name. It is reached
    // when `x::route` finds a symbol in `x::SIGS` with no `Entry` that is
    // also not a `Service::DriverOp` — `x/mod.rs:388` documents at least one
    // such contract. Whether any is actually left is a fact about the
    // registry that only a build answers, so the arm stays and says what it
    // was asked.

    // THE fn-WORLD REGISTRY, second.
    //
    // `.wiki/kernel-x/northstar.md` §5 step 3. A family that has crossed
    // into fn-world has no rows to generate an arm from and no hand-written
    // arm below — its host programs live in `kernels_cuda_new::x::<family>`,
    // one file per root `.cuh`, holding that `.cuh` by `include_str!`
    // (§5.1 ①: a `.rs` under `csrc/` would be carried to NVRTC as a header).
    //
    // # Why here and not first
    //
    // The two sets are DISJOINT by construction: a ported family's
    // contracts state no `operands`, so `emit_rust_dispatch` cannot emit an
    // arm for one, and its symbol is deleted from every row table in the
    // same change that ports it. Order is therefore not a precedence
    // question and this is not a fallback — it is the second of two
    // registries, and a symbol that is in both is a bug in the port rather
    // than a case this line resolves. It sits after the generated call so
    // that the ONE unported thing stays cheapest, which is the fire that
    // has not been migrated yet.
    //
    // # No lookup happens here
    //
    // §5 step 4 has landed: `DispatchPlan::new` resolves every symbol in
    // `lowered.kernels` to a `Route` once, at model load, and the fire
    // reads it off its own spec. The symbol string is never compared at
    // fire time.
    //
    // # Why this is a `match` on four arms and not `if let Some(entry)`
    //
    // Because two of the four are refusals the load already made, and a
    // fire that reached one is a fire the load should have stopped. Naming
    // them here rather than folding them into a fallthrough is what makes
    // `LoweredFire`'s gate checkable: if either arm below is ever taken,
    // the gate in `fire::launch` did not run, and the message says so.
    //
    // # A refusal is not a fallthrough
    //
    // A symbol the registry HOLDS is dispatched here or not at all. If its
    // bind refuses, that is the answer — `DispatchRefusal::NoArm` carrying
    // the sentence the bind wrote, not a walk down to a hand arm that
    // would fire it with different arithmetic.
    match spec.route {
        kernels_cuda_new::x::Route::Bound(entry) => {
            // Resolved before the `Cx` exists, because a `Resolver` is `&mut`
            // and `Facts` is not: a bind body that could consult the weight
            // store could also make it answer differently the second time.
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
            // The suffix reach, resolved once for the same reason `w_named`
            // is: a `Resolver` is `&mut` and `Facts` is not. THE SUFFIX SET
            // IS FIXED AND SHORT BY MEASUREMENT — `quant`'s routed MXFP4 pair
            // is the only family that reaches by suffix, and it reaches
            // exactly these three (`weight_names.rs:505`). A statement that
            // names none of them gets three nulls and `Facts::weight_suffixed`
            // filters them, which costs one scan of three entries per fire
            // and no allocation.
            //
            // Listing them here rather than deriving them from the row is
            // deliberate: `Source::WeightSuffix(s)` is going away with the
            // row world, and the fn-world spelling of the same fact is a
            // string literal in a bind body. A fourth suffix is a line here
            // and a line there, and the compiler will not connect them --
            // which is why the list is beside the argument rather than in a
            // header.
            let bank = spec.weight.as_deref();
            let mut suffixed = |suffix: &str| -> *const c_void {
                bank.and_then(|b| resolver.weight(&format!("{b}{suffix}")))
                    .unwrap_or(core::ptr::null())
            };
            let w_suffixed: [(&'static str, *const c_void); 3] = [
                ("_scales", suffixed("_scales")),
                ("_gate_bias", suffixed("_gate_bias")),
                ("_up_bias", suffixed("_up_bias")),
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
                .call(&kernels_cuda_new::x::Cx::new(&fire), ctx.stream)
                .map_err(|r| DispatchRefusal::NoArm(format!("{}: {r}", bound.kernel)));
        }
        // BOTH ALREADY REFUSED AT LOAD. Reaching one means the lowering was
        // fired without `LoweredFire`'s gate, which is a driver bug and not
        // a model's, so the message names the gate rather than the kernel.
        kernels_cuda_new::x::Route::Unbound(_, why) => {
            return Err(DispatchRefusal::NoArm(format!(
                "{}: {why} (load-time refusal; the lowering was fired without \
                 DispatchPlan::unfireable being checked)",
                bound.kernel
            )));
        }
        kernels_cuda_new::x::Route::Unknown => {
            return Err(DispatchRefusal::NoArm(format!(
                "{}: no contract and no row declares it (load-time refusal; \
                 the lowering was fired without DispatchPlan::unfireable \
                 being checked)",
                bound.kernel
            )));
        }
        // THE DRIVER'S OWN OPS, and the row world. Both fall through to the
        // match below — see the `Route::Driver` note there.
        kernels_cuda_new::x::Route::Driver | kernels_cuda_new::x::Route::Rows => {}
    }

    // The GDN arms' shared reads: the ctx itself, and the launch's state
    // layer's slab out of one of its per-layer vectors.
    let _gdn_ctx = || -> Result<&GdnCtx, DispatchRefusal> {
        gdn.ok_or_else(|| DispatchRefusal::NoGdnCtx(bound.kernel.to_string()))
    };

    // `let _rows = ...` stood here: the same expression as `rows` at the top
    // of this function, re-derived and never read. MLA's absorb arms are the
    // first consumer of the row count in this match and they read the one
    // binding. A second derivation of a value already in scope is Sec 75's
    // shape in miniature -- it cost nothing only because it was unused.
    // The op join's output placements: what a guard-region launch binds
    // for the value the GUARD owns (the recurrence three-way's core out).
    // The join's placements window with the args, or a launch reads its
    // input at the window and writes its output at the base.
    let win = if bound.kernel.ends_with("_devwin") {
        0
    } else {
        bound.rows.start
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

    // ── THE DRIVER-OP TABLE ──────────────────────────────────────
    //
    // What is left of the hand-written match, and it is NOT a leftover.
    // §5 step 4 says "delete the hand match (its one live arm,
    // `pie_lora_qkv_correction`, becomes a normal `bind: None`-plus-driver-fn
    // entry or a bind that reaches `LoraFireState::apply`)". Neither shape
    // is right, and the reason is mechanical:
    //
    // * `bind: None` is what `Entry` uses to mean "this symbol is REFUSED,
    //   here is the sentence". This symbol is not refused — it runs, every
    //   fire that stages an adapter. Spelling "it runs through something
    //   else" as "it does not run" is the one thing `Fired` exists to
    //   prevent, one level up.
    // * A bind receives a `Cx`, which is query-only by §3.3 — no device
    //   API, no allocator, no stream mutation — and this op needs
    //   `ctx.cublas`. A cuBLAS handle is a device API with a settable
    //   stream, a math mode and a workspace, so a `Cx` that could hand one
    //   over is a `Cx` a bind body could misbehave on. That is precisely
    //   the surface §3.3 says must not exist, and it is not worth spending
    //   to delete a `match` with one arm.
    //
    // So the third shape: `execution::Service::DriverOp` — already data,
    // already read by the migration census — becomes `Route::Driver`, and
    // this match is the driver-op table it names. It holds no device text,
    // and it does NOT retire with the step-5 sweep. Step 6 deletes the
    // GENERATED match beside it; this one stays and gets a better name.
    //
    // ── THE COUNT, RE-DERIVED ────────────────────────────────────────────
    //
    // This paragraph used to read *"It has TWO members here and two more in
    // the verify path."* **Measured today: ELEVEN arms** — the two
    // `gemm::mla_absorb_*`, the FlashInfer six, `moe::build_moe_ptrs_\
    // aligned_bf16` and `moe::moe_grouped_gemm_bf16`. Nine arrived after the
    // sentence was written and none of them re-derived it. A literal that
    // names a length, written once and re-derived by nothing, is the class
    // this session has now catalogued sixteen times; the count is left here
    // because a reader deciding what step 6 deletes needs a right one.
    //
    // ── AND WHAT THE COUNT DOES NOT COVER ────────────────────────────────
    //
    // `execution::SERVED` names FOUR `gemm::act_x_wt_*` symbols
    // `Service::DriverOp` — `act_x_wt_bf16`, `_bf16_out_fp32`, `_bias_bf16`,
    // `grouped_act_x_wt_bf16` — and **this match has an arm for none of
    // them**, nor does anything in `driver-cuda/src` call their
    // `bind::service` bodies.
    //
    // That is not a defect this comment is hiding: `x/gemm.rs:1100` records
    // it, because the lowering emits the PORTABLE spelling `gemm::act_x_w`
    // (`GEMM_XWT`'s `lowered_as`) and that spelling resolves to
    // `Route::Unknown` today — *"the state of the floor as this family was
    // ported, and it is in the report."* Three more `gemm::` contracts
    // (`act_x_wt_channel_scaled`, `act_x_wt_grouped_scaled`,
    // `act_x_wt_mxfp4_marlin`) are in neither `SERVED` nor `table::`, so
    // `route` answers `Route::Unknown` and the arm below prints *"no
    // contract and no row declares it"* — **which is false; a contract does
    // declare all three.** Zero goldens name any of them, so nothing can
    // observe the difference, which is why it has stayed.
    //
    // The `moe` member widens the shape rather than repeating it:
    // `moe::build_moe_ptrs_aligned_bf16` IS a `__global__`, so the "a `Cx`
    // cannot hand over a device API" argument above does not apply to it at
    // all. What it cannot hand over is a LIFETIME — see the arm.
    match bound.kernel {
        // ── THE FLASHINFER SIX ──────────────────────────────────────
        //
        // These were `execution::RUST_SERVED` rows: the generated `match`
        // in `dispatch_generated` resolved 13-19 operands from the table
        // and called into `bind::service`, which called `fire::`. They are
        // six hand arms now, and their rows are gone.
        //
        // **They are driver ops, and five of the six pass the test in one
        // line**: `DecodePlanCache` / `PrefillPlanCache` -- boxed, mutable,
        // cross-fire, raised by `Prepare::DecodePlan` / `Prepare::Prefill\
        // Plan` and owned by [`DecodePlan`] / [`PrefillPlan`] here. Name
        // the resource; that is Sec 3.3's forbidden surface exactly.
        //
        // **`attn::attention_flashinfer_prefill` names NONE**, and that is
        // the finding rather than an inconsistency. It builds a plan cache
        // on its own stack, allocates nothing, and asks `plan_device()` --
        // a read-only capability query, not a pool. It is a driver op by
        // the SECOND condition alone: it reads `qo_indptr_h` and
        // `kv_page_indptr_h`, the HOST mirrors of the CSR, and walks them
        // on the CPU. No `Cx` query answers a host pointer, and reading
        // the device CSR host-side is a synchronise, which Sec 0 forbids
        // inside a fire. That is `split_qkv_bf16_devwin`'s shape, not
        // `gemm`'s -- *name the resource* is necessary and not sufficient,
        // and this is the second instance of the pair.
        //
        // Every arm resolves every refusal before its one launch, which is
        // free here: one call each, so there is no partial device state a
        // late `Declined` could lie about.

        // args: [q (the pin)]; o is the op's arena output; the plan, the
        // workspace and the layer view are the fire's.
        // ── MLA'S ABSORB PAIR ───────────────────────────────────────
        //
        // Two crossed rows, one resolution, `ctx.cublas` named twice. See
        // `mla_absorb` above for why the callee is chosen here.
        "gemm::mla_absorb_q_to_latent_bf16" => mla_absorb(
            "gemm::mla_absorb_q_to_latent_bf16",
            bound,
            spec,
            ctx,
            rows,
            kernels_cuda_new::x::attn::mla_absorb_q_to_latent_bf16,
        )?,
        "gemm::mla_absorb_latent_to_v_bf16" => mla_absorb(
            "gemm::mla_absorb_latent_to_v_bf16",
            bound,
            spec,
            ctx,
            rows,
            kernels_cuda_new::x::attn::mla_absorb_latent_to_v_bf16,
        )?,

        "attn::dispatch_attention_flashinfer_decode" => fa2_decode(bound, spec, ctx, attn)?,
        // args: as the plain decode dispatch, plus the SCORE outputs.
        // `_capture` is capturing scores, not capturing a graph —
        // `WantsAttnScore` is the guard that selects this spelling.
        //
        // The score buffers ride the ctx rather than the statement because
        // they must be arena-STABLE: the predicate is folded, so one exec
        // serves a fire that wants scores and one that does not, and an
        // address recorded now has to still be right when it goes true.
        "attn::dispatch_attention_flashinfer_decode_capture" => fa2_decode_capture(bound, spec, ctx, attn)?,
        // args: [q]; o is guard-owned ([`AttnCtx::o_out`]); the pages are
        // the layer's bf16 MIRRORS — the native alias, the decode lesson.
        "attn::dispatch_attention_flashinfer_prefill_bf16" => fa2_prefill(bound, spec, ctx, attn)?,
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
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16" => fa2_prefill_capture(bound, spec, ctx, attn)?,
        // args: [q] with the output guard-owned, or [q, o] as SSA. The
        // custom-mask arm: `HasCustomMask` selects it, and the mask rides
        // the ctx rather than the statement for the same reason the score
        // sink does — the predicate is folded, so one exec serves the fire
        // that stages a mask and the fire that does not, and the address
        // recorded now has to still be right when it goes true.
        "attn::dispatch_attention_flashinfer_prefill_custom" => fa2_prefill_custom(bound, spec, ctx, attn)?,
        // args: [q, o] — the PLANLESS flashinfer prefill (plans
        // internally per fire; reads the host CSR mirrors).
        "attn::attention_flashinfer_prefill" => fa2_prefill_planless(bound, spec, ctx, attn, rows)?,
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
        // args: [expert_ids, aligned_in] + the THREE staging buffers it
        // declares + [gate_up_bank, down_bank] — `BoundLaunch::args` is
        // inputs, outputs, weights in the trace's stated order, and
        // `dsl::cuda::build_moe_ptrs_aligned` states two, three and two.
        //
        // THE THIRD DRIVER OP, and the only one that is a `__global__`. It is
        // here for a LIFETIME and not for a device API: the six device
        // pointer arrays it fills have no stated consumer — their only reader
        // is the batched-cuBLAS arm inside `moe::moe_grouped_gemm_bf16`,
        // which is a LOWERING of that symbol and not a statement of it — so
        // six declared-but-unread results are six values `lower.rs:1911`'s
        // liveness frees at the first op past this one, and the GEMM would
        // dereference bytes the next allocation owns. A wrong answer, not a
        // refusal. `fire::moe_ptrs` carves them from the driver's own arena
        // instead, and `execution::SERVED` carries the classification.
        //
        // It DECLARES the aligned staging, which is why it cannot be skipped
        // on a leg that takes it: step 3 of 8, and steps 4 through 8 all
        // write into buffers this statement named.
        "moe::build_moe_ptrs_aligned_bf16" => {
            need(7)?;
            let (expert_ids, aligned_in) = (bound.args[0], bound.args[1]);
            let stage = crate::fire::moe_ptrs::Stage {
                gate_up: bound.args[2].ptr,
                act: bound.args[3].ptr,
                out: bound.args[4].ptr,
            };
            // THE SHARED PAIR IS NULL ON THIS LEG AND THAT IS NOT A GAP.
            // `moe_mlp_body_aligned_cuda` runs the shared expert as its own
            // dense `gemm::act_x_w` pair (`forward/mod.rs:240-242`, under
            // `shared_expert_intermediate > 0`), so the aligned build never
            // addresses a shared tail. Passing null is what makes the host
            // program's rewrite — `routed_blocks = max_blocks`,
            // `moe_dispatch.cu:246-248` — the identity it is here, and
            // `routed_blocks` below is that same value stated rather than
            // relied upon.
            let banks = crate::fire::moe_ptrs::Banks {
                gate_up: bound.args[5].ptr.cast_const(),
                down: bound.args[6].ptr.cast_const(),
                shared_gate_up: std::ptr::null(),
                shared_down: std::ptr::null(),
            };
            // `MOE_ALIGNED_BLOCK` at `model/src/qwen_3_5/forward/mod.rs:18`,
            // and the same 16 `x::moe::supported` requires of the grouped
            // GEMM's `M` (`FRAG`). `max_blocks` is DERIVED from the padded
            // rectangle rather than read as `MOE_MAX_BLOCKS`, because the
            // rectangle is what the arrays have to cover and a constant that
            // agreed with it by construction would still be a second copy.
            const BLOCK: u32 = 16;
            let aligned_rows = bound.rows.end - bound.rows.start;
            if aligned_rows == 0 || aligned_rows % BLOCK != 0 {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!(
                        "the aligned rectangle is {aligned_rows} rows, which is not a whole \
                         number of {BLOCK}-row blocks -- the padded batch and the block size \
                         disagree, and `max_blocks <= 0` is the only shape the launcher itself \
                         refuses"
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
            // SAFETY: every pointer is a bound arg of this launch — device
            // allocations of the shapes the statement declares, live on
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
                // THE ARRAYS GO ON THE CTX AND THIS IS THE ONLY WRITE TO IT;
                // `moe::moe_grouped_gemm_bf16`'s arm below is the only read.
                // The `Cell` is what makes the write legal from a
                // `&DispatchCtx`, and the two being arms of one `match` in
                // one fire is what makes it correct: the build is step 3 of 8
                // and both grouped GEMMs are below it in the same lowering.
                //
                // **The field existed for an hour with no writer**, which is
                // the hazard its own doc warns about one level up — a `Cell`
                // is only better than the thread-local it replaced if
                // something fills it.
                crate::fire::moe_ptrs::Built::Ready(arrays) => ctx.moe_ptrs.set(Some(arrays)),
                crate::fire::moe_ptrs::Built::Declined(why) => {
                    // A DECLINE HERE MUST BE AN ERROR AND NOT A QUIET RETURN.
                    // Nothing after this op can run: the two GEMMs read arrays
                    // that were never written, and the swiglu between them
                    // writes into staging whose addresses were never baked.
                    // That is `ShapeDeclined`'s own subject — "smoothly wrong
                    // is the failure mode this tree keeps naming".
                    return Err(DispatchRefusal::ShapeDeclined {
                        kernel: bound.kernel.to_string(),
                        why: format!("{why:?}"),
                    });
                }
            }
        }
        // ── the aligned leg's steps 4 and 5 ───────────────────────────────
        //
        // One symbol, two implementations, chosen by `x::moe::supported`.
        // The choice cannot be a bind's — the batched form needs the cuBLAS
        // handle and six arrays that are nobody's operand — and cannot be
        // this arm's either, because it belongs beside both implementations.
        // So it is `fire::moe_grouped`'s and this arm only assembles.
        //
        // **This is the only symbol in the tree that reached the driver-op
        // shape from a WORKING BIND.** The bind served `down` (K=512) and
        // refused `gate_up` (K=2048) with `Refusal::Wide`, and this file's
        // own "a refusal is not a fallthrough" makes that the FINAL answer
        // rather than the first half of a choice. So both implementations
        // had to move behind one host program; leaving the bind would have
        // been a working half and a permanent refusal.
        //
        // The count is `<` and not `need(n)`: the weight may or may not ride
        // the flat run, and the bank is read from `spec.weight` — the same
        // source `cx.weight(0)` read when this was a bind.
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
            // SAFETY: every pointer is a bound arg or a resolved weight of
            // this launch, live on `ctx.stream` for the fire; the arrays are
            // this fire's own build, three arms up.
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
            if let kernels_cuda_new::x::Fired::Declined(why) = fired {
                return Err(DispatchRefusal::ShapeDeclined {
                    kernel: bound.kernel.to_string(),
                    why: format!("{why:?}"),
                });
            }
        }
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
        other => {
            // A `Route::Rows` symbol the generated match had no arm for, or
            // a `Route::Driver` with no arm above. Both are row-world
            // failures by now — the two fn-world refusals returned earlier
            // — so the message says which registry was asked, because
            // "NoArm" alone sent readers to `x/` for a symbol that never
            // reached it.
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

// `isqrt_exact` AND `stage_d2d` STOOD HERE, AND GO WITH THE GENERATED MATCH.
//
// Both were the generated arms' helpers and had no other caller.
//
// `isqrt_exact(w) -> Option<i32>` was `sqrt(w)` when `w` is a perfect square
// — how a `[K, K]` coefficient block stated its K, for an arm that had the
// block's element count and needed its side.
//
// `stage_d2d(ctx, rows, dst, src)` was the row-range device-to-device copy
// two arms shared. Its subject survives in fn-world: a copy a bind body
// needs is a `Launch` like any other, and `Cx` states the extents.
//
// They are named here rather than simply removed because a reader tracing
// the deleted `match` will find them cited in it. They were `_cuda` for one
// commit (`e88f1ffff`) on the fossil test — their bodies name no archive
// symbol, which is true and was not sufficient: a body that reaches nothing
// can still be reachable from nothing.


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
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct AttnRegions<'a> {
    /// The fire's own state, for rectangles that span it.
    pub fire: Option<&'a AttnCtx>,
    /// A peel TAIL's state. `None` on a fire with no split, where it is
    /// never selected because no rectangle is windowed.
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
        Self {
            fire: Some(fire),
            tail: Some(tail),
        }
    }

    /// The state a rectangle executes against.
    ///
    /// Keyed on whether the rectangle is WINDOWED, which is what makes it
    /// a tail: a peel's prefix starts at row zero and its tail does not.
    #[must_use]
    pub fn of(&self, rows: &std::ops::Range<u32>) -> Option<&'a AttnCtx> {
        if rows.start == 0 {
            self.fire
        } else {
            self.tail.or(self.fire)
        }
    }
}

/// Execute one fire: bind and dispatch every launch of the lowering, in
/// order. The walk the full-decode smoke proved, as the executor's entry.
///
/// # Errors
///
/// The first refusing launch, with its index and kernel — a drift
/// diagnosis, never a runtime condition to retry.
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
        dispatch(
            &bound,
            dplan.spec(i),
            frame,
            resolver,
            ctx,
            attn.of(&launch.rows),
            gdn,
        )
        .map_err(|e| RunRefusal {
            launch: i,
            kernel: kernel(),
            why: RunRefusalKind::Dispatch(e),
        })?;
    }
    Ok(lowered.launches.len())
}

/// One open conditional on the capture stack: the node, and which of its
/// two arms is currently being captured into.
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

/// Are these two tree nodes the two arms of ONE conditional?
///
/// Read off the lowering's stated pairing rather than derived from
/// `(parent, slot, param)`: a family states the same guard once per
/// layer, so those three fields identify one conditional PER LAYER, and
/// deriving would pair an arm with some other layer's else body.
#[cfg(feature = "_cuda")]
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
            builder
                .close_cond(&f.cond)
                .map_err(|e| cuda(i, &kernel, e))?;
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
        dispatch(
            &bound,
            dplan.spec(i),
            frame,
            resolver,
            &ctx,
            attn.of(&launch.rows),
            gdn,
        )
        .map_err(|e| RunRefusal {
            launch: i,
            kernel: kernel.clone(),
            why: RunRefusalKind::Dispatch(e),
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
        builder
            .close_cond(&f.cond)
            .map_err(|e| cuda(last, "<unwind>", e))?;
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
                BoundArg {
                    ptr: std::ptr::NonNull::<c_void>::dangling().as_ptr(),
                    width: 0,
                }
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
    let row = if kernel.ends_with("_devwin") {
        0
    } else {
        launch.rows.start
    };
    let mut args = Vec::with_capacity(launch.args.len());
    for arg in &lowered.args[launch.args.start as usize..launch.args.end as usize] {
        args.push(resolve_arg_windowed(arg, frame, resolver, row)?);
    }
    Ok(BoundLaunch {
        kernel,
        rows: launch.rows.clone(),
        layers: launch.layers.clone(),
        args,
    })
}
