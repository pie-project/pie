//! ② KERNEL SIGNATURES — the compiler's end of the tables.
//!
//! The rows moved out. A kernel's contract belongs beside the kernel, so the
//! CUDA table is [`kernels_cuda::sigs`] and the Metal one WAS
//! [`kernels_metal::KERNELS`], each in the crate that also holds the `.cu` /
//! `.metal` it describes — one source file and one table row, same directory,
//! same diff hunk. Metal's is empty: every family crossed to a `Routine`, and
//! what this module reads there is [`kernels_metal::declared`]. [`Stated`] is
//! the shape both planes answer in. The words a row is written in are the `kernels` crate's,
//! which is also where the reasons for `whole` / `needs` / `lacks` / `sink`
//! are written down.
//!
//! What stays here is what is about the COMPILER rather than about a kernel:
//! [`Backend`], which reads a backend out of a traced family name, and
//! [`check_plan`], which walks a [`ForwardPlan`]. Both are re-exported
//! through this module so call sites keep saying `kernels::sig(..)`.
//!
//! The tables are consumed with `default-features = false`, so reading a
//! symbol's contract does not build a single `.cu`.

use crate::trace::{ForwardPlan, OpKind};

// The vocabulary and the one remaining table, re-exported so this module
// reads as one surface. `Cap` and `Prepare` are named by model texts and by
// the tests below; `KernelSig` is what `trace` and `emit_cuda` hold a
// reference to.
pub use kernels::{Cap, KernelSig};
pub use kernels_cuda::sigs;
// The census -- every entrypoint Metal can dispatch, which is what a text may
// launch. It outlived `kernels_metal::KERNELS`, whose rows used to GENERATE
// it and which is an empty slice now.
pub use kernels_metal::entrypoints as metal_entrypoints;

/// Which backend's kernels a lowered trace states.
///
/// The table is per-BACKEND because a kernel signature is backend-owned
/// (`.wiki/tart/dsl.md` ②). A model text is written for one backend and
/// states that backend's symbols; the family name says which —
/// `llama_like.cuda.decode` is CUDA's, `llama_like.metal.decode` is Metal's.
///
/// # Two variants, four execution shells
///
/// This enum names the SURFACE a text was written against, not the device
/// that will run it, and the two are not one-to-one. `driver-vulkan` and
/// `driver-wgpu` execute plans traced by `llama_like_metal` — their tables
/// (`kernels_vulkan::KERNELS`, `kernels_wgpu::KERNELS`) are
/// `kernels-metal`'s coverage ROW FOR ROW, so a plan naming Metal's symbols
/// resolves in either of them. There is no `Backend::Vulkan` because no
/// family name would ever produce one: a variant nothing can construct is
/// not a safeguard, it is a branch that never runs.
///
/// [`check_plan`] therefore validates a Vulkan- or WGPU-bound plan against
/// **Metal's** statements, and that is sound only for as long as the three
/// stay equal. They were held equal by
/// `kernels-{vulkan,wgpu}/tests/entrypoints.rs`'s
/// `every_row_states_the_same_facts_kernels_metal_does` and
/// `every_row_asks_for_the_same_operands_kernels_metal_does`, which diff the
/// tables' source text and carry NO exception list — a comparison that only
/// runs while both sides have rows. Metal has none. What holds the three
/// equal now is `kernels/tests/shader_backends_agree.rs`, which counts each
/// backend's rows, routines and RETIREMENTS into one union and requires the
/// hundred; a name a sibling adds and Metal never had would fail there. If
/// that ever changes, this is the code that silently loses its guarantee: the
/// load-time refusal below stops firing and the failure reappears as a
/// bind-time `expect("stated")` inside the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
}

impl Backend {
    /// The backend a traced family name names, or `None` for a SEMANTIC
    /// trace — which states no kernels at all, so no table applies.
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }

}

/// What a backend states about one symbol, from whichever plane states it.
///
/// The compiler asks a table two questions and nothing else: does anything
/// declare this symbol, and what does it say about `whole` and `in_place`.
/// Both were `KernelSig` fields, which was fine while a kernel was a ROW.
/// A crossed kernel is a `Routine` instead, and its
/// [`kernels::routine::Declared`] states the same two facts — so this is the
/// answer with the plane forgotten, exactly as `Declared` is the row with the
/// backend forgotten.
///
/// It exists because `kernels-metal` finished crossing: its table is empty,
/// and a lookup that only read tables began refusing every Metal model at
/// load. `refactor-bigplan.md` §7 Stage 5 deletes `KernelSig` once the last
/// backend is here, and at that point this struct is the only shape left.
#[derive(Debug, Clone, Copy)]
pub struct Stated {
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// This statement pairs the depth-PREFIX plan, and its dedicated
    /// workspace, on a union's tail layers.
    ///
    /// A kernel property rather than an op's, which is why it is answered
    /// here: `ForwardPlan::depth_prefix_plan` asks the backend about the
    /// symbol the op launches.
    pub depth_prefix_plan: bool,
    /// `(input, output)` pairs that must be given the same address.
    pub in_place: &'static [(u32, u32)],
}

/// What `backend` states about `symbol`, or `None` if nothing does.
///
/// CUDA reads its rows, because CUDA has not crossed. Metal reads its
/// ROUTINES, resolved through [`kernels_metal::kernel_of`] so that a text
/// spelling a base name and a lowering carrying an instantiated point both
/// land on the same statement — which is what `kernels::sig_in`'s two passes
/// did for a row, and is the behaviour that has to survive the crossing.
#[must_use]
pub fn stated_in(backend: Backend, symbol: &str) -> Option<Stated> {
    match backend {
        Backend::Cuda => kernels::sig_in(sigs(), symbol).map(|k| Stated {
            whole: k.whole,
            depth_prefix_plan: k.depth_prefix_plan,
            in_place: k.in_place,
        }),
        Backend::Metal => {
            // Memoised: `declared()` builds its `Vec` from ten `ROUTINES`
            // slices on every call, and this is asked once per launched op of
            // every model that loads.
            static ROUTINES: std::sync::OnceLock<
                std::collections::BTreeMap<&'static str, kernels::routine::Declared>,
            > = std::sync::OnceLock::new();
            let routines = ROUTINES.get_or_init(|| {
                kernels_metal::declared()
                    .into_iter()
                    .map(|d| (d.name, d))
                    .collect()
            });

            let name = kernels_metal::kernel_of(symbol)?;
            routines
                .get(name)
                .map(|d| Stated {
                    whole: d.whole,
                    depth_prefix_plan: d.depth_prefix_plan,
                    in_place: d.in_place,
                })
                // One of the hundred with no routine to state it. Metal has
                // exactly one, `silu_mul_strided`, whose entrypoint leaves a
                // buffer slot empty and so cannot be given a positional
                // argument list at all -- `driver-metal` calls it DARK and
                // refuses to lower it. Declared, because the shader is real
                // and the census names it; stating nothing, because nothing
                // states it. A text that launched it would pass here and be
                // refused where the refusal is true.
                .or(Some(Stated {
                    whole: false,
                    depth_prefix_plan: false,
                    in_place: &[],
                }))
        }
    }
}

/// The ROW for one CUDA symbol.
///
/// There is no `sig_in(backend, ..)` beside it, and there was: it took a
/// backend, looked the symbol up in that backend's table, and answered `None`
/// for every Metal symbol once Metal retired its rows. A function that takes
/// a choice it can only answer one way is a trap -- `None` reads as "nothing
/// declares this" at every call site, and three callers believed it. Ask
/// [`stated_in`] anything the compiler decides; this is for the readers that
/// want the ROW, which is the CUDA emitter and the tests measuring what is
/// left of the tables.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(sigs(), symbol)
}

/// LOAD-TIME check of a traced form against the kernel table.
///
/// WHICH table is [`Backend::of_family`]'s answer, and that answer is the
/// AUTHORING surface rather than the executing device — read [`Backend`]
/// before trusting this for a Vulkan or WGPU deployment, because those two
/// are checked against Metal's rows.
///
/// Two rules, both of which are runtime failures today:
///
/// 1. a `whole` kernel may not be stated inside a [`OpKind::Peel`]'s
///    regions — the peel gives each region a row window, and a
///    fire-wide-prepared kernel has no way to honour one;
/// 2. every launched symbol must be declared, so the table cannot rot
///    while the model texts move on.
///
/// Returns the failures rather than panicking, so a caller can name the
/// family it was loading.
pub fn check_plan(plan: &ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    let backend = Backend::of_family(&plan.family);
    // Ops inside a Peel's two regions, as a countdown over the flat op
    // list (regions are consecutive: prefix then tail, right after the
    // op — `OpKind::Peel`'s doc).
    let mut peeled = 0usize;
    for op in &plan.ops {
        let inside_peel = peeled > 0;
        peeled = peeled.saturating_sub(1);
        match &op.kind {
            OpKind::Peel {
                prefix_ops,
                tail_ops,
                ..
            } => {
                peeled = peeled.max(*prefix_ops as usize + *tail_ops as usize);
            }
            OpKind::Launch { kernel, .. } => match backend.and_then(|b| stated_in(b, kernel)) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel declares",
                    plan.family,
                    match backend {
                        Some(b) => format!("{b:?}").to_lowercase(),
                        // A semantic trace states no kernels; one that
                        // does has a family name that does not say
                        // whose they are.
                        None => "backend's".to_string(),
                    }
                )),
                Some(k) if k.whole && inside_peel => problems.push(format!(
                    "{}: `{kernel}` is declared `whole` but is stated inside a Peel \
                     region, which gives it a row window it cannot honour",
                    plan.family
                )),
                Some(_) => {}
            },
            _ => {}
        }
    }
    problems
}

/// Which outputs a stated kernel writes over which inputs.
///
/// Reads the BACKEND's table, which is why it takes the plan: the family
/// name says which backend, exactly as `check_plan` reads it.
pub fn in_place_pairs(plan: &ForwardPlan, kernel: &str) -> &'static [(u32, u32)] {
    Backend::of_family(&plan.family)
        .and_then(|b| stated_in(b, kernel))
        .map_or(&[][..], |s| s.in_place)
}

/// Which outputs a SEMANTIC op writes over which inputs.
///
/// The companion to [`in_place_pairs`], for the kinds that name no
/// kernel. A stated kernel carries this fact in the `kernel!` table
/// because the symbol is the thing being described; a semantic kind is
/// described by the kind itself, so the fact lives here.
///
/// It takes no backend, and that is a claim rather than a convenience:
/// these are properties of what the kind MEANS, so a backend that
/// disagreed would not be another implementation of the kind.
pub fn semantic_in_place(kind: &OpKind) -> &'static [(u32, u32)] {
    match kind {
        // Rope ROTATES; it does not produce. Every driver's arm takes
        // one q pointer and one k pointer and no separate destination
        // -- CUDA's four families and Metal's alike -- because there is
        // no other way to write the kernel.
        //
        // The trace still names the rotated q and k as new values,
        // which is right: SSA is how a reader tells the pre-rope q from
        // the post-rope one. What was missing is that those names are
        // two names for one buffer, and while every q was pinned to the
        // same workspace field, nothing needed the distinction. A host
        // that assigns addresses does: it gave the normed k an address
        // of its own, and rope then rotated the OTHER address and left
        // the real k unread.
        OpKind::Rope { .. } => &[(0, 0), (1, 1)],
        // `beta_one` IS the accumulate: cuBLAS computes `C = A·Bᵀ + C`,
        // so C is read as well as written and the residual it folds must
        // BE C. `try_fold_residual` pushes that residual as input 1 and
        // gives the op a fresh output id, which is right for dataflow —
        // a reader after the fold wants the summed stream, not the
        // pre-fold one — and says nothing about memory. This does.
        //
        // Only when it folded: a plain matmul writes its output and
        // reads nothing of it.
        OpKind::Matmul { beta_one: true, .. } => &[(0, 1)],
        // `attn_out *= sigmoid(gate)`. The full-attention output gate,
        // and the kernel qwen3.5 states for it is spelled
        // `sigmoid_gate_inplace_bf16` -- the gate is read-only, the
        // gated value is rewritten where it lies.
        OpKind::SigmoidGateMul => &[(0, 0)],
        // `x[r, :] += bias`. One buffer in both drivers that state it --
        // gpt-oss's `o_bias`, llama_like's three attention biases -- and
        // the kernel has no destination parameter to give it another.
        OpKind::AddBias { .. } => &[(0, 0)],
        _ => &[],
    }
}
