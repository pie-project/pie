//! ② KERNEL SIGNATURES — the compiler's end of the tables.
//!
//! The rows moved out. A kernel's contract belongs beside the kernel, so the
//! CUDA table is [`kernels_cuda::KERNELS`] and the Metal one is
//! [`kernels_metal::KERNELS`], each in the crate that also holds the `.cu` /
//! `.metal` it describes — one source file and one table row, same directory,
//! same diff hunk. The words a row is written in are the `kernels` crate's,
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

// The vocabulary and the two tables, re-exported so this module reads as one
// surface. `Cap` and `Prepare` are named by model texts and by the tests
// below; `KernelSig` is what `trace` and `emit_cuda` hold a reference to.
pub use kernels::{Cap, KernelSig, Prepare};
pub use kernels_cuda::KERNELS;
pub use kernels_metal::KERNELS as KERNELS_METAL;

/// Which backend's kernels a lowered trace states.
///
/// The table is per-BACKEND because a kernel signature is backend-owned
/// (`.wiki/tart/dsl.md` ②). A model text is written for one backend and
/// states that backend's symbols; the family name says which —
/// `llama_like.cuda.decode` is CUDA's, `llama_like.metal.decode` is Metal's.
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

    pub fn table(self) -> &'static [KernelSig] {
        match self {
            Backend::Cuda => KERNELS,
            Backend::Metal => KERNELS_METAL,
        }
    }
}

/// The contract for one recorded symbol, in `backend`'s table.
pub fn sig_in(backend: Backend, symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(backend.table(), symbol)
}

/// The contract for one CUDA symbol.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    sig_in(Backend::Cuda, symbol)
}

/// LOAD-TIME check of a traced form against the kernel table.
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
            OpKind::Launch { kernel, .. } => match backend.and_then(|b| sig_in(b, kernel)) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel! signature declares",
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
                    "{}: `{kernel}` is declared `whole` (needs {:?}) but is stated \
                     inside a Peel region, which gives it a row window it cannot honour",
                    plan.family, k.needs
                )),
                Some(_) => {}
            },
            _ => {}
        }
    }
    problems
}
