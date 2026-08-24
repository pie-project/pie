//! The claim tables, joined: which points a plane answers and which symbol
//! its `canon` rows answer for.
//!
//! # What stood here
//!
//! THE TRACE-TIME SIGNATURE WALK, and R4e deleted it whole: `Stated`,
//! `Pointers`, `stated_in`, `claim_of`, `sig`, `out_shape`, `in_place_pairs`,
//! `semantic_in_place`, `ARITY_EXCEPTIONS`, `arity_problem` and `check_plan`.
//! Together they read every LEGACY COLUMN a `#[routine]` emits — the operand
//! `sources`, the `derived` names-and-nullability run, `args`, `whole`,
//! `depth_prefix_plan` and the `out_rule`s — and checked a
//! `crate::trace::ForwardPlan` against them: does this statement place as
//! many pointers as the routine reads, does its result geometry follow the
//! rule the attribute stated.
//!
//! Every one of them was reachable from exactly one place,
//! `TraceBuilder::finish`, and nothing has built a `TraceBuilder` since R3
//! deleted `model-dsl-legacy`'s `Trace`. The baker path asks none of it: a
//! statement's operands are checked against the POINT's slot list by the
//! generated dispatch, its geometry by `model_compiler::program`'s width
//! walk, and its claim by `points_dispatch::CLAIMED` before a fire.
//!
//! What is left is the join the baker path does use, and it is two
//! functions.

pub use kernels::Kind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Backend {
    Cuda,

    Metal,
}

impl Backend {
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }
}

/// Every family a cuda `#[claims]` block answers, one line per migrated
/// family. A family's line is its `*_CLAIMS` slice and nothing else, so
/// migrating one is adding one line here; the macro grows an aggregate when
/// there are enough of them to want one.
const CUDA_CLAIMS: &[&[&str]] = &[
    kernels_cuda::norm::NORM_CLAIMS,
    kernels_cuda::rope::ROPE_CLAIMS,
    kernels_cuda::mlp::MLP_CLAIMS,
    kernels_cuda::gemm::GEMM_CLAIMS,
    kernels_cuda::dist::DIST_CLAIMS,
    kernels_cuda::moe::MOE_CLAIMS,
    kernels_cuda::layout::LAYOUT_CLAIMS,
    // `GATE_CLAIMS` reads from `mlp` rather than a module of its own: the
    // impl lives beside the other gate kernel firing the same C++ namespace.
    kernels_cuda::mlp::GATE_CLAIMS,
    kernels_cuda::ssm::SSM_CLAIMS,
    // `ATTENTION_CLAIMS` reads from `attn` and not `attention`: the impl
    // lives in the module the fa2 core is filed under, which is where the
    // four delegations' neighbours already are.
    kernels_cuda::attn::ATTENTION_CLAIMS,
    // The three latent/paged families answer from `attn`, where their
    // delegates live — `mla`'s two absorbs reach across into `gemm::absorb`
    // from there, since a family is one impl block and its points may fire
    // out of two modules. `POOL_CLAIMS` is EMPTY and still listed: cuda
    // implements the family and overrides nothing, which is a measurement
    // and not an omission.
    kernels_cuda::attn::MLA_CLAIMS,
    kernels_cuda::attn::INDEX_CLAIMS,
    kernels_cuda::attn::POOL_CLAIMS,
    // `HC_CLAIMS` reads from `norm`, beside the five hyper-connection
    // kernels firing the same C++ namespace — the `GATE_CLAIMS` shape.
    kernels_cuda::norm::HC_CLAIMS,
];

/// Every family a metal `#[claims]` block answers — the same line-per-family
/// shape as [`CUDA_CLAIMS`], on the first plane to follow cuda onto the
/// declaration floor.
///
/// ALL FOURTEEN NOW, and the fourteen are the measurement rather than the
/// achievement: six of the lines are EMPTY, because a family a plane
/// implements and claims nothing of is a measured backlog while a family a
/// plane does not implement at all is a hole in the table where a measurement
/// should be. Each empty line's reason is written in its impl header, where a
/// reader arrives with the code in front of them:
///
/// * `GEMM_CLAIMS` (from `layout`) — every matmul this plane stamps is
///   QUANTIZED, so all three points wait on the floor's `Bank<R: Repr>`
///   payload, with `layout.embed` and `moe.matmul_select*` behind the same
///   gap.
/// * `DIST_CLAIMS` — no collective, and no transport under one.
/// * `MLA_CLAIMS`, `INDEX_CLAIMS`, `POOL_CLAIMS`, `HC_CLAIMS` — no `.metal`
///   kernel for any point of any of the four; these are families to write,
///   not crossings to make.
///
/// `GATE_CLAIMS` reads from `attn` for `kernels_cuda::mlp::GATE_CLAIMS`'
/// reason turned around: the impl lives beside the one kernel it fires, and
/// on this plane that kernel is filed with the attention it gates.
/// `LAYOUT_CLAIMS` reads from `layout` while both of its claimed points fire
/// out of `attn/` — a family is one impl block and its points may fire out of
/// two shader directories. `MLA_CLAIMS`, `INDEX_CLAIMS` and `POOL_CLAIMS`
/// read from `attn` and `HC_CLAIMS` from `norm`, which is where cuda files
/// the same four.
const METAL_CLAIMS: &[&[&str]] = &[
    kernels_metal::norm::NORM_CLAIMS,
    kernels_metal::mlp::MLP_CLAIMS,
    kernels_metal::layout::GEMM_CLAIMS,
    kernels_metal::dist::DIST_CLAIMS,
    kernels_metal::rope::ROPE_CLAIMS,
    kernels_metal::moe::MOE_CLAIMS,
    kernels_metal::layout::LAYOUT_CLAIMS,
    kernels_metal::attn::GATE_CLAIMS,
    kernels_metal::ssm::SSM_CLAIMS,
    kernels_metal::attn::ATTENTION_CLAIMS,
    kernels_metal::attn::MLA_CLAIMS,
    kernels_metal::attn::INDEX_CLAIMS,
    kernels_metal::attn::POOL_CLAIMS,
    kernels_metal::norm::HC_CLAIMS,
];

/// The points a plane's `#[claims]` impl blocks answer — baker's claim
/// table, consulted ahead of the routine `canon` attributes. One slice per
/// migrated family, concatenated; a family lands by adding its line, and the
/// macro grows an aggregate when the list is long enough to want one.
#[must_use]
pub fn point_claims(backend: Backend) -> &'static [&'static str] {
    static CUDA: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    static METAL: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    match backend {
        Backend::Cuda => CUDA.get_or_init(|| CUDA_CLAIMS.concat()),
        Backend::Metal => METAL.get_or_init(|| METAL_CLAIMS.concat()),
    }
}

/// The symbol that answers `claim` on `plane`, or `None` if nothing does.
///
/// THE PLANE'S OWN TABLE, READ DIRECTLY. Each kernels crate states its
/// `CANON` rows as `(claim, symbol)` pairs — cuda's two and metal's two, each
/// argued at its definition — and this is the only reader either has. This
/// was a walk over a linkme registry whose thirteen-column rows four crates
/// carried for this one question; the routine layer that filled it is folded
/// and the two tables are what is left of it.
///
/// `model_compiler::{sweep::resolve, program::call_of}` ask AFTER the tier-2
/// prefix and the point claims, so a row here is reached only by a claim no
/// `#[claims]` block answers.
#[must_use]
pub fn canon_symbol(plane: Backend, claim: &str) -> Option<&'static str> {
    let table = match plane {
        Backend::Cuda => kernels_cuda::CANON,
        Backend::Metal => kernels_metal::CANON,
    };
    table
        .iter()
        .find(|(role, _)| *role == claim)
        .map(|(_, symbol)| *symbol)
}
