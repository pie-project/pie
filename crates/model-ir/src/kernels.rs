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
//! `depth_prefix_plan` and the `out_rule`s — and checked the legacy traced
//! form, `ForwardPlan`, against them: does this statement place as many
//! pointers as the routine reads, does its result geometry follow the rule
//! the attribute stated.
//!
//! Every one of them was reachable from exactly one place,
//! `TraceBuilder::finish`, and nothing had built a `TraceBuilder` since R3
//! deleted `model-dsl-legacy`'s `Trace`. R5 deleted the pair of them and the
//! whole `trace` module with them. The baker path asks none of it: a
//! statement's operands are checked against the POINT's slot list by the
//! generated dispatch, its geometry by `model_compiler::program`'s width
//! walk, and its claim by `points_dispatch::CLAIMED` before a fire.
//!
//! What is left is the join the baker path does use, and it is two
//! functions.

pub use kernels::points;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Backend {
    Cuda,

    Metal,

    /// The WebGPU plane, added at P5b on the terms `Metal` arrived on at P5a:
    /// a row here is what makes a plane's claim tables reachable from
    /// [`point_claims`], and therefore what lets `sweep::resolve` bind — or
    /// honestly refuse — a lane against it.
    ///
    /// `driver-wgpu/src/baker/` names this variant in exactly one place, the
    /// `trace(..)` call in `Baked::of`, which is the whole of what used to be
    /// a load-time binding on this plane too.
    Wgpu,

    /// The Vulkan plane, on the terms `Wgpu` arrived on at P5b.
    ///
    /// `kernels-vulkan`'s seven `#[claims]` blocks have emitted their
    /// `*_CLAIMS` tables since they were written and NOTHING HAS EVER READ
    /// THEM: that crate's own manifest records the consequence — *"`model-ir`'s
    /// `Backend` has no vulkan arm, so `canon_symbol` never reached this
    /// crate"* — and without a row here `sweep::resolve` cannot join a lane's
    /// points against the plane at all, so every lane refuses for a reason that
    /// is about this table rather than about the plane.
    Vulkan,
}

// `Backend::of_family` STOOD HERE: the second segment of a dotted family
// name (`kernels.cuda.norm`) read back as a plane. It was the legacy
// registry's reverse lookup — a linkme row carried its plane in its own name
// — and the three claim tables below are indexed by the plane a caller
// ALREADY HAS. R5 measured zero callers.

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

/// Every family a wgpu `#[claims]` block answers — the same line-per-family
/// shape as [`CUDA_CLAIMS`], on the second shader plane to follow cuda onto
/// the declaration floor.
///
/// ALL FOURTEEN, and as with [`METAL_CLAIMS`] the empty lines are the
/// measurement. Seven of them are empty, and the two that decide what this
/// plane can serve are argued at length in their impl headers:
///
/// * `GEMM_CLAIMS` (from `quant`) — EMPTY, and this is the plane's headline
///   gap: every matmul `kernels-wgpu` stamps is QUANTIZED (`qmv.wgsl`,
///   `qmm_t.wgsl`), so all three `Gemm` points wait on the floor's
///   `Bank<R: Repr>` payload. There is no dense matmul on this plane at all,
///   which is why a dense bf16 SKU refuses at `gemm.matmul` before it can
///   refuse anywhere else.
/// * `SSM_CLAIMS` — EMPTY: the fused GDN core this plane ships cuts the
///   family differently from the floor's three points (see the impl header's
///   three options), so claiming one would be a lie about which arithmetic
///   ran.
/// * `DIST_CLAIMS`, `MLA_CLAIMS`, `INDEX_CLAIMS`, `POOL_CLAIMS`,
///   `HC_CLAIMS` — no `.wgsl` kernel for any point of any of the five; these
///   are families to write, not crossings to make.
///
/// The module each line reads from is where the impl block lives, which on
/// this plane is the file the shaders are filed under: `GEMM_CLAIMS` from
/// `quant` because every matmul here is a quantised one, `GATE_CLAIMS` from
/// `attn` because the gate kernel is filed with the attention it gates, and
/// the five empty families from `points`, which is where this plane keeps the
/// blocks it has no shader directory for.
const WGPU_CLAIMS: &[&[&str]] = &[
    kernels_wgpu::norm::NORM_CLAIMS,
    kernels_wgpu::mlp::MLP_CLAIMS,
    kernels_wgpu::quant::GEMM_CLAIMS,
    kernels_wgpu::rope::ROPE_CLAIMS,
    kernels_wgpu::moe::MOE_CLAIMS,
    kernels_wgpu::layout::LAYOUT_CLAIMS,
    kernels_wgpu::attn::ATTENTION_CLAIMS,
    kernels_wgpu::attn::GATE_CLAIMS,
    kernels_wgpu::ssm::SSM_CLAIMS,
    kernels_wgpu::points::DIST_CLAIMS,
    kernels_wgpu::points::MLA_CLAIMS,
    kernels_wgpu::points::INDEX_CLAIMS,
    kernels_wgpu::points::POOL_CLAIMS,
    kernels_wgpu::points::HC_CLAIMS,
];

/// Every family a vulkan `#[claims]` block answers — the same line-per-family
/// shape as the three above, on the third shader plane to reach this table.
///
/// EIGHT LINES AND NOT FOURTEEN, and that is the difference worth naming
/// rather than padding over. [`METAL_CLAIMS`] lists all fourteen because
/// `kernels-metal` writes an impl block for every family it does not answer,
/// so a `*_CLAIMS` const exists to be empty; `kernels-vulkan` writes EIGHT
/// impl blocks and no more, so for `Dist`, `Mla`, `Index`, `Pool` and `Hc`
/// there is no const for a line here to name. The join this function makes is
/// a concatenation, so the two spellings answer the same — what is lost is the
/// measurement, which is exactly the reading metal's own comment gives: a
/// family a plane does not implement at all is a hole in the table where a
/// measurement should be. Five such holes are this plane's backlog, and
/// `crates/points-dispatch`'s `vulkan()` says the same thing from the other
/// side.
///
/// `Gemm` AND `Ssm` BOTH LEFT THAT LIST, in the same hour and from two
/// different agents, which is why this paragraph reads as one correction and
/// not two: the list was six and is five and then four as each family got an
/// impl block.
///
/// `SSM_CLAIMS` IS FIVE POINTS, NOT SEVEN. The two conv arms, `gdn_prep` and
/// the two gated-delta scans are written; `ssm.kda_step` and
/// `ssm.kda_chunked` are not in the impl block at all. That is the difference
/// between a backlog and a lie: `#[points]` gives every point a default body
/// refusing `unclaimed`, so a point left OUT of the block refuses by name and
/// is counted nowhere, while a point written INTO it as a refusal would be
/// counted here and would make this plane read as answering a scan it does
/// not have. `driver-vulkan/tests/doors.rs` opens on that exact failure, and
/// this is what not repeating it looks like from the table's side.
///
/// `GEMM_CLAIMS` IS THE LINE THAT WAS ADDED, and adding it is the whole of
/// what this family needed from this file: a plane that answers a point and is
/// not named here answers it to nobody, because [`point_claims`] is what
/// `sweep::resolve` binds a lane against. The dense matmul gates every SKU in
/// the catalog, so the seventh hole closing is the one that lets a lane bind
/// at all.
///
/// `GATE_CLAIMS` reads from `attn` for metal's reason, and `LAYOUT_CLAIMS`
/// from `layout` while both of its claimed points fire out of `attn/`.
const VULKAN_CLAIMS: &[&[&str]] = &[
    kernels_vulkan::norm::NORM_CLAIMS,
    kernels_vulkan::mlp::MLP_CLAIMS,
    kernels_vulkan::gemm::GEMM_CLAIMS,
    kernels_vulkan::rope::ROPE_CLAIMS,
    kernels_vulkan::moe::MOE_CLAIMS,
    kernels_vulkan::layout::LAYOUT_CLAIMS,
    kernels_vulkan::attn::ATTENTION_CLAIMS,
    kernels_vulkan::attn::GATE_CLAIMS,
    kernels_vulkan::ssm::SSM_CLAIMS,
];

/// The points a plane's `#[claims]` impl blocks answer — baker's claim
/// table, consulted ahead of the routine `canon` attributes. One slice per
/// migrated family, concatenated; a family lands by adding its line, and the
/// macro grows an aggregate when the list is long enough to want one.
#[must_use]
pub fn point_claims(backend: Backend) -> &'static [&'static str] {
    static CUDA: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    static METAL: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    static WGPU: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    static VULKAN: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    match backend {
        Backend::Cuda => CUDA.get_or_init(|| CUDA_CLAIMS.concat()),
        Backend::Metal => METAL.get_or_init(|| METAL_CLAIMS.concat()),
        Backend::Wgpu => WGPU.get_or_init(|| WGPU_CLAIMS.concat()),
        Backend::Vulkan => VULKAN.get_or_init(|| VULKAN_CLAIMS.concat()),
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
        // EMPTY, AND MEASURED RATHER THAN OMITTED. `kernels-wgpu` states no
        // `CANON` const at all: the fold that deleted this plane's routine
        // layer (99/101 rows) left no row carrying a resolution, because the
        // two it de-macro'd are the sampling and delivery tail — fired
        // outside the lowered plan, so no claim ever reaches them. A plane
        // with nothing here answers every unclaimed point as unclaimed, which
        // is the honest report and is what the backlog is for.
        //
        // `Vulkan` states none for the same reason and by the same route: its
        // 101 `#[routine]` fns and the `linkme` slice that collected them are
        // deleted, and the `#[claims]` blocks that replaced them answer by
        // POINT. There is no `CANON` const in `kernels-vulkan` to name.
        Backend::Wgpu | Backend::Vulkan => &[],
    };
    table
        .iter()
        .find(|(role, _)| *role == claim)
        .map(|(_, symbol)| *symbol)
}

/// The DECLARATION `name` states — its slots, its axes, and how big each of
/// its results is.
///
/// TWO TIERS, ONE LOOKUP, and the prefix is what tells them apart. A tier-1
/// point is `family.method` and is declared on the floor, where every plane
/// owes it; a tier-2 point is one plane's inherent method, declared and
/// claimed by the same line, and a text spells it with that plane's name in
/// front. `model_compiler` is plane-agnostic and reads both through here for
/// the reason [`point_claims`] is here: the tables belong to the kernel
/// crates, and this is the one join over them.
#[must_use]
pub fn point_of(name: &str) -> Option<&'static points::Point> {
    if let Some(rest) = name.strip_prefix("cuda::") {
        return kernels_cuda::attn::TIER2_POINTS
            .iter()
            .find(|p| p.name == rest);
    }
    points::point_of(name)
}

/// Which column of a statement's PARAMS RUN the point's scalar slot named
/// `name` occupies, or `None` where the point states no such scalar.
///
/// A statement's params run is the declaration's scalar run in declaration
/// order — that is the whole rule — and this is the only honest way to read
/// one number out of it by meaning rather than by position. The failure it
/// exists to stop is one a spelled index cannot notice: `head_dim` is param
/// 1 on all five points `model::deployment`'s `ATTENDS` walks today, param 2
/// on `pool.attention_lse`, and absent from `mla.attention_{decode,prefill}`
/// — so a reader spelling `params[1]` reads `ratio` or `heads` as a head
/// width the day the walked set widens, and sizes a plausible wrong pool
/// with no refusal anywhere.
#[must_use]
pub fn param_at(point: &points::Point, name: &str) -> Option<usize> {
    let mut at = 0;
    for slot in point.slots {
        if slot.name == name {
            return (slot.mark == points::Mark::Scalar).then_some(at);
        }
        if slot.mark == points::Mark::Scalar {
            at += 1;
        }
    }
    None
}
