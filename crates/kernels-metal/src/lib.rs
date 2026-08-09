//! METAL's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## Why this is not shaped like CUDA's
//!
//! `kernels-cuda` has one row per launcher symbol, because a CUDA launcher is
//! an authored C++ function and there is nothing else it could be. An MSL
//! entrypoint is generated: `quantized_qmm_t.metal` holds one template body and
//! a macro that stamps it over `(group × bits × row tile × column tile)`, so 54
//! of its entrypoints are one kernel evaluated at 54 points.
//!
//! Measured by `scripts/metal-kernel-audit.py`: **480 entrypoints over 99
//! kernels in 28 files.** Enumerating the 480 would state the macro's job a
//! second time, by hand, and `.wiki/kernel-refactor.md` §5's own test — *would
//! the two share one C++ definition?* — answers that they are not distinct
//! kernels. So a row carries its [`Axis`]es and the product is the entrypoint
//! set. `.wiki/kernel-metal-refactor.md` §2 is the argument in full.
//!
//! The consequence worth stating on the way in: **the table is now where the
//! shader tree's coverage is written down.** `qmv_fast` is compiled for six
//! affine formats and `qmv_routed` for one; before this that difference existed
//! only as a name the driver would fail to find at model load.
//!
//! ## What keeps it honest
//!
//! Three checks, at three distances:
//!
//! * `kernels`' own unit tests pin the matcher — that a row covers every point
//!   of its axes and refuses a partial or permuted spelling.
//! * `tests/entrypoints.rs` pins the table's product against
//!   `entrypoints.generated.txt`.
//! * `scripts/metal-kernel-audit.py` pins that file against the shaders, by
//!   preprocessing them the way the Metal runtime does.
//!
//! And from the other end, `model-compiler`'s `kernels::check_plan` refuses any
//! launched symbol no row declares, so a lowered `*.metal.*` text cannot state
//! a kernel this table has not heard of.
//!
//! ## Reading this without a Mac
//!
//! All of the above runs on Linux. Metal compiles its shaders at RUN time, so
//! `default-features = false` gives the table and nothing else — which is what
//! `model-compiler` wants and all it wants — and `native` adds only the staging
//! of `ptir_rng.generated.metal` out of `tensor-compiler`.

pub use kernels::{Axis, Cap, KernelSig, Prepare};

pub mod axes;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

/// The family tables, concatenated.
///
/// A `const fn` fold rather than a `Vec`, so the whole table stays a `&'static`
/// the compiler can read at load with no allocation — the same shape
/// `kernels-cuda` uses for the same reason.
pub static KERNELS: &[KernelSig] = &CONCAT;

const FAMILIES: &[&[KernelSig]] = &[
    attn::KERNELS,
    layout::KERNELS,
    mlp::KERNELS,
    moe::KERNELS,
    norm::KERNELS,
    ptir::KERNELS,
    quant::KERNELS,
    rope::KERNELS,
    sample::KERNELS,
    ssm::KERNELS,
];

const fn total() -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < FAMILIES.len() {
        n += FAMILIES[i].len();
        i += 1;
    }
    n
}

const N: usize = total();

const EMPTY: KernelSig = KernelSig {
    name: "",
    symbol: "",
    file: None,
    launch: kernels::LaunchRule::Unstated,
    whole: false,
    needs: Prepare::None,
    lacks: &[],
    sink: None,
    in_place: &[],
    depth_prefix_plan: false,
    operands: &[],
    returns: "",
    axes: &[],
    grid_param: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name,
        symbol: k.symbol,
        file: k.file,
        launch: k.launch,
        whole: k.whole,
        needs: k.needs,
        lacks: k.lacks,
        sink: k.sink,
        in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        operands: k.operands,
        returns: k.returns,
        axes: k.axes,
        grid_param: k.grid_param,
    }
}

const CONCAT: [KernelSig; N] = {
    let mut out = [EMPTY; N];
    let mut at = 0;
    let mut f = 0;
    while f < FAMILIES.len() {
        let family = FAMILIES[f];
        let mut i = 0;
        while i < family.len() {
            out[at] = copy_sig(&family[i]);
            at += 1;
            i += 1;
        }
        f += 1;
    }
    out
};

/// Every entrypoint the table names, sorted. The set
/// `scripts/metal-kernel-audit.py` compares against the shader tree.
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = KERNELS.iter().flat_map(KernelSig::entrypoints).collect();
    out.sort();
    out
}
