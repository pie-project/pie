//! CUDA's kernel signature table — one row per launcher symbol in `csrc/`.
//!
//! The rows live here, beside the `.cu` files they describe, so that adding a
//! kernel is one source file and one table row in the same directory and the
//! same diff hunk. The words a row is written in — [`KernelSig`], `whole`,
//! `needs`, `lacks`, `sink` — are `kernels`', which is also where the reasons
//! for each of them are.
//!
//! ## Reading this without a GPU
//!
//! The table is the crate's `default-features = false` surface, and that is
//! deliberate: `model-compiler` reads it on every trace, and a compiler dev
//! loop must not pay nvcc to look up a symbol's contract. Turning on
//! `native` adds the CMake build of `csrc/` and nothing to what is below.
//!
//! The table is kept honest from the other end: `model-compiler`'s
//! `kernels::check_plan` refuses any `OpKind::Launch` symbol no row declares,
//! so a kernel cannot be stated by a model text without its contract.

pub use kernels::{Cap, KernelSig, Prepare};

pub mod abi;
pub mod adapter;
pub mod attn;
pub mod driver_internal;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod norm_device;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

/// The `pie_k_*` entry points, for the rows any caller can state.
///
/// `native` builds `libpie_launch_shim.a`, which DEFINES these; this is the
/// matching declaration, generated from the same rows in the same process, so
/// a signature cannot drift from what the shim proves against the header.
///
/// Restricted to portable rows — see
/// [`abi::emit_rust_bindings_portable`]. A row taking `KvCacheLayerView` or a
/// FlashInfer plan is absent, because its declaration would name a
/// `#[repr(C)]` mirror this crate does not hold. Those belong to the shell,
/// which generates the full set against its own mirrors; nothing stops two
/// crates from declaring one symbol, because a declaration is not a
/// definition.
#[cfg(feature = "native")]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}

/// Every kernel a lowered declaration may state.
///
/// The concatenation of the per-family tables, in the order `TABLES` lists
/// them. Order is not semantic — `sig_in` scans linearly and callers look rows
/// up by symbol — but it is stable, so a diff that adds a kernel touches one
/// module and one line.
pub static KERNELS: &[KernelSig] = &concat_tables();

/// `[&[T]] -> [T]` at compile time, because `KERNELS` must stay a `&'static
/// [KernelSig]` for every consumer that already reads it, and neither `concat`
/// nor iterator chaining is const.
const fn concat_tables() -> [KernelSig; TOTAL] {
    // `KernelSig` is not `Copy` — deriving it on a public contract type is a
    // promise this crate should not make — so the array is filled by index.
    let mut out = [EMPTY; TOTAL];
    let mut w = 0;
    let mut t = 0;
    while t < TABLES.len() {
        let table = TABLES[t];
        let mut i = 0;
        while i < table.len() {
            out[w] = copy_sig(&table[i]);
            w += 1;
            i += 1;
        }
        t += 1;
    }
    out
}

const TABLES: &[&[KernelSig]] = &[
    attn::KERNELS, rope::KERNELS, norm::KERNELS, mlp::KERNELS, gemm::KERNELS,
    moe::KERNELS, ssm::KERNELS, quant::KERNELS, layout::KERNELS,
    sample::KERNELS, adapter::KERNELS,
];

const TOTAL: usize = total();

const fn total() -> usize {
    let mut n = 0;
    let mut t = 0;
    while t < TABLES.len() {
        n += TABLES[t].len();
        t += 1;
    }
    n
}

const EMPTY: KernelSig = KernelSig {
    name: "", symbol: "", file: None, launch: kernels::LaunchRule::Unstated,
    whole: false, needs: Prepare::None,
    lacks: &[], sink: None, in_place: &[], depth_prefix_plan: false,
    operands: &[],
    returns: "", axes: &[], grid_param: None,
    head_param: None, heads_param: None, lowered_as: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name, symbol: k.symbol, file: k.file, launch: k.launch,
        whole: k.whole, needs: k.needs,
        lacks: k.lacks, sink: k.sink, in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        operands: k.operands, returns: k.returns, axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param, heads_param: k.heads_param,
        lowered_as: k.lowered_as,
    }
}
