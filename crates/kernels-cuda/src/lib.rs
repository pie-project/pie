//! Pure CUDA kernel definitions — jit unit names, argument marshalling, and
//! launch geometry over a stream; no IR types, no execution state. An engine
//! `Run` resolves plan ids to handles and calls these entry functions (design §8).

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

pub mod attn;

/// **THE VISION TOWERS' DENSE ATTENTION, RE-HOMED BY `#[path]`.**
///
/// The file is `src/attn/dense.rs`, where `.wiki/alto/multimodal.md` §2 put
/// it and where it belongs: an attention op, beside the attention family,
/// sharing nothing with it (no kv pool, no plan, no mask ladder). The module
/// PATH is here rather than `attn::dense` for one reason and it is not a
/// design one — a child module can only be declared by its parent, and the
/// campaign's conflict map closes `src/attn.rs` to this wave. `#[path]` takes
/// the declaration without the edit.
///
/// **The one line that retires this block** is `pub mod dense;` inside
/// `attn.rs`, whenever that file is open again; nothing else moves, and
/// nothing outside this declaration knows the difference.
#[path = "attn/dense.rs"]
pub mod attn_dense;

/// **THE PATCH AXIS'S ROW FOLDS, RE-HOMED BY `#[path]`** — the same detour
/// [`attn_dense`] takes, one family over.
///
/// The file is `src/layout/fold.rs`, beside the layout family whose grammar
/// its entries follow (`layout.pool_rows` and `layout.merge_rows`, as
/// `layout.scatter_rows` and `layout.split_rows` are named); the module PATH
/// is here because the campaign's conflict map closes `src/layout.rs` to this
/// wave and a child module can only be declared by its parent.
///
/// **The one line that retires this block** is `pub mod fold;` inside
/// `layout.rs`.
#[path = "layout/fold.rs"]
pub mod layout_fold;

/// **THE SCATTER THAT HONOURS A DROP SENTINEL, RE-HOMED BY `#[path]`**
/// (multimodal §8.6) — a THIRD file behind the same door, and a separate one
/// from [`layout_fold`] because a scatter is not a fold.
///
/// It exists rather than a guard inside `layout::scatter_rows` for the door's
/// own reason: `src/layout.rs` and `kernels/layout/layout.cuh` are closed to
/// this wave, and the existing op's contract — every route names a row —
/// should not be widened underneath the consumers that rely on it.
///
/// **The one line that retires this block** is `pub mod scatter_live;` inside
/// `layout.rs`.
#[path = "layout/scatter_live.rs"]
pub mod layout_scatter_live;

/// **THE GATHER THAT INTERPOLATES, RE-HOMED BY `#[path]`** (multimodal §9.2)
/// — the fourth file behind the door [`attn_dense`] opened, and separate from
/// [`layout_fold`] and [`layout_scatter_live`] because a weighted gather is
/// neither a fold nor a scatter.
///
/// It is a second entry beside `layout::embed` rather than a widening of it
/// for the reason the dropping scatter is a second entry beside
/// `layout::scatter_rows`: the existing op's operand list is one its consumers
/// rely on, and `src/layout.rs` is closed to this wave either way.
///
/// **The one line that retires this block** is `pub mod embed_weighted;`
/// inside `layout.rs`.
#[path = "layout/embed_weighted.rs"]
pub mod layout_embed_weighted;

/// **THE OBSERVABILITY DOOR'S SCORE CAPTURE, RE-HOMED BY `#[path]`.**
///
/// The file is `src/attn/score.rs`, where `.wiki/alto/attn-score.md` §5 put
/// it — "a NEW FILE outside `attn.rs`/`attn/kv.rs`", the same ownership
/// ruling the dense-attention kernel above got. The module PATH is here for
/// the same non-design reason: a child module can only be declared by its
/// parent, and `src/attn.rs` is closed to this wave.
///
/// **The one line that retires this block** is `pub mod score;` inside
/// `attn.rs`, whenever that file is open again.
#[path = "attn/score.rs"]
pub mod attn_score;

pub mod channel;
pub mod collective;
pub mod custom;
pub mod elemwise;
pub mod error;
pub mod graph;
pub mod jit;
pub mod layout;
pub mod linear;
pub mod source;
pub mod tensor;

pub use error::Error;
pub use jit::{Arg, ArgValue, Ctx, Fire, Launch, Pad, Slabs};

/// **THE RUNTIME THIS BINARY ALREADY LOADED, RE-EXPORTED FOR THIS CRATE'S OWN
/// GPU TEST TARGETS.**
///
/// A `tests/` target is a separate crate, so a device test that has to
/// allocate, copy and open a stream needs `cudarc` by some path. A
/// dev-dependency would be the wrong one twice: it puts a SECOND copy of the
/// runtime-version decision into the graph — the one thing this crate's
/// feature comment forbids, since Cargo unifies features and a `cudarc` chosen
/// here would silently decide for everybody — and, being non-optional, it
/// would drag `cudarc` into a plain `cargo check --workspace` on a box that
/// selected no version at all.
///
/// Re-exporting the dependency the `_cuda` feature already resolved has
/// neither problem: there is exactly one `cudarc` in the graph, and it is the
/// one the kernels fire through.
#[cfg(feature = "_cuda")]
#[doc(hidden)]
pub use cudarc;

pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
