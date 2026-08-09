# The multimodal encoder towers

Three encoder towers live here — Gemma-4 vision, Gemma-4 audio, and the
Qwen3-VL vision tower. They moved out of `crates/kernels-cuda/csrc/src/vision/`
in the pass recorded as `new-horizon.md` §42.

## Why they are in `driver-cuda`

Not because a tower is "driver-ish". Because of what is actually in the files.

A tower's `.cu` includes two kinds of header: `.cuh` device headers and `.hpp`
host declarations. Every `.cuh` these three include — their own five, plus
`norm/rmsnorm.cuh`, `norm/elementwise.cuh`, `mlp/swiglu.cuh` and
`ssm/causal_conv1d.cuh` — is in `crates/kernels-cuda-new/csrc/src`, the JIT
tree, reached through `-Xcompiler=-iquote`. Not one is in the archive. So
`kernels-cuda` was never compiling tower device code. It was compiling a **host
walk** over device code that already belonged to `kernels-cuda-new`.

A host walk that takes host pixel buffers, a CSR indptr and a `cudaStream_t`,
and that loops a data-dependent number of times, is this crate's kind of
object. `build.rs` makes the same argument for `csrc/supergraph.cu` one block
above the one that compiles this directory.

## Why not an `Execution` row

`Execution::Composed` is the near miss and it does not fit. `Composed` carries
a `&'static [Step]` — a symbol list fixed at build time. A tower iterates
`for im in 0..num_images` over a runtime count, walks a per-layer body whose
depth comes from the checkpoint, and interpolates the position-embedding table
**on the host, between two launches**, from a grid size known only at call
time. The three rows stay `driver_internal` with `whole = true`, which is also
what keeps `model-compiler` unable to name them at all.

## Why the headers stayed behind

`crates/kernels-cuda/csrc/src/vision/*.hpp` did not move. They are the
generated launch shim's compile-time contract: the shim is emitted in
`kernels-cuda` by design and forwards `pie_k_vision_*` into
`kernels::vision::*`. A contract is a header; the definitions are here, one
archive further along the link line. `build.rs` orders the `-l`s shim, then
`pie_vision_towers`, then `pie_kernels_cuda` — caller before callee, twice,
because these files still call `gemm::` (cuBLAS) and the FlashInfer wrappers.

`gemma4_audio.cu` reaches nothing in the archive at all.

## Toolkit-free builds

The `cc::Build` that compiles this directory is inside `build.rs`'s
`bridge`-gated `build()`. `cargo check -p driver-cuda` with no features and no
toolkit on `PATH` never reaches it, and that is a hard gate.

## One thing worth knowing before you edit these

`crates/kernels-cuda-new/tests/launch_rules.rs` pins several lines of
`gemma4_vision.cu` and `gemma4_audio.cu` **by literal text and by line number**.
Moving a line here fails that test by design, and the fix is to re-read the
launcher and correct the citation — in that order.
