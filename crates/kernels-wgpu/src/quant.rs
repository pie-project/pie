use crate::routine::Ctx;

/// The `Gemm` family: NONE of three points lands, and this is the single
/// largest finding of the wgpu migration.
///
/// # There is no dense matmul on this plane. At all.
///
/// `kernels::points::Gemm` declares three points and all three take one
/// weight of the activation's own element:
///
/// ```text
/// matmul<T>(act: In<Tensor<T>>, w: Const<Tensor<T>>, y: Out<Tensor<T>>)
/// ```
///
/// Every matmul this backend has is in `quant/qmm_t.wgsl` and `quant/qmv.wgsl`
/// and every one of them reads a bank as THREE weights: `w: array<u32>`
/// holding 4- or 8-bit blocks, `scales: array<u32>` and `biases: array<u32>`
/// one pair per group of 32, 64 or 128 elements — or, for gpt-oss, `w` plus a
/// shared-exponent `array<u8>`. The two files' group-size x bit-width x tile
/// cross is 350-odd entrypoints and there is not a `bf16 x bf16` gemm among
/// them, because there was never a reason to build one: a wgpu deployment is
/// a laptop or a browser tab and it loads a quantised checkpoint.
///
/// So `gemm.matmul`, `gemm.lm_head` and `gemm.attention_landing` are not
/// three shaders nobody wrote. They are one TYPE the floor does not have.
///
/// # `Bank<R: Repr>` is the type, and this plane needs it wider than cuda does
///
/// `.wiki/baker.md` already names it — "quantization is `Bank<R: Repr>`, not
/// `Elem`" — and `.wiki/baker-todo.md` queues it behind one point,
/// `moe.matmul_select_bias`, because that is the only place CUDA feels it:
/// cuda has dense kernels for everything else and reaches for a bank only for
/// gpt-oss's mxfp4 experts.
///
/// From here the same gap swallows FOUR families:
///
/// | family | what wants a bank |
/// | --- | --- |
/// | `Gemm` | all three points |
/// | `Layout` | `layout.embed` — a 4-bit affine embedding table |
/// | `Moe` | `matmul_select`, `matmul_select_bias` |
/// | (`Mlp`, `Norm`) | nothing — activations and norms are dense here |
///
/// Which is to say: **on this plane the bank is not an optimisation for one
/// checkpoint, it is how weights are stored.** Twelve of the twenty points
/// that would otherwise land are blocked on it, and no amount of shader work
/// moves any of them.
///
/// # What the type has to carry, measured off this file
///
/// A `Bank` here is not one handle. `qmm_t` binds three (`w`, `scales`,
/// `biases`) and `mxfp4_qmm_t_routed_bias` binds two (`w`, `exponents`), so
/// the type is a SUM over representations rather than a struct — which is
/// what `R: Repr` says. Beyond the handles it must carry two numbers the
/// entrypoint is CHOSEN by, not merely parameterised on: the symbol is
/// `affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}`, so the
/// group size and the bit width are facts about the bank that a body reads,
/// never scalars a statement restates. A statement that restated them could
/// disagree with the bytes the loader wrote, which is the whole failure a
/// declaration exists to prevent.
///
/// **SEAM (floor):** `Bank<R: Repr>` — the handles per representation, plus
/// the group size and the bit width as facts a body READS. Twelve points
/// across four families wait on it here.
///
/// The tile pair (`bm`, `bn`) is NOT part of it: those are a device tiling
/// this plane picks per fire from `m` and `n`, exactly the branch
/// `.wiki/baker.md` puts inside a body.
///
/// # The one thing that would land without `Bank`
///
/// Nothing. `quant/transcode.wgsl` (`encode_u4_*`, `mxfp4_dequant_bf16`) is
/// load-time work with no point, and `qmm_splitk_reduce` is the second half
/// of a split-K matmul whose first half is quantised too.
///
/// # Two grid facts a body cannot read off the WGSL, kept here because the
/// launchers that held them are gone
///
/// * `quant/qmv.wgsl`'s `PIE_ROWREP` (1) and `PIE_ROWW` (4) are shader
///   `const`s the HOST divides by — the y extent is
///   `ceil(out_vec / (2 * PIE_ROWW)) * 2 / PIE_ROWREP`. A host copy that
///   disagrees with the shader dispatches a fraction of the rows and writes
///   nothing to the rest; see `attn::tiled_lanes`, where exactly that
///   happened and only a workgroup census found it.
/// * That y extent exceeds `maxComputeWorkgroupsPerDimension` (65535,
///   WebGPU's guaranteed floor) at the lm head alone. `qmv.wgsl` recombines a
///   y and a z digit in base 65535, so a grid that no longer fits in y must
///   spill into z rather than be clamped.
#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {}
