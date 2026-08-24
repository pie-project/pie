use crate::points::{Payload, at_bf16};
use crate::routine::{Bind, Ctx, Fire, In, Out, elementwise};
use kernels::routine::Refusal;

/// The `Mlp` family, claimed. ONE of six points lands, and the five that do
/// not all fail on the same fact.
///
/// # A packed row cannot be cut on this plane
///
/// Five of the six `Mlp` points declare ONE operand — `packed:
/// In<Self::Tensor<T>>` — plus an `intermediate` width, and mean "the gate
/// half is columns `[0, intermediate)` and the up half is
/// `[intermediate, 2 * intermediate)`". A cuda body cuts that with a pointer:
/// `packed.ptr.add(intermediate)` is the up half's address and the kernel
/// never learns there was a seam.
///
/// `mlp/gated.wgsl` binds `gate` at `@group(0) @binding(0)` and `up` at
/// `binding(1)`, and reads `gate[i]` and `up[i]` for the SAME `i`. A wgpu
/// binding is a whole buffer and a handle carries no offset, so binding one
/// packed handle to both slots reads the gate half twice. The strided arm's
/// three pitches do not rescue it either: its addressing is
/// `m * gate_pitch + k` and `m * up_pitch + k`, with no term that could hold
/// a half-row displacement — it exists for gemma's PLE table, where the two
/// operands are genuinely separate rectangles of different widths.
///
/// So `mlp.swiglu`, `mlp.swiglu_clamp`, `mlp.swiglu_clamp_alpha`,
/// `mlp.geglu_tanh_packed` and `mlp.situ` are measured backlog rows with ONE
/// named cause between them, and there are exactly two ways to close it:
///
///  1. a `PIE_PACKED` instantiation of `gated.wgsl` that takes one binding
///     and an `intermediate` word — the W10 rule verbatim, "a packed row is
///     cut by a kernel told the packing"; or
///  2. a dynamic-offset binding, which WebGPU has
///     (`GPUBufferBinding.offset`) and this crate's `Fire`/`ArgValue` do not
///     carry.
///
/// **SEAM (P5):** (1) is a shader, (2) is an `ArgValue::Shaped` that grows an
/// offset and a `driver-wgpu` bind path that honours it. (1) is smaller and
/// is what `kernels-cuda` would call the honest one.
///
/// `mlp.geglu_tanh` is the exception because its DECLARATION states two
/// operands — gemma's gate and up arrive as separate rectangles — which is
/// the shape this file's shaders have always had.
#[kernels_macros::claims]
impl kernels::points::Mlp for Ctx<'_> {
    /// `y = gelu_tanh(gate) * up`, two rectangles in and one out.
    ///
    /// The grid is the ROUTINE's, transcribed: `elementwise` is
    /// `[width * rows, 1, 1]` in ELEMENTS while the shader's invocation owns
    /// a WORD, so the launch is twice the work there is and the shader's
    /// `if (i >= arrayLength(&out_)) return;` stops the second half. That
    /// over-dispatch is `mlp/gated.wgsl`'s stated contract ("the grid is the
    /// extent, and the extent is in WORDS") and not a mistake this claim is
    /// free to fix: the same `LaunchRule::Elementwise` row serves metal,
    /// where a threadgroup is sized at dispatch and the count is exact.
    fn geglu_tanh<T: kernels::points::Scalar>(
        &self,
        gate: In<Payload<T>>,
        up: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mlp.geglu_tanh at an element other than bf16")?;
        self.fire(
            Fire::at("mlp/gated.wgsl", "geglu_tanh_bfloat16")
                .apply(elementwise(gate.width, gate.rows)?),
            &[gate.arg(), up.arg(), y.arg()],
        )
    }
}
